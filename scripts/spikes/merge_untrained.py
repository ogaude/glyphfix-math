"""Merge an untrained LoRA adapter into the base in bf16 and check every tensor is unchanged.

Usage:
  uv run python -m lightonocr_math.guard --ram-gb 11 --timeout 1800 -- \
      .venv-lighton/bin/python -m scripts.spikes.merge_untrained [--out DIR] [--rank 32] [--alpha 8]

CPU only. The adapter has the same shape as the real training adapter; lora_B starts at zero, so
the merged weights must equal the base bit for bit. The comparison streams tensor by tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

from scripts.spikes.common import SPIKE_DIR, sha256_file, write_json

EXPORT_DIR = SPIKE_DIR / "export"
_KEEP_SKIP = {"model.safetensors.index.json", "config.json"}


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    from lightonocr_math.train import infer

    return Path(snapshot_download(infer.MODEL_ID, revision=infer.REVISION, local_files_only=True))


def _safetensor_files(d: Path) -> list[Path]:
    return sorted(Path(d).glob("*.safetensors"))


def _index(d: Path) -> dict[str, Path]:
    """Map tensor name -> file for every safetensors file in d."""
    from safetensors import safe_open

    out: dict[str, Path] = {}
    for f in _safetensor_files(d):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():  # noqa: SIM118 - safe_open handle has no __iter__
                out[k] = f
    return out


def compare_safetensors(base_dir: Path, out_dir: Path, rename: dict[str, str] | None = None) -> dict:
    """Stream-compare all tensors of two safetensors folders; `rename` maps base name -> out name."""
    import torch
    from safetensors import safe_open

    rename = rename or {}
    bidx = _index(Path(base_dir))
    oidx = _index(Path(out_dir))
    mapped = {rename.get(k, k): k for k in bidx}
    only_base = sorted(k for k in mapped if k not in oidx)
    only_out = sorted(k for k in oidx if k not in mapped)
    dtype_mm: list[str] = []
    shape_mm: list[str] = []
    value_mm: list[str] = []
    n = 0
    for oname, bname in sorted(mapped.items()):
        if oname not in oidx:
            continue
        with safe_open(str(bidx[bname]), framework="pt") as bf, safe_open(str(oidx[oname]), framework="pt") as of:
            a = bf.get_tensor(bname)
            b = of.get_tensor(oname)
        n += 1
        if a.dtype != b.dtype:
            dtype_mm.append(bname)
        elif a.shape != b.shape:
            shape_mm.append(bname)
        elif not torch.equal(a, b):
            value_mm.append(bname)
        del a, b
    bit_identical = not (only_base or only_out or dtype_mm or shape_mm or value_mm)
    return {
        "n_tensors": n,
        "n_tensors_base": len(bidx),
        "n_tensors_out": len(oidx),
        "names_only_in_base": only_base[:50],
        "names_only_in_out": only_out[:50],
        "n_names_only_in_base": len(only_base),
        "n_names_only_in_out": len(only_out),
        "dtype_mismatch": dtype_mm[:50],
        "shape_mismatch": shape_mm[:50],
        "value_mismatch": value_mm[:50],
        "n_value_mismatch": len(value_mm),
        "rename_mapping": rename,
        "bit_identical": bit_identical,
    }


def _config_diff(a: dict, b: dict, prefix: str = "") -> list[dict]:
    diffs: list[dict] = []
    for k in sorted(set(a) | set(b)):
        path = f"{prefix}{k}"
        if k not in a:
            diffs.append({"key": path, "base": None, "saved": "<added>"})
        elif k not in b:
            diffs.append({"key": path, "base": "<present>", "saved": None})
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            diffs.extend(_config_diff(a[k], b[k], path + "."))
        elif a[k] != b[k]:
            diffs.append({"key": path, "base": a[k], "saved": b[k]})
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(EXPORT_DIR / "merged-hf"))
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--evidence", default=str(EXPORT_DIR / "merge_check.json"))
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import peft
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText

    from lightonocr_math.train import infer, lora

    t0 = time.time()
    snap = snapshot_dir()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.bfloat16, device_map="cpu")
    pm = peft.get_peft_model(model, lora.lora_config(args.rank, args.alpha))
    info = lora.check_targets(pm)
    b_all_zero = all(bool((p == 0).all()) for n, p in pm.named_parameters() if "lora_B" in n)
    n_b = sum(1 for n, _ in pm.named_parameters() if "lora_B" in n)
    if not b_all_zero or n_b == 0:
        raise SystemExit("lora_B is not all zeros: the adapter is not untrained")

    adapter_dir = EXPORT_DIR / "untrained-adapter"
    pm.save_pretrained(str(adapter_dir))
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_file)

    merged = pm.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    del merged, pm, model

    copied: list[str] = []
    for f in sorted(snap.iterdir()):
        if f.name in _KEEP_SKIP or f.suffix == ".safetensors":
            continue
        if f.suffix in (".json", ".jinja", ".txt"):
            shutil.copyfile(f.resolve(), out / f.name)
            copied.append(f.name)

    base_cfg = json.loads((snap / "config.json").read_text())
    saved_cfg = json.loads((out / "config.json").read_text())
    cdiff = _config_diff(base_cfg, saved_cfg)

    cmp = compare_safetensors(snap, out)
    mapping_note = None
    if not cmp["bit_identical"] and cmp["n_names_only_in_base"] and cmp["n_names_only_in_out"]:
        mapping_note = "names differ; inspect names_only_in_* for a systematic rename"

    ev = {
        "snapshot": str(snap),
        "revision": infer.REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "rank": args.rank,
        "alpha": args.alpha,
        "n_wrapped": info["n_wrapped"],
        "lora_b_all_zero": b_all_zero,
        "n_lora_b_tensors": n_b,
        "adapter_sha256": adapter_sha,
        "copied_files": copied,
        "config_diff": cdiff,
        "compare": cmp,
        "mapping_note": mapping_note,
        "bit_identical": cmp["bit_identical"],
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(Path(args.evidence), ev)
    print("bit_identical", ev["bit_identical"], "n_tensors", cmp["n_tensors"], "config_diff", len(cdiff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
