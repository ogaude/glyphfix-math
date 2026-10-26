"""Merge a trained LoRA adapter into the bf16 base on CPU and prove which tensors changed.

Usage:
  uv run python -m glyphfix_math.guard --ram-gb 11 --timeout 1800 -- \
      .venv-lighton/bin/python -m scripts.export_merge --adapter DIR \
      --out data/models/RUN_ID/merged-hf --evidence data/models/RUN_ID/merge_check.json

CPU only. The adapter is merged into the bf16 base (never a quantized one) with merge_and_unload, the
result is saved as safetensors and compared tensor by tensor with the pinned base snapshot. The merge
passes only when exactly the 200 target-module weights differ and no tensor name appeared or vanished.
Exit codes: 0 pass, 1 merge check failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

EXPECTED_CHANGED = 200
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight")
_PEFT_PREFIX = "base_model.model."


def expected_changed_names(adapter_keys) -> list[str]:
    """Base weight names modified by an adapter: one per wrapped module, de-duplicated and sorted."""
    names: set[str] = set()
    for key in adapter_keys:
        for suffix in _LORA_SUFFIXES:
            if key.endswith(suffix):
                stem = key[: -len(suffix)]
                stem = stem.removeprefix(_PEFT_PREFIX)
                names.add(stem + ".weight")
                break
    return sorted(names)


def merge_verdict(compare: dict, expected: int = EXPECTED_CHANGED) -> bool:
    """True only when exactly `expected` tensors changed and no tensor name is only on one side."""
    return (
        compare.get("n_value_mismatch") == expected
        and compare.get("n_names_only_in_base") == 0
        and compare.get("n_names_only_in_out") == 0
        and not compare.get("dtype_mismatch")
        and not compare.get("shape_mismatch")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="adapter directory with adapter_model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args(argv)

    adapter_dir = Path(args.adapter)
    weights = adapter_dir / "adapter_model.safetensors"
    cfg_path = adapter_dir / "adapter_config.json"
    if not weights.is_file() or not cfg_path.is_file():
        print(f"adapter files missing under {adapter_dir}", file=sys.stderr)
        return 2

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import peft
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    from glyphfix_math.corpus._io import sha256_file, write_json
    from glyphfix_math.train import infer, lora
    from scripts.spikes.merge_untrained import _KEEP_SKIP, _config_diff, compare_safetensors, snapshot_dir

    t0 = time.time()
    snap = snapshot_dir()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # stale shards would be indexed by compare_safetensors and corrupt the comparison
    for stale in [*out.glob("*.safetensors"), *out.glob("*.safetensors.index.json")]:
        stale.unlink()

    adapter_sha = sha256_file(weights)
    adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    with safe_open(str(weights), framework="pt") as fh:
        expected_names = expected_changed_names(list(fh.keys()))

    model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.bfloat16, device_map="cpu")
    pm = peft.PeftModel.from_pretrained(model, str(adapter_dir))
    info = lora.check_targets(pm)
    if info["n_wrapped"] != EXPECTED_CHANGED:
        print(f"adapter wraps {info['n_wrapped']} modules, expected {EXPECTED_CHANGED}", file=sys.stderr)
        return 1

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

    cdiff = _config_diff(json.loads((snap / "config.json").read_text()), json.loads((out / "config.json").read_text()))
    cmp = compare_safetensors(snap, out)
    ok = merge_verdict(cmp)
    mismatched = set(cmp.get("value_mismatch", []))
    ev = {
        "snapshot": str(snap),
        "revision": infer.REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "adapter_path": str(adapter_dir),
        "adapter_sha256": adapter_sha,
        "adapter_config": {
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "use_rslora": adapter_cfg.get("use_rslora"),
            "target_modules": adapter_cfg.get("target_modules"),
        },
        "n_wrapped": info["n_wrapped"],
        "trainable": info["trainable"],
        "n_expected_names": len(expected_names),
        "expected_names_sample_in_mismatch": all(n in mismatched for n in expected_names[:50]),
        "expected_changed": EXPECTED_CHANGED,
        "copied_files": copied,
        "config_diff": cdiff,
        "compare": cmp,
        "pass": ok,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(Path(args.evidence), ev)
    print("merge check", "PASS" if ok else "FAIL", "n_value_mismatch", cmp["n_value_mismatch"], "n_tensors", cmp["n_tensors"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
