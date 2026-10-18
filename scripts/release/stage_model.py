"""Stage the model half of the release into release/model/ (local only, nothing is uploaded).

Usage: uv run python -m scripts.release.stage_model [--out release/model] [--projector-blob PATH]

The adapter, the merged Q8_0 GGUF and the stock projector are copied and re-hashed against their pinned digests; a
mismatch raises and leaves no destination file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.release.common import (
    ADAPTER_SHA256,
    ADAPTER_SUBDIR,
    BASE_MODEL_ID,
    BASE_REVISION,
    DISCLAIMER,
    EXPECTED,
    GGUF_NAME,
    GGUF_Q8_SHA256,
    LLAMA_CPP_COMMIT,
    MMPROJ_NAME,
    MODEL_DIR,
    OLLAMA_DIGEST,
    PROJECTOR_SHA256,
    REGISTRY_TAG,
    RESULTS_JSON,
    SCORER_VERSION,
    TRAINSET_SHA256,
    caveats,
    load_json,
    render_template,
    sha256_stream,
    write_sha256sums,
)

REGISTRY = Path("data/models/REGISTRY.json")
RUN_DIR = Path("data/models/p4-full-1")
SOURCE_MODELFILE = RUN_DIR / "Modelfile"
CARD_TEMPLATE = Path("release_templates/model_card.md")
NOTICE_TEMPLATE = Path("release_templates/NOTICE.model.txt")
LICENSE_TEMPLATE = Path("release_templates/LICENSE-Apache-2.0.txt")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PROJECTOR_SOURCE = (
    "https://huggingface.co/mradermacher/LightOnOCR-3-4B-GGUF (the mmproj that the stock Ollama tag "
    "lighton-ocr3-q8-16k was built from)"
)
OLLAMA_VERSION = "0.40.1"


class StageError(RuntimeError):
    pass


def copy_verified(src: Path | str, dst: Path | str, expected: str) -> str:
    """Copy src to dst, re-hash the copy in 1 MiB chunks and require the expected sha256; no dst file on mismatch."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    actual = sha256_stream(dst)
    if actual != expected:
        dst.unlink(missing_ok=True)
        raise StageError(f"{src.name}: sha256 {actual} differs from the pinned {expected}")
    return actual


def stage_adapter(src_dir: Path | str, dst_dir: Path | str, expected_safetensors: str = ADAPTER_SHA256) -> None:
    """Copy only the two adapter files; the auto-generated adapter README stub is never shipped."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    for name in ADAPTER_FILES:
        if not (src_dir / name).is_file():
            raise StageError(f"adapter file missing: {src_dir / name}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_dir / "adapter_config.json", dst_dir / "adapter_config.json")
    copy_verified(src_dir / "adapter_model.safetensors", dst_dir / "adapter_model.safetensors", expected_safetensors)


def make_modelfile(source_text: str) -> str:
    """Relative FROM lines; the TEMPLATE and PARAMETER lines are kept byte-identical and in order."""
    out: list[str] = []
    n_from = 0
    for line in source_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("FROM "):
            n_from += 1
            if n_from == 1:
                out.append(f"FROM ./{GGUF_NAME}")
            elif n_from == 2:
                out.append(f"FROM ./{MMPROJ_NAME}")
            else:
                raise StageError("source Modelfile has more than two FROM lines")
        else:
            out.append(line)
    if n_from != 2:
        raise StageError(f"source Modelfile must have exactly two FROM lines, found {n_from}")
    return "\n".join(out) + "\n"


def modelfile_parameter(modelfile: str, name: str) -> str:
    m = re.search(rf"^PARAMETER {re.escape(name)} (\S+)$", modelfile, re.MULTILINE)
    if not m:
        raise StageError(f"Modelfile has no PARAMETER {name}")
    return m.group(1)


def render_notice(template: str, llama_cpp_commit: str = LLAMA_CPP_COMMIT, base_revision: str = BASE_REVISION) -> str:
    return render_template(
        template,
        {
            "base_revision": base_revision,
            "llama_cpp_commit": llama_cpp_commit,
            "projector_source": PROJECTOR_SOURCE,
        },
    )


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def files_table(rows: list[tuple[str, int, str]]) -> str:
    lines = ["| File | Size (bytes) | SHA-256 |", "|------|--------------|---------|"]
    lines += [f"| `{name}` | {size} | `{sha}` |" for name, size, sha in rows]
    return "\n".join(lines)


def render_model_card(
    template: str,
    results: dict,
    entry: dict,
    file_table: list[tuple[str, int, str]],
    modelfile: str,
    train_rows: int = EXPECTED["train_rows"],
) -> str:
    h, t, rc, v = results["headline"], results["transfer"], entry["run_config"], entry["versions"]
    values = {
        "disclaimer": DISCLAIMER,
        "base_revision": entry["base_model"]["revision"],
        "llama_cpp_commit": entry["llama_cpp_commit"],
        "files_table": files_table(file_table),
        "num_ctx": modelfile_parameter(modelfile, "num_ctx"),
        "num_predict": modelfile_parameter(modelfile, "num_predict"),
        "transformers_version": v["transformers"],
        "peft_version": v["peft"],
        "min_px": rc["min_px"],
        "max_px": rc["max_px"],
        "rank": rc["rank"],
        "alpha": rc["alpha"],
        "lr": rc["lr"],
        "epochs": rc["epochs"],
        "step": entry["selection"]["chosen_step"],
        "train_rows": train_rows,
        "trainset_sha256": entry["trainset_sha256"],
        "n": h["n"],
        "n_papers": h["n_papers"],
        "stock_rate": _fmt(h["stock_rate"]),
        "tuned_rate": _fmt(h["tuned_rate"]),
        "delta": f"{h['delta']:+.3f}",
        "ci_low": f"{h['ci_low']:+.3f}",
        "ci_high": f"{h['ci_high']:+.3f}",
        "r2w": h["flip"]["right_to_wrong"],
        "w2r": h["flip"]["wrong_to_right"],
        "page_n": t["flip"]["n"],
        "page_papers": t["n_papers"],
        "page_stock_rate": _fmt(t["stock_rate"]),
        "page_tuned_rate": _fmt(t["tuned_rate"]),
        "limits": "\n".join(f"- {s}" for s in caveats(results)),
    }
    return render_template(template, values)


def provenance(entry: dict, results_sha256: str) -> dict:
    ac = entry["adapter_config"]
    return {
        "base_model": {"id": BASE_MODEL_ID, "revision": entry["base_model"]["revision"]},
        "adapter": {
            "sha256": entry["adapter_sha256"],
            "rank": ac["r"],
            "alpha": ac["alpha"],
            "use_rslora": ac["use_rslora"],
            "target_modules": ac["target_regex"],
            "peft_version": entry["versions"]["peft"],
            "step": entry["selection"]["chosen_step"],
            "run_id": entry["run_id"],
        },
        "trainset_sha256": entry["trainset_sha256"],
        "gguf": {
            "file": GGUF_NAME,
            "sha256": entry["gguf_text_sha256"],
            "llama_cpp_commit": entry["llama_cpp_commit"],
            "quantization": "Q8_0",
        },
        "projector": {
            "file": MMPROJ_NAME,
            "sha256": entry["projector_sha256"],
            "source": PROJECTOR_SOURCE,
            "modified": False,
        },
        "registered_ollama_digest": entry["ollama_digest"],
        "versions": {**entry["versions"], "ollama": OLLAMA_VERSION},
        "phase5_results_sha256": results_sha256,
        "scorer_version": SCORER_VERSION,
    }


def find_projector_blob(override: Path | None, tag: str = REGISTRY_TAG) -> Path:
    """Locate the projector blob: explicit path, else the second FROM line of `ollama show --modelfile`, else export.json."""
    if override:
        return Path(override)
    try:
        out = subprocess.run(
            ["ollama", "show", "--modelfile", tag], capture_output=True, text=True, check=True, timeout=60
        ).stdout
        froms = [ln.split(" ", 1)[1].strip() for ln in out.splitlines() if ln.startswith("FROM ")]
        if len(froms) >= 2 and Path(froms[1]).is_file():
            return Path(froms[1])
    except (OSError, subprocess.SubprocessError):
        pass
    exp = Path(load_json(RUN_DIR / "export.json")["projector_blob"])
    if exp.is_file():
        return exp
    raise StageError("projector blob not found; pass --projector-blob")


def stage(out: Path = MODEL_DIR, projector_blob: Path | None = None) -> dict:
    entry = next(e for e in load_json(REGISTRY)["entries"] if e["tag"] == REGISTRY_TAG)
    if entry["adapter_sha256"] != ADAPTER_SHA256 or entry["trainset_sha256"] != TRAINSET_SHA256:
        raise StageError("registry entry differs from the pinned adapter or trainset digest")
    results_sha = sha256_stream(RESULTS_JSON)
    results = load_json(RESULTS_JSON)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stage_adapter(Path(entry["adapter_path"]), out / ADAPTER_SUBDIR)
    copy_verified(Path(entry["gguf_path"]), out / GGUF_NAME, GGUF_Q8_SHA256)
    copy_verified(find_projector_blob(projector_blob), out / MMPROJ_NAME, PROJECTOR_SHA256)

    modelfile = make_modelfile(SOURCE_MODELFILE.read_text(encoding="utf-8"))
    (out / "Modelfile").write_text(modelfile, encoding="utf-8")
    (out / "NOTICE").write_text(render_notice(NOTICE_TEMPLATE.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copyfile(LICENSE_TEMPLATE, out / "LICENSE")
    prov = provenance(entry, results_sha)
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = [
        (p.relative_to(out).as_posix(), p.stat().st_size, sha256_stream(p))
        for p in (
            out / ADAPTER_SUBDIR / "adapter_config.json",
            out / ADAPTER_SUBDIR / "adapter_model.safetensors",
            out / GGUF_NAME,
            out / MMPROJ_NAME,
        )
    ]
    card = render_model_card(CARD_TEMPLATE.read_text(encoding="utf-8"), results, entry, table, modelfile)
    (out / "README.md").write_text(card, encoding="utf-8")
    write_sha256sums(out)
    return {"out": str(out), "ollama_digest": OLLAMA_DIGEST, "files": len(table)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    ap.add_argument("--projector-blob", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        info = stage(args.out, args.projector_blob)
    except StageError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
