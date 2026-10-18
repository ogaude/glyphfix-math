"""Render the publish runbook into release/PUBLISH.md (local only; the commands in it are run by hand later).

Usage: uv run python -m scripts.release.stage_publish [--out release/PUBLISH.md]

The placeholder tokens (@@NAME@@) stay in the file until scripts.release.apply_names replaces them.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.release.common import (
    ADAPTER_SHA256,
    ADAPTER_SUBDIR,
    BENCH_PARQUET,
    DATASET_DIR,
    GGUF_NAME,
    GGUF_Q8_SHA256,
    MMPROJ_NAME,
    MODEL_DIR,
    PROJECTOR_SHA256,
    PUBLISH_MD,
    TRAIN_PARQUET,
    load_json,
    render_template,
)

TEMPLATE = Path("release_templates/PUBLISH.md")


class StageError(RuntimeError):
    pass


def find_cli(name: str) -> str | None:
    """Locate a user-installed CLI, skipping this project's virtualenv (it ships its own, different hf)."""
    venv_bin = Path(sys.prefix).resolve() / "bin"
    dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and Path(d).resolve() != venv_bin]
    return shutil.which(name, path=os.pathsep.join(dirs))


def cli_version(cmd: str) -> str:
    exe = find_cli(cmd)
    if exe is None:
        return "unknown"
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    m = re.search(r"\d+\.\d+\.\d+", out)
    return m.group(0) if m else "unknown"


def human_size(path: Path) -> str:
    if not path.is_file():
        raise StageError(f"missing staged file {path}: run the stage scripts first")
    n = path.stat().st_size
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError


def build_values(dataset_dir: Path = DATASET_DIR, model_dir: Path = MODEL_DIR) -> dict[str, str]:
    manifest = Path(dataset_dir) / "DATASET_MANIFEST.json"
    if not manifest.is_file():
        raise StageError("release/dataset/DATASET_MANIFEST.json is missing: run stage_dataset first")
    parquet = load_json(manifest)["parquet_sha256"]
    dataset_dir, model_dir = Path(dataset_dir), Path(model_dir)
    return {
        "hf_version": cli_version("hf"),
        "gh_version": cli_version("gh"),
        "gguf_name": GGUF_NAME,
        "mmproj_name": MMPROJ_NAME,
        "gguf_sha256": GGUF_Q8_SHA256,
        "mmproj_sha256": PROJECTOR_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
        "dataset_train_sha256": parquet[TRAIN_PARQUET],
        "dataset_bench_sha256": parquet[BENCH_PARQUET],
        "gguf_size": human_size(model_dir / GGUF_NAME),
        "mmproj_size": human_size(model_dir / MMPROJ_NAME),
        "adapter_size": human_size(model_dir / ADAPTER_SUBDIR / "adapter_model.safetensors"),
        "dataset_train_size": human_size(dataset_dir / TRAIN_PARQUET),
        "dataset_bench_size": human_size(dataset_dir / BENCH_PARQUET),
    }


def stage_publish(out: Path = PUBLISH_MD, template: Path = TEMPLATE, values: dict[str, str] | None = None) -> Path:
    values = values if values is not None else build_values()
    text = render_template(Path(template).read_text(encoding="utf-8"), values)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PUBLISH_MD))
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        out = stage_publish(Path(a.out))
    except StageError as e:
        print(f"stage_publish: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
