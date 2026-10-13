"""Shared plumbing for the feasibility spikes (torch-free).

Usage:
  from scripts.spikes.common import load_legacy, strip_fences, gpu_state, write_jsonl

D-02: spikes read only the legacy dev and train rows, and drop every row whose paper is on the
benchmark list (train rows are additionally restricted to the train split minus the quarantine).
The held-out file is never opened.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from lightonocr_math import guard, splits
from lightonocr_math.ids import base_id
from lightonocr_math.paths import repo_root

SPIKE_DIR = repo_root() / "data" / "spikes"
EVIDENCE_DIR = repo_root() / ".planning" / "phases" / "02-feasibility-spikes" / "evidence"
LEGACY_DIR = repo_root() / "data" / "external" / "eq_dataset"
_FENCE = "`" * 3


def load_legacy(split: str, *, root: Path | None = None, limit: int | None = None) -> tuple[list[dict], dict]:
    """Legacy rows of `split` ("dev" or "train") without any benchmark paper; returns (rows sorted by id, stats)."""
    if split not in ("dev", "train"):
        raise ValueError(f"split must be 'dev' or 'train', got {split!r}; the eval_heldout file is never read by spikes")
    root = root or LEGACY_DIR
    rows: list[dict] = []
    with open(root / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    total = len(rows)
    kept = [r for r in rows if not splits.is_benchmark(r["arxiv_id"])]
    dropped_benchmark = total - len(kept)
    dropped_quarantine = 0
    if split == "train":
        allowed = set(splits.load_train_ids(sorted({r["arxiv_id"] for r in kept})))
        n = len(kept)
        kept = [r for r in kept if base_id(r["arxiv_id"]) in allowed]
        dropped_quarantine = n - len(kept)
    kept.sort(key=lambda r: r["id"])
    if limit is not None:
        kept = kept[:limit]
    for r in kept:
        r["image_path"] = str(repo_root() / r["image"])
    stats = {
        "total": total,
        "kept": len(kept),
        "dropped_benchmark": dropped_benchmark,
        "dropped_quarantine": dropped_quarantine,
    }
    return kept, stats


def strip_fences(text: str) -> str:
    """Remove a surrounding code fence (opening line and a closing fence line); other text is returned unchanged."""
    if not text.strip().startswith(_FENCE):
        return text
    lines = text.strip().split("\n")
    if len(lines) == 1:
        # one-line answer: fence, optional language token, content, optional closing fence on the same line
        body = lines[0][len(_FENCE):]
        body = re.sub(r"^[A-Za-z0-9_+-]+(?=\s)", "", body)
        body = body.rstrip()
        if body.endswith(_FENCE):
            body = body[: -len(_FENCE)]
        return body.strip()
    lines = lines[1:]
    if lines and lines[-1].rstrip() == _FENCE:
        lines = lines[:-1]
    return "\n".join(lines).strip("\n")


def _rocm_vram() -> tuple[int | None, int | None]:
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=30, check=False)
        out = r.stdout
        data, _ = json.JSONDecoder().raw_decode(out[out.find("{"):])
        cards = [k for k in data if str(k).startswith("card")]
        info = data[min(cards)]
        return int(info["VRAM Total Memory (B)"]), int(info["VRAM Total Used Memory (B)"])
    except Exception:  # noqa: BLE001 - unavailable tooling means unknown
        return None, None


def gpu_state() -> dict:
    """Snapshot of GPU and RAM state. Never unloads anything."""
    total, used = _rocm_vram()
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "vram_free_bytes": guard.vram_free_bytes(),
        "vram_total_bytes": total,
        "vram_used_bytes": used,
        "holders": guard.vram_holders(),
        "mem_available_bytes": guard.mem_available_bytes(),
    }


def _atomic_write(path: Path, data: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, obj: object) -> None:
    _atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
