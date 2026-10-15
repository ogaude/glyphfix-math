"""Runtime guard for the frozen benchmark (fails closed).

Usage:
  from lightonocr_math.corpus import freeze
  freeze.assert_benchmark_frozen(bench_root)            # freeze file and records.jsonl match the pinned digests
  freeze.assert_pages_frozen(bench_root)                # same for pages/PAGES_FREEZE.json and pages.jsonl
  freeze.is_frozen(bench_root)                          # True when BENCHMARK_FREEZE.json exists
  freeze.records_text_sha256(rows)                      # digest write_jsonl would produce for `rows`

Rules: the pinned digests live in _benchmark_frozen and are never changed here. Every check raises
FreezeError (a splits.LeakageError) on a mismatch or a missing file, so callers stop before overwriting or using drifted data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from lightonocr_math.corpus import _benchmark_frozen
from lightonocr_math.corpus._io import sha256_bytes, sha256_file
from lightonocr_math.splits import LeakageError

class FreezeError(LeakageError):
    def __init__(self, message: str):
        super().__init__(message, {})


FREEZE_NAME = "BENCHMARK_FREEZE.json"
PAGES_FREEZE_REL = Path("pages") / "PAGES_FREEZE.json"


def is_frozen(bench_root: Path) -> bool:
    return (Path(bench_root) / FREEZE_NAME).exists()


def records_text_sha256(rows: Iterable[dict]) -> str:
    """sha256 of the bytes corpus._io.write_jsonl writes for `rows`."""
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    return sha256_bytes(text.encode("utf-8"))


def _checked_freeze(freeze_path: Path, pinned: str, what: str) -> dict:
    if not freeze_path.exists():
        raise FreezeError(f"{what} freeze file missing: {freeze_path}")
    if sha256_file(freeze_path) != pinned:
        raise FreezeError(f"{freeze_path} differs from the pinned digest")
    return json.loads(freeze_path.read_text(encoding="utf-8"))


def assert_benchmark_frozen(bench_root: Path) -> dict:
    bench_root = Path(bench_root)
    fz = _checked_freeze(bench_root / FREEZE_NAME, _benchmark_frozen.BENCHMARK_FREEZE_SHA256, "benchmark")
    rec = bench_root / "records.jsonl"
    if not rec.exists() or sha256_file(rec) != fz["records_sha256"]:
        raise FreezeError(f"{rec} differs from the frozen records digest")
    return fz


def assert_pages_frozen(bench_root: Path) -> dict:
    bench_root = Path(bench_root)
    fz = _checked_freeze(bench_root / PAGES_FREEZE_REL, _benchmark_frozen.PAGES_FREEZE_SHA256, "pages")
    rows = bench_root / "pages" / "pages.jsonl"
    if not rows.exists() or sha256_file(rows) != fz["pages_sha256"]:
        raise FreezeError(f"{rows} differs from the frozen pages digest")
    return fz
