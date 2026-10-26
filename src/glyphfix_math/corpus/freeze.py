"""Runtime guard for the frozen benchmark (fails closed).

Usage:
  from glyphfix_math.corpus import freeze
  freeze.assert_benchmark_frozen(bench_root)            # freeze file and records.jsonl match the pinned digests
  freeze.assert_pages_frozen(bench_root)                # same for pages/PAGES_FREEZE.json and pages.jsonl
  freeze.is_frozen(bench_root)                          # True when the root is the repo benchmark, holds the pinned records, or has a freeze file
  freeze.records_are_frozen(path) / freeze.pages_are_frozen(path)   # content (sha256) matches the pinned digest
  freeze.pages_dir_is_frozen(pages_dir)                 # same idea for a pages build output directory
  freeze.records_text_sha256(rows)                      # digest write_jsonl would produce for `rows`

Rules: the pinned digests live in _benchmark_frozen and are never changed here. Every check raises
FreezeError (a splits.LeakageError) on a mismatch or a missing file, so callers stop before overwriting or using drifted data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from glyphfix_math.corpus import _benchmark_frozen
from glyphfix_math.corpus._io import sha256_bytes, sha256_file
from glyphfix_math.splits import LeakageError

class FreezeError(LeakageError):
    def __init__(self, message: str):
        super().__init__(message, {})


FREEZE_NAME = "BENCHMARK_FREEZE.json"
PAGES_FREEZE_REL = Path("pages") / "PAGES_FREEZE.json"


REPO_BENCH = Path(__file__).resolve().parents[3] / "data" / "benchmark"  # this checkout, ignoring GLYPHFIX_MATH_ROOT


def _sha_is(path: Path, pinned: str) -> bool:
    path = Path(path)
    return path.is_file() and sha256_file(path) == pinned


def records_are_frozen(path: Path) -> bool:
    """True when the file at `path` is byte-identical to the frozen benchmark records, wherever it lives."""
    return _sha_is(path, _benchmark_frozen.BENCHMARK_RECORDS_SHA256)


def pages_are_frozen(path: Path) -> bool:
    return _sha_is(path, _benchmark_frozen.PAGES_JSONL_SHA256)


def is_frozen(bench_root: Path) -> bool:
    """A benchmark root is frozen when it is this checkout's data/benchmark, holds the pinned records, or has a freeze file.

    Deleting BENCHMARK_FREEZE.json therefore does not unfreeze the repo benchmark or any copy of its records;
    callers then run assert_benchmark_frozen, which fails closed on the missing file.
    """
    root = Path(bench_root)
    return (
        (root / FREEZE_NAME).exists()
        or root.resolve() == REPO_BENCH.resolve()
        or records_are_frozen(root / "records.jsonl")
    )


def pages_dir_is_frozen(pages_dir: Path) -> bool:
    """A pages output directory is frozen when it is the repo's benchmark pages dir, has PAGES_FREEZE.json, or holds the pinned pages.jsonl."""
    d = Path(pages_dir)
    return (
        (d / "PAGES_FREEZE.json").exists()
        or d.resolve() == (REPO_BENCH / "pages").resolve()
        or pages_are_frozen(d / "pages.jsonl")
    )


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
    if fz["records_sha256"] != _benchmark_frozen.BENCHMARK_RECORDS_SHA256:
        raise FreezeError(f"{bench_root / FREEZE_NAME} records digest differs from the pinned records digest")
    return fz


def assert_pages_frozen(bench_root: Path) -> dict:
    bench_root = Path(bench_root)
    fz = _checked_freeze(bench_root / PAGES_FREEZE_REL, _benchmark_frozen.PAGES_FREEZE_SHA256, "pages")
    rows = bench_root / "pages" / "pages.jsonl"
    if not rows.exists() or sha256_file(rows) != fz["pages_sha256"]:
        raise FreezeError(f"{rows} differs from the frozen pages digest")
    if fz["pages_sha256"] != _benchmark_frozen.PAGES_JSONL_SHA256:
        raise FreezeError(f"{bench_root / PAGES_FREEZE_REL} pages digest differs from the pinned pages digest")
    return fz
