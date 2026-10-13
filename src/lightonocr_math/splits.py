"""Held-out split rule, freeze command and leakage gate.

Usage:
  python -m lightonocr_math.splits freeze --candidates data/candidates.jsonl [--out DIR] [--salt HEX]
  python -m lightonocr_math.splits verify [--candidates P] [--dir DIR]
  python -m lightonocr_math.splits info

  from lightonocr_math.splits import assert_disjoint, load_train_ids
  ids = load_train_ids(all_ids)                 # train split only, quarantine removed
  assert_disjoint(ids, context="build crops")   # raises LeakageError on any benchmark id

Rule: u = int.from_bytes(sha256((salt + base_arxiv_id).encode()).digest()[:8], "big") / 2**64.
u < 0.15 is benchmark, u < 0.20 is dev, otherwise train. The hash uses the id without its
version suffix, so v1 and v2 of one paper always land in the same split.

Frozen artefacts in data/splits/: SPLIT_RULE.json, benchmark_ids.txt, quarantine_ids.txt,
SPLIT_STATS.json and SHA256SUMS. sha256(SHA256SUMS) is pinned in _frozen.FROZEN_SPLIT_DIGEST
and checked when this module is imported. freeze() refuses to overwrite an existing benchmark
list: the benchmark never changes once written.

The gate checks both the committed list and the recomputed rule, so a paper missing from the
candidate population but hashing into the benchmark range is still refused. Quarantine
(papers sharing a normalized full author name with a benchmark paper) is applied by
load_train_ids and assert_disjoint.

Known limit: exact-id gate plus author quarantine only. Near-duplicate detection between a
benchmark paper and a training paper (reposts, extended versions under new ids) is not
covered here; it belongs to the later corpus work (COR-06, deferred).
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import secrets
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lightonocr_math import _frozen, paths
from lightonocr_math.authors import AUTHOR_RULE, quarantine
from lightonocr_math.ids import base_id, is_arxiv_id

SPLIT_RULE_VERSION = "1"
FRACTIONS = {"benchmark": 0.15, "dev": 0.05, "train": 0.80}
_BENCH_END = 0.15
_DEV_END = 0.20
HASH_DESCRIPTION = "sha256(salt + base_arxiv_id) -> first 8 bytes big-endian / 2**64"
_FILES = ("SPLIT_RULE.json", "benchmark_ids.txt", "quarantine_ids.txt", "SPLIT_STATS.json")

Split = Literal["benchmark", "dev", "train"]


class SplitNotFrozenError(RuntimeError):
    """The split has not been frozen and pinned yet."""


class SplitIntegrityError(RuntimeError):
    """A committed split file does not match its recorded digest."""


class LeakageError(ValueError):
    def __init__(self, message: str, offenders: dict[str, str]):
        super().__init__(message)
        self.offenders = offenders


def u01(base: str, salt: str) -> float:
    d = hashlib.sha256((salt + base).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def split_of_with_salt(arxiv_id: str, salt: str) -> Split:
    u = u01(base_id(arxiv_id), salt)
    if u < _BENCH_END:
        return "benchmark"
    if u < _DEV_END:
        return "dev"
    return "train"


@dataclass(frozen=True)
class FrozenSplit:
    salt: str
    rule_version: str
    benchmark: frozenset[str]
    quarantine: frozenset[str]
    digest: str
    rule: dict


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_lines(p: Path) -> frozenset[str]:
    return frozenset(line for line in p.read_text(encoding="utf-8").splitlines() if line)


def load_frozen(splits_dir: Path, expected_digest: str | None) -> FrozenSplit:
    sums_path = splits_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise SplitNotFrozenError(f"{sums_path} not found")
    sums_bytes = sums_path.read_bytes()
    digest = _sha256_bytes(sums_bytes)
    if expected_digest is not None and digest != expected_digest:
        raise SplitIntegrityError(
            f"SHA256SUMS digest {digest} does not match the pinned {expected_digest}"
        )
    listed: dict[str, str] = {}
    for line in sums_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        h, _, name = line.partition("  ")
        listed[name] = h
    if set(listed) != set(_FILES):
        raise SplitIntegrityError(f"SHA256SUMS lists {sorted(listed)}, expected {sorted(_FILES)}")
    for name, h in listed.items():
        f = splits_dir / name
        if not f.is_file() or _sha256_bytes(f.read_bytes()) != h:
            raise SplitIntegrityError(f"{name} does not match SHA256SUMS")
    rule = json.loads((splits_dir / "SPLIT_RULE.json").read_text(encoding="utf-8"))
    return FrozenSplit(
        salt=rule["salt"],
        rule_version=rule["rule_version"],
        benchmark=_read_lines(splits_dir / "benchmark_ids.txt"),
        quarantine=_read_lines(splits_dir / "quarantine_ids.txt"),
        digest=digest,
        rule=rule,
    )


@functools.cache
def _load_pinned(digest: str) -> FrozenSplit:
    return load_frozen(paths.splits_dir(), digest)


def frozen() -> FrozenSplit:
    digest = _frozen.FROZEN_SPLIT_DIGEST
    if digest is None:
        raise SplitNotFrozenError("split not frozen: _frozen.FROZEN_SPLIT_DIGEST is None")
    return _load_pinned(digest)


def rule_info() -> dict:
    return {"split_rule_version": SPLIT_RULE_VERSION, "split_digest": _frozen.FROZEN_SPLIT_DIGEST}


def split_of(arxiv_id: str, fs: FrozenSplit | None = None) -> str:
    fs = fs or frozen()
    return split_of_with_salt(arxiv_id, fs.salt)


def is_benchmark(arxiv_id: str, fs: FrozenSplit | None = None) -> bool:
    fs = fs or frozen()
    b = base_id(arxiv_id)
    return b in fs.benchmark or split_of_with_salt(b, fs.salt) == "benchmark"


def _offenders(ids: Iterable[str], fs: FrozenSplit, *, check_quarantine: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in ids:
        try:
            b = base_id(raw)
        except ValueError:
            out[str(raw)] = "invalid-id"
            continue
        if b in fs.benchmark:
            out[b] = "benchmark-list"
        elif split_of_with_salt(b, fs.salt) == "benchmark":
            out[b] = "benchmark-rule"
        elif check_quarantine and b in fs.quarantine:
            out[b] = "quarantine"
    return out


def _raise(offenders: dict[str, str], context: str) -> None:
    shown = list(offenders.items())[:10]
    detail = ", ".join(f"{k} ({v})" for k, v in shown)
    raise LeakageError(
        f"{context}: {len(offenders)} id(s) not allowed: {detail}"
        + (" ..." if len(offenders) > 10 else ""),
        offenders,
    )


def assert_disjoint(ids: Iterable[str], *, context: str, fs: FrozenSplit | None = None) -> None:
    fs = fs or frozen()
    off = _offenders(ids, fs, check_quarantine=True)
    if off:
        _raise(off, context)


def _filter(ids: Iterable[str], fs: FrozenSplit, want: str) -> list[str]:
    out = set()
    for raw in ids:
        b = base_id(raw)
        if split_of_with_salt(b, fs.salt) == want and b not in fs.benchmark and b not in fs.quarantine:
            out.add(b)
    return sorted(out)


def load_train_ids(ids: Iterable[str], fs: FrozenSplit | None = None) -> list[str]:
    fs = fs or frozen()
    result = _filter(ids, fs, "train")
    assert_disjoint(result, context="load_train_ids", fs=fs)
    return result


def load_dev_ids(ids: Iterable[str], fs: FrozenSplit | None = None) -> list[str]:
    fs = fs or frozen()
    result = _filter(ids, fs, "dev")
    off = _offenders(result, fs, check_quarantine=False)
    if off:
        _raise(off, "load_dev_ids")
    return result


def _read_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _compute(rows: list[dict], salt: str, max_authors: int) -> tuple[list[str], list[str], dict]:
    invalid = 0
    by_base: dict[str, dict] = {}
    for r in rows:
        aid = str(r.get("arxiv_id", "")).strip()
        if not is_arxiv_id(aid):
            invalid += 1
            continue
        by_base.setdefault(base_id(aid), {"arxiv_id": base_id(aid), "authors": r.get("authors") or []})
    counts = {"benchmark": 0, "dev": 0, "train": 0}
    bench: set[str] = set()
    for b in by_base:
        s = split_of_with_salt(b, salt)
        counts[s] += 1
        if s == "benchmark":
            bench.add(b)
    quar, qstats = quarantine(by_base.values(), bench, max_authors=max_authors)
    pool = counts["dev"] + counts["train"]
    stats = {
        "counts": counts,
        "candidates_unique": len(by_base),
        "invalid_ids": invalid,
        "quarantine_count": len(quar),
        "quarantine_fraction_of_train_dev": (len(quar) / pool) if pool else 0.0,
        "top_quarantine_authors": qstats["top_names"],
        "skipped_large_author_papers": qstats["skipped_large_author_papers"],
    }
    return sorted(bench), sorted(quar), stats


def _json(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def freeze(
    candidates_path: Path, out_dir: Path, *, salt: str | None = None, max_authors: int = 50
) -> dict:
    out_dir = Path(out_dir)
    if (out_dir / "benchmark_ids.txt").exists():
        raise FileExistsError(f"{out_dir / 'benchmark_ids.txt'} exists; the benchmark never changes")
    salt = salt if salt is not None else secrets.token_hex(8)
    cand_bytes = Path(candidates_path).read_bytes()
    rows = [json.loads(line) for line in cand_bytes.decode("utf-8").splitlines() if line.strip()]
    bench, quar, stats = _compute(rows, salt, max_authors)

    rule = {
        "rule_version": SPLIT_RULE_VERSION,
        "salt": salt,
        "fractions": FRACTIONS,
        "ranges": {"benchmark": [0.0, _BENCH_END], "dev": [_BENCH_END, _DEV_END], "train": [_DEV_END, 1.0]},
        "hash": HASH_DESCRIPTION,
        "author_rule": {**AUTHOR_RULE, "max_authors_per_paper": max_authors},
        "candidates_sha256": _sha256_bytes(cand_bytes),
        "candidates_count": stats["candidates_unique"],
        "created": dt.date.today().isoformat(),
    }
    contents = {
        "benchmark_ids.txt": "".join(f"{i}\n" for i in bench),
        "quarantine_ids.txt": "".join(f"{i}\n" for i in quar),
        "SPLIT_STATS.json": _json(stats),
        "SPLIT_RULE.json": _json(rule),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("benchmark_ids.txt", "quarantine_ids.txt", "SPLIT_STATS.json", "SPLIT_RULE.json"):
        (out_dir / name).write_bytes(contents[name].encode("utf-8"))
    sums = "".join(
        f"{_sha256_bytes(contents[n].encode('utf-8'))}  {n}\n" for n in sorted(contents)
    )
    (out_dir / "SHA256SUMS").write_bytes(sums.encode("utf-8"))
    stats["sha256sums_digest"] = _sha256_bytes(sums.encode("utf-8"))
    return stats


def _verify(fs: FrozenSplit, candidates: Path | None) -> list[str]:
    problems = []
    for b in sorted(fs.benchmark):
        if split_of_with_salt(b, fs.salt) != "benchmark":
            problems.append(f"{b}: listed as benchmark but the rule says otherwise")
    if candidates is None or not candidates.is_file():
        print("note: candidates file absent, skipping the recompute of lists")
        return problems
    max_authors = fs.rule.get("author_rule", {}).get("max_authors_per_paper", 50)
    bench, quar, _ = _compute(_read_candidates(candidates), fs.salt, max_authors)
    for label, expected, actual in (
        ("benchmark", set(bench), set(fs.benchmark)),
        ("quarantine", set(quar), set(fs.quarantine)),
    ):
        if expected != actual:
            problems.append(
                f"{label} list differs from recomputation: "
                f"{len(expected - actual)} missing, {len(actual - expected)} extra"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lightonocr_math.splits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--candidates", type=Path, required=True)
    f.add_argument("--out", type=Path, default=None)
    f.add_argument("--salt", default=None)
    f.add_argument("--max-authors", type=int, default=50)
    v = sub.add_parser("verify")
    v.add_argument("--candidates", type=Path, default=None)
    v.add_argument("--dir", type=Path, default=None)
    sub.add_parser("info")
    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        out = args.out or paths.splits_dir()
        stats = freeze(args.candidates, out, salt=args.salt, max_authors=args.max_authors)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    if args.cmd == "verify":
        try:
            fs = load_frozen(args.dir, None) if args.dir else frozen()
        except (SplitNotFrozenError, SplitIntegrityError) as e:
            print(f"FAIL: {e}")
            return 1
        cand = args.candidates or (paths.repo_root() / "data" / "candidates.jsonl")
        problems = _verify(fs, cand)
        for p in problems[:20]:
            print(p)
        print("OK" if not problems else f"FAIL: {len(problems)} problem(s)")
        return 1 if problems else 0
    info = rule_info()
    try:
        fs = frozen()
        info["benchmark_count"] = len(fs.benchmark)
        info["quarantine_count"] = len(fs.quarantine)
    except (SplitNotFrozenError, SplitIntegrityError) as e:
        info["frozen_error"] = str(e)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


# Fail loudly at import, except when run as ``python -m lightonocr_math.splits`` where main()
# reports an integrity failure cleanly instead of dying with a traceback.
if _frozen.FROZEN_SPLIT_DIGEST is not None and __name__ != "__main__":
    frozen()

if __name__ == "__main__":
    sys.exit(main())
