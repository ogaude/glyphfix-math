"""Repo leak scanner: finds benchmark paper ids in tracked and untracked data files.

Usage:
  python -m lightonocr_math.leakscan          # exit 1 and print path:where: id (reason) per leak

What is scanned: every file reported by `git ls-files` plus `git ls-files -o --exclude-standard`
(gitignored files are out of scope), minus ALLOW_PREFIXES (directories) and ALLOW_FILES (exact paths).
  * path segments of every file: new-style ids (2301.01234, optional vN) and old-style safe
    ids (math_0508005, converted back to math/0508005)
  * .jsonl and .json: values of the keys in ID_KEYS (strings or lists of strings), recursively
  * .csv and .tsv: columns named in ID_KEYS
  * .txt: lines that are exactly an arXiv id
  * .parquet and .arrow: columns named in ID_KEYS (needs pyarrow; a missing pyarrow is an error, never a silent skip)
Each candidate goes through ids.is_arxiv_id, ids.base_id and splits.is_benchmark, so slash,
safe_id and vN forms are all caught. Reason is "benchmark-list" when the id is in the frozen
list, "benchmark-rule" when only the hash rule says benchmark.

Allowed paths, and why (candidates.jsonl is an exact match: candidates.jsonl.bak is scanned):
  data/splits/               the split artefacts legitimately hold benchmark ids
  data/candidates.jsonl      the candidate population the split was computed from
  data/candidates_cache/     raw metadata harvest for the same population
  .planning/                 prose documents
  data/benchmark/            benchmark-side corpus artifacts (catalog, records, pages, baseline); training code never reads it

Quarantined ids are not reported: quarantine is a training-side filter enforced by
splits.load_train_ids, not a leak.

Known limit: only id-shaped values under known keys and path names are inspected, so a
benchmark id embedded in free text or under another key is missed; JSON documents larger
than 200 MB are skipped with a warning (JSONL and CSV are streamed).
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lightonocr_math import paths
from lightonocr_math.ids import ARXIV_ID_RE, base_id, from_safe_id, is_arxiv_id
from lightonocr_math.splits import FrozenSplit, frozen, is_benchmark

SCAN_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet", ".arrow"}
# Directory prefixes (startswith) and exact files (==): "data/candidates.jsonl.bak" must not inherit the exemption.
ALLOW_PREFIXES = ("data/splits/", "data/candidates_cache/", ".planning/", "data/benchmark/")
ALLOW_FILES = ("data/candidates.jsonl",)
ID_KEYS = {"arxiv_id", "paper_id", "paper", "source_id", "arxiv"}
MAX_JSON_BYTES = 200 * 1024 * 1024

_NEW_RE = re.compile(r"(?<![0-9A-Za-z.])\d{4}\.\d{4,5}(?:v\d+)?(?![0-9])")
_OLD_SAFE_RE = re.compile(r"(?<![A-Za-z0-9_\-])[a-z][a-z\-]*(?:\.[A-Z]{2})?_\d{7}(?![0-9])")


@dataclass
class Leak:
    path: str
    where: str
    arxiv_id: str
    reason: str


def _git_files(root: Path, *args: str) -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *args],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return [p for p in r.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def candidate_files(root: Path) -> list[Path]:
    rels = set(_git_files(root)) | set(_git_files(root, "-o", "--exclude-standard"))
    out = []
    for rel in sorted(rels):
        if rel.startswith(ALLOW_PREFIXES) or rel in ALLOW_FILES:
            continue
        p = root / rel
        if p.is_file() and not p.is_symlink():
            out.append(p)
    return out


def _as_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.lower().startswith("arxiv:"):
        v = v[6:].strip()
    if is_arxiv_id(v):
        return v
    alt = from_safe_id(v)
    if alt != v and is_arxiv_id(alt):
        return alt
    return None


def _check(value: object, fs: FrozenSplit) -> tuple[str, str] | None:
    v = _as_id(value)
    if v is None:
        return None
    b = base_id(v)
    if not is_benchmark(b, fs):
        return None
    return b, "benchmark-list" if b in fs.benchmark else "benchmark-rule"


def _path_leaks(rel: str, fs: FrozenSplit) -> list[Leak]:
    leaks: dict[str, Leak] = {}
    for seg in rel.split("/"):
        cands = _NEW_RE.findall(seg) + [from_safe_id(m) for m in _OLD_SAFE_RE.findall(seg)]
        for c in cands:
            hit = _check(c, fs)
            if hit and hit[0] not in leaks:
                leaks[hit[0]] = Leak(rel, "path", hit[0], hit[1])
    return list(leaks.values())


def _walk(obj: object, fs: FrozenSplit, rel: str, where: str, out: list[Leak]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ID_KEYS:
                values = v if isinstance(v, list) else [v]
                for item in values:
                    hit = _check(item, fs)
                    if hit:
                        out.append(Leak(rel, f"{where}:{k}", hit[0], hit[1]))
            if isinstance(v, dict | list):
                _walk(v, fs, rel, where, out)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict | list):
                _walk(item, fs, rel, where, out)


def _scan_delimited(path: Path, fs: FrozenSplit, rel: str, delimiter: str) -> list[Leak]:
    out: list[Leak] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for n, row in enumerate(reader, start=2):
            for k, v in row.items():
                if k in ID_KEYS:
                    hit = _check(v, fs)
                    if hit:
                        out.append(Leak(rel, f"line {n}:{k}", hit[0], hit[1]))
    return out


def _scan_arrow(path: Path, fs: FrozenSplit, rel: str) -> list[Leak]:
    """Parquet / Arrow IPC: only the ID_KEYS columns are read. Fails closed when pyarrow is missing."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(f"cannot scan {rel}: pyarrow is not installed (install it, or move the file out of the repo)") from e
    out: list[Leak] = []
    if path.suffix.lower() == ".parquet":
        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if c in ID_KEYS]
        batches = pf.iter_batches(columns=cols) if cols else []
    else:
        with pa.memory_map(str(path)) as src:
            reader = pa.ipc.open_file(src)
            batches = [reader.get_batch(i).select([c for c in reader.schema.names if c in ID_KEYS]) for i in range(reader.num_record_batches)]
    row0 = 0
    for batch in batches:
        for name in batch.schema.names:
            for i, value in enumerate(batch.column(name).to_pylist()):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    hit = _check(item, fs)
                    if hit:
                        out.append(Leak(rel, f"row {row0 + i}:{name}", hit[0], hit[1]))
        row0 += batch.num_rows
    return out


def scan_file(path: Path, fs: FrozenSplit, rel: str) -> list[Leak]:
    leaks = _path_leaks(rel, fs)
    suffix = path.suffix.lower()
    if suffix not in SCAN_SUFFIXES:
        return leaks
    if suffix in {".parquet", ".arrow"}:
        leaks += _scan_arrow(path, fs, rel)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                _walk(obj, fs, rel, f"line {n}", leaks)
    elif suffix == ".json":
        if path.stat().st_size > MAX_JSON_BYTES:
            print(f"warning: {rel} skipped (larger than 200 MB)", file=sys.stderr)
        else:
            try:
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                obj = None
            _walk(obj, fs, rel, "json", leaks)
    elif suffix in {".csv", ".tsv"}:
        leaks += _scan_delimited(path, fs, rel, "\t" if suffix == ".tsv" else ",")
    elif suffix == ".txt":
        with path.open(encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, start=1):
                s = line.strip()
                if ARXIV_ID_RE.match(s):
                    hit = _check(s, fs)
                    if hit:
                        leaks.append(Leak(rel, f"line {n}", hit[0], hit[1]))
    return leaks


def scan_repo(root: Path, fs: FrozenSplit) -> list[Leak]:
    out: list[Leak] = []
    for p in candidate_files(root):
        out += scan_file(p, fs, p.relative_to(root).as_posix())
    return out


def main() -> int:
    leaks = scan_repo(paths.repo_root(), frozen())
    for lk in leaks:
        print(f"{lk.path}:{lk.where}: {lk.arxiv_id} ({lk.reason})")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
