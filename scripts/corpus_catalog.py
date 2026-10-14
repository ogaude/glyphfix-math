"""Build the enriched, version-pinned catalog of math papers from OAI-PMH arXivRaw (metadata only).

Usage (from repo root):
  python scripts/corpus_catalog.py build --candidates data/candidates.jsonl --days-from data/candidates_cache \
      --cache data/catalog_cache --out-train-dev data/catalog/train_dev.jsonl --out-benchmark data/benchmark/catalog.jsonl \
      --out-benchmark-excluded data/benchmark/catalog_excluded.jsonl --stats data/catalog/CATALOG_STATS.json
  python scripts/corpus_catalog.py recheck --catalog data/catalog/train_dev.jsonl --ids-file ids.txt --out drift.jsonl

Reads data/candidates.jsonl read-only. Requests are at least 3.1 s apart; /e-print and /src are never requested.
Train/dev exclusions are listed in the stats file; benchmark exclusions only in the benchmark-side file, so the
stats file never names a benchmark id.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

from lightonocr_math import splits
from lightonocr_math.corpus import catalog as cat
from lightonocr_math.corpus._io import read_jsonl, write_json, write_jsonl
from lightonocr_math.ids import base_id

FLOOR_MSG = f"--interval must be at least {cat.MIN_INTERVAL_S} s (arXiv asks for at most one request per 3 s)"


def _build(args: argparse.Namespace) -> int:
    if args.interval < cat.MIN_INTERVAL_S:
        print(FLOOR_MSG, file=sys.stderr)
        return 2
    candidates = [r for r in read_jsonl(args.candidates) if r.get("set") == "math"]
    client = cat.OaiClient(args.cache, min_interval_s=args.interval)
    raw: dict[str, dict] = {}
    for day in cat.math_days(args.days_from):
        for rec in cat.list_day(client, day):
            raw[base_id(rec["id"])] = rec
        print(f"{day}: {len(raw)} records so far", flush=True)
    n_listrecords = sum(1 for c in candidates if base_id(c["arxiv_id"]) in raw)
    n_getrecord = 0
    for c in candidates:
        aid = base_id(c["arxiv_id"])
        if aid not in raw:
            rec = cat.get_record(client, aid)
            if rec is not None:
                raw[aid] = rec
                n_getrecord += 1
    missing = sorted(base_id(c["arxiv_id"]) for c in candidates if base_id(c["arxiv_id"]) not in raw)
    checked_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, excluded = cat.enrich(candidates, raw, checked_at=checked_at)
    n_td, n_b = cat.write_catalog(rows, args.out_train_dev, args.out_benchmark)

    def is_b(aid: str) -> bool:
        return splits.is_benchmark(aid)

    write_jsonl(
        args.out_benchmark_excluded,
        [e for e in excluded if is_b(e["arxiv_id"])] + [{"arxiv_id": m, "status": "missing"} for m in missing if is_b(m)],
    )
    prim = Counter(r["primary_category"].split(".")[0] for r in rows)
    stats = {
        "n_math_candidates": len(candidates),
        "n_found_listrecords": n_listrecords,
        "n_found_getrecord": n_getrecord,
        "n_missing": len(missing),
        "n_license_excluded": len(excluded),
        "n_catalog_train_dev": n_td,
        "n_catalog_benchmark": n_b,
        "per_split": dict(Counter(r["split"] for r in rows)),
        "per_license_spdx": dict(Counter(r["license_spdx"] for r in rows)),
        "primary_category_share": {k: round(v / len(rows), 4) for k, v in prim.most_common()} if rows else {},
        "n_versions_histogram": {str(k): v for k, v in sorted(Counter(r["n_versions"] for r in rows).items())},
        "excluded_train_dev": [e for e in excluded if not is_b(e["arxiv_id"])],
        "missing_train_dev": [m for m in missing if not is_b(m)],
        "scope_note": "D-04: catalog restricted to the OAI math set (primary-category mix in primary_category_share); no field-mix broadening",
        "created": checked_at,
    }
    write_json(args.stats, stats)
    print(f"catalog: {n_td} train/dev, {n_b} benchmark, {len(excluded)} license-excluded, {len(missing)} missing; {client.requests} requests")
    return 0


def _recheck(args: argparse.Namespace) -> int:
    if args.interval < cat.MIN_INTERVAL_S:
        print(FLOOR_MSG, file=sys.stderr)
        return 2
    rows = [r for p in args.catalog for r in read_jsonl(p)]
    if args.ids_file:
        want = {base_id(line.strip()) for line in Path(args.ids_file).read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r["arxiv_id"] in want]
    client = cat.OaiClient(args.cache, min_interval_s=args.interval)
    days = cat.math_days(args.days_from) if args.days_from else None
    drift = cat.recheck(rows, client, days)
    write_jsonl(args.out, drift)
    print(f"rechecked {len(rows)} rows, {len(drift)} drifted; {client.requests} requests")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--candidates", default="data/candidates.jsonl")
    b.add_argument("--days-from", default="data/candidates_cache")
    b.add_argument("--cache", default="data/catalog_cache")
    b.add_argument("--out-train-dev", default="data/catalog/train_dev.jsonl")
    b.add_argument("--out-benchmark", default="data/benchmark/catalog.jsonl")
    b.add_argument("--out-benchmark-excluded", default="data/benchmark/catalog_excluded.jsonl")
    b.add_argument("--stats", default="data/catalog/CATALOG_STATS.json")
    b.add_argument("--interval", type=float, default=cat.MIN_INTERVAL_S)
    r = sub.add_parser("recheck")
    r.add_argument("--catalog", action="append", required=True)
    r.add_argument("--ids-file")
    r.add_argument("--days-from", default=None)
    r.add_argument("--cache", default="data/catalog_cache")
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=cat.MIN_INTERVAL_S)
    args = ap.parse_args(argv)
    return _build(args) if args.cmd == "build" else _recheck(args)


if __name__ == "__main__":
    raise SystemExit(main())
