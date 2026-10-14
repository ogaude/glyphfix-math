"""Register a seeded paper order and fetch version-pinned e-print bundles, one paced request at a time.

Usage (from repo root):
  python scripts/corpus_fetch.py order --catalog data/catalog/train_dev.jsonl --out data/corpus/fetch_order.json --seed 0
  python scripts/corpus_fetch.py order --catalog data/benchmark/catalog.jsonl --out data/benchmark/fetch_order.json --seed 0
  python scripts/corpus_fetch.py fetch --order data/corpus/fetch_order.json --catalog data/catalog/train_dev.jsonl \
      --start 0 --count 100 --sources data/sources --interval 15 --policy-ack "<decision text>"

The order file is written before anything is fetched. The pilot is the first 100 ids of the corpus-side order.
fetch requires --policy-ack (robots.txt disallows /e-print and /src) and refuses --interval below 3.1 s.
Exit codes: 0 ok, 2 refused before any request, 3 arXiv answered 403, 4 too many consecutive failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lightonocr_math.corpus import fetch as F
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json
from lightonocr_math.corpus.records import LICENSE_SPDX
from lightonocr_math.ids import base_id, is_arxiv_id


def _eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("license_spdx") in LICENSE_SPDX and not r.get("quarantined")]


def _order(args: argparse.Namespace) -> int:
    rows = _eligible(read_jsonl(args.catalog))
    ids = F.fetch_order([r["arxiv_id"] for r in rows], args.seed)
    out = Path(args.out)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("ids") != ids:
            print(f"refusing to overwrite {out}: a different order is already registered", file=sys.stderr)
            return 2
        print(f"{out} already registered ({len(ids)} ids)")
        return 0
    write_json(out, {
        "seed": args.seed, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "catalog_sha256": sha256_file(args.catalog), "ids": ids,
    })
    print(f"registered {len(ids)} ids in {out}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    if not args.policy_ack or not args.policy_ack.strip():
        print("--policy-ack is required: robots.txt disallows /e-print and /src; name the decision to proceed", file=sys.stderr)
        return 2
    if args.interval < F.MIN_INTERVAL_S:
        print(f"--interval must be at least {F.MIN_INTERVAL_S} s (arXiv asks for at most one request per 3 s)", file=sys.stderr)
        return 2
    catalog_rows: dict[str, dict] = {}
    for p in args.catalog:
        for r in read_jsonl(p):
            catalog_rows[base_id(r["arxiv_id"])] = r
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))["ids"]
    chosen = order[args.start : args.start + args.count]
    for i in chosen:
        r = catalog_rows.get(i)
        if not is_arxiv_id(i) or r is None:
            print(f"refused: {i} is not in the given catalogs", file=sys.stderr)
            return 2
        if r.get("quarantined") or r.get("license_spdx") not in LICENSE_SPDX:
            print(f"refused: {i} is quarantined or not license-whitelisted", file=sys.stderr)
            return 2
    sources = Path(args.sources)
    sources.mkdir(parents=True, exist_ok=True)
    log = sources / "FETCH_LOG.jsonl"
    limiter = F.Limiter(args.interval)
    consecutive = 0
    for n, i in enumerate(chosen, 1):
        t0 = time.monotonic()
        try:
            meta = F.fetch_one(catalog_rows[i], sources, limiter=limiter, policy_ack=args.policy_ack)
            consecutive = 0 if meta["status"] != "too_large" else consecutive
        except F.FetchForbidden as e:
            print(f"STOP: {e}", file=sys.stderr)
            return 3
        except OSError as e:
            consecutive += 1
            meta = {"status": "error", "http_status": getattr(e, "code", None)}
            print(f"[{n}/{len(chosen)}] {i}: error {e!r}", file=sys.stderr)
            if consecutive >= F.MAX_CONSECUTIVE_FAILURES:
                print(f"STOP: {consecutive} consecutive failures", file=sys.stderr)
                return 4
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": i, "status": meta["status"], "http_status": meta.get("http_status"),
                                 "seconds": round(time.monotonic() - t0, 2)}) + "\n")
        print(f"[{n}/{len(chosen)}] {i}: {meta['status']} {meta.get('kind', '')} {meta.get('bytes', 0)} B", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("--catalog", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fetch")
    f.add_argument("--order", required=True)
    f.add_argument("--catalog", action="append", required=True)
    f.add_argument("--start", type=int, default=0)
    f.add_argument("--count", type=int, default=100)
    f.add_argument("--sources", default="data/sources")
    f.add_argument("--interval", type=float, default=F.DEFAULT_INTERVAL_S)
    f.add_argument("--policy-ack", default="")
    args = ap.parse_args(argv)
    return _order(args) if args.cmd == "order" else _fetch(args)


if __name__ == "__main__":
    sys.exit(main())
