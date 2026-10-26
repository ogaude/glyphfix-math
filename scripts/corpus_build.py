"""Build corpus shards for an ordered range of fetched papers, resumably, in chunks of a few papers.

Usage (from repo root):
  python scripts/corpus_build.py --order data/corpus/fetch_order.json --start 0 --count 100 \
      --catalog data/catalog/train_dev.jsonl --sources data/sources --chunk-papers 10
  add --rebuild to ignore the resume key, --keep-work to keep data/work/<safe_id>/.

Each chunk runs guard.preflight (4 GB for a Docker job above the 1.5 GB RAM floor) and two batched sandbox renders.
Papers whose shard is up to date (source sha256, builder, canon, flag and preamble versions) are skipped.
After the run, <side_root>/shards/BUILD_SUMMARY.json is recomputed from all papers/*.json of each side touched.
Exit codes: 0 ok, 1 render infrastructure error (shards already written are kept), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glyphfix_math import guard, paths, render
from glyphfix_math.corpus import build
from glyphfix_math.corpus._io import read_jsonl, write_json
from glyphfix_math.corpus.records import BUILDER_VERSION, side_of, side_root
from glyphfix_math.ids import base_id, safe_id


def _summarize(root: Path) -> dict:
    papers = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "shards" / "papers").glob("*.json"))]
    by_status: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for s in papers:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        for k, v in s.get("exclusions", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    return {
        "n_papers": len(papers), "papers_by_status": dict(sorted(by_status.items())),
        "exclusions": dict(sorted(reasons.items())), "n_kept": sum(s.get("n_kept", 0) for s in papers),
        "n_extracted": sum(s.get("n_extracted", 0) for s in papers),
        "seconds_total": round(sum(s.get("seconds", 0.0) for s in papers), 1),
        "builder_version": BUILDER_VERSION, "canon_version": build.canon.CANON_RULES_VERSION,
        "canon_fingerprint": build.canon.CANON_FINGERPRINT, "flag_rules_version": build.flags.FLAG_RULES_VERSION,
        "preamble_sha256": render.PREAMBLE_SHA256, "render_image_digest": render.IMAGE_DIGEST,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--order", required=True, help="fetch_order.json written by corpus_fetch.py order")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--sources", default="data/sources")
    ap.add_argument("--chunk-papers", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.start < 0 or args.count < 0 or args.chunk_papers < 1:
        print("--start and --count must be >= 0 and --chunk-papers >= 1", file=sys.stderr)
        return 2
    rows: dict[str, dict] = {}
    for c in args.catalog:
        for r in read_jsonl(c):
            rows[base_id(r["arxiv_id"])] = r
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))["ids"]
    sources = Path(args.sources)
    work_root = paths.repo_root() / "data" / "work"
    todo: list[build.PaperInput] = []
    sides: set[str] = set()
    for i in order[args.start : args.start + args.count]:
        meta_p = sources / safe_id(i) / "fetch.json"
        if i not in rows or not meta_p.is_file():
            print(f"{i}: skipped (not in the catalogs or not fetched)")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        side = side_of(i)
        summ = side_root(side) / "shards" / "papers" / f"{safe_id(i)}.json"
        if not args.rebuild and build.shard_up_to_date(summ, meta.get("sha256", "")):
            print(f"{i}: up to date")
            sides.add(side)
            continue
        todo.append(build.PaperInput(
            i, rows[i]["version"], rows[i], meta, sources / safe_id(i) / "src.bin", work_root / safe_id(i), keep_work=args.keep_work))
    rc = 0
    for n in range(0, len(todo), args.chunk_papers):
        chunk = todo[n : n + args.chunk_papers]
        try:
            guard.preflight(ram_gb=guard.DOCKER_MEMORY_GB, timeout_s=600)
            results = build.build_chunk(chunk)
        except render.RenderInfrastructureError as e:
            print(f"STOP: render infrastructure error: {e}", file=sys.stderr)
            rc = 1
            break
        for r in results:
            build.write_shards(r)
            sides.add(r.side)
            print(f"{r.arxiv_id}: {r.summary['status']} {r.summary['n_kept']}/{r.summary['n_extracted']}", flush=True)
    for side in sorted(sides):
        root = side_root(side)
        if (root / "shards" / "papers").is_dir():
            write_json(root / "shards" / "BUILD_SUMMARY.json", _summarize(root))
    return rc


if __name__ == "__main__":
    sys.exit(main())
