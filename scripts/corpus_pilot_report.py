"""Write the pilot report, its JSON twin and the 30-pair contact sheet from built train/dev pilot shards.

Usage (from repo root):
  python scripts/corpus_pilot_report.py --order data/corpus/fetch_order.json --count 100 \
      --shards data/corpus/shards --fetch-log data/sources/FETCH_LOG.jsonl \
      [--crop-scale <json from corpus_pages crop-scale>] \
      --out-dir .planning/phases/03-corpus-and-benchmark/reports

Reads only train/dev pilot ids (a benchmark id in the first --count ids aborts with exit 2). Writes PILOT_REPORT.md,
pilot_report.json and pilot_contact_sheet.html into --out-dir. The contact sheet embeds label text from papers and is
escaped; open it in a browser. Exit codes: 0 ok, 2 refused or bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightonocr_math import splits
from lightonocr_math.corpus import pilot
from lightonocr_math.corpus._io import atomic_write_text, read_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--order", required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--shards", default="data/corpus/shards")
    ap.add_argument("--fetch-log", default=None)
    ap.add_argument("--crop-scale", default=None)
    ap.add_argument("--out-dir", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    ids = json.loads(Path(args.order).read_text(encoding="utf-8"))["ids"][: args.count]
    try:
        splits.assert_disjoint(ids, context="pilot report")
        summaries, records, excluded = pilot.load_shards(Path(args.shards), ids)
    except splits.LeakageError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    m = pilot.pilot_metrics(Path(args.shards), ids)
    m["audit"] = pilot.automated_audit(records, excluded, summaries=summaries)
    m["near_duplicates"] = pilot.near_duplicate_profile(records)
    if args.crop_scale:
        m["crop_scale"] = json.loads(Path(args.crop_scale).read_text(encoding="utf-8"))
    if args.fetch_log and Path(args.fetch_log).is_file():
        log = read_jsonl(args.fetch_log)
        in_scope = [r for r in log if r["id"] in set(ids)]
        by: dict[str, int] = {}
        for r in in_scope:
            by[r["status"]] = by.get(r["status"], 0) + 1
        m["fetch"] = {"logged": len(in_scope), "by_status": dict(sorted(by.items()))}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pilot.render_contact_sheet(pilot.contact_sheet_sample(records), out / "pilot_contact_sheet.html")
    write_json(out / "pilot_report.json", m)
    atomic_write_text(out / "PILOT_REPORT.md", pilot.render_report(m))
    print(f"wrote {out / 'PILOT_REPORT.md'}, pilot_report.json and pilot_contact_sheet.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
