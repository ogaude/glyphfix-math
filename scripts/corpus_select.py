"""Compose the train, dev and benchmark manifests from built shards and run the corpus leak gate.

Usage (from repo root):
  python scripts/corpus_select.py run --config data/corpus/SELECTION_CONFIG.json \
      [--drift data/corpus/drift.jsonl] [--drift-benchmark data/benchmark/drift.jsonl]

Reads <side>/shards/{records,excluded,papers} for both sides, drops papers listed in the drift files, applies the
per-paper cap, quotas and notation floors, removes any overlap with the benchmark from train/dev (corpus.gate),
and writes the manifests only after the gate passes:
  data/corpus/{train,dev}/records.jsonl, data/corpus/selection_manifest.json, data/corpus/excluded_selection.jsonl
  data/benchmark/records.jsonl, data/benchmark/selection_manifest.json, data/benchmark/excluded_selection.jsonl
  data/benchmark/leak_report.json   (all offender detail: it names benchmark ids, so it never goes on the corpus side)
Exit codes: 0 ok, 1 leak or side-scan failure, 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightonocr_math import paths, splits
from lightonocr_math.corpus import gate as G
from lightonocr_math.corpus import select as S
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from lightonocr_math.corpus.flags import FLAG_RULES_VERSION
from lightonocr_math.corpus.records import BUILDER_VERSION, Exclusion, exclusion_row, side_root
from lightonocr_math.ids import base_id


def _load_side(root: Path) -> tuple[dict[str, list[dict]], list[str], dict[str, dict]]:
    by_paper: dict[str, list[dict]] = {}
    shard_hashes: list[str] = []
    for p in sorted((root / "shards" / "records").glob("*.jsonl")):
        recs = read_jsonl(p)
        shard_hashes.append(f"{p.relative_to(root).as_posix()} {sha256_file(p)}")
        for r in recs:
            by_paper.setdefault(base_id(r["arxiv_id"]), []).append(r)
    summaries: dict[str, dict] = {}
    for p in sorted((root / "shards" / "papers").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        summaries[base_id(d["arxiv_id"])] = d
    return by_paper, shard_hashes, summaries


def _drift_rows(path: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if path and Path(path).exists():
        for r in read_jsonl(path):
            reason = r.get("reason") if r.get("reason") in (Exclusion.LICENSE, Exclusion.VERSION_DRIFT) else Exclusion.VERSION_DRIFT
            out[base_id(r["arxiv_id"])] = str(reason)
    return out


def _order(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["ids"]


def _versions(summaries: dict[str, dict]) -> dict:
    return {
        "builder_version": sorted({str(s.get("builder_version")) for s in summaries.values()} or {BUILDER_VERSION}),
        "canon_version": sorted({str(s.get("canon_version")) for s in summaries.values()}),
    }


def run(args: argparse.Namespace) -> int:
    root = paths.repo_root()
    cfg = S.SelectionConfig.from_json(args.config)
    corpus_root, bench_root = side_root("corpus"), side_root("benchmark")
    for need in (corpus_root / "fetch_order.json", bench_root / "fetch_order.json"):
        if not need.exists():
            print(f"missing input: {need}", file=sys.stderr)
            return 2

    corpus_by_paper, corpus_hashes, corpus_sum = _load_side(corpus_root)
    bench_by_paper, bench_hashes, bench_sum = _load_side(bench_root)
    drift_c, drift_b = _drift_rows(args.drift), _drift_rows(args.drift_benchmark)
    drift_rows_c = [exclusion_row(p, r) for p, r in sorted(drift_c.items()) if p in corpus_by_paper]
    drift_rows_b = [exclusion_row(p, r) for p, r in sorted(drift_b.items()) if p in bench_by_paper]
    corpus_by_paper = {p: v for p, v in corpus_by_paper.items() if p not in drift_c}
    bench_by_paper = {p: v for p, v in bench_by_paper.items() if p not in drift_b}

    bsel = S.select_side(
        bench_by_paper, _order(bench_root / "fetch_order.json"), split="benchmark", quota=cfg.quotas["benchmark"],
        floors=cfg.floors.get("benchmark", {}), cap=cfg.cap, seed=cfg.seed,
    )
    corpus_order = _order(corpus_root / "fetch_order.json")
    tsel = S.select_side(
        corpus_by_paper, corpus_order, split="train", quota=cfg.quotas["train"],
        floors=cfg.floors.get("corpus", {}), cap=cfg.cap, seed=cfg.seed,
    )
    dsel = S.select_side(
        corpus_by_paper, corpus_order, split="dev", quota=cfg.quotas["dev"], floors={}, cap=cfg.cap, seed=cfg.seed,
    )

    catalog_rows: list[dict] = []
    for p in (root / "data" / "catalog" / "train_dev.jsonl", bench_root / "catalog.jsonl"):
        if p.exists():
            catalog_rows += read_jsonl(p)

    res = G.apply_gate(tsel.selected + dsel.selected, bsel.selected, catalog_rows)
    try:
        G.assert_clean(res.kept, bsel.selected, catalog_rows)
    except splits.LeakageError as e:
        print(f"LEAK: {e}", file=sys.stderr)
        return 1

    train = [r for r in res.kept if r["split"] == "train"]
    dev = [r for r in res.kept if r["split"] == "dev"]
    corpus_excluded = drift_rows_c + tsel.dropped + dsel.dropped + res.dropped
    bench_excluded = drift_rows_b + bsel.dropped

    leak_counts = {c: res.report["checks"][c]["count"] for c in G.CHECK_ORDER}
    base_manifest = {
        "config": {"seed": cfg.seed, "cap": cfg.cap, "quotas": cfg.quotas, "min_papers_per_flag": cfg.min_papers_per_flag},
        "seed": cfg.seed,
        "select_version": S.SELECT_VERSION, "gate_version": G.GATE_VERSION, "flag_rules_version": FLAG_RULES_VERSION,
    }
    corpus_manifest = {
        **base_manifest, **_versions(corpus_sum),
        "floors": cfg.floors.get("corpus", {}),
        "floor_report": S.floor_report(train, cfg.floors.get("corpus", {}), cfg.min_papers_per_flag.get("corpus", 0)),
        "stats": {"train": tsel.stats, "dev": dsel.stats, "gate_drops_by_check": leak_counts},
        "input_shards": corpus_hashes, "n_train": len(train), "n_dev": len(dev),
    }
    bench_manifest = {
        **base_manifest, **_versions(bench_sum),
        "floors": cfg.floors.get("benchmark", {}),
        "floor_report": S.floor_report(bsel.selected, cfg.floors.get("benchmark", {}), cfg.min_papers_per_flag.get("benchmark", 0)),
        "stats": bsel.stats, "input_shards": bench_hashes, "n_records": len(bsel.selected),
    }

    write_jsonl(corpus_root / "train" / "records.jsonl", train)
    write_jsonl(corpus_root / "dev" / "records.jsonl", dev)
    write_json(corpus_root / "selection_manifest.json", corpus_manifest)
    write_jsonl(corpus_root / "excluded_selection.jsonl", corpus_excluded)
    write_jsonl(bench_root / "records.jsonl", bsel.selected)
    write_json(bench_root / "selection_manifest.json", bench_manifest)
    write_jsonl(bench_root / "excluded_selection.jsonl", bench_excluded)
    write_json(bench_root / "leak_report.json", res.report)

    try:
        G.assert_side_clean(corpus_root)
    except splits.LeakageError as e:
        print(f"LEAK: {e}", file=sys.stderr)
        return 1
    short = {f: v for f, v in corpus_manifest["floor_report"].items() if v["short"]}
    short.update({f"benchmark:{f}": v for f, v in bench_manifest["floor_report"].items() if v["short"]})
    print(f"train {len(train)} dev {len(dev)} benchmark {len(bsel.selected)}; gate drops {leak_counts}; short floors {sorted(short)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="data/corpus/SELECTION_CONFIG.json")
    r.add_argument("--drift", default=None)
    r.add_argument("--drift-benchmark", default=None)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
