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
Recheck: --drift and --drift-benchmark (the output of the license/version recheck) are both required, or pass
--skip-recheck explicitly; the manifests record which was used. Every loaded shard record goes through
records.validate_record (id, license_spdx) and any problem stops the run before anything is written.
On a side-scan failure the corpus-side files written by this run are removed again.

Frozen benchmark: when data/benchmark/BENCHMARK_FREEZE.json exists, its pinned digests are verified first and the
benchmark-side records, manifest and exclusions are never rewritten. A selection that would differ from the frozen
records is refused with exit 1 and nothing is written (a changed benchmark needs a new benchmark_version, a new root
and a new stock baseline).
Exit codes: 0 ok, 1 leak, side-scan or freeze failure, 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glyphfix_math import paths, splits
from glyphfix_math.corpus import freeze as FZ
from glyphfix_math.corpus import gate as G
from glyphfix_math.corpus import select as S
from glyphfix_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from glyphfix_math.corpus.flags import FLAG_RULES_VERSION
from glyphfix_math.corpus.records import BUILDER_VERSION, Exclusion, exclusion_row, side_root, validate_record
from glyphfix_math.ids import base_id


def _load_side(root: Path) -> tuple[dict[str, list[dict]], list[str], dict[str, dict]]:
    by_paper: dict[str, list[dict]] = {}
    shard_hashes: list[str] = []
    for p in sorted((root / "shards" / "records").glob("*.jsonl")):
        recs = read_jsonl(p)
        shard_hashes.append(f"{p.relative_to(root).as_posix()} {sha256_file(p)}")
        for r in recs:
            problems = validate_record(r)
            if problems:
                raise splits.LeakageError(f"invalid record {r.get('id')!r} in {p.name}: {'; '.join(problems)}", {})
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


def _recheck_info(args: argparse.Namespace) -> dict:
    import datetime

    info: dict = {"skipped": bool(args.skip_recheck), "files": {}}
    for label, path in (("corpus", args.drift), ("benchmark", args.drift_benchmark)):
        if path and Path(path).exists():
            mtime = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime, datetime.UTC).strftime("%Y-%m-%d")
            info["files"][label] = {"path": str(path), "sha256": sha256_file(path), "date": mtime}
    return info


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
    if not args.skip_recheck and not (args.drift and args.drift_benchmark and Path(args.drift).exists() and Path(args.drift_benchmark).exists()):
        print("missing input: --drift and --drift-benchmark (license/version recheck output), or pass --skip-recheck", file=sys.stderr)
        return 2
    for need in (corpus_root / "fetch_order.json", bench_root / "fetch_order.json"):
        if not need.exists():
            print(f"missing input: {need}", file=sys.stderr)
            return 2

    # is_frozen also holds for the repo benchmark and for any root holding the pinned records, so deleting the freeze file
    # cannot unfreeze it: assert_benchmark_frozen then fails on the missing file and nothing is rewritten
    frozen = FZ.is_frozen(bench_root)
    if frozen:
        try:
            FZ.assert_benchmark_frozen(bench_root)
        except splits.LeakageError as e:
            print(f"FROZEN: {e}", file=sys.stderr)
            return 1

    try:
        corpus_by_paper, corpus_hashes, corpus_sum = _load_side(corpus_root)
        bench_by_paper, bench_hashes, bench_sum = _load_side(bench_root)
    except splits.LeakageError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
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

    try:
        res = G.apply_gate(tsel.selected + dsel.selected, bsel.selected, catalog_rows)
        G.assert_clean(res.kept, bsel.selected, catalog_rows)
    except splits.LeakageError as e:
        print(f"LEAK: {e}", file=sys.stderr)
        return 1

    if frozen and FZ.records_text_sha256(bsel.selected) != json.loads((bench_root / FZ.FREEZE_NAME).read_text(encoding="utf-8"))["records_sha256"]:
        print("FROZEN: the benchmark selection differs from the frozen records.jsonl; nothing written", file=sys.stderr)
        return 1

    train = [r for r in res.kept if r["split"] == "train"]
    dev = [r for r in res.kept if r["split"] == "dev"]
    corpus_excluded = drift_rows_c + tsel.dropped + dsel.dropped + res.dropped
    bench_excluded = drift_rows_b + bsel.dropped

    leak_counts = {c: res.report["checks"][c]["count"] for c in G.CHECK_ORDER}
    base_manifest = {
        "config": {"seed": cfg.seed, "cap": cfg.cap, "quotas": cfg.quotas, "min_papers_per_flag": cfg.min_papers_per_flag},
        "seed": cfg.seed, "recheck": _recheck_info(args),
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

    written = [
        corpus_root / "train" / "records.jsonl", corpus_root / "dev" / "records.jsonl",
        corpus_root / "selection_manifest.json", corpus_root / "excluded_selection.jsonl",
    ]
    write_jsonl(written[0], train)
    write_jsonl(written[1], dev)
    write_json(written[2], corpus_manifest)
    write_jsonl(written[3], corpus_excluded)
    if not frozen:
        write_jsonl(bench_root / "records.jsonl", bsel.selected)
        write_json(bench_root / "selection_manifest.json", bench_manifest)
        write_jsonl(bench_root / "excluded_selection.jsonl", bench_excluded)

    try:
        G.assert_side_clean(corpus_root)
    except splits.LeakageError as e:
        for f in written:
            f.unlink(missing_ok=True)  # never leave offending manifests on disk
        print(f"LEAK: {e}", file=sys.stderr)
        return 1
    write_json(bench_root / "leak_report.json", res.report)  # only after the side scan, so a failed run leaves no passing-looking report
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
    r.add_argument("--skip-recheck", action="store_true", help="run without a license/version recheck (recorded in the manifests)")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
