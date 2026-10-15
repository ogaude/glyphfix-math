"""Corpus-level leak gate between the benchmark and the train/dev side.

Usage:
  from lightonocr_math.corpus.gate import apply_gate, assert_clean, assert_side_clean, regression_overlap
  res = apply_gate(corpus_records, bench_records, catalog_rows)   # res.kept, res.dropped, res.report
  assert_clean(res.kept, bench_records, catalog_rows)             # raises splits.LeakageError
  python -m lightonocr_math.corpus.gate check --corpus data/corpus --benchmark data/benchmark
  python -m lightonocr_math.corpus.gate timing-probe --bench 3000 --train 20000 --out reports/gate_timing.json

Checks (in the order a record is charged with them; thresholds in GateConfig):
  exact            normalize.canonical equal to a benchmark record's, at least exact_min_tokens (8) tokens; shorter
                   collisions are logged under short_collisions and kept
  near             normalized Levenshtein similarity >= near_threshold (0.9) on normalize_tokens, both sides with at
                   least near_min_tokens (12) tokens
  paper_title      paper title token_set_ratio >= title_threshold (90) with a benchmark paper title
  paper_shared_eq  at least shared_eq_paper (5) exact equations shared with one benchmark paper
  same_author      catalog authors share an authors.author_keys entry with any benchmark paper (catalog rows and the
                   frozen candidates file)
  (a missing frozen candidates file raises LeakageError unless allow_missing_candidates is set; report["candidates_used"] says which)
  ids              benchmark or quarantined arxiv ids among corpus records (assert_clean only: a hard fail, never repaired)

Rules:
  * overlap is removed from the corpus side only; benchmark records are never touched
  * dropped rows are exclusion_row(arxiv_id, "leak", id=..., check=...) and nothing else, because they are written
    on the training side; the pairing detail (benchmark ids, scores, titles) is only in GateResult.report
  * the near-duplicate sweep uses a lossless length prefilter: normalized similarity is at most min/max of the two
    lengths, so pairs below the threshold on length alone are never compared

Known limit: the sweep compares every kept corpus equation of an allowed length against each benchmark equation; at
3k x 20k the measured cost is in reports/gate_timing.json.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from lightonocr_math import authors, leakscan, normalize, paths, splits
from lightonocr_math.corpus._io import read_jsonl, write_json
from lightonocr_math.corpus.catalog import split_raw_authors
from lightonocr_math.corpus.records import Exclusion, exclusion_row
from lightonocr_math.ids import base_id

GATE_VERSION = "1"
CHECK_ORDER = ("exact", "near", "paper_title", "paper_shared_eq", "same_author")
_CHUNK = 256


@dataclass
class GateConfig:
    exact_min_tokens: int = 8
    near_min_tokens: int = 12
    near_threshold: float = 0.9
    title_threshold: int = 90
    shared_eq_paper: int = 5


@dataclass
class GateResult:
    kept: list[dict]
    dropped: list[dict]
    report: dict = field(default_factory=dict)


def _default_candidates() -> Path:
    return paths.repo_root() / "data" / "candidates.jsonl"


def _rows_by_id(catalog_rows) -> dict[str, dict]:
    if isinstance(catalog_rows, dict):
        return {base_id(k): v for k, v in catalog_rows.items()}
    return {base_id(r["arxiv_id"]): r for r in catalog_rows}


def _char(idx: int) -> str:
    c = idx + 0x100
    if c >= 0xD800:
        c += 0x800
    return chr(c)


class _Encoder:
    """Maps tokens to single characters so Levenshtein on strings equals Levenshtein on token lists."""

    def __init__(self) -> None:
        self.vocab: dict[str, str] = {}

    def encode(self, tokens: list[str]) -> str:
        out = []
        for t in tokens:
            c = self.vocab.get(t)
            if c is None:
                c = self.vocab[t] = _char(len(self.vocab))
            out.append(c)
        return "".join(out)


def _sweep(
    bench: list[tuple[str, str]], train: list[tuple[str, str]], threshold: float, *, prefilter: bool = True
) -> tuple[list[tuple[str, str, float]], int]:
    """Return ([(bench_key, train_key, score)], pairs_compared); inputs are (key, encoded) pairs."""
    hits: list[tuple[str, str, float]] = []
    pairs = 0
    if not bench or not train:
        return hits, pairs
    train_sorted = sorted(train, key=lambda kv: len(kv[1]))
    train_lens = [len(s) for _, s in train_sorted]
    by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for kv in bench:
        by_len[len(kv[1])].append(kv)
    for blen, queries in by_len.items():
        if prefilter:
            lo = bisect.bisect_left(train_lens, math.ceil(blen * threshold - 1e-9))
            hi = bisect.bisect_right(train_lens, math.floor(blen / threshold + 1e-9))
        else:
            lo, hi = 0, len(train_sorted)
        window = train_sorted[lo:hi]
        if not window:
            continue
        choices = [s for _, s in window]
        for i in range(0, len(queries), _CHUNK):
            chunk = queries[i : i + _CHUNK]
            res = process.cdist(
                [s for _, s in chunk], choices, scorer=Levenshtein.normalized_similarity,
                score_cutoff=threshold, workers=-1, dtype="float32",
            )
            pairs += len(chunk) * len(choices)
            for qi, ci in np.argwhere(res >= threshold):
                hits.append((chunk[qi][0], window[ci][0], float(res[qi, ci])))
    return hits, pairs


def _bench_ids(bench_records, rows: dict[str, dict]) -> set[str]:
    ids = {base_id(r["arxiv_id"]) for r in bench_records}
    ids |= {b for b, row in rows.items() if row.get("split") == "benchmark"}
    return ids


def _candidate_authors(candidates_path: Path, *, allow_missing: bool = False) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not candidates_path.exists():
        if not allow_missing:
            raise splits.LeakageError(
                f"frozen candidates file missing: {candidates_path} (the same-author check would be weakened; "
                "pass allow_missing_candidates / --allow-missing-candidates to run without it)",
                {},
            )
    else:
        for row in read_jsonl(candidates_path):
            try:
                out[base_id(row["arxiv_id"])] = list(row.get("authors") or [])
            except ValueError:
                continue
    return out


def _findings(
    corpus_records, bench_records, catalog_rows, cfg: GateConfig, candidates_path: Path, *, allow_missing_candidates: bool = False
) -> dict:
    """Return {"by_record": {record_id: (check, detail)}, "report": {...}} over every check."""
    rows = _rows_by_id(catalog_rows)
    bench_ids = _bench_ids(bench_records, rows)
    enc = _Encoder()

    bench_info = []
    for r in bench_records:
        toks = normalize.normalize_tokens(r["latex"])
        bench_info.append((r["id"], base_id(r["arxiv_id"]), normalize.canonical(r["latex"]), toks))
    corp_info = []
    for r in corpus_records:
        toks = normalize.normalize_tokens(r["latex"])
        corp_info.append((r["id"], base_id(r["arxiv_id"]), normalize.canonical(r["latex"]), toks))

    by_record: dict[str, tuple[str, dict]] = {}
    report: dict = {
        "gate_version": GATE_VERSION,
        "thresholds": {
            "exact_min_tokens": cfg.exact_min_tokens, "near_min_tokens": cfg.near_min_tokens,
            "near_threshold": cfg.near_threshold, "title_threshold": cfg.title_threshold,
            "shared_eq_paper": cfg.shared_eq_paper,
        },
        "checks": {c: {"count": 0, "offenders": []} for c in CHECK_ORDER},
        "short_collisions": [],
    }

    def charge(check: str, rid: str, detail: dict) -> None:
        report["checks"][check]["offenders"].append({"corpus_id": rid, **detail})
        by_record.setdefault(rid, (check, detail))

    bench_by_canon: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for bid, bpaper, canon, toks in bench_info:
        bench_by_canon[canon].append((bid, bpaper))

    # exact
    shared: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rid, paper, canon, toks in corp_info:
        hit = bench_by_canon.get(canon)
        if not hit:
            continue
        if len(toks) >= cfg.exact_min_tokens:
            charge("exact", rid, {"bench_id": hit[0][0], "bench_paper": hit[0][1]})
            for _, bpaper in hit:
                shared[(paper, bpaper)].add(canon)
        else:
            report["short_collisions"].append({"corpus_id": rid, "bench_id": hit[0][0], "n_tokens": len(toks)})

    # near
    bench_enc = [(bid, enc.encode(t)) for bid, _, _, t in bench_info if len(t) >= cfg.near_min_tokens]
    corp_enc = [(rid, enc.encode(t)) for rid, _, _, t in corp_info if len(t) >= cfg.near_min_tokens]
    hits, pairs = _sweep(bench_enc, corp_enc, cfg.near_threshold)
    report["near_pairs_compared"] = pairs
    best: dict[str, tuple[str, float]] = {}
    for bid, rid, score in hits:
        if rid not in best or score > best[rid][1]:
            best[rid] = (bid, score)
    for rid, (bid, score) in sorted(best.items()):
        charge("near", rid, {"bench_id": bid, "score": round(score, 4)})

    # paper-level: title
    corp_papers = sorted({p for _, p, _, _ in corp_info})
    bench_titles = {b: rows[b].get("title", "") for b in bench_ids if b in rows and rows[b].get("title")}
    paper_flag: dict[str, tuple[str, dict]] = {}
    if bench_titles and corp_papers:
        bkeys = sorted(bench_titles)
        ckeys = [p for p in corp_papers if rows.get(p, {}).get("title")]
        if ckeys:
            res = process.cdist(
                [rows[p]["title"] for p in ckeys], [bench_titles[b] for b in bkeys],
                scorer=fuzz.token_set_ratio, score_cutoff=cfg.title_threshold, dtype="float32",
            )
            for pi, bi in np.argwhere(res >= cfg.title_threshold):
                paper_flag.setdefault(ckeys[pi], ("paper_title", {
                    "bench_paper": bkeys[bi], "score": float(res[pi, bi]), "bench_title": bench_titles[bkeys[bi]],
                    "corpus_title": rows[ckeys[pi]]["title"],
                }))
    # paper-level: shared equations (counted over every exact collision, not only charged ones)
    for (paper, bpaper), eqs in sorted(shared.items()):
        if len(eqs) >= cfg.shared_eq_paper:
            paper_flag.setdefault(paper, ("paper_shared_eq", {"bench_paper": bpaper, "n_shared": len(eqs)}))

    # same author
    cand_auth = _candidate_authors(candidates_path, allow_missing=allow_missing_candidates)
    report["candidates_used"] = Path(candidates_path).exists()
    bench_keys: dict[str, set[str]] = defaultdict(set)
    for b in bench_ids:
        ks = set()
        if b in rows:
            ks |= authors.author_keys(list(rows[b].get("authors") or []))
        if b in cand_auth:
            ks |= authors.author_keys(cand_auth[b])
        for k in ks:
            bench_keys[k].add(b)
    for b, a in cand_auth.items():
        if splits.is_benchmark(b):
            for k in authors.author_keys(a):
                bench_keys[k].add(b)
    author_flag: dict[str, tuple[str, dict]] = {}
    for p in corp_papers:
        row = rows.get(p, {})
        names = list(row.get("authors") or cand_auth.get(p, []))
        keys = authors.author_keys(names)
        if row.get("authors_raw"):
            keys |= authors.author_keys(split_raw_authors(row["authors_raw"]))
        hit = sorted(k for k in keys if k in bench_keys)
        if hit:
            author_flag[p] = ("same_author", {"author_key": hit[0], "bench_paper": min(bench_keys[hit[0]])})

    for rid, paper, _, _ in corp_info:
        if paper in paper_flag:
            charge(paper_flag[paper][0], rid, paper_flag[paper][1])
        if paper in author_flag:
            charge("same_author", rid, author_flag[paper][1])

    for c in CHECK_ORDER:
        report["checks"][c]["count"] = len(report["checks"][c]["offenders"])
    return {"by_record": by_record, "report": report}


def apply_gate(
    corpus_records, bench_records, catalog_rows, *, cfg: GateConfig | None = None, candidates_path: Path | None = None,
    allow_missing_candidates: bool = False,
) -> GateResult:
    cfg = cfg or GateConfig()
    candidates_path = candidates_path or _default_candidates()
    corpus_records = list(corpus_records)
    found = _findings(
        corpus_records, list(bench_records), catalog_rows, cfg, candidates_path, allow_missing_candidates=allow_missing_candidates
    )
    charged: dict[str, str] = {}
    # first check in CHECK_ORDER that charged the record
    for c in CHECK_ORDER:
        for off in found["report"]["checks"][c]["offenders"]:
            charged.setdefault(off["corpus_id"], c)
    kept, dropped = [], []
    for r in corpus_records:
        check = charged.get(r["id"])
        if check is None:
            kept.append(r)
        else:
            dropped.append(exclusion_row(r["arxiv_id"], Exclusion.LEAK, id=r["id"], check=check))
    found["report"]["n_dropped"] = len(dropped)
    found["report"]["n_kept"] = len(kept)
    return GateResult(kept=kept, dropped=dropped, report=found["report"])


def assert_clean(
    corpus_records, bench_records, catalog_rows, *, cfg: GateConfig | None = None, candidates_path: Path | None = None,
    allow_missing_candidates: bool = False,
) -> dict:
    cfg = cfg or GateConfig()
    candidates_path = candidates_path or _default_candidates()
    corpus_records = list(corpus_records)
    splits.assert_disjoint([r["arxiv_id"] for r in corpus_records], context="corpus records")
    found = _findings(
        corpus_records, list(bench_records), catalog_rows, cfg, candidates_path, allow_missing_candidates=allow_missing_candidates
    )
    report = found["report"]
    offenders: dict[str, str] = {}
    for c in CHECK_ORDER:
        for off in report["checks"][c]["offenders"]:
            offenders.setdefault(off["corpus_id"], c)
    if offenders:
        counts = {c: report["checks"][c]["count"] for c in CHECK_ORDER if report["checks"][c]["count"]}
        shown = ", ".join(f"{k} ({v})" for k, v in list(offenders.items())[:10])
        raise splits.LeakageError(
            f"corpus gate: {len(offenders)} corpus record(s) overlap the benchmark {counts}: {shown}"
            + (" ..." if len(offenders) > 10 else ""),
            offenders,
        )
    return report


def assert_side_clean(root: Path) -> None:
    root = Path(root)
    fs = splits.frozen()
    leaks = []
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            p = Path(dirpath, d)
            leaks += leakscan._path_leaks(p.relative_to(root).as_posix(), fs)
        for f in filenames:
            p = Path(dirpath, f)
            leaks += leakscan.scan_file(p, fs, p.relative_to(root).as_posix())
    if leaks:
        offenders = {lk.arxiv_id: f"{lk.path}:{lk.where} ({lk.reason})" for lk in leaks}
        shown = "; ".join(f"{lk.path}:{lk.where} {lk.arxiv_id} ({lk.reason})" for lk in leaks[:10])
        raise splits.LeakageError(f"side scan of {root}: {len(leaks)} leak(s): {shown}", offenders)


def regression_overlap(train_records, regression_records) -> set[str]:
    reg_ids = {r["id"] for r in regression_records}
    reg_canon = {normalize.canonical(r["latex"]) for r in regression_records}
    return {r["id"] for r in train_records if r["id"] in reg_ids or normalize.canonical(r["latex"]) in reg_canon}


def timing_probe(n_bench: int, n_train: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    vocab = [f"t{i}" for i in range(300)]
    enc = _Encoder()

    def gen(n: int, prefix: str) -> list[tuple[str, str]]:
        out = []
        for i in range(n):
            ln = rng.randint(12, 80)
            out.append((f"{prefix}{i}", enc.encode([rng.choice(vocab) for _ in range(ln)])))
        return out

    bench, train = gen(n_bench, "b"), gen(n_train, "t")
    t0 = time.perf_counter()
    hits, pairs = _sweep(bench, train, 0.9)
    dt = time.perf_counter() - t0
    return {
        "seconds": round(dt, 3), "n_bench": n_bench, "n_train": n_train, "pairs_compared": pairs,
        "pairs_possible": n_bench * n_train, "hits": len(hits), "seed": seed, "gate_version": GATE_VERSION,
    }


def _check(args: argparse.Namespace) -> int:
    corpus, bench = Path(args.corpus), Path(args.benchmark)
    train = read_jsonl(corpus / "train" / "records.jsonl") if (corpus / "train" / "records.jsonl").exists() else []
    dev = read_jsonl(corpus / "dev" / "records.jsonl") if (corpus / "dev" / "records.jsonl").exists() else []
    bench_records = read_jsonl(bench / "records.jsonl")
    cat_paths = [Path(p) for p in args.catalog] or [paths.repo_root() / "data" / "catalog" / "train_dev.jsonl", bench / "catalog.jsonl"]
    rows: list[dict] = []
    for p in cat_paths:
        if p.exists():
            rows += read_jsonl(p)
    cand = Path(args.candidates) if args.candidates else None
    try:
        report = assert_clean(train + dev, bench_records, rows, candidates_path=cand, allow_missing_candidates=args.allow_missing_candidates)
        assert_side_clean(corpus)
    except splits.LeakageError as e:
        print(f"LEAK: {e}", file=sys.stderr)
        return 1
    print(f"clean: {len(train) + len(dev)} corpus records, {len(bench_records)} benchmark records, "
          f"{report.get('near_pairs_compared', 0)} near-duplicate pairs compared")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lightonocr_math.corpus.gate", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--corpus", default="data/corpus")
    c.add_argument("--benchmark", default="data/benchmark")
    c.add_argument("--catalog", action="append", default=[])
    c.add_argument("--candidates", default=None)
    c.add_argument("--allow-missing-candidates", action="store_true", help="run without data/candidates.jsonl (weakens same_author)")
    t = sub.add_parser("timing-probe")
    t.add_argument("--bench", type=int, default=3000)
    t.add_argument("--train", type=int, default=20000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return _check(args)
    res = timing_probe(args.bench, args.train, args.seed)
    print(json.dumps(res, sort_keys=True))
    if args.out:
        write_json(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
