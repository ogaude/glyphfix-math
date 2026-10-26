"""Build the known-error regression set from train/dev equations (never from benchmark records).

Usage (from repo root):
  uv run python scripts/corpus_regression.py pool --train data/corpus/train/records.jsonl \
      --dev data/corpus/dev/records.jsonl --per-flag 300 --plain 300 --seed 0 --out data/corpus/regression/pool.jsonl
  uv run python scripts/corpus_regression.py select --scored <score>/records.jsonl --pool pool.jsonl \
      --per-class 30 --controls 75 --max-per-paper 3 --seed 0 --stock-manifest <pred>.manifest.json \
      --out regression.jsonl --manifest REGRESSION_MANIFEST.json

pool draws, with a fixed seed, up to --per-flag records for each flag (prime, greek, decoration, i_1) and --plain
records with only the plain flag. A record is pooled once, under the first flag in that order that it carries.
select takes scored pool rows (stock predictions scored by glyphfix_math.score), assigns each failing row
(exact False and render_identical not True) to its rarest known class of errors.CLASSES present in its classes
(rarity = failing rows carrying the class in the scored pool, ties by errors.CLASSES order),
keeps up to --per-class per class with at most --max-per-paper per paper, adds clean control rows spread across
flags (at most 2 per paper) and writes the set plus a manifest marked exclude_from_training.
Exit codes: 0 ok, 2 refused (benchmark or quarantined paper) or missing input.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from glyphfix_math import splits
from glyphfix_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from glyphfix_math.corpus.records import to_score_row
from glyphfix_math.errors import CLASSES
from glyphfix_math.ids import base_id
from glyphfix_math.score import SCORER_FINGERPRINT, SCORER_VERSION

FLAG_ORDER = ("prime", "greek", "decoration", "i_1")
KNOWN_CLASSES = tuple(c for c in CLASSES if c != "other")
MAX_CONTROLS_PER_PAPER = 2


def pool_rows(
    records: list[dict],
    *,
    per_flag: int,
    plain: int,
    seed: int,
    exclude: set[str] | None = None,
    flags: tuple[str, ...] = FLAG_ORDER,
) -> list[dict]:
    """Seeded draw of pool rows; no record twice, none from `exclude`; only the given flags (and plain)."""
    rng = random.Random(seed)
    taken = set(exclude or ())
    out: list[dict] = []
    for flag in (*flags, "plain"):
        limit = plain if flag == "plain" else per_flag
        if flag == "plain":
            cands = [r for r in records if set(r["flags"]) <= {"plain"} and r["id"] not in taken]
        else:
            cands = [r for r in records if flag in r["flags"] and r["id"] not in taken]
        cands.sort(key=lambda r: r["id"])
        rng.shuffle(cands)
        for r in cands[:limit]:
            taken.add(r["id"])
            out.append({**to_score_row(r), "image": r["image"], "flags": list(r["flags"]), "pool_flag": flag})
    return out


def _assert_train_dev(paper_ids: list[str], context: str) -> None:
    splits.assert_disjoint(sorted({base_id(p) for p in paper_ids}), context=context)


def cmd_pool(a: argparse.Namespace) -> int:
    for p in (a.train, a.dev):
        if not Path(p).exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2
    records = read_jsonl(a.train) + read_jsonl(a.dev)
    _assert_train_dev([r["arxiv_id"] for r in records], "corpus_regression pool input")
    exclude: set[str] = set()
    if a.exclude:
        exclude = {r["id"] for r in read_jsonl(a.exclude)}
    flags = tuple(f for f in FLAG_ORDER if f in set(a.flags.split(",")))
    rows = pool_rows(records, per_flag=a.per_flag, plain=a.plain, seed=a.seed, exclude=exclude, flags=flags)
    _assert_train_dev([r["paper_id"] for r in rows], "corpus_regression pool output")
    write_jsonl(a.out, rows)
    by_flag = Counter(r["pool_flag"] for r in rows)
    papers = Counter()
    per_paper = Counter(r["paper_id"] for r in rows)
    for n in per_paper.values():
        papers[n] += 1
    print(f"pool: {len(rows)} rows, {len(per_paper)} papers")
    for f in (*FLAG_ORDER, "plain"):
        print(f"  {f}: {by_flag.get(f, 0)}")
    return 0


def assign_class(row: dict, rarity: dict[str, int] | None = None) -> str | None:
    """Rarest eligible known class first; ties (and no rarity table) fall back to KNOWN_CLASSES order."""
    cls = set(row.get("classes") or [])
    eligible = [c for c in KNOWN_CLASSES if c in cls]
    if not eligible:
        return None
    rarity = rarity or {}
    return min(eligible, key=lambda c: (rarity.get(c, 0), KNOWN_CLASSES.index(c)))


def class_rarity(failing: list[dict]) -> dict[str, int]:
    """Failing candidates per known class across the whole scored pool."""
    counts: Counter = Counter()
    for r in failing:
        counts.update(c for c in KNOWN_CLASSES if c in set(r.get("classes") or []))
    return {c: counts.get(c, 0) for c in KNOWN_CLASSES}


def is_failure(row: dict) -> bool:
    return row.get("exact") is not True and row.get("render_identical") is not True


def is_clean(row: dict) -> bool:
    return row.get("exact") is True and row.get("render_identical") is True


def select_rows(
    scored: list[dict], pool: list[dict], *, per_class: int, controls: int, max_per_paper: int, seed: int
) -> tuple[list[dict], dict]:
    pool_by_id = {r["id"]: r for r in pool}
    rng = random.Random(seed)
    rows = sorted((r for r in scored if r["id"] in pool_by_id), key=lambda r: r["id"])
    rng.shuffle(rows)
    chosen: list[dict] = []
    counts: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    failing = [r for r in rows if is_failure(r)]
    rarity = class_rarity(failing)
    assigned = {r["id"]: assign_class(r, rarity) for r in failing}
    for cls in KNOWN_CLASSES:
        per_paper: Counter = Counter()
        n = 0
        for r in failing:
            if n >= per_class:
                break
            if assigned[r["id"]] != cls or per_paper[r["paper_id"]] >= max_per_paper:
                continue
            per_paper[r["paper_id"]] += 1
            n += 1
            chosen.append(_out_row(pool_by_id[r["id"]], r, cls))
        counts[cls] = n
        if n < per_class:
            shortfalls[cls] = per_class - n
    by_flag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if is_clean(r):
            by_flag[pool_by_id[r["id"]].get("pool_flag", "plain")].append(r)
    per_paper = Counter()
    n_controls = 0
    queues = [by_flag[f] for f in (*FLAG_ORDER, "plain")]
    idx = [0] * len(queues)
    progressed = True
    while n_controls < controls and progressed:
        progressed = False
        for qi, q in enumerate(queues):
            while idx[qi] < len(q):
                r = q[idx[qi]]
                idx[qi] += 1
                if per_paper[r["paper_id"]] >= MAX_CONTROLS_PER_PAPER:
                    continue
                per_paper[r["paper_id"]] += 1
                n_controls += 1
                chosen.append(_out_row(pool_by_id[r["id"]], r, "control"))
                progressed = True
                break
            if n_controls >= controls:
                break
    meta = {"counts": counts, "n_controls": n_controls, "shortfalls": shortfalls}
    return chosen, meta


def _out_row(pool_row: dict, scored: dict, cls: str) -> dict:
    return {
        **pool_row,
        "regression_class": cls,
        "stock_pred_latex": scored.get("pred_latex", ""),
        "stock_exact": scored.get("exact"),
        "stock_render_identical": scored.get("render_identical"),
        "stock_classes": list(scored.get("classes") or []),
        "scorer_version": SCORER_VERSION,
    }


def cmd_select(a: argparse.Namespace) -> int:
    for p in (a.scored, a.pool, a.stock_manifest):
        if not Path(p).exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2
    scored, pool = read_jsonl(a.scored), read_jsonl(a.pool)
    _assert_train_dev([r["paper_id"] for r in scored] + [r["paper_id"] for r in pool], "corpus_regression select input")
    chosen, meta = select_rows(scored, pool, per_class=a.per_class, controls=a.controls, max_per_paper=a.max_per_paper, seed=a.seed)
    chosen.sort(key=lambda r: (r["regression_class"], r["id"]))
    _assert_train_dev([r["paper_id"] for r in chosen], "corpus_regression select output")
    write_jsonl(a.out, chosen)
    write_json(
        a.manifest,
        {
            **meta,
            "exclude_from_training": True,
            "n_rows": len(chosen),
            "seed": a.seed,
            "scorer_version": SCORER_VERSION,
            "scorer_fingerprint": SCORER_FINGERPRINT,
            "stock_manifest_sha256": sha256_file(a.stock_manifest),
            "regression_sha256": sha256_file(a.out),
        },
    )
    print(f"regression: {len(chosen)} rows, counts={meta['counts']}, controls={meta['n_controls']}, shortfalls={meta['shortfalls']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="draw the candidate pool from train/dev records")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--per-flag", type=int, default=300)
    p.add_argument("--plain", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude")
    p.add_argument("--flags", default=",".join(FLAG_ORDER), help="comma list of flags to draw (default all)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pool)
    s = sub.add_parser("select", help="pick the regression items from scored stock predictions")
    s.add_argument("--scored", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--per-class", type=int, default=30)
    s.add_argument("--controls", type=int, default=75)
    s.add_argument("--max-per-paper", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--stock-manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--manifest", required=True)
    s.set_defaults(fn=cmd_select)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except splits.LeakageError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
