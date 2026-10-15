"""Stratified training-set builder: pure functions, no I/O, no model or tokenizer imports.

Usage:
  from lightonocr_math.trainset import build, subset, dev_subset, target_text, allocate_equal_share
  rows, body = build(train_records, labels, taxonomy, regression_rows, dev_ids, token_len=count_tokens, seed=0)
  small = subset(rows, n=1500, seed=0)
  ids = dev_subset(dev_records, n=150, per_paper=3, seed=0)

Rules:
  * the taxonomy must be marked reviewed; build refuses otherwise
  * rows come from train papers only: a benchmark or quarantined paper raises splits.LeakageError, a dev paper, a dev
    record id, a regression id, a record with the same canonical LaTeX as a regression row, a record without a label,
    a record whose render check is undecided and a record whose fenced target exceeds max_target_tokens are dropped
    and counted by reason
  * composition: replay_frac clean replay + control_frac controls + the rest mined error rows, of `total` rows;
    error rows use distinct crops only and are balanced by water-filling toward an equal share per taxonomy class,
    classes short of the equal-share target are recorded as {class: {target, got}} under shortfalls
  * controls are clean crops carrying a CONTROL_GLYPHS token (round-robin across the glyph queues); replay is the
    remaining clean crops, round-robin across flag queues (a row waits in the queue of its rarest flag)
  * the per-paper cap is asserted, never applied silently: a ValueError is raised if a paper would exceed it
  * every selection uses random.Random seeded from (seed, stage, key) over id-sorted input, so the output does not
    depend on input order

Known limits: no canonical-text dedupe against the benchmark (paper-level disjointness only, plus regression_overlap);
a group short of clean supply is not topped up from another group.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from lightonocr_math import splits
from lightonocr_math.corpus.gate import regression_overlap
from lightonocr_math.corpus.records import LICENSE_SPDX
from lightonocr_math.ids import base_id
from lightonocr_math.latex_tokens import tokenize

__all__ = [
    "BUILDER_VERSION",
    "CONTROL_GLYPHS",
    "FLAG_ORDER",
    "allocate_equal_share",
    "build",
    "dev_subset",
    "primary_flag",
    "subset",
    "summarize",
    "target_text",
]

BUILDER_VERSION = "1"
CONTROL_GLYPHS = ("'", "I", "l", "\\ell", "1", "|", "v", "\\nu", "\\upsilon")
FLAG_ORDER = ("prime", "greek", "decoration", "i_1", "plain")
DEV_FLAG_ORDER = ("plain", "prime", "greek", "decoration", "i_1")
GROUPS = ("error", "replay", "control")
TOTAL = 8000
REPLAY_FRAC = 0.50
CONTROL_FRAC = 0.10
MAX_PER_PAPER = 30
MAX_TARGET_TOKENS = 900

_OUT_KEYS = ("id", "arxiv_id", "image", "latex", "width", "height", "license_spdx", "license_uri", "paper_version")


def target_text(latex: str) -> str:
    return "```latex\n" + latex + "\n```"


def primary_flag(flags: Iterable[str]) -> str:
    fl = set(flags)
    for f in FLAG_ORDER:
        if f in fl:
            return f
    return "plain"


def allocate_equal_share(supply: dict[str, int], budget: int) -> dict[str, int]:
    """Water-filling: equal split of the budget, classes with less supply than their share take all they have.

    Integer remainders go to the unsaturated classes in dict order. The result never exceeds the supply and
    sums to min(budget, total supply).
    """
    out = {c: 0 for c in supply}
    active = [c for c in supply if supply[c] > 0]
    rem = max(budget, 0)
    while active and rem > 0:
        share = rem // len(active)
        saturated = [c for c in active if supply[c] <= share]
        if not saturated:
            extra = rem - share * len(active)
            for i, c in enumerate(active):
                out[c] = share + (1 if i < extra else 0)
            return out
        for c in saturated:
            out[c] = supply[c]
            rem -= supply[c]
        active = [c for c in active if c not in saturated]
    return out


def _rng(seed: int, stage: str, key: str) -> random.Random:
    return random.Random(f"{seed}/{stage}/{key}")


def _shuffled(ids: list[str], seed: int, stage: str, key: str) -> list[str]:
    out = sorted(ids)
    _rng(seed, stage, key).shuffle(out)
    return out


def _round_robin(queues: list[list[str]], n: int) -> list[tuple[int, str]]:
    """Take up to n items cycling over the queues; returns (queue index, item)."""
    idx = [0] * len(queues)
    out: list[tuple[int, str]] = []
    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for qi, q in enumerate(queues):
            if idx[qi] < len(q):
                out.append((qi, q[idx[qi]]))
                idx[qi] += 1
                progressed = True
                if len(out) >= n:
                    break
    return out


def _first_glyph(latex: str) -> str | None:
    toks = set(tokenize(latex))
    for g in CONTROL_GLYPHS:
        if g in toks:
            return g
    return None


def build(
    train_records: list[dict],
    labels: list[dict],
    taxonomy: dict,
    regression_rows: list[dict],
    dev_ids: Iterable[str],
    *,
    token_len: Callable[[str], int],
    total: int = TOTAL,
    replay_frac: float = REPLAY_FRAC,
    control_frac: float = CONTROL_FRAC,
    max_per_paper: int = MAX_PER_PAPER,
    max_target_tokens: int = MAX_TARGET_TOKENS,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    if taxonomy.get("reviewed") is not True:
        raise ValueError("taxonomy is not reviewed (reviewed != true); review data/mining/taxonomy.json first")
    classes = list(taxonomy["classes"])
    if not classes:
        raise ValueError("taxonomy has no classes")

    for r in train_records:
        if r.get("license_spdx") not in LICENSE_SPDX:
            raise ValueError(f"record {r.get('id')}: license {r.get('license_spdx')!r} not in {sorted(LICENSE_SPDX)}")
    papers = sorted({base_id(r["arxiv_id"]) for r in train_records})
    splits.assert_disjoint(papers, context="trainset.build")
    train_papers = set(splits.load_train_ids(papers))

    reg_mapped = [{"id": r["id"], "latex": r["gt_latex"]} for r in regression_rows]
    overlap = regression_overlap(train_records, reg_mapped)
    dev_set = set(dev_ids)
    label_by_id = {lb["id"]: lb for lb in labels}

    dropped: Counter = Counter()
    errors: dict[str, list[str]] = {c: [] for c in classes}
    clean: list[str] = []
    rec_by_id: dict[str, dict] = {}
    for r in sorted(train_records, key=lambda x: x["id"]):
        rid = r["id"]
        lb = label_by_id.get(rid)
        if rid in overlap or (lb is not None and lb.get("in_regression")):
            dropped["in_regression"] += 1
        elif rid in dev_set:
            dropped["dev_record"] += 1
        elif base_id(r["arxiv_id"]) not in train_papers:
            dropped["non_train_paper"] += 1
        elif lb is None:
            dropped["missing_label"] += 1
        elif lb.get("render_identical") is None:
            dropped["render_undecided"] += 1
        elif token_len(target_text(r["latex"])) > max_target_tokens:
            dropped["target_too_long"] += 1
        elif lb.get("error"):
            if lb.get("mining_class") in errors:
                errors[lb["mining_class"]].append(rid)
                rec_by_id[rid] = r
            else:
                dropped["class_not_in_taxonomy"] += 1
        elif lb.get("clean"):
            clean.append(rid)
            rec_by_id[rid] = r
        else:
            dropped["neither_error_nor_clean"] += 1
    dropped["label_without_record"] = len(set(label_by_id) - {r["id"] for r in train_records})
    if not dropped["label_without_record"]:
        del dropped["label_without_record"]

    n_replay = round(total * replay_frac)
    n_control = round(total * control_frac)
    n_error = total - n_replay - n_control
    if n_error < 0:
        raise ValueError("replay_frac + control_frac exceed 1")

    shortfalls: dict[str, dict] = {}
    chosen: dict[str, tuple[str, str | None]] = {}  # id -> (group, class/glyph)

    # error rows: equal share per class by water-filling
    supply = {c: len(errors[c]) for c in classes}
    allocation = allocate_equal_share(supply, n_error)
    target = n_error // len(classes)
    for c in classes:
        for rid in _shuffled(errors[c], seed, "error", c)[: allocation[c]]:
            chosen[rid] = ("error", c)
        if supply[c] < target:
            shortfalls[c] = {"target": target, "got": allocation[c]}

    # controls: clean crops with a genuine confusable glyph
    by_glyph: dict[str, list[str]] = {g: [] for g in CONTROL_GLYPHS}
    for rid in clean:
        g = _first_glyph(rec_by_id[rid]["latex"])
        if g is not None:
            by_glyph[g].append(rid)
    glyph_queues = [_shuffled(by_glyph[g], seed, "control", g) for g in CONTROL_GLYPHS]
    picked = _round_robin(glyph_queues, n_control)
    for qi, rid in picked:
        chosen[rid] = ("control", CONTROL_GLYPHS[qi])
    if len(picked) < n_control:
        shortfalls["control"] = {"target": n_control, "got": len(picked)}

    # replay: remaining clean crops, queued under the rarest flag
    rest = [rid for rid in clean if rid not in chosen]
    flag_count: Counter = Counter()
    for rid in clean:
        flag_count.update(set(rec_by_id[rid].get("flags") or ["plain"]))
    by_flag: dict[str, list[str]] = {f: [] for f in FLAG_ORDER}
    for rid in rest:
        fl = [f for f in FLAG_ORDER if f in set(rec_by_id[rid].get("flags") or ["plain"])] or ["plain"]
        by_flag[min(fl, key=lambda f: (flag_count[f], FLAG_ORDER.index(f)))].append(rid)
    flag_queues = [_shuffled(by_flag[f], seed, "replay", f) for f in FLAG_ORDER]
    picked_r = _round_robin(flag_queues, n_replay)
    for _, rid in picked_r:
        chosen[rid] = ("replay", None)
    if len(picked_r) < n_replay:
        shortfalls["replay"] = {"target": n_replay, "got": len(picked_r)}

    rows: list[dict] = []
    for rid in sorted(chosen):
        group, key = chosen[rid]
        r, lb = rec_by_id[rid], label_by_id[rid]
        row = {k: r.get(k) for k in _OUT_KEYS}
        row.update(
            paper_id=base_id(r["arxiv_id"]),
            group=group,
            mining_class=key if group == "error" else None,
            control_glyph=key if group == "control" else None,
            flags=list(r.get("flags") or []),
            tags=list(lb.get("tags") or []),
            target_text=target_text(r["latex"]),
        )
        rows.append(row)

    per_paper = Counter(r["paper_id"] for r in rows)
    over = {p: n for p, n in per_paper.items() if n > max_per_paper}
    if over:
        raise ValueError(f"per-paper cap {max_per_paper} exceeded: {dict(sorted(over.items())[:5])}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row ids in training set")

    body = summarize(rows, classes=classes)
    body.update(
        builder_version=BUILDER_VERSION,
        seed=seed,
        params={
            "total": total, "replay_frac": replay_frac, "control_frac": control_frac,
            "max_per_paper": max_per_paper, "max_target_tokens": max_target_tokens,
        },
        equal_share_target=target,
        allocation=allocation,
        shortfalls=shortfalls,
        dropped=dict(sorted(dropped.items())),
    )
    return rows, body


def summarize(rows: list[dict], *, classes: Iterable[str] | None = None) -> dict:
    """Row counts (group, error class, control glyph, replay flag), papers and licenses."""
    group = Counter(r["group"] for r in rows)
    err = Counter(r["mining_class"] for r in rows if r["group"] == "error")
    if classes is not None:
        err = Counter({c: err.get(c, 0) for c in classes})
    glyph = Counter(r["control_glyph"] for r in rows if r["group"] == "control")
    flag = Counter(primary_flag(r["flags"]) for r in rows if r["group"] == "replay")
    per_paper = Counter(r["paper_id"] for r in rows)
    return {
        "n_rows": len(rows),
        "n_papers": len(per_paper),
        "max_rows_per_paper": max(per_paper.values(), default=0),
        "counts": {
            "group": {g: group.get(g, 0) for g in GROUPS},
            "error_class": dict(err),
            "control_glyph": dict(glyph),
            "replay_flag": dict(flag),
        },
        "licenses": dict(Counter(r["license_spdx"] for r in rows)),
    }


def _largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    base = {k: (n * v) // total for k, v in counts.items()}
    left = n - sum(base.values())
    order = sorted(counts, key=lambda k: (-((n * counts[k]) % total), k))
    for k in order[:left]:
        base[k] += 1
    return base


def _stratum(row: dict) -> str:
    if row["group"] == "error":
        return row["mining_class"]
    if row["group"] == "control":
        return row["control_glyph"]
    return primary_flag(row["flags"])


def subset(rows: list[dict], *, n: int, seed: int = 0) -> list[dict]:
    """Group-proportional subset: largest remainder over groups, then over classes/glyphs/flags within a group."""
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["id"])
    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_group[r["group"]][_stratum(r)].append(r)
    group_quota = _largest_remainder({g: sum(len(v) for v in by_group[g].values()) for g in by_group}, n)
    out: list[dict] = []
    for g in sorted(by_group):
        strata = by_group[g]
        quota = _largest_remainder({k: len(v) for k, v in strata.items()}, group_quota[g])
        for k in sorted(strata):
            by_id = {r["id"]: r for r in strata[k]}
            out.extend(by_id[i] for i in _shuffled(list(by_id), seed, f"subset/{g}", k)[: quota[k]])
    return sorted(out, key=lambda r: r["id"])


def dev_subset(dev_records: list[dict], *, n: int = 150, per_paper: int = 3, seed: int = 0) -> list[str]:
    """Fixed mini-eval ids from dev papers only: round-robin over flags, at most per_paper per paper."""
    papers = sorted({base_id(r["arxiv_id"]) for r in dev_records})
    if set(splits.load_dev_ids(papers)) != set(papers):
        raise splits.LeakageError("dev records contain papers that are not dev papers", {})
    queues_ids: dict[str, list[str]] = {f: [] for f in DEV_FLAG_ORDER}
    paper_of: dict[str, str] = {}
    for r in dev_records:
        queues_ids[primary_flag(r.get("flags") or [])].append(r["id"])
        paper_of[r["id"]] = base_id(r["arxiv_id"])
    queues = [_shuffled(queues_ids[f], seed, "dev", f) for f in DEV_FLAG_ORDER]
    idx = [0] * len(queues)
    per: Counter = Counter()
    out: list[str] = []
    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for qi, q in enumerate(queues):
            while idx[qi] < len(q):
                rid = q[idx[qi]]
                idx[qi] += 1
                if per[paper_of[rid]] >= per_paper:
                    continue
                per[paper_of[rid]] += 1
                out.append(rid)
                progressed = True
                break
            if len(out) >= n:
                break
    return sorted(out)
