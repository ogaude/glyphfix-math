"""Error mining over scored stock predictions: error rows, confusion pairs, candidate clusters, labels, taxonomy.

Usage:
  rows = read_jsonl("data/mining/score/records.jsonl")       # score.py output on the train split
  errs = [r for r in rows if is_error_row(r)]
  cands = candidate_clusters(errs, min_papers=5, top_n=40)    # review these, then accept some by name
  doc = taxonomy_doc(cands, {"left_right": ["c01"]}, reviewer="...", reviewed_at="...", counts={}, ...)
  labels = build_labels(rows, train_by_id, accepted_patterns, regression_ids)

Rules (MINING_VERSION 1):

* An error row is a scored record whose ``render_identical`` is False. A record whose ground truth does not
  compile (``render_identical`` None) is neither error nor clean.
* Spans are the token-level differences between ``normalize_tokens`` of the ground truth and of the prediction,
  merged exactly like ``errors.classify_tokens`` (consecutive non-equal opcodes form one span). A pair is
  ``(gt_tokens, pred_tokens)``; insertions have an empty gt side, deletions an empty prediction side.
* Candidate clusters are counted over error rows whose scorer classes are a subset of ``{"other"}`` only, so a
  new cluster never takes rows from the five named glyph classes. Each distinct pair counts once per record.
* The ``NORM_EQUAL`` candidate collects rows that the normalizer calls exact but whose rendering differs.
* ``spacing_only`` is a diagnostic tag (the spacing commands differ between ground truth and prediction on an
  exact-but-render-different row). It never selects or excludes a row.
* Mining classes: the five glyph classes of ``errors.CLASSES`` first (rarest class first for a row with several),
  then accepted cluster names, then ``other``.

Known limits: pair counting is lexical, so one real confusion can split across several pairs; the reviewer
merges candidates by naming them into one cluster at accept time.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from rapidfuzz.distance import Levenshtein

from glyphfix_math.errors import CLASSES, DECORATION_COMMANDS, ERROR_RULES_VERSION, GREEK
from glyphfix_math.latex_tokens import tokenize
from glyphfix_math.normalize import SPACING_TOKENS, normalize_tokens

MINING_VERSION = "1"
GLYPH_CLASSES = tuple(c for c in CLASSES if c != "other")
NORM_EQUAL = "NORM_EQUAL"
STYLE_TOKENS = frozenset(
    {
        "\\left", "\\right", "&", "\\begin", "\\end", "aligned", "array", "{", "}", "\\tfrac", "\\dfrac", "\\frac",
        "\\mbox", "\\text", "\\textrm", "\\mathrm", "\\dots", "\\ldots", "\\cdots", "\\\\", "\\displaystyle",
        "\\limits", ".", ",",
    }
)  # fmt: skip
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

Pair = tuple[tuple[str, ...], tuple[str, ...]]


def is_error_row(row: dict) -> bool:
    """True when the render comparison failed.

    Differs from corpus_regression.is_failure on purpose: the co-primary render metric alone decides, so an
    exact-after-normalization row whose rendering differs is an error, and a row whose ground truth does not
    compile (render_identical None) is excluded.
    """
    return row.get("render_identical") is False


def is_clean_row(row: dict) -> bool:
    return row.get("render_identical") is True


def spans(gt: list[str], pred: list[str]) -> list[Pair]:
    """Merged differing spans between two token lists (same merge rule as errors.classify_tokens)."""
    merged: list[list[int]] = []
    for op in Levenshtein.opcodes(gt, pred):
        if op.tag == "equal":
            continue
        if merged and merged[-1][1] == op.src_start and merged[-1][3] == op.dest_start:
            merged[-1][1] = op.src_end
            merged[-1][3] = op.dest_end
        else:
            merged.append([op.src_start, op.src_end, op.dest_start, op.dest_end])
    return [(tuple(gt[s0:s1]), tuple(pred[d0:d1])) for s0, s1, d0, d1 in merged]


def row_pairs(row: dict) -> set[Pair]:
    return set(spans(normalize_tokens(row["gt_latex"]), normalize_tokens(row["pred_latex"])))


def _spacing(latex: str) -> list[str]:
    return [t for t in tokenize(latex) if t in SPACING_TOKENS]


def spacing_only(row: dict) -> bool:
    """Diagnostic tag: exact after normalization, render differs, and the spacing commands differ."""
    if row.get("exact") is not True or row.get("render_identical") is not False:
        return False
    return _spacing(row["gt_latex"]) != _spacing(row["pred_latex"])


def _glyphish(t: str) -> bool:
    return t in GREEK or t in DECORATION_COMMANDS or t == "'" or (len(t) == 1 and t.isascii() and t.isalnum())


def pair_family(pair: Pair) -> str:
    """style (layout tokens only), glyph (letters, digits, primes, Greek, decorations) or other."""
    toks = [*pair[0], *pair[1]]
    if toks and all(t in STYLE_TOKENS for t in toks):
        return "style"
    if any(_glyphish(t) for t in toks):
        return "glyph"
    return "other"


def _other_only(row: dict) -> bool:
    return set(row.get("classes") or []) <= {"other"}


def candidate_clusters(error_rows: list[dict], *, min_papers: int = 5, top_n: int = 40) -> list[dict]:
    """Confusion-pair candidates over error rows the scorer puts only in ``other``.

    Ids are c01, c02, ... in (-n_records, -n_papers, gt, pred) order; c00 is the NORM_EQUAL candidate.
    """
    rows = [r for r in error_rows if is_error_row(r) and _other_only(r)]
    by_pair: dict[Pair, dict[str, str]] = defaultdict(dict)  # pair -> {record id: paper}
    for r in rows:
        for p in row_pairs(r):
            by_pair[p][r["id"]] = r["paper_id"]
    ranked = [
        (p, ids) for p, ids in by_pair.items() if len(set(ids.values())) >= min_papers
    ]
    ranked.sort(key=lambda x: (-len(x[1]), -len(set(x[1].values())), list(x[0][0]), list(x[0][1])))
    out: list[dict] = []
    ne = {r["id"]: r["paper_id"] for r in rows if r.get("exact") is True}
    if len(set(ne.values())) >= min_papers:
        out.append(
            {
                "id": "c00",
                "family": "layout",
                "gt_tokens": [NORM_EQUAL],
                "pred_tokens": [NORM_EQUAL],
                "n_records": len(ne),
                "n_papers": len(set(ne.values())),
                "example_ids": sorted(ne)[:5],
            }
        )
    for i, (p, ids) in enumerate(ranked[:top_n], start=1):
        out.append(
            {
                "id": f"c{i:02d}",
                "family": pair_family(p),
                "gt_tokens": list(p[0]),
                "pred_tokens": list(p[1]),
                "n_records": len(ids),
                "n_papers": len(set(ids.values())),
                "example_ids": sorted(ids)[:5],
            }
        )
    return out


def class_rarity(error_rows: list[dict], accepted: dict[str, list] | None = None) -> dict[str, int]:
    """Error rows per glyph class (a row counts for every glyph class it carries)."""
    counts: Counter = Counter()
    for r in error_rows:
        counts.update(c for c in GLYPH_CLASSES if c in set(r.get("classes") or []))
    return {c: counts.get(c, 0) for c in GLYPH_CLASSES}


def _matches(row: dict, patterns: list) -> bool:
    pairs = None
    for pat in patterns:
        if pat == NORM_EQUAL:
            if row.get("exact") is True:
                return True
            continue
        if pairs is None:
            pairs = row_pairs(row)
        if (tuple(pat[0]), tuple(pat[1])) in pairs:
            return True
    return False


def assign_mining_class(row: dict, accepted: dict[str, list], rarity: dict[str, int]) -> str | None:
    """Mining class of a row: glyph class (rarest first), accepted cluster, ``other``; None for non-errors.

    Only rows whose scorer classes are a subset of ``{"other"}`` can enter an accepted cluster.
    """
    if not is_error_row(row):
        return None
    cls = set(row.get("classes") or [])
    eligible = [c for c in GLYPH_CLASSES if c in cls]
    if eligible:
        return min(eligible, key=lambda c: (rarity.get(c, 0), CLASSES.index(c)))
    for name, patterns in accepted.items():
        if _matches(row, patterns):
            return name
    return "other"


def build_labels(
    scored_rows: list[dict], train_by_id: dict[str, dict], accepted: dict[str, list], regression_ids: set[str]
) -> list[dict]:
    """One labels row per scored record (labels.jsonl contract), sorted by id."""
    rarity = class_rarity([r for r in scored_rows if is_error_row(r)], accepted)
    out = []
    for r in scored_rows:
        err = is_error_row(r)
        tags = ["spacing_only"] if spacing_only(r) else []
        out.append(
            {
                "id": r["id"],
                "paper_id": r["paper_id"],
                "render_identical": r.get("render_identical"),
                "exact": r.get("exact"),
                "classes": list(r.get("classes") or []),
                "error": err,
                "clean": is_clean_row(r),
                "mining_class": assign_mining_class(r, accepted, rarity) if err else None,
                "tags": tags,
                "flags": list((train_by_id.get(r["id"]) or {}).get("flags") or []),
                "in_regression": r["id"] in regression_ids,
            }
        )
    out.sort(key=lambda x: x["id"])
    return out


def taxonomy_doc(
    candidates: list[dict],
    accepted_map: dict[str, list[str]],
    *,
    reviewer: str,
    reviewed_at: str,
    counts: dict,
    candidates_sha256: str,
    scored_sha256: str,
) -> dict:
    """taxonomy.json contract; raises ValueError on an invalid cluster name or an unknown candidate id."""
    if not reviewer or not reviewer.strip():
        raise ValueError("taxonomy needs a reviewer name")
    by_id = {c["id"]: c for c in candidates}
    used: set[str] = set()
    clusters = []
    for name, ids in accepted_map.items():
        if name in CLASSES:
            raise ValueError(f"cluster name {name!r} collides with a scorer class")
        if not _NAME_RE.match(name):
            raise ValueError(f"cluster name {name!r} is not snake_case")
        if not ids:
            raise ValueError(f"cluster {name!r} has no candidates")
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"cluster {name!r} references unknown candidate id(s): {', '.join(unknown)}")
        dup = [i for i in ids if i in used]
        if dup:
            raise ValueError(f"candidate id(s) already in another cluster: {', '.join(dup)}")
        used.update(ids)
        members = [by_id[i] for i in ids]
        fams = {m["family"] for m in members}
        patterns = [NORM_EQUAL if m["id"] == "c00" else [m["gt_tokens"], m["pred_tokens"]] for m in members]
        clusters.append(
            {
                "name": name,
                "candidate_ids": list(ids),
                "family": fams.pop() if len(fams) == 1 else "mixed",
                "patterns": patterns,
            }
        )
    return {
        "mining_version": MINING_VERSION,
        "error_rules_version": ERROR_RULES_VERSION,
        "reviewed": True,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "classes": [*GLYPH_CLASSES, *accepted_map, "other"],
        "accepted_clusters": clusters,
        "rejected_candidate_ids": sorted(set(by_id) - used),
        "counts": counts,
        "candidates_sha256": candidates_sha256,
        "scored_sha256": scored_sha256,
    }


__all__ = [
    "GLYPH_CLASSES",
    "MINING_VERSION",
    "NORM_EQUAL",
    "STYLE_TOKENS",
    "assign_mining_class",
    "build_labels",
    "candidate_clusters",
    "class_rarity",
    "is_clean_row",
    "is_error_row",
    "pair_family",
    "row_pairs",
    "spacing_only",
    "spans",
    "taxonomy_doc",
]
