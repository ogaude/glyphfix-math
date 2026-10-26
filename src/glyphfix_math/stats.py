"""Paper-level statistics for the evaluation harness.

Equations inside one paper are correlated, so every interval here resamples whole papers
with replacement, never single records. Defaults: B = 10,000 replicates, percentile
intervals at 95 %, and a fixed seed that the caller records next to the result.

Rates are pooled: a replicate's rate is sum(num) / sum(den) over the drawn papers. A
replicate whose drawn papers have zero denominator is NaN and is dropped from the
percentile (numpy.nanpercentile), it is not counted as 0.

Per-class denominators (A6): a record counts toward the denominator of class c when the
ground truth contains notation where c can occur (``present``) or when the record shows
error c. The rate therefore never exceeds 1.

scipy.stats.bootstrap is deliberately not used: it resamples rows, not clusters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from glyphfix_math.errors import CLASSES

DEFAULT_B = 10_000


def _aggregate(paper_ids: Sequence[str], *cols: Sequence[float]) -> tuple[list[str], list[np.ndarray]]:
    n = len(paper_ids)
    for c in cols:
        if len(c) != n:
            raise ValueError("paper_ids and value columns must have the same length")
    index: dict[str, int] = {}
    for p in paper_ids:
        index.setdefault(p, len(index))
    out = [np.zeros(len(index), dtype=np.float64) for _ in cols]
    for i, p in enumerate(paper_ids):
        j = index[p]
        for k, c in enumerate(cols):
            out[k][j] += float(c[i])
    return list(index), out


def _draw(n_papers: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_papers, size=(B, n_papers))


def _ci(values: np.ndarray, ci: float) -> tuple[float | None, float | None]:
    if values.size == 0 or np.all(np.isnan(values)):
        return None, None
    lo, hi = np.nanpercentile(values, [(1 - ci) / 2 * 100, (1 - (1 - ci) / 2) * 100])
    return float(lo), float(hi)


def cluster_bootstrap(
    paper_ids: Sequence[str],
    num: Sequence[float],
    den: Sequence[float],
    *,
    B: int = DEFAULT_B,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Pooled rate sum(num)/sum(den) with a percentile CI from resampling papers with replacement."""
    papers, (pn, pd) = _aggregate(paper_ids, num, den)
    tn, td = float(pn.sum()), float(pd.sum())
    res = {"rate": None, "ci_low": None, "ci_high": None, "num": tn, "den": td, "n_papers": len(papers)}
    if td == 0:
        return res
    res["rate"] = tn / td
    idx = _draw(len(papers), B, seed)
    sn = pn[idx].sum(axis=1)
    sd = pd[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(sd > 0, sn / np.where(sd > 0, sd, 1), np.nan)
    res["ci_low"], res["ci_high"] = _ci(rates, ci)
    return res


def paired_bootstrap_delta(
    paper_ids: Sequence[str],
    a: Sequence[float],
    b: Sequence[float],
    den: Sequence[float],
    *,
    B: int = DEFAULT_B,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """delta = rate_b - rate_a on the same paper draw in every replicate."""
    papers, (pa, pb, pd) = _aggregate(paper_ids, a, b, den)
    td = float(pd.sum())
    res = {"delta": None, "ci_low": None, "ci_high": None, "rate_a": None, "rate_b": None, "den": td, "n_papers": len(papers)}
    if td == 0:
        return res
    res["rate_a"] = float(pa.sum()) / td
    res["rate_b"] = float(pb.sum()) / td
    res["delta"] = res["rate_b"] - res["rate_a"]
    idx = _draw(len(papers), B, seed)
    sa, sb, sd = pa[idx].sum(axis=1), pb[idx].sum(axis=1), pd[idx].sum(axis=1)
    safe = np.where(sd > 0, sd, 1)
    deltas = np.where(sd > 0, (sb - sa) / safe, np.nan)
    res["ci_low"], res["ci_high"] = _ci(deltas, ci)
    return res


def per_class_rates(rows: list[dict], *, B: int = DEFAULT_B, seed: int = 0) -> dict[str, dict]:
    """Per error class: errors, denom, rate and paper-level CI. Rows carry paper_id, classes, present."""
    papers = [r["paper_id"] for r in rows]
    out: dict[str, dict] = {}
    for c in CLASSES:
        num = [1.0 if c in r.get("classes", ()) else 0.0 for r in rows]
        den = [1.0 if (c in r.get("present", ()) or c in r.get("classes", ())) else 0.0 for r in rows]
        r = cluster_bootstrap(papers, num, den, B=B, seed=seed)
        out[c] = {
            "errors": int(r["num"]),
            "denom": int(r["den"]),
            "rate": r["rate"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        }
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def flip_table(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict:
    """Per-record correctness of run A vs run B. True means right."""
    if set(a) != set(b):
        raise ValueError("flip_table needs identical id sets in both runs")
    both_right, both_wrong, r2w, w2r = [], [], [], []
    for k in sorted(a):
        if a[k] and b[k]:
            both_right.append(k)
        elif not a[k] and not b[k]:
            both_wrong.append(k)
        elif a[k]:
            r2w.append(k)
        else:
            w2r.append(k)
    return {
        "both_right": len(both_right),
        "both_wrong": len(both_wrong),
        "right_to_wrong": len(r2w),
        "wrong_to_right": len(w2r),
        "right_to_wrong_ids": r2w,
        "wrong_to_right_ids": w2r,
        "n": len(a),
        "mcnemar_p": mcnemar_exact(len(r2w), len(w2r)),
    }


__all__ = ["cluster_bootstrap", "flip_table", "mcnemar_exact", "paired_bootstrap_delta", "per_class_rates"]
