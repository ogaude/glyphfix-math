"""Acceptance rules: pass/fail constants and pure verdict functions.

The held-out benchmark is scored exactly once, so every rule that decides the verdict lives here as
tested code before any benchmark prediction exists. This module is pure: no file reads or writes, no
child processes, no network, and it must not reference benchmark paths. Callers pass in scorer rows
(id -> row dict) and get plain dicts back.

ACCEPTANCE holds every threshold. constants_digest() hashes it, and ACCEPTANCE_SHA256 pins that hash;
the results file of a benchmark run records the same digest, so a constant changed after scoring is
detectable.
"""

from __future__ import annotations

import hashlib
import json
import re

from lightonocr_math import stats
from lightonocr_math.report import VERSION_KEYS

# Fixed before the benchmark is scored. The regression share was amended to 6% of stock-correct records
# on 2026-10-17, before any benchmark number existed.
ACCEPTANCE = {
    "version": "1",
    "headline": {
        "metric": "render_identical",
        "test": "paired_paper_cluster_bootstrap",
        "bootstrap_B": 10000,
        "bootstrap_seed": 0,
        "ci": 0.95,
        "pass_if_ci_low_gt": 0.0,
        "also_report": ["mcnemar_exact"],
    },
    "regression": {
        "max_r2w_over_w2r": 0.2,
        "max_r2w_over_stock_right": 0.06,
        "control_glyph_classes": ["prime_l", "dropped_prime", "greek_latin", "decoration", "i_1"],
        "control_glyph_rule": "tuned_error_classes_intersect",
        "empty_class_label": "render-only",
    },
    "decides": ["render_identical"],
    "secondary": ["exact", "render_tolerant", "glyph_error_mean", "spacing_insensitive_render_identical"],
    "transfer": {
        "pass_if_ci_low_gt": -0.03,
        "pass_if_delta_gt": 0.0,
        "blocks_crop_verdict": False,
        "fail_statement": "crop-level win, page-level transfer not shown",
        "unmatched_equations": "failure",
    },
    "per_class": {
        "classes": ["prime_l", "dropped_prime", "greek_latin", "decoration", "i_1", "other"],
        "min_stock_errors": 30,
        "few_events_statement": "not enough events to claim",
        "denominator": "gt_present_or_either_arm_error",
        "ids": "all_common_ids",
    },
    "spacing": {
        "commands": ["\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad"],
        "pattern": "\\\\(?:(?:qquad|quad)(?![A-Za-z])|[,;:!])",
        "replacement": " ",
        "crops": "required",
        "pages": "computed",
        "decides": False,
    },
    "generation": {"policy": "resume-only", "lock": True, "n_error": "scored-as-failure"},
    "engine": {"ollama_version": "0.40.1", "same_engine_both_arms": True},
    "sensitivity": {
        "engine": "llama-server-cpu",
        "scope": "all-flips-both-arms",
        "order": "r2w-then-w2r-sorted-by-id",
        "chunk": 50,
        "blocking": False,
    },
}

# Set after the dev dry run, before any benchmark generation; never changed after the benchmark scoring commit.
ACCEPTANCE_SHA256: str | None = "7b42234c9078a5b1005352395cfe00300536b749a56c99eb8a3ef447e02aabc2"

EPS = 1e-9


class AcceptanceError(ValueError):
    pass


def constants_digest(c: dict = ACCEPTANCE) -> str:
    blob = json.dumps(c, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_pinned() -> str:
    """Return the digest when the pin is set and matches the constants, else raise."""
    if ACCEPTANCE_SHA256 is None:
        raise AcceptanceError("ACCEPTANCE_SHA256 is not set")
    digest = constants_digest()
    if ACCEPTANCE_SHA256 != digest:
        raise AcceptanceError(f"ACCEPTANCE constants differ from the pin: {digest} != {ACCEPTANCE_SHA256}")
    return digest


# ------------------------------------------------------------------------------------ comparability

SAME_FIELDS = (
    "suite",
    "options",
    "endpoint",
    "think",
    "prompt",
    "prompt_sha256",
    "template_sha256",
    "pixel_policy",
    "ollama_version",
    "records_sha256",
    "scorer_version",
    "scorer_fingerprint",
    "render_config",
    "split_rule",
)


def comparability_problems(stock_manifest: dict, tuned_manifest: dict, *, expected_tuned_digest: str) -> list[str]:
    """Differences that would make the two arms not comparable; empty means comparable."""
    problems: list[str] = []
    for k in SAME_FIELDS:
        if stock_manifest.get(k) != tuned_manifest.get(k):
            problems.append(f"{k}: {stock_manifest.get(k)!r} vs {tuned_manifest.get(k)!r}")
    sp = (stock_manifest.get("gguf_blobs") or {}).get("projector")
    tp = (tuned_manifest.get("gguf_blobs") or {}).get("projector")
    if sp != tp:
        problems.append(f"gguf_blobs.projector: {sp!r} vs {tp!r}")
    want = ACCEPTANCE["engine"]["ollama_version"]
    for name, m in (("stock", stock_manifest), ("tuned", tuned_manifest)):
        if m.get("ollama_version") != want:
            problems.append(f"{name} ollama_version {m.get('ollama_version')!r} != {want!r}")
    if tuned_manifest.get("model_digest") != expected_tuned_digest:
        problems.append(f"tuned model_digest {tuned_manifest.get('model_digest')!r} != expected {expected_tuned_digest!r}")
    return problems


def results_problems(stock_results: dict, tuned_results: dict) -> list[str]:
    """Version keys and render image that must match between the two scored runs."""
    problems = [
        f"{k}: {stock_results.get(k)!r} vs {tuned_results.get(k)!r}" for k in VERSION_KEYS if stock_results.get(k) != tuned_results.get(k)
    ]
    si = (stock_results.get("render") or {}).get("image")
    ti = (tuned_results.get("render") or {}).get("image")
    if si != ti:
        problems.append(f"render image: {si!r} vs {ti!r}")
    return problems


# ----------------------------------------------------------------------------------------- headline


def paired_decided(stock: dict[str, dict], tuned: dict[str, dict]):
    """(ids, stock_bool, tuned_bool, excluded) over ids whose render_identical is decided in both arms."""
    if set(stock) != set(tuned):
        raise AcceptanceError("stock and tuned id sets differ")
    ids: list[str] = []
    ra: dict[str, bool] = {}
    rb: dict[str, bool] = {}
    excluded = {"stock_none": 0, "tuned_none": 0}
    for i in sorted(stock):
        x, y = stock[i].get("render_identical"), tuned[i].get("render_identical")
        if x is None:
            excluded["stock_none"] += 1
        if y is None:
            excluded["tuned_none"] += 1
        if x is None or y is None:
            continue
        ids.append(i)
        ra[i], rb[i] = bool(x), bool(y)
    return ids, ra, rb, excluded


def headline_verdict(stock: dict[str, dict], tuned: dict[str, dict], c: dict = ACCEPTANCE) -> dict:
    """Headline pass flag from render_identical only: significance plus both regression clauses."""
    ids, ra, rb, excluded = paired_decided(stock, tuned)
    n = len(ids)
    h, g = c["headline"], c["regression"]
    papers = [str(stock[i].get("paper_id", i)) for i in ids]
    boot = stats.paired_bootstrap_delta(
        papers,
        [float(ra[i]) for i in ids],
        [float(rb[i]) for i in ids],
        [1.0] * n,
        B=h["bootstrap_B"],
        seed=h["bootstrap_seed"],
        ci=h["ci"],
    )
    flips = stats.flip_table(ra, rb)
    stock_right = sum(ra.values())
    r2w, w2r = flips["right_to_wrong"], flips["wrong_to_right"]
    max_ratio = g["max_r2w_over_w2r"] * w2r
    max_share = g["max_r2w_over_stock_right"] * stock_right
    ratio_ok = r2w <= max_ratio + EPS
    share_ok = r2w <= max_share + EPS
    ci_low = boot["ci_low"]
    significant = ci_low is not None and ci_low > h["pass_if_ci_low_gt"] + EPS
    reasons: list[str] = []
    if n == 0:
        reasons.append("no common decided records")
    elif not significant:
        reasons.append(f"paired CI lower bound {ci_low} is not above {h['pass_if_ci_low_gt']}")
    if not ratio_ok:
        reasons.append(f"right-to-wrong {r2w} exceeds {g['max_r2w_over_w2r']} x wrong-to-right {w2r}")
    if not share_ok:
        reasons.append(f"right-to-wrong {r2w} exceeds {g['max_r2w_over_stock_right']} x stock-correct {stock_right}")
    regression_ok = ratio_ok and share_ok
    return {
        "n": n,
        "n_papers": boot["n_papers"],
        "stock_rate": boot["rate_a"],
        "tuned_rate": boot["rate_b"],
        "delta": boot["delta"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "stock_right": stock_right,
        "flip": {k: v for k, v in flips.items() if not k.endswith("_ids")},
        "r2w_ids": sorted(flips["right_to_wrong_ids"]),
        "w2r_ids": sorted(flips["wrong_to_right_ids"]),
        "excluded": excluded,
        "significant": significant,
        "regression": {
            "r2w": r2w,
            "w2r": w2r,
            "stock_right": stock_right,
            "max_by_ratio": max_ratio,
            "max_by_share": max_share,
            "ratio_ok": ratio_ok,
            "share_ok": share_ok,
        },
        "regression_ok": regression_ok,
        "pass": bool(n > 0 and significant and regression_ok),
        "reasons": reasons,
    }


def regression_listing(stock: dict[str, dict], tuned: dict[str, dict], r2w_ids, c: dict = ACCEPTANCE) -> dict:
    """Every right-to-wrong record with its tuned error classes and the control-glyph flag."""
    g = c["regression"]
    control = set(g["control_glyph_classes"])
    label = g["empty_class_label"]
    rows = []
    by_class: dict[str, int] = {}
    n_control = 0
    for i in sorted(r2w_ids):
        classes = list(tuned[i].get("classes") or [])
        is_control = bool(control & set(classes))
        n_control += is_control
        for k in classes or [label]:
            by_class[k] = by_class.get(k, 0) + 1
        rows.append(
            {
                "id": i,
                "paper_id": tuned[i].get("paper_id"),
                "classes": classes or [label],
                "control_glyph": is_control,
                "gt_latex": tuned[i].get("gt_latex"),
                "stock_pred": stock[i].get("pred_latex"),
                "tuned_pred": tuned[i].get("pred_latex"),
            }
        )
    return {"rows": rows, "by_class": by_class, "n_control_glyph": n_control}


# -------------------------------------------------------------------------------------- page transfer


def transfer_verdict(stock_rows: dict[str, dict], tuned_rows: dict[str, dict], *, headline_pass: bool, c: dict = ACCEPTANCE) -> dict:
    """Page-level transfer: paired delta CI lower bound above the tolerance and a positive point estimate.

    Unmatched equations arrive as rows whose render_identical is False; they stay in the pairing.
    """
    ids, ra, rb, excluded = paired_decided(stock_rows, tuned_rows)
    h, t = c["headline"], c["transfer"]
    papers = [str(stock_rows[i].get("paper_id", i)) for i in ids]
    boot = stats.paired_bootstrap_delta(
        papers,
        [float(ra[i]) for i in ids],
        [float(rb[i]) for i in ids],
        [1.0] * len(ids),
        B=h["bootstrap_B"],
        seed=h["bootstrap_seed"],
        ci=h["ci"],
    )
    flips = stats.flip_table(ra, rb)
    ci_low, delta = boot["ci_low"], boot["delta"]
    holds = bool(ci_low is not None and delta is not None and ci_low > t["pass_if_ci_low_gt"] + EPS and delta > t["pass_if_delta_gt"] + EPS)
    if holds:
        statement = "page-level transfer holds"
    elif headline_pass:
        statement = t["fail_statement"]
    else:
        statement = "page-level transfer not shown"
    return {
        "n": len(ids),
        "n_papers": boot["n_papers"],
        "stock_rate": boot["rate_a"],
        "tuned_rate": boot["rate_b"],
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": boot["ci_high"],
        "flip": {k: v for k, v in flips.items() if not k.endswith("_ids")},
        "excluded": excluded,
        "holds": holds,
        "statement": statement,
        "blocks_crop_verdict": t["blocks_crop_verdict"],
    }


# ------------------------------------------------------------------------------------------ per class


def per_class_table(stock: dict[str, dict], tuned: dict[str, dict], c: dict = ACCEPTANCE) -> list[dict]:
    """One row per error class over all common ids with one denominator shared by both arms.

    A record is in a class denominator when the ground truth has the class present or either arm shows
    the class as an error. Counts are string-tier scorer classes; the render-identical headline is separate.
    The paired delta is tuned error rate minus stock error rate.
    """
    if set(stock) != set(tuned):
        raise AcceptanceError("stock and tuned id sets differ")
    ids = sorted(stock)
    papers = [str(stock[i].get("paper_id", i)) for i in ids]
    h, p = c["headline"], c["per_class"]
    kw = {"B": h["bootstrap_B"], "seed": h["bootstrap_seed"], "ci": h["ci"]}
    out = []
    for cls in p["classes"]:
        s_err = [1.0 if cls in (stock[i].get("classes") or ()) else 0.0 for i in ids]
        t_err = [1.0 if cls in (tuned[i].get("classes") or ()) else 0.0 for i in ids]
        den = [
            1.0 if (cls in (stock[i].get("present") or ()) or cls in (tuned[i].get("present") or ()) or s_err[k] or t_err[k]) else 0.0
            for k, i in enumerate(ids)
        ]
        stock_errors, tuned_errors = int(sum(s_err)), int(sum(t_err))
        paired = stats.paired_bootstrap_delta(papers, s_err, t_err, den, **kw)
        if stock_errors < p["min_stock_errors"]:
            claim = p["few_events_statement"]
        elif paired["ci_high"] is not None and paired["ci_high"] < 0:
            claim = "fewer errors"
        elif paired["ci_low"] is not None and paired["ci_low"] > 0:
            claim = "more errors"
        else:
            claim = "no significant change"
        out.append(
            {
                "class": cls,
                "stock_errors": stock_errors,
                "tuned_errors": tuned_errors,
                "denom": int(sum(den)),
                "stock": stats.cluster_bootstrap(papers, s_err, den, **kw),
                "tuned": stats.cluster_bootstrap(papers, t_err, den, **kw),
                "paired": paired,
                "claim": claim,
            }
        )
    return out


# -------------------------------------------------------------------------------------------- spacing


def spacing_strip(latex: str, c: dict = ACCEPTANCE) -> str:
    """Replace the six spacing commands by one space; a control word followed by letters is left alone."""
    return re.sub(c["spacing"]["pattern"], c["spacing"]["replacement"], latex)


def spacing_changed_ids(*arms: dict[str, dict], c: dict = ACCEPTANCE) -> list[str]:
    """Ids where stripping changes the ground truth or the prediction in any of the given arms."""
    changed: set[str] = set()
    for rows in arms:
        for i, r in rows.items():
            for key in ("gt_latex", "pred_latex"):
                text = r.get(key) or ""
                if spacing_strip(text, c) != text:
                    changed.add(i)
    return sorted(changed)


def spacing_column(stock: dict[str, dict], tuned: dict[str, dict], rerendered: dict[str, dict], c: dict = ACCEPTANCE) -> dict:
    """Spacing-insensitive render-identical rate; secondary, it never decides the verdict.

    Unchanged ids reuse the frozen render_identical value (stripping changes nothing, so it is exact);
    changed ids take the supplied re-rendered value. Ids undecided in either arm are excluded.
    """
    if set(stock) != set(tuned):
        raise AcceptanceError("stock and tuned id sets differ")
    changed = set(spacing_changed_ids(stock, tuned, c=c))
    h = c["headline"]
    ids, papers, sa, sb = [], [], [], []
    n_rerendered = 0
    for i in sorted(stock):
        if i in changed:
            x, y = (rerendered.get(i) or {}).get("stock"), (rerendered.get(i) or {}).get("tuned")
        else:
            x, y = stock[i].get("render_identical"), tuned[i].get("render_identical")
        if x is None or y is None:
            continue
        n_rerendered += i in changed
        ids.append(i)
        papers.append(str(stock[i].get("paper_id", i)))
        sa.append(float(bool(x)))
        sb.append(float(bool(y)))
    kw = {"B": h["bootstrap_B"], "seed": h["bootstrap_seed"], "ci": h["ci"]}
    den = [1.0] * len(ids)
    return {
        "stock": stats.cluster_bootstrap(papers, sa, den, **kw),
        "tuned": stats.cluster_bootstrap(papers, sb, den, **kw),
        "paired": stats.paired_bootstrap_delta(papers, sa, sb, den, **kw),
        "n_rerendered": n_rerendered,
        "n_excluded": len(stock) - len(ids),
        "decides": c["spacing"]["decides"],
    }


# ------------------------------------------------------------------------------------- sensitivity


def sensitivity_summary(r2w_ids, w2r_ids, cpu_stock: dict, cpu_tuned: dict) -> dict:
    """How many flips found on the GPU engine also hold on the CPU engine.

    A right-to-wrong id holds when CPU stock is right and CPU tuned is wrong; a wrong-to-right id holds
    when CPU stock is wrong and CPU tuned is right. Ids missing on the CPU are not run; a None on either
    side is done but undecided.
    """
    summary: dict = {}
    n_done = n_total = 0
    for name, ids, held_pair in (("r2w", r2w_ids, (True, False)), ("w2r", w2r_ids, (False, True))):
        row = {"total": len(ids), "done": 0, "held": 0, "reversed": 0, "both_right": 0, "both_wrong": 0, "undecided": 0}
        for i in ids:
            if i not in cpu_stock or i not in cpu_tuned:
                continue
            row["done"] += 1
            x, y = cpu_stock[i], cpu_tuned[i]
            if x is None or y is None:
                row["undecided"] += 1
            elif (x, y) == held_pair:
                row["held"] += 1
            elif x and y:
                row["both_right"] += 1
            elif not x and not y:
                row["both_wrong"] += 1
            else:
                row["reversed"] += 1
        summary[name] = row
        n_done += row["done"]
        n_total += row["total"]
    summary.update(n_done=n_done, n_total=n_total, complete=n_done == n_total)
    return summary
