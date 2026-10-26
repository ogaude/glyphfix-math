"""Join HF bf16 and Ollama Q8_0 outputs on the legacy dev crops, score both, and explain every mismatch.

Usage (from repo root, project venv):
  uv run python -m scripts.spikes.parity_report --hf-pred data/spikes/hf/HF.pred.jsonl \
    --hf-sidecar data/spikes/hf/HF.sidecar.jsonl --ol-pred data/spikes/ollama/S1.pred.jsonl \
    --ol-sidecar data/spikes/ollama/S1.sidecar.jsonl [--overrides data/spikes/parity/overrides.json] [--no-render]

Each raw mismatch gets exactly one category (CATEGORIES). Committed outputs carry counts and ids only; the
texts go to the gitignored data/spikes/parity/mismatches.jsonl. Exit codes follow glyphfix_math.score:
2 invalid input, the guard's own code, 3 render infrastructure failure.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from glyphfix_math import score
from glyphfix_math.guard import GuardError
from glyphfix_math.paths import repo_root
from glyphfix_math.render import RenderInfrastructureError
from scripts.spikes import common

CATEGORIES = ("quantization", "preprocessing", "template", "truncation", "bug")
MARGIN_MAX = 2.0
TOP_K = 5


def modal_template_offset(hf_rows: dict[str, dict], ol_rows: dict[str, dict]) -> int:
    """Most common value of ollama prompt_eval_count minus HF n_prompt_tokens over the shared crops."""
    offs = [
        ol_rows[i]["prompt_eval_count"] - hf_rows[i]["n_prompt_tokens"]
        for i in hf_rows
        if i in ol_rows and ol_rows[i].get("prompt_eval_count") is not None and hf_rows[i].get("n_prompt_tokens") is not None
    ]
    return collections.Counter(offs).most_common(1)[0][0] if offs else 0


def categorize(hf_row: dict, ol_row: dict, *, template_offset: int) -> str | None:
    """Category of one crop, None when the raw texts are identical."""
    if hf_row["raw_text"] == ol_row["raw_text"]:
        return None
    if hf_row.get("pred_error") or ol_row.get("pred_error") or not hf_row["raw_text"].strip() or not ol_row["raw_text"].strip():
        return "bug"
    if hf_row.get("finish") == "length" or ol_row.get("done_reason") == "length":
        return "truncation"
    if ol_row.get("prompt_eval_count") is None or hf_row.get("n_prompt_tokens") is None:
        return "bug"  # prompt token count unmeasured (Ollama omits it on a cached prompt): cannot attribute, needs a human
    offset = ol_row["prompt_eval_count"] - hf_row["n_prompt_tokens"]
    if offset != 0:
        return "template" if offset == template_offset else "preprocessing"
    margin = hf_row.get("margin")
    rank = hf_row.get("ollama_token_rank")
    if margin is None:
        return "bug"
    if margin <= MARGIN_MAX or (rank is not None and rank < TOP_K):
        return "quantization"
    return "bug"


def apply_override(row_id: str, auto: str | None, overrides: dict) -> tuple[str | None, str | None]:
    """Replace the automatic category by an override; every override needs a category and a non-empty reason."""
    ov = overrides.get(row_id)
    if ov is None:
        return auto, None
    if ov.get("category") not in CATEGORIES:
        raise ValueError(f"override for {row_id}: category must be one of {CATEGORIES}")
    reason = (ov.get("reason") or "").strip()
    if not reason:
        raise ValueError(f"override for {row_id}: a non-empty reason is required")
    return ov["category"], reason


def summarize_categories(rows: list[dict], *, n_overridden: int) -> dict:
    counts = collections.Counter(r["category"] for r in rows if r["category"] is not None)
    return {
        "categories": dict(counts),
        "n_overridden": n_overridden,
        "blocking": counts.get("bug", 0) > 0,
    }


def _stats(vals: list[int]) -> dict:
    if not vals:
        return {"min": None, "median": None, "max": None}
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}


def _rate(rows: list[dict], key: str) -> dict:
    sel = [r for r in rows if r.get(key) is not None]
    return {"n": len(sel), "true": sum(1 for r in sel if r[key]), "rate": (sum(1 for r in sel if r[key]) / len(sel)) if sel else None}


def _by_id(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in common.read_jsonl(path)}


def build(a: argparse.Namespace) -> tuple[dict, list[dict]]:
    hf_pred = {r["id"]: r for r in score.read_predictions(a.hf_pred)}
    ol_pred = {r["id"]: r for r in score.read_predictions(a.ol_pred)}
    hf_side, ol_side = _by_id(a.hf_sidecar), _by_id(a.ol_sidecar)
    ids = sorted(set(hf_pred) & set(ol_pred) & set(hf_side) & set(ol_side))
    if not ids:
        raise ValueError("no crop is present in all four inputs")
    missing_ids = sorted((set(hf_pred) | set(ol_pred) | set(hf_side) | set(ol_side)) - set(ids))
    overrides = json.loads(a.overrides.read_text(encoding="utf-8")) if a.overrides and a.overrides.exists() else {}
    hs = {i: hf_side[i] | ({"pred_error": hf_pred[i]["pred_error"]} if hf_pred[i].get("pred_error") else {}) for i in ids}
    os_ = {i: ol_side[i] | ({"pred_error": ol_pred[i]["pred_error"]} if ol_pred[i].get("pred_error") else {}) for i in ids}
    offset = modal_template_offset(hs, os_)
    rows, n_overridden = [], 0
    for i in ids:
        auto = categorize(hs[i], os_[i], template_offset=offset)
        cat, reason = apply_override(i, auto, overrides) if auto is not None else (None, None)
        n_overridden += reason is not None
        rows.append(
            {
                "id": i,
                "auto_category": auto,
                "category": cat,
                "override_reason": reason,
                "hf_text": hs[i]["raw_text"],
                "ollama_text": os_[i]["raw_text"],
                "hf_prompt_tokens": hs[i]["n_prompt_tokens"],
                "hf_image_tokens": hs[i]["image_tokens"],
                "ollama_prompt_tokens": os_[i].get("prompt_eval_count"),
                "margin": hs[i].get("margin"),
                "ollama_token_rank": hs[i].get("ollama_token_rank"),
                "divergence_index": hs[i].get("divergence_index"),
            }
        )
    render = not a.no_render
    arm = score.score_records(
        [
            {"id": i, "paper_id": hf_pred[i]["paper_id"], "gt_latex": hf_pred[i]["pred_latex"] or " ", "pred_latex": ol_pred[i]["pred_latex"]}
            for i in ids
        ],
        render=render,
    )
    vs_gt_hf = score.score_records([hf_pred[i] for i in ids], render=render)
    vs_gt_ol = score.score_records([ol_pred[i] for i in ids], render=render)
    n_raw_equal = sum(1 for r in rows if r["auto_category"] is None and hs[r["id"]]["raw_text"] == os_[r["id"]]["raw_text"])
    summary = {
        "n": len(ids),
        "n_missing": len(missing_ids),
        "missing_ids": missing_ids,
        "n_raw_equal": n_raw_equal,
        "n_stripped_equal": sum(1 for i in ids if hf_pred[i]["pred_latex"] == ol_pred[i]["pred_latex"]),
        "normalized_agreement": _rate(arm, "exact"),
        "render_agreement_identical": _rate(arm, "render_identical") if render else "missing: --no-render",
        "render_agreement_tolerant": _rate(arm, "render_tolerant") if render else "missing: --no-render",
        "hf_vs_gt": {"exact": _rate(vs_gt_hf, "exact")} | ({"render_identical": _rate(vs_gt_hf, "render_identical")} if render else {}),
        "ollama_vs_gt": {"exact": _rate(vs_gt_ol, "exact")} | ({"render_identical": _rate(vs_gt_ol, "render_identical")} if render else {}),
        "n_prompt_match": sum(1 for i in ids if hs[i]["n_prompt_tokens"] == os_[i].get("prompt_eval_count")),
        "template_offset": offset,
        "image_tokens_hf": _stats([hs[i]["image_tokens"] for i in ids]),
        "prompt_tokens_hf": _stats([hs[i]["n_prompt_tokens"] for i in ids]),
        "prompt_tokens_ollama": _stats([os_[i]["prompt_eval_count"] for i in ids if os_[i].get("prompt_eval_count") is not None]),
        "pixel_budget": {
            "hf": {"min_px": hs[ids[0]].get("min_px"), "max_px": hs[ids[0]].get("max_px")},
            "ollama": json.loads(a.ol_budget.read_text(encoding="utf-8")) if a.ol_budget else "not recorded",
        },
        "render": render,
        "scorer_version": score.SCORER_VERSION,
        "mismatch_ids": {c: [r["id"] for r in rows if r["category"] == c] for c in CATEGORIES if any(r["category"] == c for r in rows)},
    }
    summary |= summarize_categories(rows, n_overridden=n_overridden)
    # crops absent from any input (for example an Ollama error row has no sidecar) shrink n silently otherwise
    summary["blocking"] = bool(summary["blocking"] or missing_ids)
    return summary, [r for r in rows if r["auto_category"] is not None]


def markdown(s: dict) -> str:
    L = ["# HF bf16 vs Ollama Q8_0 parity (legacy dev crops)", ""]
    L += ["| item | value |", "|---|---|"]
    for k in ("n", "n_missing", "n_raw_equal", "n_stripped_equal", "n_prompt_match", "template_offset", "n_overridden", "blocking", "scorer_version"):
        L.append(f"| {k} | {s[k]} |")
    for k in ("normalized_agreement", "render_agreement_identical", "render_agreement_tolerant"):
        L.append(f"| {k} | {json.dumps(s[k])} |")
    L += ["", "## Per-arm correctness against the legacy ground truth", "", "| arm | metric | n | true | rate |", "|---|---|---|---|---|"]
    for arm in ("hf_vs_gt", "ollama_vs_gt"):
        for m, v in s[arm].items():
            L.append(f"| {arm} | {m} | {v['n']} | {v['true']} | {v['rate']} |")
    L += ["", "## Mismatch categories", "", "| category | count | ids |", "|---|---|---|"]
    for c, n in s["categories"].items():
        L.append(f"| {c} | {n} | {', '.join(s['mismatch_ids'].get(c, []))} |")
    L += ["", "## Tokens and pixel budget", "", f"- HF image tokens: {json.dumps(s['image_tokens_hf'])}", f"- HF prompt tokens: {json.dumps(s['prompt_tokens_hf'])}", f"- Ollama prompt tokens: {json.dumps(s['prompt_tokens_ollama'])}", f"- Pixel budget: {json.dumps(s['pixel_budget'])}", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name in ("hf-pred", "hf-sidecar", "ol-pred", "ol-sidecar"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--overrides", type=Path, default=None)
    ap.add_argument("--ol-budget", type=Path, default=None, help="JSON with the Ollama-side pixel budget")
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "data" / "spikes" / "parity")
    ap.add_argument("--evidence-dir", type=Path, default=common.EVIDENCE_DIR)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    try:
        summary, mism = build(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except RenderInfrastructureError as e:
        print(f"error: render infrastructure failure: {e}", file=sys.stderr)
        return 3
    common.write_jsonl(a.out_dir / "mismatches.jsonl", mism)
    common.write_json(a.evidence_dir / "parity_summary.json", summary)
    (a.evidence_dir / "parity_report.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "n_raw_equal", "categories", "blocking")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
