"""ACCEPTANCE.md renderer: content, pass and fail variants, no mutation, no I/O."""

import copy
import re
from pathlib import Path

from scripts.acceptance_report import render_acceptance_md

RATE = {"rate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "num": 5, "den": 10, "n_papers": 3}
PAIRED = {"delta": 0.1, "ci_low": 0.01, "ci_high": 0.2}


def make_results(verdict="pass", transfer_holds=True):
    ok = verdict == "pass"
    sec = {"exact": RATE, "render_tolerant": RATE, "glyph_error_mean": 0.12}
    spacing = {"stock": RATE, "tuned": RATE, "paired": PAIRED, "n_rerendered": 4, "n_excluded": 1, "decides": False}
    return {
        "mode": "benchmark",
        "acceptance_sha256": "a" * 64,
        "git_commit": "abc1234",
        "verdict": verdict,
        "statements": {"verdict": "acceptance passed" if ok else "acceptance not passed: CI lower bound not above 0"},
        "headline": {
            "n": 2688, "n_papers": 40, "stock_rate": 0.4, "tuned_rate": 0.6 if ok else 0.41, "delta": 0.2 if ok else 0.01,
            "ci_low": 0.1 if ok else -0.02, "ci_high": 0.3, "flip": {"both_right": 900, "both_wrong": 500, "right_to_wrong": 20, "wrong_to_right": 300, "mcnemar_p": 0.001},
            "regression": {"r2w": 20, "w2r": 300, "stock_right": 1000, "max_by_ratio": 0.2, "max_by_share": 0.06, "ratio_ok": True, "share_ok": False},
            "regression_ok": False, "pass": ok, "reasons": [] if ok else ["CI lower bound not above 0"],
        },
        "regression_listing": {
            "rows": [{"id": "c01", "paper_id": "p1", "classes": ["prime_l"], "control_glyph": True, "gt_latex": "a'", "stock_pred": "a'", "tuned_pred": "al"}],
            "by_class": {"prime_l": 1}, "n_control_glyph": 1,
        },
        "secondary": {"crops": {"stock": sec, "tuned": sec}, "pages": {"stock": sec, "tuned": sec}},
        "spacing": {"crops": spacing, "pages": spacing},
        "per_class": [
            {"class": "prime_l", "stock_errors": 3, "tuned_errors": 1, "denom": 10, "stock": RATE, "tuned": RATE, "paired": PAIRED, "claim": "not enough events to claim"},
            {"class": "other", "stock_errors": 300, "tuned_errors": 100, "denom": 900, "stock": RATE, "tuned": RATE, "paired": PAIRED, "claim": "fewer errors"},
        ],
        "transfer": {
            "n": 297, "n_papers": 12, "stock_rate": 0.3, "tuned_rate": 0.4 if transfer_holds else 0.29, "delta": 0.1 if transfer_holds else -0.01,
            "ci_low": 0.02 if transfer_holds else -0.08, "ci_high": 0.2, "holds": transfer_holds,
            "statement": "page-level transfer holds" if transfer_holds else "crop-level win, page-level transfer not shown",
        },
        "page_match": {"stock": {"rate": 0.95}, "tuned": {"rate": 0.91}},
        "dev_reference": {"stock": 0.394, "candidate": 0.649, "delta": 0.255},
        "n_error": {"crops_stock": 0, "crops_tuned": 2, "pages_stock": 0, "pages_tuned": 1},
        "inputs": {"crops_tuned_preds": {"path": "x", "sha256": "f" * 64}},
    }


def test_does_not_mutate_input_and_does_no_io():
    res = make_results()
    before = copy.deepcopy(res)
    text = render_acceptance_md(res, {"r2w": {"total": 1, "done": 1, "held": 1, "reversed": 0}, "w2r": {"total": 1, "done": 1, "held": 0, "reversed": 1}, "n_done": 2, "n_total": 2, "complete": True})
    assert res == before and isinstance(text, str)
    src = Path("scripts/acceptance_report.py").read_text()
    assert not re.search(r"open\(|write_text|json\.dump", src)


def test_pass_report_has_required_content():
    t = render_acceptance_md(make_results("pass"))
    for phrase in (
        "Verdict: PASS", "pass if CI lower bound > 0", "at most 0.2", "at most 0.06", "not ok", "cannot change the verdict",
        "not enough events to claim", "95.0%", "91.0%", "spacing-insensitive", "other 210 (82%)", "+0.255", "in-sample",
        "n_error", "crops_tuned 2", "uv run python -m scripts.acceptance score --mode benchmark", "c00db734", "prime_l 1",
        "page-level transfer holds",
    ):
        assert phrase in t, phrase
    assert "Reported as is" not in t


def test_fail_report_has_d04_sentence_and_transfer_statement():
    t = render_acceptance_md(make_results("fail", transfer_holds=False))
    assert "Verdict: FAIL" in t and "Reported as is: no retraining" in t
    assert "crop-level win, page-level transfer not shown" in t


def test_sensitivity_appendix_variants():
    none = render_acceptance_md(make_results())
    assert "CPU llama-server sensitivity run: not yet run; non-blocking (D-21)" in none
    sens = {"r2w": {"total": 5, "done": 3, "held": 2, "reversed": 1}, "w2r": {"total": 9, "done": 4, "held": 4, "reversed": 0}, "n_done": 7, "n_total": 14, "complete": False}
    t = render_acceptance_md(make_results(), sens)
    assert "Coverage: 7 of 14" in t and "--resume" in t and "does not change the verdict" in t
    sens["complete"] = True
    assert "--resume" not in render_acceptance_md(make_results(), sens)


def test_no_listing_rows_message():
    res = make_results()
    res["regression_listing"] = {"rows": [], "by_class": {}, "n_control_glyph": 0}
    assert "No record that stock got right" in render_acceptance_md(res)
