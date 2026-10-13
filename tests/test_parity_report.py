"""Categorisation rules for the HF vs Ollama parity report (torch-free, render disabled)."""

from __future__ import annotations

import pytest

from scripts.spikes import parity_report as pr
from scripts.spikes.hf_infer import first_divergence


def hf(text="a", n_prompt=80, img=64, finish="eos", **kw):
    return {"raw_text": text, "n_prompt_tokens": n_prompt, "image_tokens": img, "finish": finish, **kw}


def ol(text="a", n_prompt=80, done="stop", **kw):
    return {"raw_text": text, "prompt_eval_count": n_prompt, "done_reason": done, **kw}


def test_categories_constant():
    assert pr.CATEGORIES == ("quantization", "preprocessing", "template", "truncation", "bug")


def test_identical_is_not_a_mismatch():
    assert pr.categorize(hf("x"), ol("x"), template_offset=0) is None


def test_length_on_either_side_is_truncation():
    assert pr.categorize(hf("x", finish="length"), ol("y"), template_offset=0) == "truncation"
    assert pr.categorize(hf("x"), ol("y", done="length"), template_offset=0) == "truncation"


def test_constant_offset_is_template():
    assert pr.categorize(hf("x", n_prompt=80), ol("y", n_prompt=84), template_offset=4) == "template"


def test_varying_offset_is_preprocessing():
    assert pr.categorize(hf("x", n_prompt=80), ol("y", n_prompt=90), template_offset=4) == "preprocessing"


def test_small_margin_is_quantization():
    assert pr.categorize(hf("x", margin=1.5, ollama_token_rank=9), ol("y"), template_offset=0) == "quantization"


def test_ollama_token_in_top5_is_quantization():
    assert pr.categorize(hf("x", margin=5.0, ollama_token_rank=3), ol("y"), template_offset=0) == "quantization"


def test_large_margin_outside_top5_is_bug():
    assert pr.categorize(hf("x", margin=5.0, ollama_token_rank=40), ol("y"), template_offset=0) == "bug"


def test_missing_margin_is_bug():
    assert pr.categorize(hf("x"), ol("y"), template_offset=0) == "bug"


def test_empty_or_error_is_bug():
    assert pr.categorize(hf(""), ol("y"), template_offset=0) == "bug"
    assert pr.categorize(hf("x"), ol("  "), template_offset=0) == "bug"
    assert pr.categorize(hf("x", pred_error="boom"), ol("y"), template_offset=0) == "bug"


def test_overrides_replace_and_need_reason():
    ov = {"a": {"category": "preprocessing", "reason": "resize differs"}}
    assert pr.apply_override("a", "bug", ov) == ("preprocessing", "resize differs")
    assert pr.apply_override("b", "bug", ov) == ("bug", None)
    with pytest.raises(ValueError):
        pr.apply_override("a", "bug", {"a": {"category": "template", "reason": " "}})
    with pytest.raises(ValueError):
        pr.apply_override("a", "bug", {"a": {"category": "nope", "reason": "x"}})


def test_summary_counts_overrides_and_blocking():
    rows = [{"category": "quantization"}, {"category": "bug"}, {"category": None}]
    s = pr.summarize_categories(rows, n_overridden=1)
    assert s["categories"] == {"quantization": 1, "bug": 1}
    assert s["n_overridden"] == 1
    assert s["blocking"] is True
    s2 = pr.summarize_categories([{"category": "template"}], n_overridden=0)
    assert s2["blocking"] is False


def test_template_offset_is_modal():
    hf_rows = {"a": hf(n_prompt=80), "b": hf(n_prompt=90), "c": hf(n_prompt=70)}
    ol_rows = {"a": ol(n_prompt=84), "b": ol(n_prompt=94), "c": ol(n_prompt=75)}
    assert pr.modal_template_offset(hf_rows, ol_rows) == 4


def test_first_divergence_token_space():
    tok = lambda s: [ord(c) for c in s]
    assert first_divergence(tok("abc"), tok("abd")) == 2
    assert first_divergence(tok("ab"), tok("abc")) == 2
    assert first_divergence(tok("abc"), tok("abc")) is None
