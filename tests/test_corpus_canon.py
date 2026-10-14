import json

import pytest

from lightonocr_math import render
from lightonocr_math.corpus import canon
from lightonocr_math.corpus.canon import (
    CANON_FINGERPRINT,
    CANON_GOLDEN_PATH,
    canon_fingerprint,
    canonicalize,
    layout_only_diff,
    original_for_validation,
    structural_problems,
)


def _golden():
    return [json.loads(line) for line in CANON_GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_size():
    g = _golden()
    assert len(g) >= 25
    assert sum(1 for r in g if r["validate"]) >= 15


@pytest.mark.parametrize("case", _golden(), ids=lambda c: f"{c['env']}:{c['body'][:25]!r}")
def test_golden_expected(case):
    assert canonicalize(case["body"], case["env"]) == case["expected"]


@pytest.mark.parametrize("case", _golden(), ids=lambda c: f"{c['env']}:{c['body'][:25]!r}")
def test_golden_layout_only(case):
    assert layout_only_diff(case["body"], canonicalize(case["body"], case["env"])) == []


def test_layout_removals():
    assert canonicalize("x = 1 \\label{e1} % note\n", "equation") == "x = 1"
    for cmd in ("\\tag{3}", "\\tag*{a}", "\\nonumber", "\\notag"):
        assert canonicalize(f"x = 1 {cmd}", "equation") == "x = 1"
    assert canonicalize("a &= b \\\\", "align") == "a &= b"
    assert canonicalize("a \\%b", "equation") == "a \\%b"


def test_left_right_spacing():
    assert canonicalize("\\left (x\\right ]", "equation") == "\\left(x\\right]"
    assert canonicalize("\\leftarrow x", "equation") == "\\leftarrow x"


def test_glyph_tokens_preserved():
    for s in ("\\le", "\\leq", "\\dfrac", "\\mathrm{d}", "{x}", "\\bm{v}", "\\upsilon", "x'", "x^{\\prime}"):
        assert s in canonicalize(f"a {s} b", "equation")


def test_original_for_validation():
    body = "a &= b \\label{x} % c\n \\nonumber \\\\ c &= d \\tag{2} \\notag"
    out = original_for_validation(body, "align")
    assert out.startswith("\\begin{align*}") and out.endswith("\\end{align*}")
    for bad in ("\\label", "\\tag", "\\nonumber", "\\notag", "%"):
        assert bad not in out
    assert original_for_validation("x", "equation") == "\\begin{equation*}x\\end{equation*}"
    assert original_for_validation("x", "gather") == "\\begin{gather*}x\\end{gather*}"
    with pytest.raises(ValueError):
        original_for_validation("x", "multline")


def test_layout_only_diff_detects_glyph_change():
    assert layout_only_diff("a \\le b", "a \\leq b") == ["-\\le", "+\\leq"]
    assert layout_only_diff("x'", "x") == ["-'"]
    assert layout_only_diff("a \\label{q} \\\\", "a") == []


def test_structural_problems():
    assert structural_problems("\\frac{a}{b}") == []
    assert structural_problems("\\{ a \\}") == []
    assert "unbalanced braces" in structural_problems("\\frac{a}{b")
    assert "unbalanced braces" in structural_problems("a}")
    assert "empty label" in structural_problems("  ")
    assert any("label" in p for p in structural_problems("a \\label{x}"))
    assert any("comment" in p for p in structural_problems("a % b"))
    assert structural_problems("a \\% b") == []
    assert "trailing \\\\" in structural_problems("a \\\\")
    assert any("longer" in p for p in structural_problems("x" * 1501))


def test_rules_listed():
    assert [r[0] for r in canon.RULES] == ["C1", "C2", "C3", "C4", "C5", "C6"]


def test_fingerprint_pinned_and_sensitive(tmp_path):
    assert canon_fingerprint() == CANON_FINGERPRINT
    lines = CANON_GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    changed = tmp_path / "g.jsonl"
    rec = json.loads(lines[0])
    rec["expected"] += "x"
    changed.write_text("\n".join([json.dumps(rec), *lines[1:]]) + "\n", encoding="utf-8")
    assert canon_fingerprint(changed) != CANON_FINGERPRINT


@pytest.mark.docker
def test_golden_validation_bitmap_identical(tmp_path):
    cases = [c for c in _golden() if c["validate"]]
    originals = [original_for_validation(c["body"], c["env"]) for c in cases]
    labels = [canonicalize(c["body"], c["env"]) for c in cases]
    o_res = render.render_many(originals, cache=tmp_path, wrapper="env")
    l_res = render.render_many(labels, cache=tmp_path)
    for case, o, lab in zip(cases, o_res, l_res, strict=True):
        assert o.status == "rendered" and lab.status == "rendered", (case, o.status, o.note, lab.status, lab.note)
        identical, _ = render.compare(o, lab, tmp_path)
        assert identical, case
