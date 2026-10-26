"""Adversarial pairs proving the normalizer cannot hide the target error types (EVL-02).

MUST_DIFFER rows are (ground truth, prediction, expected error class); the class
labels are reused by tests/test_errors.py. MUST_EQUAL rows are the harmless
D-03 equivalences.

Accepted string-tier leniency (D-03): ``\\bigl( x \\bigr)`` equals ``( x )`` here
although the glyph size differs; the render tier will expose that disagreement.
Likewise ``\\bm`` vs ``\\boldsymbol`` stay string-distinct while rendering alike.
"""

import time

import pytest

from glyphfix_math import normalize as N
from glyphfix_math.normalize import canonical, equivalent, normalize_tokens

MUST_DIFFER = [
    # prime vs l
    ("f'", "fl", "prime_l"),
    ("f'(x)", "fl(x)", "prime_l"),
    ("f^{\\prime}", "f^{l}", "prime_l"),
    ("f'", "f\\ell", "prime_l"),
    # dropped prime
    ("f'", "f", "dropped_prime"),
    ("f''", "f'", "dropped_prime"),
    ("u^{\\prime}_n", "u_n", "dropped_prime"),
    # Greek vs Latin
    ("\\upsilon", "v", "greek_latin"),
    ("\\nu", "v", "greek_latin"),
    ("\\alpha", "a", "greek_latin"),
    ("\\epsilon", "e", "greek_latin"),
    # decorations
    ("\\mathbf{x}", "x", "decoration"),
    ("\\boldsymbol{x}", "x", "decoration"),
    ("\\bm{x}", "x", "decoration"),
    ("\\bar{x}", "x", "decoration"),
    ("\\underline{x}", "x", "decoration"),
    ("\\hat{x}", "x", "decoration"),
    ("\\mathbf{x}", "\\mathrm{x}", "decoration"),
    ("\\mathbf{x}", "\\bm{x}", "decoration"),
    ("\\bar{x}", "\\overline{x}", "decoration"),
    ("\\mathrm{d}x", "dx", "decoration"),
    # I vs 1
    ("I", "1", "i_1"),
    ("x_I", "x_1", "i_1"),
    ("A^{-1}", "A^{-I}", "i_1"),
    # D-03 not merged
    ("\\ell", "l", "other"),
    ("\\varepsilon", "\\epsilon", "other"),
    ("\\phi", "\\varphi", "other"),
    ("\\left( x \\right)", "( x )", "other"),
    ("\\bigl( x \\bigr)", "\\left( x \\right)", "other"),
    # D-02 scope
    ("x_{ab}", "x_a b", "other"),
    ("\\frac{a}{b}c", "\\frac{a}{bc}", "other"),
    ("e^{x+1}", "e^x+1", "other"),
    ("\\sqrt{ab}", "\\sqrt{a}b", "other"),
    # dropped content must not pass as layout
    (r"\left. x \right.", r"x \right", "other"),
    (r"\left\{ a \right.", r"\left\{ a \right", "other"),
    (r"= \ln (2)", r"= \ln", "other"),
    (r"y = f (3)", r"y = f", "other"),
    # cell structure of inner environments
    (r"\begin{pmatrix}a & b\end{pmatrix}", r"\begin{pmatrix}ab\end{pmatrix}", "other"),
    (r"\begin{pmatrix}a & b & c\end{pmatrix}", r"\begin{pmatrix}a & bc\end{pmatrix}", "other"),
    (r"\begin{array}{cc} a & b \\ c & d \end{array}", r"\begin{array}{cc} ab \\ cd \end{array}", "other"),
    (
        r"\begin{cases} x & y>0 \\ 0 & y\leq 0 \end{cases}",
        r"\begin{cases} x y>0 \\ 0 y\leq 0 \end{cases}",
        "other",
    ),
    (
        r"\begin{aligned} a &= b \\ c &= d \end{aligned}",
        r"\begin{aligned} a = b \\ c = d \end{aligned}",
        "other",
    ),
]

MUST_EQUAL = [
    ("f'", "f^{\\prime}"),
    ("f'", "f^\\prime"),
    ("f'", "f^{'}"),
    ("f''", "f^{\\prime\\prime}"),
    ("u^{\\prime}_n", "u'_n"),
    ("\\le", "\\leq"),
    ("\\ge", "\\geq"),
    ("\\ne", "\\neq"),
    ("\\bigl( x \\bigr)", "( x )"),
    ("\\Big[ x \\Big]", "[ x ]"),
    ("x_{a}", "x_a"),
    ("x^{2}", "x^2"),
    ("\\frac{a}{b}", "\\frac ab"),
    ("\\sqrt{2}", "\\sqrt2"),
    ("\\hat{x}", "\\hat x"),
    ("x^a_b", "x_b^a"),
    ("x^{2}_{n}", "x_n^2"),
    ("a\\,b", "a b"),
    ("a\\;b", "a b"),
    ("a\\quad b", "a b"),
    ("a~b", "a b"),
    ("\\left. x \\right|", "x \\right|"),
    ("$$ x+1 $$", "x+1"),
    ("\\[x+1\\]", "x+1"),
    ("\\begin{equation} x+1 \\label{eq:a} \\end{equation}", "x+1"),
    ("x+1 \\tag{3}", "x+1"),
    ("x+1 \\nonumber", "x+1"),
    ("x+1,", "x+1"),
    ("x+1 \\qquad (12a)", "x+1"),
    ("a &= b", "a = b"),
    (r"\left. x \right.", "x"),
    (r"\left\{ a \right. ,", r"\left\{ a \right."),
    (r"x+1 \quad (3)", "x+1"),
    (r"x+1 \hfill (3)", "x+1"),
    (r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"),
    (r"\begin{flalign*} x+1 \end{flalign*}", "x+1"),
    (r"\begin{math} x+1 \end{math}", "x+1"),
]


@pytest.mark.parametrize(("gt", "pred", "cls"), MUST_DIFFER)
def test_must_differ(gt, pred, cls):
    assert not equivalent(gt, pred), (canonical(gt), canonical(pred))
    assert canonical(gt) != canonical(pred)


@pytest.mark.parametrize(("a", "b"), MUST_EQUAL)
def test_must_equal(a, b):
    assert equivalent(a, b), (canonical(a), canonical(b))


def test_row_counts():
    assert len(MUST_DIFFER) >= 42
    assert len(MUST_EQUAL) >= 37


def test_alias_table_is_glyph_identical_only():
    for key in ("\\le", "\\ge", "\\prime"):
        assert key in N.ALIASES
    vals = set(N.ALIASES.values())
    for forbidden in ("\\ell", "\\varepsilon", "\\varphi", "\\bm", "\\mathrm"):
        assert forbidden not in N.ALIASES
        assert forbidden not in vals


def test_rules_version():
    assert N.NORMALIZE_RULES_VERSION == "3"


def test_old_normalizer_holes_are_closed():
    triage = pytest.importorskip("experiments.exp001_baseline.triage")

    assert triage.normalize(r"\mathbf{x}") == triage.normalize("x")
    assert triage.normalize("x_{ab}") == triage.normalize("x_a b")
    assert canonical(r"\mathbf{x}") != canonical("x")
    assert canonical("x_{ab}") != canonical("x_a b")


def test_unbalanced_braces_do_not_raise():
    assert normalize_tokens("x^{a_b") == ["x", "^", "{", "a", "_", "b"]
    assert normalize_tokens("a } b {") == ["a", "}", "b", "{"]
    assert normalize_tokens("x^a_{b") == ["x", "^", "a", "_", "{", "b"]


def test_pathological_inputs_are_fast():
    start = time.monotonic()
    normalize_tokens("x^{a} + " * 2500)
    normalize_tokens("{" * 5000 + "}" * 5000)
    normalize_tokens("{" * 5000)
    normalize_tokens("x^{" * 2000)
    assert time.monotonic() - start < 2.0


def test_idempotent():
    for gt, pred, _ in MUST_DIFFER:
        for s in (gt, pred):
            assert normalize_tokens(canonical(s)) == normalize_tokens(s)


def test_unmatched_end_does_not_raise_or_go_negative():
    assert normalize_tokens(r"\end{pmatrix} a & b") == ["\\end", "{", "p", "m", "a", "t", "r", "i", "x", "}", "a", "b"]
    assert equivalent(r"\end{x} a &= b", r"\end{x} a = b")


def test_env_list_is_shared():
    from glyphfix_math import latex_tokens, render

    assert N.DISPLAY_ENVS is latex_tokens.DISPLAY_ENVS
    assert render.DISPLAY_ENVS is latex_tokens.DISPLAY_ENVS


def test_every_display_env_strips_like_render_wrap():
    from glyphfix_math import latex_tokens, render

    for name in latex_tokens.DISPLAY_ENVS:
        for star in ("", "*"):
            src = rf"\begin{{{name}{star}}} x+1 \end{{{name}{star}}}"
            assert equivalent(src, "x+1"), src
            assert render.wrap(src) == render.wrap("x+1"), src
