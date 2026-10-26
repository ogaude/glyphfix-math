import time

import pytest

from glyphfix_math import errors as E
from glyphfix_math.errors import CLASSES, classify, notation_present
from glyphfix_math.normalize import equivalent
from tests.test_normalize_adversarial import MUST_DIFFER, MUST_EQUAL


@pytest.mark.parametrize(
    ("gt", "pred", "expected"),
    [
        ("f'", "fl", {"prime_l"}),
        ("f'", "f\\ell", {"prime_l"}),
        ("f'", "f1", {"prime_l"}),
        ("f'", "f", {"dropped_prime"}),
        ("f''", "f'", {"dropped_prime"}),
        ("f", "f'", {"other"}),
        ("\\upsilon", "v", {"greek_latin"}),
        ("v", "\\nu", {"greek_latin"}),
        ("\\alpha", "\\beta", {"other"}),
        ("\\mathbf{x}", "x", {"decoration"}),
        ("\\bar{x}+1", "x+1", {"decoration"}),
        ("\\mathbf{x}", "\\bm{x}", {"decoration"}),
        ("I", "1", {"i_1"}),
        ("x_l", "x_1", {"i_1"}),
        ("f'(\\upsilon)", "f(v)", {"dropped_prime", "greek_latin"}),
    ],
)
def test_classify(gt, pred, expected):
    assert classify(gt, pred) == expected


def test_equivalent_gives_empty_set():
    assert classify("f'", "f^{\\prime}") == set()
    for a, b in MUST_EQUAL:
        assert equivalent(a, b)
        assert classify(a, b) == set(), (a, b)


@pytest.mark.parametrize(("gt", "pred", "cls"), MUST_DIFFER)
def test_adversarial_rows_get_expected_class(gt, pred, cls):
    got = classify(gt, pred)
    assert cls in got, (gt, pred, got)
    assert got <= set(CLASSES)


def test_notation_present():
    assert {"prime_l", "dropped_prime"} <= notation_present("f'(x)")
    assert "greek_latin" in notation_present("\\upsilon")
    assert "decoration" in notation_present("\\mathbf{x}")
    assert "i_1" in notation_present("I+1")
    assert notation_present("x+y") == {"other"}


def test_constants():
    assert E.ERROR_RULES_VERSION == "1"
    assert CLASSES == ("prime_l", "dropped_prime", "greek_latin", "decoration", "i_1", "other")
    assert {"\\alpha", "\\omega", "\\varepsilon", "\\Gamma", "\\Omega", "\\upalpha", "\\upomega", "\\Upgamma"} <= E.GREEK


def test_large_prediction_is_fast():
    gt = "x_{i} + " * 2500
    pred = gt.replace("x_{i}", "x_{j}", 100)
    start = time.monotonic()
    assert classify(gt, pred) == {"other"}
    assert time.monotonic() - start < 2.0
