import pytest

from glyphfix_math.corpus import flags as flags_mod
from glyphfix_math.corpus.flags import FLAGS, flags


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        ("f'(x)", "prime"),
        (r"x^{\prime}", "prime"),
        (r"\alpha", "greek"),
        (r"\bar{x}", "decoration"),
        (r"\mathbf{v}", "decoration"),
        ("I_n", "i_1"),
        (r"\mathbb{1}", "i_1"),
        ("l1", "i_1"),
        (r"\ell 1", "i_1"),
        ("1|", "i_1"),
        (r"\bm{1}", "i_1"),
    ],
)
def test_flag_present(latex, expected):
    assert expected in flags(latex)


@pytest.mark.parametrize(
    ("latex", "absent"),
    [
        (r"\mathrm{Id}", "i_1"),
        (r"\text{If } x", "i_1"),
        (r"\operatorname{I}", "i_1"),
        ("x_{1}", "i_1"),
        ("x_1 + 1", "i_1"),
        ("11", "i_1"),
        (r"\mathbb{11}", "i_1"),
    ],
)
def test_flag_absent(latex, absent):
    assert absent not in flags(latex)


def test_plain():
    assert flags("x + 1") == ["plain"]
    assert "plain" not in flags("f'")


def test_order_follows_flags_constant():
    out = flags(r"\mathbf{I}' + \alpha")
    assert out == [f for f in FLAGS if f in out]
    assert out == ["prime", "greek", "decoration", "i_1"]


def test_prime_not_folded():
    assert "prime" in flags(r"f^{\prime\prime}")


def test_reuses_error_tables():
    from glyphfix_math import errors

    assert flags_mod.GREEK is errors.GREEK
    assert flags_mod.DECORATION_COMMANDS is errors.DECORATION_COMMANDS
