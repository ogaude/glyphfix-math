import hashlib
import importlib.resources

import numpy as np
import pytest

from lightonocr_math import render


def test_wrap_equivalent_layouts():
    base = render.wrap("x+1")
    assert render.wrap("$$x+1$$") == base
    assert render.wrap(r"\[x+1\]") == base
    assert render.wrap(r"\begin{equation}x+1\label{a}\end{equation}") == base
    assert render.wrap(r"x+1 \tag{3}") == base
    assert render.wrap(r"x+1 \tag*{3} \nonumber") == base


def test_wrap_aligned():
    w = render.wrap(r"a &= b \\ c &= d")
    assert r"\begin{aligned}" in w and r"\end{aligned}" in w
    assert render.wrap(r"\begin{align}a&=b\\c&=d\end{align}") == render.wrap(r"a&=b\\c&=d")
    assert r"\begin{aligned}" not in render.wrap(r"\begin{pmatrix}a&b\\c&d\end{pmatrix}")


@pytest.mark.parametrize(
    "bad",
    [
        r"\input{/etc/passwd}", r"\include{x}", r"\openin", r"\read", r"\write18{ls}", r"\immediate", r"\catcode",
        r"\def\x{1}", r"\let", r"\gdef", r"\global", r"\newcommand", r"\renewcommand", r"\usepackage{x}", r"\documentclass",
        r"\end{document}", r"\begin{document}", r"\newpage", r"\clearpage", r"\shipout", r"\special", r"\csname",
        r"\makeatletter", r"\jobname", r"\endinput", r"\pagebreak", r"\vspace{100cm}", "a ^^5c b", "a % b", r"a \\% b",
    ],
)
def test_check_safe_rejects(bad):
    assert render.check_safe(bad)


@pytest.mark.parametrize("good", [r"\frac{a}{b}", r"\mathbf{x}", r"\text{if } x", r"a \hspace{1em} b", r"50\%", r"\left( x \right)"])
def test_check_safe_accepts(good):
    assert render.check_safe(good) is None


def test_check_safe_empty():
    assert render.check_safe("") == "empty"
    assert render.check_safe("  \n ") == "empty"


def test_ink_crop():
    a = np.full((100, 100), 255, dtype=np.uint8)
    a[30:40, 50:70] = 0
    c = render.ink_crop(a)
    assert c.shape == (10, 20) and int(c.max()) == 0
    assert render.ink_crop(np.full((10, 10), 255, dtype=np.uint8)) is None


def test_overflow_margin():
    a = np.full((100, 100), 255, dtype=np.uint8)
    a[40:50, 2:30] = 0
    assert render.ink_overflows(a)
    b = np.full((100, 100), 255, dtype=np.uint8)
    b[40:50, 40:60] = 0
    assert not render.ink_overflows(b)


def _page():
    a = np.full((120, 800), 255, dtype=np.uint8)
    for x in range(20, 780, 60):
        a[30:80, x : x + 14] = 0
    return a


def test_compare_arrays():
    a = np.full((100, 100), 255, dtype=np.uint8)
    assert render.compare_arrays(a, a.copy()) == (True, True)
    b = a.copy()
    b[5, 5] = 55
    assert render.compare_arrays(a, b) == (False, True)
    d = np.full((100, 104), 255, dtype=np.uint8)
    assert render.compare_arrays(a, d) == (False, False)
    assert render.compare_arrays(a, None) == (False, False)
    assert render.compare_arrays(None, a) == (False, False)


def test_compare_arrays_long_equation():
    a = _page()
    one = a.copy()
    one[5, 5] = 55
    assert render.compare_arrays(a, one) == (False, True)
    block = a.copy()
    block[100:112, 400:408] = 0
    assert render.compare_arrays(a, block) == (False, False)
    s1 = np.full((60, 60), 255, dtype=np.uint8)
    s1[10:40, 20:22] = 0
    s2 = s1.copy()
    s2[10:40, 22] = 0
    assert render.compare_arrays(s1, s2) == (False, False)
    wider = np.concatenate([a, np.full((120, 1), 255, dtype=np.uint8)], axis=1)
    assert render.compare_arrays(a, wider) == (False, True)
    shifted = np.full((121, 801), 255, dtype=np.uint8)
    shifted[1:, 1:] = a
    assert render.compare_arrays(a, shifted) == (False, True)


def test_compare_requires_rendered(tmp_path):
    a = render.RenderResult("k1", "compile-failed", None, None)
    b = render.RenderResult("k2", "rendered", (1, 1), "x")
    assert render.compare(a, b, cache=tmp_path) == (False, False)


def test_config_and_hash():
    raw = (importlib.resources.files("lightonocr_math") / "preamble.tex").read_bytes()
    assert render.PREAMBLE_SHA256 == hashlib.sha256(raw).hexdigest()
    c = render.render_config()
    assert c["preamble_sha256"] == render.PREAMBLE_SHA256
    assert c["filter_version"] == render.FILTER_VERSION == "2"
    assert c["wrap_version"] == "1" and c["dpi"] == 300 and c["image"] == render.IMAGE
    tol = c["tolerance"]
    assert tol == {
        "version": "2", "shape_px": 2, "pixel_delta": 64,
        "max_component_px": render.TOL_MAX_COMPONENT_PX, "max_diff_px": render.TOL_MAX_DIFF_PX,
    }
    assert "max_fraction" not in tol
    assert "upgreek" in raw.decode() and r"\begin{document}" not in raw.decode()


def test_cache_key_stability(monkeypatch):
    k = render.cache_key("x+1")
    assert k == render.cache_key("x+1")
    assert k != render.cache_key("x+2")
    monkeypatch.setattr(render, "WRAP_VERSION", "2")
    assert render.cache_key("x+1") != k
    monkeypatch.undo()
    monkeypatch.setattr(render, "PREAMBLE_SHA256", "0" * 64)
    assert render.cache_key("x+1") != k
    monkeypatch.undo()
    monkeypatch.setattr(render, "FILTER_VERSION", "3")
    assert render.cache_key("x+1") != k


def test_render_many_rejects_without_docker(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("docker must not be called")

    monkeypatch.setattr(render.subprocess, "run", boom)
    res = render.render_many([r"\input{/etc/passwd}", ""], cache=tmp_path)
    assert [r.status for r in res] == ["rejected", "empty"]


def test_image_is_digest_pinned_and_in_cache_key(monkeypatch):
    assert "@sha256:" in render.IMAGE and ":latest" not in render.IMAGE
    assert render.IMAGE.endswith(render.IMAGE_DIGEST)
    before = render.cache_key("x")
    monkeypatch.setattr(render, "IMAGE_DIGEST", "sha256:" + "0" * 64)
    assert render.cache_key("x") != before
