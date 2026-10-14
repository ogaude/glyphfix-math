import numpy as np
import pytest

from lightonocr_math import render
from lightonocr_math.corpus import canon
from lightonocr_math.corpus.glyphmatch import glyph_equal


def _page(*blobs, h=60, w=120):
    a = np.full((h, w), 255, dtype=np.uint8)
    for y, x, patch in blobs:
        a[y : y + patch.shape[0], x : x + patch.shape[1]] = np.minimum(a[y : y + patch.shape[0], x : x + patch.shape[1]], patch)
    return a


def _stem(h=20, w=4):
    return np.zeros((h, w), dtype=np.uint8)


def test_identical_pages_are_equal():
    a = _page((10, 10, _stem()), (10, 40, _stem(12, 10)))
    assert glyph_equal(a, a.copy())


def test_glyph_shifted_by_a_pixel_is_equal():
    a = _page((10, 10, _stem()), (10, 40, _stem(12, 10)))
    b = _page((10, 10, _stem()), (11, 41, _stem(12, 10)))
    assert glyph_equal(a, b)


def test_anti_aliased_edge_noise_is_equal():
    s = _stem(20, 6)
    s2 = s.copy()
    s2[:, 0] = 120  # lighter edge column, still ink
    assert glyph_equal(_page((10, 10, s)), _page((10, 10, s2)))


def test_missing_component_fails():
    a = _page((10, 10, _stem()), (10, 40, _stem(3, 3)))
    assert not glyph_equal(a, _page((10, 10, _stem())))


def test_different_size_glyph_fails():
    assert not glyph_equal(_page((10, 10, _stem(20, 4))), _page((10, 10, _stem(20, 8))))
    assert not glyph_equal(_page((10, 10, _stem(20, 4))), _page((10, 10, _stem(32, 4))))


def test_same_box_different_shape_fails():
    full = np.zeros((12, 12), dtype=np.uint8)
    hollow = full.copy()
    hollow[3:9, 3:9] = 255
    assert not glyph_equal(_page((10, 10, full)), _page((10, 10, hollow)))


def test_moved_component_fails():
    a = _page((10, 10, _stem()), (10, 40, _stem(12, 10)))
    b = _page((10, 10, _stem()), (10, 46, _stem(12, 10)))
    assert not glyph_equal(a, b)


def test_blank_or_missing_page_fails():
    blank = np.full((20, 20), 255, dtype=np.uint8)
    assert not glyph_equal(blank, blank)
    assert not glyph_equal(None, blank)


def test_thin_rule_tolerates_edge_phase():
    a = _page((40, 10, np.zeros((2, 60), dtype=np.uint8)), (10, 10, _stem()))
    b = _page((40, 10, np.zeros((3, 59), dtype=np.uint8)), (10, 10, _stem()))
    assert glyph_equal(a, b)


@pytest.mark.docker
def test_single_column_align_validates_through_glyph_tier(tmp_path):
    cases = [
        ("align", "\n\\label{eq:a}\n  \\mathcal{R}_+\\coloneqq \\{d(x)\\leq (n\\cdot v)^3\\},  \\quad \\mathcal{R}_-\\coloneqq \\{d(x)\\le 0\\}\n"),
        ("equation", "\n\\label{eq:b}\n  x'\\ \\ne y\n"),
    ]
    originals = [canon.original_for_validation(b, e) for e, b in cases]
    labels = [canon.canonicalize(b, e) for e, b in cases]
    o_res = render.render_many(originals, cache=tmp_path, wrapper="env")
    l_res = render.render_many(labels, cache=tmp_path)
    for o, lab in zip(o_res, l_res, strict=True):
        assert o.status == "rendered" and lab.status == "rendered", (o.note, lab.note)
        same = render.compare(o, lab, tmp_path)[0]
        assert same or glyph_equal(render.load_bitmap(lab, tmp_path), render.load_bitmap(o, tmp_path))


@pytest.mark.docker
def test_glyph_changes_fail_the_glyph_tier(tmp_path):
    base = "a' + \\upsilon_1 + \\bar{x} + I"
    for changed in ("a + \\upsilon_1 + \\bar{x} + I", "a' + v_1 + \\bar{x} + I", "a' + \\upsilon_1 + x + I", "a' + \\upsilon_1 + \\bar{x} + 1"):
        o = render.render_many([f"\\begin{{align*}}{base}\\end{{align*}}"], cache=tmp_path, wrapper="env")[0]
        m = render.render_many([changed], cache=tmp_path)[0]
        assert o.status == "rendered" and m.status == "rendered"
        assert not render.compare(o, m, tmp_path)[0]
        assert not glyph_equal(render.load_bitmap(m, tmp_path), render.load_bitmap(o, tmp_path)), changed
