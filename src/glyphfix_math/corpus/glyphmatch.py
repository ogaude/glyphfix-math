"""Glyph-level bitmap equivalence for canon validation: same ink components, positions free to snap by a pixel.

Usage:
  from glyphfix_math.corpus.glyphmatch import glyph_equal
  glyph_equal(bitmap_a, bitmap_b)   # True when both pages hold the same glyph shapes at the same places (+-2 px)

Why it exists: the paper's verbatim starred environment (align*, gather*) and the scorer wrapper (display math, aligned)
place the same glyphs at slightly different sub-pixel offsets. The rasterizer snaps individual glyph origins and
anti-aliases rules at those offsets, so two renders of the same formula can differ in a few pixels around one glyph while
every glyph is the same. Exact bitmap equality (the first tier, kept) calls that a mismatch; this second tier does not.

The test is deliberately strict at glyph level, so any glyph-bearing edit still fails it:
  * the ink (pixels darker than render.INK_THRESHOLD) is split into 8-connected components in both bitmaps;
  * the component counts must be equal and every component of A must pair with a distinct component of B whose
    bounding box has the same height and width (+-1 px) and whose top-left corner is within POS_TOL_PX;
  * a paired component must have the same gray values at its best +-1 px alignment: at most max(2, 1%) of its ink pixels
    may differ by more than PIXEL_DELTA (3% for components with at least LARGE_INK_PX ink pixels, the stretched
    delimiters whose strokes are anti-aliased at a different phase);
  * a thin rule (fraction bar, overline, underline: at most RULE_THICK_PX thick, at least RULE_MIN_LEN_PX long) only needs
    the same extent within RULE_TOL_PX, since its edges are anti-aliased at a sub-pixel phase.
A prime read as the letter l, upsilon as v, a missing bar, bold versus regular, I versus 1 all change a component's size
or shape, a component count, or a position by more than a pixel, and fail.

Known limits: glyphs that touch in one render and not in the other change the component count and fail (conservative);
pages of different size by more than 2 px in either axis fail.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from glyphfix_math.render import INK_THRESHOLD

POS_TOL_PX = 2
SIZE_TOL_PX = 1
PIXEL_DELTA = 128
MAX_BAD_FRACTION = 0.01
LARGE_BAD_FRACTION = 0.03
LARGE_INK_PX = 400
MIN_BAD_PX = 2
RULE_THICK_PX = 4
RULE_MIN_LEN_PX = 8
RULE_TOL_PX = 2
_EIGHT = np.ones((3, 3), dtype=bool)


def _components(arr: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """[(top, left, gray patch with non-component pixels set to 255)] for each 8-connected ink component."""
    a = np.asarray(arr, dtype=np.uint8)
    labels, _n = ndimage.label(a < INK_THRESHOLD, structure=_EIGHT)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        patch = np.where(labels[sl] == i, a[sl], 255).astype(np.uint8)
        out.append((sl[0].start, sl[1].start, patch))
    return out


def _patch_close(p: np.ndarray, q: np.ndarray) -> bool:
    """True when p and q (gray, 255 = background) agree at some +-1 px alignment."""
    ink = int((p < INK_THRESHOLD).sum())
    frac = LARGE_BAD_FRACTION if ink >= LARGE_INK_PX else MAX_BAD_FRACTION
    allowed = max(MIN_BAD_PX, int(frac * ink))
    h, w = max(p.shape[0], q.shape[0]) + 2, max(p.shape[1], q.shape[1]) + 2
    base = np.full((h, w), 255, dtype=np.int16)
    base[1 : 1 + p.shape[0], 1 : 1 + p.shape[1]] = p
    for dy in (0, -1, 1):
        for dx in (0, -1, 1):
            other = np.full((h, w), 255, dtype=np.int16)
            other[1 + dy : 1 + dy + q.shape[0], 1 + dx : 1 + dx + q.shape[1]] = q
            if int((np.abs(base - other) > PIXEL_DELTA).sum()) <= allowed:
                return True
    return False


def _is_rule(shape: tuple[int, int]) -> bool:
    return min(shape) <= RULE_THICK_PX and max(shape) >= RULE_MIN_LEN_PX


def glyph_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """True when two ink-cropped bitmaps hold the same glyph components, up to pixel snapping of their positions."""
    if a is None or b is None:
        return False
    if abs(a.shape[0] - b.shape[0]) > POS_TOL_PX or abs(a.shape[1] - b.shape[1]) > POS_TOL_PX:
        return False
    ca, cb = _components(a), _components(b)
    if len(ca) != len(cb) or not ca:
        return False
    geo_b = np.array([(t, l, p.shape[0], p.shape[1]) for t, l, p in cb], dtype=np.int32)
    taken = np.zeros(len(cb), dtype=bool)
    rule_b = np.array([_is_rule(p.shape) for _, _, p in cb], dtype=bool)
    for t, l, p in sorted(ca, key=lambda c: (c[1], c[0])):
        rule = _is_rule(p.shape)
        tol = RULE_TOL_PX if rule else SIZE_TOL_PX
        ok = (
            ~taken
            & (rule_b == rule)
            & (np.abs(geo_b[:, 0] - t) <= POS_TOL_PX)
            & (np.abs(geo_b[:, 1] - l) <= POS_TOL_PX)
            & (np.abs(geo_b[:, 2] - p.shape[0]) <= tol)
            & (np.abs(geo_b[:, 3] - p.shape[1]) <= tol)
        )
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return False
        order = idx[np.argsort(np.abs(geo_b[idx, 0] - t) + np.abs(geo_b[idx, 1] - l), kind="stable")]
        match = next((int(j) for j in order if rule or _patch_close(p, cb[int(j)][2])), None)
        if match is None:
            return False
        taken[match] = True
    return True
