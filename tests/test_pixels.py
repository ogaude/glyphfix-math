"""Pixel policy: dimensions, bounds, multiples of 32, no-op."""

import pytest
from PIL import Image

from glyphfix_math import pixels


@pytest.mark.parametrize("wh", [(811, 139), (3000, 3000), (100, 40), (5, 5), (4000, 60), (64, 1024)])
def test_dims_within_bounds(wh):
    w, h = pixels.smart_resize_dims(*wh, 65536, 602112)
    assert w % 32 == 0 and h % 32 == 0
    assert 65536 <= w * h <= 602112


def test_large_and_small_directions():
    w, h = pixels.smart_resize_dims(3000, 3000)
    assert w * h <= 602112
    w, h = pixels.smart_resize_dims(100, 40)
    assert w * h >= 65536


def test_in_range_rounds_only():
    assert pixels.smart_resize_dims(512, 256) == (512, 256)


def test_rejects_empty():
    with pytest.raises(ValueError):
        pixels.smart_resize_dims(0, 10)


def test_apply_policy_resizes_and_noop():
    img = Image.new("L", (100, 40), 255)
    out = pixels.apply_policy(img)
    assert out.mode == "RGB"
    assert out.size == pixels.smart_resize_dims(100, 40)
    again = pixels.apply_policy(out)
    assert again.size == out.size


def test_policy_constants():
    assert pixels.PIXEL_POLICY["max_px"] == 602112 and pixels.PIXEL_POLICY["min_px"] == 65536
