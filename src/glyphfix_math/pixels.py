"""Client-side pixel policy for equation crops.

Usage:
  from glyphfix_math.pixels import apply_policy, smart_resize_dims
  w, h = smart_resize_dims(811, 139)   # both sides multiples of 32, area in [65536, 602112]
  img = apply_policy(Image.open("crop.png"))

The Ollama projector GGUF carries no min/max pixel keys, so the 65,536-602,112 px window is applied to the
image before it is sent. Training crops and evaluation crops must go through this same function so the
model sees the same preprocessing at both ends.

Known limit: dimensions follow Qwen2-VL smart_resize with factor 32 (patch 16 x merge 2); Ollama may still
rescale the image internally (see the token mismatch report).
"""

from __future__ import annotations

import math

from PIL import Image

PIXEL_POLICY_VERSION = "1"
PIXEL_POLICY = {"min_px": 65536, "max_px": 602112, "factor": 32, "resample": "bicubic", "version": PIXEL_POLICY_VERSION}


def smart_resize_dims(w: int, h: int, min_px: int = 65536, max_px: int = 602112, factor: int = 32) -> tuple[int, int]:
    """Target (width, height): multiples of `factor`, area within [min_px, max_px], aspect ratio kept as far as possible."""
    if w <= 0 or h <= 0:
        raise ValueError(f"image size must be positive, got {w}x{h}")
    w_bar = max(factor, round(w / factor) * factor)
    h_bar = max(factor, round(h / factor) * factor)
    if w_bar * h_bar > max_px:
        beta = math.sqrt(w * h / max_px)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
    elif w_bar * h_bar < min_px:
        beta = math.sqrt(min_px / (w * h))
        w_bar = math.ceil(w * beta / factor) * factor
        h_bar = math.ceil(h * beta / factor) * factor
    return w_bar, h_bar


def apply_policy(img: Image.Image, policy: dict = PIXEL_POLICY) -> Image.Image:
    """RGB image resized to the policy dimensions (bicubic); the input size is kept when it already matches."""
    img = img.convert("RGB")
    w, h = smart_resize_dims(img.width, img.height, policy["min_px"], policy["max_px"], policy["factor"])
    if (w, h) == img.size:
        return img
    return img.resize((w, h), Image.BICUBIC)
