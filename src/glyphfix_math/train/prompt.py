"""Prompt construction and pixel-budget helpers shared by training and inference.

Usage:
  from glyphfix_math.train.prompt import prompt_text, size_kwargs, image_tokens
  text = prompt_text(processor)
  inputs = processor(text=[text], images=[img], return_tensors="pt", **size_kwargs(MIN_PIXELS, 401408))

The model is trained and served with one user turn holding only the image (empty text prompt) and
thinking disabled, so the generation prompt ends with an empty think block. Training uses this exact
string plus the target and the closing end-of-turn token.

Known limit: size_kwargs is the single place the per-call pixel cap is expressed; if the processor
API changes, adapt it there only.
"""

from __future__ import annotations

MIN_PIXELS = 65536  # processor floor: 64 image tokens
PATCH_PX = 32  # patch 16 x merge 2: one image token per 32x32 px
THINK_SUFFIX = "<think>\n\n</think>\n\n"


def build_messages() -> list[dict]:
    """The single user turn: one image, no text."""
    return [{"role": "user", "content": [{"type": "image"}]}]


def prompt_text(processor) -> str:
    """Chat-template prompt string for one image with an empty instruction and thinking disabled."""
    text = processor.apply_chat_template(build_messages(), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_SUFFIX):
        raise ValueError(f"prompt does not end with the empty think block: {text[-60:]!r}")
    return text


def size_kwargs(min_px: int, max_px: int) -> dict:
    """Processor kwargs that set the pixel floor and cap for one call."""
    return {"size": {"shortest_edge": min_px, "longest_edge": max_px}}


def image_tokens(grid_thw) -> int:
    """Number of image tokens after the 2x2 merge for a grid_thw (rows of t, h, w)."""
    rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    return sum(int(t) * int(h) * int(w) // 4 for t, h, w in rows)


def resize_to_budget(img, budget_px: int):
    """Resize keeping aspect so both sides are multiples of 32 and the area is as close to budget_px as possible without exceeding it."""
    from PIL import Image

    w, h = img.size
    scale = (budget_px / (w * h)) ** 0.5
    best = None
    for k in range(-2, 3):
        cw = max(PATCH_PX, int(w * scale / PATCH_PX) * PATCH_PX + k * PATCH_PX)
        for dk in range(-2, 3):
            ch = max(PATCH_PX, int(h * scale / PATCH_PX) * PATCH_PX + dk * PATCH_PX)
            if cw * ch > budget_px:
                continue
            aspect_err = abs((cw / ch) / (w / h) - 1)
            key = (aspect_err > 0.03, -(cw * ch), aspect_err)
            if best is None or key < best[0]:
                best = (key, cw, ch)
    if best is None:
        best = (None, PATCH_PX, PATCH_PX)
    return img.resize((best[1], best[2]), Image.BICUBIC)
