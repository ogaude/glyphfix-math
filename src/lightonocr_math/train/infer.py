"""Model loading and greedy single-crop generation with token accounting.

Usage:
  model, processor = load_base()
  res = generate_one(model, processor, pil_image, min_px=65536, max_px=401408)

The revision is pinned and loading is offline when HF_HUB_OFFLINE is set. Device "cuda" without a GPU
raises instead of falling back to CPU.
"""

from __future__ import annotations

import os

from lightonocr_math.train.prompt import image_tokens, prompt_text, size_kwargs

MODEL_ID = "lightonai/LightOnOCR-3-4B"
REVISION = "b71010f095dc3735de7ee6b9969bff8f8f50b39a"


def load_base(path_or_id: str = MODEL_ID, *, device: str = "cuda", dtype: str = "bfloat16", linear_patch_embed: bool = True):
    """Return (model, processor) for the pinned checkpoint, bf16, sdpa attention.

    The vision patch embed runs as a matmul (identical result, avoids the per-shape MIOpen Find).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no GPU is available; refusing to fall back to CPU")
    offline = bool(os.environ.get("HF_HUB_OFFLINE"))
    rev = REVISION if path_or_id == MODEL_ID else None
    model = AutoModelForImageTextToText.from_pretrained(
        path_or_id, revision=rev, dtype=getattr(torch, dtype), attn_implementation="sdpa", local_files_only=offline
    ).to(device)
    if linear_patch_embed:
        from lightonocr_math.train.patch_embed import linearize_patch_embed

        linearize_patch_embed(model, require=True)
    processor = AutoProcessor.from_pretrained(path_or_id, revision=rev, local_files_only=offline)
    return model, processor


def generate_one(model, processor, image, *, min_px: int, max_px: int, max_new_tokens: int = 1024) -> dict:
    """Greedy generation for one crop; returns text and token accounting."""
    import torch

    inputs = processor(text=[prompt_text(processor)], images=[image], return_tensors="pt", **size_kwargs(min_px, max_px))
    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    n_prompt = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new = out[0, n_prompt:]
    n_new = int(new.shape[0])
    grid = inputs["image_grid_thw"].tolist()
    return {
        "text": processor.tokenizer.decode(new, skip_special_tokens=True),
        "n_prompt_tokens": n_prompt,
        "image_tokens": image_tokens(grid),
        "grid_thw": grid,
        "resized_hw": [grid[0][1] * 16, grid[0][2] * 16],
        "finish": "length" if n_new >= max_new_tokens else "eos",
        "n_new_tokens": n_new,
    }
