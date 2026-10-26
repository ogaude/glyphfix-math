"""Training-set helpers: fenced targets, policy-sized images, bucketed batches, LR schedule.

Usage:
  from glyphfix_math.train import data
  rows = data.read_trainset("data/train/trainset.jsonl")
  img = data.load_image(rows[0])                       # same pixels as serving (pixels.apply_policy)
  batches = data.bucket_batches([data.policy_dims(r)[0] * data.policy_dims(r)[1] for r in rows],
                                batch_size=2, seed=0, epoch=0)
  lr = data.lr_at(step, total=total, peak=2e-4)

Rules:
  * the training target is the fenced canonical label exactly as stored in the row
  * images go through pixels.apply_policy, so the processor resize is a no-op on them
  * batch order depends only on (seed, epoch), never on global random state

Known limit: bucket_batches orders by pixel count inside windows only, so batches in one window are
similar in size but the whole epoch is not globally sorted.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image

from glyphfix_math import pixels
from glyphfix_math.paths import repo_root

__all__ = [
    "REQUIRED_KEYS",
    "bucket_batches",
    "load_image",
    "lr_at",
    "policy_dims",
    "read_trainset",
    "target_text",
]

REQUIRED_KEYS = ("id", "paper_id", "image", "latex", "target_text", "group", "width", "height")


def target_text(latex: str) -> str:
    """Fenced canonical label used as the assistant turn."""
    return "```latex\n" + latex + "\n```"


def read_trainset(path: Path | str) -> list[dict]:
    """Rows of a training-set JSONL; raises ValueError naming the line on a bad row."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing keys {missing}")
            if row["target_text"] != target_text(row["latex"]):
                raise ValueError(f"{path}:{lineno}: target_text does not match the fenced latex of row {row['id']}")
            rows.append(row)
    return rows


def policy_dims(row: dict) -> tuple[int, int]:
    """(width, height) the production pixel policy gives this row."""
    p = pixels.PIXEL_POLICY
    return pixels.smart_resize_dims(int(row["width"]), int(row["height"]), p["min_px"], p["max_px"], p["factor"])


def load_image(row: dict, root: Path | str | None = None) -> Image.Image:
    """RGB image of the row after the production pixel policy."""
    base = Path(root) if root is not None else repo_root()
    with Image.open(base / row["image"]) as im:
        im.load()
        return pixels.apply_policy(im).convert("RGB")


def bucket_batches(sizes: list[int], *, batch_size: int, seed: int, epoch: int, window: int = 64) -> list[list[int]]:
    """Index batches of similar size, reproducible per (seed, epoch), covering every index once."""
    rng = random.Random(seed * 1_000_003 + epoch)
    idx = list(range(len(sizes)))
    rng.shuffle(idx)
    span = window * batch_size
    batches: list[list[int]] = []
    for start in range(0, len(idx), span):
        chunk = sorted(idx[start : start + span], key=lambda i: (sizes[i], i))
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b : b + batch_size])
    rng.shuffle(batches)
    return batches


def lr_at(step: int, *, total: int, peak: float, warmup_frac: float = 0.05, final_frac: float = 0.10) -> float:
    """Linear warmup to peak, then cosine decay to peak * final_frac at the last step."""
    warm = max(1, round(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    span = max(1, total - 1 - warm)
    progress = min(1.0, (step - warm) / span)
    floor = peak * final_frac
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
