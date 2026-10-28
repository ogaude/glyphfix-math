"""Descriptive breakdown of benchmark render-identical failures into six buckets.

The buckets are a regex heuristic over the scorer's own rows. They are reporting only: they are not part of the
frozen scorer, never decide a verdict, and must not feed training. A new training class has to be mined from the
training split. Usage:

    uv run python -m scripts.error_buckets [scores_stock_dir scores_tuned_dir]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from glyphfix_math import mining

BUCKETS = (
    "content",
    "layout / sizing",
    "glyph classes",
    "spacing",
    "alignment wrapper",
    "same text, render differs",
)
DEFAULT_DIRS = (
    "data/benchmark/acceptance/scores/crops_stock",
    "data/benchmark/acceptance/scores/crops_tuned",
)
_WRAPPER = re.compile(r"\\(?:begin|end)\{[a-z]+\*?\}|&|\\\\|[.,]")
_LAYOUT = re.compile(
    r"\\(?:left|right|bigl|bigr|big|Big|bigg|Bigg|mbox|text|textrm|mathrm|tfrac|dfrac|frac|limits|ldots|dots|cdots"
    r"|displaystyle|textstyle|rm|vert|Vert)(?![A-Za-z])|[{}()\[\]|.,$\\]"
)


def bucket(row: dict) -> str | None:
    """Bucket of one scored row, None when the row is not a render-identical failure."""
    if not mining.is_error_row(row):
        return None
    if set(row.get("classes") or []) - {"other"}:
        return "glyph classes"
    if row.get("exact") is True:
        return "spacing" if mining.spacing_only(row) else "same text, render differs"
    kinds = set()
    for gt, pred in mining.row_pairs(row):
        a, b = "".join(gt), "".join(pred)
        if not _WRAPPER.sub("", a).strip() and not _WRAPPER.sub("", b).strip():
            kinds.add("wrapper")
        elif not _LAYOUT.sub("", a).strip() and not _LAYOUT.sub("", b).strip():
            kinds.add("layout")
        else:
            kinds.add("content")
    if not kinds or "content" in kinds:
        return "content"
    return "alignment wrapper" if kinds == {"wrapper"} else "layout / sizing"


def count(rows: list[dict]) -> Counter:
    return Counter(b for b in map(bucket, rows) if b)


def load(scores_dir: str) -> list[dict]:
    with (Path(scores_dir) / "records.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def render(stock: Counter, tuned: Counter) -> str:
    lines = ["| bucket | base model | this model | change |", "|---|---|---|---|"]
    for name in BUCKETS:
        s, t = stock.get(name, 0), tuned.get(name, 0)
        change = f"{(t - s) / s:+.0%}" if s else "n/a"
        lines.append(f"| {name} | {s} | {t} | {change} |")
    lines.append(f"| total | {sum(stock.values())} | {sum(tuned.values())} | |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    stock_dir, tuned_dir = argv[:2] if len(argv) >= 2 else DEFAULT_DIRS
    print(render(count(load(stock_dir)), count(load(tuned_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
