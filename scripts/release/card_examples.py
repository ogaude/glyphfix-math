"""Build the example images for the model card: the paper's own crop (the goal) above renders of what the base model
and the tuned model wrote.

Usage: uv run python -m scripts.release.card_examples [--out release/model/assets]

The reference crop is the benchmark image (a CC BY or CC0 paper, credited in the card). The two predictions are
compiled with the project renderer, so they show exactly what the scorer compared. Examples are fixed by id: a
curated mix that includes one the tuned model still gets wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glyphfix_math import render

BASELINE_RECORDS = Path("data/benchmark/baseline/score/records.jsonl")
TUNED_RECORDS = Path("data/benchmark/acceptance/scores/crops_tuned")
CROP_DIR = Path("data/benchmark/images")
ATTRIBUTION = Path("release/dataset/ATTRIBUTION.jsonl")
OUT_DIR = Path("release/model/assets")

# (id, class shown in the card, one line saying what the example shows)
EXAMPLES = (
    ("0bfbeba7077c27ed", "decoration", "The base model lost the sans-serif style of A and B; this model kept it."),
    ("18395b29fe0f19ef", "greek_latin", "The base model read the Greek letter chi as the Latin x; this model read it correctly."),
    ("01d56f4f68d6d327", "decoration", "A miss: both models lose the bold on the Greek letter (the reference is `\\boldsymbol{\\theta}`)."),
)
WIDTH = 720
PAD = 12
LABEL_H = 18
BG = (255, 255, 255)


class ExampleError(RuntimeError):
    pass


def _records(path: Path) -> dict[str, dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out: dict[str, dict] = {}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def _fit(img: Image.Image) -> Image.Image:
    """Scale to the common width without enlarging more than 1.6x, so every row stays readable."""
    scale = min((WIDTH - 2 * PAD) / img.width, 1.6)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)


def _bitmap_to_image(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    return img.convert("RGB")


def compose(rows: list[tuple[str, Image.Image]]) -> Image.Image:
    fitted = [(label, _fit(img)) for label, img in rows]
    height = PAD + sum(LABEL_H + im.height + PAD for _, im in fitted)
    canvas = Image.new("RGB", (WIDTH, height), BG)
    d = ImageDraw.Draw(canvas)
    y = PAD
    for label, im in fitted:
        d.text((PAD, y), label, fill=(90, 90, 90))
        y += LABEL_H
        canvas.paste(im, (PAD, y))
        y += im.height + PAD
    return canvas


def attribution(arxiv_id: str) -> dict:
    for line in ATTRIBUTION.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["arxiv_id"] == arxiv_id:
            return {k: r[k] for k in ("arxiv_id", "arxiv_url", "authors", "title", "license_spdx", "license_uri")}
    raise ExampleError(f"no attribution row for arXiv {arxiv_id}")


def build(out: Path = OUT_DIR) -> list[dict]:
    base, tuned = _records(BASELINE_RECORDS), _records(TUNED_RECORDS)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for rid, cls, note in EXAMPLES:
        if rid not in base or rid not in tuned:
            raise ExampleError(f"example {rid} is not in the benchmark records")
        cases.append((rid, cls, note, base[rid], tuned[rid]))
    latex = [x for _, _, _, b, t in cases for x in (b["pred_latex"], t["pred_latex"])]
    results = render.render_many(latex)
    manifest = []
    for n, (rid, cls, note, b, t) in enumerate(cases):
        rb, rt = results[2 * n], results[2 * n + 1]
        if rb.status != "rendered" or rt.status != "rendered":
            raise ExampleError(f"example {rid}: a prediction did not compile ({rb.status}, {rt.status})")
        crop = CROP_DIR / b["paper_id"] / f"{rid}.png"
        rows = [
            ("Goal: the paper", Image.open(crop).convert("RGB")),
            ("Base model", _bitmap_to_image(render.load_bitmap(rb))),
            ("This model", _bitmap_to_image(render.load_bitmap(rt))),
        ]
        name = f"example_{n + 1}_{cls}.png"
        compose(rows).save(out / name, optimize=True)
        manifest.append(
            {
                "file": name,
                "id": rid,
                "class": cls,
                "note": note,
                "tuned_correct": bool(t["render_identical"]),
                "base_correct": bool(b["render_identical"]),
                "source": attribution(b["paper_id"]),
            }
        )
    (out / "examples.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    try:
        m = build(args.out)
    except (ExampleError, render.RenderInfrastructureError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out), "examples": len(m)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
