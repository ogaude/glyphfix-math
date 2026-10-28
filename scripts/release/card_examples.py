"""Build the example images for the model card: the paper's own crop (the goal) above renders of what the base model
and the tuned model wrote.

Usage: uv run python -m scripts.release.card_examples [--out release/model/assets]

The reference crop is the benchmark image (a CC BY or CC0 paper, credited in the card). The two predictions are
compiled with the project renderer, so they show exactly what the scorer compared. Examples are fixed by id: a
curated mix that includes one the tuned model still gets wrong.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glyphfix_math import render

BASELINE_RECORDS = Path("data/benchmark/baseline/score/records.jsonl")
TUNED_RECORDS = Path("data/benchmark/acceptance/scores/crops_tuned")
CROP_DIR = Path("data/benchmark/images")
ATTRIBUTION = Path("release/dataset/ATTRIBUTION.jsonl")
OUT_DIR = Path("release/model/assets")

# (id, class shown in the card, one line saying what the example shows)
EXAMPLES = (
    ("8424e5e5e48206f9", "prime_l", "The base model made two errors: it read the triple prime as the Roman numeral III and dropped the minus on x_c; this model got both right."),
    ("0bfbeba7077c27ed", "decoration", "The base model lost the sans-serif style of A and B; this model kept it."),
    ("18395b29fe0f19ef", "greek_latin", "The base model read the Greek letter chi as the Latin x; this model read it correctly."),
    ("01d56f4f68d6d327", "decoration", "A miss: both models lose the bold on the Greek letter (the reference is `\\boldsymbol{\\theta}`)."),
)
WIDTH = 860
PAD = 14
LABEL_H = 34
BG = (255, 255, 255)
GOOD, BAD, GREY = (22, 128, 61), (190, 30, 45), (60, 60, 60)
_FONTS = ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf")
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]|[0-9]|\S")
_NOISE = {"{", "}", "\\left", "\\right", "\\,", "\\!", "\\;"}
_SAME = {"\\le": "\\leq", "\\ge": "\\geq"}


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
    scale = min((WIDTH - 2 * PAD) / img.width, 2.0)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)


def _bitmap_to_image(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    return img.convert("RGB")


def _font(size: int) -> ImageFont.ImageFont:
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _tokens(latex: str) -> list[str]:
    toks = [t for t in _TOKEN.findall(latex) if t not in _NOISE]
    return [_SAME.get(t, t) for t in toks]


def differs(reference: str, pred: str, limit: int = 3) -> str:
    """The few token changes that turn the paper's LaTeX into the prediction, e.g. `\\mathsf A` -> `A`."""
    a, b = _tokens(reference), _tokens(pred)
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    counts: dict[str, int] = {}
    for _, i1, i2, j1, j2 in ops:
        key = f"{' '.join(a[i1:i2]) or '(nothing)'}  ->  {' '.join(b[j1:j2]) or '(nothing)'}"
        counts[key] = counts.get(key, 0) + 1
    parts = [k + (f"  (x{n})" if n > 1 else "") for k, n in counts.items()]
    return "; ".join(parts[:limit]) + (" ..." if len(parts) > limit else "") if parts else "(same tokens)"


def compose(rows: list[tuple[str, Image.Image, bool | None, str]]) -> Image.Image:
    """rows: (label, image, matches the paper or None for the paper itself, one line saying what differs)."""
    fitted = [(label, _fit(img), ok, note) for label, img, ok, note in rows]
    note_h = 22
    height = PAD + sum(LABEL_H + im.height + (note_h if note else 0) + PAD for _, im, _, note in fitted)
    canvas = Image.new("RGB", (WIDTH, height), BG)
    d = ImageDraw.Draw(canvas)
    big, small = _font(22), _font(15)
    y = PAD
    for label, im, ok, note in fitted:
        colour = GREY if ok is None else (GOOD if ok else BAD)
        mark = "" if ok is None else ("   matches the goal" if ok else "   does not match the goal")
        mark_glyph = "" if ok is None else ("OK" if ok else "X")
        d.text((PAD, y), label, fill=colour, font=big)
        if mark:
            x = PAD + d.textlength(label, font=big) + 10
            d.text((x, y + 4), f"[{mark_glyph}]{mark}", fill=colour, font=small)
        y += LABEL_H
        d.rectangle((PAD - 4, y - 4, WIDTH - PAD + 4, y + im.height + 2), outline=colour, width=2)
        canvas.paste(im, (PAD, y))
        y += im.height
        if note:
            d.text((PAD, y + 4), note, fill=colour, font=small)
            y += note_h
        y += PAD
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
            ("Goal: the paper's LaTeX", Image.open(crop).convert("RGB"), None, ""),
            ("Base model", _bitmap_to_image(render.load_bitmap(rb)), bool(b["render_identical"]),
             "goal -> base: " + differs(b["gt_latex"], b["pred_latex"])),
            ("This model", _bitmap_to_image(render.load_bitmap(rt)), bool(t["render_identical"]),
             "goal -> this model: " + differs(b["gt_latex"], t["pred_latex"])),
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
