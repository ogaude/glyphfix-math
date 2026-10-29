"""Build the two SVG charts for the model card from the published numbers.

Usage: uv run python -m scripts.release.charts [--out release/model/assets]

chart_headline.svg  render-identical rate of the base model and this model on the frozen benchmark
chart_buckets.svg   render-identical failures by bucket (scripts/error_buckets.py), base model against this model
chart_flips.svg     crops right or wrong with each model, and the crops this model got wrong that the base model got right
chart_review.svg    hand review: how many fixed and broken crops change the mathematics, against spelling only
chart_pipeline.svg  how the dataset, the LoRA and the evaluation fit together (no data)

Every number is read from evidence/acceptance/results.json and the scored benchmark rows, never typed in. Bars start
at zero and carry their exact value. Each SVG paints its own surface so it reads on a light or a dark page, and
switches colours with prefers-color-scheme. The two series colours are categorical slots 1 and 2 of the reference
palette, checked with the dataviz validator (CVD delta E 24.7 light, 26.8 dark).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from scripts import error_buckets

RESULTS = Path("evidence/acceptance/results.json")
OUT_DIR = Path("release/model/assets")
REVIEW_JSON = Path(".planning/phases/06-publish/evidence/hand_review.json")
FILES = ("chart_headline.svg", "chart_buckets.svg", "chart_flips.svg", "chart_pipeline.svg", "chart_review.svg")
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
W = 760
STYLE = f"""<style>
svg{{font-family:{FONT}}}
.bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .sub{{fill:#52514e}} .grid{{stroke:#d9d8d4;stroke-width:1}}
.base{{fill:#eb6834}} .tuned{{fill:#2a78d6}} .box{{fill:#f1f0ec;stroke:#d9d8d4}} .arrow{{stroke:#52514e;fill:none;stroke-width:2}}
.good{{fill:#2a78d6}} .bad{{fill:#eb6834}} .neutral{{fill:#b9b8b2}} .mid{{fill:#8f8e88}}
@media (prefers-color-scheme:dark){{
.bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .sub{{fill:#c3c2b7}} .grid{{stroke:#3a3a38}}
.base{{fill:#d95926}} .tuned{{fill:#3987e5}} .box{{fill:#262625;stroke:#3a3a38}} .arrow{{stroke:#c3c2b7}}
.good{{fill:#3987e5}} .bad{{fill:#d95926}} .neutral{{fill:#6b6a66}} .mid{{fill:#8f8e88}}
}}
</style>"""


def _svg(height: int, title: str, desc: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        f'role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
        f'{STYLE}\n<rect class="bg" width="{W}" height="{height}" rx="8"/>\n{body}</svg>\n'
    )


def _text(x: float, y: float, s: str, cls: str, size: int, *, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>\n'


def _legend(x: float, y: float) -> str:
    out = ""
    for i, (cls, name) in enumerate((("base", "Base model"), ("tuned", "This model"))):
        cx = x + i * 130
        out += f'<rect class="{cls}" x="{cx}" y="{y - 10}" width="12" height="12" rx="2"/>\n'
        out += _text(cx + 18, y, name, "sub", 13)
    return out


def headline_chart(results: dict) -> str:
    h = results["headline"]
    n, papers = h["n"], h["n_papers"]
    x0, x1, h_bar = 150, 700, 34
    scale = (x1 - x0) / 1.0
    body = _text(24, 34, "Render-identical rate on the held-out benchmark", "ink", 17, weight=600)
    body += _text(24, 56, f"{n:,} equation crops from {papers} papers; higher is better", "sub", 13)
    body += _legend(24, 84)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + tick * scale
        body += f'<line class="grid" x1="{x}" y1="100" x2="{x}" y2="190"/>\n'
        body += _text(x, 208, f"{tick:.0%}", "sub", 12, anchor="middle")
    for i, (cls, name, rate) in enumerate((("base", "Base model", h["stock_rate"]), ("tuned", "This model", h["tuned_rate"]))):
        y = 110 + i * 44
        body += _text(x0 - 10, y + 22, name, "ink", 13, anchor="end")
        body += f'<rect class="{cls}" x="{x0}" y="{y}" width="{rate * scale:.1f}" height="{h_bar}" rx="3"/>\n'
        body += _text(x0 + rate * scale + 8, y + 22, f"{rate:.1%}", "ink", 14, weight=600)
    note = f"Difference {h['delta'] * 100:+.1f} points, paired paper-level 95% CI {h['ci_low'] * 100:+.1f} to {h['ci_high'] * 100:+.1f}"
    body += _text(24, 238, note, "sub", 13)
    desc = (
        f"Render-identical rate on {n} crops: base model {h['stock_rate']:.1%}, this model {h['tuned_rate']:.1%}, "
        f"difference {h['delta'] * 100:+.1f} points."
    )
    return _svg(252, "Render-identical rate, base model against this model", desc, body)


def buckets_chart(stock: Counter, tuned: Counter) -> str:
    x0, x1 = 230, 700
    top = max(max(stock.values(), default=0), max(tuned.values(), default=0), 1)
    scale = (x1 - x0 - 50) / top
    row_h, bar_h, y0 = 54, 18, 112
    body = _text(24, 34, "Where the remaining errors are", "ink", 17, weight=600)
    body += _text(24, 56, "Render-identical failures on the benchmark, by bucket; fewer is better", "sub", 13)
    body += _legend(24, 84)
    height = y0 + row_h * len(error_buckets.BUCKETS) + 44
    body += f'<line class="grid" x1="{x0}" y1="98" x2="{x0}" y2="{height - 34}"/>\n'
    body += _text(W - 24, 104, "Change", "sub", 12, anchor="end")
    for i, name in enumerate(error_buckets.BUCKETS):
        y = y0 + i * row_h
        body += _text(x0 - 10, y + 24, name, "ink", 13, anchor="end")
        for j, (cls, count) in enumerate((("base", stock.get(name, 0)), ("tuned", tuned.get(name, 0)))):
            by = y + j * (bar_h + 2)
            body += f'<rect class="{cls}" x="{x0}" y="{by}" width="{max(count * scale, 1):.1f}" height="{bar_h}" rx="3"/>\n'
            body += _text(x0 + count * scale + 6, by + 14, str(count), "ink", 12, weight=600)
        before, after = stock.get(name, 0), tuned.get(name, 0)
        change = f"{(after - before) / before:+.0%}".replace("-", "\u2212") if before else "n/a"
        body += _text(W - 24, y + 24, change, "ink", 13, anchor="end", weight=600)
    total = f"Total: {sum(stock.values()):,} failures for the base model, {sum(tuned.values()):,} for this model"
    body += _text(24, height - 14, total + ". Buckets are a pattern heuristic, for reporting only.", "sub", 12)
    desc = "; ".join(f"{n}: {stock.get(n, 0)} base, {tuned.get(n, 0)} this model" for n in error_buckets.BUCKETS)
    return _svg(height, "Render-identical failures by bucket", desc, body)


def flips_chart(results: dict, tuned_rows: dict[str, dict]) -> str:
    f = results["headline"]["flip"]
    parts = (
        ("neutral", "Right with both", f["both_right"]),
        ("good", "Wrong with the base model, right with this model", f["wrong_to_right"]),
        ("bad", "Right with the base model, wrong with this model", f["right_to_wrong"]),
        ("neutral", "Wrong with both", f["both_wrong"]),
    )
    total = sum(v for _, _, v in parts)
    x0, bar_w = 24, W - 48
    body = _text(24, 34, "Crop by crop: what changed", "ink", 17, weight=600)
    body += _text(24, 56, f"All {total:,} benchmark crops, by whether each model rendered them identically to the paper", "sub", 13)
    x = float(x0)
    for i, (cls, _, v) in enumerate(parts):
        w = v / total * bar_w
        body += f'<rect class="{cls}" x="{x:.1f}" y="74" width="{max(w - 2, 1):.1f}" height="34" rx="3"/>\n'
        body += _text(x + 4, 96, f"{v:,}", "ink", 13, weight=600) if w > 40 else ""
        x += w
    ly = 132
    for cls, label, v in parts:
        body += f'<rect class="{cls}" x="24" y="{ly - 10}" width="12" height="12" rx="2"/>\n'
        body += _text(44, ly, f"{label}: {v:,}", "sub", 13)
        ly += 22
    by_bucket = Counter(b for i in results["headline"]["r2w_ids"] if (b := error_buckets.bucket(tuned_rows[i])))
    top = max(by_bucket.values(), default=1)
    y0 = ly + 30
    body += _text(
        24, y0 - 6, f"The {f['right_to_wrong']} crops this model got wrong that the base model got right, by bucket", "ink", 14, weight=600
    )
    scale = (W - 24 - 230 - 60) / top
    for i, name in enumerate(error_buckets.BUCKETS):
        n = by_bucket.get(name, 0)
        y = y0 + 12 + i * 28
        body += _text(214, y + 14, name, "ink", 13, anchor="end")
        body += f'<rect class="bad" x="230" y="{y}" width="{max(n * scale, 1):.1f}" height="18" rx="3"/>\n'
        body += _text(230 + n * scale + 6, y + 14, str(n), "ink", 12, weight=600)
    height = y0 + 12 + 28 * len(error_buckets.BUCKETS) + 34
    body += _text(24, height - 14, "Bucket of this model's wrong output. Buckets are a pattern heuristic, for reporting only.", "sub", 12)
    desc = "; ".join(f"{label}: {v}" for _, label, v in parts)
    return _svg(height, "Crop by crop: what changed", desc, body)


def review_counts(review: dict) -> dict[str, dict[str, int]]:
    """Counts of real / borderline / style for the fixed and the broken crops, from the saved hand review."""
    out = {"fixed": dict.fromkeys(("real", "borderline", "style"), 0), "broken": dict.fromkeys(("real", "borderline", "style"), 0)}
    for v in review["labels"].values():
        out["fixed" if v["direction"] == "wrong_to_right" else "broken"][v["label"]] += 1
    out["fixed"]["style"] += review["mechanical_style_fixes"]["count"]
    return out


def review_chart(review: dict) -> str:
    c = review_counts(review)
    rows = (("Fixed", "base model wrong, this model right", c["fixed"]), ("Broken", "base model right, this model wrong", c["broken"]))
    parts = (("good", "real", "Changes the mathematics or a glyph"), ("mid", "borderline", "Borderline (trailing punctuation, roman or italic)"), ("neutral", "style", "Same formula, spelled differently"))
    x0, bar_w = 24, W - 48
    body = _text(24, 34, "Hand review: what changed in the crops that flipped", "ink", 17, weight=600)
    body += _text(24, 56, "Approximate: our own reading under one written rule, not part of the scored result", "sub", 13)
    y = 92
    for name, sub, counts in rows:
        total = sum(counts.values())
        body += _text(24, y, f"{name}: {total:,} crops", "ink", 14, weight=600)
        body += _text(24 + 190, y, sub, "sub", 12)
        x = float(x0)
        for cls, key, _ in parts:
            v = counts[key]
            w = v / total * bar_w
            if v:
                body += f'<rect class="{cls}" x="{x:.1f}" y="{y + 10}" width="{max(w - 2, 1):.1f}" height="34" rx="3"/>\n'
                if w > 70:
                    body += _text(x + 6, y + 32, f"{v:,} ({v / total:.0%})", "ink", 13, weight=600)
            x += w
        y += 78
    ly = y - 8
    for cls, _, label in parts:
        body += f'<rect class="{cls}" x="24" y="{ly - 10}" width="12" height="12" rx="2"/>\n'
        body += _text(44, ly, label, "sub", 13)
        ly += 22
    f, b = c["fixed"], c["broken"]
    ratio = f["real"] / b["real"]
    upper = (f["real"] + f["borderline"]) / b["real"]
    body += _text(24, ly + 14, f"Real fixes per real break: {ratio:.1f} (up to {upper:.1f} if every borderline fix is real).", "ink", 13, weight=600)
    height = ly + 40
    desc = "; ".join(f"{n}: " + ", ".join(f"{k} {v}" for k, v in cts.items()) for n, _, cts in rows)
    return _svg(height, "Hand review of the flipped crops", desc, body)


PIPELINE = (
    ("Papers", "arXiv papers under CC BY or CC0, with attribution"),
    ("Equation crops", "Display equations rendered from the LaTeX source; split by paper"),
    ("Mine errors", "Base model transcribes the training split; compare rendered bitmaps"),
    ("Train a LoRA", "Language model only, on the training split"),
    ("Evaluate", "Frozen held-out benchmark, same render comparison, both models"),
)


def pipeline_chart() -> str:
    n, gap = len(PIPELINE), 18
    bw = (W - 48 - gap * (n - 1)) / n
    body = _text(24, 34, "How the dataset, the LoRA and the evaluation fit together", "ink", 17, weight=600)
    body += _text(24, 56, "The benchmark papers are set aside and frozen before any mining or training", "sub", 13)
    for i, (name, detail) in enumerate(PIPELINE):
        x = 24 + i * (bw + gap)
        body += f'<rect class="box" x="{x:.1f}" y="80" width="{bw:.1f}" height="122" rx="6"/>\n'
        body += _text(x + bw / 2, 106, name, "ink", 14, anchor="middle", weight=600)
        words, line, y = detail.split(), "", 128
        for w in words:
            if len(line) + len(w) + 1 > 17:
                body += _text(x + bw / 2, y, line, "sub", 11, anchor="middle")
                line, y = w, y + 15
            else:
                line = f"{line} {w}".strip()
        body += _text(x + bw / 2, y, line, "sub", 11, anchor="middle")
        if i < n - 1:
            ax = x + bw + 2
            body += f'<path class="arrow" d="M{ax:.1f} 136 h{gap - 6} m-6 -5 l6 5 l-6 5"/>\n'
    return _svg(
        224, "Pipeline", "Papers, equation crops, mining the base model's errors, training a LoRA, evaluating on a frozen benchmark.", body
    )


def build(out: Path = OUT_DIR, results: dict | None = None) -> list[str]:
    results = results or json.loads(RESULTS.read_text(encoding="utf-8"))
    stock, tuned = (error_buckets.count(error_buckets.load(d)) for d in error_buckets.DEFAULT_DIRS)
    out.mkdir(parents=True, exist_ok=True)
    (out / FILES[0]).write_text(headline_chart(results), encoding="utf-8")
    (out / FILES[1]).write_text(buckets_chart(stock, tuned), encoding="utf-8")
    tuned_rows = {r["id"]: r for r in error_buckets.load(error_buckets.DEFAULT_DIRS[1])}
    (out / FILES[2]).write_text(flips_chart(results, tuned_rows), encoding="utf-8")
    (out / FILES[3]).write_text(pipeline_chart(), encoding="utf-8")
    (out / FILES[4]).write_text(review_chart(json.loads(REVIEW_JSON.read_text(encoding="utf-8"))), encoding="utf-8")
    return list(FILES)


def write_pipeline(out: Path) -> str:
    """Only the data-free pipeline diagram, for repositories that do not carry the benchmark charts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / FILES[3]).write_text(pipeline_chart(), encoding="utf-8")
    return FILES[3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    print(json.dumps({"out": str(args.out), "files": build(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
