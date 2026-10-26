"""Measure crop sizes under the client-side pixel policy for the benchmark and the selected train/dev crops.

Usage (from repo root):
  python scripts/crop_scale_final.py --ratio ratio.json --out reports/crop_scale_final.json

Reads image sizes from the PNG headers listed in data/benchmark/records.jsonl and data/corpus/{train,dev}/records.jsonl,
applies glyphfix_math.pixels.smart_resize_dims, and writes per-split distributions. --ratio is the output of
`corpus_pages.py crop-scale` on the frozen benchmark page set (crop ink height over page ink height). Read-only on data/.
Exit codes: 0 ok, 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from PIL import Image

from glyphfix_math.pixels import PIXEL_POLICY, smart_resize_dims

SPLITS = {
    "benchmark": "data/benchmark/records.jsonl",
    "train": "data/corpus/train/records.jsonl",
    "dev": "data/corpus/dev/records.jsonl",
}
TOKEN_BUDGET = PIXEL_POLICY["max_px"] // (PIXEL_POLICY["factor"] ** 2)


def pctl(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(int(q * (len(s) - 1) + 0.5), len(s) - 1)]


def dist(vals: list[float]) -> dict:
    return {
        "min": min(vals),
        "p5": pctl(vals, 0.05),
        "median": statistics.median(vals),
        "p95": pctl(vals, 0.95),
        "max": max(vals),
        "mean": statistics.fmean(vals),
    }


def measure(path: Path, root: Path) -> tuple[dict, list[dict]]:
    p = PIXEL_POLICY
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        with Image.open(root / rec["image"]) as im:
            w, h = im.size
        if (w, h) != (rec.get("width"), rec.get("height")):
            raise SystemExit(f"size mismatch for {rec['id']}: file {w}x{h}, record {rec.get('width')}x{rec.get('height')}")
        tw, th = smart_resize_dims(w, h, p["min_px"], p["max_px"], p["factor"])
        area, tarea = w * h, tw * th
        action = "upscale" if area < p["min_px"] else "downscale" if area > p["max_px"] else "none"
        rows.append(
            {
                "id": rec["id"],
                "w": w,
                "h": h,
                "area": area,
                "tw": tw,
                "th": th,
                "tarea": tarea,
                "action": action,
                "scale": math.sqrt(tarea / area),
                "sx": tw / w,
                "sy": th / h,
                "aspect_change": (tw / th) / (w / h) - 1.0,
                "tokens": (tw // p["factor"]) * (th // p["factor"]),
            }
        )
    n = len(rows)
    sc = [r["scale"] for r in rows]
    out = {
        "n": n,
        "orig_width_px": dist([r["w"] for r in rows]),
        "orig_height_px": dist([r["h"] for r in rows]),
        "orig_area_px": dist([r["area"] for r in rows]),
        "sent_area_px": dist([r["tarea"] for r in rows]),
        "scale_factor": dist(sc),
        "image_tokens": dist([r["tokens"] for r in rows]),
        "action_counts": {a: sum(1 for r in rows if r["action"] == a) for a in ("upscale", "none", "downscale")},
        "upscaled_gt_2x": sum(1 for r in rows if r["scale"] > 2.0),
        "upscaled_gt_3x": sum(1 for r in rows if r["scale"] > 3.0),
        "net_scale_gt_1_05": sum(1 for r in rows if r["scale"] > 1.05),
        "net_scale_lt_0_95": sum(1 for r in rows if r["scale"] < 0.95),
        "over_token_budget": sum(1 for r in rows if r["tokens"] > TOKEN_BUDGET),
        "sent_area_out_of_window": sum(1 for r in rows if not p["min_px"] <= r["tarea"] <= p["max_px"]),
        "sent_side_not_multiple": sum(1 for r in rows if r["tw"] % p["factor"] or r["th"] % p["factor"]),
        "aspect_change_abs_max": max(abs(r["aspect_change"]) for r in rows),
        "aspect_change_abs_gt_5pct": sum(1 for r in rows if abs(r["aspect_change"]) > 0.05),
        "aspect_change_abs_gt_10pct": sum(1 for r in rows if abs(r["aspect_change"]) > 0.10),
        "max_sent_side_px": max(max(r["tw"], r["th"]) for r in rows),
    }
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root)
    result: dict = {
        "pixel_policy": PIXEL_POLICY,
        "token_budget": TOKEN_BUDGET,
        "token_definition": "one image token per 32x32 px of the image sent to the model",
        "action_definition": "upscale if the area is under min_px, downscale if over max_px, none otherwise (rounding to multiples of 32 only)",
        "splits": {},
    }
    worst: dict[str, list[dict]] = {}
    for name, rel in SPLITS.items():
        path = root / rel
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2
        result["splits"][name], rows = measure(path, root)
        worst[name] = sorted(rows, key=lambda r: -r["scale"])[:5] + sorted(rows, key=lambda r: r["scale"])[:5]
    result["extremes"] = {k: [{x: r[x] for x in ("id", "w", "h", "tw", "th", "scale", "tokens", "action")} for r in v] for k, v in worst.items()}
    ratio = json.loads(Path(a.ratio).read_text(encoding="utf-8"))
    detail = [d for d in ratio["detail"] if d.get("ratio")]
    result["page_level_ratio"] = {
        "method": "scripts/corpus_pages.py crop-scale: crop ink height (300 DPI crop) / same equation ink height on the 400 DPI page raster at 2048 px longest side; run on the frozen benchmark page set (data/benchmark/pages)",
        **{k: ratio[k] for k in ("n", "median", "p10", "p90", "crop_dpi", "page_dpi", "page_longest_px")},
        "n_ratio_gt_2": sum(1 for d in detail if d["ratio"] > 2.0),
        "n_ratio_lt_0_5": sum(1 for d in detail if d["ratio"] < 0.5),
        "min": min(d["ratio"] for d in detail),
        "max": max(d["ratio"] for d in detail),
    }
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
