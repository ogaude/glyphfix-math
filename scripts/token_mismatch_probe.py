"""Timeboxed experiment: do client-side resizes make Ollama image-token counts agree with HF?

Phase 2 found that Ollama and HF disagree on the prompt token count for part of the legacy dev crops. This
probe takes those crops and sends each one twice through the production request path: variant A pre-resized
with the HF processor window (65,536 to 16,777,216 px), variant B pre-resized with the production policy
(65,536 to 602,112 px). It compares the Ollama prompt_eval_count with the HF count. Internal diagnostic on
legacy data: the output holds ids and counts only, no LaTeX and no images.

Usage (from repo root):
  uv run python -m scripts.token_mismatch_probe --out .planning/phases/03-corpus-and-benchmark/reports/token_mismatch.json

Behaviour: one run, at most two requests per mismatching crop. If the Phase 2 sidecars are absent the report says
"not run". A model that was loaded before the run is left loaded.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
from pathlib import Path

from PIL import Image

from glyphfix_math import guard
from glyphfix_math.pixels import PIXEL_POLICY, apply_policy
from scripts.baseline_infer import CROP_OPTIONS
from scripts.spikes import common
from scripts.spikes.ollama_infer import (
    DEFAULT_HOST,
    EXIT_LOW_RAM,
    RAM_FLOOR,
    OllamaError,
    _is_loaded,
    _loaded_models,
    _post,
    infer_one,
)

HF_SIDECAR = common.SPIKE_DIR / "hf" / "HF.sidecar.jsonl"
S1_SIDECAR = common.SPIKE_DIR / "ollama" / "S1.sidecar.jsonl"
HF_POLICY = {**PIXEL_POLICY, "max_px": 16777216}
AGREE_THRESHOLD = 38


def _png(path: Path, policy: dict) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as im:
        out = apply_policy(im, policy)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), out.size


def conclusion(n: int, a: int, b: int) -> str:
    if n == 0:
        return "no mismatching crops found; nothing to test"
    if a >= AGREE_THRESHOLD * n / 41:
        return (
            f"cause: Ollama's own resize differs from HF smart_resize; pre-resizing to factor-32 dims makes counts agree "
            f"on {a}/{n} (production policy agrees on {b}/{n})"
        )
    return (
        f"cause not identified; pre-resizing agrees on {a}/{n} (production policy {b}/{n}); "
        f"Ollama Go preprocessor (image_size 768) still rescales"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="lighton-ocr3-q8-16k")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args(argv)

    if not HF_SIDECAR.exists() or not S1_SIDECAR.exists():
        common.write_json(a.out, {"n": 0, "n_requests": 0, "conclusion": "not run: Phase 2 sidecars absent"})
        print("Phase 2 sidecars absent, nothing to do")
        return 0

    hf = {r["id"]: r for r in common.read_jsonl(HF_SIDECAR)}
    s1 = {r["id"]: r for r in common.read_jsonl(S1_SIDECAR)}
    legacy, _stats = common.load_legacy("dev")
    by_id = {r["id"]: r for r in legacy}
    ids = sorted(i for i in by_id if i in hf and i in s1 and hf[i]["n_prompt_tokens"] != s1[i]["prompt_eval_count"])

    was_loaded = _is_loaded(_loaded_models(a.host), a.model)
    try:
        guard.preflight(vram_gb=6, ram_gb=2, timeout_s=900)
    except guard.GuardError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    rows: list[dict] = []
    n_requests = 0
    code = 0
    try:
        for i in ids:
            if guard.mem_available_bytes() < RAM_FLOOR:
                print("MemAvailable below 1.5 GiB, stopping", file=sys.stderr)
                code = EXIT_LOW_RAM
                break
            path = Path(by_id[i]["image_path"])
            row = {
                "id": i,
                "width": by_id[i].get("width"),
                "height": by_id[i].get("height"),
                "hf_n_prompt_tokens": hf[i]["n_prompt_tokens"],
                "hf_image_tokens": hf[i].get("image_tokens"),
                "ollama_prompt_eval_count_before": s1[i]["prompt_eval_count"],
            }
            for tag, policy in (("a", HF_POLICY), ("b", PIXEL_POLICY)):
                png, (sw, sh) = _png(path, policy)
                row[f"sent_{tag}"] = [sw, sh]
                try:
                    n_requests += 1
                    out = infer_one(a.host, a.model, png, CROP_OPTIONS, "10m", a.timeout)
                    row[f"prompt_eval_count_{tag}"] = out.get("prompt_eval_count")
                except (OllamaError, urllib.error.URLError, OSError) as e:
                    row[f"prompt_eval_count_{tag}"] = None
                    row[f"error_{tag}"] = str(e)
            rows.append(row)
    finally:
        if not was_loaded:
            try:
                _post(a.host, "/api/chat", {"model": a.model, "messages": [], "keep_alive": 0}, 60)
            except (OllamaError, urllib.error.URLError, OSError) as e:
                print(f"unload failed: {e}", file=sys.stderr)

    n_a = sum(1 for r in rows if r.get("prompt_eval_count_a") == r["hf_n_prompt_tokens"])
    n_b = sum(1 for r in rows if r.get("prompt_eval_count_b") == r["hf_n_prompt_tokens"])
    # relation between pixels sent and Ollama's count: image tokens ~ area / 1024 (+ template offset)
    for r in rows:
        for tag in ("a", "b"):
            c = r.get(f"prompt_eval_count_{tag}")
            r[f"expected_tokens_{tag}"] = (r[f"sent_{tag}"][0] // 32) * (r[f"sent_{tag}"][1] // 32)
            r[f"count_minus_expected_{tag}"] = None if c is None else c - r[f"expected_tokens_{tag}"]
    common.write_json(
        a.out,
        {
            "n": len(rows),
            "n_requests": n_requests,
            "n_match_before": 0,
            "n_match_variant_a": n_a,
            "n_match_variant_b": n_b,
            "variant_a": "factor-32 resize, 65536-16777216 px (HF processor window)",
            "variant_b": f"production pixel policy {PIXEL_POLICY['min_px']}-{PIXEL_POLICY['max_px']} px",
            "model": a.model,
            "stopped_early": bool(code),
            "conclusion": conclusion(len(rows), n_a, n_b),
            "rows": rows,
        },
    )
    print(f"n {len(rows)} variant_a {n_a} variant_b {n_b} requests {n_requests}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
