"""Ollama inference step for the corpus and benchmark suites (inference only; scoring is a separate step).

Sends each record's image to POST /api/chat (empty prompt, one image, think false, temperature 0, seed 0),
writes predictions, a raw sidecar and a run manifest. Crop suites resize the image on the client with the
shared pixel policy before sending; page suites send the stored PNG. The benchmark and pages suites accept only
benchmark papers, the regression and calib suites only train/dev papers.

Usage (from repo root):
  uv run python -m scripts.baseline_infer --suite benchmark --records data/benchmark/records.jsonl \
    --model lighton-ocr3-q8-16k --out data/runs/base.pred.jsonl --sidecar data/runs/base.sidecar.jsonl
  uv run python -m scripts.baseline_infer --suite regression --records data/regression.jsonl ... --resume
  uv run python -m scripts.baseline_infer --suite pages --records data/pages/pages.jsonl ...
  uv run python -m scripts.baseline_infer --suite calib --records data/calib/pages.jsonl ...
  uv run python -m lightonocr_math.score --pred data/runs/base.pred.jsonl ...   # scoring afterwards

Behaviour: the manifest (<out stem>.manifest.json) records model digest, GGUF blob sha256s, template and prompt
hashes, options, pixel policy, scorer and render fingerprints, split rule, records hash, git commit and GPU state.
--resume skips ids already predicted and appends new rows. A model that was loaded before the run is never unloaded.

Known limits: exit 76 when MemAvailable drops below 1.5 GiB; exit 1 when any request failed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from PIL import Image

from lightonocr_math import guard, render, score, splits
from lightonocr_math.corpus import freeze
from lightonocr_math.paths import repo_root
from lightonocr_math.pixels import PIXEL_POLICY, apply_policy
from scripts.spikes.common import gpu_state, read_jsonl, sha256_file, strip_fences, write_json, write_jsonl
from scripts.spikes.ollama_infer import (
    DEFAULT_HOST,
    EXIT_LOW_RAM,
    RAM_FLOOR,
    OllamaError,
    _get,
    _is_loaded,
    _loaded_models,
    _model_digest,
    _post,
    infer_one,
)

CROP_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
PAGE_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 4096}
PAGE_POLICY = {"name": "page-2048", "dpi": 400, "longest_px": 2048}
SUITES = {
    "benchmark": {"kind": "crop", "side": "benchmark"},
    "regression": {"kind": "crop", "side": "training"},
    "calib": {"kind": "page", "side": "training"},
    "pages": {"kind": "page", "side": "benchmark"},
}
CHECKPOINT_EVERY = 25
_FROM_RE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)


def _paper(row: dict) -> str:
    return row["arxiv_id"] if "arxiv_id" in row else row["paper_id"]


def _row_id(suite: str, row: dict) -> str:
    return row["page_id"] if SUITES[suite]["kind"] == "page" else row["id"]


def load_suite(suite: str, rows: list[dict]) -> list[dict]:
    """Rows of `suite`, after checking that their papers belong on the suite's side of the split."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}")
    papers = [_paper(r) for r in rows]
    if SUITES[suite]["side"] == "benchmark":
        bad = sorted({p for p in papers if not splits.is_benchmark(p)})
        if bad:
            raise ValueError(f"suite {suite}: {len(bad)} paper(s) are not benchmark papers: {', '.join(bad[:10])}")
    else:
        splits.assert_disjoint(papers, context=f"baseline_infer:{suite}")
    return rows


def model_blobs(host: str, model: str) -> dict:
    """GGUF blob sha256s (text, projector) from the Modelfile FROM lines, plus the template hash."""
    info = _post(host, "/api/show", {"model": model}, 60)
    blobs = []
    for path in _FROM_RE.findall(info.get("modelfile", "")):
        name = Path(path).name
        if name.startswith("sha256-"):
            blobs.append(name[len("sha256-"):])
    out: dict = {"text": blobs[0] if blobs else None, "projector": blobs[1] if len(blobs) > 1 else None}
    out["template_sha256"] = hashlib.sha256(info.get("template", "").encode("utf-8")).hexdigest()
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False
        ).stdout.strip() or None
    except OSError:
        return None


def build_manifest(
    *, suite: str, model: str, host: str, options: dict, records_path: Path, ids_file: Path | None, was_loaded: bool,
    gpu_before: dict, gpu_after: dict, n_ok: int, n_error: int, n_skipped: int, wall: float,
) -> dict:
    try:
        version = _get(host, "/api/version").get("version")
    except (urllib.error.URLError, OSError, ValueError):
        version = None
    try:
        blobs = model_blobs(host, model)
    except (OllamaError, urllib.error.URLError, OSError):
        blobs = {"text": None, "projector": None, "template_sha256": None}
    template_sha = blobs.pop("template_sha256")
    return {
        "suite": suite,
        "model": model,
        "model_digest": _model_digest(host, model),
        "gguf_blobs": blobs,
        "template_sha256": template_sha,
        "prompt": "",
        "prompt_sha256": hashlib.sha256(b"").hexdigest(),
        "ollama_version": version,
        "options": dict(options),
        "endpoint": "/api/chat",
        "think": False,
        "pixel_policy": dict(PIXEL_POLICY) if SUITES[suite]["kind"] == "crop" else dict(PAGE_POLICY),
        "scorer_version": score.SCORER_VERSION,
        "scorer_fingerprint": score.SCORER_FINGERPRINT,
        "render_config": render.render_config(),
        "split_rule": splits.rule_info(),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "ids_file_sha256": sha256_file(ids_file) if ids_file else None,
        "git_commit": _git_commit(),
        "model_loaded_before": was_loaded,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "n_ok": n_ok,
        "n_error": n_error,
        "n_skipped_resume": n_skipped,
        "wall_seconds": round(wall, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def crop_png(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Crop at `path` resized with the pixel policy, re-encoded as PNG; returns (bytes, sent size)."""
    with Image.open(path) as im:
        out = apply_policy(im)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), out.size


def _write(out: Path, sidecar: Path, preds: list[dict], side: list[dict]) -> None:
    write_jsonl(out, preds)
    write_jsonl(sidecar, side)


def _assert_frozen_inputs(suite: str, records: Path, *, allow_unfrozen: bool = False) -> None:
    """Fail closed for benchmark/pages suites unless the records are the frozen ones.

    The records count as frozen when they are the repo's file or byte-identical to the pinned digest, whatever their
    path; the freeze files are then verified too. Any other records file (a copy, an edit, a subset) would be reported
    as the benchmark suite while not being it, so it needs the explicit allow_unfrozen override.
    """
    bench = repo_root() / "data" / "benchmark"
    records = Path(records)
    if suite == "benchmark":
        if records.resolve() == (bench / "records.jsonl").resolve() or freeze.records_are_frozen(records):
            freeze.assert_benchmark_frozen(bench)
        elif not allow_unfrozen:
            raise freeze.FreezeError(f"{records} is not the frozen benchmark records (digest differs); pass --allow-unfrozen-records to run on it anyway")
    elif suite == "pages":
        if records.resolve() == (bench / "pages" / "pages.jsonl").resolve() or freeze.pages_are_frozen(records):
            freeze.assert_benchmark_frozen(bench)
            freeze.assert_pages_frozen(bench)
        elif not allow_unfrozen:
            raise freeze.FreezeError(f"{records} is not the frozen pages.jsonl (digest differs); pass --allow-unfrozen-records to run on it anyway")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--model", default="lighton-ocr3-q8-16k")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-unfrozen-records", action="store_true", help="run the benchmark/pages suite on records that are not the frozen ones (never for published results)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args(argv)

    try:
        _assert_frozen_inputs(a.suite, a.records, allow_unfrozen=a.allow_unfrozen_records)
    except splits.LeakageError as e:
        print(f"FROZEN: {e}", file=sys.stderr)
        return 1

    kind = SUITES[a.suite]["kind"]
    options = CROP_OPTIONS if kind == "crop" else PAGE_OPTIONS
    rows = read_jsonl(a.records)
    if a.ids_file:
        wanted = {ln.strip() for ln in a.ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        rows = [r for r in rows if _row_id(a.suite, r) in wanted]
    rows = load_suite(a.suite, rows)
    if a.limit is not None:
        rows = rows[: a.limit]

    preds: list[dict] = []
    side: list[dict] = []
    done: set[str] = set()
    text_key = "pred_latex" if kind == "crop" else "pred_text"
    if a.resume and a.out.exists():
        preds = [p for p in read_jsonl(a.out) if text_key in p]
        done = {p["id"] for p in preds}
        if a.sidecar.exists():
            side = [s for s in read_jsonl(a.sidecar) if s["id"] in done]
    n_skipped = len(done)

    before = gpu_state()
    was_loaded = _is_loaded(_loaded_models(a.host), a.model)
    try:
        guard.preflight(vram_gb=6, ram_gb=2, timeout_s=900)
    except guard.GuardError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    t0 = time.time()
    n_ok = n_err = 0
    code = 0
    try:
        for row in rows:
            rid = _row_id(a.suite, row)
            if rid in done:
                continue
            if guard.mem_available_bytes() < RAM_FLOOR:
                print("MemAvailable below 1.5 GiB, stopping", file=sys.stderr)
                code = EXIT_LOW_RAM
                break
            img_path = repo_root() / row["image"]
            if kind == "crop":
                base = {"id": rid, "paper_id": _paper(row), "gt_latex": row["latex"] if "latex" in row else row["gt_latex"]}
            else:
                base = {"id": rid, "paper_id": _paper(row)}
            try:
                if kind == "crop":
                    png, (sw, sh) = crop_png(img_path)
                else:
                    png = img_path.read_bytes()
                    sw = sh = None
                out = infer_one(a.host, a.model, png, options, "10m", a.timeout)
            except (OllamaError, urllib.error.URLError, OSError) as e:
                preds.append({**base, "pred_error": str(e)})
                n_err += 1
                continue
            raw = out["message"]["content"]
            preds.append({**base, text_key: strip_fences(raw) if kind == "crop" else raw})
            side.append(
                {
                    "id": rid,
                    "model": a.model,
                    "raw_text": raw,
                    "done_reason": out.get("done_reason"),
                    "prompt_eval_count": out.get("prompt_eval_count"),
                    "eval_count": out.get("eval_count"),
                    "total_duration": out.get("total_duration"),
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "sent_width": sw,
                    "sent_height": sh,
                    "width": row.get("width"),
                    "height": row.get("height"),
                }
            )
            n_ok += 1
            if (n_ok + n_err) % CHECKPOINT_EVERY == 0:
                _write(a.out, a.sidecar, preds, side)
    finally:
        if not was_loaded:
            try:
                _post(a.host, "/api/chat", {"model": a.model, "messages": [], "keep_alive": 0}, 60)
            except (OllamaError, urllib.error.URLError, OSError) as e:
                print(f"unload failed: {e}", file=sys.stderr)
        _write(a.out, a.sidecar, preds, side)
    write_json(
        a.out.with_name(a.out.stem + ".manifest.json"),
        build_manifest(
            suite=a.suite, model=a.model, host=a.host, options=options, records_path=a.records, ids_file=a.ids_file,
            was_loaded=was_loaded, gpu_before=before, gpu_after=gpu_state(), n_ok=n_ok, n_error=n_err,
            n_skipped=n_skipped, wall=time.time() - t0,
        ),
    )
    print(f"ok {n_ok} error {n_err} skipped {n_skipped}")
    if code:
        return code
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
