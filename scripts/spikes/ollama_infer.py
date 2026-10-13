"""Ollama inference driver: one request shape for every arm.

Sends each legacy crop to POST /api/chat (think false, temperature 0, fixed seed, num_ctx and num_predict),
writes a scorer JSONL (id, paper_id, gt_latex, pred_latex), a parity sidecar with the raw text and token
counts, and a run manifest. A model that was not loaded before the run is unloaded at the end; a model that
someone else loaded is left alone.

Usage (from repo root):
  uv run python -m scripts.spikes.ollama_infer --model lighton-ocr3-q8-16k --split dev \
    --out data/spikes/ollama/S1.pred.jsonl --sidecar data/spikes/ollama/S1.sidecar.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from lightonocr_math import guard
from lightonocr_math.paths import repo_root
from scripts.spikes import common

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
RAM_FLOOR = int(1.5 * guard.GIB)
EXIT_LOW_RAM = 76


class OllamaError(RuntimeError):
    """A non-200 answer or an answer without message.content."""


def _post(host: str, path: str, body: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        host + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise OllamaError(f"HTTP {e.code} from {path}") from e
    if status != 200:
        raise OllamaError(f"HTTP {status} from {path}")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise OllamaError(f"unparseable answer from {path}") from e


def _get(host: str, path: str, timeout_s: float = 30) -> dict:
    with urllib.request.urlopen(urllib.request.Request(host + path), timeout=timeout_s) as resp:
        return json.loads(resp.read())


def infer_one(host: str, model: str, png_bytes: bytes, options: dict, keep_alive: str | int, timeout_s: float = 300) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "", "images": [base64.b64encode(png_bytes).decode("ascii")]}],
        "stream": False,
        "think": False,
        "options": dict(options),
        "keep_alive": keep_alive,
    }
    out = _post(host, "/api/chat", body, timeout_s)
    msg = out.get("message")
    if not isinstance(msg, dict) or "content" not in msg:
        raise OllamaError("answer has no message.content")
    return out


def _loaded_models(host: str) -> list[str]:
    try:
        return [m.get("name", "") for m in _get(host, "/api/ps").get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def _model_digest(host: str, model: str) -> str | None:
    try:
        for m in _get(host, "/api/tags").get("models", []):
            if m.get("name") == model or m.get("model") == model or m.get("name") == model + ":latest":
                return m.get("digest")
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def _is_loaded(loaded: list[str], model: str) -> bool:
    return any(n == model or n == model + ":latest" for n in loaded)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["dev", "train"], default="dev")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_OPTIONS["num_ctx"])
    ap.add_argument("--num-predict", type=int, default=DEFAULT_OPTIONS["num_predict"])
    ap.add_argument("--seed", type=int, default=DEFAULT_OPTIONS["seed"])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args(argv)

    options = {"temperature": 0, "seed": a.seed, "num_ctx": a.num_ctx, "num_predict": a.num_predict}
    rows, stats = common.load_legacy(a.split, limit=a.limit)
    before = common.gpu_state()
    was_loaded = _is_loaded(_loaded_models(a.host), a.model)
    try:
        guard.preflight(vram_gb=6, ram_gb=2, timeout_s=900)
    except guard.GuardError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    t0 = time.time()
    preds: list[dict] = []
    sidecar: list[dict] = []
    n_ok = n_err = 0
    code = 0
    try:
        for row in rows:
            if guard.mem_available_bytes() < RAM_FLOOR:
                print("MemAvailable below 1.5 GiB, stopping", file=sys.stderr)
                code = EXIT_LOW_RAM
                break
            png = Path(row["image_path"]).read_bytes()
            base = {"id": row["id"], "paper_id": row["arxiv_id"], "gt_latex": row["latex"]}
            try:
                out = infer_one(a.host, a.model, png, options, "10m", a.timeout)
            except (OllamaError, urllib.error.URLError, OSError) as e:
                preds.append({**base, "pred_error": str(e)})
                n_err += 1
                continue
            raw = out["message"]["content"]
            preds.append({**base, "pred_latex": common.strip_fences(raw)})
            sidecar.append(
                {
                    "id": row["id"],
                    "model": a.model,
                    "raw_text": raw,
                    "done_reason": out.get("done_reason"),
                    "prompt_eval_count": out.get("prompt_eval_count"),
                    "eval_count": out.get("eval_count"),
                    "total_duration": out.get("total_duration"),
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "width": row.get("width"),
                    "height": row.get("height"),
                }
            )
            n_ok += 1
    finally:
        if not was_loaded:
            try:
                _post(a.host, "/api/chat", {"model": a.model, "messages": [], "keep_alive": 0}, 60)
            except (OllamaError, urllib.error.URLError, OSError) as e:
                print(f"unload failed: {e}", file=sys.stderr)

        common.write_jsonl(a.out, preds)
        common.write_jsonl(a.sidecar, sidecar)
    try:
        version = _get(a.host, "/api/version").get("version")
    except (urllib.error.URLError, OSError, ValueError):
        version = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False
        ).stdout.strip()
    except OSError:
        commit = None
    common.write_json(
        a.out.with_name(a.out.stem + ".manifest.json"),
        {
            "model": a.model,
            "model_digest": _model_digest(a.host, a.model),
            "ollama_version": version,
            "options": options,
            "endpoint": "/api/chat",
            "git_commit": commit,
            "loader_stats": stats,
            "model_loaded_before": was_loaded,
            "gpu_before": before,
            "gpu_after": common.gpu_state(),
            "n_ok": n_ok,
            "n_error": n_err,
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print(f"ok {n_ok} error {n_err}")
    if code:
        return code
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
