"""CPU llama-server sensitivity appendix: re-run the flipped benchmark crops of both arms on a second engine.

The verdict comes from the Ollama run alone. This tool takes the ids whose outcome flipped between the arms
(headline r2w_ids and w2r_ids of the benchmark results.json), runs each through Ollama's bundled llama-server on the
CPU for the stock and the tuned model, scores both CPU prediction sets and reports how many flips hold. It never
writes results.json or any other verdict file.

Usage (from repo root; nothing else may be using the CPU heavily, 8 GiB RAM must be free):
  uv run python -m scripts.acceptance_sensitivity run --mode benchmark --results evidence/acceptance/results.json
  uv run python -m scripts.acceptance_sensitivity run --mode benchmark --results ... --resume --max-hours 6
  uv run python -m scripts.acceptance_sensitivity run --mode dev --ids-file data/runs/p5-precheck/precheck_ids.txt --limit 5 --out-dir data/runs/p5-sens-dev
  uv run python -m scripts.acceptance_sensitivity summarize --results ...

Behaviour: ids go in the pinned order (all r2w sorted, then all w2r sorted) in chunks of 50; each chunk is run for the
stock arm and then the tuned arm so every finished chunk is a complete pair. The server listens on 127.0.0.1 only and
is started per chunk and arm, CPU only. Exit codes: 0 ok (also when stopped by --max-hours or low RAM, see
progress.json), 2 bad input.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from lightonocr_math import acceptance as accmod
from lightonocr_math import guard, score
from lightonocr_math.corpus import freeze
from lightonocr_math.paths import repo_root
from scripts.baseline_infer import crop_png
from scripts.spikes.common import gpu_state, read_jsonl, strip_fences, write_json, write_jsonl

CHUNK = accmod.ACCEPTANCE["sensitivity"]["chunk"]
RAM_NEEDED = 8 * guard.GIB
DEFAULT_BINARY = "/usr/lib/ollama/llama-server"
DEFAULT_PORT = 18081
ARMS = ("stock", "tuned")
EVIDENCE = Path("evidence/acceptance")


def _now() -> float:
    return time.monotonic()


def order_ids(results: dict) -> list[str]:
    """All right-to-wrong ids sorted, then all wrong-to-right ids sorted (the pinned order)."""
    head = results["headline"]
    return sorted(head["r2w_ids"]) + sorted(head["w2r_ids"])


def chunks(ids: list[str], n: int = CHUNK) -> list[list[str]]:
    return [ids[i : i + n] for i in range(0, len(ids), n)]


def server_argv(binary: str, text_blob: str, projector_blob: str, port: int, device_args: list[str]) -> list[str]:
    """llama-server argument list (the Phase 4 parity flags) listening on loopback only."""
    return [
        binary, "-m", str(text_blob), "--mmproj", str(projector_blob), "-c", "8192", "-np", "1", "--flash-attn", "auto",
        "-b", "1024", "-ub", "1024", "--context-shift", "--keep", "4", "--host", "127.0.0.1", "--port", str(port), *device_args,
    ]


def request_payload(png_bytes: bytes) -> dict:
    """OpenAI-style chat request: one user message holding only the image, greedy, thinking off."""
    url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _help_text(binary: str) -> str:
    r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60, check=False)
    return r.stdout + r.stderr


def cpu_device_args(binary: str) -> list[str]:
    """`-dev none` when the binary has the device option, else `-ngl 0`."""
    help_text = _help_text(binary)
    return ["-dev", "none"] if "--device" in help_text or "-dev," in help_text else ["-ngl", "0"]


def llama_version(binary: str) -> str:
    r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60, check=False)
    return (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else ""


def start_server(argv: list[str]):
    return subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_server(proc) -> None:
    """Terminate the server's process group and wait for it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def wait_health(port: int, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"llama-server on port {port} not healthy after {timeout:.0f} s")


def post_chat(port: int, payload: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def blob_path(blobs_dir: Path, digest: str | None) -> Path:
    return Path(blobs_dir) / f"sha256-{digest}"


def _load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paper(row: dict) -> str:
    return row["arxiv_id"] if "arxiv_id" in row else row["paper_id"]


def _done_rows(path: Path) -> set[str]:
    """Ids with a prediction in `path`; error rows are dropped from the file so they are retried."""
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    ok = [r for r in rows if "pred_latex" in r]
    if len(ok) != len(rows):
        write_jsonl(path, ok)
    return {r["id"] for r in ok}


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _vram_used() -> int | None:
    return gpu_state().get("vram_used_bytes")


def cmd_run(a: argparse.Namespace) -> int:
    if a.mode == "benchmark":
        if a.results is None:
            print("--results is required in benchmark mode", file=sys.stderr)
            return 2
        ids = order_ids(json.loads(a.results.read_text(encoding="utf-8")))
        freeze.assert_benchmark_frozen(repo_root() / "data" / "benchmark")
    else:
        if a.ids_file is None:
            print("--ids-file is required in dev mode", file=sys.stderr)
            return 2
        ids = [ln.strip() for ln in a.ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if a.limit is not None:
        ids = ids[: a.limit]
    records = {r["id"]: r for r in read_jsonl(a.records)}
    unknown = [i for i in ids if i not in records]
    if unknown:
        print(f"{len(unknown)} id(s) not in {a.records}: {', '.join(unknown[:5])}", file=sys.stderr)
        return 2
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        print(f"--arms must be a comma list of {ARMS}", file=sys.stderr)
        return 2
    manifests = {"stock": a.stock_manifest, "tuned": a.tuned_manifest}
    blobs: dict[str, dict] = {}
    for arm in arms:
        b = _load_manifest(manifests[arm]).get("gguf_blobs", {})
        paths = {k: blob_path(a.blobs_dir, b.get(k)) for k in ("text", "projector")}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            print(f"{arm}: blob(s) not found: {', '.join(missing)}", file=sys.stderr)
            return 2
        blobs[arm] = paths
    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {arm: out_dir / f"cpu_{arm}.pred.jsonl" for arm in arms}
    for arm in arms:
        if not a.resume and preds[arm].exists() and preds[arm].stat().st_size:
            print(f"{preds[arm]} exists; pass --resume to continue it", file=sys.stderr)
            return 2
    done = {arm: _done_rows(preds[arm]) if a.resume else set() for arm in arms}
    device = cpu_device_args(a.binary)
    t0 = _now()
    deadline = None if a.max_hours is None else t0 + a.max_hours * 3600
    seconds: dict[str, list[float]] = {arm: [] for arm in arms}
    vram_before = _vram_used()
    reason = "complete"

    def timed_out() -> bool:
        return deadline is not None and _now() >= deadline

    def write_progress(stopped: str) -> None:
        n_done = sum(1 for i in ids if all(i in done[arm] for arm in arms))
        write_json(
            out_dir / "progress.json",
            {
                "n_done": n_done, "n_total": len(ids), "stopped_reason": stopped, "cpu_option": device, "arms": arms, "mode": a.mode,
                "flags": server_argv(a.binary, "<text>", "<projector>", a.port, device)[1:],
                "seconds_per_record": {arm: (sum(v) / len(v) if v else None) for arm, v in seconds.items()},
                "n_rows": {arm: len(done[arm]) for arm in arms},
                "vram_before": vram_before, "vram_after": _vram_used(),
            },
        )

    for chunk in chunks(ids, a.chunk):
        if guard.mem_available_bytes() < RAM_NEEDED:
            reason = "low_ram"
            break
        if timed_out():
            reason = "max_hours"
            break
        for arm in arms:
            todo = [i for i in chunk if i not in done[arm]]
            if not todo:
                continue
            proc = start_server(server_argv(a.binary, str(blobs[arm]["text"]), str(blobs[arm]["projector"]), a.port, device))
            try:
                wait_health(a.port)
                for rid in todo:
                    if timed_out():
                        reason = "max_hours"
                        break
                    row = records[rid]
                    base = {"id": rid, "paper_id": _paper(row), "gt_latex": row["latex"] if "latex" in row else row["gt_latex"]}
                    png, _ = crop_png(repo_root() / row["image"])
                    t = _now()
                    try:
                        out = post_chat(a.port, request_payload(png))
                        choice = out["choices"][0]
                        raw = choice["message"]["content"]
                        usage = out.get("usage", {})
                        result = {
                            **base, "pred_latex": strip_fences(raw), "raw_text": raw, "finish_reason": choice.get("finish_reason"),
                            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                            "seconds": round(_now() - t, 2),
                        }
                        done[arm].add(rid)
                    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
                        result = {**base, "pred_error": str(e)}
                    _append(preds[arm], result)
                    seconds[arm].append(_now() - t)
            finally:
                stop_server(proc)
            if reason == "max_hours":
                break
        write_progress(reason)
        if reason != "complete":
            break
    write_progress(reason)
    if reason != "complete":
        print(f"stopped: {reason}; resume with --resume")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    if a.results is None:
        print("--results is required", file=sys.stderr)
        return 2
    results = json.loads(a.results.read_text(encoding="utf-8"))
    head = results["headline"]
    cpu = {}
    for arm in ARMS:
        path = a.out_dir / f"cpu_{arm}.pred.jsonl"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2
        cpu[arm] = [r for r in read_jsonl(path) if "pred_latex" in r]
    common = {r["id"] for r in cpu["stock"]} & {r["id"] for r in cpu["tuned"]}
    flags: dict[str, dict] = {}
    for arm in ARMS:
        pairs = a.out_dir / f"pairs_{arm}.pred.jsonl"
        write_jsonl(pairs, sorted((r for r in cpu[arm] if r["id"] in common), key=lambda r: r["id"]))
        score_dir = a.out_dir / f"score_{arm}"
        score.score_file(pairs, score_dir)
        flags[arm] = {r["id"]: r.get("render_identical") for r in read_jsonl(score_dir / "records.jsonl")}
    summary = accmod.sensitivity_summary(head["r2w_ids"], head["w2r_ids"], flags["stock"], flags["tuned"])
    progress_path = a.out_dir / "progress.json"
    result = {
        "summary": summary,
        "progress": json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None,
        "seconds_per_record": {arm: _mean([r.get("seconds") for r in cpu[arm]]) for arm in ARMS},
        "flags": server_argv(a.binary, "<text>", "<projector>", a.port, ["-dev", "none"])[1:],
        "llama_server_version": llama_version(a.binary),
        "blob_digests": {arm: _load_manifest(m).get("gguf_blobs", {}) for arm, m in (("stock", a.stock_manifest), ("tuned", a.tuned_manifest))},
        "engine": accmod.ACCEPTANCE["sensitivity"]["engine"],
        "blocking": False,
    }
    write_json(a.out, result)
    print(json.dumps(summary, indent=2))
    return 0


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stock-manifest", type=Path, default=Path("data/benchmark/baseline/stock_q8.pred.manifest.json"))
    common.add_argument("--tuned-manifest", type=Path, default=Path("data/benchmark/acceptance/tuned_q8.pred.manifest.json"))
    common.add_argument("--out-dir", type=Path, default=Path("data/benchmark/acceptance/sensitivity"))
    common.add_argument("--results", type=Path, default=None, help="benchmark results.json (read only)")
    common.add_argument("--binary", default=DEFAULT_BINARY)
    common.add_argument("--port", type=int, default=DEFAULT_PORT)

    p = sub.add_parser("run", parents=[common], help="run the flipped records on the CPU server")
    p.add_argument("--mode", choices=["benchmark", "dev"], default="benchmark")
    p.add_argument("--ids-file", type=Path, default=None)
    p.add_argument("--records", type=Path, default=None)
    p.add_argument("--arms", default="stock,tuned")
    p.add_argument("--blobs-dir", type=Path, default=Path.home() / ".ollama" / "models" / "blobs")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", parents=[common], help="score the CPU predictions and write sensitivity.json")
    p.add_argument("--out", type=Path, default=EVIDENCE / "sensitivity.json")
    p.set_defaults(func=cmd_summarize)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "run" and a.records is None:
        a.records = Path("data/benchmark/records.jsonl" if a.mode == "benchmark" else "data/corpus/dev/records.jsonl")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
