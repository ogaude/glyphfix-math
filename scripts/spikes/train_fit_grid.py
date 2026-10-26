"""Training-fit grid driver (torch-free): cell list, guarded subprocess per cell, fit rule, probes, summary.

Usage:
  uv run python -m scripts.spikes.train_fit_grid [--retry-inconclusive] [--only CELL_ID] [--summarize-only]

Gate reading: "14.5 GB" is read as 14.5 GiB of the cell's own peak. Both views are reported per row:
torch max_memory_allocated / max_memory_reserved, and the rocm-smi card peak minus the usage present
right before the cell started (foreign jobs, Ollama). The usage before each cell is stored beside the
row so the result can be re-read when a foreign process goes away. A cell that was paused by the
guard, aborted by it, or hit OOM while foreign load took memory is inconclusive, never "does not fit".

Rows are appended to data/spikes/train_fit/rows.jsonl (rerun skips conclusive cells).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from glyphfix_math import guard as guardmod
from glyphfix_math.paths import repo_root
from scripts.spikes.common import EVIDENCE_DIR, _rocm_vram, gpu_state, read_jsonl, write_jsonl

GATE_GIB = 14.5
GUARD_EXITS = (75, 76, 78)
CELL_PY = ".venv-lighton/bin/python"
RESULTS = repo_root() / "data" / "spikes" / "train_fit" / "rows.jsonl"
BS = (1, 2, 4)
PIXELS = (200704, 401408, 602112)
STEPS = 20
GIB = 2**30


def grid_cells():
    for bs in BS:
        for px in PIXELS:
            for fla in ("on", "off"):
                yield {
                    "cell_id": f"bs{bs}-px{px}-fla{fla}", "bs": bs, "max_px": px, "fla": fla,
                    "rank": 32, "alpha": 8, "gc": "on", "steps": STEPS, "optim": "adamw", "quant": "none", "alloc_conf": "",
                }


def fits(row: dict) -> bool:
    return bool(
        row.get("outcome") == "ok"
        and row.get("steps_done") == row.get("steps")
        and not row.get("was_paused")
        and row.get("peak_reserved_gib") is not None
        and row["peak_reserved_gib"] <= GATE_GIB
        and row.get("own_card_peak_gib") is not None
        and row["own_card_peak_gib"] <= GATE_GIB
    )


def classify_outcome(*, guard_exit: int | None = None, outcome: str | None = None, was_paused: bool = False,
                     free_at_start_gib: float | None = None) -> dict:
    """Classify one cell run: {"outcome", "inconclusive", "verdict"}."""
    if guard_exit in GUARD_EXITS:
        return {"outcome": "guard-abort", "inconclusive": True, "verdict": "inconclusive"}
    if was_paused:
        return {"outcome": outcome, "inconclusive": True, "verdict": "inconclusive"}
    if outcome == "oom":
        if free_at_start_gib is None or free_at_start_gib < GATE_GIB:
            return {"outcome": outcome, "inconclusive": True, "verdict": "inconclusive"}
        return {"outcome": outcome, "inconclusive": False, "verdict": "does-not-fit"}
    if outcome in ("timeout", "crash"):
        return {"outcome": outcome, "inconclusive": True, "verdict": "inconclusive"}
    return {"outcome": outcome, "inconclusive": False, "verdict": "measured"}


def best_cell(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if fits(r)]
    if not ok:
        return None
    return min(ok, key=lambda r: (-r["bs"] * r["max_px"], not r.get("fla_active"), r.get("s_per_step") or 1e9))


# ---------------------------------------------------------------- launching


def _cell_cmd(spec: dict, result: Path) -> list[str]:
    cmd = [CELL_PY, "-m", "scripts.spikes.train_fit_cell", "--bs", str(spec["bs"]), "--max-px", str(spec["max_px"]),
           "--fla", spec["fla"], "--rank", str(spec["rank"]), "--alpha", str(spec["alpha"]), "--gc", spec["gc"],
           "--steps", str(spec["steps"]), "--optim", spec["optim"], "--quant", spec["quant"],
           "--cell-id", spec["cell_id"], "--result", str(result)]
    if spec.get("alloc_conf"):
        cmd += ["--alloc-conf", spec["alloc_conf"]]
    return cmd


def _ollama_loaded() -> str:
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [x for x in out.splitlines() if x.strip()]
    return "\n".join(lines[1:])


def wait_for_ollama(max_wait_s: float = 1800, poll_s: float = 30) -> str:
    deadline = time.monotonic() + max_wait_s
    loaded = _ollama_loaded()
    while loaded and time.monotonic() < deadline:
        print(f"ollama has a model loaded, waiting: {loaded.splitlines()[0][:80]}", file=sys.stderr, flush=True)
        time.sleep(poll_s)
        loaded = _ollama_loaded()
    return loaded


def _used_gib() -> float | None:
    _, used = _rocm_vram()
    return None if used is None else used / GIB


def launch_cell(spec: dict, *, cell_timeout: float = 1800, workdir: Path | None = None) -> dict:
    ollama_loaded = wait_for_ollama()
    state = gpu_state()
    holders = state.get("holders")
    holders = " | ".join(x.strip() for x in holders.replace("\t", " ").splitlines() if x.strip()) if isinstance(holders, str) else holders
    used_before = _used_gib()
    free_start = (state.get("vram_free_bytes") or 0) / GIB
    workdir = workdir or RESULTS.parent
    workdir.mkdir(parents=True, exist_ok=True)
    result = workdir / f"{spec['cell_id']}.json"
    result.unlink(missing_ok=True)

    peak = {"used": used_before or 0.0, "sampled": False}
    stop = threading.Event()

    def sampler():
        while not stop.wait(1.0):
            u = _used_gib()
            if u is not None:
                peak["sampled"] = True
                peak["used"] = max(peak["used"], u)

    th = threading.Thread(target=sampler, daemon=True)
    was_paused = False
    guard_exit = None
    timed_out = False
    returncode = None
    print(f"== cell {spec['cell_id']}: free VRAM at start {free_start:.2f} GiB, used before {used_before}", file=sys.stderr, flush=True)
    th.start()
    try:
        with guardmod.guard(vram_gb=9.5, ram_gb=8, timeout_s=1800, vram_floor_gb=0.1, grace_s=120) as g:
            proc = subprocess.Popen(_cell_cmd(spec, result), start_new_session=True, cwd=repo_root())
            g.register_child(proc)
            t0 = time.monotonic()
            try:
                while proc.poll() is None:
                    if g.paused:
                        was_paused = True
                    if time.monotonic() - t0 > cell_timeout:
                        timed_out = True
                        break
                    time.sleep(0.05)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            returncode = proc.returncode
            if g.aborted:
                guard_exit = guardmod.EXIT_WATCHDOG_KILLED
    except guardmod.GuardError as exc:
        guard_exit = exc.exit_code
        print(str(exc), file=sys.stderr)
    finally:
        stop.set()
        th.join(timeout=5)

    if result.exists():
        row = json.loads(result.read_text())
    else:
        row = {"cell_id": spec["cell_id"], "bs": spec["bs"], "max_px": spec["max_px"], "fla_arm": spec["fla"], "fla_active": None,
               "steps": spec["steps"], "steps_done": 0, "outcome": "crash", "error_tail": f"no result file, returncode {returncode}"}
    if timed_out:
        row["outcome"] = "timeout"
    if guard_exit is not None and not result.exists():
        row["outcome"] = "guard-abort"
    measured_card = used_before is not None and peak["sampled"]
    cls = classify_outcome(guard_exit=guard_exit, outcome=row.get("outcome"), was_paused=was_paused, free_at_start_gib=free_start)
    row.update(
        used_before_gib=round(used_before, 3) if used_before is not None else None,
        free_at_start_gib=round(free_start, 3),
        card_peak_used_gib=round(peak["used"], 3),
        own_card_peak_gib=round(peak["used"] - used_before, 3) if measured_card else None,
        was_paused=was_paused, returncode=returncode, guard_exit=guard_exit,
        holders_at_start=holders, ollama_loaded=ollama_loaded, inconclusive=cls["inconclusive"],
        verdict=cls["verdict"],
    )
    return row


# ---------------------------------------------------------------- running


def _latest(results: Path) -> dict[str, dict]:
    if not Path(results).exists():
        return {}
    return {r["cell_id"]: r for r in read_jsonl(Path(results))}


def _append(results: Path, row: dict) -> None:
    results = Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fla_ok(env_verify: Path) -> bool:
    try:
        return bool(json.loads(Path(env_verify).read_text())["fla"]["import_ok"])
    except (OSError, KeyError, ValueError):
        return False


def run_specs(specs, *, results: Path, env_verify: Path, launch, retry_inconclusive: bool = False, cell_timeout: float = 1800) -> list[dict]:
    fla_ok = _fla_ok(env_verify)
    stored = _latest(results)
    out: list[dict] = []
    for spec in specs:
        old = stored.get(spec["cell_id"])
        if old is not None and not (old.get("inconclusive") and retry_inconclusive):
            out.append(old)
            continue
        if spec["fla"] == "on" and not fla_ok:
            row = {"cell_id": spec["cell_id"], "bs": spec["bs"], "max_px": spec["max_px"], "fla_arm": "on", "fla_active": False,
                   "steps": spec["steps"], "steps_done": 0, "outcome": "fla-unavailable", "s_per_step": None,
                   "peak_allocated_gib": None, "peak_reserved_gib": None, "own_card_peak_gib": None, "used_before_gib": None,
                   "inconclusive": False, "error_tail": "flash-linear-attention does not import in the training venv"}
        else:
            row = launch(spec, cell_timeout=cell_timeout)
            row.setdefault("inconclusive", False)
        row["fits"] = fits(row)
        _append(results, row)
        stored[spec["cell_id"]] = row
        out.append(row)
    return out


def run_grid(*, results: Path = RESULTS, env_verify: Path = EVIDENCE_DIR / "env_verify.json", launch=launch_cell,
             only: str | None = None, retry_inconclusive: bool = False, cell_timeout: float = 1800) -> list[dict]:
    specs = [c for c in grid_cells() if only in (None, c["cell_id"])]
    return run_specs(specs, results=results, env_verify=env_verify, launch=launch,
                     retry_inconclusive=retry_inconclusive, cell_timeout=cell_timeout)


def probe_specs(best: dict) -> list[dict]:
    base = {"bs": best["bs"], "max_px": best["max_px"], "fla": "on" if best.get("fla_active") else "off", "rank": 32, "alpha": 8,
            "gc": "on", "steps": STEPS, "optim": "adamw", "quant": "none", "alloc_conf": ""}
    return [
        {**base, "cell_id": "r64-gcoff", "rank": 64, "alpha": 11, "gc": "off"},
        {**base, "cell_id": "alloc-expandable", "alloc_conf": "expandable_segments:True"},
    ]


def fallback_specs() -> list[dict]:
    base = {"bs": 1, "max_px": 200704, "fla": "off", "rank": 32, "alpha": 8, "gc": "on", "steps": STEPS, "optim": "adamw8bit",
            "quant": "none", "alloc_conf": ""}
    return [{**base, "cell_id": "fb-adamw8bit"}, {**base, "cell_id": "fb-nf4", "quant": "nf4"}]


# ---------------------------------------------------------------- evidence


def _fmt(v, nd=2):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_evidence(rows: list[dict], evidence_dir: Path, env_verify: Path) -> None:
    evidence_dir = Path(evidence_dir)
    slim = []
    for r in rows:
        r = dict(r)
        losses = r.pop("losses", None) or []
        r["first_loss"] = losses[0] if losses else None
        r["last_loss"] = losses[-1] if losses else None
        slim.append(r)
    write_jsonl(evidence_dir / "train_fit.jsonl", slim)

    grid = [r for r in rows if r["cell_id"].startswith("bs")]
    probes = [r for r in rows if not r["cell_id"].startswith("bs")]
    best = best_cell(grid)
    versions = next((r["versions"] for r in rows if r.get("versions")), {})
    try:
        fla_env = json.loads(Path(env_verify).read_text())["fla"]
    except (OSError, KeyError, ValueError):
        fla_env = {}
    lines = [
        "# Training fit grid (FEA-01)",
        "",
        ("Gate: a cell fits when it finished all steps with finite loss and grads, was never paused by the guard, and both its own "
         "torch reserved peak and its own card peak (rocm-smi peak minus usage present before the cell) are <= 14.5 GiB. "
         "D-12 says 14.5 GB; it is read here as 14.5 GiB of this process's own peak."),
        "",
        ("Foreign-usage note: the card is shared. `used before GiB` is the VRAM already in use right before each cell (foreign jobs, "
         "Ollama); `own card peak` excludes it. Guard aborts, pauses and OOMs under foreign load are marked inconclusive, not 'does not fit'."),
        "",
        f"Versions: {json.dumps(versions)}. fla in env_verify: import_ok={fla_env.get('import_ok')}, kernel_ok={fla_env.get('kernel_ok')}.",
        ("Each cell: rank 32, alpha 8 (rsLoRA), gradient checkpointing on, 20 steps, worst-case batch (largest crop resized to the pixel "
         "budget, longest label), median s/step after 3 warm-up steps."),
        "",
        "| cell | bs | max_px | image tok | label tok | fla_active | peak alloc GiB | peak reserved GiB | own card peak GiB | used before GiB | s/step | tok/s | outcome | fits |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in grid + probes:
        out = r.get("outcome", "")
        if r.get("inconclusive"):
            out += " (inconclusive)"
        lines.append(
            "| " + " | ".join([
                r["cell_id"], str(r.get("bs")), str(r.get("max_px")), _fmt(r.get("image_tokens")), _fmt(r.get("label_tokens")),
                _fmt(r.get("fla_active")), _fmt(r.get("peak_allocated_gib")), _fmt(r.get("peak_reserved_gib")),
                _fmt(r.get("own_card_peak_gib")), _fmt(r.get("used_before_gib")), _fmt(r.get("s_per_step"), 3),
                _fmt(r.get("tokens_per_s"), 1), out, _fmt(r.get("fits")),
            ]) + " |"
        )
    lines += ["", "## Best cell", ""]
    if best:
        lines.append(f"{best['cell_id']}: bs {best['bs']}, max_px {best['max_px']}, fla_active {best.get('fla_active')}, "
                     f"peak reserved {_fmt(best.get('peak_reserved_gib'))} GiB, {_fmt(best.get('s_per_step'), 3)} s/step.")
    else:
        lines.append("No grid cell fits.")
    lines += ["", "## fla comparison (same bs and max_px, both arms ok)", ""]
    by = {}
    for r in grid:
        by.setdefault((r["bs"], r["max_px"]), {})[r["fla_arm"]] = r
    for (bs, px), arms in sorted(by.items()):
        on, off = arms.get("on"), arms.get("off")
        if on and off and on.get("s_per_step") and off.get("s_per_step"):
            lines.append(f"- bs {bs}, {px} px: fla on {on['s_per_step']} s/step (reserved {on['peak_reserved_gib']} GiB), "
                         f"off {off['s_per_step']} s/step (reserved {off['peak_reserved_gib']} GiB)")
    lines += ["", "## Probes and fallbacks", ""]
    for r in probes:
        lines.append(f"- {r['cell_id']}: outcome {r.get('outcome')}, reserved {_fmt(r.get('peak_reserved_gib'))} GiB, "
                     f"own card peak {_fmt(r.get('own_card_peak_gib'))} GiB, {_fmt(r.get('s_per_step'), 3)} s/step, fits {r.get('fits')}")
    if not probes:
        lines.append("None run.")
    inc = [r["cell_id"] for r in rows if r.get("inconclusive")]
    lines += ["", f"Inconclusive rows: {', '.join(inc) if inc else 'none'}", ""]
    (evidence_dir / "train_fit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--evidence-dir", default=str(EVIDENCE_DIR))
    ap.add_argument("--retry-inconclusive", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--cell-timeout", type=float, default=1800)
    a = ap.parse_args(argv)
    results = Path(a.results)
    env_verify = Path(a.evidence_dir) / "env_verify.json"
    if not a.summarize_only:
        grid = run_grid(results=results, env_verify=env_verify, only=a.only, retry_inconclusive=a.retry_inconclusive,
                        cell_timeout=a.cell_timeout)
        if a.only is None:
            kw = {"results": results, "env_verify": env_verify, "launch": launch_cell,
                  "retry_inconclusive": a.retry_inconclusive, "cell_timeout": a.cell_timeout}
            best = best_cell(grid)
            if best:
                run_specs(probe_specs(best), **kw)
            else:
                fb = fallback_specs()
                first = run_specs(fb[:1], **kw)
                if not first[0].get("fits"):
                    run_specs(fb[1:], **kw)
    rows = list(_latest(results).values())
    write_evidence(rows, Path(a.evidence_dir), env_verify)
    print(f"{sum(1 for r in rows if r.get('fits'))} fitting rows of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
