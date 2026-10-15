"""Model-agnostic training loop: accumulation, schedule, checkpoint/resume, logging, run manifest.

Usage:
  from lightonocr_math.train import run
  cfg = run.RunConfig(run_id="r1", trainset="data/train/trainset.jsonl", trainset_sha256="...", lr=2e-4)
  result = run.train_loop(cfg, params=params, loss_fn=loss_fn, batch_plan=lambda epoch: batches,
                          save_adapter=save, load_adapter=load, run_dir=Path("runs/r1"), resume=True)
  run.write_manifest(run_dir, run.run_manifest(cfg, outcome=result["outcome"], ...))

Rules:
  * loss_fn(batch_indices) returns the mean loss of one micro-batch; the loop scales it by the real
    size of its accumulation group, so a short last group of an epoch is averaged correctly
  * a non-finite loss or gradient norm stops the run with outcome "nan" and writes no checkpoint
  * resume checkpoints (adapter plus optimizer, step, position and RNG state) are written atomically,
    read back with torch.load(weights_only=True), and only the newest keep_resume are kept
  * eval adapters (eval/step-N/adapter) are never pruned; final/adapter is written on completion
  * a stop flag is honoured at the next optimizer-step boundary after saving a resume checkpoint
  * PYTORCH_HIP_ALLOC_CONF is never set here: expandable segments gave non-finite gradients with fla

Known limit: the loop trusts the caller to restore model weights from the resume adapter through
load_adapter; the optimizer, RNG and position are restored here.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from lightonocr_math import pixels
from lightonocr_math.corpus._io import sha256_file, write_json
from lightonocr_math.paths import repo_root
from lightonocr_math.train import data

__all__ = [
    "RunConfig",
    "load_latest_resume",
    "run_manifest",
    "save_checkpoint",
    "total_opt_steps",
    "train_loop",
    "write_manifest",
]

LOCK_FILE = "requirements/train.lock.txt"


@dataclass
class RunConfig:
    run_id: str
    trainset: str
    trainset_sha256: str
    lr: float
    rank: int = 32
    alpha: int = 8
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    ckpt_every: int = 50
    eval_every: int = 0
    keep_resume: int = 2
    max_opt_steps: int | None = None
    min_px: int = 65536
    max_px: int = 602112
    empty_cache_every: int = 25


def total_opt_steps(n_batches: int, epochs: int, grad_accum: int) -> int:
    """Optimizer steps for the run; each epoch ends with a (possibly short) accumulation group."""
    return math.ceil(n_batches / grad_accum) * epochs


def save_checkpoint(
    run_dir: Path,
    kind: str,
    opt_step: int,
    *,
    save_adapter: Callable[[Path], None],
    opt=None,
    state: dict | None = None,
) -> Path:
    """Write a resume ("resume") or evaluation ("eval") checkpoint; returns its directory."""
    run_dir = Path(run_dir)
    if kind == "eval":
        target = run_dir / "eval" / f"step-{opt_step}"
        tmp = run_dir / "eval" / f".tmp-step-{opt_step}"
        adapter_rel = Path("adapter")
    elif kind == "resume":
        target = run_dir / "resume" / f"ckpt-{opt_step}"
        tmp = run_dir / "resume" / f".tmp-ckpt-{opt_step}"
        adapter_rel = Path("adapter")
    else:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / adapter_rel).mkdir(parents=True)
    save_adapter(tmp / adapter_rel)
    if kind == "resume":
        import torch

        payload = dict(state or {})
        payload["opt_step"] = opt_step
        payload["optimizer"] = opt.state_dict() if opt is not None else None
        payload["py_random"] = random.getstate()
        payload["torch_rng"] = torch.get_rng_state()
        payload["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save(payload, tmp / "state.pt")
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)
    return target


def _resume_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    root = Path(run_dir) / "resume"
    found = []
    if root.is_dir():
        for p in root.iterdir():
            if p.name.startswith("ckpt-") and (p / "state.pt").is_file() and (p / "adapter").is_dir():
                try:
                    found.append((int(p.name[5:]), p))
                except ValueError:
                    continue
    return sorted(found)


def load_latest_resume(run_dir: Path) -> dict | None:
    """Newest complete resume checkpoint as {"dir", "adapter_dir", "state"}, or None."""
    found = _resume_dirs(run_dir)
    if not found:
        return None
    import torch

    _, path = found[-1]
    state = torch.load(path / "state.pt", weights_only=True, map_location="cpu")
    return {"dir": path, "adapter_dir": path / "adapter", "state": state}


def _prune_resume(run_dir: Path, keep: int) -> None:
    found = _resume_dirs(run_dir)
    for _, path in found[: max(0, len(found) - keep)]:
        shutil.rmtree(path, ignore_errors=True)


def _truncate_log(path: Path, keep_through: int) -> None:
    """Drop log rows of optimizer steps after the resume point (written by an interrupted run)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("opt_step", 0) <= keep_through:
            kept.append(line)
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def _default_mem_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    free, _ = torch.cuda.mem_get_info()
    return {
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3),
        "vram_free_gib": round(free / gib, 3),
    }


def train_loop(
    cfg: RunConfig,
    *,
    params: list,
    loss_fn: Callable[[list[int]], object],
    batch_plan: Callable[[int], list[list[int]]],
    save_adapter: Callable[[Path], None],
    run_dir: Path,
    load_adapter: Callable[[Path], None] | None = None,
    resume: bool = False,
    on_step: Callable[[dict], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
    device_sync: Callable[[], None] | None = None,
    mem_stats: Callable[[], dict] | None = None,
) -> dict:
    """Run the optimizer loop; returns {outcome, reason, opt_steps_done, total_opt_steps, eval_steps, ...}."""
    import torch

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    plans = [batch_plan(e) for e in range(cfg.epochs)]
    total = sum(total_opt_steps(len(p), 1, cfg.grad_accum) for p in plans)
    on_cuda = any(getattr(p, "is_cuda", False) for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=on_cuda)
    mem = mem_stats or _default_mem_stats

    opt_step = 0
    epoch = 0
    pos = 0  # next micro-batch index within the epoch
    if resume:
        ck = load_latest_resume(run_dir)
        if ck is not None:
            st = ck["state"]
            if load_adapter is not None:
                load_adapter(ck["adapter_dir"])
            opt.load_state_dict(st["optimizer"])
            random.setstate(st["py_random"])
            torch.set_rng_state(st["torch_rng"].cpu())
            if st.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
            opt_step, epoch, pos = int(st["opt_step"]), int(st["epoch"]), int(st["pos"])
            _truncate_log(log_path, opt_step)
    else:
        if log_path.exists():
            log_path.unlink()

    eval_steps: list[int] = []
    step_times: list[float] = []
    peak = 0.0
    outcome, reason = "complete", ""

    def ckpt_state() -> dict:
        return {"epoch": epoch, "pos": pos}

    def stop_now(why: str) -> bool:
        nonlocal outcome, reason
        save_checkpoint(run_dir, "resume", opt_step, save_adapter=save_adapter, opt=opt, state=ckpt_state())
        _prune_resume(run_dir, cfg.keep_resume)
        outcome, reason = "stopped", why
        return True

    done = False
    with open(log_path, "a", encoding="utf-8") as log:
        while epoch < cfg.epochs and not done:
            batches = plans[epoch]
            while pos < len(batches):
                if cfg.max_opt_steps is not None and opt_step >= cfg.max_opt_steps:
                    stop_now(f"max_opt_steps {cfg.max_opt_steps} reached")
                    done = True
                    break
                t0 = time.perf_counter()
                group = batches[pos : pos + cfg.grad_accum]
                loss_sum = 0.0
                for mb in group:
                    loss = loss_fn(mb)
                    val = float(loss.detach())
                    if not math.isfinite(val):
                        outcome, reason = "nan", f"non-finite loss at optimizer step {opt_step + 1}"
                        done = True
                        break
                    (loss / len(group)).backward()
                    loss_sum += val
                if done:
                    break
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
                if not math.isfinite(grad_norm):
                    outcome, reason = "nan", f"non-finite grad norm at optimizer step {opt_step + 1}"
                    done = True
                    break
                lr = data.lr_at(opt_step, total=total, peak=cfg.lr, warmup_frac=cfg.warmup_frac, final_frac=cfg.final_lr_frac)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device_sync is not None:
                    device_sync()
                opt_step += 1
                pos += len(group)
                dt = time.perf_counter() - t0
                step_times.append(dt)
                row = {
                    "opt_step": opt_step,
                    "epoch": epoch,
                    "loss": loss_sum / len(group),
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "s_per_opt_step": round(dt, 3),
                    **mem(),
                }
                peak = max(peak, float(row.get("peak_alloc_gib", 0.0)))
                log.write(json.dumps(row) + "\n")
                log.flush()
                if on_step is not None:
                    on_step(row)
                if cfg.empty_cache_every and opt_step % cfg.empty_cache_every == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                epoch_end = pos >= len(batches)
                if (cfg.eval_every and opt_step % cfg.eval_every == 0) or (not cfg.eval_every and epoch_end):
                    save_checkpoint(run_dir, "eval", opt_step, save_adapter=save_adapter)
                    eval_steps.append(opt_step)
                if epoch_end:
                    break
                if cfg.ckpt_every and opt_step % cfg.ckpt_every == 0:
                    save_checkpoint(run_dir, "resume", opt_step, save_adapter=save_adapter, opt=opt, state=ckpt_state())
                    _prune_resume(run_dir, cfg.keep_resume)
                if stop_flag is not None and stop_flag():
                    stop_now("stop requested")
                    done = True
                    break
            if done:
                break
            epoch += 1
            pos = 0
            if epoch < cfg.epochs:
                save_checkpoint(run_dir, "resume", opt_step, save_adapter=save_adapter, opt=opt, state=ckpt_state())
                _prune_resume(run_dir, cfg.keep_resume)
                if stop_flag is not None and stop_flag():
                    stop_now("stop requested")
                    done = True

    if outcome == "complete":
        final = run_dir / "final" / "adapter"
        final.mkdir(parents=True, exist_ok=True)
        save_adapter(final)
    return {
        "outcome": outcome,
        "reason": reason,
        "opt_steps_done": opt_step,
        "total_opt_steps": total,
        "eval_steps": eval_steps,
        "peak_alloc_gib": peak,
        "mean_s_per_opt_step": round(sum(step_times) / len(step_times), 3) if step_times else None,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False)
    except OSError:
        return None
    return out.stdout.strip() or None


def _versions() -> dict:
    out: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def run_manifest(cfg: RunConfig, **extra) -> dict:
    """Run manifest with every contract key; keyword arguments override or fill the defaults."""
    lock = repo_root() / LOCK_FILE
    manifest = {
        "run_id": cfg.run_id,
        "config": asdict(cfg),
        "trainset_sha256": cfg.trainset_sha256,
        "lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "versions": _versions(),
        "base_model": None,
        "lora": None,
        "pixel_policy": dict(pixels.PIXEL_POLICY),
        "prompt_sha256": None,
        "started": None,
        "finished": None,
        "outcome": None,
        "reason": None,
        "opt_steps_done": None,
        "total_opt_steps": None,
        "eval_steps": [],
        "peak_alloc_gib": None,
        "mean_s_per_opt_step": None,
        "git_commit": _git_commit(),
        "gpu_before": None,
        "gpu_after": None,
    }
    manifest.update(extra)
    return manifest


def write_manifest(run_dir: Path, manifest: dict) -> None:
    write_json(Path(run_dir) / "run_manifest.json", manifest)
