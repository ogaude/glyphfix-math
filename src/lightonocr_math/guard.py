"""Resource guard: keep heavy jobs from starving the shared GPU and the 1.5 GiB RAM floor.

Usage:
  from lightonocr_math.guard import guard, preflight, docker_parallel_cap
  with guard(vram_gb=8, ram_gb=4, workers=2) as g:
      for batch in batches:
          g.wait_if_paused()
          g.check()          # raises GuardAbort once the watchdog gave up
          ...
  uv run python -m lightonocr_math.guard --vram-gb 8 --ram-gb 4 -- python train.py

Behaviour:
  preflight  Before work starts, wait (polling every poll_s) until the declared need fits:
             MemAvailable >= ram_gb * max(workers, 1) GiB + 1.5 GiB floor, and free VRAM
             >= vram_gb GiB. After timeout_s the guard exits 75 naming the short resource
             (and, for VRAM, who holds it: rocm-smi pids and `ollama ps`). Ollama models
             are reported, never unloaded.
  watchdog   While the job runs a thread samples every watch_s. MemAvailable below the
             floor, or free VRAM below vram_floor_gb (when a VRAM need was declared),
             pauses registered children with SIGSTOP and resumes them with SIGCONT on
             recovery. No recovery within grace_s terminates them (SIGTERM, then SIGKILL)
             and the guard exits 76.
  fail closed  A declared VRAM need with neither rocm-smi nor an already-imported torch
             able to report free VRAM exits 78; the job never runs unguarded.

Exit codes: 75 preflight timeout, 76 watchdog killed the job, 78 VRAM unmeasurable.
Otherwise the wrapped command's own exit code is returned.

Known limits:
  - SIGSTOP on a `docker run` client does not pause the container (the daemon owns it).
    Docker jobs must use `--memory 4g` (DOCKER_MEMORY) and cap parallelism with
    docker_parallel_cap().
  - The in-process guard can only pause cooperative loops (wait_if_paused / check); the
    watchdog thread never raises asynchronously.
  - Only the first GPU (card0) is sampled by default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

GIB = 1024**3
RAM_FLOOR_BYTES = int(1.5 * GIB)
DOCKER_MEMORY = "4g"
DOCKER_MEMORY_GB = 4
DOCKER_MAX_PARALLEL = 3

EXIT_PREFLIGHT_TIMEOUT = 75
EXIT_WATCHDOG_KILLED = 76
EXIT_UNMEASURABLE = 78

_TOOL_TIMEOUT_S = 10


class GuardError(RuntimeError):
    exit_code = 1


class GuardTimeout(GuardError):
    exit_code = EXIT_PREFLIGHT_TIMEOUT


class GuardAbort(GuardError):
    exit_code = EXIT_WATCHDOG_KILLED


class GuardUnmeasurable(GuardError):
    exit_code = EXIT_UNMEASURABLE


def mem_available_bytes(meminfo_path: str = "/proc/meminfo") -> int:
    with open(meminfo_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise GuardUnmeasurable(f"MemAvailable not found in {meminfo_path}")


def _run_tool(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_TOOL_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout


def vram_free_bytes(device: str = "card0") -> int | None:
    """Free VRAM on `device` from rocm-smi, or None if it cannot be read."""
    out = _run_tool(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    start = out.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(out[start:])
        info = data.get(device)
        if info is None:
            cards = [k for k in data if str(k).startswith("card")]
            if not cards:
                return None
            info = data[min(cards)]
        total = int(info["VRAM Total Memory (B)"])
        used = int(info["VRAM Total Used Memory (B)"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return max(total - used, 0)


def torch_vram_free_bytes() -> int | None:
    """Free VRAM via torch, only if torch is already imported (this module never imports it)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - any torch/driver failure means "unmeasurable"
        return None
    return int(free)


def vram_holders() -> str:
    """Short report of who holds VRAM (rocm-smi pids, ollama ps). Never raises, never unloads anything."""
    parts = []
    for label, cmd in (("rocm-smi pids", ["rocm-smi", "--showpids"]), ("ollama ps", ["ollama", "ps"])):
        out = _run_tool(cmd)
        text = out.strip() if out and out.strip() else "unavailable"
        parts.append(f"{label}:\n{text}")
    return "\n".join(parts)


@dataclass
class Sampler:
    mem: Callable[[], int]
    vram: Callable[[], int | None]
    holders: Callable[[], str]


def _default_vram() -> int | None:
    free = vram_free_bytes()
    return free if free is not None else torch_vram_free_bytes()


def default_sampler() -> Sampler:
    return Sampler(mem=mem_available_bytes, vram=_default_vram, holders=vram_holders)


def _gib(n: float) -> str:
    return f"{n / GIB:.1f} GiB"


def preflight(
    vram_gb: float = 0.0,
    ram_gb: float = 0.0,
    workers: int = 1,
    *,
    timeout_s: float = 600,
    poll_s: float = 10,
    sampler: Sampler | None = None,
    log: Callable[[str], object] = print,
) -> None:
    """Wait until the declared VRAM/RAM need fits, else raise GuardTimeout (75) or GuardUnmeasurable (78).

    Needed MemAvailable = ram_gb * max(workers, 1) GiB + the 1.5 GiB floor, so a starting job never
    pushes the system under the floor.
    """
    s = sampler or default_sampler()
    need_ram = int(ram_gb * max(workers, 1) * GIB) + RAM_FLOOR_BYTES
    need_vram = int(vram_gb * GIB)
    deadline = time.monotonic() + timeout_s
    while True:
        short: list[str] = []
        mem = s.mem()
        if mem < need_ram:
            short.append(
                f"short: MemAvailable {_gib(mem)} < needed {_gib(need_ram)} "
                f"(job {ram_gb * max(workers, 1):.1f} GiB + floor {_gib(RAM_FLOOR_BYTES)})"
            )
        if need_vram > 0:
            free = s.vram()
            if free is None:
                raise GuardUnmeasurable("VRAM unmeasurable: neither rocm-smi nor torch can report free VRAM; refusing to run unguarded")
            if free < need_vram:
                short.append(f"short: VRAM free {_gib(free)} < needed {_gib(need_vram)}; holders:\n{s.holders()}")
        if not short:
            return
        text = "\n".join(short)
        if time.monotonic() >= deadline:
            raise GuardTimeout(f"{text}\naborting, no work started")
        log(text)
        time.sleep(poll_s)


def docker_parallel_cap(
    *, sampler: Sampler | None = None, per_container_gb: float = DOCKER_MEMORY_GB, max_parallel: int = DOCKER_MAX_PARALLEL
) -> int:
    """How many Docker containers fit now: min(max_parallel, floor((MemAvailable - floor) / per_container)); may be 0."""
    s = sampler or default_sampler()
    spare = s.mem() - RAM_FLOOR_BYTES
    if spare <= 0:
        return 0
    return max(0, min(max_parallel, int(spare // (per_container_gb * GIB))))


class Guard:
    def __init__(self, *, vram_gb: float, watch_s: float, grace_s: float, vram_floor_gb: float, sampler: Sampler, log: Callable[[str], object]):
        self.paused = False
        self.aborted = False
        self._vram_gb = vram_gb
        self._watch_s = watch_s
        self._grace_s = grace_s
        self._vram_floor = int(vram_floor_gb * GIB)
        self._sampler = sampler
        self._log = log
        self._pids: list[int] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused_since = 0.0

    def register_child(self, proc_or_pid) -> None:
        pid = int(getattr(proc_or_pid, "pid", proc_or_pid))
        with self._lock:
            if pid not in self._pids:
                self._pids.append(pid)

    def check(self) -> None:
        if self.aborted:
            raise GuardAbort("watchdog terminated the job: resources stayed below the floor past the grace period")

    def wait_if_paused(self, timeout_s: float | None = None) -> None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while self.paused and not self.aborted:
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(min(self._watch_s, 0.1))

    def _signal(self, sig: int) -> None:
        with self._lock:
            pids = list(self._pids)
        for pid in pids:
            try:
                pgid = os.getpgid(pid)
                if pgid != os.getpgrp():
                    os.killpg(pgid, sig)
                else:
                    os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                continue

    def _pressure(self) -> str | None:
        mem = self._sampler.mem()
        if mem < RAM_FLOOR_BYTES:
            return f"MemAvailable {_gib(mem)} below floor {_gib(RAM_FLOOR_BYTES)}"
        if self._vram_gb > 0:
            free = self._sampler.vram()
            if free is not None and free < self._vram_floor:
                return f"free VRAM {_gib(free)} below floor {_gib(self._vram_floor)}"
        return None

    def _abort(self) -> None:
        self._log("guard: no recovery within grace, terminating job")
        self.aborted = True
        self._signal(signal.SIGTERM)
        self._signal(signal.SIGCONT)
        end = time.monotonic() + 10
        while time.monotonic() < end and self._alive():
            time.sleep(0.05)
        self._signal(signal.SIGKILL)

    def _alive(self) -> bool:
        with self._lock:
            pids = list(self._pids)
        for pid in pids:
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                    if fh.read().rsplit(")", 1)[1].split()[0] != "Z":
                        return True
            except (OSError, IndexError):
                continue
        return False

    def _loop(self) -> None:
        while not self._stop.wait(self._watch_s):
            try:
                reason = self._pressure()
            except Exception:  # noqa: BLE001, S112 - a failed sample must not kill the watchdog
                continue
            now = time.monotonic()
            if reason:
                if not self.paused:
                    self.paused = True
                    self._paused_since = now
                    self._log(f"guard: pausing job: {reason}")
                    self._signal(signal.SIGSTOP)
                elif now - self._paused_since > self._grace_s:
                    self._abort()
                    return
            elif self.paused:
                self._signal(signal.SIGCONT)
                self.paused = False
                self._log("guard: resources recovered, resuming job")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="guard-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


@contextlib.contextmanager
def guard(
    vram_gb: float = 0.0,
    ram_gb: float = 0.0,
    workers: int = 1,
    *,
    timeout_s: float = 600,
    poll_s: float = 10,
    watch_s: float = 2,
    grace_s: float = 120,
    vram_floor_gb: float = 0.25,
    sampler: Sampler | None = None,
) -> Iterator[Guard]:
    s = sampler or default_sampler()
    preflight(vram_gb, ram_gb, workers, timeout_s=timeout_s, poll_s=poll_s, sampler=s)
    g = Guard(
        vram_gb=vram_gb,
        watch_s=watch_s,
        grace_s=grace_s,
        vram_floor_gb=vram_floor_gb,
        sampler=s,
        log=lambda m: print(m, file=sys.stderr),
    )
    g.start()
    try:
        yield g
    finally:
        g.stop()
    g.check()


def run_guarded(cmd: list[str], **kwargs) -> int:
    """Run cmd under the guard in its own session; return its exit code, or 75 / 76 / 78."""
    try:
        with guard(**kwargs) as g:
            proc = subprocess.Popen(cmd, start_new_session=True)
            g.register_child(proc)
            try:
                while proc.poll() is None:
                    time.sleep(0.05)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if g.aborted:
                return EXIT_WATCHDOG_KILLED
            return proc.returncode
    except GuardError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lightonocr_math.guard", description="Run a command under the resource guard.")
    ap.add_argument("--vram-gb", type=float, default=0.0)
    ap.add_argument("--ram-gb", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600, help="preflight timeout in seconds")
    ap.add_argument("--poll", type=float, default=10, help="preflight poll interval in seconds")
    ap.add_argument("--grace", type=float, default=120, help="seconds paused before the job is terminated")
    ap.add_argument("--vram-floor-gb", type=float, default=0.25)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given (use: ... -- cmd args)")
    return run_guarded(
        cmd,
        vram_gb=args.vram_gb,
        ram_gb=args.ram_gb,
        workers=args.workers,
        timeout_s=args.timeout,
        poll_s=args.poll,
        grace_s=args.grace,
        vram_floor_gb=args.vram_floor_gb,
    )


if __name__ == "__main__":
    raise SystemExit(main())
