import json
import subprocess
import threading
import time

import pytest

from lightonocr_math import guard as G
from lightonocr_math.guard import GIB, Sampler


def fake(mem=100 * GIB, vram=100 * GIB, holders="holder-info"):
    return Sampler(mem=lambda: mem() if callable(mem) else mem, vram=lambda: vram() if callable(vram) else vram, holders=lambda: holders)


def test_mem_available(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:  9999999 kB\nMemAvailable:   2048000 kB\n")
    assert G.mem_available_bytes(str(p)) == 2048000 * 1024


def _patch_run(monkeypatch, stdout=None, exc=None):
    def run(cmd, **kw):
        if exc:
            raise exc
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(G.subprocess, "run", run)


def test_vram_free_parses_after_warning(monkeypatch):
    payload = {"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "1000"}}
    _patch_run(monkeypatch, stdout="WARNING: something\n" + json.dumps(payload) + "\n")
    assert G.vram_free_bytes() == 17163091968 - 1000


def test_vram_free_none_cases(monkeypatch):
    _patch_run(monkeypatch, exc=FileNotFoundError())
    assert G.vram_free_bytes() is None
    _patch_run(monkeypatch, stdout="no json here")
    assert G.vram_free_bytes() is None


def test_nvidia_free_parses_first_gpu(monkeypatch):
    _patch_run(monkeypatch, stdout="15000\n8000\n")
    assert G.nvidia_vram_free_bytes() == 15000 * 1024**2


def test_nvidia_free_none_cases(monkeypatch):
    _patch_run(monkeypatch, exc=FileNotFoundError())
    assert G.nvidia_vram_free_bytes() is None
    _patch_run(monkeypatch, stdout="N/A\n")
    assert G.nvidia_vram_free_bytes() is None


def test_default_vram_order(monkeypatch):
    monkeypatch.setattr(G, "vram_free_bytes", lambda *a: None)
    monkeypatch.setattr(G, "nvidia_vram_free_bytes", lambda *a: 7)
    monkeypatch.setattr(G, "torch_vram_free_bytes", lambda: 9)
    assert G._default_vram() == 7
    monkeypatch.setattr(G, "nvidia_vram_free_bytes", lambda *a: None)
    assert G._default_vram() == 9
    monkeypatch.setattr(G, "vram_free_bytes", lambda *a: 5)
    assert G._default_vram() == 5


def test_preflight_ram_timeout():
    with pytest.raises(G.GuardTimeout) as ei:
        G.preflight(ram_gb=4, timeout_s=0.3, poll_s=0.1, sampler=fake(mem=3 * GIB), log=lambda m: None)
    assert ei.value.exit_code == 75
    assert "MemAvailable" in str(ei.value)
    assert "3.0 GiB" in str(ei.value) and "5.5 GiB" in str(ei.value)


def test_preflight_vram_timeout_names_holders():
    with pytest.raises(G.GuardTimeout) as ei:
        G.preflight(vram_gb=8, timeout_s=0.2, poll_s=0.1, sampler=fake(vram=4 * GIB), log=lambda m: None)
    assert "VRAM" in str(ei.value) and "holder-info" in str(ei.value)


def test_preflight_unmeasurable():
    with pytest.raises(G.GuardUnmeasurable) as ei:
        G.preflight(vram_gb=8, sampler=fake(vram=None))
    assert ei.value.exit_code == 78


def test_preflight_ok_and_waits_until_sufficient():
    G.preflight(ram_gb=1, vram_gb=1, sampler=fake())
    calls = {"n": 0}

    def mem():
        calls["n"] += 1
        return 10 * GIB if calls["n"] >= 3 else 1 * GIB

    G.preflight(ram_gb=2, poll_s=0.01, timeout_s=5, sampler=fake(mem=mem), log=lambda m: None)
    assert calls["n"] == 3


@pytest.mark.parametrize(("mem_gib", "expected"), [(9, 1), (20, 3), (5, 0), (1, 0)])
def test_docker_parallel_cap(mem_gib, expected):
    assert G.docker_parallel_cap(sampler=fake(mem=mem_gib * GIB)) == expected


def _timed_mem(t0, low_from, low_until):
    def mem():
        t = time.monotonic() - t0
        return 0 if low_from <= t < low_until else 100 * GIB

    return mem


def _state(pid):
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().rsplit(")", 1)[1].split()[0]


def test_run_guarded_propagates_exit_code():
    assert G.run_guarded(["sh", "-c", "exit 3"], ram_gb=0.1, sampler=fake()) == 3


def test_run_guarded_pause_resume_and_kill(monkeypatch):
    # recoverable pressure: child is stopped, then resumes and finishes with 0
    seen = []
    real_popen = G.subprocess.Popen

    def popen(*a, **kw):
        p = real_popen(*a, **kw)
        seen.append(p)
        return p

    monkeypatch.setattr(G.subprocess, "Popen", popen)
    t0 = time.monotonic()
    states = []

    def watcher():
        time.sleep(0.5)
        states.append(_state(seen[0].pid))

    th = threading.Thread(target=watcher)
    s = fake(mem=_timed_mem(t0 + 0.2, 0.0, 0.8))
    th.start()
    rc = G.run_guarded(["sleep", "1.2"], ram_gb=0.1, watch_s=0.05, grace_s=5, sampler=s)
    th.join()
    assert states == ["T"]
    assert rc == 0

    # unrecoverable pressure: terminated, exit 76
    t0 = time.monotonic()
    s = fake(mem=_timed_mem(t0 + 0.2, 0.0, 1e9))
    start = time.monotonic()
    rc = G.run_guarded(["sleep", "30"], ram_gb=0.1, watch_s=0.05, grace_s=1, sampler=s)
    assert rc == 76
    assert time.monotonic() - start < 8


def test_guard_in_process_abort():
    t0 = time.monotonic()
    s = fake(mem=_timed_mem(t0 + 0.1, 0.0, 1e9))
    with pytest.raises(G.GuardAbort) as ei, G.guard(ram_gb=1, watch_s=0.05, grace_s=0.3, sampler=s) as g:
        deadline = time.monotonic() + 5
        seen_paused = False
        while time.monotonic() < deadline and not g.aborted:
            seen_paused = seen_paused or g.paused
            time.sleep(0.02)
        assert seen_paused
        g.check()
    assert ei.value.exit_code == 76


def test_run_guarded_exit_codes():
    assert G.run_guarded(["true"], ram_gb=1000, timeout_s=0.2, poll_s=0.1, sampler=fake(mem=2 * GIB)) == 75
    assert G.run_guarded(["true"], vram_gb=4, sampler=fake(vram=None)) == 78


def test_main_ok():
    assert G.main(["--ram-gb", "1", "--", "true"]) == 0
