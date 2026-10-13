import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from lightonocr_math import guard, render


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch):
    monkeypatch.setattr(guard, "preflight", lambda *a, **k: None)
    # Scenario fakes only understand `docker run`; the Docker preflight has its own tests below.
    monkeypatch.setattr(render, "docker_available", lambda: True)


def _workdir(argv):
    return Path(argv[argv.index("-v") + 1].split(":/work")[0])


class Fake:
    """Stand-in for `docker run`: scenario picks the outcome from the argv and batch.tex."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, argv, *a, **k):
        if argv[:2] == ["docker", "kill"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        with self._lock:
            self.calls += 1
        return self.scenario(argv, _workdir(argv))


def healthy(argv, wd):
    (wd / "batch.log").write_text("This is pdfTeX\nOutput written\n")
    (wd / "batch.pdf").write_bytes(b"%PDF")
    return subprocess.CompletedProcess(argv, 0, "", "")


def outage(argv, wd):
    return subprocess.CompletedProcess(argv, 125, "", "docker: Cannot connect to the Docker daemon")


def _pages(monkeypatch):
    def fake(workdir, expected):
        page = np.full((200, 200), 255, dtype=np.uint8)
        page[90:110, 80:110] = 0
        return [page.copy() for _ in range(expected)]

    monkeypatch.setattr(render, "_rasterize", fake)


def _jsons(tmp_path):
    return list((tmp_path / "render").rglob("*.json")) if (tmp_path / "render").exists() else []


def _install(monkeypatch, scenario):
    f = Fake(scenario)
    monkeypatch.setattr(render.subprocess, "run", f)
    return f


def test_outage_raises_once_and_caches_nothing(tmp_path, monkeypatch):
    f = _install(monkeypatch, outage)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many(["a", "b", "c", "d"], cache=tmp_path, max_parallel=1)
    assert f.calls == 1
    assert _jsons(tmp_path) == []


@pytest.mark.parametrize("code", sorted(render.INFRA_EXIT_CODES))
def test_infra_exit_codes_raise_even_with_error_log(tmp_path, monkeypatch, code):
    def scenario(argv, wd):
        (wd / "batch.log").write_text("! Emergency stop.\n")
        return subprocess.CompletedProcess(argv, code, "", "boom")

    _install(monkeypatch, scenario)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many(["a", "b"], cache=tmp_path, max_parallel=1)
    assert _jsons(tmp_path) == []


def test_unexplained_nonzero_exit_raises(tmp_path, monkeypatch):
    def scenario(argv, wd):
        (wd / "batch.log").write_text("no errors here\n")
        return subprocess.CompletedProcess(argv, 1, "", "")

    _install(monkeypatch, scenario)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many(["a", "b"], cache=tmp_path, max_parallel=1)
    assert _jsons(tmp_path) == []


def test_missing_pdf_raises(tmp_path, monkeypatch):
    def scenario(argv, wd):
        (wd / "batch.log").write_text("fine\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    _install(monkeypatch, scenario)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many(["a"], cache=tmp_path, max_parallel=1)
    assert _jsons(tmp_path) == []


def test_ground_truth_not_dropped_after_outage(tmp_path, monkeypatch):
    _install(monkeypatch, outage)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_equivalence(["x", "y"], ["x", "z"], cache=tmp_path, max_parallel=1)
    _install(monkeypatch, healthy)
    _pages(monkeypatch)
    rows = render.render_equivalence(["x", "y"], ["x", "z"], cache=tmp_path, max_parallel=1)
    assert [r.gt_status for r in rows] == ["rendered", "rendered"]
    assert all(r.identical is not None for r in rows)


def test_latex_error_is_cached_with_bang_note(tmp_path, monkeypatch):
    def scenario(argv, wd):
        if "\\undefinedcmd" in (wd / "batch.tex").read_text():
            (wd / "batch.log").write_text("! Undefined control sequence.\n")
            return subprocess.CompletedProcess(argv, 1, "", "")
        return healthy(argv, wd)

    _install(monkeypatch, scenario)
    _pages(monkeypatch)
    eqs = ["a", "b", r"\undefinedcmd{c}", "d"]
    res = render.render_many(eqs, batch_size=8, cache=tmp_path, max_parallel=1)
    assert [r.status for r in res] == ["rendered", "rendered", "compile-failed", "rendered"]
    d = json.loads(render._entry(tmp_path, render.cache_key(eqs[2]), ".json").read_text())
    assert d["status"] == "compile-failed" and d["note"].startswith("!")


def test_timeout_returned_but_not_cached(tmp_path, monkeypatch):
    def run(argv, *a, **k):
        if argv[:2] == ["docker", "kill"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(render.subprocess, "run", run)
    (res,) = render.render_many(["a"], cache=tmp_path, max_parallel=1)
    assert res.status == "timeout"
    assert not render._entry(tmp_path, render.cache_key("a"), ".json").exists()
    _install(monkeypatch, healthy)
    _pages(monkeypatch)
    (res,) = render.render_many(["a"], cache=tmp_path, max_parallel=1)
    assert res.status == "rendered"


def test_legacy_entries_are_cache_misses(tmp_path):
    key = render.cache_key("a")
    p = render._entry(tmp_path, key, ".json")
    p.parent.mkdir(parents=True)
    base = {"shape": None, "bitmap_sha256": None}
    p.write_text(json.dumps({**base, "status": "compile-failed", "note": "pdflatex exit 125"}))
    assert render._cache_get(tmp_path, key) is None
    p.write_text(json.dumps({**base, "status": "timeout", "note": "timeout"}))
    assert render._cache_get(tmp_path, key) is None
    p.write_text(json.dumps({**base, "status": "compile-failed", "note": "! Undefined control sequence."}))
    got = render._cache_get(tmp_path, key)
    assert got is not None and got.status == "compile-failed"


def test_abort_flag_set_blocks_container(tmp_path, monkeypatch):
    f = _install(monkeypatch, healthy)
    ev = threading.Event()
    ev.set()
    with pytest.raises(render.RenderInfrastructureError):
        render._render_group([(render.cache_key("a"), "a")], tmp_path / "work", 30, tmp_path, abort=ev)
    assert f.calls == 0


def test_abort_flag_set_on_failure(tmp_path, monkeypatch):
    _install(monkeypatch, outage)
    ev = threading.Event()
    items = [(render.cache_key("a"), "a"), (render.cache_key("b"), "b")]
    with pytest.raises(render.RenderInfrastructureError):
        render._render_group(items, tmp_path / "work", 30, tmp_path, abort=ev)
    assert ev.is_set()


def test_parallel_abort_bounds_container_calls(tmp_path, monkeypatch):
    f = _install(monkeypatch, outage)
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many([f"e{i}" for i in range(8)], batch_size=2, cache=tmp_path, max_parallel=2)
    assert f.calls <= 2
    assert _jsons(tmp_path) == []


def test_unavailable_docker_raises_before_any_container(tmp_path, monkeypatch):
    f = _install(monkeypatch, healthy)
    monkeypatch.setattr(render, "docker_available", lambda: False)
    monkeypatch.setattr(render, "docker_diagnosis", lambda: "docker daemon unreachable")
    with pytest.raises(render.RenderInfrastructureError, match="not available.*unreachable"):
        render.render_many(["a", "b"], cache=tmp_path, max_parallel=1)
    assert f.calls == 0
    assert _jsons(tmp_path) == []


def test_fully_cached_run_needs_no_docker(tmp_path, monkeypatch):
    _pages(monkeypatch)
    _install(monkeypatch, healthy)
    first = render.render_many(["a", "b"], cache=tmp_path, max_parallel=1)
    monkeypatch.setattr(render, "docker_available", lambda: False)
    f = _install(monkeypatch, outage)
    second = render.render_many(["a", "b"], cache=tmp_path, max_parallel=1)
    assert [r.key for r in second] == [r.key for r in first]
    assert f.calls == 0


def test_docker_available_false_on_hung_daemon(monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired("docker", 20)

    monkeypatch.undo()
    monkeypatch.setattr(render.subprocess, "run", hang)
    assert render.docker_available() is False


def test_docker_diagnosis_distinguishes_daemon_from_image(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(render.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 1, b"", b""))
    assert "unreachable" in render.docker_diagnosis()
    monkeypatch.setattr(render.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, b"", b""))
    assert "docker pull" in render.docker_diagnosis()


def test_oserror_becomes_infra_error_sets_abort_and_leaks_no_workdir(tmp_path, monkeypatch):
    def run(argv, *a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(render.subprocess, "run", run)
    ev = threading.Event()
    with pytest.raises(render.RenderInfrastructureError, match="PermissionError"):
        render._render_group([(render.cache_key("a"), "a")], tmp_path / "work", 30, tmp_path, abort=ev)
    assert ev.is_set()
    assert list((tmp_path / "work").glob("batch-*")) == []


def test_real_error_not_masked_by_aborted_sibling(tmp_path, monkeypatch):
    # first group hits the abort flag (raised as "aborted"), a later group holds the real failure
    _install(monkeypatch, outage)
    real = render.RenderInfrastructureError("real docker failure (exit 125)")
    second_started = threading.Event()

    def fake_inner(items, *a, abort=None, errors=None, **k):
        if items[0][1] == "a":
            second_started.wait(5)
            raise render._AbortedError("aborted: x")
        second_started.set()
        raise real

    monkeypatch.setattr(render, "_render_group_inner", fake_inner)
    with pytest.raises(render.RenderInfrastructureError, match="real docker failure"):
        render.render_many(["a", "b"], batch_size=1, cache=tmp_path, max_parallel=2)
