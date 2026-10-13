"""Driver logic of the training-fit grid (torch-free)."""

from __future__ import annotations

import json

from scripts.spikes import train_fit_grid as g


def _row(**kw):
    base = {
        "cell_id": "bs1-px200704-flaoff", "bs": 1, "max_px": 200704, "fla_arm": "off", "fla_active": False,
        "outcome": "ok", "steps": 20, "steps_done": 20, "was_paused": False, "peak_reserved_gib": 11.0,
        "own_card_peak_gib": 11.5, "s_per_step": 6.0, "free_at_start_gib": 15.8,
    }
    base.update(kw)
    return base


def test_grid_cells():
    cells = list(g.grid_cells())
    assert len(cells) == 18
    assert len({c["cell_id"] for c in cells}) == 18
    assert "bs2-px401408-flaon" in {c["cell_id"] for c in cells}
    assert all(c["rank"] == 32 and c["gc"] == "on" for c in cells)


def test_fits():
    assert g.fits(_row(peak_reserved_gib=13.9, own_card_peak_gib=14.2))
    assert not g.fits(_row(peak_reserved_gib=14.6))
    assert not g.fits(_row(own_card_peak_gib=14.6))
    assert not g.fits(_row(was_paused=True))
    assert not g.fits(_row(outcome="oom"))
    assert not g.fits(_row(steps_done=19))


def test_classify_outcome():
    for code in (75, 76, 78):
        c = g.classify_outcome(guard_exit=code)
        assert c["outcome"] == "guard-abort" and c["inconclusive"] is True
    c = g.classify_outcome(outcome="oom", free_at_start_gib=12.0)
    assert c["inconclusive"] is True
    c = g.classify_outcome(outcome="oom", free_at_start_gib=15.0)
    assert c["inconclusive"] is False and c["verdict"] == "does-not-fit"
    assert g.classify_outcome(outcome="ok", was_paused=True, free_at_start_gib=15.0)["inconclusive"] is True
    assert g.classify_outcome(outcome="ok", free_at_start_gib=15.0)["inconclusive"] is False
    for oc in ("timeout", "crash"):
        assert g.classify_outcome(outcome=oc, free_at_start_gib=15.0)["inconclusive"] is True


def test_best_cell():
    a = _row(cell_id="a", bs=1, max_px=401408)
    b = _row(cell_id="b", bs=2, max_px=401408, fla_active=False, s_per_step=5.0)
    c = _row(cell_id="c", bs=2, max_px=401408, fla_active=True, s_per_step=9.0)
    d = _row(cell_id="d", bs=2, max_px=401408, fla_active=True, s_per_step=4.0)
    e = _row(cell_id="e", bs=4, max_px=602112, outcome="oom")
    assert g.best_cell([a, b, c, d, e])["cell_id"] == "d"
    assert g.best_cell([e]) is None


def test_fla_unavailable_rows_not_launched(tmp_path):
    env = tmp_path / "env_verify.json"
    env.write_text(json.dumps({"fla": {"import_ok": False}}))
    launched = []
    rows = g.run_grid(
        results=tmp_path / "rows.jsonl", env_verify=env, launch=lambda spec, **kw: launched.append(spec["cell_id"]) or _row(cell_id=spec["cell_id"]),
        only="bs1-px200704-flaon",
    )
    assert launched == []
    assert rows[-1]["outcome"] == "fla-unavailable" and rows[-1]["fits"] is False


def test_rerun_skips_conclusive_and_retries_inconclusive(tmp_path):
    env = tmp_path / "env_verify.json"
    env.write_text(json.dumps({"fla": {"import_ok": True}}))
    results = tmp_path / "rows.jsonl"
    calls = []

    def launch(spec, **kw):
        calls.append(spec["cell_id"])
        return _row(cell_id=spec["cell_id"], inconclusive=len(calls) == 1)

    g.run_grid(results=results, env_verify=env, launch=launch, only="bs1-px200704-flaoff")
    g.run_grid(results=results, env_verify=env, launch=launch, only="bs1-px200704-flaoff")
    assert calls == ["bs1-px200704-flaoff"]  # inconclusive but no --retry-inconclusive
    g.run_grid(results=results, env_verify=env, launch=launch, only="bs1-px200704-flaoff", retry_inconclusive=True)
    assert len(calls) == 2
    g.run_grid(results=results, env_verify=env, launch=launch, only="bs1-px200704-flaoff", retry_inconclusive=True)
    assert len(calls) == 2  # now conclusive
