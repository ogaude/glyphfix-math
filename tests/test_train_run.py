import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lightonocr_math.train import run

X = torch.linspace(-1, 1, 24).reshape(12, 2).repeat(1, 2)  # 12 rows x 4 features
Y = (X.sum(1, keepdim=True) * 0.5) + 0.1


def _setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)

    def loss_fn(idx):
        return torch.nn.functional.mse_loss(model(X[idx]), Y[idx])

    def save(d: Path):
        torch.save(model.state_dict(), Path(d) / "w.pt")

    def load(d: Path):
        model.load_state_dict(torch.load(Path(d) / "w.pt", weights_only=True))

    return model, loss_fn, save, load


def _plan(epoch):
    # 6 micro-batches of 2 per epoch, rotated by epoch for variety
    base = [[2 * i, 2 * i + 1] for i in range(6)]
    return base[epoch % 6 :] + base[: epoch % 6]


def _cfg(**kw):
    base = {"run_id": "t", "trainset": "x", "trainset_sha256": "0" * 64, "lr": 0.05, "epochs": 2, "grad_accum": 3,
            "ckpt_every": 1, "empty_cache_every": 0}
    base.update(kw)
    return run.RunConfig(**base)


def _go(run_dir, cfg, resume=False, **kw):
    model, loss_fn, save, load = _setup()
    res = run.train_loop(cfg, params=list(model.parameters()), loss_fn=loss_fn, batch_plan=_plan, save_adapter=save,
                         load_adapter=load, run_dir=run_dir, resume=resume, **kw)
    return model, res


def _losses(run_dir):
    return {r["opt_step"]: r["loss"] for r in map(json.loads, (run_dir / "log.jsonl").read_text().splitlines())}


def test_total_opt_steps():
    assert run.total_opt_steps(n_batches=10, epochs=2, grad_accum=4) == 6


def test_resume_equals_uninterrupted(tmp_path):
    full_dir, part_dir = tmp_path / "full", tmp_path / "part"
    m_full, res = _go(full_dir, _cfg())
    assert res["outcome"] == "complete" and res["opt_steps_done"] == 4
    cfg = _cfg()
    _, res_a = _go(part_dir, dataclasses.replace(cfg, max_opt_steps=2))
    assert res_a["outcome"] == "stopped" and res_a["opt_steps_done"] == 2
    m_b, res_b = _go(part_dir, cfg, resume=True)
    assert res_b["outcome"] == "complete" and res_b["opt_steps_done"] == 4
    for a, b in zip(m_full.parameters(), m_b.parameters(), strict=True):
        assert torch.equal(a, b)
    lf, lp = _losses(full_dir), _losses(part_dir)
    assert lf == lp and sorted(lp) == [1, 2, 3, 4]


def test_resume_mid_epoch_with_short_group(tmp_path):
    # grad_accum 4 over 6 micro-batches: groups of 4 and 2 per epoch
    cfg = _cfg(grad_accum=4)
    m_full, _ = _go(tmp_path / "f", cfg)
    _go(tmp_path / "p", dataclasses.replace(cfg, max_opt_steps=1))
    m_b, res = _go(tmp_path / "p", cfg, resume=True)
    assert res["opt_steps_done"] == 4
    for a, b in zip(m_full.parameters(), m_b.parameters(), strict=True):
        assert torch.equal(a, b)


def test_short_last_group_scaled_by_actual_size(tmp_path):
    cfg = _cfg(grad_accum=4, epochs=1, lr=0.0)
    seen = []
    model, loss_fn, save, _ = _setup()
    params = list(model.parameters())

    def spy(idx):
        return loss_fn(idx)

    run.train_loop(cfg, params=params, loss_fn=spy, batch_plan=_plan, save_adapter=save, run_dir=tmp_path / "r",
                   on_step=lambda row: seen.append(row))
    # second group has 2 micro-batches: logged loss is the mean of those two
    expect = (loss_fn([8, 9]).item() + loss_fn([10, 11]).item()) / 2
    assert seen[1]["loss"] == pytest.approx(expect)


def test_nan_stops_without_checkpoint(tmp_path):
    model, loss_fn, save, _ = _setup()
    calls = {"n": 0}

    def bad(idx):
        calls["n"] += 1
        loss = loss_fn(idx)
        return loss * float("nan") if calls["n"] > 3 else loss

    cfg = _cfg(ckpt_every=0)
    res = run.train_loop(cfg, params=list(model.parameters()), loss_fn=bad, batch_plan=_plan, save_adapter=save,
                         run_dir=tmp_path / "r")
    assert res["outcome"] == "nan"
    assert "optimizer step 2" in res["reason"]
    assert run.load_latest_resume(tmp_path / "r") is None
    assert not (tmp_path / "r" / "final").exists()


def test_checkpoint_pruning_and_eval_and_final(tmp_path):
    cfg = _cfg(epochs=3, grad_accum=2, ckpt_every=1, keep_resume=2)
    d = tmp_path / "r"
    _, res = _go(d, cfg)
    assert res["outcome"] == "complete"
    kept = sorted(p.name for p in (d / "resume").iterdir())
    assert len(kept) == 2
    # eval_every == 0: one eval adapter per epoch end, kept
    assert res["eval_steps"] == [3, 6, 9]
    assert all((d / "eval" / f"step-{s}" / "adapter" / "w.pt").is_file() for s in res["eval_steps"])
    assert (d / "final" / "adapter" / "w.pt").is_file()


def test_eval_every(tmp_path):
    cfg = _cfg(epochs=1, grad_accum=1, eval_every=2, ckpt_every=0)
    _, res = _go(tmp_path / "r", cfg)
    assert res["eval_steps"] == [2, 4, 6]


def test_max_opt_steps_writes_resume_checkpoint(tmp_path):
    _, res = _go(tmp_path / "r", _cfg(max_opt_steps=3, ckpt_every=0))
    assert res["outcome"] == "stopped"
    ck = run.load_latest_resume(tmp_path / "r")
    assert ck is not None and ck["state"]["opt_step"] == 3
    assert not (tmp_path / "r" / "final").exists()


def test_stop_flag_saves_at_boundary(tmp_path):
    flag = {"set": False}

    def on_step(row):
        if row["opt_step"] == 1:
            flag["set"] = True

    _, res = _go(tmp_path / "r", _cfg(ckpt_every=0), on_step=on_step, stop_flag=lambda: flag["set"])
    assert res["outcome"] == "stopped" and res["opt_steps_done"] == 1
    assert run.load_latest_resume(tmp_path / "r")["state"]["opt_step"] == 1


def test_run_manifest_contract(tmp_path):
    m = run.run_manifest(_cfg(), outcome="complete")
    keys = {"run_id", "config", "trainset_sha256", "lock_sha256", "versions", "base_model", "lora", "pixel_policy",
            "prompt_sha256", "started", "finished", "outcome", "reason", "opt_steps_done", "total_opt_steps",
            "eval_steps", "peak_alloc_gib", "mean_s_per_opt_step", "git_commit", "gpu_before", "gpu_after"}
    assert keys <= set(m)
    assert m["config"]["grad_accum"] == 3 and m["outcome"] == "complete"
    assert m["pixel_policy"]["max_px"] == 602112
    run.write_manifest(tmp_path, m)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["run_id"] == "t"
