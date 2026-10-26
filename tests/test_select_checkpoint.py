import json

import pytest

from scripts import select_checkpoint as sc


def _pc(**errs):
    out = {c: {"errors": 0, "denom": 0, "rate": None} for c in sc.CLASSES}
    for c, (e, d) in errs.items():
        out[c] = {"errors": e, "denom": d, "rate": e / d if d else None}
    return out


def _entry(step, ri, gem=None):
    return {"step": step, "render_identical": ri, "glyph_error_mean": gem}


def test_choose_step_tie_goes_to_lower_glyph_error():
    steps = [_entry(100, 0.50, 0.20), _entry(200, 0.51, 0.30), _entry(300, 0.45, 0.0)]
    assert sc.choose_step(steps)["step"] == 100


def test_choose_step_equal_glyph_error_takes_earlier_step():
    steps = [_entry(300, 0.51, 0.2), _entry(100, 0.50, 0.2), _entry(200, 0.50, 0.2)]
    assert sc.choose_step(steps)["step"] == 100


def test_choose_step_clear_best_wins_over_lower_glyph_error():
    steps = [_entry(100, 0.40, 0.0), _entry(200, 0.50, 0.9)]
    assert sc.choose_step(steps)["step"] == 200


def test_glyph_error_mean_uses_only_classes_with_denominator():
    pc = _pc(prime_l=(1, 4), i_1=(0, 2), other=(9, 10))
    assert sc.glyph_error_mean(pc) == pytest.approx((0.25 + 0.0) / 2)
    assert sc.glyph_error_mean(_pc()) is None


def _cfg(run_id, ri):
    return {"run_id": run_id, "render_identical": ri}


def test_choose_winner_tie_prefers_lower_lr():
    cfgs = [_cfg("lr1e-4-a16", 0.70), _cfg("lr2e-4-a8", 0.69), _cfg("lr1e-4-a8", 0.685)]
    got = sc.choose_winner(cfgs)
    assert got["winner"] == "lr1e-4-a8" and got["tie"] is True


def test_choose_winner_clear_lead_without_pairwise():
    cfgs = [_cfg("lr1e-4-a16", 0.75), _cfg("lr2e-4-a8", 0.69), _cfg("lr1e-4-a8", 0.65)]
    got = sc.choose_winner(cfgs)
    assert got["winner"] == "lr1e-4-a16" and got["tie"] is False


def test_choose_winner_ci_including_zero_is_a_tie():
    cfgs = [_cfg("lr1e-4-a16", 0.75), _cfg("lr2e-4-a8", 0.72), _cfg("lr1e-4-a8", 0.60)]
    pw = {"lr1e-4-a16|lr2e-4-a8": {"delta": -0.03, "ci_low": -0.08, "ci_high": 0.02}}
    got = sc.choose_winner(cfgs, pw)
    assert got["tie"] is True and got["winner"] == "lr2e-4-a8"


def test_choose_winner_ci_excluding_zero_keeps_best():
    cfgs = [_cfg("lr1e-4-a16", 0.75), _cfg("lr2e-4-a8", 0.72)]
    pw = {"lr1e-4-a16|lr2e-4-a8": {"delta": -0.03, "ci_low": -0.06, "ci_high": -0.01}}
    got = sc.choose_winner(cfgs, pw)
    assert got["tie"] is False and got["winner"] == "lr1e-4-a16"


def _rows(flags, papers):
    return {f"r{i}": {"id": f"r{i}", "paper_id": papers[i], "render_identical": f} for i, f in enumerate(flags)}


def test_paired_vs_only_common_decided_ids():
    base = _rows([True, False, None, False], ["p1", "p1", "p2", "p3"])
    cand = _rows([True, True, True, None], ["p1", "p1", "p2", "p3"])
    delta, flips = sc.paired_vs(base, cand, B=200)
    assert delta["den"] == 2 and flips["n"] == 2
    assert flips["wrong_to_right"] == 1 and flips["both_right"] == 1


def _make_score_dir(d, flags, ri, per_class, papers=None, sha="x"):
    d.mkdir(parents=True)
    papers = papers or [f"p{i}" for i in range(len(flags))]
    rows = [{"id": f"r{i}", "paper_id": papers[i], "render_identical": f} for i, f in enumerate(flags)]
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = {"metrics": {"render_identical": {"rate": ri}, "exact": {"rate": ri}}, "per_class": per_class,
           "scorer_version": "9", "scorer_fingerprint": "fp", "predictions_sha256": sha}  # fmt: skip
    (d / "results.json").write_text(json.dumps(res), encoding="utf-8")


def _make_run(root, name, per_step_flags):
    run = root / name
    for step, (flags, ri, pc) in per_step_flags.items():
        _make_score_dir(run / "dev_eval" / f"step-{step}.score", flags, ri, pc)
        ad = run / "eval" / f"step-{step}" / "adapter"
        ad.mkdir(parents=True)
        (ad / "adapter_model.safetensors").write_bytes(f"w{step}".encode())
    return run


def test_select_writes_contract(tmp_path):
    base = tmp_path / "base.score"
    _make_score_dir(base, [True, False, False, False], 0.25, _pc())
    run = _make_run(tmp_path, "lr1e-4-a8", {
        100: ([True, True, False, False], 0.50, _pc(prime_l=(1, 2))),
        200: ([True, True, True, False], 0.51, _pc(prime_l=(2, 2))),
    })  # fmt: skip
    args = sc.make_parser().parse_args(["select", "--run-dir", str(run), "--base-score", str(base)])
    assert sc.main(["select", "--run-dir", str(run), "--base-score", str(base)]) == 0 and args.cmd == "select"
    sel = json.loads((run / "selection.json").read_text())
    assert sel["chosen_step"] == 100
    assert set(sel) >= {"rule", "tie_delta", "per_step", "chosen_step", "chosen_adapter", "adapter_sha256", "delta_vs_base",
                        "flip_vs_base", "scorer_version", "scorer_fingerprint"}  # fmt: skip
    assert sel["scorer_fingerprint"] == "fp" and len(sel["adapter_sha256"]) == 64
    assert sel["delta_vs_base"]["delta"] == pytest.approx(0.25)


def test_sweep_clear_lead_and_report(tmp_path):
    base = tmp_path / "base.score"
    _make_score_dir(base, [i < 10 for i in range(40)], 0.25, _pc())
    spec = {
        "lr1e-4-a8": ([i < 30 for i in range(40)], 0.75),
        "lr2e-4-a8": ([i < 30 for i in range(40)], 0.75),
        "lr1e-4-a16": ([True] * 40, 1.0),
    }
    runs = []
    for name, (flags, ri) in spec.items():
        run = _make_run(tmp_path, name, {100: (flags, ri, _pc(prime_l=(1, 4)))})
        assert sc.main(["select", "--run-dir", str(run), "--base-score", str(base)]) == 0
        runs += ["--run", str(run)]
    out, rep = tmp_path / "sweep.json", tmp_path / "sweep.md"
    assert sc.main(["sweep", *runs, "--base-score", str(base), "--out", str(out), "--report", str(rep)]) == 0
    sweep = json.loads(out.read_text())
    assert [c["run_id"] for c in sweep["configs"]] == list(spec)
    assert sweep["winner"] == "lr1e-4-a16" and sweep["tie"] is False
    assert "Winner: lr1e-4-a16" in rep.read_text()


def test_sweep_exact_tie_prefers_lower_lr(tmp_path):
    base = tmp_path / "base.score"
    _make_score_dir(base, [i < 10 for i in range(40)], 0.25, _pc())
    runs = []
    for name in ("lr1e-4-a16", "lr2e-4-a8", "lr1e-4-a8"):
        run = _make_run(tmp_path, name, {100: ([i < 30 for i in range(40)], 0.75, _pc())})
        assert sc.main(["select", "--run-dir", str(run), "--base-score", str(base)]) == 0
        runs += ["--run", str(run)]
    out = tmp_path / "sweep.json"
    assert sc.main(["sweep", *runs, "--base-score", str(base), "--out", str(out)]) == 0
    sweep = json.loads(out.read_text())
    assert sweep["winner"] == "lr1e-4-a8" and sweep["tie"] is True


def test_score_skips_unchanged_predictions(tmp_path, monkeypatch):
    from glyphfix_math import score

    pred = tmp_path / "step-1.pred.jsonl"
    pred.write_text('{"id": "a"}\n', encoding="utf-8")
    out = tmp_path / "step-1.score"
    out.mkdir()
    (out / "results.json").write_text(json.dumps({"predictions_sha256": sc.sha256_file(pred)}), encoding="utf-8")
    called = []
    monkeypatch.setattr(score, "score_file", lambda *a, **k: called.append(1))
    assert sc.score_one(pred, out) == "skipped" and not called
    pred.write_text('{"id": "b"}\n', encoding="utf-8")
    assert sc.score_one(pred, out) == "scored" and called


def test_select_bad_input_exit_2(tmp_path):
    assert sc.main(["select", "--run-dir", str(tmp_path / "none"), "--base-score", str(tmp_path / "nob")]) == 2
