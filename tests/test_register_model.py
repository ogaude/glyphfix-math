import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import register_model as rm

G = rm.GATES


# ------------------------------------------------------------------ regression gate


def _reg(fixes=None, control_flips=(), target_flips=0, extra_classes=True):
    """Synthetic regression rows and scored arms. control_flips: pool_flags with one right-to-wrong each."""
    fixes = fixes or {}
    rows, stock, cand = [], {}, {}

    def add(i, cls, flag, s, c):
        rows.append({"id": i, "regression_class": cls, "pool_flag": flag})
        stock[i] = {"render_identical": s}
        cand[i] = {"render_identical": c}

    flips = list(control_flips)
    for flag in ("plain", "prime", "greek", "decoration", "i_1"):
        for k in range(15):
            flipped = k == 0 and flag in flips
            add(f"c-{flag}-{k}", "control", flag, True, not flipped)
    # second flip inside the same group when requested twice
    for flag in {f for f in flips if flips.count(f) == 2}:
        cand[f"c-{flag}-1"]["render_identical"] = False
    for cls, n in {"decoration": 12, "dropped_prime": 12, "greek_latin": 12, "prime_l": 8, "i_1": 5}.items():
        nfix = fixes.get(cls, 0)
        for k in range(n):
            add(f"t-{cls}-{k}", cls, "plain", False, k < nfix)
    for k in range(target_flips):
        add(f"tr-{k}", "prime_l", "plain", True, False)
    return rows, stock, cand


FIX_OK = {"decoration": 2, "dropped_prime": 1, "greek_latin": 1}


def test_regression_pass_two_flips_in_different_groups():
    rows, s, c = _reg(FIX_OK, control_flips=["plain", "prime"])
    out = rm.regression_gate(rows, s, c, G["regression"])
    assert out["pass"] and out["control"]["r2w"] == 2


def test_regression_fail_two_flips_same_group():
    rows, s, c = _reg(FIX_OK, control_flips=["plain", "plain"])
    out = rm.regression_gate(rows, s, c, G["regression"])
    assert not out["pass"] and out["control"]["per_flag"]["plain"]["r2w"] == 2


def test_regression_fail_four_control_flips():
    rows, s, c = _reg(FIX_OK, control_flips=["plain", "prime", "greek", "i_1"])
    assert not rm.regression_gate(rows, s, c, G["regression"])["pass"]


def test_regression_fail_target_right_to_wrong():
    rows, s, c = _reg(FIX_OK, target_flips=1)
    out = rm.regression_gate(rows, s, c, G["regression"])
    assert not out["pass"] and out["target"]["r2w"] == 1


def test_regression_fail_class_without_fix():
    rows, s, c = _reg({"decoration": 2, "dropped_prime": 1})
    out = rm.regression_gate(rows, s, c, G["regression"])
    assert not out["pass"] and any("greek_latin" in r for r in out["reasons"])


def test_regression_small_classes_need_no_fix_and_none_excluded():
    rows, s, c = _reg(FIX_OK)
    s["t-decoration-5"]["render_identical"] = None
    out = rm.regression_gate(rows, s, c, G["regression"])
    assert out["pass"]
    assert out["target"]["excluded_none"] == 1
    assert out["target"]["per_class"]["prime_l"]["fixed"] == 0


# ------------------------------------------------------------------ export gate


def _arms(n=100, ol_gain=5, hf_gain=5, mismatch=0, unmeasured=0, same_raw=False):
    ids = [f"d{k}" for k in range(n)]
    hf_base, hf_tuned, hf_merged, ol_stock, ol_cand = {}, {}, {}, {}, {}
    for k, i in enumerate(ids):
        # base is wrong on the first max(gain) crops; tuned fixes hf_gain of them
        hb = k >= max(ol_gain, hf_gain)
        ht = hb or k < hf_gain
        os_ = hb
        oc = os_ or k < ol_gain
        hf_base[i] = {"render_identical": hb, "raw_text": "a", "n_prompt_tokens": 100, "finish": "stop"}
        hf_tuned[i] = {"render_identical": ht, "raw_text": "b", "n_prompt_tokens": 100, "finish": "stop"}
        hf_merged[i] = {"render_identical": ht, "raw_text": "b", "n_prompt_tokens": 100, "finish": "stop"}
        pe = 110 + (3 if k >= n - mismatch else 0)
        ol_stock[i] = {"render_identical": os_, "raw_text": "a", "prompt_eval_count": pe, "done_reason": "stop"}
        ol_cand[i] = {
            "render_identical": oc,
            "raw_text": "a" if same_raw else "c",
            "prompt_eval_count": None if k >= n - unmeasured else pe,
            "done_reason": "stop",
        }
    return ids, hf_base, hf_tuned, hf_merged, ol_stock, ol_cand


def _export(merge_pass=True, **kw):
    ids, hb, ht, hm, os_, oc = _arms(**kw)
    return rm.export_gate({"pass": merge_pass}, ids, hb, ht, hm, os_, oc, G["export"])


def test_export_pass_equal_gains():
    out = _export(ol_gain=5, hf_gain=5)
    assert out["pass"], out["reasons"]
    assert out["modal_offset"] == 10
    assert out["diagnostics"]["hf_merged_vs_tuned_raw_identical_frac"] == 1.0


def test_export_fail_merge_check():
    assert not _export(merge_pass=False)["pass"]


def test_export_measured_fraction_boundary():
    assert _export(unmeasured=10)["pass"]
    out = _export(unmeasured=11)
    assert not out["pass"] and any("measured" in r for r in out["reasons"])


def test_export_fail_when_offset_matches_below_95_percent():
    out = _export(mismatch=6)
    assert not out["pass"] and any("modal" in r for r in out["reasons"])


def test_export_transfer_gap_boundaries():
    assert not _export(ol_gain=1, hf_gain=5)["pass"]  # gap 0.04 > 0.03
    assert _export(ol_gain=3, hf_gain=5)["pass"]  # gap 0.02


def test_export_fail_when_candidate_identical_to_stock():
    out = _export(same_raw=True)
    assert not out["pass"] and out["n_changed_outputs"] == 0


# ------------------------------------------------------------------ dev gate


def _dev(stock, cand):
    s = {f"i{k}": {"render_identical": v, "paper_id": f"p{k % 5}"} for k, v in enumerate(stock)}
    c = {f"i{k}": {"render_identical": v, "paper_id": f"p{k % 5}"} for k, v in enumerate(cand)}
    return s, c


def test_dev_gate_equal_passes_lower_fails():
    s, c = _dev([True, False, True, False], [True, False, True, False])
    assert rm.dev_gate(s, c, G["dev"])["pass"]
    s, c = _dev([True, True, True, False], [True, True, False, False])
    assert not rm.dev_gate(s, c, G["dev"])["pass"]


def test_dev_gate_excludes_none_arms():
    s, c = _dev([True, False, None, True], [True, True, True, None])
    out = rm.dev_gate(s, c, G["dev"])
    assert out["excluded_none"] == 2 and out["n_common"] == 2 and out["pass"]


# ------------------------------------------------------------------ register


def _args(tmp_path):
    return SimpleNamespace(
        run_id="run1", run_dir=str(tmp_path / "run"), gates=str(tmp_path / "gates.json"), export=str(tmp_path / "export.json"),
        trainset_dir=str(tmp_path), sweep=str(tmp_path / "sweep.json"), registry=str(tmp_path / "REGISTRY.json"),
        merge_check=None, taxonomy=str(tmp_path / "tax.json"), mining_manifest=str(tmp_path / "mm.json"), host="h",
    )


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        assert kw.get("shell") is not True
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_register_fail_removes_candidate_and_leaves_registry(tmp_path, monkeypatch):
    a = _args(tmp_path)
    (tmp_path / "gates.json").write_text(json.dumps({"pass": False, "regression": {"pass": False, "reasons": ["x"]}}))
    (tmp_path / "export.json").write_text("{}")
    rec = _Recorder()
    monkeypatch.setattr(rm.subprocess, "run", rec)
    assert rm.cmd_register(a) == rm.EXIT_GATE
    assert rec.calls == [["ollama", "rm", "mathocr-cand:run1"]]
    reg = json.loads((tmp_path / "registration.json").read_text())
    assert reg["outcome"] == "rejected" and reg["failing"]["regression"] == ["x"]
    assert not (tmp_path / "REGISTRY.json").exists()


def test_register_pass_copies_tag_and_appends_once(tmp_path, monkeypatch):
    a = _args(tmp_path)
    (tmp_path / "gates.json").write_text(json.dumps({"pass": True}))
    (tmp_path / "export.json").write_text(json.dumps({"candidate_tag": "mathocr-cand:run1"}))
    rec = _Recorder()
    monkeypatch.setattr(rm.subprocess, "run", rec)
    monkeypatch.setattr(rm, "_model_digest", lambda host, model: "digest123")
    monkeypatch.setattr(rm, "build_entry", lambda *args, **kw: {"tag": "mathocr-lora-q8:run1"})
    monkeypatch.setattr(rm.registry, "validate", lambda e: [])
    monkeypatch.setattr(rm.registry, "verify_hashes", lambda e, root: [])
    appended = []
    monkeypatch.setattr(rm.registry, "append_entry", lambda p, e: appended.append((p, e)))
    assert rm.cmd_register(a) == rm.EXIT_OK
    assert ["ollama", "cp", "mathocr-cand:run1", "mathocr-lora-q8:run1"] in rec.calls
    assert len(appended) == 1 and appended[0][1]["tag"] == "mathocr-lora-q8:run1"


def test_bad_run_id_exits_2(capsys):
    assert rm.main(["register", "--run-id", "a;b", "--run-dir", "x", "--gates", "g", "--export", "e", "--trainset-dir", "t", "--sweep", "s"]) == 2


def test_thresholds_pre_registered():
    assert G["regression"]["max_control_r2w_total"] == 3
    assert G["regression"]["max_control_r2w_per_flag"] == 1
    assert "data/benchmark" not in Path(rm.__file__).read_text()


def test_help_runs():
    with pytest.raises(SystemExit) as e:
        rm.main(["--help"])
    assert e.value.code == 0
