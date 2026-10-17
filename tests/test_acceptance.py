import copy
import json
from pathlib import Path

import pytest

from lightonocr_math import acceptance as a

ROOT = Path(__file__).resolve().parent.parent


def _arms(stock_right: int, r2w: int, w2r: int, *, n_papers: int = 10, per: int = 10, tuned_classes=None):
    """Synthetic paired arms: n_papers x per records, flips spread round-robin across papers."""
    n = n_papers * per
    ids = [f"p{k // per}-{k % per}" for k in range(n)]
    paper = {i: i.split("-")[0] for i in ids}
    order = sorted(range(n), key=lambda k: (k % per, k // per))
    right = set(range(stock_right))
    r2w_idx = [k for k in order if k in right][:r2w]
    w2r_idx = [k for k in order if k not in right][:w2r]
    tuned_right = (right - set(r2w_idx)) | set(w2r_idx)
    stock, tuned = {}, {}
    for k, i in enumerate(ids):
        for rows, ok in ((stock, k in right), (tuned, k in tuned_right)):
            rows[i] = {
                "id": i,
                "paper_id": paper[i],
                "gt_latex": f"gt{k}",
                "pred_latex": f"pred{k}",
                "render_identical": ok,
                "classes": [] if ok else ["other"],
                "present": [],
            }
    if tuned_classes is not None:
        for k in r2w_idx:
            tuned[ids[k]]["classes"] = list(tuned_classes)
    return stock, tuned, [ids[k] for k in r2w_idx], [ids[k] for k in w2r_idx]


# ----------------------------------------------------------------------------- constants and pin


def test_thresholds_pre_registered():
    c = a.ACCEPTANCE
    assert c["headline"]["bootstrap_B"] == 10000
    assert c["headline"]["bootstrap_seed"] == 0
    assert c["headline"]["ci"] == 0.95
    assert c["headline"]["pass_if_ci_low_gt"] == 0.0
    assert c["regression"]["max_r2w_over_w2r"] == 0.2
    assert c["regression"]["max_r2w_over_stock_right"] == 0.06
    assert c["transfer"]["pass_if_ci_low_gt"] == -0.03
    assert c["transfer"]["pass_if_delta_gt"] == 0.0
    assert c["per_class"]["min_stock_errors"] == 30
    assert c["spacing"]["commands"] == ["\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad"]
    assert c["decides"] == ["render_identical"]
    assert "render_identical" not in c["secondary"]


def test_digest_changes_with_any_value():
    base = a.constants_digest()
    assert len(base) == 64
    other = copy.deepcopy(a.ACCEPTANCE)
    other["regression"]["max_r2w_over_stock_right"] = 0.07
    assert a.constants_digest(other) != base


def test_pin_matches():
    if a.ACCEPTANCE_SHA256 is None:
        pytest.skip("pin is set after the dev dry run")
    assert a.ACCEPTANCE_SHA256 == a.constants_digest()


def test_assert_pinned_refuses_none(monkeypatch):
    monkeypatch.setattr(a, "ACCEPTANCE_SHA256", None)
    with pytest.raises(a.AcceptanceError):
        a.assert_pinned()


def test_assert_pinned_refuses_mismatch_and_accepts_match(monkeypatch):
    monkeypatch.setattr(a, "ACCEPTANCE_SHA256", "0" * 64)
    with pytest.raises(a.AcceptanceError):
        a.assert_pinned()
    monkeypatch.setattr(a, "ACCEPTANCE_SHA256", a.constants_digest())
    assert a.assert_pinned() == a.constants_digest()


def test_benchmark_results_bound_to_pin():
    p = ROOT / ".planning/phases/05-acceptance-evaluation/evidence/results.json"
    if not p.is_file():
        pytest.skip("no benchmark results yet")
    res = json.loads(p.read_text())
    if res.get("mode") != "benchmark":
        pytest.skip("results are not a benchmark run")
    assert res["acceptance_sha256"] == a.ACCEPTANCE_SHA256 == a.constants_digest()


def test_module_is_pure():
    src = Path(a.__file__).read_text()
    for banned in ("data/benchmark", "open(", "read_text", "subprocess", "urllib"):
        assert banned not in src, banned


# ----------------------------------------------------------------------------- headline


def test_headline_passes_on_clear_win():
    stock, tuned, r2w, w2r = _arms(40, 2, 32)
    v = a.headline_verdict(stock, tuned)
    assert v["n"] == 100 and v["n_papers"] == 10
    assert v["stock_rate"] == pytest.approx(0.40)
    assert v["tuned_rate"] == pytest.approx(0.70)
    assert v["delta"] == pytest.approx(0.30)
    assert v["significant"] is True
    assert v["regression"]["ratio_ok"] and v["regression"]["share_ok"]
    assert v["regression_ok"] is True
    assert v["pass"] is True
    assert v["r2w_ids"] == sorted(r2w) and v["w2r_ids"] == sorted(w2r)
    assert v["flip"]["right_to_wrong"] == 2 and "right_to_wrong_ids" not in v["flip"]


def test_headline_fails_when_regressions_too_many():
    stock, tuned, _, _ = _arms(40, 8, 30)
    v = a.headline_verdict(stock, tuned)
    assert v["significant"] is True
    assert v["regression"]["ratio_ok"] is False
    assert v["pass"] is False


def test_ratio_boundary_inclusive():
    stock, tuned, _, _ = _arms(70, 4, 20)
    v = a.headline_verdict(stock, tuned)
    assert v["regression"]["ratio_ok"] is True
    stock, tuned, _, _ = _arms(70, 5, 20)
    assert a.headline_verdict(stock, tuned)["regression"]["ratio_ok"] is False


def test_share_boundary_inclusive():
    stock, tuned, _, _ = _arms(50, 3, 25)
    v = a.headline_verdict(stock, tuned)
    assert v["regression"]["share_ok"] is True
    stock, tuned, _, _ = _arms(50, 4, 25)
    v = a.headline_verdict(stock, tuned)
    assert v["regression"]["share_ok"] is False
    assert v["regression"]["ratio_ok"] is True
    assert v["pass"] is False


def test_identical_arms_do_not_pass():
    stock, tuned, _, _ = _arms(40, 0, 0)
    v = a.headline_verdict(stock, tuned)
    assert v["delta"] == 0
    assert v["ci_low"] <= 0
    assert v["pass"] is False


def test_none_render_excluded_and_id_mismatch_raises():
    stock, tuned, _, _ = _arms(40, 2, 32)
    stock["p0-0"]["render_identical"] = None
    tuned["p1-0"]["render_identical"] = None
    ids, ra, rb, excluded = a.paired_decided(stock, tuned)
    assert len(ids) == 98 and excluded == {"stock_none": 1, "tuned_none": 1}
    assert set(ra) == set(rb) == set(ids)
    assert a.headline_verdict(stock, tuned)["excluded"] == excluded
    del tuned["p9-9"]
    with pytest.raises(a.AcceptanceError):
        a.paired_decided(stock, tuned)


def test_regression_listing_flags_control_glyph_classes():
    stock, tuned, r2w, _ = _arms(40, 2, 32, tuned_classes=["decoration", "other"])
    first = min(r2w)
    tuned[first]["classes"] = []
    listing = a.regression_listing(stock, tuned, r2w)
    rows = {r["id"]: r for r in listing["rows"]}
    assert [r["id"] for r in listing["rows"]] == sorted(r2w)
    assert rows[first]["classes"] == ["render-only"] and rows[first]["control_glyph"] is False
    other = sorted(r2w)[1]
    assert rows[other]["control_glyph"] is True
    assert listing["by_class"]["decoration"] == 1 and listing["by_class"]["other"] == 1
    assert listing["by_class"]["render-only"] == 1
    assert listing["n_control_glyph"] == 1
    assert rows[other]["stock_pred"] == stock[other]["pred_latex"]
    assert rows[other]["tuned_pred"] == tuned[other]["pred_latex"]


# ----------------------------------------------------------------------------- manifests and results


def _manifest():
    return {
        "suite": "bench",
        "options": {"temperature": 0},
        "endpoint": "chat",
        "think": False,
        "prompt": "",
        "prompt_sha256": "p",
        "template_sha256": "t",
        "pixel_policy": "x",
        "ollama_version": "0.40.1",
        "records_sha256": "r",
        "scorer_version": "1",
        "scorer_fingerprint": "f",
        "render_config": {"dpi": 300},
        "split_rule": "s",
        "gguf_blobs": {"projector": "proj", "text": "text-a"},
        "model": "stock",
        "model_digest": "d-stock",
        "created": "t0",
        "wall_seconds": 1,
        "gpu_before": 1,
        "gpu_after": 1,
        "git_commit": "g0",
        "n_ok": 1,
        "n_error": 0,
        "n_skipped_resume": 0,
        "model_loaded_before": False,
    }


def test_comparability_ignores_per_arm_fields():
    s, t = _manifest(), _manifest()
    t.update(
        model="tuned",
        model_digest="d-tuned",
        created="t1",
        wall_seconds=9,
        gpu_before=5,
        gpu_after=6,
        git_commit="g1",
        n_ok=3,
        n_error=1,
        n_skipped_resume=2,
        model_loaded_before=True,
    )
    t["gguf_blobs"] = {"projector": "proj", "text": "text-b"}
    assert a.comparability_problems(s, t, expected_tuned_digest="d-tuned") == []


def test_comparability_reports_each_difference():
    s, t = _manifest(), _manifest()
    t["model_digest"] = "d-other"
    t["prompt_sha256"] = "q"
    t["gguf_blobs"] = {"projector": "other", "text": "x"}
    t["ollama_version"] = "0.41.0"
    problems = a.comparability_problems(s, t, expected_tuned_digest="d-tuned")
    text = "\n".join(problems)
    for needle in ("prompt_sha256", "gguf_blobs.projector", "ollama_version", "model_digest"):
        assert needle in text


def test_results_problems_version_keys_and_render_image():
    from lightonocr_math.report import VERSION_KEYS

    s = {k: "v" for k in VERSION_KEYS}
    s["render"] = {"image": "img"}
    t = copy.deepcopy(s)
    assert a.results_problems(s, t) == []
    t["split_digest"] = "other"
    t["render"] = {"image": "img2"}
    text = "\n".join(a.results_problems(s, t))
    assert "split_digest" in text and "render image" in text
