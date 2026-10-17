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


# ----------------------------------------------------------------------------- page transfer


def _stub_delta(monkeypatch, delta, ci_low):
    def fake(papers, x, y, den, **kw):
        return {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": 0.2,
            "rate_a": 0.5,
            "rate_b": 0.5 + delta,
            "den": float(len(x)),
            "n_papers": len(set(papers)),
        }

    monkeypatch.setattr(a.stats, "paired_bootstrap_delta", fake)


@pytest.mark.parametrize(
    ("delta", "ci_low", "holds"),
    [(0.05, -0.01, True), (0.0, -0.01, False), (0.02, -0.04, False), (0.02, -0.03, False), (0.02, -0.0299, True)],
)
def test_transfer_boundaries(monkeypatch, delta, ci_low, holds):
    stock, tuned, _, _ = _arms(40, 2, 32)
    _stub_delta(monkeypatch, delta, ci_low)
    v = a.transfer_verdict(stock, tuned, headline_pass=True)
    assert v["holds"] is holds
    assert v["blocks_crop_verdict"] is False
    if holds:
        assert v["statement"] == "page-level transfer holds"
    else:
        assert v["statement"] == a.ACCEPTANCE["transfer"]["fail_statement"]


def test_transfer_statement_when_headline_fails(monkeypatch):
    stock, tuned, _, _ = _arms(40, 2, 32)
    _stub_delta(monkeypatch, 0.0, -0.1)
    v = a.transfer_verdict(stock, tuned, headline_pass=False)
    assert v["statement"] == "page-level transfer not shown"


def test_transfer_keeps_unmatched_equations_as_failures():
    stock, tuned, _, _ = _arms(40, 0, 0)
    for r in tuned.values():
        r["render_identical"] = False
        r["pred_latex"] = ""
    v = a.transfer_verdict(stock, tuned, headline_pass=True)
    assert v["n"] == 100
    assert v["tuned_rate"] == 0.0 and v["holds"] is False


# ----------------------------------------------------------------------------- per class


def _crow(i, paper, classes=(), present=()):
    return {"id": i, "paper_id": paper, "classes": list(classes), "present": list(present), "render_identical": not classes}


def _class_arms(stock_errors: int, tuned_errors: int, cls="decoration", n=40):
    stock, tuned = {}, {}
    for k in range(n):
        i, p = f"p{k % 10}-{k}", f"p{k % 10}"
        stock[i] = _crow(i, p, [cls] if k < stock_errors else [], [cls] if k < n else [])
        tuned[i] = _crow(i, p, [cls] if k < tuned_errors else [], [])
    return stock, tuned


def test_per_class_shared_denominator_includes_tuned_only_error():
    stock = {"a-1": _crow("a-1", "a"), "b-1": _crow("b-1", "b"), "c-1": _crow("c-1", "c")}
    tuned = {"a-1": _crow("a-1", "a", ["decoration"]), "b-1": _crow("b-1", "b"), "c-1": _crow("c-1", "c")}
    rows = {r["class"]: r for r in a.per_class_table(stock, tuned)}
    d = rows["decoration"]
    assert d["denom"] == 1 and d["stock"]["den"] == 1.0 == d["tuned"]["den"] == d["paired"]["den"]
    assert d["stock_errors"] == 0 and d["tuned_errors"] == 1
    assert [r["class"] for r in a.per_class_table(stock, tuned)] == a.ACCEPTANCE["per_class"]["classes"]
    assert set(d) == {"class", "stock_errors", "tuned_errors", "denom", "stock", "tuned", "paired", "claim"}


def test_per_class_claim_boundaries():
    stock, tuned = _class_arms(29, 0)
    row = {r["class"]: r for r in a.per_class_table(stock, tuned)}["decoration"]
    assert row["claim"] == "not enough events to claim"
    stock, tuned = _class_arms(30, 0)
    row = {r["class"]: r for r in a.per_class_table(stock, tuned)}["decoration"]
    assert row["claim"] == "fewer errors"
    stock, tuned = _class_arms(30, 30)
    row = {r["class"]: r for r in a.per_class_table(stock, tuned)}["decoration"]
    assert row["claim"] == "no significant change"


def test_per_class_more_errors_claim():
    stock, tuned = _class_arms(30, 0)
    swapped = a.per_class_table(tuned, stock)
    row = {r["class"]: r for r in swapped}["decoration"]
    assert row["stock_errors"] == 0
    assert row["claim"] == "not enough events to claim"
    # enough stock errors in another direction: stock 30 errors, tuned 40 errors of 40
    stock, tuned = _class_arms(30, 40)
    row = {r["class"]: r for r in a.per_class_table(stock, tuned)}["decoration"]
    assert row["claim"] == "more errors"


def test_per_class_id_mismatch_raises():
    stock, tuned = _class_arms(1, 1)
    del tuned["p0-0"]
    with pytest.raises(a.AcceptanceError):
        a.per_class_table(stock, tuned)


# ----------------------------------------------------------------------------- spacing


def test_spacing_strip():
    assert a.spacing_strip("\\alpha\\,x") == "\\alpha x"
    assert a.spacing_strip("a\\quad b\\qquad c") == "a  b  c"
    assert a.spacing_strip("\\quadx") == "\\quadx"
    assert a.spacing_strip("a\\ b~c") == "a\\ b~c"
    for cmd in a.ACCEPTANCE["spacing"]["commands"]:
        assert cmd not in a.spacing_strip(f"x{cmd} y")


def _srow(i, paper, gt, pred, ok):
    return {"id": i, "paper_id": paper, "gt_latex": gt, "pred_latex": pred, "render_identical": ok}


def test_spacing_changed_ids_and_column_reuse_frozen_values():
    stock = {
        "p0-a": _srow("p0-a", "p0", "x", "x", True),
        "p0-b": _srow("p0-b", "p0", "a\\,b", "ab", False),
        "p1-a": _srow("p1-a", "p1", "y", "y\;", False),
        "p1-b": _srow("p1-b", "p1", "z", "z", None),
    }
    tuned = {k: dict(v) for k, v in stock.items()}
    tuned["p0-b"]["pred_latex"] = "a b"
    tuned["p1-b"]["render_identical"] = True
    assert a.spacing_changed_ids(stock, tuned) == ["p0-b", "p1-a"]
    rerendered = {"p0-b": {"stock": False, "tuned": True}, "p1-a": {"stock": True, "tuned": None}}
    col = a.spacing_column(stock, tuned, rerendered, B=200)
    assert col["decides"] is False
    assert col["n_rerendered"] == 1
    assert col["n_excluded"] == 2
    assert col["stock"]["num"] == 1 and col["stock"]["den"] == 2
    assert col["tuned"]["num"] == 2
    assert col["paired"]["delta"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- sensitivity


def test_sensitivity_summary_directions():
    r2w, w2r = ["a", "b", "c"], ["d", "e"]
    cpu_stock = {"a": True, "b": True, "c": None, "d": False, "e": True}
    cpu_tuned = {"a": False, "b": True, "c": True, "d": True, "e": True}
    s = a.sensitivity_summary(r2w, w2r, cpu_stock, cpu_tuned)
    assert s["r2w"] == {"total": 3, "done": 3, "held": 1, "reversed": 0, "both_right": 1, "both_wrong": 0, "undecided": 1}
    assert s["w2r"]["held"] == 1 and s["w2r"]["both_right"] == 1
    assert s["n_done"] == 5 and s["n_total"] == 5 and s["complete"] is True


def test_sensitivity_summary_missing_ids_are_not_run():
    s = a.sensitivity_summary(["a", "b"], ["c"], {"a": True}, {"a": True})
    assert s["r2w"]["done"] == 1 and s["r2w"]["both_right"] == 1
    assert s["w2r"]["done"] == 0
    assert s["n_done"] == 1 and s["n_total"] == 3 and s["complete"] is False
