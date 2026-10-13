import time

import pytest

from lightonocr_math.stats import (
    cluster_bootstrap,
    flip_table,
    mcnemar_exact,
    paired_bootstrap_delta,
    per_class_rates,
)


def test_all_correct_has_degenerate_ci():
    papers = [f"p{i}" for i in range(10) for _ in range(10)]
    r = cluster_bootstrap(papers, [1] * 100, [1] * 100, B=500)
    assert r["rate"] == 1.0 and r["ci_low"] == 1.0 and r["ci_high"] == 1.0
    assert r["n_papers"] == 10


def test_seed_determinism():
    papers = [f"p{i % 7}" for i in range(70)]
    num = [i % 3 == 0 for i in range(70)]
    den = [1] * 70
    a = cluster_bootstrap(papers, num, den, B=300, seed=5)
    b = cluster_bootstrap(papers, num, den, B=300, seed=5)
    c = cluster_bootstrap(papers, num, den, B=300, seed=6)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"]) or a["rate"] == c["rate"]


def test_resamples_by_paper_not_record():
    papers = ["big"] * 100 + ["small"]
    num = [0.0] * 100 + [1.0]
    r = cluster_bootstrap(papers, num, [1.0] * 101, B=2000)
    assert r["rate"] == pytest.approx(1 / 101)
    assert r["ci_high"] - r["ci_low"] > 0.3


def test_zero_denominator():
    r = cluster_bootstrap(["a", "b"], [0, 0], [0, 0], B=100)
    assert r["rate"] is None and r["ci_low"] is None and r["ci_high"] is None


def test_zero_den_replicates_dropped_not_zero():
    # paper "z" has no denominator; replicates that draw only "z" must be dropped
    r = cluster_bootstrap(["a", "z"], [1, 0], [1, 0], B=500)
    assert r["rate"] == 1.0 and r["ci_low"] == 1.0 and r["ci_high"] == 1.0


def test_per_class_rates_denominators():
    rows = [
        {"paper_id": "a", "present": {"prime_l", "dropped_prime", "other"}, "classes": {"dropped_prime"}},
        {"paper_id": "b", "present": {"other"}, "classes": {"decoration"}},  # class not in present still counts in denom
        {"paper_id": "c", "present": {"prime_l", "dropped_prime", "other"}, "classes": set()},
    ]
    out = per_class_rates(rows, B=200)
    assert out["dropped_prime"]["errors"] == 1 and out["dropped_prime"]["denom"] == 2
    assert out["decoration"]["errors"] == 1 and out["decoration"]["denom"] == 1
    assert out["decoration"]["rate"] == 1.0
    for v in out.values():
        if v["rate"] is not None:
            assert v["rate"] <= 1.0


def test_flip_table():
    a = {"a": True, "b": True, "c": False, "d": False}
    b = {"a": True, "b": False, "c": True, "d": False}
    t = flip_table(a, b)
    assert t["both_right"] == 1 and t["both_wrong"] == 1
    assert t["right_to_wrong"] == 1 and t["right_to_wrong_ids"] == ["b"]
    assert t["wrong_to_right"] == 1 and t["wrong_to_right_ids"] == ["c"]
    with pytest.raises(ValueError):
        flip_table({"a": True}, {"b": True})


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


def test_paired_delta_identical_runs():
    papers = ["a", "b", "c", "a"]
    x = [1, 0, 1, 0]
    r = paired_bootstrap_delta(papers, x, x, [1] * 4, B=200)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_paired_delta_improvement():
    papers = [f"p{i % 20}" for i in range(200)]
    a = [0.0] * 200
    b = [1.0] * 200
    r = paired_bootstrap_delta(papers, a, b, [1.0] * 200, B=200)
    assert r["delta"] == 1.0 and r["ci_low"] == 1.0


def test_bootstrap_speed():
    papers = [f"p{i % 200}" for i in range(2000)]
    num = [float(i % 5 == 0) for i in range(2000)]
    t = time.monotonic()
    cluster_bootstrap(papers, num, [1.0] * 2000, B=10000)
    assert time.monotonic() - t < 2.0
