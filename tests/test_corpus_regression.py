import json

import pytest

from lightonocr_math import splits
from scripts import corpus_regression as cr

BENCH_ID = "2604.22670"


def _score_row(i, paper, classes, exact=False, ri=False):
    return {"id": f"r{i:03d}", "paper_id": paper, "gt_latex": "x", "pred_latex": "y", "exact": exact,
            "render_identical": ri, "classes": classes}  # fmt: skip


def _pool_for(rows, flag="prime"):
    return [{"id": r["id"], "paper_id": r["paper_id"], "gt_latex": "x", "image": "i.png", "flags": [flag], "pool_flag": flag} for r in rows]


def test_class_assignment_uses_class_order():
    assert cr.assign_class({"classes": ["decoration", "dropped_prime", "other"]}) == "dropped_prime"
    assert cr.assign_class({"classes": ["other"]}) is None
    assert cr.assign_class({"classes": []}) is None


def test_rarest_class_first_for_i1_with_decoration():
    rows = [_score_row(i, f"2301.{i:05d}", ["decoration"]) for i in range(10)]
    rows += [_score_row(10 + i, f"2301.{10 + i:05d}", ["i_1", "decoration"]) for i in range(3)]
    rarity = cr.class_rarity(rows)
    assert rarity["i_1"] == 3 and rarity["decoration"] == 13
    assert cr.assign_class(rows[-1], rarity) == "i_1"
    assert cr.assign_class(rows[0], rarity) == "decoration"
    _, meta = cr.select_rows(rows, _pool_for(rows), per_class=30, controls=0, max_per_paper=3, seed=0)
    assert meta["counts"]["i_1"] == 3 and meta["counts"]["decoration"] == 10


def test_select_is_deterministic_for_same_seed():
    rows = [_score_row(i, f"2301.{i % 7:05d}", ["i_1", "decoration"] if i % 3 else ["decoration"]) for i in range(40)]
    a = cr.select_rows(rows, _pool_for(rows), per_class=10, controls=0, max_per_paper=3, seed=0)
    b = cr.select_rows(list(reversed(rows)), _pool_for(rows), per_class=10, controls=0, max_per_paper=3, seed=0)
    assert a == b


def test_failure_and_clean_predicates():
    assert cr.is_failure({"exact": False, "render_identical": None})
    assert cr.is_failure({"exact": False, "render_identical": False})
    assert not cr.is_failure({"exact": False, "render_identical": True})
    assert not cr.is_failure({"exact": True, "render_identical": True})
    assert cr.is_clean({"exact": True, "render_identical": True})
    assert not cr.is_clean({"exact": True, "render_identical": None})


def test_per_paper_limit_per_class_and_shortfall():
    rows = [_score_row(i, "2301.00001", ["prime_l"]) for i in range(10)]
    rows += [_score_row(10 + i, f"2301.1000{i}", ["greek_latin"]) for i in range(4)]
    chosen, meta = cr.select_rows(rows, _pool_for(rows), per_class=30, controls=0, max_per_paper=3, seed=0)
    assert meta["counts"]["prime_l"] == 3
    assert meta["counts"]["greek_latin"] == 4
    assert meta["shortfalls"]["prime_l"] == 27 and meta["shortfalls"]["greek_latin"] == 26
    assert meta["shortfalls"]["decoration"] == 30
    assert all(r["regression_class"] != "control" for r in chosen)
    assert {r["scorer_version"] for r in chosen} == {cr.SCORER_VERSION}


def test_per_class_cap_and_seeded_order():
    rows = [_score_row(i, f"2301.{i:05d}", ["i_1"]) for i in range(50)]
    a, _ = cr.select_rows(rows, _pool_for(rows), per_class=30, controls=0, max_per_paper=3, seed=0)
    b, _ = cr.select_rows(rows, _pool_for(rows), per_class=30, controls=0, max_per_paper=3, seed=0)
    c, _ = cr.select_rows(rows, _pool_for(rows), per_class=30, controls=0, max_per_paper=3, seed=1)
    assert len(a) == 30 and [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_controls_clean_only_spread_and_two_per_paper():
    clean, pool = [], []
    k = 0
    for flag in ("prime", "greek", "plain"):
        for j in range(6):
            r = _score_row(k, f"2301.2000{j % 3}", [], exact=True, ri=True)
            clean.append(r)
            pool.append({**_pool_for([r], flag)[0]})
            k += 1
    dirty = _score_row(99, "2301.30000", [], exact=True, ri=False)
    clean.append(dirty)
    pool += _pool_for([dirty], "plain")
    chosen, meta = cr.select_rows(clean, pool, per_class=30, controls=75, max_per_paper=3, seed=0)
    ctrl = [r for r in chosen if r["regression_class"] == "control"]
    assert len(ctrl) == meta["n_controls"] == 6  # 3 papers x 2
    assert "r099" not in {r["id"] for r in ctrl}
    assert max(sum(1 for r in ctrl if r["paper_id"] == p) for p in {r["paper_id"] for r in ctrl}) == 2
    assert {r["pool_flag"] for r in ctrl} == {"prime", "greek", "plain"}


def test_pool_rows_one_flag_each_no_duplicates_and_exclude():
    recs = [
        {"id": f"{i:016x}", "arxiv_id": f"2301.{i:05d}", "latex": "a", "image": "i.png",
         "flags": ["prime", "greek"] if i % 2 else ["plain"]}
        for i in range(40)
    ]  # fmt: skip
    rows = cr.pool_rows(recs, per_flag=5, plain=7, seed=0)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert sum(r["pool_flag"] == "prime" for r in rows) == 5
    assert sum(r["pool_flag"] == "greek" for r in rows) == 5
    assert sum(r["pool_flag"] == "plain" for r in rows) == 7
    assert rows == cr.pool_rows(recs, per_flag=5, plain=7, seed=0)
    more = cr.pool_rows(recs, per_flag=5, plain=7, seed=0, exclude=set(ids))
    assert not set(ids) & {r["id"] for r in more}


def test_pool_refuses_benchmark_paper(tmp_path):
    rec = {"id": "a" * 16, "arxiv_id": BENCH_ID, "latex": "a", "image": "i.png", "flags": ["prime"]}
    train = tmp_path / "t.jsonl"
    dev = tmp_path / "d.jsonl"
    train.write_text(json.dumps(rec) + "\n")
    dev.write_text("")
    out = tmp_path / "pool.jsonl"
    rc = cr.main(["pool", "--train", str(train), "--dev", str(dev), "--out", str(out)])
    assert rc == 2 and not out.exists()


def test_select_refuses_benchmark_paper(tmp_path):
    row = _score_row(1, BENCH_ID, ["prime_l"])
    scored, pool, man = tmp_path / "s.jsonl", tmp_path / "p.jsonl", tmp_path / "m.json"
    scored.write_text(json.dumps(row) + "\n")
    pool.write_text(json.dumps(_pool_for([row])[0]) + "\n")
    man.write_text("{}")
    out = tmp_path / "reg.jsonl"
    argv = ["select", "--scored", str(scored), "--pool", str(pool), "--stock-manifest", str(man),
            "--out", str(out), "--manifest", str(tmp_path / "rm.json")]  # fmt: skip
    assert cr.main(argv) == 2 and not out.exists()


def test_assert_disjoint_is_the_gate():
    with pytest.raises(splits.LeakageError):
        cr._assert_train_dev([BENCH_ID], "t")
