import json

import pytest

from glyphfix_math.corpus import select as S


def _rec(paper, n, flags, split="train"):
    return {"id": f"{paper}-{n}", "arxiv_id": paper, "split": split, "flags": list(flags)}


def _pool(n_papers=6, per=40, split="train"):
    pool = {}
    for p in range(n_papers):
        pid = f"2401.{p:05d}"
        recs = []
        for i in range(per):
            flags = ["prime"] if i % 10 == 0 else (["greek"] if i % 7 == 0 else ["plain"])
            recs.append(_rec(pid, i, flags, split))
        pool[pid] = recs
    return pool


def test_paper_order_deterministic_and_input_order_independent():
    ids = [f"2401.{i:05d}" for i in range(20)]
    a = S.paper_order(ids, 0)
    assert a == S.paper_order(list(reversed(ids)), 0)
    assert a != S.paper_order(ids, 1)
    assert sorted(a) == sorted(ids)


def test_cap_and_cap_rows():
    pool = _pool(3, 50)
    order = S.paper_order(list(pool), 0)
    sel = S.select_side(pool, order, split="train", quota=3, floors={}, cap=30, seed=0)
    for p in pool:
        assert sum(1 for r in sel.selected if r["arxiv_id"] == p) == 30
    cap_rows = [d for d in sel.dropped if d["reason"] == "cap"]
    assert len(cap_rows) == 3 * 20
    assert all(set(d) == {"arxiv_id", "reason", "id"} for d in cap_rows)
    assert {d["id"] for d in cap_rows}.isdisjoint({r["id"] for r in sel.selected})


def test_quota_skips_empty_papers_and_records_quota_rows():
    pool = _pool(5, 5)
    pool["2401.99999"] = []
    pool["2402.00001"] = [_rec("2402.00001", 0, ["plain"], split="dev")]
    order = S.paper_order(list(pool), 0)
    sel = S.select_side(pool, order, split="train", quota=3, floors={}, cap=30, seed=0)
    assert len(sel.papers) == 3
    assert "2401.99999" not in sel.papers and "2402.00001" not in sel.papers
    quota_rows = [d for d in sel.dropped if d["reason"] == "quota"]
    assert len(quota_rows) == 2
    assert sel.stats["quota_shortfall"] == 0
    short = S.select_side(pool, order, split="train", quota=10, floors={}, cap=30, seed=0)
    assert short.stats["quota_shortfall"] == 5


def test_floors_met_and_short():
    pool = _pool(6, 40)
    order = S.paper_order(list(pool), 0)
    sel = S.select_side(pool, order, split="train", quota=6, floors={"prime": 12, "greek": 10}, cap=5, seed=0)
    rep = S.floor_report(sel.selected, {"prime": 12, "greek": 10}, 2)
    assert not rep["prime"]["short"] and not rep["greek"]["short"]
    assert rep["prime"]["have"] >= 12 and rep["greek"]["have"] >= 10
    big = S.select_side(pool, order, split="train", quota=6, floors={"prime": 10000}, cap=30, seed=0)
    rep = S.floor_report(big.selected, {"prime": 10000}, 2)
    assert rep["prime"]["short"] is True
    assert rep["prime"]["need"] == 10000 and rep["prime"]["have"] < 10000 and rep["prime"]["papers"] == 6


def test_plain_first_fill():
    pool = {"2401.00001": [_rec("2401.00001", i, ["prime"]) for i in range(10)] + [_rec("2401.00001", 100 + i, ["plain"]) for i in range(5)]}
    sel = S.select_side(pool, ["2401.00001"], split="train", quota=1, floors={}, cap=5, seed=0)
    assert all(r["flags"] == ["plain"] for r in sel.selected)
    assert len(sel.selected) == 5


def test_determinism_byte_identical():
    pool = _pool(4, 45)
    order = S.paper_order(list(pool), 3)
    a = S.select_side(pool, order, split="train", quota=4, floors={"prime": 8}, cap=30, seed=3)
    b = S.select_side(pool, order, split="train", quota=4, floors={"prime": 8}, cap=30, seed=3)
    assert json.dumps([a.selected, a.dropped, a.stats]) == json.dumps([b.selected, b.dropped, b.stats])


def test_config_roundtrip(tmp_path):
    cfg = S.SelectionConfig(seed=2, cap=10)
    p = tmp_path / "c.json"
    cfg.to_json(p)
    assert S.SelectionConfig.from_json(p) == cfg
    assert S.SelectionConfig().cap == 30


def test_config_unknown_key_raises_but_annotations_allowed(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 1, "quota": {"train": 5}}))
    with pytest.raises(ValueError, match="quota"):
        S.SelectionConfig.from_json(p)
    p.write_text(json.dumps({"seed": 1, "evidence": {"note": "x"}, "fetch_counts": {"train": 3}, "usable_rate": 0.8}))
    assert S.SelectionConfig.from_json(p).seed == 1


def test_real_selection_config_still_loads():
    from pathlib import Path

    real = Path(__file__).resolve().parent.parent / "data" / "corpus" / "SELECTION_CONFIG.json"
    if real.exists():
        S.SelectionConfig.from_json(real)
