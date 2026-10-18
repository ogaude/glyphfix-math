import json
from collections import Counter
from pathlib import Path

import pytest

from lightonocr_math import splits
from lightonocr_math import trainset as ts

BENCH_ID = "2604.22670"
CLASSES = ["prime_l", "dropped_prime", "decoration", "other"]
ROOT = Path(__file__).resolve().parents[1]


def _token_len(s):
    return len(s.split())


@pytest.fixture(scope="module")
def real_ids():
    train, dev = [], []
    if not (ROOT / "data" / "corpus" / "train" / "records.jsonl").is_file():
        pytest.skip("reserved train and dev rows are not part of the public repository")
    for name, out in (("train", train), ("dev", dev)):
        seen = []
        with open(ROOT / "data" / "corpus" / name / "records.jsonl", encoding="utf-8") as fh:
            for line in fh:
                a = json.loads(line)["arxiv_id"]
                if a not in seen:
                    seen.append(a)
                if len(seen) >= 60:
                    break
        out.extend(seen)
    return train, dev


def _rec(i, paper, latex, flags=("plain",), lic="CC-BY-4.0"):
    return {
        "id": f"r{i:05d}", "arxiv_id": paper, "image": f"img/{i}.png", "latex": latex, "flags": list(flags),
        "width": 100, "height": 40, "license_spdx": lic, "license_uri": "http://x", "paper_version": "v1",
    }  # fmt: skip


def _label(rec, kind, cls=None, ri=True):
    return {
        "id": rec["id"], "paper_id": rec["arxiv_id"], "render_identical": ri, "exact": kind == "clean",
        "classes": [cls] if cls else [], "error": kind == "error", "clean": kind == "clean",
        "mining_class": cls, "tags": ["t"], "flags": rec["flags"], "in_regression": False,
    }  # fmt: skip


def _world(papers, n_err=40, n_clean=120):
    recs, labels = [], []
    i = 0
    for k in range(n_err):
        r = _rec(i, papers[k % len(papers)], f"e_{i} + y", flags=("prime",) if k % 2 else ("greek",))
        recs.append(r)
        labels.append(_label(r, "error", CLASSES[k % 3], ri=False))
        i += 1
    glyphs = ["x'", "I", "l", "\\ell", "1", "|", "v", "\\nu", "\\upsilon"]
    for k in range(n_clean):
        latex = f"c_{i} + {glyphs[k % 9]}" if k % 3 == 0 else f"c_{i} + z"
        flags = [("plain", "prime", "greek", "decoration", "i_1")[k % 5]]
        r = _rec(i, papers[(k * 7) % len(papers)], latex, flags=flags)
        recs.append(r)
        labels.append(_label(r, "clean"))
        i += 1
    return recs, labels


TAX = {"reviewed": True, "classes": CLASSES}


def test_target_text_is_fenced():
    assert ts.target_text("x'") == "```latex\nx'\n```"


def test_allocate_equal_share_water_filling():
    out = ts.allocate_equal_share({"a": 10, "b": 1000, "c": 1000}, budget=900)
    assert out == {"a": 10, "b": 445, "c": 445}


def test_allocate_never_exceeds_supply_and_sums():
    s = {"a": 3, "b": 5, "c": 0}
    assert ts.allocate_equal_share(s, 100) == {"a": 3, "b": 5, "c": 0}
    out = ts.allocate_equal_share({"a": 50, "b": 50, "c": 50}, 100)
    assert sum(out.values()) == 100 and max(out.values()) - min(out.values()) <= 1
    assert all(out[k] <= 50 for k in out)


def test_unreviewed_taxonomy_refused(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    with pytest.raises(ValueError, match="not reviewed"):
        ts.build(recs, labels, {"reviewed": False, "classes": CLASSES}, [], set(), token_len=_token_len, total=60)
    with pytest.raises(ValueError, match="not reviewed"):
        ts.build(recs, labels, {"classes": CLASSES}, [], set(), token_len=_token_len, total=60)


def test_benchmark_paper_raises_leakage(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    recs[0]["arxiv_id"] = BENCH_ID
    with pytest.raises(splits.LeakageError):
        ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60)


def test_dev_paper_and_dev_ids_dropped(real_ids):
    papers, dev_papers = real_ids
    recs, labels = _world(papers)
    recs[0]["arxiv_id"] = dev_papers[0]
    rows, body = ts.build(recs, labels, TAX, [], {recs[1]["id"]}, token_len=_token_len, total=60)
    ids = {r["id"] for r in rows}
    assert recs[0]["id"] not in ids and recs[1]["id"] not in ids
    assert body["dropped"]["non_train_paper"] == 1 and body["dropped"]["dev_record"] == 1


def test_regression_ids_labels_and_equal_latex_dropped(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    labels[0]["in_regression"] = True
    reg = [{"id": recs[1]["id"], "gt_latex": "zzz"}, {"id": "other", "gt_latex": recs[2]["latex"]}]
    rows, body = ts.build(recs, labels, TAX, reg, set(), token_len=_token_len, total=60)
    ids = {r["id"] for r in rows}
    assert not ({recs[0]["id"], recs[1]["id"], recs[2]["id"]} & ids)
    assert body["dropped"]["in_regression"] == 3


def test_undecided_missing_label_too_long_dropped(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    labels[0]["render_identical"] = None
    long_rec = recs[1]
    long_rec["latex"] = " ".join(["w"] * 50)
    missing = labels.pop(2)["id"]
    rows, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, max_target_tokens=20)
    ids = {r["id"] for r in rows}
    assert not ({recs[0]["id"], recs[1]["id"], missing} & ids)
    assert body["dropped"]["render_undecided"] == 1
    assert body["dropped"]["target_too_long"] == 1
    assert body["dropped"]["missing_label"] == 1


def test_default_cap_is_500_and_builder_version_2(real_ids):
    assert ts.MAX_TARGET_TOKENS == 500
    recs, labels = _world(real_ids[0])
    _, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60)
    assert body["params"]["max_target_tokens"] == 500
    assert body["builder_version"] == "2"


def test_cap_boundary_equal_kept_one_over_dropped(real_ids):
    recs, labels = _world(real_ids[0])
    cap = 20
    # fenced target counts the two fence words plus the latex words
    recs[0]["latex"] = " ".join(["w"] * (cap - 2))
    recs[1]["latex"] = " ".join(["w"] * (cap - 1))
    assert _token_len(ts.target_text(recs[0]["latex"])) == cap
    _, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, max_target_tokens=cap)
    assert body["dropped"]["target_too_long"] == 1
    assert body["target_too_long"]["total"] == 1
    assert body["target_too_long"]["by_pool"] == {"error": 1, "clean": 0, "other": 0}


def test_over_length_rows_refilled_and_broken_down(real_ids):
    recs, labels = _world(real_ids[0], n_err=60, n_clean=200)
    cap = 20
    long_err = [recs[i] for i in (0, 3, 6)]  # CLASSES[0] rows
    long_clean = [recs[60 + k] for k in (0, 3, 9)]  # control-glyph rows
    long_neutral = recs[60 + 1]  # clean row without a glyph
    for r in [*long_err, *long_clean, long_neutral]:
        r["latex"] = r["latex"] + " " + " ".join(["w"] * 40)
    long_ids = {r["id"] for r in [*long_err, *long_clean, long_neutral]}
    rows, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, max_target_tokens=cap)
    assert not (long_ids & {r["id"] for r in rows})
    assert body["counts"]["group"] == {"error": 24, "replay": 30, "control": 6}
    t = body["target_too_long"]
    assert t["max_target_tokens"] == cap and t["total"] == 7 == body["dropped"]["target_too_long"]
    assert t["by_pool"] == {"error": 3, "clean": 4, "other": 0}
    assert list(t["by_error_class"]) == CLASSES
    assert t["by_error_class"] == {"prime_l": 3, "dropped_prime": 0, "decoration": 0, "other": 0}
    assert list(t["by_control_glyph"]) == [*ts.CONTROL_GLYPHS, "none"]
    expect = Counter(ts._first_glyph(r["latex"]) or "none" for r in [*long_clean, long_neutral])
    assert t["by_control_glyph"] == {g: expect.get(g, 0) for g in [*ts.CONTROL_GLYPHS, "none"]}
    assert sum(t["by_control_glyph"].values()) == 4


def test_no_over_length_block_is_zero_filled(real_ids):
    recs, labels = _world(real_ids[0])
    _, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60)
    t = body["target_too_long"]
    assert t["total"] == 0 and "target_too_long" not in body["dropped"]
    assert t["by_pool"] == {"error": 0, "clean": 0, "other": 0}
    assert set(t["by_error_class"].values()) == {0} and list(t["by_error_class"]) == CLASSES
    assert set(t["by_control_glyph"].values()) == {0}


def test_deterministic_for_reversed_input(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    a = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=0)
    b = ts.build(list(reversed(recs)), list(reversed(labels)), TAX, [], set(), token_len=_token_len, total=60, seed=0)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=1)
    assert [r["id"] for r in a[0]] != [r["id"] for r in c[0]]


def test_group_composition_and_shortfalls(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers, n_err=40, n_clean=120)
    rows, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=0)
    g = body["counts"]["group"]
    assert g == {"error": 24, "replay": 30, "control": 6}
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert body["equal_share_target"] == 6
    # CLASSES[3] "other" has no supply, the others split the 24 error rows
    assert body["shortfalls"]["other"] == {"target": 6, "got": 0}
    assert body["counts"]["error_class"]["other"] == 0
    assert sum(body["counts"]["error_class"].values()) == 24
    assert body["allocation"]["prime_l"] + body["allocation"]["dropped_prime"] + body["allocation"]["decoration"] == 24


def test_controls_carry_glyph_and_round_robin(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    rows, _ = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=0)
    controls = [r for r in rows if r["group"] == "control"]
    assert controls and all(r["control_glyph"] in ts.CONTROL_GLYPHS for r in controls)
    from lightonocr_math.latex_tokens import tokenize

    for r in controls:
        assert r["control_glyph"] in tokenize(r["latex"])
    glyph_counts = Counter(r["control_glyph"] for r in controls)
    assert max(glyph_counts.values()) - min(glyph_counts.values()) <= 1


def test_replay_spread_across_flags(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    _, body = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=0)
    flags = body["counts"]["replay_flag"]
    assert set(flags) == {"plain", "prime", "greek", "decoration", "i_1"}
    assert max(flags.values()) - min(flags.values()) <= 2


def test_row_fields_and_unique_ids_and_class_membership(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    rows, _ = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60, seed=0)
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["target_text"] == ts.target_text(r["latex"])
        assert r["license_spdx"] == "CC-BY-4.0" and r["license_uri"] and r["paper_version"]
        assert r["group"] in ("error", "replay", "control")
        if r["group"] == "error":
            assert r["mining_class"] in CLASSES
    assert max(Counter(r["paper_id"] for r in rows).values()) <= 30


def test_bad_license_raises(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers)
    recs[5]["license_spdx"] = "CC-BY-NC-4.0"
    with pytest.raises(ValueError, match="license"):
        ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=60)


def test_per_paper_cap_raises(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers[:2], n_err=0, n_clean=40)
    with pytest.raises(ValueError, match="per-paper cap"):
        ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=40, max_per_paper=5)


def test_subset_proportional_and_deterministic(real_ids):
    papers = real_ids[0]
    recs, labels = _world(papers, n_err=90, n_clean=300)
    rows, _ = ts.build(recs, labels, TAX, [], set(), token_len=_token_len, total=200, seed=0)
    sub = ts.subset(rows, n=50, seed=0)
    assert len(sub) == 50
    assert [r["id"] for r in sub] == [r["id"] for r in ts.subset(list(reversed(rows)), n=50, seed=0)]
    assert {r["id"] for r in sub} <= {r["id"] for r in rows}
    full, part = Counter(r["group"] for r in rows), Counter(r["group"] for r in sub)
    for g in full:
        assert abs(part[g] - 50 * full[g] / len(rows)) <= 1
    fe = Counter(r["mining_class"] for r in rows if r["group"] == "error")
    pe = Counter(r["mining_class"] for r in sub if r["group"] == "error")
    ne = sum(pe.values())
    for c in fe:
        assert abs(pe[c] - ne * fe[c] / sum(fe.values())) <= 1
    assert len(ts.subset(rows, n=10_000)) == len(rows)


def test_dev_subset_only_dev_ids_and_per_paper_cap(real_ids):
    _, dev_papers = real_ids
    flags = ["plain", "prime", "greek", "decoration", "i_1"]
    recs = [_rec(i, dev_papers[i % len(dev_papers)], f"d_{i}", flags=[flags[i % 5]]) for i in range(400)]
    ids = ts.dev_subset(recs, n=50, per_paper=3, seed=0)
    assert ids == sorted(ids) and len(ids) == 50
    paper_of = {r["id"]: r["arxiv_id"] for r in recs}
    assert max(Counter(paper_of[i] for i in ids).values()) <= 3
    per_flag = Counter(flags[int(i[1:]) % 5] for i in ids)
    assert max(per_flag.values()) - min(per_flag.values()) <= 1
    assert ids == ts.dev_subset(list(reversed(recs)), n=50, per_paper=3, seed=0)


def test_dev_subset_refuses_non_dev_paper(real_ids):
    papers, _ = real_ids
    recs = [_rec(0, papers[0], "x")]
    with pytest.raises(splits.LeakageError):
        ts.dev_subset(recs, n=1)
