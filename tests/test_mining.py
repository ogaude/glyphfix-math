import pytest

from lightonocr_math import mining as m


def _row(i, paper, gt, pred, exact=False, ri=False, classes=None):
    return {"id": i, "paper_id": paper, "gt_latex": gt, "pred_latex": pred, "exact": exact,
            "render_identical": ri, "classes": classes if classes is not None else ["other"]}  # fmt: skip


def test_error_and_clean_predicates():
    assert m.is_error_row({"render_identical": False})
    assert not m.is_error_row({"render_identical": None})
    assert not m.is_error_row({"render_identical": True})
    assert m.is_clean_row({"render_identical": True})
    assert not m.is_clean_row({"render_identical": None})
    assert not m.is_clean_row({"render_identical": False})


def test_spans_basic():
    assert m.spans(["a", "'", "=", "b"], ["a", "l", "=", "b"]) == [(("'",), ("l",))]


def test_spans_merge_replace_and_insert():
    got = m.spans(["a", "b", "c"], ["a", "x", "y", "z", "c"])
    assert got == [(("b",), ("x", "y", "z"))]


def test_spans_insertion_and_deletion():
    assert m.spans(["a", "b"], ["a", "'", "b"]) == [((), ("'",))]
    assert m.spans(["a", "'", "b"], ["a", "b"]) == [(("'",), ())]


def test_spacing_only():
    assert m.spacing_only(_row("a", "p", "a\\,b", "ab", exact=True, ri=False, classes=[]))
    assert not m.spacing_only(_row("a", "p", "a+b", "a+b", exact=True, ri=False, classes=[]))
    assert not m.spacing_only(_row("a", "p", "a\\,b", "ab", exact=False, ri=False))


def test_pair_family():
    assert m.pair_family((("\\left",), ())) == "style"
    assert m.pair_family((("\\upsilon",), ("v",))) == "glyph"
    assert m.pair_family((("\\int",), ("\\oint",))) == "other"


def _cluster_rows(n_papers, pair=("\\int", "\\oint"), prefix="q"):
    return [_row(f"{prefix}{i}", f"2401.{i:05d}", f"a {pair[0]} b", f"a {pair[1]} b") for i in range(n_papers)]


def test_candidate_min_papers_drops_small_pairs():
    rows = _cluster_rows(4)
    assert m.candidate_clusters(rows, min_papers=5) == []
    got = m.candidate_clusters(rows, min_papers=4)
    assert [c["id"] for c in got] == ["c01"]
    assert got[0]["gt_tokens"] == ["\\int"] and got[0]["pred_tokens"] == ["\\oint"]
    assert got[0]["n_papers"] == 4


def test_candidates_sorted_and_ids():
    rows = _cluster_rows(5, ("\\int", "\\oint"), "a") + _cluster_rows(7, ("\\sum", "\\prod"), "b")
    got = m.candidate_clusters(rows, min_papers=5)
    assert [c["id"] for c in got] == ["c01", "c02"]
    assert got[0]["gt_tokens"] == ["\\sum"]
    assert got[0]["n_records"] == 7


def test_candidates_ignore_glyph_class_rows():
    rows = [_row(f"g{i}", f"2401.{i:05d}", "a \\int b", "a \\oint b", classes=["decoration", "other"]) for i in range(8)]
    assert m.candidate_clusters(rows, min_papers=5) == []


def test_norm_equal_candidate():
    rows = [_row(f"n{i}", f"2402.{i:05d}", "a\\,b", "ab", exact=True, ri=False, classes=[]) for i in range(5)]
    got = m.candidate_clusters(rows, min_papers=5)
    assert got[0]["id"] == "c00" and got[0]["family"] == "layout"
    assert got[0]["n_papers"] == 5


def test_assign_decoration_over_other():
    r = _row("a", "p", "x", "y", classes=["decoration", "other"])
    assert m.assign_mining_class(r, {}, {}) == "decoration"


def test_assign_rarest_glyph_class():
    r = _row("a", "p", "x", "y", classes=["prime_l", "decoration"])
    assert m.assign_mining_class(r, {}, {"prime_l": 5, "decoration": 50}) == "prime_l"
    assert m.assign_mining_class(r, {}, {"prime_l": 50, "decoration": 5}) == "decoration"
    assert m.assign_mining_class(r, {}, {}) == "prime_l"  # tie: CLASSES order


def test_assign_accepted_cluster_and_other():
    acc = {"left_right": [[["\\left"], []]]}
    r = _row("a", "p", "\\left( x \\right)", "( x \\right)", classes=["other"])
    assert m.assign_mining_class(r, acc, {}) == "left_right"
    r2 = _row("b", "p", "a \\int b", "a \\oint b", classes=["other"])
    assert m.assign_mining_class(r2, acc, {}) == "other"
    # a glyph class row never enters a cluster
    r3 = _row("c", "p", "\\left( x \\right)", "( x \\right)", classes=["decoration"])
    assert m.assign_mining_class(r3, acc, {"decoration": 1}) == "decoration"


def test_assign_norm_equal_and_clean():
    ne = _row("a", "p", "a\\,b", "ab", exact=True, ri=False, classes=[])
    assert m.assign_mining_class(ne, {"spacing": [m.NORM_EQUAL]}, {}) == "spacing"
    assert m.assign_mining_class(ne, {}, {}) == "other"
    assert m.assign_mining_class(_row("c", "p", "x", "x", exact=True, ri=True, classes=[]), {}, {}) is None


def test_build_labels():
    rows = [
        _row("b", "p2", "x", "x", exact=True, ri=True, classes=[]),
        _row("a", "p1", "a \\int b", "a \\oint b"),
        _row("c", "p3", "x", "y", ri=None),
    ]
    labels = m.build_labels(rows, {"a": {"flags": ["prime"]}}, {}, {"b"})
    assert [x["id"] for x in labels] == ["a", "b", "c"]
    keys = {"id", "paper_id", "render_identical", "exact", "classes", "error", "clean", "mining_class", "tags", "flags", "in_regression"}
    assert all(set(x) == keys for x in labels)
    a, b, c = labels
    assert a["error"] and a["mining_class"] == "other" and a["flags"] == ["prime"] and not a["in_regression"]
    assert b["clean"] and b["mining_class"] is None and b["in_regression"]
    assert not c["error"] and not c["clean"] and c["mining_class"] is None


def _cands():
    return [
        {"id": "c00", "family": "layout", "gt_tokens": ["NORM_EQUAL"], "pred_tokens": ["NORM_EQUAL"], "n_records": 5, "n_papers": 5, "example_ids": []},
        {"id": "c01", "family": "style", "gt_tokens": ["\\left"], "pred_tokens": [], "n_records": 9, "n_papers": 6, "example_ids": []},
        {"id": "c02", "family": "other", "gt_tokens": ["\\int"], "pred_tokens": ["\\oint"], "n_records": 7, "n_papers": 5, "example_ids": []},
    ]


def _doc(accepted, **kw):
    return m.taxonomy_doc(_cands(), accepted, reviewer=kw.get("reviewer", "me"), reviewed_at="t", counts={}, candidates_sha256="x", scored_sha256="y")


def test_taxonomy_doc_ok():
    d = _doc({"left_right": ["c01"], "spacing": ["c00"]})
    assert d["reviewed"] is True and d["reviewer"] == "me"
    assert d["classes"] == [*m.GLYPH_CLASSES, "left_right", "spacing", "other"]
    assert d["rejected_candidate_ids"] == ["c02"]
    assert d["accepted_clusters"][1]["patterns"] == [m.NORM_EQUAL]
    assert d["accepted_clusters"][0]["patterns"] == [[["\\left"], []]]


@pytest.mark.parametrize(
    "accepted",
    [{"decoration": ["c01"]}, {"other": ["c01"]}, {"Bad-Name": ["c01"]}, {"x": ["c99"]}, {"x": ["c01"], "y": ["c01"]}],
)
def test_taxonomy_doc_rejects(accepted):
    with pytest.raises(ValueError):
        _doc(accepted)


def test_taxonomy_needs_reviewer():
    with pytest.raises(ValueError):
        _doc({"x": ["c01"]}, reviewer=" ")


def test_taxonomy_no_clusters():
    d = _doc({})
    assert d["classes"] == [*m.GLYPH_CLASSES, "other"] and d["accepted_clusters"] == []
