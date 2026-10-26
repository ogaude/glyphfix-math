import subprocess

import pytest

from glyphfix_math import render

pytestmark = pytest.mark.docker


def _pairs(tmp_path, table):
    res = render.render_equivalence([g for g, _ in table], [p for _, p in table], cache=tmp_path)
    return res


def test_equivalent_forms_identical(tmp_path):
    table = [("f'", r"f^{\prime}"), ("f''", r"f^{\prime\prime}"), (r"\bm{x}", r"\boldsymbol{x}"), ("x+1", r"$$x+1$$")]
    for pair, r in zip(table, _pairs(tmp_path, table), strict=True):
        assert r.identical is True, pair


def test_target_error_classes_differ(tmp_path):
    table = [
        ("f'", "fl"), (r"\upsilon", "v"), ("I", "1"), (r"\mathbf{x}", "x"), (r"\bar{x}", "x"),
        (r"\underline{x}", "x"), (r"\ell", "l"), (r"\varepsilon", r"\epsilon"),
    ]
    for pair, r in zip(table, _pairs(tmp_path, table), strict=True):
        assert r.gt_status == "rendered" and r.pred_status == "rendered", pair
        assert r.identical is False, pair


def test_compile_failures_score_false(tmp_path):
    table = [("x", r"\undefinedcmd{x}"), ("x", r"\frac{a}{b")]
    for r in _pairs(tmp_path, table):
        assert r.identical is False and r.tolerant is False
    assert _pairs(tmp_path, table)[0].pred_status == "compile-failed"


def test_gt_uncompilable_excluded(tmp_path):
    (r,) = _pairs(tmp_path, [(r"\undefinedcmd x", "x")])
    assert r.identical is None and r.tolerant is None and r.gt_status == "compile-failed"


def test_bisect_isolates_bad_equation(tmp_path):
    eqs = ["a", "b", "c", r"\undefinedcmd{d}", "e", "f", "g"]
    res = render.render_many(eqs, batch_size=8, cache=tmp_path)
    assert [r.status for r in res] == ["rendered"] * 3 + ["compile-failed"] + ["rendered"] * 3


def test_rejected_and_empty_make_no_docker_call(tmp_path, monkeypatch):
    calls = []
    real = subprocess.run
    monkeypatch.setattr(render.subprocess, "run", lambda *a, **k: calls.append(a) or real(*a, **k))
    res = render.render_many([r"\input{/etc/passwd}", ""], cache=tmp_path)
    assert [r.status for r in res] == ["rejected", "empty"]
    assert calls == []
    (p,) = render.render_equivalence(["x"], [r"\input{/etc/passwd}"], cache=tmp_path)
    assert p.pred_status == "rejected" and p.identical is False
    (q,) = render.render_equivalence(["x"], [""], cache=tmp_path)
    assert q.pred_status == "empty" and q.identical is False


def test_second_call_hits_cache(tmp_path, monkeypatch):
    first = render.render_many(["x^2", "y_1"], cache=tmp_path)
    calls = []
    real = subprocess.run
    monkeypatch.setattr(render.subprocess, "run", lambda *a, **k: calls.append(a) or real(*a, **k))
    second = render.render_many(["x^2", "y_1"], cache=tmp_path)
    assert calls == [] and first == second


def test_daemon_unreachable_raises_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///nonexistent/lom-docker.sock")
    with pytest.raises(render.RenderInfrastructureError):
        render.render_many(["x+1", "y"], cache=tmp_path, max_parallel=1)
    assert list((tmp_path / "render").rglob("*.json")) == []
    monkeypatch.delenv("DOCKER_HOST")
    (r,) = render.render_equivalence(["x+1"], ["x+1"], cache=tmp_path)
    assert r.gt_status == "rendered" and r.identical is True


def test_long_equation_glyph_errors_not_tolerant(tmp_path):
    lead = "a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} + a_{12} = "
    table = [(lead + r"\mathbf{x}", lead + "x"), (lead + r"\upsilon", lead + "v"), (lead + "f'", lead + "fl"), (lead + "I", lead + "1")]
    for pair, r in zip(table, _pairs(tmp_path, table), strict=True):
        assert r.identical is False and r.tolerant is False, pair
    (ok,) = _pairs(tmp_path, [(lead + "f'", lead + r"f^{\prime}")])
    assert ok.identical is True and ok.tolerant is True
