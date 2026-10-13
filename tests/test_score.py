import json
from pathlib import Path

import pytest

from lightonocr_math import score as S

FIX = Path(__file__).parent / "fixtures" / "adversarial_predictions.jsonl"


def _fixture_rows():
    return [json.loads(line) for line in FIX.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_score_record_basic():
    r = S.score_record("f'", "fl")
    assert r["exact"] is False and r["classes"] == ["prime_l"]
    r = S.score_record("f'", "f^{\\prime}")
    assert r["exact"] is True and r["classes"] == []


def test_fingerprint_matches():
    assert S.scorer_fingerprint() == S.SCORER_FINGERPRINT, (
        "normalizer/error rules changed: bump SCORER_VERSION and update SCORER_FINGERPRINT"
    )


def test_fixture_rows_follow_expectations():
    fx = {r["id"]: r for r in _fixture_rows()}
    scored = {r["id"]: r for r in S.score_records(list(fx.values()), render=False)}
    for rid, f in fx.items():
        s = scored[rid]
        if f["expect"] == "wrong":
            assert s["exact"] is False, rid
            if "expect_class" in f:
                assert f["expect_class"] in s["classes"], (rid, s["classes"])
        else:
            assert s["exact"] is True, rid


def test_fixture_covers_all_target_errors():
    classes = {r["expect_class"] for r in _fixture_rows() if "expect_class" in r}
    assert {"prime_l", "dropped_prime", "greek_latin", "decoration", "i_1"} <= classes
    assert len({r["paper_id"] for r in _fixture_rows()}) == 4


def test_score_file_no_render(tmp_path):
    rp = S.score_file(FIX, tmp_path, render=False, B=200)
    res = json.loads(rp.read_text())
    for key in (
        "result_schema_version", "scorer_version", "normalize_rules_version", "error_rules_version",
        "split_rule_version", "split_digest", "bootstrap", "metrics", "per_class", "predictions_sha256", "n_records",
    ):
        assert key in res, key
    assert res["scorer_version"] == "1.0.0"
    assert res["render"] == {"enabled": False}
    assert res["bootstrap"] == {"B": 200, "seed": 0, "ci": 0.95}
    m = res["metrics"]["exact"]
    assert {"rate", "ci_low", "ci_high", "num", "den", "n_papers"} <= set(m)
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == len(_fixture_rows()) == res["n_records"]
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)


def test_pred_error_scores_as_empty(tmp_path):
    rows = {r["id"]: r for r in S.score_records(_fixture_rows(), render=False)}
    errs = [r for r in rows.values() if r.get("pred_error")]
    assert len(errs) == 1 and errs[0]["exact"] is False and errs[0]["pred_latex"] == ""


def test_validation_errors_name_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    good = {"id": "a", "paper_id": "p", "gt_latex": "x", "pred_latex": "x"}
    p.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="line 2.*duplicate"):
        S.read_predictions(p)
    p.write_text(json.dumps(good) + "\n" + json.dumps({**good, "id": "b", "gt_latex": ""}) + "\n")
    with pytest.raises(ValueError, match="line 2.*gt_latex"):
        S.read_predictions(p)
    p.write_text(json.dumps({k: v for k, v in good.items() if k != "gt_latex"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        S.read_predictions(p)


def test_cli_prints_summary(tmp_path, capsys):
    assert S.main([str(FIX), "--out", str(tmp_path), "--no-render", "--bootstrap", "100"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "normalized exact match" in out


@pytest.mark.docker
def test_score_file_with_render(tmp_path):
    rp = S.score_file(FIX, tmp_path, render=True, B=200, cache=tmp_path / "cache")
    res = json.loads(rp.read_text())
    assert res["scorer_version"] == "1.0.0"
    assert res["preamble_sha256"] and res["wrap_version"] and res["render_dpi"]
    assert "render_identical" in res["metrics"] and "render_tolerant" in res["metrics"]
    rows = {json.loads(line)["id"]: json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()}
    fx = {r["id"]: r for r in _fixture_rows()}
    undefined = next(i for i, f in fx.items() if f["pred_latex"] == "\\undefinedcmd{x}")
    assert rows[undefined]["pred_render_status"] == "compile-failed"
    assert rows[undefined]["exact"] is False and rows[undefined]["render_identical"] is False
    macro = next(i for i, f in fx.items() if f["gt_latex"] == "\\R^n")
    assert rows[macro]["render_identical"] is None
    assert res["n_gt_uncompilable"] >= 1
    assert res["metrics"]["render_identical"]["den"] == res["n_records"] - res["n_gt_uncompilable"]
    bm = next(i for i, f in fx.items() if f["pred_latex"] == "\\bm{x}" and f["gt_latex"] == "\\boldsymbol{x}")
    assert bm in res["disagreements"]["string_diff_render_identical"]
    err = next(i for i, f in fx.items() if f.get("pred_error"))
    assert rows[err]["render_identical"] is False
