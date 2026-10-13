import json
from pathlib import Path

import pytest

from lightonocr_math import report as R
from lightonocr_math import score as S

FIX = Path(__file__).parent / "fixtures" / "adversarial_predictions.jsonl"


def _variant(tmp_path):
    rows = [json.loads(line) for line in FIX.read_text().splitlines() if line.strip()]
    wrong = [r for r in rows if r["expect"] == "wrong" and not r.get("pred_error") and r["gt_latex"] != "\\R^n"]
    right = [r for r in rows if r["expect"] == "right"]
    fixed = {r["id"] for r in wrong[:3]}
    broken = right[0]["id"]
    out = []
    for r in rows:
        r = dict(r)
        if r["id"] in fixed:
            r["pred_latex"] = r["gt_latex"]
        if r["id"] == broken:
            r["pred_latex"] = "zzz"
        out.append(r)
    p = tmp_path / "b.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in out))
    return p, fixed, broken


def test_render_report_contents(tmp_path):
    S.score_file(FIX, tmp_path / "a", render=False, B=100)
    md = (tmp_path / "a" / "report.md").read_text()
    for k in R.VERSION_KEYS:
        assert k in md
    for s in ("normalized exact match", "Per-class error rates", "Disagreements", "bug flag", "split_digest", "seed 0, B 100"):
        assert s in md
    assert "prime_l" in md


def test_report_has_render_tiers_text(tmp_path):
    res = json.loads(S.score_file(FIX, tmp_path / "a", render=False, B=50).read_text())
    res["metrics"]["render_tolerant"] = res["metrics"]["exact"]
    assert "render tolerant (separate tier)" in R.render_report(res)


def test_compare_flips(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=100)
    p, fixed, broken = _variant(tmp_path)
    rb = S.score_file(p, tmp_path / "b", render=False, B=100)
    cmp = R.compare_runs(ra, rb, B=200)
    f = cmp["metrics"]["exact"]["flips"]
    assert f["right_to_wrong"] == 1 and f["right_to_wrong_ids"] == [broken]
    assert f["wrong_to_right"] == 3 and set(f["wrong_to_right_ids"]) == fixed
    d = cmp["metrics"]["exact"]["delta"]
    assert d["delta"] == pytest.approx(2 / 42) and d["ci_low"] is not None


def test_compare_refuses_version_mismatch(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    res = json.loads(rb.read_text())
    res["scorer_version"] = "9.9.9"
    rb.write_text(json.dumps(res))
    with pytest.raises(R.VersionMismatchError, match="scorer_version"):
        R.compare_runs(ra, rb)


def test_compare_refuses_different_ids(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    lines = (tmp_path / "b" / "records.jsonl").read_text().splitlines()
    (tmp_path / "b" / "records.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="id sets"):
        R.compare_runs(ra, rb)


def test_cli_compare(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    p, _, _ = _variant(tmp_path)
    rb = S.score_file(p, tmp_path / "b", render=False, B=50)
    out = tmp_path / "flips.md"
    assert R.main(["compare", str(ra), str(rb), "--out", str(out), "--bootstrap", "100"]) == 0
    assert "right to wrong" in out.read_text()


def _mutate(rp, **changes):
    res = json.loads(rp.read_text())
    for k, v in changes.items():
        if "." in k:
            a, b = k.split(".")
            res[a] = {**(res.get(a) or {}), b: v}
        else:
            res[k] = v
    rp.write_text(json.dumps(res))


@pytest.mark.parametrize("change", [{"scorer_fingerprint": "x"}, {"split_digest": "x"}, {"render_dpi": 150}])
def test_compare_refuses_more_version_keys(tmp_path, change):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    _mutate(rb, **change)
    with pytest.raises(R.VersionMismatchError, match=next(iter(change))):
        R.compare_runs(ra, rb)


def test_compare_refuses_render_image_and_render_enabled_mismatch(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    _mutate(rb, **{"render.enabled": True})
    with pytest.raises(R.VersionMismatchError, match="render tiers"):
        R.compare_runs(ra, rb)
    _mutate(ra, **{"render.enabled": True, "render.image": "img@a"})
    _mutate(rb, **{"render.enabled": True, "render.image": "img@b"})
    with pytest.raises(R.VersionMismatchError, match="render image"):
        R.compare_runs(ra, rb)


def test_compare_refuses_different_ground_truth(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    p = tmp_path / "b" / "records.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["gt_latex"] += " y"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="ground truth differs"):
        R.compare_runs(ra, rb)


def test_compare_refuses_different_class_denominators(tmp_path):
    ra = S.score_file(FIX, tmp_path / "a", render=False, B=50)
    rb = S.score_file(FIX, tmp_path / "b", render=False, B=50)
    res = json.loads(rb.read_text())
    res["per_class"][R.CLASSES[0]]["denom"] += 1
    rb.write_text(json.dumps(res))
    with pytest.raises(ValueError, match="denominators differ"):
        R.compare_runs(ra, rb)
