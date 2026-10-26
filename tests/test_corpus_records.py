import hashlib

import pytest

from glyphfix_math import splits
from glyphfix_math.corpus import records as R


def test_record_id_matches_definition_and_uses_base_id():
    want = hashlib.sha256(b"2301.01234|v2|x^2").hexdigest()[:16]
    assert R.record_id("2301.01234v2", "v2", "x^2") == want
    assert R.record_id("2301.01234", "v2", "x^2") == want


def test_record_id_sensitive_to_label_and_version():
    base = R.record_id("2301.01234", "v2", "x^2")
    assert R.record_id("2301.01234", "v2", "x^3") != base
    assert R.record_id("2301.01234", "v3", "x^2") != base


def test_exclusion_enum_is_closed():
    assert set(R.EXCLUSION_REASONS) == {
        "macros", "engine", "render", "duplicate", "no_source", "no_main", "bad_archive", "short_long",
        "canon_mismatch", "unsafe", "license", "version_drift", "cap", "quota", "leak",
    }
    assert R.exclusion_row("2301.01234", R.Exclusion.MACROS, line=3) == {"arxiv_id": "2301.01234", "reason": "macros", "line": 3}
    with pytest.raises(ValueError):
        R.exclusion_row("2301.01234", "because")


def _ids():
    fs = splits.frozen()
    bench = min(i for i in fs.benchmark if "/" not in i)
    other = next(f"2401.{n:05d}" for n in range(1, 10000) if splits.split_of(f"2401.{n:05d}") != "benchmark")
    return bench, other


def test_side_routing():
    bench, other = _ids()
    assert R.side_of(bench) == "benchmark"
    assert R.side_of(other) == "corpus"
    assert R.side_root("benchmark").parts[-2:] == ("data", "benchmark")
    assert R.side_root("corpus").parts[-2:] == ("data", "corpus")
    with pytest.raises(ValueError):
        R.side_root("x")


def _rec(**kw):
    rec = {"arxiv_id": "2401.00001", "paper_version": "v1", "latex": "x^2", "license_spdx": "CC-BY-4.0"}
    rec["id"] = R.record_id(rec["arxiv_id"], rec["paper_version"], rec["latex"])
    rec.update(kw)
    return rec


def test_validate_record():
    assert R.validate_record(_rec()) == []
    assert any("arxiv_id" in p for p in R.validate_record(_rec(arxiv_id="")))
    assert any("record_id" in p for p in R.validate_record(_rec(id="0" * 16)))
    assert any("license_spdx" in p for p in R.validate_record(_rec(license_spdx="CC-BY-NC-4.0")))
    assert any("latex" in p for p in R.validate_record(_rec(latex="")))


def test_to_score_row():
    rec = _rec(arxiv_id="2401.00001v1")
    assert R.to_score_row(rec) == {"id": rec["id"], "paper_id": "2401.00001", "gt_latex": "x^2"}
