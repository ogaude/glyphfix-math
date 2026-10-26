"""Runtime freeze guard: digests of the real frozen files hold; tampered copies are refused."""

import json
import shutil
from pathlib import Path

import pytest

from glyphfix_math.corpus import freeze
from glyphfix_math.corpus._io import read_jsonl, write_jsonl

BENCH = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def _copy(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    shutil.copy(BENCH / "BENCHMARK_FREEZE.json", root / "BENCHMARK_FREEZE.json")
    shutil.copy(BENCH / "records.jsonl", root / "records.jsonl")
    (root / "pages").mkdir()
    shutil.copy(BENCH / "pages" / "PAGES_FREEZE.json", root / "pages" / "PAGES_FREEZE.json")
    shutil.copy(BENCH / "pages" / "pages.jsonl", root / "pages" / "pages.jsonl")
    return root


def test_real_files_pass():
    assert freeze.is_frozen(BENCH)
    freeze.assert_benchmark_frozen(BENCH)
    freeze.assert_pages_frozen(BENCH)


def test_records_text_digest_matches_write_jsonl(tmp_path):
    rows = read_jsonl(BENCH / "records.jsonl")
    write_jsonl(tmp_path / "r.jsonl", rows)
    assert freeze.records_text_sha256(rows) == json.loads((BENCH / "BENCHMARK_FREEZE.json").read_text())["records_sha256"]


def test_tampered_records_refused(tmp_path):
    root = _copy(tmp_path)
    freeze.assert_benchmark_frozen(root)
    with open(root / "records.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(freeze.FreezeError):
        freeze.assert_benchmark_frozen(root)


def test_tampered_freeze_file_refused(tmp_path):
    root = _copy(tmp_path)
    p = root / "BENCHMARK_FREEZE.json"
    p.write_text(p.read_text().replace('"n_records"', '"n_records "'))
    with pytest.raises(freeze.FreezeError):
        freeze.assert_benchmark_frozen(root)


def test_tampered_pages_refused(tmp_path):
    root = _copy(tmp_path)
    freeze.assert_pages_frozen(root)
    with open(root / "pages" / "pages.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(freeze.FreezeError):
        freeze.assert_pages_frozen(root)


def test_missing_files_refused(tmp_path):
    with pytest.raises(freeze.FreezeError):
        freeze.assert_benchmark_frozen(tmp_path)
    assert not freeze.is_frozen(tmp_path)


def test_baseline_infer_refuses_drifted_frozen_records(tmp_path, monkeypatch):
    from scripts import baseline_infer as bi

    def boom(_root):
        raise freeze.FreezeError("drift")

    monkeypatch.setattr(freeze, "assert_benchmark_frozen", boom)
    argv = ["--suite", "benchmark", "--records", str(BENCH / "records.jsonl"), "--out", str(tmp_path / "o"), "--sidecar", str(tmp_path / "s")]
    assert bi.main(argv) == 1
    assert not (tmp_path / "o").exists()


def test_pinned_content_digests_match_freeze_files():
    from glyphfix_math.corpus import _benchmark_frozen as pin

    assert pin.BENCHMARK_RECORDS_SHA256 == json.loads((BENCH / "BENCHMARK_FREEZE.json").read_text())["records_sha256"]
    assert pin.PAGES_JSONL_SHA256 == json.loads((BENCH / "pages" / "PAGES_FREEZE.json").read_text())["pages_sha256"]


def test_deleting_freeze_file_does_not_unfreeze(tmp_path):
    root = _copy(tmp_path)
    (root / freeze.FREEZE_NAME).unlink()
    assert freeze.is_frozen(root)  # pinned records still recognised by content
    with pytest.raises(freeze.FreezeError):
        freeze.assert_benchmark_frozen(root)
    assert freeze.is_frozen(BENCH)


def test_unrelated_root_is_not_frozen(tmp_path):
    (tmp_path / "records.jsonl").write_text("{}\n")
    assert not freeze.is_frozen(tmp_path)
    assert not freeze.pages_dir_is_frozen(tmp_path)


def test_pages_dir_frozen_by_content_or_location(tmp_path):
    root = _copy(tmp_path)
    assert freeze.pages_dir_is_frozen(root / "pages")
    (root / "pages" / "PAGES_FREEZE.json").unlink()
    assert freeze.pages_dir_is_frozen(root / "pages")  # pinned pages.jsonl content
    assert freeze.pages_dir_is_frozen(BENCH / "pages")


def test_baseline_infer_refuses_unfrozen_copy_without_override(tmp_path):
    from scripts import baseline_infer as bi

    rec = tmp_path / "copy.jsonl"
    rec.write_text('{"id": "x"}\n')
    with pytest.raises(freeze.FreezeError):
        bi._assert_frozen_inputs("benchmark", rec)
    with pytest.raises(freeze.FreezeError):
        bi._assert_frozen_inputs("pages", rec)
    bi._assert_frozen_inputs("benchmark", rec, allow_unfrozen=True)
    bi._assert_frozen_inputs("regression", rec)  # training-side suites are unaffected


def test_baseline_infer_checks_frozen_copy_by_content(tmp_path):
    from scripts import baseline_infer as bi

    copy = tmp_path / "elsewhere.jsonl"
    shutil.copy(BENCH / "records.jsonl", copy)
    bi._assert_frozen_inputs("benchmark", copy)  # identical bytes, other path: verified against the repo freeze


def test_corpus_pages_build_refuses_frozen_output(tmp_path, capsys):
    from scripts import corpus_pages as cp

    ids = tmp_path / "ids.txt"
    ids.write_text("")
    a = cp.argparse.Namespace(papers=str(ids), side="benchmark", out=str(BENCH / "pages"), sources="s", shards="h", n=1, min_eqs=3)
    assert cp.cmd_build(a) == 1
    assert "frozen" in capsys.readouterr().err
