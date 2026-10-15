"""Runtime freeze guard: digests of the real frozen files hold; tampered copies are refused."""

import json
import shutil
from pathlib import Path

import pytest

from lightonocr_math.corpus import freeze
from lightonocr_math.corpus._io import read_jsonl, write_jsonl

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
