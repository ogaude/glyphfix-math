"""Guards the committed catalog files: side separation, required fields, unique ids."""

import re

import pytest

from lightonocr_math import splits
from lightonocr_math.corpus.records import LICENSE_SPDX
from lightonocr_math.paths import repo_root

TD = repo_root() / "data" / "catalog" / "train_dev.jsonl"
BM = repo_root() / "data" / "benchmark" / "catalog.jsonl"

pytestmark = pytest.mark.skipif(not TD.exists(), reason="catalog not built (scripts/corpus_catalog.py build)")


def _rows(p):
    from lightonocr_math.corpus._io import read_jsonl

    return read_jsonl(p)


def test_train_dev_has_no_benchmark_id():
    rows = _rows(TD)
    assert rows
    assert not [r["arxiv_id"] for r in rows if splits.is_benchmark(r["arxiv_id"])]


def test_benchmark_file_has_only_benchmark_ids():
    rows = _rows(BM)
    assert rows
    assert all(splits.is_benchmark(r["arxiv_id"]) for r in rows)


def test_required_fields_and_unique_ids():
    rows = _rows(TD) + _rows(BM)
    for r in rows:
        assert re.fullmatch(r"v\d+", r["version"]), r["arxiv_id"]
        assert r["license_spdx"] in LICENSE_SPDX
        assert r["title"] and r["authors"] and r["license_checked_at"]
        assert r["n_versions"] >= 1
    ids = [r["arxiv_id"] for r in rows]
    assert len(ids) == len(set(ids))
