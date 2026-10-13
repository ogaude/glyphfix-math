import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from lightonocr_math import _frozen
from lightonocr_math.splits import (
    LeakageError,
    SplitIntegrityError,
    assert_disjoint,
    frozen,
    load_frozen,
    load_train_ids,
    split_of_with_salt,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
CANDIDATES = ROOT / "data" / "candidates.jsonl"


def test_digest_is_hex_and_matches_sums_file():
    d = _frozen.FROZEN_SPLIT_DIGEST
    assert d is not None and re.fullmatch(r"[0-9a-f]{64}", d)
    assert d == hashlib.sha256((SPLITS / "SHA256SUMS").read_bytes()).hexdigest()


def test_benchmark_count_matches_stats():
    fs = frozen()
    stats = json.loads((SPLITS / "SPLIT_STATS.json").read_text(encoding="utf-8"))
    assert len(fs.benchmark) == stats["counts"]["benchmark"]
    assert len(fs.quarantine) == stats["quarantine_count"]


def test_every_benchmark_id_hashes_to_benchmark():
    fs = frozen()
    bad = [i for i in fs.benchmark if split_of_with_salt(i, fs.salt) != "benchmark"]
    assert bad == []


def test_benchmark_and_quarantine_are_disjoint():
    fs = frozen()
    assert not (fs.benchmark & fs.quarantine)


def test_tampering_with_a_split_file_is_detected(tmp_path):
    tmp = tmp_path / "splits"
    shutil.copytree(SPLITS, tmp)
    with open(tmp / "benchmark_ids.txt", "a", encoding="utf-8") as f:
        f.write("9999.99999\n")
    with pytest.raises(SplitIntegrityError):
        load_frozen(tmp, _frozen.FROZEN_SPLIT_DIGEST)


def test_candidates_file_matches_recorded_sha256():
    if not CANDIDATES.is_file():
        pytest.skip("data/candidates.jsonl not present")
    rule = json.loads((SPLITS / "SPLIT_RULE.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(CANDIDATES.read_bytes()).hexdigest() == rule["candidates_sha256"]


def test_gate_rejects_benchmark_and_quarantine_ids():
    fs = frozen()
    first = sorted(fs.benchmark)[:5]
    with pytest.raises(LeakageError):
        assert_disjoint(first, context="test", fs=fs)
    assert load_train_ids(sorted(fs.benchmark | fs.quarantine), fs=fs) == []
