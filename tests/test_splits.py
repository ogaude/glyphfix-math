import hashlib
import json

import pytest

from glyphfix_math import _frozen, splits
from glyphfix_math.splits import (
    LeakageError,
    SplitIntegrityError,
    SplitNotFrozenError,
    assert_disjoint,
    freeze,
    load_dev_ids,
    load_frozen,
    load_train_ids,
    rule_info,
    split_of_with_salt,
    u01,
)

SALT = "00ff00ff00ff00ff"


def _write_candidates(path, n=300):
    with path.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"arxiv_id": f"2301.{i:05d}", "authors": [f"First{i % 40} Last{i % 40}"]}) + "\n")
        fh.write(json.dumps({"arxiv_id": "not an id", "authors": []}) + "\n")
    return path


@pytest.fixture
def fs(tmp_path):
    cand = _write_candidates(tmp_path / "cand.jsonl")
    out = tmp_path / "splits"
    freeze(cand, out, salt=SALT)
    return load_frozen(out, None)


def test_version_independence():
    a = split_of_with_salt("2301.01234v1", SALT)
    assert a == split_of_with_salt("2301.01234v3", SALT) == split_of_with_salt("2301.01234", SALT)


def test_u01_golden():
    expected = int.from_bytes(hashlib.sha256((SALT + "2301.01234").encode()).digest()[:8], "big") / 2**64
    assert u01("2301.01234", SALT) == expected
    assert 0.0 <= expected < 1.0


def test_fractions():
    n = 20000
    counts = {"benchmark": 0, "dev": 0, "train": 0}
    for i in range(n):
        counts[split_of_with_salt(f"2301.{i:05d}", SALT)] += 1
    assert abs(counts["benchmark"] / n - 0.15) < 0.01
    assert abs(counts["dev"] / n - 0.05) < 0.01


def test_freeze_writes_files(tmp_path):
    cand = _write_candidates(tmp_path / "cand.jsonl")
    out = tmp_path / "s"
    stats = freeze(cand, out, salt=SALT)
    assert stats["invalid_ids"] == 1
    for name in ("SPLIT_RULE.json", "benchmark_ids.txt", "quarantine_ids.txt", "SPLIT_STATS.json", "SHA256SUMS"):
        assert (out / name).is_file()
    text = (out / "benchmark_ids.txt").read_text()
    lines = text.splitlines()
    assert text.endswith("\n") and lines == sorted(lines) and lines
    rule = json.loads((out / "SPLIT_RULE.json").read_text())
    for key in ("salt", "rule_version", "fractions", "hash", "author_rule", "candidates_sha256", "candidates_count"):
        assert key in rule
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert [s.split("  ")[1] for s in sums] == sorted(
        ["SPLIT_RULE.json", "benchmark_ids.txt", "quarantine_ids.txt", "SPLIT_STATS.json"]
    )
    for s in sums:
        h, name = s.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h


def test_freeze_refuses_overwrite_and_is_deterministic(tmp_path):
    cand = _write_candidates(tmp_path / "cand.jsonl")
    a, b = tmp_path / "a", tmp_path / "b"
    freeze(cand, a, salt=SALT)
    with pytest.raises(FileExistsError):
        freeze(cand, a, salt=SALT)
    freeze(cand, b, salt=SALT)
    for name in ("SPLIT_RULE.json", "benchmark_ids.txt", "quarantine_ids.txt", "SPLIT_STATS.json", "SHA256SUMS"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_load_frozen_integrity(tmp_path):
    cand = _write_candidates(tmp_path / "cand.jsonl")
    out = tmp_path / "s"
    freeze(cand, out, salt=SALT)
    digest = hashlib.sha256((out / "SHA256SUMS").read_bytes()).hexdigest()
    assert load_frozen(out, digest).digest == digest
    with pytest.raises(SplitIntegrityError):
        load_frozen(out, "0" * 64)
    with (out / "benchmark_ids.txt").open("a") as fh:
        fh.write("2301.99999\n")
    with pytest.raises(SplitIntegrityError):
        load_frozen(out, digest)
    with pytest.raises(SplitIntegrityError):
        load_frozen(out, None)


def _ids_by_split(fs, n=300):
    out = {"benchmark": [], "dev": [], "train": []}
    for i in range(n):
        b = f"2301.{i:05d}"
        out[split_of_with_salt(b, fs.salt)].append(b)
    return out


def test_assert_disjoint_reasons(fs):
    groups = _ids_by_split(fs)
    bench = groups["benchmark"][0]
    train = next(t for t in groups["train"] if t not in fs.quarantine)
    with pytest.raises(LeakageError) as ei:
        assert_disjoint([bench + "v2", train], context="t", fs=fs)
    assert ei.value.offenders == {bench: "benchmark-list"}

    unlisted = next(
        f"2399.{i:05d}" for i in range(100000) if split_of_with_salt(f"2399.{i:05d}", fs.salt) == "benchmark"
    )
    with pytest.raises(LeakageError) as ei:
        assert_disjoint([unlisted], context="t", fs=fs)
    assert ei.value.offenders == {unlisted: "benchmark-rule"}

    assert fs.quarantine, "synthetic data must produce quarantine"
    q = min(fs.quarantine)
    with pytest.raises(LeakageError) as ei:
        assert_disjoint([q], context="t", fs=fs)
    assert ei.value.offenders == {q: "quarantine"}

    with pytest.raises(LeakageError) as ei:
        assert_disjoint(["garbage"], context="t", fs=fs)
    assert ei.value.offenders == {"garbage": "invalid-id"}

    assert_disjoint([train], context="t", fs=fs)


def test_load_train_and_dev(fs):
    groups = _ids_by_split(fs)
    allids = [i + "v1" for ids in groups.values() for i in ids]
    train = load_train_ids(allids, fs)
    assert train == sorted(set(train))
    assert set(train) == {t for t in groups["train"] if t not in fs.quarantine}
    dev = load_dev_ids(allids, fs)
    assert set(dev) == {t for t in groups["dev"] if t not in fs.quarantine}


def test_not_frozen(monkeypatch):
    monkeypatch.setattr(_frozen, "FROZEN_SPLIT_DIGEST", None)
    with pytest.raises(SplitNotFrozenError):
        splits.frozen()
    assert rule_info() == {"split_rule_version": "1", "split_digest": None}


def test_cli_verify_and_info(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(_frozen, "FROZEN_SPLIT_DIGEST", None)
    cand = _write_candidates(tmp_path / "cand.jsonl")
    out = tmp_path / "s"
    assert splits.main(["freeze", "--candidates", str(cand), "--out", str(out), "--salt", SALT]) == 0
    assert splits.main(["verify", "--dir", str(out), "--candidates", str(cand)]) == 0
    assert splits.main(["verify", "--dir", str(out), "--candidates", str(tmp_path / "missing.jsonl")]) == 0
    assert splits.main(["info"]) == 0
    assert '"split_digest": null' in capsys.readouterr().out
