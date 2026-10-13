import json
import subprocess

import pytest

from lightonocr_math import _frozen
from lightonocr_math.ids import safe_id
from lightonocr_math.leakscan import ALLOW_PREFIXES, scan_repo
from lightonocr_math.paths import repo_root
from lightonocr_math.splits import freeze, frozen, load_frozen, split_of_with_salt

SALT = "1234abcd1234abcd"


@pytest.fixture
def env(tmp_path):
    rows = [{"arxiv_id": f"2301.{i:05d}", "authors": [f"A{i} B{i}"]} for i in range(200)]
    rows.append({"arxiv_id": "math/0508005", "authors": ["Old Timer"]})
    rows += [{"arxiv_id": f"math/05{i:05d}", "authors": []} for i in range(1, 60)]
    cand = tmp_path / "cand.jsonl"
    cand.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "splits"
    freeze(cand, out, salt=SALT)
    fs = load_frozen(out, None)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    old = next(i for i in sorted(fs.benchmark) if "/" in i)
    new = next(i for i in sorted(fs.benchmark) if "/" not in i)
    return repo, fs, new, old


def _write(repo, rel, text):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_clean_repo_has_no_leaks(env):
    repo, fs, _new, _ = env
    _write(repo, "data/train.jsonl", json.dumps({"arxiv_id": "fixture-a", "year": "2024"}) + "\n")
    assert scan_repo(repo, fs) == []


def test_jsonl_leak_with_version(env):
    repo, fs, new, _ = env
    _write(repo, "data/train.jsonl", json.dumps({"arxiv_id": new + "v2"}) + "\n")
    leaks = scan_repo(repo, fs)
    assert len(leaks) == 1
    assert leaks[0].arxiv_id == new and leaks[0].reason == "benchmark-list"
    assert leaks[0].path == "data/train.jsonl"


def test_csv_old_style_id(env):
    repo, fs, _, old = env
    _write(repo, "manifests/x.csv", f"paper_id,other\n{old},1\n")
    leaks = scan_repo(repo, fs)
    assert [lk.arxiv_id for lk in leaks] == [old]


def test_path_segment_safe_id(env):
    repo, fs, new, old = env
    _write(repo, f"crops/{safe_id(old)}/eq1.png", "x")
    _write(repo, f"crops/{new}v3/eq1.png", "x")
    assert {lk.arxiv_id for lk in scan_repo(repo, fs)} == {old, new}


def test_txt_line(env):
    repo, fs, new, _ = env
    _write(repo, "ids.txt", f"hello\n{new}\n")
    leaks = scan_repo(repo, fs)
    assert len(leaks) == 1 and leaks[0].where == "line 2"


def test_rule_only_leak(env):
    repo, fs, _, _ = env
    unlisted = next(
        f"2399.{i:05d}" for i in range(100000) if split_of_with_salt(f"2399.{i:05d}", fs.salt) == "benchmark"
    )
    assert unlisted not in fs.benchmark
    _write(repo, "data/m.json", json.dumps({"rows": [{"arxiv_id": unlisted}]}))
    leaks = scan_repo(repo, fs)
    assert len(leaks) == 1 and leaks[0].reason == "benchmark-rule"


def test_allowlist_titles_and_ignored(env):
    repo, fs, new, _ = env
    assert "data/splits/" in ALLOW_PREFIXES
    _write(repo, "data/splits/benchmark_ids.txt", new + "\n")
    _write(repo, "data/candidates.jsonl", json.dumps({"arxiv_id": new}) + "\n")
    _write(repo, "data/titles.jsonl", json.dumps({"title": new, "arxiv_id": "fixture-a"}) + "\n")
    _write(repo, ".gitignore", "ignored/\n")
    _write(repo, "ignored/leak.jsonl", json.dumps({"arxiv_id": new}) + "\n")
    assert scan_repo(repo, fs) == []


def test_tracked_and_untracked_both_scanned(env):
    repo, fs, new, old = env
    _write(repo, "tracked.jsonl", json.dumps({"arxiv_id": new}) + "\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.jsonl"], check=True)
    _write(repo, "untracked.jsonl", json.dumps({"arxiv_id": old}) + "\n")
    assert {lk.path for lk in scan_repo(repo, fs)} == {"tracked.jsonl", "untracked.jsonl"}


def test_repo_has_no_benchmark_leaks():
    if _frozen.FROZEN_SPLIT_DIGEST is None:
        pytest.skip("split not frozen yet (plan 05)")
    assert scan_repo(repo_root(), frozen()) == []
