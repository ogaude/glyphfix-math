"""The frozen-scorer harness: one proven pin-commit substitution, refusals otherwise (throwaway repositories only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.acceptance as acc
from scripts.release import frozen_rescore as fr
from scripts.release import rename_map

OLD = fr.OLD_MODULE
NEW = fr.NEW_MODULE
OLD_PKG = Path(OLD).parts[1]
TEXT = f"import {OLD_PKG}\nNAME = '{OLD_PKG}'\n"
D1 = "2026-10-17T09:59:12+02:00"
D2 = "2026-10-26T10:00:00+02:00"


def git(root: Path, *args: str, date: str | None = None) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
        "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(root),
    }  # fmt: skip
    if date:
        env["GIT_COMMITTER_DATE"] = date
        env["GIT_AUTHOR_DATE"] = date
    r = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def make_repo(root: Path, new_text: str | None = None, extra_commit: bool = False) -> dict:
    git(root, "init", "-q")
    p_old = root / OLD
    p_old.parent.mkdir(parents=True)
    p_old.write_text(TEXT, encoding="utf-8")
    git(root, "add", OLD)
    git(root, "commit", "-q", "-m", "one", date=D1)
    c1 = git(root, "rev-parse", "HEAD")
    blob = git(root, "rev-parse", f"{c1}:{OLD}")
    git(root, "rm", "-q", OLD)
    p_new = root / NEW
    p_new.parent.mkdir(parents=True)
    p_new.write_text(rename_map.rewrite_text(TEXT) if new_text is None else new_text, encoding="utf-8")
    git(root, "add", NEW)
    git(root, "commit", "-q", "-m", "two", date=D2)
    if extra_commit:
        p_new.write_text(p_new.read_text(encoding="utf-8") + "# x\n", encoding="utf-8")
        git(root, "commit", "-q", "-am", "three", date=D2)
    published = {"code_freeze": {"commits": {OLD: [c1, D1]}}}
    return {"c1": c1, "blob": blob, "published": published}


def sub(root: Path, info: dict, **kw):
    kw.setdefault("commit_map", {})
    return fr.pin_substitution(root, published=info["published"], module_blob=info["blob"], **kw)


def test_substitution_returns_the_pre_move_commit(tmp_path):
    info = make_repo(tmp_path)
    r = sub(tmp_path, info)
    assert r["path"] == NEW and r["commit"] == info["c1"] and r["date"] == D1
    assert r["module_blob_before"] == info["blob"]


def test_substitution_uses_the_commit_map(tmp_path):
    info = make_repo(tmp_path)
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "twin", date=D1)
    x = git(tmp_path, "rev-parse", "HEAD")
    # the twin carries the same date but not the blob at the old path: refused
    with pytest.raises(fr.SubstitutionRefused):
        sub(tmp_path, info, commit_map={info["c1"]: x})


def test_substitution_maps_to_a_twin_commit_with_the_same_date_and_blob(tmp_path):
    info = make_repo(tmp_path)
    # a twin of the first commit (as in a public history): same tree, same date, other hash
    tree = git(tmp_path, "rev-parse", f"{info['c1']}^{{tree}}")
    twin = git(tmp_path, "commit-tree", tree, "-m", "twin", date=D1)
    assert twin != info["c1"]
    assert sub(tmp_path, info, commit_map={info["c1"]: twin})["commit"] == twin


def test_refuses_when_the_map_lacks_the_pin_commit(tmp_path):
    info = make_repo(tmp_path)
    with pytest.raises(fr.SubstitutionRefused, match="commit map"):
        sub(tmp_path, info, commit_map={"f" * 40: info["c1"]})


def test_refuses_two_commits_on_the_new_path(tmp_path):
    info = make_repo(tmp_path, extra_commit=True)
    with pytest.raises(fr.SubstitutionRefused, match="exactly one"):
        sub(tmp_path, info)


def test_refuses_one_extra_byte(tmp_path):
    info = make_repo(tmp_path, new_text=rename_map.rewrite_text(TEXT) + " ")
    with pytest.raises(fr.SubstitutionRefused, match="pinned blob"):
        sub(tmp_path, info)


def test_refuses_a_different_pin_date(tmp_path):
    info = make_repo(tmp_path)
    info["published"]["code_freeze"]["commits"][OLD][1] = "2026-10-17T10:00:00+02:00"
    with pytest.raises(fr.SubstitutionRefused, match="committer date"):
        sub(tmp_path, info)


def test_refuses_an_unexpected_old_blob(tmp_path):
    info = make_repo(tmp_path)
    with pytest.raises(fr.SubstitutionRefused, match="expected"):
        fr.pin_substitution(tmp_path, {}, published=info["published"], module_blob="0" * 40)


def test_refuses_uncommitted_changes_to_the_new_module(tmp_path):
    info = make_repo(tmp_path)
    (tmp_path / NEW).write_text("changed\n", encoding="utf-8")
    with pytest.raises(fr.SubstitutionRefused, match="uncommitted"):
        sub(tmp_path, info)


def test_refuses_when_the_published_results_are_not_the_recorded_ones(tmp_path):
    make_repo(tmp_path)
    p = tmp_path / fr.PUBLISHED_PATH
    p.parent.mkdir(parents=True)
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(fr.SubstitutionRefused, match="sha256"):
        fr.pin_substitution(tmp_path, {})


def test_refuses_without_a_recorded_pin(tmp_path):
    info = make_repo(tmp_path)
    with pytest.raises(fr.SubstitutionRefused, match="no pin commit"):
        fr.pin_substitution(tmp_path, {}, published={"code_freeze": {"commits": {}}}, module_blob=info["blob"])


# ---------------------------------------------------------------------------------------------- main()

def test_main_before_the_move_runs_the_scorer_unpatched(monkeypatch):
    original = acc.file_commit
    seen = {}

    def fake_main(argv):
        seen["file_commit"] = acc.file_commit
        seen["argv"] = list(argv)
        return 7

    monkeypatch.setattr(acc, "FROZEN_CODE", (OLD, "scripts/acceptance.py"))
    monkeypatch.setattr(acc, "main", fake_main)
    assert fr.main(["score", "--mode", "benchmark"]) == 7
    assert seen["file_commit"] is original and seen["argv"] == ["score", "--mode", "benchmark"]


def test_main_after_the_move_substitutes_only_the_module_and_restores(monkeypatch, capsys):
    original = acc.file_commit
    seen = {}

    def fake_main(argv):
        seen["module"] = acc.file_commit(NEW)
        seen["module_posix"] = acc.file_commit(Path(NEW))
        seen["other"] = acc.file_commit("scripts/acceptance.py")
        return 0

    monkeypatch.setattr(acc, "FROZEN_CODE", (NEW, "scripts/acceptance.py"))
    monkeypatch.setattr(acc, "main", fake_main)
    monkeypatch.setattr(fr, "pin_substitution", lambda *a, **k: {"path": NEW, "commit": "c" * 40, "date": D1, "module_blob_before": "b" * 40})
    monkeypatch.setattr(acc, "file_commit", lambda p: ("orig", "d"))
    wrapped_original = acc.file_commit
    assert fr.main(["score"]) == 0
    assert seen["module"] == ("c" * 40, D1) and seen["module_posix"] == ("c" * 40, D1)
    assert seen["other"] == ("orig", "d")
    assert acc.file_commit is wrapped_original
    assert capsys.readouterr().err.startswith("frozen_rescore: module pin commit")
    assert original is not None


def test_main_refuses_on_a_refused_substitution(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(acc, "FROZEN_CODE", (NEW, "scripts/acceptance.py"))
    monkeypatch.setattr(acc, "main", lambda argv: called.append(1) or 0)

    def refuse(*a, **k):
        raise fr.SubstitutionRefused("nope")

    monkeypatch.setattr(fr, "pin_substitution", refuse)
    assert fr.main(["score"]) == 2
    assert not called and capsys.readouterr().err.startswith("REFUSED: nope")


def test_main_refuses_an_unknown_module_path(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(acc, "FROZEN_CODE", ("src/other/acceptance.py", "scripts/acceptance.py"))
    monkeypatch.setattr(acc, "main", lambda argv: called.append(1) or 0)
    assert fr.main(["score"]) == 2
    assert not called and "REFUSED" in capsys.readouterr().err


def test_source_has_no_literal_old_name_and_writes_no_file():
    src = Path("scripts/release/frozen_rescore.py").read_text(encoding="utf-8")
    assert "write_text(" not in src and "write_bytes(" not in src and "open(" not in src
