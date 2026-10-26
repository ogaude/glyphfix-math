"""export_public: filtered history, kept metadata, source repository untouched."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.release import common as C
from scripts.release import export_public as E
from scripts.release import public_paths, rename_map
from scripts.release import rename_amendment as RA

SPEC = """\
[history]
src/**
scripts/**

[latest-only]
data/**

[overlay]
README.md
LICENSE
RELEASE_HISTORY.json
results/**
"""
ADDR = "olivier.gaude" + "@gmail.com"
DAYS = {i: f"2026-02-0{i}T10:00:0{i}+02:00" for i in range(1, 7)}


def sh(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_AUTHOR_NAME": "dev", "GIT_COMMITTER_NAME": "dev", "GIT_AUTHOR_EMAIL": ADDR, "GIT_COMMITTER_EMAIL": ADDR}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env, check=True).stdout.decode().strip()


def commit(repo: Path, files: dict, msg: str, day: int) -> str:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", msg, date=DAYS[day])
    return sh(repo, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    sh(src, "init", "-q", "-b", "master")
    (tmp_path / "spec.txt").write_text(SPEC)
    c = {}
    c[1] = commit(src, {"src/a.py": "v1\n", "README.md": "old readme\n", "data/x.txt": "v1\n", "misc/n.md": "n\n"}, "initial copy from Nexa" + "ris", 1)
    c[2] = commit(src, {"data/x.txt": "v2\n"}, "two", 2)
    c[3] = commit(src, {"src/a.py": f'UA = "{ADDR}"\n', "data/x.txt": "v3\n"}, "three", 3)
    c[4] = commit(src, {"misc/n.md": "changed\n"}, "four", 4)
    c[5] = commit(src, {"src/a.py": "clean\n", "data/y.txt": "y\n"}, "five", 5)
    ov = tmp_path / "overlay"
    (ov / "results").mkdir(parents=True)
    (ov / "README.md").write_text("public readme\n")
    (ov / "results" / "r.md").write_text("r\n")
    (ov / "SHA256SUMS").write_text("stale\n")
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"code_freeze": {"commits": {"src/a.py": [c[3], DAYS[3]], "scripts/b.py": [c[1], DAYS[1]]}}}))
    am = tmp_path / "amendment.json"
    am.write_text(json.dumps({"amendment_commit": [c[5], DAYS[5]]}))
    return {"src": src, "out": tmp_path / "out", "ov": ov, "res": res, "am": am, "spec": tmp_path / "spec.txt", "c": c}


def run(w, **kw):
    args = {"source": w["src"], "out": w["out"], "overlay": w["ov"], "published_results": w["res"], "amendment": w["am"], "spec_path": w["spec"]}
    args.update(kw)
    return E.export(**args)


def pub(w) -> Path:
    return w["out"] / "public.git"


def test_dropped_and_latest_only_paths(world):
    run(world)
    names = set(sh(pub(world), "log", "--all", "--name-only", "--format=").splitlines())
    assert "misc/n.md" not in names and "README.md" in names  # README.md only from the release commit
    assert sh(pub(world), "log", "--format=%s", "--", "README.md") == "release: add release documents and results"
    # data/x.txt only from the commit that introduced the HEAD blob; earlier versions never appear
    assert len(sh(pub(world), "log", "--format=%H", "--", "data/x.txt").splitlines()) == 1
    blobs = sh(pub(world), "log", "--all", "-p", "--format=", "--", "data/x.txt")
    assert "v1" not in blobs and "v2" not in blobs and "v3" in blobs


def test_history_files_keep_all_versions_and_empty_commits_are_pruned(world):
    rep = run(world)
    assert len(sh(pub(world), "log", "--format=%H", "--", "src/a.py").splitlines()) == 3
    assert rep["source_commits"] == 5 and rep["pruned_commits"] == 2  # commits two and four changed dropped paths only
    assert rep["public_commits"] == 3 + 1


def test_personal_address_is_scrubbed_from_blobs_and_identities(world):
    rep = run(world)
    assert rep["scrubbed_blobs"] == 1
    everything = sh(pub(world), "log", "--all", "-p", "--format=fuller")
    assert ADDR not in everything
    assert "contact@example.invalid" in everything
    for line in sh(pub(world), "log", "--format=%an <%ae> %cn <%ce>").splitlines():
        assert line == f"dev <{C.PUBLIC_IDENTITY_EMAIL}> dev <{C.PUBLIC_IDENTITY_EMAIL}>"
    assert C.PERSONAL_ADDRESS.search(ADDR)


def test_messages_are_scrubbed_and_otherwise_kept(world):
    rep = run(world)
    first = rep["mapped_commits"][world["c"][1]]
    assert sh(pub(world), "show", "-s", "--format=%s", first) == "initial copy from a private project"
    assert "Nexa" + "ris" not in sh(pub(world), "log", "--all", "--format=%B")


def test_dates_names_and_messages_survive(world):
    rep = run(world)
    c = world["c"]
    new = rep["mapped_commits"][c[3]]
    assert sh(pub(world), "show", "-s", "--format=%aI|%cI|%an|%s", new) == f"{DAYS[3]}|{DAYS[3]}|dev|three"
    assert sh(pub(world), "log", "-1", "--format=%cI", "main", "--", "src/a.py").startswith("2026-02-05")


def test_repository_shape_and_release_commit(world):
    rep = run(world)
    p = pub(world)
    assert sh(p, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert sh(p, "remote") == "" and sh(p, "for-each-ref", "refs/original") == "" and sh(p, "tag", "--list") == ""
    assert sh(p, "branch", "--list").split() == ["*", "main"] or sh(p, "branch", "--list").split() == ["main"]
    assert sh(p, "show", "--name-only", "--format=", "main").splitlines() == ["README.md", "RELEASE_HISTORY.json", "results/r.md"]
    doc = json.loads(sh(p, "show", "main:RELEASE_HISTORY.json"))
    assert set(doc["commit_map"]) == {world["c"][1], world["c"][3], world["c"][5]}
    assert doc["n_source_commits"] == 5 and doc["n_public_commits"] == rep["public_commits"]
    assert "SHA256SUMS" not in sh(p, "ls-tree", "-r", "--name-only", "main")
    assert (world["out"] / "EXPORT.json").is_file()


def test_source_repository_is_unchanged(world):
    src = world["src"]
    before = (E.snapshot(src), sh(src, "status", "--porcelain"))
    run(world)
    assert (E.snapshot(src), sh(src, "status", "--porcelain")) == before
    assert sh(world["out"] / "work", "remote") == ""


def test_published_commit_without_public_counterpart_fails(world):
    pruned = sh(world["src"], "rev-parse", world["c"][4])
    world["res"].write_text(json.dumps({"code_freeze": {"commits": {"src/a.py": [pruned, DAYS[4]]}}}))
    with pytest.raises(E.ExportError, match="no public counterpart"):
        run(world)


def test_refuses_modified_tracked_files(world):
    (world["src"] / "src/a.py").write_text("edited\n")
    with pytest.raises(E.ExportError, match="modified tracked files"):
        run(world)


def test_refuses_overlay_file_that_is_not_overlay(world):
    (world["ov"] / "stray.txt").write_text("x\n")
    with pytest.raises(E.ExportError, match="not classified as overlay"):
        run(world)


# ---- rename amendment commit
RENAME_SPEC = """\
[history]
src/**

[latest-only]
data/models/REGISTRY.json

[overlay]
README.md
RELEASE_HISTORY.json
"""
OLD_PKG = RA.OLD_MODULE.split("/")[1]
REG_PATH = "data/models/REGISTRY.json"
REG_OLD = json.dumps({"entries": [{"tag": RA.OLD_TAG}]})


def test_amendment2_adds_the_rename_role(tmp_path):
    res, am, am2 = tmp_path / "results.json", tmp_path / "FREEZE_AMENDMENT.json", tmp_path / "FREEZE_AMENDMENT_2.json"
    res.write_text(json.dumps({"code_freeze": {"commits": {"src/a.py": ["a" * 40, DAYS[1]]}}}))
    am.write_text(json.dumps({"amendment_commit": ["b" * 40, DAYS[2]]}))
    assert E._published_hashes(res, am) == {"a" * 40: "code_freeze:src/a.py", "b" * 40: "amendment_commit"}
    am2.write_text(json.dumps({"amendment_commit": ["c" * 40, DAYS[3]]}))
    roles = E._published_hashes(res, am)
    assert roles["c" * 40] == "rename_amendment_commit" and len(roles) == 3
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps({"amendment_commit": ["d" * 40, DAYS[3]]}))
    assert E._published_hashes(res, am, other)["d" * 40] == "rename_amendment_commit"


def test_rename_commit_without_public_counterpart_fails(world):
    world["am"].with_name("FREEZE_AMENDMENT_2.json").write_text(json.dumps({"amendment_commit": [world["c"][4], DAYS[4]]}))
    with pytest.raises(E.ExportError, match="rename_amendment_commit"):
        run(world)


def test_rename_commit_passes_pure_rewrite_check_after_export(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    sh(src, "init", "-q", "-b", "master")
    spec_path = tmp_path / "spec.txt"
    spec_path.write_text(RENAME_SPEC)
    spec = public_paths.load_spec(spec_path)
    new_module = rename_map.map_path(RA.OLD_MODULE)
    c1 = commit(src, {RA.OLD_MODULE: f"import {OLD_PKG}\n", REG_PATH: REG_OLD, "src/other.py": "v1\n"}, "initial", 1)
    sh(src, "mv", os.path.dirname(RA.OLD_MODULE), os.path.dirname(new_module))
    c2 = commit(
        src,
        {new_module: rename_map.rewrite_text(f"import {OLD_PKG}\n"), REG_PATH: rename_map.rewrite_text(REG_OLD)},
        "rename",
        2,
    )
    commit(src, {"src/other.py": "v2\n"}, "other", 3)

    before = RA.check_commit_rewrite(src, c2, exceptions=(), spec=spec)
    assert before["problems"] == [] and before["pure_rewrite"] is True
    reg_blob = sh(src, "rev-parse", f"HEAD:{REG_PATH}")
    proven = before["latest_only_rewritten"]
    assert proven == [{"path": REG_PATH, "new_blob": reg_blob}]

    ov = tmp_path / "overlay"
    ov.mkdir()
    (ov / "README.md").write_text("public readme\n")
    res, am, am2 = tmp_path / "results.json", tmp_path / "FREEZE_AMENDMENT.json", tmp_path / "FREEZE_AMENDMENT_2.json"
    res.write_text(json.dumps({"code_freeze": {"commits": {RA.OLD_MODULE: [c1, DAYS[1]]}}}))
    am.write_text(json.dumps({"amendment_commit": [c2, DAYS[2]]}))
    am2.write_text(json.dumps({"amendment_commit": [c2, DAYS[2]]}))
    out = tmp_path / "out"
    E.export(source=src, out=out, overlay=ov, published_results=res, amendment=am, amendment2=am2, spec_path=spec_path)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(out / "public.git"), str(clone)], check=True)
    mapped = json.loads(sh(clone, "show", "HEAD:RELEASE_HISTORY.json"))["commit_map"][c2]
    status = sh(clone, "diff", "--name-status", "--no-renames", f"{mapped}~1", mapped).splitlines()
    assert f"A\t{REG_PATH}" in status

    after = RA.check_commit_rewrite(clone, mapped, exceptions=(), spec=spec, proven_latest_only=proven)
    assert after["problems"] == []
    assert [e["path"] for e in after["accepted_latest_only"]] == [REG_PATH]
    assert after["pure_rewrite"] == "partial"
    unproven = RA.check_commit_rewrite(clone, mapped, exceptions=(), spec=spec, proven_latest_only=[])
    assert any(REG_PATH in p for p in unproven["problems"]) and unproven["pure_rewrite"] is False
    unfiltered = RA.check_commit_rewrite(clone, mapped, exceptions=(), spec=spec, proven_latest_only=proven, filtered=False)
    assert any(REG_PATH in p and "addition without a paired deletion" in p for p in unfiltered["problems"])
