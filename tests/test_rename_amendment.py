"""Tests of the rename amendment proof core: names-only proof, record edits, tree prediction, commit check.

Nothing here reads the worktree's frozen files, RUN_LOCK.json or REGISTRY.json: real frozen texts come from git
objects (they stay reachable after the rename) and the lock and registry are synthetic fixtures.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import freeze_amendment as fa
from scripts.release import rename_amendment as ra
from scripts.release import rename_map

ROOT = Path(__file__).resolve().parent.parent
OLD_PKG = "lightonocr_math"
NEW_PKG = "glyphfix_math"
OLD_DIR = f"src/{OLD_PKG}"
NEW_DIR = f"src/{NEW_PKG}"


def git(repo: Path, *args: str, input: bytes | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input, check=True)
    return r.stdout.decode("utf-8", "replace").strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    for k, v in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        git(path, "config", k, v)
    return path


def commit_all(repo: Path, msg: str = "c") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def blob_text(sha: str) -> str:
    return subprocess.run(["git", "cat-file", "-p", sha], cwd=ROOT, capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------ prove_rename

OLD_TEXT = f'''"""doc"""
from {OLD_PKG} import guard
from {OLD_PKG}.corpus import freeze

# a comment
TAG = "{ra.OLD_TAG}"
FROZEN = ("{ra.OLD_MODULE}", "x.py")


def f(x):
    return x + 1  # keep
'''
NEW_TEXT = rename_map.rewrite_text(OLD_TEXT)


def test_prove_rename_positive():
    p = ra.prove_rename(OLD_TEXT, NEW_TEXT)
    assert p["changed_lines"] == [2, 3, 6, 7]
    assert p["changed_tokens"] == {"NAME": 2, "STRING": 2}
    assert p["ast_equal_after_mapping"] is True
    assert {s["line"] for s in p["substitutions"]} == {2, 3, 6, 7}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("# a comment", "# another comment"),
        lambda t: t.replace("x + 1", "x + 2"),
        lambda t: t + "\nX = 1\n",
        lambda t: t.replace("\n\n\ndef f", "\n\ndef f"),
        lambda t: t.replace(f"from {NEW_PKG} import", f"from {NEW_PKG}x import"),
        lambda t: t.replace(f"from {NEW_PKG}.corpus", f"from {OLD_PKG}.corpus"),
        lambda t: t.replace('"x.py"', '"y.py"'),
        lambda t: t.replace(f"from {NEW_PKG} import guard\nfrom {NEW_PKG}.corpus import freeze",
                            f"from {NEW_PKG}.corpus import freeze\nfrom {NEW_PKG} import guard"),
    ],
    ids=["comment", "number", "added-line", "whitespace", "wrong-name", "token-left", "string", "swapped-imports"],
)
def test_prove_rename_rejects_other_changes(mutate):
    mutated = mutate(NEW_TEXT)
    assert mutated != NEW_TEXT
    with pytest.raises(ra.AmendmentError):
        ra.prove_rename(OLD_TEXT, mutated)


def test_prove_rename_rejects_identity():
    with pytest.raises(ra.AmendmentError):
        ra.prove_rename("x = 1\n", "x = 1\n")


def test_prove_rename_real_cli_from_git_object():
    old = blob_text(ra.CLI_OLD_BLOB)
    p = ra.prove_rename(old, rename_map.rewrite_text(old))
    assert p["changed_lines"] == [40, 41, 42, 43, 44, 45, 46, 47, 55, 62, 313]


def test_prove_rename_real_module_from_git_object():
    old = blob_text(ra.MODULE_OLD_BLOB)
    p = ra.prove_rename(old, rename_map.rewrite_text(old))
    assert p["changed_lines"] == [19, 20]


# --------------------------------------------------------------- lock and registry

def _lock():
    arm = {
        "acceptance_cli_blob": ra.CLI_OLD_BLOB,
        "acceptance_module_blob": ra.MODULE_OLD_BLOB,
        "acceptance_cli_blob_generated": ra.CLI_GENERATED_BLOB,
        "acceptance_commit": "a" * 40,
        "freeze_amendment": "evidence/acceptance/FREEZE_AMENDMENT.json",
        "model": ra.OLD_TAG,
        "model_digest": "d" * 64,
        "started": "2026-10-17T08:23:33Z",
        "finished": "2026-10-17T09:37:17Z",
        "n_ok": 3,
    }
    return {"arms": {"tuned-crops": dict(arm), "tuned-pages": dict(arm)}}


def test_amended_lock_rename_changes_exactly_the_blob_keys():
    lock = _lock()
    out = ra.amended_lock_rename(lock, cli_new="c" * 40, module_new="m" * 40)
    assert lock == _lock()
    for arm in ("tuned-crops", "tuned-pages"):
        before, after = lock["arms"][arm], out["arms"][arm]
        assert after["acceptance_cli_blob"] == "c" * 40
        assert after["acceptance_module_blob"] == "m" * 40
        assert after["acceptance_module_blob_generated"] == ra.MODULE_OLD_BLOB
        assert after["freeze_amendment"] == "evidence/acceptance/FREEZE_AMENDMENT_2.json"
        assert after["acceptance_cli_blob_generated"] == ra.CLI_GENERATED_BLOB
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        assert changed == {
            "acceptance_cli_blob", "acceptance_module_blob", "acceptance_module_blob_generated", "freeze_amendment",
        }


@pytest.mark.parametrize("key,value", [
    ("acceptance_cli_blob", "x" * 40),
    ("acceptance_module_blob", "y" * 40),
    ("acceptance_cli_blob_generated", "z" * 40),
])
def test_amended_lock_rename_rejects_unexpected_blobs(key, value):
    bad = copy.deepcopy(_lock())
    bad["arms"]["tuned-pages"][key] = value
    with pytest.raises(ra.AmendmentError):
        ra.amended_lock_rename(bad, cli_new="c" * 40, module_new="m" * 40)


def _registry(extra_entry=None, tag=ra.OLD_TAG, cand=ra.OLD_CAND_TAG):
    entries = [{"tag": tag, "candidate_tag": cand, "n": 1}, {"tag": "other:1", "note": "unrelated"}]
    if extra_entry:
        entries.append(extra_entry)
    return json.dumps({"entries": entries}, indent=2) + "\n"


def test_renamed_registry_text_exactly_two_leaves():
    text = _registry()
    new = ra.renamed_registry_text(text)
    assert new == rename_map.rewrite_text(text)
    changes = ra.registry_changes(text, new)
    assert {(c["old"], c["new"]) for c in changes} == {(ra.OLD_TAG, ra.NEW_TAG), (ra.OLD_CAND_TAG, ra.NEW_CAND_TAG)}
    assert len(changes) == 2
    assert ra.NEW_TAG == "glyphfix-math-lora-q8:p4-full-1" and ra.NEW_CAND_TAG == "glyphfix-math-cand:p4-full-1"


def test_renamed_registry_text_rejects_a_third_change_or_a_missing_one():
    with pytest.raises(ra.AmendmentError):
        ra.renamed_registry_text(_registry({"tag": "t", "path": f"src/{OLD_PKG}/x.py"}))
    with pytest.raises(ra.AmendmentError):
        ra.renamed_registry_text(_registry(cand="plain"))


# -------------------------------------------------------------------- predict_tree

def test_predict_tree_equals_apply_and_touches_neither_index_nor_worktree(tmp_path):
    repo = init_repo(tmp_path / "r")
    exp = repo / ra.EXP001_OLD
    (exp / "sub").mkdir(parents=True)
    (exp / "a.py").write_text(f"from {OLD_PKG} import x\n", encoding="utf-8")
    (exp / "sub" / "b.txt").write_text("keep\nexp001_lighton\n", encoding="utf-8")
    (exp / "bin.dat").write_bytes(b"\0\x01 " + OLD_PKG.encode())
    commit_all(repo)
    tree = git(repo, "rev-parse", f"HEAD:{ra.EXP001_OLD}")
    status_before = git(repo, "status", "--porcelain")

    report: list = []
    predicted = ra.predict_tree(repo, tree, report=report)
    assert git(repo, "status", "--porcelain") == status_before == ""
    assert {r["path"] for r in report} == {"a.py", "sub/b.txt"}

    rename_map.apply(repo, write=True)
    git(repo, "add", "-A")
    applied = git(repo, "rev-parse", f"{git(repo, 'write-tree')}:{ra.EXP001_NEW}")
    assert predicted == applied
    assert (repo / ra.EXP001_NEW / "bin.dat").read_bytes() == b"\0\x01 " + OLD_PKG.encode()


# ------------------------------------------------------------ check_commit_rewrite

MOD_LINES = [f"x{i} = {i}" for i in range(59)]


def _module(old: bool = True) -> str:
    return "\n".join([f"from {OLD_PKG} import helper"] + MOD_LINES) + "\n"


def _build_old(repo: Path) -> None:
    d = repo / OLD_DIR
    (d / "sub").mkdir(parents=True)
    (d / "mod.py").write_text(_module(), encoding="utf-8")
    (d / "__init__.py").write_text(f"from {OLD_PKG}.mod import x0\n", encoding="utf-8")
    (d / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (d / "data.bin").write_bytes(b"\0abc " + OLD_PKG.encode())
    (repo / "README.md").write_text("plain readme\n", encoding="utf-8")
    (repo / "data" / "benchmark" / "acceptance").mkdir(parents=True)
    (repo / ra.COMMIT_EXCEPTIONS[0]).write_text('{"a": 1}\n', encoding="utf-8")
    commit_all(repo, "old")


def _rename(repo: Path, *, target: str = NEW_DIR, tweak=None) -> str:
    git(repo, "mv", OLD_DIR, target)
    d = repo / target
    for f in d.rglob("*.py"):
        f.write_text(rename_map.rewrite_text(f.read_text(encoding="utf-8")), encoding="utf-8")
    if tweak:
        tweak(repo, d)
    return commit_all(repo, "rename")


@pytest.fixture
def old_repo(tmp_path):
    repo = init_repo(tmp_path / "dev")
    _build_old(repo)
    return repo


def _problem_names(res: dict, path: str) -> bool:
    return any(path in p for p in res["problems"])


def test_check_commit_rewrite_passes_moves_and_tiny_files(old_repo):
    c = _rename(old_repo)
    res = ra.check_commit_rewrite(old_repo, c)
    assert res["problems"] == []
    assert res["pure_rewrite"] is True and res["filtered"] is False
    assert res["commit"] == c and res["exceptions"] == list(ra.COMMIT_EXCEPTIONS)
    assert res["latest_only_rewritten"] == [] and res["accepted_latest_only"] == [] and res["latest_only_unproven_in_clone"] == []
    names = ["__init__.py", "data.bin", "mod.py", "sub/__init__.py"]
    assert res["renames"] == [[f"{OLD_DIR}/{n}", f"{NEW_DIR}/{n}"] for n in names]


def test_check_commit_rewrite_exception_file_is_not_checked(old_repo):
    def tweak(repo, d):
        (repo / ra.COMMIT_EXCEPTIONS[0]).write_text('{"a": 2, "anything": true}\n', encoding="utf-8")
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, tweak=tweak))
    assert res["problems"] == [] and res["pure_rewrite"] is True


def test_check_commit_rewrite_wrong_target_dir(old_repo):
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, target=NEW_DIR + "x"))
    assert res["pure_rewrite"] is False
    assert _problem_names(res, f"{OLD_DIR}/mod.py") and _problem_names(res, NEW_DIR + "x/mod.py")


def test_check_commit_rewrite_extra_byte_in_moved_module(old_repo):
    def tweak(repo, d):
        with open(d / "mod.py", "a", encoding="utf-8") as fh:
            fh.write("# extra\n")
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, tweak=tweak))
    assert res["pure_rewrite"] is False and _problem_names(res, f"{NEW_DIR}/mod.py")


def test_check_commit_rewrite_rewritten_binary(old_repo):
    def tweak(repo, d):
        (d / "data.bin").write_bytes(b"\0abc " + NEW_PKG.encode())
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, tweak=tweak))
    assert res["pure_rewrite"] is False and _problem_names(res, f"{NEW_DIR}/data.bin")


def test_check_commit_rewrite_unpaired_deletion(old_repo):
    def tweak(repo, d):
        (d / "data.bin").unlink()
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, tweak=tweak))
    assert res["pure_rewrite"] is False and _problem_names(res, f"{OLD_DIR}/data.bin")


def test_check_commit_rewrite_unpaired_addition(old_repo):
    def tweak(repo, d):
        (d / "new_file.py").write_text("y = 1\n", encoding="utf-8")
    res = ra.check_commit_rewrite(old_repo, _rename(old_repo, tweak=tweak))
    assert res["pure_rewrite"] is False and _problem_names(res, f"{NEW_DIR}/new_file.py")


def test_check_commit_rewrite_in_place_non_name_change(old_repo):
    (old_repo / "README.md").write_text("plain readme!\n", encoding="utf-8")
    git(old_repo, "add", "-A")
    git(old_repo, "commit", "-q", "-m", "touch")
    res = ra.check_commit_rewrite(old_repo, "HEAD")
    assert res["pure_rewrite"] is False and _problem_names(res, "README.md")


def test_check_commit_rewrite_reports_latest_only_in_place_rewrites(old_repo):
    reg = old_repo / "data" / "models" / "REGISTRY.json"
    reg.parent.mkdir(parents=True)
    reg.write_text(_registry(), encoding="utf-8")
    commit_all(old_repo, "registry")
    reg.write_text(rename_map.rewrite_text(_registry()), encoding="utf-8")
    c = commit_all(old_repo, "rewrite registry")
    spec = old_repo.parent / "spec.txt"
    spec.write_text("[history]\nsrc/**\n[latest-only]\ndata/models/REGISTRY.json\n", encoding="utf-8")
    res = ra.check_commit_rewrite(old_repo, c, spec=spec)
    assert res["problems"] == [] and res["pure_rewrite"] is True
    blob = git(old_repo, "rev-parse", "HEAD:data/models/REGISTRY.json")
    assert res["latest_only_rewritten"] == [{"path": "data/models/REGISTRY.json", "new_blob": blob}]


# ---- filtered public history

REG = "data/models/REGISTRY.json"


def _filtered_repo(tmp_path: Path, *, reg_text: str | None = None, commit3_reg: str | None = None, extra_src: bool = False,
                   release_history: bool = True):
    repo = init_repo(tmp_path / "pub")
    spec = tmp_path / "spec.txt"
    spec.write_text("[history]\nsrc/**\n[latest-only]\n" + REG + "\n", encoding="utf-8")
    (repo / OLD_DIR).mkdir(parents=True)
    (repo / OLD_DIR / "mod.py").write_text(_module(), encoding="utf-8")
    commit_all(repo, "old")
    git(repo, "mv", OLD_DIR, NEW_DIR)
    mod = repo / NEW_DIR / "mod.py"
    mod.write_text(rename_map.rewrite_text(mod.read_text(encoding="utf-8")), encoding="utf-8")
    (repo / REG).parent.mkdir(parents=True)
    text = reg_text if reg_text is not None else rename_map.rewrite_text(_registry())
    (repo / REG).write_text(text, encoding="utf-8")
    if extra_src:
        (repo / NEW_DIR / "extra.py").write_text("z = 1\n", encoding="utf-8")
    c2 = commit_all(repo, "rename")
    if commit3_reg is not None:
        (repo / REG).write_text(commit3_reg, encoding="utf-8")
    if release_history:
        (repo / "RELEASE_HISTORY.json").write_text("{}\n", encoding="utf-8")
    commit_all(repo, "release")
    return repo, c2, spec


def _proven(repo: Path):
    return [{"path": REG, "new_blob": git(repo, "rev-parse", f"{ra_head(repo)}:{REG}")}]


def ra_head(repo: Path) -> str:
    return "HEAD"


def test_check_commit_rewrite_filtered_accepts_proven_latest_only(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path)
    blob = git(repo, "rev-parse", f"HEAD:{REG}")
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=_proven(repo))
    assert res["problems"] == []
    assert res["accepted_latest_only"] == [{"path": REG, "blob": blob}]
    assert res["latest_only_unproven_in_clone"] == [REG]
    assert res["pure_rewrite"] == "partial" and res["filtered"] is True


@pytest.mark.parametrize("proven", [None, []])
def test_check_commit_rewrite_filtered_refuses_unrecorded_path(tmp_path, proven):
    repo, c2, spec = _filtered_repo(tmp_path)
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=proven)
    assert res["pure_rewrite"] is False and _problem_names(res, REG)


def test_check_commit_rewrite_filtered_refuses_other_recorded_blob(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path)
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=[{"path": REG, "new_blob": "f" * 40}])
    assert res["pure_rewrite"] is False and _problem_names(res, REG)


def test_check_commit_rewrite_filtered_refuses_blob_that_differs_from_head(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path, commit3_reg=rename_map.rewrite_text(_registry()) + "\n")
    blob2 = git(repo, "rev-parse", f"{c2}:{REG}")
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=[{"path": REG, "new_blob": blob2}])
    assert res["pure_rewrite"] is False and _problem_names(res, REG)


def test_check_commit_rewrite_filtered_refuses_old_name_in_addition(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path, reg_text=_registry())
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=_proven(repo))
    assert res["pure_rewrite"] is False and _problem_names(res, REG)


def test_check_commit_rewrite_filtered_refuses_history_class_addition(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path, extra_src=True)
    res = ra.check_commit_rewrite(repo, c2, spec=spec, proven_latest_only=_proven(repo))
    assert res["pure_rewrite"] is False and _problem_names(res, f"{NEW_DIR}/extra.py")


def test_check_commit_rewrite_unfiltered_reports_the_addition(tmp_path):
    repo, c2, spec = _filtered_repo(tmp_path)
    explicit = ra.check_commit_rewrite(repo, c2, spec=spec, filtered=False, proven_latest_only=_proven(repo))
    assert explicit["pure_rewrite"] is False and _problem_names(explicit, REG)
    repo2, c2b, spec2 = _filtered_repo(tmp_path / "second", release_history=False)
    auto = ra.check_commit_rewrite(repo2, c2b, spec=spec2, proven_latest_only=_proven(repo2))
    assert auto["filtered"] is False and auto["pure_rewrite"] is False and _problem_names(auto, REG)


def test_check_commit_rewrite_filtered_without_additions_is_pure(tmp_path):
    repo = init_repo(tmp_path / "plain")
    (repo / OLD_DIR).mkdir(parents=True)
    (repo / OLD_DIR / "mod.py").write_text(_module(), encoding="utf-8")
    commit_all(repo, "old")
    c = _rename(repo)
    (repo / "RELEASE_HISTORY.json").write_text("{}\n", encoding="utf-8")
    spec = tmp_path / "spec.txt"
    spec.write_text("[history]\nsrc/**\n", encoding="utf-8")
    commit_all(repo, "release")
    res = ra.check_commit_rewrite(repo, c, spec=spec)
    assert res["filtered"] is True and res["pure_rewrite"] is True and res["problems"] == []


def test_constants():
    assert ra.NEW_MODULE == "src/glyphfix_math/acceptance.py"
    assert ra.NEW_TAG == "glyphfix-math-lora-q8:p4-full-1"
    assert len(ra.RENAME_KEYS) == 8
    assert ra.AMENDMENT2_PATH.as_posix() == "evidence/acceptance/FREEZE_AMENDMENT_2.json"


# ---------------------------------------------------------------- verify_rename

DIGEST = "c06474f6789e896e8acbddf0cb91dad88b0a29c563ee58d76bd31d846351e737"
RENAME_SPEC = "[history]\nsrc/**\n[latest-only]\n" + REG + "\n"
GOOD_RERUN = {
    "compare": "identical_modulo_amendment",
    "differing_keys": list(ra.RENAME_KEYS),
    "report_rerender_identical": True,
    "score_exit": 0,
    "cold_cache": True,
}


def real_blob(sha: str) -> bytes:
    return subprocess.run(["git", "cat-file", "blob", sha], cwd=ROOT, capture_output=True, check=True).stdout


def _vr_registry(tag=ra.OLD_TAG, cand=ra.OLD_CAND_TAG) -> str:
    return json.dumps({"entries": [{"tag": tag, "candidate_tag": cand, "ollama_digest": DIGEST}], "schema_version": 1}, indent=2) + "\n"


def _vr_lock(renamed: dict | None = None) -> str:
    arm = {
        "acceptance_cli_blob": ra.CLI_OLD_BLOB,
        "acceptance_module_blob": ra.MODULE_OLD_BLOB,
        "acceptance_cli_blob_generated": ra.CLI_GENERATED_BLOB,
        "freeze_amendment": fa.AMENDMENT_REF,
        "model": ra.OLD_TAG,
        "n_ok": 3,
    }
    lock = {"arms": {"tuned-crops": dict(arm), "tuned-pages": dict(arm)}}
    if renamed:
        lock = ra.amended_lock_rename(lock, cli_new=renamed["cli"], module_new=renamed["module"])
    return json.dumps(lock, indent=2, sort_keys=True) + "\n"


SIDECAR = "data/benchmark/acceptance/tuned_q8.sidecar.jsonl"
NOTE = "extra/note.txt"


def _build_verify_repo(
    tmp_path: Path, *, registry_in_first_commit: bool = True, filtered: bool = False, sidecar: str | None = None, a_extra: bool = False
):
    """Throwaway repository: commit 1 holds the real frozen files, commit 2 is the rename; returns (repo, a1, a2, published)."""
    repo = init_repo(tmp_path / "vr")
    cli_old, mod_old = real_blob(ra.CLI_OLD_BLOB), real_blob(ra.MODULE_OLD_BLOB)
    (repo / "scripts" / "release").mkdir(parents=True)
    (repo / ra.FROZEN_CLI).write_bytes(cli_old)
    (repo / ra.OLD_MODULE).parent.mkdir(parents=True)
    (repo / ra.OLD_MODULE).write_bytes(mod_old)
    (repo / "scripts/release/public_paths.txt").write_text(RENAME_SPEC, encoding="utf-8")
    (repo / ra.COMMIT_EXCEPTIONS[0]).parent.mkdir(parents=True)
    (repo / ra.COMMIT_EXCEPTIONS[0]).write_text(_vr_lock(), encoding="utf-8")
    old_reg = _vr_registry()
    if registry_in_first_commit:
        (repo / REG).parent.mkdir(parents=True)
        (repo / REG).write_text(old_reg, encoding="utf-8")
    sidecar_old = f'{{"model": "{ra.OLD_TAG}"}}\n'
    if sidecar:
        (repo / SIDECAR).parent.mkdir(parents=True, exist_ok=True)
        (repo / SIDECAR).write_text(sidecar_old, encoding="utf-8")
    if a_extra:
        (repo / NOTE).parent.mkdir(parents=True)
        (repo / NOTE).write_text("note\n", encoding="utf-8")
    h1 = commit_all(repo, "old")
    d1 = git(repo, "log", "-1", "--format=%cI")

    cli_new = rename_map.rewrite_text(cli_old.decode()).encode()
    mod_new = rename_map.rewrite_text(mod_old.decode()).encode()
    (repo / ra.FROZEN_CLI).write_bytes(cli_new)
    (repo / ra.NEW_MODULE).parent.mkdir(parents=True, exist_ok=True)
    git(repo, "mv", ra.OLD_MODULE, ra.NEW_MODULE)
    (repo / ra.NEW_MODULE).write_bytes(mod_new)
    blobs = {
        "cli": git(repo, "hash-object", "--no-filters", ra.FROZEN_CLI),
        "module": git(repo, "hash-object", "--no-filters", ra.NEW_MODULE),
    }
    new_reg = ra.renamed_registry_text(old_reg)
    (repo / REG).parent.mkdir(parents=True, exist_ok=True)
    (repo / REG).write_text(new_reg, encoding="utf-8")
    lock_new = _vr_lock(blobs)
    (repo / ra.COMMIT_EXCEPTIONS[0]).write_text(lock_new, encoding="utf-8")
    if sidecar in ("rewritten", "bad"):
        (repo / SIDECAR).write_text(rename_map.rewrite_text(sidecar_old) + ("x" if sidecar == "bad" else ""), encoding="utf-8")
    if a_extra:
        with open(repo / NOTE, "a", encoding="utf-8") as fh:
            fh.write("x")
    h2 = commit_all(repo, "rename")
    d2 = git(repo, "log", "-1", "--format=%cI")
    if filtered:
        (repo / "RELEASE_HISTORY.json").write_text("{}\n", encoding="utf-8")
        commit_all(repo, "release")

    res = ra.check_commit_rewrite(repo, h2)
    a1 = {"old_blob": ra.CLI_GENERATED_BLOB, "new_blob": ra.CLI_OLD_BLOB}
    lock_changes = [
        {"arm": arm, "key": k, "old": None, "new": None}
        for arm in fa.ARMS
        for k in sorted(ra.LOCK_RENAME_KEYS)
    ]
    a2 = {
        "schema": 1,
        "kind": "rename-only",
        "previous": fa.AMENDMENT_PATH.as_posix(),
        "token_map": [list(x) for x in rename_map.TOKEN_MAP],
        "path_moves": [list(x) for x in rename_map.PATH_MOVES],
        "files": {
            "cli": {"old_blob": ra.CLI_OLD_BLOB, "new_blob": blobs["cli"]},
            "module": {"old_blob": ra.MODULE_OLD_BLOB, "new_blob": blobs["module"]},
        },
        "module_pin_commit": [h1, d1],
        "amendment_commit": [h2, d2],
        "run_lock": {
            "path": fa.LOCK_PATH.as_posix(),
            "sha256_after": fa._sha256(repo / ra.COMMIT_EXCEPTIONS[0]),
            "changes": lock_changes,
        },
        "registry": {"path": REG, "changes": ra.registry_changes(old_reg, new_reg)},
        "exp001_tree": {"new_tree": "0" * 40},
        "commit_proof": {
            "commit": res["commit"],
            "changed": res["changed"],
            "pure_rewrite": res["pure_rewrite"],
            "exceptions": res["exceptions"],
            "renames": res["renames"],
            "latest_only_rewritten": res["latest_only_rewritten"],
        },
        "recorded_data_kept": [],
        "kept_unpinned": [],
        "rewritten_unpinned": [],
        "rerun_equivalence": dict(GOOD_RERUN),
    }
    published = {"code_freeze": {"commits": {ra.OLD_MODULE: [h1, d1]}}}
    return repo, a1, a2, published


def _verify(repo, a1, a2, published, **kw):
    return ra.verify_rename(repo, a2, a1=a1, published=published, **kw)


def test_verify_rename_sound_record(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    assert a2["commit_proof"]["latest_only_rewritten"][0]["path"] == REG
    assert _verify(repo, a1, a2, pub) == []


def test_verify_rename_sound_record_after_a_later_release_commit(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path, filtered=True)
    info: dict = {}
    assert _verify(repo, a1, a2, pub, info=info) == []
    assert info["commit_rewrite"]["filtered"] is True and info["commit_rewrite"]["pure_rewrite"] is True


def test_verify_rename_public_clone_reports_partial(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path, registry_in_first_commit=False, filtered=True)
    a2["commit_proof"]["latest_only_rewritten"] = [{"path": REG, "new_blob": git(repo, "rev-parse", f"HEAD:{REG}")}]
    info: dict = {}
    assert _verify(repo, a1, a2, pub, info=info) == []
    res = info["commit_rewrite"]
    assert res["pure_rewrite"] == "partial" and res["latest_only_unproven_in_clone"] == [REG]


def test_verify_rename_public_clone_refuses_unlisted_addition(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path, registry_in_first_commit=False, filtered=True)
    a2["commit_proof"]["latest_only_rewritten"] = []
    assert any(REG in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_module_changed_by_one_byte(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    with open(repo / ra.NEW_MODULE, "a", encoding="utf-8") as fh:
        fh.write("#")
    assert any("module" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_old_module_path_present(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    (repo / ra.OLD_MODULE).write_bytes(real_blob(ra.MODULE_OLD_BLOB))
    assert any("still exists" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_amendment_commit_elsewhere(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    a2["amendment_commit"] = [a2["module_pin_commit"][0], a2["amendment_commit"][1]]
    assert any("commits:" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_second_commit_touching_the_module(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    with open(repo / ra.NEW_MODULE, "a", encoding="utf-8") as fh:
        fh.write("#")
    commit_all(repo, "touch module")
    assert any("exactly the recorded amendment commit" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_run_lock_model_field_changed(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    lock = json.loads((repo / fa.LOCK_PATH).read_text(encoding="utf-8"))
    lock["arms"]["tuned-crops"]["model"] = "other"
    (repo / fa.LOCK_PATH).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert any("RUN_LOCK" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_run_lock_blob_field_changed(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    a1["old_blob"] = "e" * 40
    assert any("acceptance_cli_blob_generated" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_third_registry_change(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    a2["registry"]["changes"].append({"key": "entries[tag=x].other", "old": 1, "new": 2})
    assert any("exactly the tag and candidate_tag" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_registry_head_value_differs(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    (repo / REG).write_text(_vr_registry(tag=ra.NEW_TAG, cand="something-else:p4-full-1"), encoding="utf-8")
    assert any("candidate_tag" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_registry_old_tag_remains(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    (repo / REG).write_text(_vr_registry(), encoding="utf-8")
    assert any("registry" in p for p in _verify(repo, a1, a2, pub))


@pytest.mark.parametrize("how", ["removed", "blob-changed"])
def test_verify_rename_latest_only_list_differs_from_recomputed(tmp_path, how):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    if how == "removed":
        a2["commit_proof"]["latest_only_rewritten"] = []
    else:
        a2["commit_proof"]["latest_only_rewritten"][0]["new_blob"] = "f" * 40
    assert any("latest_only_rewritten" in p for p in _verify(repo, a1, a2, pub))


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda a: a.__setitem__("kind", "other"), "kind"),
        (lambda a: a.__setitem__("token_map", []), "token_map"),
        (lambda a: a.__setitem__("path_moves", []), "path_moves"),
        (lambda a: a["rerun_equivalence"].__setitem__("compare", "different"), "rerun"),
        (lambda a: a["rerun_equivalence"].__setitem__("differing_keys", []), "rerun"),
        (lambda a: a["rerun_equivalence"].__setitem__("cold_cache", False), "rerun"),
        (lambda a: a["rerun_equivalence"].__setitem__("report_rerender_identical", False), "rerun"),
        (lambda a: a["files"]["cli"].__setitem__("new_blob", "a" * 40), "cli"),
        (lambda a: a["module_pin_commit"].__setitem__(1, "other-date"), "commits"),
        (lambda a: a["commit_proof"].__setitem__("exceptions", []), "exceptions"),
    ],
)
def test_verify_rename_rejects_record_changes(tmp_path, mutate, needle):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    mutate(a2)
    assert any(needle in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_recorded_data(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    f = repo / "data" / "kept.txt"
    f.write_text("kept\n", encoding="utf-8")
    a2["recorded_data_kept"] = [{"path": "data/kept.txt", "sha256": fa._sha256(f)}]
    a2["kept_unpinned"] = [{"path": "data/gone.txt", "sha256": "0" * 64, "reason": "absent paths are skipped"}]
    assert _verify(repo, a1, a2, pub) == []
    f.write_text("changed\n", encoding="utf-8")
    assert any("data/kept.txt" in p for p in _verify(repo, a1, a2, pub))
    f.write_text("kept\n", encoding="utf-8")
    a2["rewritten_unpinned"] = [{"path": "data/gone.txt", "sha256_before": "0" * 64, "sha256_after": "1" * 64}]
    assert any("both" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_rewritten_unpinned_must_hold_no_old_name(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    f = repo / "data" / "side.jsonl"
    f.write_text(f"{NEW_PKG}\n", encoding="utf-8")
    a2["rewritten_unpinned"] = [{"path": "data/side.jsonl", "sha256_before": "0" * 64, "sha256_after": fa._sha256(f)}]
    assert _verify(repo, a1, a2, pub) == []
    f.write_text(f"{OLD_PKG}\n", encoding="utf-8")
    a2["rewritten_unpinned"][0]["sha256_after"] = fa._sha256(f)
    assert any("old name" in p for p in _verify(repo, a1, a2, pub))


def test_verify_rename_exp001_tree(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    (repo / ra.EXP001_NEW).mkdir(parents=True)
    (repo / ra.EXP001_NEW / "a.txt").write_text("x\n", encoding="utf-8")
    (repo / ra.EXP001_TEST).parent.mkdir(parents=True, exist_ok=True)
    (repo / ra.EXP001_TEST).write_text('EXP001_TREE = "0000"\n', encoding="utf-8")
    commit_all(repo, "exp001")
    tree = git(repo, "rev-parse", f"HEAD:{ra.EXP001_NEW}")
    a2["exp001_tree"]["new_tree"] = tree
    assert any("literal" in p for p in _verify(repo, a1, a2, pub))
    (repo / ra.EXP001_TEST).write_text(f'EXP001_TREE = "{tree}"\n', encoding="utf-8")
    assert _verify(repo, a1, a2, pub) == []
    a2["exp001_tree"]["new_tree"] = "1" * 40
    assert any("exp001" in p for p in _verify(repo, a1, a2, pub))


def test_verify_chain_returns_rename_problems_in_a_throwaway_repository(tmp_path):
    repo, a1, a2, pub = _build_verify_repo(tmp_path)
    assert fa.verify_chain(repo, published=pub, commit_map={}) != []  # amendment 1 files are absent here
    (repo / fa.AMENDMENT_PATH).parent.mkdir(parents=True)
    (repo / fa.AMENDMENT_PATH).write_text(json.dumps(a1), encoding="utf-8")
    (repo / ra.AMENDMENT2_PATH).write_text(json.dumps(a2), encoding="utf-8")
    problems = fa.verify_chain(repo, published=pub, commit_map={})
    assert not any(p.startswith("cannot load") for p in problems)
    a2["kind"] = "other"
    (repo / ra.AMENDMENT2_PATH).write_text(json.dumps(a2), encoding="utf-8")
    assert any("kind" in p for p in fa.verify_chain(repo, published=pub, commit_map={}))


def test_verify_chain_before_the_rename_is_amendment_1_only(tmp_path):
    assert fa.verify_chain(tmp_path, published={}, commit_map={}) != []
    assert not (tmp_path / ra.AMENDMENT2_PATH).exists()


# ----------------------------------------------------------------- record

PIN = "2026-10-17T10:09:32+02:00"
PRED = "data/benchmark/acceptance/tuned_q8.pred.jsonl"
SCHEMA_KEYS = {
    "schema", "kind", "previous", "reason", "approved", "token_map", "path_moves", "files", "module_pin_commit",
    "previous_amendment_commit", "amendment_commit", "tags", "run_lock", "registry", "exp001_tree", "commit_proof",
    "recorded_data_kept", "kept_unpinned", "rewritten_unpinned", "rerun_equivalence", "effects",
}


def fake_compare(published, fresh):
    return "different: " + ", ".join(ra.RENAME_KEYS)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_setup(tmp_path: Path, **kw):
    """Throwaway repository after the rename commit, plus a preview directory and a score-only rerun directory."""
    repo, a1, a2, _ = _build_verify_repo(tmp_path, **kw)
    h1, d1 = a2["module_pin_commit"]
    h2, d2 = a2["amendment_commit"]
    a1 = {**a1, "old_commit": ["c" * 40, PIN], "amendment_commit": ["a" * 40, "2026-10-18T18:34:42+02:00"]}
    cli_new, mod_new = a2["files"]["cli"]["new_blob"], a2["files"]["module"]["new_blob"]
    published = {
        "headline": {"exact": 117},
        "tag": ra.OLD_TAG,
        "code_freeze": {
            "commits": {ra.FROZEN_CLI: list(a1["old_commit"]), ra.OLD_MODULE: [h1, d1]},
            "current_blobs": {"cli": a1["old_blob"], "module": ra.MODULE_OLD_BLOB},
            "lock_blobs": {"cli": a1["old_blob"], "module": ra.MODULE_OLD_BLOB},
        },
    }
    fresh = copy.deepcopy(published)
    fresh["tag"] = ra.NEW_TAG
    fcf = fresh["code_freeze"]
    fcf["commits"] = {ra.FROZEN_CLI: [h2, d2], ra.NEW_MODULE: [h1, d1]}
    fcf["current_blobs"] = {"cli": cli_new, "module": mod_new}
    fcf["lock_blobs"] = {"cli": cli_new, "module": mod_new}

    acc = repo / "evidence" / "acceptance"
    _dump(acc / "results.json", published)
    (acc / "ACCEPTANCE.md").write_text("# acceptance\n", encoding="utf-8")
    _dump(acc / "FREEZE_AMENDMENT.json", a1)
    _dump(acc / "env_check.json", {"python": "3"})
    (repo / PRED).parent.mkdir(parents=True, exist_ok=True)
    (repo / PRED).write_text(f'{{"model": "{ra.OLD_TAG}"}}\n', encoding="utf-8")
    (repo / ra.EXP001_NEW).mkdir(parents=True)
    (repo / ra.EXP001_NEW / "a.txt").write_text("x\n", encoding="utf-8")
    commit_all(repo, "experiments")
    tree = git(repo, "rev-parse", f"HEAD:{ra.EXP001_NEW}")
    (repo / ra.EXP001_TEST).parent.mkdir(parents=True, exist_ok=True)
    (repo / ra.EXP001_TEST).write_text(f'EXP001_TREE = "{tree}"\n', encoding="utf-8")
    commit_all(repo, "record inputs")

    preview = tmp_path / "preview"
    _dump(preview / "blobs.json", {"cli_old": ra.CLI_OLD_BLOB, "cli_new": cli_new, "module_old": ra.MODULE_OLD_BLOB, "module_new": mod_new})
    (preview / "RUN_LOCK.json").write_bytes((repo / fa.LOCK_PATH).read_bytes())
    (preview / "REGISTRY.json").write_bytes((repo / REG).read_bytes())
    _dump(preview / "exp001.json", {"old_tree": ra.EXP001_OLD_TREE, "predicted_new_tree": tree, "files": []})
    rerun = tmp_path / "rerun"
    _dump(rerun / "evidence" / "results.json", fresh)
    (rerun / "ACCEPTANCE.md").write_bytes((acc / "ACCEPTANCE.md").read_bytes())
    report = {
        "tier": "score-only", "cold_cache": True, "render_image": "texlive/texlive:latest", "git_head": "f" * 40,
        "score_exit": 0, "seconds": 12, "harness": "scripts.release.frozen_rescore",
        "pin_substitution": {"path": ra.OLD_MODULE, "commit": h1, "date": d1}, "compare": "not_run: score-only",
    }
    _dump(rerun / "reproduce_report.json", report)
    return SimpleNamespace(
        repo=repo, a1=a1, published=published, fresh=fresh, preview=preview, rerun=rerun, report=report, commit=h2, tree=tree
    )


def _check(s, **over):
    kw = {"preview_dir": s.preview, "amendment_commit": s.commit, "approved": "approved (test)", "compare_runner": fake_compare}
    kw.update(over)
    return ra.check_record_inputs(s.repo, s.rerun, **kw)


def _report(s, **changes):
    _dump(s.rerun / "reproduce_report.json", {**s.report, **changes})


def test_check_record_inputs_valid_setup_has_no_problem(tmp_path):
    s = _record_setup(tmp_path)
    problems, draft = _check(s)
    assert problems == []
    res = draft["commit_rewrite"]
    assert res["pure_rewrite"] is True and res["filtered"] is False
    assert res["latest_only_rewritten"] == [{"path": REG, "new_blob": git(s.repo, "rev-parse", f"{s.commit}:{REG}")}]


def test_check_record_inputs_writes_nothing(tmp_path):
    s = _record_setup(tmp_path)
    before = git(s.repo, "status", "--porcelain")
    _check(s)
    _check(s, approved="")
    assert git(s.repo, "status", "--porcelain") == before == ""


def test_check_record_inputs_default_compare_runner_is_the_frozen_cli(tmp_path):
    _dump(tmp_path / "a.json", {"x": 1})
    _dump(tmp_path / "b.json", {"x": 2})
    assert ra._run_compare(ROOT, tmp_path / "a.json", tmp_path / "b.json") == "different: x"


def _mut_fresh_headline(s):
    fresh = json.loads((s.rerun / "evidence" / "results.json").read_text(encoding="utf-8"))
    fresh["headline"]["exact"] = 116
    _dump(s.rerun / "evidence" / "results.json", fresh)


def _mut_acceptance_md(s):
    with open(s.rerun / "ACCEPTANCE.md", "a", encoding="utf-8") as fh:
        fh.write("x")


def _mut_module(s):
    with open(s.repo / ra.NEW_MODULE, "a", encoding="utf-8") as fh:
        fh.write("#")


def _mut_run_lock(s):
    with open(s.repo / fa.LOCK_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n")


def _mut_preview_registry(s):
    with open(s.preview / "REGISTRY.json", "a", encoding="utf-8") as fh:
        fh.write("\n")


def _mut_exp_tree(s):
    _dump(s.preview / "exp001.json", {"predicted_new_tree": "1" * 40})


def _mut_release_history(s):
    (s.repo / "RELEASE_HISTORY.json").write_text("{}\n", encoding="utf-8")
    commit_all(s.repo, "release")


def _mut_no_public_paths(s):
    git(s.repo, "rm", "-q", "scripts/release/public_paths.txt")
    commit_all(s.repo, "drop spec")


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda s: _report(s, score_exit=1), "score_exit"),
        (lambda s: _report(s, cold_cache=False), "not cold"),
        (lambda s: _report(s, harness="scripts.acceptance"), "harness"),
        (_mut_fresh_headline, "not identical_modulo_amendment"),
        (_mut_acceptance_md, "ACCEPTANCE.md"),
        (_mut_module, "preview module blob"),
        (_mut_run_lock, "RUN_LOCK.json"),
        (_mut_preview_registry, "REGISTRY.json"),
        (_mut_exp_tree, "experiments tree"),
        (_mut_release_history, "RELEASE_HISTORY.json"),
        (_mut_no_public_paths, "public_paths.txt"),
    ],
)
def test_check_record_inputs_refuses(tmp_path, mutate, needle):
    s = _record_setup(tmp_path)
    mutate(s)
    problems, _ = _check(s)
    assert any(needle in p for p in problems), problems


def test_check_record_inputs_refuses_a_different_frozen_compare_output(tmp_path):
    s = _record_setup(tmp_path)
    problems, _ = _check(s, compare_runner=lambda p, f: "different: tag")
    assert any("frozen compare-results" in p for p in problems), problems


def test_check_record_inputs_refuses_empty_approval(tmp_path):
    s = _record_setup(tmp_path)
    assert any("approved" in p for p in _check(s, approved="  ")[0])


def test_check_record_inputs_refuses_a_commit_that_changes_a_non_name_byte(tmp_path):
    s = _record_setup(tmp_path, a_extra=True)
    problems, _ = _check(s)
    assert any("commit rewrite" in p and NOTE in p for p in problems), problems


def test_check_record_inputs_refuses_a_rewritten_unpinned_sidecar_that_is_not_rewrite_text(tmp_path):
    s = _record_setup(tmp_path, sidecar="bad")
    problems, _ = _check(s)
    assert any("rewritten unpinned file is not rewrite_text of its parent" in p and SIDECAR in p for p in problems), problems


def test_check_record_inputs_refuses_an_unknown_commit(tmp_path):
    s = _record_setup(tmp_path)
    problems, draft = _check(s, amendment_commit="0" * 40)
    assert any("amendment commit" in p for p in problems) and draft == {}


# ------------------------------------------------------------------ build_record

def _build(s, **kw):
    problems, draft = _check(s)
    assert problems == []
    return ra.build_record(s.repo, draft, approved="approved (test)", amendment_commit=s.commit), draft


def test_build_record_has_exactly_the_schema_keys_and_the_proof_of_the_commit(tmp_path):
    s = _record_setup(tmp_path)
    rec, draft = _build(s)
    assert set(rec) == SCHEMA_KEYS and rec["schema"] == 1 and rec["kind"] == "rename-only"
    assert rec["amendment_commit"] == [s.commit, git(s.repo, "log", "-1", "--format=%cI", s.commit)]
    assert rec["commit_proof"] == {k: draft["commit_rewrite"][k] for k in ra.COMMIT_PROOF_KEYS}
    assert rec["commit_proof"]["pure_rewrite"] is True
    assert rec["commit_proof"]["latest_only_rewritten"][0]["path"] == REG
    assert rec["tags"] == [
        {"old": ra.OLD_TAG, "new": ra.NEW_TAG, "ollama_digest": DIGEST},
        {"old": ra.OLD_CAND_TAG, "new": ra.NEW_CAND_TAG, "ollama_digest": None},
    ]
    assert rec["files"]["cli"]["proof"]["changed_lines"] and rec["files"]["module"]["proof"]["changed_lines"]
    assert rec["module_pin_commit"] == s.published["code_freeze"]["commits"][ra.OLD_MODULE]
    assert rec["previous_amendment_commit"] == s.a1["amendment_commit"]
    assert len(rec["effects"]) == 4 and "frozen_rescore" in rec["effects"][0]
    assert {c["key"] for c in rec["run_lock"]["changes"]} == set(ra.LOCK_RENAME_KEYS)
    assert [c["new"] for c in rec["registry"]["changes"] if c["key"].endswith("candidate_tag")] == [ra.NEW_CAND_TAG]
    re = rec["rerun_equivalence"]
    assert re["compare"] == "identical_modulo_amendment" and re["differing_keys"] == list(ra.RENAME_KEYS)
    assert re["frozen_compare_output"] == "different: " + ", ".join(ra.RENAME_KEYS)
    assert re["published_sha256"] == fa._sha256(s.repo / "evidence/acceptance/results.json")
    assert re["fresh_sha256"] == fa._sha256(s.rerun / "evidence/results.json")
    assert re["report_rerender_identical"] is True and re["harness"] == "scripts.release.frozen_rescore"
    json.dumps(rec)


def test_build_record_kept_data_lists(tmp_path):
    s = _record_setup(tmp_path, sidecar="kept")
    rec, _ = _build(s)
    paths = [e["path"] for e in rec["recorded_data_kept"]]
    assert paths == sorted([PRED, "evidence/acceptance/env_check.json", "evidence/acceptance/results.json"])
    assert [e["path"] for e in rec["kept_unpinned"]] == [SIDECAR] and rec["rewritten_unpinned"] == []
    assert rec["kept_unpinned"][0]["sha256"] == fa._sha256(s.repo / SIDECAR)


def test_build_record_rewritten_sidecar_moves_to_rewritten_unpinned(tmp_path):
    s = _record_setup(tmp_path, sidecar="rewritten")
    rec, _ = _build(s)
    assert rec["kept_unpinned"] == []
    (entry,) = rec["rewritten_unpinned"]
    assert entry["path"] == SIDECAR and entry["sha256_after"] == fa._sha256(s.repo / SIDECAR)
    assert entry["sha256_before"] != entry["sha256_after"]
    assert SIDECAR not in [e["path"] for e in rec["recorded_data_kept"]]


def test_build_record_default_has_no_unpinned_entries(tmp_path):
    rec, _ = _build(_record_setup(tmp_path))
    assert rec["kept_unpinned"] == [] and rec["rewritten_unpinned"] == []


@pytest.mark.parametrize("sidecar", [None, "kept", "rewritten"])
def test_a_built_record_passes_verify_rename(tmp_path, sidecar):
    s = _record_setup(tmp_path, sidecar=sidecar)
    rec, _ = _build(s)
    assert ra.verify_rename(s.repo, rec, a1=s.a1, published=s.published) == []


def test_build_record_refuses_another_commit(tmp_path):
    s = _record_setup(tmp_path)
    _, draft = _check(s)
    with pytest.raises(ra.AmendmentError):
        ra.build_record(s.repo, draft, approved="x", amendment_commit="0" * 7)


# ------------------------------------------------------------------ record CLI

def _record_cli(s, monkeypatch, *extra):
    monkeypatch.chdir(s.repo)
    monkeypatch.setattr(ra, "_run_compare", lambda root, p, f: fake_compare(p, f))
    return ra.main(["record", "--rerun", str(s.rerun), "--approved", "approved (test)", "--amendment-commit", s.commit,
                    "--preview", str(s.preview), *extra])


def test_record_cli_writes_only_the_record_and_never_overwrites(tmp_path, monkeypatch, capsys):
    s = _record_setup(tmp_path)
    assert _record_cli(s, monkeypatch) == 0
    assert git(s.repo, "status", "--porcelain") == f"?? {ra.AMENDMENT2_REF}"
    first = (s.repo / ra.AMENDMENT2_PATH).read_bytes()
    assert set(json.loads(first)) == SCHEMA_KEYS
    capsys.readouterr()
    assert _record_cli(s, monkeypatch) == 2
    assert "record exists" in capsys.readouterr().err
    assert (s.repo / ra.AMENDMENT2_PATH).read_bytes() == first


def test_record_cli_refuses_and_writes_nothing(tmp_path, monkeypatch, capsys):
    s = _record_setup(tmp_path)
    _report(s, score_exit=1)
    assert _record_cli(s, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "score_exit" in err
    assert not (s.repo / ra.AMENDMENT2_PATH).exists()
    assert git(s.repo, "status", "--porcelain") == ""


def test_record_help_lists_its_options(capsys):
    with pytest.raises(SystemExit):
        ra.main(["record", "--help"])
    out = capsys.readouterr().out
    assert all(o in out for o in ("--rerun", "--approved", "--amendment-commit", "--preview"))


def test_verify_rename_real_repository():
    if not (ROOT / ra.AMENDMENT2_PATH).exists():
        pytest.skip("record not written yet")
    assert fa.verify_chain(ROOT) == []
