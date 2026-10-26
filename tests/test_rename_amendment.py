"""Tests of the rename amendment proof core: names-only proof, record edits, tree prediction, commit check.

Nothing here reads the worktree's frozen files, RUN_LOCK.json or REGISTRY.json: real frozen texts come from git
objects (they stay reachable after the rename) and the lock and registry are synthetic fixtures.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

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
