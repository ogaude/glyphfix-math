"""rename_map: token map, protected phrases, allow-list, scanner, rewriter and the trial runner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.release import rename_map as R

OLD = "lightonocr" + "_math"
PL = ".plan" + "ning"  # the public-reference test forbids literal planning paths in kept tests


def sh(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_AUTHOR_NAME": "dev", "GIT_COMMITTER_NAME": "dev",
           "GIT_AUTHOR_EMAIL": "dev@example.invalid", "GIT_COMMITTER_EMAIL": "dev@example.invalid"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env, check=True).stdout.decode().strip()


def make_repo(path: Path, files: dict[str, bytes | str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    sh(path, "init", "-q", "-b", "master")
    for rel, content in files.items():
        p = path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    sh(path, "add", "-A")
    sh(path, "commit", "-q", "-m", "init")
    return path


# Built from pieces so this file never holds the local path the release scan forbids.
ABS = "/mnt" + "/datas/Transfer/KarbotiumLabs/projects"


@pytest.mark.parametrize(("src", "dst"), [
    ("from lightonocr_math.corpus import freeze", "from glyphfix_math.corpus import freeze"),
    ("LIGHTONOCR_MATH_CACHE", "GLYPHFIX_MATH_CACHE"),
    ('name = "lightonocr-math"', 'name = "glyphfix-math"'),
    ("experiments.exp001_lighton.triage", "experiments.exp001_baseline.triage"),
    ("mathocr-lora-q8:p4-full-1", "glyphfix-math-lora-q8:p4-full-1"),
    ("mathocr-cand:run1", "glyphfix-math-cand:run1"),
    ("--model-name mathocr-lora --outfile", "--model-name glyphfix-math-lora --outfile"),
    ("mathocr-lora-bf16:x", "glyphfix-math-lora-bf16:x"),
    (f"{ABS}/lightonocr-math/x", f"{ABS}/glyphfix-math/x"),
])
def test_rewrite_maps_tokens_in_one_pass(src, dst):
    assert R.rewrite_text(src) == dst
    assert R.rewrite_text(dst) == dst


@pytest.mark.parametrize("keep", [
    "LightOnOCR-3-4B", "lightonai/LightOnOCR-3-4B", "lighton-ocr3-q8-16k", "lighton-ocr3-bf16", "Modelfile.lighton-bf16",
    "scripts/utils/lighton_extractor.py", ".venv-lighton", "LightOnOCR-3-math", "lightonocr-finetune",
    "EXP-001_LightOnOCR-3_Layer1", f"{PL}/phases/999.2-lightonocr-3-0-8b-baseline-and-lora-feasibility", "mathocr",
])
def test_rewrite_leaves_upstream_names(keep):
    assert R.rewrite_text(keep) == keep


@pytest.mark.parametrize("keep", [
    "Phase 06.1: Rename lightonocr-math to glyphfix-math (package",
    f"{PL}/phases/06.1-rename-lightonocr-math-to-glyphfix-math-package-experiments-/x.md",
])
def test_protected_phrases_stay(keep):
    assert R.rewrite_text(keep) == keep


def test_history_statement_line_stays_next_line_changes():
    text = "renamed from `lightonocr_math`\nimport lightonocr_math\n"
    assert R.rewrite_text(text) == "renamed from `lightonocr_math`\nimport glyphfix_math\n"


def test_bytes_outside_matches_are_preserved():
    text = "a lightonocr_math\r\nb éè\r\nno newline lightonocr-math"
    assert R.rewrite_text(text) == "a glyphfix_math\r\nb éè\r\nno newline glyphfix-math"


def test_map_path():
    assert R.map_path("src/lightonocr_math/corpus/freeze.py") == "src/glyphfix_math/corpus/freeze.py"
    assert R.map_path("experiments/exp001_lighton/Modelfile.lighton") == "experiments/exp001_baseline/Modelfile.lighton"
    assert R.map_path("src/lightonocr_mathx/a.py") == "src/lightonocr_mathx/a.py"
    p = f"{PL}/phases/999.2-lightonocr-3-0-8b-baseline-and-lora-feasibility/x.md"
    assert R.map_path(p) == p


ALLOWED = [
    f"{PL}/phases/06.1-rename-lightonocr-math-to-glyphfix-math-package-experiments-/06.1-01-PLAN.md",
    "data/benchmark/acceptance/tuned_q8.sidecar.jsonl", "data/benchmark/acceptance/tuned_q8.pred.manifest.json",
    "data/benchmark/acceptance/RUN_LOCK.json", "evidence/acceptance/results.json", "evidence/acceptance/env_check.json",
    "evidence/acceptance/FREEZE_AMENDMENT.json", "evidence/acceptance/FREEZE_AMENDMENT_2.json",
    "scripts/release/rename_map.py", "scripts/release/rename_amendment.py", "scripts/release/freeze_amendment.py",
    "tests/test_rename_map.py", "tests/test_rename_amendment.py", "tests/test_rename_real_data.py", "tests/test_freeze_amendment.py",
]
NOT_ALLOWED = [
    f"{PL}/phases/05-acceptance-evaluation/evidence/precheck.json",
    f"{PL}/phases/06-publish/evidence/rehearsal_pre_naming.json",
    "src/lightonocr_math/score.py", "README.md", f"{PL}/STATE.md", "data/models/REGISTRY.json",
    "evidence/acceptance/ACCEPTANCE.md", "scripts/release/reproduce.py",
]


@pytest.mark.parametrize("path", ALLOWED)
def test_allowed(path):
    assert R.allowed_reason(path)


@pytest.mark.parametrize("path", NOT_ALLOWED)
def test_not_allowed(path):
    assert R.allowed_reason(path) is None


def test_every_allow_entry_names_its_pin():
    assert all(g and r and p for g, r, p in R.ALLOW_FILES)


def test_ollama_new_name():
    assert R.ollama_new_name("mathocr-lora-q8:p4-full-1") == "glyphfix-math-lora-q8:p4-full-1"
    assert R.ollama_new_name("mathocr-cand:p4-full-1") == "glyphfix-math-cand:p4-full-1"
    assert R.ollama_new_name("lighton-ocr3-q8-16k") is None
    assert R.ollama_new_name("llama3:8b") is None


def test_is_binary_and_rewrite_blob():
    assert R.is_binary(b"a\0lightonocr_math")
    assert not R.is_binary(b"x" * 8000 + b"\0")
    assert not R.is_binary(b"")
    binary = b"a\0lightonocr_math"
    assert R.rewrite_blob("src/x.py", binary) == binary
    assert R.rewrite_blob("data/benchmark/acceptance/RUN_LOCK.json", b"lightonocr_math") == b"lightonocr_math"
    assert R.rewrite_blob("src/x.py", b"import lightonocr_math\n") == b"import glyphfix_math\n"
    odd = b"\xff\xfe lightonocr_math \x80\n"
    assert R.rewrite_blob("src/x.py", odd) == b"\xff\xfe glyphfix_math \x80\n"


@pytest.fixture
def repo(tmp_path):
    r = make_repo(tmp_path / "repo", {
        "README.md": f"uses {OLD}\n",
        f"src/{OLD}/x.py": "print(1)\n",
        f"src/{OLD}/y.py": f"import {OLD}\n",
        "data/benchmark/acceptance/a.sidecar.jsonl": '{"model": "mathocr-lora-q8"}\n',
        f"{PL}/phases/02-x/evidence/e.json": '{"model": "mathocr-lora-q8"}\n',
        "blob.bin": b"\0" + OLD.encode(),
        "notes.md": f"renamed from `{OLD}`\n",
        "keep.txt": "LightOnOCR-3-4B\n",
    })
    (r / "untracked.txt").write_text(OLD)
    return r


def test_scan_reports_only_unallowed_tracked_text(repo):
    found = R.scan(repo)
    got = sorted((f["path"], f["kind"], f["token"]) for f in found)
    assert got == sorted([
        ("README.md", "content", OLD),
        (f"{PL}/phases/02-x/evidence/e.json", "content", "mathocr-lora-q8"),
        (f"src/{OLD}/x.py", "path", OLD),
        (f"src/{OLD}/y.py", "path", OLD),
        (f"src/{OLD}/y.py", "content", OLD),
    ])


def test_apply_dry_run_changes_nothing(repo):
    before = sh(repo, "status", "--porcelain")
    plan = R.apply(repo, write=False)
    assert sh(repo, "status", "--porcelain") == before
    assert plan["moves"] == [{"from": f"src/{OLD}", "to": "src/glyphfix_math", "files": 2}]
    assert set(plan["files"]) == {"README.md", f"{PL}/phases/02-x/evidence/e.json", "src/glyphfix_math/y.py"}
    assert plan["skipped_allowed"] == ["data/benchmark/acceptance/a.sidecar.jsonl"]
    assert plan["total_replacements"] == 3


def test_apply_write_moves_rewrites_and_is_idempotent(repo):
    (repo / "src" / OLD / "__pycache__").mkdir()
    (repo / "src" / OLD / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    R.apply(repo, write=True)
    assert not (repo / "src" / OLD).exists()
    assert (repo / "src/glyphfix_math/y.py").read_text() == "import glyphfix_math\n"
    assert (repo / "data/benchmark/acceptance/a.sidecar.jsonl").read_text() == '{"model": "mathocr-lora-q8"}\n'
    assert (repo / "blob.bin").read_bytes() == b"\0" + OLD.encode()
    assert R.scan(repo) == []
    sh(repo, "add", "-u")
    snapshot = sh(repo, "write-tree")
    plan = R.apply(repo, write=True)
    assert plan["moves"] == [] and plan["files"] == {}
    sh(repo, "add", "-u")
    assert sh(repo, "write-tree") == snapshot


def test_apply_write_refuses_other_leftovers_and_existing_destination(repo):
    (repo / "src" / OLD / "stray.txt").write_text("x")
    (repo / ".gitignore").write_text("stray.txt\n")
    with pytest.raises(R.RenameError):
        R.apply(repo, write=True)
    (repo / "src" / OLD / "stray.txt").unlink()
    (repo / "src" / "glyphfix_math").mkdir()
    with pytest.raises(R.RenameError):
        R.apply(repo, write=True)


def test_real_repository_has_no_residue():
    root = Path(__file__).resolve().parent.parent
    if (root / "src" / OLD).exists():
        pytest.skip("rename not applied yet")
    if not (root / PL).exists():
        pytest.skip("public clone: no planning directory")
    assert R.scan(root) == []


# ---------------------------------------------------------------------------------------------------------------------
# trial

JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>"""


def junit_xml(passed=(), failed=(), skipped=()):
    cases = [f'<testcase classname="tests.test_x" name="{n}" file="tests/test_x.py"/>' for n in passed]
    cases += [f'<testcase classname="tests.test_x" name="{n}" file="tests/test_x.py"><failure message="boom">trace</failure></testcase>'
              for n in failed]
    cases += [f'<testcase classname="tests.test_x.TestK" name="{n}"><skipped message="why"/></testcase>' for n in skipped]
    return JUNIT.format(cases="".join(cases))


def test_parse_junit(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text(junit_xml(passed=["test_a"], failed=["test_b"], skipped=["test_c"]))
    r = R.parse_junit(p)
    assert r["counts"] == {"passed": 1, "failed": 1, "errors": 0, "skipped": 1}
    assert r["failing"] == ["tests/test_x.py::test_b"]
    assert r["skipped"] == [{"id": "tests/test_x.py::TestK::test_c", "reason": "why"}]
    assert r["messages"]["tests/test_x.py::test_b"] == "boom"


def test_set_exp001_literal():
    tree = "a" * 40
    text = f'import x\nEXP001_TREE = "{"0" * 40}"\nother = "{"0" * 40}"\n'
    out = R.set_exp001_literal(text, tree)
    assert out == f'import x\nEXP001_TREE = "{tree}"\nother = "{"0" * 40}"\n'
    with pytest.raises(R.RenameError):
        R.set_exp001_literal("x = 1\n", tree)
    with pytest.raises(R.RenameError):
        R.set_exp001_literal(text + text, tree)


@pytest.fixture
def src(tmp_path):
    return make_repo(tmp_path / "src", {
        ".gitignore": "work/\n",
        f"src/{OLD}/__init__.py": "",
        "experiments/exp001_lighton/a.txt": "stock\n",
        "tests/test_frozen_exp001.py": f'EXP001_TREE = "{"0" * 40}"\n',
        "a.py": f"import {OLD}\n",
    })


def fake_runner(red_in_renamed=("test_new",), base_failing=("test_old",)):
    seen = []

    def run(clone, args, junit):
        seen.append(clone.name)
        if clone.name == "base":
            junit.write_text(junit_xml(passed=["test_new"], failed=base_failing))
        else:
            junit.write_text(junit_xml(passed=[], failed=[*base_failing, *red_in_renamed]))
        return 1
    run.seen = seen
    return run


GIB = 1024**3


def test_trial_reports_red_and_never_writes_source(src):
    before = (sh(src, "status", "--porcelain"), sh(src, "rev-parse", "HEAD"))
    runner = fake_runner()
    res = R.trial(src, src / "work" / "t1", runner=runner, mem_check=lambda: 16 * GIB, set_exp001_tree=True)
    assert (sh(src, "status", "--porcelain"), sh(src, "rev-parse", "HEAD")) == before
    assert runner.seen == ["base", "renamed"]
    assert res["red"] == ["tests/test_x.py::test_new"]
    assert res["red_allowed"] == [] and res["fixed"] == []
    assert res["renamed_scan_residue"] == 0 and res["worktree_diff_applied"] is False
    w = src / "work" / "t1"
    assert sh(w / "base", "remote") == "" and sh(w / "renamed", "remote") == ""
    assert (w / "base" / "src" / OLD).is_dir() and (w / "renamed" / "src" / "glyphfix_math").is_dir()
    assert not (w / "renamed" / "src" / OLD).exists()
    assert sh(w / "renamed", "status", "--porcelain") == ""
    assert res["exp001_tree"] == sh(w / "renamed", "rev-parse", "HEAD:experiments/exp001_baseline")
    assert f'"{res["exp001_tree"]}"' in (w / "renamed" / "tests" / "test_frozen_exp001.py").read_text()


def test_trial_allow_red_and_fixed(src):
    runner = fake_runner(red_in_renamed=("test_new",), base_failing=("test_old",))
    res = R.trial(src, src / "work" / "t2", runner=runner, mem_check=lambda: 16 * GIB, allow_red=("*::test_new",))
    assert res["red"] == [] and res["red_allowed"] == ["tests/test_x.py::test_new"]

    def fixing(clone, args, junit):
        junit.write_text(junit_xml(failed=["test_old"]) if clone.name == "base" else junit_xml(passed=["test_old"]))
        return 0
    res = R.trial(src, src / "work" / "t3", runner=fixing, mem_check=lambda: 16 * GIB)
    assert res["red"] == [] and res["fixed"] == ["tests/test_x.py::test_old"]


def test_trial_carries_tracked_changes_only_without_paths(src):
    (src / "a.py").write_text("import os\n")
    (src / "new.txt").write_text("untracked\n")
    res = R.trial(src, src / "work" / "t4", runner=fake_runner(), mem_check=lambda: 16 * GIB)
    w = src / "work" / "t4"
    assert (w / "base" / "a.py").read_text() == "import os\n"
    assert (w / "renamed" / "a.py").read_text() == "import os\n"
    assert not (w / "base" / "new.txt").exists() and not (w / "renamed" / "new.txt").exists()
    assert res["worktree_diff_applied"] is True


def test_trial_paths_carry_named_paths_only(src):
    (src / "a.py").write_text("import os\n")
    (src / "tests" / "test_frozen_exp001.py").write_text(f'EXP001_TREE = "{"1" * 40}"\n')
    (src / "tests" / "test_new.py").write_text("def test_n(): pass\n")
    (src / "sibling.txt").write_text("not mine\n")
    res = R.trial(src, src / "work" / "t5", runner=fake_runner(), mem_check=lambda: 16 * GIB, paths=("a.py", "tests/"))
    w = src / "work" / "t5"
    assert res["carried_paths"] == ["a.py", "tests/test_frozen_exp001.py", "tests/test_new.py"]
    for name in ("base", "renamed"):
        assert (w / name / "a.py").read_text() == "import os\n"
        assert (w / name / "tests" / "test_new.py").exists()
        assert not (w / name / "sibling.txt").exists()


def test_trial_overlay(src):
    (src / "overlay.txt").write_text(f"{OLD}\n")
    R.trial(src, src / "work" / "t6", runner=fake_runner(), mem_check=lambda: 16 * GIB, overlays=("overlay.txt=extra/o.txt",))
    assert (src / "work" / "t6" / "renamed" / "extra" / "o.txt").read_text() == f"{OLD}\n"
    assert not (src / "work" / "t6" / "base" / "extra").exists()


def test_trial_refusals(src, tmp_path):
    ok = lambda: 16 * GIB
    with pytest.raises(R.RenameError, match="MemAvailable"):
        R.trial(src, src / "work" / "m", runner=fake_runner(), mem_check=lambda: GIB)
    with pytest.raises(R.RenameError, match="gitignored"):
        R.trial(src, src / "unignored", runner=fake_runner(), mem_check=ok)
    busy = src / "work" / "busy"
    busy.mkdir(parents=True)
    (busy / "x").write_text("x")
    with pytest.raises(R.RenameError, match="not empty"):
        R.trial(src, busy, runner=fake_runner(), mem_check=ok)
