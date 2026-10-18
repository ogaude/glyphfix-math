"""The public-history allow-list: classification, import closure, and fail-closed handling of every historical path."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from scripts.release import common
from scripts.release import public_paths as pp

ROOT = Path(__file__).resolve().parent.parent

HISTORY = [
    "src/lightonocr_math/acceptance.py", "scripts/acceptance.py",
    "scripts/release/freeze_amendment.py", "tests/test_freeze_amendment.py", "tests/test_acceptance.py",
    "pyproject.toml", "uv.lock", ".python-version", ".gitignore", "requirements/train.lock.txt",
    "requirements/export_pins.json",
]
LATEST_ONLY = [
    "scripts/release/common.py", "scripts/release/stage_results.py", "tests/test_release_check.py",
    "data/benchmark/records.jsonl", "data/benchmark/acceptance/RUN_LOCK.json",
    "data/train/sets/full/trainset.jsonl", "data/train/sets/sweep1500/trainset.jsonl", "data/mining/labels.jsonl",
    "data/mining/taxonomy.json", "data/models/REGISTRY.json", "data/splits/SHA256SUMS",
    "data/catalog/train_dev.jsonl", *common.PUBLIC_EVIDENCE_FILES,
]
OVERLAY = [
    "README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", "ATTRIBUTION.jsonl", "RELEASE_HISTORY.json",
    "results/results.json", "REPRODUCE.md", "REBUILD.md",
]
DROPPED = [
    "data/corpus/train/records.jsonl", "data/corpus/dev/records.jsonl", "data/corpus/regression/pool.jsonl",
    "data/corpus/pages_calib/pages.jsonl", "data/train/dev_eval_ids.txt", "scripts/utils/models.py",
    "experiments/exp001_lighton/triage.py", "experiments/__init__.py", "docs/results/exp002_groupnorm_vs_kept.png",
    "docs/results/summary.md", "docs/EXP-001_LightOnOCR-3_Layer1.md", "docs/notes/miopen-patch-embed.md",
    "docs/notes/miopen-patch-embed/run3.sh", "docs/notes/vram-cache-guard-freeze.md", "docs/REPRODUCE.md",
    "docs/REBUILD.md", "ORIGIN.md", "CLAUDE.md", ".planning/STATE.md", ".planning/ROADMAP.md",
    ".planning/phases/05-acceptance-evaluation/evidence/results.json",
    ".planning/phases/05-acceptance-evaluation/evidence/env_check.json",
    ".planning/phases/05-acceptance-evaluation/evidence/sensitivity.json",
    ".planning/phases/05-acceptance-evaluation/reports/ACCEPTANCE.md",
    ".planning/phases/05-acceptance-evaluation/MOVED.md", "evidence/acceptance/unexpected.json",
    ".planning/phases/05-acceptance-evaluation/05-CONTEXT.md",
    ".planning/phases/05-acceptance-evaluation/evidence/dryrun_dev_results.json",
    ".planning/phases/05-acceptance-evaluation/evidence/precheck.json",
    ".planning/phases/05-acceptance-evaluation/evidence/env_check_precheck.json",
    ".planning/phases/05-acceptance-evaluation/evidence/flips.jsonl",
    ".planning/phases/05-acceptance-evaluation/evidence/rerun_check.json",
    ".planning/phases/04-error-mining-and-lora-training/reports/DEV_ANALYSIS.md",
    ".planning/phases/06-publish/evidence/reproduce_source_warm.json", "tests/test_frozen_exp001.py",
    "scripts/synth_glyph_probe.py", "tests/test_synth_glyph_probe.py", "data/formula_sources.jsonl",
]


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [x for x in out.splitlines() if x]


TRACKED = _git("ls-files")
HISTORICAL = sorted(set(_git("log", "--all", "--name-only", "--format=")))
SPEC = pp.load_spec()


@pytest.mark.parametrize("path", HISTORY)
def test_history(path):
    assert pp.classify(path, SPEC) == "history"


@pytest.mark.parametrize("path", LATEST_ONLY)
def test_latest_only(path):
    assert pp.classify(path, SPEC) == "latest-only"


@pytest.mark.parametrize("path", OVERLAY)
def test_overlay(path):
    assert pp.classify(path, SPEC) == "overlay"


@pytest.mark.parametrize("path", DROPPED)
def test_dropped(path):
    assert pp.classify(path, SPEC) is None


def _kept(path: str) -> bool:
    return pp.classify(path, SPEC) in ("history", "latest-only")


def _module_file(mod: str) -> str | None:
    base = mod.replace(".", "/")
    for cand in (f"{base}.py", f"{base}/__init__.py"):
        if (ROOT / cand).exists():
            return cand
    return None


def test_import_closure():
    problems = []
    for path in TRACKED:
        if not path.endswith(".py") or not _kept(path):
            continue
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
                mods += [f"{node.module}.{a.name}" for a in node.names]
            for mod in mods:
                if mod.split(".")[0] not in ("scripts", "experiments"):
                    continue
                f = _module_file(mod)
                if f is None:
                    continue  # an imported name inside a module, not a module
                if not _kept(f):
                    problems.append(f"{path} imports {mod} ({f} is dropped)")
                # parent packages
                parts = mod.split(".")
                for i in range(1, len(parts)):
                    pkg = _module_file(".".join(parts[:i]))
                    if pkg and not _kept(pkg):
                        problems.append(f"{path} imports {mod}: package file {pkg} is dropped")
    assert problems == []


def test_every_historical_path_is_classified_and_kept_ones_are_clean():
    for path in HISTORICAL:
        cls = pp.classify(path, SPEC)
        assert cls in (None, *pp.SECTIONS)
        if cls in ("history", "latest-only"):
            assert not any(rx.search(path) for _, rx in common.FORBIDDEN_FILE_PATTERNS), path


def test_spikes_code_is_kept_and_private_data_is_not():
    for p in ("scripts/spikes/ollama_infer.py", "scripts/spikes/common.py", "scripts/spikes/parity_report.py",
              "scripts/spikes/export_gguf.sh"):
        assert pp.classify(p, SPEC) == "history"
        assert not any(rx.search(p) for _, rx in common.FORBIDDEN_FILE_PATTERNS)
    private = dict(common.FORBIDDEN_FILE_PATTERNS)["private_data"]
    for p in ("data/probes/x.json", "data/spikes/x", "data/external/x"):
        assert private.search(p)
        assert pp.classify(p, SPEC) is None


def test_no_planning_no_docs_and_evidence_is_exact():
    for path in {*TRACKED, *HISTORICAL}:
        if path.startswith((".planning/", "docs/")):
            assert pp.classify(path, SPEC) is None, path
    kept_evidence = {p for p in TRACKED if p.startswith("evidence/") and _kept(p)}
    assert kept_evidence == set(common.PUBLIC_EVIDENCE_FILES)
    assert all(pp.classify(p, SPEC) == "latest-only" for p in kept_evidence)


def test_glob_semantics():
    rules = pp.load_spec.__globals__["_glob_to_regex"]
    assert rules("a/**").match("a/b/c")
    assert not rules("a/*").match("a/b/c")
    assert rules("**/__pycache__/**").match("scripts/x/__pycache__/y.pyc")
    assert rules("*.py").match("x.py") and not rules("*.py").match("d/x.py")
