"""Reference closure: no kept file reads an excluded .planning path or any docs/ path."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.release import common
from scripts.release import public_paths as pp

ROOT = Path(__file__).resolve().parent.parent

# (file, substring, reason): write-only defaults, usage docstrings and optional reads that fall back when absent.
# A read of a .planning file that the code needs may never be in this table.
WRITE_ONLY = [
    ("scripts/corpus_pilot_report.py", ".planning/phases/03-corpus-and-benchmark/reports", "usage docstring, output dir"),
    ("scripts/dev_gain_by_class.py", ".planning/phases/04-error-mining-and-lora-training/evidence/dev_gain_by_class.json", "usage docstring, --out"),
    ("scripts/mine_train.py", ".planning/phases/04-error-mining-and-lora-training/reports/MINING.md", "--report output default"),
    ("scripts/setup_train_env.sh", ".planning/phases/02-feasibility-spikes/evidence/env_verify.json", "--out of the verifier"),
    ("scripts/token_mismatch_probe.py", ".planning/phases/03-corpus-and-benchmark/reports/token_mismatch.json", "usage docstring, --out"),
    ("scripts/verify_train_env.py", ".planning/phases/02-feasibility-spikes/evidence/env_verify.json", "usage docstring, --out"),
    ("scripts/spikes/common.py", '".planning"', "spike evidence write dir; the readers in train_lora.py, overfit_check.py and train_fit_grid.py fall back when the file is absent"),
]
SCAN_SUFFIXES = (".py", ".sh", ".toml", ".md")
EXEMPT_PREFIXES = ("scripts/release/", "src/glyphfix_math/leakscan.py")
EXEMPT_TESTS = (
    "tests/test_release_", "tests/test_public_", "tests/test_freeze_amendment.py", "tests/test_leak_scan.py",
)


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def _scanned() -> list[str]:
    spec = pp.load_spec()
    res = []
    for p in _tracked():
        if not p.endswith(SCAN_SUFFIXES) or pp.classify(p, spec) not in ("history", "latest-only"):
            continue
        if not p.startswith(("src/", "scripts/", "tests/", "release_templates/")):
            continue
        if p.startswith(EXEMPT_PREFIXES) or p.startswith(EXEMPT_TESTS):
            continue
        res.append(p)
    return res


def test_no_docs_and_only_table_planning_references():
    allowed: dict[str, list[str]] = {}
    for f, sub, _ in WRITE_ONLY:
        allowed.setdefault(f, []).append(sub)
    problems = []
    for p in _scanned():
        for n, line in enumerate((ROOT / p).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(?<![\w.])docs/", line):
                problems.append(f"{p}:{n}: docs/ reference")
            if ".planning" in line and not any(s in line for s in allowed.get(p, [])):
                problems.append(f"{p}:{n}: .planning reference")
    assert problems == []


def test_table_entries_exist():
    for f, sub, _ in WRITE_ONLY:
        assert sub in (ROOT / f).read_text(encoding="utf-8"), (f, sub)


def test_export_scripts_read_pins_file():
    for f in ("scripts/export_lora.sh", "scripts/spikes/export_gguf.sh"):
        body = (ROOT / f).read_text(encoding="utf-8")
        assert "requirements/export_pins.json" in body
        assert "02-feasibility-spikes" not in body


def test_export_pins_match_registry_and_phase2_evidence():
    pins = json.loads((ROOT / "requirements/export_pins.json").read_text())
    reg = json.loads((ROOT / "data/models/REGISTRY.json").read_text())
    e = next(x for x in reg["entries"] if x["tag"] == common.REGISTRY_TAG)
    assert pins["llama_cpp_commit"] == e["llama_cpp_commit"]
    assert pins["projector_blob_sha256"] == e["projector_sha256"]
    ev = ROOT / ".planning/phases/02-feasibility-spikes/evidence"
    if not (ev / "llama_cpp.json").exists() or not (ev / "roundtrip.json").exists():
        pytest.skip("Phase 2 evidence absent (public clone)")
    assert pins["llama_cpp_commit"] == json.loads((ev / "llama_cpp.json").read_text())["commit"]
    assert pins["projector_blob_sha256"] == json.loads((ev / "roundtrip.json").read_text())["projector_blob_sha256"]
