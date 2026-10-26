"""Pre-rename real-data tests of the rename amendment: real RUN_LOCK, real REGISTRY, preview on the real repository.

These need the repository in its pre-rename state and skip once the rename is applied; afterwards the same facts are
checked by `rename_amendment verify`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release import freeze_amendment as fa
from scripts.release import rename_amendment as ra

ROOT = Path(__file__).resolve().parent.parent


def _pre_rename_only() -> None:
    if (ROOT / ra.NEW_MODULE).exists():
        pytest.skip("rename applied: pre-rename real data is now checked by rename_amendment verify")


def _status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_real_run_lock_amendment_changes_the_expected_keys():
    _pre_rename_only()
    lock = json.loads((ROOT / fa.LOCK_PATH).read_text(encoding="utf-8"))
    out = ra.amended_lock_rename(lock, cli_new="c" * 40, module_new="m" * 40)
    for arm in fa.ARMS:
        before, after = lock["arms"][arm], out["arms"][arm]
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        assert changed == {
            "acceptance_cli_blob", "acceptance_module_blob", "acceptance_module_blob_generated", "freeze_amendment",
        }
        assert after["model"] == before["model"] == ra.OLD_TAG


def test_real_registry_renames_exactly_two_leaves():
    _pre_rename_only()
    text = (ROOT / fa.REGISTRY_PATH).read_text(encoding="utf-8")
    new = ra.renamed_registry_text(text)
    changes = ra.registry_changes(text, new)
    assert {(c["old"], c["new"]) for c in changes} == {(ra.OLD_TAG, ra.NEW_TAG), (ra.OLD_CAND_TAG, ra.NEW_CAND_TAG)}


def test_preview_on_the_real_repository_writes_everything_and_changes_nothing_tracked(tmp_path):
    _pre_rename_only()
    before = _status()
    out = tmp_path / "preview"
    r = subprocess.run(
        [sys.executable, "-m", "scripts.release.rename_amendment", "preview", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    names = {
        "acceptance.py", "acceptance_module.py", "proof_cli.json", "proof_module.json", "blobs.json", "RUN_LOCK.json",
        "REGISTRY.json", "exp001.json", "rewrite_plan.json", "acceptance.py.diff", "acceptance_module.py.diff",
        "RUN_LOCK.json.diff", "REGISTRY.json.diff", "SUMMARY.txt",
    }
    assert names <= {p.name for p in out.iterdir()}
    assert json.loads((out / "proof_cli.json").read_text())["changed_lines"] == [40, 41, 42, 43, 44, 45, 46, 47, 55, 62, 313]
    assert json.loads((out / "proof_module.json").read_text())["changed_lines"] == [19, 20]
    blobs = json.loads((out / "blobs.json").read_text())
    assert blobs["cli_old"] == ra.CLI_OLD_BLOB and blobs["module_old"] == ra.MODULE_OLD_BLOB
    exp = json.loads((out / "exp001.json").read_text())
    assert exp["old_tree"] == ra.EXP001_OLD_TREE and exp["predicted_new_tree"] != exp["old_tree"]
    summary = (out / "SUMMARY.txt").read_text()
    assert summary.startswith("RENAME AMENDMENT PREVIEW (nothing tracked has changed)")
    assert "Kept with old names without a hash pin" in summary and "frozen_rescore" in summary
    assert _status() == before
