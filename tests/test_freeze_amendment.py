"""Tests for the freeze amendment tool: proof rejections, comparison, lock edit, record recompute."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.release import freeze_amendment as fa

ROOT = Path(__file__).resolve().parent.parent
OLD_E, OLD_R = fa.REPLACEMENTS[0][1], fa.REPLACEMENTS[1][1]
NEW_E, NEW_R = fa.REPLACEMENTS[0][2], fa.REPLACEMENTS[1][2]

OLD_TEXT = f'''"""doc"""
from pathlib import Path

# comment
ACC_DIR = Path("data/benchmark/acceptance")
EVIDENCE_DIR = Path("{OLD_E}")
REPORT_PATH = Path("{OLD_R}")


def f(x):
    return x  # keep
'''
NEW_TEXT = OLD_TEXT.replace(f'"{OLD_E}"', f'"{NEW_E}"').replace(f'"{OLD_R}"', f'"{NEW_R}"')


def test_prove_positive():
    p = fa.prove(OLD_TEXT, NEW_TEXT)
    assert len(p["changed_lines"]) == 2 and p["changed_tokens"] == 2 and p["ast_equal_after_substitution"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("# keep", "# changed"),
        lambda t: t.replace("return x", "return  x"),
        lambda t: t + "\nX = 1\n",
        lambda t: t.replace("\n\n\ndef f", "\n\ndef f"),
        lambda t: t.replace("data/benchmark/acceptance", "data/other"),
        lambda t: t.replace(f'"{NEW_E}"', '"evidence/other"'),
        lambda t: t.replace("# comment", "# comment 2"),
    ],
)
def test_prove_rejects_other_changes(mutate):
    with pytest.raises(fa.AmendmentError):
        fa.prove(OLD_TEXT, mutate(NEW_TEXT))


def test_prove_rejects_acc_dir_instead():
    bad = OLD_TEXT.replace("data/benchmark/acceptance", NEW_E)
    with pytest.raises(fa.AmendmentError):
        fa.prove(OLD_TEXT, bad)


def test_prove_rejects_duplicate_old_literal():
    dup = OLD_TEXT + f'\nEXTRA = "{OLD_E}"\n'
    with pytest.raises(fa.AmendmentError):
        fa.prove(dup, dup.replace(f'"{OLD_E}"', f'"{NEW_E}"'))


def test_prove_rejects_replacement_inside_function():
    old = f'''from pathlib import Path


def g():
    EVIDENCE_DIR = Path("{OLD_E}")
    return EVIDENCE_DIR


REPORT_PATH = Path("{OLD_R}")
'''
    with pytest.raises(fa.AmendmentError):
        fa.prove(old, old.replace(f'"{OLD_E}"', f'"{NEW_E}"').replace(f'"{OLD_R}"', f'"{NEW_R}"'))


def _lock():
    arm = {"acceptance_cli_blob": fa.OLD_CLI_BLOB, "acceptance_module_blob": fa.MODULE_BLOB, "n_ok": 3}
    return {"arms": {"tuned-crops": dict(arm), "tuned-pages": dict(arm)}}


def test_amended_lock():
    lock = _lock()
    out = fa.amended_lock(lock, old_blob=fa.OLD_CLI_BLOB, new_blob="n" * 40)
    assert lock == _lock()
    for arm in fa.ARMS:
        a = out["arms"][arm]
        assert a["acceptance_cli_blob"] == "n" * 40
        assert a["acceptance_cli_blob_generated"] == fa.OLD_CLI_BLOB
        assert a["freeze_amendment"] == fa.AMENDMENT_REF
        assert a["n_ok"] == 3 and a["acceptance_module_blob"] == fa.MODULE_BLOB
    bad = _lock()
    bad["arms"]["tuned-pages"]["acceptance_cli_blob"] = "x" * 40
    with pytest.raises(fa.AmendmentError):
        fa.amended_lock(bad, old_blob=fa.OLD_CLI_BLOB, new_blob="n" * 40)
    bad = _lock()
    bad["arms"]["tuned-crops"]["acceptance_module_blob"] = "y" * 40
    with pytest.raises(fa.AmendmentError):
        fa.amended_lock(bad, old_blob=fa.OLD_CLI_BLOB, new_blob="n" * 40)


NEW_BLOB = "n" * 40
AM = {
    "old_blob": fa.OLD_CLI_BLOB,
    "new_blob": NEW_BLOB,
    "old_commit": list(fa.OLD_CLI_COMMIT),
    "amendment_commit": ["a" * 40, "2026-10-18T10:00:00+02:00"],
}


def _res():
    return {
        "headline": {"delta": 0.2},
        "per_class": {"a": 1},
        "transfer": {"holds": True},
        "spacing": {"x": 1},
        "inputs": {"crops": {"sha256": "s"}},
        "created": "now",
        "code_freeze": {
            "pin_commit_time": "t",
            "lock_started": "l",
            "commits": {
                fa.FROZEN_CLI: list(fa.OLD_CLI_COMMIT),
                fa.FROZEN_MODULE: ["m" * 40, "2026-10-17T09:59:12+02:00"],
            },
            "current_blobs": {"cli": fa.OLD_CLI_BLOB, "module": fa.MODULE_BLOB},
            "lock_blobs": {"cli": fa.OLD_CLI_BLOB, "module": fa.MODULE_BLOB},
        },
    }


def _amend(r):
    r = copy.deepcopy(r)
    r["code_freeze"]["commits"][fa.FROZEN_CLI] = list(AM["amendment_commit"])
    r["code_freeze"]["current_blobs"]["cli"] = NEW_BLOB
    r["code_freeze"]["lock_blobs"]["cli"] = NEW_BLOB
    return r


def test_compare_identical_and_amendment():
    p = _res()
    assert fa.compare_published(p, copy.deepcopy(p))["compare"] == "identical"
    r = fa.compare_published(p, _amend(p), amendment=AM)
    assert r["compare"] == "identical_modulo_amendment"
    assert r["differing_keys"] == list(fa.AMENDED_KEYS)
    assert fa.compare_published(p, _amend(p))["compare"] == "different"


def test_compare_history():
    p = _res()
    f = copy.deepcopy(p)
    f["code_freeze"]["commits"][fa.FROZEN_MODULE][0] = "z" * 40
    cmap = {"m" * 40: "z" * 40}
    r = fa.compare_published(p, f, commit_map=cmap)
    assert r["compare"] == "identical_modulo_history"
    assert r["history_mapped"][fa.FROZEN_MODULE]["public_commit"] == "z" * 40
    assert fa.compare_published(p, f)["compare"] == "different"
    f["code_freeze"]["commits"][fa.FROZEN_MODULE][1] = "other"
    assert fa.compare_published(p, f, commit_map=cmap)["compare"] == "different"


def test_compare_history_and_amendment():
    p = _res()
    f = _amend(p)
    f["code_freeze"]["commits"][fa.FROZEN_CLI][0] = "q" * 40
    f["code_freeze"]["commits"][fa.FROZEN_MODULE][0] = "z" * 40
    cmap = {"m" * 40: "z" * 40, AM["amendment_commit"][0]: "q" * 40}
    r = fa.compare_published(p, f, amendment=AM, commit_map=cmap)
    assert r["compare"] == "identical_modulo_history_and_amendment"
    assert r["history_mapped"][fa.FROZEN_CLI]["source_commit"] == AM["amendment_commit"][0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f["headline"].update(delta=0.3),
        lambda f: f["per_class"].update(a=2),
        lambda f: f["transfer"].update(holds=False),
        lambda f: f["spacing"].update(x=2),
        lambda f: f["inputs"]["crops"].update(sha256="t"),
        lambda f: f["code_freeze"]["current_blobs"].update(module="q" * 40),
        lambda f: f["code_freeze"]["lock_blobs"].update(module="q" * 40),
        lambda f: f["code_freeze"].update(pin_commit_time="x"),
        lambda f: f["code_freeze"].update(lock_started="x"),
        lambda f: f["code_freeze"]["current_blobs"].update(cli="w" * 40),
        lambda f: f["code_freeze"]["commits"][fa.FROZEN_CLI].__setitem__(1, "other-date"),
    ],
)
def test_compare_rejects(mutate):
    p = _res()
    f = _amend(p)
    mutate(f)
    r = fa.compare_published(p, f, amendment=AM)
    assert r["compare"] == "different" and r["problems"]


def test_compare_rejects_wrong_published_cli_commit():
    p = _res()
    p["code_freeze"]["commits"][fa.FROZEN_CLI] = ["b" * 40, "d"]
    r = fa.compare_published(p, _amend(_res()), amendment=AM)
    assert r["compare"] == "different"


def test_load_commit_map(tmp_path):
    assert fa.load_commit_map(tmp_path) == {}
    (tmp_path / "RELEASE_HISTORY.json").write_text(json.dumps({"commit_map": {"a": "b"}}))
    assert fa.load_commit_map(tmp_path) == {"a": "b"}


def test_verify_uses_default_commit_map(tmp_path, monkeypatch):
    # The amendment commit is reachable only through the map: the same setup without the file reports it.
    import subprocess

    def sh(*a):
        subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)

    sh("git", "init", "-q")
    sh("git", "config", "user.email", "t@t")
    sh("git", "config", "user.name", "t")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/acceptance.py").write_text(NEW_TEXT)
    sh("git", "add", ".")
    sh("git", "commit", "-qm", "x")
    h, d = fa._git("log", "-1", "--format=%H %cI", root=tmp_path).split()
    am = {"amendment_commit": ["f" * 40, d]}
    without = fa.verify_amendment(tmp_path, am, published={})
    assert any("last commit touching" in p for p in without)
    (tmp_path / "RELEASE_HISTORY.json").write_text(json.dumps({"commit_map": {"f" * 40: h}}))
    with_map = fa.verify_amendment(tmp_path, am, published={})
    assert not any("last commit touching" in p for p in with_map)


def test_verify_real_repository():
    pytest.skip("temporary: the amendment is not recorded yet") if not (ROOT / fa.AMENDMENT_PATH).is_file() else None
    assert fa.verify_amendment(ROOT) == []
