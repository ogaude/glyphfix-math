"""Public-history release check on tiny throwaway repositories: one rejection test per rule."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import common as C
from scripts.release import freeze_amendment as FA
from scripts.release import release_check as rc
from scripts.release import release_check_git as rg
from scripts.release import rename_amendment as RA
from scripts.release import rename_map

SPEC = """\
[history]
src/**
scripts/**
tests/**
pyproject.toml
evidence/**
data/probes/**
data/mining/**
docs/**
.planning/**

[latest-only]
data/benchmark/**
data/catalog/**

[overlay]
README.md
ATTRIBUTION.jsonl
RELEASE_HISTORY.json
"""

OLD_CLI = (
    "from pathlib import Path\n\n"
    'EVIDENCE_DIR = Path(".planning/phases/05-acceptance-evaluation/evidence")\n'
    'REPORT_PATH = Path(".planning/phases/05-acceptance-evaluation/reports/ACCEPTANCE.md")\n\n\n'
    "def main():\n    return 0\n"
)
MODULE = "def score():\n    return 1\n"
DATES = {i: f"2026-01-0{i}T10:00:00+02:00" for i in range(1, 9)}
EVIDENCE = ("results.json", "flips.jsonl", "env_check.json", "sensitivity.json", "ACCEPTANCE.md", "FREEZE_AMENDMENT.json")
BASE_ROWS = [{"id": "r1", "arxiv_id": "2001.00001", "latex": "a"}, {"id": "r2", "arxiv_id": "2001.00001", "latex": "b"}]


def sh(repo: Path, *args: str, date: str | None = None, input: bytes | None = None) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false", *args],
                       capture_output=True, env=env, check=True, input=input)
    return r.stdout.decode().strip()


def blob_of(repo: Path, text: str) -> str:
    return sh(repo, "hash-object", "--stdin", input=text.encode())


def jl(rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


class B:
    """Builds the clean repository; options mutate exactly one rule."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        sh(self.repo, "init", "-q", "-b", "main")
        self.spec = tmp / "spec.txt"
        self.spec.write_text(SPEC)
        self.n = 0
        (tmp / "res").mkdir()
        self.train_records = tmp / "res" / "train.jsonl"
        self.dev_records = tmp / "res" / "dev.jsonl"
        self.trainset = tmp / "res" / "trainset.jsonl"
        self.train_records.write_text(jl([{"id": "tr_in"}, {"id": "tr_res"}]))
        self.dev_records.write_text(jl([{"id": "devrow1"}]))
        self.trainset.write_text(jl([{"id": "tr_in"}]))

    def commit(self, files: dict, msg: str = "c", day: int | None = None) -> str:
        self.n += 1
        for rel, content in files.items():
            p = self.repo / rel
            if content is None:
                p.unlink()
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content if isinstance(content, bytes) else content.encode())
        sh(self.repo, "add", "-A")
        sh(self.repo, "commit", "-q", "-m", msg, date=DATES[day or min(self.n, 8)])
        return sh(self.repo, "rev-parse", "HEAD")

    def build(self, *, pre=None, post=(), amend=None, lock=None, history=None, third_line=False, module_v2=False,
              drop_evidence=None, no_history=False, no_amendment=False) -> SimpleNamespace:
        r = self.repo
        if pre:
            self.commit(pre, "pre", day=1)
        base = {
            FA.FROZEN_MODULE: MODULE, "scripts/acceptance.py": OLD_CLI, "scripts/spikes/common.py": "X = 1\n",
            "scripts/spikes/export_gguf.sh": "echo gguf\n", "tests/test_spikes_common.py": "def test_x():\n    pass\n", "pyproject.toml": "[x]\n",
        }
        c1 = self.commit(base, "kept code", day=1)
        new_cli = OLD_CLI
        for _, old, new in FA.REPLACEMENTS:
            new_cli = new_cli.replace(f'"{old}"', f'"{new}"')
        if third_line:
            new_cli = new_cli.replace("return 0", "return 1")
        c2 = self.commit({"scripts/acceptance.py": new_cli}, "amend", day=2)
        old_blob, new_blob, mod_blob = blob_of(r, OLD_CLI), blob_of(r, new_cli), blob_of(r, MODULE)
        records = jl(BASE_ROWS)
        published = {
            "inputs": {},
            "code_freeze": {
                "commits": {"scripts/acceptance.py": [c1, DATES[1]], FA.FROZEN_MODULE: [c1, DATES[1]]},
                "current_blobs": {"cli": old_blob, "module": mod_blob}, "lock_blobs": {"cli": old_blob, "module": mod_blob},
            },
        }
        module_text = MODULE + "# v2\n" if module_v2 else MODULE
        mod_for_lock = blob_of(r, module_text)
        arm = {"acceptance_cli_blob": new_blob, "acceptance_cli_blob_generated": old_blob, "acceptance_module_blob": mod_for_lock}
        lock_doc = {"arms": {"tuned-crops": dict(arm), "tuned-pages": dict(arm)}}
        if lock:
            lock(lock_doc)
        amendment = {
            "old_blob": old_blob, "new_blob": new_blob, "module_blob": mod_blob, "old_commit": [c1, DATES[1]], "amendment_commit": [c2, DATES[2]],
            "replacements": [{"name": n, "old": o, "new": nw} for n, o, nw in FA.REPLACEMENTS],
            "rerun_equivalence": {"compare": "identical_modulo_amendment", "differing_keys": list(FA.AMENDED_KEYS)},
        }
        if amend:
            amend(amendment)
        files = {
            "data/benchmark/records.jsonl": records,
            "data/benchmark/BENCHMARK_FREEZE.json": json.dumps({"records_sha256": C.sha256_stream(self._tmpfile(records))}),
            "data/benchmark/acceptance/RUN_LOCK.json": json.dumps(lock_doc),
            "evidence/acceptance/results.json": json.dumps(published),
            "evidence/acceptance/flips.jsonl": "{}\n",
            "evidence/acceptance/env_check.json": "{}",
            "evidence/acceptance/sensitivity.json": "{}",
            "evidence/acceptance/ACCEPTANCE.md": "# a\n",
            "evidence/acceptance/FREEZE_AMENDMENT.json": json.dumps(amendment),
        }
        if module_v2:
            files[FA.FROZEN_MODULE] = module_text
        if drop_evidence:
            del files[f"evidence/acceptance/{drop_evidence}"]
        if no_amendment:
            files.pop("evidence/acceptance/FREEZE_AMENDMENT.json", None)
        c3 = self.commit(files, "data and evidence", day=3)
        overlay = {
            "ATTRIBUTION.jsonl": jl([{"arxiv_id": "2001.00001", "version": "v1"}]),
            "README.md": "# readme\n",
        }
        cmap = {c1: c1, c2: c2}
        if history:
            history(cmap, c1, c2, c3)
        if not no_history:
            overlay["RELEASE_HISTORY.json"] = json.dumps({"schema": 1, "commit_map": cmap})
        c4 = self.commit(overlay, "overlay", day=4)
        for i, (extra, msg) in enumerate(post):
            self.commit(extra, msg, day=5 + i)
        return SimpleNamespace(c1=c1, c2=c2, c3=c3, c4=c4, old_blob=old_blob, new_blob=new_blob)

    def _tmpfile(self, text: str) -> Path:
        p = self.tmp / "recs.tmp"
        p.write_text(text)
        return p

    def check(self, **kw) -> dict:
        kw.setdefault("reserved_sources", [self.train_records, self.dev_records])
        kw.setdefault("trainset", self.trainset)
        kw.setdefault("spec", rg.public_paths.load_spec(self.spec))
        rep = rg.check_git(self.repo, **kw)
        return {c["name"]: c["status"] for c in rep["checks"]} | {"_status": rep["status"], "_report": rep}


def failed(res: dict) -> set[str]:
    return {k for k, v in res.items() if not k.startswith("_") and v == "fail"}


@pytest.fixture
def b(tmp_path, monkeypatch) -> B:
    # B builds a repository without a rename record, so its evidence list is the one before the record existed.
    monkeypatch.setattr(C, "PUBLIC_EVIDENCE_FILES", tuple(f"evidence/acceptance/{n}" for n in EVIDENCE))
    return B(tmp_path)


def test_clean_repo_passes(b):
    b.build()
    res = b.check()
    assert failed(res) == set(), res["_report"]
    assert res["_status"] == "pass"
    assert set(res) >= {"paths", "forbidden_history", "attributed_content", "reserved_rows", "binaries", "frozen_code",
                        "freeze_amendment", "history_map", "remotes", "planning_docs"}


def test_check_never_writes_the_repository(b):
    b.build()
    before = (sh(b.repo, "status", "--porcelain"), sh(b.repo, "rev-parse", "HEAD"), sh(b.repo, "count-objects", "-v"))
    b.check()
    assert (sh(b.repo, "status", "--porcelain"), sh(b.repo, "rev-parse", "HEAD"), sh(b.repo, "count-objects", "-v")) == before


def test_works_on_a_bare_clone(b, tmp_path):
    b.build()
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(b.repo), str(bare)], check=True)
    sh(bare, "remote", "remove", "origin")
    rep = rg.check_git(bare, reserved_sources=[b.train_records, b.dev_records], trainset=b.trainset, spec=rg.public_paths.load_spec(b.spec))
    assert rep["status"] == "pass", rep


# ---- paths
def test_dropped_path_in_old_commit(b):
    b.build(post=[({"data/corpus/train/records.jsonl": "{}\n"}, "x"), ({"data/corpus/train/records.jsonl": None}, "rm")])
    assert "paths" in failed(b.check())


def test_older_different_blob_of_latest_only_path(b):
    b.build(pre={"data/benchmark/records.jsonl": jl([{"id": "old", "latex": "z"}])})
    assert "paths" in failed(b.check())


def test_overlay_path_before_overlay_commit(b):
    b.build(pre={"README.md": "old\n"})
    assert "paths" in failed(b.check())


# ---- forbidden_history
def test_local_path_in_deleted_blob(b):
    b.build(post=[({"scripts/leak.txt": "see /mnt" "/datas/x\n"}, "add"), ({"scripts/leak.txt": None}, "rm")])
    res = b.check()
    assert failed(res) == {"forbidden_history"}
    assert "/mnt" + "/datas" not in json.dumps(res["_report"])


def test_local_path_in_commit_message(b):
    b.build(post=[({"scripts/ok.txt": "ok\n"}, "message mentions /mnt" + "/datas/secret")])
    assert failed(b.check()) == {"forbidden_history"}


def test_personal_address_in_a_deleted_blob(b):
    addr = "olivier.gaude" + "@gmail.com"
    b.build(post=[({"scripts/ua.py": f'UA = "x ({addr})"\n'}, "add"), ({"scripts/ua.py": None}, "rm")])
    res = b.check()
    assert failed(res) == {"forbidden_history"}
    assert addr not in json.dumps(res["_report"])


def test_personal_address_as_commit_identity(b):
    addr = "olivier.gaude" + "@gmail.com"
    b.build()
    sh(b.repo, "-c", f"user.email={addr}", "commit", "-q", "--allow-empty", "-m", "empty", date=DATES[8])
    assert failed(b.check()) == {"forbidden_history"}


# ---- attributed_content / reserved_rows
def test_content_row_from_unattributed_paper(b):
    b.build(post=[({"data/mining/y.jsonl": jl([{"id": "zz", "arxiv_id": "2999.99999", "latex": "a"}])}, "x")])
    assert failed(b.check()) == {"attributed_content"}


def test_content_row_nested_under_a_list(b):
    b.build(post=[({"data/mining/y.json": json.dumps({"rows": [{"id": "zz", "paper_id": "2999.99999v2", "tuned_pred": "a"}]})}, "x")])
    assert failed(b.check()) == {"attributed_content"}


def test_attributed_content_exempts_tests(b):
    b.build(post=[({"tests/fixtures/rows.jsonl": jl([{"id": "t", "arxiv_id": "2999.99999", "latex": "a"}])}, "x")])
    assert failed(b.check()) == set()


def test_reserved_row_with_content(b):
    b.build(post=[({"data/mining/z.jsonl": jl([{"id": "devrow1", "latex": "a"}])}, "x")])
    assert failed(b.check()) == {"reserved_rows"}


def test_reserved_id_without_content_is_allowed(b):
    b.build(post=[({"data/mining/labels.jsonl": jl([{"id": "devrow1", "mining_class": "prime"}])}, "x")])
    assert failed(b.check()) == set()


def test_reserved_row_in_tests_still_fails(b):
    b.build(post=[({"tests/fixtures/rows.jsonl": jl([{"id": "tr_res", "latex": "a"}])}, "x")])
    assert failed(b.check()) == {"reserved_rows"}


def test_reserved_rows_not_run_without_sources(b):
    b.build()
    res = b.check(reserved_sources=None)
    assert res["reserved_rows"] == "not_run"
    assert res["_status"] == "pass"
    assert b.check(reserved_sources=None, final=True)["_status"] == "fail"


# ---- binaries
def test_png_in_history(b):
    b.build(post=[({"scripts/x.png": b"\x89PNG"}, "x")])
    assert failed(b.check()) == {"binaries"}


def test_gguf_in_history(b):
    b.build(post=[({"scripts/x.gguf": b"gguf"}, "x")])
    assert failed(b.check()) == {"binaries"}


def test_blob_over_50_mb(b):
    b.build(post=[({"scripts/big.txt": b"a" * (51 * 1024 * 1024)}, "x")])
    assert failed(b.check()) == {"binaries"}


def test_private_data_file_name_rule(b):
    b.build(post=[({"data/probes/x.json": "{}"}, "x")])
    assert failed(b.check()) == {"binaries"}


# ---- frozen_code
def test_head_cli_blob_differs_from_run_lock(b):
    def lock(doc):
        doc["arms"]["tuned-crops"]["acceptance_cli_blob"] = "0" * 40

    b.build(lock=lock)
    assert "frozen_code" in failed(b.check())


def test_benchmark_records_differ_from_freeze(b):
    b.build(post=[({"data/benchmark/records.jsonl": jl([{"id": "other", "arxiv_id": "2001.00001", "latex": "q"}])}, "x")])
    assert "frozen_code" in failed(b.check())


# ---- freeze_amendment
def test_new_blob_differs_from_head(b):
    b.build(amend=lambda a: a.update(new_blob="1" * 40))
    assert "freeze_amendment" in failed(b.check())


def test_proof_does_not_recompute(b):
    b.build(third_line=True)
    res = b.check()
    assert "freeze_amendment" in failed(res)
    assert any("proof" in d for c in res["_report"]["checks"] if c["name"] == "freeze_amendment" for d in c["details"])
    assert "frozen_code" not in failed(res)


def test_old_blob_unreachable(b):
    def both(a):
        a["old_blob"] = "2" * 40

    b.build(amend=both)
    assert "freeze_amendment" in failed(b.check())


def test_run_lock_generated_blob_differs(b):
    def lock(doc):
        doc["arms"]["tuned-pages"]["acceptance_cli_blob_generated"] = "3" * 40

    b.build(lock=lock)
    assert failed(b.check()) == {"freeze_amendment"}


def test_module_blob_changed(b):
    b.build(module_v2=True)
    assert "freeze_amendment" in failed(b.check())


def test_rerun_equivalence_not_identical(b):
    def amend(a):
        a["rerun_equivalence"]["compare"] = "different"

    b.build(amend=amend)
    assert failed(b.check()) == {"freeze_amendment"}


def test_no_amendment_with_changed_cli_fails(b):
    b.build(no_amendment=True)
    assert "freeze_amendment" in failed(b.check())


# ---- history_map
def test_history_map_date_mismatch(b):
    def hist(cmap, c1, c2, c3):
        cmap[c1] = c3

    b.build(history=hist)
    assert "history_map" in failed(b.check())


def test_history_map_missing_commit(b):
    def hist(cmap, c1, c2, c3):
        del cmap[c1]

    b.build(history=hist)
    assert "history_map" in failed(b.check())


def test_history_map_not_run_without_release_history(b):
    b.build(no_history=True)
    res = b.check()
    assert res["history_map"] == "not_run"
    assert b.check(final=True)["_status"] == "fail"


# ---- remotes
def test_remote_configured(b):
    b.build()
    sh(b.repo, "remote", "add", "origin", "https://example.org/x.git")
    assert failed(b.check()) == {"remotes"}


# ---- planning_docs
def test_planning_path_in_history(b):
    b.build(post=[({".planning/phases/05-acceptance-evaluation/evidence/results.json": "{}"}, "x")])
    assert failed(b.check()) == {"planning_docs"}


def test_docs_path_in_history(b):
    b.build(post=[({"docs/notes/a.md": "x\n"}, "x")])
    assert failed(b.check()) == {"planning_docs"}


def test_head_missing_evidence_file(b):
    b.build(drop_evidence="flips.jsonl")
    assert "planning_docs" in failed(b.check())


def test_extra_evidence_file(b):
    b.build(post=[({"evidence/acceptance/extra.json": "{}"}, "x")])
    assert failed(b.check()) == {"planning_docs"}


# ---- CLI wiring
def test_cli_git_subcommand(b, tmp_path, capsys):
    b.build()
    report = tmp_path / "rep.json"
    code = rc.main(["git", "--repo", str(b.repo), "--report", str(report), "--reserved-source", str(b.train_records)])
    out = capsys.readouterr().out
    assert "planning_docs" in out
    assert code in (0, 1)
    assert json.loads(report.read_text())["reports"][0]["surface"] == "git"


# ---- rename amendment chain (amendment 2)
OLD_PKG = RA.OLD_MODULE.split("/")[1]
CHAIN_MODULE = f"import {OLD_PKG}\n\n" + MODULE
CHAIN_CLI = f"import {OLD_PKG}\n\n" + OLD_CLI
CHAIN_SPEC = SPEC.replace("[latest-only]\n", "[latest-only]\ndata/models/REGISTRY.json\n")
REG_PATH = FA.REGISTRY_PATH.as_posix()
REG_OLD = json.dumps(
    {"entries": [{"tag": RA.OLD_TAG, "candidate_tag": RA.OLD_CAND_TAG, "ollama_digest": RA.REGISTRY_DIGEST}]}, indent=1
)
REG_NEW = rename_map.rewrite_text(REG_OLD)
LOCK_PATH = FA.LOCK_PATH.as_posix()


class Chain(B):
    """History with amendment 1 and the rename commit; the options break exactly one condition of the chain."""

    def build_chain(self, *, registry_before=True, registry="ok", listed="ok", a2=None, lock=None, rename_extra=None, post=()):
        r = self.repo
        c1_files = {
            "scripts/release/public_paths.txt": CHAIN_SPEC,
            RA.OLD_MODULE: CHAIN_MODULE,
            "scripts/acceptance.py": CHAIN_CLI,
            "scripts/spikes/common.py": "X = 1\n",
            "pyproject.toml": "[x]\n",
        }
        if registry_before:
            c1_files[REG_PATH] = REG_OLD
        c1 = self.commit(c1_files, "kept code", day=1)
        cli_a1 = CHAIN_CLI
        for _, old, new in FA.REPLACEMENTS:
            cli_a1 = cli_a1.replace(f'"{old}"', f'"{new}"')
        c2 = self.commit({"scripts/acceptance.py": cli_a1}, "amend 1", day=2)
        old1, new1, mod_old = blob_of(r, CHAIN_CLI), blob_of(r, cli_a1), blob_of(r, CHAIN_MODULE)
        cli_new, mod_new = rename_map.rewrite_text(cli_a1), rename_map.rewrite_text(CHAIN_MODULE)
        cli_new_blob, mod_new_blob = blob_of(r, cli_new), blob_of(r, mod_new)
        records = jl(BASE_ROWS)
        published = {
            "inputs": {},
            "code_freeze": {
                "commits": {"scripts/acceptance.py": [c1, DATES[1]], RA.OLD_MODULE: [c1, DATES[1]]},
                "current_blobs": {"cli": old1, "module": mod_old},
                "lock_blobs": {"cli": old1, "module": mod_old},
            },
        }
        arm = {"acceptance_cli_blob": new1, "acceptance_cli_blob_generated": old1, "acceptance_module_blob": mod_old}
        a1 = {
            "old_blob": old1, "new_blob": new1, "module_blob": mod_old, "old_commit": [c1, DATES[1]], "amendment_commit": [c2, DATES[2]],
            "replacements": [{"name": n, "old": o, "new": nw} for n, o, nw in FA.REPLACEMENTS],
            "rerun_equivalence": {"compare": "identical_modulo_amendment", "differing_keys": list(FA.AMENDED_KEYS)},
        }
        self.commit({
            "data/benchmark/records.jsonl": records,
            "data/benchmark/BENCHMARK_FREEZE.json": json.dumps({"records_sha256": C.sha256_stream(self._tmpfile(records))}),
            LOCK_PATH: json.dumps({"arms": {"tuned-crops": arm, "tuned-pages": dict(arm)}}),
            "evidence/acceptance/results.json": json.dumps(published),
            "evidence/acceptance/FREEZE_AMENDMENT.json": json.dumps(a1),
        }, "data and evidence", day=3)

        new_arm = {
            "acceptance_cli_blob": cli_new_blob, "acceptance_module_blob": mod_new_blob, "acceptance_cli_blob_generated": old1,
            RA.LOCK_NEW_KEY: mod_old, "freeze_amendment": RA.AMENDMENT2_REF,
        }
        lock_doc = {"arms": {"tuned-crops": new_arm, "tuned-pages": dict(new_arm)}}
        if lock:
            lock(lock_doc)
        lock_text = json.dumps(lock_doc)
        files = {RA.OLD_MODULE: None, RA.NEW_MODULE: mod_new, "scripts/acceptance.py": cli_new, LOCK_PATH: lock_text}
        if registry_before or registry != "absent":
            files[REG_PATH] = REG_OLD if registry == "old_tag" else REG_NEW
        files.update(rename_extra or {})
        c4 = self.commit(files, "rename", day=4)
        reg_blob = blob_of(r, REG_NEW)
        proof_list = {"ok": [{"path": REG_PATH, "new_blob": reg_blob}], "missing": [], "other_blob": [{"path": REG_PATH, "new_blob": "4" * 40}]}[listed]
        record = {
            "kind": "rename-only",
            "previous": FA.AMENDMENT_PATH.as_posix(),
            "token_map": [list(x) for x in rename_map.TOKEN_MAP],
            "path_moves": [list(x) for x in rename_map.PATH_MOVES],
            "files": {"cli": {"old_blob": new1, "new_blob": cli_new_blob}, "module": {"old_blob": mod_old, "new_blob": mod_new_blob}},
            "run_lock": {"changes": [{"key": "acceptance_cli_blob"}], "sha256_after": hashlib_sha256(lock_text)},
            "registry": {"changes": [
                {"key": f"entries[tag={RA.OLD_TAG}].tag", "old": RA.OLD_TAG, "new": RA.NEW_TAG},
                {"key": f"entries[tag={RA.OLD_TAG}].candidate_tag", "old": RA.OLD_CAND_TAG, "new": RA.NEW_CAND_TAG},
            ]},
            "amendment_commit": [c4, DATES[4]],
            "previous_amendment_commit": [c2, DATES[2]],
            "module_pin_commit": [c1, DATES[1]],
            "commit_proof": {"commit": c4, "exceptions": list(RA.COMMIT_EXCEPTIONS), "latest_only_rewritten": proof_list},
            "rerun_equivalence": {
                "compare": "identical_modulo_amendment", "differing_keys": list(RA.RENAME_KEYS),
                "report_rerender_identical": True, "score_exit": 0, "cold_cache": True,
            },
        }
        if a2:
            a2(record)
        c5 = self.commit({"evidence/acceptance/FREEZE_AMENDMENT_2.json": json.dumps(record)}, "amendment 2 record", day=5)
        cmap = {c: c for c in (c1, c2, c4, c5)}
        self.commit({
            "README.md": "# readme\n",
            "ATTRIBUTION.jsonl": jl([{"arxiv_id": "2001.00001", "version": "v1"}]),
            "RELEASE_HISTORY.json": json.dumps({"schema": 1, "commit_map": cmap}),
        }, "overlay", day=6)
        for i, (extra, msg) in enumerate(post):
            self.commit(extra, msg, day=min(7 + i, 8))
        self.mod_old = mod_old
        return SimpleNamespace(c1=c1, c2=c2, c4=c4, c5=c5)

    def chain_checks(self, monkeypatch, *, bare=False) -> dict:
        monkeypatch.setattr(FA, "MODULE_BLOB", self.mod_old)
        repo = self.repo
        if bare:
            repo = self.tmp / "bare.git"
            subprocess.run(["git", "clone", "-q", "--bare", str(self.repo), str(repo)], check=True)
            sh(repo, "remote", "remove", "origin")
        rep = rg.check_git(repo, reserved_sources=[self.train_records, self.dev_records], trainset=self.trainset,
                           spec=rg.public_paths.load_spec(self.spec))
        return {c["name"]: c for c in rep["checks"]}


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


CHAIN_CHECKS = ("frozen_code", "freeze_amendment", "history_map")


def chain_failed(checks: dict) -> set[str]:
    return {n for n in CHAIN_CHECKS if checks[n]["status"] == "fail"}


def chain_text(checks: dict, name: str) -> str:
    return " | ".join(checks[name]["details"])


@pytest.fixture
def chain(tmp_path) -> Chain:
    return Chain(tmp_path)


@pytest.mark.parametrize("bare", [False, True])
def test_chain_passes(chain, monkeypatch, bare):
    chain.build_chain()
    checks = chain.chain_checks(monkeypatch, bare=bare)
    assert chain_failed(checks) == set(), {n: checks[n]["details"] for n in CHAIN_CHECKS}
    assert "pure_rewrite=True" in chain_text(checks, "freeze_amendment")


def test_chain_passes_on_a_filtered_history(chain, monkeypatch):
    """The export drops the registry from the parent of the rename commit: the rename commit adds it, listed in the record."""
    chain.build_chain(registry_before=False)
    checks = chain.chain_checks(monkeypatch, bare=True)
    assert chain_failed(checks) == set(), {n: checks[n]["details"] for n in CHAIN_CHECKS}
    text = chain_text(checks, "freeze_amendment")
    assert "pure_rewrite=partial" in text
    assert f"latest_only_unproven_in_clone=['{REG_PATH}']" in text


@pytest.mark.parametrize(
    "opts",
    [
        {"post": [({REG_PATH: REG_NEW + "\n"}, "edit registry")]},
        {"registry": "old_tag"},
        {"listed": "missing"},
        {"listed": "other_blob"},
    ],
    ids=["differs_from_head", "old_tag_kept", "not_listed", "listed_with_another_blob"],
)
def test_filtered_registry_addition_is_refused(chain, monkeypatch, opts):
    chain.build_chain(registry_before=False, **opts)
    checks = chain.chain_checks(monkeypatch, bare=True)
    assert "freeze_amendment" in chain_failed(checks)
    assert REG_PATH in chain_text(checks, "freeze_amendment")


def test_chain_module_with_an_extra_byte(chain, monkeypatch):
    chain.build_chain(post=[({RA.NEW_MODULE: rename_map.rewrite_text(CHAIN_MODULE) + "#"}, "tamper")])
    assert {"freeze_amendment", "frozen_code"} <= chain_failed(chain.chain_checks(monkeypatch))


def test_chain_fourth_commit_on_the_module(chain, monkeypatch):
    new = rename_map.rewrite_text(CHAIN_MODULE)
    chain.build_chain(post=[({RA.NEW_MODULE: new + "#"}, "touch"), ({RA.NEW_MODULE: new}, "revert")])
    checks = chain.chain_checks(monkeypatch)
    assert chain.repo.joinpath(RA.NEW_MODULE).read_text() == new
    assert {"freeze_amendment", "history_map"} <= chain_failed(checks)
    assert "frozen_code" not in chain_failed(checks)


def test_chain_rename_commit_with_an_unrelated_change(chain, monkeypatch):
    chain.build_chain(rename_extra={"scripts/spikes/common.py": "X = 2\n"})
    checks = chain.chain_checks(monkeypatch)
    assert chain_failed(checks) == {"freeze_amendment"}
    assert "scripts/spikes/common.py" in chain_text(checks, "freeze_amendment")


def test_chain_rerun_equivalence_without_the_tag_key(chain, monkeypatch):
    chain.build_chain(a2=lambda rec: rec["rerun_equivalence"].update(differing_keys=[k for k in RA.RENAME_KEYS if k != "tag"]))
    assert chain_failed(chain.chain_checks(monkeypatch)) == {"freeze_amendment"}


def test_chain_run_lock_module_blob_left_at_the_old_blob(chain, monkeypatch):
    def lock(doc):
        for a in doc["arms"].values():
            a["acceptance_module_blob"] = chain_old_module_blob(chain)

    chain.build_chain(lock=lock)
    assert "frozen_code" in chain_failed(chain.chain_checks(monkeypatch))


def chain_old_module_blob(chain: Chain) -> str:
    return blob_of(chain.repo, CHAIN_MODULE)
