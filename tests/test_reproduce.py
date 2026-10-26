"""Tier 1 reproduce wrapper and the Tier 2 image extraction helper (no Docker, no scorer run)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from glyphfix_math import guard
from scripts.release import extract_images as ei
from scripts.release import freeze_amendment as fa
from scripts.release import reproduce as rp

PUB = json.loads(Path("evidence/acceptance/results.json").read_text(encoding="utf-8"))
AM = json.loads(Path("evidence/acceptance/FREEZE_AMENDMENT.json").read_text(encoding="utf-8"))
MOD = fa.FROZEN_MODULE
CLI = fa.FROZEN_CLI
PUBLIC_MOD = ("a" * 40)
PUBLIC_CLI = ("b" * 40)
CMAP = {PUB["code_freeze"]["commits"][MOD][0]: PUBLIC_MOD, AM["amendment_commit"][0]: PUBLIC_CLI}


def amended(fresh_public: bool = False) -> dict:
    f = copy.deepcopy(PUB)
    cf = f["code_freeze"]
    cf["commits"][CLI] = [PUBLIC_CLI if fresh_public else AM["amendment_commit"][0], AM["amendment_commit"][1]]
    cf["current_blobs"]["cli"] = AM["new_blob"]
    cf["lock_blobs"]["cli"] = AM["new_blob"]
    return f


def history(f: dict) -> dict:
    f["code_freeze"]["commits"][MOD] = [PUBLIC_MOD, PUB["code_freeze"]["commits"][MOD][1]]
    return f


def test_identical_up_to_volatile_keys():
    fresh = copy.deepcopy(PUB)
    fresh["timings"] = {"x": 1}
    fresh["created"] = "other"
    fresh["git_commit"] = "other"
    assert rp.compare_results(PUB, fresh)["compare"] == "identical"


def test_modulo_history():
    v = rp.compare_results(PUB, history(copy.deepcopy(PUB)), CMAP)
    assert v["compare"] == "identical_modulo_history"
    assert v["differing_keys"] == [fa.HISTORY_KEY]


def test_modulo_amendment():
    v = rp.compare_results(PUB, amended(), {}, AM)
    assert v["compare"] == "identical_modulo_amendment"
    assert v["differing_keys"] == list(fa.AMENDED_KEYS)


def test_modulo_history_and_amendment():
    v = rp.compare_results(PUB, history(amended(fresh_public=True)), CMAP, AM)
    assert v["compare"] == "identical_modulo_history_and_amendment"


def test_amended_keys_without_amendment_are_different():
    assert rp.compare_results(PUB, amended(), {}, None)["compare"] == "different"


def test_unmapped_hash_is_different():
    assert rp.compare_results(PUB, history(copy.deepcopy(PUB)), {})["compare"] == "different"


def test_mapped_hash_with_other_date_is_different():
    f = history(copy.deepcopy(PUB))
    f["code_freeze"]["commits"][MOD][1] = "2026-10-18T00:00:00+02:00"
    assert rp.compare_results(PUB, f, CMAP)["compare"] == "different"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f["code_freeze"].__setitem__("pin_commit_time", "2026-10-17T12:00:00+02:00"),
        lambda f: f["code_freeze"].__setitem__("lock_started", "2026-10-17T09:00:00Z"),
        lambda f: f["code_freeze"]["current_blobs"].__setitem__("module", "0" * 40),
        lambda f: f["headline"].__setitem__("delta", 0.5),
        lambda f: f["transfer"].__setitem__("delta", 0.5),
        lambda f: next(iter(f["inputs"].values())).__setitem__("sha256", "0" * 64),
    ],
)
def test_other_changes_are_different(mutate):
    f = amended()
    mutate(f)
    v = rp.compare_results(PUB, f, CMAP, AM)
    assert v["compare"] == "different"
    assert v["problems"]


def test_fresh_cli_blob_not_the_predicted_one_is_different():
    f = amended()
    f["code_freeze"]["current_blobs"]["cli"] = "1" * 40
    assert rp.compare_results(PUB, f, {}, AM)["compare"] == "different"


def test_per_class_change_is_different():
    f = amended()
    f["per_class"][0]["stock_right"] = f["per_class"][0].get("stock_right", 0) + 1 if "stock_right" in f["per_class"][0] else 1
    f["per_class"][0]["__changed__"] = True
    assert rp.compare_results(PUB, f, {}, AM)["compare"] == "different"


# ------------------------------------------------------------------------------------------------ main()

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "docker_image_ok", lambda image=None: True)
    monkeypatch.setattr(guard, "mem_available_bytes", lambda *a, **k: 20 * guard.GIB)
    monkeypatch.setattr(rp.fa, "verify_chain", lambda *a, **k: [])
    seen: dict = {}

    def fake_score(work, env, cwd=None):
        seen["env"] = env
        (work / "evidence").mkdir(parents=True, exist_ok=True)
        (work / "evidence" / "results.json").write_text(json.dumps(seen["fresh"]), encoding="utf-8")
        return seen.get("exit", 0)

    monkeypatch.setattr(rp, "run_score", fake_score)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pub.json").write_text(json.dumps(PUB), encoding="utf-8")
    (tmp_path / "am.json").write_text(json.dumps(AM), encoding="utf-8")
    return seen


def run_main(tmp_path, *extra):
    work = tmp_path / "work"
    code = rp.main(["--work", str(work), "--published", "pub.json", "--amendment", "am.json", *extra])
    return code, json.loads((work / "reproduce_report.json").read_text(encoding="utf-8")), work


def test_main_identical_exit0_cold_cache(tmp_path, env):
    env["fresh"] = copy.deepcopy(PUB)
    code, rep, work = run_main(tmp_path)
    assert code == 0 and rep["compare"] == "identical"
    assert rep["cold_cache"] is True and rep["tier"] == 1 and rep["amendment"]["verified"] is True
    cache = Path(env["env"]["GLYPHFIX_MATH_CACHE"])
    assert cache == work / "render-cache" and cache.is_dir() and not any(cache.iterdir())
    assert rep["published_sha256"] and rep["fresh_sha256"]


def test_main_warm_cache_keeps_the_environment(tmp_path, env, monkeypatch):
    monkeypatch.delenv("GLYPHFIX_MATH_CACHE", raising=False)
    env["fresh"] = amended()
    code, rep, _ = run_main(tmp_path, "--warm-cache")
    assert code == 0 and rep["compare"] == "identical_modulo_amendment" and rep["cold_cache"] is False
    assert "GLYPHFIX_MATH_CACHE" not in env["env"]
    assert rep["differing_keys"] == list(fa.AMENDED_KEYS)
    assert set(rep["allowed"].values()) == {"amendment"}


def test_main_history_map_from_file(tmp_path, env):
    (tmp_path / "hist.json").write_text(json.dumps({"commit_map": CMAP}), encoding="utf-8")
    env["fresh"] = history(amended(fresh_public=True))
    code, rep, _ = run_main(tmp_path, "--history", "hist.json")
    assert code == 0 and rep["compare"] == "identical_modulo_history_and_amendment"
    assert set(rep["history_mapped"]) == {MOD, CLI}


def test_main_different_exit1(tmp_path, env):
    f = copy.deepcopy(PUB)
    f["headline"]["delta"] = 0.0
    env["fresh"] = f
    code, rep, _ = run_main(tmp_path)
    assert code == 1 and rep["compare"] == "different"


def test_main_unverified_amendment_stops_before_scoring(tmp_path, env, monkeypatch):
    monkeypatch.setattr(rp.fa, "verify_chain", lambda *a, **k: ["proof: broken"])
    called = []
    monkeypatch.setattr(rp, "run_score", lambda *a, **k: called.append(1) or 0)
    code, rep, _ = run_main(tmp_path)
    assert code == 1 and not called
    assert rep["amendment"]["problems"] == ["proof: broken"] and rep["amendment"]["verified"] is False


def test_main_docker_missing_exit3(tmp_path, env, monkeypatch):
    monkeypatch.setattr(rp, "docker_image_ok", lambda image=None: False)
    code, rep, _ = run_main(tmp_path)
    assert code == 3 and rep["score_exit"] is None


def test_main_low_ram_exit2(tmp_path, env, monkeypatch):
    monkeypatch.setattr(guard, "mem_available_bytes", lambda *a, **k: guard.RAM_FLOOR_BYTES + 3 * guard.GIB)
    code, _, _ = run_main(tmp_path)
    assert code == 2


def test_main_scorer_failure_passes_the_exit_code(tmp_path, env):
    env["fresh"] = copy.deepcopy(PUB)
    env["exit"] = 3
    code, rep, _ = run_main(tmp_path)
    assert code == 3 and rep["score_exit"] == 3


def test_score_command_is_an_argument_list_with_the_phase5_flags(tmp_path):
    cmd = rp.score_command(tmp_path)
    assert isinstance(cmd, list) and cmd[1:6] == ["-m", "scripts.release.frozen_rescore", "score", "--mode", "benchmark"]
    assert "--ref-score-crops-tuned" in cmd and "--ref-score-pages-tuned" in cmd


def test_reproduce_source_has_no_shell_and_no_frozen_edit():
    src = Path("scripts/release/reproduce.py").read_text(encoding="utf-8")
    assert "shell=True" not in src and "os.system" not in src
    assert "write_text(" not in src.replace("(work / \"reproduce_report.json\").write_text", "")


def test_preflight_reports_missing_image_and_low_ram(monkeypatch):
    monkeypatch.setattr(guard, "mem_available_bytes", lambda *a, **k: 20 * guard.GIB)
    monkeypatch.setattr(rp, "docker_image_ok", lambda image=None: False)
    assert rp.preflight()[0] == rp.EXIT_NO_IMAGE
    monkeypatch.setattr(rp, "docker_image_ok", lambda image=None: True)
    assert rp.preflight() == (0, "")
    monkeypatch.setattr(guard, "mem_available_bytes", lambda *a, **k: guard.RAM_FLOOR_BYTES + guard.GIB)
    code, msg = rp.preflight()
    assert code == rp.EXIT_LOW_RAM and "below" in msg


def test_cold_env_points_the_cache_into_work_unless_warm(tmp_path, monkeypatch):
    monkeypatch.delenv(rp.CACHE_ENV, raising=False)
    cold = rp.cold_env(tmp_path, False)
    assert Path(cold[rp.CACHE_ENV]) == tmp_path / "render-cache" and (tmp_path / "render-cache").is_dir()
    assert rp.CACHE_ENV not in rp.cold_env(tmp_path, True)


def test_score_command_goes_through_the_harness(tmp_path):
    assert rp.score_command(tmp_path)[1:3] == ["-m", "scripts.release.frozen_rescore"]


def test_main_with_a_rename_record_verifies_the_chain_and_compares_with_it(tmp_path, env, monkeypatch):
    rec = {"marker": "rename"}
    (tmp_path / rp.ra.AMENDMENT2_PATH).parent.mkdir(parents=True)
    (tmp_path / rp.ra.AMENDMENT2_PATH).write_text(json.dumps(rec), encoding="utf-8")
    calls = {}
    monkeypatch.setattr(rp.fa, "verify_chain", lambda *a, **k: calls.setdefault("chain", []) or [])
    seen = {}

    def compare(published, fresh, commit_map=None, amendment=None, rename=None):
        seen["rename"] = rename
        return {"compare": "identical_modulo_history_and_amendment", "differing_keys": [], "allowed": {}, "history_mapped": {}, "problems": []}

    monkeypatch.setattr(rp, "compare_results", compare)
    env["fresh"] = copy.deepcopy(PUB)
    code, rep, _ = run_main(tmp_path)
    assert code == 0 and seen["rename"] == rec
    assert rep["amendment"]["rename_present"] is True and rep["amendment"]["rename_verified"] is True
    assert rep["harness"] == "scripts.release.frozen_rescore" and rep["pin_substitution"] is None


def test_main_rename_record_not_verified_stops_before_scoring(tmp_path, env, monkeypatch):
    (tmp_path / rp.ra.AMENDMENT2_PATH).parent.mkdir(parents=True)
    (tmp_path / rp.ra.AMENDMENT2_PATH).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rp.fa, "verify_chain", lambda *a, **k: ["rename: broken"])
    called = []
    monkeypatch.setattr(rp, "run_score", lambda *a, **k: called.append(1) or 0)
    code, rep, _ = run_main(tmp_path)
    assert code == 1 and not called and rep["amendment"]["rename_verified"] is False


def test_main_renamed_tree_records_the_pin_substitution(tmp_path, env, monkeypatch):
    new = tmp_path / rp.ra.NEW_MODULE
    new.parent.mkdir(parents=True)
    new.write_text("x", encoding="utf-8")
    sub = {"path": rp.ra.NEW_MODULE, "commit": "c" * 40, "date": "d", "module_blob_before": "b", "proof_changed_lines": [1]}
    monkeypatch.setattr(rp.frozen_rescore, "pin_substitution", lambda *a, **k: sub)
    env["fresh"] = copy.deepcopy(PUB)
    _, rep, _ = run_main(tmp_path)
    assert rep["pin_substitution"] == {"path": sub["path"], "commit": sub["commit"], "date": "d"}


def test_main_refused_pin_substitution_stops_before_scoring(tmp_path, env, monkeypatch):
    new = tmp_path / rp.ra.NEW_MODULE
    new.parent.mkdir(parents=True)
    new.write_text("x", encoding="utf-8")

    def refuse(*a, **k):
        raise rp.frozen_rescore.SubstitutionRefused("no")

    monkeypatch.setattr(rp.frozen_rescore, "pin_substitution", refuse)
    called = []
    monkeypatch.setattr(rp, "run_score", lambda *a, **k: called.append(1) or 0)
    code, rep, _ = run_main(tmp_path)
    assert code == 1 and not called and rep["compare"] == "not_run: pin substitution refused"


def test_score_only_never_verifies_or_compares(tmp_path, env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    for name in ("verify_chain", "compare_published"):
        monkeypatch.setattr(rp.fa, name, boom)
    monkeypatch.setattr(rp, "compare_results", boom)
    env["fresh"] = {"anything": 1}
    work = tmp_path / "work"
    assert rp.main(["--work", str(work), "--score-only"]) == 0
    rep = json.loads((work / "reproduce_report.json").read_text(encoding="utf-8"))
    assert rep["compare"] == "not_run: score-only" and rep["score_exit"] == 0
    assert rep["harness"] == "scripts.release.frozen_rescore" and rep["pin_substitution"] is None
    assert rep["compare"] not in rp.IDENTICAL and rep["fresh_sha256"] is None
    assert Path(env["env"][rp.CACHE_ENV]) == work / "render-cache"


def test_score_only_passes_the_score_exit_code_and_honours_preflight(tmp_path, env, monkeypatch):
    env["fresh"] = {}
    env["exit"] = 5
    assert rp.main(["--work", str(tmp_path / "w1"), "--score-only"]) == 5
    monkeypatch.setattr(rp, "docker_image_ok", lambda image=None: False)
    assert rp.main(["--work", str(tmp_path / "w2"), "--score-only"]) == 3


def test_score_only_help_text():
    with pytest.raises(SystemExit):
        rp.main(["--help"])


# ----------------------------------------------------------------------------------------- extract_images

def make_fixture(tmp_path, n=2):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, recs = [], []
    for i in range(n):
        data = f"png-bytes-{i}".encode()
        sha = hashlib.sha256(data).hexdigest()
        rows.append({"id": f"id{i}", "image": {"bytes": data, "path": f"id{i}.png"}, "image_sha256": sha})
        recs.append({"id": f"id{i}", "image": f"data/benchmark/images/p/id{i}.png", "image_sha256": sha})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "b.parquet")
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return recs


def test_extract_writes_and_is_idempotent(tmp_path):
    recs = make_fixture(tmp_path)
    assert ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path) == {"written": 2, "skipped": 0, "records": 2}
    for r in recs:
        assert hashlib.sha256((tmp_path / r["image"]).read_bytes()).hexdigest() == r["image_sha256"]
    assert ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)["skipped"] == 2


def test_extract_refuses_a_different_existing_file(tmp_path):
    recs = make_fixture(tmp_path)
    t = tmp_path / recs[0]["image"]
    t.parent.mkdir(parents=True)
    t.write_bytes(b"other")
    with pytest.raises(ei.ExtractError, match="not overwritten"):
        ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)
    assert t.read_bytes() == b"other"


def test_extract_refuses_paths_outside_images(tmp_path):
    recs = make_fixture(tmp_path, 1)
    recs[0]["image"] = "data/benchmark/records.jsonl"
    (tmp_path / "records.jsonl").write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")
    with pytest.raises(ei.ExtractError, match="outside"):
        ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)
    recs[0]["image"] = "data/benchmark/images/../../evil.png"
    (tmp_path / "records.jsonl").write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")
    with pytest.raises(ei.ExtractError, match="outside"):
        ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)
    assert not (tmp_path / "evil.png").exists()


def test_extract_detects_a_hash_mismatch(tmp_path):
    recs = make_fixture(tmp_path, 1)
    recs[0]["image_sha256"] = "0" * 64
    (tmp_path / "records.jsonl").write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")
    with pytest.raises(ei.ExtractError, match="sha256"):
        ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)
    assert ei.main(["--parquet", str(tmp_path / "b.parquet"), "--records", str(tmp_path / "records.jsonl")]) == 1


def test_extract_detects_a_missing_image(tmp_path):
    recs = make_fixture(tmp_path, 2)
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in [*recs, {**recs[0], "id": "extra"}]) + "\n", encoding="utf-8")
    with pytest.raises(ei.ExtractError, match="no image"):
        ei.extract(tmp_path / "b.parquet", tmp_path / "records.jsonl", tmp_path)
