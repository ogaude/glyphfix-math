"""Acceptance CLI run side: environment check, pre-check and lock-protected generation (fakes only)."""

import json

import pytest

from lightonocr_math import acceptance as accmod
from scripts import acceptance as acc

STOCK_MANIFEST = {
    "model_digest": "stockdig",
    "gguf_blobs": {"text": "tt", "projector": "pp"},
    "template_sha256": "tpl",
}
ENV = {"OLLAMA_HOST": "127.0.0.1:11434", "HIP_VISIBLE_DEVICES": "0"}
BLOB = {"src/lightonocr_math/acceptance.py": "blob-mod", "scripts/acceptance.py": "blob-cli"}


class Server:
    """Fake Ollama daemon: answers for the patched helpers of scripts.acceptance."""

    def __init__(self):
        self.version = "0.40.1"
        self.digests = {acc.STOCK_TAG: "stockdig", acc.TUNED_TAG: "tuneddig"}
        self.blobs = {"text": "tt", "projector": "pp", "template_sha256": "tpl"}
        self.params = "num_ctx    32768\nnum_predict    24576\n"
        self.loaded = []


@pytest.fixture
def env(tmp_path, monkeypatch):
    srv = Server()
    monkeypatch.setattr(acc, "ACC_DIR", tmp_path / "acc")
    monkeypatch.setattr(acc, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(acc, "LOCK_PATH", tmp_path / "acc" / "RUN_LOCK.json")
    reg = tmp_path / "REGISTRY.json"
    reg.write_text(json.dumps({"entries": [{"tag": acc.TUNED_TAG, "ollama_digest": "tuneddig"}]}))
    monkeypatch.setattr(acc, "REGISTRY_PATH", reg)
    man = tmp_path / "stock.manifest.json"
    man.write_text(json.dumps(STOCK_MANIFEST))
    srv.manifest = man

    def get(host, path, timeout_s=30):
        if path == "/api/version":
            return {"version": srv.version}
        if path == "/api/ps":
            return {"models": [{"name": n} for n in srv.loaded]}
        raise AssertionError(path)

    monkeypatch.setattr(acc, "_get", get)
    monkeypatch.setattr(acc, "_model_digest", lambda host, model: srv.digests.get(model))
    monkeypatch.setattr(acc, "model_blobs", lambda host, model: dict(srv.blobs))
    monkeypatch.setattr(acc, "_show", lambda host, model: {"parameters": srv.params})
    monkeypatch.setattr(acc, "read_server_env", lambda: dict(srv.env) if srv.env is not None else None)
    srv.env = dict(ENV)
    monkeypatch.setattr(acc, "gpu_state", lambda: {"vram_free_bytes": 1})
    monkeypatch.setattr(acc.guard, "mem_available_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(acc, "git_blob", lambda p: BLOB[str(p)])
    monkeypatch.setattr(acc, "is_committed_clean", lambda p: True)
    monkeypatch.setattr(acc, "file_commit", lambda p: ("abc123", "2026-10-17T00:00:00+00:00"))
    monkeypatch.setattr(acc, "git_head", lambda: "headcommit")
    monkeypatch.setattr(accmod, "ACCEPTANCE_SHA256", accmod.constants_digest())
    return srv


# ------------------------------------------------------------------------------------------ verify-env


def _verify(env, tmp_path):
    out = tmp_path / "env_check.json"
    rc = acc.main(["verify-env", "--stock-manifest", str(env.manifest), "--out", str(out)])
    return rc, json.loads(out.read_text())


def test_verify_env_passes(env, tmp_path):
    rc, res = _verify(env, tmp_path)
    assert rc == 0 and res["pass"] is True
    assert set(res["checks"]) == {"ollama_version", "stock_digest", "projector_blob", "text_blob", "template_sha256", "stock_params", "tuned_digest"}
    assert all(c["ok"] for c in res["checks"].values())
    ready = res["readiness"]
    assert ready["server_env"] == ENV and len(ready["server_env_sha256"]) == 64
    assert {"ollama_ps", "gpu", "mem_available_bytes", "pin", "frozen_code"} <= ready.keys()
    assert ready["pin"]["acceptance_sha256"] == accmod.ACCEPTANCE_SHA256 and ready["pin"]["clean"] is True
    assert set(ready["frozen_code"]) == set(acc.FROZEN_CODE)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("ollama_version", lambda s: setattr(s, "version", "0.40.2")),
        ("stock_digest", lambda s: s.digests.__setitem__(acc.STOCK_TAG, "other")),
        ("projector_blob", lambda s: s.blobs.__setitem__("projector", "zz")),
        ("text_blob", lambda s: s.blobs.__setitem__("text", "zz")),
        ("template_sha256", lambda s: s.blobs.__setitem__("template_sha256", "zz")),
        ("stock_params", lambda s: setattr(s, "params", "num_ctx 8192\nnum_predict 24576\n")),
        ("tuned_digest", lambda s: s.digests.__setitem__(acc.TUNED_TAG, "other")),
    ],
)
def test_verify_env_fails_naming_field(env, tmp_path, capsys, field, mutate):
    mutate(env)
    rc, res = _verify(env, tmp_path)
    assert rc == 1 and res["pass"] is False
    assert res["checks"][field]["ok"] is False
    assert [k for k, c in res["checks"].items() if not c["ok"]] == [field]
    assert field in capsys.readouterr().out


def test_verify_env_unreadable_server_env(env, tmp_path):
    env.env = None
    rc, res = _verify(env, tmp_path)
    assert rc == 1 and res["pass"] is False
    assert res["readiness"]["server_env"] is None and res["readiness"]["server_env_sha256"] is None


# -------------------------------------------------------------------------------------------- precheck


def _row(i, text="x", n=10):
    return {"id": i, "raw_text": text, "prompt_eval_count": n}


def test_raw_text_cmp():
    ids = ["a", "b"]
    a = {i: _row(i) for i in ids}
    b = {i: _row(i) for i in ids}
    r = acc.raw_text_cmp(a, b, ids)
    assert r == {"n": 2, "n_identical": 2, "differing_ids": [], "missing_ids": [], "prompt_count_mismatch_ids": []}
    b["b"] = _row("b", "y", 11)
    r = acc.raw_text_cmp(a, b, ids)
    assert r["n_identical"] == 1 and r["differing_ids"] == ["b"] and r["prompt_count_mismatch_ids"] == ["b"]
    del b["a"]
    assert acc.raw_text_cmp(a, b, ids)["missing_ids"] == ["a"]


def test_prompt_offset_check():
    ids = ["a", "b"]
    hf = {i: {"id": i, "n_prompt_tokens": 10} for i in ids}
    ol = {i: _row(i, n=10) for i in ids}
    r = acc.prompt_offset_check(ol, hf, ids)
    assert r["pass"] is True and r["measured"] == 2 and r["modal_offset"] == 0 and r["n_offset_zero"] == 2
    ol["b"]["prompt_eval_count"] = None
    assert acc.prompt_offset_check(ol, hf, ids)["pass"] is False
    ol["b"]["prompt_eval_count"] = 11
    assert acc.prompt_offset_check(ol, hf, ids)["pass"] is False


def _write_side(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _precheck_setup(tmp_path, monkeypatch, *, tuned_b_differs=False, stock_differs=False):
    ids = [f"id{i}" for i in range(4)]
    src = tmp_path / "ids.txt"
    src.write_text("\n".join(ids + ["id9"]) + "\n")
    hf = tmp_path / "hf.jsonl"
    stored_t = tmp_path / "stored_t.jsonl"
    stored_s = tmp_path / "stored_s.jsonl"
    _write_side(hf, [{"id": i, "raw_text": "t", "n_prompt_tokens": 10} for i in ids + ["id9"]])
    _write_side(stored_t, [_row(i, "tuned") for i in ids + ["id9"]])
    _write_side(stored_s, [_row(i, "stock") for i in ids + ["id9"]])
    monkeypatch.setattr(acc, "PRECHECK_N", 4)
    calls = []

    def run_cmd(cmd):
        calls.append(cmd)
        out = cmd[cmd.index("--out") + 1]
        side = cmd[cmd.index("--sidecar") + 1]
        model = cmd[cmd.index("--model") + 1]
        text = "tuned" if model == acc.TUNED_TAG else "stock"
        if tuned_b_differs and out.endswith("tuned_b.pred.jsonl"):
            text = "other"
        if stock_differs and model == acc.STOCK_TAG:
            text = "other"
        _write_side(__import__("pathlib").Path(side), [_row(i, text) for i in ids])
        return 0

    monkeypatch.setattr(acc, "run_cmd", run_cmd)
    argv = [
        "precheck", "--ids-source", str(src), "--hf-sidecar", str(hf), "--dev-records", "dev.jsonl", "--stored-tuned", str(stored_t),
        "--stored-stock", str(stored_s), "--work", str(tmp_path / "work"), "--out", str(tmp_path / "precheck.json"),
    ]
    return argv, calls, ids


def test_precheck_passes(env, tmp_path, monkeypatch):
    argv, calls, ids = _precheck_setup(tmp_path, monkeypatch)
    assert acc.main(argv) == 0
    res = json.loads((tmp_path / "precheck.json").read_text())
    assert res["pass"] is True and set(res["checks"]) == {"tuned_ab", "tuned_vs_stored", "prompt_offset", "stock_vs_stored"}
    assert res["frozen_code"] == BLOB and res["digests"] == {"tuned": "tuneddig", "stock": "stockdig"}
    assert res["server_env_sha256"] and res["ollama_version"] == "0.40.1"
    assert [c[c.index("--model") + 1] for c in calls] == [acc.TUNED_TAG, acc.TUNED_TAG, acc.STOCK_TAG]
    assert (tmp_path / "work" / "precheck_ids.txt").read_text().split() == ids
    assert all(c[:3] == [acc.sys.executable, "-m", "lightonocr_math.guard"] and "scripts.baseline_infer" in c and "--suite" in c for c in calls)
    assert all(c[c.index("--suite") + 1] == "dev" for c in calls)


def test_precheck_fails_when_tuned_runs_differ(env, tmp_path, monkeypatch):
    argv, _, _ = _precheck_setup(tmp_path, monkeypatch, tuned_b_differs=True)
    assert acc.main(argv) == 1
    res = json.loads((tmp_path / "precheck.json").read_text())
    assert res["pass"] is False and res["checks"]["tuned_ab"]["ok"] is False


def test_precheck_fails_when_stock_differs_from_stored(env, tmp_path, monkeypatch):
    argv, _, _ = _precheck_setup(tmp_path, monkeypatch, stock_differs=True)
    assert acc.main(argv) == 1
    res = json.loads((tmp_path / "precheck.json").read_text())
    assert res["checks"]["stock_vs_stored"]["ok"] is False and res["checks"]["tuned_ab"]["ok"] is True


def test_precheck_refuses_loaded_model(env, tmp_path, monkeypatch):
    argv, calls, _ = _precheck_setup(tmp_path, monkeypatch)
    env.loaded.append("someone-else")
    assert acc.main(argv) == 2 and calls == []


# ------------------------------------------------------------------------------------------ generate


@pytest.fixture
def gen(env, tmp_path, monkeypatch):
    monkeypatch.setattr(acc, "EXPECTED", {"crops": 2, "pages": 1, "page_rows": 3})
    acc.EVIDENCE_DIR.mkdir(parents=True)
    good = {"pass": True, "server_env_sha256": acc.env_sha(ENV), "frozen_code": dict(BLOB)}
    (acc.EVIDENCE_DIR / "precheck.json").write_text(json.dumps(good))
    (acc.EVIDENCE_DIR / "env_check.json").write_text(json.dumps({"pass": True}))
    calls = []
    state = {"rc": 0, "digest": "tuneddig"}

    def run_cmd(cmd):
        calls.append(cmd)
        if "scripts.corpus_pages" in cmd:
            from pathlib import Path

            Path(cmd[cmd.index("--out") + 1]).write_text("{}\n{}\n{}\n")
            Path(cmd[cmd.index("--stats") + 1]).write_text(json.dumps({"page_match_version": "2"}))
            return 0
        out = __import__("pathlib").Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        key = "pred_text" if cmd[cmd.index("--suite") + 1] == "pages" else "pred_latex"
        n = 1 if key == "pred_text" else 2
        out.write_text("".join(json.dumps({"id": f"r{i}", key: "x"}) + "\n" for i in range(n)))
        out.with_name(out.stem + ".manifest.json").write_text(
            json.dumps({"model_digest": state["digest"], "n_ok": n, "n_error": 0, "n_skipped_resume": 0})
        )
        return state["rc"]

    monkeypatch.setattr(acc, "run_cmd", run_cmd)
    return calls, state


def _lock():
    return json.loads(acc.LOCK_PATH.read_text())


def test_generate_refused_with_pin_none(gen, monkeypatch):
    calls, _ = gen
    monkeypatch.setattr(accmod, "ACCEPTANCE_SHA256", None)
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2
    assert calls == [] and not acc.LOCK_PATH.exists()


def test_generate_refused_on_dirty_cli(gen, monkeypatch):
    calls, _ = gen
    monkeypatch.setattr(acc, "is_committed_clean", lambda p: str(p) != "scripts/acceptance.py")
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []


def test_generate_refused_without_precheck(gen):
    calls, _ = gen
    (acc.EVIDENCE_DIR / "precheck.json").unlink()
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []
    (acc.EVIDENCE_DIR / "precheck.json").write_text(json.dumps({"pass": False}))
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []


def test_generate_refused_when_cli_changed_since_precheck(gen, monkeypatch):
    calls, _ = gen
    monkeypatch.setattr(acc, "git_blob", lambda p: "changed" if str(p) == "scripts/acceptance.py" else BLOB[str(p)])
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []


def test_generate_refused_on_server_env_mismatch(gen, env):
    calls, _ = gen
    env.env = {**ENV, "OLLAMA_FLASH_ATTENTION": "1"}
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []


def test_generate_refused_on_tuned_digest_mismatch(gen, env):
    calls, _ = gen
    env.digests[acc.TUNED_TAG] = "other"
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and calls == []


def test_generate_writes_lock_then_runs(gen):
    calls, _ = gen
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 0
    entry = _lock()["arms"]["tuned-crops"]
    assert entry["completed"] is True and entry["n_ok"] == 2 and entry["n_error"] == 0
    assert entry["model"] == acc.TUNED_TAG and entry["model_digest"] == "tuneddig"
    assert entry["acceptance_module_blob"] == "blob-mod" and entry["acceptance_cli_blob"] == "blob-cli"
    assert entry["acceptance_sha256"] == accmod.ACCEPTANCE_SHA256 and entry["server_env_sha256"] == acc.env_sha(ENV)
    assert entry["git_commit"] == "headcommit" and entry["acceptance_commit"] == "abc123"
    cmd = calls[0]
    assert cmd[:3] == [acc.sys.executable, "-m", "lightonocr_math.guard"] and "--vram-gb" in cmd and "--" in cmd
    assert cmd[cmd.index("--suite") + 1] == "benchmark" and cmd[cmd.index("--model") + 1] == acc.TUNED_TAG
    assert "--resume" not in cmd and "shell" not in cmd
    assert (acc.ACC_DIR / "tuned_q8.run_env.json").is_file()


def test_lock_is_written_before_subprocess(gen, monkeypatch):
    calls, _ = gen
    seen = {}
    inner = acc.run_cmd

    def spy(cmd):
        seen["lock"] = json.loads(acc.LOCK_PATH.read_text())["arms"]["tuned-crops"]["completed"]
        return inner(cmd)

    monkeypatch.setattr(acc, "run_cmd", spy)
    acc.main(["generate", "--arm", "tuned-crops"])
    assert seen["lock"] is False


def test_second_fresh_generate_refused(gen):
    calls, state = gen
    state["rc"] = 1
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 1
    assert _lock()["arms"]["tuned-crops"]["completed"] is False
    n = len(calls)
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 2 and len(calls) == n


def test_resume_passes_through(gen, capsys):
    calls, state = gen
    state["rc"] = 1
    acc.main(["generate", "--arm", "tuned-crops"])
    assert "--resume" in capsys.readouterr().out
    state["rc"] = 0
    assert acc.main(["generate", "--arm", "tuned-crops", "--resume"]) == 0
    assert "--resume" in calls[-1]
    assert _lock()["arms"]["tuned-crops"]["completed"] is True


def test_resume_refused_on_changed_code(gen, monkeypatch):
    calls, state = gen
    state["rc"] = 1
    acc.main(["generate", "--arm", "tuned-crops"])
    n = len(calls)
    lock = _lock()
    lock["arms"]["tuned-crops"]["acceptance_module_blob"] = "older"
    acc.LOCK_PATH.write_text(json.dumps(lock))
    assert acc.main(["generate", "--arm", "tuned-crops", "--resume"]) == 2 and len(calls) == n


def test_resume_refused_on_digest_change(gen, env):
    calls, state = gen
    state["rc"] = 1
    acc.main(["generate", "--arm", "tuned-crops"])
    lock = _lock()
    lock["arms"]["tuned-crops"]["model_digest"] = "older"
    acc.LOCK_PATH.write_text(json.dumps(lock))
    n = len(calls)
    assert acc.main(["generate", "--arm", "tuned-crops", "--resume"]) == 2 and len(calls) == n


def test_completed_arm_refused_even_with_resume(gen):
    calls, _ = gen
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 0
    n = len(calls)
    assert acc.main(["generate", "--arm", "tuned-crops", "--resume"]) == 2 and len(calls) == n


def test_resume_without_lock_entry_refused(gen):
    calls, _ = gen
    assert acc.main(["generate", "--arm", "tuned-crops", "--resume"]) == 2 and calls == []


def test_tuned_pages_needs_completed_crops(gen):
    calls, _ = gen
    assert acc.main(["generate", "--arm", "tuned-pages"]) == 2 and calls == []
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 0
    assert acc.main(["generate", "--arm", "tuned-pages"]) == 0
    rows_cmds = [c for c in calls if "scripts.corpus_pages" in c]
    assert len(rows_cmds) == 1 and rows_cmds[0][-1] != ""
    assert _lock()["arms"]["tuned-pages"]["completed"] is True


def test_pages_arm_fails_on_wrong_match_version(gen, monkeypatch):
    acc.main(["generate", "--arm", "tuned-crops"])
    inner = acc.run_cmd

    def bad(cmd):
        rc = inner(cmd)
        if "scripts.corpus_pages" in cmd:
            from pathlib import Path

            Path(cmd[cmd.index("--stats") + 1]).write_text(json.dumps({"page_match_version": "1"}))
        return rc

    monkeypatch.setattr(acc, "run_cmd", bad)
    assert acc.main(["generate", "--arm", "tuned-pages"]) == 1
    assert _lock()["arms"]["tuned-pages"]["completed"] is False


def test_stock_arms_refused_while_env_check_passes(gen):
    calls, _ = gen
    assert acc.main(["generate", "--arm", "stock-crops"]) == 2 and calls == []
    (acc.EVIDENCE_DIR / "env_check.json").write_text(json.dumps({"pass": False}))
    assert acc.main(["generate", "--arm", "stock-crops"]) == 0
    assert _lock()["arms"]["stock-crops"]["model"] == acc.STOCK_TAG


def test_wrong_digest_in_manifest_leaves_incomplete(gen):
    calls, state = gen
    state["digest"] = "unexpected"
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 1
    assert _lock()["arms"]["tuned-crops"]["completed"] is False


def test_no_benchmark_data_touched(gen):
    import subprocess

    acc.main(["generate", "--arm", "tuned-crops"])
    out = subprocess.run(["git", "status", "--porcelain", "data/benchmark"], capture_output=True, text=True, check=False).stdout
    assert out.strip() == ""
