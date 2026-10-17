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
    state = {"rc": 0, "digest": None}

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
            json.dumps({"model_digest": state["digest"] or env.digests[cmd[cmd.index("--model") + 1]], "n_ok": n, "n_error": 0, "n_skipped_resume": 0})
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
    assert (acc.ACC_DIR / "tuned_q8.pred.run_env.json").is_file()


def test_lock_is_written_before_subprocess(gen, monkeypatch):
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
    assert "--resume" in capsys.readouterr().err
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
    _, state = gen
    state["digest"] = "unexpected"
    assert acc.main(["generate", "--arm", "tuned-crops"]) == 1
    assert _lock()["arms"]["tuned-crops"]["completed"] is False


def test_no_benchmark_data_touched(gen):
    import subprocess

    acc.main(["generate", "--arm", "tuned-crops"])
    out = subprocess.run(["git", "status", "--porcelain", "data/benchmark"], capture_output=True, text=True, check=False).stdout
    assert out.strip() == ""


# ------------------------------------------------------------------------------------------------ score

PAPERS = [f"2601.0000{k}" for k in range(6)]
GLYPH = ("prime_l", "dropped_prime", "greek_latin", "decoration", "i_1")
VERSIONS = {
    "scorer_version": "1.2.0", "scorer_fingerprint": "fp", "normalize_rules_version": "n1", "error_rules_version": "e1",
    "preamble_sha256": "pre", "wrap_version": "w1", "split_rule_version": "s1", "split_digest": "sd", "render_dpi": 300,
}
MANIFEST = {
    "suite": "benchmark", "options": {"temperature": 0}, "endpoint": "/api/chat", "think": False, "prompt": "", "prompt_sha256": "ps",
    "template_sha256": "tpl", "pixel_policy": "pp", "ollama_version": "0.40.1", "records_sha256": "rs", "scorer_version": "1.2.0",
    "scorer_fingerprint": "fp", "render_config": {"dpi": 300}, "split_rule": "s1", "gguf_blobs": {"projector": "pp", "text": "tt"},
}


def synth_rows(prefix, stock_ri, tuned_ri, *, tuned_exact=None, strip_ids=()):
    """(stock rows, tuned rows) with scorer hints (_ri, _exact, _classes) read by the fake score_file."""
    stock, tuned = [], []
    for k, paper in enumerate(PAPERS):
        for j in range(5):
            rid = f"{prefix}{k}{j}"
            gt = r"a\,b" if rid in strip_ids else f"x_{rid}"
            base = {"id": rid, "paper_id": paper, "gt_latex": gt}
            stock.append({**base, "pred_latex": f"s_{rid}", "_ri": stock_ri(k, j), "_exact": stock_ri(k, j), "_classes": [] if stock_ri(k, j) else ["prime_l"]})
            te = tuned_ri(k, j) if tuned_exact is None else tuned_exact(k, j)
            tuned.append({**base, "pred_latex": f"t_{rid}", "_ri": tuned_ri(k, j), "_exact": te, "_classes": [] if te else ["greek_latin"]})
    return stock, tuned


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def fake_score_file(pred_path, out_dir, *, render=True, B=10000, seed=0, cache=None):
    rows = [json.loads(ln) for ln in open(pred_path) if ln.strip()]
    recs = []
    for r in rows:
        recs.append({
            "id": r["id"], "paper_id": r["paper_id"], "gt_latex": r["gt_latex"], "pred_latex": r["pred_latex"],
            "exact": r["_exact"], "render_identical": r["_ri"], "render_tolerant": r.get("_rt", r["_ri"]),
            "classes": r["_classes"], "present": list(GLYPH),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rate = lambda key: {"rate": sum(bool(r[key]) for r in recs) / len(recs), "den": len(recs)}  # noqa: E731
    res = {
        **VERSIONS, "render": {"image": "img"}, "per_class": {c: {"errors": sum(c in r["classes"] for r in recs), "denom": len(recs)} for c in GLYPH},
        "metrics": {"exact": rate("exact"), "render_identical": rate("render_identical"), "render_tolerant": rate("render_tolerant")},
    }
    (out_dir / "results.json").write_text(json.dumps(res))
    return out_dir / "results.json"


class Score:
    """Synthetic arms on disk plus the argument list of one score invocation."""

    def __init__(self, tmp_path, monkeypatch, env):
        self.tmp = tmp_path
        self.env = env
        self.calls = []
        self.d = tmp_path / "in"
        monkeypatch.setattr(acc, "EXPECTED", {"crops": 30, "pages": 6, "page_rows": 30})

        def score_file(pred, out, **kw):
            self.calls.append(str(out))
            return fake_score_file(pred, out, **kw)

        monkeypatch.setattr(acc, "score_file", score_file)
        monkeypatch.setattr(acc.freeze, "assert_benchmark_frozen", lambda root: {})
        monkeypatch.setattr(acc.freeze, "assert_pages_frozen", lambda root: {})
        monkeypatch.setattr(acc.splits, "is_benchmark", lambda pid: True)
        monkeypatch.setattr(acc.paths, "cache_dir", lambda: tmp_path / "cache")
        pf = tmp_path / "PAGES_FREEZE.json"
        pf.write_text(json.dumps({"match_threshold": 0.5, "page_match_version": "2"}))
        monkeypatch.setattr(acc, "PAGES_FREEZE_PATH", pf)
        self.build(lambda k, j: j == 0, lambda k, j: j < 4)

    def build(self, stock_ri, tuned_ri, **kw):
        d = self.d
        cs, ct = synth_rows("c", stock_ri, tuned_ri, **kw)
        ps, pt = synth_rows("p", stock_ri, tuned_ri)
        self.files = {
            "crops_stock": write_rows(d / "stock.pred.jsonl", cs), "crops_tuned": write_rows(d / "tuned.pred.jsonl", ct),
            "pages_stock_rows": write_rows(d / "stock_pages_rows.jsonl", ps), "pages_tuned_rows": write_rows(d / "tuned_pages_rows.jsonl", pt),
            "pages_stock_preds": write_rows(d / "stock_pages.pred.jsonl", [{"id": "pg"}]),
            "pages_tuned_preds": write_rows(d / "tuned_pages.pred.jsonl", [{"id": "pg"}]),
        }
        self.manifests = {"stock": dict(MANIFEST, model_digest="stockdig"), "tuned": dict(MANIFEST, model_digest="tuneddig")}
        for arm in ("stock", "tuned"):
            for suffix, key in (("", "crops"), ("_pages", "pages")):
                p = self.files[f"crops_{arm}" if key == "crops" else f"pages_{arm}_preds"]
                m = dict(self.manifests[arm], n_error=0)
                p.with_name(p.name.replace(".jsonl", ".manifest.json")).write_text(json.dumps(m))
        for arm in ("stock", "tuned"):
            (d / f"{arm}_stats.json").write_text(json.dumps({"page_match_version": "2", "threshold": 0.5, "match_rate": 0.95, "n_gt": 30, "n_matched": 29}))
        (d / "env_check.json").write_text(json.dumps({"pass": True}))
        (self.tmp / "evidence").mkdir(exist_ok=True)
        (self.tmp / "acc").mkdir(exist_ok=True)
        self.lock = {"arms": {"tuned-crops": {
            "started": "2026-10-17T12:00:00Z", "acceptance_module_blob": "blob-mod", "acceptance_cli_blob": "blob-cli", "completed": True}}}
        self.write_lock()

    def write_lock(self):
        acc.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        acc.LOCK_PATH.write_text(json.dumps(self.lock))

    def args(self, mode="benchmark", **over):
        f = self.files
        opts = {
            "--crops-stock": f["crops_stock"], "--crops-tuned": f["crops_tuned"],
            "--pages-stock-preds": f["pages_stock_preds"], "--pages-tuned-preds": f["pages_tuned_preds"],
            "--pages-stock-rows": f["pages_stock_rows"], "--pages-tuned-rows": f["pages_tuned_rows"],
            "--pages-stock-stats": self.d / "stock_stats.json", "--pages-tuned-stats": self.d / "tuned_stats.json",
            "--ref-score-crops-stock": "none", "--ref-score-pages-stock": "none",
            "--scores-dir": self.tmp / "scores", "--evidence-dir": self.tmp / "evidence", "--report": self.tmp / "report.md",
            "--env-check": self.d / "env_check.json", "--registry": self.tmp / "REGISTRY.json",
        }
        opts.update({k if k.startswith("--") else "--" + k.replace("_", "-"): v for k, v in over.items()})
        out = ["score", "--mode", mode]
        for k, v in opts.items():
            out += [k, str(v)]
        return out

    def run(self, mode="benchmark", **over):
        return acc.main(self.args(mode, **over))


@pytest.fixture
def sc(env, tmp_path, monkeypatch):
    return Score(tmp_path, monkeypatch, env)


def test_score_help_lists_options(capsys):
    with pytest.raises(SystemExit):
        acc.main(["score", "--help"])
    out = capsys.readouterr().out
    for opt in ("--mode", "--scores-dir", "--evidence-dir", "--report", "--ref-score-crops-stock", "--ref-score-pages-tuned", "--env-check"):
        assert opt in out


def test_score_benchmark_scores_four_arms(sc):
    assert sc.run() == 0
    assert sorted(p.rsplit("/", 1)[1] for p in sc.calls) == ["crops_stock", "crops_tuned", "pages_stock", "pages_tuned"]
    assert (sc.tmp / "scores" / "crops_tuned" / "results.json").is_file()


def test_score_pages_arms_use_rows_files(sc, monkeypatch):
    seen = []
    monkeypatch.setattr(acc, "score_file", lambda pred, out, **kw: (seen.append((out.name, pred.name)), fake_score_file(pred, out, **kw))[1])
    assert sc.run() == 0
    assert dict(seen)["pages_stock"] == "stock_pages_rows.jsonl" and dict(seen)["pages_tuned"] == "tuned_pages_rows.jsonl"


def test_refused_with_pin_none(sc, monkeypatch):
    monkeypatch.setattr(accmod, "ACCEPTANCE_SHA256", None)
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_dirty_cli(sc, monkeypatch):
    monkeypatch.setattr(acc, "is_committed_clean", lambda p: p != "scripts/acceptance.py")
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_cli_blob_differs_from_lock(sc, monkeypatch):
    sc.lock["arms"]["tuned-crops"]["acceptance_cli_blob"] = "other"
    sc.write_lock()
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_module_blob_differs_from_lock(sc):
    sc.lock["arms"]["tuned-crops"]["acceptance_module_blob"] = "other"
    sc.write_lock()
    assert sc.run() == 2 and sc.calls == []


def test_refused_when_lock_started_before_pin_commit(sc):
    sc.lock["arms"]["tuned-crops"]["started"] = "2026-10-16T23:59:59Z"
    sc.write_lock()
    assert sc.run() == 2 and sc.calls == []


def test_refused_without_lock_entry(sc):
    sc.lock = {"arms": {}}
    sc.write_lock()
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_non_benchmark_paper(sc, monkeypatch):
    monkeypatch.setattr(acc.splits, "is_benchmark", lambda pid: pid != PAPERS[2])
    assert sc.run() == 2 and sc.calls == []


def test_refused_when_freeze_check_fails(sc, monkeypatch):
    def bad(root):
        raise acc.freeze.FreezeError("changed")

    monkeypatch.setattr(acc.freeze, "assert_pages_frozen", bad)
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_wrong_match_version(sc):
    (sc.d / "tuned_stats.json").write_text(json.dumps({"page_match_version": "1", "threshold": 0.5}))
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_comparability_problem(sc):
    m = dict(MANIFEST, model_digest="tuneddig", prompt="other")
    p = sc.files["crops_tuned"]
    p.with_name("tuned.pred.manifest.json").write_text(json.dumps(m))
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_wrong_tuned_digest(sc):
    m = dict(MANIFEST, model_digest="unregistered")
    sc.files["crops_tuned"].with_name("tuned.pred.manifest.json").write_text(json.dumps(m))
    sc.files["pages_tuned_preds"].with_name("tuned_pages.pred.manifest.json").write_text(json.dumps(m))
    assert sc.run() == 2


def test_refused_on_wrong_row_counts(sc, monkeypatch):
    monkeypatch.setattr(acc, "EXPECTED", {"crops": 31, "pages": 6, "page_rows": 30})
    assert sc.run() == 2 and sc.calls == []
    monkeypatch.setattr(acc, "EXPECTED", {"crops": 30, "pages": 6, "page_rows": 29})
    assert sc.run() == 2 and sc.calls == []


def test_refused_on_results_problem(sc, monkeypatch):
    def drift(pred, out, **kw):
        path = fake_score_file(pred, out, **kw)
        if out.name == "crops_tuned":
            res = json.loads(path.read_text())
            res["scorer_version"] = "9"
            path.write_text(json.dumps(res))
        return path

    monkeypatch.setattr(acc, "score_file", drift)
    assert sc.run() == 2


def test_benchmark_needs_env_check_and_stock_rerun_paths(sc):
    (sc.d / "env_check.json").write_text(json.dumps({"pass": False}))
    assert sc.run() == 2 and sc.calls == []
    (sc.d / "env_check.json").unlink()
    assert sc.run() == 2 and sc.calls == []


def test_dev_mode_skips_pin_lock_freeze_and_paper_checks(sc, monkeypatch):
    monkeypatch.setattr(accmod, "ACCEPTANCE_SHA256", None)
    monkeypatch.setattr(acc, "is_committed_clean", lambda p: False)
    monkeypatch.setattr(acc.splits, "is_benchmark", lambda pid: False)
    sc.lock = {"arms": {}}
    sc.write_lock()
    (sc.d / "env_check.json").unlink()
    monkeypatch.setattr(acc, "EXPECTED", {"crops": 1, "pages": 1, "page_rows": 1})
    assert sc.run("dev") == 0 and len(sc.calls) == 4


def test_dev_mode_still_checks_comparability(sc):
    m = dict(MANIFEST, model_digest="unregistered")
    sc.files["crops_tuned"].with_name("tuned.pred.manifest.json").write_text(json.dumps(m))
    assert sc.run("dev") == 2


def test_ref_drift_exits_3_with_ids(sc, capsys):
    ref = sc.tmp / "ref"
    rows = [json.loads(ln) for ln in sc.files["crops_stock"].read_text().splitlines()]
    recs = [{"id": r["id"], "gt_latex": r["gt_latex"], "pred_latex": r["pred_latex"], "render_identical": r["_ri"]} for r in rows]
    recs[3]["render_identical"] = not recs[3]["render_identical"]
    write_rows(ref / "records.jsonl", recs)
    assert sc.run(**{"ref-score-crops-stock": ref}) == 3
    assert recs[3]["id"] in capsys.readouterr().err


def test_ref_equal_passes_and_ignores_changed_predictions(sc):
    ref = sc.tmp / "ref"
    rows = [json.loads(ln) for ln in sc.files["crops_stock"].read_text().splitlines()]
    recs = [{"id": r["id"], "gt_latex": r["gt_latex"], "pred_latex": r["pred_latex"], "render_identical": r["_ri"]} for r in rows]
    recs[3]["pred_latex"] = "different prediction text"
    recs[3]["render_identical"] = not recs[3]["render_identical"]
    write_rows(ref / "records.jsonl", recs)
    assert sc.run(**{"ref-score-crops-stock": ref}) == 0


def test_render_infrastructure_error_exits_3(sc, monkeypatch):
    def boom(pred, out, **kw):
        raise acc.RenderInfrastructureError("docker down")

    monkeypatch.setattr(acc, "score_file", boom)
    assert sc.run() == 3


def test_no_compare_runs_and_no_shell():
    from pathlib import Path

    text = Path(acc.__file__).read_text()
    assert "compare_runs" not in text and "shell=True" not in text
