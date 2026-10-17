"""CPU llama-server sensitivity tool (fakes only: no server, no GPU, no scorer)."""

import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from lightonocr_math import acceptance as accmod
from scripts import acceptance_sensitivity as sens

RESULTS = {"headline": {"r2w_ids": ["c", "a"], "w2r_ids": ["z", "b"]}}


def test_order_ids_and_chunks():
    assert sens.order_ids(RESULTS) == ["a", "c", "b", "z"]
    assert accmod.ACCEPTANCE["sensitivity"]["chunk"] == sens.CHUNK == 50
    ids = [str(i) for i in range(120)]
    assert [len(c) for c in sens.chunks(ids, sens.CHUNK)] == [50, 50, 20]
    assert [i for c in sens.chunks(ids, sens.CHUNK) for i in c] == ids


def test_server_argv():
    argv = sens.server_argv("/bin/llama-server", "/b/text", "/b/proj", 18081, ["-dev", "none"])
    assert isinstance(argv, list) and argv[0] == "/bin/llama-server"
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
    assert pairs["-m"] == "/b/text" and pairs["--mmproj"] == "/b/proj" and pairs["-c"] == "8192" and pairs["-np"] == "1"
    assert pairs["--flash-attn"] == "auto" and pairs["-b"] == "1024" and pairs["-ub"] == "1024" and pairs["--keep"] == "4"
    assert pairs["--host"] == "127.0.0.1" and pairs["--port"] == "18081" and pairs["-dev"] == "none"
    assert "--context-shift" in argv


def test_request_payload():
    p = sens.request_payload(b"PNG")
    assert p["temperature"] == 0 and p["seed"] == 0 and p["max_tokens"] == 1024
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(p["messages"]) == 1 and p["messages"][0]["role"] == "user"
    content = p["messages"][0]["content"]
    assert len(content) == 1 and content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"PNG").decode()


def test_cpu_device_args(monkeypatch):
    class R:
        stdout = "-dev,  --device <dev1,dev2,..>  comma-separated\n"
        stderr = ""

    monkeypatch.setattr(sens.subprocess, "run", lambda *a, **k: R())
    assert sens.cpu_device_args("x") == ["-dev", "none"]
    R.stdout = "-ngl N\n"
    assert sens.cpu_device_args("x") == ["-ngl", "0"]


# ------------------------------------------------------------------------------------------ run


@pytest.fixture
def world(tmp_path, monkeypatch):
    ids = ["a", "c", "b", "z"]
    rows = []
    for i in ids:
        img = tmp_path / f"{i}.png"
        Image.new("L", (100, 40), 255).save(img)
        rows.append({"id": i, "arxiv_id": f"p-{i}", "latex": f"gt{i}", "image": str(img)})
    rec = tmp_path / "records.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows))
    results = tmp_path / "results.json"
    results.write_text(json.dumps(RESULTS))
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    manifests = {}
    for arm, text in (("stock", "st"), ("tuned", "tu")):
        (blobs / f"sha256-{text}").write_text("x")
        man = tmp_path / f"{arm}.manifest.json"
        man.write_text(json.dumps({"gguf_blobs": {"text": text, "projector": "pj"}}))
        manifests[arm] = man
    (blobs / "sha256-pj").write_text("x")
    state = {"started": [], "stopped": 0, "posts": [], "mem": 64 * 1024**3, "clock": 0.0}

    monkeypatch.setattr(sens, "cpu_device_args", lambda binary: ["-dev", "none"])
    monkeypatch.setattr(sens.freeze, "assert_benchmark_frozen", lambda p: {})
    monkeypatch.setattr(sens.guard, "mem_available_bytes", lambda: state["mem"])
    monkeypatch.setattr(sens, "gpu_state", lambda: {"vram_used_bytes": 100})
    monkeypatch.setattr(sens, "start_server", lambda argv: state["started"].append(argv) or object())
    monkeypatch.setattr(sens, "wait_health", lambda port, timeout=300: None)
    monkeypatch.setattr(sens, "stop_server", lambda proc: state.__setitem__("stopped", state["stopped"] + 1))

    def post(port, payload, timeout=900):
        state["posts"].append(payload)
        state["clock"] += 1.0
        n = len(state["posts"])
        return {"choices": [{"message": {"content": f"```latex\ny{n}\n```"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    monkeypatch.setattr(sens, "post_chat", post)
    monkeypatch.setattr(sens, "_now", lambda: state["clock"])

    def argv(*extra):
        return [
            "run", "--mode", "benchmark", "--results", str(results), "--records", str(rec), "--stock-manifest", str(manifests["stock"]),
            "--tuned-manifest", str(manifests["tuned"]), "--blobs-dir", str(blobs), "--out-dir", str(tmp_path / "sens"), *extra,
        ]

    state["argv"] = argv
    state["results"] = results
    state["out"] = tmp_path / "sens"
    return state


def _rows(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines()]


def test_run_both_arms_in_pinned_order(world):
    assert sens.main(world["argv"]("--chunk", "2")) == 0
    out = world["out"]
    stock, tuned = _rows(out / "cpu_stock.pred.jsonl"), _rows(out / "cpu_tuned.pred.jsonl")
    assert [r["id"] for r in stock] == ["a", "c", "b", "z"] == [r["id"] for r in tuned]
    assert {"id", "paper_id", "gt_latex", "pred_latex", "raw_text", "finish_reason", "prompt_tokens", "completion_tokens", "seconds"} <= stock[0].keys()
    assert stock[0]["gt_latex"] == "gta" and stock[0]["paper_id"] == "p-a" and stock[0]["pred_latex"].startswith("y")
    # chunk alternation: stock then tuned per chunk
    assert len(world["started"]) == 4 and world["stopped"] == 4
    assert "st" in world["started"][0][world["started"][0].index("-m") + 1]
    assert "tu" in world["started"][1][world["started"][1].index("-m") + 1]
    prog = json.loads((out / "progress.json").read_text())
    assert prog["n_done"] == 4 and prog["n_total"] == 4 and prog["stopped_reason"] == "complete"
    assert prog["cpu_option"] == ["-dev", "none"] and "vram_before" in prog and "vram_after" in prog


def test_resume_skips_done_ids(world):
    assert sens.main(world["argv"]("--limit", "2")) == 0
    n = len(world["posts"])
    assert n == 4  # 2 ids x 2 arms
    assert sens.main(world["argv"]()) == 2  # existing predictions need --resume
    assert sens.main(world["argv"]("--resume")) == 0
    assert len(world["posts"]) - n == 4
    assert [r["id"] for r in _rows(world["out"] / "cpu_stock.pred.jsonl")] == ["a", "c", "b", "z"]


def test_max_hours_stops_between_records(world):
    assert sens.main(world["argv"]("--max-hours", "0.0007")) == 0  # 2.52 s: two requests fit
    prog = json.loads((world["out"] / "progress.json").read_text())
    assert prog["stopped_reason"] == "max_hours" and prog["n_done"] < prog["n_total"]
    assert len(world["posts"]) == 3 and world["stopped"] >= 1


def test_low_ram_stops_before_chunk(world):
    world["mem"] = 1024**3
    assert sens.main(world["argv"]()) == 0
    prog = json.loads((world["out"] / "progress.json").read_text())
    assert prog["stopped_reason"] == "low_ram" and world["posts"] == [] and world["started"] == []


def test_missing_blob_is_exit_2(world, tmp_path):
    (tmp_path / "blobs" / "sha256-tu").unlink()
    assert sens.main(world["argv"]()) == 2 and world["started"] == []


def test_results_json_never_written(world):
    before = world["results"].read_bytes()
    sens.main(world["argv"]())
    assert world["results"].read_bytes() == before
    assert not any(p.name == "results.json" for p in world["out"].rglob("*"))


# ------------------------------------------------------------------------------------------ summarize


def test_summarize(world, tmp_path, monkeypatch):
    sens.main(world["argv"]())
    scored = []

    def fake_score(pred, out_dir, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arm = "stock" if "stock" in Path(pred).name else "tuned"
        scored.append((arm, [json.loads(ln)["id"] for ln in Path(pred).read_text().splitlines()]))
        flags = {"stock": {"a": True, "c": True, "b": False, "z": False}, "tuned": {"a": False, "c": True, "b": True, "z": False}}[arm]
        (out_dir / "records.jsonl").write_text("".join(json.dumps({"id": i, "render_identical": v}) + "\n" for i, v in flags.items()))
        return out_dir

    monkeypatch.setattr(sens.score, "score_file", fake_score)
    monkeypatch.setattr(sens, "llama_version", lambda binary: "version: test")
    out = tmp_path / "sensitivity.json"
    rc = sens.main([
        "summarize", "--results", str(world["results"]), "--out-dir", str(world["out"]), "--out", str(out),
        "--stock-manifest", str(tmp_path / "stock.manifest.json"), "--tuned-manifest", str(tmp_path / "tuned.manifest.json"),
    ])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["summary"]["r2w"] == {"total": 2, "done": 2, "held": 1, "reversed": 0, "both_right": 1, "both_wrong": 0, "undecided": 0}
    assert res["summary"]["w2r"]["held"] == 1 and res["summary"]["w2r"]["both_wrong"] == 1
    assert res["summary"]["complete"] is True
    assert res["progress"]["n_done"] == 4 and res["llama_server_version"] == "version: test"
    assert res["blob_digests"]["stock"]["text"] == "st"
    assert all(ids == ["a", "b", "c", "z"] or sorted(ids) == ["a", "b", "c", "z"] for _, ids in scored)
    assert json.loads(world["results"].read_text()) == RESULTS
