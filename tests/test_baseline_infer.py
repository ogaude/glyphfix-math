"""baseline_infer with a fake Ollama (no network, no GPU)."""

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from lightonocr_math import splits
from scripts import baseline_infer as bi
from scripts.spikes import ollama_infer
from tests.test_ollama_infer import _Resp

F = "`" * 3
D12_KEYS = {
    "model_digest", "gguf_blobs", "template_sha256", "prompt_sha256", "ollama_version", "options", "pixel_policy",
    "scorer_version", "scorer_fingerprint", "render_config", "split_rule", "records_sha256", "git_commit",
    "gpu_before", "gpu_after",
}


def _png(path, size=(100, 40)):
    Image.new("L", size, 255).save(path)


def _rows(tmp_path, kind="crop", n=2):
    rows = []
    for i in range(1, n + 1):
        img = tmp_path / f"{i}.png"
        _png(img)
        if kind == "crop":
            rows.append({"id": f"p{i}_eq0", "arxiv_id": f"2501.0000{i}", "latex": f"x{i}", "image": str(img), "width": 100, "height": 40})
        else:
            rows.append({"page_id": f"p{i}_pg1", "arxiv_id": f"2501.0000{i}", "image": str(img)})
    rec = tmp_path / "records.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rec


@pytest.fixture
def fake(tmp_path, monkeypatch):
    calls = {"chat": [], "unload": 0, "loaded": []}
    monkeypatch.setattr(bi.guard, "preflight", lambda **k: None)
    monkeypatch.setattr(bi.guard, "mem_available_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(bi, "gpu_state", dict)
    monkeypatch.setattr(bi.splits, "is_benchmark", lambda i, fs=None: True)
    monkeypatch.setattr(bi.splits, "assert_disjoint", lambda ids, context, fs=None: None)
    monkeypatch.setattr(bi.splits, "rule_info", lambda: {"split_rule_version": "1", "split_digest": "d"})
    n = [0]

    def urlopen(req, timeout=None):
        url = req.full_url
        if url.endswith("/api/chat"):
            body = json.loads(req.data)
            if not body["messages"]:
                calls["unload"] += 1
                return _Resp({})
            n[0] += 1
            calls["chat"].append(body)
            return _Resp({"message": {"content": F + f"latex\ny{n[0]}\n" + F}, "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 3})
        if url.endswith("/api/ps"):
            return _Resp({"models": [{"name": m} for m in calls["loaded"]]})
        if url.endswith("/api/version"):
            return _Resp({"version": "0.40.1"})
        if url.endswith("/api/show"):
            return _Resp({"modelfile": "FROM /x/blobs/sha256-aaa\nFROM /x/blobs/sha256-bbb\nTEMPLATE x", "template": "{{ .Prompt }}"})
        return _Resp({"models": [{"name": "m", "digest": "dig"}]})

    monkeypatch.setattr(ollama_infer.urllib.request, "urlopen", urlopen)
    return calls


def _argv(tmp_path, rec, suite="benchmark", *extra):
    return ["--suite", suite, "--records", str(rec), "--allow-unfrozen-records", "--model", "m", "--out", str(tmp_path / "o.jsonl"), "--sidecar", str(tmp_path / "s.jsonl"), *extra]


def test_crop_run(tmp_path, fake):
    rec = _rows(tmp_path)
    assert bi.main(_argv(tmp_path, rec)) == 0
    preds = [json.loads(ln) for ln in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert [p["pred_latex"] for p in preds] == ["y1", "y2"]
    assert preds[0]["gt_latex"] == "x1" and preds[0]["paper_id"] == "2501.00001"
    side = [json.loads(ln) for ln in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert {"raw_text", "prompt_eval_count", "eval_count", "done_reason", "sent_width", "sent_height", "image_sha256"} <= side[0].keys()
    assert side[0]["sent_width"] % 32 == 0 and side[0]["sent_height"] % 32 == 0
    sent = json.loads(json.dumps(fake["chat"][0]))
    img = Image.open(io.BytesIO(base64.b64decode(sent["messages"][0]["images"][0])))
    assert img.size == (side[0]["sent_width"], side[0]["sent_height"])
    assert sent["options"] == bi.CROP_OPTIONS and sent["think"] is False and sent["messages"][0]["content"] == ""
    man = json.loads((tmp_path / "o.manifest.json").read_text())
    assert D12_KEYS <= man.keys()
    assert man["gguf_blobs"] == {"text": "aaa", "projector": "bbb"}
    assert man["pixel_policy"]["max_px"] == 602112
    assert fake["unload"] == 1


def test_pages_run(tmp_path, fake):
    rec = _rows(tmp_path, kind="page")
    assert bi.main(_argv(tmp_path, rec, "pages")) == 0
    preds = [json.loads(ln) for ln in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert preds[0]["id"] == "p1_pg1" and "pred_text" in preds[0]
    assert fake["chat"][0]["options"] == bi.PAGE_OPTIONS
    man = json.loads((tmp_path / "o.manifest.json").read_text())
    assert man["pixel_policy"]["name"] == "page-2048"


def test_resume_skips_done(tmp_path, fake):
    rec = _rows(tmp_path, n=3)
    assert bi.main(_argv(tmp_path, rec, "benchmark", "--limit", "1")) == 0
    n_before = len(fake["chat"])
    assert bi.main(_argv(tmp_path, rec, "benchmark", "--resume")) == 0
    assert len(fake["chat"]) - n_before == 2
    preds = [json.loads(ln) for ln in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert [p["id"] for p in preds] == ["p1_eq0", "p2_eq0", "p3_eq0"]
    assert json.loads((tmp_path / "o.manifest.json").read_text())["n_skipped_resume"] == 1


def test_low_ram_stops_with_76(tmp_path, fake, monkeypatch):
    rec = _rows(tmp_path)
    monkeypatch.setattr(bi.guard, "mem_available_bytes", lambda: 1024**3)
    assert bi.main(_argv(tmp_path, rec)) == 76
    assert fake["chat"] == []


def test_preloaded_model_not_unloaded(tmp_path, fake):
    fake["loaded"].append("m")
    rec = _rows(tmp_path)
    assert bi.main(_argv(tmp_path, rec)) == 0
    assert fake["unload"] == 0


def test_benchmark_suite_rejects_training_paper(tmp_path, fake, monkeypatch):
    rec = _rows(tmp_path)
    monkeypatch.setattr(bi.splits, "is_benchmark", lambda i, fs=None: i == "2501.00001")
    with pytest.raises(ValueError):
        bi.main(_argv(tmp_path, rec, "benchmark"))


def test_regression_suite_rejects_benchmark_paper(tmp_path, fake, monkeypatch):
    rec = _rows(tmp_path)

    def boom(ids, context, fs=None):
        raise splits.LeakageError("leak", {"2501.00001": "benchmark-list"})

    monkeypatch.setattr(bi.splits, "assert_disjoint", boom)
    with pytest.raises(splits.LeakageError):
        bi.main(_argv(tmp_path, rec, "regression"))


def _first_paper(split):
    if not Path(f"data/corpus/{split}/records.jsonl").is_file():
        pytest.skip("reserved train and dev rows are not part of the public repository")
    with open(f"data/corpus/{split}/records.jsonl", encoding="utf-8") as fh:
        return json.loads(fh.readline())["arxiv_id"]


def _bench_paper():
    return min(splits.frozen().benchmark)


def test_mining_suite_accepts_train_only():
    train = {"id": "a", "arxiv_id": _first_paper("train")}
    assert bi.load_suite("mining", [train]) == [train]
    dev = {"id": "b", "arxiv_id": _first_paper("dev")}
    with pytest.raises(ValueError, match=_first_paper("dev")):
        bi.load_suite("mining", [train, dev])
    with pytest.raises(splits.LeakageError):
        bi.load_suite("mining", [train, {"id": "c", "arxiv_id": _bench_paper()}])


def test_dev_suite_accepts_dev_only():
    dev = {"id": "b", "arxiv_id": _first_paper("dev")}
    assert bi.load_suite("dev", [dev]) == [dev]
    with pytest.raises(ValueError, match="not dev-split"):
        bi.load_suite("dev", [dev, {"id": "a", "arxiv_id": _first_paper("train")}])


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("data/mining/stock_train.pred.jsonl", True),
        ("data/train/runs/x/ckpt-1/a.bin", True),
        ("data/models/r1/text-q8_0.gguf", True),
        ("data/runs/a.pred.jsonl", True),
        ("data/mining/taxonomy.json", False),
        ("data/mining/labels.jsonl", False),
        ("data/mining/MINING_MANIFEST.json", False),
        ("data/mining/taxonomy_candidates.json", False),
        ("data/models/REGISTRY.json", False),
        ("data/train/sets/full/trainset.jsonl", False),
    ],
)
def test_phase4_gitignore(path, ignored):
    import subprocess

    rc = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], check=False).returncode
    assert (rc == 0) is ignored
