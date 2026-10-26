"""Ollama driver and comparator with a fake urlopen (no network)."""

import base64
import json

from glyphfix_math import score
from scripts.spikes import common, compare_raw, ollama_infer

F = "`" * 3


class _Resp:
    status = 200

    def __init__(self, obj):
        self._b = json.dumps(obj).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_infer_one_request_shape(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        return _Resp({"message": {"content": "x"}})

    monkeypatch.setattr(ollama_infer.urllib.request, "urlopen", fake)
    opts = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
    ollama_infer.infer_one("http://h", "m", b"PNGBYTES", opts, "10m")
    assert len(seen) == 1
    assert seen[0].full_url == "http://h/api/chat"
    body = json.loads(seen[0].data)
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == opts
    assert body["messages"][0]["content"] == ""
    assert body["messages"][0]["images"] == [base64.b64encode(b"PNGBYTES").decode()]


def _setup(tmp_path, monkeypatch, fail_second=False):
    rows = []
    for i in (1, 2):
        img = tmp_path / f"{i}.png"
        img.write_bytes(b"png%d" % i)
        rows.append({"id": f"p{i}_eq0", "arxiv_id": f"p{i}", "latex": f"x{i}", "image_path": str(img), "width": 1, "height": 2})
    monkeypatch.setattr(common, "load_legacy", lambda split, limit=None: (rows, {"total": 2, "kept": 2}))
    monkeypatch.setattr(ollama_infer.guard, "preflight", lambda **k: None)
    monkeypatch.setattr(common, "gpu_state", dict)
    n = [0]

    def fake(req, timeout=None):
        url = req.full_url
        if url.endswith("/api/chat"):
            body = json.loads(req.data)
            if not body["messages"]:
                return _Resp({})
            n[0] += 1
            if fail_second and n[0] == 2:
                raise ollama_infer.urllib.error.HTTPError(url, 500, "boom", {}, None)
            return _Resp(
                {"message": {"content": F + f"latex\ny{n[0]}\n" + F}, "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 3}
            )
        if url.endswith("/api/ps"):
            return _Resp({"models": []})
        if url.endswith("/api/version"):
            return _Resp({"version": "0"})
        return _Resp({"models": []})

    monkeypatch.setattr(ollama_infer.urllib.request, "urlopen", fake)


def test_main_writes_outputs(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out, side = tmp_path / "o.jsonl", tmp_path / "s.jsonl"
    assert ollama_infer.main(["--model", "m", "--out", str(out), "--sidecar", str(side)]) == 0
    preds = score.read_predictions(out)
    assert [p["pred_latex"] for p in preds] == ["y1", "y2"]
    assert [p["gt_latex"] for p in preds] == ["x1", "x2"]
    sc = common.read_jsonl(side)
    assert sc[0]["raw_text"].startswith(F)
    assert {"done_reason", "prompt_eval_count", "eval_count", "image_sha256"} <= sc[0].keys()
    assert (tmp_path / "o.manifest.json").exists()


def test_http_error_continues(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, fail_second=True)
    out, side = tmp_path / "o.jsonl", tmp_path / "s.jsonl"
    assert ollama_infer.main(["--model", "m", "--out", str(out), "--sidecar", str(side)]) == 1
    preds = common.read_jsonl(out)
    assert "pred_latex" in preds[0] and "pred_error" in preds[1] and "pred_latex" not in preds[1]


def test_compare():
    a = [{"id": "a", "raw_text": "x", "prompt_eval_count": 1}, {"id": "b", "raw_text": "y", "prompt_eval_count": 1}]
    r = compare_raw.compare(a, a)
    assert r["n"] == 2 and r["n_identical"] == 2
    b = [{"id": "a", "raw_text": "x", "prompt_eval_count": 2}, {"id": "b", "raw_text": "z", "prompt_eval_count": 1}, {"id": "c", "raw_text": "", "prompt_eval_count": 1}]
    r = compare_raw.compare(a, b)
    assert r["differing_ids"] == ["b"] and r["prompt_count_mismatch_ids"] == ["a"] and r["missing_ids"] == ["c"]
