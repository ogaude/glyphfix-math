import json
from pathlib import Path

import pytest

from glyphfix_math import splits
from glyphfix_math.corpus._io import read_jsonl, sha256_file, write_jsonl
from glyphfix_math.train import data
from scripts import train_lora as tl

ROOT = Path(__file__).resolve().parents[1]


def _real_rows(name: str, n: int) -> list[dict]:
    if not (ROOT / "data" / "corpus" / name / "records.jsonl").is_file():
        pytest.skip("reserved train and dev rows are not part of the public repository")
    out, seen = [], set()
    with open(ROOT / "data" / "corpus" / name / "records.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arxiv_id"] in seen:
                continue
            seen.add(r["arxiv_id"])
            out.append(r)
            if len(out) >= n:
                break
    return out


def _train_row(rec: dict) -> dict:
    return {
        "id": rec["id"], "paper_id": rec["arxiv_id"], "image": rec["image"], "latex": rec["latex"],
        "target_text": data.target_text(rec["latex"]), "group": "clean", "width": rec["width"], "height": rec["height"],
    }  # fmt: skip


def _write_set(tmp_path: Path, rows: list[dict], *, sha: str | None = None) -> Path:
    p = tmp_path / "trainset.jsonl"
    write_jsonl(p, rows)
    (tmp_path / tl.MANIFEST_FILE).write_text(json.dumps({"trainset_sha256": sha or sha256_file(p)}), encoding="utf-8")
    return p


def test_verify_accepts_clean_set(tmp_path):
    rows = [_train_row(r) for r in _real_rows("train", 3)]
    p = _write_set(tmp_path, rows)
    got, manifest = tl.verify_trainset(p, dev_ids=set(), eval_ids=set(), regression_ids=set())
    assert [r["id"] for r in got] == [r["id"] for r in rows]
    assert manifest["trainset_sha256"] == sha256_file(p)


def test_verify_rejects_hash_mismatch(tmp_path):
    p = _write_set(tmp_path, [_train_row(r) for r in _real_rows("train", 2)], sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        tl.verify_trainset(p, dev_ids=set(), eval_ids=set(), regression_ids=set())


@pytest.mark.parametrize("which", ["dev", "eval", "regression"])
def test_verify_rejects_listed_ids(tmp_path, which):
    rows = [_train_row(r) for r in _real_rows("train", 2)]
    p = _write_set(tmp_path, rows)
    sets = {"dev_ids": set(), "eval_ids": set(), "regression_ids": set()}
    sets[{"dev": "dev_ids", "eval": "eval_ids", "regression": "regression_ids"}[which]] = {rows[0]["id"]}
    with pytest.raises(splits.LeakageError):
        tl.verify_trainset(p, **sets)


def test_verify_rejects_regression_latex_overlap(tmp_path):
    rows = [_train_row(r) for r in _real_rows("train", 2)]
    p = _write_set(tmp_path, rows)
    reg = [{"id": "other-id", "gt_latex": rows[0]["latex"]}]
    with pytest.raises(splits.LeakageError):
        tl.verify_trainset(p, dev_ids=set(), eval_ids=set(), regression_ids=set(), regression_rows=reg)


def test_verify_rejects_dev_paper_row(tmp_path):
    rows = [_train_row(r) for r in _real_rows("dev", 1)]
    p = _write_set(tmp_path, rows)
    with pytest.raises(splits.LeakageError):
        tl.verify_trainset(p, dev_ids=set(), eval_ids=set(), regression_ids=set())


def test_cmd_train_refuses_when_ollama_busy(monkeypatch, capsys):
    monkeypatch.setattr(tl, "ollama_busy", lambda host=tl.DEFAULT_HOST: ["lighton-ocr3-q8-16k"])
    args = tl.make_parser().parse_args(["train", "--run-id", "x", "--lr", "2e-4"])
    assert tl.cmd_train(args) == 75
    assert "lighton-ocr3-q8-16k" in capsys.readouterr().err


def test_cmd_train_refuses_without_gpu(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    rows = [_train_row(r) for r in _real_rows("train", 2)]
    p = _write_set(tmp_path, rows)
    monkeypatch.setattr(tl, "ollama_busy", lambda host=tl.DEFAULT_HOST: [])
    monkeypatch.setattr(tl, "_context_sets", lambda root: (set(), set(), set(), []))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = tl.make_parser().parse_args(["train", "--run-id", "x", "--lr", "2e-4", "--trainset", str(p)])
    assert tl.cmd_train(args) == 2


def test_cmd_train_refuses_hash_mismatch(monkeypatch, tmp_path):
    p = _write_set(tmp_path, [_train_row(r) for r in _real_rows("train", 2)], sha="f" * 64)
    monkeypatch.setattr(tl, "ollama_busy", lambda host=tl.DEFAULT_HOST: [])
    monkeypatch.setattr(tl, "_context_sets", lambda root: (set(), set(), set(), []))
    args = tl.make_parser().parse_args(["train", "--run-id", "x", "--lr", "2e-4", "--trainset", str(p)])
    assert tl.cmd_train(args) == 2


def test_load_eval_records_refuses_non_dev_paper(tmp_path):
    train_rec = _real_rows("train", 1)[0]
    recs = tmp_path / "records.jsonl"
    write_jsonl(recs, [train_rec])
    ids = tmp_path / "ids.txt"
    ids.write_text(train_rec["id"] + "\n", encoding="utf-8")
    with pytest.raises(tl.Refused):
        tl.load_eval_records(ids, recs)


def test_load_eval_records_accepts_dev(tmp_path):
    dev = _real_rows("dev", 2)
    recs = tmp_path / "records.jsonl"
    write_jsonl(recs, dev)
    ids = tmp_path / "ids.txt"
    ids.write_text("".join(r["id"] + "\n" for r in dev), encoding="utf-8")
    assert [r["id"] for r in tl.load_eval_records(ids, recs)] == [r["id"] for r in dev]


def test_cmd_eval_exit_2_on_non_dev_ids(monkeypatch, tmp_path):
    train_rec = _real_rows("train", 1)[0]
    recs = ROOT / "data" / "corpus" / "train" / "records.jsonl"
    ids = tmp_path / "ids.txt"
    ids.write_text(train_rec["id"] + "\n", encoding="utf-8")
    monkeypatch.setattr(tl, "ollama_busy", lambda host=tl.DEFAULT_HOST: [])
    args = tl.make_parser().parse_args(["eval", "--base", "--ids", str(ids), "--records", str(recs), "--out-dir", str(tmp_path)])
    assert tl.cmd_eval(args) == 2


def test_eval_row_keys_and_fence_stripping():
    rec = {"id": "a", "arxiv_id": "2301.01234v2", "latex": "x'"}
    res = {"text": "```latex\nx'\n```", "n_prompt_tokens": 90, "image_tokens": 40, "grid_thw": [[1, 8, 16]], "finish": "eos",
           "n_new_tokens": 9, "resized_hw": [128, 256]}  # fmt: skip
    pred, side = tl.eval_row(rec, res)
    assert set(pred) == {"id", "paper_id", "gt_latex", "pred_latex"}
    assert pred["pred_latex"] == "x'" and pred["paper_id"] == "2301.01234"
    assert side["sent_width"] == 256 and side["sent_height"] == 128 and side["raw_text"].startswith("```")


def test_generate_file_resume_skips_written(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from glyphfix_math.train import infer

    calls = []

    def fake_generate(model, processor, image, *, min_px, max_px, max_new_tokens):
        calls.append(1)
        return {"text": "x", "n_prompt_tokens": 1, "image_tokens": 1, "grid_thw": [[1, 2, 2]], "finish": "eos", "n_new_tokens": 1,
                "resized_hw": [32, 32]}  # fmt: skip

    monkeypatch.setattr(infer, "generate_one", fake_generate)
    monkeypatch.setattr(data, "load_image", lambda row, root=None: None)
    recs = [{"id": f"r{i}", "arxiv_id": "2301.01234", "latex": "x", "image": "i.png"} for i in range(3)]
    stem = tmp_path / "step-1"
    assert tl.generate_file(None, None, recs[:2], stem, resume=False) == 2
    assert tl.generate_file(None, None, recs, stem, resume=True) == 1
    rows = read_jsonl(tmp_path / "step-1.pred.jsonl")
    assert [r["id"] for r in rows] == ["r0", "r1", "r2"]
    assert len(calls) == 3


def test_train_args_build_config_with_defaults():
    args = tl.make_parser().parse_args(
        ["train", "--lr", "2e-4", "--alpha", "8", "--rank", "32", "--epochs", "1", "--run-id", "x", "--trainset", "p"]
    )
    cfg = tl.build_config(args)
    assert (cfg.lr, cfg.alpha, cfg.rank, cfg.epochs, cfg.run_id, cfg.trainset) == (2e-4, 8, 32, 1, "x", "p")
    assert (cfg.batch_size, cfg.grad_accum, cfg.min_px, cfg.max_px) == (2, 8, 65536, 602112)


def test_step_dirs_filters(tmp_path):
    for n in (100, 200, 300):
        (tmp_path / "eval" / f"step-{n}" / "adapter").mkdir(parents=True)
    assert [s for s, _ in tl._step_dirs(tmp_path, "all")] == [100, 200, 300]
    assert [s for s, _ in tl._step_dirs(tmp_path, "200,300")] == [200, 300]
