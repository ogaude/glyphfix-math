"""Spike common module: fence stripping, legacy loader filters, atomic IO (no network, no GPU)."""

import json
from pathlib import Path

import pytest

from scripts.spikes import common

F = "`" * 3


def test_strip_fences():
    assert common.strip_fences(F + "latex\nx+1\n" + F) == "x+1"
    assert common.strip_fences(F + "\nx\n" + F + "\n") == "x"
    assert common.strip_fences("x+1") == "x+1"
    assert common.strip_fences(F + "latex\na\nb\n" + F) == "a\nb"
    assert common.strip_fences(F + "latex\n  a \t b\n" + F) == "  a \t b"
    assert common.strip_fences(F + "latex x^2" + F) == "x^2"
    assert common.strip_fences(F + "x^2" + F) == "x^2"
    assert common.strip_fences(F + "latex x^2") == "x^2"


def _write_rows(root: Path, split: str, papers: list[str]) -> None:
    rows = [
        {"id": f"{p}_eq0", "arxiv_id": p, "latex": "x", "image": f"img/{p}.png", "split": split} for p in papers
    ]
    (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_refuses_heldout(tmp_path):
    with pytest.raises(ValueError, match="eval_heldout"):
        common.load_legacy("eval_heldout", root=tmp_path)


def test_drops_benchmark(tmp_path, monkeypatch):
    _write_rows(tmp_path, "dev", ["2501.00001", "2501.00002", "2501.00003"])
    monkeypatch.setattr(common.splits, "is_benchmark", lambda i, fs=None: i == "2501.00002")
    rows, stats = common.load_legacy("dev", root=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["2501.00001", "2501.00003"]
    assert stats == {"total": 3, "kept": 2, "dropped_benchmark": 1, "dropped_quarantine": 0}


def test_train_drops_quarantine(tmp_path, monkeypatch):
    _write_rows(tmp_path, "train", ["2501.00001", "2501.00002"])
    monkeypatch.setattr(common.splits, "is_benchmark", lambda i, fs=None: False)
    monkeypatch.setattr(common.splits, "load_train_ids", lambda ids, fs=None: ["2501.00001"])
    rows, stats = common.load_legacy("train", root=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["2501.00001"]
    assert stats["dropped_quarantine"] == 1


def test_jsonl_roundtrip_and_atomic(tmp_path):
    p = tmp_path / "a.jsonl"
    rows = [{"id": "a", "t": "é"}, {"id": "b"}]
    common.write_jsonl(p, rows)
    assert common.read_jsonl(p) == rows

    class Boom:
        def __iter__(self):
            raise RuntimeError

    q = tmp_path / "b.jsonl"
    with pytest.raises(RuntimeError):
        common.write_jsonl(q, Boom())  # type: ignore[arg-type]
    assert not q.exists()
    assert list(tmp_path.glob("*.tmp")) == []
