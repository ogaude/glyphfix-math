import json
from pathlib import Path

import pytest

from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from scripts import build_trainset as bt

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["prime_l", "decoration", "other"]


def _arxiv_ids(name, n):
    seen = []
    with open(ROOT / "data" / "corpus" / name / "records.jsonl", encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)["arxiv_id"]
            if a not in seen:
                seen.append(a)
            if len(seen) >= n:
                break
    return seen


def _rec(i, paper, latex, flags):
    return {
        "id": f"r{i:05d}", "arxiv_id": paper, "image": f"img/{i}.png", "latex": latex, "flags": flags,
        "width": 100, "height": 40, "license_spdx": "CC-BY-4.0", "license_uri": "http://x", "paper_version": "v1",
    }  # fmt: skip


@pytest.fixture
def inputs(tmp_path):
    train_p, dev_p = _arxiv_ids("train", 40), _arxiv_ids("dev", 10)
    recs, labels = [], []
    glyphs = ["x'", "I", "l", "1"]
    for i in range(160):
        flags = [("plain", "prime", "greek", "decoration", "i_1")[i % 5]]
        err = i < 40
        latex = f"e_{i} + y" if err else (f"c_{i} + {glyphs[i % 4]}" if i % 2 else f"c_{i} + z")
        r = _rec(i, train_p[i % 40], latex, flags)
        recs.append(r)
        cls = CLASSES[i % 2] if err else None
        labels.append(
            {"id": r["id"], "paper_id": r["arxiv_id"], "render_identical": not err, "exact": not err, "classes": [],
             "error": err, "clean": not err, "mining_class": cls, "tags": [], "flags": flags, "in_regression": False}
        )  # fmt: skip
    dev = [_rec(1000 + i, dev_p[i % 10], f"d_{i}", [("plain", "prime", "greek")[i % 3]]) for i in range(60)]
    reg = [{"id": "reg1", "paper_id": train_p[0], "gt_latex": "q + q"}]
    tax = {"reviewed": True, "reviewer": "t", "reviewed_at": "now", "classes": CLASSES}
    d = tmp_path / "in"
    d.mkdir()
    write_jsonl(d / "train.jsonl", recs)
    write_jsonl(d / "labels.jsonl", labels)
    write_jsonl(d / "dev.jsonl", dev)
    write_jsonl(d / "reg.jsonl", reg)
    write_json(d / "tax.json", tax)
    return d


def _argv(d, out, *extra):
    return [
        "build", "--train", str(d / "train.jsonl"), "--labels", str(d / "labels.jsonl"),
        "--taxonomy", str(d / "tax.json"), "--regression", str(d / "reg.jsonl"), "--dev", str(d / "dev.jsonl"),
        "--total", "50", "--token-counter", "chars", "--out-dir", str(out), *extra,
    ]  # fmt: skip


def test_build_writes_set_and_hashed_manifest(inputs, tmp_path):
    out = tmp_path / "set"
    assert bt.main(_argv(inputs, out)) == 0
    m = json.loads((out / "TRAINSET_MANIFEST.json").read_text())
    assert m["trainset_sha256"] == sha256_file(out / "trainset.jsonl")
    assert m["n_rows"] == len(read_jsonl(out / "trainset.jsonl")) == 50
    assert m["sources"]["labels_sha256"] == sha256_file(inputs / "labels.jsonl")
    for k, f in (("train_records_sha256", "train.jsonl"), ("taxonomy_sha256", "tax.json"),
                 ("regression_sha256", "reg.jsonl"), ("dev_records_sha256", "dev.jsonl")):  # fmt: skip
        assert m["sources"][k] == sha256_file(inputs / f)
    assert m["tokenizer"]["model_id"] and m["tokenizer"]["revision"]
    assert m["scorer_version"] and m["scorer_fingerprint"]
    assert set(m["shortfalls"]) <= {*CLASSES, "control", "replay"}


def test_build_refuses_unreviewed_taxonomy(inputs, tmp_path, capsys):
    write_json(inputs / "tax.json", {"reviewed": False, "classes": CLASSES})
    out = tmp_path / "set"
    assert bt.main(_argv(inputs, out)) == 2
    assert "refused:" in capsys.readouterr().err
    assert not (out / "trainset.jsonl").exists()


def test_build_refuses_benchmark_paper(inputs, tmp_path, capsys):
    recs = read_jsonl(inputs / "train.jsonl")
    recs[0]["arxiv_id"] = "2604.22670"
    write_jsonl(inputs / "train.jsonl", recs)
    assert bt.main(_argv(inputs, tmp_path / "set")) == 2
    assert "refused:" in capsys.readouterr().err


def test_build_refuses_existing_set_without_force(inputs, tmp_path):
    out = tmp_path / "set"
    assert bt.main(_argv(inputs, out)) == 0
    assert bt.main(_argv(inputs, out)) == 2
    assert bt.main(_argv(inputs, out, "--force")) == 0


def test_build_missing_input_exits_2(inputs, tmp_path):
    (inputs / "labels.jsonl").unlink()
    assert bt.main(_argv(inputs, tmp_path / "set")) == 2


def test_subset_records_parent_hash(inputs, tmp_path):
    out, sub = tmp_path / "set", tmp_path / "sub"
    assert bt.main(_argv(inputs, out)) == 0
    assert bt.main(["subset", "--from", str(out), "--n", "20", "--out-dir", str(sub)]) == 0
    pm = json.loads((out / "TRAINSET_MANIFEST.json").read_text())
    sm = json.loads((sub / "TRAINSET_MANIFEST.json").read_text())
    assert sm["parent_sha256"] == pm["trainset_sha256"]
    assert sm["n_rows"] == 20 and sm["trainset_sha256"] == sha256_file(sub / "trainset.jsonl")
    parent_ids = {r["id"] for r in read_jsonl(out / "trainset.jsonl")}
    assert {r["id"] for r in read_jsonl(sub / "trainset.jsonl")} <= parent_ids


def test_subset_refuses_tampered_parent(inputs, tmp_path):
    out = tmp_path / "set"
    assert bt.main(_argv(inputs, out)) == 0
    with open(out / "trainset.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert bt.main(["subset", "--from", str(out), "--n", "20", "--out-dir", str(tmp_path / "sub")]) == 2


def test_devsubset_writes_sorted_ids_and_manifest(inputs, tmp_path):
    out = tmp_path / "t" / "dev_eval_ids.txt"
    assert bt.main(["devsubset", "--dev", str(inputs / "dev.jsonl"), "--n", "30", "--out", str(out)]) == 0
    ids = out.read_text().splitlines()
    assert ids == sorted(ids) and len(ids) == 30
    assert {r["id"] for r in read_jsonl(inputs / "dev.jsonl")} >= set(ids)
    m = json.loads((out.parent / "DEV_EVAL_MANIFEST.json").read_text())
    assert m["n"] == 30 and m["max_per_paper"] == 3 and m["seed"] == 0
    assert m["dev_records_sha256"] == sha256_file(inputs / "dev.jsonl")
    assert m["ids_sha256"] == sha256_file(out)
