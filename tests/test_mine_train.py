import json
from pathlib import Path

import pytest

from lightonocr_math.pixels import PIXEL_POLICY
from scripts import baseline_infer
from scripts import mine_train as mt

OPTIONS = dict(baseline_infer.CROP_OPTIONS)


def _first_papers(split, n):
    out = []
    with open(f"data/corpus/{split}/records.jsonl", encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)["arxiv_id"]
            if p not in out:
                out.append(p)
            if len(out) == n:
                break
    return out


TRAIN = _first_papers("train", 6)
DEV = _first_papers("dev", 1)[0]


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _train_rows(papers=TRAIN, per=2):
    return [
        {"id": f"{p}_{i}", "arxiv_id": p, "latex": f"x_{i} + {j}", "flags": ["plain"], "image": "i.png"}
        for j, p in enumerate(papers)
        for i in range(per)
    ]


def _manifest(digest="D"):
    return {"model": "m", "model_digest": digest, "template_sha256": "T", "options": OPTIONS, "pixel_policy": dict(PIXEL_POLICY)}


def _pred(rows):
    return [{"id": r["id"], "paper_id": r.get("arxiv_id") or r["paper_id"], "gt_latex": "g", "pred_latex": "p"} for r in rows]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "_model_digest", lambda host, model: "D")
    train = tmp_path / "train.jsonl"
    _w(train, _train_rows())
    pool = tmp_path / "pool"
    dev_row = {"id": "devrow", "paper_id": DEV, "gt_latex": "g", "pred_latex": "p"}
    pool_rows = _pred(_train_rows()[:5]) + [dev_row]
    _w(pool / "stock_pool.pred.jsonl", pool_rows)
    _w(pool / "stock_pool.sidecar.jsonl", [{"id": r["id"], "raw_text": "p"} for r in pool_rows])
    (pool / "stock_pool.pred.manifest.json").write_text(json.dumps(_manifest()))
    stock = tmp_path / "stock.manifest.json"
    stock.write_text(json.dumps(_manifest()))
    mdir = tmp_path / "mining"
    return {
        "tmp": tmp_path, "train": train, "pool": pool, "stock": stock, "mdir": mdir,
        "args": ["--train", str(train), "--pool-dir", str(pool), "--stock-manifest", str(stock),
                 "--out", str(mdir / "p.jsonl"), "--sidecar", str(mdir / "s.jsonl"),
                 "--score-dir", str(mdir / "score"), "--candidates", str(mdir / "cands.json"),
                 "--taxonomy", str(mdir / "tax.json"), "--labels", str(mdir / "labels.jsonl"),
                 "--manifest", str(mdir / "MM.json"), "--regression", str(tmp_path / "reg.jsonl"),
                 "--report", str(tmp_path / "MINING.md")],
    }  # fmt: skip


def test_seed_copies_train_rows_only(env):
    assert mt.main(["seed", *env["args"]]) == 0
    out = [json.loads(x) for x in (env["mdir"] / "p.jsonl").read_text().splitlines()]
    assert len(out) == 5 and all(r["paper_id"] in TRAIN for r in out)
    side = (env["mdir"] / "s.jsonl").read_text().splitlines()
    assert len(side) == 5
    carried = json.loads((env["mdir"] / "carried_over.json").read_text())
    assert carried["n"] == 5 and len(carried["ids_sha256"]) == 64 and len(carried["pool_manifest_sha256"]) == 64
    assert mt.main(["seed", *env["args"]]) == 2  # never overwrites


def test_seed_digest_mismatch_writes_nothing(env, monkeypatch, capsys):
    monkeypatch.setattr(mt, "_model_digest", lambda host, model: "OTHER")
    assert mt.main(["seed", *env["args"]]) == 2
    assert "refused:" in capsys.readouterr().err
    assert not (env["mdir"] / "p.jsonl").exists()


def test_infer_chunks_and_stops_on_failure(env, monkeypatch):
    calls = []

    def fake(argv):
        calls.append(argv)
        ids = Path(argv[argv.index("--ids-file") + 1]).read_text().split()
        out = env["mdir"] / "p.jsonl"
        prev = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        by_id = {r["id"]: r for r in _train_rows()}
        prev += _pred([{"id": i, "paper_id": by_id[i]["arxiv_id"]} for i in ids])
        _w(out, prev)
        (env["mdir"] / "p.manifest.json").write_text(json.dumps(_manifest()))
        return 0 if len(calls) < 2 else 76

    monkeypatch.setattr(mt.baseline_infer, "main", fake)
    assert mt.main(["infer", "--chunk", "5", *env["args"]]) == 76
    assert len(calls) == 2
    assert calls[0][calls[0].index("--suite") + 1] == "mining" and "--resume" in calls[0]
    assert sorted(p.name for p in (env["mdir"] / "chunks").iterdir()) == ["chunk-00.txt", "chunk-01.txt"]
    assert (env["mdir"] / "manifests" / "chunk-00.manifest.json").exists()
    ids0 = (env["mdir"] / "chunks" / "chunk-00.txt").read_text().split()
    assert ids0 == sorted(ids0) and len(ids0) == 5


def _scored(env):
    rows = []
    train = _train_rows()
    for k, t in enumerate(train):
        rows.append({"id": t["id"], "paper_id": t["arxiv_id"], "gt_latex": "a \\int b", "pred_latex": "a \\oint b",
                     "exact": False, "render_identical": False, "classes": ["other"]})  # fmt: skip
    rows[0].update(render_identical=True, exact=True, pred_latex="a \\int b", classes=[])
    rows[1].update(render_identical=None)
    rows[2].update(classes=["decoration"])
    _w(env["mdir"] / "score" / "records.jsonl", rows)
    _w(env["tmp"] / "reg.jsonl", [{"id": train[3]["id"], "gt_latex": "zzz"}])
    return rows


def test_report_then_accept(env):
    _scored(env)
    assert mt.main(["report", *env["args"]]) == 0
    cands = json.loads((env["mdir"] / "cands.json").read_text())
    assert [c["id"] for c in cands["candidates"]] == ["c01"]
    c1 = cands["candidates"][0]
    assert c1["gt_tokens"] == ["\\int"] and c1["n_papers"] == 5
    assert cands["available_for_training"]["c01"] == c1["n_records"] - 1  # one record is in the regression set
    md = (env["tmp"] / "MINING.md").read_text()
    assert "| c01 |" in md and md.rstrip().splitlines()[-1].endswith("|  |")
    mm = json.loads((env["mdir"] / "MM.json").read_text())
    assert mm["n_error"] == 10 and mm["n_clean"] == 1 and mm["n_gt_uncompilable"] == 1

    assert mt.main(["accept", "--reviewer", "olivier", "--cluster", "integral_variants=c01", *env["args"]]) == 0
    tax = json.loads((env["mdir"] / "tax.json").read_text())
    assert tax["reviewed"] is True and tax["reviewer"] == "olivier"
    assert tax["classes"][-2:] == ["integral_variants", "other"]
    labels = [json.loads(x) for x in (env["mdir"] / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 12 and labels == sorted(labels, key=lambda x: x["id"])
    by = {lb["id"]: lb for lb in labels}
    assert by[_train_rows()[0]["id"]]["clean"]
    assert by[_train_rows()[2]["id"]]["mining_class"] == "decoration"
    assert by[_train_rows()[3]["id"]]["in_regression"]
    assert by[_train_rows()[4]["id"]]["mining_class"] == "integral_variants"
    mm = json.loads((env["mdir"] / "MM.json").read_text())
    assert len(mm["taxonomy_sha256"]) == 64 and len(mm["labels_sha256"]) == 64


@pytest.mark.parametrize("spec", [["--cluster", "x=c99"], ["--cluster", "decoration=c01"], ["--cluster", "other=c01"], []])
def test_accept_refuses_bad_input(env, spec, capsys):
    _scored(env)
    assert mt.main(["report", *env["args"]]) == 0
    assert mt.main(["accept", "--reviewer", "o", *spec, *env["args"]]) == 2
    assert "refused:" in capsys.readouterr().err
    assert not (env["mdir"] / "tax.json").exists()


def test_accept_none(env):
    _scored(env)
    assert mt.main(["report", *env["args"]]) == 0
    assert mt.main(["accept", "--reviewer", "o", "--none", *env["args"]]) == 0
    tax = json.loads((env["mdir"] / "tax.json").read_text())
    assert tax["accepted_clusters"] == [] and tax["rejected_candidate_ids"] == ["c01"]


def test_non_train_paper_refused(env, capsys):
    _w(env["train"], [*_train_rows(), {"id": "d1", "arxiv_id": DEV, "latex": "x", "flags": [], "image": "i.png"}])
    for cmd in ("seed", "infer", "report", "accept"):
        assert mt.main([cmd, *(["--reviewer", "o", "--none"] if cmd == "accept" else []), *env["args"]]) == 2
        assert "refused:" in capsys.readouterr().err


def test_score_refuses_non_train_predictions(env, capsys):
    _w(env["mdir"] / "p.jsonl", [{"id": "d1", "paper_id": DEV, "gt_latex": "x", "pred_latex": "x"}])
    assert mt.main(["score", *env["args"]]) == 2
    assert "refused:" in capsys.readouterr().err
