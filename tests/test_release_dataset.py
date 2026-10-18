"""Dataset staging is fail-closed on license, version, catalog, hash and image errors (synthetic fixtures only)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from scripts.release import stage_dataset as sd
from scripts.release.common import (
    BENCH_COLUMNS,
    LICENSE_URI,
    PLACEHOLDERS,
    STRIPPED_BENCH_FIELDS,
    TRAIN_COLUMNS,
    render_template,
)

ROOT = Path(__file__).resolve().parent.parent
BY, CC0 = "CC-BY-4.0", "CC0-1.0"


def _png(color: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8, 6), color).save(buf, format="PNG")
    return buf.getvalue()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    papers = [("2001.00001", "v2", BY), ("2001.00002", "v1", CC0)]
    cat, trainset, corpus, bench = [], [], [], []
    for n, (a, v, spdx) in enumerate(papers):
        cat.append({"arxiv_id": a, "title": f"Paper {a}", "authors": ["A One", "B Two"], "authors_raw": "A One and B Two",
                    "version": v, "license_spdx": spdx, "license_uri": LICENSE_URI[spdx]})
        for k in range(2):
            rid = f"r{n}{k}"
            data = _png(10 * (n * 2 + k))
            rel = f"img/{a}/{rid}.png"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            base = {"arxiv_id": a, "id": rid, "latex": f"x_{{{k}}}", "license_spdx": spdx, "license_uri": LICENSE_URI[spdx],
                    "paper_version": v, "flags": ["greek"], "width": 8, "height": 6, "image": rel}
            trainset.append(dict(base, paper_id=a, target_text="```latex\nx\n```", group="replay", mining_class=None,
                                 control_glyph=None, tags=[]))
            corpus.append(dict(base, image_sha256=sha, bitmap_sha256="b" * 64, raw_body="raw", source_file="p.tex",
                               source_line=3))
            bench.append(dict(base, image_sha256=sha, bitmap_sha256="b" * 64, raw_body="raw", source_file="p.tex",
                              source_line=3, render_image_digest="sha256:x", preamble_sha256="p" * 64, canon_version="2",
                              builder_version="1", env="equation", split="benchmark"))
    _write(tmp_path / "trainset.jsonl", trainset)
    _write(tmp_path / "corpus.jsonl", corpus)
    _write(tmp_path / "bench.jsonl", bench)
    _write(tmp_path / "catalog.jsonl", cat)
    return tmp_path


def _train(w):
    return sd.build_train_rows(w / "trainset.jsonl", w / "corpus.jsonl", w / "catalog.jsonl", w)


def _edit(path: Path, fn) -> None:
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    fn(rows)
    _write(path, rows)


def test_train_rows_shape(world):
    rows = _train(world)
    assert len(rows) == 4
    for r in rows:
        assert tuple(r) == TRAIN_COLUMNS
    r = rows[0]
    assert r["title"] == "Paper 2001.00001" and r["authors"] == ["A One", "B Two"]
    assert r["arxiv_url"] == "https://arxiv.org/abs/2001.00001v2"
    assert r["image"]["path"] == "r00.png" and r["image"]["bytes"][:4] == b"\x89PNG"


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r[0].update(license_spdx="CC-BY-NC-4.0"), "not allowed"),
    (lambda r: r[0].update(license_spdx="CC-BY-SA-4.0"), "not allowed"),
    (lambda r: r[0].update(license_uri="http://example.org/"), "license_uri"),
    (lambda r: r[0].update(paper_version="2"), "vN"),
])
def test_license_and_version_gates(world, mutate, match):
    def both(rows):
        mutate(rows)

    _edit(world / "trainset.jsonl", both)
    _edit(world / "corpus.jsonl", both)
    with pytest.raises(sd.StageError, match=match):
        _train(world)


def test_catalog_disagreement_raises(world):
    _edit(world / "catalog.jsonl", lambda r: r[0].update(version="v9"))
    with pytest.raises(sd.StageError, match="catalog version"):
        _train(world)
    _edit(world / "catalog.jsonl", lambda r: r[0].update(version="v2", license_spdx=CC0, license_uri=LICENSE_URI[CC0]))
    with pytest.raises(sd.StageError, match="catalog license"):
        _train(world)


def test_png_hash_and_missing(world):
    p = world / "img/2001.00001/r00.png"
    p.write_bytes(_png(250))
    with pytest.raises(sd.StageError, match="do not hash"):
        _train(world)
    p.unlink()
    with pytest.raises(sd.StageError, match="missing"):
        _train(world)


def test_trainset_latex_must_match_corpus(world):
    _edit(world / "trainset.jsonl", lambda r: r[0].update(latex="y"))
    with pytest.raises(sd.StageError, match="latex differs"):
        _train(world)


def test_bench_rows_strip_fields(world):
    rows = sd.build_bench_rows(world / "bench.jsonl", world / "catalog.jsonl", world)
    assert len(rows) == 4
    for r in rows:
        assert tuple(r) == BENCH_COLUMNS
        assert not set(STRIPPED_BENCH_FIELDS) & set(r)


def test_attribution(world):
    train = _train(world)
    bench = sd.build_bench_rows(world / "bench.jsonl", world / "catalog.jsonl", world)
    ent = sd.attribution_entries(train, bench)
    assert [e["arxiv_id"] for e in ent] == ["2001.00001", "2001.00002"]
    assert all(e["splits"] == ["benchmark", "train"] for e in ent)
    assert sum(e["n_rows"] for e in ent) == len(train) + len(bench)
    md = sd.render_attribution_md(ent)
    for e in ent:
        assert e["arxiv_id"] in md and e["version"] in md
    assert "indication of modification" in md
    assert "needs no attribution" in md or "need no attribution" in md


def test_attribution_authors_parse():
    assert sd._authors({"arxiv_id": "x", "authors": "['A', 'B']"}) == ["A", "B"]
    assert sd._authors({"arxiv_id": "x", "authors": None, "authors_raw": "A, B and C"}) == ["A", "B", "C"]
    with pytest.raises(sd.StageError):
        sd._authors({"arxiv_id": "x", "authors": [], "authors_raw": ""})


def test_card_template():
    text = (ROOT / "release_templates/dataset_card.md").read_text(encoding="utf-8")
    assert not re.search(r"\b(8000|2688|7560|440|767|150)\b", text)
    for k in ("license: cc-by-4.0", "Do not train on it", "@@DATASET_NAME@@", "@@CONTACT_EMAIL@@"):
        assert k in text
    vals = {k: 1 for k in ("train_rows", "train_papers", "bench_rows", "bench_papers", "train_cc0", "bench_cc0", "mined_records")}
    out = render_template(text, vals)
    assert "{{" not in out
    for tok in PLACEHOLDERS.values():
        if tok in text:
            assert tok in out


def test_parquet_roundtrip(world, tmp_path):
    pytest.importorskip("datasets")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = _train(world)
    out = tmp_path / "out" / "t.parquet"
    sd._write_parquet(rows, out, False, tmp_path / "cache")
    t = pq.read_table(out)
    assert t.num_rows == 4
    assert all("/" not in x["path"] for x in t.column("image").to_pylist())
    assert set(t.column_names) == set(TRAIN_COLUMNS)
