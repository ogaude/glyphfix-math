"""Pilot metrics, engine stop rule, audit, contact sheet and report on synthetic shards."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from lightonocr_math import render, splits
from lightonocr_math.corpus import pilot
from lightonocr_math.corpus._io import write_json, write_jsonl
from lightonocr_math.corpus.records import BUILDER_VERSION


def _ids(side, n):
    out = []
    for k in range(1, 20000):
        i = f"2302.{k:05d}"
        if splits.split_of(i) == side and i not in splits.frozen().quarantine:
            out.append(i)
            if len(out) == n:
                return out
    raise AssertionError


TRAIN = _ids("train", 20)
NB = 16
BENCH = _ids("benchmark", 1)[0]
FLAG_SETS = [["prime"], ["greek"], ["decoration"], ["i_1"], ["plain"], ["prime", "greek"]]


def _rec(aid, n, flags, tmp_path, latex=None, raw=None):
    latex = latex or f"x_{n} + y_{n} = z"
    rid = hashlib.sha256(f"{aid}{n}{latex}".encode()).hexdigest()[:16]
    rel = f"data/corpus/images/{aid}/{rid}.png"
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((30, 60), 255, dtype=np.uint8), mode="L").save(p)
    return {
        "id": rid, "arxiv_id": aid, "paper_version": "v1", "license_uri": "u", "license_spdx": "CC-BY-4.0", "split": "train",
        "env": "equation", "source_file": "main.tex", "source_line": n + 1, "raw_body": raw or latex, "latex": latex,
        "flags": flags, "image": rel, "width": 60, "height": 30, "bitmap_sha256": "ab" * 32, "image_sha256": "cd" * 32,
        "builder_version": BUILDER_VERSION, "canon_version": "1", "preamble_sha256": "p", "render_image_digest": "d",
        "engine_flagged": False, "orig_bitmap_sha256": "ab" * 32,
    }


def _summary(aid, kept, *, status="built", engine=False, extracted=None, split="train", exclusions=None):
    return {
        "arxiv_id": aid, "side": "corpus", "split": split, "status": status, "n_extracted": kept if extracted is None else extracted,
        "n_kept": kept, "engine_flagged": engine, "exclusions": exclusions or {}, "redefined_core": [], "seconds": 2.0,
    }


def _write(root, summaries, records, excluded=()):
    for s in summaries:
        sid = s["arxiv_id"]
        write_json(root / "papers" / f"{sid}.json", s)
        write_jsonl(root / "records" / f"{sid}.jsonl", [r for r in records if r["arxiv_id"] == sid])
        write_jsonl(root / "excluded" / f"{sid}.jsonl", [r for r in excluded if r["arxiv_id"] == sid])


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    recs, sums, exc = [], [], []
    for pi, aid in enumerate(TRAIN[:NB]):
        n = 4 + pi
        flags = [FLAG_SETS[(pi + k) % len(FLAG_SETS)] for k in range(n)]
        recs += [_rec(aid, k, flags[k], tmp_path) for k in range(n)]
        sums.append(_summary(aid, n, extracted=n + 1, exclusions={"duplicate": 1}))
        exc.append({"arxiv_id": aid, "reason": "duplicate", "eq_index": 0, "source_file": "main.tex", "source_line": 1, "env": "equation", "detail": {}})
    sums.append(_summary(TRAIN[NB], 0, status="no_source", exclusions={"no_source": 1}))
    exc.append({"arxiv_id": TRAIN[NB], "reason": "no_source", "eq_index": None, "source_file": None, "source_line": None, "env": None, "detail": {}})
    _write(root, sums, recs, exc)
    return root, recs, sums, exc, tmp_path


def test_metrics(shards):
    root, recs, sums, _exc, _ = shards
    m = pilot.pilot_metrics(root, TRAIN[:NB + 1])
    assert m["papers_by_status"] == {"built": NB, "no_source": 1}
    assert m["equation_exclusions"] == {"duplicate": NB} and m["paper_level_exclusions"] == {"no_source": 1}
    assert m["n_kept"] == len(recs)
    assert m["yield"]["per_paper"]["median"] == 11.5
    assert sum(m["yield"]["histogram"].values()) == NB
    assert m["yield"]["capped"]["total"] == sum(min(s["n_kept"], 30) for s in sums if s["status"] == "built")
    assert m["flags"]["prime"]["equations"] > 0 and m["flags"]["plain"]["papers"] > 0
    assert m["crop"]["share_upscaled_to_floor"] == 1.0
    assert m["crop"]["image_tokens"]["median"] >= 64.0


def test_benchmark_id_refused(shards):
    root, *_ = shards
    with pytest.raises(splits.LeakageError):
        pilot.pilot_metrics(root, [*TRAIN[:2], BENCH])


def test_benchmark_record_in_shard_refused(tmp_path):
    root = tmp_path / "shards"
    rec = _rec(BENCH, 0, ["plain"], tmp_path)
    _write(root, [_summary(TRAIN[0], 1)], [{**rec, "arxiv_id": TRAIN[0]}])
    # a record whose arxiv_id is a benchmark paper is rejected even when filed under a train paper
    write_jsonl(root / "records" / f"{TRAIN[0]}.jsonl", [rec])
    with pytest.raises(splits.LeakageError):
        pilot.load_shards(root, [TRAIN[0]])


def _eng(n_good, n_flag, zero=0):
    sums = [_summary(f"p{i}", 3, engine=i < n_flag) for i in range(n_good)]
    for s in sums[:zero]:
        s["n_kept"] = 0
    return pilot.engine_numbers(sums, [])


def test_engine_stop_rule_at_ten_percent():
    assert _eng(20, 2)["stop"] is False and _eng(20, 2)["engine_flagged_share"] == 0.10
    assert _eng(20, 3)["stop"] is True
    e = _eng(10, 2, zero=1)
    assert e["engine_flagged_zero_kept"] == 1 and e["n_good_papers"] == 10


def test_engine_good_excludes_macro_and_short_only_papers():
    sums = [
        _summary("a", 2, engine=True),
        _summary("b", 0, extracted=3, exclusions={"macros": 2, "short_long": 1}, engine=True),
        _summary("c", 0, status="no_main"),
    ]
    e = pilot.engine_numbers(sums, [{"reason": "engine"}] * 4)
    assert e["n_good_papers"] == 1 and e["n_engine_flagged_good"] == 1 and e["equations_lost_to_engine"] == 4


def test_contact_sheet_sample_size_and_paper_cap(shards):
    _, recs, *_ = shards
    s = pilot.contact_sheet_sample(recs)
    assert len(s) == 30 and len({r["id"] for r in s}) == 30
    per: dict[str, int] = {}
    for r in s:
        per[r["arxiv_id"]] = per.get(r["arxiv_id"], 0) + 1
    assert max(per.values()) <= 2
    for f in ("prime", "greek", "decoration", "i_1", "plain"):
        assert sum(1 for r in s if f in r["flags"]) >= 4
    assert s == pilot.contact_sheet_sample(recs)


class FakeRender:
    def __init__(self):
        self.calls = []

    def render_many(self, items, *, cache=None, wrapper="display", **kw):
        self.calls.append((list(items), wrapper, cache))
        return [render.RenderResult(hashlib.sha256(x.encode()).hexdigest(), "rendered", (10, 20), hashlib.sha256(x.encode()).hexdigest(), "") for x in items]

    def load_bitmap(self, res, cache=None):
        return np.full((10, 20), 128, dtype=np.uint8)


def test_contact_sheet_escapes_html(tmp_path, monkeypatch):
    fake = FakeRender()
    monkeypatch.setattr(render, "render_many", fake.render_many)
    monkeypatch.setattr(render, "load_bitmap", fake.load_bitmap)
    evil = "x <script>alert(1)</script> + y"
    rec = _rec(TRAIN[0], 0, ["plain"], tmp_path, latex=evil, raw=evil + " \\label{a}")
    out = tmp_path / "sheet.html"
    pilot.render_contact_sheet([rec], out, cache=tmp_path / "c", repo_root=tmp_path)
    text = out.read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "data:image/png;base64," in text and "http" not in text.replace("http-equiv", "")
    assert fake.calls[0][1] == "env"


def test_audit(shards, tmp_path, monkeypatch):
    _, recs, sums, _exc, _ = shards
    fake = FakeRender()
    monkeypatch.setattr(render, "render_many", fake.render_many)
    mism = [
        {"arxiv_id": TRAIN[i % 6], "reason": "canon_mismatch", "eq_index": 1, "env": "align", "source_file": "m", "source_line": 1,
         "detail": {"cause": "bitmap-differs", "multi_column": True}} for i in range(12)
    ]
    # the fake bitmaps differ from the stored ones, so determinism reports 0 identical
    a = pilot.automated_audit(recs, mism, summaries=sums, sample=10)
    cv = a["canon_validation"]
    assert cv["validated"] == len(recs) + 12 and cv["mismatched"] == 12 and cv["passed"] == len(recs)
    assert "bitmap-differs | align | multi_column=True" in cv["by_cause"]
    assert a["systematic_causes"] == ["bitmap-differs"]
    assert a["determinism"]["sampled"] == 10 and a["determinism"]["identical"] == 0
    assert a["structural_failures"]["count"] == 0 and a["layout_only_diff_failures"]["count"] == 0
    # the determinism render uses a fresh temporary cache, not a shared one
    assert fake.calls[0][2] is not None and "pilot_determinism_" in str(fake.calls[0][2])


def test_audit_flags_structural_and_layout_failures(tmp_path, monkeypatch):
    fake = FakeRender()
    monkeypatch.setattr(render, "render_many", fake.render_many)
    bad = _rec(TRAIN[0], 0, ["plain"], tmp_path, latex="\\frac{a}{b", raw="\\frac{a}{b")
    changed = _rec(TRAIN[1], 0, ["plain"], tmp_path, latex="a \\le b", raw="a \\leq b")
    a = pilot.automated_audit([bad, changed], [], sample=0)
    assert a["structural_failures"]["count"] == 1 and a["layout_only_diff_failures"]["count"] == 1


def test_report_sections_and_phrases(shards, monkeypatch):
    root, recs, sums, exc, _tmp = shards
    fake = FakeRender()
    monkeypatch.setattr(render, "render_many", fake.render_many)
    m = pilot.pilot_metrics(root, TRAIN[:NB + 1])
    m["audit"] = pilot.automated_audit(recs, exc, summaries=sums, sample=5)
    m["near_duplicates"] = pilot.near_duplicate_profile(recs)
    text = pilot.render_report(m)
    for h in ("## Papers", "## Exclusions", "## Yield", "## Notation mix", "## Engine go/no-go", "## Label audit",
              "## Crop sizes and scale", "## Near-duplicates", "## Projections"):
        assert h in text
    assert "Manual audit: 30 pairs per D-09 (user choice); the roadmap text says 100-200" in text
    assert "tautological under D-10" in text
    assert "continue (under the threshold)" in text and "10%" in text
    m["engine"] = _eng(10, 5)
    assert "STOP for user decision" in pilot.render_report(m)
    assert "150" in text and "800" in text and "50" in text
    json.dumps(m)


def test_near_duplicates_cross_paper_only(tmp_path):
    a = _rec(TRAIN[0], 0, ["plain"], tmp_path, latex="x + y = z")
    a2 = _rec(TRAIN[0], 1, ["plain"], tmp_path, latex="x + y = z ")
    b = _rec(TRAIN[1], 0, ["plain"], tmp_path, latex="x + y = z + 1")
    c = _rec(TRAIN[2], 0, ["plain"], tmp_path, latex="\\int_0^1 f'(t) dt")
    nd = pilot.near_duplicate_profile([a, a2, b, c], threshold=0.7)
    assert nd["n"] == 4 and nd["n_above"] >= 2
    assert all({p["a"], p["b"]} != {a["id"], a2["id"]} for p in nd["top_pairs"])
