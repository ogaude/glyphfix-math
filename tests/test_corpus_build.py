"""Corpus builder logic with a faked render sandbox."""

import hashlib
import io
import json
import re
import tarfile

import numpy as np
import pytest
from PIL import Image

from lightonocr_math import render, splits
from lightonocr_math.corpus import build, canon
from lightonocr_math.corpus.records import EXCLUSION_REASONS, validate_record


def _pick(side):
    for n in range(1, 5000):
        i = f"2301.{n:05d}"
        if splits.split_of(i) == side:
            return i
    raise AssertionError(side)


TRAIN_ID = _pick("train")
BENCH_ID = _pick("benchmark")

INNER_RE = re.compile(r"\\begin\{(?:equation|align|gather)\*\}(.*)\\end\{(?:equation|align|gather)\*\}", re.DOTALL)


class FakeSandbox:
    """Statuses by content; bitmaps derived from the formula text so the original and the label match."""

    def __init__(self):
        self.calls = []

    def _inner(self, latex):
        m = INNER_RE.match(latex.strip())
        return m.group(1).strip() if m else latex.strip()

    def render_many(self, items, *, batch_size=100, cache=None, timeout_s=300, max_parallel=None, wrapper="display"):
        self.calls.append((list(items), wrapper))
        out = []
        for x in items:
            key = hashlib.sha256((wrapper + x).encode()).hexdigest()
            inner = self._inner(x)
            if "UNDEF" in inner:
                out.append(render.RenderResult(key, "compile-failed", None, None, "! Undefined"))
            elif wrapper == "env" and "ORIGFAIL" in inner:
                out.append(render.RenderResult(key, "compile-failed", None, None, "! boom"))
            else:
                h = hashlib.sha256(("DIFF" if wrapper == "env" and "DIFFERS" in inner else inner).encode()).hexdigest()
                out.append(render.RenderResult(key, "rendered", (20, 40), h, ""))
        return out

    def load_bitmap(self, result, cache=None):
        if result.status != "rendered":
            return None
        seed = int(result.bitmap_sha256[:8], 16)
        return np.random.default_rng(seed).integers(0, 255, size=(20, 40), dtype=np.uint8)

    def compare(self, a, b, cache=None):
        same = a.status == "rendered" and b.status == "rendered" and a.bitmap_sha256 == b.bitmap_sha256
        return same, same


@pytest.fixture
def sandbox(monkeypatch):
    fake = FakeSandbox()
    monkeypatch.setattr(render, "render_many", fake.render_many)
    monkeypatch.setattr(render, "load_bitmap", fake.load_bitmap)
    monkeypatch.setattr(render, "compare", fake.compare)
    return fake


def _bundle(tmp_path, name, files, mode="w:gz"):
    p = tmp_path / name
    with tarfile.open(p, mode) as tf:
        for fname, data in files.items():
            b = data.encode()
            ti = tarfile.TarInfo(fname)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return p


def _paper(tmp_path, arxiv_id, files, *, kind="tar.gz", fetch_status="ok", split="train"):
    src = _bundle(tmp_path, f"{arxiv_id}.tgz", files)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    return build.PaperInput(
        arxiv_id, "v1",
        {"arxiv_id": arxiv_id, "version": "v1", "license_uri": "u", "license_spdx": "CC-BY-4.0", "split": split},
        {"status": fetch_status, "kind": kind, "sha256": sha}, src, tmp_path / "work" / arxiv_id,
    )


DOC = "\\documentclass{article}\n{macros}\\begin{document}\n{body}\\end{document}\n"


def _doc(body, macros=""):
    return DOC.replace("{macros}", macros).replace("{body}", body)


def test_every_reason_and_kept_record(tmp_path, sandbox):
    body = (
        "\\begin{equation}\\label{a} x+y=z \\end{equation}\n"          # kept
        "\\begin{equation} x+y=z \\end{equation}\n"                      # duplicate
        "\\begin{equation} \\R + 1 \\end{equation}\n"                    # macros
        "\\begin{equation} a \\end{equation}\n"                          # short_long
        "\\begin{equation} \\input{/etc/passwd} xx \\end{equation}\n"    # unsafe
        "\\begin{equation} UNDEF+1 \\end{equation}\n"                    # render
        "\\begin{equation} DIFFERS+1 \\end{equation}\n"                  # canon_mismatch bitmap
        "\\begin{equation} ORIGFAIL+1 \\end{equation}\n"                 # canon_mismatch original
        "\\begin{align} a&=b \\\\ c&=d \\end{align}\n"                   # kept
    )
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc(body, "\\newcommand{\\R}{\\mathbb{R}}\n")})
    (res,) = build.build_chunk([p], cache=tmp_path / "c", out_roots={"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"})
    reasons = {r["reason"] for r in res.excluded}
    assert reasons == {"duplicate", "macros", "short_long", "unsafe", "render", "canon_mismatch"}
    assert reasons <= set(EXCLUSION_REASONS)
    assert len(res.records) == 2 and res.summary["n_kept"] == 2 and res.summary["n_extracted"] == 9
    causes = sorted(r["detail"]["cause"] for r in res.excluded if r["reason"] == "canon_mismatch")
    assert causes == ["bitmap-differs", "original-compile-failed"]
    assert res.summary["validation_tiers"] == {"identical": 2, "glyph_equal": 0}
    for rec in res.records:
        assert validate_record(rec) == []
        assert rec["license_spdx"] == "CC-BY-4.0" and rec["builder_version"] == build.BUILDER_VERSION
        assert rec["canon_version"] == canon.CANON_RULES_VERSION and rec["preamble_sha256"] == render.PREAMBLE_SHA256
        assert rec["orig_bitmap_sha256"] == rec["bitmap_sha256"]
        assert rec["flags"] and rec["raw_body"]
    first = res.records[0]
    assert first["latex"] == "x+y=z" and first["source_line"] == 4
    assert not p.work_dir.exists()


def test_infra_error_not_an_exclusion(tmp_path, sandbox, monkeypatch):
    def boom(*a, **k):
        raise render.RenderInfrastructureError("docker down")

    monkeypatch.setattr(render, "render_many", boom)
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    roots = {"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"}
    with pytest.raises(render.RenderInfrastructureError):
        build.build_chunk([p], cache=tmp_path / "c", out_roots=roots)
    assert not (tmp_path / "corpus").exists()


def test_paper_level_reasons(tmp_path, sandbox):
    missing = _paper(tmp_path, "2301.00001", {"main.tex": "x"}, fetch_status="missing")
    pdf = _paper(tmp_path, "2301.00002", {"main.tex": "x"}, kind="pdf")
    nomain = _paper(tmp_path, "2301.00003", {"a.tex": "no document class"})
    bad = _paper(tmp_path, "2301.00004", {"../evil.tex": "x"})
    res = build.build_chunk([missing, pdf, nomain, bad], cache=tmp_path / "c")
    got = {r.arxiv_id: (r.summary["status"], r.excluded[0]["reason"]) for r in res}
    assert got == {
        "2301.00001": ("no_source", "no_source"), "2301.00002": ("no_source", "no_source"),
        "2301.00003": ("no_main", "no_main"), "2301.00004": ("bad_archive", "bad_archive"),
    }
    assert sandbox.calls == []


def test_engine_flag_changes_reason(tmp_path, sandbox):
    body = "\\begin{equation} UNDEF+1 \\end{equation}\n"
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc(body, "\\usepackage{fontspec}\n")})
    (res,) = build.build_chunk([p], cache=tmp_path / "c")
    assert res.summary["engine_flagged"] is True
    assert [r["reason"] for r in res.excluded] == ["engine"]


def test_two_batched_calls_per_chunk(tmp_path, sandbox):
    ps = [
        _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")}),
    ]
    build.build_chunk(ps, cache=tmp_path / "c")
    assert [w for _, w in sandbox.calls] == ["display", "env"]
    assert sandbox.calls[1][0][0].startswith("\\begin{equation*}")


def test_png_padding_mode_and_determinism(tmp_path, sandbox):
    p1 = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    roots = {"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"}
    (res,) = build.build_chunk([p1], cache=tmp_path / "c", out_roots=roots)
    rec = res.records[0]
    png = roots["corpus"] / "images" / TRAIN_ID.replace("/", "_") / f"{rec['id']}.png"
    img = Image.open(png)
    assert img.mode == "L" and img.size == (40 + 2 * build.PAD_PX, 20 + 2 * build.PAD_PX)
    assert (rec["width"], rec["height"]) == img.size
    arr = np.asarray(img)
    assert (arr[: build.PAD_PX] == 255).all() and (arr[:, : build.PAD_PX] == 255).all()
    assert hashlib.sha256(png.read_bytes()).hexdigest() == rec["image_sha256"]
    p2 = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    (res2,) = build.build_chunk([p2], cache=tmp_path / "c", out_roots=roots)
    assert res2.records[0]["image_sha256"] == rec["image_sha256"]
    assert png.read_bytes() == build._crop_png(sandbox.load_bitmap(render.RenderResult("k", "rendered", (20, 40), res2.records[0]["bitmap_sha256"])))[0]


def test_write_shards_routes_by_side(tmp_path, sandbox):
    roots = {"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"}
    tr = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    bm = _paper(tmp_path, BENCH_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")}, split="benchmark")
    for r in build.build_chunk([tr, bm], cache=tmp_path / "c", out_roots=roots):
        build.write_shards(r, roots)
    sid_t, sid_b = TRAIN_ID.replace("/", "_"), BENCH_ID.replace("/", "_")
    assert (roots["corpus"] / "shards" / "records" / f"{sid_t}.jsonl").is_file()
    assert (roots["corpus"] / "images" / sid_t).is_dir()
    assert (roots["benchmark"] / "shards" / "papers" / f"{sid_b}.json").is_file()
    assert (roots["benchmark"] / "images" / sid_b).is_dir()
    assert not (roots["corpus"] / "shards" / "records" / f"{sid_b}.jsonl").exists()
    assert not (roots["benchmark"] / "shards" / "records" / f"{sid_t}.jsonl").exists()
    rec = json.loads((roots["benchmark"] / "shards" / "records" / f"{sid_b}.jsonl").read_text().splitlines()[0])
    assert rec["image"].startswith("data/benchmark/images/")


def test_shard_up_to_date_resume_key(tmp_path, sandbox, monkeypatch):
    roots = {"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"}
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    (res,) = build.build_chunk([p], cache=tmp_path / "c", out_roots=roots)
    build.write_shards(res, roots)
    summ = roots["corpus"] / "shards" / "papers" / f"{TRAIN_ID.replace('/', '_')}.json"
    sha = res.summary["src_sha256"]
    assert build.shard_up_to_date(summ, sha)
    assert not build.shard_up_to_date(summ, "0" * 64)
    assert not build.shard_up_to_date(tmp_path / "nope.json", sha)
    for mod, name, val in [
        (build, "BUILDER_VERSION", "99"), (build.canon, "CANON_RULES_VERSION", "99"), (build.canon, "CANON_FINGERPRINT", "x"),
        (build.flags, "FLAG_RULES_VERSION", "99"), (build.render, "PREAMBLE_SHA256", "x"),
    ]:
        with monkeypatch.context() as m:
            m.setattr(mod, name, val)
            assert not build.shard_up_to_date(summ, sha), name


def test_glyph_equal_tier_keeps_a_label_whose_bitmap_only_shifts(tmp_path, sandbox, monkeypatch):
    monkeypatch.setattr(render, "compare", lambda a, b, cache=None: (False, False))
    monkeypatch.setattr(build.glyphmatch, "glyph_equal", lambda a, b: True)
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    (res,) = build.build_chunk([p], cache=tmp_path / "c", out_roots={"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"})
    assert len(res.records) == 1 and res.summary["validation_tiers"] == {"identical": 0, "glyph_equal": 1}
    monkeypatch.setattr(build.glyphmatch, "glyph_equal", lambda a, b: False)
    p2 = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    (res2,) = build.build_chunk([p2], cache=tmp_path / "c2", out_roots={"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"})
    assert res2.records == [] and [r["detail"]["cause"] for r in res2.excluded] == ["bitmap-differs"]


def test_stale_work_tree_is_cleared(tmp_path):
    p = _paper(tmp_path, TRAIN_ID, {"main.tex": _doc("\\begin{equation} x+y=z \\end{equation}\n")})
    stale = p.work_dir / "src" / "stale.tex"
    stale.parent.mkdir(parents=True)
    stale.write_text(_doc("\\begin{equation} STALE \\end{equation}\n"))
    res = build.prepare_paper(p)
    assert not stale.exists()
    assert res.summary["n_files"] == 1
