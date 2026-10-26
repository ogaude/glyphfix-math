import pytest

from glyphfix_math import render
from glyphfix_math.corpus import pages


def test_latexmk_cmd_is_hardened(tmp_path):
    cmd = pages.latexmk_docker_cmd("n", tmp_path, "main.tex")
    assert cmd[cmd.index("--network") + 1] == "none"
    for flag in ("--read-only", "--cap-drop", "ALL", render.IMAGE, "latexmk", "-pdf", "-synctex=1", "-interaction=nonstopmode"):
        assert flag in cmd
    assert "-norc" in cmd  # a paper-supplied latexmkrc is Perl and must not be read
    assert "-pdflatex=pdflatex -no-shell-escape %O %S" in cmd
    assert not any("-shell-escape" in a and "-no-shell-escape" not in a for a in cmd)


CANNED = """##Q e1
SyncTeX result begin
Output:main.pdf
Input:/work/main.tex
Line:5
Column:0
Page:1
x:100.5
y:200.25
h:90
v:200.25
W:300
H:12.5
before:
offset:
middle:
SyncTeX result end
##Q e2
SyncTeX result begin
Page:2
x:10
y:20
h:10
v:20
W:30
H:4
SyncTeX result end
##Q e3
"""


def test_parse_synctex():
    got = pages.parse_synctex(CANNED)
    assert got["e1"] == [pages.Box(1, 100.5, 200.25, 300.0, 12.5)]
    assert got["e2"] == [pages.Box(2, 10.0, 20.0, 30.0, 4.0)]
    assert got["e3"] == []


def test_synctex_script_quotes_names():
    s = pages.synctex_script([("a b", 3, "it's.tex")], "main.pdf")
    assert "'##Q a b'" in s
    assert "it'\"'\"'s.tex" in s


def test_equation_pages():
    b1, b2 = pages.Box(1, 0, 0, 1, 1), pages.Box(2, 0, 0, 1, 1)
    assert pages.equation_pages([b1, b1]) == 1
    assert pages.equation_pages([b1, b2]) is None
    assert pages.equation_pages([]) is None


def test_choose_page():
    assert pages.choose_page({"a": 2, "b": 2, "c": 1, "d": 2, "e": None}) == 2
    assert pages.choose_page({"a": 3, "b": 1, "c": 3, "d": 1, "e": 1, "f": 3}) == 1  # tie to the lower page
    assert pages.choose_page({"a": 1, "b": 1}) is None
    assert pages.choose_page({}) is None


def test_rasterize_page(tmp_path):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    pdf = tmp_path / "p.pdf"
    doc.save(str(pdf))
    doc.close()
    img = pages.rasterize_page(pdf, 0)
    assert img.mode == "RGB"
    assert max(img.size) == pages.PAGE_LONGEST_PX
    assert abs(img.width / img.height - 612 / 792) < 2 / 2048


def test_extract_display_blocks():
    text = (
        "intro $x$ text\n$$a+b$$\nmid \\[ c = d \\]\n"
        "\\begin{align*}\ne &= f\n\\end{align*}\n"
        "```\n$$fenced$$\n```\n"
        "cost is 5\\$$ and 6\\$$ done\n"
        "\\begin{equation}g\\end{equation}"
    )
    assert pages.extract_display_blocks(text) == ["a+b", "c = d", "\\begin{align*}\ne &= f\n\\end{align*}", "\\begin{equation}g\\end{equation}"]


def test_match_threshold_frozen_value():
    assert pages.MATCH_THRESHOLD == 0.5
    assert pages.PAGE_MATCH_VERSION == "2"


def test_extract_display_blocks_model_page_shape():
    # shape observed in stock page output: $$ blocks on their own lines, numbered with \tag or a leading label
    text = "Lemma.\n\n$$\n(3.22.2) \\qquad 0 \\longrightarrow S_1 \\longrightarrow P_2,\n$$\n\nthen\n\n$$\nm = \\left\\lceil \\frac{Mn}{N} \\right\\rceil. \\tag{3}\n$$\n"
    blocks = pages.extract_display_blocks(text)
    assert len(blocks) == 2 and blocks[0].startswith("(3.22.2)") and blocks[1].endswith("\\tag{3}")
    m = pages.match_equations(["0 \\longrightarrow S_1 \\longrightarrow P_2,", "m = \\Bigg{\\lceil} \\frac{Mn}{N} \\Bigg{\\rceil}."], blocks)
    assert [x[1] for x in m] == [0, 1]


def test_extract_display_blocks_empty():
    assert pages.extract_display_blocks("no math here $x$") == []


def test_match_one_to_one_and_threshold():
    gt = ["a+b=c", "x^2+y^2=z^2", "\\int_0^1 f(t) dt"]
    pred = ["x^2+y^2=z^2", "a+b=c", "completely different \\alpha \\beta \\gamma \\delta"]
    m = pages.match_equations(gt, pred, 0.7)
    assert [x[1] for x in m] == [1, 0, None]
    # one prediction cannot serve two GT equations
    m2 = pages.match_equations(["a+b=c", "a+b=c"], ["a+b=c"], 0.7)
    assert sorted(x[1] is not None for x in m2) == [False, True]


def test_match_no_predictions():
    assert pages.match_equations(["a", "b"], []) == [(0, None, 0.0), (1, None, 0.0)]
    assert pages.match_equations([], ["a"]) == []


def test_page_score_rows_pass_read_predictions(tmp_path):
    from glyphfix_math import score
    from glyphfix_math.corpus._io import write_jsonl

    recs = [{"id": "r1", "latex": "a+b=c"}, {"id": "r2", "latex": "x^2=y"}]
    row = pages.page_score_rows({"page_id": "p1", "arxiv_id": "2301.01234", "paper_id": "2301.01234"}, recs, "$$a+b=c$$")
    assert [r["id"] for r in row] == ["r1", "r2"]
    assert row[0]["pred_latex"] == "a+b=c" and row[0]["matched"]
    assert row[1]["pred_latex"] == "" and not row[1]["matched"]
    f = tmp_path / "rows.jsonl"
    write_jsonl(f, row)
    assert len(score.read_predictions(f)) == 2


def test_crop_scale(tmp_path):
    from PIL import Image

    page = Image.new("RGB", (1000, 1000), "white")
    for y in range(200, 240):
        for x in range(100, 300):
            page.putpixel((x, y), (0, 0, 0))
    crop = Image.new("L", (400, 200), 255)
    for y in range(60, 140):
        for x in range(50, 350):
            crop.putpixel((x, y), 0)
    p = tmp_path / "c.png"
    crop.save(p)
    out = pages.crop_scale(page, pages.Box(1, 50, 120, 100, 20), (500.0, 500.0), p, pad_px=0)
    assert out["crop_ink_px"] == 80 and out["page_ink_px"] == 40
    assert out["ratio"] == pytest.approx(2.0, rel=0.05)


SYNTH = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
Text one.
\begin{equation}
a+b=c
\end{equation}
Text two.
\begin{equation}
x^2+y^2=z^2
\end{equation}
\newpage
Page two.
\begin{equation}
e=mc^2
\end{equation}
\end{document}
"""


@pytest.mark.docker
def test_compile_and_synctex_two_pages(tmp_path):
    (tmp_path / "main.tex").write_text(SYNTH)
    res = pages.compile_paper(tmp_path, "main.tex")
    assert res.status == "ok", res.log_tail
    assert res.pdf is not None and res.pdf.exists()
    queries = [("e1", 6, "main.tex"), ("e2", 10, "main.tex"), ("e3", 15, "main.tex")]
    got = pages.parse_synctex(pages.run_synctex(tmp_path, queries, "main.pdf"))
    assert [pages.equation_pages(got[q]) for q, _, _ in queries] == [1, 1, 2]


@pytest.mark.docker
def test_compile_failure_reported(tmp_path):
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\begin{document}\\undefinedcmd\\end{document}")
    res = pages.compile_paper(tmp_path, "main.tex")
    assert res.status == "failed"
    assert res.pdf is None


# ---------------------------------------------------------------- CLI

import io
import json
import tarfile

from glyphfix_math import score, splits
from glyphfix_math.corpus._io import read_jsonl, write_jsonl
from glyphfix_math.ids import safe_id
from scripts import corpus_pages as cli


def _ids():
    bench, corp = [], []
    for n in range(1, 400):
        aid = f"2301.{n:05d}"
        (bench if splits.is_benchmark(aid) else corp).append(aid)
        if bench and len(corp) >= 3:
            break
    return bench[0], corp[:3]


def _make_source(sources, aid):
    sdir = sources / safe_id(aid)
    sdir.mkdir(parents=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = SYNTH.encode()
        ti = tarfile.TarInfo("main.tex")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    (sdir / "src.bin").write_bytes(buf.getvalue())
    (sdir / "fetch.json").write_text(json.dumps({"kind": "tar.gz", "license_uri": "u", "license_spdx": "CC-BY-4.0", "version": "v1"}))


def _shard(shards, aid, n=3):
    recs = [
        {"id": f"{safe_id(aid)}-{i}", "arxiv_id": aid, "paper_version": "v1", "env": "equation", "source_file": "main.tex",
         "source_line": 5 + i, "latex": f"x_{i}=y", "image": f"img/{i}.png", "width": 10, "height": 10}
        for i in range(n)
    ]
    write_jsonl(shards / "records" / f"{safe_id(aid)}.jsonl", recs)
    return recs


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GLYPHFIX_MATH_ROOT", str(tmp_path))
    bench, corp = _ids()
    sources, shards = tmp_path / "src", tmp_path / "shards"
    for aid in corp:
        _make_source(sources, aid)
        _shard(shards, aid)
    return tmp_path, bench, corp, sources, shards


def _fake_pipeline(monkeypatch, mapped):
    """Replace Docker compile and SyncTeX; mapped: arxiv-safe-id -> page per record index (None = unmapped)."""
    from PIL import Image

    def fake_compile(workdir, main_rel, *, timeout_s=0):
        if safe_id_of(workdir) in mapped and mapped[safe_id_of(workdir)] == "fail":
            return pages.CompileResult("failed", None, "boom", 0.1)
        pdf = workdir / "main.pdf"
        import pypdfium2 as pdfium

        d = pdfium.PdfDocument.new()
        d.new_page(612, 792)
        d.save(str(pdf))
        d.close()
        return pages.CompileResult("ok", pdf, "", 0.1)

    def safe_id_of(workdir):
        return workdir.name

    def fake_synctex(workdir, queries, pdf_rel):
        pg = mapped[workdir.name]
        out = []
        for qid, _line, _f in queries:
            idx = int(qid.rsplit("-", 1)[1])
            p = pg[idx] if isinstance(pg, list) else None
            out.append(f"##Q {qid}")
            if p is not None:
                out.append(f"SyncTeX result begin\nPage:{p}\nx:100\ny:200\nh:100\nv:200\nW:50\nH:10\nSyncTeX result end")
        return "\n".join(out) + "\n"

    monkeypatch.setattr(pages, "compile_paper", fake_compile)
    monkeypatch.setattr(pages, "run_synctex", fake_synctex)
    monkeypatch.setattr(pages, "rasterize_page", lambda pdf, i: Image.new("RGB", (100, 140), "white"))


def test_build_skips_and_logs(cli_env, monkeypatch):
    tmp, _bench, corp, sources, shards = cli_env
    _fake_pipeline(monkeypatch, {safe_id(corp[0]): [1, 1, 1], safe_id(corp[1]): "fail", safe_id(corp[2]): [1, 2, None]})
    ids = tmp / "ids.txt"
    ids.write_text("\n".join(corp))
    out = tmp / "out"
    rc = cli.main(["build", "--side", "calib", "--papers", str(ids), "--sources", str(sources), "--shards", str(shards), "--out", str(out), "--n", "5"])
    assert rc == 0
    built = read_jsonl(out / "pages.jsonl")
    assert [r["arxiv_id"] for r in built] == [corp[0]]
    row = built[0]
    for k in ("page_id", "arxiv_id", "paper_version", "license_uri", "license_spdx", "page_index", "n_pages", "image", "width", "height",
              "image_sha256", "gt_ids", "page_version"):
        assert k in row
    log = {e["arxiv_id"]: e for e in read_jsonl(out / "build_log.jsonl")}
    assert log[corp[1]]["reason"] == "compile_failed"
    assert log[corp[2]]["reason"] == "few_eqs"
    assert json.loads((out / "PAGES_SUMMARY.json").read_text())["n_built"] == 1


def test_build_stops_after_n(cli_env, monkeypatch):
    tmp, _b, corp, sources, shards = cli_env
    _fake_pipeline(monkeypatch, {safe_id(a): [1, 1, 1] for a in corp})
    ids = tmp / "ids.txt"
    ids.write_text("\n".join(corp))
    cli.main(["build", "--side", "calib", "--papers", str(ids), "--sources", str(sources), "--shards", str(shards), "--out", str(tmp / "o"), "--n", "2"])
    assert [r["arxiv_id"] for r in read_jsonl(tmp / "o" / "pages.jsonl")] == corp[:2]


def test_build_refuses_wrong_side(cli_env):
    tmp, bench, corp, sources, shards = cli_env
    ids = tmp / "ids.txt"
    ids.write_text("\n".join([corp[0], bench]))
    args = ["--papers", str(ids), "--sources", str(sources), "--shards", str(shards), "--out", str(tmp / "o")]
    assert cli.main(["build", "--side", "calib", *args]) == 2
    ids.write_text(corp[0])
    assert cli.main(["build", "--side", "benchmark", *args]) == 2


def test_rows_and_stats(cli_env, monkeypatch):
    tmp, _b, corp, sources, shards = cli_env
    _fake_pipeline(monkeypatch, {safe_id(a): [1, 1, 1] for a in corp})
    ids = tmp / "ids.txt"
    ids.write_text("\n".join(corp[:2]))
    out = tmp / "out"
    cli.main(["build", "--side", "calib", "--papers", str(ids), "--sources", str(sources), "--shards", str(shards), "--out", str(out)])
    built = read_jsonl(out / "pages.jsonl")
    preds = tmp / "preds.jsonl"
    write_jsonl(preds, [{"id": built[0]["page_id"], "paper_id": built[0]["paper_id"], "pred_text": "$$x_0=y$$\n$$x_1=y$$"},
                        {"id": built[1]["page_id"], "paper_id": built[1]["paper_id"], "pred_error": "timeout"}])
    rows_f, stats_f = tmp / "rows.jsonl", tmp / "stats.json"
    rc = cli.main(["rows", "--pages", str(out / "pages.jsonl"), "--records", str(shards / "records"), "--preds", str(preds),
                   "--out", str(rows_f), "--stats", str(stats_f)])
    assert rc == 0
    assert len(score.read_predictions(rows_f)) == 6
    stats = json.loads(stats_f.read_text())
    assert stats["n_gt"] == 6 and stats["n_matched"] == 2
    assert stats["threshold"] == pages.MATCH_THRESHOLD and stats["page_match_version"] == pages.PAGE_MATCH_VERSION
    assert [p["n_matched"] for p in stats["pages"]] == [2, 0]


def test_crop_scale_cli(cli_env, monkeypatch):
    from PIL import Image

    tmp, _b, corp, sources, shards = cli_env
    _fake_pipeline(monkeypatch, {safe_id(a): [1, 1, 1] for a in corp})
    ids = tmp / "ids.txt"
    ids.write_text(corp[0])
    out = tmp / "out"
    cli.main(["build", "--side", "calib", "--papers", str(ids), "--sources", str(sources), "--shards", str(shards), "--out", str(out)])
    for i in range(3):
        p = tmp / "img" / f"{i}.png"
        p.parent.mkdir(exist_ok=True)
        Image.new("L", (50, 20), 0).save(p)
    res = tmp / "cs.json"
    assert cli.main(["crop-scale", "--pages", str(out / "pages.jsonl"), "--pdf-dir", str(out / "pdf"), "--records", str(shards / "records"),
                     "--images-root", str(tmp), "--out", str(res)]) == 0
    d = json.loads(res.read_text())
    assert set(d) >= {"median", "p10", "p90", "n"}


def _fake_run(returncode, write_pdf, workdir):
    import subprocess as sp

    def run(cmd, **kw):
        if write_pdf:
            (workdir / "main.pdf").write_bytes(b"%PDF-partial")
        return sp.CompletedProcess(cmd, returncode, "", "")

    return run


def test_nonzero_exit_with_partial_pdf_is_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(pages.guard, "preflight", lambda **k: None)
    monkeypatch.setattr(pages.subprocess, "run", _fake_run(12, True, tmp_path))
    (tmp_path / "main.pdf").write_bytes(b"%PDF-stale")
    res = pages.compile_paper(tmp_path, "main.tex")
    assert res.status == "failed" and res.pdf is None
    assert not (tmp_path / "main.pdf").exists()


def test_stale_pdf_removed_before_compile(tmp_path, monkeypatch):
    monkeypatch.setattr(pages.guard, "preflight", lambda **k: None)
    monkeypatch.setattr(pages.subprocess, "run", _fake_run(0, False, tmp_path))
    (tmp_path / "main.pdf").write_bytes(b"%PDF-stale")
    assert pages.compile_paper(tmp_path, "main.tex").status == "failed"


def test_clean_exit_with_pdf_is_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(pages.guard, "preflight", lambda **k: None)
    monkeypatch.setattr(pages.subprocess, "run", _fake_run(0, True, tmp_path))
    res = pages.compile_paper(tmp_path, "main.tex")
    assert res.status == "ok" and res.pdf is not None
