import pytest

from lightonocr_math import render
from lightonocr_math.corpus import pages


def test_latexmk_cmd_is_hardened(tmp_path):
    cmd = pages.latexmk_docker_cmd("n", tmp_path, "main.tex")
    assert cmd[cmd.index("--network") + 1] == "none"
    for flag in ("--read-only", "--cap-drop", "ALL", render.IMAGE, "latexmk", "-pdf", "-synctex=1", "-interaction=nonstopmode"):
        assert flag in cmd
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
    from lightonocr_math import score
    from lightonocr_math.corpus._io import write_jsonl

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
