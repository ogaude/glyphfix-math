"""Safe extraction and static equation extraction on hand-built fixtures."""

import gzip
import io
import tarfile

import pytest

from glyphfix_math.corpus import extract as X


def _make_tar(path, members, mode="w:gz"):
    """members: list of (name, bytes | TarInfo-kind tuple)."""
    with tarfile.open(path, mode) as tf:
        for name, spec in members:
            ti = tarfile.TarInfo(name)
            if isinstance(spec, tuple):
                kind, target = spec
                ti.type = kind
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    ti.linkname = target
                tf.addfile(ti)
            else:
                ti.size = len(spec)
                tf.addfile(ti, io.BytesIO(spec))
    return path


def test_safe_extract_ok(tmp_path):
    a = _make_tar(tmp_path / "a.tgz", [("main.tex", b"x"), ("sec/one.tex", b"y")])
    dest = tmp_path / "out"
    assert X.safe_extract(a, "tar.gz", dest) == ["main.tex", "sec/one.tex"]
    assert (dest / "sec" / "one.tex").read_bytes() == b"y"


@pytest.mark.parametrize(
    "member",
    [
        ("../evil", b"x"),
        ("/abs/evil", b"x"),
        ("a/../../evil", b"x"),
        ("ln", (tarfile.SYMTYPE, "/etc/passwd")),
        ("hl", (tarfile.LNKTYPE, "main.tex")),
        ("dev", (tarfile.CHRTYPE, "")),
        ("fifo", (tarfile.FIFOTYPE, "")),
    ],
)
def test_safe_extract_rejects(tmp_path, member):
    a = _make_tar(tmp_path / "a.tgz", [("main.tex", b"x"), member])
    dest = tmp_path / "out"
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(a, "tar.gz", dest)
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()
    assert not dest.exists() or not any(dest.iterdir())


def test_safe_extract_caps(tmp_path):
    a = _make_tar(tmp_path / "a.tgz", [("a.tex", b"x" * 600), ("b.tex", b"x" * 600)])
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(a, "tar.gz", tmp_path / "o1", max_bytes=1000)
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(a, "tar.gz", tmp_path / "o2", max_members=1)
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


def test_safe_extract_gz_and_other(tmp_path):
    g = tmp_path / "s.gz"
    g.write_bytes(gzip.compress(b"\\documentclass{a}"))
    dest = tmp_path / "o"
    assert X.safe_extract(g, "gz", dest) == ["main.tex"]
    assert (dest / "main.tex").read_bytes() == b"\\documentclass{a}"
    g.write_bytes(gzip.compress(b"x" * 5000))
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(g, "gz", tmp_path / "o2", max_bytes=100)
    assert not (tmp_path / "o2" / "main.tex").exists()
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(g, "pdf", tmp_path / "o3")


def test_safe_extract_corrupt(tmp_path):
    b = tmp_path / "bad"
    b.write_bytes(b"not an archive")
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(b, "tar.gz", tmp_path / "o")


def test_safe_extract_corrupt_gzip_stream(tmp_path):
    import gzip

    good = gzip.compress(b"x" * 100000)
    b = tmp_path / "bad.gz"
    b.write_bytes(good[:20] + b"\xff" * 40 + good[60:])
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(b, "gz", tmp_path / "o")
    assert not (tmp_path / "o" / "main.tex").exists()


def _w(root, name, text):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


DOC = "\\documentclass{article}\n\\begin{document}\n"


def test_find_main_prefers_not_included(tmp_path):
    _w(tmp_path, "paper.tex", DOC + "\\input{sub}\n\\end{document}\n")
    _w(tmp_path, "sub.tex", DOC + "x" * 500 + "\n\\end{document}\n")
    _w(tmp_path, "frag.tex", "no class here" * 100)
    assert X.find_main(tmp_path).name == "paper.tex"


def test_find_main_largest_and_none(tmp_path):
    assert X.find_main(tmp_path) is None
    _w(tmp_path, "a.tex", DOC + "x\\end{document}")
    _w(tmp_path, "b.tex", DOC + "x" * 300 + "\\end{document}")
    assert X.find_main(tmp_path).name == "b.tex"
    _w(tmp_path / "n", "c.tex", "%\\documentclass{a}\n\\begin{document}")
    assert X.find_main(tmp_path / "n") is None


def test_document_files(tmp_path):
    main = _w(tmp_path, "main.tex", DOC + "\\input{a}\n\\input b\n\\include{sec/c.tex}\n\\subfile{d}\n% \\input{ghost}\n\\input{../outside}\n")
    _w(tmp_path, "a.tex", "\\input{main}\n")
    _w(tmp_path, "b.tex", "b")
    _w(tmp_path, "sec/c.tex", "c")
    _w(tmp_path, "d.tex", "d")
    _w(tmp_path, "ghost.tex", "g")
    _w(tmp_path.parent, "outside.tex", "o")
    names = sorted(p.name for p in X.document_files(tmp_path, main))
    assert names == ["a.tex", "b.tex", "c.tex", "d.tex", "main.tex"]


def test_strip_comments():
    s = "a % c\nb \\% keep\nc \\\\% gone\nd"
    out = X.strip_comments(s)
    assert out.split("\n") == ["a ", "b \\% keep", "c \\\\", "d"]
    assert out.count("\n") == s.count("\n")


def test_blank_disabled():
    s = "keep\n\\iffalse\nhidden\n\\if x nested \\fi\nstill\n\\fi\nafter\n\\begin{comment}\nzz\n\\end{comment}\nend"
    out = X.blank_disabled(s)
    assert out.count("\n") == s.count("\n")
    assert "hidden" not in out and "still" not in out and "zz" not in out
    assert "keep" in out and "after" in out and "end" in out


def test_extract_display(tmp_path):
    main = _w(tmp_path, "main.tex", DOC + (
        "text \\begin{equation} a=b \\label{x}\n\\end{equation}\n"
        "\\begin{align*}\nx &= \\begin{aligned} y \\end{aligned}\\\\\nz\n\\end{align*}\n"
        "\\begin{gather} g \\end{gather}\n"
        "\\begin{equation} \\begin{split} s \\end{split} \\end{equation}\n"
        "% \\begin{equation} commented \\end{equation}\n"
        "\\iffalse \\begin{equation} off \\end{equation} \\fi\n"
        "\\begin{aligned} not extracted \\end{aligned}\n"
        "\\begin{multline} m \\end{multline}\n"
        "\\input{sub}\n"))
    _w(tmp_path, "sub.tex", "\\begin{equation*} q \\end{equation*}\n")
    _w(tmp_path, "unused.tex", "\\begin{equation} u \\end{equation}\n")
    files = X.document_files(tmp_path, main)
    eqs = X.extract_display(files, tmp_path)
    got = [(e.file, e.line, e.env, e.starred) for e in eqs]
    assert got == [
        ("main.tex", 3, "equation", False),
        ("main.tex", 5, "align", True),
        ("main.tex", 9, "gather", False),
        ("main.tex", 10, "equation", False),
        ("sub.tex", 1, "equation", True),
    ]
    assert eqs[0].body.strip().startswith("a=b") and "\\end{equation}" in eqs[0].raw
    assert "\\begin{split}" in eqs[3].body
    assert "\\begin{aligned}" in eqs[1].body


def test_other_display_counts(tmp_path):
    f = _w(tmp_path, "m.tex", "\\begin{multline}a\\end{multline} \\begin{eqnarray*}\\end{eqnarray*} \\[ x \\] $$y$$ $$z$$ a\\\\[2pt] % \\[\n"
           "\\begin{flalign}\\end{flalign}\\begin{alignat}{2}\\end{alignat}")
    c = X.other_display_counts([f])
    assert c == {"multline": 1, "flalign": 1, "alignat": 1, "eqnarray": 1, "\\[": 1, "$$": 2}


def test_own_macros_and_uses(tmp_path):
    f = _w(tmp_path, "a.tex", "\\newcommand{\\foo}[1]{#1}\n\\renewcommand\\bar{b}\n\\def\\baz{z}\n\\DeclareMathOperator{\\op}{op}\n"
           "\\DeclareRobustCommand{\\rc}{r}\n% \\newcommand{\\ghost}{}\n\\def\\a{x}\n")
    s = _w(tmp_path, "x.sty", "\\providecommand*{\\fromsty}{1}")
    m = X.own_macros([f, s])
    assert m == {"\\a", "\\foo", "\\bar", "\\baz", "\\op", "\\rc", "\\fromsty"}
    assert X.uses_macro("\\foo x", {"\\foo"}) is True
    assert X.uses_macro("\\foobar", {"\\foo"}) is False
    assert X.uses_macro("a\\foo{", {"\\foo"}) is True


def test_macro_files_adds_sty(tmp_path):
    main = _w(tmp_path, "m.tex", DOC)
    _w(tmp_path, "my.sty", "x")
    _w(tmp_path, "c.cls", "x")
    assert sorted(p.name for p in X.macro_files(tmp_path, [main])) == ["c.cls", "m.tex", "my.sty"]


def test_redefined_core(tmp_path):
    f = _w(tmp_path, "a.tex", "\\renewcommand{\\alpha}{a}\n\\def\\prime{p}\n\\renewcommand\\mathbf[1]{#1}\n\\renewcommand{\\foo}{}\n\\def\\bar{x}\n"
           "% \\renewcommand{\\beta}{}\n")
    assert X.redefined_core([f]) == {"\\alpha", "\\prime", "\\mathbf", "\\bar"}


def test_engine_hints(tmp_path):
    f = _w(tmp_path, "a.tex", "%!TEX program = xelatex\n\\usepackage{fontspec}\n\\usepackage{unicode-math}\n% \\usepackage{polyglossia}\n")
    g = _w(tmp_path, "b.tex", "\\usepackage{amsmath}\n")
    assert X.engine_hints([f]) == ["fontspec", "program=xelatex", "unicode-math"]
    assert X.engine_hints([g]) == []


def test_large_files_skipped_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "MAX_SOURCE_BYTES", 50)
    f = _w(tmp_path, "big.tex", "\\begin{equation} a \\end{equation}" + " " * 100)
    skipped = []
    assert X.extract_display([f], tmp_path, skipped) == []
    assert skipped == [str(f)]


def test_safe_extract_file_dir_collision_is_unsafe_archive(tmp_path):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in ("a", "a/b"):
            ti = tarfile.TarInfo(name)
            ti.size = 1
            tf.addfile(ti, io.BytesIO(b"x"))
    src = tmp_path / "src.bin"
    src.write_bytes(buf.getvalue())
    dest = tmp_path / "out"
    with pytest.raises(X.UnsafeArchive):
        X.safe_extract(src, "tar.gz", dest)
    assert not dest.exists()


def test_input_with_nul_byte_does_not_crash(tmp_path):
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\begin{document}\\input{a\x00b}\\end{document}\n")
    main = X.find_main(tmp_path)
    assert main is not None
    assert main in X.document_files(tmp_path, main)
