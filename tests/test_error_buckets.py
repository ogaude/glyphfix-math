from scripts import error_buckets as eb


def _row(gt, pred, *, classes=(), exact=False, ok=False):
    return {"gt_latex": gt, "pred_latex": pred, "classes": list(classes), "exact": exact, "render_identical": ok}


def test_not_an_error_is_none():
    assert eb.bucket(_row("x", "x", exact=True, ok=True)) is None


def test_glyph_class_wins():
    assert eb.bucket(_row("\\mathbf{x}", "x", classes=["decoration"])) == "glyph classes"


def test_spacing_only():
    assert eb.bucket(_row("a\\,b", "a b", exact=True)) == "spacing"


def test_wrapper_layout_and_content():
    assert eb.bucket(_row("\\begin{split} a \\end{split}", "a")) == "alignment wrapper"
    assert eb.bucket(_row("\\left( a \\right)", "( a )")) == "layout / sizing"
    assert eb.bucket(_row("a + b", "a - b")) == "content"


def test_render_table_totals():
    out = eb.render(eb.count([_row("a + b", "a - b")]), eb.count([]))
    assert "| content | 1 | 0 | -100% |" in out and "| total | 1 | 0 |" in out
