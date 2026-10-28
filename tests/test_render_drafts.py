import pytest

from scripts.release import render_drafts as rd


def test_fill_names_replaces_every_token():
    text = " ".join(rd.PREVIEW_NAMES)
    out = rd.fill_names(text)
    assert "@@" not in out and "Olivier Gaude" in out


def test_to_html_keeps_front_matter_and_rewrites_image_paths():
    pytest.importorskip("markdown")
    html = rd.to_html("T", "n", "---\nlicense: x\n---\n# H\n\n![a](assets/c.svg)\n\n![b](results/charts/d.svg)\n")
    assert "<pre>license: x</pre>" in html and 'src="card-examples/c.svg"' in html and 'src="card-examples/d.svg"' in html
