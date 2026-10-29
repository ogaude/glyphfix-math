import xml.dom.minidom
from collections import Counter

from scripts import error_buckets
from scripts.release import charts

RESULTS = {"headline": {"n": 100, "n_papers": 10, "stock_rate": 0.4, "tuned_rate": 0.65, "delta": 0.25, "ci_low": 0.2, "ci_high": 0.3}}


def test_headline_chart_carries_the_numbers_and_is_valid_xml():
    svg = charts.headline_chart(RESULTS)
    xml.dom.minidom.parseString(svg)
    assert "40.0%" in svg and "65.0%" in svg and "+25.0 points" in svg and "+20.0 to +30.0" in svg


def test_buckets_chart_has_every_bucket_and_count():
    stock = Counter({"content": 10, "spacing": 4})
    tuned = Counter({"content": 6})
    svg = charts.buckets_chart(stock, tuned)
    xml.dom.minidom.parseString(svg)
    for name in error_buckets.BUCKETS:
        assert name in svg
    assert ">10<" in svg and ">6<" in svg and "Total: 14 failures" in svg


def test_bars_start_at_zero_and_scale_with_count():
    svg = charts.buckets_chart(Counter({"content": 10, "spacing": 5}), Counter())
    widths = [float(w) for w in __import__("re").findall(r'class="base" x="230" y="[\d.]+" width="([\d.]+)"', svg)]
    assert widths[0] > widths[3] > 0 and abs(widths[0] / widths[3] - 2) < 0.05


def test_flips_chart_counts_and_regression_buckets():
    results = {"headline": {"flip": {"both_right": 5, "wrong_to_right": 3, "right_to_wrong": 2, "both_wrong": 4}, "r2w_ids": ["a", "b"]}}
    rows = {
        "a": {"gt_latex": "a + b", "pred_latex": "a - b", "classes": [], "exact": False, "render_identical": False},
        "b": {"gt_latex": "a\\,b", "pred_latex": "a b", "classes": [], "exact": True, "render_identical": False},
    }
    svg = charts.flips_chart(results, rows)
    xml.dom.minidom.parseString(svg)
    assert "All 14 benchmark crops" in svg and "The 2 crops this model got wrong" in svg
    assert ">1<" in svg


def test_pipeline_chart_is_valid_xml_with_every_step():
    svg = charts.pipeline_chart()
    xml.dom.minidom.parseString(svg)
    for name, _ in charts.PIPELINE:
        assert name in svg


def test_write_pipeline_writes_only_the_pipeline_svg(tmp_path):
    name = charts.write_pipeline(tmp_path / "assets")
    assert name == "chart_pipeline.svg" and [p.name for p in (tmp_path / "assets").iterdir()] == [name]
    xml.dom.minidom.parseString((tmp_path / "assets" / name).read_text(encoding="utf-8"))


REVIEW = {
    "labels": {
        "a": {"direction": "wrong_to_right", "label": "real", "why": ""},
        "b": {"direction": "wrong_to_right", "label": "borderline", "why": ""},
        "c": {"direction": "wrong_to_right", "label": "style", "why": ""},
        "d": {"direction": "right_to_wrong", "label": "real", "why": ""},
        "e": {"direction": "right_to_wrong", "label": "style", "why": ""},
    },
    "mechanical_style_fixes": {"count": 5},
}


def test_review_chart_counts_and_ratio():
    assert charts.review_counts(REVIEW) == {
        "fixed": {"real": 1, "borderline": 1, "style": 6},
        "broken": {"real": 1, "borderline": 0, "style": 1},
    }
    svg = charts.review_chart(REVIEW)
    xml.dom.minidom.parseString(svg)
    assert "Fixed: 8 crops" in svg and "Broken: 2 crops" in svg and "Real fixes per real break: 1.0 (up to 2.0" in svg


def test_review_chart_drops_the_borderline_bar_once_resolved():
    review = {"labels": {"a": {"direction": "wrong_to_right", "label": "real", "why": ""}, "b": {"direction": "right_to_wrong", "label": "real", "why": ""}}, "mechanical_style_fixes": {"count": 0}}
    svg = charts.review_chart(review)
    xml.dom.minidom.parseString(svg)
    assert "Borderline" not in svg and "Real fixes per real break: 1.0." in svg
