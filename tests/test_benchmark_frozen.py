import hashlib
import json
from collections import Counter
from pathlib import Path

from lightonocr_math import splits
from lightonocr_math.corpus import _benchmark_frozen
from lightonocr_math.corpus._io import read_jsonl, sha256_file
from lightonocr_math.corpus.records import validate_record

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmark"
FREEZE = BENCH / "BENCHMARK_FREEZE.json"
RECORDS = BENCH / "records.jsonl"


def _freeze() -> dict:
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def test_freeze_file_matches_pinned_digest():
    assert _benchmark_frozen.BENCHMARK_FREEZE_SHA256 == hashlib.sha256(FREEZE.read_bytes()).hexdigest()


def test_records_match_freeze():
    f = _freeze()
    recs = read_jsonl(RECORDS)
    assert f["records_sha256"] == sha256_file(RECORDS)
    assert f["n_records"] == len(recs)
    assert f["n_papers"] == len({r["arxiv_id"] for r in recs})


def test_only_benchmark_papers():
    fs = splits.frozen()
    bad = sorted({r["arxiv_id"] for r in read_jsonl(RECORDS) if not splits.is_benchmark(r["arxiv_id"], fs)})
    assert not bad, bad


def test_records_valid_and_capped():
    recs = read_jsonl(RECORDS)
    problems = {r["id"]: p for r in recs if (p := validate_record(r))}
    assert not problems, list(problems.items())[:5]
    assert len({r["id"] for r in recs}) == len(recs)
    per_paper = Counter(r["arxiv_id"] for r in recs)
    assert max(per_paper.values()) <= 30


def test_images_digest_matches_freeze():
    recs = sorted(read_jsonl(RECORDS), key=lambda r: r["id"])
    lines = "".join(f"{r['id']} {r['image_sha256']}\n" for r in recs)
    assert _freeze()["images_sha256"] == hashlib.sha256(lines.encode()).hexdigest()


def test_image_files_match_records_when_present():
    import pytest

    if not (BENCH / "images").is_dir():
        pytest.skip("data/benchmark/images/ absent (gitignored)")
    bad = []
    for r in read_jsonl(RECORDS):
        p = ROOT / r["image"]
        if p.exists() and sha256_file(p) != r["image_sha256"]:
            bad.append(r["id"])
    assert not bad, bad[:5]


PAGES = BENCH / "pages"
PAGES_FREEZE = PAGES / "PAGES_FREEZE.json"
PAGES_ROWS = PAGES / "pages.jsonl"


def _pages_freeze() -> dict:
    return json.loads(PAGES_FREEZE.read_text(encoding="utf-8"))


def test_pages_freeze_matches_pinned_digest():
    assert _benchmark_frozen.PAGES_FREEZE_SHA256 == hashlib.sha256(PAGES_FREEZE.read_bytes()).hexdigest()


def test_pages_match_freeze():
    from lightonocr_math.corpus import pages as pages_mod

    f = _pages_freeze()
    rows = read_jsonl(PAGES_ROWS)
    assert f["pages_sha256"] == sha256_file(PAGES_ROWS)
    assert f["n_pages"] == len(rows)
    assert f["n_papers"] == len({r["arxiv_id"] for r in rows})
    assert f["n_gt_equations"] == sum(len(r["gt_ids"]) for r in rows)
    assert f["match_threshold"] == pages_mod.MATCH_THRESHOLD
    assert f["page_match_version"] == pages_mod.PAGE_MATCH_VERSION
    assert f["benchmark_freeze_sha256"] == _benchmark_frozen.BENCHMARK_FREEZE_SHA256
    lines = "".join(f"{r['page_id']} {r['image_sha256']}\n" for r in sorted(rows, key=lambda r: r["page_id"]))
    assert f["images_sha256"] == hashlib.sha256(lines.encode()).hexdigest()


def test_pages_are_benchmark_papers_one_page_each():
    rows = read_jsonl(PAGES_ROWS)
    bench_papers = {r["arxiv_id"] for r in read_jsonl(RECORDS)}
    assert {r["arxiv_id"] for r in rows} <= bench_papers
    assert len({r["page_id"] for r in rows}) == len(rows)
    assert len({r["arxiv_id"] for r in rows}) == len(rows)
    assert all(len(r["gt_ids"]) >= 3 and r["license_spdx"] and r["image_sha256"] for r in rows)


def test_page_images_match_rows_when_present():
    import pytest

    if not (PAGES / "images").is_dir():
        pytest.skip("data/benchmark/pages/images/ absent (gitignored)")
    bad = [r["page_id"] for r in read_jsonl(PAGES_ROWS) if (ROOT / r["image"]).exists() and sha256_file(ROOT / r["image"]) != r["image_sha256"]]
    assert not bad, bad[:5]
