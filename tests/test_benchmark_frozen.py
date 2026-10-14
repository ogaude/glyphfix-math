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
