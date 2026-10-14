from pathlib import Path

import pytest

from lightonocr_math.corpus import gate
from lightonocr_math.corpus._io import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
BENCH = ROOT / "data" / "benchmark"
TRAIN = CORPUS / "train" / "records.jsonl"
DEV = CORPUS / "dev" / "records.jsonl"
BRECS = BENCH / "records.jsonl"

pytestmark = pytest.mark.skipif(
    not (TRAIN.exists() and DEV.exists() and BRECS.exists()), reason="built manifests absent"
)


def _catalog_rows() -> list[dict]:
    rows: list[dict] = []
    for p in (ROOT / "data" / "catalog" / "train_dev.jsonl", BENCH / "catalog.jsonl"):
        if p.exists():
            rows += read_jsonl(p)
    return rows


def test_gate_clean_over_built_manifests():
    gate.assert_clean(read_jsonl(TRAIN) + read_jsonl(DEV), read_jsonl(BRECS), _catalog_rows())


def test_corpus_side_scan_clean():
    gate.assert_side_clean(CORPUS)


def test_no_record_id_in_more_than_one_split():
    sets = {name: {r["id"] for r in read_jsonl(p)} for name, p in (("train", TRAIN), ("dev", DEV), ("benchmark", BRECS))}
    assert not (sets["train"] & sets["dev"])
    assert not (sets["train"] & sets["benchmark"])
    assert not (sets["dev"] & sets["benchmark"])
