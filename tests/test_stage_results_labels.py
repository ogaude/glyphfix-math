from pathlib import Path

from scripts.release import common as C

strip_mining_labels = C.strip_mining_labels

MINING = Path(".planning/phases/04-error-mining-and-lora-training/reports/MINING.md")


def test_mining_labels_removed_and_text_kept():
    out = strip_mining_labels(MINING.read_text(encoding="utf-8"))
    assert not C.INTERNAL_LABEL.search(out)
    assert "D-06" not in out and "D-09" not in out and "D-10" not in out and "D-11" not in out
    assert "still trained whether or not you accept a cluster" in out
    assert "count as errors and are tagged" in out
