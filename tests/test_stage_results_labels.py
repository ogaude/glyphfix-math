from scripts.release import common as C

SAMPLE = (
    "- An error row does not render identically to the ground truth (D-06). 8,477 rows.\n"
    "- Style rows are still trained under D-09 whether or not you accept a cluster (D-11).\n"
    "- Spacing-only rows count as errors (D-10) and are tagged.\n"
)


def test_mining_labels_removed_and_text_kept():
    out = C.strip_mining_labels(SAMPLE)
    assert not C.INTERNAL_LABEL.search(out)
    for label in ("D-06", "D-09", "D-10", "D-11"):
        assert label not in out
    assert "ground truth. 8,477 rows." in out
    assert "still trained whether or not you accept a cluster." in out
    assert "count as errors and are tagged" in out
