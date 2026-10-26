"""The canonicalizer is frozen: any rule, version or golden-file change fails here until the freeze file is replaced on purpose."""

import json
from pathlib import Path

from glyphfix_math.corpus import canon

FREEZE = Path(__file__).resolve().parents[1] / "data" / "corpus" / "CANON_FREEZE.json"


def _freeze():
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def test_freeze_file_has_required_keys():
    f = _freeze()
    for k in ("canon_rules_version", "canon_fingerprint", "rules", "frozen_at", "pilot_report_sha256", "decision"):
        assert f[k]


def test_version_and_fingerprint_match_freeze():
    f = _freeze()
    assert canon.CANON_RULES_VERSION == f["canon_rules_version"]
    assert canon.CANON_FINGERPRINT == f["canon_fingerprint"]


def test_recomputed_fingerprint_matches_freeze():
    f = _freeze()
    assert canon.canon_fingerprint() == f["canon_fingerprint"] == canon.CANON_FINGERPRINT


def test_rules_match_freeze():
    assert [list(r) for r in canon.RULES] == _freeze()["rules"]
