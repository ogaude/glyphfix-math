"""rehearse: summary parsing and the pass rule."""

from __future__ import annotations

import copy

from scripts.release import rehearse as R

GOOD = {
    "absent_inputs_ok": True, "planning_paths_ok": True, "evidence_paths_ok": True,
    "uv_sync_exit": 0, "leakscan_exit": 0, "amendment_verify_exit": 0,
    "pytest": {"passed": 10, "failed": 0, "errors": 0, "skipped": 3, "exit": 0},
    "freeze_amendment_tests": {"passed": 30, "failed": 0, "errors": 0, "skipped": 0, "exit": 0},
    "reproduce": {
        "score_exit": 0, "compare": R.VERDICT, "cold_cache": True,
        "history_mapped": {"src/lightonocr_math/acceptance.py": {}, "scripts/acceptance.py": {}},
        "allowed": {"a": "history", **dict.fromkeys(R.AMENDED, "amendment")},
        "amendment": {"verified": True},
    },
    "release_check_git": {"status": "pass"},
}


def test_parse_summary():
    assert R.parse_pytest_summary("..\n1 failed, 10 passed, 2 skipped, 1 error in 2.10s\n") == {
        "passed": 10, "failed": 1, "errors": 1, "skipped": 2}
    assert R.parse_pytest_summary("30 passed, 17 deselected in 1.5s")["passed"] == 30
    assert R.parse_pytest_summary("no summary")["passed"] == 0


def test_good_report_passes():
    assert R.evaluate(GOOD) == []


def test_each_failure_is_named():
    def mutate(path, value):
        r = copy.deepcopy(GOOD)
        d = r
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        return R.evaluate(r)

    assert "pytest" in mutate(("pytest", "failed"), 1)
    assert "freeze_amendment_tests" in mutate(("freeze_amendment_tests", "skipped"), 1)
    assert "reproduce" in mutate(("reproduce", "compare"), "different")
    assert "reproduce.allowed" in mutate(("reproduce", "allowed"), {"code_freeze.x": "amendment"})
    assert "release_check_git" in mutate(("release_check_git", "status"), "fail")
    assert "planning_paths_ok" in mutate(("planning_paths_ok",), False)
    assert "reproduce" in mutate(("reproduce",), None)
