"""Tier 1 reproduction: rescore the published predictions with the frozen scorer and compare to results.json.

Usage (from the repo root; CPU and Docker only, no GPU):
  uv run python -m scripts.release.reproduce [--work DIR] [--published PATH] [--history PATH]
                                             [--amendment PATH] [--warm-cache]

The frozen scorer is run exactly as the published rescore ran it (`scripts.acceptance score --mode benchmark`). The
fresh results.json is compared to the published one through freeze_amendment.compare_published: identical up to the
volatile keys, plus (a) the module commit hash when RELEASE_HISTORY.json maps it with an equal committer date and
(b) the three code_freeze keys of the recorded freeze amendment, after the amendment was re-proven from git objects.

Exit codes: 0 identical (any of the four identical verdicts), 1 different or amendment not verified, 2 not enough
free RAM, 3 the pinned render image is missing, or the scorer's own non-zero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lightonocr_math import guard, render
from scripts.release import freeze_amendment as fa

EXIT_OK = 0
EXIT_DIFFERENT = 1
EXIT_LOW_RAM = 2
EXIT_NO_IMAGE = 3
IDENTICAL = (
    "identical",
    "identical_modulo_history",
    "identical_modulo_amendment",
    "identical_modulo_history_and_amendment",
)
PUBLISHED_DEFAULT = Path("evidence/acceptance/results.json")
HISTORY_DEFAULT = Path("RELEASE_HISTORY.json")
CROPS_REF = "data/benchmark/acceptance/scores/crops_tuned"
PAGES_REF = "data/benchmark/acceptance/scores/pages_tuned"
RAM_HEADROOM_BYTES = 4 * guard.GIB


def compare_results(published: dict, fresh: dict, commit_map: dict | None = None, amendment: dict | None = None) -> dict:
    """The verdict of freeze_amendment.compare_published, unchanged."""
    return fa.compare_published(published, fresh, amendment=amendment, commit_map=commit_map or {})


def load_commit_map(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    return dict(json.loads(Path(path).read_text(encoding="utf-8")).get("commit_map") or {})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def docker_image_ok(image: str = render.IMAGE) -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def score_command(work: Path) -> list[str]:
    return [
        sys.executable, "-m", "scripts.acceptance", "score", "--mode", "benchmark",
        "--scores-dir", str(work / "scores"),
        "--evidence-dir", str(work / "evidence"),
        "--report", str(work / "ACCEPTANCE.md"),
        "--ref-score-crops-tuned", CROPS_REF,
        "--ref-score-pages-tuned", PAGES_REF,
    ]  # fmt: skip


def run_score(work: Path, env: dict | None, cwd: Path | None = None) -> int:
    """Run the frozen scorer (argument list, never a shell string); output streams to <work>/score.log."""
    with open(work / "score.log", "wb") as log:
        return subprocess.run(score_command(work), env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def _write_report(work: Path, report: dict) -> None:
    (work / "reproduce_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import os

    ap = argparse.ArgumentParser(prog="reproduce", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=None, help="work directory (default data/runs/reproduce-<UTC stamp>)")
    ap.add_argument("--published", type=Path, default=PUBLISHED_DEFAULT)
    ap.add_argument("--history", type=Path, default=None, help="RELEASE_HISTORY.json (default: the repo root one when present)")
    ap.add_argument("--amendment", type=Path, default=None, help="FREEZE_AMENDMENT.json (default: evidence/acceptance one when present)")
    ap.add_argument("--warm-cache", action="store_true", help="reuse the normal render cache instead of a fresh empty one")
    a = ap.parse_args(argv)

    root = Path.cwd()
    work = a.work or Path("data/runs") / f"reproduce-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True, exist_ok=True)
    history = a.history if a.history is not None else (HISTORY_DEFAULT if HISTORY_DEFAULT.is_file() else None)
    amendment_path = a.amendment if a.amendment is not None else (fa.AMENDMENT_PATH if fa.AMENDMENT_PATH.is_file() else None)
    start = time.monotonic()

    report: dict = {
        "tier": 1,
        "cold_cache": not a.warm_cache,
        "render_image": render.IMAGE,
        "git_head": _git_head(),
        "published_sha256": _sha256(a.published) if a.published.is_file() else None,
        "fresh_sha256": None,
        "score_exit": None,
        "compare": None,
        "differing_keys": [],
        "allowed": {},
        "history_mapped": {},
        "amendment": {"present": amendment_path is not None, "verified": False, "problems": [], "old_blob": None, "new_blob": None},
        "seconds": None,
    }

    def finish(code: int) -> int:
        report["seconds"] = round(time.monotonic() - start, 1)
        _write_report(work, report)
        print(f"compare: {report['compare']}  score_exit: {report['score_exit']}  report: {work / 'reproduce_report.json'}")
        return code

    if not docker_image_ok():
        print(f"Docker image {render.IMAGE} is not available. Run: docker pull {render.IMAGE}", file=sys.stderr)
        report["compare"] = "not_run: render image missing"
        return finish(EXIT_NO_IMAGE)
    avail = guard.mem_available_bytes()
    if avail < guard.RAM_FLOOR_BYTES + RAM_HEADROOM_BYTES:
        print(f"MemAvailable {avail / guard.GIB:.1f} GiB is below the {(guard.RAM_FLOOR_BYTES + RAM_HEADROOM_BYTES) / guard.GIB:.1f} GiB needed", file=sys.stderr)
        report["compare"] = "not_run: low memory"
        return finish(EXIT_LOW_RAM)

    try:
        published = json.loads(a.published.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"cannot read the published results: {e}", file=sys.stderr)
        report["compare"] = "not_run: published results unreadable"
        return finish(EXIT_DIFFERENT)
    cmap = load_commit_map(history)
    amendment = None
    if amendment_path is not None:
        try:
            amendment = json.loads(Path(amendment_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            report["amendment"]["problems"] = [f"cannot read the amendment: {e}"]
            report["compare"] = "not_run: amendment unreadable"
            return finish(EXIT_DIFFERENT)
        report["amendment"]["old_blob"] = amendment.get("old_blob")
        report["amendment"]["new_blob"] = amendment.get("new_blob")
        problems = fa.verify_amendment(root, amendment, published=published, commit_map=cmap)
        report["amendment"]["problems"] = problems
        report["amendment"]["verified"] = not problems
        if problems:
            for p in problems:
                print(f"amendment: {p}", file=sys.stderr)
            report["compare"] = "not_run: amendment not verified"
            return finish(EXIT_DIFFERENT)

    env = dict(os.environ)
    if not a.warm_cache:
        cache = work / "render-cache"
        cache.mkdir(parents=True, exist_ok=True)
        env["LIGHTONOCR_MATH_CACHE"] = str(cache)

    code = run_score(work, env)
    report["score_exit"] = code
    if code != 0:
        report["compare"] = "not_run: scorer failed"
        return finish(code)

    fresh_path = work / "evidence" / "results.json"
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    report["fresh_sha256"] = _sha256(fresh_path)
    verdict = compare_results(published, fresh, cmap, amendment)
    report["compare"] = verdict["compare"]
    report["differing_keys"] = verdict["differing_keys"]
    report["allowed"] = verdict["allowed"]
    report["history_mapped"] = verdict["history_mapped"]
    if verdict["problems"]:
        report["problems"] = verdict["problems"]
        for p in verdict["problems"]:
            print(f"differs: {p}", file=sys.stderr)
    return finish(EXIT_OK if verdict["compare"] in IDENTICAL else EXIT_DIFFERENT)


if __name__ == "__main__":
    raise SystemExit(main())
