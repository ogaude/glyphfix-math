"""Tier 1 reproduction: rescore the published predictions with the frozen scorer and compare to results.json.

Usage (from the repo root; CPU and Docker only, no GPU):
  uv run python -m scripts.release.reproduce [--work DIR] [--published PATH] [--history PATH]
                                             [--amendment PATH] [--warm-cache] [--score-only]

The frozen scorer is run exactly as the published rescore ran it (`scripts.acceptance score --mode benchmark`), through
scripts.release.frozen_rescore: the frozen pin check refuses a module whose last commit is later than the generation
start, and the project rename made the rename commit that last commit; the harness supplies the pre-rename pin commit
after proving the module is the pinned blob renamed, and changes nothing else. The whole amendment chain is verified
from git objects before scoring. The fresh results.json is compared to the published one through
freeze_amendment.compare_published: identical up to the volatile keys, plus (a) the module commit hash when
RELEASE_HISTORY.json maps it with an equal committer date, (b) the three code_freeze keys of the recorded freeze
amendment and (c) the eight keys of the recorded rename amendment, when those records are present.

--score-only runs the scorer and writes its outputs and reproduce_report.json, nothing more: it neither verifies nor
compares, and its report can never hold an identical verdict. It exists to record an amendment.

Exit codes: 0 identical (any of the four identical verdicts) or, with --score-only, a successful score, 1 different or
the amendment chain not verified, 2 not enough free RAM, 3 the pinned render image is missing, or the scorer's own
non-zero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from glyphfix_math import guard, render
from scripts.release import freeze_amendment as fa
from scripts.release import frozen_rescore
from scripts.release import rename_amendment as ra

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
CACHE_ENV = "GLYPHFIX_MATH_CACHE"
HARNESS = "scripts.release.frozen_rescore"


def compare_results(
    published: dict, fresh: dict, commit_map: dict | None = None, amendment: dict | None = None, rename: dict | None = None
) -> dict:
    """The verdict of freeze_amendment.compare_published, unchanged."""
    return fa.compare_published(published, fresh, amendment=amendment, commit_map=commit_map or {}, rename=rename)


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
        sys.executable, "-m", HARNESS, "score", "--mode", "benchmark",
        "--scores-dir", str(work / "scores"),
        "--evidence-dir", str(work / "evidence"),
        "--report", str(work / "ACCEPTANCE.md"),
        "--ref-score-crops-tuned", CROPS_REF,
        "--ref-score-pages-tuned", PAGES_REF,
    ]  # fmt: skip


def preflight() -> tuple[int, str]:
    """(0, "") when the render image exists and enough RAM is free, else (exit code, message)."""
    if not docker_image_ok():
        return EXIT_NO_IMAGE, f"Docker image {render.IMAGE} is not available. Run: docker pull {render.IMAGE}"
    avail = guard.mem_available_bytes()
    need = guard.RAM_FLOOR_BYTES + RAM_HEADROOM_BYTES
    if avail < need:
        return EXIT_LOW_RAM, f"MemAvailable {avail / guard.GIB:.1f} GiB is below the {need / guard.GIB:.1f} GiB needed"
    return 0, ""


def cold_env(work: Path, warm: bool) -> dict:
    """The environment for the scorer: a fresh empty render cache under work unless warm."""
    env = dict(os.environ)
    if not warm:
        cache = Path(work) / "render-cache"
        cache.mkdir(parents=True, exist_ok=True)
        env[CACHE_ENV] = str(cache)
    return env


def run_score(work: Path, env: dict | None, cwd: Path | None = None) -> int:
    """Run the frozen scorer (argument list, never a shell string); output streams to <work>/score.log."""
    with open(work / "score.log", "wb") as log:
        return subprocess.run(score_command(work), env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def _write_report(work: Path, report: dict) -> None:
    (work / "reproduce_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pin_substitution(root: Path, cmap: dict) -> dict | None:
    """The harness substitution when the tree is renamed, else None; raises SubstitutionRefused when not provable."""
    if not (root / ra.NEW_MODULE).is_file():
        return None
    sub = frozen_rescore.pin_substitution(root, cmap)
    return {k: sub[k] for k in ("path", "commit", "date")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reproduce", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=None, help="work directory (default data/runs/reproduce-<UTC stamp>)")
    ap.add_argument("--published", type=Path, default=PUBLISHED_DEFAULT)
    ap.add_argument("--history", type=Path, default=None, help="RELEASE_HISTORY.json (default: the repo root one when present)")
    ap.add_argument("--amendment", type=Path, default=None, help="FREEZE_AMENDMENT.json (default: evidence/acceptance one when present)")
    ap.add_argument("--warm-cache", action="store_true", help="reuse the normal render cache instead of a fresh empty one")
    ap.add_argument("--score-only", action="store_true", help="score only, for recording an amendment; never a reproduction verdict")
    a = ap.parse_args(argv)

    root = Path.cwd()
    work = a.work or Path("data/runs") / f"reproduce-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True, exist_ok=True)
    history = a.history if a.history is not None else (HISTORY_DEFAULT if HISTORY_DEFAULT.is_file() else None)
    amendment_path = a.amendment if a.amendment is not None else (fa.AMENDMENT_PATH if fa.AMENDMENT_PATH.is_file() else None)
    rename_path = root / ra.AMENDMENT2_PATH
    start = time.monotonic()

    report: dict = {
        "tier": 1,
        "cold_cache": not a.warm_cache,
        "render_image": render.IMAGE,
        "git_head": _git_head(),
        "harness": HARNESS,
        "pin_substitution": None,
        "published_sha256": _sha256(a.published) if a.published.is_file() else None,
        "fresh_sha256": None,
        "score_exit": None,
        "compare": None,
        "differing_keys": [],
        "allowed": {},
        "history_mapped": {},
        "amendment": {
            "present": amendment_path is not None,
            "verified": False,
            "problems": [],
            "old_blob": None,
            "new_blob": None,
            "rename_present": rename_path.is_file(),
            "rename_verified": False,
        },
        "seconds": None,
    }

    def finish(code: int) -> int:
        report["seconds"] = round(time.monotonic() - start, 1)
        _write_report(work, report)
        print(f"compare: {report['compare']}  score_exit: {report['score_exit']}  report: {work / 'reproduce_report.json'}")
        return code

    code, message = preflight()
    if code:
        print(message, file=sys.stderr)
        report["compare"] = "not_run: render image missing" if code == EXIT_NO_IMAGE else "not_run: low memory"
        return finish(code)

    cmap = load_commit_map(history)

    if a.score_only:
        try:
            report["pin_substitution"] = _pin_substitution(root, cmap)
        except frozen_rescore.SubstitutionRefused as e:
            print(f"pin substitution refused: {e}", file=sys.stderr)  # the scorer refuses too, with exit 2
        report["compare"] = "not_run: score-only"
        report["score_exit"] = run_score(work, cold_env(work, a.warm_cache))
        return finish(report["score_exit"])

    try:
        published = json.loads(a.published.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"cannot read the published results: {e}", file=sys.stderr)
        report["compare"] = "not_run: published results unreadable"
        return finish(EXIT_DIFFERENT)
    amendment = None
    rename = None
    try:
        if amendment_path is not None:
            amendment = json.loads(Path(amendment_path).read_text(encoding="utf-8"))
            report["amendment"]["old_blob"] = amendment.get("old_blob")
            report["amendment"]["new_blob"] = amendment.get("new_blob")
        if rename_path.is_file():
            rename = json.loads(rename_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        report["amendment"]["problems"] = [f"cannot read an amendment record: {e}"]
        report["compare"] = "not_run: amendment unreadable"
        return finish(EXIT_DIFFERENT)
    if amendment is not None or rename is not None:
        problems = fa.verify_chain(root, published=published, commit_map=cmap)
        report["amendment"]["problems"] = problems
        report["amendment"]["verified"] = not problems if amendment is not None else False
        report["amendment"]["rename_verified"] = (not problems) if rename is not None else False
        if problems:
            for p in problems:
                print(f"amendment: {p}", file=sys.stderr)
            report["compare"] = "not_run: amendment not verified"
            return finish(EXIT_DIFFERENT)

    try:
        report["pin_substitution"] = _pin_substitution(root, cmap)
    except frozen_rescore.SubstitutionRefused as e:
        print(f"pin substitution refused: {e}", file=sys.stderr)
        report["amendment"]["problems"] = [f"pin substitution refused: {e}"]
        report["compare"] = "not_run: pin substitution refused"
        return finish(EXIT_DIFFERENT)

    code = run_score(work, cold_env(work, a.warm_cache))
    report["score_exit"] = code
    if code != 0:
        report["compare"] = "not_run: scorer failed"
        return finish(code)

    fresh_path = work / "evidence" / "results.json"
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    report["fresh_sha256"] = _sha256(fresh_path)
    verdict = compare_results(published, fresh, cmap, amendment, rename)
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
