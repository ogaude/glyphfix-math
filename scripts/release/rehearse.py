"""Fresh-clone rehearsal: clone the public repository, install, test and reproduce the published numbers there.

Usage (CPU and Docker only, no GPU; nothing is uploaded):
  uv run python -m scripts.release.rehearse [--repo release/code/public.git] [--work release/rehearsal/<UTC>]
                                            [--skip-reproduce]

The clone has none of the gitignored inputs (images, shards, model files, reserved rows) and starts with an empty
render cache. REHEARSAL.json in the work directory records every exit status and count; paths in it are relative.
`--skip-reproduce` exists for iterating on the other steps and never produces a passing report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.release import common as C
from scripts.release import freeze_amendment as FA
from scripts.release import rename_amendment as RA

ABSENT_INPUTS = (
    "data/corpus/images",
    "data/benchmark/images",
    "data/benchmark/shards",
    "data/models/p4-full-1",
    "data/corpus/train/records.jsonl",
)
VERDICT = "identical_modulo_history_and_amendment"
AMENDED = set(FA.AMENDED_KEYS)
MIN_AVAILABLE_BYTES = 3 * 1024**3


def parse_pytest_summary(text: str) -> dict:
    """Counts from the last pytest summary line, for example `3 failed, 10 passed, 2 skipped, 1 error in 2.1s`."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    lines = [ln for ln in text.splitlines() if re.search(r"\b(passed|failed|error|errors|skipped)\b.* in [\d.]+s", ln)]
    if lines:
        for n, word in re.findall(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)", lines[-1]):
            key = {"error": "errors"}.get(word, word)
            if key in counts:
                counts[key] = int(n)
    return counts


def evaluate(r: dict) -> list[str]:
    """The reasons a rehearsal report does not pass (empty when it passes)."""
    bad: list[str] = []
    for k in ("absent_inputs_ok", "planning_paths_ok", "evidence_paths_ok"):
        if not r.get(k):
            bad.append(k)
    for k in ("uv_sync_exit", "leakscan_exit", "amendment_verify_exit"):
        if r.get(k) != 0:
            bad.append(k)
    p = r.get("pytest") or {}
    if p.get("failed") != 0 or p.get("errors") != 0 or not p.get("passed"):
        bad.append("pytest")
    fa = r.get("freeze_amendment_tests") or {}
    if fa.get("exit") != 0 or fa.get("failed") != 0 or fa.get("errors") != 0 or fa.get("skipped") != 0 or not fa.get("passed"):
        bad.append("freeze_amendment_tests")
    renamed = bool(r.get("rename_amendment_present"))
    if renamed:
        ra = r.get("rename_amendment_tests") or {}
        if ra.get("exit") != 0 or ra.get("failed") != 0 or ra.get("errors") != 0 or ra.get("skipped") != 0 or not ra.get("passed"):
            bad.append("rename_amendment_tests")
    history_module = RA.NEW_MODULE if renamed else FA.FROZEN_MODULE
    amended = set(RA.RENAME_KEYS) if renamed else AMENDED
    rp = r.get("reproduce")
    if not isinstance(rp, dict) or rp.get("score_exit") != 0 or rp.get("compare") != VERDICT or not rp.get("cold_cache"):
        bad.append("reproduce")
    else:
        if set(rp.get("history_mapped") or {}) != {history_module, FA.FROZEN_CLI}:
            bad.append("reproduce.history_mapped")
        if {k for k, v in (rp.get("allowed") or {}).items() if v == "amendment"} != amended:
            bad.append("reproduce.allowed")
        am = rp.get("amendment") or {}
        if not am.get("verified") or (renamed and not am.get("rename_verified")):
            bad.append("reproduce.amendment")
    if (r.get("release_check_git") or {}).get("status") != "pass":
        bad.append("release_check_git")
    return bad


class Rehearsal:
    def __init__(self, repo: Path, work: Path):
        self.repo = repo.resolve()
        self.work = work.resolve()
        self.clone = self.work / "clone"
        self.logs = self.work / "logs"
        self.uv_cache = subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True, check=True).stdout.strip()
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        (self.work / "cache" / "render").mkdir(parents=True, exist_ok=True)
        (self.work / "cache" / "xdg").mkdir(parents=True, exist_ok=True)
        env["LIGHTONOCR_MATH_CACHE"] = str(self.work / "cache" / "render")
        env["XDG_CACHE_HOME"] = str(self.work / "cache" / "xdg")
        # The development checkout's virtualenv must not leak into the clone through PATH.
        env["PATH"] = os.pathsep.join(d for d in env.get("PATH", "").split(os.pathsep) if not re.search(r"/\.venv[^/]*/bin/?$", d))
        env["UV_CACHE_DIR"] = self.uv_cache  # package downloads are not part of what is being rehearsed
        self.env = env

    def run(self, name: str, cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
        self.logs.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        r = subprocess.run(cmd, cwd=cwd or self.clone, env=self.env, capture_output=True, text=True, check=False)
        out = r.stdout + r.stderr
        (self.logs / f"{name}.log").write_text(out, encoding="utf-8")
        print(f"[{name}] exit {r.returncode} in {time.monotonic() - t:.0f}s", flush=True)
        return r.returncode, out

    def git_lines(self, *args: str) -> list[str]:
        out = subprocess.run(["git", "-C", str(self.clone), *args], capture_output=True, text=True, check=True).stdout
        return [ln for ln in out.splitlines() if ln]


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def rehearse(repo: Path, work: Path, skip_reproduce: bool = False) -> dict:
    if mem_available() < MIN_AVAILABLE_BYTES:
        raise SystemExit("MemAvailable is below 3 GiB; not starting the rehearsal")
    start = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    h = Rehearsal(repo, work)
    subprocess.run(["git", "clone", "-q", "--no-local", str(h.repo), str(h.clone)], check=True)
    # The release check requires a repository without remotes; the throwaway clone does not need its origin.
    subprocess.run(["git", "-C", str(h.clone), "remote", "remove", "origin"], check=True)
    report: dict = {"public_head": h.git_lines("rev-parse", "HEAD")[0]}
    report["absent_inputs_ok"] = not any((h.clone / p).exists() for p in ABSENT_INPUTS)
    report["planning_paths_ok"] = h.git_lines("ls-files", ".planning", "docs") == []
    report["evidence_paths_ok"] = sorted(h.git_lines("ls-files", "evidence")) == sorted(C.PUBLIC_EVIDENCE_FILES)

    report["uv_sync_exit"], _ = h.run("uv_sync", ["uv", "sync", "--frozen"])
    code, out = h.run("pytest", ["uv", "run", "--frozen", "pytest", "-q", "-m", "not docker", "-p", "no:cacheprovider"])
    report["pytest"] = {**parse_pytest_summary(out), "exit": code}
    code, out = h.run(
        "freeze_amendment_tests",
        ["uv", "run", "--frozen", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests/test_freeze_amendment.py"],
    )
    report["freeze_amendment_tests"] = {**parse_pytest_summary(out), "exit": code}
    report["rename_amendment_present"] = (h.clone / RA.AMENDMENT2_PATH).is_file()
    if report["rename_amendment_present"]:
        code, out = h.run(
            "rename_amendment_tests",
            ["uv", "run", "--frozen", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests/test_rename_amendment.py"],
        )
        report["rename_amendment_tests"] = {**parse_pytest_summary(out), "exit": code}
    report["leakscan_exit"], _ = h.run("leakscan", ["uv", "run", "--frozen", "python", "-m", "lightonocr_math.leakscan"])
    report["amendment_verify_exit"], _ = h.run(
        "amendment_verify", ["uv", "run", "--frozen", "python", "-m", "scripts.release.freeze_amendment", "verify"]
    )
    if skip_reproduce:
        report["reproduce"] = None
    else:
        if mem_available() < MIN_AVAILABLE_BYTES:
            raise SystemExit("MemAvailable is below 3 GiB before the cold rerun; not starting it")
        rep_dir = Path("data/runs/reproduce")
        code, _ = h.run("reproduce", ["uv", "run", "--frozen", "python", "-m", "scripts.release.reproduce", "--work", str(rep_dir)])
        path = h.clone / rep_dir / "reproduce_report.json"
        report["reproduce"] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"score_exit": code, "compare": None}
        report["reproduce_exit"] = code
    rc_path = h.work / "release_check_git.json"
    code, _ = h.run(
        "release_check_git", ["uv", "run", "--frozen", "python", "-m", "scripts.release.release_check", "git", "--repo", ".", "--report", str(rc_path)]
    )
    rc = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.is_file() else {}
    git_report = next((x for x in rc.get("reports", []) if x.get("surface") == "git"), {})
    report["release_check_git"] = {
        "status": rc.get("status"),
        "exit": code,
        "checks": [{"name": c["name"], "status": c["status"]} for c in git_report.get("checks", [])],
    }
    report["seconds"] = round(time.monotonic() - start, 1)
    report["failures"] = evaluate(report) if not skip_reproduce else ["reproduce skipped"]
    text = json.dumps(report, indent=2, sort_keys=True).replace(str(h.work), "<work>").replace(str(h.repo), "<repo>")
    (h.work / "REHEARSAL.json").write_text(text + "\n", encoding="utf-8")
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rehearse", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("release/code/public.git"))
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--skip-reproduce", action="store_true")
    a = ap.parse_args(argv)
    work = a.work or Path("release/rehearsal") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rep = rehearse(a.repo, work, a.skip_reproduce)
    print(json.dumps({"failures": rep["failures"], "report": str(work / "REHEARSAL.json")}, indent=2))
    return 0 if not rep["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
