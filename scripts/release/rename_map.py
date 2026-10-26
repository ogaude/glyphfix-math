"""The one definition of the project rename: token map, directory moves, allow-list, scanner and rewriter.

Usage:
  uv run python -m scripts.release.rename_map scan [--root DIR] [--json OUT]
  uv run python -m scripts.release.rename_map apply (--dry-run | --write) [--root DIR] [--json OUT]
  uv run python -m scripts.release.rename_map trial --work DIR [--tests PATH ...] [--paths PATHSPEC ...]
                                                    [--overlay SRC=DEST ...] [--set-exp001-tree]
                                                    [--allow-red PATTERN ...] [--json OUT]

`scan` lists every old-name occurrence (file content and tracked path names) outside the allow-list and exits 1 when
any remains. `apply --dry-run` reports what would change and writes nothing. `apply --write` performs the directory
moves and the rewrite on tracked files and never commits. `trial` performs the rename in a throwaway clone, runs the
test suite there and in an unrenamed clone, and reports the tests that only the rename turns red (the source
repository is never written).

This file spells the old names as data and is therefore itself on the allow-list.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.release.public_paths import _glob_to_regex

TOKEN_MAP: tuple[tuple[str, str], ...] = (
    ("mathocr-lora-q8", "glyphfix-math-lora-q8"),
    ("mathocr-cand", "glyphfix-math-cand"),
    ("mathocr-lora", "glyphfix-math-lora"),
    ("exp001_lighton", "exp001_baseline"),
    ("lightonocr_math", "glyphfix_math"),
    ("LIGHTONOCR_MATH", "GLYPHFIX_MATH"),
    ("lightonocr-math", "glyphfix-math"),
)
PATH_MOVES: tuple[tuple[str, str], ...] = (
    ("src/lightonocr_math", "src/glyphfix_math"),
    ("experiments/exp001_lighton", "experiments/exp001_baseline"),
)
PHASE_DIR = ".planning/phases/06.1-rename-lightonocr-math-to-glyphfix-math-package-experiments-"

# (glob, reason, pinned_by): files that keep their old names because something verifies them byte for byte.
ALLOW_FILES: tuple[tuple[str, str, str], ...] = (
    (f"{PHASE_DIR}/**", "this phase's own documents describe the rename", "none: documents of the rename itself"),
    ("data/benchmark/acceptance/*.pred.manifest.json", "recorded prediction manifests",
     "sha256 in evidence/acceptance/results.json inputs"),
    ("data/benchmark/acceptance/*.pred.jsonl", "recorded predictions", "sha256 in evidence/acceptance/results.json inputs"),
    ("data/benchmark/acceptance/RUN_LOCK.json",
     "generation lock; its blob fields are edited by freeze amendment 2, its model field keeps the tag that generated the predictions",
     "blob fields verified by freeze_amendment and rename_amendment verify"),
    ("data/benchmark/acceptance/*.sidecar.jsonl",
     "per-row record of the same run; kept so it agrees with the pinned manifests and the RUN_LOCK model field",
     "none: unpinned exemption, listed for user acceptance at the plan 08 checkpoint"),
    ("evidence/acceptance/results.json", "published results",
     "sha256 f506e7b4... in freeze_amendment.PUBLISHED_RESULTS_SHA256 and data/models/REGISTRY.json"),
    ("evidence/acceptance/env_check.json", "published environment check", "sha256 in FREEZE_AMENDMENT.json moved list"),
    ("evidence/acceptance/FREEZE_AMENDMENT.json", "amendment 1 record",
     "verification input: module_file and registry key fields are compared with freeze_amendment's amendment-1 constants"),
    ("evidence/acceptance/FREEZE_AMENDMENT_2.json", "amendment 2 record states the token map",
     "verification input of rename_amendment verify"),
    ("scripts/release/rename_map.py", "old names as data for the recorded history and its proofs",
     "verification code: its constants must equal the recorded history"),
    ("scripts/release/rename_amendment.py", "old names as data for the recorded history and its proofs",
     "verification code: its constants must equal the recorded history"),
    ("scripts/release/freeze_amendment.py", "old names as data for the recorded history and its proofs",
     "verification code: its constants must equal the recorded history"),
    ("tests/test_rename_map.py", "tests of that tooling", "fixtures for the verification code"),
    ("tests/test_rename_amendment.py", "tests of that tooling", "fixtures for the verification code"),
    ("tests/test_rename_real_data.py", "tests of that tooling", "fixtures for the verification code"),
    ("tests/test_freeze_amendment.py", "tests of that tooling", "fixtures for the verification code"),
)
_ALLOW_RE = [(_glob_to_regex(g), g, r, p) for g, r, p in ALLOW_FILES]

_BY_LENGTH = sorted(TOKEN_MAP, key=lambda t: -len(t[0]))
_NEW = dict(TOKEN_MAP)
OLD_RE = re.compile("|".join(re.escape(old) for old, _ in _BY_LENGTH))
PROTECTED_RE = re.compile(r"rename[ -]lightonocr-math[ -]to[ -]glyphfix-math", re.IGNORECASE)
_COMBINED = re.compile(f"(?P<prot>(?i:{PROTECTED_RE.pattern}))|(?P<tok>{OLD_RE.pattern})")
_HISTORY_MARK = "renamed from"


class RenameError(RuntimeError):
    pass


def _is_history_line(line: str) -> bool:
    return _HISTORY_MARK in line.lower()


def _hits_in_line(line: str) -> list[str]:
    """Old tokens in one line that the rewrite would change (protected phrases excluded, case-sensitive tokens)."""
    if _is_history_line(line):
        return []
    out = []
    for m in _COMBINED.finditer(line):
        tok = m.group(0)
        if m.group("prot") is None:
            out.append(tok)
    return out


def _rewrite_counted(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()

    def sub(m: re.Match) -> str:
        tok = m.group(0)
        if m.group("prot") is not None:
            return tok
        counts[tok] += 1
        return _NEW[tok]

    parts = []
    for line in text.splitlines(keepends=True):
        parts.append(line if _is_history_line(line) else _COMBINED.sub(sub, line))
    return "".join(parts), counts


def rewrite_text(text: str) -> str:
    """Map the project's old names to the new ones in a single pass; upstream names and history statements stay."""
    return _rewrite_counted(text)[0]


def map_path(p: str) -> str:
    for old, new in PATH_MOVES:
        if p == old:
            return new
        if p.startswith(old + "/"):
            return new + p[len(old):]
    return p


def allowed_entry(path: str) -> tuple[str, str, str] | None:
    for rx, glob, reason, pinned in _ALLOW_RE:
        if rx.match(path):
            return glob, reason, pinned
    return None


def allowed_reason(path: str) -> str | None:
    e = allowed_entry(path)
    return e[1] if e else None


def ollama_new_name(name: str) -> str | None:
    if not name.startswith("mathocr-"):
        return None
    new = rewrite_text(name)
    return new if new != name else None


def is_binary(data: bytes) -> bool:
    """The one binary-file rule: a NUL byte in the first 8000 bytes."""
    return b"\0" in data[:8000]


def rewrite_blob(path: str, data: bytes) -> bytes:
    """The one per-file rewrite decision: binary and allow-listed files stay byte-identical."""
    if is_binary(data) or allowed_reason(path) is not None:
        return data
    return rewrite_text(data.decode("utf-8", "surrogateescape")).encode("utf-8", "surrogateescape")


def _git(root: Path, *args: str, check: bool = True, input: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, input=input, check=check)


def tracked_files(root: Path) -> list[str]:
    out = _git(root, "ls-files", "-z").stdout.decode("utf-8", "surrogateescape")
    return [p for p in out.split("\0") if p]


def _read_regular(root: Path, rel: str) -> bytes | None:
    p = root / rel
    if p.is_symlink() or not p.is_file():
        return None
    return p.read_bytes()


def scan(root: Path = Path("."), *, include_paths: bool = True) -> list[dict]:
    """Old-name occurrences in tracked files outside the allow-list."""
    root = Path(root)
    found: list[dict] = []
    for rel in tracked_files(root):
        if allowed_reason(rel) is not None:
            continue
        if include_paths:
            for tok in _hits_in_line(rel):
                found.append({"path": rel, "line": 0, "token": tok, "text": rel[:200], "kind": "path"})
        data = _read_regular(root, rel)
        if data is None or is_binary(data):
            continue
        text = data.decode("utf-8", "surrogateescape")
        for n, line in enumerate(text.splitlines(), 1):
            for tok in _hits_in_line(line):
                found.append({"path": rel, "line": n, "token": tok, "text": line[:200], "kind": "content"})
    return found


def _clean_old_dir(root: Path, old: str) -> None:
    """Delete bytecode left under an old directory; any other untracked or ignored file is a refusal."""
    left = _git(root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", old).stdout
    left += _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", old).stdout
    names = sorted({x for x in left.decode("utf-8", "surrogateescape").split("\0") if x})
    other = [n for n in names if "__pycache__" not in n.split("/") and not n.endswith(".pyc")]
    if other:
        raise RenameError(f"untracked or ignored files under {old}: {', '.join(other[:5])}")
    for n in names:
        (root / n).unlink(missing_ok=True)
    base = root / old
    if base.is_dir():
        for d in list(base.rglob("__pycache__")):
            shutil.rmtree(d, ignore_errors=True)


def apply(root: Path = Path("."), *, write: bool) -> dict:
    """Plan (write=False) or perform (write=True) the directory moves and the rewrite. Never commits."""
    root = Path(root)
    tracked = tracked_files(root)
    moves = []
    for old, new in PATH_MOVES:
        n = sum(1 for t in tracked if t == old or t.startswith(old + "/"))
        if n:
            if (root / new).exists():
                raise RenameError(f"destination exists: {new}")
            moves.append({"from": old, "to": new, "files": n})
    files: dict[str, dict] = {}
    skipped: list[str] = []
    total = 0
    for rel in tracked:
        mapped = map_path(rel)
        data = _read_regular(root, rel)
        if data is None or is_binary(data):
            continue
        text = data.decode("utf-8", "surrogateescape")
        _, counts = _rewrite_counted(text)
        if allowed_reason(mapped) is not None or allowed_reason(rel) is not None:
            if counts or _hits_in_line(rel):
                skipped.append(rel)
            continue
        if counts:
            files[mapped] = dict(counts)
            total += sum(counts.values())
    plan = {"moves": moves, "files": files, "skipped_allowed": sorted(skipped), "total_replacements": total}
    if not write:
        return plan
    for m in moves:
        _clean_old_dir(root, m["from"])
        r = _git(root, "mv", m["from"], m["to"], check=False)
        if r.returncode != 0:
            raise RenameError(f"git mv {m['from']} failed: {r.stderr.decode(errors='replace').strip()}")
        left = root / m["from"]
        if left.exists():
            if any(p.is_file() for p in left.rglob("*")):
                raise RenameError(f"files remain under {m['from']} after the move")
            shutil.rmtree(left)
    for rel in tracked_files(root):
        data = _read_regular(root, rel)
        if data is None:
            continue
        new = rewrite_blob(rel, data)
        if new != data:
            (root / rel).write_bytes(new)
    return plan


# ---------------------------------------------------------------------------------------------------------------------
# trial: the rename in throwaway clones

MIN_AVAILABLE_BYTES = 6 * 1024**3
DEFAULT_PYTEST_ARGS = ("-q", "-m", "not docker", "-p", "no:cacheprovider", "-o", "junit_family=xunit1")
_TRIAL_GIT = ("-c", "user.name=trial", "-c", "user.email=trial@example.invalid", "-c", "commit.gpgsign=false")


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _test_id(case: ET.Element) -> str:
    name = case.get("name", "")
    file = case.get("file")
    classname = case.get("classname", "")
    if file:
        cls = [c for c in classname.split(".") if c[:1].isupper()]
        return "::".join([file, *cls, name])
    parts = classname.split(".")
    mod = [x for x in parts if not x[:1].isupper()]
    cls = parts[len(mod):]
    return "::".join(["/".join(mod) + ".py", *cls, name])


def parse_junit(path: Path) -> dict:
    """Counts, failing ids (failures and errors), skipped ids with reasons and the first message line of each failure."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    failing: list[str] = []
    skipped: list[dict] = []
    messages: dict[str, str] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        tid = _test_id(case)
        bad = case.find("failure")
        err = case.find("error")
        skip = case.find("skipped")
        if bad is not None or err is not None:
            node = bad if bad is not None else err
            counts["failed" if bad is not None else "errors"] += 1
            failing.append(tid)
            messages[tid] = ((node.get("message") or node.text or "").strip().splitlines() or [""])[0][:300]
        elif skip is not None:
            counts["skipped"] += 1
            skipped.append({"id": tid, "reason": skip.get("message") or ""})
        else:
            counts["passed"] += 1
    return {"counts": counts, "failing": sorted(failing), "skipped": skipped, "messages": messages}


_EXP001_RE = re.compile(r'^(EXP001_TREE\s*=\s*")([0-9a-f]{40})(")', re.MULTILINE)


def set_exp001_literal(text: str, tree: str) -> str:
    if len(_EXP001_RE.findall(text)) != 1:
        raise RenameError("EXP001_TREE line missing or not unique")
    return _EXP001_RE.sub(lambda m: m.group(1) + tree + m.group(3), text)


def _default_runner(clone: Path, args: list[str], junit: Path) -> int:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = os.pathsep.join(d for d in env.get("PATH", "").split(os.pathsep) if not re.search(r"/\.venv[^/]*/bin/?$", d))
    env["UV_CACHE_DIR"] = subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True, check=True).stdout.strip()
    log = junit.with_suffix("").with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        sync = subprocess.run(["uv", "sync", "--frozen"], cwd=clone, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
        if sync.returncode != 0:
            return sync.returncode
        r = subprocess.run(["uv", "run", "--frozen", "pytest", *args, "--junitxml", str(junit)],
                           cwd=clone, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return r.returncode


def _carry(root: Path, clones: list[Path], paths: tuple[str, ...] | None) -> list[str]:
    """Commit the source worktree's uncommitted changes (limited to `paths` when given) in each clone."""
    spec = ["--", *paths] if paths else []
    diff = _git(root, "diff", "HEAD", "--binary", *spec).stdout
    carried = [x for x in _git(root, "diff", "HEAD", "--name-only", "-z", *spec).stdout.decode().split("\0") if x]
    untracked: list[str] = []
    if paths:
        out = _git(root, "ls-files", "-o", "--exclude-standard", "-z", *spec).stdout.decode()
        untracked = [x for x in out.split("\0") if x]
    for clone in clones:
        if diff:
            r = _git(clone, "apply", "--index", "-", input=diff, check=False)
            if r.returncode != 0:
                raise RenameError(f"cannot apply the worktree diff: {r.stderr.decode(errors='replace').strip()}")
        for rel in untracked:
            dest = clone / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dest)
            _git(clone, "add", "--", rel)
        if _git(clone, "diff", "--cached", "--quiet", check=False).returncode != 0:
            _git(clone, *_TRIAL_GIT, "commit", "-q", "-m", "trial: carried worktree changes")
    return sorted(set(carried) | set(untracked))


def trial(root: Path, work: Path, *, tests: tuple[str, ...] = (), pytest_args: tuple[str, ...] = DEFAULT_PYTEST_ARGS,
          overlays: tuple[str, ...] = (), set_exp001_tree: bool = False, allow_red: tuple[str, ...] = (),
          paths: tuple[str, ...] | None = None, runner=None, mem_check=None) -> dict:
    """Run the suite in a baseline clone and in a renamed clone; report the tests only the rename turns red."""
    root, work = Path(root).resolve(), Path(work).resolve()
    runner = runner or _default_runner
    avail = (mem_check or mem_available)()
    if avail < MIN_AVAILABLE_BYTES:
        raise RenameError(f"MemAvailable {avail / 1024**3:.1f} GiB is below 6 GiB; not starting the trial")
    if _git(root, "check-ignore", "-q", str(work), check=False).returncode != 0:
        raise RenameError(f"work directory is not gitignored in the source repository: {work}")
    if work.exists() and any(work.iterdir()):
        raise RenameError(f"work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    status = _git(root, "status", "--porcelain").stdout
    head = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    clones = {"base": work / "base", "renamed": work / "renamed"}
    for c in clones.values():
        subprocess.run(["git", "clone", "-q", "--no-local", str(root), str(c)], check=True, capture_output=True)
        _git(c, "remote", "remove", "origin")
    carried = _carry(root, list(clones.values()), paths)
    renamed = clones["renamed"]
    apply(renamed, write=True)
    for ov in overlays:
        src, _, dest = ov.partition("=")
        if not src or not dest:
            raise RenameError(f"bad overlay (expected SRC=DEST): {ov}")
        s, d = root / src, renamed / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        _git(renamed, "add", "-A", "--", dest)
    _git(renamed, "add", "-u")
    tree = None
    if set_exp001_tree:
        t = _git(renamed, "write-tree").stdout.decode().strip()
        tree = _git(renamed, "rev-parse", f"{t}:experiments/exp001_baseline").stdout.decode().strip()
        f = renamed / "tests" / "test_frozen_exp001.py"
        f.write_text(set_exp001_literal(f.read_text(encoding="utf-8"), tree), encoding="utf-8")
        _git(renamed, "add", "-u")
    _git(renamed, *_TRIAL_GIT, "commit", "-q", "--allow-empty", "-m", "trial rename")
    results = {}
    for name, c in clones.items():  # one after the other: RAM
        junit = work / f"{name}.junit.xml"
        runner(c, [*pytest_args, *tests], junit)
        if not junit.exists():
            raise RenameError(f"the {name} run wrote no junit report (see {work / (name + '.log')})")
        results[name] = parse_junit(junit)
    base_fail, ren_fail = set(results["base"]["failing"]), set(results["renamed"]["failing"])
    red_all = sorted(ren_fail - base_fail)
    red = [r for r in red_all if not any(fnmatch.fnmatchcase(r, pat) for pat in allow_red)]
    out = {
        "source_head": head,
        "worktree_diff_applied": bool(carried),
        "carried_paths": carried,
        "overlays": list(overlays),
        "exp001_tree": tree,
        "base": results["base"],
        "renamed": results["renamed"],
        "red": red,
        "red_allowed": [r for r in red_all if r not in red],
        "fixed": sorted(base_fail - ren_fail),
        "renamed_scan_residue": len(scan(renamed)),
    }
    if _git(root, "status", "--porcelain").stdout != status or _git(root, "rev-parse", "HEAD").stdout.decode().strip() != head:
        raise RenameError("the source repository changed during the trial")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# CLI


def _summary(found: list[dict]) -> str:
    by_tok = Counter(f["token"] for f in found)
    by_file = Counter(f["path"] for f in found)
    lines = [f"{len(found)} occurrences in {len(by_file)} files"]
    lines += [f"  token {t}: {n}" for t, n in by_tok.most_common()]
    lines += [f"  file {p}: {n}" for p, n in by_file.most_common(15)]
    return "\n".join(lines)


def _write_json(path: str | None, obj) -> None:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rename_map")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", default=".")
    s.add_argument("--json")
    a = sub.add_parser("apply")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a.add_argument("--root", default=".")
    a.add_argument("--json")
    t = sub.add_parser("trial")
    t.add_argument("--root", default=".")
    t.add_argument("--work", required=True)
    t.add_argument("--tests", nargs="+", default=[])
    t.add_argument("--paths", nargs="+")
    t.add_argument("--overlay", nargs="+", default=[], metavar="SRC=DEST")
    t.add_argument("--set-exp001-tree", action="store_true")
    t.add_argument("--allow-red", nargs="+", default=[], metavar="PATTERN")
    t.add_argument("--json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "trial":
            res = trial(Path(args.root), Path(args.work), tests=tuple(args.tests), overlays=tuple(args.overlay),
                        set_exp001_tree=args.set_exp001_tree, allow_red=tuple(args.allow_red),
                        paths=tuple(args.paths) if args.paths else None)
            for name in ("base", "renamed"):
                print(f"{name}: {res[name]['counts']}")
            print(f"carried paths: {len(res['carried_paths'])}; residue in renamed clone: {res['renamed_scan_residue']}")
            print(f"red (turned red by the rename alone): {len(res['red'])}")
            for r in res["red"]:
                print(f"  {r}: {res['renamed']['messages'].get(r, '')}")
            if res["red_allowed"]:
                print(f"red but allowed: {res['red_allowed']}")
            if res["fixed"]:
                print(f"fixed by the rename: {res['fixed']}")
            _write_json(args.json, res)
            return 1 if res["red"] else 0
        if args.cmd == "scan":
            found = scan(Path(args.root))
            print(_summary(found))
            _write_json(args.json, found)
            return 1 if found else 0
        plan = apply(Path(args.root), write=args.write)
        print(f"moves: {[(m['from'], m['to'], m['files']) for m in plan['moves']]}")
        print(f"files to change: {len(plan['files'])}, replacements: {plan['total_replacements']}, "
              f"allow-listed with old names: {len(plan['skipped_allowed'])}")
        _write_json(args.json, plan)
        return 0
    except RenameError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
