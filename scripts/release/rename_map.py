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
import json
import re
import shutil
import subprocess
import sys
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
    args = ap.parse_args(argv)
    try:
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
