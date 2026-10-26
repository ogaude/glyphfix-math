"""Recorded amendment 2 of the frozen acceptance files: the project rename changes names only.

Amendment 1 (scripts/release/freeze_amendment.py) changed two path literals in the frozen CLI. The project rename
changes the package name in the same two frozen files (the CLI scripts/acceptance.py and the module
acceptance.py inside the package). This tool proves, mechanically, that the new text differs from the old text only by
the token map of scripts/release/rename_map.py (byte substitution, line diff, token stream and AST with identifiers
and string constants mapped), builds the amended RUN_LOCK.json and REGISTRY.json, predicts the git tree of the moved
experiments directory, and checks a whole commit as a pure rename-rewrite.

  uv run python -m scripts.release.rename_amendment preview [--out DIR]
  uv run python -m scripts.release.rename_amendment verify    (chain verification)
  uv run python -m scripts.release.rename_amendment record    (writes the amendment record)

`preview` changes nothing tracked: it writes the renamed frozen files, their blobs, the amended lock and registry,
the predicted experiments tree and the rewrite plan under data/runs/rename-amendment/preview.

This file spells the old names as data (derived from freeze_amendment and rename_map where possible) and is on the
rename allow-list. It never imports the project package, so it works before and after the rename.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

from scripts.release import freeze_amendment as fa
from scripts.release import public_paths, rename_map
from scripts.spikes.common import write_json

AmendmentError = fa.AmendmentError

AMENDMENT2_PATH = fa.NEW_EVIDENCE_DIR / "FREEZE_AMENDMENT_2.json"
AMENDMENT2_REF = AMENDMENT2_PATH.as_posix()
FROZEN_CLI = fa.FROZEN_CLI
OLD_MODULE = fa.FROZEN_MODULE
NEW_MODULE = rename_map.map_path(OLD_MODULE)
assert NEW_MODULE == "src/glyphfix_math/acceptance.py", NEW_MODULE

CLI_OLD_BLOB = "611504261674544c9296806ba3b6d51630bcf536"
CLI_GENERATED_BLOB = fa.OLD_CLI_BLOB
MODULE_OLD_BLOB = fa.MODULE_BLOB
OLD_TAG = "mathocr-lora-q8:p4-full-1"
NEW_TAG = rename_map.rewrite_text(OLD_TAG)
OLD_CAND_TAG = "mathocr-cand:p4-full-1"
NEW_CAND_TAG = rename_map.rewrite_text(OLD_CAND_TAG)
EXP001_OLD = "experiments/exp001_lighton"
EXP001_NEW = rename_map.map_path(EXP001_OLD)
EXP001_TEST = "tests/test_frozen_exp001.py"
EXP001_OLD_TREE = "934e94975a80805ad3c14a5a3f2c6b493ff26736"
COMMIT_EXCEPTIONS = (fa.LOCK_PATH.as_posix(), EXP001_TEST)
PREVIOUS_COMMIT = ("8bff9421db6ab8f95e69e527b7775f1356e7b25c", "2026-10-18T18:34:42+02:00")
MODULE_PIN_COMMIT = ("0a3aff0e66504ff2b8d69f40c61cda7221c28c9d", "2026-10-17T09:59:12+02:00")
LOCK_NEW_KEY = "acceptance_module_blob_generated"

# the keys scripts.acceptance._diff reports between the published and a fresh results.json, in its sorted order
RENAME_KEYS = (
    f"code_freeze.commits.{FROZEN_CLI}",
    f"code_freeze.commits.{NEW_MODULE}",
    f"code_freeze.commits.{OLD_MODULE}",
    "code_freeze.current_blobs.cli",
    "code_freeze.current_blobs.module",
    "code_freeze.lock_blobs.cli",
    "code_freeze.lock_blobs.module",
    "tag",
)
assert list(RENAME_KEYS) == sorted(RENAME_KEYS) and len(RENAME_KEYS) == 8

DEFAULT_PREVIEW_DIR = "data/runs/rename-amendment/preview"
PUBLIC_PATHS_REF = "scripts/release/public_paths.txt"
RELEASE_HISTORY_REF = "RELEASE_HISTORY.json"


# ---------------------------------------------------------------------- proof

_NAME_FIELDS = ("id", "attr", "name", "asname", "module", "arg")


def _map_tree(tree: ast.AST) -> None:
    """Map identifiers and string constants of a tree through the rename, in place."""
    for node in ast.walk(tree):
        for f in _NAME_FIELDS:
            v = getattr(node, f, None)
            if isinstance(v, str):
                setattr(node, f, rename_map.rewrite_text(v))
        if isinstance(node, ast.Global | ast.Nonlocal):
            node.names = [rename_map.rewrite_text(n) for n in node.names]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = rename_map.rewrite_text(node.value)


_MAPPED_TOKEN_TYPES = {tokenize.NAME, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}


def prove_rename(old_text: str, new_text: str) -> dict:
    """Prove new_text is old_text with exactly the project's old names mapped, and nothing else."""
    expected = rename_map.rewrite_text(old_text)
    if old_text == new_text:
        raise AmendmentError("new text equals old text: nothing was renamed")
    if new_text != expected:
        raise AmendmentError("byte substitution: new text differs from rewrite_text(old text)")

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    if len(old_lines) != len(new_lines):
        raise AmendmentError("line count differs")
    changed = [i + 1 for i, (a, b) in enumerate(zip(old_lines, new_lines, strict=True)) if a != b]
    should = [i + 1 for i, a in enumerate(old_lines) if rename_map.rewrite_text(a) != a]
    if changed != should:
        raise AmendmentError(f"changed lines {changed} are not the lines the token map changes {should}")

    ot = list(tokenize.generate_tokens(io.StringIO(old_text).readline))
    nt = list(tokenize.generate_tokens(io.StringIO(new_text).readline))
    if len(ot) != len(nt):
        raise AmendmentError("token count differs")
    changed_tokens: dict[str, int] = {}
    substitutions: list[dict] = []
    for a, b in zip(ot, nt, strict=True):
        if a.type != b.type or a.start[0] != b.start[0]:
            raise AmendmentError(f"token type or line differs at line {a.start[0]}")
        if a.string != b.string:
            if a.type not in _MAPPED_TOKEN_TYPES or b.string != rename_map.rewrite_text(a.string):
                raise AmendmentError(f"unexpected token change at line {a.start[0]}: {a.string!r} -> {b.string!r}")
            name = tokenize.tok_name[a.type]
            changed_tokens[name] = changed_tokens.get(name, 0) + 1
    for n in changed:
        substitutions.append({"line": n, "old": old_lines[n - 1].strip(), "new": new_lines[n - 1].strip()})

    otree, ntree = ast.parse(old_text), ast.parse(new_text)
    _map_tree(otree)
    if ast.dump(otree, include_attributes=False) != ast.dump(ntree, include_attributes=False):
        raise AmendmentError("AST differs after mapping identifiers and string constants")

    return {
        "method": "rewrite_text+line-diff+tokenize+ast",
        "changed_lines": changed,
        "changed_tokens": changed_tokens,
        "substitutions": substitutions,
        "ast_equal_after_mapping": True,
        "python": sys.version,
    }


# ------------------------------------------------------------ record edits

def amended_lock_rename(lock: dict, *, cli_new: str, module_new: str) -> dict:
    """RUN_LOCK with both arms re-pinned to the renamed frozen files; the generating blobs are kept next to them."""
    out = copy.deepcopy(lock)
    arms = out.get("arms") or {}
    for arm in fa.ARMS:
        if arm not in arms:
            raise AmendmentError(f"RUN_LOCK has no arm {arm}")
        a = arms[arm]
        if a.get("acceptance_cli_blob") != CLI_OLD_BLOB:
            raise AmendmentError(f"{arm}: acceptance_cli_blob {a.get('acceptance_cli_blob')} is not {CLI_OLD_BLOB}")
        if a.get("acceptance_module_blob") != MODULE_OLD_BLOB:
            raise AmendmentError(f"{arm}: acceptance_module_blob {a.get('acceptance_module_blob')} is not {MODULE_OLD_BLOB}")
        if a.get("acceptance_cli_blob_generated") != CLI_GENERATED_BLOB:
            raise AmendmentError(
                f"{arm}: acceptance_cli_blob_generated {a.get('acceptance_cli_blob_generated')} is not {CLI_GENERATED_BLOB}"
            )
        a["acceptance_cli_blob"] = cli_new
        a["acceptance_module_blob"] = module_new
        a[LOCK_NEW_KEY] = MODULE_OLD_BLOB
        a["freeze_amendment"] = AMENDMENT2_REF
    return out


def _leaf_changes(a, b, path: tuple = ()) -> list[tuple[tuple, object, object]]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + (k,), a.get(k), b.get(k)))
            else:
                out += _leaf_changes(a[k], b[k], path + (k,))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _leaf_changes(x, y, path + (i,))
        return out
    return [] if a == b else [(path, a, b)]


def registry_changes(old_text: str, new_text: str) -> list[dict]:
    """Changed leaves of the registry as [{key, old, new}], entries addressed by their tag."""
    old, new = json.loads(old_text), json.loads(new_text)
    out = []
    for path, a, b in _leaf_changes(old, new):
        key = ""
        node = old
        for p in path:
            if isinstance(p, int):
                tag = node[p].get("tag") if isinstance(node[p], dict) else None
                key += f"[tag={tag}]" if tag is not None else f"[{p}]"
            else:
                key += ("." if key else "") + str(p)
            node = node[p] if not isinstance(node, dict) or p in node else None
        out.append({"key": key, "old": a, "new": b})
    return out


def renamed_registry_text(text: str) -> str:
    """The registry text after the rename; refuses unless exactly the tag and candidate_tag leaves change."""
    new = rename_map.rewrite_text(text)
    changes = _leaf_changes(json.loads(text), json.loads(new))
    got = sorted((str(p[-1]), a, b) for p, a, b in changes)
    want = sorted([("candidate_tag", OLD_CAND_TAG, NEW_CAND_TAG), ("tag", OLD_TAG, NEW_TAG)])
    if got != want:
        raise AmendmentError(f"registry changes are not exactly the tag and candidate_tag leaves: {got}")
    return new


# ------------------------------------------------------------------ git helpers

def _run(root: Path, *args: str, input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, input=input, check=False)
    if check and r.returncode != 0:
        raise AmendmentError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r


def _cat_blob(root: Path, sha: str) -> bytes:
    return _run(root, "cat-file", "blob", sha).stdout


def _ls_tree(root: Path, tree: str, recursive: bool = False) -> dict[str, tuple[str, str, str]]:
    """path -> (mode, type, sha); with recursive, blobs of the whole tree (no tree entries)."""
    args = ["ls-tree", "-z"] + (["-r"] if recursive else []) + [tree]
    out = _run(root, *args).stdout
    res: dict[str, tuple[str, str, str]] = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, name = rec.partition(b"\t")
        mode, typ, sha = meta.decode().split()
        res[name.decode("utf-8", "surrogateescape")] = (mode, typ, sha)
    return res


def predict_tree(root: Path, tree: str, *, base: str = EXP001_OLD, report: list | None = None) -> str:
    """Tree id the directory `tree` would have after the rename; objects go to the object store only.

    Every regular blob goes through rename_map.rewrite_blob at its post-move path, the same decision `apply` makes.
    `report` receives {path, old_blob, new_blob} for each blob that changes.
    """
    root = Path(root)

    def walk(t: str, prefix: str) -> str:
        entries = []
        for name, (mode, typ, sha) in sorted(_ls_tree(root, t).items(), key=lambda kv: kv[0].encode("utf-8", "surrogateescape")):
            rel = f"{prefix}/{name}" if prefix else name
            if typ == "tree":
                new_sha = walk(sha, rel)
            elif typ == "blob" and mode in ("100644", "100755"):
                data = _cat_blob(root, sha)
                new = rename_map.rewrite_blob(rename_map.map_path(f"{base}/{rel}"), data)
                if new == data:
                    new_sha = sha
                else:
                    new_sha = _run(root, "hash-object", "-w", "--stdin", input=new).stdout.decode().strip()
                    if report is not None:
                        report.append({"path": rel, "old_blob": sha, "new_blob": new_sha})
            else:
                new_sha = sha
            entries.append(f"{mode} {typ} {new_sha}\t{name}\0".encode("utf-8", "surrogateescape"))
        return _run(root, "mktree", "-z", input=b"".join(entries)).stdout.decode().strip()

    return walk(tree, "")


# ------------------------------------------------------- whole-commit check

def _load_spec(root: Path, spec, head: str):
    if spec is not None:
        return public_paths.load_spec(Path(spec)) if isinstance(spec, str | Path) else spec
    r = _run(root, "show", f"{head}:scripts/release/public_paths.txt", check=False)
    if r.returncode != 0:
        return None
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "public_paths.txt"
        p.write_bytes(r.stdout)
        return public_paths.load_spec(p)


def _name_status(root: Path, commit: str) -> dict[str, str]:
    out = _run(root, "diff", "--name-status", "--no-renames", "-z", f"{commit}~1", commit).stdout.split(b"\0")
    res: dict[str, str] = {}
    i = 0
    while i + 1 < len(out):
        status, path = out[i].decode(), out[i + 1].decode("utf-8", "surrogateescape")
        res[path] = status[0]
        i += 2
    return res


def _expected_blob(old: bytes) -> bytes:
    if rename_map.is_binary(old):
        return old
    return rename_map.rewrite_text(old.decode("utf-8", "surrogateescape")).encode("utf-8", "surrogateescape")


def check_commit_rewrite(
    root: Path,
    commit: str,
    exceptions=COMMIT_EXCEPTIONS,
    *,
    filtered: bool | None = None,
    spec=None,
    head: str = "HEAD",
    proven_latest_only: list | None = None,
) -> dict:
    """Check that `commit` is a pure rename-rewrite of its parent.

    A deleted path pairs with an added path only when map_path(old) == new (git similarity is never used). Each
    pair and each in-place modification outside `exceptions` must equal rewrite_text of the parent blob (binary blobs
    byte-identical, same mode). In a filtered public clone an unpaired latest-only addition is accepted only when it
    equals the blob at `head`, holds no old name and is listed in `proven_latest_only` (the development repository's
    commit_proof.latest_only_rewritten).
    """
    root = Path(root)
    exceptions = tuple(exceptions)
    full = _run(root, "rev-parse", commit).stdout.decode().strip()
    status = _name_status(root, full)
    parent_blobs = _ls_tree(root, f"{full}~1", recursive=True)
    new_blobs = _ls_tree(root, full, recursive=True)

    if filtered is None:
        filtered = _run(root, "cat-file", "-e", f"{head}:RELEASE_HISTORY.json", check=False).returncode == 0
    rules = _load_spec(root, spec, head)
    problems: list[str] = []
    if filtered and rules is None:
        problems.append("public_paths.txt is missing at head: cannot apply the filtered-history rule")

    deleted = sorted(p for p, s in status.items() if s == "D")
    added = {p for p, s in status.items() if s == "A"}
    modified = sorted(p for p, s in status.items() if s not in ("A", "D"))

    pairs: list[tuple[str, str]] = []
    unpaired_deleted: list[str] = []
    for d in deleted:
        m = rename_map.map_path(d)
        if m != d and m in added:
            pairs.append((d, m))
            added.discard(m)
        else:
            unpaired_deleted.append(d)
    for d in unpaired_deleted:
        problems.append(f"deletion without a paired addition: {d}")

    latest_only: list[dict] = []

    def check_rewrite(old_path: str, new_path: str) -> None:
        if new_path in exceptions:
            return
        om, nm = parent_blobs.get(old_path), new_blobs.get(new_path)
        if om is None or nm is None:
            problems.append(f"{new_path}: missing on one side")
            return
        if om[0] != nm[0]:
            problems.append(f"{new_path}: file mode {om[0]} -> {nm[0]}")
            return
        if _expected_blob(_cat_blob(root, om[2])) != _cat_blob(root, nm[2]):
            problems.append(f"{new_path}: content is not rewrite_text of {old_path}")
            return
        if rules is not None and public_paths.classify(new_path, rules) == "latest-only":
            latest_only.append({"path": new_path, "new_blob": nm[2]})

    for old_path, new_path in pairs:
        check_rewrite(old_path, new_path)
    for m in modified:
        if rename_map.map_path(m) != m:
            problems.append(f"{m}: modified inside a moved directory")
            continue
        check_rewrite(m, m)

    accepted: list[dict] = []
    proven = {(e.get("path"), e.get("new_blob")) for e in (proven_latest_only or [])}
    for a in sorted(added):
        if a in exceptions:
            continue
        if not filtered:
            problems.append(f"addition without a paired deletion: {a}")
            continue
        if rules is None or public_paths.classify(a, rules) != "latest-only":
            problems.append(f"addition without a paired deletion (not latest-only): {a}")
            continue
        blob = new_blobs[a][2]
        h = _run(root, "rev-parse", f"{head}:{a}", check=False)
        if h.returncode != 0 or h.stdout.decode().strip() != blob:
            problems.append(f"latest-only addition {a} differs from the blob at head")
            continue
        data = _cat_blob(root, blob)
        if rename_map.is_binary(data) or rename_map.rewrite_text(data.decode("utf-8", "surrogateescape")) != data.decode(
            "utf-8", "surrogateescape"
        ):
            problems.append(f"latest-only addition {a} still holds an old name")
            continue
        if (a, blob) not in proven:
            problems.append(f"latest-only addition not proven in the development repository record: {a}")
            continue
        accepted.append({"path": a, "blob": blob})

    if problems:
        pure: bool | str = False
    elif accepted:
        pure = "partial"
    else:
        pure = True
    return {
        "commit": full,
        "changed": len(status),
        "pure_rewrite": pure,
        "exceptions": list(exceptions),
        "renames": sorted([o, n] for o, n in pairs),
        "latest_only_rewritten": sorted(latest_only, key=lambda e: e["path"]),
        "accepted_latest_only": accepted,
        "latest_only_unproven_in_clone": [e["path"] for e in accepted],
        "filtered": bool(filtered),
        "problems": problems,
    }


# ----------------------------------------------------------------- verification

REGISTRY_DIGEST = "c06474f6789e896e8acbddf0cb91dad88b0a29c563ee58d76bd31d846351e737"
LOCK_RENAME_KEYS = frozenset({"acceptance_cli_blob", "acceptance_module_blob", LOCK_NEW_KEY, "freeze_amendment"})
RECORD_COMMIT_PROOF_EXCEPTIONS = list(COMMIT_EXCEPTIONS)


def _reachable(root: Path, sha: object) -> bool:
    return fa._object_exists(root, sha)


def _blob_at(root: Path, rel: str) -> str | None:
    p = Path(root) / rel
    return fa._blob_of(p, Path(root)) if p.is_file() else None


def _cat_text(root: Path, sha: str) -> str:
    return fa._git("cat-file", "-p", sha, root=root)


def _commit_ok(found: str, recorded: object, cmap: dict) -> bool:
    """`found` is "<hash> <iso date>"; it must be the recorded commit (or its mapping) with the recorded date."""
    if not (isinstance(recorded, list | tuple) and len(recorded) == 2):
        return False
    try:
        h, d = found.split()
    except ValueError:
        return False
    return d == recorded[1] and h in (recorded[0], cmap.get(recorded[0]))


_KEY_SEG = re.compile(r"\[tag=([^\]]*)\]|\[(\d+)\]|([^.\[\]]+)")


def _resolve_record_key(doc, key: str):
    """Value at a registry change key such as `entries[tag=T].candidate_tag`; (found, value)."""
    node = doc
    for m in _KEY_SEG.finditer(key):
        tag, idx, name = m.groups()
        if tag is not None:
            if not isinstance(node, list):
                return False, None
            hit = [e for e in node if isinstance(e, dict) and e.get("tag") in (tag, rename_map.rewrite_text(tag))]
            if len(hit) != 1:
                return False, None
            node = hit[0]
        elif idx is not None:
            if not isinstance(node, list) or int(idx) >= len(node):
                return False, None
            node = node[int(idx)]
        else:
            if not isinstance(node, dict) or name not in node:
                return False, None
            node = node[name]
    return True, node


def _check_shape(a2: dict, a1: dict, problems: list[str]) -> None:
    if a2.get("kind") != "rename-only":
        problems.append("rename record: kind is not rename-only")
    if a2.get("previous") != fa.AMENDMENT_PATH.as_posix():
        problems.append("rename record: previous is not amendment 1")
    if a2.get("token_map") != [list(x) for x in rename_map.TOKEN_MAP]:
        problems.append("rename record: token_map differs from the token map in rename_map")
    if a2.get("path_moves") != [list(x) for x in rename_map.PATH_MOVES]:
        problems.append("rename record: path_moves differs from the moves in rename_map")


def _check_frozen_files(root: Path, a2: dict, a1: dict, problems: list[str]) -> None:
    files = a2.get("files") or {}
    cli, mod = files.get("cli") or {}, files.get("module") or {}
    try:
        cur = _blob_at(root, FROZEN_CLI)
        if cur != cli.get("new_blob"):
            problems.append(f"cli: current {FROZEN_CLI} blob is not the recorded new_blob")
        if cli.get("old_blob") != a1.get("new_blob") or not _reachable(root, cli.get("old_blob")):
            problems.append("cli: recorded old_blob is not amendment 1's new_blob or is not reachable")
        else:
            prove_rename(_cat_text(root, cli["old_blob"]), (Path(root) / FROZEN_CLI).read_text(encoding="utf-8"))
    except (AmendmentError, OSError) as e:
        problems.append(f"cli proof: {e}")
    try:
        if (Path(root) / OLD_MODULE).exists():
            problems.append(f"module: {OLD_MODULE} still exists")
        cur = _blob_at(root, NEW_MODULE)
        if cur is None:
            problems.append(f"module: {NEW_MODULE} is missing")
        elif cur != mod.get("new_blob"):
            problems.append(f"module: current {NEW_MODULE} blob is not the recorded new_blob")
        if mod.get("old_blob") != fa.MODULE_BLOB or not _reachable(root, mod.get("old_blob")):
            problems.append("module: recorded old_blob is not the frozen module blob or is not reachable")
        elif cur is not None:
            prove_rename(_cat_text(root, mod["old_blob"]), (Path(root) / NEW_MODULE).read_text(encoding="utf-8"))
    except (AmendmentError, OSError) as e:
        problems.append(f"module proof: {e}")


def _check_run_lock(root: Path, a2: dict, a1: dict, problems: list[str]) -> None:
    files = a2.get("files") or {}
    rl = a2.get("run_lock") or {}
    p = Path(root) / fa.LOCK_PATH
    try:
        arms = json.loads(p.read_text(encoding="utf-8")).get("arms", {})
    except (OSError, ValueError) as e:
        problems.append(f"RUN_LOCK unreadable: {e}")
        return
    want = {
        "acceptance_cli_blob": (files.get("cli") or {}).get("new_blob"),
        "acceptance_module_blob": (files.get("module") or {}).get("new_blob"),
        "acceptance_cli_blob_generated": a1.get("old_blob"),
        LOCK_NEW_KEY: fa.MODULE_BLOB,
        "freeze_amendment": AMENDMENT2_REF,
    }
    for arm in fa.ARMS:
        a = arms.get(arm) or {}
        for k, v in want.items():
            if a.get(k) != v:
                problems.append(f"RUN_LOCK {arm}: {k} is {a.get(k)!r}, expected {v!r}")
    extra = {c.get("key") for c in rl.get("changes") or []} - LOCK_RENAME_KEYS
    if extra:
        problems.append(f"rename record: RUN_LOCK changes touch other keys: {sorted(extra)}")
    if fa._sha256(p) != rl.get("sha256_after"):
        problems.append("RUN_LOCK.json sha256 is not the recorded sha256_after")


def _check_registry(root: Path, a2: dict, problems: list[str]) -> None:
    reg = a2.get("registry") or {}
    try:
        doc = json.loads((Path(root) / fa.REGISTRY_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        problems.append(f"REGISTRY.json unreadable: {e}")
        return
    entries = doc.get("entries") or []
    new = [e for e in entries if isinstance(e, dict) and e.get("tag") == NEW_TAG]
    if not new or any(e.get("ollama_digest") != REGISTRY_DIGEST for e in new):
        problems.append(f"registry: no entry with tag {NEW_TAG} and the recorded ollama digest")
    if any(isinstance(e, dict) and e.get("tag") == OLD_TAG for e in entries):
        problems.append(f"registry: an entry still has tag {OLD_TAG}")
    changes = reg.get("changes") or []
    leaves = sorted((str(c.get("key", "")).rsplit(".", 1)[-1], c.get("old"), c.get("new")) for c in changes)
    want = sorted([("candidate_tag", OLD_CAND_TAG, NEW_CAND_TAG), ("tag", OLD_TAG, NEW_TAG)])
    if leaves != want:
        problems.append("registry: the recorded changes are not exactly the tag and candidate_tag leaves")
        return
    for c in changes:
        found, value = _resolve_record_key(doc, c["key"])
        if not found or value != c["new"]:
            problems.append(f"registry: {c['key']} does not hold the recorded new value at HEAD")


def _check_commits(root: Path, a2: dict, published: dict, cmap: dict, problems: list[str]) -> str | None:
    """Commit order and dates; returns the (mapped) amendment commit for the rewrite check."""
    ac = a2.get("amendment_commit")
    mapped = None
    if isinstance(ac, list | tuple) and len(ac) == 2:
        mapped = cmap.get(ac[0]) or ac[0]
    try:
        lines = fa._git("log", "--format=%H %cI", "--", NEW_MODULE, root=root).splitlines()
        if len(lines) != 1 or not _commit_ok(lines[0], ac, cmap):
            problems.append(f"commits: {NEW_MODULE} is not touched by exactly the recorded amendment commit")
        last = fa._git("log", "-1", "--format=%H %cI", "--", FROZEN_CLI, root=root).strip()
        if not _commit_ok(last, ac, cmap):
            problems.append(f"commits: the last commit touching {FROZEN_CLI} is not the recorded amendment commit")
    except AmendmentError as e:
        problems.append(f"commits: {e}")
    pin = a2.get("module_pin_commit")
    pub = ((published.get("code_freeze") or {}).get("commits") or {}).get(OLD_MODULE)
    if not (isinstance(pin, list | tuple) and len(pin) == 2 and list(pin) == list(pub or [])):
        problems.append("commits: module_pin_commit is not the published code_freeze commit of the module")
    else:
        pm = cmap.get(pin[0]) or pin[0]
        try:
            if fa._git("rev-parse", f"{pm}:{OLD_MODULE}", root=root).strip() != fa.MODULE_BLOB:
                problems.append("commits: the module at the pin commit is not the frozen module blob")
            if fa._git("log", "-1", "--format=%cI", pm, root=root).strip() != pin[1]:
                problems.append("commits: the pin commit date differs from the recorded one")
        except AmendmentError as e:
            problems.append(f"commits: module pin commit not found: {e}")
    return mapped


def _check_rewrite(root: Path, a2: dict, mapped: str | None, info: dict, problems: list[str]) -> None:
    proof = a2.get("commit_proof") or {}
    if mapped is None:
        problems.append("commit rewrite: no amendment commit recorded")
        return
    if proof.get("exceptions") != RECORD_COMMIT_PROOF_EXCEPTIONS:
        problems.append("commit rewrite: recorded exceptions differ from the expected ones")
    ac = a2.get("amendment_commit")
    if proof.get("commit") != (ac[0] if ac else None):
        problems.append("commit rewrite: commit_proof.commit is not the amendment commit")
    recorded = proof.get("latest_only_rewritten")
    if not isinstance(recorded, list):
        problems.append("commit rewrite: commit_proof.latest_only_rewritten is missing")
        return
    try:
        res = check_commit_rewrite(root, mapped, proven_latest_only=recorded)
    except AmendmentError as e:
        problems.append(f"commit rewrite: {e}")
        return
    info["commit_rewrite"] = res
    problems += [f"commit rewrite: {p}" for p in res["problems"]]
    if res["filtered"]:
        if res["pure_rewrite"] not in (True, "partial"):
            problems.append("commit rewrite: the rename commit is not a pure rewrite in this clone")
    else:
        if res["pure_rewrite"] is not True:
            problems.append("commit rewrite: the rename commit is not a pure rewrite")
        if res["latest_only_rewritten"] != recorded:
            problems.append("commit rewrite: recorded latest_only_rewritten differs from the recomputed list")


def _check_exp001(root: Path, a2: dict, problems: list[str]) -> None:
    new_dir, test = Path(root) / EXP001_NEW, Path(root) / EXP001_TEST
    if not (new_dir.exists() and test.is_file()):
        return
    want = (a2.get("exp001_tree") or {}).get("new_tree")
    try:
        got = fa._git("rev-parse", f"HEAD:{EXP001_NEW}", root=root).strip()
    except AmendmentError as e:
        problems.append(f"exp001: {e}")
        return
    if got != want:
        problems.append("exp001: tree at HEAD is not the recorded new_tree")
    m = re.search(r'^EXP001_TREE\s*=\s*"([0-9a-f]+)"', test.read_text(encoding="utf-8"), re.MULTILINE)
    if not m or m.group(1) != want:
        problems.append(f"exp001: the tree literal in {EXP001_TEST} is not the recorded new_tree")


def _check_recorded_data(root: Path, a2: dict, problems: list[str]) -> None:
    kept = a2.get("kept_unpinned") or []
    rewritten = a2.get("rewritten_unpinned") or []
    both = {e.get("path") for e in kept} & {e.get("path") for e in rewritten}
    if both:
        problems.append(f"recorded data: in both kept_unpinned and rewritten_unpinned: {sorted(both)}")
    for e in list(a2.get("recorded_data_kept") or []) + list(kept):
        p = Path(root) / e["path"]
        if p.is_file() and fa._sha256(p) != e.get("sha256"):
            problems.append(f"recorded data: {e['path']} differs from its recorded sha256")
    for e in rewritten:
        p = Path(root) / e["path"]
        if not p.is_file():
            continue
        if fa._sha256(p) != e.get("sha256_after"):
            problems.append(f"recorded data: {e['path']} differs from its recorded sha256_after")
        text = p.read_bytes().decode("utf-8", "surrogateescape")
        if rename_map.rewrite_text(text) != text:
            problems.append(f"recorded data: {e['path']} still holds an old name")


def _check_rerun(a2: dict, problems: list[str]) -> None:
    r = a2.get("rerun_equivalence") or {}
    if r.get("compare") != "identical_modulo_amendment" or r.get("differing_keys") != list(RENAME_KEYS):
        problems.append("rerun: not identical_modulo_amendment with exactly the eight rename keys")
    if r.get("report_rerender_identical") is not True:
        problems.append("rerun: the report re-render is not recorded as identical")
    if r.get("score_exit") != 0 or r.get("cold_cache") is not True:
        problems.append("rerun: score_exit is not 0 or the cache was not cold")


def verify_rename(
    root: Path,
    a2: dict,
    *,
    a1: dict,
    published: dict,
    commit_map: dict | None = None,
    info: dict | None = None,
) -> list[str]:
    """Every problem found when checking the rename record against the tree and git objects (empty when sound)."""
    root = Path(root)
    cmap = commit_map or {}
    info = info if info is not None else {}
    problems: list[str] = []
    _check_shape(a2, a1, problems)
    _check_frozen_files(root, a2, a1, problems)
    _check_run_lock(root, a2, a1, problems)
    _check_registry(root, a2, problems)
    mapped = _check_commits(root, a2, published, cmap, problems)
    _check_rewrite(root, a2, mapped, info, problems)
    _check_exp001(root, a2, problems)
    _check_recorded_data(root, a2, problems)
    _check_rerun(a2, problems)
    return problems


# -------------------------------------------------------------------- preview

def _blob_of_bytes(data: bytes, path: Path) -> str:
    path.write_bytes(data)
    return fa._blob_of(path)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AmendmentError(msg)


def cmd_preview(args: argparse.Namespace) -> int:
    out = Path(args.out)
    _require(fa._blob_of(Path(FROZEN_CLI)) == CLI_OLD_BLOB, f"{FROZEN_CLI} is not at the amendment-1 blob {CLI_OLD_BLOB}")
    _require(Path(OLD_MODULE).is_file() and fa._blob_of(Path(OLD_MODULE)) == MODULE_OLD_BLOB,
             f"{OLD_MODULE} is missing or not at blob {MODULE_OLD_BLOB}")
    _require(not Path(NEW_MODULE).exists(), f"{NEW_MODULE} already exists: the rename is applied")
    _require(fa._git("rev-parse", f"HEAD:{EXP001_OLD}").strip() == EXP001_OLD_TREE,
             f"{EXP001_OLD} is not at tree {EXP001_OLD_TREE}")
    before = fa._git("status", "--porcelain", "--untracked-files=no")

    out.mkdir(parents=True, exist_ok=True)
    cli_old = Path(FROZEN_CLI).read_text(encoding="utf-8")
    mod_old = Path(OLD_MODULE).read_text(encoding="utf-8")
    cli_new, mod_new = rename_map.rewrite_text(cli_old), rename_map.rewrite_text(mod_old)
    cli_path, mod_path = out / "acceptance.py", out / "acceptance_module.py"
    cli_path.write_text(cli_new, encoding="utf-8")
    mod_path.write_text(mod_new, encoding="utf-8")
    proof_cli, proof_mod = prove_rename(cli_old, cli_new), prove_rename(mod_old, mod_new)
    for name, p in (("proof_cli.json", proof_cli), ("proof_module.json", proof_mod)):
        (out / name).write_text(json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blobs = {
        "cli_old": CLI_OLD_BLOB,
        "cli_new": fa._blob_of(cli_path),
        "module_old": MODULE_OLD_BLOB,
        "module_new": fa._blob_of(mod_path),
    }
    (out / "blobs.json").write_text(json.dumps(blobs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lock = json.loads(fa.LOCK_PATH.read_text(encoding="utf-8"))
    write_json(out / "RUN_LOCK.json", amended_lock_rename(lock, cli_new=blobs["cli_new"], module_new=blobs["module_new"]))
    reg_text = fa.REGISTRY_PATH.read_text(encoding="utf-8")
    (out / "REGISTRY.json").write_text(renamed_registry_text(reg_text), encoding="utf-8")

    changed_files: list[dict] = []
    new_tree = predict_tree(Path("."), EXP001_OLD_TREE, report=changed_files)
    (out / "exp001.json").write_text(
        json.dumps({"old_tree": EXP001_OLD_TREE, "predicted_new_tree": new_tree, "files": changed_files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = rename_map.apply(Path("."), write=False)
    (out / "rewrite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fa._unified_diff(Path(FROZEN_CLI), cli_path, out / "acceptance.py.diff")
    fa._unified_diff(Path(OLD_MODULE), mod_path, out / "acceptance_module.py.diff")
    fa._unified_diff(fa.LOCK_PATH, out / "RUN_LOCK.json", out / "RUN_LOCK.json.diff")
    fa._unified_diff(fa.REGISTRY_PATH, out / "REGISTRY.json", out / "REGISTRY.json.diff")

    (out / "SUMMARY.txt").write_text(_summary(blobs, proof_cli, proof_mod, new_tree, plan, changed_files), encoding="utf-8")
    after = fa._git("status", "--porcelain", "--untracked-files=no")
    _require(before == after, "preview changed a tracked file")
    print(f"preview written to {out}; cli {blobs['cli_new']}, module {blobs['module_new']}, experiments tree {new_tree}")
    return 0


def _summary(blobs: dict, proof_cli: dict, proof_mod: dict, new_tree: str, plan: dict, exp_files: list[dict]) -> str:
    L = ["RENAME AMENDMENT PREVIEW (nothing tracked has changed)", "", "Moves (git mv, content rewritten by the token map):"]
    L += [f"  {m['from']} -> {m['to']} ({m['files']} files)" for m in plan["moves"]]
    L += [
        "",
        "Frozen files (names only, proven by byte substitution, line diff, token stream and AST):",
        f"  {FROZEN_CLI}: blob {blobs['cli_old']} -> {blobs['cli_new']}, lines {proof_cli['changed_lines']}",
        f"  {OLD_MODULE} -> {NEW_MODULE}: blob {blobs['module_old']} -> {blobs['module_new']}, lines {proof_mod['changed_lines']}",
        "",
        (f"{fa.LOCK_PATH}: in both arms acceptance_cli_blob and acceptance_module_blob take the new blobs, "
        f"{LOCK_NEW_KEY} (the generating module blob) and the freeze_amendment pointer ({AMENDMENT2_REF}) are added; "
        "acceptance_cli_blob_generated stays. The model field keeps the tag that generated the predictions."),
        (f"{fa.REGISTRY_PATH}: tag {OLD_TAG} -> {NEW_TAG}; candidate_tag {OLD_CAND_TAG} -> {NEW_CAND_TAG}. "
        "Its sha256 is recorded in amendment 1 (registry.sha256_after), so that field stops equalling the file; "
        "amendment 2 records the new hash."),
        "",
        (f"{EXP001_OLD}: tree {EXP001_OLD_TREE} -> predicted {new_tree} at {EXP001_NEW} "
        f"({len(exp_files)} files change). {EXP001_TEST} updates its tree constant."),
        "",
        (f"Rewrite totals: {len(plan['files'])} files, {plan['total_replacements']} replacements "
        "(planning prose and planning evidence are rewritten in commit C)."),
    ]
    per_token: dict[str, int] = {}
    for counts in plan["files"].values():
        for tok, n in counts.items():
            per_token[tok] = per_token.get(tok, 0) + n
    L += [f"  {tok}: {n}" for tok, n in sorted(per_token.items(), key=lambda kv: -kv[1])]
    kept = plan["skipped_allowed"]
    phase = [p for p in kept if p.startswith(rename_map.PHASE_DIR)]
    others = [p for p in kept if not p.startswith(rename_map.PHASE_DIR)]
    L += ["", f"Allow-listed files kept byte-identical with old names ({len(kept)}; {len(phase)} are this phase's own documents):"]
    unpinned = []
    for p in others:
        e = rename_map.allowed_entry(p)
        reason, pinned = (e[1], e[2]) if e else ("", "")
        L.append(f"  {p}: {reason} (pinned by: {pinned})")
        if pinned.startswith("none"):
            unpinned.append(p)
    L += ["", "Kept with old names without a hash pin (needs your acceptance):"]
    L += [f"  {p}" for p in unpinned] or ["  (none)"]
    L += [
        "Also kept: the model field of RUN_LOCK.json, and history lines (\"renamed from\" statements and the protected phase title).",
        "",
        f"Ollama re-tag to perform: {OLD_TAG} -> {NEW_TAG} (same digest).",
        "",
        ("The frozen scorer refuses benchmark scoring when the module's last commit is later than the generation start "
        "(scripts/acceptance.py lines 612-614). Moving the module makes the rename commit that last commit, so the frozen "
        "logic cannot score after the rename. The frozen files change in names only. The rescore and the public reproduce "
        "command run the frozen scorer through scripts.release.frozen_rescore, which supplies the module's pre-rename pin "
        f"commit {MODULE_PIN_COMMIT[0][:7]} ({MODULE_PIN_COMMIT[1]}) after proving from git objects that the current module "
        f"is blob {MODULE_OLD_BLOB[:8]} renamed. Calling scripts.acceptance score --mode benchmark directly will refuse; "
        "that is recorded as an effect of the amendment."),
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- record preconditions

RECORD_REASON = (
    "rename the project, package, experiments directory and our Ollama tags to the published names "
    "before the first release (user decision 2026-10-26)"
)
RECORD_EFFECTS = [
    (
        "scripts.acceptance score --mode benchmark refuses when called directly: the module's last commit is the rename "
        "commit, later than the generation start; the published-claim rescore runs through scripts.release.frozen_rescore"
    ),
    "scripts.acceptance generate refuses for every arm, as after amendment 1",
    "results.json, ACCEPTANCE.md and the recorded predictions are unchanged; they describe the run as scored under the old names",
    "historical commits keep the old names; amendment 1 is verified from git objects",
]
RECORD_HARNESS = "scripts.release.frozen_rescore"
PUBLISHED_RESULTS = fa.NEW_EVIDENCE_DIR / "results.json"
PUBLISHED_ACCEPTANCE_MD = fa.NEW_EVIDENCE_DIR / "ACCEPTANCE.md"
RECORDED_DATA_DIR = "data/benchmark/acceptance/"
RECORDED_DATA_FILES = ("evidence/acceptance/results.json", "evidence/acceptance/env_check.json")
COMMIT_PROOF_KEYS = ("commit", "changed", "pure_rewrite", "exceptions", "renames", "latest_only_rewritten")


@contextlib.contextmanager
def _guard(problems: list[str], label: str):
    try:
        yield
    except (AmendmentError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        problems.append(f"{label}: {e}")


def _run_compare(root: Path, published: Path, fresh: Path) -> str:
    """Output of the frozen scorer's own results comparison (`scripts.acceptance compare-results`)."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.acceptance", "compare-results", str(published), str(fresh)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    return r.stdout.strip()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tracked(root: Path, ref: str = "HEAD") -> list[str]:
    out = _run(root, "ls-tree", "-r", "--name-only", "-z", ref).stdout
    return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]


def _outside_phase(path: str) -> bool:
    return not path.startswith(rename_map.PHASE_DIR + "/")


def _unpinned_rewrites(root: Path, commit: str, res: dict, problems: list[str]) -> list[dict]:
    """Unpinned allow-listed files `commit` changed; each must equal rewrite_text of its parent version."""
    old_of = {new: old for old, new in res.get("renames", [])}
    out: list[dict] = []
    for path, st in sorted(_name_status(root, commit).items()):
        if st == "D":
            continue
        entry = rename_map.allowed_entry(path)
        if not (entry and entry[2].startswith("none")):
            continue
        old = _run(root, "show", f"{commit}~1:{old_of.get(path, path)}", check=False)
        if old.returncode != 0:
            continue
        new = _cat_blob_at(root, commit, path)
        if _expected_blob(old.stdout) != new:
            problems.append(f"rewritten unpinned file is not rewrite_text of its parent: {path}")
        elif _outside_phase(path):
            out.append({"path": path, "sha256_before": _sha256_bytes(old.stdout), "sha256_after": _sha256_bytes(new)})
    return out


def _cat_blob_at(root: Path, ref: str, path: str) -> bytes:
    return _run(root, "show", f"{ref}:{path}").stdout


def _kept_data(root: Path, rewritten: list[dict]) -> tuple[list[dict], list[dict]]:
    """(recorded_data_kept, kept_unpinned) of the tracked allow-listed files that keep their old names."""
    gone = {e["path"] for e in rewritten}
    skip = {fa.LOCK_PATH.as_posix(), AMENDMENT2_REF}
    kept, unpinned = [], []
    for path in _tracked(root):
        entry = rename_map.allowed_entry(path)
        p = Path(root) / path
        if entry is None or path in skip or path in gone or not _outside_phase(path) or not p.is_file():
            continue
        if entry[2].startswith("none"):
            data = p.read_bytes()
            if not rename_map.is_binary(data):
                text = data.decode("utf-8", "surrogateescape")
                if rename_map.rewrite_text(text) != text:
                    unpinned.append({"path": path, "sha256": _sha256_bytes(data), "reason": entry[1]})
        elif path.startswith(RECORDED_DATA_DIR) or path in RECORDED_DATA_FILES:
            kept.append({"path": path, "sha256": fa._sha256(p)})
    return kept, unpinned


def check_record_inputs(
    root: Path,
    rerun_dir: Path,
    *,
    preview_dir: Path,
    amendment_commit: str,
    approved: str,
    compare_runner=None,
) -> tuple[list[str], dict]:
    """Every reason the amendment record must not be written yet, and the values the record is built from.

    Collects all problems instead of stopping at the first. Reads files and git objects; writes nothing.
    """
    root, rerun, preview = Path(root), Path(rerun_dir), Path(preview_dir)
    if compare_runner is None:
        def compare_runner(published: Path, fresh: Path) -> str:
            return _run_compare(root, published, fresh)
    problems: list[str] = []
    draft: dict = {}

    if not str(approved or "").strip():
        problems.append("approved text is empty")

    try:
        full = _run(root, "rev-parse", "--verify", f"{amendment_commit}^{{commit}}").stdout.decode().strip()
        _run(root, "rev-parse", "--verify", f"{full}~1")
        date = _run(root, "log", "-1", "--format=%cI", full).stdout.decode().strip()
        if _run(root, "merge-base", "--is-ancestor", full, "HEAD", check=False).returncode != 0:
            problems.append(f"amendment commit {full} is not an ancestor of HEAD")
    except AmendmentError as e:
        problems.append(f"amendment commit: {e}")
        return problems, draft
    draft.update(commit=full, commit_date=date, amendment_commit=[full, date])

    blobs: dict = {}
    with _guard(problems, "preview blobs.json"):
        blobs = _load_json(preview / "blobs.json")
    draft["blobs"] = blobs

    report: dict = {}
    with _guard(problems, "rerun report"):
        report = _load_json(rerun / "reproduce_report.json")
        if report.get("score_exit") != 0:
            problems.append(f"rerun report: score_exit is {report.get('score_exit')!r}, not 0")
        if report.get("cold_cache") is not True:
            problems.append("rerun report: the render cache was not cold")
        if report.get("harness") != RECORD_HARNESS:
            problems.append(f"rerun report: harness is {report.get('harness')!r}, not {RECORD_HARNESS}")
    draft["report"] = report

    published_path, fresh_path = root / PUBLISHED_RESULTS, rerun / "evidence" / "results.json"
    with _guard(problems, "rerun comparison"):
        published, fresh = _load_json(published_path), _load_json(fresh_path)
        a1 = _load_json(root / fa.AMENDMENT_PATH)
        proposal = {
            "files": {"cli": {"new_blob": blobs.get("cli_new")}, "module": {"new_blob": blobs.get("module_new")}},
            "amendment_commit": [full, date],
        }
        cmp = fa.compare_published(published, fresh, amendment=a1, rename=proposal)
        draft.update(published=published, fresh=fresh, a1=a1, compare=cmp)
        if cmp["compare"] != "identical_modulo_amendment":
            problems.append(f"rerun is {cmp['compare']}, not identical_modulo_amendment: {cmp['problems']}")
        elif cmp["differing_keys"] != list(RENAME_KEYS):
            problems.append(f"rerun differing keys are not exactly the rename keys: {cmp['differing_keys']}")
        draft["published_sha256"], draft["fresh_sha256"] = fa._sha256(published_path), fa._sha256(fresh_path)

    with _guard(problems, "frozen compare-results"):
        out = compare_runner(published_path.resolve(), fresh_path.resolve()).strip()
        want = "different: " + ", ".join(RENAME_KEYS)
        draft["frozen_compare_output"] = out
        if out != want:
            problems.append(f"frozen compare-results output is {out!r}, expected {want!r}")

    with _guard(problems, "ACCEPTANCE.md"):
        if (rerun / "ACCEPTANCE.md").read_bytes() != (root / PUBLISHED_ACCEPTANCE_MD).read_bytes():
            problems.append(f"re-rendered ACCEPTANCE.md differs from {PUBLISHED_ACCEPTANCE_MD}")

    with _guard(problems, "frozen file blobs"):
        for label, rel, key in (("cli", FROZEN_CLI, "cli_new"), ("module", NEW_MODULE, "module_new")):
            cur = fa._blob_of(root / rel, root)
            if cur != blobs.get(key):
                problems.append(f"current {rel} blob {cur} is not the preview {label} blob {blobs.get(key)}")
        proofs = {}
        for label, old_blob, rel in (("cli", CLI_OLD_BLOB, FROZEN_CLI), ("module", MODULE_OLD_BLOB, NEW_MODULE)):
            try:
                proofs[label] = prove_rename(_cat_text(root, old_blob), (root / rel).read_text(encoding="utf-8"))
            except AmendmentError as e:
                problems.append(f"{label} names-only proof: {e}")
        draft["proofs"] = proofs

    with _guard(problems, "RUN_LOCK.json"):
        if (root / fa.LOCK_PATH).read_bytes() != (preview / "RUN_LOCK.json").read_bytes():
            problems.append("current RUN_LOCK.json differs from the preview RUN_LOCK.json")

    with _guard(problems, "REGISTRY.json"):
        at_commit = _cat_blob_at(root, full, fa.REGISTRY_PATH.as_posix())
        if at_commit != (preview / "REGISTRY.json").read_bytes():
            problems.append(f"REGISTRY.json at {full} differs from the preview REGISTRY.json")
        draft["registry_new_bytes"] = at_commit

    with _guard(problems, "experiments tree"):
        exp = _load_json(preview / "exp001.json")
        got = fa._git("rev-parse", f"HEAD:{EXP001_NEW}", root=root).strip()
        draft["exp001"] = exp
        if got != exp.get("predicted_new_tree"):
            problems.append(f"experiments tree {got} at HEAD is not the preview predicted_new_tree {exp.get('predicted_new_tree')}")

    if _run(root, "cat-file", "-e", f"HEAD:{RELEASE_HISTORY_REF}", check=False).returncode == 0:
        problems.append(f"HEAD tracks {RELEASE_HISTORY_REF}: the record is written in the development repository, not in a public clone")
    if _run(root, "cat-file", "-e", f"HEAD:{PUBLIC_PATHS_REF}", check=False).returncode != 0:
        problems.append(f"HEAD does not track {PUBLIC_PATHS_REF}: the latest-only files cannot be classified")

    with _guard(problems, "commit rewrite"):
        res = check_commit_rewrite(root, full)
        draft["commit_rewrite"] = res
        problems += [f"commit rewrite: {p}" for p in res["problems"]]
        if res["filtered"]:
            problems.append("commit rewrite: HEAD is a filtered public clone")
        if res["pure_rewrite"] is not True:
            problems.append(f"commit rewrite: {full} is not a pure rewrite (pure_rewrite is {res['pure_rewrite']!r})")
        rewritten = _unpinned_rewrites(root, full, res, problems)
        kept, unpinned = _kept_data(root, rewritten)
        draft.update(rewritten_unpinned=rewritten, recorded_data_kept=kept, kept_unpinned=unpinned)
    return problems, draft


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rename_amendment", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("preview")
    sp.add_argument("--out", default=DEFAULT_PREVIEW_DIR)
    sp.set_defaults(fn=cmd_preview)
    sv = sub.add_parser("verify")
    sv.add_argument("--history", default=None)
    sv.set_defaults(fn=fa.cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except AmendmentError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
