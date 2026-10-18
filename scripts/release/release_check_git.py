"""Public-history checks on the exported repository (bare or non-bare).

Every check reads git objects through read-only plumbing (`git -C <repo> ...`); nothing in the repository is written.
A finding names the path, commit and rule, never the matched secret.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

from scripts.release import common as C
from scripts.release import freeze_amendment as FA
from scripts.release import public_paths
from scripts.release.release_check import overall, result

BINARY_SUFFIXES = (".png", ".pdf", ".gguf", ".safetensors", ".bin", ".pt", ".parquet")
MAX_BLOB = 50 * 1024 * 1024
JSON_SUFFIXES = (".json", ".jsonl")
PAPER_KEYS = ("arxiv_id", "paper_id", "paper")
LOCK_PATH = FA.LOCK_PATH.as_posix()
FREEZE_PATH = "data/benchmark/BENCHMARK_FREEZE.json"
BENCH_RECORDS = "data/benchmark/records.jsonl"
TRAINSET = Path("data/train/sets/full/trainset.jsonl")
HISTORY_FILE = "RELEASE_HISTORY.json"
AMENDMENT_FILE = FA.AMENDMENT_PATH.as_posix()
RESULTS_FILE = C.RESULTS_JSON.as_posix()


class GitError(RuntimeError):
    pass


class Repo:
    """Read-only view of a git repository with cached tree snapshots."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        self._snap: dict[str, list[tuple[str, str, int]]] | None = None
        self.commits: list[str] = []

    def git(self, *args: str, check: bool = True) -> bytes:
        r = subprocess.run(["git", "-C", self.path, *args], capture_output=True, check=False, env=self._env)
        if check and r.returncode != 0:
            raise GitError(f"git {args[0]} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
        return r.stdout if r.returncode == 0 else b""

    def text(self, *args: str, check: bool = True) -> str:
        return self.git(*args, check=check).decode("utf-8", "replace")

    def head(self) -> str:
        return self.text("rev-parse", "HEAD").strip()

    def has_object(self, sha: str) -> bool:
        r = subprocess.run(["git", "-C", self.path, "cat-file", "-e", sha], capture_output=True, check=False, env=self._env)
        return r.returncode == 0

    def blob_id(self, rev_path: str) -> str | None:
        out = self.git("rev-parse", "--verify", "-q", rev_path, check=False).decode().strip()
        return out or None

    def show(self, rev_path: str) -> bytes | None:
        bid = self.blob_id(rev_path)
        return self.git("cat-file", "blob", bid) if bid else None

    def show_json(self, rev_path: str):
        data = self.show(rev_path)
        if data is None:
            return None
        try:
            return json.loads(data)
        except ValueError:
            return None

    def snapshot(self) -> dict[str, list[tuple[str, str, int]]]:
        """commit -> [(path, blob, size)] for every commit of `rev-list --reverse HEAD`."""
        if self._snap is None:
            self.commits = [c for c in self.text("rev-list", "--reverse", "HEAD").split() if c]
            snap: dict[str, list[tuple[str, str, int]]] = {}
            for c in self.commits:
                out = self.git("ls-tree", "-r", "-l", "-z", c)
                rows = []
                for rec in out.split(b"\0"):
                    if not rec:
                        continue
                    meta, _, path = rec.decode("utf-8", "surrogateescape").partition("\t")
                    _mode, typ, sha, size = meta.split()
                    if typ == "blob":
                        rows.append((path, sha, int(size)))
                snap[c] = rows
            self._snap = snap
        return self._snap

    def iter_objects(self, shas: list[str]):
        """Yield (sha, type, bytes|None) from one `cat-file --batch` stream."""
        p = subprocess.Popen(
            ["git", "-C", self.path, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self._env
        )

        def feed():
            try:
                for s in shas:
                    p.stdin.write(s.encode() + b"\n")
            finally:
                p.stdin.close()

        t = threading.Thread(target=feed, daemon=True)
        t.start()
        try:
            for _ in shas:
                header = p.stdout.readline()
                if not header:
                    break
                parts = header.split()
                if len(parts) == 2 and parts[1] == b"missing":
                    yield parts[0].decode(), "missing", None
                    continue
                sha, typ, size = parts[0].decode(), parts[1].decode(), int(parts[2])
                data = p.stdout.read(size)
                p.stdout.read(1)
                yield sha, typ, data
        finally:
            t.join()
            p.stdout.close()
            p.wait()


def _is_text(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def _head_paths(repo: Repo) -> dict[str, str]:
    snap = repo.snapshot()
    return {p: b for p, b, _ in snap[repo.commits[-1]]} if repo.commits else {}


# ------------------------------------------------------------------------------------------------------- checks
def check_paths(repo: Repo, spec) -> dict:
    snap = repo.snapshot()
    problems: list[str] = []
    overlay_idx = next((i for i, c in enumerate(repo.commits) if any(p == HISTORY_FILE for p, _, _ in snap[c])), None)
    head = _head_paths(repo)
    cache: dict[str, str | None] = {}

    def cls(path: str) -> str | None:
        if path not in cache:
            cache[path] = public_paths.classify(path, spec)
        return cache[path]

    reported: set[tuple[str, str]] = set()
    for i, c in enumerate(repo.commits):
        pre = overlay_idx is None or i < overlay_idx
        for path, blob, _ in snap[c]:
            k = cls(path)
            if k is None:
                msg = "path is not allowed"
            elif k == "overlay" and pre:
                msg = "overlay path present before the overlay commit"
            elif k == "latest-only" and head.get(path) != blob:
                msg = "latest-only path has a blob that differs from HEAD"
            else:
                continue
            if (path, msg) not in reported:
                reported.add((path, msg))
                problems.append(f"{path} in {c[:10]}: {msg}")
    return result("paths", problems)


def check_forbidden_history(repo: Repo) -> dict:
    repo.snapshot()
    names: dict[str, str] = {}
    first_commit: dict[str, str] = {}
    for c in repo.commits:
        for path, blob, _ in repo.snapshot()[c]:
            names.setdefault(blob, path)
            first_commit.setdefault(blob, c)
    shas: list[str] = []
    for line in repo.text("rev-list", "--objects", "--all").splitlines():
        sha, _, path = line.partition(" ")
        shas.append(sha)
        if path:
            names.setdefault(sha, path)
    problems: list[str] = []
    for sha, typ, data in repo.iter_objects(sorted(set(shas))):
        if typ not in ("blob", "commit", "tag") or data is None or len(data) > MAX_BLOB or not _is_text(data):
            continue
        text = data.decode("utf-8", "replace")
        where = names.get(sha, f"{typ} {sha[:10]}") if typ == "blob" else f"{typ} {sha[:10]} (message or header)"
        commit = f" in {first_commit[sha][:10]}" if sha in first_commit else ""
        for name, pat in C.FORBIDDEN_PATTERNS:
            if pat.search(text):
                problems.append(f"{where}{commit}: {name}")
    return result("forbidden_history", problems)


def _walk_content(obj, found: list[tuple[str | None, str | None]]) -> None:
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if any(isinstance(o.get(k), str) and o[k].strip() for k in C.CONTENT_KEYS):
                rid = o.get("id") if isinstance(o.get("id"), str) else None
                paper = next((o[k] for k in PAPER_KEYS if isinstance(o.get(k), str) and o[k]), None)
                found.append((rid, paper))
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)


def _content_rows(repo: Repo) -> list[tuple[str, set[str], str | None, str | None]]:
    """(blob, paths, row id, paper id) for every public JSON/JSONL dict that carries LaTeX or model text."""
    if getattr(repo, "_content", None) is not None:
        return repo._content
    paths: dict[str, set[str]] = {}
    for c in repo.commits:
        for path, blob, size in repo.snapshot()[c]:
            if path.endswith(JSON_SUFFIXES) and size <= MAX_BLOB:
                paths.setdefault(blob, set()).add(path)
    out: list[tuple[str, set[str], str | None, str | None]] = []
    for sha, typ, data in repo.iter_objects(sorted(paths)):
        if typ != "blob" or data is None:
            continue
        text = data.decode("utf-8", "replace")
        objs = []
        if any(p.endswith(".jsonl") for p in paths[sha]):
            for line in text.splitlines():
                if line.strip():
                    try:
                        objs.append(json.loads(line))
                    except ValueError:
                        continue
        else:
            try:
                objs.append(json.loads(text))
            except ValueError:
                pass
        found: list[tuple[str | None, str | None]] = []
        for o in objs:
            _walk_content(o, found)
        out += [(sha, paths[sha], rid, paper) for rid, paper in found]
    repo._content = out
    return out


def _attributed_ids(repo: Repo, attribution: Path | None) -> set[str] | None:
    data = repo.show("HEAD:ATTRIBUTION.jsonl")
    if data is None and attribution is not None and Path(attribution).is_file():
        data = Path(attribution).read_bytes()
    if data is None:
        return None
    return {json.loads(x)["arxiv_id"] for x in data.decode("utf-8").splitlines() if x.strip()}


def _base(paper: str) -> str | None:
    from lightonocr_math.ids import base_id, from_safe_id

    for cand in (paper, from_safe_id(paper)):
        try:
            return base_id(cand)
        except ValueError:
            continue
    return None


def check_attributed_content(repo: Repo, attribution: Path | None) -> dict:
    repo.snapshot()
    allowed = _attributed_ids(repo, attribution)
    if allowed is None:
        return result("attributed_content", not_run="no ATTRIBUTION.jsonl at HEAD and none given")
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for blob, paths, _rid, paper in _content_rows(repo):
        if all(p.startswith("tests/") for p in paths) or paper is None:
            continue
        base = _base(paper)
        if base is None or base not in allowed:
            key = (blob, base or paper)
            if key not in seen:
                seen.add(key)
                problems.append(f"{min(paths)}: content row of paper {base or paper!r} is not in ATTRIBUTION.jsonl")
    return result("attributed_content", problems)


def _reserved_ids(reserved_sources, trainset: Path) -> set[str] | None:
    if not reserved_sources or not Path(trainset).is_file():
        return None
    ids: set[str] = set()
    for p in reserved_sources:
        if not Path(p).is_file():
            return None
        ids |= {r["id"] for r in C.read_jsonl(p)}
    return ids - {r["id"] for r in C.read_jsonl(trainset)}


def check_reserved_rows(repo: Repo, reserved_sources, trainset: Path) -> dict:
    reserved = _reserved_ids(reserved_sources, trainset)
    if reserved is None:
        return result("reserved_rows", not_run="reserved train/dev sources are not available")
    repo.snapshot()
    problems = []
    seen = set()
    for blob, paths, rid, _paper in _content_rows(repo):
        if rid in reserved and (blob, rid) not in seen:
            seen.add((blob, rid))
            problems.append(f"{min(paths)}: a row with content has a reserved train/dev id")
    return result("reserved_rows", problems)


def check_binaries(repo: Repo) -> dict:
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for c, rows in repo.snapshot().items():
        for path, blob, size in rows:
            if (path, blob) in seen:
                continue
            seen.add((path, blob))
            if path.lower().endswith(BINARY_SUFFIXES):
                problems.append(f"{path} in {c[:10]}: binary file type")
            if size > MAX_BLOB:
                problems.append(f"{path} in {c[:10]}: blob over 50 MB")
            for name, pat in C.FORBIDDEN_FILE_PATTERNS:
                if pat.search(path):
                    problems.append(f"{path} in {c[:10]}: forbidden file name ({name})")
    return result("binaries", problems)


def _published(repo: Repo, published_results) -> dict | None:
    if published_results is not None:
        try:
            return json.loads(Path(published_results).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return repo.show_json(f"HEAD:{RESULTS_FILE}")


def check_frozen_code(repo: Repo) -> dict:
    problems: list[str] = []
    lock = repo.show_json(f"HEAD:{LOCK_PATH}")
    arm = ((lock or {}).get("arms") or {}).get("tuned-crops")
    if not arm:
        return result("frozen_code", [f"{LOCK_PATH} or its tuned-crops arm is missing at HEAD"])
    if repo.blob_id(f"HEAD:{FA.FROZEN_MODULE}") != arm.get("acceptance_module_blob"):
        problems.append(f"{FA.FROZEN_MODULE} blob differs from RUN_LOCK.json")
    if repo.blob_id(f"HEAD:{FA.FROZEN_CLI}") != arm.get("acceptance_cli_blob"):
        problems.append(f"{FA.FROZEN_CLI} blob differs from RUN_LOCK.json")
    freeze = repo.show_json(f"HEAD:{FREEZE_PATH}")
    records = repo.show(f"HEAD:{BENCH_RECORDS}")
    if freeze is None or records is None:
        problems.append("BENCHMARK_FREEZE.json or benchmark records.jsonl is missing at HEAD")
    elif hashlib.sha256(records).hexdigest() != freeze.get("records_sha256"):
        problems.append("benchmark records.jsonl sha256 differs from BENCHMARK_FREEZE.json")
    return result("frozen_code", problems)


def _commit_map(repo: Repo) -> dict | None:
    h = repo.show_json(f"HEAD:{HISTORY_FILE}")
    return None if h is None else dict(h.get("commit_map") or {})


def check_freeze_amendment(repo: Repo, published_results) -> dict:
    problems: list[str] = []
    pub = _published(repo, published_results)
    lock = repo.show_json(f"HEAD:{LOCK_PATH}")
    if pub is None or lock is None:
        return result("freeze_amendment", ["published results.json or RUN_LOCK.json is unreadable"])
    cf = pub.get("code_freeze") or {}
    arms = [(lock.get("arms") or {}).get(a) or {} for a in FA.ARMS]
    cli_head = repo.blob_id(f"HEAD:{FA.FROZEN_CLI}")
    mod_head = repo.blob_id(f"HEAD:{FA.FROZEN_MODULE}")
    pub_mod = {(cf.get("current_blobs") or {}).get("module"), (cf.get("lock_blobs") or {}).get("module")}
    if pub_mod != {mod_head}:
        problems.append("module blob at HEAD differs from the published code_freeze record")
    if any(a.get("acceptance_module_blob") != mod_head for a in arms):
        problems.append("RUN_LOCK acceptance_module_blob differs from the HEAD module blob")
    pub_cli = {(cf.get("current_blobs") or {}).get("cli"), (cf.get("lock_blobs") or {}).get("cli")}
    am = repo.show_json(f"HEAD:{AMENDMENT_FILE}")
    if am is None:
        if pub_cli != {cli_head}:
            problems.append("no amendment record and the HEAD scripts/acceptance.py blob differs from the published one")
        if any(a.get("acceptance_cli_blob") != cli_head for a in arms):
            problems.append("RUN_LOCK acceptance_cli_blob differs from the HEAD blob")
        return result("freeze_amendment", problems)

    old, new = am.get("old_blob"), am.get("new_blob")
    if pub_cli != {old}:
        problems.append("old_blob differs from the published code_freeze record")
    if any(a.get("acceptance_cli_blob_generated") != old for a in arms):
        problems.append("RUN_LOCK acceptance_cli_blob_generated differs from old_blob")
    if new != cli_head:
        problems.append("new_blob differs from the HEAD scripts/acceptance.py blob")
    if any(a.get("acceptance_cli_blob") != new for a in arms):
        problems.append("RUN_LOCK acceptance_cli_blob differs from new_blob")
    if am.get("module_blob") != mod_head:
        problems.append("amendment module_blob differs from the HEAD module blob")

    cmap = _commit_map(repo)
    old_commit, amend_commit = am.get("old_commit") or [None, None], am.get("amendment_commit") or [None, None]
    mapped_old = old_commit[0] if cmap is None else cmap.get(old_commit[0])
    if not mapped_old:
        problems.append("old_commit is not in RELEASE_HISTORY.json")
    else:
        at_old = repo.blob_id(f"{mapped_old}:{FA.FROZEN_CLI}")
        if at_old != old:
            problems.append("old_blob is not scripts/acceptance.py at the public old commit")
    if not old or not repo.has_object(str(old)):
        problems.append("old_blob is not reachable in the public history")
    else:
        try:
            old_text = repo.text("cat-file", "blob", old)
            new_text = (repo.show(f"HEAD:{FA.FROZEN_CLI}") or b"").decode("utf-8")
            recorded = tuple((r["name"], r["old"], r["new"]) for r in am.get("replacements", []))
            if recorded != FA.REPLACEMENTS:
                problems.append("recorded replacements differ from the expected path constants")
            FA.prove(old_text, new_text, FA.REPLACEMENTS)
        except (FA.AmendmentError, GitError, KeyError, SyntaxError, UnicodeDecodeError) as e:
            problems.append(f"proof does not recompute: {e}")
    last = repo.text("log", "-1", "--format=%H %cI", "HEAD", "--", FA.FROZEN_CLI).split()
    mapped_amend = amend_commit[0] if cmap is None else (cmap or {}).get(amend_commit[0])
    if len(last) != 2 or last[0] != mapped_amend or last[1] != amend_commit[1]:
        problems.append("the last commit touching scripts/acceptance.py is not the recorded amendment commit and date")
    eq = am.get("rerun_equivalence") or {}
    if eq.get("compare") != "identical_modulo_amendment" or eq.get("differing_keys") != list(FA.AMENDED_KEYS):
        problems.append("rerun_equivalence is not identical_modulo_amendment with exactly the amended keys")
    return result("freeze_amendment", problems)


def check_history_map(repo: Repo, published_results) -> dict:
    cmap = _commit_map(repo)
    if cmap is None:
        return result("history_map", not_run="RELEASE_HISTORY.json is not at HEAD")
    pub = _published(repo, published_results)
    if pub is None:
        return result("history_map", ["published results.json is unreadable"])
    problems: list[str] = []
    am = repo.show_json(f"HEAD:{AMENDMENT_FILE}") or {}
    for path, (commit, date) in ((pub.get("code_freeze") or {}).get("commits") or {}).items():
        mapped = cmap.get(commit)
        if not mapped:
            problems.append(f"{path}: published commit {commit[:10]} is not in the commit map")
            continue
        got = repo.text("show", "-s", "--format=%cI", mapped, check=False).strip()
        if got != date:
            problems.append(f"{path}: committer date of the mapped commit differs from the published date")
        last = repo.text("log", "-1", "--format=%H %cI", "HEAD", "--", path).split()
        if path == FA.FROZEN_CLI and am:
            ac = am.get("amendment_commit") or [None, None]
            want = (cmap.get(ac[0]), ac[1])
        else:
            want = (mapped, date)
        if len(last) != 2 or (last[0], last[1]) != want:
            problems.append(f"{path}: last commit at HEAD is not the mapped frozen commit")
    return result("history_map", problems)


def check_remotes(repo: Repo) -> dict:
    out = repo.text("remote").strip()
    return result("remotes", [f"remote configured: {n}" for n in out.split()])


def check_planning_docs(repo: Repo) -> dict:
    problems: list[str] = []
    seen: set[str] = set()
    for c, rows in repo.snapshot().items():
        for path, _, _ in rows:
            if (path.startswith((".planning/", "docs/")) or path in (".planning", "docs")) and path not in seen:
                seen.add(path)
                problems.append(f"{path} in {c[:10]}: planning and docs paths are never published")
    head = _head_paths(repo)
    have = {p for p in head if p.startswith("evidence/")}
    want = set(C.PUBLIC_EVIDENCE_FILES)
    for p in sorted(want - have):
        problems.append(f"HEAD lacks evidence file {p}")
    for p in sorted(have - want):
        problems.append(f"HEAD has an unexpected evidence file {p}")
    return result("planning_docs", problems)


def check_git(
    repo,
    *,
    final: bool = False,
    reserved_sources=None,
    attribution=None,
    published_results=None,
    spec=None,
    trainset: Path = TRAINSET,
) -> dict:
    r = Repo(repo)
    if not r.snapshot():
        return {"surface": "git", "final": final, "status": "fail", "checks": [result("paths", ["repository has no commits"])]}
    spec = spec if spec is not None else public_paths.load_spec()
    published = published_results if (published_results and Path(published_results).is_file()) else None
    checks = [
        check_paths(r, spec),
        check_forbidden_history(r),
        check_attributed_content(r, attribution),
        check_reserved_rows(r, reserved_sources, trainset),
        check_binaries(r),
        check_frozen_code(r),
        check_freeze_amendment(r, published),
        check_history_map(r, published),
        check_remotes(r),
        check_planning_docs(r),
    ]
    return {"surface": "git", "final": final, "status": overall(checks, final), "checks": checks}
