"""Export the filtered public code history into a local bare repository.

Usage:
  python -m scripts.release.export_public [--source .] [--out release/code] [--overlay release/code/overlay]
      [--published-results PATH] [--amendment PATH] [--identity-email ADDR]

The development repository is only read. History is rewritten with `git filter-branch` in a throwaway clone that has
no remote: every commit keeps the paths the allow-list classifies as `history`, or `latest-only` where the blob equals
the source HEAD blob; kept blobs with a scrub pattern (the personal address) are rewritten; author and committer
dates and names are kept while the email of every commit becomes one neutral address. One release commit with the
overlay files and RELEASE_HISTORY.json is added on top, and the result is cloned bare into `<out>/public.git`.
Nothing is pushed or uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.release import public_paths

TOOL_ROOT = Path(__file__).resolve().parents[2]
RELEASE_COMMIT_MESSAGE = "release: add release documents and results"
OVERLAY_SKIP = {"SHA256SUMS"}


class ExportError(RuntimeError):
    pass


def git(repo: Path | str, *args: str, input: bytes | None = None, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        input=input,
        env={**os.environ, **(env or {})},
        check=False,
    )
    if check and r.returncode != 0:
        raise ExportError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode("utf-8", "replace").rstrip("\n")


def snapshot(repo: Path) -> dict:
    return {
        "head": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "refs": hashlib.sha256(git(repo, "for-each-ref", "--format=%(refname) %(objectname)").encode()).hexdigest(),
        "count": int(git(repo, "rev-list", "--count", "--all")),
    }


# ----------------------------------------------------------------------------------------------- index filter
def index_filter(spec_path: Path, head_blobs_path: Path, scrub_map_path: Path | None = None) -> int:
    """Rewrite the temporary index of one commit (runs inside the throwaway clone with GIT_INDEX_FILE set)."""
    spec = public_paths.load_spec(spec_path)
    head_blobs = json.loads(Path(head_blobs_path).read_text(encoding="utf-8"))
    scrub = json.loads(Path(scrub_map_path).read_text(encoding="utf-8")) if scrub_map_path else {}
    out = subprocess.run(["git", "ls-files", "-s", "-z"], capture_output=True, check=True).stdout
    drop: list[bytes] = []
    rewrite: list[bytes] = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, raw_path = rec.partition(b"\t")
        mode, sha, _stage = meta.decode().split(" ")
        path = raw_path.decode("utf-8", "surrogateescape")
        cls = public_paths.classify(path, spec)
        keep = cls == "history" or (cls == "latest-only" and head_blobs.get(path) == sha)
        if not keep:
            drop.append(raw_path)
        elif sha in scrub:
            rewrite.append(f"{mode} {scrub[sha]}\t".encode() + raw_path + b"\0")
    if drop:
        subprocess.run(["git", "update-index", "--force-remove", "-z", "--stdin"], input=b"\0".join(drop) + b"\0", check=True)
    if rewrite:
        subprocess.run(["git", "update-index", "-z", "--index-info"], input=b"".join(rewrite), check=True)
    return 0


# ----------------------------------------------------------------------------------------------- helpers
def _head_blobs(source: Path, spec) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in git(source, "ls-tree", "-r", "HEAD").splitlines():
        meta, _, path = line.partition("\t")
        if public_paths.classify(path, spec) == "latest-only":
            out[path] = meta.split()[2]
    return out


def _historical_paths(source: Path) -> list[str]:
    return sorted({p for p in git(source, "log", "--all", "--name-only", "--format=").splitlines() if p})


def _scrub_blobs(work: Path, spec, replacements) -> dict[str, str]:
    """Map every blob under a kept path that matches a scrub pattern to its rewritten blob."""
    paths: dict[str, str] = {}
    for line in git(work, "rev-list", "--objects", "--all").splitlines():
        sha, _, path = line.partition(" ")
        if path and public_paths.classify(path, spec) in ("history", "latest-only"):
            paths.setdefault(sha, path)
    if not paths:
        return {}
    ids = sorted(paths)
    proc = subprocess.run(
        ["git", "-C", str(work), "cat-file", "--batch"], input=("\n".join(ids) + "\n").encode(), capture_output=True, check=True
    )
    buf, pos, mapping = proc.stdout, 0, {}
    while pos < len(buf):
        nl = buf.index(b"\n", pos)
        head = buf[pos:nl].decode().split()
        pos = nl + 1
        if len(head) < 3:
            continue  # "<sha> missing"
        size = int(head[2])
        data = buf[pos : pos + size]
        pos += size + 1
        if head[1] != "blob":
            continue
        if b"\0" in data[:8000]:
            continue
        text = data.decode("utf-8", "surrogateescape")
        new = text
        for pat, repl in replacements:
            new = pat.sub(repl, new)
        if new != text:
            mapping[head[0]] = git(work, "hash-object", "-w", "--stdin", input=new.encode("utf-8", "surrogateescape"))
    return mapping


def _read_map(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    return [tuple(line.split()) for line in path.read_text().splitlines() if line.strip()]  # type: ignore[misc]


def _published_hashes(published: Path, amendment: Path) -> dict[str, str]:
    """Source commit hash -> role, for every commit RELEASE_HISTORY.json must map."""
    res = json.loads(Path(published).read_text(encoding="utf-8"))
    am = json.loads(Path(amendment).read_text(encoding="utf-8"))
    roles = {v[0]: f"code_freeze:{k}" for k, v in (res.get("code_freeze") or {}).get("commits", {}).items()}
    roles[am["amendment_commit"][0]] = "amendment_commit"
    return roles


def _overlay_files(overlay: Path, spec) -> list[str]:
    files = sorted(p.relative_to(overlay).as_posix() for p in overlay.rglob("*") if p.is_file())
    files = [f for f in files if f not in OVERLAY_SKIP]
    bad = [f for f in files if public_paths.classify(f, spec) != "overlay"]
    if bad:
        raise ExportError(f"overlay files not classified as overlay: {bad}")
    return files


# ----------------------------------------------------------------------------------------------- export
def export(
    source: Path = Path("."),
    out: Path = Path("release/code"),
    overlay: Path | None = None,
    published_results: Path = Path("evidence/acceptance/results.json"),
    amendment: Path = Path("evidence/acceptance/FREEZE_AMENDMENT.json"),
    spec_path: Path | None = None,
    identity_email: str | None = None,
    scrub=None,
) -> dict:
    from scripts.release import common as C

    source, out = Path(source).resolve(), Path(out).resolve()
    overlay = Path(overlay).resolve() if overlay else out / "overlay"
    spec_path = Path(spec_path).resolve() if spec_path else public_paths.DEFAULT_SPEC.resolve()
    identity_email = identity_email or C.PUBLIC_IDENTITY_EMAIL
    scrub = C.SCRUB_REPLACEMENTS if scrub is None else scrub
    published_results, amendment = Path(published_results), Path(amendment)
    if not published_results.is_absolute():
        published_results = source / published_results
    if not amendment.is_absolute():
        amendment = source / amendment

    top = Path(git(source, "rev-parse", "--show-toplevel")).resolve()
    if top != source:
        raise ExportError(f"--source must be the repository root, got {source} (root {top})")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ExportError("the source worktree has modified tracked files")
    work = out / "work"
    if work.resolve() == source:
        raise ExportError("work directory resolves to the source repository")

    spec = public_paths.load_spec(spec_path)
    files = _overlay_files(overlay, spec)
    roles = _published_hashes(published_results, amendment)
    before = snapshot(source)

    out.mkdir(parents=True, exist_ok=True)
    head_blobs_path, map_path, scrub_path = out / "head_blobs.json", out / "commit_map.txt", out / "scrub_map.json"
    head_blobs_path.write_text(json.dumps(_head_blobs(source, spec), indent=0, sort_keys=True), encoding="utf-8")
    map_path.unlink(missing_ok=True)
    if work.exists():
        shutil.rmtree(work)
    git(source, "clone", "--no-local", "--single-branch", "--branch", before["branch"], str(source), str(work))
    git(work, "remote", "remove", "origin")
    if Path(git(work, "rev-parse", "--show-toplevel")).resolve() == top:
        raise ExportError("throwaway clone resolves to the source repository")
    tags = git(work, "tag", "--list").split()
    for t in tags:
        git(work, "tag", "-d", t)

    scrub_map = _scrub_blobs(work, spec, scrub)
    scrub_path.write_text(json.dumps(scrub_map, indent=0, sort_keys=True), encoding="utf-8")

    tool = f"{sys.executable} -P -m scripts.release.export_public index-filter --spec {spec_path} --head-blobs {head_blobs_path} --scrub-map {scrub_path}"
    commit_filter = (
        f"export GIT_AUTHOR_EMAIL={identity_email} GIT_COMMITTER_EMAIL={identity_email}; "
        f'n=$(git_commit_non_empty_tree "$@") && echo "$GIT_COMMIT $n" >> {map_path} && echo "$n"'
    )
    env = {"FILTER_BRANCH_SQUELCH_WARNING": "1", "PYTHONPATH": str(TOOL_ROOT)}
    git(work, "filter-branch", "--index-filter", tool, "--commit-filter", commit_filter, "--", before["branch"], env=env)
    shutil.rmtree(work / ".git" / "refs" / "original", ignore_errors=True)
    git(work, "reflog", "expire", "--expire=now", "--all")
    git(work, "gc", "--prune=now", "-q")
    git(work, "branch", "-m", "main")

    pairs = _read_map(map_path)
    first: dict[str, str] = {}
    for src, new in pairs:
        first.setdefault(new, src)
    real = {src: new for new, src in first.items()}
    n_filtered = int(git(work, "rev-list", "--count", "main"))
    if n_filtered != len(first):
        raise ExportError(f"commit map covers {len(first)} public commits, the repository has {n_filtered}")

    # Dates, names and messages survive; only the email becomes the neutral identity.
    fmt = "%an%n%aI%n%cn%n%cI%n%B"
    for src, new in real.items():
        a = git(source, "show", "-s", f"--format={fmt}", src)
        b = git(work, "show", "-s", f"--format={fmt}", new)
        if a != b:
            raise ExportError(f"commit {src[:10]} changed identity names, dates or message in the rewrite")
        for f in ("%ae", "%ce"):
            if git(work, "show", "-s", f"--format={f}", new) != identity_email:
                raise ExportError(f"commit {new[:10]} does not carry the public identity email")
    # The kept HEAD tree is the source HEAD tree restricted to kept paths.
    src_head = {
        line.partition("\t")[2]: line.split()[2] for line in git(source, "ls-tree", "-r", "HEAD").splitlines()
    }
    for line in git(work, "ls-tree", "-r", "main").splitlines():
        path = line.partition("\t")[2]
        if src_head.get(path) != line.split()[2]:
            raise ExportError(f"{path} at the public HEAD differs from the source HEAD blob")

    history_map = {}
    for h, role in roles.items():
        if h not in real:
            raise ExportError(f"{role} commit {h} has no public counterpart (dropped or pruned)")
        history_map[h] = real[h]

    release_head = _release_commit(work, source, before, files, overlay, spec_path, history_map, identity_email)
    public = out / "public.git"
    if public.exists():
        shutil.rmtree(public)
    git(work, "clone", "--bare", "--no-local", str(work), str(public))
    git(public, "remote", "remove", "origin", check=False)
    if git(public, "remote"):
        raise ExportError("the public repository has a remote")

    after = snapshot(source)
    if after != before:
        raise ExportError(f"source repository changed during the export: {before} -> {after}")

    hist = _historical_paths(source)
    counts = Counter(public_paths.classify(p, spec) or "dropped" for p in hist)
    report = {
        "source_head": before["head"],
        "source_branch": before["branch"],
        "source_commits": before["count"],
        "public_head": release_head,
        "public_commits": int(git(public, "rev-list", "--count", "main")),
        "pruned_commits": before["count"] - len(first),
        "historical_paths_by_class": {k: counts.get(k, 0) for k in (*public_paths.SECTIONS, "dropped")},
        "overlay_files": files,
        "scrubbed_blobs": len(scrub_map),
        "tags_removed": len(tags),
        "identity_email": identity_email,
        "mapped_commits": history_map,
        "public_paths_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }
    (out / "EXPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _release_commit(work, source, before, files, overlay, spec_path, history_map, identity_email) -> str:
    for rel in files:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(overlay / rel, dst)
    n_public = int(git(work, "rev-list", "--count", "main"))
    doc = {
        "schema": 1,
        "source_head": before["head"],
        "source_branch": before["branch"],
        "public_branch": "main",
        "public_paths_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "commit_map": history_map,
        "n_source_commits": before["count"],
        "n_public_commits": n_public + 1,
    }
    (work / "RELEASE_HISTORY.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for rel in [*files, "RELEASE_HISTORY.json"]:
        git(work, "add", "--", rel)
    name = git(source, "log", "-1", "--format=%an")
    env = {
        "GIT_AUTHOR_NAME": name,
        "GIT_COMMITTER_NAME": name,
        "GIT_AUTHOR_EMAIL": identity_email,
        "GIT_COMMITTER_EMAIL": identity_email,
    }
    git(work, "commit", "-q", "-m", RELEASE_COMMIT_MESSAGE, env=env)
    return git(work, "rev-parse", "HEAD")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_public", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("index-filter")
    f.add_argument("--spec", required=True)
    f.add_argument("--head-blobs", required=True)
    f.add_argument("--scrub-map")
    ap.add_argument("--source", default=".")
    ap.add_argument("--out", default="release/code")
    ap.add_argument("--overlay")
    ap.add_argument("--published-results", default="evidence/acceptance/results.json")
    ap.add_argument("--amendment", default="evidence/acceptance/FREEZE_AMENDMENT.json")
    ap.add_argument("--identity-email")
    args = ap.parse_args(argv)
    if args.cmd == "index-filter":
        return index_filter(Path(args.spec), Path(args.head_blobs), Path(args.scrub_map) if args.scrub_map else None)
    try:
        rep = export(
            Path(args.source), Path(args.out), Path(args.overlay) if args.overlay else None,
            Path(args.published_results), Path(args.amendment), identity_email=args.identity_email,
        )
    except ExportError as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: rep[k] for k in ("source_commits", "public_commits", "pruned_commits", "public_head")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
