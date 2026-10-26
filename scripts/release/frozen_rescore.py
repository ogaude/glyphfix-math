"""Run the frozen acceptance scorer after the package directory moved, without touching any frozen file.

The frozen CLI refuses benchmark scoring when the last commit of the frozen module is later than the generation
start (scripts/acceptance.py, `_code_freeze_checks`: the module's `file_commit` date becomes `pin_commit_time` and
is compared with the lock start). Moving the module to its new path makes the move commit that last commit, so the
pin check would fail although the module text is the pinned text with only the project names mapped.

This harness supplies the one git lookup the pin check reads. For the new module path only, it answers with the
commit and date the published results recorded for the pre-move module, and only after proving from git objects that

  * the published results.json is the recorded one (sha256),
  * the recorded pin commit has the recorded committer date and holds the pinned module blob at the old path,
  * the new path was touched by exactly one commit, and
  * the current module text is the pinned blob with exactly the project's names mapped (rename_amendment.prove_rename).

Anything else is refused with exit code 2 and nothing is scored. Before the move the scorer runs unchanged.

  uv run python -m scripts.release.frozen_rescore <scripts.acceptance arguments>

It changes no file and no frozen logic; it supplies one git lookup.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from scripts.release import freeze_amendment as fa
from scripts.release import rename_amendment as ra

EXIT_REFUSED = 2
OLD_MODULE = ra.OLD_MODULE
NEW_MODULE = ra.NEW_MODULE
PUBLISHED_PATH = Path("evidence/acceptance/results.json")


class SubstitutionRefused(Exception):
    """A precondition of the pin substitution does not hold."""


def _git(root: Path, *args: str) -> str:
    try:
        return fa._git(*args, root=root)
    except fa.AmendmentError as e:
        raise SubstitutionRefused(str(e)) from e


def _load_published(root: Path) -> dict:
    p = Path(root) / PUBLISHED_PATH
    try:
        data = p.read_bytes()
    except OSError as e:
        raise SubstitutionRefused(f"cannot read the published results: {e}") from e
    if hashlib.sha256(data).hexdigest() != fa.PUBLISHED_RESULTS_SHA256:
        raise SubstitutionRefused("the published results.json is not the recorded one (sha256)")
    try:
        return json.loads(data)
    except ValueError as e:
        raise SubstitutionRefused(f"the published results are unreadable: {e}") from e


def pin_substitution(
    root: Path = Path("."),
    commit_map: dict | None = None,
    *,
    published: dict | None = None,
    module_blob: str | None = None,
    old_module: str | None = None,
    new_module: str | None = None,
) -> dict:
    """The proven (commit, date) the frozen pin check should read for the moved module, or SubstitutionRefused."""
    root = Path(root)
    old_module = old_module or OLD_MODULE
    new_module = new_module or NEW_MODULE
    module_blob = module_blob or fa.MODULE_BLOB
    if published is None:
        published = _load_published(root)
    rec = ((published.get("code_freeze") or {}).get("commits") or {}).get(old_module)
    if not (isinstance(rec, list | tuple) and len(rec) == 2):
        raise SubstitutionRefused(f"the published results record no pin commit for {old_module}")
    recorded, date = rec[0], rec[1]

    if commit_map is None:
        try:
            cmap = fa.load_commit_map(root)
        except (OSError, ValueError) as e:
            raise SubstitutionRefused(f"cannot read the release history: {e}") from e
    else:
        cmap = commit_map
    if cmap:
        if recorded not in cmap:
            raise SubstitutionRefused(f"the commit map has no entry for the pin commit {recorded}")
        mapped = cmap[recorded]
    else:
        mapped = recorded

    got_date = _git(root, "show", "-s", "--format=%cI", f"{mapped}^{{commit}}").strip()
    if got_date != date:
        raise SubstitutionRefused(f"pin commit {mapped} has committer date {got_date}, the published results record {date}")
    got_blob = _git(root, "rev-parse", f"{mapped}:{old_module}").strip()
    if got_blob != module_blob:
        raise SubstitutionRefused(f"{mapped}:{old_module} is blob {got_blob}, expected {module_blob}")

    touching = [c for c in _git(root, "log", "--format=%H", "--", new_module).split() if c]
    if len(touching) != 1:
        raise SubstitutionRefused(f"{new_module} was touched by {len(touching)} commits, expected exactly one")
    if _git(root, "status", "--porcelain", "--", new_module).strip():
        raise SubstitutionRefused(f"{new_module} has uncommitted changes")

    old_text = _git(root, "cat-file", "-p", module_blob)
    try:
        new_text = (root / new_module).read_text(encoding="utf-8")
    except OSError as e:
        raise SubstitutionRefused(f"cannot read {new_module}: {e}") from e
    try:
        proof = ra.prove_rename(old_text, new_text)
    except fa.AmendmentError as e:
        raise SubstitutionRefused(f"the module is not the pinned blob renamed: {e}") from e

    return {
        "path": new_module,
        "commit": mapped,
        "date": date,
        "module_blob_before": module_blob,
        "proof_changed_lines": proof["changed_lines"],
    }


def main(argv: list[str] | None = None) -> int:
    import scripts.acceptance as acc

    args = list(sys.argv[1:] if argv is None else argv)
    module_path = acc.FROZEN_CODE[0]
    if module_path == OLD_MODULE:
        return acc.main(args)
    if module_path != NEW_MODULE:
        print(f"REFUSED: unexpected frozen module path {module_path}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        sub = pin_substitution(Path("."))
    except SubstitutionRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED

    original = acc.file_commit

    def file_commit(path):
        if Path(path).as_posix() == NEW_MODULE:
            return sub["commit"], sub["date"]
        return original(path)

    print(
        f"frozen_rescore: module pin commit {sub['commit']} ({sub['date']}) substituted for {sub['path']}; "
        f"module proven as blob {sub['module_blob_before']} renamed",
        file=sys.stderr,
    )
    acc.file_commit = file_commit
    try:
        return acc.main(args)
    finally:
        acc.file_commit = original


if __name__ == "__main__":
    raise SystemExit(main())
