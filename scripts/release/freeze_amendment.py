"""One-time, recorded amendment of the frozen acceptance CLI (path constants only).

The frozen scorer scripts/acceptance.py hardcodes where it reads and writes the
acceptance evidence. Moving that evidence out of .planning/ needs the two path
literals changed. This tool proves the edit touches nothing else, builds the
amended lock and registry, compares a fresh rescore with the published results
and re-verifies the whole record from git objects.

  uv run python -m scripts.release.freeze_amendment preview [--out DIR]
  uv run python -m scripts.release.freeze_amendment verify [--history PATH]
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path

from scripts.acceptance import _comparable, _diff
from scripts.spikes.common import write_json

FROZEN_CLI = "scripts/acceptance.py"
FROZEN_MODULE = "src/lightonocr_math/acceptance.py"
NEW_EVIDENCE_DIR = Path("evidence/acceptance")
AMENDMENT_PATH = NEW_EVIDENCE_DIR / "FREEZE_AMENDMENT.json"
LOCK_PATH = Path("data/benchmark/acceptance/RUN_LOCK.json")
REGISTRY_PATH = Path("data/models/REGISTRY.json")
REGISTRY_TAG = "mathocr-lora-q8:p4-full-1"
OLD_EVIDENCE = ".planning/phases/05-acceptance-evaluation/evidence"
OLD_REPORTS = ".planning/phases/05-acceptance-evaluation/reports"

REPLACEMENTS = (
    ("EVIDENCE_DIR", ".planning/phases/05-acceptance-evaluation/evidence", "evidence/acceptance"),
    ("REPORT_PATH", ".planning/phases/05-acceptance-evaluation/reports/ACCEPTANCE.md", "evidence/acceptance/ACCEPTANCE.md"),
)
MOVES = (
    (f"{OLD_EVIDENCE}/results.json", "evidence/acceptance/results.json"),
    (f"{OLD_EVIDENCE}/flips.jsonl", "evidence/acceptance/flips.jsonl"),
    (f"{OLD_EVIDENCE}/env_check.json", "evidence/acceptance/env_check.json"),
    (f"{OLD_EVIDENCE}/sensitivity.json", "evidence/acceptance/sensitivity.json"),
    (f"{OLD_REPORTS}/ACCEPTANCE.md", "evidence/acceptance/ACCEPTANCE.md"),
)
OLD_CLI_BLOB = "9eb9e4bb6fbc015eb402246f794ab7f36434e29b"
MODULE_BLOB = "c7cbaf36158a9e091e1f9f80ca1da81ad4e2a3ac"
OLD_CLI_COMMIT = ("83e897688a87fc2848b8ebd772ca595677500a29", "2026-10-17T10:09:32+02:00")
PUBLISHED_RESULTS_SHA256 = "f506e7b408338858f74e6d6a786d85d5faa475f35631ab220e0459f2435ecfa3"
AMENDED_KEYS = (
    "code_freeze.commits.scripts/acceptance.py",
    "code_freeze.current_blobs.cli",
    "code_freeze.lock_blobs.cli",
)
HISTORY_KEY = "code_freeze.commits.src/lightonocr_math/acceptance.py"
AMENDMENT_REF = "evidence/acceptance/FREEZE_AMENDMENT.json"
ARMS = ("tuned-crops", "tuned-pages")


class AmendmentError(Exception):
    pass


# ---------------------------------------------------------------- git helpers

def _git(*args: str, root: Path = Path("."), text: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=text, check=False)
    if r.returncode != 0:
        err = r.stderr if text else r.stderr.decode("utf-8", "replace")
        raise AmendmentError(f"git {' '.join(args)} failed: {err.strip()}")
    return r.stdout


def _blob_of(path: Path, root: Path = Path(".")) -> str:
    return _git("hash-object", "--no-filters", str(path), root=root).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- proof

def _assign_literals(tree: ast.Module, names: set[str]) -> dict[str, ast.Constant]:
    """Constants that are the sole argument of Path(...) in a module-level `NAME = Path(...)`."""
    out: dict[str, ast.Constant] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        v = node.value
        if (
            name in names
            and isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id == "Path"
            and len(v.args) == 1
            and not v.keywords
            and isinstance(v.args[0], ast.Constant)
            and isinstance(v.args[0].value, str)
        ):
            out[name] = v.args[0]
    return out


def prove(old_text: str, new_text: str, replacements=REPLACEMENTS) -> dict:
    """Prove new_text is old_text with exactly the given path literals replaced."""
    expected = old_text
    for name, old, new in replacements:
        quoted = f'"{old}"'
        if expected.count(quoted) != 1:
            raise AmendmentError(f"{name}: old literal {quoted} occurs {expected.count(quoted)} times, expected exactly once")
        expected = expected.replace(quoted, f'"{new}"')
    if new_text != expected:
        raise AmendmentError("byte substitution: new text differs from old text with only the literals replaced")

    old_lines, new_lines = old_text.splitlines(keepends=True), new_text.splitlines(keepends=True)
    if len(old_lines) != len(new_lines):
        raise AmendmentError("line count differs")
    changed = [i + 1 for i, (a, b) in enumerate(zip(old_lines, new_lines, strict=True)) if a != b]
    if len(changed) != len(replacements):
        raise AmendmentError(f"{len(changed)} lines changed, expected {len(replacements)}")

    ot = list(tokenize.generate_tokens(io.StringIO(old_text).readline))
    nt = list(tokenize.generate_tokens(io.StringIO(new_text).readline))
    if len(ot) != len(nt):
        raise AmendmentError("token count differs")
    wanted = {f'"{o}"': f'"{n}"' for _, o, n in replacements}
    changed_tokens = 0
    for a, b in zip(ot, nt, strict=True):
        if a.type != b.type or a.start[0] != b.start[0]:
            raise AmendmentError(f"token type or line differs at line {a.start[0]}")
        if a.string != b.string:
            if a.type != tokenize.STRING or wanted.get(a.string) != b.string:
                raise AmendmentError(f"unexpected token change at line {a.start[0]}: {a.string!r} -> {b.string!r}")
            changed_tokens += 1
    if changed_tokens != len(replacements):
        raise AmendmentError(f"{changed_tokens} string tokens changed, expected {len(replacements)}")

    names = {n for n, _, _ in replacements}
    otree, ntree = ast.parse(old_text), ast.parse(new_text)
    olit, nlit = _assign_literals(otree, names), _assign_literals(ntree, names)
    for name, old, new in replacements:
        if name not in olit or name not in nlit:
            raise AmendmentError(f"{name}: not a module-level `{name} = Path(<str>)` assignment")
        if olit[name].value != old or nlit[name].value != new:
            raise AmendmentError(f"{name}: assigned literal is not the recorded old/new value")
        olit[name].value = new
    if ast.dump(otree, include_attributes=False) != ast.dump(ntree, include_attributes=False):
        raise AmendmentError("AST differs after substituting the two literals")

    return {
        "method": "byte-substitution+line-diff+tokenize+ast",
        "replacements": [{"name": n, "old": o, "new": nw, "line": changed[i]} for i, (n, o, nw) in enumerate(replacements)],
        "changed_lines": changed,
        "changed_tokens": changed_tokens,
        "ast_equal_after_substitution": True,
        "python": sys.version,
    }


# ------------------------------------------------------------------ lock edit

def amended_lock(lock: dict, *, old_blob: str, new_blob: str, module_blob: str = MODULE_BLOB) -> dict:
    out = copy.deepcopy(lock)
    arms = out.get("arms") or {}
    for arm in ARMS:
        if arm not in arms:
            raise AmendmentError(f"RUN_LOCK has no arm {arm}")
        a = arms[arm]
        if a.get("acceptance_cli_blob") != old_blob:
            raise AmendmentError(f"{arm}: acceptance_cli_blob {a.get('acceptance_cli_blob')} is not {old_blob}")
        if a.get("acceptance_module_blob") != module_blob:
            raise AmendmentError(f"{arm}: acceptance_module_blob {a.get('acceptance_module_blob')} is not {module_blob}")
        a["acceptance_cli_blob"] = new_blob
        a["acceptance_cli_blob_generated"] = old_blob
        a["freeze_amendment"] = AMENDMENT_REF
    return out


# ----------------------------------------------------------------- comparison

def compare_published(published: dict, fresh: dict, *, amendment: dict | None = None, commit_map: dict | None = None) -> dict:
    cmap = commit_map or {}
    diff = _diff(_comparable(published, published.get("scores_dir")), _comparable(fresh, fresh.get("scores_dir")))
    pcf, fcf = published.get("code_freeze") or {}, fresh.get("code_freeze") or {}
    allowed: dict[str, str] = {}
    history_mapped: dict[str, dict] = {}
    problems: list[str] = []

    def commit(cf: dict, path: str):
        v = (cf.get("commits") or {}).get(path)
        return list(v) if isinstance(v, list | tuple) and len(v) == 2 else None

    for key in diff:
        if key == HISTORY_KEY:
            p, f = commit(pcf, FROZEN_MODULE), commit(fcf, FROZEN_MODULE)
            if p and f and p[1] == f[1] and cmap.get(p[0]) == f[0]:
                allowed[key] = "history"
                history_mapped[FROZEN_MODULE] = {"source_commit": p[0], "public_commit": f[0], "committer_date": f[1]}
            else:
                problems.append(f"{key}: not explained by the commit map with an equal committer date")
        elif key == AMENDED_KEYS[0]:
            p, f = commit(pcf, FROZEN_CLI), commit(fcf, FROZEN_CLI)
            am = amendment or {}
            ac, oc = am.get("amendment_commit"), am.get("old_commit")
            ok = bool(am and p and f and ac and oc and p == list(oc) and f[1] == ac[1] and f[0] in (ac[0], cmap.get(ac[0])))
            if ok:
                allowed[key] = "amendment"
                if f[0] != ac[0]:
                    history_mapped[FROZEN_CLI] = {"source_commit": ac[0], "public_commit": f[0], "committer_date": f[1]}
            else:
                problems.append(f"{key}: not the predicted amendment commit")
        elif key in AMENDED_KEYS[1:]:
            part = key.split(".")[1]
            am = amendment or {}
            p = (pcf.get(part) or {}).get("cli")
            f = (fcf.get(part) or {}).get("cli")
            if am and p == am.get("old_blob") and f == am.get("new_blob"):
                allowed[key] = "amendment"
            else:
                problems.append(f"{key}: published {p!r} / fresh {f!r} is not the predicted old/new blob")
        else:
            problems.append(f"{key}: differs")

    kinds = set(allowed.values())
    if problems:
        compare = "different"
    elif not diff:
        compare = "identical"
    elif kinds == {"history"}:
        compare = "identical_modulo_history"
    elif kinds == {"amendment"}:
        compare = "identical_modulo_amendment"
    else:
        compare = "identical_modulo_history_and_amendment"
    return {"compare": compare, "differing_keys": diff, "allowed": allowed, "history_mapped": history_mapped, "problems": problems}


def load_commit_map(root: Path = Path(".")) -> dict:
    p = Path(root) / "RELEASE_HISTORY.json"
    if not p.is_file():
        return {}
    return dict(json.loads(p.read_text(encoding="utf-8")).get("commit_map") or {})


# --------------------------------------------------------------------- verify

def verify_amendment(root: Path = Path("."), amendment: dict | None = None, *, published: dict | None = None, commit_map: dict | None = None) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    cmap = load_commit_map(root) if commit_map is None else commit_map
    try:
        if amendment is None:
            amendment = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
        if published is None:
            published = json.loads((root / "evidence/acceptance/results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"cannot load amendment or published results: {e}"]

    old_blob, new_blob = amendment.get("old_blob"), amendment.get("new_blob")
    cf = published.get("code_freeze") or {}
    if not (old_blob == OLD_CLI_BLOB == (cf.get("lock_blobs") or {}).get("cli") == (cf.get("current_blobs") or {}).get("cli")):
        problems.append("old_blob does not match the published code_freeze record")
    try:
        if _blob_of(root / FROZEN_CLI, root) != new_blob:
            problems.append("current scripts/acceptance.py blob is not new_blob")
        if _blob_of(root / FROZEN_MODULE, root) != MODULE_BLOB or amendment.get("module_blob") != MODULE_BLOB:
            problems.append("src/lightonocr_math/acceptance.py blob changed")
        old_text = _git("cat-file", "-p", str(old_blob), root=root)
        new_text = (root / FROZEN_CLI).read_text(encoding="utf-8")
        recorded = tuple((r["name"], r["old"], r["new"]) for r in amendment.get("replacements", []))
        if recorded != REPLACEMENTS:
            problems.append("recorded replacements differ from the expected ones")
        prove(old_text, new_text, recorded or REPLACEMENTS)
    except (AmendmentError, OSError, KeyError, SyntaxError) as e:
        problems.append(f"proof: {e}")

    try:
        arms = json.loads((root / LOCK_PATH).read_text(encoding="utf-8")).get("arms", {})
        for arm in ARMS:
            a = arms.get(arm, {})
            if a.get("acceptance_cli_blob") != new_blob or a.get("acceptance_cli_blob_generated") != old_blob:
                problems.append(f"RUN_LOCK {arm}: blob fields inconsistent with the amendment")
    except (OSError, ValueError) as e:
        problems.append(f"RUN_LOCK unreadable: {e}")

    for m in amendment.get("moved", []):
        new_p, old_p = root / m["new"], root / m["old"]
        if old_p.exists():
            problems.append(f"{m['old']} still exists")
        if not new_p.is_file():
            problems.append(f"{m['new']} missing")
            continue
        if _sha256(new_p) != m["sha256"]:
            problems.append(f"{m['new']} sha256 differs from the record")
        try:
            if _blob_of(new_p, root) != m["git_blob"]:
                problems.append(f"{m['new']} git blob differs from the record")
        except AmendmentError as e:
            problems.append(str(e))
    if len(amendment.get("moved", [])) != len(MOVES):
        problems.append("moved list does not hold the five expected files")
    try:
        if _sha256(root / "evidence/acceptance/results.json") != PUBLISHED_RESULTS_SHA256:
            problems.append("results.json sha256 is not the published value")
    except OSError as e:
        problems.append(str(e))

    try:
        h, d = _git("log", "-1", "--format=%H %cI", "--", FROZEN_CLI, root=root).split()
        ac = amendment.get("amendment_commit") or []
        if len(ac) != 2 or d != ac[1] or h not in (ac[0], cmap.get(ac[0])):
            problems.append("last commit touching scripts/acceptance.py is not the recorded amendment commit")
    except (AmendmentError, ValueError) as e:
        problems.append(f"amendment commit lookup: {e}")

    re_ = amendment.get("rerun_equivalence") or {}
    if re_.get("compare") != "identical_modulo_amendment" or re_.get("differing_keys") != list(AMENDED_KEYS):
        problems.append("rerun_equivalence is not identical_modulo_amendment with exactly the three amended keys")
    return problems


# -------------------------------------------------------------------- preview

def _unified_diff(a: Path, b: Path, out: Path) -> None:
    r = subprocess.run(["git", "diff", "--no-index", "--", str(a), str(b)], capture_output=True, text=True, check=False)
    if r.returncode not in (0, 1):
        raise AmendmentError(f"git diff failed: {r.stderr.strip()}")
    out.write_text(r.stdout, encoding="utf-8")


def cmd_preview(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old_text = Path(FROZEN_CLI).read_text(encoding="utf-8")
    if _blob_of(Path(FROZEN_CLI)) != OLD_CLI_BLOB:
        raise AmendmentError("scripts/acceptance.py is not at the pre-amendment blob")
    new_text = old_text
    for _, old, new in REPLACEMENTS:
        new_text = new_text.replace(f'"{old}"', f'"{new}"')
    cli = out / "acceptance.py"
    cli.write_text(new_text, encoding="utf-8")
    proof = prove(old_text, new_text)
    (out / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    new_blob = _blob_of(cli)
    (out / "new_blob.txt").write_text(new_blob + "\n", encoding="utf-8")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    write_json(out / "RUN_LOCK.json", amended_lock(lock, old_blob=OLD_CLI_BLOB, new_blob=new_blob))

    reg_text = REGISTRY_PATH.read_text(encoding="utf-8")
    line = f'"evidence": "{OLD_EVIDENCE}",'
    if reg_text.count(line) != 1:
        raise AmendmentError("REGISTRY.json benchmark.evidence line not found exactly once")
    (out / "REGISTRY.json").write_text(reg_text.replace(line, '"evidence": "evidence/acceptance",'), encoding="utf-8")

    _unified_diff(Path(FROZEN_CLI), cli, out / "acceptance.py.diff")
    _unified_diff(LOCK_PATH, out / "RUN_LOCK.json", out / "RUN_LOCK.json.diff")
    _unified_diff(REGISTRY_PATH, out / "REGISTRY.json", out / "REGISTRY.json.diff")

    lines = ["FREEZE AMENDMENT PREVIEW (nothing tracked has changed)", "", "Moves (git mv, byte-identical):"]
    lines += [f"  {o} -> {n}" for o, n in MOVES]
    lines += [
        "",
        f"scripts/acceptance.py: blob {OLD_CLI_BLOB} -> {new_blob} (lines {proof['changed_lines']} only)",
        f"{LOCK_PATH}: acceptance_cli_blob -> new blob in both arms; added acceptance_cli_blob_generated and freeze_amendment",
        f"{REGISTRY_PATH}: entry {REGISTRY_TAG} benchmark.evidence -> evidence/acceptance",
        "Non-frozen edits: scripts/acceptance_sensitivity.py, scripts/acceptance_report.py, tests/test_acceptance.py, README.md, leakscan allow-list",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"preview written to {out}; new blob {new_blob}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cmap = None
    if args.history:
        cmap = dict(json.loads(Path(args.history).read_text(encoding="utf-8")).get("commit_map") or {})
    problems = verify_amendment(Path("."), commit_map=cmap)
    for p in problems:
        print(f"PROBLEM: {p}")
    if not problems:
        print("freeze amendment verified")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="freeze_amendment", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("preview")
    sp.add_argument("--out", default="data/runs/freeze-amendment/preview")
    sp.set_defaults(fn=cmd_preview)
    sv = sub.add_parser("verify")
    sv.add_argument("--history", default=None)
    sv.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except AmendmentError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
