"""The single naming step: replace the placeholder tokens in the staged text files under release/.

Usage:
  uv run python -m scripts.release.apply_names --dataset-name N --model-name N --repo-name N \
      --hf-namespace NS --gh-owner O --contact-email E --copyright-holder H
  uv run python -m scripts.release.apply_names --names-file names.json

The chosen names never enter code or source history: only files inside release/ (gitignored) are rewritten, the
SHA256SUMS the rewrite invalidates are regenerated, and the choice is recorded in release/NAMES.json. Binary payloads
(Parquet, GGUF, safetensors) are never opened.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.release import common as C

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".cff"}
TEXT_NAMES = {"Modelfile", "NOTICE", "LICENSE"}
TOKEN_RE = re.compile(r"@@[A-Z_]+@@")

_REPO_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")


class NamesError(ValueError):
    pass


def validate_names(names: dict) -> dict:
    """Return the cleaned names or raise NamesError listing every problem."""
    problems: list[str] = []
    out: dict[str, str] = {}
    for key in C.PLACEHOLDERS:
        v = names.get(key)
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{key} is missing or empty")
            continue
        v = v.strip()
        out[key] = v
        if "@@" in v:
            problems.append(f"{key} must not contain '@@'")
        if "\n" in v or "\r" in v:
            problems.append(f"{key} must be a single line")
        if C.NAME_FORBIDDEN.search(v):
            problems.append(f"{key} must not contain the LightOn name in any spelling")
    for key in ("dataset_name", "model_name", "repo_name"):
        if key in out and not _REPO_NAME_RE.match(out[key]):
            problems.append(f"{key} must match {_REPO_NAME_RE.pattern} (lowercase letters, digits, hyphens, 3+ chars)")
    if "hf_namespace" in out and not _NAMESPACE_RE.match(out["hf_namespace"]):
        problems.append("hf_namespace is not a valid Hugging Face namespace")
    if "gh_owner" in out and not _OWNER_RE.match(out["gh_owner"]):
        problems.append("gh_owner is not a valid GitHub account name (max 39 characters)")
    if "contact_email" in out:
        mail = out["contact_email"]
        if not _EMAIL_RE.match(mail):
            problems.append("contact_email is not a valid address")
        elif mail in C.CONTACT_FORBIDDEN:
            problems.append("contact_email is the arXiv User-Agent address, which is never the release contact")
    if problems:
        raise NamesError("; ".join(problems))
    return out


def _is_text_file(p: Path) -> bool:
    if p.name == C.SHA256SUMS:
        return False
    if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in TEXT_NAMES:
        return False
    with open(p, "rb") as f:
        return b"\x00" not in f.read(8192)


def _inside(path: Path, base: Path) -> bool:
    path, base = path.resolve(), base.resolve()
    return path == base or base in path.parents


def _collect(roots: list[Path], extra_files: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in C.iter_files(root) if _is_text_file(p))
    files.extend(p for p in extra_files if p.is_file() and _is_text_file(p))
    return files


def apply_names(
    names: dict,
    roots: tuple[Path, ...] | None = None,
    extra_files: tuple[Path, ...] | None = None,
    staging: Path | None = None,
    names_json: Path | None = None,
) -> dict:
    """Replace every placeholder token. Returns {"changed": {path: n_tokens}, "noop": bool}."""
    names = validate_names(names)
    staging = Path(staging) if staging is not None else C.STAGING
    names_json = Path(names_json) if names_json is not None else staging / "NAMES.json"
    roots = tuple(Path(r) for r in (roots if roots is not None else (C.DATASET_DIR, C.MODEL_DIR, C.OVERLAY_DIR)))
    extra = tuple(Path(p) for p in (extra_files if extra_files is not None else (C.PUBLISH_MD,)))

    for p in (*roots, *extra, names_json):
        if not _inside(p, staging):
            raise NamesError(f"refusing to touch {p}: it is outside {staging}")

    files = _collect(list(roots), list(extra))
    texts = {p: p.read_bytes().decode("utf-8") for p in files}
    with_tokens = [p for p, t in texts.items() if TOKEN_RE.search(t)]

    if names_json.is_file():
        old = json.loads(names_json.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in names.items()):
            if not with_tokens:
                return {"changed": {}, "noop": True}
        else:
            baked = [
                k for k in ("dataset_name", "model_name", "repo_name", "hf_namespace", "gh_owner", "contact_email")
                if isinstance(old.get(k), str) and old[k] != names[k] and any(old[k] in t for t in texts.values())
            ]
            if baked:
                raise NamesError(
                    "different names were already applied; re-run the stage scripts to regenerate the templates "
                    f"with tokens, then apply again (still present from the earlier choice: {', '.join(baked)})"
                )

    known = {tok: names[key] for key, tok in C.PLACEHOLDERS.items()}
    changed: dict[str, int] = {}
    for p in with_tokens:
        unknown = sorted({t for t in TOKEN_RE.findall(texts[p]) if t not in known})
        if unknown:
            raise NamesError(f"{p}: unknown placeholder {unknown[0]}")
        new, count = TOKEN_RE.subn(lambda m: known[m.group(0)], texts[p])
        p.write_bytes(new.encode("utf-8"))
        changed[p.as_posix()] = count
        texts[p] = new

    left = [p.as_posix() for p, t in texts.items() if TOKEN_RE.search(t)]
    if left:
        raise NamesError(f"placeholder tokens left in: {', '.join(left)}")

    for root in roots:
        if (root / C.SHA256SUMS).is_file():
            C.write_sha256sums(root)

    record = dict(names)
    record["applied_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    names_json.parent.mkdir(parents=True, exist_ok=True)
    names_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"changed": changed, "noop": False}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names-file", help="JSON object with the seven name keys")
    for key in C.PLACEHOLDERS:
        ap.add_argument("--" + key.replace("_", "-"))
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if a.names_file:
        names = json.loads(Path(a.names_file).read_text(encoding="utf-8"))
    else:
        names = {k: getattr(a, k) for k in C.PLACEHOLDERS}
    try:
        result = apply_names(names)
    except NamesError as e:
        print(f"apply_names: {e}", file=sys.stderr)
        return 1
    if result["noop"]:
        print("apply_names: names already applied, nothing changed")
        return 0
    for path, n in sorted(result["changed"].items()):
        print(f"{n:4d} token(s)  {path}")
    print(f"apply_names: {len(result['changed'])} file(s) changed, recorded in {C.NAMES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
