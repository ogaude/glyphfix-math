"""Public-history allow-list: classify a repo path as history, latest-only, overlay or dropped.

Usage:
  python -m scripts.release.public_paths report
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SECTIONS = ("history", "latest-only", "overlay")
DEFAULT_SPEC = Path(__file__).with_name("public_paths.txt")


@dataclass(frozen=True)
class Rule:
    section: str
    negated: bool
    pattern: str
    regex: re.Pattern


def _glob_to_regex(pat: str) -> re.Pattern:
    out: list[str] = []
    i = 0
    while i < len(pat):
        if pat.startswith("/**", i) and i + 3 == len(pat):
            out.append("/.*")
            i += 3
        elif pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def load_spec(path: Path | None = None) -> list[Rule]:
    path = Path(path) if path else DEFAULT_SPEC
    rules: list[Rule] = []
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section not in SECTIONS:
                raise ValueError(f"unknown section {line}")
            continue
        if section is None:
            raise ValueError(f"rule before any section: {line}")
        neg = line.startswith("!")
        pat = line[1:] if neg else line
        rules.append(Rule(section, neg, pat, _glob_to_regex(pat)))
    return rules


def classify(path: str, spec: list[Rule] | None = None) -> str | None:
    if spec is None:
        spec = _default_spec()
    result: str | None = None
    for r in spec:
        if r.regex.match(path):
            result = None if r.negated else r.section
    return result


_CACHE: list[Rule] | None = None


def _default_spec() -> list[Rule]:
    global _CACHE
    if _CACHE is None:
        _CACHE = load_spec()
    return _CACHE


def _git_lines(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    return [x for x in out.splitlines() if x]


def tracked_paths() -> list[str]:
    return _git_lines("ls-files")


def historical_paths() -> list[str]:
    return sorted(set(_git_lines("log", "--all", "--name-only", "--format=")))


def report() -> int:
    spec = load_spec()
    for title, paths in (("tracked", tracked_paths()), ("historical", historical_paths())):
        counts = Counter(classify(p, spec) or "dropped" for p in paths)
        print(f"{title}: {len(paths)} paths; " + ", ".join(f"{k}={counts[k]}" for k in (*SECTIONS, "dropped")))
    print("dropped tracked paths:")
    for p in tracked_paths():
        if classify(p, spec) is None:
            print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="public_paths")
    ap.add_argument("cmd", choices=["report"])
    ap.parse_args(argv)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
