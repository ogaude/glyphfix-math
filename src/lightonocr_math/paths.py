"""Repository path helpers.

Usage:
  from lightonocr_math.paths import repo_root, splits_dir, cache_dir

repo_root() honours LIGHTONOCR_MATH_ROOT, else walks up from this file to the first
directory whose pyproject.toml declares name = "lightonocr-math".
cache_dir() honours LIGHTONOCR_MATH_CACHE, else uses $XDG_CACHE_HOME (or ~/.cache)/lightonocr_math.

Known limit: the walk-up lookup needs an editable (source tree) install; set
LIGHTONOCR_MATH_ROOT when the package is installed elsewhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_NAME_RE = re.compile(r'^\s*name\s*=\s*"lightonocr-math"\s*$', re.MULTILINE)


def repo_root() -> Path:
    env = os.environ.get("LIGHTONOCR_MATH_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        pp = parent / "pyproject.toml"
        if pp.is_file() and _NAME_RE.search(pp.read_text(encoding="utf-8")):
            return parent
    raise RuntimeError("repo root not found: set LIGHTONOCR_MATH_ROOT")


def splits_dir() -> Path:
    return repo_root() / "data" / "splits"


def cache_dir() -> Path:
    env = os.environ.get("LIGHTONOCR_MATH_CACHE")
    if env:
        p = Path(env)
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        p = Path(base) / "lightonocr_math"
    p.mkdir(parents=True, exist_ok=True)
    return p
