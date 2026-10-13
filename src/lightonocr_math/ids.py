"""arXiv id normalization: the only place ids are parsed or turned into file names.

Usage:
  base_id("2301.01234v2")      -> "2301.01234"
  safe_id("math/0508005v1")    -> "math_0508005"
  from_safe_id("math_0508005") -> "math/0508005"

The strict regex rejects "..", spaces and extra slashes, so safe_id output is safe
to use as a single path component.

Known limit: from_safe_id relies on new-style ids starting with a digit and old-style
ids starting with a letter.
"""

from __future__ import annotations

import re

ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z][a-z\-]*(\.[A-Z]{2})?/\d{7})(v\d+)?$")
_VERSION_RE = re.compile(r"v\d+$")


def is_arxiv_id(s: str) -> bool:
    m = ARXIV_ID_RE.match(s)
    if not m:
        return False
    head = m.group(1)
    if "/" not in head:
        yymm, num = head.split(".")
        # 4-digit numbers exist only up to 1412; from 1501 on arXiv uses 5 digits
        if len(num) == 4 and yymm > "1412":
            return False
        if len(num) == 5 and yymm < "1501":
            return False
    return True


def base_id(arxiv_id: str) -> str:
    s = arxiv_id.strip()
    if not is_arxiv_id(s):
        raise ValueError(f"not an arXiv id: {arxiv_id!r}")
    return _VERSION_RE.sub("", s)


def safe_id(arxiv_id: str) -> str:
    return base_id(arxiv_id).replace("/", "_")


def from_safe_id(s: str) -> str:
    if s and not s[0].isdigit():
        return s.replace("_", "/", 1)
    return s
