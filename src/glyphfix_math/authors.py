"""Author normalization and quarantine computation.

Usage:
  normalize_author("José  Müller-Lüdenscheidt") -> "jose muller ludenscheidt"
  quarantine(candidates, benchmark_ids) -> (quarantined_ids, stats)

Rules (recorded verbatim in SPLIT_RULE.json through AUTHOR_RULE):
  * fold: Unicode NFKD, combining marks dropped, ascii only
  * case: lower
  * punctuation: periods, commas, apostrophes and hyphens become spaces, other punctuation is
    removed, whitespace is collapsed
  * a name needs at least 2 tokens, and a name whose tokens before the last are all single
    letters ("J. Smith") is dropped: initial-only names are too ambiguous to match on
  * middle initials are kept ("john r smith")
  * papers with more than max_authors authors (default 50) are skipped on both sides: large
    collaborations would otherwise quarantine a large part of the pool

A non-benchmark candidate is quarantined when any of its normalized names also belongs to a
benchmark paper. Benchmark papers are never in the quarantine set.

Known limit: matching on the full normalized name is a deliberate "slightly leaky but honest"
choice. Surname plus initial matching quarantined 55% of the candidate pool in research, which
would leave too little training data. Two different people with the same full name are
quarantined together (over-quarantine), and one person publishing under spelling variants can
slip through.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from glyphfix_math.ids import base_id

AUTHOR_RULE: dict = {
    "version": "1",
    "fold": "NFKD ascii",
    "case": "lower",
    "punctuation": "removed, hyphen->space",
    "min_tokens": 2,
    "initial_only": "dropped",
    "max_authors_per_paper": 50,
}

_SEPARATORS_RE = re.compile(r"[-'’.,]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def normalize_author(name: str) -> str | None:
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.encode("ascii", "ignore").decode("ascii").lower()
    folded = _SEPARATORS_RE.sub(" ", folded)
    folded = _NON_ALNUM_RE.sub("", folded)
    tokens = folded.split()
    if len(tokens) < AUTHOR_RULE["min_tokens"]:
        return None
    if all(len(t) == 1 for t in tokens[:-1]):
        return None
    return " ".join(tokens)


def author_keys(authors: list[str]) -> set[str]:
    keys = set()
    for a in authors:
        k = normalize_author(a)
        if k is not None:
            keys.add(k)
    return keys


def _safe_base(arxiv_id: str) -> str | None:
    try:
        return base_id(arxiv_id)
    except ValueError:
        return None


def quarantine(
    candidates: Iterable[dict], benchmark: set[str], *, max_authors: int = 50
) -> tuple[set[str], dict]:
    rows = []
    for c in candidates:
        b = _safe_base(c.get("arxiv_id", ""))
        if b is not None:
            rows.append((b, list(c.get("authors") or [])))

    index: dict[str, set[str]] = {}
    skipped_large = 0
    for b, authors in rows:
        if len(authors) > max_authors:
            skipped_large += 1
            continue
        if b in benchmark:
            for k in author_keys(authors):
                index.setdefault(k, set()).add(b)

    quarantined: set[str] = set()
    caused: Counter[str] = Counter()
    for b, authors in rows:
        if b in benchmark or len(authors) > max_authors:
            continue
        hits = author_keys(authors) & index.keys()
        if hits:
            quarantined.add(b)
            for k in hits:
                caused[k] += 1

    stats = {
        "quarantined": len(quarantined),
        "benchmark_author_names": len(index),
        "skipped_large_author_papers": skipped_large,
        "top_names": [
            {"name": n, "quarantined_papers": c}
            for n, c in sorted(caused.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        ],
    }
    return quarantined, stats
