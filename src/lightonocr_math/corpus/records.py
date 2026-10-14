"""Corpus record schema, content-addressed ids and the closed set of exclusion reasons.

Usage:
  from lightonocr_math.corpus.records import record_id, Exclusion, exclusion_row, side_of, side_root
  rid = record_id("2301.01234v2", "v2", canonical_latex)   # 16 hex chars
  exclusion_row("2301.01234", Exclusion.MACROS, line=12)

Rules:
  * the id is the first 16 hex of sha256("<base arxiv id>|<version>|<canonical latex>"); file and
    line are provenance and never part of the id, so a re-extraction gives the same id
  * exclusion reasons are the Exclusion enum, never free text
  * benchmark papers go to side "benchmark" (data/benchmark/), all others to side "corpus" (data/corpus/)

Known limit: validate_record checks shape and id consistency only, not that the image matches the label.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from lightonocr_math import paths, splits
from lightonocr_math.ids import base_id

BUILDER_VERSION = "1"


class Exclusion(StrEnum):
    MACROS = "macros"
    ENGINE = "engine"
    RENDER = "render"
    DUPLICATE = "duplicate"
    NO_SOURCE = "no_source"
    NO_MAIN = "no_main"
    BAD_ARCHIVE = "bad_archive"
    SHORT_LONG = "short_long"
    CANON_MISMATCH = "canon_mismatch"
    UNSAFE = "unsafe"
    LICENSE = "license"
    VERSION_DRIFT = "version_drift"
    CAP = "cap"
    QUOTA = "quota"
    LEAK = "leak"


EXCLUSION_REASONS = tuple(e.value for e in Exclusion)

RECORD_KEYS = (
    "id", "arxiv_id", "paper_version", "license_uri", "license_spdx", "split", "env", "source_file", "source_line",
    "raw_body", "latex", "flags", "image", "width", "height", "bitmap_sha256", "image_sha256", "builder_version",
    "canon_version", "preamble_sha256", "render_image_digest", "engine_flagged",
)

LICENSE_SPDX = frozenset({"CC-BY-4.0", "CC-BY-3.0", "CC0-1.0"})


def record_id(arxiv_id: str, version: str, canonical_latex: str) -> str:
    return hashlib.sha256(f"{base_id(arxiv_id)}|{version}|{canonical_latex}".encode()).hexdigest()[:16]


def side_of(arxiv_id: str) -> str:
    return "benchmark" if splits.is_benchmark(arxiv_id) else "corpus"


def side_root(side: str) -> Path:
    if side not in ("benchmark", "corpus"):
        raise ValueError(f"side must be 'benchmark' or 'corpus', got {side!r}")
    return paths.repo_root() / "data" / ("benchmark" if side == "benchmark" else "corpus")


def exclusion_row(arxiv_id: str, reason: str, **detail: object) -> dict:
    if reason not in EXCLUSION_REASONS:
        raise ValueError(f"unknown exclusion reason {reason!r}; allowed: {EXCLUSION_REASONS}")
    return {"arxiv_id": arxiv_id, "reason": str(reason), **detail}


def validate_record(rec: dict) -> list[str]:
    problems: list[str] = []
    arxiv_id = rec.get("arxiv_id")
    latex = rec.get("latex")
    if not arxiv_id:
        problems.append("missing arxiv_id")
    if not latex:
        problems.append("empty latex")
    if arxiv_id and latex:
        try:
            expected = record_id(arxiv_id, rec.get("paper_version", ""), latex)
        except ValueError:
            problems.append("invalid arxiv_id")
        else:
            if rec.get("id") != expected:
                problems.append(f"id {rec.get('id')!r} is not record_id(arxiv_id, paper_version, latex) = {expected!r}")
    if rec.get("license_spdx") not in LICENSE_SPDX:
        problems.append(f"license_spdx {rec.get('license_spdx')!r} not in {sorted(LICENSE_SPDX)}")
    return problems


def to_score_row(rec: dict) -> dict:
    return {"id": rec["id"], "paper_id": base_id(rec["arxiv_id"]), "gt_latex": rec["latex"]}
