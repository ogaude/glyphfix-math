"""Notation-stratified, per-paper-capped selection of corpus and benchmark records.

Usage:
  from glyphfix_math.corpus.select import SelectionConfig, paper_order, select_side, floor_report
  order = paper_order(ids, seed)
  sel = select_side(records_by_paper, order, split="train", quota=800, floors={"prime": 800}, cap=30, seed=0)

Rules:
  * papers are taken in the seeded order (sha256 of "<seed>|<base id>", the same rule as fetch.fetch_order);
    papers with no record of the requested split are skipped and do not count toward the quota; usable papers
    beyond the quota get one "quota" exclusion row each
  * pass 1 (floors): flags are served from the rarest in the pool to the most common, round-robin over the selected
    papers, one record per turn, never beyond the per-paper cap
  * pass 2 (fill): each paper is filled up to the cap with plain records first, then with the remaining flagged ones,
    both in seeded hash order
  * every record not selected becomes a "cap" exclusion row carrying its record id; nothing is deleted from shards
  * the cap applies identically to the corpus and the benchmark (D-05)

Known limit: the floor pass is greedy per flag; a record carrying several flags counts toward each of them, but the
pass does not search for a minimal set, so totals may exceed the floors.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from glyphfix_math.corpus.records import Exclusion, exclusion_row
from glyphfix_math.ids import base_id

SELECT_VERSION = "1"
# planning notes stored next to the settings in SELECTION_CONFIG.json; they never change a selection
ANNOTATION_KEYS = frozenset({"evidence", "fetch_counts", "usable_rate"})


def _default_quotas() -> dict[str, int]:
    return {"benchmark": 150, "dev": 50, "train": 800}


def _default_floors() -> dict[str, dict[str, int]]:
    return {
        "corpus": {"prime": 800, "greek": 800, "decoration": 800, "i_1": 800},
        "benchmark": {"prime": 150, "greek": 150, "decoration": 150, "i_1": 150},
    }


def _default_min_papers() -> dict[str, int]:
    return {"benchmark": 40, "corpus": 0}


@dataclass
class SelectionConfig:
    seed: int = 0
    cap: int = 30
    quotas: dict[str, int] = field(default_factory=_default_quotas)
    floors: dict[str, dict[str, int]] = field(default_factory=_default_floors)
    min_papers_per_flag: dict[str, int] = field(default_factory=_default_min_papers)

    @classmethod
    def from_json(cls, path: Path | str) -> SelectionConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(data) - set(cls.__dataclass_fields__) - ANNOTATION_KEYS)
        if unknown:
            raise ValueError(f"unknown key(s) in selection config {path}: {unknown}; allowed: {sorted(cls.__dataclass_fields__)}")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class Selection:
    selected: list[dict]
    dropped: list[dict]
    papers: list[str]
    stats: dict


def _hash_key(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def paper_order(ids: list[str], seed: int) -> list[str]:
    bases = sorted({base_id(i) for i in ids})
    return sorted(bases, key=lambda b: _hash_key(seed, b))


def select_side(
    records_by_paper: dict[str, list[dict]],
    order: list[str],
    *,
    split: str,
    quota: int,
    floors: dict[str, int],
    cap: int,
    seed: int,
) -> Selection:
    pools: dict[str, list[dict]] = {}
    for paper in order:
        recs = [r for r in records_by_paper.get(paper, []) if r.get("split") == split]
        if recs:
            pools[paper] = sorted(recs, key=lambda r: _hash_key(seed, r["id"]))
    usable = list(pools)
    papers = usable[:quota]
    dropped: list[dict] = [exclusion_row(p, Exclusion.QUOTA) for p in usable[quota:]]

    chosen: dict[str, list[dict]] = {p: [] for p in papers}
    taken: set[str] = set()

    prevalence = Counter(f for p in papers for r in pools[p] for f in set(r.get("flags", [])))
    for flag in sorted(floors, key=lambda f: (prevalence.get(f, 0), f)):
        need = floors[flag]
        have = sum(1 for p in papers for r in chosen[p] if flag in r.get("flags", []))
        progress = True
        while have < need and progress:
            progress = False
            for p in papers:
                if have >= need:
                    break
                if len(chosen[p]) >= cap:
                    continue
                for r in pools[p]:
                    if r["id"] not in taken and flag in r.get("flags", []):
                        chosen[p].append(r)
                        taken.add(r["id"])
                        have += 1
                        progress = True
                        break

    for p in papers:
        rest = [r for r in pools[p] if r["id"] not in taken]
        plain = [r for r in rest if "plain" in r.get("flags", []) or not r.get("flags")]
        other = [r for r in rest if r not in plain]
        for r in plain + other:
            if len(chosen[p]) >= cap:
                break
            chosen[p].append(r)
            taken.add(r["id"])

    selected: list[dict] = []
    for p in papers:
        selected.extend(chosen[p])
        for r in pools[p]:
            if r["id"] not in taken:
                dropped.append(exclusion_row(p, Exclusion.CAP, id=r["id"]))

    per_flag = Counter(f for r in selected for f in set(r.get("flags", [])))
    hist = Counter(len(chosen[p]) for p in papers)
    stats = {
        "split": split,
        "n_papers": len(papers),
        "n_records": len(selected),
        "per_flag": dict(sorted(per_flag.items())),
        "per_paper_histogram": {str(k): v for k, v in sorted(hist.items())},
        "usable_papers": len(usable),
        "quota": quota,
        "quota_shortfall": max(0, quota - len(usable)),
        "cap": cap,
    }
    return Selection(selected=selected, dropped=dropped, papers=papers, stats=stats)


def floor_report(selected: list[dict], floors: dict[str, int], min_papers: int = 0) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for flag, need in floors.items():
        have = sum(1 for r in selected if flag in r.get("flags", []))
        papers = len({r["arxiv_id"] for r in selected if flag in r.get("flags", [])})
        out[flag] = {
            "have": have,
            "need": need,
            "papers": papers,
            "min_papers": min_papers,
            "short": have < need or papers < min_papers,
        }
    return out
