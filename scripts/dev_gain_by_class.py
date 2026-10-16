"""Break the dev gain of a tuned run down by error class from frozen per-record scores (CPU only).

Usage (repo root):
  .venv-lighton/bin/python -m scripts.dev_gain_by_class --out .planning/phases/04-error-mining-and-lora-training/evidence/dev_gain_by_class.json

Reads the frozen scorer outputs (records.jsonl with render_identical) of the stock and tuned arms; the scorer
is not run or changed. Each crop is labelled with the mining class of the STOCK arm's error (same rules and the
same accepted symbol_swap patterns as data/mining/labels.jsonl, rarity taken from those labels); a crop the
stock arm got right is labelled stock_right (its class, if the tuned arm broke it, is the tuned error's class).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from lightonocr_math import mining

STOCK_HF = "data/train/dev_eval/base.score/records.jsonl"
TUNED_HF = "data/train/runs/p4-full-1/dev_eval/step-875.score/records.jsonl"
STOCK_OL = "data/runs/p4-full-1/dev_stock.score/records.jsonl"
TUNED_OL = "data/runs/p4-full-1/dev_cand.score/records.jsonl"
TAXONOMY = "data/mining/taxonomy.json"
CANDIDATES = "data/mining/taxonomy_candidates.json"
LABELS = "data/mining/labels.jsonl"


def _read(p: str) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def setup():
    tax = json.loads(Path(TAXONOMY).read_text())
    accepted = {c["name"]: c["patterns"] for c in tax["accepted_clusters"]}
    cand = {c["id"]: c for c in json.loads(Path(CANDIDATES).read_text())["candidates"]}
    cluster_of = {}
    for c in tax["accepted_clusters"]:
        for cid, pat in zip(c["candidate_ids"], c["patterns"]):
            cluster_of[(tuple(pat[0]), tuple(pat[1]))] = (cid, " ".join(cand[cid]["gt_tokens"]) + " -> " + " ".join(cand[cid]["pred_tokens"]))
    labels = _read(LABELS)
    rarity = collections.Counter()
    for r in labels:
        if r["error"]:
            rarity.update(c for c in mining.GLYPH_CLASSES if c in r["classes"])
    return accepted, dict(rarity), cluster_of


def label(row, accepted, rarity, cluster_of):
    cls = mining.assign_mining_class(row, accepted, rarity)
    clu = None
    if cls == "symbol_swap":
        pairs = mining.row_pairs(row)
        hits = sorted(v[0] + " (" + v[1] + ")" for k, v in cluster_of.items() if k in pairs)
        clu = hits[0] if hits else "unresolved"
    return cls, clu


def breakdown(stock: list[dict], tuned: list[dict], accepted, rarity, cluster_of) -> dict:
    s = {r["id"]: r for r in stock}
    t = {r["id"]: r for r in tuned}
    ids = sorted(set(s) & set(t))
    by_class: dict = collections.defaultdict(lambda: collections.Counter())
    by_cluster: dict = collections.defaultdict(lambda: collections.Counter())
    r2w_class: collections.Counter = collections.Counter()
    ids_by = collections.defaultdict(list)
    excluded = 0
    for i in ids:
        a, b = s[i].get("render_identical"), t[i].get("render_identical")
        if a is None or b is None:
            excluded += 1
            continue
        if a:
            key = "both_right" if b else "right_to_wrong"
            cls = "stock_right"
            if not b:
                tc, _ = label(t[i], accepted, rarity, cluster_of)
                r2w_class[tc] += 1
                cls = "stock_right"
            by_class[cls][key] += 1
            ids_by[(cls, key)].append(i)
            continue
        cls, clu = label(s[i], accepted, rarity, cluster_of)
        key = "wrong_to_right" if b else "both_wrong"
        by_class[cls][key] += 1
        ids_by[(cls, key)].append(i)
        if clu:
            by_cluster[clu][key] += 1
    pair_fix: collections.Counter = collections.Counter()
    pair_all: collections.Counter = collections.Counter()
    sp = collections.Counter()
    for i in ids:
        if s[i].get("render_identical") is False and t[i].get("render_identical") is not None:
            fixed = bool(t[i]["render_identical"])
            tag = "spacing_only" if mining.spacing_only(s[i]) else ("exact_norm_other" if s[i].get("exact") else "token_diff")
            sp[(tag, "fixed" if fixed else "still_wrong")] += 1
            for pr in mining.row_pairs(s[i]):
                k = " ".join(pr[0]) + " -> " + " ".join(pr[1])
                pair_all[k] += 1
                if fixed:
                    pair_fix[k] += 1
    def fin(d):
        return {k: {"n": sum(v.values()), "wrong_to_right": v["wrong_to_right"], "right_to_wrong": v["right_to_wrong"],
                    "both_wrong": v["both_wrong"], "both_right": v["both_right"]} for k, v in sorted(d.items())}
    tot = collections.Counter()
    for v in by_class.values():
        tot.update(v)
    return {
        "n_common": len(ids), "excluded_none": excluded,
        "totals": {k: tot[k] for k in ("wrong_to_right", "right_to_wrong", "both_right", "both_wrong")},
        "stock_right": sum(1 for i in ids if s[i].get("render_identical")),
        "tuned_right": sum(1 for i in ids if t[i].get("render_identical")),
        "by_class_of_stock_error": fin(by_class),
        "symbol_swap_clusters": fin(by_cluster),
        "right_to_wrong_by_tuned_error_class": dict(r2w_class),
        "stock_error_kind": {f"{a}/{b}": n for (a, b), n in sorted(sp.items())},
        "top_stock_pairs_in_fixed_crops": [[k, n, pair_all[k]] for k, n in pair_fix.most_common(15)],
        "right_to_wrong_ids": sorted(ids_by[("stock_right", "right_to_wrong")]),
        "wrong_to_right_ids_by_class": {c: sorted(ids_by[(c, "wrong_to_right")]) for c in sorted(by_class) if ids_by[(c, "wrong_to_right")]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    accepted, rarity, cluster_of = setup()
    out = {
        "run_id": "p4-full-1", "checkpoint": "step-875", "rarity_from_train_labels": rarity,
        "sources": {"hf_stock": STOCK_HF, "hf_tuned": TUNED_HF, "ollama_stock": STOCK_OL, "ollama_tuned": TUNED_OL},
        "hf_mini_136": breakdown(_read(STOCK_HF), _read(TUNED_HF), accepted, rarity, cluster_of),
        "ollama_dev_926": breakdown(_read(STOCK_OL), _read(TUNED_OL), accepted, rarity, cluster_of),
    }
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for k in ("hf_mini_136", "ollama_dev_926"):
        print(k, out[k]["n_common"], out[k]["totals"], out[k]["stock_right"], out[k]["tuned_right"])
        for c, v in out[k]["by_class_of_stock_error"].items():
            print("  ", c, v)
        for c, v in out[k]["symbol_swap_clusters"].items():
            print("   cl", c, v)
        print("   r2w by tuned class", out[k]["right_to_wrong_by_tuned_error_class"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
