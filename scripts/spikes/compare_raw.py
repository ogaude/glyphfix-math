"""Byte-level comparison of two Ollama sidecars (raw, unstripped text).

Usage:  uv run python -m scripts.spikes.compare_raw --a A.sidecar.jsonl --b B.sidecar.jsonl --out cmp.json
The output holds ids and counts only, never LaTeX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.spikes import common


def compare(a_rows: list[dict], b_rows: list[dict]) -> dict:
    a = {r["id"]: r for r in a_rows}
    b = {r["id"]: r for r in b_rows}
    shared = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    differing = [i for i in shared if a[i]["raw_text"].encode() != b[i]["raw_text"].encode()]
    count_mismatch = [i for i in shared if a[i].get("prompt_eval_count") != b[i].get("prompt_eval_count")]
    return {
        "n": len(shared),
        "n_identical": len(shared) - len(differing),
        "differing_ids": differing,
        "prompt_count_mismatch_ids": count_mismatch,
        "missing_ids": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    res = compare(common.read_jsonl(args.a), common.read_jsonl(args.b))
    common.write_json(args.out, res)
    print(f"identical {res['n_identical']}/{res['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
