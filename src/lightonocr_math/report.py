"""Markdown report for one run and comparison (flip table) of two runs.

Usage (from repo root):
  uv run python -m lightonocr_math.report compare A/results.json B/results.json [--out flips.md]

Two runs are comparable only when scorer, normalizer, error-rule, preamble, wrap and split
versions match; otherwise ``compare`` refuses (VersionMismatchError). The record id sets must
be identical. Per metric the flip table counts right->wrong, wrong->right and unchanged
records with the ids, an exact McNemar p-value and a paired paper-level bootstrap delta
(rate_B - rate_A).

Known limit: records whose ground truth does not render are excluded from the render tiers
and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightonocr_math import stats
from lightonocr_math.errors import CLASSES

VERSION_KEYS = (
    "scorer_version",
    "scorer_fingerprint",
    "normalize_rules_version",
    "error_rules_version",
    "preamble_sha256",
    "wrap_version",
    "split_rule_version",
    "split_digest",
    "render_dpi",
)
METRICS = ("exact", "render_identical", "render_tolerant")
METRIC_LABELS = {
    "exact": "normalized exact match",
    "render_identical": "render identical",
    "render_tolerant": "render tolerant (separate tier)",
}


class VersionMismatchError(ValueError):
    pass


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(m: dict) -> str:
    return f"[{_pct(m.get('ci_low'))}, {_pct(m.get('ci_high'))}]"


def render_report(results: dict) -> str:
    out = ["# Evaluation report", "", "## Versions", ""]
    for k in VERSION_KEYS:
        out.append(f"- {k}: {results.get(k)}")
    b = results["bootstrap"]
    out += [
        f"- render image: {(results.get('render') or {}).get('image')}",
        f"- bootstrap: seed {b['seed']}, B {b['B']}, CI {b['ci']} (paper-level cluster bootstrap)",
        f"- predictions_sha256: {results.get('predictions_sha256')}",
        f"- records: {results['n_records']}, ground truth not compilable (excluded from render tiers): {results.get('n_gt_uncompilable', 0)}",
    ]
    counts = results.get("gt_render_status_counts")
    if counts:
        out.append("- ground truth render status: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    out += [
        "",
        "## Metrics",
        "",
        "| metric | rate | 95% CI | n | papers |",
        "|---|---|---|---|---|",
    ]
    for key in METRICS:
        m = results["metrics"].get(key)
        if m:
            out.append(f"| {METRIC_LABELS[key]} | {_pct(m['rate'])} | {_ci(m)} | {int(m['den'])} | {m['n_papers']} |")
    out += ["", "## Per-class error rates", "", "| class | errors | denominator | rate | 95% CI |", "|---|---|---|---|---|"]
    for c in CLASSES:
        pc = results["per_class"][c]
        out.append(f"| {c} | {pc['errors']} | {pc['denom']} | {_pct(pc['rate'])} | {_ci(pc)} |")
    d = results.get("disagreements", {})
    a, e = d.get("string_diff_render_identical", []), d.get("string_equal_render_diff", [])
    out += [
        "",
        "## Disagreements between tiers",
        "",
        f"String differs but render identical (harmless), {len(a)}:",
        "",
        ", ".join(a) if a else "none",
        "",
        f"String equal but render differs (bug flag), {len(e)}:",
        "",
        ", ".join(e) if e else "none",
        "",
    ]
    return "\n".join(out)


def _load(path: Path) -> tuple[dict, dict[str, dict]]:
    path = Path(path)
    res = json.loads(path.read_text(encoding="utf-8"))
    recs = {}
    for line in (path.parent / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    return res, recs


def compare_runs(a_path: Path, b_path: Path, *, B: int = 10000, seed: int = 0) -> dict:
    ra, recs_a = _load(a_path)
    rb, recs_b = _load(b_path)
    diffs = [f"{k}: {ra.get(k)!r} vs {rb.get(k)!r}" for k in VERSION_KEYS if ra.get(k) != rb.get(k)]
    if diffs:
        raise VersionMismatchError("results are not comparable, differing versions: " + "; ".join(diffs))
    ren_a, ren_b = ra.get("render") or {}, rb.get("render") or {}
    if bool(ren_a.get("enabled")) != bool(ren_b.get("enabled")):
        raise VersionMismatchError("results are not comparable: only one run has the render tiers (--no-render on one side)")
    if ren_a.get("image") != ren_b.get("image"):
        raise VersionMismatchError(f"results are not comparable, render image: {ren_a.get('image')!r} vs {ren_b.get('image')!r}")
    if set(recs_a) != set(recs_b):
        raise ValueError("record id sets differ between the two runs")
    gt_diff = sorted(i for i in recs_a if recs_a[i].get("gt_latex") != recs_b[i].get("gt_latex"))
    if gt_diff:
        raise ValueError(f"ground truth differs between the two runs for {len(gt_diff)} id(s), e.g. {gt_diff[:5]}")
    out: dict = {"n": len(recs_a), "metrics": {}, "per_class": {}}
    for key in METRICS:
        if key not in ra["metrics"] or key not in rb["metrics"]:
            continue
        ids = sorted(i for i in recs_a if recs_a[i].get(key) is not None and recs_b[i].get(key) is not None)
        excluded = len(recs_a) - len(ids)
        flips = stats.flip_table({i: bool(recs_a[i][key]) for i in ids}, {i: bool(recs_b[i][key]) for i in ids})
        papers = [recs_a[i]["paper_id"] for i in ids]
        delta = stats.paired_bootstrap_delta(
            papers,
            [float(bool(recs_a[i][key])) for i in ids],
            [float(bool(recs_b[i][key])) for i in ids],
            [1.0] * len(ids),
            B=B,
            seed=seed,
        )
        out["metrics"][key] = {"flips": flips, "delta": delta, "excluded": excluded}
    for c in CLASSES:
        da, db = ra["per_class"][c]["denom"], rb["per_class"][c]["denom"]
        if da != db:
            raise ValueError(f"per-class denominators differ for {c}: {da} vs {db}; the runs are not comparable")
        out["per_class"][c] = {"a": ra["per_class"][c]["errors"], "b": rb["per_class"][c]["errors"], "denom": db}
    return out


def render_comparison(cmp: dict) -> str:
    out = ["# Run comparison (A to B)", "", f"Records: {cmp['n']}", ""]
    for key, m in cmp["metrics"].items():
        f, d = m["flips"], m["delta"]
        out += [
            f"## {METRIC_LABELS[key]}",
            "",
            "| both right | both wrong | right to wrong | wrong to right | McNemar p |",
            "|---|---|---|---|---|",
            f"| {f['both_right']} | {f['both_wrong']} | {f['right_to_wrong']} | {f['wrong_to_right']} | {f['mcnemar_p']:.4g} |",
            "",
            f"Paired delta (B minus A): {_pct(d['delta'])} {_ci(d)}; A {_pct(d['rate_a'])}, B {_pct(d['rate_b'])}; "
            + f"records excluded (ground truth not renderable): {m['excluded']}",
            "",
            f"right to wrong ids: {', '.join(f['right_to_wrong_ids']) or 'none'}",
            "",
            f"wrong to right ids: {', '.join(f['wrong_to_right_ids']) or 'none'}",
            "",
        ]
    out += ["## Per-class errors", "", "| class | errors A | errors B | denominator (B) |", "|---|---|---|---|"]
    for c, v in cmp["per_class"].items():
        out.append(f"| {c} | {v['a']} | {v['b']} | {v['denom']} |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lightonocr_math.report", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="flip table between two results.json files")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--bootstrap", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        cmp = compare_runs(a.a, a.b, B=a.bootstrap, seed=a.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = render_comparison(cmp)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
