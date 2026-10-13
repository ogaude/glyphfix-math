"""Versioned scorer: a predictions file in, results.json + records.jsonl + report.md out.

Usage (from repo root):
  uv run python -m lightonocr_math.score PRED.jsonl --out DIR [--no-render] [--bootstrap 10000] [--seed 0] [--cache DIR]

Predictions file (JSONL, one object per line):
  {"id": str, "paper_id": str, "gt_latex": str, "pred_latex": str}
  optional "pred_error": str  (inference failure, scored as an empty prediction)

Tiers reported side by side (D-12): normalized exact match, render identical (co-primary),
render tolerant (separate tier). A prediction that does not compile is a failure in both
metrics. A ground truth that does not compile is excluded from the render metric and counted
in ``n_gt_uncompilable``. Intervals come from a paper-level cluster bootstrap.

Every result file stores the scorer, normalizer, error-rule, render (preamble hash, wrap
version, dpi) and split versions plus the bootstrap seed and B, so two result files can be
checked for comparability (see ``report compare``).

Known limit: paper macros are not loaded, so a ground truth that uses one is excluded from the
render metric and counted; the string tier still scores it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from lightonocr_math import paths, stats
from lightonocr_math.errors import CLASSES, ERROR_RULES_VERSION, classify, notation_present
from lightonocr_math.guard import GuardError
from lightonocr_math.normalize import NORMALIZE_RULES_VERSION, canonical, equivalent
from lightonocr_math.render import RenderInfrastructureError

SCORER_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "1"
SCORER_FINGERPRINT = "8e462d68b3ee89eb2a2429ca6b1ddd1a9c04ac3d3e340fb68188b139394274e8"
GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scorer_golden.jsonl"


def scorer_fingerprint(golden: Path = GOLDEN_PATH) -> str:
    """sha256 over normalized forms and error classes of the golden pairs plus the rule versions."""
    h = hashlib.sha256()
    h.update(f"{NORMALIZE_RULES_VERSION}|{ERROR_RULES_VERSION}\n".encode())
    for line in golden.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gt, pred = r["gt"], r["pred"]
        item = [canonical(gt), canonical(pred), sorted(classify(gt, pred)), sorted(notation_present(gt))]
        h.update(json.dumps(item, ensure_ascii=False).encode() + b"\n")
    return h.hexdigest()


def score_record(gt: str, pred: str) -> dict:
    exact = equivalent(gt, pred)
    return {
        "exact": exact,
        "gt_norm": canonical(gt),
        "pred_norm": canonical(pred),
        "classes": [] if exact else sorted(classify(gt, pred)),
        "present": sorted(notation_present(gt)),
    }


def score_records(records: list[dict], *, render: bool = True, cache: Path | None = None) -> list[dict]:
    rows = []
    for rec in records:
        pred = "" if rec.get("pred_error") else rec["pred_latex"]
        row = {"id": rec["id"], "paper_id": rec["paper_id"], "gt_latex": rec["gt_latex"], "pred_latex": pred}
        if rec.get("pred_error"):
            row["pred_error"] = rec["pred_error"]
        row.update(score_record(rec["gt_latex"], pred))
        rows.append(row)
    if not render or not rows:
        return rows
    from lightonocr_math.render import render_equivalence

    cache = cache or paths.cache_dir()
    gts = [r["gt_latex"] for r in rows]
    preds = [r["pred_latex"] for r in rows]
    # No Guard watchdog here: it would register no child and could only fail a finished run. render checks
    # MemAvailable (guard.preflight) before every container start, which is where the RAM floor matters.
    pairs = render_equivalence(gts, preds, cache=cache)
    for row, pr in zip(rows, pairs, strict=True):
        row["gt_render_status"] = pr.gt_status
        row["pred_render_status"] = pr.pred_status
        if pr.identical is None:
            row["render_identical"] = None
            row["render_tolerant"] = None
        else:
            row["render_identical"] = bool(pr.identical)
            row["render_tolerant"] = bool(pr.tolerant)
    return rows


def read_predictions(path: Path) -> list[dict]:
    """Strict streaming JSONL read; errors name the line number."""
    records: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: invalid JSON ({e})") from e
            if not isinstance(r, dict):
                raise ValueError(f"line {n}: expected a JSON object")  # noqa: TRY004
            for key in ("id", "paper_id"):
                if not isinstance(r.get(key), str) or not r[key]:
                    raise ValueError(f"line {n}: missing or empty {key}")
            if not isinstance(r.get("gt_latex"), str) or not r["gt_latex"].strip():
                raise ValueError(f"line {n}: missing gt_latex")
            if not isinstance(r.get("pred_latex"), str) and not r.get("pred_error"):
                raise ValueError(f"line {n}: need pred_latex (string) or pred_error")
            if r["id"] in seen:
                raise ValueError(f"line {n}: duplicate id {r['id']!r}")
            seen.add(r["id"])
            records.append(r)
    return records


def _metric(rows: list[dict], key: str, B: int, seed: int) -> dict:
    sel = [r for r in rows if r.get(key) is not None]
    return stats.cluster_bootstrap(
        [r["paper_id"] for r in sel], [1.0 if r[key] else 0.0 for r in sel], [1.0] * len(sel), B=B, seed=seed
    )


def score_file(
    pred_path: Path, out_dir: Path, *, render: bool = True, B: int = 10000, seed: int = 0, cache: Path | None = None
) -> Path:
    from lightonocr_math import splits
    from lightonocr_math.report import render_report

    pred_path, out_dir = Path(pred_path), Path(out_dir)
    records = read_predictions(pred_path)
    rows = sorted(score_records(records, render=render, cache=cache), key=lambda r: r["id"])

    metrics = {"exact": _metric(rows, "exact", B, seed)}
    disagreements = {"string_diff_render_identical": [], "string_equal_render_diff": []}
    n_gt_uncompilable = 0
    render_cfg: dict = {"enabled": False}
    if render:
        from lightonocr_math.render import render_config

        cfg = render_config()
        render_cfg = {"enabled": True, **cfg}
        metrics["render_identical"] = _metric(rows, "render_identical", B, seed)
        metrics["render_tolerant"] = _metric(rows, "render_tolerant", B, seed)
        n_gt_uncompilable = sum(1 for r in rows if r.get("render_identical") is None)
        for r in rows:
            if r.get("render_identical") is None:
                continue
            if not r["exact"] and r["render_identical"]:
                disagreements["string_diff_render_identical"].append(r["id"])
            if r["exact"] and not r["render_identical"]:
                disagreements["string_equal_render_diff"].append(r["id"])
    per_class = stats.per_class_rates(rows, B=B, seed=seed)

    info = splits.rule_info()
    results = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "scorer_version": SCORER_VERSION,
        "scorer_fingerprint": scorer_fingerprint(),
        "normalize_rules_version": NORMALIZE_RULES_VERSION,
        "error_rules_version": ERROR_RULES_VERSION,
        "split_rule_version": info["split_rule_version"],
        "split_digest": info["split_digest"],
        "render": render_cfg,
        "preamble_sha256": render_cfg.get("preamble_sha256"),
        "wrap_version": render_cfg.get("wrap_version"),
        "render_dpi": render_cfg.get("dpi"),
        "bootstrap": {"B": B, "seed": seed, "ci": 0.95},
        "predictions_path": str(pred_path),
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_records": len(rows),
        "n_gt_uncompilable": n_gt_uncompilable,
        "metrics": metrics,
        "per_class": per_class,
        "disagreements": {k: sorted(v) for k, v in disagreements.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    rp = out_dir / "results.json"
    rp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    return rp


def _fmt_ci(m: dict) -> str:
    if m["rate"] is None:
        return f"{'n/a':>8}  {'':>17}"
    lo = "n/a" if m["ci_low"] is None else f"{m['ci_low']:.3f}"
    hi = "n/a" if m["ci_high"] is None else f"{m['ci_high']:.3f}"
    return f"{m['rate']:8.3f}  [{lo}, {hi}]"


def summary_table(results: dict) -> str:
    lines = [f"{'metric':<34}{'rate':>8}  {'95% CI':<17}{'n':>6}"]
    names = {"exact": "normalized exact match", "render_identical": "render identical", "render_tolerant": "render tolerant (separate tier)"}
    for key, label in names.items():
        m = results["metrics"].get(key)
        if m:
            lines.append(f"{label:<34}{_fmt_ci(m)}{int(m['den']):>6}")
    for c in CLASSES:
        pc = results["per_class"][c]
        rate = "n/a" if pc["rate"] is None else f"{pc['rate']:.3f}"
        lines.append(f"  class {c:<26}{rate:>8}  {pc['errors']}/{pc['denom']}")
    lines.append(f"{'TOTAL':<34}{'':>8}  {'':<17}{results['n_records']:>6}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lightonocr_math.score", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-render", action="store_true", help="string tier only, no Docker")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        rp = score_file(a.predictions, a.out, render=not a.no_render, B=a.bootstrap, seed=a.seed, cache=a.cache)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except RenderInfrastructureError as e:
        print(f"error: render infrastructure failure: {e}", file=sys.stderr)
        return 3
    results = json.loads(rp.read_text(encoding="utf-8"))
    print(summary_table(results))
    print(f"\nwrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
