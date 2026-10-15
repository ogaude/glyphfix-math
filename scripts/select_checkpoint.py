"""Score dev mini-eval predictions, pick the checkpoint of a run and the winner of the 3-config sweep.

Usage (from repo root, .venv-lighton or uv environment; scoring renders through the Docker sandbox):
  uv run python -m scripts.select_checkpoint score --run-dir data/train/runs/RUN
  uv run python -m scripts.select_checkpoint score --pred data/train/dev_eval/base.pred.jsonl --out data/train/dev_eval/base.score
  uv run python -m scripts.select_checkpoint select --run-dir data/train/runs/RUN --base-score data/train/dev_eval/base.score
  uv run python -m scripts.select_checkpoint sweep --run RUN_A --run RUN_B --run RUN_C \
      --base-score data/train/dev_eval/base.score --out sweep.json --report sweep.md

Selection rule (written before any run; assumptions A-04-1 and A-04-2):
  * checkpoint: highest dev render-identical rate on the 150-crop mini-eval; steps within TIE_DELTA (0.02, 3 of 150
    crops) of the best are tied; ties go to the lowest mean error rate over the five glyph classes (classes with a
    non-zero denominator), then to the earliest step
  * sweep: same metric on each config's last eval adapter; when the best two differ by less than TIE_DELTA, or the
    paired bootstrap interval of their difference contains 0, they are tied and the preference order is
    SWEEP_PREFERENCE (lower learning rate and the stock alpha first)
Only dev mini-eval scores are read. Generation and render equivalence decide; no training-loss number is used and no
benchmark file is read.

Exit codes: 0 ok, 2 bad input, 3 render infrastructure failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from lightonocr_math import stats
from lightonocr_math.errors import CLASSES

TIE_DELTA = 0.02
GLYPH_CLASSES = tuple(c for c in CLASSES if c != "other")
SWEEP_PREFERENCE = ("lr1e-4-a8", "lr2e-4-a8", "lr1e-4-a16")
RULE_CHECKPOINT = (
    "best dev mini-eval render-identical; steps within 0.02 of the best are tied; ties go to the lowest mean glyph-class "
    "error rate, then the earliest step"
)
RULE_SWEEP = (
    "same metric on each final adapter; best and second best within 0.02 or with a paired bootstrap interval containing 0 "
    "are tied; ties follow the order lr1e-4/alpha8, lr2e-4/alpha8, lr1e-4/alpha16"
)
EXIT_OK, EXIT_BAD_INPUT, EXIT_RENDER = 0, 2, 3


class BadInput(Exception):
    pass


def glyph_error_mean(per_class: dict) -> float | None:
    """Mean error rate over the five glyph classes that have a non-zero denominator; None when none do."""
    rates = []
    for c in GLYPH_CLASSES:
        pc = per_class.get(c) or {}
        den = pc.get("denom") or 0
        if den > 0:
            rates.append(pc["errors"] / den)
    return sum(rates) / len(rates) if rates else None


def choose_step(per_step: list[dict]) -> dict:
    """The chosen per_step entry under the checkpoint rule."""
    if not per_step:
        raise ValueError("no steps to choose from")
    best = max(e["render_identical"] for e in per_step)
    tied = [e for e in per_step if best - e["render_identical"] <= TIE_DELTA + 1e-9]
    inf = float("inf")
    return min(tied, key=lambda e: (e["glyph_error_mean"] if e.get("glyph_error_mean") is not None else inf, e["step"]))


def _pref_rank(run_id: str) -> int:
    return SWEEP_PREFERENCE.index(run_id) if run_id in SWEEP_PREFERENCE else len(SWEEP_PREFERENCE)


def choose_winner(configs: list[dict], pairwise: dict | None = None) -> dict:
    """{"winner": run_id, "tie": bool, "tied": [run_id]} for the sweep.

    pairwise maps (run_a, run_b) or "run_a|run_b" to a paired_bootstrap_delta result; when it holds the top two configs
    and its interval contains 0, they are tied even if the point estimates are 0.02 or more apart.
    """
    if not configs:
        raise ValueError("no configs")
    ranked = sorted(configs, key=lambda c: (-c["render_identical"], _pref_rank(c["run_id"])))
    best = ranked[0]
    tied = [c for c in ranked if best["render_identical"] - c["render_identical"] <= TIE_DELTA + 1e-9]
    if len(tied) == 1 and len(ranked) > 1:
        second = ranked[1]
        pw = _lookup_pair(pairwise, best["run_id"], second["run_id"])
        if pw is not None and pw.get("ci_low") is not None and pw["ci_low"] <= 0 <= pw["ci_high"]:
            tied = [best, second]
    is_tie = len(tied) > 1
    if is_tie:
        winner = min(tied, key=lambda c: (_pref_rank(c["run_id"]), -c["render_identical"]))
    else:
        winner = best
    return {"winner": winner["run_id"], "tie": is_tie, "tied": [c["run_id"] for c in tied]}


def _lookup_pair(pairwise: dict | None, a: str, b: str) -> dict | None:
    if not pairwise:
        return None
    for key in ((a, b), (b, a), f"{a}|{b}", f"{b}|{a}"):
        if key in pairwise:
            return pairwise[key]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(score_dir: Path) -> dict[str, dict]:
    p = Path(score_dir) / "records.jsonl"
    if not p.is_file():
        raise BadInput(f"{p} missing; run the score subcommand first")
    with open(p, encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, fh))}


def read_results(score_dir: Path) -> dict:
    p = Path(score_dir) / "results.json"
    if not p.is_file():
        raise BadInput(f"{p} missing; run the score subcommand first")
    return json.loads(p.read_text(encoding="utf-8"))


def paired_vs(base_rows: dict[str, dict], rows: dict[str, dict], *, B: int = 10000, seed: int = 0) -> tuple[dict, dict]:
    """(paired bootstrap delta, flip table) over ids present in both runs with a decided render result."""
    common = sorted(i for i in base_rows if i in rows and base_rows[i].get("render_identical") is not None and rows[i].get("render_identical") is not None)
    if not common:
        raise BadInput("no common ids with a render result in both runs")
    papers = [base_rows[i]["paper_id"] for i in common]
    a = [1.0 if base_rows[i]["render_identical"] else 0.0 for i in common]
    b = [1.0 if rows[i]["render_identical"] else 0.0 for i in common]
    delta = stats.paired_bootstrap_delta(papers, a, b, [1.0] * len(common), B=B, seed=seed)
    flips = stats.flip_table({i: bool(base_rows[i]["render_identical"]) for i in common}, {i: bool(rows[i]["render_identical"]) for i in common})
    return delta, flips


def step_entry(step: int, results: dict) -> dict:
    m = results["metrics"]
    return {
        "step": step,
        "render_identical": m["render_identical"]["rate"],
        "exact": m["exact"]["rate"],
        "glyph_error_mean": glyph_error_mean(results["per_class"]),
        "per_class": results["per_class"],
    }


def _score_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for p in (Path(run_dir) / "dev_eval").glob("step-*.score"):
        m = re.fullmatch(r"step-(\d+)\.score", p.name)
        if m and (p / "results.json").is_file():
            found.append((int(m.group(1)), p))
    return sorted(found)


def score_one(pred: Path, out: Path) -> str:
    """Score a prediction file unless an up-to-date result exists; returns "skipped" or "scored"."""
    from lightonocr_math import score

    res = Path(out) / "results.json"
    if res.is_file():
        try:
            if json.loads(res.read_text(encoding="utf-8")).get("predictions_sha256") == sha256_file(pred):
                return "skipped"
        except ValueError:
            pass
    score.score_file(pred, out, render=True)
    return "scored"


def cmd_score(a: argparse.Namespace) -> int:
    if a.pred:
        if not a.out:
            raise BadInput("--out is required with --pred")
        jobs = [(Path(a.pred), Path(a.out))]
    elif a.run_dir:
        preds = sorted((Path(a.run_dir) / "dev_eval").glob("step-*.pred.jsonl"))
        if not preds:
            raise BadInput(f"no step-*.pred.jsonl under {a.run_dir}/dev_eval")
        jobs = [(p, p.with_name(p.name.removesuffix(".pred.jsonl") + ".score")) for p in preds]
    else:
        raise BadInput("give --run-dir or --pred")
    for pred, out in jobs:
        print(f"{pred}: {score_one(pred, out)}", flush=True)
    return EXIT_OK


def cmd_select(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    base_results, base_rows = read_results(a.base_score), read_rows(a.base_score)
    dirs = _score_dirs(run_dir)
    if not dirs:
        raise BadInput(f"no scored steps under {run_dir}/dev_eval (expected step-N.score)")
    per_step = [step_entry(s, read_results(d)) for s, d in dirs]
    chosen = choose_step(per_step)
    cstep = chosen["step"]
    adapter = run_dir / "eval" / f"step-{cstep}" / "adapter"
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise BadInput(f"{weights} missing")
    delta, flips = paired_vs(base_rows, read_rows(dict(dirs)[cstep]))
    out = {
        "rule": RULE_CHECKPOINT,
        "tie_delta": TIE_DELTA,
        "per_step": per_step,
        "chosen_step": cstep,
        "chosen_adapter": str(adapter),
        "adapter_sha256": sha256_file(weights),
        "delta_vs_base": delta,
        "flip_vs_base": flips,
        "scorer_version": base_results.get("scorer_version"),
        "scorer_fingerprint": base_results.get("scorer_fingerprint"),
    }
    (run_dir / "selection.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"chosen step {cstep}: render identical {chosen['render_identical']:.3f}, delta vs base {delta['delta']:+.3f}")
    return EXIT_OK


def _run_lr_alpha(run_dir: Path) -> tuple[float | None, int | None]:
    mf = run_dir / "run_manifest.json"
    if mf.is_file():
        cfg = json.loads(mf.read_text(encoding="utf-8")).get("config", {})
        if "lr" in cfg and "alpha" in cfg:
            return cfg["lr"], cfg["alpha"]
    m = re.search(r"lr([0-9.]+e-?\d+)-a(\d+)", run_dir.name)
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def sweep_report(sweep: dict) -> str:
    lines = [
        "# LoRA sweep result",
        "",
        f"Rule: {sweep['rule']}",
        "",
        "| config | LR | alpha | step | render identical | exact | " + " | ".join(GLYPH_CLASSES) + " | delta vs base [95% CI] |",
        "|---|---|---|---|---|---|" + "---|" * len(GLYPH_CLASSES) + "---|",
    ]
    for c in sweep["configs"]:
        pcs = " | ".join(
            "n/a" if not (c["per_class"].get(k) or {}).get("denom") else f"{c['per_class'][k]['errors']}/{c['per_class'][k]['denom']}"
            for k in GLYPH_CLASSES
        )
        d = c["delta_vs_base"]
        lines.append(
            f"| {c['run_id']} | {c['lr']} | {c['alpha']} | {c['step']} | {_fmt(c['render_identical'])} | {_fmt(c['exact'])} | {pcs} | "
            f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] |"
        )
    lines += ["", "Pairwise paired deltas (second minus first):", ""]
    for key, pw in sweep["pairwise"].items():
        lines.append(f"- {key}: {pw['delta']:+.3f} [{pw['ci_low']:+.3f}, {pw['ci_high']:+.3f}]")
    lines += ["", f"Winner: {sweep['winner']} ({'tie, preference order applied' if sweep['tie'] else 'clear lead'})", ""]
    return "\n".join(lines)


def cmd_sweep(a: argparse.Namespace) -> int:
    base_rows = read_rows(a.base_score)
    configs, rows_by_run = [], {}
    for run in a.run:
        run_dir = Path(run)
        sel_p = run_dir / "selection.json"
        if not sel_p.is_file():
            raise BadInput(f"{sel_p} missing; run select for {run_dir} first")
        sel = json.loads(sel_p.read_text(encoding="utf-8"))
        last = max(sel["per_step"], key=lambda e: e["step"])
        dirs = dict(_score_dirs(run_dir))
        if last["step"] not in dirs:
            raise BadInput(f"no score directory for step {last['step']} of {run_dir}")
        rows = read_rows(dirs[last["step"]])
        rows_by_run[run_dir.name] = rows
        delta, flips = paired_vs(base_rows, rows)
        lr, alpha = _run_lr_alpha(run_dir)
        configs.append({
            "run_id": run_dir.name, "lr": lr, "alpha": alpha, "step": last["step"], "render_identical": last["render_identical"],
            "exact": last["exact"], "glyph_error_mean": last["glyph_error_mean"], "per_class": last["per_class"],
            "delta_vs_base": delta, "flip_vs_base": flips,
        })  # fmt: skip
    pairwise = {}
    names = [c["run_id"] for c in configs]
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            pairwise[f"{x}|{y}"] = paired_vs(rows_by_run[x], rows_by_run[y])[0]
    verdict = choose_winner(configs, pairwise)
    sweep = {"configs": configs, "pairwise": pairwise, "rule": RULE_SWEEP, "winner": verdict["winner"], "tie": verdict["tie"], "tied": verdict["tied"]}
    Path(a.out).write_text(json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if a.report:
        Path(a.report).write_text(sweep_report(sweep), encoding="utf-8")
    print(f"winner {sweep['winner']} (tie={sweep['tie']})")
    return EXIT_OK


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score prediction files with the frozen scorer")
    s.add_argument("--run-dir")
    s.add_argument("--pred")
    s.add_argument("--out")
    s.set_defaults(func=cmd_score)
    c = sub.add_parser("select", help="choose the checkpoint of one run")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--base-score", required=True)
    c.set_defaults(func=cmd_select)
    w = sub.add_parser("sweep", help="choose the sweep winner")
    w.add_argument("--run", action="append", required=True)
    w.add_argument("--base-score", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--report")
    w.set_defaults(func=cmd_sweep)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return args.func(args)
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except Exception as e:
        from lightonocr_math.render import RenderInfrastructureError

        if isinstance(e, RenderInfrastructureError):
            print(f"error: render infrastructure failure: {e}", file=sys.stderr)
            return EXIT_RENDER
        raise


if __name__ == "__main__":
    raise SystemExit(main())
