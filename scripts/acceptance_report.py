"""Text only. Reads a results dict, returns markdown. This is the only acceptance code that may change after the generation lock; it cannot change the verdict or any number because it never writes results.json and receives a copy."""

from __future__ import annotations

RERUN = "uv run python -m scripts.acceptance score --mode benchmark --scores-dir <dir> --evidence-dir <dir> --report <file>"
RESUME = (
    "uv run python -m scripts.acceptance_sensitivity run --mode benchmark "
    "--results .planning/phases/05-acceptance-evaluation/evidence/results.json "
    "--records data/benchmark/records.jsonl --resume"
)
D04 = (
    "Reported as is: no retraining, checkpoint or model change follows from these numbers; "
    "a new attempt would be a new LoRA version scored on a new benchmark version (D-04)."
)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _d(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _ci(lo, hi) -> str:
    return f"[{_f(lo)}, {_f(hi)}]"


def _rate(block: dict | None) -> str:
    if not block:
        return "n/a"
    return f"{_pct(block.get('rate'))} ({_pct(block.get('ci_low'))} to {_pct(block.get('ci_high'))})"


def _ok(flag) -> str:
    return "ok" if flag else "not ok"


def _headline(res: dict) -> list[str]:
    h = res["headline"]
    return [
        "## Headline: render-identical on the benchmark crops",
        "",
        "| | stock | tuned | delta | 95% CI of delta | records | papers | McNemar p |",
        "|---|---|---|---|---|---|---|---|",
        f"| render-identical | {_pct(h['stock_rate'])} | {_pct(h['tuned_rate'])} | {_d(h['delta'])} | {_ci(h['ci_low'], h['ci_high'])} | {h['n']} | {h['n_papers']} | {_f(h['flip']['mcnemar_p'])} |",
        "",
        "Rule: pass if CI lower bound > 0, a paired bootstrap that resamples papers, not equations (D-02).",
        "",
        (f"Paired outcomes: both right {h['flip']['both_right']}, both wrong {h['flip']['both_wrong']}, "
        f"right-to-wrong {h['flip']['right_to_wrong']}, wrong-to-right {h['flip']['wrong_to_right']}."),
        "",
    ]


def _regression(res: dict) -> list[str]:
    r = res["headline"]["regression"]
    return [
        "## Regression clauses (D-02)",
        "",
        "| clause | observed | limit | result |",
        "|---|---|---|---|",
        f"| right-to-wrong over wrong-to-right | {r['r2w']} of {r['w2r']} | at most {r['max_by_ratio']} | {_ok(r['ratio_ok'])} |",
        f"| right-to-wrong over stock-correct | {r['r2w']} of {r['stock_right']} | at most {r['max_by_share']} | {_ok(r['share_ok'])} |",
        "",
        f"Regression clauses: {_ok(res['headline']['regression_ok'])}.",
        "",
    ]


def _secondary(res: dict) -> list[str]:
    sec, sp = res["secondary"]["crops"], res["spacing"]["crops"]
    rows = [
        "## Secondary metrics",
        "",
        "Secondary metrics cannot change the verdict (D-03). Only the render-identical headline and the D-02 clauses decide.",
        "",
        "| metric (crops) | stock | tuned |",
        "|---|---|---|",
        f"| normalized exact | {_rate(sec['stock']['exact'])} | {_rate(sec['tuned']['exact'])} |",
        f"| render tolerant | {_rate(sec['stock'].get('render_tolerant'))} | {_rate(sec['tuned'].get('render_tolerant'))} |",
        f"| glyph error mean | {_f(sec['stock'].get('glyph_error_mean'))} | {_f(sec['tuned'].get('glyph_error_mean'))} |",
        f"| spacing-insensitive render identical | {_rate(sp['stock'])} | {_rate(sp['tuned'])} |",
        "",
        (f"Spacing column: paired delta {_d(sp['paired'].get('delta'))}, {sp['n_rerendered']} records re-rendered, "
        f"{sp['n_excluded']} excluded as undecided (D-03)."),
        "",
    ]
    return rows


def _per_class(res: dict) -> list[str]:
    rows = [
        "## Per-class table",
        "",
        ("Counts are string-tier scorer classes over all common ids; the render-identical headline is separate (D-14, D-20). "
        "A class with fewer stock errors than the minimum is reported as: not enough events to claim."),
        "",
        "| class | stock errors | tuned errors | denominator | stock error rate | tuned error rate | paired delta (CI) | claim |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["per_class"]:
        p = r["paired"]
        rows.append(
            f"| {r['class']} | {r['stock_errors']} | {r['tuned_errors']} | {r['denom']} | {_rate(r['stock'])} | {_rate(r['tuned'])} "
            f"| {_d(p.get('delta'))} ({_ci(p.get('ci_low'), p.get('ci_high'))}) | {r['claim']} |"
        )
    rows.append("")
    return rows


def _pages(res: dict) -> list[str]:
    t, pm, sp = res["transfer"], res["page_match"], res["spacing"]["pages"]
    sec = res["secondary"]["pages"]
    return [
        "## Page-level transfer (D-11..D-13)",
        "",
        "| | stock | tuned |",
        "|---|---|---|",
        f"| render-identical on pages | {_pct(t['stock_rate'])} | {_pct(t['tuned_rate'])} |",
        f"| equation match rate | {_pct(pm['stock']['rate'])} | {_pct(pm['tuned']['rate'])} |",
        f"| normalized exact | {_rate(sec['stock']['exact'])} | {_rate(sec['tuned']['exact'])} |",
        f"| spacing-insensitive render identical | {_rate(sp['stock'])} | {_rate(sp['tuned'])} |",
        "",
        (f"Paired delta {_d(t['delta'])}, 95% CI {_ci(t['ci_low'], t['ci_high'])}, {t['n']} equations in {t['n_papers']} papers. "
        "Unmatched equations count as failures for both arms."),
        "",
        f"Transfer statement: {t['statement']}.",
        "",
        "The transfer result is reported beside the verdict and does not block it.",
        "",
    ]


def _r2w(res: dict) -> list[str]:
    lst = res["regression_listing"]
    rows = ["## Right-to-wrong records (D-09)", ""]
    if not lst["rows"]:
        rows += ["No record that stock got right was lost by the tuned model.", ""]
    else:
        by = ", ".join(f"{k} {v}" for k, v in sorted(lst["by_class"].items()))
        rows += [
            (f"{len(lst['rows'])} records, by tuned error class: {by}. "
            f"Control-glyph flagged (prime, Greek, decoration, I vs 1 classes): {lst['n_control_glyph']}."),
            "",
            "| id | classes | control glyph | ground truth | stock | tuned |",
            "|---|---|---|---|---|---|",
        ]
        for r in lst["rows"]:
            rows.append(
                f"| {r['id']} | {', '.join(r['classes'])} | {'yes' if r['control_glyph'] else 'no'} "
                f"| `{_cell(r['gt_latex'])}` | `{_cell(r['stock_pred'])}` | `{_cell(r['tuned_pred'])}` |"
            )
        rows.append("")
    rows += [
        ("Near-tie caveat: greedy decoding is not stable across engines. The HF bf16 head rounds logits and takes the lowest token id on a tie, "
        "while llama.cpp keeps f32 logits (PARITY_DIAGNOSIS.md, example c00db734), so a single flip can be a near-tie and not a learned error. "
        "The paired paper-level statistics carry the verdict, not any single flip."),
        "",
    ]
    return rows


def _cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("`", "'").replace("\n", " ")


def _gain(res: dict) -> list[str]:
    return [
        "## Where the gain comes from (D-16, D-22)",
        "",
        ("Phase 4 dev analysis (DEV_ANALYSIS.md), Ollama dev, 256 wrong-to-right crops: other 210 (82%), decoration 38 (15%), "
        "symbol_swap 6, dropped_prime 2. The 20 right-to-wrong crops: other 17, decoration 3. "
        "The gain is mostly the model learning corpus conventions (delimiter sizing, alignment wrappers, spacing), not the rare glyph classes. "
        "The per-class table above shows the benchmark view of the same question."),
        "",
    ]


def _dev(res: dict) -> list[str]:
    ref, h = res.get("dev_reference"), res["headline"]
    rows = ["## Dev vs benchmark (D-17)", ""]
    if ref:
        rows.append(
            f"Dev (Ollama, 926 crops): stock {_pct(ref.get('stock'))}, tuned {_pct(ref.get('candidate'))}, delta {_d(ref.get('delta'))}. "
            f"Benchmark: stock {_pct(h['stock_rate'])}, tuned {_pct(h['tuned_rate'])}, delta {_d(h['delta'])}."
        )
    else:
        rows.append(f"No dev reference stored. Benchmark delta {_d(h['delta'])}.")
    rows += [
        "",
        ("Selection caveat: the dev split guided the hyperparameter sweep and the checkpoint choice, so the dev gain is in-sample. "
        "The benchmark is the only held-out claim."),
        "",
    ]
    return rows


def _errors(res: dict) -> list[str]:
    ne = res["n_error"]
    return [
        "## Generation errors (D-19)",
        "",
        "n_error per arm (requests that failed, scored as failures): " + ", ".join(f"{k} {v}" for k, v in sorted(ne.items())) + ".",
        "",
    ]


def _rerun(res: dict) -> list[str]:
    rows = [
        "## Rerun (D-18)",
        "",
        f"Mode {res['mode']}, git commit {res.get('git_commit')}, acceptance constants sha256 {res['acceptance_sha256']}.",
        "",
        "```",
        RERUN,
        "```",
        "",
        "| input | sha256 |",
        "|---|---|",
    ]
    for k, v in sorted(res["inputs"].items()):
        rows.append(f"| {k} | {v['sha256']} |")
    rows.append("")
    return rows


def _sensitivity(sens: dict | None) -> list[str]:
    rows = ["## Appendix: CPU llama-server sensitivity", ""]
    if not sens:
        return rows + ["CPU llama-server sensitivity run: not yet run; non-blocking (D-21).", ""]
    r2w, w2r = sens["r2w"], sens["w2r"]
    rows += [
        f"Coverage: {sens['n_done']} of {sens['n_total']} flipped records run on CPU.",
        "",
        "| direction | run | held on CPU | reversed |",
        "|---|---|---|---|",
        f"| right-to-wrong | {r2w['done']} of {r2w['total']} | {r2w['held']} | {r2w['reversed']} |",
        f"| wrong-to-right | {w2r['done']} of {w2r['total']} | {w2r['held']} | {w2r['reversed']} |",
        "",
    ]
    if not sens.get("complete"):
        rows += ["The run is incomplete. Resume with:", "", "```", RESUME, "```", ""]
    rows += ["The sensitivity run does not change the verdict (D-21).", ""]
    return rows


def render_acceptance_md(results: dict, sensitivity: dict | None = None) -> str:
    """Markdown for the acceptance report; a pure function of its arguments."""
    h = results["headline"]
    verdict = results["verdict"]
    lines = [
        "# Acceptance: tuned LoRA vs stock on the held-out benchmark",
        "",
        f"Verdict: {verdict.upper()}. {results['statements']['verdict']}.",
        "",
    ]
    if h.get("reasons"):
        lines += ["Reasons: " + "; ".join(h["reasons"]) + ".", ""]
    for part in (_headline, _regression, _secondary, _per_class, _pages, _r2w, _gain, _dev, _errors):
        lines += part(results)
    if verdict == "fail":
        lines += ["## Outcome on a miss", "", D04, ""]
    lines += _rerun(results)
    lines += _sensitivity(sensitivity)
    return "\n".join(lines).rstrip("\n") + "\n"
