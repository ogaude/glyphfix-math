"""Pre-registered gates for a merged candidate model, then registration or rollback.

Usage (from repo root, uv environment; scoring of the arms is done beforehand with score / select_checkpoint):
  uv run python -m scripts.register_model gates --run-id RUN_ID --run-dir data/train/runs/RUN_ID \
      --regression data/corpus/regression/regression.jsonl --reg-stock DIR --reg-cand DIR \
      --dev-stock DIR --dev-cand DIR --dev-stock-sidecar F --dev-cand-sidecar F \
      --hf-base PRED --hf-tuned PRED --hf-merged PRED --ids data/train/dev_eval_ids.txt \
      --merge-check data/models/RUN_ID/merge_check.json --out data/models/RUN_ID/gates.json
  uv run python -m scripts.register_model register --run-id RUN_ID --run-dir data/train/runs/RUN_ID \
      --gates data/models/RUN_ID/gates.json --export data/models/RUN_ID/export.json \
      --trainset-dir DIR --sweep sweep.json --registry data/models/REGISTRY.json

Score directories hold records.jsonl (render_identical, paper_id) written by lightonocr_math.score. Each HF
PRED is OUT_STEM.pred.jsonl with OUT_STEM.sidecar.jsonl and OUT_STEM.score next to it. Ollama arms on the full
dev split are filtered to the dev-subset ids for the export gate.

Gates (thresholds in GATES, fixed before any training run):
  regression  control rows (stock right) must not flip to wrong beyond tolerance, target rows must not regress
              and must show fixes
  export      merge check passes; Ollama and HF see the same prompt-token offsets; the Ollama gain tracks the HF
              gain; the LoRA delta survived Q8_0
  dev         candidate render-identical rate over the full dev split is at least stock's
A pass copies the candidate tag to the final tag, removes the candidate tag and appends a verified registry
entry. A failure removes the candidate tag and writes registration.json.

Exit codes: 0 registered (gates: all pass), 1 gate failed (candidate removed), 2 bad input.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from lightonocr_math import score, stats
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json
from lightonocr_math.paths import repo_root
from lightonocr_math.train import registry
from scripts.export_merge import merge_verdict
from scripts.spikes import parity_report
from scripts.spikes.ollama_infer import DEFAULT_HOST, _model_digest

# D-16: thresholds fixed in code before any training run. Control tolerance is the user-approved default:
# at most 3 right-to-wrong flips over the 75 control rows and at most 1 per pool_flag group of 15.
GATES = {
    "regression": {
        "max_control_r2w_total": 3,
        "max_control_r2w_per_flag": 1,
        "max_target_r2w": 0,
        "min_target_fixed_total": 1,
        "min_fixed_per_class": 1,
        "min_rows_for_class_requirement": 10,
    },
    "export": {
        "expected_changed_tensors": 200,
        "min_measured_frac": 0.90,
        "min_offset_match_frac": 0.95,
        "max_transfer_gap": 0.03,
        "min_changed_outputs": 1,
    },
    "dev": {"min_delta": 0.0},
}

FINAL_PREFIX = "mathocr-lora-q8"
CAND_PREFIX = "mathocr-cand"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EPS = 1e-9
EXIT_OK, EXIT_GATE, EXIT_BAD = 0, 1, 2


class BadInput(Exception):
    pass


# --------------------------------------------------------------------------------------- pure gate logic


def _decided(scored: dict[str, dict], i: str) -> bool | None:
    row = scored.get(i)
    if row is None:
        return None
    v = row.get("render_identical")
    return None if v is None else bool(v)


def _paired(ids, a: dict[str, dict], b: dict[str, dict]) -> tuple[dict[str, bool], dict[str, bool], int]:
    """Boolean maps for ids decided in both arms, and the count of excluded ids."""
    ra: dict[str, bool] = {}
    rb: dict[str, bool] = {}
    excluded = 0
    for i in ids:
        x, y = _decided(a, i), _decided(b, i)
        if x is None or y is None:
            excluded += 1
        else:
            ra[i], rb[i] = x, y
    return ra, rb, excluded


def regression_gate(regression_rows: list[dict], stock_scored: dict, cand_scored: dict, thresholds: dict) -> dict:
    t = thresholds
    controls = [r for r in regression_rows if r["regression_class"] == "control"]
    targets = [r for r in regression_rows if r["regression_class"] != "control"]

    c_ids = [r["id"] for r in controls]
    ca, cb, c_excl = _paired(c_ids, stock_scored, cand_scored)
    c_flip = stats.flip_table(ca, cb)
    flag_of = {r["id"]: r["pool_flag"] for r in controls}
    per_flag: dict[str, dict] = {}
    for flag in sorted(set(flag_of.values())):
        ids = [i for i in ca if flag_of[i] == flag]
        ft = stats.flip_table({i: ca[i] for i in ids}, {i: cb[i] for i in ids})
        per_flag[flag] = {"n": ft["n"], "r2w": ft["right_to_wrong"], "w2r": ft["wrong_to_right"], "r2w_ids": ft["right_to_wrong_ids"]}

    t_ids = [r["id"] for r in targets]
    ta, tb, t_excl = _paired(t_ids, stock_scored, cand_scored)
    t_flip = stats.flip_table(ta, tb)
    cls_of = {r["id"]: r["regression_class"] for r in targets}
    cls_rows = collections.Counter(cls_of.values())
    per_class: dict[str, dict] = {}
    for cls in sorted(cls_rows):
        ids = [i for i in ta if cls_of[i] == cls]
        ft = stats.flip_table({i: ta[i] for i in ids}, {i: tb[i] for i in ids})
        per_class[cls] = {
            "rows": cls_rows[cls],
            "n_scored": ft["n"],
            "fixed": ft["wrong_to_right"],
            "r2w": ft["right_to_wrong"],
            "required": cls_rows[cls] >= t["min_rows_for_class_requirement"],
        }

    reasons: list[str] = []
    if c_flip["right_to_wrong"] > t["max_control_r2w_total"]:
        reasons.append(f"control r2w {c_flip['right_to_wrong']} > {t['max_control_r2w_total']}")
    for flag, d in per_flag.items():
        if d["r2w"] > t["max_control_r2w_per_flag"]:
            reasons.append(f"control r2w in flag {flag}: {d['r2w']} > {t['max_control_r2w_per_flag']}")
    if t_flip["right_to_wrong"] > t["max_target_r2w"]:
        reasons.append(f"target r2w {t_flip['right_to_wrong']} > {t['max_target_r2w']}")
    if t_flip["wrong_to_right"] < t["min_target_fixed_total"]:
        reasons.append(f"target fixed {t_flip['wrong_to_right']} < {t['min_target_fixed_total']}")
    for cls, d in per_class.items():
        if d["required"] and d["fixed"] < t["min_fixed_per_class"]:
            reasons.append(f"class {cls}: fixed {d['fixed']} < {t['min_fixed_per_class']}")

    return {
        "control": {
            "n": c_flip["n"],
            "excluded_none": c_excl,
            "r2w": c_flip["right_to_wrong"],
            "w2r": c_flip["wrong_to_right"],
            "r2w_ids": c_flip["right_to_wrong_ids"],
            "mcnemar_p": c_flip["mcnemar_p"],
            "per_flag": per_flag,
        },
        "target": {
            "n": t_flip["n"],
            "excluded_none": t_excl,
            "r2w": t_flip["right_to_wrong"],
            "r2w_ids": t_flip["right_to_wrong_ids"],
            "fixed": t_flip["wrong_to_right"],
            "mcnemar_p": t_flip["mcnemar_p"],
            "per_class": per_class,
        },
        "reasons": reasons,
        "pass": not reasons,
    }


def _rate(arm: dict[str, dict], ids) -> float | None:
    vals = [arm[i]["render_identical"] for i in ids if i in arm and arm[i].get("render_identical") is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def export_gate(
    merge_check: dict,
    sub_ids: list[str],
    hf_base: dict,
    hf_tuned: dict,
    hf_merged: dict,
    ol_stock: dict,
    ol_cand: dict,
    thresholds: dict,
) -> dict:
    t = thresholds
    reasons: list[str] = []
    merge_ok = bool(merge_check.get("pass")) if "compare" not in merge_check else merge_verdict(merge_check["compare"], t["expected_changed_tensors"])
    if not merge_ok:
        reasons.append("merge check failed")

    sub = list(sub_ids)
    hf_sub = {i: hf_tuned[i] for i in sub if i in hf_tuned}
    ol_sub = {i: ol_cand[i] for i in sub if i in ol_cand}
    measured = [
        i
        for i in sub
        if i in hf_sub
        and i in ol_sub
        and ol_sub[i].get("prompt_eval_count") is not None
        and hf_sub[i].get("n_prompt_tokens") is not None
    ]
    measured_frac = len(measured) / len(sub) if sub else 0.0
    modal = parity_report.modal_template_offset(hf_sub, ol_sub)
    offsets = {i: ol_sub[i]["prompt_eval_count"] - hf_sub[i]["n_prompt_tokens"] for i in measured}
    matched = [i for i in measured if offsets[i] == modal]
    match_frac = len(matched) / len(measured) if measured else 0.0
    if measured_frac + EPS < t["min_measured_frac"]:
        reasons.append(f"offsets measured on {measured_frac:.3f} of crops < {t['min_measured_frac']}")
    if match_frac + EPS < t["min_offset_match_frac"]:
        reasons.append(f"offset equals modal on {match_frac:.3f} of measured crops < {t['min_offset_match_frac']}")

    arms = (hf_base, hf_tuned, ol_stock, ol_cand)
    usable = [i for i in matched if all(_decided_arm(a, i) is not None for a in arms)]
    r_hb, r_ht, r_os, r_oc = (_rate(a, usable) for a in arms)
    ol_gain = None if r_oc is None or r_os is None else r_oc - r_os
    hf_gain = None if r_ht is None or r_hb is None else r_ht - r_hb
    if ol_gain is None or hf_gain is None:
        reasons.append("no matched crops with decided render results in all four arms")
    elif ol_gain + EPS < hf_gain - t["max_transfer_gap"]:
        reasons.append(f"Ollama gain {ol_gain:.3f} < HF gain {hf_gain:.3f} - {t['max_transfer_gap']}")

    changed = [i for i in sub if i in ol_cand and i in ol_stock and ol_cand[i].get("raw_text") != ol_stock[i].get("raw_text")]
    if len(changed) < t["min_changed_outputs"]:
        reasons.append(f"candidate raw output differs from stock on {len(changed)} crops < {t['min_changed_outputs']}")

    both = [i for i in sub if i in hf_merged and i in hf_tuned]
    raw_same = sum(1 for i in both if hf_merged[i].get("raw_text") == hf_tuned[i].get("raw_text"))
    cats: collections.Counter = collections.Counter()
    for i in sub:
        if i in ol_cand and i in hf_merged:
            try:
                cats[str(parity_report.categorize(hf_merged[i], ol_cand[i], template_offset=modal))] += 1
            except KeyError:
                cats["unmeasured"] += 1
    r_mg, r_tu = _rate(hf_merged, both), _rate(hf_tuned, both)

    return {
        "merge_pass": merge_ok,
        "n_subset": len(sub),
        "measured_frac": measured_frac,
        "modal_offset": modal,
        "offset_match_frac": match_frac,
        "n_matched": len(matched),
        "n_usable": len(usable),
        "rates_on_matched": {"hf_base": r_hb, "hf_tuned": r_ht, "ollama_stock": r_os, "ollama_candidate": r_oc},
        "hf_gain": hf_gain,
        "ollama_gain": ol_gain,
        "n_changed_outputs": len(changed),
        "diagnostics": {
            "hf_merged_vs_tuned_raw_identical_frac": raw_same / len(both) if both else None,
            "hf_merged_minus_tuned_render_identical": None if r_mg is None or r_tu is None else r_mg - r_tu,
            "ollama_cand_vs_hf_merged_categories": dict(cats),
        },
        "reasons": reasons,
        "pass": not reasons,
    }


def _decided_arm(arm: dict, i: str) -> bool | None:
    row = arm.get(i)
    if row is None or row.get("render_identical") is None:
        return None
    return bool(row["render_identical"])


def dev_gate(stock_scored: dict, cand_scored: dict, thresholds: dict) -> dict:
    ids = sorted(set(stock_scored) | set(cand_scored))
    ra, rb, excluded = _paired(ids, stock_scored, cand_scored)
    common = sorted(ra)
    flips = stats.flip_table(ra, rb)
    papers = [str((stock_scored.get(i) or cand_scored[i]).get("paper_id", i)) for i in common]
    boot = stats.paired_bootstrap_delta(papers, [float(ra[i]) for i in common], [float(rb[i]) for i in common], [1.0] * len(common))
    n = len(common)
    rate_s = sum(ra.values()) / n if n else None
    rate_c = sum(rb.values()) / n if n else None
    delta = None if n == 0 else rate_c - rate_s
    reasons = []
    if delta is None:
        reasons.append("no common decided ids")
    elif delta + EPS < thresholds["min_delta"]:
        reasons.append(f"candidate rate {rate_c:.4f} < stock rate {rate_s:.4f} + {thresholds['min_delta']}")
    return {
        "n_common": n,
        "excluded_none": excluded,
        "stock_rate": rate_s,
        "candidate_rate": rate_c,
        "delta": delta,
        "bootstrap": boot,
        "flip": {k: v for k, v in flips.items() if not k.endswith("_ids")},
        "r2w_ids": flips["right_to_wrong_ids"],
        "w2r_ids": flips["wrong_to_right_ids"],
        "reasons": reasons,
        "pass": not reasons,
    }


# ------------------------------------------------------------------------------------------- IO helpers


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def read_scored(score_dir: Path) -> dict[str, dict]:
    p = Path(score_dir) / "records.jsonl"
    if not p.is_file():
        raise BadInput(f"{p} missing")
    return _by_id(read_jsonl(p))


def _stem(pred: Path) -> Path:
    name = pred.name
    if not name.endswith(".pred.jsonl"):
        raise BadInput(f"{pred} must end with .pred.jsonl")
    return pred.with_name(name.removesuffix(".pred.jsonl"))


def build_hf_arm(pred: Path) -> dict[str, dict]:
    """Score records joined with the sidecar of an HF generation run."""
    stem = _stem(Path(pred))
    scored = read_scored(stem.with_name(stem.name + ".score"))
    side_p = stem.with_name(stem.name + ".sidecar.jsonl")
    if not side_p.is_file():
        raise BadInput(f"{side_p} missing")
    side = _by_id(read_jsonl(side_p))
    return {i: {**s, **{k: side[i].get(k) for k in ("raw_text", "n_prompt_tokens", "finish")}} for i, s in scored.items() if i in side}


def build_ollama_arm(score_dir: Path, sidecar: Path) -> dict[str, dict]:
    scored = read_scored(score_dir)
    if not Path(sidecar).is_file():
        raise BadInput(f"{sidecar} missing")
    side = _by_id(read_jsonl(sidecar))
    return {i: {**s, **{k: side[i].get(k) for k in ("raw_text", "prompt_eval_count", "done_reason")}} for i, s in scored.items() if i in side}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False).stdout.strip()
    except OSError:
        return None
    return out or None


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id or ""):
        raise BadInput("run id must match [A-Za-z0-9._-]+")


def _ollama(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ollama", *args], capture_output=True, text=True, check=False)


# -------------------------------------------------------------------------------------------- subcommands


def cmd_gates(a: argparse.Namespace) -> int:
    _check_run_id(a.run_id)
    reg_rows = read_jsonl(a.regression)
    regression = regression_gate(reg_rows, read_scored(a.reg_stock), read_scored(a.reg_cand), GATES["regression"])

    ids_path = Path(a.ids)
    if not ids_path.is_file():
        raise BadInput(f"{ids_path} missing")
    sub_ids = [x.strip() for x in ids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    mc_path = Path(a.merge_check)
    if not mc_path.is_file():
        raise BadInput(f"{mc_path} missing")
    merge_check = json.loads(mc_path.read_text(encoding="utf-8"))

    ol_stock = build_ollama_arm(a.dev_stock, a.dev_stock_sidecar)
    ol_cand = build_ollama_arm(a.dev_cand, a.dev_cand_sidecar)
    sub = set(sub_ids)
    export = export_gate(
        merge_check,
        sub_ids,
        build_hf_arm(Path(a.hf_base)),
        build_hf_arm(Path(a.hf_tuned)),
        build_hf_arm(Path(a.hf_merged)),
        {i: r for i, r in ol_stock.items() if i in sub},
        {i: r for i, r in ol_cand.items() if i in sub},
        GATES["export"],
    )
    dev = dev_gate(read_scored(a.dev_stock), read_scored(a.dev_cand), GATES["dev"])
    out = {
        "run_id": a.run_id,
        "thresholds": GATES,
        "scorer_version": score.SCORER_VERSION,
        "scorer_fingerprint": score.SCORER_FINGERPRINT,
        "regression": regression,
        "export": export,
        "dev": dev,
        "pass": bool(regression["pass"] and export["pass"] and dev["pass"]),
    }
    write_json(Path(a.out), out)
    print(f"gates: regression {regression['pass']} export {export['pass']} dev {dev['pass']} -> pass {out['pass']}")
    return EXIT_OK if out["pass"] else EXIT_GATE


def _load(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise BadInput(f"{p} missing")
    return json.loads(p.read_text(encoding="utf-8"))


def build_entry(a: argparse.Namespace, gates: dict, export: dict, final_tag: str, digest: str | None) -> dict:
    run_dir = Path(a.run_dir)
    rm_path = run_dir / "run_manifest.json"
    manifest = _load(rm_path)
    selection = _load(run_dir / "selection.json")
    sweep_path = Path(a.sweep)
    sweep = _load(sweep_path)
    trainset_dir = Path(a.trainset_dir)
    ts_manifest_path = trainset_dir / "TRAINSET_MANIFEST.json"
    ts_manifest = _load(ts_manifest_path)
    merge_check = _load(Path(a.merge_check)) if a.merge_check else _load(Path(a.export).with_name("merge_check.json"))
    taxonomy = Path(a.taxonomy)
    mining_manifest = Path(a.mining_manifest)
    for p in (taxonomy, mining_manifest):
        if not p.is_file():
            raise BadInput(f"{p} missing")
    cfg = manifest.get("config", {})
    lora = manifest.get("lora") or {}
    mc_cfg = merge_check.get("adapter_config", {})
    base = manifest.get("base_model") or {}
    sweep_winner = sweep.get("winner") if isinstance(sweep, dict) else None
    return {
        "tag": final_tag,
        "ollama_digest": digest,
        "candidate_tag": export["candidate_tag"],
        "gguf_path": export["gguf_path"],
        "gguf_text_sha256": export["text_q8_0_sha256"],
        "projector_sha256": export["projector_sha256"],
        "adapter_path": selection["chosen_adapter"],
        "adapter_sha256": selection["adapter_sha256"],
        "adapter_config": {
            "r": mc_cfg.get("r", cfg.get("r")),
            "alpha": mc_cfg.get("lora_alpha", cfg.get("alpha")),
            "use_rslora": mc_cfg.get("use_rslora"),
            "target_regex": mc_cfg.get("target_modules"),
            "n_wrapped": lora.get("n_wrapped", merge_check.get("n_wrapped")),
            "trainable": lora.get("trainable", merge_check.get("trainable")),
        },
        "base_model": {"id": base.get("id"), "revision": base.get("revision")},
        "lock_sha256": manifest.get("lock_sha256"),
        "versions": manifest.get("versions"),
        "llama_cpp_commit": export["llama_cpp_commit"],
        "trainset_path": str(trainset_dir / "trainset.jsonl"),
        "trainset_sha256": manifest.get("trainset_sha256") or ts_manifest.get("trainset_sha256"),
        "trainset_manifest_sha256": sha256_file(ts_manifest_path),
        "taxonomy_sha256": sha256_file(taxonomy),
        "mining_manifest_sha256": sha256_file(mining_manifest),
        "run_id": a.run_id,
        "run_manifest_sha256": sha256_file(rm_path),
        "run_config": cfg,
        "sweep": {"path": str(sweep_path), "sha256": sha256_file(sweep_path), "winner": sweep_winner},
        "selection": {
            "chosen_step": selection["chosen_step"],
            "rule": selection.get("rule"),
            "tie_delta": selection.get("tie_delta"),
        },
        "scorer_version": gates.get("scorer_version"),
        "scorer_fingerprint": gates.get("scorer_fingerprint"),
        "scores": {
            "dev_mini_eval_hf": {"delta_vs_base": selection.get("delta_vs_base"), "per_step": selection.get("per_step")},
            "dev_full_ollama": {
                "stock": gates["dev"]["stock_rate"],
                "candidate": gates["dev"]["candidate_rate"],
                "delta": gates["dev"]["delta"],
                "ci": [gates["dev"]["bootstrap"].get("ci_low"), gates["dev"]["bootstrap"].get("ci_high")],
            },
            "regression": {"per_group": {"control": gates["regression"]["control"]["per_flag"], "target": gates["regression"]["target"]["per_class"]}},
        },
        "gates": {
            "thresholds": gates["thresholds"],
            "results": {k: gates[k] for k in ("regression", "export", "dev")},
            "pass": gates["pass"],
        },
        "benchmark": dict(registry.BENCHMARK_PENDING),
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cmd_register(a: argparse.Namespace) -> int:
    _check_run_id(a.run_id)
    gates = _load(a.gates)
    export = _load(a.export)
    cand = f"{CAND_PREFIX}:{a.run_id}"
    final = f"{FINAL_PREFIX}:{a.run_id}"
    work = Path(a.export).parent

    if not gates.get("pass"):
        r = _ollama("rm", cand)
        write_json(
            work / "registration.json",
            {
                "run_id": a.run_id,
                "outcome": "rejected",
                "candidate_tag": cand,
                "candidate_removed": r.returncode == 0,
                "failing": {k: gates.get(k, {}).get("reasons") for k in ("regression", "export", "dev") if not gates.get(k, {}).get("pass")},
                "gates": gates,
            },
        )
        print(f"gates failed: removed {cand}, wrote registration.json")
        return EXIT_GATE

    r = _ollama("cp", cand, final)
    if r.returncode != 0:
        raise BadInput(f"ollama cp {cand} {final} failed: {r.stderr.strip()}")
    digest = _model_digest(a.host, final)
    entry = build_entry(a, gates, export, final, digest)
    problems = registry.validate(entry) + registry.verify_hashes(entry, repo_root())
    if problems:
        _ollama("rm", final)
        raise BadInput("registry entry rejected: " + "; ".join(problems))
    registry.append_entry(Path(a.registry), entry)
    _ollama("rm", cand)
    write_json(work / "registration.json", {"run_id": a.run_id, "outcome": "registered", "tag": final, "ollama_digest": digest})
    print(f"registered {final} ({digest})")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gates", help="evaluate the three gates and write gates.json")
    g.add_argument("--run-id", required=True)
    g.add_argument("--run-dir", required=True)
    g.add_argument("--regression", default="data/corpus/regression/regression.jsonl")
    for name in ("reg-stock", "reg-cand", "dev-stock", "dev-cand", "dev-stock-sidecar", "dev-cand-sidecar", "hf-base", "hf-tuned", "hf-merged"):
        g.add_argument(f"--{name}", required=True)
    g.add_argument("--ids", default="data/train/dev_eval_ids.txt")
    g.add_argument("--merge-check", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_gates)

    r = sub.add_parser("register", help="register on pass, roll back on failure")
    r.add_argument("--run-id", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--gates", required=True)
    r.add_argument("--export", required=True)
    r.add_argument("--trainset-dir", required=True)
    r.add_argument("--sweep", required=True)
    r.add_argument("--registry", default="data/models/REGISTRY.json")
    r.add_argument("--merge-check", default=None)
    r.add_argument("--taxonomy", default="data/mining/taxonomy.json")
    r.add_argument("--mining-manifest", default="data/mining/MINING_MANIFEST.json")
    r.add_argument("--host", default=DEFAULT_HOST)
    r.set_defaults(fn=cmd_register)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD


if __name__ == "__main__":
    raise SystemExit(main())
