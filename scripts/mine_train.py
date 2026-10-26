"""Error-mining driver over the training split: seed, infer, score, report, accept.

Usage (from repo root):
  uv run python -m scripts.mine_train seed      # carry over stock predictions already made for train ids
  uv run python -m scripts.mine_train infer --chunk 2000      # stock model on the remaining train crops (resumable)
  uv run python -m scripts.mine_train score     # frozen scorer with render comparison
  uv run python -m scripts.mine_train report    # candidate clusters, rare-class supply, MINING_MANIFEST.json, MINING.md
  uv run python -m scripts.mine_train accept --reviewer NAME --cluster left_right=c01,c02 [--cluster ...]
  uv run python -m scripts.mine_train accept --reviewer NAME --none        # accept no new class

Rules:
  * Train papers only. Every subcommand checks its input and output papers with splits.assert_disjoint and
    load_train_ids, and report/accept finish with gate.assert_side_clean over the mining directory.
  * seed refuses (exit 2, nothing written) when the pool predictions came from a different model digest, template,
    options or pixel policy than the Phase 3 baseline manifest and the live model. It never overwrites a
    non-empty --out.
  * An error row is a scored record with render_identical False; candidate clusters come from the error rows the
    scorer leaves in ``other``. taxonomy.json and labels.jsonl are written only by accept, which needs --reviewer.
  * "Available for training" excludes ids and canonical LaTeX that are in the regression set.

Exit codes: 0 ok, 1 request failures, 2 refused or bad input, 3 render infrastructure failure, 75 / 76 passed
through from the GPU guard / RAM floor.

Known limits: the review table lists ids and token spans only (no images); the reviewer decides which candidate
ids to merge into a named cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from glyphfix_math import mining, score, splits
from glyphfix_math.corpus import gate
from glyphfix_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from glyphfix_math.errors import CLASSES, ERROR_RULES_VERSION
from glyphfix_math.guard import GuardError
from glyphfix_math.pixels import PIXEL_POLICY
from glyphfix_math.render import RenderInfrastructureError
from scripts import baseline_infer
from scripts.spikes.ollama_infer import DEFAULT_HOST, _model_digest

MINING_DIR = "data/mining"
D11_ERROR_ROWS = 3200
D11_SET_ROWS = 8000
IDENTITY_KEYS = ("model_digest", "template_sha256", "options", "pixel_policy")


class Refused(Exception):
    """Input that must not be used (non-train paper, mismatched stock build, missing file)."""


def _assert_train(papers, context: str) -> None:
    uniq = sorted(set(papers))
    splits.assert_disjoint(uniq, context=context)
    allowed = set(splits.load_train_ids(uniq))
    bad = [p for p in uniq if p not in allowed]
    if bad:
        raise Refused(f"{context}: {len(bad)} paper(s) are not train-split papers: {', '.join(bad[:10])}")


def _need(*paths) -> None:
    for p in paths:
        if not Path(p).exists():
            raise Refused(f"missing input: {p}")


def _paper(row: dict) -> str:
    return row["arxiv_id"] if "arxiv_id" in row else row["paper_id"]


def _identity(manifest: dict) -> dict:
    return {k: manifest.get(k) for k in IDENTITY_KEYS}


def _expected_identity(stock_manifest: dict, digest: str | None) -> dict:
    exp = _identity(stock_manifest)
    exp["options"] = dict(baseline_infer.CROP_OPTIONS)
    exp["pixel_policy"] = dict(PIXEL_POLICY)
    if digest is not None:
        exp["model_digest"] = digest
    return exp


def _check_identity(label: str, manifest: dict, expected: dict) -> None:
    got = _identity(manifest)
    diff = [k for k in IDENTITY_KEYS if got[k] != expected[k]]
    if diff:
        raise Refused(f"{label} differs from the stock baseline build in: {', '.join(diff)}")


def _train_records(a) -> list[dict]:
    _need(a.train)
    rows = read_jsonl(a.train)
    _assert_train([_paper(r) for r in rows], "mine_train train records")
    return rows


# ---------------------------------------------------------------- seed
def cmd_seed(a) -> int:
    pool_pred = Path(a.pool_dir) / "stock_pool.pred.jsonl"
    pool_side = Path(a.pool_dir) / "stock_pool.sidecar.jsonl"
    pool_man = Path(a.pool_dir) / "stock_pool.pred.manifest.json"
    _need(pool_pred, pool_side, pool_man, a.stock_manifest)
    out, sidecar = Path(a.out), Path(a.sidecar)
    if out.exists() and out.stat().st_size > 0:
        raise Refused(f"{out} is not empty; seed never overwrites predictions")
    stock = read_json(a.stock_manifest)
    digest = _model_digest(a.host, a.model)
    if digest is None:
        raise Refused(f"cannot read the live digest of model {a.model!r} from {a.host}")
    expected = _expected_identity(stock, digest)
    pool_manifest = read_json(pool_man)
    _check_identity("pool manifest", pool_manifest, expected)
    _check_identity("baseline manifest", stock, expected)
    train_ids = {r["id"] for r in _train_records(a)}
    pool = [p for p in read_jsonl(pool_pred) if "pred_latex" in p]
    keep = [p for p in pool if p["id"] in train_ids]
    _assert_train([p["paper_id"] for p in keep], "mine_train seed output")
    kept = {p["id"] for p in keep}
    side = [s for s in read_jsonl(pool_side) if s["id"] in kept]
    keep.sort(key=lambda r: r["id"])
    side.sort(key=lambda r: r["id"])
    write_jsonl(out, keep)
    write_jsonl(sidecar, side)
    ids_sha = _ids_sha(sorted(kept))
    write_json(
        out.parent / "carried_over.json",
        {"pool_manifest_sha256": sha256_file(pool_man), "n": len(keep), "ids_sha256": ids_sha},
    )
    print(f"seed: carried over {len(keep)} train predictions (pool had {len(pool)})")
    return 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- infer
def _done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {p["id"] for p in read_jsonl(out) if "pred_latex" in p}


def cmd_infer(a) -> int:
    rows = _train_records(a)
    ids = sorted(r["id"] for r in rows)
    out, sidecar = Path(a.out), Path(a.sidecar)
    chunks_dir = out.parent / "chunks"
    man_dir = out.parent / "manifests"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    for n, start in enumerate(range(0, len(ids), a.chunk)):
        chunk = ids[start : start + a.chunk]
        if set(chunk) <= _done_ids(out):
            print(f"chunk {n:02d}: already done")
            continue
        idf = chunks_dir / f"chunk-{n:02d}.txt"
        idf.write_text("".join(f"{i}\n" for i in chunk), encoding="utf-8")
        code = baseline_infer.main(
            [
                "--suite", "mining", "--records", str(a.train), "--model", a.model, "--out", str(out),
                "--sidecar", str(sidecar), "--ids-file", str(idf), "--resume", "--host", a.host,
            ]
        )  # fmt: skip
        man = out.with_name(out.stem + ".manifest.json")
        if man.exists():
            shutil.copyfile(man, man_dir / f"chunk-{n:02d}.manifest.json")
        if code:
            print(f"chunk {n:02d}: baseline_infer exit {code}, stopping", file=sys.stderr)
            return code
    if out.exists():
        _assert_train([p["paper_id"] for p in read_jsonl(out)], "mine_train infer output")
    return 0


# ---------------------------------------------------------------- score
def cmd_score(a) -> int:
    _need(a.out)
    _assert_train([p["paper_id"] for p in read_jsonl(a.out)], "mine_train score input")
    try:
        rp = score.score_file(Path(a.out), Path(a.score_dir), render=True)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except RenderInfrastructureError as e:
        print(f"error: render infrastructure failure: {e}", file=sys.stderr)
        return 3
    print(f"wrote {rp}")
    return 0


# ---------------------------------------------------------------- report
def _regression_ids(a, train: list[dict]) -> set[str]:
    _need(a.regression)
    reg = [{"id": r["id"], "latex": r["gt_latex"]} for r in read_jsonl(a.regression)]
    return gate.regression_overlap([{"id": t["id"], "latex": t["latex"]} for t in train], reg)


def _scored(a) -> tuple[list[dict], Path]:
    rec = Path(a.score_dir) / "records.jsonl"
    _need(rec)
    rows = read_jsonl(rec)
    _assert_train([r["paper_id"] for r in rows], "mine_train scored rows")
    return rows, rec


def _candidate_availability(cands: list[dict], other_err: list[dict], reg_ids: set[str]) -> dict[str, int]:
    pairs = {r["id"]: mining.row_pairs(r) for r in other_err}
    out = {}
    for c in cands:
        if c["id"] == "c00":
            out["c00"] = sum(1 for r in other_err if r.get("exact") is True and r["id"] not in reg_ids)
            continue
        key = (tuple(c["gt_tokens"]), tuple(c["pred_tokens"]))
        out[c["id"]] = sum(1 for r in other_err if key in pairs[r["id"]] and r["id"] not in reg_ids)
    return out


def _report_md(c: dict, supply: list[dict], stats: dict, avail: dict[str, int]) -> str:
    lines = [
        "# Error mining report",
        "",
        (
            f"Scored records: {stats['n_records']}; error rows (render differs): {stats['n_error']}; "
            f"clean: {stats['n_clean']}; ground truth uncompilable: {stats['n_gt_uncompilable']}; "
            f"prediction errors: {stats['n_pred_error']}."
        ),
        f"Spacing-only diagnostic tag: {stats['n_spacing_only']} rows. Candidate families: {stats['families']}.",
        "",
        "## Rare-class supply",
        "",
        "| class | error rows | available after regression exclusion | equal-share target | short |",
        "|---|---|---|---|---|",
    ]
    for s in supply:
        lines.append(f"| {s['class']} | {s['total']} | {s['available']} | {s['target']} | {'yes' if s['short'] else 'no'} |")
    lines += [
        "",
        "## Candidate clusters (review)",
        "",
        "| id | family | gt tokens | pred tokens | records | papers | available | examples | decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in c["candidates"]:
        gt = " ".join(k["gt_tokens"]).replace("|", "\\|")
        pr = " ".join(k["pred_tokens"]).replace("|", "\\|")
        lines.append(
            f"| {k['id']} | {k['family']} | `{gt}` | `{pr}` | {k['n_records']} | {k['n_papers']} | "
            f"{avail.get(k['id'], 0)} | {', '.join(k['example_ids'])} |  |"
        )
    return "\n".join(lines) + "\n"


def cmd_report(a) -> int:
    scored, rec_path = _scored(a)
    train = _train_records(a)
    reg_ids = _regression_ids(a, train)
    errs = [r for r in scored if mining.is_error_row(r)]
    other_err = [r for r in errs if set(r.get("classes") or []) <= {"other"}]
    cands = mining.candidate_clusters(errs, min_papers=5, top_n=40)
    avail = _candidate_availability(cands, other_err, reg_ids)
    rarity = mining.class_rarity(errs)
    counts: Counter = Counter()
    counts_avail: Counter = Counter()
    for r in errs:
        cls = mining.assign_mining_class(r, {}, rarity)
        counts[cls] += 1
        if r["id"] not in reg_ids:
            counts_avail[cls] += 1
    target = D11_ERROR_ROWS // len(CLASSES)
    supply = [
        {
            "class": c,
            "total": counts.get(c, 0),
            "available": counts_avail.get(c, 0),
            "target": target,
            "short": counts_avail.get(c, 0) < target,
        }
        for c in CLASSES
    ]
    families = dict(Counter(k["family"] for k in cands))
    stats = {
        "n_records": len(scored),
        "n_error": len(errs),
        "n_clean": sum(1 for r in scored if mining.is_clean_row(r)),
        "n_gt_uncompilable": sum(1 for r in scored if r.get("render_identical") is None),
        "n_pred_error": sum(1 for r in scored if r.get("pred_error")),
        "n_spacing_only": sum(1 for r in scored if mining.spacing_only(r)),
        "families": families,
    }
    cand_doc = {
        "mining_version": mining.MINING_VERSION,
        "min_papers": 5,
        "top_n": 40,
        "scored_sha256": sha256_file(rec_path),
        "candidates": cands,
        "available_for_training": avail,
        "rare_class_supply": supply,
        "d11": {"set_rows": D11_SET_ROWS, "error_rows": D11_ERROR_ROWS, "classes": len(CLASSES)},
    }
    write_json(a.candidates, cand_doc)

    out_dir = Path(a.out).parent
    man_dir = out_dir / "manifests"
    stock = read_json(a.stock_manifest) if Path(a.stock_manifest).exists() else {}
    chunk_manifests = sorted(man_dir.glob("*.manifest.json")) if man_dir.exists() else []
    expected = _identity(stock)
    for cm in chunk_manifests:
        _check_identity(str(cm), read_json(cm), {**expected, "model_digest": expected["model_digest"]})
    carried = out_dir / "carried_over.json"
    manifest = {
        "mining_version": mining.MINING_VERSION,
        "model": stock.get("model", a.model),
        "model_digest": stock.get("model_digest"),
        "template_sha256": stock.get("template_sha256"),
        "options": stock.get("options"),
        "pixel_policy": stock.get("pixel_policy"),
        "train_records_sha256": sha256_file(a.train),
        "predictions_sha256": sha256_file(a.out) if Path(a.out).exists() else None,
        "sidecar_sha256": sha256_file(a.sidecar) if Path(a.sidecar).exists() else None,
        "carried_over": read_json(carried) if carried.exists() else None,
        "chunk_manifest_sha256": [sha256_file(p) for p in chunk_manifests],
        "scorer_version": score.SCORER_VERSION,
        "scorer_fingerprint": score.SCORER_FINGERPRINT,
        "error_rules_version": ERROR_RULES_VERSION,
        "scored_sha256": sha256_file(rec_path),
        "n_records": stats["n_records"],
        "n_error": stats["n_error"],
        "n_clean": stats["n_clean"],
        "n_gt_uncompilable": stats["n_gt_uncompilable"],
        "n_pred_error": stats["n_pred_error"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": baseline_infer._git_commit(),
    }
    write_json(a.manifest, manifest)
    md = Path(a.report)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_report_md(cand_doc, supply, stats, avail), encoding="utf-8")
    if out_dir.exists():
        gate.assert_side_clean(out_dir)
    print(f"report: {stats['n_error']} error rows, {len(cands)} candidates -> {a.candidates}")
    return 0


# ---------------------------------------------------------------- accept
def _parse_clusters(specs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in specs:
        name, sep, ids = s.partition("=")
        if not sep or not name or not ids:
            raise Refused(f"bad --cluster {s!r}; expected NAME=cID,cID")
        if name in out:
            raise Refused(f"cluster {name!r} given twice")
        out[name] = [i for i in ids.split(",") if i]
    return out


def cmd_accept(a) -> int:
    if not a.reviewer or not a.reviewer.strip():
        raise Refused("accept needs --reviewer")
    if bool(a.cluster) == bool(a.none):
        raise Refused("give --cluster NAME=cID,... (repeatable) or --none")
    accepted_map = _parse_clusters(a.cluster or [])
    _need(a.candidates)
    cand_doc = read_json(a.candidates)
    scored, rec_path = _scored(a)
    train = _train_records(a)
    train_by_id = {r["id"]: r for r in train}
    reg_ids = _regression_ids(a, train)
    try:
        doc = mining.taxonomy_doc(
            cand_doc["candidates"], accepted_map, reviewer=a.reviewer,
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), counts={},
            candidates_sha256=sha256_file(a.candidates), scored_sha256=sha256_file(rec_path),
        )  # fmt: skip
    except ValueError as e:
        raise Refused(str(e)) from e
    patterns = {c["name"]: c["patterns"] for c in doc["accepted_clusters"]}
    labels = mining.build_labels(scored, train_by_id, patterns, reg_ids)
    total: Counter = Counter()
    avail: Counter = Counter()
    for lb in labels:
        if lb["error"]:
            total[lb["mining_class"]] += 1
            if not lb["in_regression"]:
                avail[lb["mining_class"]] += 1
    doc["counts"] = {c: {"total": total.get(c, 0), "available_for_training": avail.get(c, 0)} for c in doc["classes"]}
    _assert_train([lb["paper_id"] for lb in labels], "mine_train accept output")
    write_json(a.taxonomy, doc)
    write_jsonl(a.labels, labels)
    manifest = read_json(a.manifest) if Path(a.manifest).exists() else {}
    manifest["taxonomy_sha256"] = sha256_file(a.taxonomy)
    manifest["labels_sha256"] = sha256_file(a.labels)
    write_json(a.manifest, manifest)
    mdir = Path(a.out).parent
    if mdir.exists():
        gate.assert_side_clean(mdir)
    print(f"accept: {len(accepted_map)} cluster(s), {len(labels)} labels, classes={doc['classes']}")
    return 0


# ---------------------------------------------------------------- cli
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train", default="data/corpus/train/records.jsonl")
    p.add_argument("--pool-dir", default="data/corpus/regression", help="stock_pool.* predictions (seed)")
    p.add_argument("--regression", default="data/corpus/regression/regression.jsonl")
    p.add_argument("--out", default=f"{MINING_DIR}/stock_train.pred.jsonl")
    p.add_argument("--sidecar", default=f"{MINING_DIR}/stock_train.sidecar.jsonl")
    p.add_argument("--model", default="lighton-ocr3-q8-16k")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--score-dir", default=f"{MINING_DIR}/score")
    p.add_argument("--candidates", default=f"{MINING_DIR}/taxonomy_candidates.json")
    p.add_argument("--taxonomy", default=f"{MINING_DIR}/taxonomy.json")
    p.add_argument("--labels", default=f"{MINING_DIR}/labels.jsonl")
    p.add_argument("--manifest", default=f"{MINING_DIR}/MINING_MANIFEST.json")
    p.add_argument("--stock-manifest", default="data/benchmark/baseline/stock_q8.pred.manifest.json")
    p.add_argument("--report", default=".planning/phases/04-error-mining-and-lora-training/reports/MINING.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed", help="carry over stock predictions already made for train ids")
    _common(p)
    p.set_defaults(fn=cmd_seed)
    p = sub.add_parser("infer", help="stock inference on train crops in resumable chunks")
    _common(p)
    p.add_argument("--chunk", type=int, default=2000)
    p.set_defaults(fn=cmd_infer)
    p = sub.add_parser("score", help="score predictions with the frozen scorer and render comparison")
    _common(p)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("report", help="candidate clusters, supply table, manifest, markdown report")
    _common(p)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("accept", help="write taxonomy.json and labels.jsonl after review")
    _common(p)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--cluster", action="append", default=[], metavar="NAME=cID,cID")
    p.add_argument("--none", action="store_true", help="accept no new class")
    p.set_defaults(fn=cmd_accept)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (splits.LeakageError, Refused, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
