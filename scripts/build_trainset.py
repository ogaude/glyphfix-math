"""Build the stratified training set, a sweep subset and the dev mini-eval id list.

Usage (from repo root; the default token counter needs transformers, so use .venv-lighton):
  HF_HUB_OFFLINE=1 .venv-lighton/bin/python -m scripts.build_trainset build --out-dir data/train/sets/full
  HF_HUB_OFFLINE=1 .venv-lighton/bin/python -m scripts.build_trainset subset --from data/train/sets/full --n 1500 \
      --out-dir data/train/sets/sweep1500
  HF_HUB_OFFLINE=1 .venv-lighton/bin/python -m scripts.build_trainset devsubset --out data/train/dev_eval_ids.txt

build reads the mining labels and the reviewed taxonomy (data/mining), the train records, the regression set and the
dev records (ids only), and writes OUT_DIR/trainset.jsonl plus TRAINSET_MANIFEST.json holding the sha256 of every input
and of the set. A written set is immutable: a non-empty OUT_DIR is refused unless --force. The output directory is
scanned with gate.assert_side_clean. subset draws a group-proportional subset of a built set and records the parent
set's hash. devsubset writes sorted dev ids (one per line) and DEV_EVAL_MANIFEST.json next to them.
--token-counter qwen loads the pinned tokenizer offline; chars uses len(target) // 3 for tests and dry runs.
Exit codes: 0 ok, 2 refused (unreviewed taxonomy, leakage, existing set) or missing input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from lightonocr_math import splits, trainset
from lightonocr_math.corpus import gate
from lightonocr_math.corpus._io import read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl
from lightonocr_math.paths import repo_root
from lightonocr_math.score import SCORER_FINGERPRINT, SCORER_VERSION
from lightonocr_math.train.infer import MODEL_ID, REVISION

SET_FILE = "trainset.jsonl"
MANIFEST_FILE = "TRAINSET_MANIFEST.json"


class Refused(Exception):
    pass


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def _token_len(kind: str):
    if kind == "chars":
        return lambda s: len(s) // 3
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def _need(*paths: str) -> None:
    for p in paths:
        if not Path(p).exists():
            raise Refused(f"missing input: {p}")


def _check_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise Refused(f"{out_dir} is not empty; a built set is immutable (use --force to overwrite)")


def _write_set(out_dir: Path, rows: list[dict], manifest: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_path = out_dir / SET_FILE
    write_jsonl(set_path, rows)
    manifest["trainset_sha256"] = sha256_file(set_path)
    manifest["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["git_commit"] = _git_commit()
    write_json(out_dir / MANIFEST_FILE, manifest)
    try:
        gate.assert_side_clean(out_dir)
    except splits.LeakageError:
        set_path.unlink(missing_ok=True)
        (out_dir / MANIFEST_FILE).unlink(missing_ok=True)
        raise
    return manifest


def _print_summary(manifest: dict) -> None:
    c = manifest["counts"]
    print(f"trainset: {manifest['n_rows']} rows, {manifest['n_papers']} papers, sha256 {manifest['trainset_sha256'][:12]}")
    print(f"  group: {c['group']}")
    print(f"  error_class: {c['error_class']}")
    print(f"  control_glyph: {c['control_glyph']}")
    print(f"  replay_flag: {c['replay_flag']}")
    if manifest.get("shortfalls"):
        print(f"  shortfalls: {manifest['shortfalls']}")


def cmd_build(a: argparse.Namespace) -> int:
    _need(a.train, a.labels, a.taxonomy, a.regression, a.dev)
    out_dir = Path(a.out_dir)
    _check_out_dir(out_dir, a.force)
    taxonomy = json.loads(Path(a.taxonomy).read_text(encoding="utf-8"))
    if taxonomy.get("reviewed") is not True:
        raise Refused(f"{a.taxonomy}: taxonomy not reviewed (reviewed != true)")
    train, labels = read_jsonl(a.train), read_jsonl(a.labels)
    regression = read_jsonl(a.regression)
    dev_ids = {r["id"] for r in read_jsonl(a.dev)}
    rows, body = trainset.build(
        train, labels, taxonomy, regression, dev_ids, token_len=_token_len(a.token_counter),
        total=a.total, replay_frac=a.replay_frac, control_frac=a.control_frac,
        max_per_paper=a.max_per_paper, max_target_tokens=a.max_target_tokens, seed=a.seed,
    )  # fmt: skip
    manifest = {
        **body,
        "sources": {
            "train_records_sha256": sha256_file(a.train),
            "labels_sha256": sha256_file(a.labels),
            "taxonomy_sha256": sha256_file(a.taxonomy),
            "regression_sha256": sha256_file(a.regression),
            "dev_records_sha256": sha256_file(a.dev),
        },
        "tokenizer": {"model_id": MODEL_ID, "revision": REVISION, "counter": a.token_counter},
        "scorer_version": SCORER_VERSION,
        "scorer_fingerprint": SCORER_FINGERPRINT,
    }
    _print_summary(_write_set(out_dir, rows, manifest))
    return 0


def cmd_subset(a: argparse.Namespace) -> int:
    parent = Path(a.from_dir)
    _need(parent / SET_FILE, parent / MANIFEST_FILE)
    out_dir = Path(a.out_dir)
    _check_out_dir(out_dir, a.force)
    pm = json.loads((parent / MANIFEST_FILE).read_text(encoding="utf-8"))
    if sha256_file(parent / SET_FILE) != pm["trainset_sha256"]:
        raise Refused(f"{parent / SET_FILE} does not match its manifest hash")
    rows = trainset.subset(read_jsonl(parent / SET_FILE), n=a.n, seed=a.seed)
    classes = list(pm["counts"]["error_class"])
    manifest = {
        **trainset.summarize(rows, classes=classes),
        "builder_version": trainset.BUILDER_VERSION,
        "seed": a.seed,
        "params": {**pm["params"], "subset_n": a.n},
        "sources": pm["sources"],
        "tokenizer": pm["tokenizer"],
        "scorer_version": pm["scorer_version"],
        "scorer_fingerprint": pm["scorer_fingerprint"],
        "parent_sha256": pm["trainset_sha256"],
    }
    _print_summary(_write_set(out_dir, rows, manifest))
    return 0


def cmd_devsubset(a: argparse.Namespace) -> int:
    _need(a.dev)
    dev = read_jsonl(a.dev)
    ids = trainset.dev_subset(dev, n=a.n, per_paper=a.per_paper, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{i}\n" for i in ids)
    out.write_text(text, encoding="utf-8")
    by_id = {r["id"]: r for r in dev}
    per_flag = Counter(trainset.primary_flag(by_id[i]["flags"]) for i in ids)
    write_json(
        out.parent / "DEV_EVAL_MANIFEST.json",
        {
            "n": len(ids),
            "per_flag": dict(per_flag),
            "max_per_paper": a.per_paper,
            "seed": a.seed,
            "dev_records_sha256": sha256_file(a.dev),
            "ids_sha256": sha256_bytes(text.encode("utf-8")),
        },
    )
    print(f"dev mini-eval: {len(ids)} ids, per_flag={dict(per_flag)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the full training set")
    b.add_argument("--train", default="data/corpus/train/records.jsonl")
    b.add_argument("--labels", default="data/mining/labels.jsonl")
    b.add_argument("--taxonomy", default="data/mining/taxonomy.json")
    b.add_argument("--regression", default="data/corpus/regression/regression.jsonl")
    b.add_argument("--dev", default="data/corpus/dev/records.jsonl")
    b.add_argument("--total", type=int, default=trainset.TOTAL)
    b.add_argument("--replay-frac", type=float, default=trainset.REPLAY_FRAC)
    b.add_argument("--control-frac", type=float, default=trainset.CONTROL_FRAC)
    b.add_argument("--max-per-paper", type=int, default=trainset.MAX_PER_PAPER)
    b.add_argument("--max-target-tokens", type=int, default=trainset.MAX_TARGET_TOKENS)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--token-counter", choices=("qwen", "chars"), default="qwen")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("subset", help="group-proportional subset of a built set")
    s.add_argument("--from", dest="from_dir", required=True)
    s.add_argument("--n", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_subset)
    d = sub.add_parser("devsubset", help="fixed dev mini-eval id list")
    d.add_argument("--dev", default="data/corpus/dev/records.jsonl")
    d.add_argument("--n", type=int, default=150)
    d.add_argument("--per-paper", type=int, default=3)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="data/train/dev_eval_ids.txt")
    d.set_defaults(fn=cmd_devsubset)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (Refused, splits.LeakageError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
