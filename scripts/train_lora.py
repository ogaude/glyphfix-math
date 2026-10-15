"""LoRA training, out-of-process dev evaluation and the preflight probe on the real model.

Usage (always under the resource guard; the GPU is shared):
  uv run python -m lightonocr_math.guard --vram-gb 14.5 --ram-gb 8 --timeout 1800 --vram-floor-gb 0.1 -- \
      .venv-lighton/bin/python -m scripts.train_lora train --run-id lr2e-4-a8 --trainset data/train/sets/full/trainset.jsonl \
      --lr 2e-4 --alpha 8 --rank 32 --epochs 1 [--resume] [--max-opt-steps N]
  uv run python -m lightonocr_math.guard --vram-gb 14.5 --ram-gb 8 --timeout 1800 --vram-floor-gb 0.1 -- \
      .venv-lighton/bin/python -m scripts.train_lora eval --base --out-dir data/train/dev_eval
  ... eval --run-dir data/train/runs/RUN [--steps all|100,200]       (every saved eval adapter of a run)
  ... eval --adapter DIR --name NAME --out-dir DIR                   (one adapter)
  ... eval --model-dir MERGED_HF_DIR --name NAME --out-dir DIR       (a merged HF folder)
  ... probe --trainset data/train/sets/full/trainset.jsonl --out data/train/probe.json

train refuses to start when Ollama holds a model, when the training set does not match its manifest hash, or when a
row is a dev, dev-mini-eval, regression or non-train record. It wraps the language model with 200 LoRA modules
(rsLoRA, vision tower frozen), enables gradient checkpointing (use_reentrant False), checks the label mask on the
first row, and runs the model-agnostic loop in lightonocr_math.train.run. eval writes scorer-format predictions
({id, paper_id, gt_latex, pred_latex}) and a sidecar with token counts; it only reads dev papers. probe measures
peak memory and timing on the longest-target and largest-crop batches and fails above 14.5 GiB.

Exit codes: 0 complete, 1 nan / error / mask failure / wrong module count / probe over budget,
2 refused (no GPU, leak, hash mismatch, bad ids), 75 stopped or Ollama busy (resumable).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lightonocr_math import splits
from lightonocr_math.corpus import gate
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json
from lightonocr_math.ids import base_id
from lightonocr_math.paths import repo_root
from lightonocr_math.train import data, run
from scripts.spikes.common import EVIDENCE_DIR, gpu_state, strip_fences
from scripts.spikes.ollama_infer import DEFAULT_HOST, _loaded_models

ENV_VERIFY = EVIDENCE_DIR / "env_verify.json"
MANIFEST_FILE = "TRAINSET_MANIFEST.json"
DEV_EVAL_IDS = "data/train/dev_eval_ids.txt"
DEV_RECORDS = "data/corpus/dev/records.jsonl"
REGRESSION = "data/corpus/regression/regression.jsonl"
MIN_PX = 65536
MAX_PX = 602112
MAX_NEW = 1024
PEAK_BUDGET_GIB = 14.5
EXPECTED_WRAPPED = 200
EXPECTED_TRAINABLE = 60_948_480
END_OF_TURN = "<|im_end|>"

EXIT_OK, EXIT_FAIL, EXIT_REFUSED, EXIT_BUSY = 0, 1, 2, 75


class Refused(Exception):
    """A pre-flight check failed; the message says which."""


def _clean_state(state: dict) -> dict:
    out = dict(state)
    if isinstance(out.get("holders"), str):
        out["holders"] = " | ".join(x.strip() for x in out["holders"].replace("\t", " ").splitlines() if x.strip())
    return out


def _fla_default() -> str:
    try:
        return "on" if json.load(open(ENV_VERIFY))["fla"]["kernel_ok"] else "off"
    except (OSError, KeyError, ValueError):
        return "off"


def ollama_busy(host: str = DEFAULT_HOST) -> list[str]:
    """Names of models Ollama currently holds; empty when it is idle or unreachable."""
    return [m for m in _loaded_models(host) if m]


def read_ids(path: Path | str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def verify_trainset(
    path: Path | str,
    *,
    dev_ids: set[str],
    eval_ids: set[str],
    regression_ids: set[str],
    regression_rows: list[dict] | None = None,
    fs: splits.FrozenSplit | None = None,
) -> tuple[list[dict], dict]:
    """Rows and manifest of a training set after the hash, split, dev, dev-eval and regression checks."""
    path = Path(path)
    mpath = path.parent / MANIFEST_FILE
    if not mpath.is_file():
        raise ValueError(f"{mpath} missing; refusing a training set without a manifest")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if actual != manifest.get("trainset_sha256"):
        raise ValueError(f"{path} sha256 {actual[:12]} does not match {MANIFEST_FILE} {str(manifest.get('trainset_sha256'))[:12]}")
    rows = data.read_trainset(path)
    ids = {r["id"] for r in rows}
    for label, bad in (("dev", dev_ids), ("dev-eval", eval_ids), ("regression", regression_ids)):
        hit = sorted(ids & bad)
        if hit:
            raise splits.LeakageError(f"training set contains {len(hit)} {label} record(s): {hit[:5]}", {i: label for i in hit})
    if regression_rows:
        reg_mapped = [{"id": r["id"], "latex": r["gt_latex"]} for r in regression_rows]
        overlap = gate.regression_overlap(rows, reg_mapped)
        if overlap:
            raise splits.LeakageError(
                f"training set has {len(overlap)} row(s) equal to a regression record: {sorted(overlap)[:5]}", {i: "regression" for i in overlap}
            )
    papers = sorted({base_id(r["paper_id"]) for r in rows})
    splits.assert_disjoint(papers, context="train_lora.verify_trainset", fs=fs)
    train_papers = set(splits.load_train_ids(papers, fs))
    outside = sorted(set(papers) - train_papers)
    if outside:
        raise splits.LeakageError(f"{len(outside)} paper(s) are not train papers: {outside[:5]}", {p: "not train" for p in outside})
    return rows, manifest


def build_config(args: argparse.Namespace, *, trainset_sha256: str = "") -> run.RunConfig:
    return run.RunConfig(
        run_id=args.run_id,
        trainset=str(args.trainset),
        trainset_sha256=trainset_sha256,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        max_opt_steps=args.max_opt_steps,
        min_px=args.min_px,
        max_px=args.max_px,
    )


def _context_sets(root: Path) -> tuple[set[str], set[str], set[str], list[dict]]:
    dev = {r["id"] for r in read_jsonl(root / DEV_RECORDS)}
    eval_p = root / DEV_EVAL_IDS
    eval_ids = set(read_ids(eval_p)) if eval_p.is_file() else set()
    reg_rows = read_jsonl(root / REGRESSION)
    return dev, eval_ids, {r["id"] for r in reg_rows}, reg_rows


def _busy_exit(models: list[str]) -> int:
    print(f"refusing: Ollama holds {models}; unload it (ollama stop) before GPU work", file=sys.stderr)
    return EXIT_BUSY


def load_eval_records(ids_path: Path, records_path: Path) -> list[dict]:
    """Dev records for the mini-eval ids; every paper must be a dev paper."""
    ids = read_ids(ids_path)
    by_id = {r["id"]: r for r in read_jsonl(records_path)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise Refused(f"{len(missing)} id(s) not in {records_path}: {missing[:5]}")
    recs = [by_id[i] for i in ids]
    papers = sorted({base_id(r["arxiv_id"]) for r in recs})
    dev_papers = set(splits.load_dev_ids(papers))
    outside = sorted(set(papers) - dev_papers)
    if outside:
        raise Refused(f"{len(outside)} paper(s) are not dev papers: {outside[:5]}")
    return recs


def eval_row(rec: dict, res: dict) -> tuple[dict, dict]:
    """(scorer prediction row, sidecar row) for one generation."""
    pred = {"id": rec["id"], "paper_id": base_id(rec["arxiv_id"]), "gt_latex": rec["latex"], "pred_latex": strip_fences(res["text"])}
    side = {
        "id": rec["id"], "raw_text": res["text"], "n_prompt_tokens": res["n_prompt_tokens"], "image_tokens": res["image_tokens"],
        "grid_thw": res["grid_thw"], "finish": res["finish"], "n_new_tokens": res["n_new_tokens"],
        "sent_width": res["resized_hw"][1], "sent_height": res["resized_hw"][0],
    }  # fmt: skip
    return pred, side


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path)} if path.is_file() else set()


def generate_file(model, processor, recs: list[dict], out_stem: Path, *, resume: bool) -> int:
    """Generate for every record into OUT_STEM.pred.jsonl and OUT_STEM.sidecar.jsonl; returns rows written."""
    from lightonocr_math.train import infer

    pred_p = out_stem.with_name(out_stem.name + ".pred.jsonl")
    side_p = out_stem.with_name(out_stem.name + ".sidecar.jsonl")
    pred_p.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        pred_p.unlink(missing_ok=True)
        side_p.unlink(missing_ok=True)
    done = _done_ids(pred_p) & _done_ids(side_p)
    n = 0
    for rec in recs:
        if rec["id"] in done:
            continue
        img = data.load_image({"image": rec["image"]})
        res = infer.generate_one(model, processor, img, min_px=MIN_PX, max_px=MAX_PX, max_new_tokens=MAX_NEW)
        pred, side = eval_row(rec, res)
        _append_jsonl(side_p, side)
        _append_jsonl(pred_p, pred)
        n += 1
    return n


def _require_gpu() -> bool:
    import torch

    if not torch.cuda.is_available():
        print("no GPU available; refusing to run on CPU", file=sys.stderr)
        return False
    return True


def _step_dirs(run_dir: Path, steps: str) -> list[tuple[int, Path]]:
    found = []
    for p in (run_dir / "eval").glob("step-*"):
        if (p / "adapter").is_dir():
            found.append((int(p.name[5:]), p / "adapter"))
    found.sort()
    if steps != "all":
        want = {int(s) for s in steps.split(",") if s}
        found = [f for f in found if f[0] in want]
    return found


def cmd_eval(a: argparse.Namespace) -> int:
    root = repo_root()
    busy = ollama_busy()
    if busy:
        return _busy_exit(busy)
    try:
        recs = load_eval_records(root / a.ids, root / a.records)
    except (Refused, splits.LeakageError, OSError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    modes = [bool(a.base), bool(a.adapter), bool(a.model_dir), bool(a.run_dir)]
    if sum(modes) != 1:
        print("give exactly one of --base, --adapter, --model-dir, --run-dir", file=sys.stderr)
        return EXIT_REFUSED
    if (a.adapter or a.model_dir) and not a.name:
        print("--name is required with --adapter / --model-dir", file=sys.stderr)
        return EXIT_REFUSED
    if a.run_dir:
        steps = _step_dirs(Path(a.run_dir), a.steps)
        if not steps:
            print(f"no eval adapters under {a.run_dir}/eval", file=sys.stderr)
            return EXIT_REFUSED
        out_dir = Path(a.run_dir) / "dev_eval"
    else:
        out_dir = Path(a.out_dir)
    if not _require_gpu():
        return EXIT_REFUSED

    from lightonocr_math.train import infer

    before = _clean_state(gpu_state())
    model, processor = infer.load_base(a.model_dir) if a.model_dir else infer.load_base()
    model.config.use_cache = True
    t0 = time.perf_counter()
    if a.base:
        n = generate_file(model, processor, recs, out_dir / "base", resume=a.resume)
    elif a.model_dir:
        n = generate_file(model, processor, recs, out_dir / a.name, resume=a.resume)
    elif a.adapter:
        import peft

        pm = peft.PeftModel.from_pretrained(model, a.adapter)
        pm.eval()
        n = generate_file(pm, processor, recs, out_dir / a.name, resume=a.resume)
    else:
        import peft

        n = 0
        pm = None
        for step, adir in steps:
            if pm is not None:
                pm.unload()  # recover the plain base before attaching the next adapter
            pm = peft.PeftModel.from_pretrained(model, str(adir))
            pm.eval()
            n += generate_file(pm, processor, recs, out_dir / f"step-{step}", resume=a.resume)
            print(f"step-{step}: done", flush=True)
        if pm is not None:
            pm.unload()
    print(f"eval wrote {n} rows in {time.perf_counter() - t0:.0f}s; gpu before {before.get('vram_free_bytes')}", flush=True)
    return EXIT_OK


def _setup_model(args: argparse.Namespace, fla: str):
    import peft

    from lightonocr_math.train import infer, lora

    model, processor = infer.load_base()
    model.config.use_cache = False
    model.enable_input_require_grads()
    pm = peft.get_peft_model(model, lora.lora_config(args.rank, args.alpha))
    info = lora.check_targets(pm)
    if info["n_wrapped"] != EXPECTED_WRAPPED:
        raise RuntimeError(f"expected {EXPECTED_WRAPPED} wrapped modules, got {info['n_wrapped']}")
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.train()
    return pm, processor, info


def check_mask(collator, processor, row: dict, image) -> bool:
    """True when the unmasked label tokens of the row decode to the target plus the end-of-turn token."""
    probe = collator([{"image": image, "target_text": row["target_text"]}])
    keep = probe["labels"][0] != -100
    return processor.tokenizer.decode(probe["input_ids"][0][keep]) == row["target_text"] + END_OF_TURN


def cmd_train(a: argparse.Namespace) -> int:
    root = repo_root()
    busy = ollama_busy()
    if busy:
        return _busy_exit(busy)
    try:
        dev, eval_ids, reg_ids, reg_rows = _context_sets(root)
        rows, manifest = verify_trainset(a.trainset, dev_ids=dev, eval_ids=eval_ids, regression_ids=reg_ids, regression_rows=reg_rows)
    except (ValueError, OSError) as e:  # LeakageError is a ValueError
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    import torch

    if not torch.cuda.is_available():
        print("no GPU available; refusing to run on CPU", file=sys.stderr)
        return EXIT_REFUSED

    fla = a.fla or _fla_default()
    if fla == "off":
        sys.modules["fla"] = None  # force the torch reference delta rule before transformers imports
    cfg = build_config(a, trainset_sha256=manifest["trainset_sha256"])
    run_dir = Path(a.run_dir) if a.run_dir else root / "data" / "train" / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    before = _clean_state(gpu_state())
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run.write_manifest(run_dir, run.run_manifest(cfg, outcome="running", started=started, gpu_before=before, fla=fla))

    from lightonocr_math.train import prompt
    from lightonocr_math.train.collate import LabelMaskCollator, label_only_loss

    def fail(reason: str, **extra) -> int:
        print(f"FAIL: {reason}", file=sys.stderr)
        run.write_manifest(run_dir, run.run_manifest(cfg, outcome="error", reason=reason, started=started, gpu_before=before, fla=fla, **extra))
        return EXIT_FAIL

    try:
        pm, processor, info = _setup_model(a, fla)
    except RuntimeError as e:
        return fail(str(e))
    lora_info = {k: info[k] for k in ("n_wrapped", "trainable", "total")}
    collator = LabelMaskCollator(processor, min_px=cfg.min_px, max_px=cfg.max_px, target_key="target_text")
    if not check_mask(collator, processor, rows[0], data.load_image(rows[0])):
        return fail("label mask does not match target + <|im_end|>", lora=lora_info)

    dev_t = next(pm.parameters()).device
    params = [p for p in pm.parameters() if p.requires_grad]
    sizes = [w * h for w, h in (data.policy_dims(r) for r in rows)]

    def batch_plan(epoch: int) -> list[list[int]]:
        return data.bucket_batches(sizes, batch_size=cfg.batch_size, seed=cfg.seed, epoch=epoch)

    def loss_fn(idx: list[int]):
        samples = [{"image": data.load_image(rows[i]), "target_text": rows[i]["target_text"]} for i in idx]
        batch = {k: v.to(dev_t) for k, v in collator(samples).items()}
        return label_only_loss(pm, batch)

    def save_adapter(d: Path) -> None:
        pm.save_pretrained(str(d))

    def load_adapter(d: Path) -> None:
        import peft

        peft.set_peft_model_state_dict(pm, _load_adapter_state(Path(d)))

    stop = {"flag": False}

    def _on_signal(_sig, _frm) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    def mem_stats() -> dict:
        gib = 1024**3
        free, _ = torch.cuda.mem_get_info()
        return {
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3),
            "vram_free_gib": round(free / gib, 3),
        }

    torch.cuda.reset_peak_memory_stats()
    result = run.train_loop(
        cfg, params=params, loss_fn=loss_fn, batch_plan=batch_plan, save_adapter=save_adapter, load_adapter=load_adapter,
        run_dir=run_dir, resume=a.resume, on_step=lambda r: print(json.dumps(r), flush=True),
        stop_flag=lambda: stop["flag"], device_sync=torch.cuda.synchronize, mem_stats=mem_stats,
    )  # fmt: skip
    final_manifest = run.run_manifest(
        cfg, outcome=result["outcome"], reason=result["reason"], started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        opt_steps_done=result["opt_steps_done"], total_opt_steps=result["total_opt_steps"], eval_steps=result["eval_steps"],
        peak_alloc_gib=result["peak_alloc_gib"], mean_s_per_opt_step=result["mean_s_per_opt_step"], lora=lora_info,
        prompt_sha256=_sha256_text(prompt.prompt_text(processor)), gpu_before=before, gpu_after=_clean_state(gpu_state()), fla=fla,
        base_model={"id": _model_id(), "revision": _revision()},
    )  # fmt: skip
    run.write_manifest(run_dir, final_manifest)
    print(f"outcome {result['outcome']} {result['reason']}", flush=True)
    return {"complete": EXIT_OK, "stopped": EXIT_BUSY}.get(result["outcome"], EXIT_FAIL)


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_id() -> str:
    from lightonocr_math.train.infer import MODEL_ID

    return MODEL_ID


def _revision() -> str:
    from lightonocr_math.train.infer import REVISION

    return REVISION


def _load_adapter_state(adir: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(adir / "adapter_model.safetensors"))


def cmd_probe(a: argparse.Namespace) -> int:
    root = repo_root()
    busy = ollama_busy()
    if busy:
        return _busy_exit(busy)
    try:
        dev, eval_ids, reg_ids, reg_rows = _context_sets(root)
        rows, _manifest = verify_trainset(a.trainset, dev_ids=dev, eval_ids=eval_ids, regression_ids=reg_ids, regression_rows=reg_rows)
        recs = load_eval_records(root / a.ids, root / a.records)
    except (ValueError, Refused, OSError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    import torch

    if not torch.cuda.is_available():
        print("no GPU available; refusing to run on CPU", file=sys.stderr)
        return EXIT_REFUSED
    fla = a.fla or _fla_default()
    if fla == "off":
        sys.modules["fla"] = None
    from lightonocr_math.train import infer
    from lightonocr_math.train.collate import LabelMaskCollator, label_only_loss

    try:
        pm, processor, _info = _setup_model(a, fla)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return EXIT_FAIL
    tok = processor.tokenizer
    by_target = sorted(range(len(rows)), key=lambda i: -len(tok(rows[i]["target_text"], add_special_tokens=False)["input_ids"]))[:4]
    areas = [w * h for w, h in (data.policy_dims(r) for r in rows)]
    by_area = sorted(range(len(rows)), key=lambda i: -areas[i])[:4]
    collator = LabelMaskCollator(processor, min_px=MIN_PX, max_px=MAX_PX, target_key="target_text")
    dev_t = next(pm.parameters()).device
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)

    def micro(idx: list[int], step_opt: bool = False) -> float:
        t0 = time.perf_counter()
        samples = [{"image": data.load_image(rows[i]), "target_text": rows[i]["target_text"]} for i in idx]
        batch = {k: v.to(dev_t) for k, v in collator(samples).items()}
        loss = label_only_loss(pm, batch)
        loss.backward()
        if step_opt:
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    worst_batches = [by_target[0:2], by_target[2:4], by_area[0:2], by_area[2:4]] * 2
    worst_times = [micro(b, step_opt=(i == 0)) for i, b in enumerate(worst_batches)]
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    rng = random.Random(0)
    mixed_times = [micro(rng.sample(range(len(rows)), 2)) for _ in range(40)]
    pm.eval()
    pm.config.use_cache = True
    pm.gradient_checkpointing_disable()
    gen_times = []
    for rec in recs[:10]:
        t0 = time.perf_counter()
        infer.generate_one(pm, processor, data.load_image({"image": rec["image"]}), min_px=MIN_PX, max_px=MAX_PX, max_new_tokens=MAX_NEW)
        gen_times.append(time.perf_counter() - t0)

    def med(xs: list[float]) -> float:
        return sorted(xs)[len(xs) // 2]

    s_mixed = med(mixed_times)
    steps_per_epoch = len(rows) / (a.batch_size * a.grad_accum)
    out = {
        "peak_alloc_gib": round(peak_alloc, 3), "peak_reserved_gib": round(peak_res, 3),
        "s_per_micro_step_worst": round(med(worst_times), 3), "s_per_micro_step_mixed": round(s_mixed, 3),
        "s_per_generate": round(med(gen_times), 3),
        "est_hours_per_epoch": round(len(rows) / a.batch_size * s_mixed / 3600, 2),
        "n_rows": len(rows), "opt_steps_per_epoch": round(steps_per_epoch, 1), "fla": fla,
        "pass": peak_alloc <= PEAK_BUDGET_GIB,
    }  # fmt: skip
    write_json(Path(a.out), out)
    print(json.dumps(out), flush=True)
    return EXIT_OK if out["pass"] else EXIT_FAIL


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trainset", default="data/train/sets/full/trainset.jsonl")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--fla", choices=["on", "off"], default=None)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="LoRA training run")
    _add_model_args(t)
    t.add_argument("--run-id", required=True)
    t.add_argument("--run-dir", default=None)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ckpt-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--max-opt-steps", type=int, default=None)
    t.add_argument("--min-px", type=int, default=MIN_PX)
    t.add_argument("--max-px", type=int, default=MAX_PX)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="dev mini-eval generation")
    e.add_argument("--base", action="store_true")
    e.add_argument("--adapter", default=None)
    e.add_argument("--model-dir", default=None)
    e.add_argument("--run-dir", default=None)
    e.add_argument("--steps", default="all")
    e.add_argument("--name", default=None)
    e.add_argument("--out-dir", default="data/train/dev_eval")
    e.add_argument("--ids", default=DEV_EVAL_IDS)
    e.add_argument("--records", default=DEV_RECORDS)
    e.add_argument("--resume", action="store_true")
    e.set_defaults(func=cmd_eval)

    p = sub.add_parser("probe", help="peak memory and timing preflight")
    _add_model_args(p)
    p.add_argument("--out", default="data/train/probe.json")
    p.add_argument("--ids", default=DEV_EVAL_IDS)
    p.add_argument("--records", default=DEV_RECORDS)
    p.set_defaults(func=cmd_probe)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
