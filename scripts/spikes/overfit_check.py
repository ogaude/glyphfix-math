"""Overfit check on 16 legacy train crops: proves masking, template, LoRA targets and gradient paths.

Run in .venv-lighton under the resource guard (shared GPU):
  uv run python -m lightonocr_math.guard --vram-gb 10 --ram-gb 8 --timeout 1800 --vram-floor-gb 0.1 -- \
      .venv-lighton/bin/python -m scripts.spikes.overfit_check

A pass means the training core is correct, not that the model got better. Evidence holds counts and ids only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from scripts.spikes.common import EVIDENCE_DIR, gpu_state, load_legacy, write_json

ENV_VERIFY = EVIDENCE_DIR / "env_verify.json"


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


def _delta_rule_module() -> str:
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    fn = mq.torch_chunk_gated_delta_rule
    cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
    return getattr(cells.get("implementation"), "__module__", None) or "none"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--min-px", type=int, default=65536)
    ap.add_argument("--max-px", type=int, default=401408)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--fla", choices=["on", "off"], default=_fla_default())
    ap.add_argument("--out", default=str(EVIDENCE_DIR / "overfit.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.fla == "off":
        sys.modules["fla"] = None  # force the torch reference delta-rule before transformers imports

    import torch
    from PIL import Image

    from lightonocr_math.normalize import equivalent
    from lightonocr_math.train import infer, lora
    from lightonocr_math.train.collate import LabelMaskCollator, label_only_loss
    from scripts.spikes.common import strip_fences

    if not torch.cuda.is_available():
        print("no GPU available; overfit check refuses to run on CPU", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    before = _clean_state(gpu_state())

    rows, _stats = load_legacy("train")
    rows = [r for r in rows if r["width"] * r["height"] <= args.max_px][: args.n]
    ids = [r["id"] for r in rows]
    images = [Image.open(r["image_path"]).convert("RGB") for r in rows]
    for im in images:
        im.load()

    model, processor = infer.load_base()
    result: dict = {
        "n": len(rows), "ids": ids, "fla_arm": args.fla, "fla_active": _delta_rule_module(), "rank": args.rank, "alpha": args.alpha,
        "lr": args.lr, "steps": args.steps, "bs": args.bs, "min_px": args.min_px, "max_px": args.max_px, "seed": args.seed,
        "versions": {"torch": torch.__version__},
    }
    import peft
    import transformers

    result["versions"].update({"transformers": transformers.__version__, "peft": peft.__version__})

    def exact_count(tag: str) -> int:
        model.eval()
        n = 0
        for r, im in zip(rows, images):
            res = infer.generate_one(model, processor, im, min_px=args.min_px, max_px=args.max_px, max_new_tokens=args.max_new)
            n += int(equivalent(r["latex"], strip_fences(res["text"])))
        print(f"{tag}_exact {n}/{len(rows)}", flush=True)
        return n

    result["pre_exact"] = exact_count("pre")

    model.config.use_cache = False
    model.enable_input_require_grads()
    pm = peft.get_peft_model(model, lora.lora_config(args.rank, args.alpha))
    info = lora.check_targets(pm)
    result["lora"] = {k: info[k] for k in ("n_wrapped", "trainable", "total")}
    if info["n_wrapped"] != 200:
        result.update(outcome="fail", reason=f"expected 200 wrapped modules, got {info['n_wrapped']}")
        write_json(args.out, result)
        return 1
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.train()

    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    collate = LabelMaskCollator(processor, min_px=args.min_px, max_px=args.max_px, target_key="target_text")
    samples = [{"image": im, "target_text": r["target_text"]} for r, im in zip(rows, images)]
    dev = next(pm.parameters()).device

    probe = collate(samples[:1])
    keep = probe["labels"][0] != -100
    decoded = processor.tokenizer.decode(probe["input_ids"][0][keep])
    result["mask_ok"] = decoded == rows[0]["target_text"] + "<|im_end|>"

    losses: list[float] = []
    step_times: list[float] = []
    grad_b = grad_a = None
    outcome, reason = "pass", ""
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        start = ((step - 1) * args.bs) % len(samples)
        chunk = [samples[(start + j) % len(samples)] for j in range(args.bs)]
        batch = {k: v.to(dev) for k, v in collate(chunk).items()}
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / 5)
        loss = label_only_loss(pm, batch)
        if not torch.isfinite(loss):
            outcome, reason = "nan", f"non-finite loss at step {step}"
            break
        loss.backward()
        if step == 1:
            grad_b = all(
                p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.float().norm()) > 0
                for n, p in pm.named_parameters() if "lora_B" in n
            )
        if step == 3:
            grad_a = all(
                p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.float().norm()) > 0
                for n, p in pm.named_parameters() if "lora_A" in n
            )
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            outcome, reason = "nan", f"non-finite grad norm at step {step}"
            break
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
        if step % 10 == 0 or step == 1:
            print(f"step {step} loss {losses[-1]:.4f} {step_times[-1]:.2f}s", flush=True)

    result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    late = sorted(step_times[3:])
    result["s_per_step"] = round(late[len(late) // 2], 3) if late else None
    result["losses"] = [round(x, 5) for x in losses]
    result["grad_b_nonzero_step1"] = bool(grad_b)
    result["grad_a_nonzero_step3"] = bool(grad_a)
    if losses:
        result["first_loss"] = losses[0]
        result["last10_mean"] = sum(losses[-10:]) / len(losses[-10:])
        result["loss_ratio"] = result["last10_mean"] / losses[0]

    if outcome == "pass":
        pm.eval()
        pm.config.use_cache = True
        model.config.use_cache = True
        result["post_exact"] = exact_count("post")
    else:
        result["post_exact"] = 0
    pm.gradient_checkpointing_disable() if hasattr(pm, "gradient_checkpointing_disable") else None

    if outcome == "pass":
        problems = []
        if result["loss_ratio"] > 0.25:
            problems.append(f"loss_ratio {result['loss_ratio']:.3f} > 0.25")
        if not (result["grad_b_nonzero_step1"] and result["grad_a_nonzero_step3"]):
            problems.append("a LoRA parameter had no gradient")
        if not result["mask_ok"]:
            problems.append("label mask does not match target + <|im_end|>")
        if result["post_exact"] < 15 or result["post_exact"] < result["pre_exact"]:
            problems.append(f"post_exact {result['post_exact']} (pre {result['pre_exact']})")
        if problems:
            outcome, reason = "fail", "; ".join(problems)
    result["outcome"], result["reason"] = outcome, reason
    result["gpu_state_before"] = before
    result["gpu_state_after"] = _clean_state(gpu_state())
    write_json(args.out, result)
    print(f"outcome {outcome} {reason}", flush=True)
    return 0 if outcome == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
