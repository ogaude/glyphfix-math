"""One training-fit grid cell: worst-case batch, label-only loss, peak memory and seconds per step.

Run in .venv-lighton under the resource guard (shared GPU), normally launched by train_fit_grid:
  .venv-lighton/bin/python -m scripts.spikes.train_fit_cell --bs 1 --max-px 200704 --fla off --cell-id c --result out.json

The fla choice is made at import time, so every cell is a fresh process. The outcome is data: any
recorded outcome exits 0; a nonzero exit means the row could not be written (or no GPU, exit 2).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--max-px", type=int, required=True)
    ap.add_argument("--fla", choices=["on", "off"], required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--gc", choices=["on", "off"], default="on")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--quant", choices=["none", "nf4"], default="none")
    ap.add_argument("--alloc-conf", default="")
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--result", required=True)
    return ap.parse_args(argv)


def _delta_rule_module() -> str:
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    fn = mq.torch_chunk_gated_delta_rule
    cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
    return getattr(cells.get("implementation"), "__module__", None) or "none"


def _tail(text: str, n: int = 2000) -> str:
    return text[-n:]


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.alloc_conf:
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = args.alloc_conf

    fla_import_error = ""
    if args.fla == "off":
        sys.modules["fla"] = None  # force the torch reference delta-rule before transformers imports
    else:
        try:
            import fla  # noqa: F401
        except Exception:  # noqa: BLE001 - recorded, not fatal
            fla_import_error = _tail(traceback.format_exc())

    import torch

    from scripts.spikes.common import load_legacy, write_json

    row: dict = {
        "cell_id": args.cell_id, "bs": args.bs, "max_px": args.max_px, "fla_arm": args.fla, "fla_active": None,
        "delta_rule_impl": None, "fla_import_error": fla_import_error, "rank": args.rank, "alpha": args.alpha,
        "gc": args.gc, "optim": args.optim, "quant": args.quant, "alloc_conf": args.alloc_conf, "steps": args.steps,
        "steps_done": 0, "outcome": "error", "error_tail": "",
    }
    if not torch.cuda.is_available():
        row.update(outcome="no-gpu", error_tail="torch.cuda.is_available() is False")
        write_json(args.result, row)
        return 2

    GIB = 2**30
    row["torch_mem_free_start_gib"] = round(torch.cuda.mem_get_info()[0] / GIB, 3)
    losses: list[float] = []
    step_times: list[float] = []
    try:
        import peft
        import transformers
        from PIL import Image

        from lightonocr_math.train import infer, lora, prompt
        from lightonocr_math.train.collate import LabelMaskCollator, label_only_loss

        row["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
        try:
            import importlib.metadata as md

            row["versions"]["fla"] = md.version("flash-linear-attention")
        except Exception:  # noqa: BLE001
            row["versions"]["fla"] = None

        rows, _ = load_legacy("train")
        if args.quant == "nf4":
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForImageTextToText.from_pretrained(
                infer.MODEL_ID, revision=infer.REVISION, quantization_config=qc, attn_implementation="sdpa",
                device_map={"": 0}, local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(infer.MODEL_ID, revision=infer.REVISION, local_files_only=True)
        else:
            model, processor = infer.load_base()
        row["delta_rule_impl"] = _delta_rule_module()
        row["fla_active"] = row["delta_rule_impl"].startswith("fla.")

        # worst case: largest crop resized to the budget, longest label
        biggest = max(rows, key=lambda r: r["width"] * r["height"])
        tok = processor.tokenizer
        longest = max(rows, key=lambda r: len(tok(r["target_text"])["input_ids"]))
        img = prompt.resize_to_budget(Image.open(biggest["image_path"]).convert("RGB"), args.max_px)
        collate = LabelMaskCollator(processor, min_px=prompt.MIN_PIXELS, max_px=args.max_px, target_key="target_text")
        batch_cpu = collate([{"image": img, "target_text": longest["target_text"]} for _ in range(args.bs)])
        row["image_tokens"] = prompt.image_tokens(batch_cpu["image_grid_thw"][:1])
        row["label_tokens"] = int((batch_cpu["labels"][0] != -100).sum())
        row["seq_len"] = int(batch_cpu["input_ids"].shape[1])

        model.config.use_cache = False
        if args.quant == "nf4":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.enable_input_require_grads()
        pm = peft.get_peft_model(model, lora.lora_config(args.rank, args.alpha))
        lora.check_targets(pm)
        if args.gc == "on":
            pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm.train()
        params = [p for p in pm.parameters() if p.requires_grad]
        if args.optim == "adamw8bit":
            import bitsandbytes as bnb

            opt = bnb.optim.PagedAdamW8bit(params, lr=1e-4, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0, fused=True)
        dev = next(pm.parameters()).device
        batch = {k: v.to(dev) for k, v in batch_cpu.items()}

        torch.cuda.reset_peak_memory_stats()
        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()
            loss = label_only_loss(pm, batch)
            if not torch.isfinite(loss):
                row.update(outcome="nan", error_tail=f"non-finite loss at step {step}")
                break
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):
                row.update(outcome="nan", error_tail=f"non-finite grad norm at step {step}")
                break
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach()))
            row["steps_done"] = step
            if step == args.warmup:
                row["peak_warmup_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
                row["peak_warmup_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
            print(f"step {step} loss {losses[-1]:.4f} {step_times[-1]:.2f}s", flush=True)
        else:
            row["outcome"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row.update(outcome="oom", error_tail=_tail(traceback.format_exc()))
    except Exception:  # noqa: BLE001 - the outcome is data
        tb = traceback.format_exc()
        low = tb.lower()
        if "out of memory" in low:
            kind = "oom"
        elif re.search(r"\bfla[./]|/fla/|triton", low):
            kind = "fla-error"
        else:
            kind = "error"
        row.update(outcome=kind, error_tail=_tail(tb))

    row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
    late = step_times[args.warmup:]
    row["s_per_step"] = round(statistics.median(late), 3) if late else None
    row["tokens_per_s"] = round(args.bs * row.get("seq_len", 0) / row["s_per_step"], 1) if row["s_per_step"] and row.get("seq_len") else None
    row["losses"] = [round(x, 5) for x in losses]
    write_json(args.result, row)
    print(json.dumps({k: row[k] for k in ("cell_id", "outcome", "peak_reserved_gib", "s_per_step", "fla_active")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
