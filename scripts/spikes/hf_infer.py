"""HF bf16 greedy inference over the legacy dev crops, with token accounting and a teacher-forced divergence margin.

Usage (from repo root, training venv):
  .venv-lighton/bin/python -m scripts.spikes.hf_infer --split dev --out data/spikes/hf/HF.pred.jsonl \
    --sidecar data/spikes/hf/HF.sidecar.jsonl --teacher-force data/spikes/ollama/S1.sidecar.jsonl

Shared GPU: prefer launching under the resource guard, for example
  uv run python -m lightonocr_math.guard --vram-gb 10 --ram-gb 8 --timeout 1800 --vram-floor-gb 0.1 -- \
    .venv-lighton/bin/python -m scripts.spikes.hf_infer ...
The script also runs a guard preflight (free VRAM and RAM floor) before loading the model. The default
--max-px is the processor's stock cap (up to about 16k image tokens), kept so the run mirrors stock HF
behaviour; pass a smaller --max-px on a busy card (the value is recorded in the sidecar).

Batch size is 1 on purpose: padding changes the linear-attention state. The pixel caps default to the
processor's shipped values, which is the stock HF behaviour. Without a GPU the run aborts with exit 2.
first_divergence is a pure function and imports without torch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from lightonocr_math.paths import repo_root
from scripts.spikes import common


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where two id lists differ (a shorter prefix diverges at its length); None if equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _margin(model, processor, image, ol_text: str, hf_text: str, min_px: int, max_px: int) -> dict:
    """Teacher-forced forward over prompt + Ollama continuation up to the first divergent token."""
    import torch

    from lightonocr_math.train.prompt import prompt_text, size_kwargs

    tok = processor.tokenizer
    a = tok(hf_text, add_special_tokens=False)["input_ids"]
    b = tok(ol_text, add_special_tokens=False)["input_ids"]
    d = first_divergence(a, b)
    if d is None:
        return {"divergence_index": None}
    inputs = processor(text=[prompt_text(processor)], images=[image], return_tensors="pt", **size_kwargs(min_px, max_px))
    dev = next(model.parameters()).device
    n_prompt = int(inputs["input_ids"].shape[1])
    if d >= len(b):
        return {"divergence_index": d, "ollama_token": None, "note": "ollama text ends here"}
    ext = torch.tensor([b[: d + 1]], dtype=inputs["input_ids"].dtype)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key not in inputs:
            continue
        add = ext if key == "input_ids" else torch.full_like(ext, 1 if key == "attention_mask" else 0)
        inputs[key] = torch.cat([inputs[key], add], dim=1)
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, n_prompt + d - 1].float()
    ol_tok = b[d]
    top1 = int(torch.argmax(logits))
    return {
        "divergence_index": d,
        "hf_top1_token": top1,
        "ollama_token": int(ol_tok),
        "margin": float(logits[top1] - logits[ol_tok]),
        "ollama_token_rank": int((logits > logits[ol_tok]).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--min-px", type=int, default=65536)
    ap.add_argument("--max-px", type=int, default=16777216)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--teacher-force", type=Path, default=None, help="Ollama sidecar JSONL for divergence margins")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    if not torch.cuda.is_available():
        print("error: no GPU available; refusing to run HF inference on CPU", file=sys.stderr)
        return 2
    from PIL import Image

    from lightonocr_math import guard
    from lightonocr_math.train import infer

    try:
        guard.preflight(vram_gb=10, ram_gb=4, timeout_s=900)
    except guard.GuardError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    rows, stats = common.load_legacy(a.split, limit=a.limit)
    gpu_before = common.gpu_state()
    model, processor = infer.load_base()
    model.eval()
    ol_side = {r["id"]: r for r in common.read_jsonl(a.teacher_force)} if a.teacher_force else {}
    preds, sides = [], []
    t0 = time.time()
    for r in rows:
        img = Image.open(r["image_path"]).convert("RGB")
        res = infer.generate_one(model, processor, img, min_px=a.min_px, max_px=a.max_px, max_new_tokens=a.max_new_tokens)
        side = {
            "id": r["id"],
            "raw_text": res["text"],
            "n_prompt_tokens": res["n_prompt_tokens"],
            "image_tokens": res["image_tokens"],
            "grid_thw": res["grid_thw"],
            "resized_hw": res["resized_hw"],
            "min_px": a.min_px,
            "max_px": a.max_px,
            "finish": res["finish"],
            "n_new_tokens": res["n_new_tokens"],
        }
        o = ol_side.get(r["id"])
        if o is not None and o["raw_text"] != res["text"]:
            side.update(_margin(model, processor, img, o["raw_text"], res["text"], a.min_px, a.max_px))
        preds.append(
            {"id": r["id"], "paper_id": r["arxiv_id"], "gt_latex": r["latex"], "pred_latex": common.strip_fences(res["text"])}
        )
        sides.append(side)
    common.write_jsonl(a.out, preds)
    common.write_jsonl(a.sidecar, sides)
    import peft
    import transformers

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root(), check=False).stdout.strip()
    common.write_json(
        a.out.with_name(a.out.stem + ".manifest.json"),
        {
            "git_commit": commit,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "model": infer.MODEL_ID,
            "revision": infer.REVISION,
            "min_px": a.min_px,
            "max_px": a.max_px,
            "max_new_tokens": a.max_new_tokens,
            "batch_size": 1,
            "n": len(rows),
            "loader_stats": stats,
            "wall_seconds": round(time.time() - t0, 1),
            "gpu_before": gpu_before,
            "gpu_after": common.gpu_state(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
