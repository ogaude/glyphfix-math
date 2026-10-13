"""LoRA target selection for the language model only.

Usage:
  from lightonocr_math.train.lora import lora_config, check_targets, expected_target_count
  model = get_peft_model(base, lora_config(32, 8))
  info = check_targets(model)

Targets: q/k/v/o of the full-attention layers, in_proj_qkv/in_proj_z/out_proj of the Gated DeltaNet
layers, gate/up/down of every MLP. The vision tower, embed_tokens and the tied output head are never
wrapped. rsLoRA scales by alpha / sqrt(r), so a small alpha (8 at r=32) gives a scaling near 1.4.
"""

from __future__ import annotations

import re

TARGET_REGEX = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)"
)
_BAD = re.compile(r"visual|embed_tokens|lm_head")


def lora_config(r: int, alpha: int, *, dropout: float = 0.0):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, use_rslora=True, lora_dropout=dropout, target_modules=TARGET_REGEX, bias="none")


def expected_target_count(config) -> int:
    """Wrapped-module count implied by the text config: 4 per full-attention layer, 3 per linear-attention layer, 3 MLP per layer."""
    tc = getattr(config, "text_config", config)
    types = list(tc.layer_types)
    full = sum(1 for t in types if t == "full_attention")
    linear = sum(1 for t in types if t == "linear_attention")
    return full * 4 + linear * 3 + tc.num_hidden_layers * 3


def check_targets(peft_model) -> dict:
    """Count wrapped modules and trainable parameters; raise if a trainable name hits the vision tower, embeddings or head."""
    n_wrapped = sum(1 for _, m in peft_model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B"))
    trainable = 0
    total = 0
    bad: list[str] = []
    for name, p in peft_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if _BAD.search(name):
                bad.append(name)
    if bad:
        raise ValueError(f"trainable parameters outside the language-model targets: {bad[:5]}")
    return {"n_wrapped": n_wrapped, "trainable": trainable, "total": total, "bad_names": bad}
