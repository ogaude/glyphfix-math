"""Label-masking collator and label-only loss.

Usage:
  collate = LabelMaskCollator(processor, min_px=65536, max_px=401408)
  batch = collate([{"image": pil, "target_text": fenced_latex}, ...])
  loss = label_only_loss(model, batch)

Sequence layout per sample: prompt_text(processor) (image placeholder, generation prompt with the
empty think block) + target + "<|im_end|>". Labels are -100 everywhere except the target tokens and
the closing <|im_end|>; the boundary is taken from the token length of the processed prompt prefix,
never by searching for a token string.

label_only_loss resolves the backbone through the PEFT wrapper (peft_model.get_base_model() gives
Qwen3_5ForConditionalGeneration, whose `.model` returns hidden states without the LM head), selects
the hidden states that predict a label token, and applies the tied output embedding only to those
rows, so the batch x seq x 248,320 logits tensor is never built.
"""

from __future__ import annotations

from lightonocr_math.train.prompt import prompt_text, size_kwargs

END_OF_TURN = "<|im_end|>"


class LabelMaskCollator:
    def __init__(self, processor, *, min_px: int, max_px: int, target_key: str = "target_text"):
        self.processor = processor
        self.min_px = min_px
        self.max_px = max_px
        self.target_key = target_key

    def _process(self, texts, images, **extra):
        return self.processor(text=texts, images=images, return_tensors="pt", **size_kwargs(self.min_px, self.max_px), **extra)

    def __call__(self, samples: list[dict]) -> dict:
        import torch

        prompt = prompt_text(self.processor)
        images = [s["image"] for s in samples]
        full = [prompt + s[self.target_key] + END_OF_TURN for s in samples]
        batch = self._process(full, images, padding=True, padding_side="right")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, img in enumerate(images):
            prefix_ids = self._process([prompt], [img])["input_ids"][0]
            n = int(prefix_ids.shape[0])
            if not torch.equal(batch["input_ids"][i, :n], prefix_ids):
                raise ValueError(f"sample {i}: prompt tokens changed when the target was appended; label boundary is unsafe")
            labels[i, :n] = -100
        batch["labels"] = labels
        return batch


def _backbone_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def label_only_loss(model, batch: dict):
    """Mean float32 cross entropy over label positions, with the LM head applied only to the gathered rows."""
    import torch
    import torch.nn.functional as F

    backbone, head = _backbone_and_head(model)
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    hidden = backbone(**inputs).last_hidden_state
    labels = batch["labels"]
    shifted = labels[:, 1:]
    sel = shifted != -100
    rows = hidden[:, :-1][sel]
    logits = head(rows).float()
    return F.cross_entropy(logits, shifted[sel].to(logits.device), reduction="mean") if rows.shape[0] else torch.zeros((), device=hidden.device)
