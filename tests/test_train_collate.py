import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from PIL import Image

from glyphfix_math.train import infer
from glyphfix_math.train.collate import LabelMaskCollator, label_only_loss

MIN_PX, MAX_PX = 65536, 401408
TARGETS = ["```latex\nx^{\\prime} = \\Upsilon + 1\n```", "```latex\n\\bar{a}\n```"]


@pytest.fixture(scope="module")
def processor():
    try:
        return transformers.AutoProcessor.from_pretrained(infer.MODEL_ID, revision=infer.REVISION, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"model snapshot not in local cache: {exc}")


@pytest.fixture(scope="module")
def real_config():
    return transformers.AutoConfig.from_pretrained(infer.MODEL_ID, revision=infer.REVISION, local_files_only=True)


def _samples():
    return [
        {"image": Image.new("RGB", (333, 91), "white"), "target_text": TARGETS[0]},
        {"image": Image.new("RGB", (200, 60), "white"), "target_text": TARGETS[1]},
    ]


def _tiny_model(real_config, monkeypatch):
    # CPU run: use the torch reference delta-rule ops instead of the Triton kernels
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(mq, name)
        monkeypatch.setattr(mq, name, getattr(fn, "__wrapped__", fn))
    cfg = copy.deepcopy(real_config)
    tc = cfg.text_config
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.head_dim = 32
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.linear_key_head_dim = 16
    tc.linear_value_head_dim = 16
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.num_hidden_layers = 2
    tc.layer_types = ["linear_attention", "full_attention"]
    tc.rope_parameters["mrope_section"] = [2, 1, 1]
    vc = cfg.vision_config
    vc.depth = 1
    vc.hidden_size = 32
    vc.num_heads = 2
    vc.intermediate_size = 64
    vc.out_hidden_size = 64
    torch.manual_seed(0)
    return transformers.AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32).eval()


def test_collator_masks_prompt_and_padding(processor):
    batch = LabelMaskCollator(processor, min_px=MIN_PX, max_px=MAX_PX)(_samples())
    assert set(batch) >= {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert (att[0] != att[1]).any(), "samples must have different lengths"
    assert att[0].tolist() == sorted(att[0].tolist(), reverse=True), "right padding expected"
    assert (labels[att == 0] == -100).all()
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert (labels[ids == image_id] == -100).all()
    tok = processor.tokenizer
    for i, t in enumerate(TARGETS):
        keep = labels[i] != -100
        assert tok.decode(ids[i][keep]) == t + "<|im_end|>"
        first = int(keep.nonzero()[0])
        assert (labels[i, :first] == -100).all()
    # the prefix up to the first label decodes to the prompt (image pads aside)
    first0 = int((labels[0] != -100).nonzero()[0])
    assert tok.decode(ids[0][:first0]).endswith("<think>\n\n</think>\n\n")


def test_label_only_loss_matches_full_logits(processor, real_config, monkeypatch):
    import torch.nn.functional as F
    from peft import get_peft_model

    from glyphfix_math.train.lora import lora_config

    model = _tiny_model(real_config, monkeypatch)
    batch = LabelMaskCollator(processor, min_px=MIN_PX, max_px=MAX_PX)(_samples())
    ref_logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.float()
    shifted = batch["labels"][:, 1:]
    sel = shifted != -100
    ref = F.cross_entropy(ref_logits[:, :-1][sel], shifted[sel])
    got = label_only_loss(model, batch)
    assert abs(float(got.detach()) - float(ref.detach())) < 1e-4

    pm = get_peft_model(model, lora_config(4, 8))
    loss = label_only_loss(pm, batch)
    loss.backward()
    lora_b = [p for n, p in pm.named_parameters() if "lora_B" in n]
    assert lora_b and all(p.grad is not None for p in lora_b)
    # B starts at zero, so only B gets gradient on step one; after B moves A must too
    with torch.no_grad():
        for p in lora_b:
            p.add_(0.01)
    pm.zero_grad()
    label_only_loss(pm, batch).backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in pm.named_parameters() if "lora_A" in n)
    assert all(float(p.grad.abs().sum()) > 0 for p in lora_b)


def test_generate_one_accounting(processor, real_config, monkeypatch):
    model = _tiny_model(real_config, monkeypatch)
    res = infer.generate_one(model, processor, Image.new("RGB", (333, 91), "white"), min_px=MIN_PX, max_px=MAX_PX, max_new_tokens=3)
    assert set(res) == {"text", "n_prompt_tokens", "image_tokens", "grid_thw", "resized_hw", "finish", "n_new_tokens"}
    assert res["finish"] == "length" and res["n_new_tokens"] == 3
    assert res["image_tokens"] >= 64
    assert res["n_prompt_tokens"] > res["image_tokens"]


def test_load_base_refuses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        infer.load_base(device="cuda")
