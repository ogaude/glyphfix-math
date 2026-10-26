import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from glyphfix_math.train import lora as L
from glyphfix_math.train.infer import MODEL_ID, REVISION


@pytest.fixture(scope="module")
def config():
    try:
        return transformers.AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"config not in local cache: {exc}")


def test_expected_count(config):
    assert L.expected_target_count(config) == 200


def test_lora_config_fields():
    c = L.lora_config(32, 8)
    assert c.use_rslora is True
    assert c.lora_dropout == 0.0
    assert c.r == 32 and c.lora_alpha == 8
    assert c.target_modules == L.TARGET_REGEX


def test_wraps_only_language_model(config):
    import torch
    from accelerate import init_empty_weights
    from peft import get_peft_model

    with init_empty_weights():
        model = transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    pm = get_peft_model(model, L.lora_config(32, 8))
    info = L.check_targets(pm)
    assert info["n_wrapped"] == L.expected_target_count(config) == 200
    assert info["trainable"] > 0
    assert info["bad_names"] == []
