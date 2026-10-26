import subprocess
import sys

import pytest

from glyphfix_math.train import prompt as P
from glyphfix_math.train.infer import MODEL_ID, REVISION


@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"model snapshot not in local cache: {exc}")


def _tokens(processor, size, px):
    from PIL import Image

    img = Image.new("RGB", size, "white")
    out = processor(text=[P.prompt_text(processor)], images=[img], return_tensors="pt", **P.size_kwargs(P.MIN_PIXELS, px))
    return P.image_tokens(out["image_grid_thw"])


def test_prompt_ends_with_empty_think(processor):
    text = P.prompt_text(processor)
    assert text.endswith("<think>\n\n</think>\n\n")
    assert text.count("<|image_pad|>") == 1
    assert text.count("<|vision_start|>") == 1


def test_size_kwargs():
    assert P.size_kwargs(65536, 401408) == {"size": {"shortest_edge": 65536, "longest_edge": 401408}}


def test_pixel_cap_honoured(processor):
    n = _tokens(processor, (2000, 1000), 401408)
    assert 300 < n <= 392
    assert _tokens(processor, (2000, 1000), 200704) <= 196
    assert _tokens(processor, (256, 256), 401408) == 64
    # the floor upsizes tiny crops to at least 64 tokens; rounding to 32 px multiples at a 2.5 aspect gives a few more
    assert 64 <= _tokens(processor, (100, 40), 401408) <= 80


def test_image_tokens():
    assert P.image_tokens([[1, 20, 30]]) == 150


def test_resize_to_budget():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (333, 91), "white")
    out = P.resize_to_budget(img, 401408)
    w, h = out.size
    assert w * h <= 401408
    assert w % 32 == 0 and h % 32 == 0
    assert abs((w / h) / (333 / 91) - 1) < 0.05
    assert w * h >= 0.9 * 401408


def test_import_is_torch_free():
    r = subprocess.run(
        [sys.executable, "-c", "import glyphfix_math, sys; assert 'torch' not in sys.modules"],
        check=False,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr
