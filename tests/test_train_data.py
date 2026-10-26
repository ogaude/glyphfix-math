import json
import random

import pytest
from PIL import Image

from glyphfix_math import pixels
from glyphfix_math.train import data

MIN_PX, MAX_PX = 65536, 602112


def _row(**kw):
    row = {"id": "a", "paper_id": "p", "image": "x.png", "latex": "a_1", "target_text": data.target_text("a_1"),
           "group": "plain", "width": 470, "height": 139}
    row.update(kw)
    return row


def test_target_text_is_fenced():
    assert data.target_text("a_1") == "```latex\na_1\n```"


def test_read_trainset_ok_and_errors(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(_row()) + "\n")
    assert data.read_trainset(good)[0]["id"] == "a"

    bad_key = _row()
    del bad_key["group"]
    p = tmp_path / "k.jsonl"
    p.write_text(json.dumps(_row()) + "\n" + json.dumps(bad_key) + "\n")
    with pytest.raises(ValueError, match=r":2:"):
        data.read_trainset(p)

    p2 = tmp_path / "t.jsonl"
    p2.write_text(json.dumps(_row(target_text="a_1")) + "\n")
    with pytest.raises(ValueError, match=r":1:"):
        data.read_trainset(p2)


def test_policy_dims():
    assert data.policy_dims({"width": 470, "height": 139}) == pixels.smart_resize_dims(470, 139)


def test_load_image_matches_policy(tmp_path):
    Image.new("L", (470, 139), 255).save(tmp_path / "x.png")
    row = _row()
    img = data.load_image(row, root=tmp_path)
    assert img.mode == "RGB"
    assert img.size == data.policy_dims(row)


def test_bucket_batches_cover_and_reproduce():
    rng = random.Random(1)
    sizes = [rng.randint(1000, 600000) for _ in range(301)]
    a = data.bucket_batches(sizes, batch_size=2, seed=0, epoch=0)
    assert sorted(i for b in a for i in b) == list(range(301))
    assert sum(len(b) != 2 for b in a) <= 1
    assert a == data.bucket_batches(sizes, batch_size=2, seed=0, epoch=0)
    assert a != data.bucket_batches(sizes, batch_size=2, seed=0, epoch=1)


def test_bucket_batches_group_by_size_in_window():
    sizes = list(range(256))
    full = data.bucket_batches(sizes, batch_size=2, seed=3, epoch=0, window=128)
    assert all(abs(sizes[b[0]] - sizes[b[1]]) == 1 for b in full)
    small = data.bucket_batches(sizes, batch_size=2, seed=3, epoch=0, window=8)
    mean_spread = sum(abs(sizes[b[0]] - sizes[b[1]]) for b in small) / len(small)
    assert mean_spread < 40  # random pairing of 256 sizes averages about 85


def test_lr_schedule():
    total, peak = 100, 2e-4
    lrs = [data.lr_at(s, total=total, peak=peak) for s in range(total)]
    assert lrs[0] == pytest.approx(peak / 5)
    assert lrs[4] == pytest.approx(peak)
    assert max(lrs) == pytest.approx(peak)
    assert lrs[99] == pytest.approx(2e-5, abs=1e-9)
    assert all(lrs[i] >= lrs[i + 1] - 1e-15 for i in range(4, total - 1))


def test_agrees_with_trainset_target_text():
    ts = pytest.importorskip("glyphfix_math.trainset")
    for s in ("a_1", "x^{\\prime}", ""):
        assert ts.target_text(s) == data.target_text(s)


def test_processor_resize_is_noop_on_policy_output():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from glyphfix_math.paths import repo_root
    from glyphfix_math.train import infer
    from glyphfix_math.train.prompt import prompt_text, size_kwargs

    try:
        proc = transformers.AutoProcessor.from_pretrained(infer.MODEL_ID, revision=infer.REVISION, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"model snapshot not in local cache: {exc}")
    recs = repo_root() / "data" / "corpus" / "train" / "records.jsonl"
    if not recs.exists():
        pytest.skip("train records not present")
    rows = [json.loads(line) for line in recs.open() if line.strip()]
    rows = [r for r in rows if (repo_root() / r["image"]).exists()]
    if len(rows) < 50:
        pytest.skip("images not present")
    biggest = max(rows, key=lambda r: r["width"] * r["height"])
    sample = random.Random(0).sample(rows, 49) + [biggest]
    text = prompt_text(proc)
    for r in sample:
        img = data.load_image(r)
        w, h = data.policy_dims(r)
        assert img.size == (w, h)
        out = proc(text=[text], images=[img], return_tensors="pt", **size_kwargs(MIN_PX, MAX_PX))
        assert out["image_grid_thw"].tolist() == [[1, h // 16, w // 16]], r["id"]
