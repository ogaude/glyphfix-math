import copy
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from lightonocr_math.train import infer
from lightonocr_math.train.patch_embed import linearize_patch_embed


class FakePatchEmbed(nn.Module):
    """Same structure and forward as the transformers vision patch embed."""

    def __init__(self, in_channels=3, t=2, p=4, embed=8):
        super().__init__()
        self.in_channels, self.t, self.p, self.embed = in_channels, t, p, embed
        self.proj = nn.Conv3d(in_channels, embed, kernel_size=[t, p, p], stride=[t, p, p], bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.in_channels, self.t, self.p, self.p)
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed)
        return hidden_states


class Visual(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.patch_embed = FakePatchEmbed(**kw)


class Container(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.visual = Visual(**kw)
        self.lin = nn.Linear(4, 4)


def _x(n, c=3, t=2, p=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c * t * p * p, generator=g)


def test_fp32_matches_conv3d():
    torch.manual_seed(0)
    m = Container()
    x = _x(7)
    ref = m.visual.patch_embed(x)
    assert linearize_patch_embed(m) == 1
    out = m.visual.patch_embed(x)
    assert out.shape == (7, 8)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_bf16_within_one_step():
    torch.manual_seed(0)
    m = Container().to(torch.bfloat16)
    x = _x(7)
    ref = m.visual.patch_embed(x)
    assert linearize_patch_embed(m) == 1
    out = m.visual.patch_embed(x)
    assert out.dtype == torch.bfloat16 and out.shape == (7, 8)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1e-2)


def test_real_class_matches():
    transformers = pytest.importorskip("transformers")
    mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    cfg = types.SimpleNamespace(patch_size=16, temporal_patch_size=2, in_channels=3, hidden_size=1024)
    torch.manual_seed(0)
    pe = mod.Qwen3_5VisionPatchEmbed(cfg)
    holder = nn.Module()
    holder.patch_embed = pe
    x = torch.randn(50, 3 * 2 * 16 * 16)
    ref = pe(x)
    assert linearize_patch_embed(holder) == 1
    torch.testing.assert_close(pe(x), ref, rtol=1e-5, atol=1e-5)

    holder16 = copy.deepcopy(holder).to(torch.bfloat16)
    pe16 = holder16.patch_embed
    # compare against the unpatched class forward on the same weights
    ref16 = mod.Qwen3_5VisionPatchEmbed.forward(pe16, x)
    out16 = pe16(x)
    assert out16.dtype == torch.bfloat16
    torch.testing.assert_close(out16.float(), ref16.float(), rtol=1.6e-2, atol=1e-2)
    assert transformers is not None


def test_state_dict_unchanged():
    m = Container()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    names = [n for n, _ in m.named_parameters()]
    bufs = [n for n, _ in m.named_buffers()]
    linearize_patch_embed(m)
    after = m.state_dict()
    assert list(before) == list(after)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert names == [n for n, _ in m.named_parameters()]
    assert bufs == [n for n, _ in m.named_buffers()]


def test_gradient_equivalence_fp32():
    torch.manual_seed(0)
    m = Container()
    x = _x(5)
    m.visual.patch_embed(x).sum().backward()
    g_conv = m.visual.patch_embed.proj.weight.grad.clone()
    b_conv = m.visual.patch_embed.proj.bias.grad.clone()
    m.zero_grad()
    linearize_patch_embed(m)
    m.visual.patch_embed(x).sum().backward()
    torch.testing.assert_close(m.visual.patch_embed.proj.weight.grad, g_conv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m.visual.patch_embed.proj.bias.grad, b_conv, rtol=1e-5, atol=1e-5)


def test_idempotent():
    torch.manual_seed(0)
    m = Container()
    x = _x(7)
    ref = m.visual.patch_embed(x)
    assert linearize_patch_embed(m) == 1
    fwd = m.visual.patch_embed.forward
    assert linearize_patch_embed(m) == 1
    assert m.visual.patch_embed.forward.__func__ is fwd.__func__
    torch.testing.assert_close(m.visual.patch_embed(x), ref, rtol=1e-5, atol=1e-5)


def test_absent_module():
    m = nn.Sequential(nn.Linear(2, 2))
    assert linearize_patch_embed(m) == 0
    with pytest.raises(RuntimeError, match="patch-embed Conv3d"):
        linearize_patch_embed(m, require=True)


def test_non_matching_conv_not_patched():
    class Holder(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.pe = nn.Module()
            self.pe.proj = nn.Conv3d(3, 8, kernel_size=2, stride=1, **kw)

    assert linearize_patch_embed(Holder()) == 0
    assert linearize_patch_embed(Holder(padding=1)) == 0
    h = Holder()
    h.pe.proj = nn.Conv3d(3, 8, kernel_size=2, stride=2, padding=1)
    assert linearize_patch_embed(h) == 0


def _patch_loaders(monkeypatch, model):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: model)
    sentinel = object()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: sentinel)
    return sentinel


def test_load_base_wires_linearization(monkeypatch):
    model = Container()
    sentinel = _patch_loaders(monkeypatch, model)
    got, proc = infer.load_base(device="cpu")
    assert got is model and proc is sentinel
    assert getattr(model.visual.patch_embed, "_lom_linear_patch_embed", False) is True


def test_load_base_can_skip_linearization(monkeypatch):
    model = Container()
    _patch_loaders(monkeypatch, model)
    infer.load_base(device="cpu", linear_patch_embed=False)
    assert not getattr(model.visual.patch_embed, "_lom_linear_patch_embed", False)
