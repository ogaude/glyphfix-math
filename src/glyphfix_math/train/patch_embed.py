"""Run the vision patch embed as a matmul instead of a Conv3d.

The vision tower embeds flattened patches with a Conv3d whose kernel equals its stride and which has no
padding. On gfx1200 MIOpen re-runs its Find step for every new total patch count N (about 28.7 ms per
patch), which makes variable-size crops very slow. A Conv3d of that shape is exactly a matmul over the
flattened (channel, time, height, width) patch vectors, so the forward is replaced by F.linear on the
reshaped weight. Numerics differ by at most one bf16 step. No parameter or buffer changes, so state_dict
keys, adapters, merge and GGUF export are unaffected.
"""

from __future__ import annotations

import types


def _is_patch_embed(module) -> bool:
    """Structural match: a module with a `proj` Conv3d whose kernel equals its stride, unpadded."""
    from torch import nn

    proj = getattr(module, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        return False
    return (
        tuple(proj.kernel_size) == tuple(proj.stride)
        and all(p == 0 for p in proj.padding)
        and all(d == 1 for d in proj.dilation)
        and proj.groups == 1
        and proj.padding_mode == "zeros"
    )


def _linear_forward(self, hidden_states):
    import torch.nn.functional as F

    w = self.proj.weight
    flat = hidden_states.reshape(-1, w[0].numel()).to(w.dtype)
    return F.linear(flat, w.reshape(w.shape[0], -1), self.proj.bias)


def linearize_patch_embed(model, *, require: bool = False) -> int:
    """Patch every matching patch-embed module to the matmul forward; return how many are patched."""
    count = 0
    for _, module in model.named_modules():
        if not _is_patch_embed(module):
            continue
        if not getattr(module, "_lom_linear_patch_embed", False):
            module.forward = types.MethodType(_linear_forward, module)
            module._lom_linear_patch_embed = True
        count += 1
    if count == 0 and require:
        raise RuntimeError("no vision patch-embed Conv3d (kernel == stride, unpadded) found; refusing to run without the linear forward")
    return count
