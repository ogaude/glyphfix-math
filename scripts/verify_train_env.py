"""Verify the training env: GPU visibility, fla delta-rule kernel, bitsandbytes, lock match.

Usage (from repo root, with .venv-lighton/bin/python):
  python scripts/verify_train_env.py --lock requirements/train.lock.txt \
      --out .planning/phases/02-feasibility-spikes/evidence/env_verify.json

Writes one JSON object. fla and bitsandbytes failures are recorded, not raised. A missing GPU
or a non-gfx12 device is an error (exit 2); kernel checks never run on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path

PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s\\;]+)", re.MULTILINE)

PROBE = r"""
import json, logging, sys, io
if sys.argv[1] == "off":
    sys.modules["fla"] = None
buf = io.StringIO()
h = logging.StreamHandler(buf)
logging.getLogger().addHandler(h)
logging.getLogger("transformers").addHandler(h)
import transformers.models.qwen3_5.modeling_qwen3_5 as m
attr = "torch_chunk_gated_delta_rule"
fn = getattr(m, attr)
cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
impl = cells.get("implementation")
mod = getattr(impl, "__module__", None) or "none"
print(json.dumps({"attr": attr, "resolved_via": "closure variable 'implementation' of the hub-kernel wrapper",
                  "module": mod, "is_new_implementation": cells.get("is_new_implementation"),
                  "fallback_warning_at_import": "reference PyTorch implementation" in buf.getvalue()}))
"""


def canon(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def lock_check(lock: Path) -> tuple[bool, list[str]]:
    mism = []
    for name, ver in PIN_RE.findall(lock.read_text(encoding="utf-8")):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            mism.append(f"{name}: lock {ver}, not installed")
            continue
        if have != ver:
            mism.append(f"{canon(name)}: lock {ver}, installed {have}")
    return not mism, mism


def ver(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def fla_check() -> dict:
    out = {"import_ok": False, "import_error": "", "kernel_ok": False, "kernel_error": "", "kernel_path": ""}
    try:
        import fla  # noqa: F401

        out["import_ok"] = True
    except Exception:  # noqa: BLE001 - the probe reports any import failure
        out["import_error"] = traceback.format_exc()[-2000:]
        return out
    try:
        import torch

        mod = importlib.import_module("fla.ops.gated_delta_rule")
        fn = mod.chunk_gated_delta_rule
        out["kernel_path"] = f"{fn.__module__}.{fn.__name__}"
        B, T, H, D = 1, 64, 2, 64
        dev = "cuda"
        q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(B, T, H, device=dev, dtype=torch.float32)).requires_grad_(True)
        beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16).requires_grad_(True)
        o, _ = fn(q, torch.nn.functional.normalize(k, dim=-1), v, g, beta, output_final_state=False, use_qk_l2norm_in_kernel=False)
        o.float().sum().backward()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(o).all())
        for t in (q, k, v, g, beta):
            ok = ok and t.grad is not None and bool(torch.isfinite(t.grad).all())
        out["kernel_ok"] = ok
        if not ok:
            out["kernel_error"] = "non-finite output or gradient"
    except Exception:  # noqa: BLE001 - the probe reports any kernel failure
        out["kernel_error"] = traceback.format_exc()[-2000:]
    return out


def tf_probe(arm: str) -> dict:
    r = subprocess.run([sys.executable, "-c", PROBE, arm], check=False, capture_output=True, text=True)
    if r.returncode != 0:
        return {"module": "error", "error": r.stderr[-2000:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"module": "error", "error": (r.stdout + r.stderr)[-2000:]}


def gpu_state() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from glyphfix_math import guard

    r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], check=False, capture_output=True, text=True)
    try:
        smi = json.loads(r.stdout)
    except ValueError:
        smi = {"raw": r.stdout[-500:]}
    return {"rocm_smi": smi, "holders": guard.vram_holders(), "mem_available_bytes": guard.mem_available_bytes()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import torch

    res: dict = {"python": sys.version, "platform": platform.platform()}
    res["lock_sha256"] = hashlib.sha256(a.lock.read_bytes()).hexdigest()
    res["lock_match"], res["lock_mismatches"] = lock_check(a.lock)
    res["torchvision_version"] = ver("torchvision")
    t = {"version": torch.__version__, "hip": torch.version.hip, "cuda_available": bool(torch.cuda.is_available()),
         "device_name": None, "gcn_arch": None, "bf16_matmul_ok": False}
    if t["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        t["device_name"], t["gcn_arch"] = p.name, getattr(p, "gcnArchName", "")
        free, total = torch.cuda.mem_get_info()
        t["mem_free_bytes"], t["mem_total_bytes"] = free, total
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        t["bf16_matmul_ok"] = bool(torch.isfinite(y).all())
    res["torch"] = t
    for n in ("transformers", "peft", "accelerate", "bitsandbytes", "flash-linear-attention", "fla-core"):
        res[f"version_{n}"] = ver(n)
    res["version_triton"] = ver("triton") or ver("triton-rocm")
    res["gpu_state"] = gpu_state()

    gpu_ok = t["cuda_available"] and str(t["gcn_arch"]).startswith("gfx12")
    if gpu_ok:
        res["fla"] = fla_check()
    else:
        res["fla"] = {"import_ok": False, "import_error": "skipped: no gfx12 GPU", "kernel_ok": False,
                      "kernel_error": "skipped: no gfx12 GPU"}
    res["transformers_delta_rule"] = {"on": tf_probe("on"), "off": tf_probe("off")}
    r = subprocess.run([sys.executable, "-m", "bitsandbytes"], check=False, capture_output=True, text=True)
    res["bitsandbytes"] = {"returncode": r.returncode, "tail": "\n".join((r.stdout + r.stderr).splitlines()[-40:])}
    res["ok"] = bool(t["cuda_available"] and t["bf16_matmul_ok"] and res["lock_match"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not gpu_ok:
        print(f"error: no gfx12 GPU visible to torch (cuda_available={t['cuda_available']}, arch={t['gcn_arch']})", file=sys.stderr)
        return 2
    print(f"wrote {a.out} ok={res['ok']} fla_import={res['fla']['import_ok']} fla_kernel={res['fla']['kernel_ok']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
