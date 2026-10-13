"""Metadata and tensor-data diff between two GGUF files (hashes only, no tensor contents).

Usage (in .venv-convert, which has the checkout's gguf package):
  .venv-convert/bin/python -m scripts.spikes.gguf_diff --a A.gguf --b B.gguf --out diff.json
Memory: tensors are hashed in 64 MiB slices; on the shared machine launch it under the guard, for example
  uv run python -m lightonocr_math.guard --ram-gb 4 -- .venv-convert/bin/python -m scripts.spikes.gguf_diff ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _field_value(field):
    """Scalar python value of a GGUF KV field, or an {type, length, sha256} summary for arrays."""
    from gguf import GGUFValueType

    types = field.types
    if types and types[0] == GGUFValueType.ARRAY:
        h = hashlib.sha256()
        for idx in field.data:
            h.update(bytes(field.parts[idx]))
        return {"type": "array", "length": len(field.data), "sha256": h.hexdigest()}
    if types and types[0] == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    val = field.parts[field.data[0]]
    return val.item() if hasattr(val, "item") else val


def _hash_tensor(data, chunk: int = 64 * 2**20) -> str:
    """sha256 of the tensor bytes, fed in 64 MiB slices of the memmap (no full copy of the tensor)."""
    import numpy as np

    flat = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    h = hashlib.sha256()
    for i in range(0, flat.size, chunk):
        h.update(memoryview(flat[i : i + chunk]))
    return h.hexdigest()


def summarize(path: str | Path) -> dict:
    from gguf import GGUFReader

    r = GGUFReader(str(path))
    kv = {name: _field_value(f) for name, f in r.fields.items()}
    tensors = {}
    for t in r.tensors:
        tensors[t.name] = {
            "type": t.tensor_type.name,
            "shape": [int(x) for x in t.shape],
            "sha256": _hash_tensor(t.data),
        }
    return {"kv": kv, "tensors": tensors}


def diff(a: dict, b: dict) -> dict:
    ka, kb = a["kv"], b["kv"]
    kv_changed = [{"key": k, "a": ka[k], "b": kb[k]} for k in sorted(set(ka) & set(kb)) if ka[k] != kb[k]]
    ta, tb = a["tensors"], b["tensors"]
    common = sorted(set(ta) & set(tb))
    return {
        "only_in_a": sorted(set(ka) - set(kb)),
        "only_in_b": sorted(set(kb) - set(ka)),
        "tensors_only_in_a": sorted(set(ta) - set(tb)),
        "tensors_only_in_b": sorted(set(tb) - set(ta)),
        "kv_changed": kv_changed,
        "tensor_type_changed": [n for n in common if ta[n]["type"] != tb[n]["type"]],
        "tensor_shape_changed": [n for n in common if ta[n]["shape"] != tb[n]["shape"]],
        "tensor_data_changed": [n for n in common if ta[n]["sha256"] != tb[n]["sha256"]],
        "n_tensors_a": len(ta),
        "n_tensors_b": len(tb),
        "n_tensors_common": len(common),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    d = diff(summarize(args.a), summarize(args.b))
    d["a"], d["b"] = args.a, args.b
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(d, indent=2, sort_keys=True, default=str) + "\n")
    print({k: (len(v) if isinstance(v, list) else v) for k, v in d.items() if k not in ("a", "b")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
