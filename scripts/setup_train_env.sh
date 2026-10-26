#!/usr/bin/env bash
# Build the training env .venv-lighton from requirements/train.lock.txt (ROCm torch first).
#
# Usage (from repo root):
#   scripts/setup_train_env.sh [--recreate]
#
# Recompile the lock after editing requirements/train.in:
#   uv pip compile requirements/train.in -o requirements/train.lock.txt --python-version 3.12 \
#     --python-platform x86_64-manylinux_2_28 --index-url https://pypi.org/simple \
#     --extra-index-url https://download.pytorch.org/whl/rocm7.2 --index-strategy unsafe-best-match \
#     --emit-index-annotation --generate-hashes
set -euo pipefail

VENV=.venv-lighton
ROCM_INDEX=https://download.pytorch.org/whl/rocm7.2
LOCK=requirements/train.lock.txt
TORCH_PIN=$(grep -m1 '^torch==' "$LOCK" | sed -E 's/^torch==([^ \\]+).*/\1/' || true)
[ -n "$TORCH_PIN" ] || { echo "error: no torch== line in $LOCK" >&2; exit 1; }
TORCHVISION_PIN=0.29.1+rocm7.2
PY="$VENV/bin/python"

if [ "${1:-}" = "--recreate" ]; then
  echo "== remove $VENV"
  rm -rf "$VENV"
fi

echo "== (a) create venv"
uv venv "$VENV" --python 3.12

echo "== (b) ROCm torch $TORCH_PIN"
uv pip install --python "$PY" --no-deps "torch==$TORCH_PIN" --index-url "$ROCM_INDEX"

echo "== (c) torchvision $TORCHVISION_PIN (--no-deps)"
uv pip install --python "$PY" --no-deps "torchvision==$TORCHVISION_PIN" --index-url "$ROCM_INDEX"

echo "== (d) rest of the lock"
uv pip install --python "$PY" -r "$LOCK" --index-url https://pypi.org/simple \
  --extra-index-url "$ROCM_INDEX" --index-strategy unsafe-best-match

echo "== (e) project (no deps)"
uv pip install --python "$PY" --no-deps -e .

echo "== (f) verify under the resource guard"
rocm-smi --showmeminfo vram || true
ollama ps || true
uv run python -m glyphfix_math.guard --vram-gb 1 --ram-gb 4 --timeout 600 -- \
  env HF_HUB_OFFLINE=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 "$PY" scripts/verify_train_env.py \
  --lock "$LOCK" --out .planning/phases/02-feasibility-spikes/evidence/env_verify.json
