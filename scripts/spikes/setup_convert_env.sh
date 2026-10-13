#!/usr/bin/env bash
# Create .venv-convert (llama.cpp converter env, CPU torch), separate from .venv-lighton.
# Usage: scripts/spikes/setup_convert_env.sh   (LLAMA_CPP_DIR defaults to $HOME/src/llama.cpp)
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
uv venv .venv-convert --python 3.12
uv pip install --python .venv-convert/bin/python -r requirements/convert.lock.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --python .venv-convert/bin/python --no-deps -e "$LLAMA_CPP_DIR/gguf-py"
.venv-convert/bin/python - <<'PY'
import importlib.metadata as m
for p in ("torch", "transformers", "numpy", "safetensors", "protobuf", "sentencepiece", "gguf"):
    print(p, m.version(p))
PY
