#!/usr/bin/env bash
# Pinned CPU build of llama.cpp (conversion + quantization tools), outside the repo.
#
# Usage:  scripts/spikes/build_llama_cpp.sh
# Env:    LLAMA_CPP_DIR (default $HOME/src/llama.cpp), LLAMA_CPP_COMMIT (40-char pin)
set -euo pipefail

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT:-a17e94d229fc7fed1567a799baebd11b771fe5af}
JOBS=${JOBS:-6}

avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt $((3 * 1024 * 1024)) ]; then
  echo "MemAvailable ${avail_kb} kB is too low to build safely (floor 1.5 GiB plus build headroom)" >&2
  exit 75
fi

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --quiet origin
git -C "$LLAMA_CPP_DIR" checkout --quiet --detach "$LLAMA_CPP_COMMIT"
head=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
if [ "$head" != "$LLAMA_CPP_COMMIT" ]; then
  echo "checkout is at $head, expected $LLAMA_CPP_COMMIT" >&2
  exit 1
fi

if ! grep -rlq "Qwen3_5ForConditionalGeneration" "$LLAMA_CPP_DIR/conversion"; then
  echo "no converter under conversion/ registers Qwen3_5ForConditionalGeneration at $head" >&2
  exit 1
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_HIP=OFF -DLLAMA_CURL=OFF
nice -n 10 cmake --build "$LLAMA_CPP_DIR/build" -j "$JOBS" --target llama-quantize llama-server llama-mtmd-cli

for b in llama-quantize llama-server llama-mtmd-cli; do
  echo "$LLAMA_CPP_DIR/build/bin/$b"
done
