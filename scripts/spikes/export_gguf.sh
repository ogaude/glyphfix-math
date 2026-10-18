#!/usr/bin/env bash
# Convert one HF folder to a Q8_0 GGUF with the pinned llama.cpp and register it in Ollama.
#
# Usage: scripts/spikes/export_gguf.sh ARM SRC_DIR     ARM: u | m | uf16
#   u, m   bf16 intermediate, Q8_0, stock Modelfile with only the text FROM replaced, ollama create spike-lighton-ARM
#   uf16   diagnostic: f16 intermediate, Q8_0, CPU only, no ollama
# Env: LLAMA_CPP_DIR (default $HOME/src/llama.cpp). Stock tags are never touched.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM=${1:-}
SRC=${2:-}
case "$ARM" in u | m | uf16) ;; *) echo "ARM must be u, m or uf16" >&2; exit 2 ;; esac
[ -d "$SRC" ] || { echo "SRC_DIR $SRC not found" >&2; exit 2; }

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
PIN=$(python3 -c "import json;print(json.load(open('requirements/export_pins.json'))['llama_cpp_commit'])")
[ "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$PIN" ] || { echo "llama.cpp is not at pinned commit $PIN" >&2; exit 3; }

WORK=data/spikes/export/$ARM
mkdir -p "$WORK"
OUTTYPE=bf16
[ "$ARM" = uf16 ] && OUTTYPE=f16
GUARD="uv run python -m lightonocr_math.guard --ram-gb 10 --timeout 1800 --"

$GUARD .venv-convert/bin/python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$SRC" \
  --outtype "$OUTTYPE" --no-mtp --model-name spike-lighton --outfile "$WORK/text-$OUTTYPE.gguf"
$GUARD nice -n 10 "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$WORK/text-$OUTTYPE.gguf" "$WORK/text-q8_0.gguf" Q8_0
sha256sum "$WORK/text-$OUTTYPE.gguf" "$WORK/text-q8_0.gguf" > "$WORK/sha256.txt"
[ "$ARM" = uf16 ] && { echo "uf16 done"; exit 0; }

STOCK=lighton-ocr3-q8-16k
ollama show --modelfile "$STOCK" > "$WORK/stock.Modelfile"
mapfile -t FROMS < <(grep '^FROM ' "$WORK/stock.Modelfile" | cut -d' ' -f2-)
TEXT_BLOB=""
PROJ_BLOB=""
for b in "${FROMS[@]}"; do
  arch=$(.venv-convert/bin/python - "$b" <<'PY'
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields["general.architecture"]
print(bytes(f.parts[f.data[0]]).decode())
PY
)
  if [ "$arch" = clip ]; then PROJ_BLOB=$b; else TEXT_BLOB=$b; fi
done
[ -n "$TEXT_BLOB" ] && [ -n "$PROJ_BLOB" ] || { echo "could not identify text and projector blobs" >&2; exit 4; }
[ -r "$PROJ_BLOB" ] || { echo "projector blob unreadable" >&2; exit 4; }
echo "$TEXT_BLOB" > data/spikes/export/stock_text_blob.path
echo "$PROJ_BLOB" > data/spikes/export/stock_proj_blob.path
sha256sum "$PROJ_BLOB" > data/spikes/export/stock_proj_blob.sha256

ABS_TEXT=$(realpath "$WORK/text-q8_0.gguf")
awk -v old="FROM $TEXT_BLOB" -v new="FROM $ABS_TEXT" '$0 == old { print new; next } { print }' "$WORK/stock.Modelfile" > "$WORK/Modelfile"
changed=$(diff "$WORK/stock.Modelfile" "$WORK/Modelfile" | grep -c '^>' || true)
[ "$changed" = 1 ] || { echo "expected exactly one changed Modelfile line, got $changed" >&2; exit 5; }

rocm-smi --showmeminfo vram | grep -E 'Total (Used )?Memory' || true
ollama ps
TAG=spike-lighton-$ARM
ollama rm "$TAG" >/dev/null 2>&1 || true
ollama create "$TAG" -f "$WORK/Modelfile"
echo "created $TAG"
