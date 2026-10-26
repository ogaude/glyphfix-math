#!/usr/bin/env bash
# Convert a merged HF folder to a Q8_0 GGUF with the pinned llama.cpp and create a candidate Ollama tag.
#
# Usage: scripts/export_lora.sh RUN_ID MERGED_DIR
#   bf16 intermediate (--no-mtp), Q8_0, stock Modelfile with only the text FROM line replaced,
#   ollama create glyphfix-math-cand:RUN_ID, then data/models/RUN_ID/export.json.
# Env: LLAMA_CPP_DIR (default $HOME/src/llama.cpp).
# Exit codes: 2 bad arguments, 3 llama.cpp not at the pin, 4 blobs not identified or projector hash differs,
#             5 Modelfile differs from stock in other than one line.
# Only the tag glyphfix-math-cand:RUN_ID is removed or created. The stock tag is read with `ollama show` only.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID=${1:-}
SRC=${2:-}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_ID must match [A-Za-z0-9._-]+" >&2; exit 2; }
[ -d "$SRC" ] || { echo "MERGED_DIR $SRC not found" >&2; exit 2; }

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}
PIN=$(python3 -c "import json;print(json.load(open('requirements/export_pins.json'))['llama_cpp_commit'])")
[ "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$PIN" ] || { echo "llama.cpp is not at pinned commit $PIN" >&2; exit 3; }

WORK=data/models/$RUN_ID
mkdir -p "$WORK"
GUARD="uv run python -m glyphfix_math.guard --ram-gb 10 --timeout 1800 --"
STOCK=lighton-ocr3-q8-16k
CAND=glyphfix-math-cand:$RUN_ID

$GUARD .venv-convert/bin/python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$SRC" \
  --outtype bf16 --no-mtp --model-name glyphfix-math-lora --outfile "$WORK/text-bf16.gguf"
$GUARD nice -n 10 "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$WORK/text-bf16.gguf" "$WORK/text-q8_0.gguf" Q8_0
sha256sum "$WORK/text-bf16.gguf" "$WORK/text-q8_0.gguf" > "$WORK/sha256.txt"

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

PROJ_SHA=$(sha256sum "$PROJ_BLOB" | cut -d' ' -f1)
WANT_PROJ=$(python3 -c "import json;print(json.load(open('requirements/export_pins.json'))['projector_blob_sha256'])")
[ "$PROJ_SHA" = "$WANT_PROJ" ] || { echo "projector sha256 $PROJ_SHA differs from recorded $WANT_PROJ" >&2; exit 4; }

ABS_TEXT=$(realpath "$WORK/text-q8_0.gguf")
awk -v old="FROM $TEXT_BLOB" -v new="FROM $ABS_TEXT" '$0 == old { print new; next } { print }' "$WORK/stock.Modelfile" > "$WORK/Modelfile"
changed=$(diff "$WORK/stock.Modelfile" "$WORK/Modelfile" | grep -c '^>' || true)
[ "$changed" = 1 ] || { echo "expected exactly one changed Modelfile line, got $changed" >&2; exit 5; }

rocm-smi --showmeminfo vram | grep -E 'Total (Used )?Memory' || true
ollama ps
ollama rm "$CAND" >/dev/null 2>&1 || true
ollama create "$CAND" -f "$WORK/Modelfile"
echo "created $CAND"

RUN_ID="$RUN_ID" WORK="$WORK" CAND="$CAND" PIN="$PIN" PROJ_BLOB="$PROJ_BLOB" PROJ_SHA="$PROJ_SHA" python3 - <<'PY'
import hashlib, json, os, pathlib

work = pathlib.Path(os.environ["WORK"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


out = {
    "run_id": os.environ["RUN_ID"],
    "text_bf16_sha256": sha(work / "text-bf16.gguf"),
    "text_q8_0_sha256": sha(work / "text-q8_0.gguf"),
    "gguf_path": str(work / "text-q8_0.gguf"),
    "projector_blob": os.environ["PROJ_BLOB"],
    "projector_sha256": os.environ["PROJ_SHA"],
    "modelfile_sha256": sha(work / "Modelfile"),
    "llama_cpp_commit": os.environ["PIN"],
    "candidate_tag": os.environ["CAND"],
}
(work / "export.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
PY
