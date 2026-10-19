# Rebuilding the pipeline

This is the whole pipeline in script order. A rebuild re-derives the artifacts from arXiv. The published benchmark stays the frozen one in `data/benchmark` and `data/splits`: the split rule is hash based and the benchmark list is never rewritten, so a rebuild cannot move a paper across the split. Each script's module docstring has the full options and exit codes; the commands below show the shape of each step.

Versions and digests (Python, locked packages, llama.cpp commit, Ollama, render image, scorer, canon rules) are listed in [results/PINS.md](results/PINS.md).

## Rules that apply to every step

- arXiv politeness: metadata is read through `export.arxiv.org` and `oaipmh.arxiv.org`, and e-prints are fetched one at a time. Keep at least 3.1 s between two requests (the scripts refuse a smaller interval). Send a User-Agent with a contact address of your own (the constant `UA` in `src/lightonocr_math/corpus/catalog.py`, and the one in `scripts/build_candidates.py`). Fetch metadata and e-prints only, no bulk transfers.
- License: only papers under CC BY (3.0 or 4.0) and CC0 enter the dataset. The license is per paper version, so the version is pinned.
- GPU: a local 16 GB card. Training, export and acceptance were run and tested only on AMD ROCm (gfx1200); other GPUs are untested. The GPU may be shared, so check the free VRAM first (`rocm-smi --showmeminfo vram` on AMD, `nvidia-smi --query-gpu=memory.free --format=csv` on NVIDIA), size batches to it, and wait or stop instead of running out of memory. Ollama may hold a model in VRAM.
- RAM: keep `MemAvailable` above 1.5 GB at all times. Check it before a job and cap workers and Docker memory. The renderer runs at most 3 containers at 4 GB each; the training and export steps run under `lightonocr_math.guard`.
- Rendering uses the Docker image `texlive/texlive` at the digest in PINS.md, with no network.

## Pipeline

1. Candidates: `uv run python -m scripts.build_candidates` harvests license-filtered paper metadata into `data/candidates.jsonl`.
2. Split: `uv run python -m lightonocr_math.splits verify` checks the frozen split rule and benchmark list against the candidates (`freeze` creates them once and refuses to overwrite).
3. Catalog: `uv run python -m scripts.corpus_catalog build` writes the version-pinned catalogs of the training side and the benchmark side.
4. Fetch: `uv run python -m scripts.corpus_fetch order` registers a seeded paper order, then `uv run python -m scripts.corpus_fetch fetch` downloads e-print bundles, paced.
5. Build: `uv run python -m scripts.corpus_build` extracts equations from the sources, renders them in the sandbox and writes shards.
6. Select: `uv run python -m scripts.corpus_select run --config data/corpus/SELECTION_CONFIG.json` composes the train, dev and benchmark manifests and runs the leakage gate.
7. Pages: `uv run python -m scripts.corpus_pages build` makes page-level evaluation sets, and `uv run python -m scripts.corpus_pages rows` turns page transcriptions into scorer rows. The page images are not shipped, so page-level work starts here.
8. Regression set: `uv run python -m scripts.corpus_regression pool` and `select` build the known-error regression set from training and dev equations only.
9. Baseline: `uv run python -m scripts.baseline_infer --suite benchmark ...` runs the stock model through Ollama; `uv run python -m lightonocr_math.score` scores predictions.
10. Mining: `uv run python -m scripts.mine_train seed`, `infer`, `score`, `report` and `accept` find the error classes of the stock model on the training split only.
11. Training set: `scripts.build_trainset build` builds the stratified training set (use the training environment below).
12. Training: `scripts.train_lora train` trains the LoRA under the resource guard, with the training environment below.
13. Checkpoint selection: `uv run python -m scripts.select_checkpoint score`, `select` and `sweep`.
14. Merge and export: `uv run python -m scripts.export_merge` merges the adapter into the bf16 base on CPU; `scripts/export_lora.sh RUN_ID MERGED_DIR` converts it to a Q8_0 GGUF with the pinned llama.cpp and creates the candidate Ollama tag.
15. Registration: `uv run python -m scripts.register_model gates` and `register` run the pre-registered gates, then register or remove the candidate.
16. Acceptance: `uv run python -m scripts.acceptance generate` and `score --mode benchmark` generate and score the benchmark arms under the run lock; `report` renders `evidence/acceptance/ACCEPTANCE.md`.

## Environments

- Default environment (everything except training and export): `uv sync --frozen`.
- Training environment: `scripts/setup_train_env.sh` builds `.venv-lighton` from `requirements/train.lock.txt` (ROCm PyTorch first; on another GPU, install the matching PyTorch and torchvision build instead, untested). Commands that name `.venv-lighton/bin/python` use it.
- Export environment: `requirements/convert.lock.txt`, plus a llama.cpp checkout at the commit named in PINS.md (`LLAMA_CPP_DIR`, default `$HOME/src/llama.cpp`).
