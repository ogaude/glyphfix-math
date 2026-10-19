# @@REPO_NAME@@

This is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn.

## What the project is

A LoRA adapter for LightOnOCR-3-4B that reduces math-notation errors when the model transcribes equations from research papers: a prime read as the letter l or dropped, a Greek letter read as a Latin one (upsilon as v), decorations such as bold, bar and underline lost, and I read as 1.

The repository holds the code, the evaluation and the evidence behind three published artifacts:

- the adapter and a merged Q8_0 GGUF for Ollama: https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@
- the license-clean equation dataset, {{train_rows}} training rows from {{train_papers}} papers and a frozen held-out benchmark of {{bench_rows}} rows from {{bench_papers}} papers, every row from a CC BY or CC0 arXiv paper with its attribution: https://huggingface.co/datasets/@@HF_NAMESPACE@@/@@DATASET_NAME@@
- the evaluation: the frozen scorer, the predictions of both models and the scored results, all under this repository

## Results

{{results_md}}
Full report: [evidence/acceptance/ACCEPTANCE.md](evidence/acceptance/ACCEPTANCE.md). Index of every result file with its sha256: [results/README.md](results/README.md).

## Reproduce

Tier 1 rescores the published predictions with the frozen scorer on CPU and Docker and compares the outcome with the published `evidence/acceptance/results.json`. It needs [uv](https://docs.astral.sh/uv/) and Docker, and uses no GPU.

```bash
git clone https://github.com/@@GH_OWNER@@/@@REPO_NAME@@.git
cd @@REPO_NAME@@
uv sync --frozen
docker pull {{texlive_image}}
uv run python -m scripts.release.reproduce
```

The command prints a verdict and writes a report. [REPRODUCE.md](REPRODUCE.md) explains each verdict and also documents Tier 2, the regeneration of predictions with Ollama.

## Hardware

The published model files and the benchmark rerun do not depend on a particular GPU: the GGUF runs in Ollama or llama.cpp on any backend they support, and Tier 1 above uses no GPU. Training, export and the full rebuild were done and tested only on AMD ROCm (a gfx1200 card with 16 GB). The pinned training environment installs the ROCm build of PyTorch, so other GPUs need the matching PyTorch build and have not been tried. The resource guard also reads free VRAM through `nvidia-smi` and torch, but that path was tested only with mocked output, not on NVIDIA hardware.

## Rebuild

[REBUILD.md](REBUILD.md) lists the whole pipeline in script order: candidate harvesting, fetching, building and selecting equations, error mining, training, export and acceptance. The versions and digests it relies on are in [results/PINS.md](results/PINS.md). A rebuild re-derives the artifacts; the published benchmark stays the frozen one in `data/benchmark` and `data/splits`.

## Evidence

`evidence/acceptance/` holds the published `results.json`, `flips.jsonl`, `ACCEPTANCE.md`, `sensitivity.json` and `env_check.json` at the paths the frozen scorer reads, and `evidence/acceptance/FREEZE_AMENDMENT.json`. The latter is the record of the one change made to the frozen scorer after scoring: two path constants in `scripts/acceptance.py` moved from the development repository's planning folder (not published) to `evidence/acceptance/`. The change was approved, was proven to alter nothing else, and a cold rescore afterwards matched the published numbers.

`results/` holds the error-mining and development reports, `PINS.md` and an index of the result files whose rerun record cites the cold rescore in `evidence/acceptance/FREEZE_AMENDMENT.json`.

The Python import package is still named `lightonocr_math` because the frozen scoring code imports it and renaming it would change the pinned blobs.

## History

This history was filtered from the development repository to leave out unreleased data rows, so the commit hashes differ from those recorded inside `results.json`. `RELEASE_HISTORY.json` maps them. The reproduce command checks that mapping and the freeze amendment, which is why it reports `identical_modulo_history_and_amendment` rather than `identical`.

## Licenses

- Code: Apache-2.0 (see `LICENSE` and `NOTICE`).
- Data rows: CC BY 4.0 or CC0 1.0 per record, as listed with authors and sources in `ATTRIBUTION.md` and `ATTRIBUTION.jsonl`.
- Our labels and annotations: CC BY 4.0.
- The base model LightOnOCR-3-4B is Apache-2.0, by LightOn.

## Disclaimer

This is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn.

## Contact and takedown

Open an issue on GitHub or write to @@CONTACT_EMAIL@@. If you are an author of a paper in the dataset and want a row removed, say which paper and we remove it.
