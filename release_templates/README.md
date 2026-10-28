# @@REPO_NAME@@

This is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn.

## Status: an initial release

This is our first release, and we want to be honest about what it is. The adapter scores higher than the base model on the frozen benchmark, with a paper-level confidence interval that excludes zero, but most of that gain comes from a broad group of errors we have not yet split into named classes. Of the classes that motivated the project, decorations show fewer errors, dropped primes show no significant change, and prime read as l, Greek read as Latin and I read as 1 have too few events in the benchmark to claim anything. The per-class table in the results shows each one. Treat the results as a starting point, not a finished product.

![Render-identical rate, base model against this model](results/charts/chart_headline.svg)

![Crop by crop: what changed](results/charts/chart_flips.svg)

Of the 1,543 crops the base model gets wrong on the benchmark, 935 remain wrong with the adapter. The largest remaining buckets are real content differences (304) and layout or sizing (236); the model card has the full six-bucket table, and `scripts/error_buckets.py` reproduces it. The buckets are a pattern heuristic for reporting, not part of the frozen scorer.

We will build on it, including more training on the layout, spacing and content buckets above: a training set built around each error class, a new and separate benchmark for those classes, and a closer look at what the large `other` group contains. The plan may change as we learn more. Reports of equations the model gets wrong are welcome as issues on this repository.

Versions: this is project release v1.0.0 (see `VERSIONING.md`), with tags `lora-v1.0.0`, `dataset-v1.0.0` and `benchmark-v1`.

## What the project is

A LoRA adapter for LightOnOCR-3-4B, trained to reduce math-notation errors when the model transcribes equations from research papers. The errors that motivated it are a prime read as the letter l or dropped, a Greek letter read as a Latin one (upsilon as v), decorations such as bold, bar and underline lost, and I read as 1. See the status above for what the benchmark does and does not show for each.

The repository holds the code, the evaluation and the evidence behind three published artifacts:

- the adapter and a merged Q8_0 GGUF for Ollama: https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@
- the equation dataset, built only from CC BY and CC0 arXiv papers with the license checked per paper version as far as we could, {{train_rows}} training rows from {{train_papers}} papers and a frozen held-out benchmark of {{bench_rows}} rows from {{bench_papers}} papers, every row with its paper's attribution: https://huggingface.co/datasets/@@HF_NAMESPACE@@/@@DATASET_NAME@@
- the evaluation: the frozen scorer, the predictions of both models and the scored results, all under this repository

## Use the model

You do not need this repository to use the model. For Ollama:

```bash
hf download @@HF_NAMESPACE@@/@@MODEL_NAME@@ --local-dir @@MODEL_NAME@@
cd @@MODEL_NAME@@
ollama create @@MODEL_NAME@@ -f Modelfile
```

The model page has the full steps, including llama.cpp and PEFT: https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@

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

The published model files and the benchmark rerun do not depend on a particular GPU: the GGUF is meant to run in Ollama or llama.cpp on any backend they support (we tested it with Ollama on ROCm, and with Ollama's bundled `llama-server` on CPU), and Tier 1 above uses no GPU. Training, export and the full rebuild were done and tested only on AMD ROCm (a gfx1200 card with 16 GB). The pinned training environment installs the ROCm build of PyTorch, so other GPUs need the matching PyTorch build and have not been tried. The resource guard also reads free VRAM through `nvidia-smi` and torch, but that path was tested only with mocked output, not on NVIDIA hardware.

## Rebuild

[REBUILD.md](REBUILD.md) lists the whole pipeline in script order: candidate harvesting, fetching, building and selecting equations, error mining, training, export and acceptance. The versions and digests it relies on are in [results/PINS.md](results/PINS.md). A rebuild re-derives the artifacts; the published benchmark stays the frozen one in `data/benchmark` and `data/splits`.

## Evidence

`evidence/acceptance/` holds the published `results.json`, `flips.jsonl`, `ACCEPTANCE.md`, `sensitivity.json` and `env_check.json` at the paths the frozen scorer reads, and `evidence/acceptance/FREEZE_AMENDMENT.json`. The latter is the record of the one change made to the frozen scorer after scoring: two path constants in `scripts/acceptance.py` moved from the development repository's planning folder (not published) to `evidence/acceptance/`. The change was approved, was proven to alter nothing else, and a cold rescore afterwards matched the published numbers. `evidence/acceptance/FREEZE_AMENDMENT_2.json` is the record of the rename made before the first release, in which the project, the Python package and our Ollama tags got their published names. The two frozen scoring files changed in names only, proven token by token from the git objects, and a cold rescore matched the published numbers.

`results/` holds the error-mining and development reports, `PINS.md` and an index of the result files whose rerun record cites the cold rescores in `evidence/acceptance/FREEZE_AMENDMENT.json` and `evidence/acceptance/FREEZE_AMENDMENT_2.json`.

The Python package was renamed from `lightonocr_math` to `glyphfix_math` before the first release; `FREEZE_AMENDMENT_2.json` records it.

## History

This history was filtered from the development repository to leave out unreleased data rows, so the commit hashes differ from those recorded inside `results.json`. `RELEASE_HISTORY.json` maps them. The reproduce command checks that mapping and both freeze amendments, which is why it reports `identical_modulo_history_and_amendment` rather than `identical`. Commits before the rename keep the old names because the published results were scored from them.

## Licenses

- Code: Apache-2.0 (see `LICENSE` and `NOTICE`). The charts under `results/charts/` are ours, also Apache-2.0.
- Model files (the adapter and the merged GGUF) on the model page: Apache-2.0. Its example images keep the license of their papers.
- Data rows: CC BY 4.0 or CC0 1.0 per record, as listed with authors and sources in `ATTRIBUTION.md` and `ATTRIBUTION.jsonl`.
- Our labels and annotations: CC BY 4.0.
- The base model LightOnOCR-3-4B is Apache-2.0, by LightOn.

## Disclaimer

This is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn.

## Contact and takedown

Open an issue on GitHub or write to @@CONTACT_EMAIL@@. If you are an author of a paper in the dataset and want a row removed, say which paper and we remove it.
