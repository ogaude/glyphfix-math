# Reproducing the published results

There are two tiers. Tier 1 is what the published claim rests on. Tier 2 regenerates the predictions and is documented but is not part of pass or fail.

## Tier 1: rescore the published predictions (required)

The scorer is deterministic. Tier 1 takes the published predictions of the stock and the tuned model, scores them again with the frozen scorer and compares the new `results.json` with the published `evidence/acceptance/results.json`. It runs on CPU plus Docker. No GPU and no Ollama are needed.

Prerequisites:

- [uv](https://docs.astral.sh/uv/) and Docker
- the pinned render image: `docker pull {{texlive_image}}`
- free memory: at least 1.5 GB must stay available. The renderer runs up to 3 containers at 4 GB each, so the command refuses to start with less than 5.5 GB available

Commands, from the repository root:

```bash
uv sync --frozen
uv run python -m scripts.release.reproduce
```

The first run renders from a cold cache and takes minutes, not seconds. The command writes `data/runs/reproduce-<UTC time>/reproduce_report.json` (the work folder is printed) and exits 0 when the verdict is one of the identical ones below, 1 when it is different or the freeze amendment could not be verified, 2 when memory is short, 3 when the render image is missing.

What the verdicts mean:

- `identical`: the new `results.json` equals the published one except for the volatile keys (creation time, timings, git commit, output paths).
- `identical_modulo_history`: additionally, only the commit hash of the frozen acceptance module differs. `RELEASE_HISTORY.json` maps the published hash to the hash in this repository and the committer dates are equal.
- `identical_modulo_amendment` and `identical_modulo_history_and_amendment`: additionally, the `code_freeze` keys (and the tag) of the recorded freeze amendments differ exactly as they predict. Amendment 1 (`evidence/acceptance/FREEZE_AMENDMENT.json`) changed two path constants and moves three keys. Amendment 2 (`evidence/acceptance/FREEZE_AMENDMENT_2.json`) is the rename: the CLI commit, the module path key moving to its new path with the same commit and date, the CLI and module blobs in `current_blobs` and `lock_blobs`, and the tag; eight keys in total. Before comparing, the command proves again from the git objects that each frozen file changed only as its record says. A clone of the public repository reports `identical_modulo_history_and_amendment`.
- `different`: anything else, including any changed number, input digest, module blob, pin time or lock time. The differing keys are listed in the report.

The command runs the frozen scorer through `scripts.release.frozen_rescore`. The frozen scorer refuses to score when its module's last commit is later than the prediction run, and the rename moved the module (renamed from `src/lightonocr_math/acceptance.py` to `src/glyphfix_math/acceptance.py`), so the wrapper supplies the module's pre-rename commit after proving from git objects that the module is that pinned file with only names changed. Calling `scripts.acceptance score --mode benchmark` directly refuses for that reason.

The frozen `uv run python -m scripts.acceptance compare-results` stays strict on purpose. Given the published file and a fresh one, it lists the keys of the amendments (eight after the rename) and nothing more; `scripts.release.reproduce` is the comparison that knows the amendment and the history mapping.

## Tier 2: regenerate the predictions (documented, not pass or fail)

Tier 2 asks whether the published predictions can be produced again. It needs a GPU or a machine that can run a 4B vision model through Ollama {{ollama_version}}, the same engine version as the published predictions. Greedy decoding differs across engines and hardware at near-ties, so the regenerated predictions are not expected to be identical, and Tier 2 is not part of pass or fail.

1. Download the benchmark split of the dataset and write the crops to the paths the frozen `data/benchmark/records.jsonl` expects. Each image is checked against its sha256:

   ```bash
   uv run python -m scripts.release.extract_images --parquet <path to data/benchmark-00000-of-00001.parquet>
   ```

2. Build the tuned model from the model repository folder:

   ```bash
   ollama create <tuned tag> -f Modelfile
   ```

3. Build the stock model from the community GGUF of LightOnOCR-3-4B, `hf.co/mradermacher/LightOnOCR-3-4B-GGUF` (Q8_0 with its projector), with the stock parameters `num_ctx {{stock_num_ctx}}`, `num_predict {{stock_num_predict}}` and `TEMPLATE {{ .Prompt }}`.

4. Regenerate the crop predictions with the inference CLI, one run per model (`--resume` continues an interrupted run):

   ```bash
   uv run python -m scripts.baseline_infer --suite benchmark --records data/benchmark/records.jsonl \
       --model <tag> --out <new>.pred.jsonl --sidecar <new>.sidecar.jsonl
   ```

   `scripts.acceptance generate` is protected by the run lock of the published evaluation and refuses every arm, so it is not the Tier 2 path.

5. Score the new files with explicit paths and a separate output folder, so the published evidence is not touched:

   ```bash
   uv run python -m scripts.acceptance score --mode dev \
       --crops-stock <new stock>.pred.jsonl --crops-tuned <new tuned>.pred.jsonl \
       --scores-dir <out>/scores --evidence-dir <out>/evidence --report <out>/ACCEPTANCE.md
   ```

   The page-level inputs keep their published defaults. Page-level regeneration needs the page images, which are not shipped, so it requires the rebuild in [REBUILD.md](REBUILD.md).

Tolerance: a CPU re-run of every flipped record (the records where the two models disagree about being right) held {{r2w_held}} of {{r2w_total}} right-to-wrong flips and {{w2r_held}} of {{w2r_total}} wrong-to-right flips, with {{reversed}} reversed (from `evidence/acceptance/sensitivity.json`). The diagnosis of the differences is in [results/PARITY_DIAGNOSIS.md](results/PARITY_DIAGNOSIS.md). The pinned versions are in [results/PINS.md](results/PINS.md).
