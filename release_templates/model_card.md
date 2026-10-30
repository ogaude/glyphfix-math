---
license: apache-2.0
base_model: lightonai/LightOnOCR-3-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - peft
  - ocr
  - latex
  - math
  - gguf
language:
  - en
datasets:
  - @@HF_NAMESPACE@@/@@DATASET_NAME@@
model-index:
  - name: @@MODEL_NAME@@
    results:
      - task:
          type: image-text-to-text
          name: Equation image to LaTeX
        dataset:
          type: @@HF_NAMESPACE@@/@@DATASET_NAME@@
          name: @@DATASET_NAME@@ benchmark split
          split: benchmark
        metrics:
          - type: render_identical_rate
            name: Render-identical rate
            value: {{tuned_rate}}
---

# @@MODEL_NAME@@

{{disclaimer}}

Base model: lightonai/LightOnOCR-3-4B at revision {{base_revision}}.

A LoRA adapter, and a merged Q8_0 GGUF for Ollama, that move the base model toward transcribing equation images
into LaTeX closer to what the paper's author actually typed. Input is an image of one display equation, output is LaTeX.

**This is a first, early release.** It is the first model of this project and a first attempt at the problem, so
expect rough edges. We plan to update it in a later version, with no date yet, with a dataset aimed more specifically at each error class
(primes, Greek letters, decorations, I and 1), because this version's training data was not built around them and the
results below show it. Numbers on this page describe this version only and will change.

## Why this project

Mathematical formulas are the part of a PDF where OCR mistakes cost the most. One wrong glyph, a prime read as the
letter l, a Greek letter read as a Latin one, a lost bold or bar, an I read as 1, can change which mathematical object
is meant, and a reader may not notice. We want OCR of formulas in PDFs to be as close to correct as we can get it.

This release is a first step. We work on images of single display equations because they are easy to measure,
and we score them by whether the output typesets to the same bitmap as the paper's LaTeX. We checked full pages only on
50 pages, and on full pages we have not shown an improvement yet (see Results).

## The problem

A general OCR model reads an equation and returns LaTeX that usually looks right at a glance. On a closer look the
notation is sometimes not what the author wrote, and in mathematics a different glyph can be a different object:

- a prime dropped or read as the letter l, or a Greek letter read as a Latin one (upsilon as v)
- a decoration lost or swapped: bold, sans-serif, script, bar and underline (`\mathsf{A}` becomes `A`, `\mathscr{A}`
  becomes `\mathcal{A}`)
- the letter I read as the digit 1

Plain text metrics hide this: two LaTeX strings can differ in one glyph and score as nearly identical. This project
mines the base model's own errors on real papers, trains a LoRA on them, and scores the result by compiling both the
reference and the prediction and comparing the rendered bitmaps ("render-identical").

What we can and cannot say so far (see Results): on this benchmark the base model made few errors on a prime read as l
(1), Greek read as Latin (18) or I read as 1 (3), so there was little to measure and we make no claim for those
classes. The scorer flagged 31 rows as dropped primes and the change was not significant; the prime count is unchanged
in most of those rows, so the flag is probably partly an artifact of the scorer. What we did measure is a smaller
number of `decoration` errors (see the table below) and of errors in the large `other` group, which includes
typesetting choices such as `\mathrm` against `\rm` or which script letter the author used. We are still learning
what that group contains.

![How the dataset, the LoRA and the evaluation fit together](assets/chart_pipeline.svg)

### Examples

Each image is a real benchmark equation. The top image (the goal) is our rendering of the equation from the paper's
LaTeX source with one fixed preamble, cropped to the ink, so it is not a scan of the printed paper and can differ from
it in size and spacing. Below it are what the base model wrote and what this model wrote, both compiled with the same
preamble, so you see what each output looks like.

{{examples_md}}

Some of these classes had very few base-model errors on the benchmark (see the table below). An example here shows that
the model can do something in one case, not that it does so reliably: the prime and Greek-letter examples are single
events, and we make no claim for those classes. We show no example for a dropped prime, because the benchmark has no
clear case: most rows the scorer flags still have all their primes, and the rest lost whole lines or cells. We show none
for I read as 1 either, because the cases we found are long multi-line equations that are hard to read at this size.

These examples were picked by hand to be readable, not drawn at random. The model also makes mistakes: {{r2w}} benchmark crops that the
base model got right are wrong with this model. See Results and the limits below.

## Model details

| | |
|---|---|
| Base model | lightonai/LightOnOCR-3-4B (Apache-2.0), pinned revision above |
| Adapter | LoRA on the language model, rank {{rank}}, alpha {{alpha}}, rsLoRA; vision tower and projector untouched |
| Input / output | one equation image, no text prompt / LaTeX |
| Language | English mathematics papers typeset with LaTeX |
| Formats | PEFT adapter (safetensors) and merged Q8_0 GGUF for Ollama or llama.cpp |
| Developed by | an independent author, not LightOn |

## Files

{{files_table}}

The adapter is safetensors only and the repository contains no custom code and no pickle files.

## Download

You need the Hugging Face CLI (`pip install -U huggingface_hub`). For Ollama or llama.cpp, download the whole
repository. It is about 5 GB, mostly the Q8_0 text GGUF (see the file sizes above):

```
hf download @@HF_NAMESPACE@@/@@MODEL_NAME@@ --local-dir @@MODEL_NAME@@
```

For PEFT you only need the adapter files, about 250 MB. The base model, about 9 GB in bf16, is
downloaded separately by transformers the first time you load it:

```
hf download @@HF_NAMESPACE@@/@@MODEL_NAME@@ --include "adapter_*" --local-dir @@MODEL_NAME@@-adapter
```

Without the CLI, the same from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("@@HF_NAMESPACE@@/@@MODEL_NAME@@", local_dir="@@MODEL_NAME@@")
```

The PEFT snippet below loads the adapter by repository id, so it needs no manual download at all.

## Use with Ollama

From the downloaded folder, build the model:

```
cd @@MODEL_NAME@@
ollama create @@MODEL_NAME@@ -f Modelfile
ollama run @@MODEL_NAME@@
```

The Modelfile uses relative paths, so run the command from inside the downloaded folder. The prompt is the image alone
with no text, and decoding is greedy (temperature 0). The Modelfile sets the context to {{num_ctx}} tokens and the
output cap to {{num_predict}} tokens.

## Use with llama.cpp

The two GGUF files in the repository (`text-q8_0.gguf` and `mmproj-stock-q8_0.gguf`) also load in `llama-server`:

```
llama-server -m text-q8_0.gguf --mmproj mmproj-stock-q8_0.gguf -c 8192 --host 127.0.0.1 --port 8080
```

Send one user message that holds only the image, with `temperature` 0 and `"chat_template_kwargs": {"enable_thinking": false}`
in the request to `/v1/chat/completions`. We tested this path only with the `llama-server` build that ships with
Ollama {{ollama_version}}, on CPU, as a second-engine check of the benchmark (see the code repository); we did not test
other llama.cpp builds.

## Use with PEFT

Load the base model at the pinned revision with transformers {{transformers_version}}, attach the adapter in
this repository's root (`adapter_config.json`, `adapter_model.safetensors`) with peft {{peft_version}}, send a single user turn that contains only the image (no text prompt), and apply
the chat template with `enable_thinking=False`. Use the same processor pixel limits as in training
({{min_px}} to {{max_px}} pixels).

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE, REV = "lightonai/LightOnOCR-3-4B", "{{base_revision}}"
ADAPTER = "@@HF_NAMESPACE@@/@@MODEL_NAME@@"

processor = AutoProcessor.from_pretrained(
    BASE, revision=REV, size={"shortest_edge": {{min_px}}, "longest_edge": {{max_px}}}
)
model = AutoModelForImageTextToText.from_pretrained(BASE, revision=REV, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

messages = [{"role": "user", "content": [{"type": "image"}]}]  # the image alone, no text
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=prompt, images=Image.open("equation.png").convert("RGB"), return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

This snippet follows the settings used for training and evaluation but is not itself part of the tested pipeline;
the tested paths are the Ollama model and the PEFT run in the code repository.

Hardware: the GGUF runs on any backend Ollama or llama.cpp supports, but it was tested only on an AMD ROCm GPU
(gfx1200). Training and export were also done only on ROCm.

## How the GGUF was made

The adapter was merged into the bf16 base weights, the merged model was converted with llama.cpp {{llama_cpp_commit}}
(with `--no-mtp`) and quantized to Q8_0, and the stock multimodal projector was reused unchanged. The export scripts
are `scripts/export_merge.py` and `scripts/export_lora.sh` in the code repository.

## Training data

Trained on {{train_rows}} equation crops from {{train_papers}} arXiv papers, selected from the base model's own
errors and oversampled by error class. Every paper is CC BY or CC0 and the dataset keeps per-record attribution:
https://huggingface.co/datasets/@@HF_NAMESPACE@@/@@DATASET_NAME@@. The benchmark papers are split off by paper and
frozen before any mining or training. We did not train on it, and a leak scan in the code repository checks this.

## Training

LoRA on the language model only (rank {{rank}}, alpha {{alpha}}, rsLoRA), learning rate {{lr}}, {{epochs}} epochs,
checkpoint selected at step {{step}}, {{train_rows}} training rows with SHA-256 {{trainset_sha256}}. The vision tower
and the projector were not trained.

## Results

Held-out benchmark ({{n}} equation crops from {{n_papers}} papers, split by paper and frozen before mining and
training), decided on the render-identical metric:

- render-identical rate: {{stock_rate}} for the base model, {{tuned_rate}} with this model
- difference: {{delta}}, paired paper-level 95% CI [{{ci_low}}, {{ci_high}}]
- crops the base model got right and this model got wrong: {{r2w}}; wrong to right: {{w2r}}
- in relative terms: {{fewer_wrong_pct}} fewer wrong crops ({{wrong_before}} down to {{wrong_after}}), {{more_right_pct}} more
  correct crops, and {{w2r}} fixed for every {{r2w}} broken (about {{flip_ratio}} to 1). The {{r2w}} broken crops are
  {{r2w_of_right_pct}} of the crops the base model got right.
- page-level transfer ({{page_n}} equations from {{page_papers}} papers): {{page_stock_rate}} for the base model,
  {{page_tuned_rate}} with this model

![Render-identical rate, base model against this model](assets/chart_headline.svg)

![Crop by crop: what changed](assets/chart_flips.svg)

"Render-identical" means the reference LaTeX and the prediction are compiled with the same preamble and the two
rendered bitmaps are identical. It is stricter than string matching and catches a changed glyph that a text metric
would forgive. The confidence interval resamples papers, not equations, because equations in one paper are correlated.

### By error class

Counts are from the string-level scorer over the benchmark. A class with too few base-model errors is reported as
"not enough events to claim" instead of a result. `other` collects everything that is not one of the named classes.

{{per_class_table}}

### What changed, by our own hand review

This model is not strictly better: on {{r2w}} benchmark crops the base model was right and this model is wrong
(each one is listed in `evidence/acceptance/ACCEPTANCE.md`). We also read the crops that flipped. The labels were made
by language-model labelling under one written rule; we then read the 46 borderline fixes ourselves and spot-checked the rest. They are approximate and are not part of
the scored result. The per-crop labels are not published yet.

![Hand review: what changed in the crops that flipped](assets/chart_review.svg)

- About 21% of the {{w2r}} fixed crops (139) and about a third of the {{r2w}} broken crops (21) change the mathematics
  or a glyph, including the cases where upright against italic carries meaning (`\\max`, `\\mathrm{d}x`). The rest differ only in how the same formula is spelled: delimiter sizing (`\left` against `\bigl`),
  `cases` against `dcases`, `\le` against `\leq`, `\text` against `\mathrm`, an alignment wrapper, spacing.
- That is roughly 6.6 real fixes for every real break. In points, the real net gain is about 4 to 5 of the
  {{delta}} difference above; most of the rest is matching the papers' LaTeX conventions.
- Matching the author's spelling is a gain for a reader who wants the author's LaTeX, but it is not the same as fewer
  notation mistakes, which was the motivation for this project.
- The real breaks include five glyph swaps: `\Lambda` read as `\wedge`, `\#` as `\sharp`, `\boxplus` as `\oplus`,
  `\vdash` as `\dashv`, and `\bar f_\Gamma` as `\overline{f_\Gamma}`.
- On the dev split, 82% of the crops fixed are in `other` (layout, wrappers, spacing) and 15% in `decoration`.
- Greedy decoding is not stable across engines, so a single flip can be a near-tie. All {{r2w}} breaks were rerun on CPU
  and none reversed.

### Where the remaining errors are

The `other` class is large, so we also sorted every render-identical failure on the benchmark into six buckets. The
buckets come from a simple pattern heuristic, are for reporting only, are not part of the frozen scorer, and were not
used for training. A row falls in the first bucket that fits, so the buckets do not overlap.

![Render-identical failures by bucket, base model against this model](assets/chart_buckets.svg)

{{buckets_table}}

About half of the drop in failures is in layout / sizing, spacing and alignment wrapper rows, which are partly the
author's typesetting choices. The
`content` bucket is still the largest and we have not yet read it closely. The last bucket grew from 35 to 51 crops
and we have not yet found out why. `scripts/error_buckets.py` in the code repository reproduces the table.

## Intended use and limits

Intended for transcribing images of single display equations from mathematics papers into LaTeX. It was trained on
crops rendered from arXiv papers typeset with LaTeX; handwriting, scans, tables and full pages were not part of the
training data. Always check the output of any OCR model before relying on it.

{{limits}}

## What comes next

This release is meant to be updated, with no date yet. The plan, which may change as we learn more:

- a training set built around each error class (primes, Greek letters, decorations, I against 1), so that these classes
  have enough events to measure
- possibly more training on the largest remaining buckets (layout / sizing, spacing, alignment wrappers, content),
  after we have read the `content` bucket more closely
- a new, separate benchmark, because the current one is frozen and its counts for the named classes are too small;
  any claim about the new training would need it
- a look at why the "same text, render differs" bucket grew
- more evidence on full pages, since that is the real target: a larger and separate set of pages, and a look at the
  unbalanced `$$` fences that cost this model equations on two test pages

We would be glad to hear about equations this model gets wrong. Issues on the code repository are welcome.

## Evaluation and reproduction

The benchmark, the scorer and the exact commands are in https://github.com/@@GH_OWNER@@/@@REPO_NAME@@ (see
REPRODUCE.md at the repository root).

## License

Apache License 2.0 for the adapter, the merged GGUF and this packaging. See NOTICE and LICENSE. Third-party credits,
including the base model, are in NOTICE. The example images (`example_*.png`) are our renderings of equations from
papers licensed CC BY 3.0, CC BY 4.0 or CC0 and keep the license of their paper, credited under each image; the charts
(`chart_*.svg`) are ours, built from the published results, under Apache 2.0.

Versions: this is project release v1.0.0, with adapter tag `lora-v1.0.0`, dataset tag `dataset-v1.0.0` and benchmark
`benchmark-v1` (see VERSIONING.md in the code repository).

Contact: @@CONTACT_EMAIL@@

## Citation

```bibtex
@misc{@@MODEL_NAME@@,
  title  = {@@MODEL_NAME@@: a LoRA for mathematical notation in LightOnOCR-3},
  author = {@@COPYRIGHT_HOLDER@@},
  year   = {2026},
  url    = {https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@}
}
```

If you use the benchmark or the scorer, also cite the dataset (https://huggingface.co/datasets/@@HF_NAMESPACE@@/@@DATASET_NAME@@) and the code repository (https://github.com/@@GH_OWNER@@/@@REPO_NAME@@, which has a `CITATION.cff`).
