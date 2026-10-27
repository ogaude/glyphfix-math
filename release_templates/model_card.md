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

A small LoRA adapter, and a merged Q8_0 GGUF for Ollama, that nudge the base model toward transcribing equation images
into LaTeX closer to what the paper's author actually typed. Input is an image of one display equation, output is LaTeX.

**This is a first, early release.** It is the first model of this project and a first attempt at the problem, so
expect rough edges. We plan to update it soon with a dataset aimed more specifically at each error class (primes,
Greek letters, decorations, I and 1), because this version's training data was not built around them and the results
below show it. Numbers on this page describe this version only and will change.

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

What we can and cannot say so far (see Results): on this benchmark the base model made very few errors on primes,
Greek letters or I and 1, so there was little to improve and we make no claim there. What we did measure is a smaller
number of `decoration` errors (see the table below) and of errors in the large `other` group, which includes
typesetting choices such as `\mathrm` against `\rm` or which script letter the author used. We are still learning
what that group contains.

### Examples

Each image is a real benchmark equation: the paper's own typesetting on top (the goal), then what the base model wrote
and what this model wrote, both compiled with LaTeX so you see what each output actually looks like.

{{examples_md}}

Some of these classes had very few base-model errors on the benchmark (see the table below). An example here shows that
the model can do something in one case, not that it does so reliably: the prime and Greek-letter examples are single
events, and we make no claim for those classes. We show no example for I read as 1, because the cases we found are long
multi-line equations that are hard to read at this size.

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

## Use with Ollama

Download the whole folder, change into it and build the model:

```
cd <downloaded folder>
ollama create @@MODEL_NAME@@ -f Modelfile
```

The Modelfile uses relative paths, so run the command from inside the downloaded folder. The prompt is the image alone
with no text, and decoding is greedy (temperature 0). The Modelfile sets the context to {{num_ctx}} tokens and the
output cap to {{num_predict}} tokens.

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
frozen before any mining or training, and never used to train.

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
- page-level transfer ({{page_n}} equations from {{page_papers}} papers): {{page_stock_rate}} for the base model,
  {{page_tuned_rate}} with this model

"Render-identical" means the reference LaTeX and the prediction are compiled with the same preamble and the two
rendered bitmaps are identical. It is stricter than string matching and catches a changed glyph that a text metric
would forgive. The confidence interval resamples papers, not equations, because equations in one paper are correlated.

### By error class

Counts are from the string-level scorer over the benchmark. A class with too few base-model errors is reported as
"not enough events to claim" instead of a result. `other` collects everything that is not one of the named classes.

{{per_class_table}}

## Intended use and limits

Intended for transcribing images of single display equations from mathematics papers into LaTeX. It was trained on
crops rendered from arXiv papers typeset with LaTeX; handwriting, scans, tables and full pages were not part of the
training data. Always check the output of any OCR model before relying on it.

{{limits}}

## What comes next

This release will be updated soon. The plan, which may change as we learn more:

- a training set built around each error class, with more examples of primes, Greek letters, decorations and I
  against 1, so that these classes have enough events to measure
- a new, separate benchmark for those classes, because the current one is frozen and has too few errors in them
- a check of what the large `other` group contains, so that it can be split into named classes

We would be glad to hear about equations this model gets wrong. Issues on the code repository are welcome.

## Evaluation and reproduction

The benchmark, the scorer and the exact commands are in https://github.com/@@GH_OWNER@@/@@REPO_NAME@@ (see
REPRODUCE.md at the repository root).

## License

Apache License 2.0. See NOTICE and LICENSE. Third-party credits, including the base model, are in NOTICE.

Contact: @@CONTACT_EMAIL@@

## Citation

```bibtex
@misc{@@MODEL_NAME@@,
  title  = {@@MODEL_NAME@@: a LoRA for mathematical notation in LightOnOCR-3},
  author = {@@GH_OWNER@@},
  year   = {2026},
  url    = {https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@}
}
```
