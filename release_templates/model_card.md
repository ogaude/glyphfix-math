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
datasets:
  - @@HF_NAMESPACE@@/@@DATASET_NAME@@
---

# @@MODEL_NAME@@

{{disclaimer}}

Base model: lightonai/LightOnOCR-3-4B at revision {{base_revision}}.

A LoRA adapter, and a merged Q8_0 GGUF for Ollama, that reduce mistakes on mathematical notation when the base model
transcribes equation images: a prime read as the letter l, a dropped prime, a Greek letter read as a Latin one,
decorations such as bold, bar and underline, and I read as 1. These are the error classes the training data targets.
The results below say which of them were actually measured to improve.

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
`adapter/` with peft {{peft_version}}, send a single user turn that contains only the image (no text prompt), and apply
the chat template with `enable_thinking=False`. Use the same processor pixel limits as in training
({{min_px}} to {{max_px}} pixels).

## How the GGUF was made

The adapter was merged into the bf16 base weights, the merged model was converted with llama.cpp {{llama_cpp_commit}}
(with `--no-mtp`) and quantized to Q8_0, and the stock multimodal projector was reused unchanged. The export scripts
are `scripts/export_merge.py` and `scripts/export_lora.sh` in the code repository.

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

## Limits

{{limits}}

## Evaluation and reproduction

The benchmark, the scorer and the exact commands are in https://github.com/@@GH_OWNER@@/@@REPO_NAME@@ (see
REPRODUCE.md at the repository root).

## License

Apache License 2.0. See NOTICE and LICENSE. Third-party credits, including the base model, are in NOTICE.

Contact: @@CONTACT_EMAIL@@
