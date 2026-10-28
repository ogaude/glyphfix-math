---
license: cc-by-4.0
pretty_name: @@DATASET_NAME@@
task_categories:
  - image-to-text
language:
  - en
multilinguality:
  - monolingual
source_datasets:
  - original
annotations_creators:
  - machine-generated
tags:
  - ocr
  - latex
  - math
  - formula-recognition
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: benchmark
        path: data/benchmark-*.parquet
---

# @@DATASET_NAME@@

Images of display equations from open-access mathematics papers, paired with the canonicalized LaTeX that produced
them. The images are rendered by us from the papers' own LaTeX source, so each label should match the rendered glyphs,
though we have not checked every row by hand. The set was put together with glyph-level notation errors of OCR models
in mind: primes, Greek letters read as Latin ones, bold, bar and underline decorations, and the letter I read as the
digit 1. It does not yet contain enough examples of those classes to measure them (see the status below).

This dataset is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn.

## Status: an initial release

This is our first release and we want to be honest about it. The train split is what the first LoRA was trained on, and
the benchmark is the frozen set it was scored on. They were not built class by class: on this benchmark the base model
made few errors on a prime read as l, Greek read as Latin or I read as 1 (and the dropped-prime flags are probably
partly a scorer artifact), so those classes have too few examples to measure anything, and about half of the drop in
failures was in layout, spacing and alignment wrapper rows. The model card
has the per-class table and a breakdown of the remaining errors.

We want to build on it, with no date yet. The plan, which may change as we learn more:

- a training set built around each error class (primes, Greek letters, decorations, I against 1), so that these classes
  have enough events to measure
- possibly more training on the largest remaining buckets (layout / sizing, spacing, alignment wrappers, content),
  after we have read the `content` bucket more closely
- a new, separate benchmark, because the current one is frozen and its counts for the named classes are too small;
  any claim about the new training would need it
- a look at why the "same text, render differs" bucket grew

New rows would come as a new version of this dataset. The current benchmark stays frozen so that results on it remain
comparable.

## Splits

| Split | Rows | Papers | CC0-1.0 rows |
|-------|------|--------|--------------|
| train | {{train_rows}} | {{train_papers}} | {{train_cc0}} |
| benchmark | {{bench_rows}} | {{bench_papers}} | {{bench_cc0}} |

The splits are disjoint by paper. The benchmark split is an evaluation set: it is held out by paper and was frozen
before any mining or training. Do not train on it; a model trained on it can no longer be compared on this benchmark.
The leak scan that enforces this lives in the code repository (`python -m glyphfix_math.leakscan`).

The train split is the training set of the released LoRA and nothing else: other crops that were mined or considered
during development are not released. The mining labels in the code repository cover {{mined_records}} mined train
crops, which is more than the train split holds.

## Fields

| Field | Meaning |
|-------|---------|
| id | stable crop identifier |
| image | PNG crop of the rendered equation |
| latex | canonicalized LaTeX of the equation |
| target_text | train only: the training target, the LaTeX in a fenced block |
| arxiv_id, paper_version | source paper and the pinned version the license was checked on |
| title, authors | title and full author list of the source paper |
| license_spdx, license_uri | license of the source paper (CC-BY-4.0, CC-BY-3.0 or CC0-1.0) |
| arxiv_url | versioned link to the source paper |
| flags | notation classes found in the equation |
| image_sha256, bitmap_sha256 | hash of the PNG file and of the rendered bitmap |
| width, height | image size in pixels |
| group, mining_class, control_glyph, tags | train only: how the row was selected |
| render_image_digest, preamble_sha256, canon_version, builder_version, env | benchmark only: render provenance |

## Download and use

Download everything with the Hugging Face CLI (`pip install -U huggingface_hub`):

```
hf download @@HF_NAMESPACE@@/@@DATASET_NAME@@ --repo-type dataset --local-dir @@DATASET_NAME@@
```

Or load a split with the `datasets` library:

```python
from datasets import load_dataset

train = load_dataset("@@HF_NAMESPACE@@/@@DATASET_NAME@@", split="train")
benchmark = load_dataset("@@HF_NAMESPACE@@/@@DATASET_NAME@@", split="benchmark")
print(train[0]["latex"])
```

Keep the `ATTRIBUTION.md` file with any rows you reuse.

## Intended use and limitations

Intended for training and evaluating OCR models on display equations, in particular on glyph-level notation. All
equations come from arXiv mathematics papers typeset in LaTeX and rendered with one pinned preamble, so the images
are clean renderings: no handwriting, scans, noise or page context. Papers are limited to CC BY 3.0, CC BY 4.0 and
CC0 versions, which is a small and self-selected part of arXiv. The `latex` label is canonicalized, so a model can
be correct and still differ from it as a string; the released benchmark scores by rendering, not by string match.
If you slice the benchmark by the `flags` field, expect small counts for primes, Greek letters and I against 1: a
class with few rows cannot support a claim about a model, so check the counts before you compare.

## License

The dataset repository is licensed CC BY 4.0. Every record carries the license of its source paper in
`license_spdx` and `license_uri`; we included only papers that our checks found under CC BY 3.0, CC BY 4.0 or CC0, with the version pinned because the license is per paper version. Please check the license of a paper yourself before you rely on it. {{train_cc0}} train rows
and {{bench_cc0}} benchmark rows come from CC0-1.0 papers. Our added labels and annotations are CC BY 4.0, Copyright 2026 @@COPYRIGHT_HOLDER@@.

Versions: this is project release v1.0.0, dataset tag `dataset-v1.0.0`, benchmark `benchmark-v1` (see VERSIONING.md in the code repository).

## Attribution

`ATTRIBUTION.md` and `ATTRIBUTION.jsonl` list every source paper with title, full author list, version, license, link,
split and row count. If you reuse rows, keep the attribution of their papers.

## How it was built

![How the dataset, the LoRA and the evaluation fit together](assets/chart_pipeline.svg)

Equations were extracted from each paper's LaTeX source, canonicalized, and re-rendered to images with pdflatex in a
pinned TeX Live image. The `latex` column is the canonicalized LaTeX, not the author's original source, which is an
indication of modification under CC BY 4.0 section 3(a)(1)(B). The page images used for page-level checks are not part
of this dataset. The raw equation body and its source file position are not shipped.

## Citation

```bibtex
@misc{@@DATASET_NAME@@,
  title  = {@@DATASET_NAME@@: equation crops for glyph-level math OCR},
  author = {@@COPYRIGHT_HOLDER@@},
  year   = {2026},
  url    = {https://huggingface.co/datasets/@@HF_NAMESPACE@@/@@DATASET_NAME@@}
}
```

Please also cite the source papers listed in `ATTRIBUTION.md` for any rows you reuse.

## Links and contact

- Model: https://huggingface.co/@@HF_NAMESPACE@@/@@MODEL_NAME@@
- Code: https://github.com/@@GH_OWNER@@/@@REPO_NAME@@

To ask for a record to be removed, open an issue on the GitHub repository or write to @@CONTACT_EMAIL@@. Records are
removed on request.
