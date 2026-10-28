"""Render review drafts (HTML) of the model card, the dataset card and the repository README.

Usage: uv run --with markdown python -m scripts.release.render_drafts [--out docs]

The drafts come from the release templates with preview names filled in, the way staging would fill them, but without
copying the 4 GB GGUF: the file table is read from the already staged release/model folder. Images and charts are read
from docs/card-examples/ (regenerate them with scripts.release.card_examples and scripts.release.charts). The drafts
are for reading only and are never published. Needs the `markdown` package, which is not a project dependency.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts import error_buckets
from scripts.release import stage_model, stage_results
from scripts.release.common import (
    PHASE5_EVIDENCE,
    REGISTRY_TAG,
    RESULTS_JSON,
    SENSITIVITY_JSON,
    load_json,
    render_template,
)

PREVIEW_NAMES = {
    "@@DATASET_NAME@@": "glyphfix-math-equations",
    "@@MODEL_NAME@@": "glyphfix-math-lora",
    "@@REPO_NAME@@": "glyphfix-math",
    "@@HF_NAMESPACE@@": "ogaude",
    "@@GH_OWNER@@": "ogaude",
    "@@CONTACT_EMAIL@@": "contact@example.invalid",
    "@@COPYRIGHT_HOLDER@@": "Olivier Gaude",
}
TEMPLATES = Path("release_templates")
MODEL_DIR = Path("release/model")
DATASET_MANIFEST = Path("release/dataset/DATASET_MANIFEST.json")
EXAMPLES = Path("docs/card-examples/examples.json")
CSS = """body{font:16px/1.6 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px;color:#1c1c1c;background:#fff}
h1,h2{border-bottom:1px solid #ddd;padding-bottom:.3em}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#f3f3f3}
code,pre{font-family:ui-monospace,monospace;background:#f3f3f3;border-radius:4px}code{padding:1px 4px;font-size:.9em;word-break:break-all}pre{padding:12px;overflow-x:auto}pre code{padding:0}img{max-width:100%}
details{margin:1rem 0;border:1px solid #ddd;border-radius:6px;padding:.5rem 1rem}summary{cursor:pointer;color:#555}
@media(prefers-color-scheme:dark){body{background:#16181c;color:#e4e4e4}h1,h2,th,td,details{border-color:#444}th,code,pre{background:#262a30}summary{color:#aaa}a{color:#7cb7ff}}"""


def fill_names(text: str) -> str:
    for token, value in PREVIEW_NAMES.items():
        text = text.replace(token, value)
    return text


def to_html(title: str, note: str, markdown_text: str) -> str:
    import markdown

    front = ""
    m = re.match(r"---\n(.*?)\n---\n", markdown_text, re.DOTALL)
    if m:
        front = f"<details><summary>Hub metadata (YAML front matter)</summary><pre>{m.group(1)}</pre></details>"
        markdown_text = markdown_text[m.end() :]
    markdown_text = markdown_text.replace("](assets/", "](card-examples/").replace("](results/charts/", "](card-examples/")
    body = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{CSS}</style></head><body><p><strong>Draft for review.</strong> {note}</p>{front}{body}</body></html>\n"
    )


def _file_table() -> list[tuple[str, int, str]]:
    sums = dict(reversed(line.split("  ", 1)) for line in (MODEL_DIR / "SHA256SUMS").read_text().splitlines() if line)
    rels = ("adapter_config.json", "adapter_model.safetensors", "text-q8_0.gguf", "mmproj-stock-q8_0.gguf")
    return [(rel, (MODEL_DIR / rel).stat().st_size, sums[rel]) for rel in rels]


def model_card() -> str:
    entry = next(e for e in load_json(Path("data/models/REGISTRY.json"))["entries"] if e["tag"] == REGISTRY_TAG)
    stock, tuned = (error_buckets.count(error_buckets.load(d)) for d in error_buckets.DEFAULT_DIRS)
    return stage_model.render_model_card(
        (TEMPLATES / "model_card.md").read_text(encoding="utf-8"),
        load_json(RESULTS_JSON),
        entry,
        _file_table(),
        (MODEL_DIR / "Modelfile").read_text(encoding="utf-8"),
        examples=json.loads(EXAMPLES.read_text(encoding="utf-8")),
        buckets_table=error_buckets.render(stock, tuned),
    )


def dataset_card() -> str:
    m = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    values = {
        "train_rows": m["rows"]["train"],
        "train_papers": m["papers"]["train"],
        "bench_rows": m["rows"]["benchmark"],
        "bench_papers": m["papers"]["benchmark"],
        "train_cc0": m["licenses"]["train"].get("CC0-1.0", 0),
        "bench_cc0": m["licenses"]["benchmark"].get("CC0-1.0", 0),
        "mined_records": m["mined_records"],
    }
    return render_template((TEMPLATES / "dataset_card.md").read_text(encoding="utf-8"), values)


def repo_readme() -> str:
    values = stage_results.slot_values(load_json(RESULTS_JSON), load_json(SENSITIVITY_JSON))
    return render_template((TEMPLATES / "README.md").read_text(encoding="utf-8"), values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args(argv)
    note = "Rendered from the release templates with preview names filled in. Not staged or published. "
    drafts = {
        "huggingface-model-readme-draft.html": ("glyphfix-math-lora (Hugging Face model card, DRAFT)", model_card()),
        "huggingface-dataset-readme.html": ("glyphfix-math-equations (Hugging Face dataset card, DRAFT)", dataset_card()),
        "github-readme-draft.html": (
            "glyphfix-math (GitHub README, DRAFT)",
            repo_readme() + "\n\n(Relative links to repository files do not resolve in this preview.)\n",
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (title, text) in drafts.items():
        (args.out / name).write_text(to_html(title, note, fill_names(text)), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "files": list(drafts), "evidence": str(PHASE5_EVIDENCE)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
