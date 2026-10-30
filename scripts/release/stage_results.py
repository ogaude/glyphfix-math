"""Stage the results bundle and the code-repo documents into release/code/overlay/ (local only, nothing is uploaded).

Usage: uv run python -m scripts.release.stage_results [--out release/code/overlay] [--allow-missing-attribution]

The overlay is what the export adds on top of the filtered history: README.md, REPRODUCE.md, REBUILD.md, NOTICE,
LICENSE, ATTRIBUTION.md/.jsonl and results/. The scorer evidence is already tracked at evidence/acceptance/ and is
indexed here, never copied. Every number in the documents is read from results.json or sensitivity.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

from scripts.release import charts, public_paths, rename_amendment
from scripts.release.common import (
    DATASET_DIR,
    EXPECTED,
    FREEZE_AMENDMENT_JSON,
    OVERLAY_DIR,
    PHASE4_SOURCES,
    PUBLIC_EVIDENCE_FILES,
    RESULTS_JSON,
    SENSITIVITY_JSON,
    caveats,
    load_json,
    phase4_public_bytes,
    relative_summary,
    render_template,
    sha256_stream,
    strip_internal_labels,
    write_sha256sums,
)

TEMPLATES = Path("release_templates")
REGISTRY = Path("data/models/REGISTRY.json")
STOCK_MODELFILE = Path("data/models/p4-full-1/stock.Modelfile")
SENSITIVITY_DIR = Path("data/benchmark/acceptance/sensitivity")
MINING_FILES = (
    "data/mining/taxonomy.json",
    "data/mining/labels.jsonl",
    "data/mining/MINING_MANIFEST.json",
    "data/mining/taxonomy_candidates.json",
)
UV_PACKAGES = ("numpy", "pillow", "pyarrow", "rapidfuzz", "scipy", "pypdfium2", "pymupdf", "datasets", "huggingface-hub")
TRAIN_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "flash-linear-attention")
FORBIDDEN_IN_OVERLAY = (".planning", "docs/", "/mnt" + "/datas", "/home" + "/ogaude")
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".txt", ""}


class StageError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------- pins
def _lock_versions(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", line)
        if m and m.group(1).lower() in names:
            found[m.group(1).lower()] = m.group(2)
    return found


def _uv_versions(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in data.get("package", []) if p["name"] in names}


def ollama_version(results: dict) -> str:
    versions = set()
    for item in results["inputs"].values():
        if item["path"].endswith(".pred.manifest.json"):
            v = load_json(item["path"]).get("ollama_version")
            if v:
                versions.add(v)
    if len(versions) != 1:
        raise StageError(f"the prediction manifests name {sorted(versions)} as Ollama versions, expected exactly one")
    return versions.pop()


def pins() -> list[tuple[str, str, str]]:
    """(pin, value, source file) triples, each read from its source."""
    from glyphfix_math import render
    from glyphfix_math.acceptance import ACCEPTANCE_SHA256
    from glyphfix_math.normalize import NORMALIZE_RULES_VERSION
    from glyphfix_math.score import SCORER_VERSION, scorer_fingerprint

    results = load_json(RESULTS_JSON)
    amend = load_json(FREEZE_AMENDMENT_JSON)
    reg = next(e for e in load_json(REGISTRY)["entries"] if e["tag"] == "glyphfix-math-lora-q8:p4-full-1")
    canon = load_json("data/corpus/CANON_FREEZE.json")
    bench = load_json("data/benchmark/BENCHMARK_FREEZE.json")
    out: list[tuple[str, str, str]] = [
        ("Python", Path(".python-version").read_text(encoding="utf-8").strip(), ".python-version"),
        ("uv.lock sha256", sha256_stream("uv.lock"), "uv.lock"),
    ]
    for name, ver in sorted(_uv_versions(Path("uv.lock"), UV_PACKAGES).items()):
        out.append((f"{name} (default environment)", ver, "uv.lock"))
    for name, ver in sorted(_lock_versions(Path("requirements/train.lock.txt"), TRAIN_PACKAGES).items()):
        out.append((f"{name} (training environment)", ver, "requirements/train.lock.txt"))
    out += [
        ("requirements/train.lock.txt sha256", sha256_stream("requirements/train.lock.txt"), "requirements/train.lock.txt"),
        ("requirements/convert.lock.txt sha256", sha256_stream("requirements/convert.lock.txt"), "requirements/convert.lock.txt"),
        ("llama.cpp commit", reg["llama_cpp_commit"], "data/models/REGISTRY.json"),
        ("Ollama", ollama_version(results), "prediction manifests listed in results.json inputs"),
        ("texlive image", render.IMAGE, "src/glyphfix_math/render.py"),
        ("scorer version", SCORER_VERSION, "src/glyphfix_math/score.py"),
        ("scorer fingerprint", scorer_fingerprint(), "src/glyphfix_math/score.py"),
        ("canon rules version", canon["canon_rules_version"], "data/corpus/CANON_FREEZE.json"),
        ("canon fingerprint", canon["canon_fingerprint"], "data/corpus/CANON_FREEZE.json"),
        ("normalizer rules version", NORMALIZE_RULES_VERSION, "src/glyphfix_math/normalize.py"),
        ("ACCEPTANCE_SHA256", ACCEPTANCE_SHA256 or "", "src/glyphfix_math/acceptance.py"),
        ("split rule sha256", sha256_stream("data/splits/SPLIT_RULE.json"), "data/splits/SPLIT_RULE.json"),
        ("split SHA256SUMS sha256", sha256_stream("data/splits/SHA256SUMS"), "data/splits/SHA256SUMS"),
        ("benchmark records sha256", bench["records_sha256"], "data/benchmark/BENCHMARK_FREEZE.json"),
        ("benchmark images sha256", bench["images_sha256"], "data/benchmark/BENCHMARK_FREEZE.json"),
        ("base model", f"{reg['base_model']['id']} @ {reg['base_model']['revision']}", "data/models/REGISTRY.json"),
        ("adapter sha256", reg["adapter_sha256"], "data/models/REGISTRY.json"),
        ("training set sha256", reg["trainset_sha256"], "data/models/REGISTRY.json"),
        ("frozen CLI blob before the amendment", amend["old_blob"], "evidence/acceptance/FREEZE_AMENDMENT.json"),
        ("frozen CLI blob after the amendment", amend["new_blob"], "evidence/acceptance/FREEZE_AMENDMENT.json"),
        ("frozen module blob", amend["module_blob"], "evidence/acceptance/FREEZE_AMENDMENT.json"),
    ]
    for key in ("torch", "transformers", "peft"):
        if reg["versions"].get(key) != _lock_versions(Path("requirements/train.lock.txt"), (key,)).get(key):
            raise StageError(f"REGISTRY.json and requirements/train.lock.txt disagree on {key}")
    return out


def render_pins(rows: list[tuple[str, str, str]]) -> str:
    lines = [
        "# Pins",
        "",
        "Versions and digests the published results depend on. Each value is read from the source file named in its row.",
        "",
        "| Pin | Value | Source |",
        "|-----|-------|--------|",
    ]
    lines += [f"| {name} | `{value}` | `{src}` |" for name, value, src in rows]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- results/README.md
def _files_table(paths: list[str], sha_of: dict[str, str] | None = None) -> list[str]:
    lines = ["| Path | sha256 |", "|------|--------|"]
    for p in paths:
        sha = (sha_of or {}).get(p) or sha256_stream(p)
        lines.append(f"| `{p}` | `{sha}` |")
    return lines


def sensitivity_files() -> list[str]:
    return sorted(p.as_posix() for p in SENSITIVITY_DIR.rglob("*") if p.is_file())


def _rename_rerun_lines(amendment2: dict | None) -> list[str]:
    """The rerun record of the rename amendment, when the repository carries it (nothing otherwise)."""
    if amendment2 is None:
        return []
    rr = amendment2["rerun_equivalence"]
    return [
        ("The rename made before the first release was rescored from a cold render cache as well. The record is "
        f"`rerun_equivalence` in `{rename_amendment.AMENDMENT2_REF}`; the values below are rendered from it."),
        "",
        f"- compare: `{rr['compare']}`",
        f"- differing keys: {', '.join(f'`{k}`' for k in rr['differing_keys'])}",
        f"- score exit code: {rr['score_exit']}",
        f"- cold render cache: {str(rr['cold_cache']).lower()}",
        f"- seconds: {rr['seconds']}",
        f"- published results.json sha256: `{rr['published_sha256']}`",
        f"- fresh results.json sha256: `{rr['fresh_sha256']}`",
        "",
    ]


def render_results_index(results: dict, amendment: dict, amendment2: dict | None = None) -> str:
    rr = amendment["rerun_equivalence"]
    inputs = {v["path"]: v["sha256"] for v in results["inputs"].values()}
    lines = [
        "# Results index",
        "",
        ("Results are published exactly as scored. The scorer evidence is tracked at `evidence/acceptance/` in this "
        "repository (at the paths the frozen scorer reads) and is indexed here, not copied. This folder adds the "
        "development reports, the pins and this index."),
        "",
        "## Evidence tracked at evidence/acceptance/",
        "",
        "- `evidence/acceptance/results.json`: the headline, the per-class results, the page-level transfer, the spacing-insensitive column and the code-freeze record",
        "- `evidence/acceptance/flips.jsonl`: the paired flip table, one row per record whose outcome changed",
        "- `evidence/acceptance/sensitivity.json`: the CPU re-run of every flipped record (appendix of the report)",
        "- `evidence/acceptance/env_check.json`: the engine and stock-model check",
        "- `evidence/acceptance/ACCEPTANCE.md`: the report rendered from `results.json` and `sensitivity.json`",
        "- `evidence/acceptance/FREEZE_AMENDMENT.json`: the record of the one change to the frozen scorer, two path constants",
        "",
        *_files_table(list(PUBLIC_EVIDENCE_FILES)),
        "",
        "## Rerun record",
        "",
        ("After the freeze amendment the published numbers were rescored from a cold render cache. The record is "
        "`rerun_equivalence` in `evidence/acceptance/FREEZE_AMENDMENT.json`; the values below are rendered from it."),
        "",
        f"- compare: `{rr['compare']}`",
        f"- differing keys: {', '.join(f'`{k}`' for k in rr['differing_keys'])}",
        f"- score exit code: {rr['score_exit']}",
        f"- cold render cache: {str(rr['cold_cache']).lower()}",
        f"- seconds: {rr['seconds']}",
        f"- published results.json sha256: `{rr['published_sha256']}`",
        f"- fresh results.json sha256: `{rr['fresh_sha256']}`",
        "",
        *_rename_rerun_lines(amendment2),
        "## Inputs of results.json",
        "",
        ("The predictions, manifests and page rows that `results.json` was scored from, with the sha256 recorded in it. "
        "They stay where they are tracked and are not duplicated here."),
        "",
        *_files_table(list(inputs), inputs),
        "",
        "## CPU sensitivity predictions",
        "",
        "The predictions behind `evidence/acceptance/sensitivity.json`:",
        "",
        *_files_table(sensitivity_files()),
        "",
        "## Mining taxonomy and labels",
        "",
        ("`labels.jsonl` covers more crops than the released training rows: it lists every mined training-split crop, "
        "and the dataset keeps the stratified subset the model was trained on."),
        "",
        *_files_table(list(MINING_FILES)),
        "",
        "## Files in this folder",
        "",
        "- `DEV_ANALYSIS.md`: gain on the development split, by class",
        "- `MINING.md`: the error-mining report",
        "- `PARITY_DIAGNOSIS.md`: why the bf16 model and the GGUF served by Ollama differ at near-ties",
        "- `dev_gain_by_class.json`: the data behind `DEV_ANALYSIS.md`",
        "- `PINS.md`: pinned versions and digests",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- documents
def public_wording(line: str) -> str:
    """Column headers of the generated tables say base model and this model, as the cards do (the frozen report keeps stock and tuned)."""
    return line.replace("| | stock | tuned |", "| | base model | this model |") if line.startswith("| | stock | tuned |") else line


MATCH_RATE_NOTE = (
    "The equation match rate is the share of the reference display equations on the test pages that could be paired with "
    "an equation in the model's output. It says nothing about whether the LaTeX was right. An unmatched equation counts "
    "as a failure for both models in the page-level rate, so a lower match rate lowers the page-level score."
)


def results_markdown(results: dict) -> str:
    from scripts import acceptance_report as ar

    res = copy.deepcopy(results)
    parts = [f"Verdict: {res['verdict'].upper()}. {res['statements']['verdict']}.", ""]
    parts += [public_wording(strip_internal_labels(x)) for x in ar._headline(res)]
    parts += [public_wording(strip_internal_labels(x)) for x in ar._pages(copy.deepcopy(results))]
    parts += [MATCH_RATE_NOTE, ""]
    parts += ["**Limits of these results**", ""]
    parts += [f"- {s}" for s in caveats(results)]
    parts.append("")
    # the README puts these under its own "## Results" heading
    return "\n".join("#" + x if x.startswith("## ") else x for x in parts)


def _stock_param(name: str) -> str:
    m = re.search(rf"^PARAMETER {name} (\S+)$", STOCK_MODELFILE.read_text(encoding="utf-8"), re.MULTILINE)
    if not m:
        raise StageError(f"{STOCK_MODELFILE} has no PARAMETER {name}")
    return m.group(1)


def slot_values(results: dict, sensitivity: dict) -> dict[str, str]:
    from glyphfix_math import render

    s = sensitivity["summary"]
    return {
        "results_md": results_markdown(results),
        **relative_summary(results["headline"]),
        "r2w": str(results["headline"]["flip"]["right_to_wrong"]),
        "w2r": str(results["headline"]["flip"]["wrong_to_right"]),
        "texlive_image": render.IMAGE,
        "train_rows": f"{EXPECTED['train_rows']:,}",
        "train_papers": str(EXPECTED["train_papers"]),
        "bench_rows": f"{EXPECTED['bench_rows']:,}",
        "bench_papers": str(EXPECTED["bench_papers"]),
        "ollama_version": ollama_version(results),
        "stock_num_ctx": _stock_param("num_ctx"),
        "stock_num_predict": _stock_param("num_predict"),
        "r2w_held": str(s["r2w"]["held"]),
        "r2w_total": str(s["r2w"]["total"]),
        "w2r_held": str(s["w2r"]["held"]),
        "w2r_total": str(s["w2r"]["total"]),
        "reversed": str(s["r2w"]["reversed"] + s["w2r"]["reversed"]),
    }


def _is_text(p: Path) -> bool:
    return p.suffix in TEXT_SUFFIXES and p.name != "SHA256SUMS"


def scan_overlay(root: Path) -> list[str]:
    hits = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and _is_text(p):
            text = p.read_text(encoding="utf-8", errors="replace")
            for needle in FORBIDDEN_IN_OVERLAY:
                if needle in text:
                    hits.append(f"{p.relative_to(root).as_posix()}: contains {needle!r}")
    return hits


def stage(out: Path = OVERLAY_DIR, attribution_dir: Path = DATASET_DIR, allow_missing_attribution: bool = False) -> dict:
    results = load_json(RESULTS_JSON)
    sensitivity = load_json(SENSITIVITY_JSON)
    amendment = load_json(FREEZE_AMENDMENT_JSON)
    attribution = [attribution_dir / "ATTRIBUTION.md", attribution_dir / "ATTRIBUTION.jsonl"]
    missing = [str(p) for p in attribution if not p.is_file()]
    if missing and not allow_missing_attribution:
        raise StageError(f"missing {', '.join(missing)}; stage the dataset first or pass --allow-missing-attribution")
    for src in PHASE4_SOURCES:
        if not src.is_file():
            raise StageError(f"missing source {src}")

    if out.exists():
        shutil.rmtree(out)
    res = out / "results"
    res.mkdir(parents=True)
    for src in PHASE4_SOURCES:
        (res / src.name).write_bytes(phase4_public_bytes(src))
    (res / "PINS.md").write_text(render_pins(pins()), encoding="utf-8")
    charts.build(res / "charts", results)
    amendment2 = load_json(rename_amendment.AMENDMENT2_PATH) if rename_amendment.AMENDMENT2_PATH.is_file() else None
    (res / "README.md").write_text(render_results_index(results, amendment, amendment2), encoding="utf-8")

    values = slot_values(results, sensitivity)
    for name in ("README.md", "REPRODUCE.md", "REBUILD.md"):
        (out / name).write_text(render_template((TEMPLATES / name).read_text(encoding="utf-8"), values), encoding="utf-8")
    (out / "CITATION.cff").write_text(render_template((TEMPLATES / "CITATION.cff").read_text(encoding="utf-8"), values), encoding="utf-8")
    (out / "NOTICE").write_text(render_template((TEMPLATES / "NOTICE.code.txt").read_text(encoding="utf-8"), values), encoding="utf-8")
    shutil.copyfile(TEMPLATES / "LICENSE-Apache-2.0.txt", out / "LICENSE")
    for p in attribution:
        if p.is_file():
            shutil.copyfile(p, out / p.name)

    hits = scan_overlay(out)
    if hits:
        shutil.rmtree(out)
        raise StageError("forbidden strings in the overlay: " + "; ".join(hits))
    write_sha256sums(out)
    kept = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    return {"out": str(out), "files": len(kept), "attribution": not missing}


def referenced_files(text: str) -> list[str]:
    """Repository files a document refers to: `-m` modules and scripts/ or evidence/ paths."""
    refs: set[str] = set()
    for mod in re.findall(r"python -m ([A-Za-z_][\w.]*)", text):
        base = mod.replace(".", "/")
        top = mod.split(".")[0]
        refs.add(f"{base}.py" if top == "scripts" else f"src/{base}.py")
    refs.update(re.findall(r"\b(?:scripts|evidence)/[\w./-]*\.\w+", text))
    return sorted(refs)


def unresolved_references(text: str) -> list[str]:
    bad = []
    for ref in referenced_files(text):
        if public_paths.classify(ref) is None:
            bad.append(ref)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OVERLAY_DIR)
    ap.add_argument("--allow-missing-attribution", action="store_true")
    a = ap.parse_args(argv)
    try:
        info = stage(a.out, allow_missing_attribution=a.allow_missing_attribution)
    except (StageError, OSError, KeyError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
