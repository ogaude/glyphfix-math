"""Results bundle, pins and code-repo documents staged into the overlay."""

from __future__ import annotations

import copy
import json
import re
import tomllib
from pathlib import Path

import pytest

from glyphfix_math import render
from scripts import acceptance_report as ar
from scripts.release import charts, public_paths
from scripts.release import common as C
from scripts.release import rename_amendment as RA
from scripts.release import stage_results as SR

RESULTS = json.loads(Path("evidence/acceptance/results.json").read_text(encoding="utf-8"))
SENS = json.loads(Path("evidence/acceptance/sensitivity.json").read_text(encoding="utf-8"))
AMEND = json.loads(Path("evidence/acceptance/FREEZE_AMENDMENT.json").read_text(encoding="utf-8"))
TEMPLATES = ("README.md", "REPRODUCE.md", "REBUILD.md", "NOTICE.code.txt")

needs_phase4 = pytest.mark.skipif(not all(p.is_file() for p in C.PHASE4_SOURCES), reason="development-only Phase 4 sources are not in this checkout")


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    base = tmp_path_factory.mktemp("overlay")
    attr = base / "attr"
    attr.mkdir()
    (attr / "ATTRIBUTION.md").write_text("# Attribution\n", encoding="utf-8")
    (attr / "ATTRIBUTION.jsonl").write_text('{"arxiv_id": "x"}\n', encoding="utf-8")
    out = base / "overlay"
    SR.stage(out, attribution_dir=attr)
    return out


def read(overlay, rel):
    return (overlay / rel).read_text(encoding="utf-8")


@needs_phase4
def test_phase4_files_are_byte_copies_and_nothing_else_is_duplicated(overlay):
    for src in C.PHASE4_SOURCES:
        assert (overlay / "results" / src.name).read_bytes() == C.phase4_public_bytes(src)
    names = {p.name for p in (overlay / "results").iterdir()}
    assert names == {*(p.name for p in C.PHASE4_SOURCES), "PINS.md", "README.md", "charts"}
    assert {p.name for p in (overlay / "results" / "charts").iterdir()} == set(charts.FILES)
    for name in ("chart_headline.svg", "chart_flips.svg"):
        assert f"results/charts/{name}" in read(overlay, "README.md")
    for banned in ("rerun_check.json", "results.json", "flips.jsonl", "ACCEPTANCE.md", "sensitivity.json", "env_check.json", "FREEZE_AMENDMENT.json"):
        assert not (overlay / "results" / banned).exists()
    assert not any("dryrun" in n or "precheck" in n for n in names)
    for d in ("docs", "evidence"):
        assert not (overlay / d).exists()


@needs_phase4
def test_results_index_lists_every_file_with_its_sha256(overlay):
    text = read(overlay, "results/README.md")
    for f in C.PUBLIC_EVIDENCE_FILES:
        assert f"`{f}` | `{C.sha256_stream(f)}`" in text
    for v in RESULTS["inputs"].values():
        assert f"`{v['path']}` | `{v['sha256']}`" in text
        assert public_paths.classify(v["path"]) is not None
    for p in SR.sensitivity_files():
        assert f"`{p}` | `{C.sha256_stream(p)}`" in text and public_paths.classify(p) is not None
    for p in SR.MINING_FILES:
        assert f"`{p}` | `{C.sha256_stream(p)}`" in text and public_paths.classify(p) is not None
    assert "labels.jsonl" in text and "more crops than the released training rows" in text


@needs_phase4
def test_rerun_record_is_rendered_from_the_amendment(overlay):
    text = read(overlay, "results/README.md")
    rr = AMEND["rerun_equivalence"]
    assert "evidence/acceptance/FREEZE_AMENDMENT.json" in text and "rerun_equivalence" in text
    assert f"compare: `{rr['compare']}`" in text
    for k in rr["differing_keys"]:
        assert f"`{k}`" in text
    assert f"score exit code: {rr['score_exit']}" in text
    assert f"cold render cache: {str(rr['cold_cache']).lower()}" in text
    assert rr["published_sha256"] in text and rr["fresh_sha256"] in text
    assert "cold render cache" in text


@needs_phase4
def test_release_check_rule_for_inputs_digest_holds(overlay):
    text = read(overlay, "results/README.md")
    inputs = {v["path"]: v["sha256"] for v in RESULTS["inputs"].values()}
    hexre = re.compile(r"\b[0-9a-f]{64}\b")
    for line in text.splitlines():
        for path, sha in inputs.items():
            if path in line:
                found = hexre.findall(line)
                assert not found or sha in found


@needs_phase4
def test_pins_equal_values_recomputed_from_their_sources(overlay):
    from glyphfix_math.acceptance import ACCEPTANCE_SHA256
    from glyphfix_math.normalize import NORMALIZE_RULES_VERSION
    from glyphfix_math.score import SCORER_VERSION, scorer_fingerprint

    text = read(overlay, "results/PINS.md")
    uv = {p["name"]: p["version"] for p in tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))["package"]}
    lock = Path("requirements/train.lock.txt").read_text(encoding="utf-8")
    expected = [
        Path(".python-version").read_text(encoding="utf-8").strip(),
        uv["pyarrow"],
        uv["rapidfuzz"],
        re.search(r"^torch==(\S+)", lock, re.MULTILINE).group(1),
        re.search(r"^transformers==(\S+)", lock, re.MULTILINE).group(1),
        re.search(r"^peft==(\S+)", lock, re.MULTILINE).group(1),
        render.IMAGE,
        SCORER_VERSION,
        scorer_fingerprint(),
        NORMALIZE_RULES_VERSION,
        ACCEPTANCE_SHA256,
        json.loads(Path("data/corpus/CANON_FREEZE.json").read_text(encoding="utf-8"))["canon_fingerprint"],
        json.loads(Path("data/corpus/CANON_FREEZE.json").read_text(encoding="utf-8"))["canon_rules_version"],
        json.loads(Path("data/models/REGISTRY.json").read_text(encoding="utf-8"))["entries"][0]["llama_cpp_commit"],
        C.sha256_stream("uv.lock"),
        C.sha256_stream("data/splits/SPLIT_RULE.json"),
        AMEND["old_blob"],
        AMEND["new_blob"],
        json.loads(Path("data/benchmark/baseline/stock_q8.pred.manifest.json").read_text(encoding="utf-8"))["ollama_version"],
    ]
    for value in expected:
        assert f"`{value}`" in text, value
    assert "frozen CLI blob before the amendment" in text and "frozen CLI blob after the amendment" in text


def test_pins_fail_when_registry_and_lock_disagree(monkeypatch):
    real = SR.load_json

    def patched(path):
        d = real(path)
        if str(path).endswith("REGISTRY.json"):
            d = copy.deepcopy(d)
            for e in d["entries"]:
                e["versions"]["torch"] = "0.0.0"
        return d

    monkeypatch.setattr(SR, "load_json", patched)
    with pytest.raises(SR.StageError, match="disagree"):
        SR.pins()


@needs_phase4
@pytest.mark.parametrize("name", ["README.md", "REPRODUCE.md", "REBUILD.md", "NOTICE"])
def test_documents_render_without_slots_and_keep_placeholders(overlay, name):
    text = read(overlay, name)
    assert not re.search(r"\{\{[a-z_][a-z0-9_]*\}\}", text)
    if name == "README.md":
        for token in ("@@REPO_NAME@@", "@@HF_NAMESPACE@@", "@@CONTACT_EMAIL@@", "@@GH_OWNER@@", "@@DATASET_NAME@@", "@@MODEL_NAME@@"):
            assert token in text
    if name == "NOTICE":
        assert "@@REPO_NAME@@" in text and "@@COPYRIGHT_HOLDER@@" in text
        assert "not affiliated with, endorsed by, or sponsored by LightOn" in text


@needs_phase4
def test_citation_cff_staged_with_placeholders(overlay):
    t = read(overlay, "CITATION.cff")
    assert not re.search(r"\{\{[a-z_][a-z0-9_]*\}\}", t)
    for token in ("@@REPO_NAME@@", "@@GH_OWNER@@", "@@HF_NAMESPACE@@", "@@DATASET_NAME@@", "@@MODEL_NAME@@", "@@COPYRIGHT_HOLDER@@"):
        assert token in t


@needs_phase4
def test_readme_content(overlay):
    t = read(overlay, "README.md")
    # (a) what the project is
    for s in ("LightOnOCR-3-4B", "LoRA", "prime", "Greek", "bold, bar and underline", "I read as 1", "dataset"):
        assert s in t
    # (b) results: verdict, exact generator lines, caveats, links
    assert f"Verdict: {RESULTS['verdict'].upper()}. {RESULTS['statements']['verdict']}." in t
    gen = [C.strip_internal_labels(x) for x in [*ar._headline(copy.deepcopy(RESULTS)), *ar._pages(copy.deepcopy(RESULTS))]]
    # the generator lines appear verbatim, except that heading lines sit one level lower under the README's Results heading
    gen = [SR.public_wording("#" + x if x.startswith("## ") else x) for x in gen]
    for line in gen:
        assert line in t.splitlines() or line == ""
    block = "\n".join(gen)
    assert block in t
    assert "(D-" not in t
    for s in C.caveats(RESULTS):
        assert s in t
    assert "evidence/acceptance/ACCEPTANCE.md" in t and "results/README.md" in t
    # (c) reproduce
    assert "uv run python -m scripts.release.reproduce" in t and "REPRODUCE.md" in t and "REBUILD.md" in t
    assert f"docker pull {render.IMAGE}" in t
    # (d) evidence
    assert "evidence/acceptance/" in t and "FREEZE_AMENDMENT.json" in t
    # (e) licenses
    for s in ("Apache-2.0", "CC BY 4.0", "CC0 1.0", "Our labels and annotations"):
        assert s in t
    # (f) not affiliated
    assert "not affiliated with, endorsed by, or sponsored by LightOn" in t
    # (g) no chosen name, no docs/, no planning, no private projects
    assert "docs/" not in t and ".planning" not in t and "nexa" + "ris" not in t.lower()
    # section order
    order = [t.index(h) for h in ("## What is where", "## Results", "## Reproduce", "## Rebuild", "## Licenses", "## Disclaimer", "## Contact")]
    assert order == sorted(order)


@needs_phase4
def test_reproduce_md_numbers_come_from_sensitivity(overlay):
    t = read(overlay, "REPRODUCE.md")
    s = SENS["summary"]
    assert f"{s['r2w']['held']} of {s['r2w']['total']} right-to-wrong" in t
    assert f"{s['w2r']['held']} of {s['w2r']['total']} wrong-to-right" in t
    assert "uv run python -m scripts.release.reproduce" in t
    assert "not part of pass or fail" in t
    assert "identical_modulo_history_and_amendment" in t and "scripts.acceptance compare-results" in t
    assert "PARAMETER" not in t and "num_ctx" in t and "num_predict" in t
    assert "{{ .Prompt }}" in t


@needs_phase4
def test_rebuild_md_mentions_the_rules(overlay):
    t = read(overlay, "REBUILD.md")
    for s in ("export.arxiv.org", "3.1 s", "MemAvailable", "16 GB", "rocm-smi", "requirements/train.lock.txt", "requirements/convert.lock.txt", "setup_train_env.sh"):
        assert s in t
    for step in ("build_candidates", "corpus_catalog", "corpus_fetch", "corpus_build", "corpus_select", "corpus_pages", "corpus_regression", "baseline_infer", "mine_train", "build_trainset", "train_lora", "select_checkpoint", "export_merge", "export_lora.sh", "register_model", "acceptance"):
        assert step in t
    order = [t.index(s) for s in ("build_candidates", "corpus_catalog", "corpus_fetch", "corpus_build", "corpus_select", "mine_train", "train_lora", "export_merge", "register_model")]
    assert order == sorted(order)


@needs_phase4
@pytest.mark.parametrize("name", ["README.md", "REPRODUCE.md", "REBUILD.md"])
def test_every_referenced_file_is_kept_in_the_public_tree(overlay, name):
    text = read(overlay, name)
    # The rename record is written, listed in the public paths and tracked together; until then the documents may cite it.
    pending = set() if RA.AMENDMENT2_PATH.is_file() else {RA.AMENDMENT2_REF}
    refs = [r for r in SR.referenced_files(text) if r not in pending]
    assert refs
    assert [r for r in SR.unresolved_references(text) if r not in pending] == []
    for ref in refs:
        assert Path(ref).is_file(), ref


@needs_phase4
def test_no_forbidden_string_anywhere_in_the_overlay(overlay):
    for p in overlay.rglob("*"):
        if p.is_file() and p.name != "SHA256SUMS":
            text = p.read_text(encoding="utf-8")
            for needle in (".planning", "docs/", "/mnt" + "/datas", "/home" + "/ogaude", "Nexa" + "ris", "nexa" + "ris"):
                assert needle not in text, f"{p}: {needle}"
    assert SR.scan_overlay(overlay) == []


def test_scan_overlay_catches_a_planted_string(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.json").write_text('{"p": ".planning/phases"}', encoding="utf-8")
    (tmp_path / "REPRODUCE.md").write_text("see docs/notes", encoding="utf-8")
    hits = SR.scan_overlay(tmp_path)
    assert len(hits) == 2


@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_hold_no_literal_result_numbers_or_blob_hashes(name):
    text = (Path("release_templates") / name).read_text(encoding="utf-8")
    for num in ("0.652", "0.426", "0.226", "59", "646", "672"):
        assert not re.search(rf"(?<![\w.]){re.escape(num)}(?![\w.])", text), num
    assert not re.search(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b", text)
    assert "docs/" not in text and ".planning" not in text and "Nexa" + "ris" not in text


def test_attribution_is_required_unless_allowed(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(SR.StageError, match="ATTRIBUTION"):
        SR.stage(tmp_path / "o1", attribution_dir=empty)
    if all(p.is_file() for p in C.PHASE4_SOURCES):
        info = SR.stage(tmp_path / "o2", attribution_dir=empty, allow_missing_attribution=True)
        assert info["attribution"] is False
        assert not (tmp_path / "o2" / "ATTRIBUTION.md").exists()


@needs_phase4
def test_checksums_and_license_files(overlay):
    assert C.verify_sha256sums(overlay) == []
    assert (overlay / "LICENSE").read_bytes() == Path("release_templates/LICENSE-Apache-2.0.txt").read_bytes()
    assert (overlay / "ATTRIBUTION.md").is_file() and (overlay / "ATTRIBUTION.jsonl").is_file()
    assert not (overlay / "RELEASE_HISTORY.json").exists()


A2_RERUN = {
    "compare": "identical_modulo_amendment", "differing_keys": list(RA.RENAME_KEYS), "score_exit": 0, "cold_cache": True,
    "seconds": 321.5, "published_sha256": "12" * 32, "fresh_sha256": "34" * 32,
}


def test_results_index_without_the_rename_record_is_unchanged():
    assert SR.render_results_index(RESULTS, AMEND) == SR.render_results_index(RESULTS, AMEND, None)
    # the evidence file table lists the record file itself; its rerun prose appears only when the record is given
    assert "The rename made before the first release" not in SR.render_results_index(RESULTS, AMEND)
    assert SR.render_results_index(RESULTS, AMEND, {"rerun_equivalence": A2_RERUN}) != SR.render_results_index(RESULTS, AMEND)


def test_results_index_cites_the_rename_record_when_given():
    text = SR.render_results_index(RESULTS, AMEND, {"rerun_equivalence": A2_RERUN})
    assert text.startswith(SR.render_results_index(RESULTS, AMEND).split("## Inputs of results.json")[0].rstrip("\n"))
    assert f"`{RA.AMENDMENT2_REF}`" in text
    for k in RA.RENAME_KEYS:
        assert f"`{k}`" in text
    assert "12" * 32 in text and "34" * 32 in text and "seconds: 321.5" in text


@needs_phase4
def test_results_block_headings_sit_below_the_readme_results_heading(overlay):
    text = read(overlay, "README.md")
    block = text.split("## Results", 1)[1].split("## Reproduce", 1)[0]
    assert "### Headline: render-identical on the benchmark crops" in block
    assert "### Page-level transfer" in block
    assert "\n## " not in block


def test_public_wording_relabels_only_the_table_headers():
    assert SR.public_wording("| | stock | tuned | delta |") == "| | base model | this model | delta |"
    assert SR.public_wording("| render-identical | 42.6% | 65.2% |") == "| render-identical | 42.6% | 65.2% |"
    assert "stock" not in SR.results_markdown(RESULTS).split("**Limits")[0].replace("Rule:", "")
