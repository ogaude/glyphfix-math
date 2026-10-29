"""Release constants are cross-checked against their sources of truth; helpers behave fail-closed."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from glyphfix_math import render
from glyphfix_math.corpus import catalog
from scripts.release import common

ROOT = Path(__file__).resolve().parent.parent


def _entry() -> dict:
    reg = json.loads((ROOT / "data/models/REGISTRY.json").read_text())
    return next(e for e in reg["entries"] if e["tag"] == common.REGISTRY_TAG)


def test_digests_match_sources_of_truth():
    e = _entry()
    assert common.ADAPTER_SHA256 == e["adapter_sha256"]
    assert common.GGUF_Q8_SHA256 == e["gguf_text_sha256"]
    assert common.PROJECTOR_SHA256 == e["projector_sha256"]
    assert common.OLLAMA_DIGEST == e["ollama_digest"]
    assert common.BASE_MODEL_ID == e["base_model"]["id"]
    assert common.BASE_REVISION == e["base_model"]["revision"]
    assert common.LLAMA_CPP_COMMIT == e["llama_cpp_commit"]
    assert common.TRAINSET_SHA256 == e["trainset_sha256"]
    man = json.loads((ROOT / "data/train/sets/full/TRAINSET_MANIFEST.json").read_text())
    assert common.TRAINSET_SHA256 == man["trainset_sha256"]
    assert common.EXPECTED["train_rows"] == man["n_rows"]
    assert common.EXPECTED["train_papers"] == man["n_papers"]
    assert common.EXPECTED["train_licenses"] == man["licenses"]
    assert common.TEXLIVE_IMAGE == render.IMAGE
    freeze = json.loads((ROOT / "data/benchmark/BENCHMARK_FREEZE.json").read_text())
    assert common.TEXLIVE_IMAGE == freeze["render_config"]["image"]
    assert common.EXPECTED["bench_papers"] == freeze["n_papers"]
    assert common.EXPECTED["bench_rows"] == freeze["n_records"]
    assert len(common.PLACEHOLDERS) == 7


def test_render_template():
    assert common.render_template("a {{x}} b {{ .Prompt }} @@T@@", {"x": "1"}) == "a 1 b {{ .Prompt }} @@T@@"
    with pytest.raises(KeyError):
        common.render_template("{{y}}", {"x": "1"})
    with pytest.raises(ValueError):
        common.render_template("{{x}}", {"x": "{{z}}"})


def test_sha256sums_roundtrip(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "d").mkdir()
    (tmp_path / "d/b.txt").write_bytes(b"world")
    assert common.sha256_stream(tmp_path / "a.bin") == hashlib.sha256(b"hello").hexdigest()
    common.write_sha256sums(tmp_path)
    line = (tmp_path / "SHA256SUMS").read_text().splitlines()[0]
    assert line == f"{hashlib.sha256(b'hello').hexdigest()}  a.bin"
    assert common.verify_sha256sums(tmp_path) == []
    (tmp_path / "a.bin").write_bytes(b"hellO")
    assert common.verify_sha256sums(tmp_path) == ["changed: a.bin"]
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "extra").write_bytes(b"x")
    assert common.verify_sha256sums(tmp_path) == ["unlisted: extra"]
    (tmp_path / "extra").unlink()
    (tmp_path / "d/b.txt").unlink()
    assert common.verify_sha256sums(tmp_path) == ["missing: d/b.txt"]


def test_forbidden_patterns():
    samples = {
        "personal_email": "olivier.gaude" + "@gmail.com",
        "work_email": "olivier.gaude" + "@karbo" + "tium.com",
        "local_path_mnt": "/mnt" + "/datas/x",
        "local_path_home": "/home" + "/ogaude/x",
        "hf_token": "hf_" + "a" * 34,
        "github_token": "ghp_" + "b" * 36,
        "private_key": "-----BEGIN OPENSSH" + " PRIVATE KEY-----",
        "origin_project": "Nexa" + "ris",
        "comparison_dataset_a": "im2" + "latex",
        "comparison_dataset_b": "LaTeX" + "_OCR",
        "unsloth_studio": "Unsloth" + " Studio",
    }
    assert {n for n, _ in common.FORBIDDEN_PATTERNS} == set(samples)
    for name, rx in common.FORBIDDEN_PATTERNS:
        assert rx.search(samples[name]), name
        if name != "work_email":  # the source UA holds the work address; only the scrubbed public history may not
            assert not rx.search(catalog.UA), name
        assert not rx.search("lightonai/LightOnOCR-3-4B"), name
    files = {
        "env": ".env",
        "key": "a/b.key",
        "pem": "c.pem",
        "bf16_gguf": "text-bf16.gguf",
        "private_data": "data/probes/x.json",
    }
    assert {n for n, _ in common.FORBIDDEN_FILE_PATTERNS} == set(files)
    for name, rx in common.FORBIDDEN_FILE_PATTERNS:
        assert rx.search(files[name]), name
    assert common.FORBIDDEN_FILE_PATTERNS[-1][1].search("data/spikes/x")
    assert common.FORBIDDEN_FILE_PATTERNS[-1][1].search("data/external/x")
    for ok in (
        "scripts/spikes/common.py",
        "scripts/spikes/export_gguf.sh",
        "tests/test_spikes_common.py",
        ".planning/phases/02-feasibility-spikes/evidence/llama_cpp.json",
        "data/train-00000-of-00001.parquet",
    ):
        assert not any(rx.search(ok) for _, rx in common.FORBIDDEN_FILE_PATTERNS), ok


def test_caveats_from_real_results():
    results = json.loads((ROOT / "evidence/acceptance/results.json").read_text())
    out = common.caveats(results)
    text = "\n".join(out)
    weak = [c["class"] for c in results["per_class"] if c["claim"] == "not enough events to claim"]
    assert weak
    for cls in weak:
        assert cls in text
    for c in results["per_class"]:
        if c["claim"] != "not enough events to claim":
            assert c["class"] not in text
    assert f"{results['transfer']['ci_low']:.4f}" in text
    assert f"{results['dev_reference']['delta']:.4f}" in text
    allowed = {
        f"{results['transfer']['ci_low']:.4f}",
        f"{results['transfer']['ci_high']:.4f}",
        f"{results['dev_reference']['delta']:.4f}",
        str(results["page_match"]["tuned"]["n_matched"]),
        str(results["page_match"]["tuned"]["n_gt"]),
        str(results["page_match"]["stock"]["n_matched"]),
        "95",
        "1.2.0",
    }
    scrubbed = text
    for cls in weak:
        scrubbed = scrubbed.replace(cls, "")
    for num in re.findall(r"-?\d+(?:\.\d+)*", scrubbed):
        assert num in allowed, num


def test_public_evidence_and_sources():
    tracked = subprocess.run(
        ["git", "ls-files", "evidence"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert set(common.PUBLIC_EVIDENCE_FILES) == set(tracked)
    amend = json.loads((ROOT / "evidence/acceptance/FREEZE_AMENDMENT.json").read_text())
    moved = {m["new"] for m in amend["moved"]}
    assert len(moved) == 5
    assert set(common.PUBLIC_EVIDENCE_FILES) == moved | {
        "evidence/acceptance/FREEZE_AMENDMENT.json",
        "evidence/acceptance/FREEZE_AMENDMENT_2.json",
    }
    assert "evidence/acceptance/flips.jsonl" in common.PUBLIC_EVIDENCE_FILES
    assert not any(p.startswith((".planning/", "docs/")) for p in common.PUBLIC_EVIDENCE_FILES)
    assert not hasattr(common, "PUBLIC_PLANNING_FILES")
    assert not hasattr(common, "RERUN_CHECK_SRC")
    assert common.FLIPS_JSONL.as_posix() == "evidence/acceptance/flips.jsonl"
    for src in common.PHASE4_SOURCES:
        p = ROOT / src
        if not p.exists():
            pytest.skip(f"{src} not present")
        body = p.read_text(encoding="utf-8")
        assert ".planning" not in body, src
        assert "docs/" not in body, src
    assert common.CONTACT_FORBIDDEN
    assert common.CONTACT_FORBIDDEN == frozenset(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", catalog.UA))


def test_strip_internal_labels():
    f = common.strip_internal_labels
    assert f("a bootstrap (D-02).") == "a bootstrap."
    assert f("## Page-level transfer (D-11..D-13)") == "## Page-level transfer"
    assert f("## Where (D-16, D-22)") == "## Where"
    assert f("no label (here)") == "no label (here)"
    assert not common.INTERNAL_LABEL.search(f("x (D-02) y (D-11..D-13)"))
