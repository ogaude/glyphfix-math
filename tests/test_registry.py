import copy
import json

import pytest

from glyphfix_math import score
from glyphfix_math.corpus._io import sha256_file
from glyphfix_math.train import registry as r


@pytest.fixture
def entry(tmp_path):
    ad = tmp_path / "adapter"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(b"adapter")
    ts = tmp_path / "trainset.jsonl"
    ts.write_text("{}\n")
    gg = tmp_path / "text-q8_0.gguf"
    gg.write_bytes(b"gguf")
    e = {k: "x" for k in r.REQUIRED_KEYS}
    e.update(
        tag="glyphfix-math-lora-q8:run1",
        candidate_tag="glyphfix-math-cand:run1",
        adapter_path=str(ad),
        adapter_sha256=sha256_file(ad / "adapter_model.safetensors"),
        trainset_path=str(ts),
        trainset_sha256=sha256_file(ts),
        gguf_path=str(gg),
        gguf_text_sha256=sha256_file(gg),
        scorer_fingerprint=score.SCORER_FINGERPRINT,
        benchmark={"status": "pending-phase-5"},
        adapter_config={k: 1 for k in r.NESTED_KEYS["adapter_config"]},
        base_model={"id": "m", "revision": "rev"},
        versions={"torch": "1", "transformers": "5", "peft": "0"},
        sweep={"path": "s", "sha256": "h", "winner": "A"},
        selection={"chosen_step": 100, "rule": "r", "tie_delta": 0.02},
        scores={"dev_mini_eval_hf": {}, "dev_full_ollama": {}, "regression": {}},
        gates={"thresholds": {}, "results": {}, "pass": True},
    )
    return e


def test_required_keys_count():
    assert len(r.REQUIRED_KEYS) >= 30
    assert r.REGISTRY_SCHEMA_VERSION == "1"


def test_validate_complete_entry(entry):
    assert r.validate(entry) == []


def test_validate_names_missing_keys(entry):
    del entry["ollama_digest"]
    del entry["selection"]["rule"]
    p = r.validate(entry)
    assert "missing key: ollama_digest" in p
    assert "missing key: selection.rule" in p


def test_validate_flags_lighton_tag(entry):
    entry["tag"] = "lighton-lora:run1"
    assert any("lighton" in p for p in r.validate(entry))


def test_validate_flags_scorer_fingerprint(entry):
    entry["scorer_fingerprint"] = "deadbeef"
    assert any("scorer_fingerprint" in p for p in r.validate(entry))


def test_validate_flags_failed_gates_and_benchmark(entry):
    entry["gates"]["pass"] = False
    entry["benchmark"] = {"status": "done"}
    p = r.validate(entry)
    assert "gates.pass is not True" in p
    assert any(x.startswith("benchmark must be") for x in p)


def test_verify_hashes_clean(entry, tmp_path):
    assert r.verify_hashes(entry, tmp_path) == []


def test_verify_hashes_reports_mismatch_and_missing(entry, tmp_path):
    entry["adapter_sha256"] = "0" * 64
    (tmp_path / "trainset.jsonl").unlink()
    p = r.verify_hashes(entry, tmp_path)
    assert any("adapter_sha256 mismatch" in x for x in p)
    assert any(x.startswith("trainset_path: file not found") for x in p)
    assert not any("gguf" in x for x in p)


def test_append_entry_writes_and_refuses_duplicate(entry, tmp_path):
    reg = tmp_path / "models" / "REGISTRY.json"
    r.append_entry(reg, entry)
    data = json.loads(reg.read_text())
    assert data["schema_version"] == "1"
    assert len(data["entries"]) == 1
    with pytest.raises(ValueError, match="already registered"):
        r.append_entry(reg, copy.deepcopy(entry))
    assert len(json.loads(reg.read_text())["entries"]) == 1


def test_append_entry_refuses_invalid(entry, tmp_path):
    reg = tmp_path / "REGISTRY.json"
    del entry["git_commit"]
    with pytest.raises(ValueError, match="invalid registry entry"):
        r.append_entry(reg, entry)
    assert not reg.exists()


def _scored():
    return {
        "status": "scored",
        "verdict": "pass",
        "transfer_holds": True,
        "headline": {k: 0.5 for k in r.HEADLINE_KEYS},
        "evidence": "e",
        "results_sha256": "a" * 64,
        "tuned_preds_sha256": {"crops": "b" * 64, "pages": "c" * 64},
        "acceptance_sha256": "d" * 64,
        "scored_at": "2026-10-20T00:00:00Z",
    }


def test_scored_block_validates(entry):
    entry["benchmark"] = _scored()
    assert r.validate(entry) == []


@pytest.mark.parametrize("mutate", ["drop_key", "bad_verdict", "bad_headline", "bad_preds", "bad_status"])
def test_invalid_scored_blocks_rejected(entry, mutate):
    b = _scored()
    if mutate == "drop_key":
        del b["results_sha256"]
    elif mutate == "bad_verdict":
        b["verdict"] = "maybe"
    elif mutate == "bad_headline":
        del b["headline"]["ci_low"]
    elif mutate == "bad_preds":
        b["tuned_preds_sha256"] = {"crops": "x"}
    else:
        b["status"] = "done"
    entry["benchmark"] = b
    assert any(x.startswith("benchmark must be") for x in r.validate(entry))


def test_update_benchmark_replaces_pending_once(entry, tmp_path):
    reg = tmp_path / "REGISTRY.json"
    r.append_entry(reg, entry)
    r.update_benchmark(reg, entry["tag"], _scored())
    got = r.load(reg)["entries"][0]
    assert got["benchmark"]["verdict"] == "pass"
    assert r.validate(got) == []
    before = reg.read_bytes()
    with pytest.raises(ValueError, match="benchmark already scored"):
        r.update_benchmark(reg, entry["tag"], _scored())
    assert reg.read_bytes() == before


def test_update_benchmark_refusals_leave_file_unchanged(entry, tmp_path):
    reg = tmp_path / "REGISTRY.json"
    r.append_entry(reg, entry)
    before = reg.read_bytes()
    with pytest.raises(ValueError, match="not in registry"):
        r.update_benchmark(reg, "nope:1", _scored())
    bad = _scored()
    bad["verdict"] = "maybe"
    with pytest.raises(ValueError, match="invalid"):
        r.update_benchmark(reg, entry["tag"], bad)
    assert reg.read_bytes() == before
