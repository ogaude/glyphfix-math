"""Model registry: entry schema, validation and hash rebuild.

Usage:
  from lightonocr_math.train import registry
  problems = registry.validate(entry) + registry.verify_hashes(entry, root)
  registry.append_entry(Path("data/models/REGISTRY.json"), entry)

The registry file is {"schema_version": "1", "entries": [...]}. An entry links the adapter, the training
set, the GGUF, the Ollama digest, the configuration and the dev and regression scores. validate checks the
schema and the invariants (internal tag names, scorer fingerprint, all gates passed, benchmark pending);
verify_hashes recomputes adapter, training set and GGUF sha256 from the paths in the entry. append_entry
refuses an invalid entry or a duplicate tag and writes atomically. No torch import.
"""

from __future__ import annotations

from pathlib import Path

from lightonocr_math import score
from lightonocr_math.corpus._io import sha256_file, write_json

REGISTRY_SCHEMA_VERSION = "1"
BENCHMARK_PENDING = {"status": "pending-phase-5"}
ADAPTER_FILE = "adapter_model.safetensors"

REQUIRED_KEYS = (
    "tag",
    "ollama_digest",
    "candidate_tag",
    "gguf_path",
    "gguf_text_sha256",
    "projector_sha256",
    "adapter_path",
    "adapter_sha256",
    "adapter_config",
    "base_model",
    "lock_sha256",
    "versions",
    "llama_cpp_commit",
    "trainset_path",
    "trainset_sha256",
    "trainset_manifest_sha256",
    "taxonomy_sha256",
    "mining_manifest_sha256",
    "run_id",
    "run_manifest_sha256",
    "run_config",
    "sweep",
    "selection",
    "scorer_version",
    "scorer_fingerprint",
    "scores",
    "gates",
    "benchmark",
    "git_commit",
    "created",
)

# Required sub-keys of the nested blocks.
NESTED_KEYS = {
    "adapter_config": ("r", "alpha", "use_rslora", "target_regex", "n_wrapped", "trainable"),
    "base_model": ("id", "revision"),
    "versions": ("torch", "transformers", "peft"),
    "sweep": ("path", "sha256", "winner"),
    "selection": ("chosen_step", "rule", "tie_delta"),
    "scores": ("dev_mini_eval_hf", "dev_full_ollama", "regression"),
    "gates": ("thresholds", "results", "pass"),
}


def validate(entry: dict) -> list[str]:
    """Problems found in an entry; an empty list means the entry is complete and consistent."""
    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in entry:
            problems.append(f"missing key: {key}")
    for block, subkeys in NESTED_KEYS.items():
        sub = entry.get(block)
        if block in entry and not isinstance(sub, dict):
            problems.append(f"{block} must be an object")
            continue
        for sk in subkeys if isinstance(sub, dict) else ():
            if sk not in sub:
                problems.append(f"missing key: {block}.{sk}")
    for key in ("tag", "candidate_tag"):
        if "lighton" in str(entry.get(key, "")).lower():
            problems.append(f"{key} must not contain 'lighton': {entry.get(key)!r}")
    if "scorer_fingerprint" in entry and entry["scorer_fingerprint"] != score.SCORER_FINGERPRINT:
        problems.append("scorer_fingerprint differs from the current scorer")
    gates = entry.get("gates")
    if isinstance(gates, dict) and "pass" in gates and gates["pass"] is not True:
        problems.append("gates.pass is not True")
    if "benchmark" in entry and entry["benchmark"] != BENCHMARK_PENDING:
        problems.append(f"benchmark must be {BENCHMARK_PENDING}")
    return problems


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(root) / path


def verify_hashes(entry: dict, root: Path) -> list[str]:
    """Recompute adapter, training set and GGUF sha256 from disk; list each mismatch or missing file."""
    problems: list[str] = []
    for path_key, hash_key in (
        ("adapter_path", "adapter_sha256"),
        ("trainset_path", "trainset_sha256"),
        ("gguf_path", "gguf_text_sha256"),
    ):
        raw = entry.get(path_key)
        if not raw:
            problems.append(f"{path_key} missing")
            continue
        path = _resolve(root, raw)
        if path.is_dir():
            path = path / ADAPTER_FILE
        if not path.is_file():
            problems.append(f"{path_key}: file not found: {path}")
            continue
        actual = sha256_file(path)
        if actual != entry.get(hash_key):
            problems.append(f"{hash_key} mismatch: entry {entry.get(hash_key)} != disk {actual}")
    return problems


def load(path: Path) -> dict:
    import json

    p = Path(path)
    if not p.is_file():
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": []}
    return json.loads(p.read_text(encoding="utf-8"))


def append_entry(path: Path, entry: dict) -> None:
    """Append a validated entry atomically; refuses an invalid entry or a tag already registered."""
    problems = validate(entry)
    if problems:
        raise ValueError("invalid registry entry: " + "; ".join(problems))
    reg = load(path)
    if any(e.get("tag") == entry["tag"] for e in reg["entries"]):
        raise ValueError(f"tag already registered: {entry['tag']}")
    reg["entries"].append(entry)
    reg["schema_version"] = REGISTRY_SCHEMA_VERSION
    write_json(path, reg)
