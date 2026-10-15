import subprocess
from pathlib import Path

from scripts import export_merge as em

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_lora.sh"
P = "base_model.model.model.language_model.layers.0."


def _keys(*mods):
    out = []
    for m in mods:
        out += [f"{P}{m}.lora_A.weight", f"{P}{m}.lora_B.weight"]
    return out


def test_expected_names_attention_and_mlp():
    names = em.expected_changed_names(_keys("self_attn.q_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in names
    assert "model.language_model.layers.0.mlp.down_proj.weight" in names
    assert len(names) == 4


def test_expected_names_deltanet_and_dedup():
    keys = _keys("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
    keys += keys  # duplicates collapse
    names = em.expected_changed_names(keys)
    assert len(names) == 3
    assert all(n.endswith(".weight") and ".lora_" not in n for n in names)


def test_expected_names_ignores_non_lora_keys():
    assert em.expected_changed_names(["something.else.weight"]) == []


def _cmp(**kw):
    base = {"n_value_mismatch": 200, "n_names_only_in_base": 0, "n_names_only_in_out": 0}
    base.update(kw)
    return base


def test_merge_verdict_pass():
    assert em.merge_verdict(_cmp()) is True


def test_merge_verdict_fails_on_count_or_names():
    assert em.merge_verdict(_cmp(n_value_mismatch=199)) is False
    assert em.merge_verdict(_cmp(n_value_mismatch=201)) is False
    assert em.merge_verdict(_cmp(n_names_only_in_base=1)) is False
    assert em.merge_verdict(_cmp(n_names_only_in_out=2)) is False


def test_script_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_script_missing_merged_dir_exits_2(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "run1", str(tmp_path / "nope")], capture_output=True, text=True, check=False)
    assert r.returncode == 2


def test_script_bad_run_id_exits_2(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "bad;id", str(tmp_path)], capture_output=True, text=True, check=False)
    assert r.returncode == 2
