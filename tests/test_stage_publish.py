import re
import subprocess
from pathlib import Path

import pytest

from scripts.release import common as C
from scripts.release import stage_publish as SP

TOKEN_RE = re.compile(r"@@[A-Z_]+@@")
SLOT_RE = re.compile(r"\{\{[a-z_][a-z0-9_]*\}\}")


def fake_values():
    return {
        s.strip("{}"): "x"
        for s in set(SLOT_RE.findall(SP.TEMPLATE.read_text(encoding="utf-8")))
    }


@pytest.fixture
def rendered(tmp_path):
    out = SP.stage_publish(tmp_path / "PUBLISH.md", values=fake_values())
    return out.read_text(encoding="utf-8")


def test_render_has_no_slot_and_all_tokens(rendered):
    assert not SLOT_RE.search(rendered)
    for tok in C.PLACEHOLDERS.values():
        if tok in ("@@CONTACT_EMAIL@@", "@@COPYRIGHT_HOLDER@@"):
            continue  # not needed by the runbook
        assert tok in rendered, tok


def test_required_content(rendered):
    for needle in (
        "hf repos create",
        "--private",
        "release_check all --repo release/code/public.git --final",
        "gh repo create",
        "Do not run any step without the user's go-ahead",
        "identical_modulo_history_and_amendment",
        "freeze_amendment verify",
    ):
        assert needle in rendered, needle
    assert not re.search(r"hf_[A-Za-z0-9]+", rendered)
    for line in rendered.splitlines():
        if re.match(r"(hf repos create|gh repo create) ", line):
            assert "--private" in line, line


def _help(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60).stdout


@pytest.mark.skipif(not (SP.find_cli("hf") and SP.find_cli("gh")), reason="hf or gh CLI absent")
def test_commands_exist_in_local_help(rendered):
    seen = 0
    for line in rendered.splitlines():
        toks = line.split()
        if not toks or toks[0] not in ("hf", "gh") or len(toks) < 3:
            continue
        seen += 1
        if toks[0] == "hf":
            first = subprocess.run([SP.find_cli("hf"), toks[1], "--help"], capture_output=True, text=True, check=False, timeout=60)
            assert first.returncode == 0, line
            if toks[1] in ("repos", "auth"):
                assert re.search(rf"^\s+{re.escape(toks[2])}\b", first.stdout, re.MULTILINE), line
        else:
            sub = _help(SP.find_cli("gh"), toks[1], "--help")
            assert re.search(rf"\b{re.escape(toks[2])}\b", sub), line
    assert seen >= 8


def test_no_script_publishes():
    gate = re.compile(r"upload_folder|create_repo|\"push\"|'push'|hf upload|gh repo create|git push")
    for p in Path("scripts/release").glob("*.py"):
        assert not gate.search(p.read_text(encoding="utf-8")), p


def test_missing_dataset_fails_closed(tmp_path):
    with pytest.raises(SP.StageError, match="stage_dataset"):
        SP.build_values(dataset_dir=tmp_path, model_dir=tmp_path)
