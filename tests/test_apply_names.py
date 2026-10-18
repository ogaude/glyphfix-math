import json
import subprocess

import pytest

from scripts.release import apply_names as AN
from scripts.release import common as C

GOOD = {
    "dataset_name": "texglyph-equations",
    "model_name": "texglyph-lora",
    "repo_name": "texglyph",
    "hf_namespace": "example-user",
    "gh_owner": "example-user",
    "contact_email": "a@b.org",
    "copyright_holder": "Example Holder",
}


def names(**kw):
    return {**GOOD, **kw}


@pytest.mark.parametrize(
    "triple",
    [
        ("texglyph-lora", "texglyph-equations", "texglyph"),
        ("glyphmath-ocr-lora", "glyphmath-equations", "glyphmath-ocr"),
        ("eqglyph-lora", "eqglyph-bench", "eqglyph"),
    ],
)
def test_candidates_accepted(triple):
    m, d, r = triple
    AN.validate_names(names(model_name=m, dataset_name=d, repo_name=r))


@pytest.mark.parametrize(
    "kw",
    [
        {"model_name": "lighton-math"},
        {"model_name": "LightOn_ocr"},
        {"repo_name": "light-on-glyph"},
        {"dataset_name": "light on"},
        {"copyright_holder": "Light On Labs"},
        {"model_name": "Upper-case"},
        {"model_name": "has space"},
        {"model_name": "ab"},
        {"dataset_name": ""},
        {"copyright_holder": "x @@TOKEN@@"},
        {"contact_email": "not-an-email"},
        {"gh_owner": "a" * 40},
        {"copyright_holder": "two\nlines"},
    ],
)
def test_invalid_rejected(kw):
    with pytest.raises(AN.NamesError):
        AN.validate_names(names(**kw))


def test_arxiv_contact_rejected():
    assert C.CONTACT_FORBIDDEN
    with pytest.raises(AN.NamesError):
        AN.validate_names(names(contact_email=min(C.CONTACT_FORBIDDEN)))


@pytest.fixture
def tree(tmp_path):
    stg = tmp_path / "release"
    ds, md, ov = stg / "dataset", stg / "model", stg / "code" / "overlay"
    for d in (ds, md, ov):
        d.mkdir(parents=True)
    (ds / "README.md").write_text("# @@DATASET_NAME@@ by @@COPYRIGHT_HOLDER@@ at @@HF_NAMESPACE@@\n")
    (ds / "ATTRIBUTION.jsonl").write_text('{"c": "@@CONTACT_EMAIL@@"}\n')
    (ds / "data").mkdir()
    pq = b"PAR1\x00\x01@@DATASET_NAME@@\x00PAR1"
    (ds / "data" / "t.parquet").write_bytes(pq)
    (md / "README.md").write_text("@@MODEL_NAME@@ @@MODEL_NAME@@\n")
    (md / "Modelfile").write_text("FROM ./x\n# @@MODEL_NAME@@\n")
    (md / "NOTICE").write_text("@@COPYRIGHT_HOLDER@@\n")
    (md / "text.gguf").write_bytes(b"GGUF\x00@@MODEL_NAME@@")
    (ov / "README.md").write_text("https://github.com/@@GH_OWNER@@/@@REPO_NAME@@\n")
    pub = stg / "PUBLISH.md"
    pub.write_text("hf repos create @@HF_NAMESPACE@@/@@DATASET_NAME@@\n")
    for d in (ds, md, ov):
        C.write_sha256sums(d)
    return stg


def run(stg, nm=None):
    return AN.apply_names(
        nm or GOOD,
        roots=(stg / "dataset", stg / "model", stg / "code" / "overlay"),
        extra_files=(stg / "PUBLISH.md",),
        staging=stg,
    )


def test_apply(tree):
    pq = (tree / "dataset" / "data" / "t.parquet").read_bytes()
    gg = (tree / "model" / "text.gguf").read_bytes()
    res = run(tree)
    assert not res["noop"]
    for p in tree.rglob("*"):
        if p.is_file() and p.suffix not in (".parquet", ".gguf"):
            assert "@@" not in p.read_text(), p
    assert (tree / "dataset" / "data" / "t.parquet").read_bytes() == pq
    assert (tree / "model" / "text.gguf").read_bytes() == gg
    assert "texglyph-equations by Example Holder" in (tree / "dataset" / "README.md").read_text()
    for sub in ("dataset", "model", "code/overlay"):
        assert C.verify_sha256sums(tree / sub) == []
    rec = json.loads((tree / "NAMES.json").read_text())
    for k, v in GOOD.items():
        assert rec[k] == v


def test_twice_is_noop(tree):
    run(tree)
    before = {p: p.read_bytes() for p in tree.rglob("*") if p.is_file() and p.name != "NAMES.json"}
    res = run(tree)
    assert res["noop"]
    assert before == {p: p.read_bytes() for p in tree.rglob("*") if p.is_file() and p.name != "NAMES.json"}


def test_different_names_refused(tree):
    run(tree)
    with pytest.raises(AN.NamesError, match="re-run the stage scripts"):
        run(tree, names(model_name="eqglyph-lora", dataset_name="eqglyph-bench", repo_name="eqglyph"))


def test_different_names_allowed_after_restage(tree):
    run(tree)
    (tree / "dataset" / "README.md").write_text("# @@DATASET_NAME@@\n")
    (tree / "model" / "README.md").write_text("@@MODEL_NAME@@\n")
    (tree / "model" / "Modelfile").write_text("FROM ./x\n")
    (tree / "model" / "NOTICE").write_text("n\n")
    (tree / "dataset" / "ATTRIBUTION.jsonl").write_text("{}\n")
    (tree / "code" / "overlay" / "README.md").write_text("@@REPO_NAME@@ @@GH_OWNER@@\n")
    (tree / "PUBLISH.md").write_text("@@HF_NAMESPACE@@ @@CONTACT_EMAIL@@\n")
    run(tree, names(model_name="eqglyph-lora", dataset_name="eqglyph-bench", repo_name="eqglyph"))
    assert json.loads((tree / "NAMES.json").read_text())["repo_name"] == "eqglyph"


def test_outside_release_refused(tree, tmp_path):
    other = tmp_path / "release_templates"
    other.mkdir()
    (other / "x.md").write_text("@@DATASET_NAME@@\n")
    with pytest.raises(AN.NamesError, match="outside"):
        AN.apply_names(GOOD, roots=(other,), extra_files=(), staging=tree)
    assert (other / "x.md").read_text() == "@@DATASET_NAME@@\n"
    with pytest.raises(AN.NamesError, match="outside"):
        AN.apply_names(GOOD, roots=(tree / ".." / "release_templates",), extra_files=(), staging=tree)


def test_unknown_token_fails(tree):
    (tree / "dataset" / "README.md").write_text("@@NOPE@@\n")
    with pytest.raises(AN.NamesError, match="unknown"):
        run(tree)


def test_source_tree_unchanged_outside_release():
    before = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    # a refused run on the real layout must not write anything
    with pytest.raises(AN.NamesError):
        AN.apply_names(names(model_name="lighton-math"))
    after = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    assert before == after
