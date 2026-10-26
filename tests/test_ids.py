import pytest

from glyphfix_math.ids import base_id, from_safe_id, is_arxiv_id, safe_id
from glyphfix_math.paths import repo_root


def test_base_id():
    assert base_id("2301.01234v2") == "2301.01234"
    assert base_id(" math/0508005v1 ") == "math/0508005"
    assert base_id("math.GT/0309136") == "math.GT/0309136"


@pytest.mark.parametrize("bad", ["../etc/passwd", "2301.0123", "", "a b", "math/0508005/../x"])
def test_base_id_rejects(bad):
    assert not is_arxiv_id(bad)
    with pytest.raises(ValueError):
        base_id(bad)


def test_safe_id_roundtrip():
    assert safe_id("math/0508005") == "math_0508005"
    assert from_safe_id("math_0508005") == "math/0508005"
    assert from_safe_id("2301.01234") == "2301.01234"


def test_repo_root(tmp_path, monkeypatch):
    assert (repo_root() / "pyproject.toml").is_file()
    monkeypatch.setenv("GLYPHFIX_MATH_ROOT", str(tmp_path))
    assert repo_root() == tmp_path
