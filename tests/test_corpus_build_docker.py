"""End-to-end builder run in the real render sandbox on a synthetic paper."""

import hashlib
import io
import tarfile

import pytest
from PIL import Image

from glyphfix_math import splits
from glyphfix_math.corpus import build

pytestmark = pytest.mark.docker

MAIN = r"""\documentclass{article}
\usepackage{amsmath}
\newcommand{\R}{\mathbb{R}}
\begin{document}
\begin{equation}\label{e1} f'(x) = \alpha + \beta \end{equation}
\begin{align} a &= b + 1 \\ c &= d - 2 \end{align}
\begin{equation} x \in \R \end{equation}
\begin{equation} f'(x) = \alpha + \beta \end{equation}
\begin{equation} \undefinedcs{x} + y \end{equation}
\begin{gather} p + q = r \\ s + t = u \end{gather}
\begin{multline} a + b + c \\ + d + e \end{multline}
\end{document}
"""


def _train_id():
    for n in range(1, 5000):
        i = f"2301.{n:05d}"
        if splits.split_of(i) == "train":
            return i
    raise AssertionError


def test_synthetic_paper_end_to_end(tmp_path):
    aid = _train_id()
    src = tmp_path / "src.bin"
    with tarfile.open(src, "w:gz") as tf:
        data = MAIN.encode()
        ti = tarfile.TarInfo("main.tex")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    paper = build.PaperInput(
        aid, "v1", {"arxiv_id": aid, "version": "v1", "license_uri": "u", "license_spdx": "CC-BY-4.0", "split": "train"},
        {"status": "ok", "kind": "tar.gz", "sha256": sha}, src, tmp_path / "work",
    )
    roots = {"corpus": tmp_path / "corpus", "benchmark": tmp_path / "bench"}
    (res,) = build.build_chunk([paper], cache=tmp_path / "cache", out_roots=roots)
    by_reason = {}
    for r in res.excluded:
        by_reason.setdefault(r["reason"], []).append(r)
    assert len(by_reason["macros"]) == 1
    assert len(by_reason["duplicate"]) == 1
    assert len(by_reason["render"]) == 1
    assert res.summary["other_display_counts"]["multline"] == 1
    assert res.summary["n_extracted"] == 6
    kept_latex = {r["latex"] for r in res.records}
    assert "f'(x) = \\alpha + \\beta" in kept_latex
    assert any("c &= d" in x for x in kept_latex)
    gather_total = len([r for r in res.records if r["env"] == "gather"]) + len(
        [r for r in by_reason.get("canon_mismatch", []) if r["env"] == "gather"])
    assert gather_total == 1
    assert len(res.records) >= 2
    sid = aid
    for rec in res.records:
        img = Image.open(roots["corpus"] / "images" / sid / f"{rec['id']}.png")
        assert img.mode == "L" and img.size == (rec["width"], rec["height"])
    build.write_shards(res, roots)
    summ = roots["corpus"] / "shards" / "papers" / f"{sid}.json"
    assert build.shard_up_to_date(summ, sha)
    assert not (roots["benchmark"] / "shards").exists()
