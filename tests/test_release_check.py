"""Staging-surface release check on a synthetic package: one rejection test per rule."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from glyphfix_math import splits
from scripts.release import common as C
from scripts.release import release_check as rc
from scripts.release import rename_amendment

ROOT = Path(__file__).resolve().parent.parent
BENCH_ID = "2001.00099"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _train_ids() -> list[str]:
    fs = splits.FrozenSplit(salt="t", rule_version="1", benchmark=frozenset(), quarantine=frozenset(), digest="x", rule={})
    out, n = [], 1
    while len(out) < 2:
        cand = f"2002.{n:05d}"
        if splits.split_of_with_salt(cand, fs.salt) != "benchmark":
            out.append(cand)
        n += 1
    return out


class Tree:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.root = tmp / "release"
        self.t1, self.t2 = _train_ids()
        self.fs = splits.FrozenSplit(
            salt="t", rule_version="1", benchmark=frozenset({BENCH_ID}), quarantine=frozenset(), digest="x", rule={}
        )
        self.train = [self._row("tr1", self.t1, "v2", "CC-BY-4.0", 1), self._row("tr2", self.t1, "v2", "CC-BY-4.0", 2),
                      self._row("tr3", self.t2, "v1", "CC0-1.0", 3)]
        self.bench = [self._row("be1", BENCH_ID, "v1", "CC-BY-4.0", 4, bench=True), self._row("be2", BENCH_ID, "v1", "CC-BY-4.0", 5, bench=True)]
        self.bench_columns = list(C.BENCH_COLUMNS)
        self.expected = {
            "train_rows": 3, "train_papers": 2, "train_licenses": {"CC-BY-4.0": 2, "CC0-1.0": 1},
            "bench_rows": 2, "bench_papers": 1, "bench_licenses": {"CC-BY-4.0": 2},
        }
        self.model_files = {
            C.GGUF_NAME: b"gguf-bytes", C.MMPROJ_NAME: b"mmproj-bytes", "adapter_model.safetensors": b"adapter-bytes",
        }
        self.adapter_sha = _sha(b"adapter-bytes")
        self.gguf_sha = _sha(b"gguf-bytes")
        self.proj_sha = _sha(b"mmproj-bytes")
        self.contact_forbidden = frozenset({"arxiv@example.org"})
        self.build()

    def _row(self, rid, paper, version, spdx, n, bench=False):
        data = f"png-{rid}-{n}".encode()
        r = {
            "id": rid, "image": {"bytes": data, "path": f"{rid}.png"}, "latex": f"x_{n}", "arxiv_id": paper, "paper_version": version,
            "title": f"Title {paper}", "authors": ["A One", "B Two"], "license_spdx": spdx, "license_uri": C.LICENSE_URI[spdx],
            "arxiv_url": f"https://arxiv.org/abs/{paper}{version}", "flags": ["prime"], "image_sha256": _sha(data),
            "bitmap_sha256": _sha(f"bm{n}".encode()), "width": 10, "height": 5,
        }
        if bench:
            r.update(render_image_digest="sha256:d", preamble_sha256="p", canon_version="2", builder_version="1", env="equation")
        else:
            r.update(target_text=f"```latex\nx_{n}\n```", group="replay", mining_class=None, control_glyph=None, tags=[])
        return r

    # ---- writers
    def _write_parquet(self, path: Path, rows: list[dict], columns: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        types = {
            "image": pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            "authors": pa.list_(pa.string()), "flags": pa.list_(pa.string()), "tags": pa.list_(pa.string()),
            "width": pa.int32(), "height": pa.int32(),
        }
        arrays = [pa.array([r.get(c) for r in rows], type=types.get(c, pa.string())) for c in columns]
        pq.write_table(pa.table(arrays, names=columns), path)

    def write(self, rel: str, content: str | bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def build(self) -> None:
        ds, md, ov = self.root / "dataset", self.root / "model", self.root / "code" / "overlay"
        self.write_data()
        self.write_dataset_text()
        for name, data in self.model_files.items():
            self.write(f"model/{name}", data)
        self.write("model/adapter_config.json", "{}")
        self.write(
            "model/README.md",
            f"---\nlicense: apache-2.0\nbase_model: {C.BASE_MODEL_ID}\nbase_model_relation: adapter\n---\n# Model\n"
            f"Base revision {C.BASE_REVISION}. {C.DISCLAIMER}\n",
        )
        self.write("model/NOTICE", f"@@MODEL_NAME@@\nBuilt on LightOn {C.BASE_REVISION}\n")
        self.write("model/LICENSE", "Apache License\n Version 2.0, January 2004\n")
        self.write("model/Modelfile", f"FROM ./{C.GGUF_NAME}\nFROM ./{C.MMPROJ_NAME}\nPARAMETER num_ctx 1024\n")
        self.write("model/PROVENANCE.json", "{}")
        self.write_overlay()
        self.write_sources()
        for r in (ds, md, ov):
            C.write_sha256sums(r)
        self.write("NAMES.json", json.dumps({
            "dataset_name": "math-crops", "model_name": "math-lora", "repo_name": "math-ocr", "hf_namespace": "someone",
            "gh_owner": "someone", "contact_email": "me@example.com", "copyright_holder": "Someone",
        }))

    def write_data(self) -> None:
        ds = self.root / "dataset"
        self._write_parquet(ds / C.TRAIN_PARQUET, self.train, list(C.TRAIN_COLUMNS))
        self._write_parquet(ds / C.BENCH_PARQUET, self.bench, self.bench_columns)

    def write_dataset_text(self) -> None:
        papers: dict[str, dict] = {}
        for split, rows in (("train", self.train), ("benchmark", self.bench)):
            for r in rows:
                e = papers.setdefault(r["arxiv_id"], {
                    "arxiv_id": r["arxiv_id"], "version": r["paper_version"], "title": r["title"], "authors": r["authors"],
                    "license_spdx": r["license_spdx"], "license_uri": r["license_uri"], "url": r["arxiv_url"], "splits": [split],
                    "n_rows": 0,
                })
                e["n_rows"] += 1
        self.attr = list(papers.values())
        self.write("dataset/ATTRIBUTION.jsonl", "".join(json.dumps(e) + "\n" for e in self.attr))
        self.write("dataset/ATTRIBUTION.md", "# Attribution\n" + "".join(f"| [{e['arxiv_id']}]({e['url']}) |\n" for e in self.attr))
        self.write("dataset/README.md", "---\nlicense: cc-by-4.0\n---\n# Data\nDo not train on it.\n")
        self.write("dataset/DATASET_MANIFEST.json", json.dumps({"parquet_sha256": {
            C.TRAIN_PARQUET: C.sha256_stream(self.root / "dataset" / C.TRAIN_PARQUET),
            C.BENCH_PARQUET: C.sha256_stream(self.root / "dataset" / C.BENCH_PARQUET),
        }}))
        ov = self.root / "code" / "overlay"
        ov.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root / "dataset" / "ATTRIBUTION.jsonl", ov / "ATTRIBUTION.jsonl")

    def write_sources(self) -> None:
        s = self.tmp / "src"
        s.mkdir(exist_ok=True)
        trainset = "".join(json.dumps({"id": r["id"]}) + "\n" for r in self.train)
        (s / "trainset.jsonl").write_text(trainset)
        (s / "records.jsonl").write_text("".join(json.dumps({"id": r["id"]}) + "\n" for r in self.bench))
        for name, rows, split in (("train_cat.jsonl", self.train, "train"), ("bench_cat.jsonl", self.bench, "benchmark")):
            seen = {}
            for r in rows:
                seen[r["arxiv_id"]] = {"arxiv_id": r["arxiv_id"], "version": r["paper_version"], "license_spdx": r["license_spdx"]}
            (s / name).write_text("".join(json.dumps(x) + "\n" for x in seen.values()))
        (s / "freeze.json").write_text(json.dumps({
            "render_config": {"image": C.TEXLIVE_IMAGE}, "render_image_digest": C.TEXLIVE_IMAGE.split("@", 1)[1],
        }))
        self.inputs = {"a": {"path": "data/x.pred.jsonl", "sha256": "ab" * 32}}
        (s / "results.json").write_text(json.dumps({"inputs": self.inputs}))
        self.rerun = {"compare": "identical_modulo_amendment", "published_sha256": "cd" * 32, "fresh_sha256": "ef" * 32}
        (s / "amendment.json").write_text(json.dumps({"rerun_equivalence": self.rerun}))
        for n in ("DEV_ANALYSIS.md", "MINING.md", "PARITY_DIAGNOSIS.md", "dev_gain_by_class.json"):
            (s / "p4").mkdir(exist_ok=True)
            (s / "p4" / n).write_text(f"source {n}\n")

    def write_overlay(self) -> None:
        for n in ("README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", "REPRODUCE.md", "REBUILD.md"):
            self.write(f"code/overlay/{n}", f"{n}\n")
        s = self.tmp / "src" / "p4"
        s.mkdir(parents=True, exist_ok=True)
        for n in ("DEV_ANALYSIS.md", "MINING.md", "PARITY_DIAGNOSIS.md", "dev_gain_by_class.json"):
            (s / n).write_text(f"source {n}\n")
            self.write(f"code/overlay/results/{n}", f"source {n}\n")
        self.write(
            "code/overlay/results/README.md",
            "Rerun record: evidence/acceptance/FREEZE_AMENDMENT.json\n"
            f"compare identical_modulo_amendment; published {'cd' * 32}; fresh {'ef' * 32}\n"
            f"- data/x.pred.jsonl {'ab' * 32}\n",
        )

    def resum(self) -> None:
        for r in ("dataset", "model", "code/overlay"):
            if (self.root / r).is_dir():
                C.write_sha256sums(self.root / r)

    def refresh_data(self, resum: bool = True) -> None:
        self.write_data()
        self.write_dataset_text()
        (self.tmp / "dummy").mkdir(exist_ok=True)
        if resum:
            self.resum()

    def config(self, **kw) -> rc.Config:
        s = self.tmp / "src"
        base = {
            "root": self.root, "expected": self.expected, "trainset": s / "trainset.jsonl", "trainset_sha256": C.sha256_stream(s / "trainset.jsonl"),
            "bench_records": s / "records.jsonl", "train_catalog": s / "train_cat.jsonl", "bench_catalog": s / "bench_cat.jsonl",
            "freeze_json": s / "freeze.json", "results_json": s / "results.json", "amendment_json": s / "amendment.json", "amendment2_json": s / "amendment2.json",
            "phase4": tuple(s / "p4" / n for n in ("DEV_ANALYSIS.md", "MINING.md", "PARITY_DIAGNOSIS.md", "dev_gain_by_class.json")),
            "adapter_sha": self.adapter_sha, "gguf_sha": self.gguf_sha, "proj_sha": self.proj_sha, "fs": self.fs,
            "contact_forbidden": self.contact_forbidden,
        }
        base.update(kw)
        return rc.Config(**base)

    def run(self, final: bool = False, **kw) -> dict:
        render = kw.pop("render", False)
        rep = rc.check_staging(final=final, render=render, cfg=self.config(**kw))
        return {c["name"]: c["status"] for c in rep["checks"]} | {"_status": rep["status"], "_report": rep}


@pytest.fixture
def tree(tmp_path) -> Tree:
    return Tree(tmp_path)


def failed(res: dict) -> set[str]:
    return {k for k, v in res.items() if not k.startswith("_") and v == "fail"}


def test_valid_tree_passes(tree):
    res = tree.run()
    assert failed(res) == set(), res["_report"]
    assert res["_status"] == "pass"
    assert res["render_sample"] == "not_run"
    assert res["results_overlay"] == "pass"


# ---- records_license / records_fields / splits
def test_license_not_allowed(tree):
    tree.train[0].update(license_spdx="CC-BY-NC-4.0")
    tree.refresh_data()
    assert "records_license" in failed(tree.run())


def test_license_uri_mismatch(tree):
    tree.train[0]["license_uri"] = C.LICENSE_URI["CC0-1.0"]
    tree.refresh_data()
    assert "records_license" in failed(tree.run())


def test_paper_version_malformed(tree):
    tree.train[2].update(paper_version="3", arxiv_url=f"https://arxiv.org/abs/{tree.t2}3")
    tree.refresh_data()
    assert "records_license" in failed(tree.run())


def test_raw_body_column_in_benchmark(tree):
    tree.bench_columns = [*C.BENCH_COLUMNS, "raw_body"]
    for r in tree.bench:
        r["raw_body"] = "secret"
    tree.refresh_data()
    assert "records_fields" in failed(tree.run())


def test_train_paper_also_benchmark_paper(tree):
    tree.bench.append(tree._row("be3", tree.t1, "v2", "CC-BY-4.0", 6, bench=True))
    tree.refresh_data()
    assert "splits" in failed(tree.run())


def test_benchmark_id_in_train_parquet(tree):
    tree.train[0].update(arxiv_id=BENCH_ID, arxiv_url=f"https://arxiv.org/abs/{BENCH_ID}v2")
    tree.refresh_data()
    res = tree.run()
    assert "leak" in failed(res)
    assert "splits" in failed(res)


def test_counts_differ_from_expected(tree):
    assert "splits" in failed(tree.run(expected={**tree.expected, "train_rows": 4}))


# ---- images
def test_image_bytes_do_not_match_hash(tree):
    tree.train[0]["image"] = {"bytes": b"other", "path": "tr1.png"}
    tree.refresh_data()
    assert failed(tree.run()) == {"images"}


def test_image_path_with_slash(tree):
    tree.train[0]["image"] = {"bytes": tree.train[0]["image"]["bytes"], "path": "data/x/y.png"}
    tree.refresh_data()
    assert failed(tree.run()) == {"images"}


# ---- attribution
def test_attribution_missing_paper(tree):
    p = tree.root / "dataset" / "ATTRIBUTION.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in tree.attr[1:]))
    shutil.copy(p, tree.root / "code/overlay/ATTRIBUTION.jsonl")
    tree.resum()
    assert "attribution" in failed(tree.run())


def test_attribution_extra_paper(tree):
    extra = {**tree.attr[0], "arxiv_id": "2003.00001"}
    p = tree.root / "dataset" / "ATTRIBUTION.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in [*tree.attr, extra]))
    shutil.copy(p, tree.root / "code/overlay/ATTRIBUTION.jsonl")
    tree.resum()
    assert "attribution" in failed(tree.run())


def test_catalog_version_differs(tree):
    p = tree.tmp / "src" / "train_cat.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["version"] = "v9"
    p.write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert "attribution" in failed(tree.run())


def test_overlay_attribution_differs(tree):
    (tree.root / "code/overlay/ATTRIBUTION.jsonl").write_text("{}\n")
    tree.resum()
    assert "attribution" in failed(tree.run())


# ---- allow list and hashes
def test_unlisted_extra_file(tree):
    tree.write("dataset/extra.txt", "x")
    assert failed(tree.run()) == {"staging_allow_list", "hashes"}


def test_changed_byte_in_listed_file(tree):
    (tree.root / "dataset" / "README.md").write_text("---\nlicense: cc-by-4.0\n---\nDo not train on it.!\n")
    assert failed(tree.run()) == {"hashes"}


def test_adapter_digest_mismatch(tree):
    assert failed(tree.run(adapter_sha="0" * 64)) == {"hashes"}


def test_parquet_digest_differs_from_manifest(tree):
    m = tree.root / "dataset" / "DATASET_MANIFEST.json"
    d = json.loads(m.read_text())
    d["parquet_sha256"][C.TRAIN_PARQUET] = "0" * 64
    m.write_text(json.dumps(d))
    tree.resum()
    assert failed(tree.run()) == {"hashes"}


# ---- model card
def test_model_card_without_base_model(tree):
    p = tree.root / "model" / "README.md"
    p.write_text(p.read_text().replace(f"base_model: {C.BASE_MODEL_ID}\n", ""))
    tree.resum()
    assert failed(tree.run()) == {"model_card"}


def test_model_card_without_disclaimer(tree):
    p = tree.root / "model" / "README.md"
    p.write_text(p.read_text().replace(C.DISCLAIMER, ""))
    tree.resum()
    assert failed(tree.run()) == {"model_card"}


def test_modelfile_absolute_from(tree):
    (tree.root / "model" / "Modelfile").write_text(f"FROM /abs/{C.GGUF_NAME}\nFROM ./{C.MMPROJ_NAME}\n")
    tree.resum()
    assert failed(tree.run()) == {"model_card"}


# ---- forbidden content
def test_local_path_in_card(tree):
    p = tree.root / "dataset" / "README.md"
    p.write_text(p.read_text() + "see /home" + "/ogaude/x\n")
    tree.resum()
    assert failed(tree.run()) == {"forbidden_content"}


def test_token_in_json(tree):
    tree.write("model/PROVENANCE.json", json.dumps({"t": "hf_" + "a" * 34}))
    tree.resum()
    assert failed(tree.run()) == {"forbidden_content"}


def test_bf16_gguf_name(tree):
    tree.write("model/text-bf16.gguf", "x")
    tree.resum()
    assert {"forbidden_content", "staging_allow_list"} <= failed(tree.run())


def test_probe_data_under_dataset(tree):
    tree.write("dataset/data/probes/x.json", "{}")
    tree.resum()
    assert {"forbidden_content", "staging_allow_list"} == failed(tree.run())


def test_forbidden_string_in_parquet(tree):
    tree.train[0]["title"] = "from im2" + "latex"
    tree.refresh_data()
    assert "forbidden_content" in failed(tree.run())


# ---- overlay
def test_overlay_results_json_present(tree):
    tree.write("code/overlay/results/results.json", "{}")
    tree.resum()
    assert "results_overlay" in failed(tree.run())


def test_overlay_rerun_check_present(tree):
    tree.write("code/overlay/results/rerun_check.json", "{}")
    tree.resum()
    assert "results_overlay" in failed(tree.run())


def test_overlay_phase4_copy_differs(tree):
    tree.write("code/overlay/results/MINING.md", "edited\n")
    tree.resum()
    assert failed(tree.run()) == {"results_overlay"}


def test_overlay_planning_string(tree):
    tree.write("code/overlay/results/x.json", json.dumps({"p": ".planning/phases"}))
    tree.resum()
    assert "results_overlay" in failed(tree.run())


def test_overlay_internal_decision_label(tree):
    p = tree.root / "code/overlay/README.md"
    p.write_text(p.read_text() + "\nRule: pass if CI lower bound > 0 (D-02).\n")
    tree.resum()
    assert failed(tree.run()) == {"results_overlay"}


def test_overlay_readme_digest_differs(tree):
    p = tree.root / "code/overlay/results/README.md"
    p.write_text(p.read_text().replace("ab" * 32, "12" * 32))
    tree.resum()
    assert failed(tree.run()) == {"results_overlay"}


def _write_amendment2(tree, rerun=None):
    rerun = rerun or {"compare": "identical_modulo_amendment", "published_sha256": "12" * 32, "fresh_sha256": "34" * 32}
    p = tree.tmp / "src" / "amendment2.json"
    p.write_text(json.dumps({"rerun_equivalence": rerun}))
    return rerun


def _set_results_readme(tree, text):
    p = tree.root / "code/overlay/results/README.md"
    p.write_text(p.read_text() + text)
    tree.resum()


def test_overlay_cites_rename_record(tree):
    _write_amendment2(tree)
    _set_results_readme(tree, f"Rename record: {rename_amendment.AMENDMENT2_REF}\nidentical_modulo_amendment {'12' * 32} {'34' * 32}\n")
    assert failed(tree.run()) == set()


def test_overlay_without_rename_citation_fails(tree):
    _write_amendment2(tree)
    res = tree.run()
    assert failed(res) == {"results_overlay"}
    details = next(c["details"] for c in res["_report"]["checks"] if c["name"] == "results_overlay")
    assert any(rename_amendment.AMENDMENT2_REF in x for x in details), details


def test_overlay_rename_record_digest_missing_fails(tree):
    _write_amendment2(tree)
    _set_results_readme(tree, f"Rename record: {rename_amendment.AMENDMENT2_REF}\nidentical_modulo_amendment {'12' * 32}\n")
    assert failed(tree.run()) == {"results_overlay"}


def test_overlay_docs_path(tree):
    tree.write("code/overlay/docs/REPRODUCE.md", "x")
    tree.resum()
    assert {"planning_docs", "staging_allow_list"} <= failed(tree.run())


def test_overlay_planning_path(tree):
    tree.write("code/overlay/.planning/x.md", "x")
    tree.resum()
    assert "planning_docs" in failed(tree.run())


def test_overlay_absent_is_not_run(tree):
    shutil.rmtree(tree.root / "code")
    res = tree.run()
    assert res["results_overlay"] == "not_run"
    assert tree.run(final=True)["_status"] == "fail"


# ---- final mode
def test_final_valid_tree_with_render_not_run_fails(tree):
    res = tree.run(final=True)
    assert res["render_sample"] == "not_run"
    assert res["_status"] == "fail"


def test_placeholders_pre_final_pass_final_fail(tree):
    assert tree.run()["placeholders"] == "pass"
    assert tree.run(final=True)["placeholders"] == "fail"


def test_leftover_contact_placeholder_alone_fails(tree):
    for n in ("model/NOTICE",):
        (tree.root / n).write_text("Built on LightOn " + C.BASE_REVISION + "\n")
    tree.write("dataset/README.md", "---\nlicense: cc-by-4.0\n---\nDo not train on it. @@CONTACT_EMAIL@@\n")
    tree.resum()
    assert tree.run()["placeholders"] == "pass"
    assert tree.run(final=True)["placeholders"] == "fail"


def test_names_lighton_name_fails(tree):
    tree.write("NAMES.json", json.dumps({**json.loads((tree.root / "NAMES.json").read_text()), "model_name": "light-on-math"}))
    assert tree.run(final=True)["names"] == "fail"
    assert tree.run()["names"] == "pass"


def test_names_forbidden_contact_fails(tree):
    tree.write("NAMES.json", json.dumps({**json.loads((tree.root / "NAMES.json").read_text()), "contact_email": "arxiv@example.org"}))
    assert tree.run(final=True)["names"] == "fail"


def test_names_valid_passes(tree):
    assert tree.run(final=True)["names"] == "pass"


# ---- render sample
def test_render_not_run_without_docker(tree, monkeypatch):
    monkeypatch.setattr(rc, "docker_available", lambda image: False)
    res = tree.run(render=True)
    assert res["render_sample"] == "not_run"
    assert res["_status"] == "pass"
    assert tree.run(final=True, render=True)["_status"] == "fail"


def test_render_image_mismatch_fails(tree):
    assert tree.run(render_image="texlive/texlive:latest")["render_sample"] == "fail"


def test_render_freeze_image_mismatch_fails(tree):
    p = tree.tmp / "src" / "freeze.json"
    p.write_text(json.dumps({"render_config": {"image": "other"}, "render_image_digest": "sha256:0"}))
    assert tree.run()["render_sample"] == "fail"


def test_cli_help_lists_subcommands(capsys):
    assert rc.main(["--help"]) == 0
    assert rc.main(["bogus"]) == 2
    out = capsys.readouterr().out
    assert all(w in out for w in ("staging", "git", "all"))


@pytest.mark.docker
def test_render_sample_reproduces_real_records(tree):
    real = [json.loads(x) for x in (ROOT / "data/benchmark/records.jsonl").read_text().splitlines()[:3]]
    for rows in (tree.train, tree.bench):
        for r, src in zip(rows, real, strict=False):
            r.update(latex=src["latex"], bitmap_sha256=src["bitmap_sha256"])
    tree.refresh_data()
    cfg = tree.config()
    out = rc.check_render_sample(cfg, rc.Tables(cfg), render_n=2, render=True)
    assert out["status"] == "pass", out
    assert out["details"] == []


def test_final_names_reject_placeholder_contact(tmp_path):
    cfg = rc.Config(root=tmp_path)
    cfg.names_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.names_json.write_text(
        json.dumps({"dataset_name": "abc-data", "model_name": "abc-model", "repo_name": "abc-code",
                    "hf_namespace": "ns", "gh_owner": "own", "contact_email": "contact@example.invalid",
                    "copyright_holder": "PENDING copyright holder"}),
        encoding="utf-8",
    )
    out = rc.check_names(cfg, final=True)
    assert out["status"] == "fail"
    assert any("contact_email" in p and "placeholder" in p for p in out["details"])
    assert any("copyright_holder" in p and "placeholder" in p for p in out["details"])


def test_repository_allowlists_admit_the_chart_files_and_nothing_nearby():
    ok_ds = lambda rel: any(r.match(rel) for r in rc.DATASET_ALLOW)
    ok_model = lambda rel: any(r.match(rel) for r in rc.MODEL_ALLOW)
    assert ok_ds("assets/chart_pipeline.svg") and not ok_ds("assets/chart_buckets.svg") and not ok_ds("assets/x.png")
    assert ok_model("assets/chart_flips.svg") and not ok_model("assets/script.js")
