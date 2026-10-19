"""Allow-list release check for the staged package (and, through `git`/`all`, the public history).

Usage:
  python -m scripts.release.release_check staging [--final] [--render-n N] [--no-render] [--report PATH]
  python -m scripts.release.release_check git --repo PATH [--final] [--reserved-source P ...] [--attribution PATH]
  python -m scripts.release.release_check all --repo PATH [--final] [--render-n N] [--no-render]

Every rule fails closed: a staged file no rule allows is an error. The check recomputes each guarantee from pinned
digests and source records instead of trusting the staging code. Exit 0 pass, 1 fail, 2 usage error. It only reads.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scripts.release import common as C
from scripts.release import public_paths

MAX_DETAILS = 20
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".txt", ".yaml", ".yml"}
TEXT_NAMES = {"Modelfile", "NOTICE", "LICENSE", "SHA256SUMS"}
PLACEHOLDER_RE = re.compile(r"@@[A-Z_]+@@")
VERSION_RE = re.compile(r"^v\d+$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")

DATASET_ALLOW = (
    re.compile(r"^(README\.md|ATTRIBUTION\.md|ATTRIBUTION\.jsonl|DATASET_MANIFEST\.json|SHA256SUMS)$"),
    re.compile(r"^data/train-[^/]+\.parquet$"),
    re.compile(r"^data/benchmark-[^/]+\.parquet$"),
)
MODEL_ALLOW = (
    re.compile(
        r"^(README\.md|NOTICE|LICENSE|PROVENANCE\.json|Modelfile|SHA256SUMS|text-q8_0\.gguf|mmproj-stock-q8_0\.gguf"
        r"|adapter_config\.json|adapter_model\.safetensors)$"
    ),
)


def result(name: str, problems: list[str] | None = None, *, not_run: str | None = None, note: str | None = None) -> dict:
    if not_run is not None:
        return {"name": name, "status": "not_run", "details": [not_run]}
    problems = problems or []
    details = list(problems[:MAX_DETAILS])
    if len(problems) > MAX_DETAILS:
        details.append(f"(+{len(problems) - MAX_DETAILS} more)")
    if not problems and note:
        details.append(note)
    return {"name": name, "status": "fail" if problems else "pass", "details": details}


def overall(results: list[dict], final: bool) -> str:
    if any(r["status"] == "fail" for r in results):
        return "fail"
    if final and any(r["status"] == "not_run" for r in results):
        return "fail"
    return "pass"


def docker_available(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@dataclass
class Config:
    root: Path = C.STAGING
    expected: dict = field(default_factory=lambda: dict(C.EXPECTED))
    trainset: Path = Path("data/train/sets/full/trainset.jsonl")
    trainset_sha256: str = C.TRAINSET_SHA256
    bench_records: Path = Path("data/benchmark/records.jsonl")
    train_catalog: Path = Path("data/catalog/train_dev.jsonl")
    bench_catalog: Path = Path("data/benchmark/catalog.jsonl")
    freeze_json: Path = Path("data/benchmark/BENCHMARK_FREEZE.json")
    results_json: Path = C.RESULTS_JSON
    amendment_json: Path = C.FREEZE_AMENDMENT_JSON
    phase4: tuple = C.PHASE4_SOURCES
    adapter_sha: str = C.ADAPTER_SHA256
    gguf_sha: str = C.GGUF_Q8_SHA256
    proj_sha: str = C.PROJECTOR_SHA256
    base_revision: str = C.BASE_REVISION
    texlive_image: str = C.TEXLIVE_IMAGE
    render_image: str | None = None
    fs: object = None
    spec: list | None = None
    contact_forbidden: frozenset = field(default_factory=lambda: frozenset(C.CONTACT_FORBIDDEN))

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def overlay(self) -> Path:
        return self.root / "code" / "overlay"

    @property
    def names_json(self) -> Path:
        return self.root / "NAMES.json"


# ------------------------------------------------------------------------------------------------ shared readers
def _is_text(rel: str) -> bool:
    p = Path(rel)
    return p.suffix.lower() in TEXT_SUFFIXES or p.name in TEXT_NAMES


def _walk(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _front_matter(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i])
    return None


class Tables:
    """Parquet readers: rows without the image column, plus the schema names."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rows: dict[str, list[dict] | None] = {}
        self.errors: list[str] = []

    def path(self, split: str) -> Path:
        return self.cfg.dataset / (C.TRAIN_PARQUET if split == "train" else C.BENCH_PARQUET)

    def schema(self, split: str) -> list[str] | None:
        import pyarrow.parquet as pq

        p = self.path(split)
        if not p.is_file():
            return None
        return list(pq.ParquetFile(p).schema_arrow.names)

    def rows(self, split: str) -> list[dict] | None:
        import pyarrow.parquet as pq

        if split not in self._rows:
            p = self.path(split)
            if not p.is_file():
                self.errors.append(f"missing {p.relative_to(self.cfg.dataset).as_posix()}")
                self._rows[split] = None
            else:
                cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c != "image"]
                self._rows[split] = pq.read_table(p, columns=cols).to_pylist()
        return self._rows[split]


def _need(t: Tables, *splits: str) -> list[str]:
    return [m for s in splits if t.rows(s) is None for m in [f"{s} Parquet is missing"]]


# --------------------------------------------------------------------------------------------------- the checks
def check_staging_allow_list(cfg: Config) -> dict:
    problems: list[str] = []
    spec = cfg.spec if cfg.spec is not None else public_paths.load_spec()
    for rel in _walk(cfg.dataset):
        if not any(r.match(rel) for r in DATASET_ALLOW):
            problems.append(f"dataset/{rel} is not allowed")
    for rel in _walk(cfg.model):
        if not any(r.match(rel) for r in MODEL_ALLOW):
            problems.append(f"model/{rel} is not allowed")
    for rel in _walk(cfg.overlay):
        if rel != C.SHA256SUMS and public_paths.classify(rel, spec) != "overlay":
            problems.append(f"overlay/{rel} is not an overlay path")
    if not cfg.dataset.is_dir():
        problems.append("dataset root is missing")
    if not cfg.model.is_dir():
        problems.append("model root is missing")
    return result("staging_allow_list", problems)


def check_records_license(cfg: Config, t: Tables) -> dict:
    from lightonocr_math.ids import is_arxiv_id

    problems = _need(t, "train", "benchmark")
    for split in ("train", "benchmark"):
        bad: Counter = Counter()
        for r in t.rows(split) or []:
            spdx = r.get("license_spdx")
            if spdx not in C.ALLOWED_SPDX:
                bad[f"{split}: license {spdx!r}"] += 1
            elif r.get("license_uri") != C.LICENSE_URI[spdx]:
                bad[f"{split}: license_uri does not match {spdx}"] += 1
            v = r.get("paper_version")
            if not isinstance(v, str) or not VERSION_RE.match(v):
                bad[f"{split}: paper_version {v!r}"] += 1
            aid = r.get("arxiv_id")
            if not isinstance(aid, str) or not is_arxiv_id(aid):
                bad[f"{split}: arxiv_id {aid!r}"] += 1
            if not (isinstance(r.get("title"), str) and r["title"].strip()):
                bad[f"{split}: empty title"] += 1
            if not (isinstance(r.get("authors"), list) and r["authors"] and all(isinstance(a, str) and a.strip() for a in r["authors"])):
                bad[f"{split}: empty authors"] += 1
            if r.get("arxiv_url") != f"https://arxiv.org/abs/{aid}{v}":
                bad[f"{split}: arxiv_url is not https://arxiv.org/abs/<id><version>"] += 1
        problems += [f"{k} ({n} rows)" for k, n in bad.items()]
    return result("records_license", problems)


def check_records_fields(cfg: Config, t: Tables) -> dict:
    problems: list[str] = []
    for split, want in (("train", C.TRAIN_COLUMNS), ("benchmark", C.BENCH_COLUMNS)):
        names = t.schema(split)
        if names is None:
            problems.append(f"{split} Parquet is missing")
            continue
        if set(names) != set(want) or len(names) != len(want):
            problems.append(f"{split}: columns differ (extra {sorted(set(names) - set(want))}, missing {sorted(set(want) - set(names))})")
    names = t.schema("benchmark") or []
    for f in C.STRIPPED_BENCH_FIELDS:
        if f in names:
            problems.append(f"benchmark: stripped field {f} is present")
    return result("records_fields", problems)


def _fs(cfg: Config):
    if cfg.fs is None:
        from lightonocr_math import splits

        cfg.fs = splits.frozen()
    return cfg.fs


def _jsonl_ids(path: Path, key: str = "id") -> list[str] | None:
    if not path.is_file():
        return None
    return [str(r[key]) for r in C.read_jsonl(path)]


def check_splits(cfg: Config, t: Tables) -> dict:
    from lightonocr_math import splits

    problems = _need(t, "train", "benchmark")
    train, bench = t.rows("train") or [], t.rows("benchmark") or []
    exp = cfg.expected
    fs = _fs(cfg)
    tp, bp = {r["arxiv_id"] for r in train}, {r["arxiv_id"] for r in bench}
    for label, rows, papers in (("train", train, tp), ("bench", bench, bp)):
        if len(rows) != exp[f"{label}_rows"]:
            problems.append(f"{label}: {len(rows)} rows, expected {exp[f'{label}_rows']}")
        if len(papers) != exp[f"{label}_papers"]:
            problems.append(f"{label}: {len(papers)} papers, expected {exp[f'{label}_papers']}")
        lic = dict(Counter(r["license_spdx"] for r in rows))
        if lic != exp[f"{label}_licenses"]:
            problems.append(f"{label}: license counts {lic}, expected {exp[f'{label}_licenses']}")
    both = tp & bp
    if both:
        problems.append(f"{len(both)} papers are in both splits, e.g. {sorted(both)[:3]}")
    for p in sorted(bp):
        if not splits.is_benchmark(p, fs):
            problems.append(f"benchmark paper {p} is not a frozen benchmark paper")
    try:
        splits.assert_disjoint(tp, context="train papers", fs=fs)
    except splits.LeakageError as e:
        problems.append(str(e))

    if not cfg.trainset.is_file():
        problems.append(f"{cfg.trainset} is missing")
    else:
        if C.sha256_stream(cfg.trainset) != cfg.trainset_sha256:
            problems.append("trainset.jsonl sha256 differs from the pinned digest")
        ids = [r["id"] for r in train]
        want = _jsonl_ids(cfg.trainset) or []
        if len(set(ids)) != len(ids) or set(ids) != set(want) or len(ids) != len(want):
            problems.append("train ids differ from the pinned trainset ids")
    want = _jsonl_ids(cfg.bench_records)
    if want is None:
        problems.append(f"{cfg.bench_records} is missing")
    else:
        ids = [r["id"] for r in bench]
        if len(set(ids)) != len(ids) or set(ids) != set(want) or len(ids) != len(want):
            problems.append("benchmark ids differ from data/benchmark/records.jsonl")
    return result("splits", problems)


def check_images(cfg: Config, t: Tables) -> dict:
    import hashlib

    import pyarrow.parquet as pq

    problems: list[str] = []
    for split in ("train", "benchmark"):
        p = t.path(split)
        if not p.is_file():
            problems.append(f"{split} Parquet is missing")
            continue
        bad_hash = bad_path = 0
        for batch in pq.ParquetFile(p).iter_batches(columns=["id", "image", "image_sha256"], batch_size=500):
            for r in batch.to_pylist():
                img = r["image"] or {}
                data = img.get("bytes")
                if not isinstance(data, bytes | bytearray) or hashlib.sha256(data).hexdigest() != r["image_sha256"]:
                    bad_hash += 1
                path = img.get("path")
                if not path or "/" in path or ".." in path or "\\" in path:
                    bad_path += 1
        if bad_hash:
            problems.append(f"{split}: {bad_hash} image bytes do not hash to image_sha256")
        if bad_path:
            problems.append(f"{split}: {bad_path} image paths are not a bare file name")
    return result("images", problems)


def check_attribution(cfg: Config, t: Tables) -> dict:
    problems = _need(t, "train", "benchmark")
    ajl, amd = cfg.dataset / "ATTRIBUTION.jsonl", cfg.dataset / "ATTRIBUTION.md"
    if not ajl.is_file() or not amd.is_file():
        return result("attribution", problems + ["dataset ATTRIBUTION.jsonl or ATTRIBUTION.md is missing"])
    entries = C.read_jsonl(ajl)
    by_id = {e["arxiv_id"]: e for e in entries}
    if len(by_id) != len(entries):
        problems.append("duplicate arxiv_id in ATTRIBUTION.jsonl")
    papers: dict[str, dict] = {}
    for split in ("train", "benchmark"):
        for r in t.rows(split) or []:
            p = papers.setdefault(r["arxiv_id"], {"versions": set(), "licenses": set(), "splits": Counter()})
            p["versions"].add(r["paper_version"])
            p["licenses"].add(r["license_spdx"])
            p["splits"][split] += 1
    for pid in sorted(set(papers) - set(by_id)):
        problems.append(f"{pid} has rows but no ATTRIBUTION entry")
    for pid in sorted(set(by_id) - set(papers)):
        problems.append(f"{pid} is in ATTRIBUTION but has no rows")
    md = _read(amd)
    for pid in sorted(set(papers) & set(by_id)):
        p, e = papers[pid], by_id[pid]
        if p["versions"] != {e.get("version")}:
            problems.append(f"{pid}: version differs between rows and ATTRIBUTION")
        if p["licenses"] != {e.get("license_spdx")}:
            problems.append(f"{pid}: license differs between rows and ATTRIBUTION")
        if sum(p["splits"].values()) != e.get("n_rows") or sorted(p["splits"]) != sorted(e.get("splits", [])):
            problems.append(f"{pid}: n_rows or splits differ from the rows")
        if f"{pid}{e.get('version')}" not in md:
            problems.append(f"{pid}{e.get('version')} does not appear in ATTRIBUTION.md")
    for split in ("train", "benchmark"):
        got = sum(e["n_rows"] for e in entries if e.get("splits") == [split])
        if got != len(t.rows(split) or []):
            problems.append(f"{split}: ATTRIBUTION n_rows sum {got} differs from the row count")
    for cat_path, split in ((cfg.train_catalog, "train"), (cfg.bench_catalog, "benchmark")):
        if not cat_path.is_file():
            problems.append(f"{cat_path} is missing")
            continue
        cat = {r["arxiv_id"]: r for r in C.read_jsonl(cat_path)}
        for pid, p in papers.items():
            if split not in p["splits"]:
                continue
            c = cat.get(pid)
            if c is None:
                problems.append(f"{pid}: not in the {split} catalog")
            elif {c.get("version")} != p["versions"] or {c.get("license_spdx")} != p["licenses"]:
                problems.append(f"{pid}: catalog version or license differs from the rows")
    ov = cfg.overlay / "ATTRIBUTION.jsonl"
    if ov.is_file() and ov.read_bytes() != ajl.read_bytes():
        problems.append("overlay ATTRIBUTION.jsonl differs from the dataset one")
    return result("attribution", problems)


def check_leak(cfg: Config, t: Tables) -> dict:
    from lightonocr_math import leakscan

    p = t.path("train")
    if not p.is_file():
        return result("leak", ["train Parquet is missing"])
    leaks = leakscan.scan_file(p, _fs(cfg), p.as_posix())
    return result("leak", [f"{x.where}: {x.arxiv_id} ({x.reason})" for x in leaks])


def check_render_sample(cfg: Config, t: Tables, *, render_n: int, render: bool) -> dict:
    from lightonocr_math import render as R

    problems: list[str] = []
    image = cfg.render_image or R.IMAGE
    if image != cfg.texlive_image:
        problems.append("render.IMAGE differs from the pinned TeX Live image")
    try:
        freeze = C.load_json(cfg.freeze_json)
        if freeze["render_config"]["image"] != image or freeze["render_image_digest"] not in image:
            problems.append("BENCHMARK_FREEZE.json render image differs from render.IMAGE")
    except (OSError, KeyError, ValueError) as e:
        problems.append(f"BENCHMARK_FREEZE.json unreadable: {e}")
    if problems:
        return result("render_sample", problems)
    if not render:
        return result("render_sample", not_run="rendering disabled (--no-render)")
    if not docker_available(image):
        return result("render_sample", not_run=f"Docker image {image} is not available")
    problems += _need(t, "train", "benchmark")
    if problems:
        return result("render_sample", problems)
    cache = Path(tempfile.mkdtemp(prefix="release_check_render_"))
    try:
        for split in ("train", "benchmark"):
            rows = sorted(t.rows(split) or [], key=lambda r: r["id"])
            n = min(render_n, len(rows))
            picked = [rows[(i * len(rows)) // n] for i in range(n)] if n else []
            try:
                res = R.render_many([r["latex"] for r in picked], cache=cache)
            except R.RenderInfrastructureError as e:
                return result("render_sample", not_run=f"render infrastructure error: {e}")
            for r, x in zip(picked, res, strict=True):
                if x.status != "rendered" or x.bitmap_sha256 != r["bitmap_sha256"]:
                    problems.append(f"{split} {r['id']}: re-render is {x.status}, bitmap hash {'differs' if x.status == 'rendered' else 'absent'}")
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return result("render_sample", problems)


def check_hashes(cfg: Config, t: Tables) -> dict:
    problems: list[str] = []
    for label, root in (("dataset", cfg.dataset), ("model", cfg.model), ("overlay", cfg.overlay)):
        if label == "overlay" and not root.is_dir():
            continue
        problems += [f"{label}: {m}" for m in C.verify_sha256sums(root)]
    for rel, want in (
        ("adapter_model.safetensors", cfg.adapter_sha),
        (C.GGUF_NAME, cfg.gguf_sha),
        (C.MMPROJ_NAME, cfg.proj_sha),
    ):
        p = cfg.model / rel
        if not p.is_file():
            problems.append(f"model/{rel} is missing")
        elif C.sha256_stream(p) != want:
            problems.append(f"model/{rel} differs from the pinned digest")
    man = cfg.dataset / "DATASET_MANIFEST.json"
    if not man.is_file():
        problems.append("DATASET_MANIFEST.json is missing")
    else:
        sums = C.load_json(man).get("parquet_sha256", {})
        for rel in (C.TRAIN_PARQUET, C.BENCH_PARQUET):
            p = cfg.dataset / rel
            if not p.is_file():
                problems.append(f"dataset/{rel} is missing")
            elif sums.get(rel) != C.sha256_stream(p):
                problems.append(f"dataset/{rel} differs from DATASET_MANIFEST.json")
    return result("hashes", problems)


def check_model_card(cfg: Config) -> dict:
    problems: list[str] = []
    readme = cfg.model / "README.md"
    if not readme.is_file():
        problems.append("model README.md is missing")
    else:
        text = _read(readme)
        fm = _front_matter(text)
        lines = {x.strip() for x in (fm or "").splitlines()}
        if fm is None:
            problems.append("model card has no front matter")
        for want in ("license: apache-2.0", f"base_model: {C.BASE_MODEL_ID}", "base_model_relation: adapter"):
            if want not in lines:
                problems.append(f"model card front matter lacks `{want}`")
        if cfg.base_revision not in text:
            problems.append("model card does not name the base revision")
        if C.DISCLAIMER not in text:
            problems.append("model card lacks the independence disclaimer")
    notice = cfg.model / "NOTICE"
    if not notice.is_file() or "LightOn" not in _read(notice) or cfg.base_revision not in _read(notice):
        problems.append("model NOTICE is missing, does not name LightOn or does not name the base revision")
    lic = cfg.model / "LICENSE"
    if not lic.is_file() or not _read(lic).startswith("Apache License") or "Version 2.0, January 2004" not in _read(lic):
        problems.append("model LICENSE is not the Apache License 2.0 text")
    mf = cfg.model / "Modelfile"
    if not mf.is_file():
        problems.append("Modelfile is missing")
    else:
        froms = [x.strip() for x in _read(mf).splitlines() if x.strip().upper().startswith("FROM")]
        if froms != [f"FROM ./{C.GGUF_NAME}", f"FROM ./{C.MMPROJ_NAME}"]:
            problems.append("Modelfile FROM lines are not exactly the two relative GGUF files")
    dcard = cfg.dataset / "README.md"
    if not dcard.is_file():
        problems.append("dataset README.md is missing")
    else:
        text = _read(dcard)
        fm = _front_matter(text) or ""
        if "license: cc-by-4.0" not in {x.strip() for x in fm.splitlines()}:
            problems.append("dataset card front matter lacks `license: cc-by-4.0`")
        if "Do not train on it" not in text:
            problems.append("dataset card lacks the eval-only warning")
    return result("model_card", problems)


def check_planning_docs(cfg: Config) -> dict:
    problems = []
    for rel in _walk(cfg.overlay):
        if rel.startswith(("docs/", ".planning/")) or rel in ("docs", ".planning"):
            problems.append(f"overlay/{rel}: planning and docs paths never leave the machine")
    return result("planning_docs", problems)


def check_results_overlay(cfg: Config) -> dict:
    if not cfg.overlay.is_dir():
        return result("results_overlay", not_run="overlay is not staged yet")
    problems: list[str] = []
    res = cfg.overlay / "results"
    for f in C.PUBLIC_EVIDENCE_FILES:
        name = Path(f).name
        if (res / name).exists():
            problems.append(f"results/{name} is tracked at evidence/acceptance/, not in the overlay")
    if (res / "rerun_check.json").exists():
        problems.append("results/rerun_check.json is not published")
    for src in cfg.phase4:
        dst = res / Path(src).name
        if not dst.is_file():
            problems.append(f"results/{Path(src).name} is missing")
        elif not Path(src).is_file() or dst.read_bytes() != C.phase4_public_bytes(Path(src)):
            problems.append(f"results/{Path(src).name} differs from its source")
    for rel in _walk(res):
        if _is_text(rel) and ".planning" in _read(res / rel):
            problems.append(f"results/{rel} contains a .planning string")
    byte_copies = {Path(src).name for src in cfg.phase4} - {"MINING.md"}  # MINING.md is label-stripped and scanned
    for rel in _walk(cfg.overlay):
        if rel.startswith("results/") and Path(rel).name in byte_copies:
            continue  # byte copies of development reports are compared with their source above
        if _is_text(rel) and C.INTERNAL_LABEL.search(_read(cfg.overlay / rel)):
            problems.append(f"{rel} contains an internal decision label such as (D-NN)")
    readme = res / "README.md"
    if not readme.is_file():
        problems.append("results/README.md is missing")
    else:
        text = _read(readme)
        try:
            am = C.load_json(cfg.amendment_json)["rerun_equivalence"]
            published = C.load_json(cfg.results_json)
        except (OSError, KeyError, ValueError) as e:
            problems.append(f"FREEZE_AMENDMENT.json or results.json unreadable: {e}")
            am, published = None, None
        if am is not None:
            if "evidence/acceptance/FREEZE_AMENDMENT.json" not in text:
                problems.append("results/README.md does not cite evidence/acceptance/FREEZE_AMENDMENT.json")
            for key in ("compare", "published_sha256", "fresh_sha256"):
                if str(am.get(key)) not in text:
                    problems.append(f"results/README.md rerun record lacks {key}={am.get(key)}")
            inputs = {v["path"]: v["sha256"] for v in (published.get("inputs") or {}).values()}
            hexre = re.compile(r"\b[0-9a-f]{64}\b")
            for line in text.splitlines():
                for path, sha in inputs.items():
                    if path in line:
                        found = hexre.findall(line)
                        if found and sha not in found:
                            problems.append(f"results/README.md lists {path} with a digest that differs from results.json")
    return result("results_overlay", problems)


def _parquet_strings(path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = [
        f.name
        for f in pf.schema_arrow
        if pa.types.is_string(f.type) or (pa.types.is_list(f.type) and pa.types.is_string(f.type.value_type))
    ]
    if not cols:
        return
    for batch in pf.iter_batches(columns=cols, batch_size=1000):
        for name in cols:
            for v in batch.column(name).to_pylist():
                if v is None:
                    continue
                yield name, _join(v)


def _join(v) -> str:
    return "\n".join(v) if isinstance(v, list) else v


def check_forbidden_content(cfg: Config) -> dict:
    problems: list[str] = []
    for label, root in (("dataset", cfg.dataset), ("model", cfg.model), ("overlay", cfg.overlay)):
        for rel in _walk(root):
            for name, pat in C.FORBIDDEN_FILE_PATTERNS:
                if pat.search(rel):
                    problems.append(f"{label}/{rel}: forbidden file name ({name})")
            p = root / rel
            if _is_text(rel):
                text = _read(p)
                for name, pat in C.FORBIDDEN_PATTERNS:
                    if pat.search(text):
                        problems.append(f"{label}/{rel}: forbidden content ({name})")
            elif p.suffix == ".parquet":
                seen: set[tuple[str, str]] = set()
                for col, s in _parquet_strings(p):
                    for name, pat in C.FORBIDDEN_PATTERNS:
                        if (col, name) not in seen and pat.search(s):
                            seen.add((col, name))
                            problems.append(f"{label}/{rel}: forbidden content ({name}) in column {col}")
    return result("forbidden_content", problems)


def check_placeholders(cfg: Config, final: bool) -> dict:
    counts: Counter = Counter()
    for label, root in (("dataset", cfg.dataset), ("model", cfg.model), ("overlay", cfg.overlay)):
        for rel in _walk(root):
            if _is_text(rel):
                n = len(PLACEHOLDER_RE.findall(_read(root / rel)))
                if n:
                    counts[f"{label}/{rel}"] = n
    if final:
        return result("placeholders", [f"{rel}: {n} placeholder(s) left" for rel, n in counts.items()])
    return result("placeholders", [], note=f"not final: {sum(counts.values())} placeholder(s) in {len(counts)} file(s)")


def check_names(cfg: Config, final: bool) -> dict:
    if not final:
        return result("names", [], note="not final: names are not checked yet")
    if not cfg.names_json.is_file():
        return result("names", ["NAMES.json is missing"])
    try:
        names = C.load_json(cfg.names_json)
    except ValueError as e:
        return result("names", [f"NAMES.json unreadable: {e}"])
    problems = []
    for key in C.PLACEHOLDERS:
        v = names.get(key)
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{key} is missing or empty")
            continue
        if C.NAME_FORBIDDEN.search(v):
            problems.append(f"{key} contains the LightOn name")
        if key in ("dataset_name", "model_name", "repo_name") and not NAME_RE.match(v):
            problems.append(f"{key} is not a valid repository name")
    mail = names.get("contact_email")
    if isinstance(mail, str) and mail.strip() and (not EMAIL_RE.match(mail) or mail in cfg.contact_forbidden):
        problems.append("contact_email is invalid or is the arXiv User-Agent address")
    elif isinstance(mail, str) and mail.strip().lower().endswith(".invalid"):
        problems.append("contact_email is still a placeholder (.invalid); supply the real takedown address")
    return result("names", problems)


def check_staging(
    root: Path = C.STAGING,
    final: bool = False,
    render_n: int = 25,
    render: bool = True,
    expected: dict | None = None,
    cfg: Config | None = None,
) -> dict:
    cfg = cfg or Config(root=Path(root))
    if expected:
        cfg.expected = {**cfg.expected, **expected}
    t = Tables(cfg)
    checks = [
        check_staging_allow_list(cfg),
        check_records_license(cfg, t),
        check_records_fields(cfg, t),
        check_splits(cfg, t),
        check_images(cfg, t),
        check_attribution(cfg, t),
        check_leak(cfg, t),
        check_render_sample(cfg, t, render_n=render_n, render=render),
        check_hashes(cfg, t),
        check_model_card(cfg),
        check_planning_docs(cfg),
        check_results_overlay(cfg),
        check_forbidden_content(cfg),
        check_placeholders(cfg, final),
        check_names(cfg, final),
    ]
    return {"surface": "staging", "final": final, "status": overall(checks, final), "checks": checks}


# -------------------------------------------------------------------------------------------------------- CLI
def _print(report: dict) -> None:
    for c in report["checks"]:
        first = c["details"][0] if c["details"] else ""
        print(f"{c['status'].upper():8} {c['name']}" + (f": {first}" if first else ""))
    print(f"{report['surface']}: {report['status'].upper()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="release_check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p, git: bool, staging: bool):
        p.add_argument("--final", action="store_true")
        p.add_argument("--report", default=str(C.STAGING / "RELEASE_CHECK.json"))
        if staging:
            p.add_argument("--render-n", type=int, default=25)
            p.add_argument("--no-render", action="store_true")
        if git:
            p.add_argument("--repo", required=True)
            p.add_argument("--reserved-source", action="append", default=None)
            p.add_argument("--attribution", default=None)

    common_args(sub.add_parser("staging", help="check release/dataset, release/model and the overlay"), False, True)
    common_args(sub.add_parser("git", help="check the exported public repository"), True, False)
    common_args(sub.add_parser("all", help="both surfaces"), True, True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0

    reports = []
    if a.cmd in ("staging", "all"):
        reports.append(check_staging(final=a.final, render_n=a.render_n, render=not a.no_render))
    if a.cmd in ("git", "all"):
        from scripts.release.release_check_git import check_git

        reserved = [Path(p) for p in a.reserved_source] if a.reserved_source else [p for p in C.RESERVED_SOURCES if p.is_file()]
        reports.append(
            check_git(Path(a.repo), final=a.final, reserved_sources=reserved or None, attribution=Path(a.attribution) if a.attribution else None)
        )
    for r in reports:
        _print(r)
    status = "fail" if any(r["status"] == "fail" for r in reports) else "pass"
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": status, "final": a.final, "reports": reports}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
