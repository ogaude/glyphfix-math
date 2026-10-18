"""Stage the dataset half of the release into release/dataset/ (local only, nothing is uploaded).

Usage: uv run --group release python -m scripts.release.stage_dataset [--out release/dataset]

Every gate fails closed with a message: license, version, catalog, hash and split errors never skip a row.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from scripts.release.common import (
    ALLOWED_SPDX,
    BENCH_COLUMNS,
    BENCH_PARQUET,
    DATASET_DIR,
    EXPECTED,
    LICENSE_DEED,
    LICENSE_URI,
    PLACEHOLDERS,
    STAGING,
    STRIPPED_BENCH_FIELDS,
    TRAIN_COLUMNS,
    TRAIN_PARQUET,
    TRAINSET_SHA256,
    read_jsonl,
    render_template,
    sha256_stream,
    write_sha256sums,
)

TRAINSET = Path("data/train/sets/full/trainset.jsonl")
CORPUS_TRAIN = Path("data/corpus/train/records.jsonl")
BENCH_RECORDS = Path("data/benchmark/records.jsonl")
TRAIN_CATALOG = Path("data/catalog/train_dev.jsonl")
BENCH_CATALOG = Path("data/benchmark/catalog.jsonl")
MINING_MANIFEST = Path("data/mining/MINING_MANIFEST.json")
CARD_TEMPLATE = Path("release_templates/dataset_card.md")

_VERSION_RE = re.compile(r"^v\d+$")


class StageError(RuntimeError):
    pass


def _authors(row: dict) -> list[str]:
    a = row.get("authors")
    if isinstance(a, str):
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            a = None
    if isinstance(a, list) and a and all(isinstance(x, str) and x for x in a):
        return list(a)
    raw = row.get("authors_raw")
    if isinstance(raw, str) and raw.strip():
        parts = [p.strip() for p in re.split(r",|\band\b", raw) if p.strip()]
        if parts:
            return parts
    raise StageError(f"catalog row {row.get('arxiv_id')}: no usable author list")


def load_catalog(path: Path | str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        out[r["arxiv_id"]] = {
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "authors": _authors(r),
            "version": r["version"],
            "license_spdx": r["license_spdx"],
            "license_uri": r.get("license_uri"),
        }
    return out


def _check_license(rec: dict, cat: dict[str, dict], where: str) -> None:
    spdx, uri, ver = rec.get("license_spdx"), rec.get("license_uri"), rec.get("paper_version")
    if spdx not in ALLOWED_SPDX:
        raise StageError(f"{where}: license {spdx!r} is not allowed")
    if uri != LICENSE_URI[spdx]:
        raise StageError(f"{where}: license_uri {uri!r} does not match {spdx}")
    if not isinstance(ver, str) or not _VERSION_RE.match(ver):
        raise StageError(f"{where}: paper_version {ver!r} is not of the form vN")
    c = cat.get(rec["arxiv_id"])
    if c is None:
        raise StageError(f"{where}: no catalog row for {rec['arxiv_id']}")
    if c["version"] != ver:
        raise StageError(f"{where}: catalog version {c['version']} differs from record version {ver}")
    if c["license_spdx"] != spdx:
        raise StageError(f"{where}: catalog license {c['license_spdx']} differs from record license {spdx}")


def _read_png(root: Path, rel: str, want_sha: str, where: str) -> bytes:
    p = root / rel
    if not p.is_file():
        raise StageError(f"{where}: image file missing: {rel}")
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != want_sha:
        raise StageError(f"{where}: image bytes do not hash to image_sha256")
    return data


def _common(rec: dict, cat: dict[str, dict], data: bytes) -> dict:
    c = cat[rec["arxiv_id"]]
    return {
        "id": rec["id"],
        "image": {"bytes": data, "path": f"{rec['id']}.png"},
        "latex": rec["latex"],
        "arxiv_id": rec["arxiv_id"],
        "paper_version": rec["paper_version"],
        "title": c["title"],
        "authors": list(c["authors"]),
        "license_spdx": rec["license_spdx"],
        "license_uri": rec["license_uri"],
        "arxiv_url": f"https://arxiv.org/abs/{rec['arxiv_id']}{rec['paper_version']}",
        "flags": list(rec.get("flags") or []),
        "image_sha256": rec["image_sha256"],
        "bitmap_sha256": rec["bitmap_sha256"],
        "width": int(rec["width"]),
        "height": int(rec["height"]),
    }


def build_train_rows(trainset_path, corpus_records_path, catalog_path, root: Path | str = ".") -> list[dict]:
    root = Path(root)
    cat = load_catalog(catalog_path) if not isinstance(catalog_path, dict) else catalog_path
    corpus = {r["id"]: r for r in read_jsonl(corpus_records_path)}
    rows = []
    for t in read_jsonl(trainset_path):
        where = f"train row {t['id']}"
        c = corpus.get(t["id"])
        if c is None:
            raise StageError(f"{where}: not in the corpus records")
        if c["latex"] != t["latex"]:
            raise StageError(f"{where}: trainset latex differs from the corpus record")
        for k in ("arxiv_id", "paper_version", "license_spdx", "license_uri"):
            if c.get(k) != t.get(k):
                raise StageError(f"{where}: {k} differs between trainset and corpus record")
        _check_license(t, cat, where)
        data = _read_png(root, t["image"], c["image_sha256"], where)
        rec = dict(t, image_sha256=c["image_sha256"], bitmap_sha256=c["bitmap_sha256"])
        row = _common(rec, cat, data)
        row.update(
            target_text=t["target_text"],
            group=t.get("group"),
            mining_class=t.get("mining_class"),
            control_glyph=t.get("control_glyph"),
            tags=list(t.get("tags") or []),
        )
        rows.append({k: row[k] for k in TRAIN_COLUMNS})
    return rows


def build_bench_rows(records_path, catalog_path, root: Path | str = ".") -> list[dict]:
    root = Path(root)
    cat = load_catalog(catalog_path) if not isinstance(catalog_path, dict) else catalog_path
    rows = []
    for r in read_jsonl(records_path):
        where = f"benchmark row {r['id']}"
        _check_license(r, cat, where)
        data = _read_png(root, r["image"], r["image_sha256"], where)
        row = _common(r, cat, data)
        for k in ("render_image_digest", "preamble_sha256", "canon_version", "builder_version", "env"):
            row[k] = r[k]
        rows.append({k: row[k] for k in BENCH_COLUMNS if k not in STRIPPED_BENCH_FIELDS})
    return rows


def attribution_entries(train_rows: list[dict], bench_rows: list[dict]) -> list[dict]:
    papers: dict[str, dict] = {}
    splits: dict[str, set] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for split, rows in (("train", train_rows), ("benchmark", bench_rows)):
        for r in rows:
            a = r["arxiv_id"]
            e = papers.setdefault(a, {
                "arxiv_id": a,
                "version": r["paper_version"],
                "title": r["title"],
                "authors": list(r["authors"]),
                "license_spdx": r["license_spdx"],
                "license_uri": r["license_uri"],
                "license_legalcode": LICENSE_DEED[r["license_spdx"]],
                "arxiv_url": r["arxiv_url"],
            })
            if e["version"] != r["paper_version"] or e["license_spdx"] != r["license_spdx"]:
                raise StageError(f"paper {a}: rows disagree on version or license")
            splits[a].add(split)
            counts[a] += 1
    out = []
    for a in sorted(papers):
        e = dict(papers[a], splits=sorted(splits[a]), n_rows=counts[a])
        out.append(e)
    if sum(e["n_rows"] for e in out) != len(train_rows) + len(bench_rows):
        raise StageError("attribution row counts do not sum to the row total")
    return out


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_attribution_md(entries: list[dict]) -> str:
    head = (
        "# Attribution\n\n"
        "The equations in this dataset were extracted from the LaTeX source of the papers listed below, canonicalized, "
        "and re-rendered to images with pdflatex in a pinned TeX Live image. The `latex` column is the canonicalized "
        "LaTeX, not the authors' original source: this is an indication of modification under CC BY 4.0 "
        "section 3(a)(1)(B) (and the equivalent clause of CC BY 3.0).\n\n"
        "Each paper's own license applies to the rows taken from it. The license column links to the legal code of "
        "that license, including its disclaimer of warranties. Rows from CC0-1.0 papers need no attribution and are "
        "listed here for traceability.\n\n"
        "| arXiv | Version | Title | Authors | License | Split | Rows |\n"
        "|-------|---------|-------|---------|---------|-------|------|\n"
    )
    lines = []
    for e in entries:
        lines.append(
            f"| [{e['arxiv_id']}]({e['arxiv_url']}) | {e['version']} | {_md_cell(e['title'])} | "
            f"{_md_cell(', '.join(e['authors']))} | [{e['license_spdx']}]({e['license_legalcode']}) | "
            f"{', '.join(e['splits'])} | {e['n_rows']} |"
        )
    return head + "\n".join(lines) + "\n"


def _count(rows: list[dict], key: str) -> dict:
    return dict(Counter(r[key] for r in rows))


def _gate_counts(train_rows: list[dict], bench_rows: list[dict]) -> None:
    got = {
        "train_rows": len(train_rows),
        "train_papers": len({r["arxiv_id"] for r in train_rows}),
        "train_licenses": _count(train_rows, "license_spdx"),
        "bench_rows": len(bench_rows),
        "bench_papers": len({r["arxiv_id"] for r in bench_rows}),
        "bench_licenses": _count(bench_rows, "license_spdx"),
    }
    if got != EXPECTED:
        raise StageError(f"counts differ from the pinned expectation: got {got}, expected {EXPECTED}")


def _gate_splits(train_rows: list[dict], bench_rows: list[dict]) -> None:
    from lightonocr_math import splits

    train_ids = sorted({r["arxiv_id"] for r in train_rows})
    bench_ids = sorted({r["arxiv_id"] for r in bench_rows})
    splits.assert_disjoint(train_ids, context="release train split")
    bad = [a for a in bench_ids if not splits.is_benchmark(a)]
    if bad:
        raise StageError(f"benchmark rows from non-benchmark papers: {bad[:5]}")
    both = set(train_ids) & set(bench_ids)
    if both:
        raise StageError(f"papers in both splits: {sorted(both)[:5]}")


def _features(bench: bool):
    from datasets import Features, Image, Sequence, Value

    f = {
        "id": Value("string"),
        "image": Image(),
        "latex": Value("string"),
        "arxiv_id": Value("string"),
        "paper_version": Value("string"),
        "title": Value("string"),
        "authors": Sequence(Value("string")),
        "license_spdx": Value("string"),
        "license_uri": Value("string"),
        "arxiv_url": Value("string"),
        "flags": Sequence(Value("string")),
        "image_sha256": Value("string"),
        "bitmap_sha256": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
    }
    if bench:
        for k in ("render_image_digest", "preamble_sha256", "canon_version", "builder_version", "env"):
            f[k] = Value("string")
    else:
        for k in ("target_text", "group", "mining_class", "control_glyph"):
            f[k] = Value("string")
        f["tags"] = Sequence(Value("string"))
    cols = BENCH_COLUMNS if bench else TRAIN_COLUMNS
    return Features({k: f[k] for k in cols if k in f})


def _write_parquet(rows: list[dict], path: Path, bench: bool, cache_dir: Path) -> None:
    from datasets import Dataset

    def gen(rows):
        yield from rows

    ds = Dataset.from_generator(gen, features=_features(bench), gen_kwargs={"rows": rows}, cache_dir=str(cache_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(path))


def stage(out: Path = DATASET_DIR, root: Path = Path(".")) -> dict:
    root = Path(root)
    actual = sha256_stream(root / TRAINSET)
    if actual != TRAINSET_SHA256:
        raise StageError(f"trainset.jsonl sha256 {actual} differs from the pinned {TRAINSET_SHA256}")
    train = build_train_rows(root / TRAINSET, root / CORPUS_TRAIN, root / TRAIN_CATALOG, root)
    bench = build_bench_rows(root / BENCH_RECORDS, root / BENCH_CATALOG, root)
    _gate_counts(train, bench)
    _gate_splits(train, bench)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cache = STAGING / ".hf-cache"
    try:
        _write_parquet(train, out / TRAIN_PARQUET, False, cache)
        _write_parquet(bench, out / BENCH_PARQUET, True, cache)
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    from lightonocr_math import leakscan, splits

    leaks = leakscan.scan_file(out / TRAIN_PARQUET, splits.frozen(), TRAIN_PARQUET)
    if leaks:
        raise StageError(f"leak scan found {len(leaks)} benchmark id(s) in the train Parquet")

    entries = attribution_entries(train, bench)
    (out / "ATTRIBUTION.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8"
    )
    (out / "ATTRIBUTION.md").write_text(render_attribution_md(entries), encoding="utf-8")

    train_lic, bench_lic = _count(train, "license_spdx"), _count(bench, "license_spdx")
    mined = json.loads((root / MINING_MANIFEST).read_text(encoding="utf-8"))["n_records"]
    manifest = {
        "rows": {"train": len(train), "benchmark": len(bench)},
        "papers": {
            "train": len({r["arxiv_id"] for r in train}),
            "benchmark": len({r["arxiv_id"] for r in bench}),
            "total": len(entries),
        },
        "licenses": {"train": train_lic, "benchmark": bench_lic},
        "mined_records": mined,
        "source_sha256": {
            "trainset.jsonl": actual,
            "corpus_train_records.jsonl": sha256_stream(root / CORPUS_TRAIN),
            "benchmark_records.jsonl": sha256_stream(root / BENCH_RECORDS),
        },
        "parquet_sha256": {
            TRAIN_PARQUET: sha256_stream(out / TRAIN_PARQUET),
            BENCH_PARQUET: sha256_stream(out / BENCH_PARQUET),
        },
        "columns": {"train": list(TRAIN_COLUMNS), "benchmark": [c for c in BENCH_COLUMNS]},
        "stripped_benchmark_fields": list(STRIPPED_BENCH_FIELDS),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    values = {
        "train_rows": len(train),
        "train_papers": manifest["papers"]["train"],
        "bench_rows": len(bench),
        "bench_papers": manifest["papers"]["benchmark"],
        "train_cc0": train_lic.get("CC0-1.0", 0),
        "bench_cc0": bench_lic.get("CC0-1.0", 0),
        "mined_records": mined,
    }
    card = render_template((root / CARD_TEMPLATE).read_text(encoding="utf-8"), values)
    for tok in PLACEHOLDERS.values():
        if tok in ("@@COPYRIGHT_HOLDER@@",):
            continue
        if tok not in card:
            raise StageError(f"rendered card lost placeholder {tok}")
    (out / "README.md").write_text(card, encoding="utf-8")

    write_sha256sums(out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    args = ap.parse_args(argv)
    try:
        m = stage(args.out)
    except StageError as e:
        print(f"stage_dataset: FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: m[k] for k in ("rows", "papers", "licenses")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
