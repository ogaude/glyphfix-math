"""Per-paper corpus builder: fetched bundle -> records shard, excluded shard, paper summary and 300 DPI crops.

Usage:
  from lightonocr_math.corpus import build
  results = build.build_chunk([build.PaperInput(arxiv_id, version, catalog_row, fetch_meta, src_bin, work_dir)])
  for r in results: build.write_shards(r)
  build.shard_up_to_date(summary_path, src_sha256)      # resume key

Per-equation decision order (the first rule that fires gives the single exclusion reason):
  1. uses a macro the paper defines itself                          -> macros
  2. canonicalize(body, env); check_safe(canonical) is not None     -> unsafe   (detail = reason)
  3. fewer than MIN_COMPARABLE_CHARS non-space chars or more than MAX_LABEL_CHARS chars -> short_long
  4. canon.structural_problems(canonical) is not empty              -> canon_mismatch (cause "structural")
  5. canonical label already seen in this paper                     -> duplicate
  6. canonical label not rendered by the sandbox                    -> engine when the paper uses a different TeX
     engine package, else render (detail = status and note)
  7. the paper's own starred environment (wrapper "env") not rendered, or its bitmap differs from the canonical
     label rendered through the scorer wrapper                      -> canon_mismatch (cause "original-<status>" or
     "bitmap-differs", with env and multi_column)
  Survivors are kept: the crop is the label's ink bitmap padded with PAD_PX white pixels, written as a grayscale PNG.
  Static hints (macros, engine packages) are recorded for reporting; the render sandbox is the filter.

Shard contract (side_root = data/benchmark for benchmark papers, data/corpus for all others):
  <side_root>/shards/records/<safe_id>.jsonl   kept records (RECORD_KEYS + orig_bitmap_sha256)
  <side_root>/shards/excluded/<safe_id>.jsonl  {arxiv_id, reason, eq_index, source_file, source_line, env, detail}
  <side_root>/shards/papers/<safe_id>.json     paper summary (written last: a crash leaves the paper "not built")
  <side_root>/images/<safe_id>/<record_id>.png ; record "image" = "data/<corpus|benchmark>/images/<safe_id>/<id>.png"

Resume: a paper is rebuilt only when its source sha256, builder version, canon version or fingerprint, flag version or
preamble sha256 differ from its summary.

Known limit: a RenderInfrastructureError aborts the chunk and no shard of that chunk is written; the caller reruns.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from lightonocr_math import paths, render
from lightonocr_math.corpus import canon, extract, flags
from lightonocr_math.corpus._io import read_jsonl, sha256_bytes, write_json, write_jsonl
from lightonocr_math.corpus.records import BUILDER_VERSION, Exclusion, exclusion_row, side_of, side_root, validate_record
from lightonocr_math.ids import base_id, safe_id

PAD_PX = 10
MIN_COMPARABLE_CHARS = 4
MAX_LABEL_CHARS = 1500
RENDER_BATCH = 150
_ARCHIVE_KINDS = ("tar.gz", "tar", "gz")


@dataclass
class PaperInput:
    arxiv_id: str
    version: str
    catalog_row: dict
    fetch_meta: dict
    src_bin: Path
    work_dir: Path
    keep_work: bool = False


@dataclass
class PaperResult:
    arxiv_id: str
    side: str
    summary: dict
    records: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)


def _ex(arxiv_id: str, reason: Exclusion, *, eq_index: int | None = None, eq: extract.Equation | None = None, **detail: object) -> dict:
    return exclusion_row(
        arxiv_id, reason.value, eq_index=eq_index, source_file=eq.file if eq else None,
        source_line=eq.line if eq else None, env=eq.env if eq else None, detail=detail,
    )


def _multi_column(body: str, env: str) -> bool:
    if env != "align":
        return False
    import re

    for row in re.split(r"(?<!\\)\\\\", body):
        if len(re.findall(r"(?<!\\)&", row)) >= 2:
            return True
    return False


def _summary_base(p: PaperInput, side: str) -> dict:
    return {
        "arxiv_id": base_id(p.arxiv_id), "version": p.version, "side": side, "split": p.catalog_row.get("split"),
        "status": "built", "kind": p.fetch_meta.get("kind"), "src_sha256": p.fetch_meta.get("sha256", ""),
        "main_file": None, "n_files": 0, "n_extracted": 0, "n_kept": 0, "exclusions": {}, "engine_hints": [],
        "engine_flagged": False, "n_own_macros": 0, "redefined_core": [], "other_display_counts": {},
        "builder_version": BUILDER_VERSION, "extract_version": extract.EXTRACT_VERSION,
        "canon_version": canon.CANON_RULES_VERSION, "canon_fingerprint": canon.CANON_FINGERPRINT,
        "flag_rules_version": flags.FLAG_RULES_VERSION, "preamble_sha256": render.PREAMBLE_SHA256,
        "render_image_digest": render.IMAGE_DIGEST, "seconds": 0.0,
    }


def _count(excluded: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in excluded:
        out[r["reason"]] = out.get(r["reason"], 0) + 1
    return dict(sorted(out.items()))


def prepare_paper(p: PaperInput) -> PaperResult:
    """Host-only static work: extraction, own-macro and engine hints, pre-render filters."""
    t0 = time.monotonic()
    aid = base_id(p.arxiv_id)
    side = side_of(aid)
    res = PaperResult(aid, side, _summary_base(p, side))
    s = res.summary

    def paper_level(reason: Exclusion, status: str, **detail: object) -> PaperResult:
        s["status"] = status
        res.excluded.append(_ex(aid, reason, **detail))
        s["exclusions"] = _count(res.excluded)
        s["seconds"] = round(time.monotonic() - t0, 3)
        return res

    kind = p.fetch_meta.get("kind")
    if p.fetch_meta.get("status") != "ok" or kind not in _ARCHIVE_KINDS or not Path(p.src_bin).is_file():
        return paper_level(Exclusion.NO_SOURCE, "no_source", fetch_status=p.fetch_meta.get("status"), kind=kind)
    root = Path(p.work_dir) / "src"
    try:
        extract.safe_extract(Path(p.src_bin), kind, root)
    except extract.UnsafeArchive as e:
        return paper_level(Exclusion.BAD_ARCHIVE, "bad_archive", error=str(e))
    main = extract.find_main(root)
    if main is None:
        return paper_level(Exclusion.NO_MAIN, "no_main")
    files = extract.document_files(root, main)
    macros = extract.own_macros(extract.macro_files(root, files))
    hints = extract.engine_hints(files)
    eqs = extract.extract_display(files, root)
    s.update(
        main_file=str(main.relative_to(root)) if main.is_relative_to(root) else str(main), n_files=len(files),
        n_extracted=len(eqs), engine_hints=hints, engine_flagged=bool(hints), n_own_macros=len(macros),
        redefined_core=sorted(extract.redefined_core(files)), other_display_counts=extract.other_display_counts(files),
    )
    seen: set[str] = set()
    for i, eq in enumerate(eqs):
        if extract.uses_macro(eq.body, macros):
            res.excluded.append(_ex(aid, Exclusion.MACROS, eq_index=i, eq=eq))
            continue
        label = canon.canonicalize(eq.body, eq.env)
        unsafe = render.check_safe(label)
        if unsafe is not None:
            res.excluded.append(_ex(aid, Exclusion.UNSAFE, eq_index=i, eq=eq, why=unsafe))
            continue
        n_chars = len("".join(label.split()))
        if n_chars < MIN_COMPARABLE_CHARS or len(label) > MAX_LABEL_CHARS:
            res.excluded.append(_ex(aid, Exclusion.SHORT_LONG, eq_index=i, eq=eq, chars=len(label)))
            continue
        problems = canon.structural_problems(label)
        if problems:
            res.excluded.append(_ex(aid, Exclusion.CANON_MISMATCH, eq_index=i, eq=eq, cause="structural", problems=problems, multi_column=False))
            continue
        if label in seen:
            res.excluded.append(_ex(aid, Exclusion.DUPLICATE, eq_index=i, eq=eq))
            continue
        seen.add(label)
        res.pending.append({
            "eq_index": i, "eq": eq, "canonical": label, "original": canon.original_for_validation(eq.body, eq.env),
            "flags": flags.flags(label), "multi_column": _multi_column(eq.body, eq.env),
        })
    s["exclusions"] = _count(res.excluded)
    s["seconds"] = round(time.monotonic() - t0, 3)
    return res


def _crop_png(bitmap: np.ndarray) -> tuple[bytes, int, int]:
    padded = np.pad(np.asarray(bitmap, dtype=np.uint8), PAD_PX, mode="constant", constant_values=255)
    img = Image.fromarray(padded, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), padded.shape[1], padded.shape[0]


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _root(side: str, out_roots: dict[str, Path] | None) -> Path:
    if out_roots and side in out_roots:
        return Path(out_roots[side])
    return side_root(side)


def _finish(p: PaperInput, res: PaperResult, canon_res: dict, orig_res: dict, cache: Path, out_roots: dict[str, Path] | None) -> None:
    from lightonocr_math.corpus.records import record_id

    aid, s = res.arxiv_id, res.summary
    sid = safe_id(aid)
    img_dir = _root(res.side, out_roots) / "images" / sid
    if img_dir.exists():
        shutil.rmtree(img_dir)
    side_dir = "benchmark" if res.side == "benchmark" else "corpus"
    cat = p.catalog_row
    for item in res.pending:
        eq: extract.Equation = item["eq"]
        rc, ro = canon_res[item["canonical"]], orig_res[item["original"]]
        if rc.status != "rendered":
            reason = Exclusion.ENGINE if s["engine_flagged"] else Exclusion.RENDER
            res.excluded.append(_ex(aid, reason, eq_index=item["eq_index"], eq=eq, status=rc.status, note=rc.note))
            continue
        cause = None
        if ro.status != "rendered":
            cause = f"original-{ro.status}"
        elif not render.compare(rc, ro, cache)[0]:
            cause = "bitmap-differs"
        if cause is not None:
            res.excluded.append(_ex(aid, Exclusion.CANON_MISMATCH, eq_index=item["eq_index"], eq=eq, cause=cause,
                                    multi_column=item["multi_column"]))
            continue
        bitmap = render.load_bitmap(rc, cache)
        if bitmap is None:
            raise render.RenderInfrastructureError(f"bitmap of a rendered entry is unreadable: {rc.key}")
        png, w, h = _crop_png(bitmap)
        rid = record_id(aid, p.version, item["canonical"])
        _write_bytes_atomic(img_dir / f"{rid}.png", png)
        res.records.append({
            "id": rid, "arxiv_id": aid, "paper_version": p.version, "license_uri": cat.get("license_uri"),
            "license_spdx": cat.get("license_spdx"), "split": cat.get("split"), "env": eq.env, "source_file": eq.file,
            "source_line": eq.line, "raw_body": eq.body, "latex": item["canonical"], "flags": item["flags"],
            "image": f"data/{side_dir}/images/{sid}/{rid}.png", "width": w, "height": h,
            "bitmap_sha256": rc.bitmap_sha256, "image_sha256": sha256_bytes(png), "builder_version": BUILDER_VERSION,
            "canon_version": canon.CANON_RULES_VERSION, "preamble_sha256": render.PREAMBLE_SHA256,
            "render_image_digest": render.IMAGE_DIGEST, "engine_flagged": s["engine_flagged"],
            "orig_bitmap_sha256": ro.bitmap_sha256,
        })
        bad = validate_record(res.records[-1])
        if bad:
            raise ValueError(f"built an invalid record for {aid}: {bad}")
    res.excluded.sort(key=lambda r: (r.get("eq_index") is not None, r.get("eq_index") or 0))
    s["n_kept"] = len(res.records)
    s["exclusions"] = _count(res.excluded)
    res.pending = []


def build_chunk(
    papers: list[PaperInput], *, cache: Path | None = None, out_roots: dict[str, Path] | None = None,
    max_parallel: int | None = None,
) -> list[PaperResult]:
    """Prepare every paper, render all labels in two batched sandbox runs, decide, write crops. Shards are not written."""
    cache = Path(cache) if cache is not None else paths.cache_dir()
    t0 = time.monotonic()
    results: list[PaperResult] = []
    try:
        for p in papers:
            results.append(prepare_paper(p))
        canonicals = list(dict.fromkeys(i["canonical"] for r in results for i in r.pending))
        originals = list(dict.fromkeys(i["original"] for r in results for i in r.pending))
        canon_map: dict[str, render.RenderResult] = {}
        orig_map: dict[str, render.RenderResult] = {}
        if canonicals:
            canon_map = dict(zip(canonicals, render.render_many(
                canonicals, batch_size=RENDER_BATCH, cache=cache, max_parallel=max_parallel), strict=True))
            orig_map = dict(zip(originals, render.render_many(
                originals, batch_size=RENDER_BATCH, cache=cache, max_parallel=max_parallel, wrapper="env"), strict=True))
        t_render = (time.monotonic() - t0) / max(len(papers), 1)
        for p, r in zip(papers, results, strict=True):
            _finish(p, r, canon_map, orig_map, cache, out_roots)
            r.summary["seconds"] = round(r.summary["seconds"] + t_render, 3)
    finally:
        for p in papers:
            if not p.keep_work:
                shutil.rmtree(p.work_dir, ignore_errors=True)
    return results


def write_shards(result: PaperResult, out_roots: dict[str, Path] | None = None) -> None:
    """Records, excluded rows and finally the paper summary, atomically, under the paper's side root."""
    sid = safe_id(result.arxiv_id)
    shards = _root(result.side, out_roots) / "shards"
    write_jsonl(shards / "records" / f"{sid}.jsonl", result.records)
    write_jsonl(shards / "excluded" / f"{sid}.jsonl", result.excluded)
    write_json(shards / "papers" / f"{sid}.json", result.summary)


def shard_up_to_date(summary_path: Path, src_sha256: str) -> bool:
    import json

    try:
        s = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        s.get("src_sha256") == src_sha256 and s.get("builder_version") == BUILDER_VERSION
        and s.get("canon_version") == canon.CANON_RULES_VERSION and s.get("canon_fingerprint") == canon.CANON_FINGERPRINT
        and s.get("flag_rules_version") == flags.FLAG_RULES_VERSION and s.get("preamble_sha256") == render.PREAMBLE_SHA256
    )


def load_records(shard_root: Path, sid: str) -> list[dict]:
    p = Path(shard_root) / "records" / f"{sid}.jsonl"
    return read_jsonl(p) if p.exists() else []
