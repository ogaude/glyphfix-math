"""Build page-level evaluation sets and turn page transcriptions into scorer rows.

Usage (from repo root):
  python scripts/corpus_pages.py build --side calib --papers ids.txt --sources data/sources \
      --shards data/corpus/shards --out data/corpus/pages_calib --n 50
  python scripts/corpus_pages.py rows --pages <out>/pages.jsonl --records <shards>/records \
      --preds page_preds.jsonl --out rows.jsonl --stats rows_stats.json
  python scripts/corpus_pages.py crop-scale --pages <out>/pages.jsonl --pdf-dir <out>/pdf \
      --records <shards>/records --images-root . --out crop_scale.json

build compiles each paper from its stored source (no arXiv request) in the sandbox, maps kept equations to pages
with SyncTeX, rasterizes the page with the most mapped equations and writes pages.jsonl, build_log.jsonl and
PAGES_SUMMARY.json. --side calib accepts only train/dev papers, --side benchmark only benchmark papers.
Exit codes: 0 ok, 2 refused or missing input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

from lightonocr_math import paths
from lightonocr_math.corpus import extract, freeze, pages
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json, write_jsonl
from lightonocr_math.corpus.records import side_of
from lightonocr_math.ids import base_id, safe_id

SIDE_FOR = {"benchmark": "benchmark", "calib": "corpus"}
QUERY_LINES = 6


def _read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def _queries(recs: list[dict]) -> list[tuple[str, int, str]]:
    out = []
    for r in recs:
        span = min(r["latex"].count("\n") + 3, QUERY_LINES)
        out.extend((r["id"], int(r["source_line"]) + k, r["source_file"]) for k in range(span))
    return out


def _union(boxes: list[pages.Box]) -> list[float]:
    x0 = min(b.x for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    base = max(b.y for b in boxes)
    top = min(b.y - b.h for b in boxes)
    return [round(x0, 2), round(base, 2), round(x1 - x0, 2), round(base - top, 2)]


def _build_paper(aid: str, sources: Path, shards: Path, out: Path, work_root: Path) -> tuple[dict | None, str]:
    sid = safe_id(aid)
    shard = shards / "records" / f"{sid}.jsonl"
    if not shard.exists():
        return None, "no_shard"
    recs = read_jsonl(shard)
    sdir = sources / sid
    try:
        fetch = json.loads((sdir / "fetch.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "no_source"
    workdir = work_root / sid
    shutil.rmtree(workdir, ignore_errors=True)
    try:
        try:
            extract.safe_extract(sdir / "src.bin", fetch["kind"], workdir)
        except (extract.UnsafeArchive, OSError, KeyError):
            return None, "bad_archive"
        main = extract.find_main(workdir)
        if main is None:
            return None, "no_main"
        main_rel = main.relative_to(workdir).as_posix()
        res = pages.compile_paper(workdir, main_rel)
        if res.status != "ok" or res.pdf is None:
            return None, f"compile_{res.status}"
        pdf_rel = res.pdf.relative_to(workdir).as_posix()
        boxes = pages.parse_synctex(pages.run_synctex(workdir, _queries(recs), pdf_rel))
        eq_pages = {r["id"]: pages.equation_pages(boxes.get(r["id"], [])) for r in recs}
        page = pages.choose_page(eq_pages)
        if page is None:
            return None, "few_eqs"
        img = pages.rasterize_page(res.pdf, page - 1)
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(res.pdf))
        try:
            n_pages = len(doc)
        finally:
            doc.close()
        page_id = f"{sid}_p{page - 1}"
        img_path = out / "images" / f"{page_id}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)
        (out / "pdf").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.pdf, out / "pdf" / f"{sid}.pdf")
        gt_ids = [r["id"] for r in recs if eq_pages[r["id"]] == page]
        row = {
            "page_id": page_id,
            "arxiv_id": recs[0]["arxiv_id"],
            "paper_id": base_id(aid),
            "paper_version": recs[0]["paper_version"],
            "license_uri": fetch.get("license_uri"),
            "license_spdx": fetch.get("license_spdx"),
            "page_index": page - 1,
            "n_pages": n_pages,
            "image": img_path.as_posix(),
            "width": img.width,
            "height": img.height,
            "image_sha256": sha256_file(img_path),
            "gt_ids": gt_ids,
            "gt_boxes": {i: _union(boxes[i]) for i in gt_ids},
            "page_version": pages.PAGE_VERSION,
            "compile_seconds": round(res.seconds, 2),
        }
        return row, ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cmd_build(a: argparse.Namespace) -> int:
    ids = _read_ids(Path(a.papers))
    want = SIDE_FOR[a.side]
    bad = [i for i in ids if side_of(i) != want]
    if bad:
        print(f"refusing: --side {a.side} needs {want} papers, got {len(bad)} from the other side (first: {bad[0]})", file=sys.stderr)
        return 2
    out = Path(a.out)
    if freeze.pages_dir_is_frozen(out):
        print(f"refusing: {out} is the frozen pages set (PAGES_FREEZE.json, pinned pages.jsonl or the repo benchmark pages dir); build elsewhere", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    work_root = paths.repo_root() / "data" / "work" / "pages"
    built: list[dict] = []
    log: list[dict] = []
    tried = 0
    for aid in ids:
        if len(built) >= a.n:
            break
        tried += 1
        row, reason = _build_paper(aid, Path(a.sources), Path(a.shards), out, work_root)
        if row is not None and len(row["gt_ids"]) < a.min_eqs:
            row, reason = None, "few_eqs"
        if row is None:
            log.append({"arxiv_id": aid, "status": "skipped", "reason": reason})
        else:
            built.append(row)
            log.append({"arxiv_id": aid, "status": "built", "page_id": row["page_id"], "n_gt": len(row["gt_ids"])})
    write_jsonl(out / "pages.jsonl", built)
    write_jsonl(out / "build_log.jsonl", log)
    write_json(
        out / "PAGES_SUMMARY.json",
        {
            "n_tried": tried,
            "n_built": len(built),
            "skip_reasons": dict(Counter(e["reason"] for e in log if e["status"] == "skipped")),
            "compile_seconds": round(sum(r["compile_seconds"] for r in built), 1),
            "page_version": pages.PAGE_VERSION,
            "side": a.side,
        },
    )
    print(f"built {len(built)} pages from {tried} papers -> {out}")
    return 0


def _records_by_id(records_dir: Path, arxiv_id: str) -> dict[str, dict]:
    p = records_dir / f"{safe_id(arxiv_id)}.jsonl"
    return {r["id"]: r for r in read_jsonl(p)} if p.exists() else {}


def cmd_rows(a: argparse.Namespace) -> int:
    page_rows = read_jsonl(a.pages)
    preds = {r["id"]: r for r in read_jsonl(a.preds)}
    rows: list[dict] = []
    per_page: list[dict] = []
    for pr in page_rows:
        by_id = _records_by_id(Path(a.records), pr["arxiv_id"])
        gt = [by_id[i] for i in pr["gt_ids"] if i in by_id]
        pred = preds.get(pr["page_id"])
        text = "" if pred is None or pred.get("pred_error") else (pred.get("pred_text") or "")
        out = pages.page_score_rows(pr, gt, text, a.threshold)
        rows.extend(out)
        n_m = sum(1 for r in out if r["matched"])
        per_page.append({"page_id": pr["page_id"], "n_gt": len(out), "n_matched": n_m, "match_rate": (n_m / len(out)) if out else None})
    n_gt = sum(p["n_gt"] for p in per_page)
    n_m = sum(p["n_matched"] for p in per_page)
    write_jsonl(a.out, rows)
    write_json(
        a.stats,
        {
            "n_gt": n_gt,
            "n_matched": n_m,
            "match_rate": (n_m / n_gt) if n_gt else None,
            "threshold": a.threshold,
            "page_match_version": pages.PAGE_MATCH_VERSION,
            "pages": per_page,
        },
    )
    print(f"{len(rows)} rows, match rate {n_m}/{n_gt}")
    return 0


def _pctl(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(int(q * (len(s) - 1) + 0.5), len(s) - 1)]


def cmd_crop_scale(a: argparse.Namespace) -> int:
    import pypdfium2 as pdfium

    ratios: list[float] = []
    detail: list[dict] = []
    for pr in read_jsonl(a.pages):
        by_id = _records_by_id(Path(a.records), pr["arxiv_id"])
        pdf = Path(a.pdf_dir) / f"{safe_id(pr['arxiv_id'])}.pdf"
        if not pdf.exists():
            continue
        img = pages.rasterize_page(pdf, pr["page_index"])
        doc = pdfium.PdfDocument(str(pdf))
        try:
            pg = doc[pr["page_index"]]
            size = pg.get_size()
            pg.close()
        finally:
            doc.close()
        for rid in pr["gt_ids"]:
            rec = by_id.get(rid)
            if rec is None or rec.get("env") != "equation" or "\\\\" in rec["latex"]:
                continue
            x, y, w, h = pr["gt_boxes"][rid]
            crop = Path(a.images_root) / rec["image"]
            if not crop.exists():
                continue
            m = pages.crop_scale(img, pages.Box(pr["page_index"] + 1, x, y, w, h), size, crop, pad_px=2)
            detail.append({"id": rid, "page_id": pr["page_id"], **m})
            if m["ratio"]:
                ratios.append(m["ratio"])
    summary = {
        "n": len(ratios),
        "median": statistics.median(ratios) if ratios else None,
        "p10": _pctl(ratios, 0.1) if ratios else None,
        "p90": _pctl(ratios, 0.9) if ratios else None,
        "crop_dpi": 300,
        "page_dpi": pages.PAGE_DPI,
        "page_longest_px": pages.PAGE_LONGEST_PX,
        "detail": detail,
    }
    write_json(a.out, summary)
    print(f"crop/page ink-height ratio: n={summary['n']} median={summary['median']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compile papers and build the page set")
    b.add_argument("--side", choices=sorted(SIDE_FOR), required=True)
    b.add_argument("--papers", required=True)
    b.add_argument("--sources", default="data/sources")
    b.add_argument("--shards", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--min-eqs", type=int, default=pages.MIN_GT_EQS)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("rows", help="page predictions to scorer rows")
    r.add_argument("--pages", required=True)
    r.add_argument("--records", required=True)
    r.add_argument("--preds", required=True)
    r.add_argument("--threshold", type=float, default=pages.MATCH_THRESHOLD)
    r.add_argument("--out", required=True)
    r.add_argument("--stats", required=True)
    r.set_defaults(fn=cmd_rows)

    c = sub.add_parser("crop-scale", help="crop vs page glyph height")
    c.add_argument("--pages", required=True)
    c.add_argument("--pdf-dir", required=True)
    c.add_argument("--records", required=True)
    c.add_argument("--images-root", default=".")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_crop_scale)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
