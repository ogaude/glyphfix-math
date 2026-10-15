"""Page-level evaluation machinery: compile a paper from its own source, map equations to pages, score page output.

Usage:
  from lightonocr_math.corpus import pages
  res = pages.compile_paper(workdir, "main.tex")                       # sandboxed latexmk, no arXiv request
  raw = pages.run_synctex(workdir, [(rid, line, "main.tex")], "main.pdf")
  eq_pages = {qid: pages.equation_pages(b) for qid, b in pages.parse_synctex(raw).items()}
  page = pages.choose_page(eq_pages)                                   # 1-based, None when < MIN_GT_EQS
  img = pages.rasterize_page(res.pdf, page - 1)                        # 400 DPI, then longest side 2048 px
  rows = pages.page_score_rows(page_row, gt_records, model_text)       # rows for lightonocr_math.score

Rules:
  * the paper's own LaTeX source is compiled in the pinned texlive container with the same hardening as the
    render sandbox (no network, read-only root, capabilities dropped, memory cap) and a host timeout; the pdflatex
    call carries -no-shell-escape (D-01)
  * equations whose SyncTeX boxes span two pages are left out of that page's ground truth
  * the page raster follows the model card: 400 DPI, then thumbnail to 2048 px on the longest side (D-02)
  * page output is scored by one-to-one equation matching on normalized token similarity; unmatched ground truth
    equations score as misses and the match rate is reported on its own (D-03)

Known limit: two-column layouts and figure-heavy pages break the reading-order assumption of display-block
extraction and lower the match rate; MATCH_THRESHOLD was calibrated on train/dev pages before
any benchmark page was scored.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightonocr_math import guard, render
from lightonocr_math.normalize import normalize_tokens

PAGE_VERSION = "1"
PAGE_MATCH_VERSION = "2"
PAGE_DPI = 400
PAGE_LONGEST_PX = 2048
MATCH_THRESHOLD = 0.5
MIN_GT_EQS = 3
COMPILE_TIMEOUT_S = 300
SYNCTEX_TIMEOUT_S = 300
LOG_TAIL_CHARS = 1500


@dataclass(frozen=True)
class Box:
    page: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class CompileResult:
    status: str  # "ok" | "failed" | "timeout" | "infra"
    pdf: Path | None
    log_tail: str
    seconds: float


# ---------------------------------------------------------------- sandbox


def _hardened_prefix(name: str, workdir: Path) -> list[str]:
    """docker run flags identical to render._docker_cmd up to and including the image."""
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--memory", guard.DOCKER_MEMORY, "--memory-swap", guard.DOCKER_MEMORY,
        "--cpus", "2", "--pids-limit", "512",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "TEXMFVAR=/tmp/texmf-var",
        "-v", f"{workdir}:/work", "-w", "/work",
        render.IMAGE,
    ]


def latexmk_docker_cmd(name: str, workdir: Path, main_rel: str) -> list[str]:
    return [
        *_hardened_prefix(name, Path(workdir)),
        "latexmk", "-pdf", "-synctex=1", "-interaction=nonstopmode", "-halt-on-error",
        "-pdflatex=pdflatex -no-shell-escape %O %S",
        main_rel,
    ]


def _first_error(log: str) -> str:
    for line in log.splitlines():
        if line.startswith("!"):
            return line[:160]
    return ""


def compile_paper(workdir: Path, main_rel: str, *, timeout_s: int = COMPILE_TIMEOUT_S) -> CompileResult:
    workdir = Path(workdir).resolve()
    guard.preflight(ram_gb=guard.DOCKER_MEMORY_GB, timeout_s=600)
    name = f"lom-pages-{uuid.uuid4().hex}"
    pdf = workdir / (Path(main_rel).stem + ".pdf")  # latexmk writes outputs to the working directory
    pdf.unlink(missing_ok=True)  # a stale or truncated PDF from an earlier attempt must never count as output
    pdf.with_suffix(".synctex.gz").unlink(missing_ok=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            latexmk_docker_cmd(name, workdir, main_rel), capture_output=True, text=True, errors="replace", timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        return CompileResult("timeout", None, "", time.monotonic() - t0)
    except OSError as e:
        return CompileResult("infra", None, f"{type(e).__name__}: {e}", time.monotonic() - t0)
    secs = time.monotonic() - t0
    if proc.returncode in render.INFRA_EXIT_CODES:
        return CompileResult("infra", None, (proc.stderr or "")[-LOG_TAIL_CHARS:], secs)
    log_path = pdf.with_suffix(".log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else (proc.stdout or "")
    if proc.returncode != 0 or not pdf.exists():
        pdf.unlink(missing_ok=True)  # -halt-on-error leaves a truncated PDF of the pages before the error
        return CompileResult("failed", None, _first_error(log) or log[-LOG_TAIL_CHARS:], secs)
    return CompileResult("ok", pdf, log[-LOG_TAIL_CHARS:], secs)


# ---------------------------------------------------------------- SyncTeX


def synctex_script(queries: list[tuple[str, int, str]], pdf_rel: str) -> str:
    lines = ["#!/bin/sh"]
    for qid, line, file in queries:
        lines.append(f"echo {shlex.quote('##Q ' + qid)}")
        lines.append(f"synctex view -i {shlex.quote(f'{int(line)}:0:{file}')} -o {shlex.quote(pdf_rel)} 2>/dev/null")
    return "\n".join(lines) + "\n"


def run_synctex(workdir: Path, queries: list[tuple[str, int, str]], pdf_rel: str) -> str:
    workdir = Path(workdir).resolve()
    (workdir / "_synctex_q.sh").write_text(synctex_script(queries, pdf_rel), encoding="utf-8")
    name = f"lom-synctex-{uuid.uuid4().hex}"
    cmd = [*_hardened_prefix(name, workdir), "sh", "_synctex_q.sh"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=SYNCTEX_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        return ""
    return proc.stdout or ""


def parse_synctex(raw: str) -> dict[str, list[Box]]:
    """Boxes per query id; x,y is the box origin (y = baseline from the page top), the box extends up by h."""
    out: dict[str, list[Box]] = {}
    cur_id: str | None = None
    cur: dict[str, float] = {}

    def flush() -> None:
        if cur_id is not None and {"Page", "x", "y", "W", "H"} <= cur.keys():
            out[cur_id].append(Box(int(cur["Page"]), cur["x"], cur["y"], cur["W"], cur["H"]))
        cur.clear()

    for ln in raw.splitlines():
        if ln.startswith("##Q "):
            flush()
            cur_id = ln[4:].strip()
            out.setdefault(cur_id, [])
        elif ln.startswith("SyncTeX result begin"):
            cur.clear()
        elif ln.startswith("SyncTeX result end"):
            flush()
        elif ":" in ln:
            k, _, v = ln.partition(":")
            if k in ("Page", "x", "y", "W", "H"):
                try:
                    cur[k] = float(v)
                except ValueError:
                    pass
    flush()
    return out


def equation_pages(boxes: list[Box]) -> int | None:
    pages = {b.page for b in boxes}
    return next(iter(pages)) if len(pages) == 1 else None


def choose_page(eq_pages: dict[str, int | None], min_eqs: int = MIN_GT_EQS) -> int | None:
    counts: dict[int, int] = {}
    for p in eq_pages.values():
        if p is not None:
            counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best = min(counts, key=lambda p: (-counts[p], p))
    return best if counts[best] >= min_eqs else None


# ---------------------------------------------------------------- raster


def rasterize_page(pdf_path: Path, page_index: int):
    """RGB PIL image: 400 DPI render, then longest side shrunk to 2048 px (model card recipe)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        try:
            img = page.render(scale=PAGE_DPI / 72).to_pil().convert("RGB")
        finally:
            page.close()
    finally:
        doc.close()
    img.thumbnail((PAGE_LONGEST_PX, PAGE_LONGEST_PX))
    return img


# ---------------------------------------------------------------- model output

_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "aligned")
_ENV_ALT = "|".join(re.escape(e) + r"\*?" for e in _ENVS)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_RE = re.compile(
    r"(?<!\\)\$\$(?P<dd>.+?)(?<!\\)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    rf"|\\begin\{{(?P<env>{_ENV_ALT})\}}(?P<body>.*?)\\end\{{(?P=env)\}}",
    re.DOTALL,
)


def extract_display_blocks(text: str) -> list[str]:
    text = _FENCE_RE.sub(" ", text)
    out: list[str] = []
    for m in _DISPLAY_RE.finditer(text):
        if m.group("dd") is not None:
            block = m.group("dd")
        elif m.group("br") is not None:
            block = m.group("br")
        else:
            block = m.group(0)
        block = block.strip()
        if block:
            out.append(block)
    return out


def match_equations(
    gt_latex: list[str], pred_blocks: list[str], threshold: float = MATCH_THRESHOLD
) -> list[tuple[int, int | None, float]]:
    """One (gt_index, pred_index or None, similarity) per GT equation, one-to-one."""
    if not gt_latex:
        return []
    if not pred_blocks:
        return [(i, None, 0.0) for i in range(len(gt_latex))]
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    from scipy.optimize import linear_sum_assignment

    gt_tok = [normalize_tokens(g) for g in gt_latex]
    pr_tok = [normalize_tokens(p) for p in pred_blocks]
    sim = np.asarray(cdist(gt_tok, pr_tok, scorer=Levenshtein.normalized_similarity), dtype=float)
    rows, cols = linear_sum_assignment(-sim)
    assigned = {int(r): (int(c), float(sim[r, c])) for r, c in zip(rows, cols, strict=True)}
    result: list[tuple[int, int | None, float]] = []
    for i in range(len(gt_latex)):
        c, s = assigned.get(i, (None, 0.0))
        if c is not None and s >= threshold:
            result.append((i, c, s))
        else:
            result.append((i, None, s if c is not None else 0.0))
    return result


def page_score_rows(
    page_row: dict, gt_records: list[dict], pred_text: str, threshold: float = MATCH_THRESHOLD
) -> list[dict]:
    blocks = extract_display_blocks(pred_text or "")
    matches = match_equations([r["latex"] for r in gt_records], blocks, threshold)
    rows = []
    for (i, j, sim), rec in zip(matches, gt_records, strict=True):
        rows.append(
            {
                "id": rec["id"],
                "paper_id": page_row.get("paper_id") or page_row["arxiv_id"],
                "gt_latex": rec["latex"],
                "pred_latex": blocks[j] if j is not None else "",
                "match_sim": round(sim, 4),
                "matched": j is not None,
                "page_id": page_row["page_id"],
            }
        )
    return rows


# ---------------------------------------------------------------- crop scale


def _ink_height(arr: np.ndarray) -> int | None:
    ink = render.ink_crop(arr)
    return None if ink is None else int(ink.shape[0])


def crop_scale(page_img, box: Box, pdf_page_size_pt: tuple[float, float], crop_png_path: Path, pad_px: int = 0) -> dict:
    """Ink height of an equation in its 300 DPI crop vs the same equation on the 2048 px page raster.

    box is a SyncTeX box in PDF points measured from the top-left of the page.
    """
    from PIL import Image

    crop = np.asarray(Image.open(crop_png_path).convert("L"))
    crop_ink = _ink_height(crop)
    pw, ph = pdf_page_size_pt
    sx, sy = page_img.width / pw, page_img.height / ph
    x0, x1 = max(int(box.x * sx) - pad_px, 0), min(int((box.x + box.w) * sx) + pad_px, page_img.width)
    y0, y1 = max(int((box.y - box.h) * sy) - pad_px, 0), min(int((box.y + 0.35 * box.h) * sy) + pad_px, page_img.height)
    region = np.asarray(page_img.convert("L"))[y0:y1, x0:x1]
    page_ink = _ink_height(region) if region.size else None
    ratio = (crop_ink / page_ink) if crop_ink and page_ink else None
    return {"crop_ink_px": crop_ink, "page_ink_px": page_ink, "ratio": ratio}
