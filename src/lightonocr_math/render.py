"""Render-equivalence metric: do two LaTeX strings produce the same ink?

Usage:
  from lightonocr_math.render import render_equivalence, render_config
  pairs = render_equivalence(gt_list, pred_list)      # list[PairResult]
  pairs[0].identical      # True / False, None when the ground truth itself does not compile
  render_config()         # preamble sha256, wrap version, dpi, tolerance: store in every result file

Behaviour:
  Ground truth and prediction go through one fixed preamble (preamble.tex) and one wrap()
  function, are compiled with pdflatex in batches inside a Docker sandbox, rasterized at
  300 DPI with pypdfium2 and cropped to the ink bounding box. "identical" means exact bitmap
  equality after the crop. A separate "tolerant" flag allows a few pixels of shape and
  intensity noise; it is reported on its own and never merged into the identical rate.
  A prediction that does not compile (or is rejected, empty, blank or overflows the page)
  scores identical=False and tolerant=False. Results are cached by
  sha256(preamble sha256, wrap version, dpi, latex).

Security model:
  Equations are untrusted. They are only ever written into batch.tex, never into a shell
  string or argv. Inputs containing file-I/O, catcode, definition or page-control
  primitives are rejected before rendering (check_safe). pdflatex runs in
  texlive/texlive:latest with no network, a memory cap, a pid cap, dropped capabilities,
  a read-only root file system, a non-root user, -no-shell-escape and a timeout.

Known limit: paper-specific macros are not available. A ground truth that uses them does not
compile and is reported as gt_uncompilable (identical=None) and excluded from the rate.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightonocr_math.paths import cache_dir

IMAGE = "texlive/texlive:latest"
RENDER_DPI = 300
INK_THRESHOLD = 250
WRAP_VERSION = "1"
TOL_SHAPE_PX = 2
TOL_PIXEL_DELTA = 64
TOL_MAX_FRACTION = 0.002
OVERFLOW_MARGIN_PX = 5

STATUSES = ("rendered", "compile-failed", "blank-page", "rejected", "overflow", "timeout", "empty")

_PREAMBLE_BYTES = (importlib.resources.files("lightonocr_math") / "preamble.tex").read_bytes()
PREAMBLE = _PREAMBLE_BYTES.decode("utf-8")
PREAMBLE_SHA256 = hashlib.sha256(_PREAMBLE_BYTES).hexdigest()


def render_config() -> dict:
    return {
        "image": IMAGE,
        "preamble_sha256": PREAMBLE_SHA256,
        "wrap_version": WRAP_VERSION,
        "dpi": RENDER_DPI,
        "ink_threshold": INK_THRESHOLD,
        "tolerance": {
            "shape_px": TOL_SHAPE_PX,
            "pixel_delta": TOL_PIXEL_DELTA,
            "max_fraction": TOL_MAX_FRACTION,
        },
    }


# --------------------------------------------------------------------------- safety filter

_FORBIDDEN_WORDS = ["input", "include", "openin", "openout", "read", "readline", "write", "immediate", "catcode", "lccode", "uccode", "def", "edef", "gdef", "xdef", "let", "futurelet", "global", "outer", "newcommand", "renewcommand", "providecommand", "DeclareMathOperator", "DeclareRobustCommand", "newenvironment", "renewenvironment", "usepackage", "RequirePackage", "documentclass", "newpage", "clearpage", "cleardoublepage", "pagebreak", "shipout", "special", "csname", "endcsname", "makeatletter", "makeatother", "jobname", "endinput", "vspace", "vskip", "pdfprimitive", "directlua", "scantokens", "everymath", "everydisplay", "afterassignment", "expandafter"]
_FORBIDDEN_RE = re.compile(r"\\(?:" + "|".join(_FORBIDDEN_WORDS) + r")(?![A-Za-z])")
_DOCUMENT_RE = re.compile(r"\\(?:begin|end)\s*\{\s*document\s*\}")
_CONTROL_SYMBOL_RE = re.compile(r"\\[^A-Za-z]")


def check_safe(latex: str) -> str | None:
    """Reason the input must not be rendered, or None. Blank input returns "empty"."""
    if not latex or not latex.strip():
        return "empty"
    if "^^" in latex:
        return "TeX character escape ^^"
    if _DOCUMENT_RE.search(latex):
        return "document environment"
    # Control symbols (\\ \% \{ ...) are removed first so "\\%" is seen as a line break plus a comment.
    plain = _CONTROL_SYMBOL_RE.sub(" ", latex)
    if "%" in plain:
        return "unescaped %"
    m = _FORBIDDEN_RE.search(plain)
    if m:
        return f"forbidden primitive {m.group(0)}"
    return None


# --------------------------------------------------------------------------- wrap

_ENV_NAMES = r"(?:equation|align|alignat|flalign|gather|multline|eqnarray|displaymath|math)"
_ENV_RE = re.compile(r"^\\begin\{(" + _ENV_NAMES + r")(\*?)\}(.*)\\end\{\1\2\}$", re.DOTALL)
_LAYOUT_RES = [
    re.compile(r"\\label\{[^{}]*\}"),
    re.compile(r"\\tag\*?\{[^{}]*\}"),
    re.compile(r"\\(?:nonumber|notag)(?![A-Za-z])"),
]
_INNER_ENV_RE = re.compile(r"\\begin\{([^{}]*)\}(?:(?!\\begin\{).)*?\\end\{\1\}", re.DOTALL)
_BRACE_GROUP_RE = re.compile(r"\{[^{}]*\}")
_ALIGN_RE = re.compile(r"(?<!\\)&|\\\\")


def _strip_outer(s: str) -> str:
    for _ in range(6):
        t = s.strip()
        if t.startswith("$$") and t.endswith("$$") and len(t) >= 4 or t.startswith(r"\[") and t.endswith(r"\]") or t.startswith(r"\(") and t.endswith(r"\)"):
            s = t[2:-2]
        elif t.startswith("$") and t.endswith("$") and len(t) >= 2 and not t.startswith("$$"):
            s = t[1:-1]
        else:
            m = _ENV_RE.match(t)
            if not m:
                return t
            s = m.group(3)
    return s.strip()


def _needs_aligned(body: str) -> bool:
    flat = body
    prev = None
    while prev != flat:
        prev = flat
        flat = _INNER_ENV_RE.sub("", flat)
    prev = None
    while prev != flat:
        prev = flat
        flat = _BRACE_GROUP_RE.sub("", flat)
    return bool(_ALIGN_RE.search(flat))


def wrap(latex: str) -> str:
    """Display body placed in the batch document. Identical for ground truth and prediction."""
    s = latex
    for rx in _LAYOUT_RES:
        s = rx.sub("", s)
    body = _strip_outer(s)
    if _needs_aligned(body):
        body = "\\begin{aligned}\n" + body + "\n\\end{aligned}"
    return "\\[\n" + body + "\n\\]"


# --------------------------------------------------------------------------- bitmaps

def _ink_bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(arr < INK_THRESHOLD)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def ink_crop(arr: np.ndarray) -> np.ndarray | None:
    """The ink bounding box of a grayscale page, or None for a blank page."""
    box = _ink_bbox(arr)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    return arr[y0:y1, x0:x1]


def ink_overflows(arr: np.ndarray) -> bool:
    """True when ink touches the page edge, so the page may have clipped the equation."""
    box = _ink_bbox(arr)
    if box is None:
        return False
    y0, y1, x0, x1 = box
    h, w = arr.shape
    m = OVERFLOW_MARGIN_PX
    return y0 < m or x0 < m or h - y1 < m or w - x1 < m


def compare_arrays(a: np.ndarray | None, b: np.ndarray | None) -> tuple[bool, bool]:
    """(identical, tolerant). Identical is exact equality; tolerant is reported separately (D-10)."""
    if a is None or b is None:
        return False, False
    if a.shape == b.shape and np.array_equal(a, b):
        return True, True
    if abs(a.shape[0] - b.shape[0]) > TOL_SHAPE_PX or abs(a.shape[1] - b.shape[1]) > TOL_SHAPE_PX:
        return False, False
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.full((h, w), 255, dtype=np.uint8)
    pb = np.full((h, w), 255, dtype=np.uint8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    diff = np.abs(pa.astype(np.int16) - pb.astype(np.int16)) > TOL_PIXEL_DELTA
    return False, bool(diff.mean() <= TOL_MAX_FRACTION)


# --------------------------------------------------------------------------- cache

@dataclass
class RenderResult:
    key: str
    status: str
    shape: tuple[int, int] | None
    bitmap_sha256: str | None
    note: str = ""


def cache_key(latex: str) -> str:
    payload = PREAMBLE_SHA256 + "\n" + WRAP_VERSION + "\n" + str(RENDER_DPI) + "\n" + latex
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_root(cache: Path | None) -> Path:
    return (Path(cache) if cache is not None else cache_dir()) / "render"


def _entry(cache: Path | None, key: str, suffix: str) -> Path:
    return _cache_root(cache) / key[:2] / f"{key}{suffix}"


def _cache_get(cache: Path | None, key: str) -> RenderResult | None:
    p = _entry(cache, key, ".json")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shape = tuple(d["shape"]) if d.get("shape") else None
    return RenderResult(key, d["status"], shape, d.get("bitmap_sha256"), d.get("note", ""))


def _cache_put(cache: Path | None, res: RenderResult, bitmap: np.ndarray | None = None) -> None:
    jp = _entry(cache, res.key, ".json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    if bitmap is not None:
        tmp = jp.with_suffix(".npy.tmp")
        with tmp.open("wb") as fh:
            np.save(fh, bitmap)
        os.replace(tmp, _entry(cache, res.key, ".npy"))
    payload = {"status": res.status, "shape": list(res.shape) if res.shape else None, "bitmap_sha256": res.bitmap_sha256, "note": res.note}
    tmpj = jp.with_suffix(".json.tmp")
    tmpj.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmpj, jp)  # json last: a partial write is a cache miss


def load_bitmap(result: RenderResult, cache: Path | None = None) -> np.ndarray | None:
    if result.status != "rendered":
        return None
    try:
        return np.load(_entry(cache, result.key, ".npy"))
    except (OSError, ValueError):
        return None


def compare(a: RenderResult, b: RenderResult, cache: Path | None = None) -> tuple[bool, bool]:
    if a.status != "rendered" or b.status != "rendered":
        return False, False
    return compare_arrays(load_bitmap(a, cache), load_bitmap(b, cache))
