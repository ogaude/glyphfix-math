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
  equality after the crop. A separate "tolerant" flag allows only isolated anti-alias noise
  (no differing region above TOL_MAX_COMPONENT_PX pixels, at most TOL_MAX_DIFF_PX differing
  pixels in total, best alignment within TOL_SHAPE_PX). It is a diagnostic tier reported on
  its own, never merged into the identical rate, and it must never be used as an acceptance
  gate; render identical is the co-primary metric.
  A prediction that does not compile (or is rejected, empty, blank or overflows the page)
  scores identical=False and tolerant=False.

Cache policy:
  Results are cached by sha256(preamble sha256, wrap version, dpi, latex). Rendered, blank-page,
  overflow, rejected, empty and pdflatex "!" compile failures are cached. Timeouts and failures
  not attributable to a LaTeX error are returned for the current call but never cached.
  Infrastructure failures (Docker unreachable, container killed, no pdflatex log) raise
  RenderInfrastructureError and leave the cache untouched.

Security model:
  Equations are untrusted. They are only ever written into batch.tex, never into a shell
  string or argv. Inputs containing file-I/O, catcode, definition or page-control
  primitives are rejected before rendering (check_safe). pdflatex runs in
  the pinned texlive/texlive image (by digest) with no network, a memory cap, a pid cap, dropped capabilities,
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
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightonocr_math import guard
from lightonocr_math.paths import cache_dir

# Pinned by digest, not the floating :latest tag: a silent image update would change rendering mid-project and
# make cached bitmaps incomparable to new ones. The digest is part of cache_key(), so bumping it re-renders.
IMAGE_DIGEST = "sha256:0f4884ca7d287f47ab466d2dd405d4c901c576585363f628863a7ca7eea21301"
IMAGE = f"texlive/texlive@{IMAGE_DIGEST}"
RENDER_DPI = 300
INK_THRESHOLD = 250
WRAP_VERSION = "1"
TOL_SHAPE_PX = 2
TOL_PIXEL_DELTA = 64
TOLERANCE_VERSION = "2"
TOL_MAX_COMPONENT_PX = 4
TOL_MAX_DIFF_PX = 24
OVERFLOW_MARGIN_PX = 5

# docker run errors, command not executable / not found, SIGKILL (OOM), SIGSEGV
INFRA_EXIT_CODES = frozenset({125, 126, 127, 137, 139})

STATUSES = ("rendered", "compile-failed", "blank-page", "rejected", "overflow", "timeout", "empty")

_PREAMBLE_BYTES = (importlib.resources.files("lightonocr_math") / "preamble.tex").read_bytes()
PREAMBLE = _PREAMBLE_BYTES.decode("utf-8")
PREAMBLE_SHA256 = hashlib.sha256(_PREAMBLE_BYTES).hexdigest()


class RenderInfrastructureError(RuntimeError):
    """Docker or the container could not produce a pdflatex verdict.

    Never cached: the caller must fix the infrastructure and rerun.
    """


def render_config() -> dict:
    return {
        "image": IMAGE,
        "preamble_sha256": PREAMBLE_SHA256,
        "wrap_version": WRAP_VERSION,
        "dpi": RENDER_DPI,
        "ink_threshold": INK_THRESHOLD,
        "tolerance": {
            "version": TOLERANCE_VERSION,
            "shape_px": TOL_SHAPE_PX,
            "pixel_delta": TOL_PIXEL_DELTA,
            "max_component_px": TOL_MAX_COMPONENT_PX,
            "max_diff_px": TOL_MAX_DIFF_PX,
        },
    }


# --------------------------------------------------------------------------- safety filter

_FORBIDDEN_WORDS = ["input", "include", "openin", "openout", "read", "readline", "write", "immediate", "catcode", "lccode", "uccode", "def", "edef", "gdef", "xdef", "let", "futurelet", "global", "outer", "newcommand", "renewcommand", "providecommand", "DeclareMathOperator", "DeclareRobustCommand", "newenvironment", "renewenvironment", "usepackage", "RequirePackage", "documentclass", "newpage", "clearpage", "cleardoublepage", "pagebreak", "shipout", "special", "csname", "endcsname", "makeatletter", "makeatother", "jobname", "endinput", "vspace", "vskip", "pdfprimitive", "directlua", "scantokens", "everymath", "everydisplay", "afterassignment", "expandafter",
    # global-state, file and counter primitives: one prediction must not change how later items in the batch render
    "NewDocumentCommand", "RenewDocumentCommand", "ProvideDocumentCommand", "DeclareDocumentCommand", "NewDocumentEnvironment", "RenewDocumentEnvironment", "NewCommandCopy", "RenewCommandCopy", "DeclareCommandCopy",
    "ExplSyntaxOn", "ExplSyntaxOff", "newcount", "newdimen", "newskip", "newtoks", "newbox", "newif", "newlength", "newcounter", "newtheorem", "newlinechar",
    "setcounter", "addtocounter", "stepcounter", "refstepcounter", "setlength", "addtolength", "setbox", "loop", "repeat",
    "mathchardef", "chardef", "mathcode", "delcode", "sfcode", "DeclareMathSymbol", "DeclareSymbolFont", "DeclareMathAlphabet", "SetMathAlphabet", "DeclareMathAccent", "DeclareMathDelimiter", "DeclareMathRadical", "DeclareFontFamily", "DeclareFontShape",
    "AtEndDocument", "AtBeginDocument", "InputIfFileExists", "IfFileExists", "includegraphics", "includepdf", "pdfximage", "pdfliteral", "pdfobj", "pdfannot", "pdfstartlink", "pdfinfo"]
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
    from scipy import ndimage

    pad = TOL_SHAPE_PX
    h, w = max(a.shape[0], b.shape[0]) + 2 * pad, max(a.shape[1], b.shape[1]) + 2 * pad
    pa = np.full((h, w), 255, dtype=np.uint8)
    pa[pad : pad + a.shape[0], pad : pad + a.shape[1]] = a
    pa16 = pa.astype(np.int16)
    best: np.ndarray | None = None
    best_count = -1
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            pb = np.full((h, w), 255, dtype=np.uint8)
            pb[pad + dy : pad + dy + b.shape[0], pad + dx : pad + dx + b.shape[1]] = b
            mask = np.abs(pa16 - pb.astype(np.int16)) > TOL_PIXEL_DELTA
            count = int(mask.sum())
            if best is None or count < best_count:
                best, best_count = mask, count
    assert best is not None
    if best_count > TOL_MAX_DIFF_PX:
        return False, False
    labels, n = ndimage.label(best, structure=np.ones((3, 3), dtype=bool))
    largest = int(np.bincount(labels.ravel())[1:].max()) if n else 0
    return False, largest <= TOL_MAX_COMPONENT_PX


# --------------------------------------------------------------------------- cache

@dataclass
class RenderResult:
    key: str
    status: str
    shape: tuple[int, int] | None
    bitmap_sha256: str | None
    note: str = ""


def cache_key(latex: str) -> str:
    payload = IMAGE_DIGEST + "\n" + PREAMBLE_SHA256 + "\n" + WRAP_VERSION + "\n" + str(RENDER_DPI) + "\n" + latex
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
    try:
        status = d.get("status")
        # entries poisoned by pre-fix code (infrastructure failures cached as LaTeX failures) are re-rendered
        if status == "timeout" or (status == "compile-failed" and not str(d.get("note", "")).startswith("!")):
            return None
        # a "rendered" entry whose bitmap was lost is a miss (re-render), not a silent non-match
        if status == "rendered" and not _entry(cache, key, ".npy").is_file():
            return None
        shape = tuple(d["shape"]) if d.get("shape") else None
        return RenderResult(key, d["status"], shape, d.get("bitmap_sha256"), d.get("note", ""))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


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
    except (OSError, ValueError, EOFError):
        return None


def compare(a: RenderResult, b: RenderResult, cache: Path | None = None) -> tuple[bool, bool]:
    if a.status != "rendered" or b.status != "rendered":
        return False, False
    ba, bb = load_bitmap(a, cache), load_bitmap(b, cache)
    if ba is None or bb is None:
        # a rendered entry whose bitmap is unreadable must not score as "not identical"
        raise RenderInfrastructureError("cached bitmap missing or unreadable for a rendered entry; delete the render cache entry and rerun")
    return compare_arrays(ba, bb)


# --------------------------------------------------------------------------- Docker rendering

def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def docker_diagnosis() -> str:
    """Why docker_available() failed: daemon unreachable, or image not pulled yet. Never raises."""
    try:
        r = subprocess.run(["docker", "info"], check=False, capture_output=True, timeout=20)
        if r.returncode != 0:
            return "docker daemon unreachable (`docker info` failed): start Docker"
        return f"image not present locally: run `docker pull {IMAGE}`"
    except Exception:  # noqa: BLE001 - diagnosis only
        return f"docker unreachable or slow; if the daemon is up, run `docker pull {IMAGE}`"


def _batch_source(latexes: Sequence[str]) -> str:
    parts = [PREAMBLE, "\\begin{document}\n"]
    for latex in latexes:
        parts.append(wrap(latex) + "\n\\newpage\n")
    parts.append("\\end{document}\n")
    return "".join(parts)


def _docker_cmd(name: str, workdir: Path) -> list[str]:
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
        IMAGE,
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "batch.tex",
    ]


def _first_error(log: str) -> str:
    for line in log.splitlines():
        if line.startswith("!"):
            return line[:120]
    return ""


def _infra_error(workdir: Path, what: str, proc: subprocess.CompletedProcess) -> RenderInfrastructureError:
    shutil.rmtree(workdir, ignore_errors=True)
    return RenderInfrastructureError(f"{what} (exit {proc.returncode}): {(proc.stderr or '')[:300]}")


_START_LOCK = threading.Lock()
_inflight = [0]  # containers started by this process and not yet finished (guarded by _START_LOCK)


def _run_batch(latexes: Sequence[str], work_root: Path, timeout_s: int) -> tuple[Path | None, str, bool]:
    """Compile one batch. Returns (workdir with batch.pdf or None, first error line, timed_out).

    Only outcomes attributable to LaTeX are returned; infrastructure failures raise RenderInfrastructureError.
    """
    with _START_LOCK:
        # Serialized: the check and the reservation are one step, and containers already running are counted at their
        # full cap (conservative), so N workers cannot all pass the MemAvailable check at the same moment.
        guard.preflight(ram_gb=guard.DOCKER_MEMORY_GB, workers=_inflight[0] + 1, timeout_s=600)
        _inflight[0] += 1
    try:
        return _run_batch_reserved(latexes, work_root, timeout_s)
    finally:
        with _START_LOCK:
            _inflight[0] -= 1


def _run_batch_reserved(latexes: Sequence[str], work_root: Path, timeout_s: int) -> tuple[Path | None, str, bool]:
    keep = False
    workdir: Path | None = None
    try:
        work_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="batch-", dir=work_root))
        (workdir / "batch.tex").write_text(_batch_source(latexes), encoding="utf-8")
        name = f"lom-render-{uuid.uuid4().hex}"
        try:
            proc = subprocess.run(
                _docker_cmd(name, workdir.resolve()), capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            # equivalent of `docker kill <name>`: the host timeout is the only limit, so stop the container explicitly
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return None, "timeout", True
        if proc.returncode in INFRA_EXIT_CODES:
            raise _infra_error(workdir, "docker or container failure", proc)
        log_path = workdir / "batch.log"
        if not log_path.exists():
            raise _infra_error(workdir, "no pdflatex log produced", proc)
        err = _first_error(log_path.read_text(encoding="utf-8", errors="replace"))
        if err:
            return None, err, False
        if proc.returncode != 0:
            raise _infra_error(workdir, "non-zero exit without a LaTeX error", proc)
        if not (workdir / "batch.pdf").exists():
            raise _infra_error(workdir, "no pdf produced", proc)
        keep = True
        return workdir, "", False
    except OSError as e:
        # PermissionError, disk full, docker binary vanished: the sandbox could not give a verdict
        raise RenderInfrastructureError(f"{type(e).__name__} while running a render batch: {e}") from e
    finally:
        # the workdir is only handed to the caller on success; every other path (including raw exceptions) removes it
        if workdir is not None and not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def _rasterize(workdir: Path, expected: int) -> list[np.ndarray] | None:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(workdir / "batch.pdf"))
    try:
        if len(doc) != expected:
            return None
        pages = []
        for i in range(expected):
            page = doc[i]
            try:
                arr = np.asarray(page.render(scale=RENDER_DPI / 72, grayscale=True).to_numpy())
            finally:
                page.close()
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            pages.append(np.ascontiguousarray(arr, dtype=np.uint8))
        return pages
    finally:
        doc.close()


def _bitmap_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{arr.shape[0]}x{arr.shape[1]}:".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


class _AbortedError(RenderInfrastructureError):
    """A sibling batch already failed; never the root cause, so it must not mask the real error."""


_ERR_LOCK = threading.Lock()


def _render_group(
    items: list[tuple[str, str]],
    work_root: Path,
    timeout_s: int,
    cache: Path | None,
    *,
    abort: threading.Event | None = None,
    errors: list[BaseException] | None = None,
) -> dict[str, RenderResult]:
    """Render a group; any failure (not only RenderInfrastructureError) stops sibling threads.

    The first real exception is kept in ``errors`` so render_many can raise it instead of an "aborted" one.
    """
    try:
        return _render_group_inner(items, work_root, timeout_s, cache, abort=abort, errors=errors)
    except Exception as e:
        if abort is not None:
            abort.set()
        if errors is not None and not isinstance(e, _AbortedError):
            with _ERR_LOCK:
                if not errors:
                    errors.append(e)
        raise


def _render_group_inner(
    items: list[tuple[str, str]],
    work_root: Path,
    timeout_s: int,
    cache: Path | None,
    *,
    abort: threading.Event | None = None,
    errors: list[BaseException] | None = None,
    first: tuple[Path | None, str, bool] | None = None,
) -> dict[str, RenderResult]:
    """Render (key, latex) items; write cache entries. A failing group is bisected down to singletons.

    Returns the uncached (transient) results. Raises RenderInfrastructureError, never bisecting on it.
    ``first`` is an already computed _run_batch outcome for these items (used when probing both halves of a timeout).
    """
    if abort is not None and abort.is_set():
        if first is not None and first[0] is not None:
            shutil.rmtree(first[0], ignore_errors=True)
        raise _AbortedError("aborted: an earlier batch hit an infrastructure failure")
    workdir, err, timed_out = first if first is not None else _run_batch([latex for _, latex in items], work_root, timeout_s)
    pages = None
    if workdir is not None:
        try:
            pages = _rasterize(workdir, len(items))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if pages is None:
            err = "page count mismatch"
    if pages is not None:
        for (key, _), arr in zip(items, pages, strict=True):
            crop = ink_crop(arr)
            if crop is None:
                _cache_put(cache, RenderResult(key, "blank-page", None, None))
            elif ink_overflows(arr):
                _cache_put(cache, RenderResult(key, "overflow", None, None, "ink touches the page edge"))
            else:
                _cache_put(
                    cache, RenderResult(key, "rendered", (int(crop.shape[0]), int(crop.shape[1])), _bitmap_hash(crop)), crop
                )
        return {}
    if len(items) == 1:
        key = items[0][0]
        if timed_out:
            return {key: RenderResult(key, "timeout", None, None, f"timeout after {timeout_s}s")}
        res = RenderResult(key, "compile-failed", None, None, err)
        if err.startswith("!"):
            _cache_put(cache, res)
            return {}
        return {key: res}
    mid = len(items) // 2
    left_first = right_first = None
    if timed_out:
        # If both halves also time out at their first attempt the host is too slow (swap, saturated CPU): bisecting
        # further would burn ~N timeouts and report every item as failed. Stop and call it infrastructure.
        left_first = _run_batch([latex for _, latex in items[:mid]], work_root, timeout_s)
        try:
            right_first = _run_batch([latex for _, latex in items[mid:]], work_root, timeout_s)
        except BaseException:
            if left_first[0] is not None:
                shutil.rmtree(left_first[0], ignore_errors=True)
            raise
        if left_first[2] and right_first[2]:
            raise RenderInfrastructureError(
                f"both halves of a timed-out batch of {len(items)} also timed out after {timeout_s}s: host too slow "
                "or overloaded (check swap and CPU load), refusing to score every item as a timeout"
            )
    out = _render_group_inner(items[:mid], work_root, timeout_s, cache, abort=abort, errors=errors, first=left_first)
    out.update(_render_group_inner(items[mid:], work_root, timeout_s, cache, abort=abort, errors=errors, first=right_first))
    return out


def render_many(
    latex_list: Sequence[str],
    *,
    batch_size: int = 100,
    cache: Path | None = None,
    timeout_s: int = 300,
    max_parallel: int | None = None,
) -> list[RenderResult]:
    """Render every input (aligned with the input order); cached, batched, sandboxed, failures bisected."""
    keys = [cache_key(x) for x in latex_list]
    todo: dict[str, str] = {}
    for key, latex in zip(keys, latex_list, strict=True):
        if key in todo:
            continue
        # check_safe runs before the cache lookup, so entries written under an older, laxer filter are not served
        reason = check_safe(latex)
        cached = _cache_get(cache, key)
        if reason is not None:
            status = "empty" if reason == "empty" else "rejected"
            if cached is None or cached.status != status:
                _cache_put(cache, RenderResult(key, status, None, None, reason))
            continue
        if cached is not None:
            continue
        todo[key] = latex
    transient: dict[str, RenderResult] = {}
    if todo:
        # Only when something must be rendered: fully cached runs work without Docker. A missing image or a
        # hung daemon would otherwise surface as per-item timeouts and a silently bogus result file.
        if not docker_available():
            raise RenderInfrastructureError(f"Docker daemon or image {IMAGE} is not available: {docker_diagnosis()}")
        size = batch_size if batch_size < 50 else min(max(batch_size, 50), 200)
        size = max(size, 1)
        items = list(todo.items())
        groups = [items[i : i + size] for i in range(0, len(items), size)]
        parallel = max_parallel if max_parallel is not None else guard.docker_parallel_cap()
        if parallel <= 0:
            guard.preflight(ram_gb=guard.DOCKER_MEMORY_GB, timeout_s=600)
            parallel = 1
        work_root = (Path(cache) if cache is not None else cache_dir()) / "work"
        abort = threading.Event()
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(groups)))) as ex:
            futs = [ex.submit(_render_group, g, work_root, timeout_s, cache, abort=abort, errors=errors) for g in groups]
            for fut in futs:
                try:
                    transient.update(fut.result())
                except BaseException as exc:
                    ex.shutdown(wait=True, cancel_futures=True)
                    # prefer the first real failure over an "aborted" one raised by a sibling that started later
                    if errors and errors[0] is not exc:
                        raise errors[0] from exc
                    raise
    out = []
    for key in keys:
        res = _cache_get(cache, key) or transient.get(key)
        if res is None:
            raise RuntimeError(f"render produced no result for key {key}")
        out.append(res)
    return out


@dataclass
class PairResult:
    gt_status: str
    pred_status: str
    identical: bool | None
    tolerant: bool | None


def render_equivalence(gt: Sequence[str], pred: Sequence[str], **render_kwargs) -> list[PairResult]:
    """Per-pair render comparison. identical/tolerant are None when the ground truth does not render."""
    if len(gt) != len(pred):
        raise ValueError("gt and pred must have the same length")
    cache = render_kwargs.get("cache")
    res = render_many([*gt, *pred], **render_kwargs)
    g_res, p_res = res[: len(gt)], res[len(gt) :]
    out = []
    for g, p in zip(g_res, p_res, strict=True):
        if g.status != "rendered":
            out.append(PairResult(g.status, p.status, None, None))
            continue
        identical, tolerant = compare(g, p, cache)
        out.append(PairResult(g.status, p.status, identical, tolerant))
    return out
