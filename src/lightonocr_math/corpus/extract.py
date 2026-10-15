"""Safe archive extraction and static display-equation extraction from LaTeX sources.

Usage:
  from lightonocr_math.corpus import extract as X
  X.safe_extract(Path("data/sources/2301.01234/src.bin"), "tar.gz", Path("data/work/2301.01234/src"))
  main = X.find_main(root); files = X.document_files(root, main)
  eqs = X.extract_display(files, root)          # equation / align / gather and starred forms, with file and line
  X.own_macros(X.macro_files(root, files)); X.engine_hints(files)

Rules:
  * static text analysis only: nothing here executes TeX or any code from a source; the render
    sandbox is the authoritative judge of whether an equation compiles
  * archives are validated member by member before anything is written: regular files and
    directories only, normalized path inside dest, total size and member count capped
  * only files reachable from the main file through \\input, \\include or \\subfile are scanned
  * multline, flalign, alignat, eqnarray, \\[..\\] and $$..$$ are counted, never extracted

Known limit: \\iffalse blanking tracks plain \\if... nesting only; \\newif-defined conditionals are not understood,
and \\input paths built from macros are not followed.
"""

from __future__ import annotations

import gzip
import re
import shutil
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from lightonocr_math.errors import DECORATION_COMMANDS, GREEK

EXTRACT_ENVS = ("equation", "align", "gather")
OTHER_DISPLAY = ("multline", "flalign", "alignat", "eqnarray", "\\[", "$$")
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 5000
MAX_SOURCE_BYTES = 5 * 1024 * 1024
EXTRACT_VERSION = "1"

DEF_PATTERNS = [
    r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*(\\[A-Za-z@]+)",
    r"\\DeclareMathOperator\*?\s*\{?\s*(\\[A-Za-z]+)",
    r"\\def\s*(\\[A-Za-z@]+)",
    r"\\DeclareRobustCommand\*?\s*\{?\s*(\\[A-Za-z]+)",
]
_REDEF_PATTERNS = [r"\\renewcommand\*?\s*\{?\s*(\\[A-Za-z@]+)", r"\\def\s*(\\[A-Za-z@]+)"]
ENGINE_PATTERNS = {
    name: re.compile(r"(?<![A-Za-z-])" + re.escape(name) + r"(?![A-Za-z-])")
    for name in ("fontspec", "unicode-math", "polyglossia", "luacode", "xltxtra", "xunicode")
}
_MAGIC_ENGINE = re.compile(r"^[ \t]*%\s*!\s*TEX\s+program\s*=\s*(xelatex|lualatex)", re.IGNORECASE | re.MULTILINE)
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*(?:\{([^{}]*)\}|([^\s{}\\]+))")
_DOCCLASS_RE = re.compile(r"^[ \t]*\\documentclass", re.MULTILINE)


class UnsafeArchive(ValueError):
    pass


# ---------------------------------------------------------------- archives


def _check_member(m: tarfile.TarInfo, dest: Path) -> None:
    if not (m.isreg() or m.isdir()):
        raise UnsafeArchive(f"member {m.name!r} is not a regular file or directory")
    name = m.name
    if not name or "\x00" in name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        raise UnsafeArchive(f"absolute or empty path {name!r}")
    if ".." in Path(name.replace("\\", "/")).parts:
        raise UnsafeArchive(f"path traversal in {name!r}")
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise UnsafeArchive(f"member {name!r} escapes the destination")


def safe_extract(
    src_bin: Path, kind: str, dest: Path, *, max_bytes: int = MAX_ARCHIVE_BYTES, max_members: int = MAX_ARCHIVE_MEMBERS
) -> list[str]:
    src_bin, dest = Path(src_bin), Path(dest)
    if kind in ("tar.gz", "tar"):
        try:
            tf = tarfile.open(src_bin, mode="r:*")  # noqa: SIM115 (closed by the with below)
        except (tarfile.TarError, OSError, EOFError) as e:
            raise UnsafeArchive(f"unreadable archive: {e}") from e
        with tf:
            try:
                members = tf.getmembers()
            except (tarfile.TarError, OSError, EOFError) as e:
                raise UnsafeArchive(f"unreadable archive: {e}") from e
            if len(members) > max_members:
                raise UnsafeArchive(f"{len(members)} members exceeds the cap of {max_members}")
            total = 0
            for m in members:
                _check_member(m, dest)
                total += m.size if m.isreg() else 0
                if total > max_bytes:
                    raise UnsafeArchive(f"uncompressed size exceeds the cap of {max_bytes} bytes")
            dest.mkdir(parents=True, exist_ok=True)
            try:
                tf.extractall(dest, filter="data")
            except (OSError, tarfile.TarError, EOFError, zlib.error) as e:
                shutil.rmtree(dest, ignore_errors=True)
                raise UnsafeArchive(f"extraction failed: {e}") from e
        return sorted(m.name for m in members if m.isreg())
    if kind == "gz":
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "main.tex"
        written = 0
        try:
            with gzip.open(src_bin, "rb") as fin, open(out, "wb") as fout:
                while chunk := fin.read(1 << 16):
                    written += len(chunk)
                    if written > max_bytes:
                        raise UnsafeArchive(f"decompressed size exceeds the cap of {max_bytes} bytes")
                    fout.write(chunk)
        except (OSError, EOFError) as e:
            out.unlink(missing_ok=True)
            raise UnsafeArchive(f"unreadable gzip: {e}") from e
        except UnsafeArchive:
            out.unlink(missing_ok=True)
            raise
        return ["main.tex"]
    raise UnsafeArchive(f"unsupported bundle kind {kind!r}")


# ---------------------------------------------------------------- text helpers


def read_source(path: Path, skipped: list[str] | None = None) -> str | None:
    """UTF-8 text with replacement; None (and recorded in skipped) when the file exceeds MAX_SOURCE_BYTES."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            if skipped is not None:
                skipped.append(str(path))
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _strip_line(line: str) -> str:
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if c == "%":
            return line[:i]
        i += 1
    return line


def strip_comments(text: str) -> str:
    """Remove unescaped % to end of line; the number of lines is unchanged."""
    return "\n".join(_strip_line(ln) for ln in text.split("\n"))


def _blank(s: str) -> str:
    return re.sub(r"[^\n]", " ", s)


_IF_OPEN = re.compile(r"\\if(?!f(?![A-Za-z@]))[A-Za-z@]*")
_FI = re.compile(r"\\fi(?![A-Za-z@])")


def blank_disabled(text: str) -> str:
    """Blank \\iffalse..\\fi and comment environments with spaces, keeping every newline."""
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", lambda m: _blank(m.group(0)), text, flags=re.DOTALL)
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\\iffalse(?![A-Za-z@])", text):
        if m.start() < pos:
            continue
        depth = 1
        j = m.end()
        end = len(text)
        while True:
            nxt_fi = _FI.search(text, j)
            if nxt_fi is None:
                break
            nxt_if = _IF_OPEN.search(text, j)
            if nxt_if is not None and nxt_if.start() < nxt_fi.start():
                depth += 1
                j = nxt_if.end()
                continue
            depth -= 1
            j = nxt_fi.end()
            if depth == 0:
                end = j
                break
        out.append(text[pos : m.start()])
        out.append(_blank(text[m.start() : end]))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _clean(text: str) -> str:
    return blank_disabled(strip_comments(text))


# ---------------------------------------------------------------- main file and document tree


def _input_names(text: str) -> list[str]:
    return [(m.group(1) or m.group(2)).strip() for m in _INPUT_RE.finditer(_clean(text))]


def _resolve(root: Path, base_dir: Path, name: str) -> Path | None:
    root_r = root.resolve()
    for d in (root, base_dir):
        for cand in ([name] if name.endswith(".tex") else [name + ".tex", name]):
            p = (d / cand)
            try:
                r = p.resolve()
            except OSError:
                continue
            if r.is_file() and r.is_relative_to(root_r) and r.suffix == ".tex":
                return r
    return None


def find_main(root: Path) -> Path | None:
    root = Path(root)
    cands: list[tuple[Path, str]] = []
    for f in sorted(root.rglob("*.tex")):
        t = read_source(f)
        if t is None:
            continue
        c = _clean(t)
        if _DOCCLASS_RE.search(c) and "\\begin{document}" in c:
            cands.append((f, t))
    if not cands:
        return None
    included: set[Path] = set()
    for f, t in cands:
        for name in _input_names(t):
            r = _resolve(root, f.parent, name)
            if r is not None and r != f.resolve():
                included.add(r)
    free = [f for f, _ in cands if f.resolve() not in included] or [f for f, _ in cands]
    return max(free, key=lambda f: (f.stat().st_size, str(f)))


def document_files(root: Path, main: Path) -> list[Path]:
    root = Path(root)
    seen: list[Path] = []
    seen_set: set[Path] = set()
    stack = [Path(main).resolve()]
    while stack:
        f = stack.pop()
        if f in seen_set:
            continue
        seen_set.add(f)
        seen.append(f)
        t = read_source(f)
        if t is None:
            continue
        for name in reversed(_input_names(t)):
            r = _resolve(root, f.parent, name)
            if r is not None and r not in seen_set:
                stack.append(r)
    return seen


def macro_files(root: Path, files: list[Path]) -> list[Path]:
    """Document files plus every .sty/.cls shipped in the bundle."""
    extra = [p for p in sorted(Path(root).rglob("*")) if p.suffix in (".sty", ".cls") and p.is_file()]
    return list(files) + extra


# ---------------------------------------------------------------- macros and engines


def own_macros(files: list[Path]) -> set[str]:
    names: set[str] = set()
    for f in files:
        t = read_source(f)
        if t is None:
            continue
        c = _clean(t)
        for pat in DEF_PATTERNS:
            names.update(re.findall(pat, c))
    return {n for n in names if len(n) > 1}


def redefined_core(files: list[Path]) -> set[str]:
    core = GREEK | DECORATION_COMMANDS | {"\\prime"}
    hit: set[str] = set()
    for f in files:
        t = read_source(f)
        if t is None:
            continue
        c = _clean(t)
        for pat in _REDEF_PATTERNS:
            hit.update(n for n in re.findall(pat, c) if n in core)
    return hit


def engine_hints(files: list[Path]) -> list[str]:
    hints: set[str] = set()
    for f in files:
        t = read_source(f)
        if t is None:
            continue
        for m in _MAGIC_ENGINE.finditer(t):
            hints.add(f"program={m.group(1).lower()}")
        c = strip_comments(t)
        hints.update(name for name, pat in ENGINE_PATTERNS.items() if pat.search(c))
    return sorted(hints)


def uses_macro(body: str, macros: set[str]) -> bool:
    return any(re.search(re.escape(m) + r"(?![A-Za-z])", body) for m in macros)


# ---------------------------------------------------------------- display equations


@dataclass
class Equation:
    file: str
    line: int
    env: str
    starred: bool
    body: str
    raw: str


_EQ_RE = re.compile(r"\\begin\{(" + "|".join(EXTRACT_ENVS) + r")(\*?)\}(.*?)\\end\{\1\2\}", re.DOTALL)


def extract_display(files: list[Path], root: Path, skipped: list[str] | None = None) -> list[Equation]:
    root_r = Path(root).resolve()
    out: list[Equation] = []
    for f in files:
        t = read_source(f, skipped)
        if t is None:
            continue
        c = _clean(t)
        lines = t.split("\n")
        try:
            rel = str(Path(f).resolve().relative_to(root_r))
        except ValueError:
            rel = str(f)
        for m in _EQ_RE.finditer(c):
            start = c.count("\n", 0, m.start())
            end = c.count("\n", 0, m.end())
            out.append(Equation(
                file=rel, line=start + 1, env=m.group(1), starred=m.group(2) == "*", body=m.group(3),
                raw="\n".join(lines[start : end + 1]),
            ))
    return out


def other_display_counts(files: list[Path]) -> dict[str, int]:
    counts = dict.fromkeys(OTHER_DISPLAY, 0)
    for f in files:
        t = read_source(f)
        if t is None:
            continue
        c = _clean(t)
        for env in ("multline", "flalign", "alignat", "eqnarray"):
            counts[env] += len(re.findall(r"\\begin\{" + env + r"\*?\}", c))
        counts["\\["] += len(re.findall(r"(?<!\\)\\\[", c))
        counts["$$"] += len(re.findall(r"(?<!\\)\$\$", c)) // 2
    return counts
