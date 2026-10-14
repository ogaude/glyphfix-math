"""Layout-only label canonicalizer for extracted display equations (D-11).

Usage:
  from lightonocr_math.corpus.canon import canonicalize, original_for_validation, layout_only_diff, structural_problems
  label = canonicalize(body, env)                  # body: text between \\begin{env} and \\end{env}
  orig = original_for_validation(body, env)        # the paper's own starred environment, verbatim
  layout_only_diff(body, label) == []              # True when only layout tokens differ
  structural_problems(label) == []                 # braces balanced, nothing residual, not empty or too long
  # validation: render_many([orig], wrapper="env") and render_many([label]) must give identical bitmaps

Rules (applied in this order; RULES lists them for the dataset card):
  C1  strip comments: an unescaped % to the end of the line, the newline and the leading blanks of the next line
      (what TeX does), implemented locally
  C2  remove \\label{..}, \\tag{..}, \\tag*{..}, \\nonumber, \\notag
  C3  drop trailing \\\\ rows at the end of the body
  C4  collapse whitespace runs to one space and strip both ends
  C5  remove whitespace between \\left or \\right and the delimiter token that follows
  C6  the outer environment is removed (the builder records the env name in the record's env field); a gather body
      with a top-level \\\\ is wrapped in \\begin{gathered}..\\end{gathered} so wrap() keeps its rows centered;
      equation and align bodies are left as they are (wrap() adds aligned for a top-level & or \\\\)

Never: alias rewrites (\\le -> \\leq), \\dfrac -> \\frac, brace removal, \\mathrm or \\text edits, anything in GREEK or
DECORATION_COMMANDS, prime spelling. Glyph-bearing tokens are byte-preserved; layout_only_diff proves it per record.

Known limits:
  * a multi-column align body (several & pairs per row) is kept as is, but wrap() puts it in aligned, whose column
    spacing differs from align*: such labels can fail the bitmap validation and are reported as canon_mismatch
  * comment removal does not know about verbatim-like environments; display math has none
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from lightonocr_math.latex_tokens import tokenize
from lightonocr_math.render import _needs_aligned

CANON_RULES_VERSION = "2"
CANON_GOLDEN_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "canon_golden.jsonl"
CANON_FINGERPRINT = "1d7d79739aef5d0534f51bf13d0372c5f095f51bac23b0244ea92729e55e7ba4"
MAX_LABEL_CHARS = 1500

RULES = (
    ("C1", "strip comments (unescaped % to end of line, the newline and the next line's leading blanks)"),
    ("C2", "remove \\label{..}, \\tag{..}, \\tag*{..}, \\nonumber, \\notag"),
    ("C3", "drop trailing \\\\ at the end of the body"),
    ("C4", "collapse whitespace runs to one space, strip the ends"),
    ("C5", "remove whitespace between \\left or \\right and the following delimiter"),
    ("C6", "outer env removed; gather bodies with a top-level \\\\ wrapped in gathered"),
)

_LAYOUT_RES = (
    re.compile(r"\\label\{[^{}]*\}"),
    re.compile(r"\\tag\*?\{[^{}]*\}"),
    re.compile(r"\\(?:nonumber|notag)(?![A-Za-z])"),
)
_TRAILING_BREAK_RE = re.compile(r"(?<!\\)\\\\\s*$")
_LEFT_RIGHT_RE = re.compile(r"(\\(?:left|right))(?![A-Za-z])\s+(?=[^\sA-Za-z])")
_WS_RE = re.compile(r"\s+")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*(?=\n)")
_CONTROL_SEQ_RE = re.compile(r"\\(?:[A-Za-z]+|.)", re.DOTALL)
_VALIDATION_ENVS = ("equation", "align", "gather")
_GATHERED_OPEN = "\\begin{gathered} "
_GATHERED_CLOSE = " \\end{gathered}"


def _strip_comments(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append(s[i : i + 2])
            i += 2
        elif c == "%":
            j = s.find("\n", i)
            if j < 0:
                break
            i = j + 1
            while i < n and s[i] in " \t":
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _strip_layout_commands(s: str) -> str:
    for rx in _LAYOUT_RES:
        s = rx.sub("", s)
    return s


def _drop_trailing_breaks(s: str) -> str:
    s = s.rstrip()
    while True:
        t = _TRAILING_BREAK_RE.sub("", s)
        if t == s:
            return s
        s = t.rstrip()


def canonicalize(body: str, env: str) -> str:
    """Layout-only canonical label for the body of a display environment ``env``."""
    s = _strip_layout_commands(_strip_comments(body))
    s = _drop_trailing_breaks(s)
    s = _WS_RE.sub(" ", s).strip()
    s = _LEFT_RIGHT_RE.sub(r"\1", s)
    if env == "gather" and _needs_aligned(s):
        s = _GATHERED_OPEN + s + _GATHERED_CLOSE
    return s


def original_for_validation(body: str, env: str) -> str:
    """The paper's own starred environment: comments, labels, tags and numbering switches removed, nothing else.

    Removing a \\label or \\tag that sat on its own line leaves a whitespace-only line, which TeX reads as a paragraph
    break and rejects inside align* and gather*; such lines are dropped so the check sees the equation, not the gap.
    """
    if env not in _VALIDATION_ENVS:
        raise ValueError(f"unsupported env {env!r} for validation; expected one of {_VALIDATION_ENVS}")
    s = _BLANK_LINE_RE.sub("", _strip_layout_commands(_strip_comments(body)))
    return f"\\begin{{{env}*}}{s}\\end{{{env}*}}"


def _control_space(s: str) -> str:
    """Spell a backslash followed by a whitespace character (a control space) as backslash and a blank."""
    return _CONTROL_SEQ_RE.sub(lambda m: "\\ " if m.group(0)[1:].isspace() else m.group(0), s)


def _layout_tokens(s: str) -> list[str]:
    return tokenize(_control_space(_drop_trailing_breaks(_strip_layout_commands(_strip_comments(s)))))


def layout_only_diff(raw_body: str, canonical: str) -> list[str]:
    """Tokens that differ between the raw body and the canonical label ([] when only layout changed).

    Entries are "-tok" (in the raw body only) and "+tok" (in the label only). Token order is not compared:
    canonicalization never reorders, and a multiset check keeps the diff readable.
    """
    own_wrapper = _WS_RE.sub(" ", _strip_comments(raw_body)).strip().startswith(_GATHERED_OPEN.strip())
    if not own_wrapper and canonical.startswith(_GATHERED_OPEN) and canonical.endswith(_GATHERED_CLOSE):
        canonical = canonical[len(_GATHERED_OPEN) : len(canonical) - len(_GATHERED_CLOSE)]
    raw, can = Counter(_layout_tokens(raw_body)), Counter(tokenize(_control_space(canonical)))
    diff = [f"-{t}" for t in sorted((raw - can).elements())] + [f"+{t}" for t in sorted((can - raw).elements())]
    return diff


def structural_problems(label: str) -> list[str]:
    """Reasons a canonical label is not usable as ground truth; [] for a clean label."""
    problems: list[str] = []
    if not label.strip():
        problems.append("empty label")
    if len(label) > MAX_LABEL_CHARS:
        problems.append(f"label longer than {MAX_LABEL_CHARS} chars")
    depth, i, n, bad = 0, 0, len(label), False
    while i < n:
        c = label[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                bad = True
                break
        i += 1
    if bad or depth != 0:
        problems.append("unbalanced braces")
    if re.search(r"\\(?:label|tag)(?![A-Za-z])", label):
        problems.append("residual \\label or \\tag")
    if re.search(r"(?<!\\)(?:\\\\)*%", label):
        problems.append("residual comment %")
    if _TRAILING_BREAK_RE.search(label.rstrip()):
        problems.append("trailing \\\\")
    return problems


def canon_fingerprint(golden: Path = CANON_GOLDEN_PATH) -> str:
    """sha256 over the rules version and each golden case with the label the current rules produce."""
    h = hashlib.sha256()
    h.update(f"{CANON_RULES_VERSION}\n".encode())
    for line in golden.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        item = [r["env"], r["body"], r["expected"], canonicalize(r["body"], r["env"]), bool(r["validate"])]
        h.update(json.dumps(item, ensure_ascii=False).encode() + b"\n")
    return h.hexdigest()
