"""Strict, tokenizer-based LaTeX normalizer.

It merges only strings that render to the same glyphs, so that prime/l, dropped
primes, Greek/Latin, bold/bar/underline and I/1 errors are never hidden.

Rules (NORMALIZE_RULES_VERSION 2), applied in this fixed order:

1. ``strip_layout`` on the string: outer ``$$..$$``, ``$..$``, ``\\[..\\]``,
   ``\\(..\\)`` and outer ``\\begin{X}..\\end{X}`` for equation/align-like
   environments; remove ``\\label{}``, ``\\tag{}``, ``\\tag*{}``, ``\\nonumber``,
   ``\\notag``; trailing equation number ``(12a)`` after a long gap (``\\quad``,
   ``\\qquad``, ``\\hfill``, ``\\;\\;``, two control spaces or ``~~``); trailing
   ``, . ;`` unless the dot closes ``\\left.`` or ``\\right.``.
2. Tokenize; drop ``&`` (alignment only). ``\\\\`` and inner environments stay.
3. Drop spacing tokens (SPACING_TOKENS) and the invisible ``\\left.`` /
   ``\\right.`` pairs.
4. Drop ``\\big``-family sizing commands (_SIZING_RE). ``\\left``/``\\right``
   are kept: they render at a different size.
5. Glyph-identical aliases (ALIASES), for example ``\\le`` -> ``\\leq``,
   ``\\prime`` -> ``'``. Nothing that changes a glyph is aliased.
6. Redundant braces: a ``{X}`` group with exactly one non-brace token is
   unwrapped only when it is the argument of ``^``/``_`` or of a command in
   ONE_ARG_COMMANDS / TWO_ARG_COMMANDS. Commands are never removed, free
   groups and multi-token groups are never unwrapped.
7. Primes: ``^'`` and ``^{' '}`` become plain ``'`` tokens.
8. Script order: ``^A _B`` becomes ``_B ^A``. Primes are not reordered.

Unbalanced braces never raise; unmatched groups are left as they are.

Changes in 2: a bare trailing ``\\right`` no longer equals ``\\right.``, and a real
trailing argument such as ``\\ln (2)`` or ``f (3)`` is no longer stripped as an
equation number.
"""

from __future__ import annotations

import re

from lightonocr_math.latex_tokens import detokenize, tokenize

NORMALIZE_RULES_VERSION = "2"

SPACING_TOKENS = frozenset(
    {"\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad", "~", "\\enspace", "\\thinspace", "\\medspace", "\\thickspace"}
)

_SIZING_RE = re.compile(r"\\(?:big|Big|bigg|Bigg)[lrm]?")

ALIASES = {
    "\\le": "\\leq",
    "\\ge": "\\geq",
    "\\ne": "\\neq",
    "\\prime": "'",
    "\\to": "\\rightarrow",
    "\\gets": "\\leftarrow",
    "\\lbrace": "\\{",
    "\\rbrace": "\\}",
    "\\land": "\\wedge",
    "\\lor": "\\vee",
    "\\lnot": "\\neg",
}

ONE_ARG_COMMANDS = frozenset(
    {
        "\\hat", "\\bar", "\\tilde", "\\vec", "\\dot", "\\ddot", "\\check", "\\breve", "\\acute", "\\grave",
        "\\widehat", "\\widetilde", "\\overline", "\\underline", "\\mathbf", "\\mathrm", "\\mathit", "\\mathsf",
        "\\mathtt", "\\mathcal", "\\mathbb", "\\mathfrak", "\\mathscr", "\\boldsymbol", "\\bm", "\\sqrt", "\\text",
        "\\operatorname",
    }
)  # fmt: skip
TWO_ARG_COMMANDS = frozenset({"\\frac", "\\dfrac", "\\tfrac", "\\binom", "\\dbinom", "\\tbinom"})

_ENVS = (
    "equation", "equation*", "displaymath", "align", "align*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*",
)  # fmt: skip
_ENV_RE = re.compile(r"^\\begin\{(" + "|".join(re.escape(e) for e in _ENVS) + r")\}(.*)\\end\{\1\}$", re.DOTALL)
_LABEL_RE = re.compile(r"\\label\s*\{[^{}]*\}")
_TAG_RE = re.compile(r"\\tag\*?\s*\{[^{}]*\}")
_NONUM_RE = re.compile(r"\\(?:nonumber|notag)(?![A-Za-z])")
_LONG_GAP = r"(?:\\(?:q?quad|hfill)(?![A-Za-z])|(?:\\;\s*){2,}|(?:\\ \s*){2,}|(?:~\s*){2,})"
_TRAIL_NUM_RE = re.compile(_LONG_GAP + r"\s*\(\s*\d+[a-z]?\s*\)\s*$")
_TRAIL_PUNCT_RE = re.compile(r"[,.;]\s*$")
_DELIM_END_RE = re.compile(r"\\(?:right|left)$")


def _strip_trailing_punct(s: str) -> str:
    m = _TRAIL_PUNCT_RE.search(s)
    if m is None:
        return s
    head = s[: m.start()].rstrip()
    if s[: m.start()].endswith("\\") or _DELIM_END_RE.search(head):
        return s
    return s[: m.start()]


def _strip_outer(s: str) -> str:
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for left, right in (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")):
            if len(s) >= len(left) + len(right) and s.startswith(left) and s.endswith(right):
                s = s[len(left) : len(s) - len(right)].strip()
                changed = True
                break
        else:
            m = _ENV_RE.match(s)
            if m:
                s = m.group(2).strip()
                changed = True
    return s


def strip_layout(latex: str) -> str:
    """Remove equation wrappers, labels, tags, numbers and trailing punctuation."""
    s = _strip_outer(latex)
    s = _LABEL_RE.sub("", s)
    s = _TAG_RE.sub("", s)
    s = _NONUM_RE.sub("", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = _TRAIL_NUM_RE.sub("", s).strip()
        s = _strip_trailing_punct(s).strip()
    return s


def _match_braces(tokens: list[str]) -> dict[int, int]:
    """Map index of each matched ``{`` to the index of its ``}`` (iterative)."""
    stack: list[int] = []
    match: dict[int, int] = {}
    for i, t in enumerate(tokens):
        if t == "{":
            stack.append(i)
        elif t == "}" and stack:
            match[stack.pop()] = i
    return match


def _drop_spacing_and_sizing(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t == "&" or t in SPACING_TOKENS or _SIZING_RE.fullmatch(t):
            i += 1
            continue
        if t in ("\\left", "\\right") and i + 1 < n and tokens[i + 1] == ".":
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _unwrap_braces(tokens: list[str]) -> list[str]:
    match = _match_braces(tokens)
    n = len(tokens)
    drop: set[int] = set()

    def candidate(start: int) -> None:
        end = match.get(start)
        if end is not None and end - start == 2 and tokens[start + 1] not in ("{", "}"):
            drop.add(start)
            drop.add(end)

    for k, t in enumerate(tokens):
        if t in ("^", "_") or t in ONE_ARG_COMMANDS:
            if k + 1 < n and tokens[k + 1] == "{":
                candidate(k + 1)
        elif t in TWO_ARG_COMMANDS and k + 1 < n:
            if tokens[k + 1] == "{":
                candidate(k + 1)
                end = match.get(k + 1)
                nxt = end + 1 if end is not None else None
            else:
                nxt = k + 2
            if nxt is not None and nxt < n and tokens[nxt] == "{":
                candidate(nxt)
    if not drop:
        return tokens
    return [t for i, t in enumerate(tokens) if i not in drop]


def _fold_primes(tokens: list[str]) -> list[str]:
    match = _match_braces(tokens)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t == "^" and i + 1 < n:
            if tokens[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            if tokens[i + 1] == "{":
                end = match.get(i + 1)
                if end is not None and end > i + 2 and all(x == "'" for x in tokens[i + 2 : end]):
                    out.extend(tokens[i + 2 : end])
                    i = end + 1
                    continue
        out.append(t)
        i += 1
    return out


def _reorder_scripts(tokens: list[str]) -> list[str]:
    depth = 0
    for t in tokens:
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            if depth < 0:
                return tokens
    if depth != 0:
        return tokens
    match = _match_braces(tokens)
    n = len(tokens)

    def arg_end(i: int) -> int | None:
        if i >= n:
            return None
        if tokens[i] == "{":
            end = match.get(i)
            return None if end is None else end + 1
        return i + 1

    out = list(tokens)
    for p in [i for i, t in enumerate(tokens) if t == "^"][::-1]:
        a_end = arg_end(p + 1)
        if a_end is None or a_end >= n or out[a_end] != "_":
            continue
        b_end = arg_end(a_end + 1)
        if b_end is None:
            continue
        out[p:b_end] = out[a_end:b_end] + out[p:a_end]
    return out


def normalize_tokens(latex: str) -> list[str]:
    """Normalize ``latex`` to a canonical token list (never raises)."""
    tokens = tokenize(strip_layout(latex))
    tokens = _drop_spacing_and_sizing(tokens)
    tokens = [ALIASES.get(t, t) for t in tokens]
    tokens = _unwrap_braces(tokens)
    tokens = _fold_primes(tokens)
    return _reorder_scripts(tokens)


def canonical(latex: str) -> str:
    return detokenize(normalize_tokens(latex))


def equivalent(a: str, b: str) -> bool:
    return normalize_tokens(a) == normalize_tokens(b)
