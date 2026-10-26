"""Per-equation notation flags used to stratify the benchmark and training selection (D-06).

Usage:
  from glyphfix_math.corpus.flags import flags, FLAGS, FLAG_RULES_VERSION
  flags("f'(x) + \\alpha")      # ["prime", "greek"]
  flags("x + 1")               # ["plain"]

Rules (computed from the raw tokens of the canonical label; primes are never folded):
  prime       a token "'" or "\\prime"
  greek       a token in errors.GREEK
  decoration  a token in errors.DECORATION_COMMANDS
  i_1         (a) a bare "I" token outside the argument of \\text, \\mathrm, \\operatorname, \\textrm, \\mathit, \\mathsf, or
              (b) a "1" token whose nearest neighbour on either side is "l", "I", "|" or "\\ell", or
              (c) "1" as the only content of the argument of \\mathbb, \\mathbf, \\mathds, \\mathbbm, \\boldsymbol, \\bm
  plain       none of the above

The i_1 rule is deliberately narrower than the scorer denominator in errors.py, whose i_1 fires on any 1, l or |: that is a
denominator for scoring, not a stratification flag. Flags are ordered as in FLAGS.

Known limit: a "1" next to "|" is flagged even in expressions such as |1\\rangle; the flag selects candidates for
stratified sampling, it is not a claim that the glyph is ambiguous.
"""

from __future__ import annotations

from glyphfix_math.errors import DECORATION_COMMANDS, GREEK
from glyphfix_math.latex_tokens import tokenize

FLAGS = ("prime", "greek", "decoration", "i_1", "plain")
FLAG_RULES_VERSION = "1"

_TEXT_ARG_COMMANDS = frozenset({"\\text", "\\mathrm", "\\operatorname", "\\textrm", "\\mathit", "\\mathsf"})
_ONE_ARG_COMMANDS = frozenset({"\\mathbb", "\\mathbf", "\\mathds", "\\mathbbm", "\\boldsymbol", "\\bm"})
_ONE_NEIGHBOURS = frozenset({"l", "I", "|", "\\ell"})


def _matching_close(tokens: list[str], open_idx: int) -> int | None:
    depth = 0
    for j in range(open_idx, len(tokens)):
        if tokens[j] == "{":
            depth += 1
        elif tokens[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return None


def _text_argument_positions(tokens: list[str]) -> set[int]:
    inside: set[int] = set()
    for i, t in enumerate(tokens):
        if t in _TEXT_ARG_COMMANDS and i + 1 < len(tokens) and tokens[i + 1] == "{":
            close = _matching_close(tokens, i + 1)
            if close is not None:
                inside.update(range(i + 2, close))
    return inside


def _has_i_1(tokens: list[str]) -> bool:
    inside = _text_argument_positions(tokens)
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "I" and i not in inside:
            return True
        if t == "1":
            if (i > 0 and tokens[i - 1] in _ONE_NEIGHBOURS) or (i + 1 < n and tokens[i + 1] in _ONE_NEIGHBOURS):
                return True
            if i >= 2 and tokens[i - 1] == "{" and tokens[i - 2] in _ONE_ARG_COMMANDS and i + 1 < n and tokens[i + 1] == "}":
                return True
    return False


def flags(latex: str) -> list[str]:
    """Notation flags of a canonical label, in FLAGS order; ["plain"] when none apply."""
    tokens = tokenize(latex)
    present = set(tokens)
    out = []
    if "'" in present or "\\prime" in present:
        out.append("prime")
    if present & GREEK:
        out.append("greek")
    if present & DECORATION_COMMANDS:
        out.append("decoration")
    if _has_i_1(tokens):
        out.append("i_1")
    return out or ["plain"]
