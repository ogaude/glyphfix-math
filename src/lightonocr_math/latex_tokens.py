"""Minimal LaTeX tokenizer used by the normalizer and the error classifier.

Tokens are control words (``\\alpha``), control symbols (``\\,``, ``\\\\``,
``\\{``), or any other single character. Whitespace is skipped. The regex has
no nested quantifiers, so tokenizing is linear in the input length.
"""

from __future__ import annotations

import re

# Outer display environments stripped by both normalize._ENV_RE and render._ENV_RE; a starred
# variant is matched by the regex, not listed here.
DISPLAY_ENVS = ("equation", "align", "alignat", "flalign", "gather", "multline", "eqnarray", "displaymath", "math")

_TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\s+|.", re.DOTALL)


def tokenize(latex: str) -> list[str]:
    """Split ``latex`` into tokens, dropping whitespace."""
    return [m for m in _TOKEN_RE.findall(latex) if not m.isspace()]


def detokenize(tokens: list[str]) -> str:
    """Join tokens with single spaces."""
    return " ".join(tokens)
