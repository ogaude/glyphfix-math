"""Error classification of normalized LaTeX mismatches by token alignment.

Rules (ERROR_RULES_VERSION 1). ``classify`` works on ``normalize_tokens`` output.

* Equal token lists give the empty set.
* Whole-record check: if removing decoration commands (DECORATION_COMMANDS and the
  braces directly following them) makes both sides equal, the result is
  ``{"decoration"}``.
* Otherwise the token lists are aligned with ``Levenshtein.opcodes``; consecutive
  non-equal opcodes are merged into one span (g = ground-truth tokens, p = prediction
  tokens). A span may receive several classes:

  - prime_l: replace span where ``'`` is on one side and one of ``l \\ell 1 | I``
    on the other.
  - dropped_prime: more ``'`` in g than in p and the span is not prime_l.
  - greek_latin: replace span with a GREEK token on one side and a single ASCII
    letter on the other.
  - decoration: either side contains a DECORATION_COMMANDS token.
  - i_1: replace span with ``1`` on one side and one of ``I l |`` on the other, and
    no ``'`` involved.
  - other: nothing else fired for the span. Extra (inserted) primes and
    Greek-Greek / Latin-Latin swaps land here.

``notation_present`` gives the per-class denominators from the ground truth alone
(A6). The consumer uses denominator = records where the class is present OR the
record has that error, so rates never exceed 1. The Latin lookalike list makes the
greek_latin denominator broad on purpose. ``x`` is left out of that list: it is a
lookalike of chi, but including it would mark nearly every record as a greek_latin
candidate.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from glyphfix_math.normalize import normalize_tokens

ERROR_RULES_VERSION = "1"
CLASSES = ("prime_l", "dropped_prime", "greek_latin", "decoration", "i_1", "other")

_LOWER = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
_UPPER = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]


def _build_greek() -> frozenset[str]:
    names = {f"\\{n}" for n in _LOWER if n != "omicron"}
    names |= {f"\\{n}" for n in _UPPER}
    names |= {f"\\var{n}" for n in ("epsilon", "theta", "pi", "rho", "sigma", "phi", "kappa")}
    names |= {f"\\var{n}" for n in _UPPER}
    names.add("\\digamma")
    names |= {f"\\up{n}" for n in _LOWER if n != "omicron"}
    names |= {f"\\Up{n.lower()}" for n in _UPPER}
    return frozenset(names)


GREEK = _build_greek()

DECORATION_COMMANDS = frozenset(
    {
        "\\mathbf", "\\boldsymbol", "\\bm", "\\mathit", "\\mathrm", "\\mathsf", "\\mathtt", "\\mathcal", "\\mathbb",
        "\\mathfrak", "\\mathscr", "\\bar", "\\overline", "\\underline", "\\hat", "\\widehat", "\\tilde", "\\widetilde",
        "\\vec", "\\dot", "\\ddot", "\\check", "\\breve", "\\acute", "\\grave",
    }
)  # fmt: skip

_PRIME_PARTNERS = frozenset({"l", "\\ell", "1", "|", "I"})
_I_LIKE = frozenset({"I", "l", "|"})
_LATIN_LOOKALIKES = frozenset("vunwaeopki")


def strip_decorations(tokens: list[str]) -> list[str]:
    """Drop decoration commands and the braces of the group directly after each."""
    n = len(tokens)
    stack: list[int] = []
    match: dict[int, int] = {}
    for i, t in enumerate(tokens):
        if t == "{":
            stack.append(i)
        elif t == "}" and stack:
            match[stack.pop()] = i
    skip: set[int] = set()
    for i, t in enumerate(tokens):
        if t in DECORATION_COMMANDS:
            skip.add(i)
            if i + 1 < n and tokens[i + 1] == "{" and (i + 1) in match:
                skip.add(i + 1)
                skip.add(match[i + 1])
    return [t for i, t in enumerate(tokens) if i not in skip]


def _is_latin_letter(t: str) -> bool:
    return len(t) == 1 and t.isascii() and t.isalpha()


def _span_classes(g: list[str], p: list[str]) -> set[str]:
    out: set[str] = set()
    replace = bool(g) and bool(p)
    prime_involved = "'" in g or "'" in p
    if replace and (
        ("'" in g and any(t in _PRIME_PARTNERS for t in p)) or ("'" in p and any(t in _PRIME_PARTNERS for t in g))
    ):
        out.add("prime_l")
    if g.count("'") > p.count("'") and "prime_l" not in out:
        out.add("dropped_prime")
    if replace and (
        (any(t in GREEK for t in g) and any(_is_latin_letter(t) for t in p))
        or (any(t in GREEK for t in p) and any(_is_latin_letter(t) for t in g))
    ):
        out.add("greek_latin")
    if any(t in DECORATION_COMMANDS for t in g) or any(t in DECORATION_COMMANDS for t in p):
        out.add("decoration")
    if replace and not prime_involved and (
        ("1" in g and any(t in _I_LIKE for t in p)) or ("1" in p and any(t in _I_LIKE for t in g))
    ):
        out.add("i_1")
    if not out:
        out.add("other")
    return out


def classify_tokens(gt: list[str], pred: list[str]) -> set[str]:
    if gt == pred:
        return set()
    if strip_decorations(gt) == strip_decorations(pred):
        return {"decoration"}
    spans: list[list[int]] = []
    for op in Levenshtein.opcodes(gt, pred):
        if op.tag == "equal":
            continue
        if spans and spans[-1][1] == op.src_start and spans[-1][3] == op.dest_start:
            spans[-1][1] = op.src_end
            spans[-1][3] = op.dest_end
        else:
            spans.append([op.src_start, op.src_end, op.dest_start, op.dest_end])
    classes: set[str] = set()
    for s0, s1, d0, d1 in spans:
        classes |= _span_classes(gt[s0:s1], pred[d0:d1])
    return classes or {"other"}


def classify(gt_latex: str, pred_latex: str) -> set[str]:
    return classify_tokens(normalize_tokens(gt_latex), normalize_tokens(pred_latex))


def notation_present(gt_latex: str) -> set[str]:
    """Classes the ground truth could exhibit (denominators); always includes ``other``."""
    toks = normalize_tokens(gt_latex)
    present = {"other"}
    if "'" in toks:
        present |= {"prime_l", "dropped_prime"}
    if any(t in GREEK or t in _LATIN_LOOKALIKES for t in toks):
        present.add("greek_latin")
    if any(t in DECORATION_COMMANDS for t in toks):
        present.add("decoration")
    if any(t in ("I", "1", "l", "|") for t in toks):
        present.add("i_1")
    return present


__all__ = [
    "CLASSES",
    "DECORATION_COMMANDS",
    "ERROR_RULES_VERSION",
    "GREEK",
    "classify",
    "classify_tokens",
    "notation_present",
    "strip_decorations",
]
