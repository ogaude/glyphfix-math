from glyphfix_math.latex_tokens import detokenize, tokenize


def test_frac():
    assert tokenize(r"\frac{a}{b}") == ["\\frac", "{", "a", "}", "{", "b", "}"]


def test_control_symbols_kept():
    assert tokenize(r"a\,b\\c") == ["a", "\\,", "b", "\\\\", "c"]


def test_whitespace_dropped():
    assert tokenize("a  \n b\t c") == ["a", "b", "c"]


def test_control_word_stops_at_digit():
    assert tokenize(r"\alpha2") == ["\\alpha", "2"]


def test_escaped_braces_and_space():
    assert tokenize(r"\{ a \} \ b") == ["\\{", "a", "\\}", "\\ ", "b"]


def test_detokenize_roundtrip():
    assert detokenize(tokenize(r"\frac{a}{b}")) == "\\frac { a } { b }"
