"""Tests for interp.lexer: token kinds, literal values and types, positions, errors."""
from __future__ import annotations

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Severity
from pycc.interp.c_types import DOUBLE, FLOAT, INT, LLONG, LONG, UINT, ULLONG, ULONG
from pycc.interp.lexer import (
    KEYWORDS, PUNCTUATORS, Token, TokenKind, tokenize, tokenize_text, tokenize_tolerant,
)
from pycc.interp.source import PPLine


def lex(text: str) -> tuple[list[Token], list[Diagnostic]]:
    """Tokenize text and return (tokens without EOF, diagnostics)."""
    diagnostics: list[Diagnostic] = []
    tokens = tokenize_text(text, "main.c", diagnostics)
    assert tokens[-1].kind is TokenKind.EOF
    return tokens[:-1], diagnostics


def only(text: str) -> Token:
    """The single token of ``text`` (asserting there were no diagnostics)."""
    tokens, diagnostics = lex(text)
    assert diagnostics == []
    assert len(tokens) == 1
    return tokens[0]


# --- kinds ------------------------------------------------------------------------------

def test_identifiers_and_keywords() -> None:
    tokens, _ = lex("int count _Bool restrict inline my_var2")
    kinds = [t.kind for t in tokens]
    assert kinds == [TokenKind.KEYWORD, TokenKind.IDENT, TokenKind.KEYWORD, TokenKind.KEYWORD,
                     TokenKind.KEYWORD, TokenKind.IDENT]


@pytest.mark.parametrize("word", sorted(KEYWORDS))
def test_every_keyword_is_a_keyword(word: str) -> None:
    assert only(word).kind is TokenKind.KEYWORD


@pytest.mark.parametrize("punct", PUNCTUATORS)
def test_every_punctuator(punct: str) -> None:
    token = only(punct)
    assert token.kind is TokenKind.PUNCT
    assert token.value == punct


def test_longest_punctuator_wins() -> None:
    tokens, _ = lex("a<<=b->c...d++--e")
    assert [t.value for t in tokens] == ["a", "<<=", "b", "->", "c", "...", "d", "++", "--", "e"]


def test_eof_token_is_after_last_character() -> None:
    tokens = tokenize_text("int x;", "main.c")
    assert tokens[-1].kind is TokenKind.EOF
    assert tokens[-1].loc.column == 7


def test_empty_input_gives_only_eof() -> None:
    assert [t.kind for t in tokenize([])] == [TokenKind.EOF]


# --- positions --------------------------------------------------------------------------

def test_locations_are_one_based_with_end_column() -> None:
    tokens, _ = lex("int x;\n  return 42;")
    ret = tokens[3]
    assert (ret.loc.line, ret.loc.column, ret.end_column) == (2, 3, 9)
    assert str(tokens[4].loc) == "main.c:2:10"
    assert tokens[4].end_loc.column == 12


def test_locations_follow_pplines() -> None:
    lines = [PPLine("int a;", "farm.h", 7), PPLine("int b;", "main.c", 2)]
    tokens = tokenize(lines)
    assert str(tokens[0].loc) == "farm.h:7:1"
    assert str(tokens[3].loc) == "main.c:2:1"


def test_comments_are_skipped_even_across_lines() -> None:
    tokens, diagnostics = lex("a // note\nb /* one\ntwo */ c")
    assert [t.value for t in tokens] == ["a", "b", "c"]
    assert diagnostics == []


def test_unterminated_comment_is_reported() -> None:
    _, diagnostics = lex("int a; /* never closed")
    assert [d.code for d in diagnostics] == ["unterminated_comment"]
    assert str(diagnostics[0].location) == "main.c:1:8"


# --- integer literals -------------------------------------------------------------------

@pytest.mark.parametrize("text, value, ctype", [
    ("0", 0, INT),
    ("42", 42, INT),
    ("2147483647", 2147483647, INT),
    ("2147483648", 2147483648, LONG),
    ("0xFFFFFFFF", 0xFFFFFFFF, UINT),
    ("0x7FFFFFFF", 0x7FFFFFFF, INT),
    ("0x100000000", 0x100000000, LONG),
    ("0xFFFFFFFFFFFFFFFF", 0xFFFFFFFFFFFFFFFF, ULONG),
    ("017", 15, INT),
    ("10u", 10, UINT),
    ("10U", 10, UINT),
    ("5L", 5, LONG),
    ("5l", 5, LONG),
    ("5ul", 5, ULONG),
    ("5LU", 5, ULONG),
    ("7ll", 7, LLONG),
    ("7LL", 7, LLONG),
    ("7ull", 7, ULLONG),
    ("7LLU", 7, ULLONG),
    ("4294967295u", 4294967295, UINT),
    ("4294967296u", 4294967296, ULONG),
])
def test_integer_literal_value_and_type(text: str, value: int, ctype: object) -> None:
    token = only(text)
    assert token.kind is TokenKind.INT
    assert token.value == value
    assert token.ctype == ctype


def test_integer_literal_keeps_base_and_suffix() -> None:
    hex_token = only("0x1Fu")
    assert (hex_token.base, hex_token.suffix, hex_token.value) == (16, "u", 31)
    assert only("010").base == 8
    assert only("10").base == 10


def test_decimal_too_big_for_signed_is_unsigned_with_warning() -> None:
    tokens, diagnostics = lex("18446744073709551615")
    assert tokens[0].ctype == ULLONG
    assert [(d.code, d.severity) for d in diagnostics] == [
        ("integer_overflow_constant", Severity.WARNING)]


def test_integer_too_large_for_any_type() -> None:
    _, diagnostics = lex("x = 99999999999999999999999;")
    assert diagnostics[0].code == "invalid_number"
    assert diagnostics[0].message == "integer constant is too large for its type"
    assert str(diagnostics[0].location) == "main.c:1:5"


def test_invalid_suffix_on_integer() -> None:
    tokens, diagnostics = lex("int x = 123abc;")
    assert [d.code for d in diagnostics] == ["invalid_suffix"]
    assert diagnostics[0].message == 'invalid suffix "abc" on integer constant'
    assert str(diagnostics[0].location) == "main.c:1:9"
    assert diagnostics[0].concept is Concept.INTEGER_TYPES
    # Scanning goes on after the error: ';' is still there.
    assert tokens[-1].value == ";"


def test_mixed_case_ll_suffix_is_invalid() -> None:
    _, diagnostics = lex("5lL")
    assert [d.code for d in diagnostics] == ["invalid_suffix"]


def test_invalid_octal_digit() -> None:
    _, diagnostics = lex("int m = 09;")
    assert diagnostics[0].code == "invalid_number"
    assert diagnostics[0].message == 'invalid digit "9" in octal constant'
    assert "octal" in diagnostics[0].friendly


def test_hex_without_digits() -> None:
    _, diagnostics = lex("0x;")
    assert diagnostics[0].message == 'invalid suffix "x" on integer constant'


# --- floating literals ------------------------------------------------------------------

@pytest.mark.parametrize("text, value", [
    ("1.5", 1.5), ("1e3", 1000.0), (".5", 0.5), ("2.", 2.0), ("1.5e-2", 0.015), ("3E+2", 300.0),
])
def test_float_literals_are_double(text: str, value: float) -> None:
    token = only(text)
    assert token.kind is TokenKind.FLOAT
    assert token.value == pytest.approx(value)
    assert token.ctype == DOUBLE


def test_float_suffix_f_gives_float_rounded_to_single() -> None:
    token = only("0.1f")
    assert token.ctype == FLOAT
    assert token.value != 0.1  # the float nearest 0.1 is not the double nearest 0.1
    assert token.value == pytest.approx(0.1, rel=1e-7)


def test_long_double_suffix_is_double() -> None:
    assert only("2.5L").ctype == DOUBLE


def test_float_with_bad_suffix() -> None:
    _, diagnostics = lex("1.5x")
    assert diagnostics[0].message == 'invalid suffix "x" on floating constant'


def test_exponent_without_digits() -> None:
    _, diagnostics = lex("1e+;")
    assert diagnostics[0].code == "invalid_number"
    assert diagnostics[0].message == "exponent has no digits"


# --- character literals -----------------------------------------------------------------

@pytest.mark.parametrize("text, value", [
    ("'a'", 97), ("'\\n'", 10), ("'\\t'", 9), ("'\\\\'", 92), ("'\\''", 39), ("'\\\"'", 34),
    ("'\\0'", 0), ("'\\r'", 13), ("'\\a'", 7), ("'\\b'", 8), ("'\\x41'", 65), ("'\\101'", 65),
    ("'\"'", 34), ("'\\x7f'", 127),
])
def test_char_escapes(text: str, value: int) -> None:
    token = only(text)
    assert token.kind is TokenKind.CHAR
    assert token.value == value
    assert token.ctype == INT  # in C a character constant has type int


def test_high_char_is_negative_because_char_is_signed() -> None:
    assert only("'\\xff'").value == -1


def test_multichar_constant_warns_and_packs() -> None:
    tokens, diagnostics = lex("'ab'")
    assert tokens[0].value == 0x6162
    assert [(d.code, d.severity) for d in diagnostics] == [("multichar_constant", Severity.WARNING)]


def test_empty_char_constant() -> None:
    _, diagnostics = lex("char c = '';")
    assert diagnostics[0].code == "empty_char_constant"
    assert diagnostics[0].message == "empty character constant"
    assert diagnostics[0].concept is Concept.STRINGS


def test_unterminated_char() -> None:
    _, diagnostics = lex("char c = 'a;")
    assert diagnostics[0].code == "unterminated_char"
    assert diagnostics[0].message == "missing terminating ' character"
    assert str(diagnostics[0].location) == "main.c:1:10"


def test_unknown_escape_is_a_warning_with_the_letter() -> None:
    tokens, diagnostics = lex("'\\q'")
    assert tokens[0].value == ord("q")
    assert diagnostics[0].code == "invalid_escape"
    assert diagnostics[0].severity is Severity.WARNING
    assert diagnostics[0].message == "unknown escape sequence: '\\q'"


def test_hex_escape_without_digits_is_an_error() -> None:
    _, diagnostics = lex('"\\xg"')
    assert diagnostics[0].code == "invalid_escape"
    assert diagnostics[0].severity is Severity.ERROR


# --- string literals --------------------------------------------------------------------

def test_string_value_is_bytes_with_escapes_decoded() -> None:
    token = only('"hi\\n\\tthere"')
    assert token.kind is TokenKind.STRING
    assert token.value == b"hi\n\tthere"
    assert token.text == '"hi\\n\\tthere"'


def test_string_keeps_embedded_nul() -> None:
    assert only('"a\\0b"').value == b"a\x00b"


def test_string_octal_and_hex_escapes() -> None:
    assert only('"\\101\\x42\\7"').value == b"AB\x07"


def test_string_non_ascii_becomes_utf8() -> None:
    assert only('"café"').value == "café".encode()


def test_unterminated_string_reports_at_the_quote_and_continues() -> None:
    tokens, diagnostics = lex('printf("hello);\nreturn 0;')
    assert [d.code for d in diagnostics] == ["unterminated_string"]
    assert diagnostics[0].message == 'missing terminating " character'
    assert str(diagnostics[0].location) == "main.c:1:8"
    assert tokens[2].value == b"hello);"
    assert [t.value for t in tokens[3:]] == ["return", 0, ";"]
    assert tokens[3].after_error  # the parser will not pile on here


# --- stray characters -------------------------------------------------------------------

def test_stray_backslash() -> None:
    _, diagnostics = lex("x = \\ 2;")
    assert diagnostics[0].code == "stray_character"
    assert diagnostics[0].message == "stray '\\' in program"
    assert str(diagnostics[0].location) == "main.c:1:5"
    assert diagnostics[0].concept is Concept.SYNTAX


def test_each_stray_character_is_reported_and_lexing_continues() -> None:
    tokens, diagnostics = lex("a @ b $ c")
    assert [t.value for t in tokens] == ["a", "b", "c"]
    assert [str(d.location) for d in diagnostics] == ["main.c:1:3", "main.c:1:7"]
    assert not tokens[0].after_error
    assert tokens[1].after_error and tokens[2].after_error


def test_curly_quote_gets_a_friendly_hint() -> None:
    _, diagnostics = lex("printf(“hi”);")
    assert diagnostics[0].code == "stray_character"
    assert "curly quote" in diagnostics[0].friendly


def test_every_diagnostic_has_code_message_friendly_and_concept() -> None:
    _, diagnostics = lex("''\n'a\n\"b\n12xyz 09 1e @ '\\q'")
    assert len(diagnostics) == 8
    for diag in diagnostics:
        assert diag.code and diag.message and diag.friendly
        assert isinstance(diag.concept, Concept)
        assert diag.location is not None and diag.location.column > 0


# --- tolerant mode ------------------------------------------------------------------------

def test_tolerant_keeps_comments_with_positions() -> None:
    tokens = tokenize_tolerant("int x; // the count\n/* a\nb */ int y;")
    comments = [t for t in tokens if t.kind is TokenKind.COMMENT]
    assert [c.text for c in comments] == ["// the count", "/* a", "b */"]
    assert (comments[0].loc.line, comments[0].loc.column) == (1, 8)
    assert comments[2].loc.line == 3


def test_tolerant_marks_errors_as_tokens() -> None:
    tokens = tokenize_tolerant('x = @;\ns = "open')
    errors = [t for t in tokens if t.kind is TokenKind.ERROR]
    assert [e.text for e in errors] == ["@", '"open']
    assert errors[1].loc.line == 2


def test_tolerant_directive_and_header() -> None:
    tokens = tokenize_tolerant("#include <stdio.h>\n#define N 3")
    assert tokens[0].kind is TokenKind.DIRECTIVE and tokens[0].text == "#include"
    assert tokens[1].kind is TokenKind.STRING and tokens[1].text == "<stdio.h>"
    assert tokens[2].kind is TokenKind.DIRECTIVE and tokens[2].text == "#define"
    assert [t.text for t in tokens[3:]] == ["N", "3"]


def test_tolerant_never_raises_on_garbage() -> None:
    tokens = tokenize_tolerant("\u0000☃ '\\ \"\\")
    assert all(isinstance(t, Token) for t in tokens)
    assert tokens[-1].kind is not TokenKind.EOF


def test_tolerant_turns_bad_numbers_into_error_tokens() -> None:
    tokens = tokenize_tolerant("0x 09 12abc 7")
    assert [t.kind for t in tokens] == [TokenKind.ERROR] * 3 + [TokenKind.INT]
