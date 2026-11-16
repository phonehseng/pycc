"""Review tests for the lexer: literal typing, escapes and non-ASCII input."""
from __future__ import annotations

import pytest

from pycc.core.diagnostics import Diagnostic, Severity
from pycc.interp.c_types import INT, LONG, UINT, ULLONG, ULONG
from pycc.interp.lexer import Token, TokenKind, tokenize_text, tokenize_tolerant


def lex(text: str) -> tuple[list[Token], list[Diagnostic]]:
    """Tokenize ``text`` and return the tokens and the diagnostics."""
    diagnostics: list[Diagnostic] = []
    return tokenize_text(text, diagnostics=diagnostics), diagnostics


@pytest.mark.parametrize(("spelling", "ctype"), [
    ("2147483647", INT),
    ("2147483648", LONG),          # decimal: signed types only
    ("0x7FFFFFFF", INT),
    ("0x80000000", UINT),          # hex: unsigned int fits first
    ("0xFFFFFFFF", UINT),
    ("4294967296", LONG),
    ("0x100000000", LONG),
    ("0xFFFFFFFFFFFFFFFF", ULONG),
    ("18446744073709551615u", ULONG),
    ("037777777777", UINT),        # octal behaves like hex
])
def test_integer_literal_types(spelling: str, ctype: object) -> None:
    tokens, diagnostics = lex(spelling)
    assert tokens[0].ctype == ctype
    assert not diagnostics


def test_decimal_too_big_for_signed_warns_and_is_unsigned() -> None:
    tokens, diagnostics = lex("18446744073709551615")
    assert tokens[0].value == 18446744073709551615
    assert tokens[0].ctype == ULLONG
    assert [d.code for d in diagnostics] == ["integer_overflow_constant"]
    assert diagnostics[0].severity is Severity.WARNING


def test_literal_beyond_unsigned_long_long_is_an_error() -> None:
    _, diagnostics = lex("18446744073709551616")
    assert diagnostics[0].code == "invalid_number"
    assert diagnostics[0].message == "integer constant is too large for its type"


def test_string_with_embedded_nul_keeps_every_byte() -> None:
    tokens, diagnostics = lex(r'"a\0b\x41"')
    assert tokens[0].value == b"a\x00bA"
    assert not diagnostics


def test_hex_escape_is_greedy_and_out_of_range_warns() -> None:
    tokens, diagnostics = lex(r'"\x41BC"')
    assert tokens[0].value == bytes([0xBC])
    assert diagnostics[0].severity is Severity.WARNING
    # The hint explains greediness and how to split the string.
    assert '"\\x41" "BC"' in diagnostics[0].friendly


def test_accented_identifier_is_a_stray_character_not_a_crash() -> None:
    tokens, diagnostics = lex("int café = 1;")
    assert [d.code for d in diagnostics] == ["stray_character"]
    assert diagnostics[0].location.column == 8
    assert "a-z" in diagnostics[0].friendly
    assert tokens[1].value == "caf"


def test_superscript_digit_does_not_crash() -> None:
    _, diagnostics = lex("int x = ²;")
    assert [d.code for d in diagnostics] == ["stray_character"]


def test_non_ascii_digit_is_not_a_number() -> None:
    tokens, diagnostics = lex("int x = ٣;")
    assert diagnostics and diagnostics[0].code == "stray_character"
    assert all(token.kind is not TokenKind.INT for token in tokens)


def test_tolerant_lexer_keeps_colouring_after_odd_line() -> None:
    tokens = tokenize_tolerant("int café;\nint ok;")
    second_line = [token for token in tokens if token.loc.line == 2]
    assert [token.text for token in second_line] == ["int", "ok", ";"]
    assert any(token.kind is TokenKind.ERROR for token in tokens)


def test_eof_token_sits_after_last_character() -> None:
    tokens, _ = lex("int x;\nx")
    assert tokens[-1].kind is TokenKind.EOF
    assert (tokens[-1].loc.line, tokens[-1].loc.column) == (2, 2)
