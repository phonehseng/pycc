"""The C lexer: turns preprocessed lines into tokens.

The lexer works line by line over ``PPLine`` records, so every token knows the file,
line and column the user actually typed. It never stops at the first mistake: an
error is recorded as a gcc-style diagnostic and scanning continues right after the bad
spot, so one typo does not hide the rest of the file.

Two entry points:

* ``tokenize(lines, diagnostics)`` for the compiler. Comments are skipped (normally the
  preprocessor already removed them), errors go into ``diagnostics``.
* ``tokenize_tolerant(text)`` for the editor's syntax colouring. It never raises and
  keeps comments, directives and errors as tokens of their own, with positions.

Columns are 1-based character positions in the line; ``end_column`` is the column just
after the token's last character, which is where gcc points for a missing ';'.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation
from pycc.interp.c_types import (
    CHAR, DOUBLE, FLOAT, INT, LLONG, LONG, UINT, ULLONG, ULONG, CType, convert_value, wrap_int,
)
from pycc.interp.friendly import make_diagnostic
from pycc.interp.source import PPLine, lines_from_text


class TokenKind(StrEnum):
    """What a token is. COMMENT, DIRECTIVE and ERROR appear only in tolerant mode."""

    IDENT = "ident"
    KEYWORD = "keyword"
    INT = "int"
    FLOAT = "float"
    CHAR = "char"
    STRING = "string"
    PUNCT = "punct"
    EOF = "eof"
    COMMENT = "comment"
    DIRECTIVE = "directive"
    ERROR = "error"


@dataclass(frozen=True)
class Token:
    """One token.

    ``value`` depends on the kind: the spelling for IDENT/KEYWORD/PUNCT, a Python int
    for INT and CHAR, a float for FLOAT and ``bytes`` (without the final NUL) for
    STRING. Numeric tokens also carry the C type the literal gets, its suffix and base.
    ``after_error`` marks the first token after a lexer error, so the parser does not
    pile a second complaint onto the same mistake.
    """

    kind: TokenKind
    value: object
    loc: SourceLocation
    end_column: int
    text: str
    ctype: CType | None = None
    suffix: str = ""
    base: int = 10
    after_error: bool = False

    @property
    def end_loc(self) -> SourceLocation:
        """The position just after the token, on the same line."""
        return SourceLocation(self.loc.file, self.loc.line, self.end_column)

    def is_punct(self, *spellings: str) -> bool:
        """True if this is a punctuator with one of the given spellings."""
        return self.kind is TokenKind.PUNCT and self.value in spellings

    def is_keyword(self, *words: str) -> bool:
        """True if this is a keyword with one of the given spellings."""
        return self.kind is TokenKind.KEYWORD and self.value in words


KEYWORDS: frozenset[str] = frozenset({
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool",
    "_Complex", "_Imaginary",
})

# Longest first, so that '<<=' wins over '<<' and '<'.
PUNCTUATORS: tuple[str, ...] = (
    "...", "<<=", ">>=",
    "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=",
    "[", "]", "(", ")", "{", "}", ".", "&", "*", "+", "-", "~", "!", "/", "%",
    "<", ">", "^", "|", "?", ":", ";", "=", ",",
)

SIMPLE_ESCAPES: dict[str, int] = {
    "n": 10, "t": 9, "r": 13, "a": 7, "b": 8, "f": 12, "v": 11,
    "\\": 92, "'": 39, '"': 34, "?": 63,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# re.ASCII: without it Python's \d also matches non-ASCII digits, which C rejects.
_FLOAT_RE = re.compile(r"(\d+\.\d*|\.\d+)([eE][+-]?\d*)?|\d+([eE][+-]?\d*)", re.ASCII)
_DIGITS = frozenset("0123456789")
_VALID_INT_SUFFIXES = frozenset({"", "u", "l", "ul", "lu", "ll", "ull", "llu"})


def _is_ident_start(char: str) -> bool:
    """True for the characters a C name may start with (ASCII letters and '_').

    str.isalpha() alone is not enough: it accepts accented letters that C rejects.
    """
    return char.isascii() and (char.isalpha() or char == "_")


def _is_pp_number_char(text: str, j: int) -> bool:
    """True if ``text[j]`` can continue a preprocessing number (ASCII only)."""
    char = text[j]
    return char.isascii() and (char.isalnum() or char in "_.")


class _Scanner:
    """Shared scanning machinery for the strict and the tolerant lexer."""

    def __init__(self, tolerant: bool) -> None:
        self.tolerant = tolerant
        self.tokens: list[Token] = []
        self.diagnostics: list[Diagnostic] = []
        self.in_comment: SourceLocation | None = None   # where an open /* started
        # Set while scanning one token: did that token contain an error?
        self._token_had_error = False
        # Set after an error: the next token emitted gets ``after_error``.
        self.flag_next = False
        self.file = ""
        self.line = 0
        self.text = ""

    # --- reporting -------------------------------------------------------------
    def _loc(self, index: int) -> SourceLocation:
        return SourceLocation(self.file, self.line, index + 1)

    def report(self, code: str, message: str, index: int, *,
               severity: Severity = Severity.ERROR, **details: object) -> None:
        """Record a lexer diagnostic at character ``index`` of the current line."""
        if severity is Severity.ERROR:
            self._token_had_error = True
        if self.tolerant:
            return
        self.diagnostics.append(make_diagnostic(
            code, message, self._loc(index), severity=severity, phase=Phase.PARSE, **details))

    def emit(self, kind: TokenKind, value: object, start: int, end: int, **extra: object) -> None:
        """Append a token covering ``text[start:end]``.

        In tolerant mode a token that contained an error becomes an ERROR token, so the
        editor can underline exactly that span.
        """
        if self.tolerant and self._token_had_error:
            kind = TokenKind.ERROR
        self.tokens.append(Token(kind, value, self._loc(start), end + 1, self.text[start:end],
                                 after_error=self.flag_next, **extra))  # type: ignore[arg-type]
        self.flag_next = self._token_had_error
        self._token_had_error = False

    # --- the main loop ----------------------------------------------------------
    def scan_line(self, text: str, file: str, line: int) -> None:
        """Tokenize one line, continuing a block comment from earlier lines if open."""
        self.text, self.file, self.line = text, file, line
        index = 0
        if self.in_comment is not None:
            index = self._continue_comment(0)
        elif self.tolerant:
            index = self._scan_directive_start()
        while index < len(text):
            index = self._scan_one(index)

    def _scan_one(self, i: int) -> int:
        text = self.text
        char = text[i]
        if char.isspace():
            return i + 1
        if text.startswith("//", i):
            if self.tolerant:
                self.emit(TokenKind.COMMENT, text[i:], i, len(text))
            return len(text)
        if text.startswith("/*", i):
            self.in_comment = self._loc(i)
            return self._continue_comment(i, opened_at=i)
        if char in _DIGITS or (char == "." and i + 1 < len(text) and text[i + 1] in _DIGITS):
            return self._scan_number(i)
        if _is_ident_start(char):
            return self._scan_word(i)
        if char == '"':
            return self._scan_string(i)
        if char == "'":
            return self._scan_char(i)
        for punct in PUNCTUATORS:
            if text.startswith(punct, i):
                self.emit(TokenKind.PUNCT, punct, i, i + len(punct))
                return i + len(punct)
        return self._scan_stray(i)

    # --- comments and directives ------------------------------------------------------
    def _continue_comment(self, i: int, opened_at: int | None = None) -> int:
        """Skip to the end of a block comment; ``opened_at`` is where '/*' sits on this line."""
        search_from = i + 2 if opened_at is not None else i
        close = self.text.find("*/", search_from)
        end = len(self.text) if close < 0 else close + 2
        if self.tolerant and end > i:
            self.emit(TokenKind.COMMENT, self.text[i:end], i, end)
        if close >= 0:
            self.in_comment = None
        return end

    def _scan_directive_start(self) -> int:
        """Tolerant mode only: colour '#include' and its <header> as single tokens."""
        stripped = self.text.lstrip()
        if not stripped.startswith("#"):
            return 0
        start = len(self.text) - len(stripped)
        match = re.match(r"#\s*([A-Za-z_]\w*)?", self.text[start:])
        end = start + (match.end() if match else 1)
        self.emit(TokenKind.DIRECTIVE, self.text[start:end], start, end)
        if match and match.group(1) == "include":
            rest = self.text[end:]
            header = re.match(r"\s*(<[^>\n]*>?)", rest)
            if header:
                h_start = end + header.start(1)
                h_end = end + header.end(1)
                self.emit(TokenKind.STRING, header.group(1).encode(), h_start, h_end)
                return h_end
        return end

    # --- identifiers and keywords -------------------------------------------------------
    def _scan_word(self, i: int) -> int:
        match = _IDENT_RE.match(self.text, i)
        assert match is not None  # the caller saw a letter or '_'
        word = match.group(0)
        kind = TokenKind.KEYWORD if word in KEYWORDS else TokenKind.IDENT
        self.emit(kind, word, i, match.end())
        return match.end()

    # --- stray characters -------------------------------------------------------
    def _scan_stray(self, i: int) -> int:
        char = self.text[i]
        self.report("stray_character", f"stray '{char}' in program", i,
                    friendly=_stray_friendly(char))
        if self.tolerant:
            self.emit(TokenKind.ERROR, char, i, i + 1)
        self.flag_next = True
        self._token_had_error = False
        return i + 1

    # --- numbers --------------------------------------------------------------------
    def _scan_number(self, i: int) -> int:
        """Scan a 'preprocessing number' the way cpp does, then classify it.

        cpp first grabs the longest run of digits, letters, '.', and 'e+'/'p-' pairs,
        and only then decides what it means. That is why gcc says
        'invalid suffix "abc" on integer constant' for 123abc instead of two tokens.
        """
        text = self.text
        j = i
        while j < len(text):
            if text[j] in "eEpP" and j + 1 < len(text) and text[j + 1] in "+-":
                j += 2
            elif _is_pp_number_char(text, j):
                j += 1
            else:
                break
        spelling = text[i:j]
        if spelling[:2].lower() == "0x":
            self._classify_hex(spelling, i, j)
        else:
            self._classify_decimal(spelling, i, j)
        return j

    def _classify_hex(self, spelling: str, start: int, end: int) -> None:
        match = re.match(r"0[xX]([0-9a-fA-F]*)", spelling)
        assert match is not None
        digits = match.group(1)
        suffix = spelling[match.end():]
        if not digits:
            self.report("invalid_suffix", f'invalid suffix "{spelling[1:]}" on integer constant',
                        start, suffix=spelling[1:])
            self.emit(TokenKind.INT, 0, start, end, ctype=INT, base=16)
            return
        if "." in suffix or suffix[:1] in ("p", "P"):
            self.report("invalid_number", "hexadecimal floating constants are not supported",
                        start, detail="Write decimal numbers like 1.5 for floating point values.")
            self.emit(TokenKind.FLOAT, 0.0, start, end, ctype=DOUBLE, base=16)
            return
        self._finish_integer(int(digits, 16), 16, suffix, start, end)

    def _classify_decimal(self, spelling: str, start: int, end: int) -> None:
        match = _FLOAT_RE.match(spelling)
        if match is not None and ("." in match.group(0) or match.group(2) or match.group(3)):
            self._finish_float(spelling, match.group(0), start, end)
            return
        digits_match = re.match(r"[0-9]+", spelling)
        assert digits_match is not None  # a pp-number starting with a digit
        digits = digits_match.group(0)
        suffix = spelling[len(digits):]
        if len(digits) > 1 and digits.startswith("0"):
            bad = next((d for d in digits if d in "89"), None)
            if bad is not None:
                self.report("invalid_number", f'invalid digit "{bad}" in octal constant', start,
                            detail="A number starting with 0 is octal (base 8), so it may only "
                                   "use digits 0-7. Remove the leading 0.")
                self.emit(TokenKind.INT, int(digits, 10), start, end, ctype=INT, base=8)
                return
            self._finish_integer(int(digits, 8), 8, suffix, start, end)
            return
        self._finish_integer(int(digits, 10), 10, suffix, start, end)

    def _finish_float(self, spelling: str, number: str, start: int, end: int) -> None:
        suffix = spelling[len(number):]
        if re.search(r"[eE][+-]?$", number):
            self.report("invalid_number", "exponent has no digits", start,
                        detail="After 'e' in a number, write the power of ten, like 1e3.")
            number = re.sub(r"[eE][+-]?$", "", number)
        if suffix not in ("", "f", "F", "l", "L"):
            self.report("invalid_suffix", f'invalid suffix "{suffix}" on floating constant',
                        start, suffix=suffix)
        value = float(number)
        ctype: CType = DOUBLE
        if suffix in ("f", "F"):
            ctype = FLOAT
            # Round to single precision now, so 0.1f really is the float nearest 0.1.
            value = float(convert_value(value, DOUBLE, FLOAT))
        self.emit(TokenKind.FLOAT, value, start, end, ctype=ctype, suffix=suffix)

    def _finish_integer(self, value: int, base: int, suffix: str, start: int, end: int) -> None:
        lowered = suffix.lower()
        mixed_ll = "ll" in lowered and "ll" not in suffix and "LL" not in suffix
        if lowered not in _VALID_INT_SUFFIXES or mixed_ll:
            self.report("invalid_suffix", f'invalid suffix "{suffix}" on integer constant',
                        start, suffix=suffix)
            lowered = ""
        unsigned = "u" in lowered
        longs = lowered.count("l")
        ctype = _integer_literal_type(value, base == 10, unsigned, longs)
        if ctype is None:
            self.report("invalid_number", "integer constant is too large for its type", start,
                        detail="This number does not fit in any C integer type (the largest "
                               "is unsigned long long).")
            ctype = ULLONG
            value = wrap_int(value, ULLONG)
        elif base == 10 and not unsigned and not ctype.signed:  # type: ignore[attr-defined]
            self.report("integer_overflow_constant",
                        "integer constant is so large that it is unsigned", start,
                        severity=Severity.WARNING)
        self.emit(TokenKind.INT, value, start, end, ctype=ctype, suffix=suffix, base=base)

    # --- character and string literals -------------------------------------------------
    def _read_body(self, i: int, quote: str) -> tuple[list[int], int, bool]:
        """Read literal contents after the opening quote at ``i``.

        Returns ``(bytes, index after the literal, terminated)``. Escapes are decoded
        here; characters beyond ASCII become their UTF-8 bytes.
        """
        text = self.text
        values: list[int] = []
        j = i + 1
        while j < len(text):
            char = text[j]
            if char == quote:
                return values, j + 1, True
            if char == "\\":
                if j + 1 >= len(text):
                    break  # a backslash at the very end: the literal is unterminated
                decoded, j = self._read_escape(j)
                values.append(decoded)
                continue
            values.extend(char.encode("utf-8"))
            j += 1
        return values, len(text), False

    def _read_escape(self, j: int) -> tuple[int, int]:
        """Decode the escape sequence whose backslash is at ``j``; return (byte, next index)."""
        text = self.text
        char = text[j + 1]
        if char in SIMPLE_ESCAPES:
            return SIMPLE_ESCAPES[char], j + 2
        if char in "01234567":
            match = re.match(r"[0-7]{1,3}", text[j + 1:])
            assert match is not None
            return int(match.group(0), 8) & 0xFF, j + 1 + match.end()
        if char == "x":
            match = re.match(r"[0-9a-fA-F]*", text[j + 2:])
            digits = match.group(0) if match else ""
            if not digits:
                self.report("invalid_escape", "\\x used with no following hex digits", j,
                            char="x")
                return ord("x"), j + 2
            value = int(digits, 16)
            if value > 0xFF:
                # \x is greedy: "\x41BC" is one escape, not 'A' then "BC". Say so,
                # because the generic "unknown escape" text would mislead here.
                self.report("invalid_escape", "hex escape sequence out of range", j,
                            severity=Severity.WARNING,
                            friendly=f"'\\x' takes every hex digit after it, so '\\x{digits}' "
                                     f"is too big for one char. To end the escape early, "
                                     f"split the string: \"\\x{digits[:2]}\" \"{digits[2:]}\".")
            return value & 0xFF, j + 2 + len(digits)
        self.report("invalid_escape", f"unknown escape sequence: '\\{char}'", j,
                    severity=Severity.WARNING, char=char)
        return ord(char) & 0xFF, j + 2

    def _scan_string(self, i: int) -> int:
        values, end, terminated = self._read_body(i, '"')
        if not terminated:
            self.report("unterminated_string", 'missing terminating " character', i)
        self.emit(TokenKind.STRING, bytes(values), i, end)
        return end

    def _scan_char(self, i: int) -> int:
        values, end, terminated = self._read_body(i, "'")
        if not terminated:
            self.report("unterminated_char", "missing terminating ' character", i)
        elif not values:
            self.report("empty_char_constant", "empty character constant", i)
        elif len(values) > 1:
            self.report("multichar_constant", "multi-character character constant", i,
                        severity=Severity.WARNING, text=self.text[i + 1:end - 1])
        self.emit(TokenKind.CHAR, _char_value(values), i, end, ctype=INT)
        return end


def _char_value(values: list[int]) -> int:
    """The int value of a character constant, as gcc computes it.

    A one-byte constant is a (signed) char widened to int, so '\\xff' is -1. Several
    bytes are packed big-endian into an int ('ab' == 0x6162), gcc's documented choice.
    """
    if not values:
        return 0
    if len(values) == 1:
        return wrap_int(values[0], CHAR)
    packed = 0
    for byte in values:
        packed = (packed << 8) | byte
    return wrap_int(packed, INT)


def _integer_literal_type(value: int, decimal: bool, unsigned: bool, longs: int) -> CType | None:
    """The first type in C99's list (6.4.4.1) that can hold ``value``, or None.

    Decimal literals without 'u' only try signed types, which is why 2147483648 is a
    long; hex and octal ones also try unsigned, which is why 0xFFFFFFFF is unsigned int.
    """
    if unsigned:
        candidates = [UINT, ULONG, ULLONG][longs:]
    elif decimal:
        candidates = [INT, LONG, LLONG][longs:]
        candidates.append(ULLONG)  # gcc's fallback, reported with a warning
    else:
        candidates = [INT, UINT, LONG, ULONG, LLONG, ULLONG][2 * longs:]
    for ctype in candidates:
        if value <= ctype.max:  # type: ignore[attr-defined]
            return ctype
    return None


def _stray_friendly(char: str) -> str:
    """A beginner-sized hint for a character C does not know."""
    if char in "\u201c\u201d\u2018\u2019":
        return ("This is a curly quote, probably pasted from a document. C only "
                "understands straight quotes: \" and '.")
    if char == "\\":
        return "A backslash only belongs inside quotes, as in \"\\n\". Delete this one."
    if char == "#":
        return "'#' only starts a line like #include or #define. Delete this one."
    if char.isalpha():
        return (f"C names may only use the letters a-z and A-Z, the digits 0-9 and '_'. "
                f"Replace '{char}' with a plain letter.")
    return f"The character '{char}' has no meaning in C here. Delete it."


def tokenize(lines: Iterable[PPLine], diagnostics: list[Diagnostic] | None = None) -> list[Token]:
    """Tokenize preprocessed lines. Diagnostics are appended to ``diagnostics``.

    The result always ends with one EOF token placed just after the last character.
    """
    scanner = _Scanner(tolerant=False)
    last = PPLine("", "<input>", 1)
    for pp_line in lines:
        scanner.scan_line(pp_line.text, pp_line.file, pp_line.line)
        last = pp_line
    if scanner.in_comment is not None:
        scanner.diagnostics.append(make_diagnostic(
            "unterminated_comment", "unterminated comment", scanner.in_comment))
    end_column = len(last.text) + 1
    eof_loc = SourceLocation(last.file, last.line, end_column)
    scanner.tokens.append(Token(TokenKind.EOF, "", eof_loc, end_column, "",
                                after_error=scanner.flag_next))
    if diagnostics is not None:
        diagnostics.extend(scanner.diagnostics)
    return scanner.tokens


def tokenize_text(text: str, filename: str = "main.c",
                  diagnostics: list[Diagnostic] | None = None) -> list[Token]:
    """Tokenize raw text without preprocessing (handy in tests and tools)."""
    return tokenize(lines_from_text(text, filename), diagnostics)


def tokenize_tolerant(text: str, filename: str = "<editor>") -> list[Token]:
    """Tokenize editor text for colouring. Never raises; no EOF token is added.

    Comments become COMMENT tokens (one per line they cover), '#include'-style words
    become DIRECTIVE tokens and anything malformed becomes an ERROR token.
    """
    scanner = _Scanner(tolerant=True)
    for pp_line in lines_from_text(text, filename):
        # Guard each line on its own: one line we cannot colour must not leave the rest
        # of the file uncoloured.
        try:
            scanner.scan_line(pp_line.text, pp_line.file, pp_line.line)
        except Exception:  # noqa: BLE001 - the editor must keep colouring, whatever we typed
            line_text = pp_line.text
            scanner.tokens.append(Token(TokenKind.ERROR, line_text, SourceLocation(
                filename, pp_line.line, 1), len(line_text) + 1, line_text))
            scanner.in_comment = None
    return scanner.tokens
