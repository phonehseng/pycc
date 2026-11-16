"""The C preprocessor: turns the user's text into ``PPLine`` records.

Like ``cpp`` it works on lines of text, before the lexer ever sees a token. The work
happens in four phases, in the same order the C standard describes:

1. **Splice** - a backslash at the very end of a line glues the next line on. The
   glued line keeps the number of its *first* physical line.
2. **Strip comments** - ``/* ... */`` and ``// ...`` become one space. Text inside
   string and character literals is never treated as a comment. Code after a block
   comment that spans several lines keeps the number of the line it is really on.
3. **Directives** - ``#include``, ``#define``, ``#undef``, ``#if`` and friends,
   ``#pragma once``, ``#error``, ``#warning``, ``#line`` (accepted and ignored).
   Directive lines never produce output.
4. **Macro expansion** - object-like and function-like macros, with arguments that
   may span lines, nested calls and rescanning. A macro is never expanded again
   inside its own expansion (each token carries a "hide set" of the macros that
   produced it), so ``#define X X+1`` gives ``X+1`` instead of looping forever.

Every output line remembers its origin file and line number, so an error in a header
points at the header, not at the ``#include`` line.

Built-in macros: ``__FILE__``, ``__LINE__``, ``__PYCC__`` (1) and ``__DATE__`` /
``__TIME__``, which are fixed strings so that builds are reproducible.

Deliberately NOT supported (a ``#define`` that uses them gets an
``invalid_directive`` error that says so):

* the ``#`` stringizing operator and the ``##`` token-pasting operator;
* variadic macros (``...`` and ``__VA_ARGS__``).

Other simplifications: ``#if`` arithmetic is done in signed 64-bit integers (there is
no unsigned arithmetic), and ``includes`` lists every included file once, in the order
it was first included.
"""
from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Protocol

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation, concept_for_code
from pycc.interp.source import PPLine

MAX_INCLUDE_DEPTH = 32
SYSTEM_INCLUDE_DIR = "/usr/include"

# __DATE__ and __TIME__ are fixed so the same program always builds to the same output
# (tests compare outputs, and make must stay deterministic).
FIXED_DATE = '"Jan  1 2026"'
FIXED_TIME = '"12:00:00"'

PREDEFINED_MACROS: dict[str, str] = {"__PYCC__": "1"}
BUILTIN_NAMES = frozenset({"__FILE__", "__LINE__", "__DATE__", "__TIME__"})

_BUILTIN_FILE = "<built-in>"
# <assert.h> defines ASSERT_FLAG; then assert(e) becomes a call of ASSERT_FUNCTION.
ASSERT_NAME = "assert"
ASSERT_FLAG = "__CV_ASSERT__"
ASSERT_FUNCTION = "__cv_assert"
_CONDITIONAL_DIRECTIVES = frozenset({"if", "ifdef", "ifndef", "elif", "else", "endif"})
_OPENING_DIRECTIVES = frozenset({"if", "ifdef", "ifndef"})


# --------------------------------------------------------------------------------------
# Public data model
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Macro:
    """One ``#define``. ``params`` is ``None`` for object-like macros.

    ``file`` and ``line`` remember where it was defined, so a redefinition warning can
    point at the first definition the way gcc does.
    """

    name: str
    params: list[str] | None
    body: str
    file: str = ""
    line: int = 0

    @property
    def is_function_like(self) -> bool:
        """True for ``#define F(x) ...``, false for ``#define N 10``."""
        return self.params is not None

    def same_definition(self, other: Macro) -> bool:
        """C allows redefining a macro only with an identical definition."""
        return self.params == other.params and self.body == other.body


@dataclass
class PreprocessResult:
    """Everything the preprocessor learned about one translation unit."""

    lines: list[PPLine] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    macros: dict[str, Macro] = field(default_factory=dict)
    # Per included file: was it protected by an include guard or #pragma once?
    guards: dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no error (warnings are fine) was reported."""
        return not any(diagnostic.is_error for diagnostic in self.diagnostics)

    @property
    def text(self) -> str:
        """The output as plain text, handy for tests and for ``cc -E``."""
        return "\n".join(line.text for line in self.lines)


class IncludeResolver(Protocol):
    """Finds the file an ``#include`` names.

    Returns ``(resolved_path, text)`` or ``None`` when no such file exists.
    """

    def resolve(self, name: str, quoted: bool, including_file: str) -> tuple[str, str] | None:
        """Look up ``name`` as written in ``#include "name"`` or ``#include <name>``."""
        ...


class DictResolver:
    """An ``IncludeResolver`` over plain dictionaries, using gcc's search order.

    * ``#include "x.h"``: the directory of the including file, then each ``-I``
      directory, then the system headers.
    * ``#include <x.h>``: the system headers, then each ``-I`` directory.

    ``files`` maps POSIX paths to text. ``system_headers`` maps bare names such as
    ``"stdio.h"`` to text; they resolve to paths under ``/usr/include``.
    """

    def __init__(
        self,
        files: Mapping[str, str] | None = None,
        system_headers: Mapping[str, str] | None = None,
        include_dirs: Iterable[str] | None = None,
    ) -> None:
        # Normalising once means "src/../inc/a.h" and "inc/a.h" are the same key.
        self.files = {posixpath.normpath(path): text for path, text in (files or {}).items()}
        self.system_headers = dict(system_headers or {})
        self.include_dirs = list(include_dirs or [])

    def resolve(self, name: str, quoted: bool, including_file: str) -> tuple[str, str] | None:
        """Search for ``name`` in gcc's order; see the class docstring."""
        if name.startswith("/"):
            return self._lookup_file(name)
        if quoted:
            here = posixpath.dirname(including_file)
            return (
                self._lookup_file(posixpath.join(here, name))
                or self._search_include_dirs(name)
                or self._lookup_system(name)
            )
        return self._lookup_system(name) or self._search_include_dirs(name)

    def _lookup_file(self, path: str) -> tuple[str, str] | None:
        """Find a project file by its normalised path."""
        normalised = posixpath.normpath(path)
        if normalised in self.files:
            return normalised, self.files[normalised]
        return None

    def _search_include_dirs(self, name: str) -> tuple[str, str] | None:
        """Try each ``-I`` directory in the order it was given."""
        for directory in self.include_dirs:
            found = self._lookup_file(posixpath.join(directory, name))
            if found is not None:
                return found
        return None

    def _lookup_system(self, name: str) -> tuple[str, str] | None:
        """Find a simulated system header such as ``stdio.h``."""
        if name in self.system_headers:
            return posixpath.join(SYSTEM_INCLUDE_DIR, name), self.system_headers[name]
        return None


def preprocess(
    text: str,
    filename: str,
    resolver: IncludeResolver,
    defines: Mapping[str, str | None] | None = None,
) -> PreprocessResult:
    """Preprocess one translation unit.

    ``defines`` holds command-line ``-D`` macros: ``{"DEBUG": "1"}``; a ``None`` value
    means ``-DNAME`` with no value, which gcc defines as 1. A key may carry
    parameters, as in ``{"SQ(x)": "((x)*(x))"}``.
    """
    return _Preprocessor(resolver, defines).run(text, filename)


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------

def _make_diagnostic(
    severity: Severity,
    code: str,
    message: str,
    friendly: str,
    file: str | None,
    line: int,
    notes: Iterable[str] = (),
    column: int = 0,
) -> Diagnostic:
    """Build a preprocessor diagnostic; codes not in CODE_CONCEPTS count as preprocessor."""
    location = SourceLocation(file, line, column) if file else None
    return Diagnostic(
        severity=severity,
        code=code,
        message=message,
        friendly=friendly,
        location=location,
        concept=concept_for_code(code, Concept.PREPROCESSOR),
        phase=Phase.PREPROCESS,
        notes=list(notes),
    )


# Headers of POSIX (Unix), not of standard C, with what to do instead.
POSIX_HEADERS: dict[str, str] = {
    "unistd.h": "There is no getopt, sleep or read here: parse argv by hand, comparing "
                "each argv[i] with strcmp.",
    "getopt.h": "There is no getopt here: parse argv by hand, comparing each argv[i] "
                "with strcmp.",
    "strings.h": "For strcasecmp, compare with strcmp after tolower() from <ctype.h>.",
    "sys/types.h": "The standard types are in <stddef.h> (size_t) and <stdint.h>.",
    "sys/stat.h": "Files are opened with fopen from <stdio.h>; it fails when a file "
                  "does not exist.",
    "fcntl.h": "Open files with fopen from <stdio.h> instead of open.",
    "dirent.h": "There is no way to list folders from C here; read a file of names "
                "with fopen instead.",
    "pthread.h": "pycc runs one thread: write the work as a loop.",
    "sys/time.h": "For time use <time.h>.",
}


class _DirectiveError(Exception):
    """A malformed directive; becomes an ``invalid_directive`` diagnostic.

    ``unsupported`` marks C that is valid but beyond this preprocessor (``#``, ``##``,
    variadic macros). Those stop the file, as a fatal error does: every later use of
    the macro would only add confusing follow-on errors.
    """

    def __init__(self, message: str, friendly: str, unsupported: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.friendly = friendly
        self.unsupported = unsupported


class _ExpressionError(Exception):
    """An ``#if`` condition the evaluator could not compute."""

    def __init__(self, message: str, code: str = "invalid_directive", friendly: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.friendly = friendly or (
            "The preprocessor could not read this #if condition. It understands whole "
            "numbers, character constants like 'a', defined(NAME), macros and the usual "
            "C operators."
        )


class _FatalStop(Exception):
    """Raised after a fatal error: like gcc, preprocessing stops right there."""


# --------------------------------------------------------------------------------------
# Phases 1 and 2: splicing and comment removal
# --------------------------------------------------------------------------------------

def _include_name_column(text: str) -> int:
    """The 1-based column of the ``<`` or ``"`` of an ``#include`` line (0 if none).

    gcc points a missing header at its name: ``main.c:1:10: fatal error: device.h: No
    such file or directory`` for ``#include <device.h>``.
    """
    positions = [index for index in (text.find("<"), text.find('"')) if index >= 0]
    return min(positions) + 1 if positions else 0


@dataclass
class _Line:
    """A logical line of one file, with the number of the line it starts on."""

    text: str
    line: int


def _splice_lines(text: str) -> list[_Line]:
    """Join lines ending in a backslash with the next one, keeping the first number."""
    physical = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result: list[_Line] = []
    pieces: list[str] = []
    start = 0
    for number, raw in enumerate(physical, start=1):
        if not pieces:
            start = number
        # Editors often leave invisible spaces after the backslash; gcc still splices
        # (with a warning), so a student's macro does not silently break in two.
        without_blanks = raw.rstrip(" \t")
        if without_blanks.endswith("\\"):
            pieces.append(without_blanks[:-1])
            continue
        pieces.append(raw)
        result.append(_Line("".join(pieces), start))
        pieces = []
    if pieces:  # the file ended with a backslash
        result.append(_Line("".join(pieces), start))
    return result


class _CommentStripper:
    """Replace each comment by one space, across lines, ignoring quotes' contents.

    When a block comment spans lines, the code after ``*/`` normally becomes its own
    line with its real line number, so errors there point at the right line. A
    directive is the exception: it must stay a single line, so its tail is glued back
    onto the directive (as cpp does).
    """

    def __init__(self) -> None:
        self.output: list[_Line] = []
        self.in_block = False
        self.block_start = 0
        self.pieces: list[str] = []
        self.current_line = 0

    def run(self, lines: list[_Line]) -> tuple[list[_Line], int | None]:
        """Return the stripped lines and the start line of an unclosed comment, if any."""
        for line in lines:
            if not (self.in_block and self._is_directive_so_far()):
                self.pieces = []
                self.current_line = line.line
            self._scan(line)
            if self.in_block and self._is_directive_so_far():
                continue  # keep gluing the directive together
            self.output.append(_Line("".join(self.pieces), self.current_line))
        if self.in_block:
            if self._is_directive_so_far():
                self.output.append(_Line("".join(self.pieces), self.current_line))
            return self.output, self.block_start
        return self.output, None

    def _is_directive_so_far(self) -> bool:
        """Is the line being built a directive (so a comment must not split it)?"""
        return "".join(self.pieces).lstrip().startswith("#")

    def _scan(self, line: _Line) -> None:
        """Copy one line into ``pieces``, dropping comment text."""
        text = line.text
        index = 0
        quote: str | None = None
        while index < len(text):
            if self.in_block:
                end = text.find("*/", index)
                if end < 0:
                    return
                self.in_block = False
                index = end + 2
                continue
            char = text[index]
            if quote is not None:
                index = self._copy_quoted(text, index, quote)
                if char == quote:
                    quote = None
                continue
            if char in "\"'":
                quote = char
                self.pieces.append(char)
                index += 1
            elif text.startswith("//", index):
                self.pieces.append(" ")
                return
            elif text.startswith("/*", index):
                self.pieces.append(" ")
                self.in_block = True
                self.block_start = line.line
                index += 2
            else:
                self.pieces.append(char)
                index += 1

    def _copy_quoted(self, text: str, index: int, quote: str) -> int:
        """Copy one character inside a literal; an escape copies two (so \\" stays)."""
        if text[index] == "\\" and index + 1 < len(text):
            self.pieces.append(text[index:index + 2])
            return index + 2
        self.pieces.append(text[index])
        return index + 1


def _strip_comments(lines: list[_Line]) -> tuple[list[_Line], int | None]:
    """Phase 2: see ``_CommentStripper``."""
    return _CommentStripper().run(lines)


def _is_directive(text: str) -> bool:
    """Does this logical line start with '#' (after optional blanks)?"""
    return text.lstrip().startswith("#")


_DIRECTIVE_NAME = re.compile(r"\s*([A-Za-z_]\w*|\d+)?")


def _split_directive(text: str) -> tuple[str, str]:
    """``'  # define X 1'`` -> ``('define', 'X 1')``. A bare ``#`` gives ``('', '')``."""
    body = text.lstrip()[1:]
    match = _DIRECTIVE_NAME.match(body)  # every part is optional, so this always matches
    assert match is not None
    return match.group(1) or "", body[match.end():].strip()


# --------------------------------------------------------------------------------------
# Preprocessing tokens
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Token:
    """A preprocessing token.

    ``hide`` is the set of macros whose expansion produced this token; such a macro
    name is never expanded again (the recursion guard). ``expanded`` marks tokens that
    came out of a macro, so the joiner knows where to add a separating space.
    """

    kind: str
    text: str
    hide: frozenset[str] = frozenset()
    expanded: bool = False


_PP_TOKEN = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<ident>[A-Za-z_]\w*)
    | (?P<number>\.?\d(?:[eEpP][+-]|[\w.])*)
    | (?P<string>"(?:\\.|[^"\\])*"?)
    | (?P<char>'(?:\\.|[^'\\])*'?)
    | (?P<punct>
          # Multi-character operators must stay one token: if '==' were two '='
          # tokens, the joiner would separate them and 'a==b' would become 'a= =b'.
          ->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|\#\#|\.\.\.|.
      )
    """,
    re.VERBOSE | re.DOTALL,
)


@lru_cache(maxsize=4096)
def _tokenize(text: str) -> tuple[_Token, ...]:
    """Split text into preprocessing tokens, whitespace included (so text round-trips)."""
    return tuple(_Token(match.lastgroup or "punct", match.group()) for match in _PP_TOKEN.finditer(text))


# Two tokens that would melt into one token if written side by side.
_PASTING_PAIRS = frozenset({
    "++", "--", "+=", "-=", "->", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "&=", "|=", "^=", "*=", "/=", "%=", "//", "/*", "##", "..",
})


def _is_word_char(char: str) -> bool:
    """Letters, digits and '_' glue together into one identifier or number."""
    return char.isalnum() or char == "_"


def _would_paste(left: str, right: str) -> bool:
    """Would ``left`` followed directly by ``right`` lex as something else?"""
    last, first = left[-1], right[0]
    if _is_word_char(last) and _is_word_char(first):
        return True
    if (last.isdigit() and first == ".") or (last == "." and first.isdigit()):
        return True
    return last + first in _PASTING_PAIRS


def _join_tokens(tokens: Iterable[_Token]) -> str:
    """Turn tokens back into text.

    Where a macro's output touches its neighbours we add a space if they would
    otherwise merge: ``-NEG`` with ``#define NEG -1`` must become ``- -1``, not the
    decrement operator ``--1``.
    """
    parts: list[str] = []
    previous: _Token | None = None
    for token in tokens:
        if (
            previous is not None
            and "ws" not in (token.kind, previous.kind)
            and (token.expanded or previous.expanded)
            and _would_paste(previous.text, token.text)
        ):
            parts.append(" ")
        parts.append(token.text)
        previous = token
    return "".join(parts)


def _c_string(text: str) -> str:
    """Quote text as a C string literal (for __FILE__)."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --------------------------------------------------------------------------------------
# Phase 4: macro expansion
# --------------------------------------------------------------------------------------

@dataclass
class _ExpandContext:
    """Where an expansion happens, plus the diagnostics it produced.

    Diagnostics are collected here instead of reported straight away because a line
    may be expanded twice: once to discover that a macro call continues on the next
    line, then again with that line glued on. Only the final attempt's are kept.
    """

    file: str
    line: int
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def report(self, code: str, message: str, friendly: str) -> None:
        """Record an error at this context's position."""
        self.diagnostics.append(
            _make_diagnostic(Severity.ERROR, code, message, friendly, self.file, self.line)
        )


class _NeedMoreInput(Exception):
    """A function-like macro call may continue on the next source line.

    ``definite`` is True when an argument list is open (``F(1,`` at line end) and False
    when only the macro name ended the line (``F`` - a ``(`` may follow).
    ``open_parens`` says how many ``)`` are still missing, so the caller can glue on
    lines until they could have arrived instead of re-expanding after every line
    (which made a forgotten ``)`` in a long file quadratically slow).
    """

    def __init__(self, definite: bool, open_parens: int = 0) -> None:
        super().__init__("macro call continues on the next line")
        self.definite = definite
        self.open_parens = open_parens


def _with_hide(token: _Token, hide: frozenset[str]) -> _Token:
    """Copy a token produced by a macro, adding that macro to its hide set."""
    return _Token(token.kind, token.text, token.hide | hide, True)


# The shape assert(e) is read with: one argument, like the real macro.
ASSERT_MACRO = Macro(ASSERT_NAME, ["expression"], "", "/usr/include/assert.h")


class _MacroExpander:
    """Expands macros in token lists, using hide sets to stop recursion."""

    def __init__(self, macros: dict[str, Macro]) -> None:
        self.macros = macros

    def expand_text(self, text: str, context: _ExpandContext, allow_more: bool) -> str:
        """Expand every macro in ``text``; returns the text unchanged if there is none."""
        tokens = _tokenize(text)
        if not any(self._is_expandable(token) for token in tokens):
            return text  # fast path, and keeps the user's columns exactly
        return _join_tokens(self.expand_tokens(tokens, context, allow_more))

    def _is_expandable(self, token: _Token) -> bool:
        """Could this token start a macro expansion?"""
        return token.kind == "ident" and (token.text in self.macros or token.text in BUILTIN_NAMES
                                          or self._is_assert(token))

    def _is_assert(self, token: _Token) -> bool:
        """``assert`` while ``<assert.h>`` is active (and no #define of its own)."""
        return (token.text == ASSERT_NAME and ASSERT_FLAG in self.macros
                and ASSERT_NAME not in self.macros)

    def expand_tokens(
        self, tokens: Iterable[_Token], context: _ExpandContext, allow_more: bool
    ) -> list[_Token]:
        """Expand and rescan. A replacement goes back on the input so it is rescanned."""
        stream = list(reversed(list(tokens)))  # a stack: the next token is at the end
        output: list[_Token] = []
        while stream:
            token = stream.pop()
            replacement = self._replacement_for(token, stream, context, allow_more)
            if replacement is None:
                output.append(token)
            else:
                stream.extend(reversed(replacement))
        return output

    def _replacement_for(
        self, token: _Token, stream: list[_Token], context: _ExpandContext, allow_more: bool
    ) -> list[_Token] | None:
        """The tokens that replace ``token``, or None if it is not expanded."""
        if token.kind != "ident" or token.text in token.hide:
            return None
        macro = self.macros.get(token.text)
        if macro is None and self._is_assert(token):
            return self._assert_call(token, stream, context, allow_more)
        if macro is None:
            builtin = self._builtin(token, context)
            return [builtin] if builtin is not None else None
        if macro.params is None:
            hide = token.hide | {token.text}
            return [_with_hide(part, hide) for part in _tokenize(macro.body)]
        call = self._collect_call(macro, stream, context, allow_more)
        if call is None:
            return None
        arguments, close_paren = call
        # Prosser's rule: only macros that hide BOTH the name and its closing ')' stay
        # hidden. The ')' may come from outside an earlier expansion, and then that
        # earlier macro may expand again (the C standard's f(2)(9) example).
        hide = (token.hide & close_paren.hide) | {token.text}
        # Arguments are fully expanded before they are substituted (C99 6.10.3.1).
        expanded = [self.expand_tokens(argument, context, allow_more=False) for argument in arguments]
        return self._substitute(macro, expanded, hide)

    def _assert_call(self, token: _Token, stream: list[_Token], context: _ExpandContext,
                     allow_more: bool) -> list[_Token] | None:
        """``assert(e)`` -> ``__cv_assert((e), "e", __FILE__, __LINE__, __func__)``.

        A real <assert.h> quotes its argument with the ``#`` operator, which this
        preprocessor does not offer to users; this one rewrite does the same job, so
        the failure message can show the condition exactly as the user wrote it
       . Like any function-like macro, a bare ``assert`` with
        no ``(`` after it stays a plain word.
        """
        call = self._collect_call(ASSERT_MACRO, stream, context, allow_more)
        if call is None:
            return None
        arguments, _close_paren = call
        written = _join_tokens(arguments[0]).strip()
        hide = token.hide | {ASSERT_NAME}
        condition = self.expand_tokens(arguments[0], context, allow_more=False)
        tail = f"), {_c_string(written)}, __FILE__, __LINE__, __func__)"
        parts = [*_tokenize(f"{ASSERT_FUNCTION}(("), *condition, *_tokenize(tail)]
        return [_with_hide(part, hide) for part in parts]

    @staticmethod
    def _builtin(token: _Token, context: _ExpandContext) -> _Token | None:
        """Value of ``__LINE__``, ``__FILE__``, ``__DATE__`` or ``__TIME__``."""
        match token.text:
            case "__LINE__":
                return _Token("number", str(context.line), expanded=True)
            case "__FILE__":
                return _Token("string", _c_string(context.file), expanded=True)
            case "__DATE__":
                return _Token("string", FIXED_DATE, expanded=True)
            case "__TIME__":
                return _Token("string", FIXED_TIME, expanded=True)
        return None

    def _collect_call(
        self, macro: Macro, stream: list[_Token], context: _ExpandContext, allow_more: bool
    ) -> tuple[list[list[_Token]], _Token] | None:
        """Read ``( args )`` after a function-like macro name.

        Returns the arguments plus the closing ``)`` token and removes them from the
        stream, or returns ``None`` (stream untouched) when the name is not a call, so
        it stays a plain word.
        """
        open_index = self._next_significant(stream, len(stream) - 1)
        if open_index < 0:
            if allow_more:
                raise _NeedMoreInput(definite=False)
            return None
        if stream[open_index].text != "(":
            return None
        split = self._split_arguments(stream, open_index)
        if split is None:
            if allow_more:
                raise _NeedMoreInput(definite=True, open_parens=_open_depth(stream, open_index))
            context.report(
                "macro_args_mismatch",
                f'unterminated argument list invoking macro "{macro.name}"',
                f"The call to {macro.name}( is missing its closing ')'. Every '(' "
                "needs a matching ')'.",
            )
            return None
        arguments, close_index = split
        if not self._argument_count_ok(macro, arguments, context):
            return None
        close_paren = stream[close_index]
        del stream[close_index:]
        return (arguments if macro.params else []), close_paren

    @staticmethod
    def _next_significant(stream: list[_Token], start: int) -> int:
        """Index of the next non-whitespace token at or below ``start``."""
        index = start
        while index >= 0 and stream[index].kind == "ws":
            index -= 1
        return index

    @staticmethod
    def _split_arguments(stream: list[_Token], open_index: int) -> tuple[list[list[_Token]], int] | None:
        """Split at top-level commas; commas inside nested parentheses do not count."""
        arguments: list[list[_Token]] = [[]]
        depth = 0
        for index in range(open_index - 1, -1, -1):
            token = stream[index]
            if token.text == "(":
                depth += 1
            elif token.text == ")" and depth == 0:
                return [_strip_ws(argument) for argument in arguments], index
            elif token.text == ")":
                depth -= 1
            elif token.text == "," and depth == 0:
                arguments.append([])
                continue
            arguments[-1].append(token)
        return None

    @staticmethod
    def _argument_count_ok(macro: Macro, arguments: list[list[_Token]], context: _ExpandContext) -> bool:
        """Check the call's argument count; report a mismatch like gcc."""
        expected = len(macro.params or [])
        given = len(arguments)
        if expected == 0 and arguments == [[]]:
            return True  # F() calls a macro with no parameters
        if given == expected:
            return True
        if given < expected:
            message = f'macro "{macro.name}" requires {expected} arguments, but only {given} given'
        else:
            message = f'macro "{macro.name}" passed {given} arguments, but takes just {expected}'
        context.report(
            "macro_args_mismatch",
            message,
            f"{macro.name} was defined with {expected} parameter(s) but called with {given} "
            "argument(s). Arguments are split at commas; wrap an argument that contains a "
            "comma in parentheses.",
        )
        return False

    @staticmethod
    def _substitute(macro: Macro, arguments: list[list[_Token]], hide: frozenset[str]) -> list[_Token]:
        """Put the (already expanded) arguments into the macro body."""
        positions = {name: index for index, name in enumerate(macro.params or [])}
        result: list[_Token] = []
        for token in _tokenize(macro.body):
            if token.kind == "ident" and token.text in positions:
                result.extend(_with_hide(part, hide) for part in arguments[positions[token.text]])
            else:
                result.append(_with_hide(token, hide))
        return result


def _open_depth(stream: list[_Token], open_index: int) -> int:
    """How many ``)`` are missing after the ``(`` at ``open_index`` (the stack top is last)."""
    depth = 0
    for index in range(open_index, -1, -1):
        if stream[index].text == "(":
            depth += 1
        elif stream[index].text == ")":
            depth -= 1
    return max(depth, 1)


def _paren_balance(text: str) -> int:
    """``(`` minus ``)`` in one line, ignoring brackets inside string literals."""
    balance = 0
    for token in _tokenize(text):
        if token.text == "(":
            balance += 1
        elif token.text == ")":
            balance -= 1
    return balance


def _strip_ws(tokens: list[_Token]) -> list[_Token]:
    """Drop leading and trailing whitespace tokens (around a macro argument)."""
    start, end = 0, len(tokens)
    while start < end and tokens[start].kind == "ws":
        start += 1
    while end > start and tokens[end - 1].kind == "ws":
        end -= 1
    return tokens[start:end]


# --------------------------------------------------------------------------------------
# #define parsing
# --------------------------------------------------------------------------------------

_MACRO_HEAD = re.compile(r"([A-Za-z_]\w*)(\()?")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")

_DEFINE_HELP = (
    "A #define needs a name and then its replacement, like '#define WIDTH 10' or "
    "'#define SQ(x) ((x) * (x))'. For a function-like macro the '(' must touch the name."
)


def _parse_define(text: str, file: str, line: int) -> Macro:
    """Parse the text after ``#define``; raises ``_DirectiveError`` if it is malformed."""
    if not text:
        raise _DirectiveError("no macro name given in #define directive", _DEFINE_HELP)
    head = _MACRO_HEAD.match(text)
    if head is None:
        raise _DirectiveError("macro names must be identifiers", _DEFINE_HELP)
    name = head.group(1)
    if name == "defined":
        raise _DirectiveError('"defined" cannot be used as a macro name', _DEFINE_HELP)
    params: list[str] | None = None
    body_start = head.end()
    if head.group(2):
        close = text.find(")", body_start)
        if close < 0:
            raise _DirectiveError("missing ')' in macro parameter list", _DEFINE_HELP)
        params = _parse_params(text[body_start:close])
        body_start = close + 1
    body = _normalise_body(text[body_start:])
    _reject_unsupported_operators(body, params)
    return Macro(name, params, body, file, line)


def _parse_params(inner: str) -> list[str]:
    """Parse the text between the parentheses of ``#define F(a, b)``."""
    if not inner.strip():
        return []
    params: list[str] = []
    for raw in inner.split(","):
        param = raw.strip()
        if param.endswith("..."):
            raise _DirectiveError(
                "variadic macros are not supported",
                "pycc's preprocessor does not support '...' in macro parameters. "
                "Give the macro a fixed number of parameters, or write a function.",
                unsupported=True,
            )
        if not _IDENTIFIER.match(param):
            raise _DirectiveError(f'expected parameter name, found "{param}"', _DEFINE_HELP)
        if param in params:
            raise _DirectiveError(f'duplicate macro parameter "{param}"', _DEFINE_HELP)
        params.append(param)
    return params


def _normalise_body(text: str) -> str:
    """Collapse whitespace runs so redefinition checks ignore spacing (as C requires)."""
    tokens = _tokenize(text.strip())
    return "".join(" " if token.kind == "ws" else token.text for token in tokens)


def _reject_unsupported_operators(body: str, params: list[str] | None) -> None:
    """Refuse ``#`` and ``##`` in a body with a clear message instead of misbehaving."""
    significant = [token.text for token in _tokenize(body)]
    if "##" in significant:  # the tokenizer keeps '##' as one token
        raise _DirectiveError(
            "the '##' operator is not supported",
            "pycc's preprocessor does not support token pasting with ##. "
            "Write the full name out instead.",
            unsupported=True,
        )
    if params is not None and "#" in significant:
        raise _DirectiveError(
            "the '#' operator is not supported",
            "pycc's preprocessor does not support turning a macro argument into a "
            "string with #. Pass the string as an argument instead.",
            unsupported=True,
        )


# --------------------------------------------------------------------------------------
# #if expressions
# --------------------------------------------------------------------------------------

_EXPRESSION_TOKEN = re.compile(
    r"""
      (?P<number>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*(?![\w.])
    | (?P<char>'(?:\\.|[^'\\])+')
    | (?P<ident>[A-Za-z_]\w*)
    | (?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>?:()])
    """,
    re.VERBOSE,
)

_SIMPLE_ESCAPES = {
    "n": 10, "t": 9, "r": 13, "0": 0, "a": 7, "b": 8, "f": 12, "v": 11,
    "\\": 92, "'": 39, '"': 34, "?": 63,
}

# Binding strength of each binary operator; higher binds tighter (C's precedence table).
_BINARY_PRECEDENCE = {
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6,
    "<": 7, "<=": 7, ">": 7, ">=": 7, "<<": 8, ">>": 8,
    "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
}


def _wrap64(value: int) -> int:
    """Keep results in signed 64-bit range, like a real compiler's intmax_t."""
    return (value + 2**63) % 2**64 - 2**63


def _c_divide(left: int, right: int) -> int:
    """C division truncates toward zero; Python's // rounds down."""
    quotient = abs(left) // abs(right)
    return quotient if (left < 0) == (right < 0) else -quotient


def _c_shift_left(left: int, right: int) -> int:
    """Left shift; shifting by 64 or more (undefined in C) gives 0 here."""
    return left << right if 0 <= right < 64 else 0


def _c_shift_right(left: int, right: int) -> int:
    """Arithmetic right shift; an out-of-range count gives 0 or -1."""
    return left >> right if 0 <= right < 64 else (-1 if left < 0 else 0)


_BINARY_OPERATIONS: dict[str, Callable[[int, int], int]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _c_divide,
    "%": lambda a, b: a - _c_divide(a, b) * b,
    "<<": _c_shift_left,
    ">>": _c_shift_right,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "&&": lambda a, b: int(bool(a) and bool(b)),
    "||": lambda a, b: int(bool(a) or bool(b)),
}


def _char_value(literal: str) -> int:
    """Value of a character constant such as 'a', '\\n' or '\\x41'."""
    body = literal[1:-1]
    if not body.startswith("\\"):
        return ord(body[0])
    escape = body[1:]
    if escape[0] in "xX":
        return int(escape[1:], 16) if len(escape) > 1 else 0
    if escape[0] in "01234567":
        return int(escape[:3], 8)
    return _SIMPLE_ESCAPES.get(escape[0], ord(escape[0]))


def _number_value(text: str) -> int:
    """Value of an integer literal (suffix already removed): hex, binary, octal or decimal."""
    lowered = text.lower()
    if lowered.startswith("0x"):
        return int(lowered[2:], 16)
    if lowered.startswith("0b"):
        return int(lowered[2:], 2)
    if len(lowered) > 1 and lowered.startswith("0"):
        # A leading 0 means octal, so "08" is a mistake, not eight (a common surprise).
        bad_digits = [digit for digit in lowered if digit in "89"]
        if bad_digits:
            raise _ExpressionError(
                f'invalid digit "{bad_digits[0]}" in octal constant',
                friendly=f"A number that starts with 0 is octal (base 8), which only "
                f"uses the digits 0-7. Write {lowered.lstrip('0') or '0'} without the "
                "leading zero if you meant a decimal number.",
            )
        return int(lowered, 8)
    return int(lowered)


def _expression_tokens(text: str) -> list[tuple[str, str, int]]:
    """Tokens of an #if expression as ``(kind, text, value)``; names count as 0."""
    tokens: list[tuple[str, str, int]] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return tokens
        match = _EXPRESSION_TOKEN.match(text, index)
        if match is None:
            bad = text[index:].split()[0]
            raise _ExpressionError(f'token "{bad}" is not valid in preprocessor expressions')
        tokens.append(_classify_expression_token(match))
        index = match.end()


def _classify_expression_token(match: re.Match[str]) -> tuple[str, str, int]:
    """Turn one regex match into a ``(kind, text, value)`` triple."""
    kind = match.lastgroup
    text = match.group()
    if kind == "number":
        return "value", text, _number_value(match.group("number"))
    if kind == "char":
        return "value", text, _char_value(text)
    if kind == "ident":
        return "value", text, 0  # C says: identifiers left after expansion are 0
    return "op", text, 0


class _ConditionEvaluator:
    """Recursive-descent evaluator for ``#if`` conditions.

    It evaluates while it parses. The ``live`` flag is False inside a branch that C
    would not evaluate (the right side of ``0 && ...``, the unused arm of ``?:``), so
    ``#if 0 && 1/0`` is fine while ``#if 1/0`` reports division by zero.
    """

    def __init__(self, text: str) -> None:
        self.tokens = _expression_tokens(text)
        self.position = 0

    def evaluate(self) -> int:
        """Compute the whole condition; raises ``_ExpressionError`` on bad input."""
        if not self.tokens:
            raise _ExpressionError("#if with no expression")
        value = self._ternary(live=True)
        if self.position < len(self.tokens):
            raise _ExpressionError(f'missing binary operator before token "{self.tokens[self.position][1]}"')
        return value

    def _peek(self) -> str | None:
        """The next operator, or None if the next token is a value or the end."""
        if self.position < len(self.tokens) and self.tokens[self.position][0] == "op":
            return self.tokens[self.position][1]
        return None

    def _expect(self, operator: str, message: str) -> None:
        """Consume ``operator`` or fail with ``message``."""
        if self._peek() != operator:
            raise _ExpressionError(message)
        self.position += 1

    def _ternary(self, live: bool) -> int:
        """Parse ``cond ? a : b`` (the lowest-precedence form allowed in #if)."""
        condition = self._binary(1, live)
        if self._peek() != "?":
            return condition
        self.position += 1
        when_true = self._ternary(live and condition != 0)
        self._expect(":", "'?' without following ':'")
        when_false = self._ternary(live and condition == 0)
        return when_true if condition else when_false

    def _binary(self, min_precedence: int, live: bool) -> int:
        """Precedence climbing: one loop handles every binary operator level."""
        left = self._unary(live)
        while (operator := self._peek()) in _BINARY_PRECEDENCE:
            precedence = _BINARY_PRECEDENCE[operator]
            if precedence < min_precedence:
                break
            self.position += 1
            short_circuited = (operator == "&&" and left == 0) or (operator == "||" and left != 0)
            right_live = live and not short_circuited
            right = self._binary(precedence + 1, right_live)
            left = self._apply(operator, left, right, right_live)
        return left

    @staticmethod
    def _apply(operator: str, left: int, right: int, live: bool) -> int:
        """Apply a binary operator; division by zero only matters when evaluated."""
        if operator in ("/", "%") and right == 0:
            if live:
                raise _ExpressionError(
                    "division by zero in #if",
                    code="division_by_zero",
                    friendly="This #if condition divides by zero, so the preprocessor "
                    "cannot decide which lines to keep. Check the macros it uses.",
                )
            return 0
        return _wrap64(_BINARY_OPERATIONS[operator](left, right))

    def _unary(self, live: bool) -> int:
        """Parse a value, a parenthesised expression or a unary operator."""
        if self.position >= len(self.tokens):
            raise _ExpressionError("expected a value at the end of the #if expression")
        kind, text, value = self.tokens[self.position]
        self.position += 1
        if kind == "value":
            return value
        if text in ("+", "-", "~", "!"):
            operand = self._unary(live)
            return _wrap64({"+": operand, "-": -operand, "~": ~operand, "!": int(operand == 0)}[text])
        if text == "(":
            inner = self._ternary(live)
            self._expect(")", "missing ')' in expression")
            return inner
        if text in _BINARY_PRECEDENCE:
            raise _ExpressionError(f"operator '{text}' has no left operand")
        raise _ExpressionError(f"unexpected '{text}' in #if expression")


def _replace_defined(tokens: tuple[_Token, ...], is_defined: Callable[[str], bool]) -> list[_Token]:
    """Turn ``defined X`` and ``defined(X)`` into 1 or 0 before macros are expanded."""
    result: list[_Token] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "ident" and token.text == "defined":
            name, index = _read_defined_operand(tokens, index + 1)
            result.append(_Token("number", "1" if is_defined(name) else "0"))
        else:
            result.append(token)
            index += 1
    return result


def _read_defined_operand(tokens: tuple[_Token, ...], index: int) -> tuple[str, int]:
    """Read ``X`` or ``( X )`` after ``defined``; returns the name and the next index."""
    def skip_ws(position: int) -> int:
        """Skip whitespace tokens."""
        while position < len(tokens) and tokens[position].kind == "ws":
            position += 1
        return position

    index = skip_ws(index)
    parenthesised = index < len(tokens) and tokens[index].text == "("
    if parenthesised:
        index = skip_ws(index + 1)
    if index >= len(tokens) or tokens[index].kind != "ident":
        raise _ExpressionError('operator "defined" requires an identifier')
    name = tokens[index].text
    index += 1
    if parenthesised:
        index = skip_ws(index)
        if index >= len(tokens) or tokens[index].text != ")":
            raise _ExpressionError("missing ')' after \"defined\"")
        index += 1
    return name, index


# --------------------------------------------------------------------------------------
# Include guards
# --------------------------------------------------------------------------------------

_GUARD_OPEN = re.compile(
    r"#\s*(?:ifndef\s+([A-Za-z_]\w*)|if\s*!\s*defined\s*\(?\s*([A-Za-z_]\w*)\s*\)?)\s*$"
)
_GUARD_DEFINE = re.compile(r"#\s*define\s+([A-Za-z_]\w*)(?:\s.*)?$")


def _has_include_guard(lines: list[_Line]) -> bool:
    """Is the whole file wrapped in ``#ifndef X`` / ``#define X`` ... ``#endif``?"""
    significant = [line.text.strip() for line in lines if line.text.strip()]
    if len(significant) < 3:
        return False
    opening = _GUARD_OPEN.match(significant[0])
    if opening is None:
        return False
    guard = opening.group(1) or opening.group(2)
    define = _GUARD_DEFINE.match(significant[1])
    if define is None or define.group(1) != guard:
        return False
    return _matching_endif(significant) == len(significant) - 1


def _matching_endif(texts: list[str]) -> int:
    """Index of the #endif that closes the conditional opened by ``texts[0]``."""
    depth = 0
    for index, text in enumerate(texts):
        if not text.startswith("#"):
            continue
        name, _ = _split_directive(text)
        if name in _OPENING_DIRECTIVES:
            depth += 1
        elif name == "endif":
            depth -= 1
            if depth == 0:
                return index
    return -1


# --------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------

@dataclass
class _Conditional:
    """One open ``#if`` block. ``active`` already includes the enclosing blocks."""

    directive: str
    line: int
    parent_active: bool
    taken: bool
    active: bool
    seen_else: bool = False


@dataclass
class _FileState:
    """Per-file state: conditionals cannot cross file boundaries."""

    path: str
    depth: int
    lines: list[_Line]
    conditionals: list[_Conditional] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """Are lines at this point of the file kept (every enclosing #if true)?"""
        return not self.conditionals or self.conditionals[-1].active


_INCLUDE_QUOTED = re.compile(r'"([^"]*)"')
_INCLUDE_ANGLED = re.compile(r"<([^>]*)>")
_FIRST_IDENTIFIER = re.compile(r"([A-Za-z_]\w*)")


class _Preprocessor:
    """Runs the phases over the main file and, recursively, every included file."""

    def __init__(self, resolver: IncludeResolver, defines: Mapping[str, str | None] | None) -> None:
        self.resolver = resolver
        self.macros: dict[str, Macro] = {}
        self.expander = _MacroExpander(self.macros)
        self.diagnostics: list[Diagnostic] = []
        self.lines: list[PPLine] = []
        self.includes: list[str] = []
        self.guards: dict[str, bool] = {}
        self.once_files: set[str] = set()
        for name, body in PREDEFINED_MACROS.items():
            self.macros[name] = Macro(name, None, body, _BUILTIN_FILE, 0)
        for key, value in (defines or {}).items():
            self._define_from_command_line(key, value)

    def run(self, text: str, filename: str) -> PreprocessResult:
        """Preprocess the main file and package up the results."""
        try:
            self._process_file(text, filename, depth=0)
        except _FatalStop:
            pass
        return PreprocessResult(
            lines=self.lines,
            diagnostics=self.diagnostics,
            includes=self.includes,
            macros=dict(self.macros),
            guards=self.guards,
        )

    # --- reporting -------------------------------------------------------------------
    def _report(
        self,
        severity: Severity,
        code: str,
        message: str,
        friendly: str,
        file: str | None,
        line: int,
        notes: Iterable[str] = (),
        column: int = 0,
    ) -> None:
        """Record one diagnostic (``column`` 0: the whole line, gcc's ``file:line:``)."""
        self.diagnostics.append(
            _make_diagnostic(severity, code, message, friendly, file, line, notes, column))

    def _report_invalid(self, error: _DirectiveError, state: _FileState, line: _Line) -> None:
        """Report a malformed directive as ``invalid_directive``."""
        self._report(
            Severity.ERROR, "invalid_directive", error.message, error.friendly, state.path, line.line
        )

    # --- files -----------------------------------------------------------------------
    def _process_file(self, text: str, path: str, depth: int) -> None:
        """Run every phase over one file; includes recurse into here."""
        lines, open_comment = _strip_comments(_splice_lines(text))
        if depth > 0 and path not in self.guards:
            self.guards[path] = _has_include_guard(lines)
        state = _FileState(path, depth, lines)
        index = 0
        while index < len(lines):
            line = lines[index]
            if _is_directive(line.text):
                self._handle_directive(line, state)
                index += 1
            elif state.active:
                index = self._emit_text(state, index)
            else:
                index += 1
        if open_comment is not None:
            self._report(
                Severity.ERROR, "unterminated_comment", "unterminated comment",
                "This /* comment is never closed with */, so everything after it was "
                "ignored. Add */ where the comment should end.",
                path, open_comment,
            )
        self._report_unclosed_conditionals(state)

    def _report_unclosed_conditionals(self, state: _FileState) -> None:
        """Report each #if still open at the end of the file."""
        for conditional in state.conditionals:
            self._report(
                Severity.ERROR, "unterminated_if", f"unterminated #{conditional.directive}",
                f"This #{conditional.directive} starts a block that is never closed. Add a "
                "matching #endif in the same file (for an include guard, at the very end).",
                state.path, conditional.line,
            )

    # --- ordinary lines --------------------------------------------------------------
    def _emit_text(self, state: _FileState, index: int) -> int:
        """Expand one line (plus any lines a macro call spills onto); returns next index."""
        first = state.lines[index]
        text = first.text
        last = index
        while True:
            context = _ExpandContext(state.path, first.line)
            try:
                expanded = self.expander.expand_text(
                    text, context, allow_more=self._can_join(state, last)
                )
                break
            except _NeedMoreInput as need:
                new_last = self._lines_to_join(state, last, need)
                if new_last is None:
                    # The macro name ended the line but is not called: keep it as a word.
                    context = _ExpandContext(state.path, first.line)
                    expanded = self.expander.expand_text(text, context, allow_more=False)
                    break
                joined = (line.text for line in state.lines[last + 1:new_last + 1])
                text = " ".join([text, *joined])
                last = new_last
        self.diagnostics.extend(context.diagnostics)
        if expanded.strip():
            self.lines.append(PPLine(expanded.rstrip(), state.path, first.line))
        return last + 1

    @staticmethod
    def _can_join(state: _FileState, last: int) -> bool:
        """A macro call may continue onto the next line unless that line is a directive."""
        return last + 1 < len(state.lines) and not _is_directive(state.lines[last + 1].text)

    def _lines_to_join(self, state: _FileState, last: int, need: _NeedMoreInput) -> int | None:
        """Index of the last line to glue on for a macro call, or None if it is no call."""
        if need.definite:
            return self._last_line_closing(state, last, need.open_parens)
        # Only the name ended the line: it is a call only if the next line with any
        # code starts with '('. Blank lines in between do not matter to C.
        candidate = last + 1
        while self._can_join(state, candidate - 1) and not state.lines[candidate].text.strip():
            candidate += 1
        if not self._can_join(state, candidate - 1):
            return None
        return candidate if state.lines[candidate].text.lstrip().startswith("(") else None

    def _last_line_closing(self, state: _FileState, last: int, open_parens: int) -> int:
        """Glue lines until enough ')' could have appeared (or a directive / EOF stops us)."""
        index = last + 1
        # A ')' lowers the balance, so adding the balance counts the missing ones down.
        missing = open_parens + _paren_balance(state.lines[index].text)
        while missing > 0 and self._can_join(state, index):
            index += 1
            missing += _paren_balance(state.lines[index].text)
        return index

    # --- directives ------------------------------------------------------------------
    def _handle_directive(self, line: _Line, state: _FileState) -> None:
        """Dispatch one directive line to its handler."""
        name, rest = _split_directive(line.text)
        if name in _CONDITIONAL_DIRECTIVES:
            self._handle_conditional(name, rest, line, state)
            return
        if not state.active:
            return  # C ignores everything except conditionals in skipped blocks
        handlers: dict[str, Callable[[str, _Line, _FileState], None]] = {
            "include": self._handle_include,
            "define": self._handle_define,
            "undef": self._handle_undef,
            "error": self._handle_error,
            "warning": self._handle_warning,
            "pragma": self._handle_pragma,
            "line": lambda rest, line, state: None,  # accepted, has no effect here
        }
        if name in handlers:
            handlers[name](rest, line, state)
        elif name.isdigit() or (not name and not rest):
            return  # a line marker ("# 12") or the null directive "#"
        else:
            self._report(
                Severity.ERROR, "invalid_directive",
                f"invalid preprocessing directive #{name or rest.split()[0]}",
                "Lines that start with # are preprocessor directives. Known ones are "
                "#include, #define, #undef, #if, #ifdef, #ifndef, #elif, #else, #endif, "
                "#pragma, #error and #warning. Check the spelling.",
                state.path, line.line,
            )

    def _handle_define(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#define``."""
        try:
            macro = _parse_define(rest, state.path, line.line)
        except _DirectiveError as error:
            if error.unsupported:
                self._report(Severity.FATAL, "invalid_directive", error.message,
                             error.friendly + " Compilation stops here.", state.path, line.line)
                raise _FatalStop from None
            self._report_invalid(error, state, line)
            return
        self._install_macro(macro)

    def _define_from_command_line(self, key: str, value: str | None) -> None:
        """Install one ``-D`` macro from the cc command line."""
        body = "1" if value is None else value
        try:
            macro = _parse_define(f"{key} {body}", "<command-line>", 0)
        except _DirectiveError as error:
            self._report(Severity.ERROR, "invalid_directive", error.message, error.friendly, None, 0)
            return
        self._install_macro(macro)

    def _install_macro(self, macro: Macro) -> None:
        """Store a macro, warning if it changes an earlier definition."""
        previous = self.macros.get(macro.name)
        if previous is not None and not previous.same_definition(macro):
            self._report(
                Severity.WARNING, "macro_redefined", f'"{macro.name}" redefined',
                f"{macro.name} was already defined with a different value. The new "
                "definition wins from here on. Use #undef first if this is on purpose, "
                "or make both definitions identical.",
                macro.file or None, macro.line,
                notes=[f"this is the location of the previous definition: {previous.file}:{previous.line}"],
            )
        self.macros[macro.name] = macro

    def _handle_undef(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#undef NAME``."""
        match = _FIRST_IDENTIFIER.match(rest)
        if match is None:
            self._report_invalid(
                _DirectiveError(
                    "no macro name given in #undef directive", "Write the macro to forget: #undef NAME."
                ),
                state, line,
            )
            return
        self.macros.pop(match.group(1), None)

    def _handle_error(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#error``: an error the author wrote on purpose."""
        self._report(
            Severity.ERROR, "error_directive", f"#error {rest}".rstrip(),
            "The build reached an #error line, which stops it on purpose. Usually a "
            "required #define is missing; the message says what is wrong.",
            state.path, line.line,
        )

    def _handle_warning(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#warning``: a warning that lets the build go on."""
        self._report(
            Severity.WARNING, "warning_directive", f"#warning {rest}".rstrip(),
            "A #warning line printed this message. The build continues; it is a note "
            "left by whoever wrote the code.",
            state.path, line.line,
        )

    def _handle_pragma(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#pragma once``; other pragmas are ignored."""
        if rest.split() == ["once"]:
            self.once_files.add(state.path)
            if state.depth > 0:
                self.guards[state.path] = True
        # Other pragmas are ignored, as gcc ignores pragmas it does not know.

    # --- #include --------------------------------------------------------------------
    def _handle_include(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#include``: find the file and preprocess it in place."""
        try:
            name, quoted = self._parse_include_target(rest, line, state)
        except _DirectiveError as error:
            self._report_invalid(error, state, line)
            return
        if state.depth + 1 > MAX_INCLUDE_DEPTH:
            self._report(
                Severity.ERROR, "include_depth",
                f"#include nested depth {state.depth + 1} exceeds maximum of {MAX_INCLUDE_DEPTH}",
                "Headers are including each other in a loop. Add an include guard "
                "(#ifndef NAME_H / #define NAME_H ... #endif) or #pragma once to each header.",
                state.path, line.line,
            )
            return
        found = self.resolver.resolve(name, quoted, state.path)
        if found is None:
            self._report_missing_include(name, quoted, line, state)
            raise _FatalStop
        path, text = found
        if path not in self.includes:
            self.includes.append(path)
        if path in self.once_files:
            return
        self._process_file(text, path, state.depth + 1)

    def _parse_include_target(self, rest: str, line: _Line, state: _FileState) -> tuple[str, bool]:
        """Read ``"name"`` or ``<name>`` (expanding a macro if needed)."""
        target = self._match_include_name(rest)
        if target is None and rest:
            # "#include HEADER" where HEADER is a macro: expand once, then try again.
            context = _ExpandContext(state.path, line.line)
            expanded = self.expander.expand_text(rest, context, allow_more=False)
            target = self._match_include_name(expanded.strip())
        if target is None:
            raise _DirectiveError(
                '#include expects "FILENAME" or <FILENAME>',
                'Write #include "myfile.h" for your own headers or #include <stdio.h> '
                "for system headers.",
            )
        if not target[0]:
            raise _DirectiveError("empty filename in #include", "Put the header's name between the quotes.")
        return target

    @staticmethod
    def _match_include_name(rest: str) -> tuple[str, bool] | None:
        """Match ``"name"`` or ``<name>``; returns ``(name, quoted)``."""
        for pattern, quoted in ((_INCLUDE_QUOTED, True), (_INCLUDE_ANGLED, False)):
            match = pattern.match(rest)
            if match is not None:
                return match.group(1), quoted
        return None

    def _report_missing_include(self, name: str, quoted: bool, line: _Line, state: _FileState) -> None:
        """Report a missing header as a fatal error, like gcc."""
        self._report(
            Severity.FATAL, "include_not_found", f"{name}: No such file or directory",
            self._missing_include_friendly(name, quoted, state.path),
            state.path, line.line, column=_include_name_column(line.text),
        )

    def _missing_include_friendly(self, name: str, quoted: bool, including_file: str) -> str:
        """Explain where the header was looked for, in plain words."""
        folder = posixpath.dirname(including_file) or "the current folder"
        if quoted:
            return (
                f'No file called "{name}" was found. With quotes the preprocessor looks in '
                f"{folder} (next to {posixpath.basename(including_file)}), then in each -I "
                "folder, then in the system headers. Check the spelling, or add "
                "-I <folder> to the cc command if the header lives elsewhere."
            )
        posix = POSIX_HEADERS.get(name)
        if posix is not None:
            return (f"<{name}> is POSIX, not standard C: pycc has only the "
                    f"standard C library. {posix} Type `ls /usr/include` to see the "
                    "headers there are.")
        if self.resolver.resolve(name, True, including_file) is not None:
            return (
                f"<{name}> is not a system header, but your project has a file with that "
                f'name. Angle brackets skip your folders; write #include "{name}" for your '
                "own headers."
            )
        return (
            f"<{name}> is not a system header. Angle brackets search only the system "
            f"headers and -I folders. If {name} is your own header, write "
            f'#include "{name}" and keep it in {folder}.'
        )

    # --- conditionals ----------------------------------------------------------------
    def _handle_conditional(self, name: str, rest: str, line: _Line, state: _FileState) -> None:
        """Dispatch #if, #ifdef, #ifndef, #elif, #else and #endif."""
        if name in _OPENING_DIRECTIVES:
            self._open_conditional(name, rest, line, state)
        elif name == "elif":
            self._handle_elif(rest, line, state)
        elif name == "else":
            self._handle_else(line, state)
        else:
            self._handle_endif(line, state)

    def _open_conditional(self, name: str, rest: str, line: _Line, state: _FileState) -> None:
        """Push a new conditional block."""
        parent_active = state.active
        # Conditions inside a skipped block are not evaluated (they may be nonsense).
        value = self._evaluate(name, rest, line, state) if parent_active else False
        state.conditionals.append(
            _Conditional(name, line.line, parent_active, value, parent_active and value)
        )

    def _handle_elif(self, rest: str, line: _Line, state: _FileState) -> None:
        """Handle ``#elif``: evaluated only if no earlier branch was taken."""
        conditional = self._innermost(state, line, "#elif without #if", "else_without_if")
        if conditional is None:
            return
        if conditional.seen_else:
            self._report_misplaced(state, line, "#elif after #else")
            return
        conditional.directive, conditional.line = "elif", line.line
        if conditional.parent_active and not conditional.taken:
            conditional.active = self._evaluate("if", rest, line, state)
            conditional.taken = conditional.active
        else:
            conditional.active = False

    def _handle_else(self, line: _Line, state: _FileState) -> None:
        """Handle ``#else``: active only if no earlier branch was taken."""
        conditional = self._innermost(state, line, "#else without #if", "else_without_if")
        if conditional is None:
            return
        if conditional.seen_else:
            self._report_misplaced(state, line, "#else after #else")
            return
        conditional.directive, conditional.line = "else", line.line
        conditional.seen_else = True
        conditional.active = conditional.parent_active and not conditional.taken
        conditional.taken = True

    def _handle_endif(self, line: _Line, state: _FileState) -> None:
        """Handle ``#endif``: close the innermost block."""
        if self._innermost(state, line, "#endif without #if", "endif_without_if") is not None:
            state.conditionals.pop()

    def _innermost(self, state: _FileState, line: _Line, message: str, code: str) -> _Conditional | None:
        """The innermost open block, or None (and an error) if there is none."""
        if state.conditionals:
            return state.conditionals[-1]
        self._report(
            Severity.ERROR, code, message,
            f"This {message.split()[0]} has no #if, #ifdef or #ifndef above it in this "
            "file. Each one must close a block opened earlier in the same file.",
            state.path, line.line,
        )
        return None

    def _report_misplaced(self, state: _FileState, line: _Line, message: str) -> None:
        """Report an #elif or #else that comes after #else."""
        self._report(
            Severity.ERROR, "invalid_directive", message,
            "A conditional block can have only one #else, and it must come last, after "
            "every #elif. Close the block with #endif.",
            state.path, line.line,
        )

    def _evaluate(self, name: str, rest: str, line: _Line, state: _FileState) -> bool:
        """Decide an opening directive's condition."""
        if name in ("ifdef", "ifndef"):
            match = _FIRST_IDENTIFIER.match(rest)
            if match is None:
                self._report_invalid(
                    _DirectiveError(
                        f"no macro name given in #{name} directive",
                        f"Write the macro to test: #{name} NAME.",
                    ),
                    state, line,
                )
                return False
            defined = self._is_defined(match.group(1))
            return defined if name == "ifdef" else not defined
        return self._evaluate_if(rest, line, state)

    def _is_defined(self, name: str) -> bool:
        """Does ``defined(name)`` hold? Built-ins count as defined."""
        return name in self.macros or name in BUILTIN_NAMES

    def _evaluate_if(self, rest: str, line: _Line, state: _FileState) -> bool:
        """Expand and compute an #if / #elif condition."""
        context = _ExpandContext(state.path, line.line)
        try:
            tokens = _replace_defined(_tokenize(rest), self._is_defined)
            expanded = _join_tokens(self.expander.expand_tokens(tokens, context, allow_more=False))
            value = _ConditionEvaluator(expanded).evaluate()
        except _ExpressionError as error:
            self._report(Severity.ERROR, error.code, error.message, error.friendly, state.path, line.line)
            return False
        finally:
            self.diagnostics.extend(context.diagnostics)
        return value != 0
