"""A recursive-descent parser for the C subset pycc runs.

It turns tokens into the AST of ``interp.astnodes`` and, at the same time, resolves
every *declaration* to a real C type: ``int (*fp)(int, char *)`` becomes "pointer to
function (int, char *) returning int" right here, so nothing later re-reads type syntax.

To do that the parser keeps scopes (the file plus one per block) holding typedef
names, struct/union/enum tags and enum constants. C needs this: ``(T) * x`` is a cast
if ``T`` is a typedef name and a multiplication if it is a variable.

Error handling is the main feature for beginners, so it gets real care:

* Messages copy gcc ("expected ';' before '}' token") and carry a friendly text.
* A missing ';' or ')' is reported where gcc reports it (just after the previous
  token) and parsing simply continues as if it had been typed, which avoids a
  cascade of follow-up errors.
* Harder errors raise ``_ParseError`` after reporting; the nearest statement or
  declaration then skips to the next ';' or '}' at the same brace depth and goes on.
* Two errors at the same token are never both reported, and parsing stops after
  ``MAX_ERRORS`` errors. An AST, possibly partial, is always returned.
"""
from __future__ import annotations

import functools
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import NoReturn, TypeVar

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Severity, SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.c_types import (
    BOOL, CHAR, DOUBLE, FLOAT, INT, LLONG, LONG, SCHAR, SHORT, UCHAR, UINT, ULLONG, ULONG,
    USHORT, VOID, ArrayType, CType, EnumType, FunctionType, IntType, Member, PointerType,
    StructType,
)
from pycc.interp.constexpr import eval_const_expr
from pycc.interp.friendly import (
    did_you_mean, friendly_text, make_diagnostic, redefinition_friendly, unknown_type_hint,
)
from pycc.interp.lexer import Token, TokenKind, tokenize
from pycc.interp.source import lines_from_text

MAX_ERRORS = 20

# How deeply brackets, blocks and statements may nest. Every level costs several Python
# frames in a recursive-descent parser, so without a cap a file of 2000 '(' would crash
# with RecursionError instead of getting a diagnostic. 512 is far beyond real code.
MAX_NESTING = 512

# Python frames one nesting level may use (expression -> assignment -> conditional ->
# binary -> cast -> unary -> primary ...), with room to spare. The parser makes sure the
# interpreter's recursion limit allows MAX_NESTING levels of that.
_FRAMES_PER_LEVEL = 12
_RECURSION_HEADROOM = 2000

# Tokens where skipping after a missing ')' or ']' must stop: they end the statement or
# open/close a block, so the statement and block parsers should see them.
_SKIP_STOPS = frozenset({";", "{", "}"})

STORAGE_CLASSES = frozenset({"static", "extern", "typedef", "register", "auto"})
QUALIFIERS = frozenset({"const", "volatile", "restrict"})
FUNC_NAME = "__func__"
BASIC_TYPE_WORDS = frozenset({
    "void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "_Bool",
})
TAG_KEYWORDS = frozenset({"struct", "union", "enum"})
TYPE_START_WORDS = BASIC_TYPE_WORDS | TAG_KEYWORDS | QUALIFIERS
DECL_START_WORDS = TYPE_START_WORDS | STORAGE_CLASSES | {"inline", "_Complex", "_Imaginary"}
ASSIGN_OPS = frozenset({"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="})
UNARY_OPS = frozenset({"&", "*", "+", "-", "~", "!"})
STATEMENT_KEYWORDS = ("return", "break", "continue", "while", "for", "if", "else", "switch",
                      "case", "default", "do")

# Binding strength of binary operators; a higher number binds tighter. This one table
# encodes C's precedence: 'a & b == c' is 'a & (b == c)' because '==' (6) beats '&' (5).
BINARY_PRECEDENCE: dict[str, int] = {
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5,
    "==": 6, "!=": 6, "<": 7, ">": 7, "<=": 7, ">=": 7,
    "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
}


class _ParseError(Exception):
    """Raised after an error was reported, to unwind to the nearest recovery point."""


class _TooManyErrors(Exception):
    """Raised when MAX_ERRORS errors were reported; parsing stops."""


_Result = TypeVar("_Result")


def _nested(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """Decorator for parse methods that can recurse into themselves (brackets, blocks).

    It counts the nesting level so absurdly deep input gets a diagnostic, not a crash.
    """
    @functools.wraps(method)
    def guarded(self: Parser, *args: object, **kwargs: object) -> _Result:
        self.enter_nesting()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.leave_nesting()
    return guarded


# --- scopes ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """What an ordinary identifier means in a scope.

    ``kind`` is 'typedef' (then ``ctype``/``is_const`` describe the type), 'enum' (then
    ``value`` is the constant) or 'var' (variables and functions). For a 'var' the
    declared ``ctype`` is kept too: the checker owns type rules, but constant folding
    needs it for the common ``sizeof arr / sizeof arr[0]`` array-size idiom.
    """

    kind: str
    ctype: CType | None = None
    value: int | None = None
    is_const: bool = False


@dataclass
class _Scope:
    names: dict[str, Symbol] = field(default_factory=dict)
    tags: dict[str, CType] = field(default_factory=dict)


class ScopeStack:
    """Nested scopes: index 0 is the file scope, then one entry per open block."""

    def __init__(self) -> None:
        self._scopes: list[_Scope] = [_Scope()]

    @property
    def depth(self) -> int:
        """How many scopes are open (1 = only the file scope)."""
        return len(self._scopes)

    def push(self) -> None:
        """Open a new innermost scope."""
        self._scopes.append(_Scope())

    def truncate(self, depth: int) -> None:
        """Close scopes until only ``depth`` remain (used after errors, too)."""
        del self._scopes[max(depth, 1):]

    def declare(self, name: str, symbol: Symbol) -> None:
        """Declare an ordinary identifier in the innermost scope."""
        self._scopes[-1].names[name] = symbol

    def lookup_current(self, name: str) -> Symbol | None:
        """The meaning of ``name`` in the innermost scope only (redeclaration checks)."""
        return self._scopes[-1].names.get(name)

    def lookup(self, name: str) -> Symbol | None:
        """The innermost visible meaning of ``name``."""
        for scope in reversed(self._scopes):
            if name in scope.names:
                return scope.names[name]
        return None

    def declare_tag(self, tag: str, ctype: CType) -> None:
        """Declare a struct/union/enum tag in the innermost scope."""
        self._scopes[-1].tags[tag] = ctype

    def lookup_tag(self, tag: str, current_only: bool = False) -> CType | None:
        """Find a tag, either anywhere visible or only in the innermost scope."""
        scopes = self._scopes[-1:] if current_only else reversed(self._scopes)
        for scope in scopes:
            if tag in scope.tags:
                return scope.tags[tag]
        return None

    def is_typedef(self, name: str) -> bool:
        """True if ``name`` currently names a type."""
        symbol = self.lookup(name)
        return symbol is not None and symbol.kind == "typedef"

    def enum_constant(self, name: str) -> int | None:
        """ConstScope protocol: the value of a visible enum constant."""
        symbol = self.lookup(name)
        return symbol.value if symbol is not None and symbol.kind == "enum" else None

    def object_type(self, name: str) -> CType | None:
        """ConstScope protocol: the declared type of a visible variable, if known."""
        symbol = self.lookup(name)
        return symbol.ctype if symbol is not None and symbol.kind == "var" else None

    def typedef_names(self) -> list[str]:
        """Every visible typedef name (for did-you-mean suggestions)."""
        return [name for scope in self._scopes for name, symbol in scope.names.items()
                if symbol.kind == "typedef"]


# --- declaration pieces ---------------------------------------------------------------

@dataclass
class DeclSpec:
    """The declaration specifiers before the declarators: 'static const unsigned int'."""

    loc: SourceLocation
    words: list[str] = field(default_factory=list)   # basic type keywords, in order
    named_type: CType | None = None                  # struct/union/enum/typedef type
    ctype: CType = INT                               # the resolved type, set at the end
    storage: str | None = None
    is_const: bool = False
    is_inline: bool = False
    declares_tag: bool = False      # 'struct S {...}' or 'struct S;' declares something
    bodies: int = 0                 # how many struct/union/enum bodies were parsed
    ended_early: bool = False       # a struct body without its ';' ended the declaration
    consumed: int = 0               # how many specifier tokens were read


@dataclass
class _PtrOp:
    is_const: bool = False


@dataclass
class _ArrayOp:
    length: int | None


@dataclass
class _Param:
    name: str | None
    ctype: CType
    is_const: bool
    loc: SourceLocation


@dataclass
class _FuncOp:
    params: list[_Param]
    variadic: bool
    has_prototype: bool


_DeclOp = _PtrOp | _ArrayOp | _FuncOp


@dataclass
class Declarator:
    """A parsed declarator: the declared name plus the type operations around it.

    ``ops`` are applied to the base type first to last. For ``*a[8]`` they are
    [pointer, array 8] (array of pointers); for ``(*a)[8]`` they are [array 8,
    pointer] (pointer to an array). This ordering is the whole trick of C declarators.
    """

    name: str | None
    loc: SourceLocation
    ops: list[_DeclOp] = field(default_factory=list)

    def outer_function(self) -> _FuncOp | None:
        """The function suffix the name itself has (``f(int x)``), if any."""
        if self.ops and isinstance(self.ops[-1], _FuncOp):
            return self.ops[-1]
        return None


def _qualifiers(base_const: bool, ops: list[_DeclOp]) -> tuple[bool, bool]:
    """(is_const, points_to_const) of the declared object.

    Arrays are looked through: the constness of an array is that of its elements.
    """
    trimmed = list(ops)
    while trimmed and isinstance(trimmed[-1], _ArrayOp):
        trimmed.pop()
    if not trimmed:
        return base_const, False
    last = trimmed[-1]
    if not isinstance(last, _PtrOp):
        return False, False
    before = trimmed[-2] if len(trimmed) >= 2 else None
    if before is None:
        return last.is_const, base_const
    return last.is_const, isinstance(before, _PtrOp) and before.is_const


def describe_token(tok: Token) -> str:
    """How gcc names a token in 'expected X before Y': "'}' token", "'x'", ..."""
    if tok.kind is TokenKind.PUNCT:
        return f"'{tok.value}' token"
    if tok.kind in (TokenKind.INT, TokenKind.FLOAT):
        return "numeric constant"
    if tok.kind is TokenKind.STRING:
        return "string constant"
    if tok.kind is TokenKind.CHAR:
        return tok.text
    return f"'{tok.value}'"


# --- results ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """The AST (always present, maybe partial) and the diagnostics found on the way."""

    ast: ast.TranslationUnit
    diagnostics: list[Diagnostic]

    @property
    def errors(self) -> list[Diagnostic]:
        """Only the diagnostics that stop compilation."""
        return [diag for diag in self.diagnostics if diag.is_error]

    @property
    def ok(self) -> bool:
        """True when there were no errors (warnings are fine)."""
        return not self.errors


class Parser:
    """Parses one translation unit. Use ``parse()`` or ``parse_text()`` rather than this."""

    def __init__(self, tokens: Iterable[Token], unit_name: str) -> None:
        self.tokens = list(tokens)
        if not self.tokens or self.tokens[-1].kind is not TokenKind.EOF:
            last = self.tokens[-1].end_loc if self.tokens else SourceLocation(unit_name, 1, 1)
            self.tokens.append(Token(TokenKind.EOF, "", last, last.column, ""))
        self.unit_name = unit_name
        self.pos = 0
        self.diagnostics: list[Diagnostic] = []
        self.scopes = ScopeStack()
        self._error_count = 0
        self._last_error_pos = -1
        self._pending_tags: list[ast.Decl] = []
        self._eof_brace_reported = False
        self._implicit_warning: Diagnostic | None = None
        self._nesting = 0   # open brackets/blocks/statements, checked against MAX_NESTING
        # Where each struct/union body was written, keyed by id() of its StructType, so
        # a redefinition can point back at the first one (and spot a double #include).
        self._tag_defined_at: dict[int, SourceLocation] = {}
        # The function whose body is being parsed: '__func__' is its name (C99 6.4.2.2).
        self._function_name: str | None = None
        # Where each enumerator of each scope depth was declared, for redeclarations.
        self._enumerator_at: dict[tuple[int, str], SourceLocation] = {}

    # ===================================================================== tokens
    @property
    def tok(self) -> Token:
        """The current token."""
        return self.tokens[self.pos]

    @property
    def prev(self) -> Token:
        """The token just consumed (the current one at the very start)."""
        return self.tokens[self.pos - 1] if self.pos > 0 else self.tokens[0]

    def peek(self, offset: int = 1) -> Token:
        """Look ahead without consuming; past the end this is the EOF token."""
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        """Consume and return the current token (EOF is never consumed)."""
        token = self.tok
        if token.kind is not TokenKind.EOF:
            self.pos += 1
        return token

    def at_punct(self, *spellings: str) -> bool:
        """Is the current token one of these punctuators?"""
        return self.tok.is_punct(*spellings)

    def accept(self, spelling: str) -> Token | None:
        """Consume the punctuator if it is next."""
        return self.advance() if self.at_punct(spelling) else None

    @property
    def at_eof(self) -> bool:
        """True at the end of input."""
        return self.tok.kind is TokenKind.EOF

    def _before(self) -> str:
        """'before X' for the current token, or 'at end of input'."""
        return "at end of input" if self.at_eof else f"before {describe_token(self.tok)}"

    def _found(self) -> str:
        return "the end of the file" if self.at_eof else describe_token(self.tok)

    def _missing_loc(self) -> SourceLocation:
        """Where gcc reports a missing token: just after the previous token."""
        return self.prev.end_loc if self.pos > 0 else self.tok.loc

    # ================================================================== reporting
    def error(self, code: str, message: str, loc: SourceLocation, *,
              notes: list[str] | None = None, friendly: str | None = None,
              concept: Concept | None = None, **details: object) -> None:
        """Record an error, unless this token already has one (here or in the lexer)."""
        if self.pos == self._last_error_pos or self.tok.after_error:
            return  # a second complaint about the same spot is almost always a cascade
        self._last_error_pos = self.pos
        self.diagnostics.append(make_diagnostic(
            code, message, loc, friendly=friendly, notes=notes, concept=concept, **details))
        self._error_count += 1
        if self._error_count >= MAX_ERRORS:
            raise _TooManyErrors

    def warning(self, code: str, message: str, loc: SourceLocation, *,
                friendly: str | None = None, **details: object) -> Diagnostic:
        """Record a warning and return it (so it can be withdrawn later)."""
        diag = make_diagnostic(code, message, loc, friendly=friendly,
                               severity=Severity.WARNING, **details)
        self.diagnostics.append(diag)
        return diag

    def fail(self, code: str, message: str, loc: SourceLocation, **kwargs: object) -> NoReturn:
        """Report an error and unwind to the nearest recovery point."""
        self.error(code, message, loc, **kwargs)  # type: ignore[arg-type]
        raise _ParseError

    def expect_semicolon(self) -> None:
        """Consume ';' or report it missing (and carry on as if it were there)."""
        if self.accept(";"):
            return
        self.error("missing_semicolon", f"expected ';' {self._before()}", self._missing_loc(),
                   friendly=self._semicolon_friendly())

    def _semicolon_friendly(self) -> str:
        text = friendly_text("missing_semicolon", line=self._missing_loc().line)
        previous = self.prev
        if previous.kind is TokenKind.IDENT:
            keyword = did_you_mean(str(previous.value), STATEMENT_KEYWORDS, cutoff=0.7)
            if keyword is not None:
                text = (f"C reads '{previous.value}' as a name, not a keyword. "
                        f"Did you mean '{keyword}'?")
        return text

    def expect_close(self, closer: str, opener: Token) -> None:
        """Consume ``closer`` or report it missing, pointing back at its opener."""
        if self.accept(closer):
            return
        where = f"line {opener.loc.line}, column {opener.loc.column}"
        code = "unbalanced_paren" if closer == ")" else "expected_token"
        self.error(code, f"expected '{closer}' {self._before()}", self._missing_loc(),
                   notes=[f"to match this '{opener.value}' at {where}"],
                   line=opener.loc.line, expected=f"'{closer}'", found=self._found())
        if closer in (")", "]"):
            self._skip_to_closer(closer)

    def _skip_to_closer(self, closer: str) -> None:
        """After a missing ')' or ']': drop the rest of the bracket, up to its closer.

        In 'printf("%d" x);' gcc reports only "expected ')' before 'x'". Parsing 'x' as
        the next statement would add a bogus "expected ';'" about the ')' after it, so
        we skip 'x' and take the ')'. We never skip past ';', '{' or '}', so a ')' that
        is really missing ('if (x > 3 {') costs nothing.
        """
        depth = 0
        while not self.at_eof:
            token = self.tok
            if depth == 0 and token.is_punct(closer):
                self.advance()
                return
            if token.kind is TokenKind.PUNCT and token.value in _SKIP_STOPS:
                return
            if token.is_punct("(", "["):
                depth += 1
            elif token.is_punct(")", "]"):
                depth = max(depth - 1, 0)
            self.advance()

    def expect_token(self, spelling: str, *, fatal: bool = False) -> None:
        """Consume ``spelling`` or report "expected 'X' before Y"."""
        if self.accept(spelling):
            return
        report = self.fail if fatal else self.error
        report("expected_token", f"expected '{spelling}' {self._before()}", self._missing_loc(),
               expected=f"'{spelling}'", found=self._found())

    def enter_nesting(self) -> None:
        """Count one more open level; past MAX_NESTING report it and unwind."""
        if self._nesting >= MAX_NESTING:
            self.fail("nesting_too_deep",
                      f"code is nested more than {MAX_NESTING} levels deep",
                      self.tok.loc, limit=MAX_NESTING)
        self._nesting += 1

    def leave_nesting(self) -> None:
        """Close the level opened by ``enter_nesting``."""
        self._nesting -= 1

    def _report_eof_brace(self, opener: Token) -> None:
        """'expected '}' at end of input', reported once however many blocks are open."""
        if self._eof_brace_reported:
            return
        self._eof_brace_reported = True
        self.error("unbalanced_brace", "expected '}' at end of input", self._missing_loc(),
                   notes=[f"to match this '{{' at line {opener.loc.line}, "
                          f"column {opener.loc.column}"],
                   line=opener.loc.line)

    # ================================================================== recovery
    def _skip_statement(self) -> None:
        """Skip to just after the next ';' at this brace depth, or stop before a '}'."""
        depth = 0
        while not self.at_eof:
            token = self.tok
            if token.is_punct("{"):
                depth += 1
            elif token.is_punct("}"):
                if depth == 0:
                    return  # it closes the enclosing block: leave it for that block
                depth -= 1
                self.advance()
                if depth == 0:
                    return
                continue
            elif token.is_punct(";") and depth == 0:
                self.advance()
                return
            self.advance()

    def _skip_external(self) -> None:
        """At file scope: skip past the next ';' or the '}' closing a skipped block."""
        depth = 0
        while not self.at_eof:
            token = self.advance()
            if token.is_punct("{"):
                depth += 1
            elif token.is_punct("}"):
                depth = max(depth - 1, 0)
                if depth == 0:
                    return
            elif token.is_punct(";") and depth == 0:
                return

    def _take_pending(self) -> list[ast.Decl]:
        """Struct/enum definitions met while parsing the current declaration."""
        pending, self._pending_tags = self._pending_tags, []
        return pending

    # ============================================================ translation unit
    def parse_translation_unit(self) -> ast.TranslationUnit:
        """Parse the whole token list."""
        unit = ast.TranslationUnit(SourceLocation(self.unit_name, 1, 1), self.unit_name, [])
        try:
            while not self.at_eof:
                unit.decls.extend(self._parse_external_guarded())
        except _TooManyErrors:
            self.diagnostics.append(make_diagnostic(
                "too_many_errors", f"compilation terminated due to -fmax-errors={MAX_ERRORS}.",
                None, severity=Severity.NOTE))
        except RecursionError:
            # Only reachable through chains MAX_NESTING does not count, such as thousands
            # of unary minus signs in a row. Report it rather than crash.
            self.scopes.truncate(1)
            self.diagnostics.append(make_diagnostic(
                "nesting_too_deep", "expression is nested too deeply to compile",
                self.tok.loc, limit=MAX_NESTING))
        return unit

    def _parse_external_guarded(self) -> list[ast.Decl]:
        errors_before = self._error_count
        self._implicit_warning = None
        self._pending_tags = []
        start = self.pos
        try:
            decls = self._parse_external_declaration()
        except _ParseError:
            self.scopes.truncate(1)
            self._skip_external()
            decls = []
        if self.pos == start:
            self.advance()  # never get stuck on one token
        # An 'implicit int' warning next to a real error is noise: the error explains it.
        warning = self._implicit_warning
        if warning is not None and self._error_count > errors_before:
            self.diagnostics.remove(warning)
        return decls

    def _parse_external_declaration(self) -> list[ast.Decl]:
        if self.accept(";"):
            return []
        spec = self._parse_decl_specifiers("file")
        if spec is None:
            self.fail("bad_declarator", f"expected identifier or '(' {self._before()}",
                      self.tok.loc, detail=self._file_scope_friendly())
        return self._parse_declaration_rest(spec, "file")

    def _file_scope_friendly(self) -> str:
        token = self.tok
        if (token.is_punct("{") and self.prev.is_punct(";")
                and self.pos >= 2 and self.tokens[self.pos - 2].is_punct(")")):
            return ("There is a ';' between the function header and its '{'. Remove it: "
                    "write 'int main(void) {'.")
        if token.is_punct("}"):
            return "This '}' has no matching '{'. Delete it or check the braces above."
        return ("C expected a declaration here, such as 'int x;' or a function. "
                "Statements belong inside a function body.")

    # ============================================================== declarations
    def _parse_declaration_rest(self, spec: DeclSpec, context: str) -> list[ast.Decl]:
        """Everything after the specifiers: declarators, initializers, ';' or a body."""
        if spec.ended_early:
            return self._take_pending()
        if self.accept(";"):
            if not spec.declares_tag:
                self.warning("useless_declaration", "useless type name in empty declaration",
                             spec.loc)
            return self._take_pending()
        decls: list[ast.Decl] = []
        first = True
        while True:
            declarator = self._parse_declarator("decl")
            ctype = self._build_type(spec.ctype, declarator)
            could_define = (first and isinstance(ctype, FunctionType)
                            and spec.storage != "typedef")
            if could_define and (self.at_punct("{")
                                 or self._body_brace_missing(declarator, context)):
                function = self._parse_function_definition(spec, declarator, ctype, context)
                return self._take_pending() + ([function] if function else [])
            decls.append(self._make_declaration(spec, declarator, ctype))
            first = False
            if self.accept(","):
                continue
            if not self.accept(";"):
                self._report_declaration_end(decls[-1])
            break
        return self._take_pending() + decls

    def _body_brace_missing(self, declarator: Declarator, context: str) -> bool:
        """'int main(void)' then 'return 0;' on the next line: the '{' was forgotten.

        Only statement keywords count. A type word there more likely means a missing
        ';' after a prototype ('int f(void)' then 'int main...'), so we leave that case
        to the usual "expected ';'" error.
        """
        token = self.tok
        if (context != "file" or declarator.outer_function() is None
                or token.kind is not TokenKind.KEYWORD
                or str(token.value) not in STATEMENT_KEYWORDS):
            return False
        self.error("expected_token", f"expected '{{' {self._before()}", self._missing_loc(),
                   friendly=f"A function body starts with '{{'. Add '{{' at the end of line "
                            f"{self._missing_loc().line}, after the function header.")
        return True

    def _report_declaration_end(self, last: ast.Decl) -> None:
        """gcc's wording after a declarator: "expected '=', ',' or ';' before ..."."""
        has_init = isinstance(last, ast.VarDecl) and last.init is not None
        expected = "',' or ';'" if has_init else "'=', ',' or ';'"
        self.error("missing_semicolon", f"expected {expected} {self._before()}",
                   self._missing_loc(), friendly=self._semicolon_friendly())

    def _make_declaration(self, spec: DeclSpec, declarator: Declarator,
                          ctype: CType) -> ast.Decl:
        name = declarator.name or ""
        is_const, points_to_const = _qualifiers(spec.is_const, declarator.ops)
        if spec.storage == "typedef":
            self._reject_initializer(f"typedef '{name}' is initialized")
            self.scopes.declare(name, Symbol("typedef", ctype, is_const=is_const))
            return ast.TypedefDecl(declarator.loc, name, ctype, is_const)
        self.scopes.declare(name, Symbol("var", ctype))
        if isinstance(ctype, FunctionType):
            self._reject_initializer(f"function '{name}' is initialized like a variable")
            function = declarator.outer_function()
            names = [param.name for param in function.params] if function else []
            return ast.FunctionDecl(declarator.loc, name, ctype, spec.storage, names,
                                    spec.is_inline)
        if ctype.is_void:
            self.error("bad_declarator", f"variable or field '{name}' declared void",
                       declarator.loc, detail="Variables cannot have type void; void only means "
                                              "'nothing' as a return type or in (void).")
        init = self._parse_initializer() if self.accept("=") else None
        completed = self._complete_array_type(ctype, init, declarator, spec)
        if completed is not ctype:
            # 'int a[] = {1, 2}' only gets its length from the initializer; remember it,
            # so a later 'sizeof a' sees int[2].
            self.scopes.declare(name, Symbol("var", completed))
        ctype = completed
        return ast.VarDecl(declarator.loc, name, ctype, init, spec.storage,
                           spec.storage == "extern", is_const, points_to_const)

    def _reject_initializer(self, message: str) -> None:
        if self.at_punct("="):
            self.error("bad_initializer", message, self.tok.loc,
                       detail="Only variables can be given a value with '='.")
            self.advance()
            self._parse_initializer()

    def _parse_function_definition(self, spec: DeclSpec, declarator: Declarator,
                                   ftype: FunctionType, context: str) -> ast.FunctionDef | None:
        name = declarator.name or ""
        function = declarator.outer_function()
        self.scopes.declare(name, Symbol("var", ftype))
        if function is None:
            self.fail("missing_semicolon", f"expected ',' or ';' {self._before()}",
                      self._missing_loc(), friendly=self._semicolon_friendly())
        if context != "file":
            self.error("nested_function", "ISO C forbids nested functions", declarator.loc,
                       name=name)
            self._parse_function_body(function)  # consume it so parsing can go on
            return None
        for param in function.params:
            if param.name is None:
                self.error("bad_declarator", "parameter name omitted", param.loc,
                           detail="In a function definition every parameter needs a name, "
                                  "like 'int count'.")
        outer, self._function_name = self._function_name, name
        try:
            body = self._parse_function_body(function)
        finally:
            self._function_name = outer
        return ast.FunctionDef(
            declarator.loc, name, ftype, [p.name for p in function.params], body,
            spec.storage, spec.is_inline, [p.is_const for p in function.params],
            [p.loc for p in function.params])

    def _parse_function_body(self, function: _FuncOp) -> ast.CompoundStmt:
        # The parameters and the outermost block of the body share one scope in C.
        depth = self.scopes.depth
        self.scopes.push()
        try:
            for param in function.params:
                if param.name:
                    self.scopes.declare(param.name, Symbol("var", param.ctype))
            # A forgotten '{' was already reported; parse the body as if it were there.
            return self._parse_compound(new_scope=False,
                                        brace_missing=not self.at_punct("{"))
        finally:
            self.scopes.truncate(depth)

    # ----------------------------------------------------------- specifiers
    def _parse_decl_specifiers(self, context: str) -> DeclSpec | None:
        """Read 'static const unsigned long', 'struct S {...}', 'Tile', ...

        ``context`` is 'file', 'block', 'member', 'param' or 'typename'; it decides
        which words are allowed and how an unknown name is treated. Returns None when
        no specifier is present at all.
        """
        spec = DeclSpec(loc=self.tok.loc)
        while True:
            token = self.tok
            if token.kind is TokenKind.KEYWORD:
                bodies_before = spec.bodies
                if not self._take_keyword_specifier(spec, context):
                    break
                if spec.bodies > bodies_before and self._tag_body_missing_semicolon(context):
                    spec.ended_early = True
                    break
            elif token.kind is TokenKind.IDENT and not spec.words and spec.named_type is None:
                if not self._take_type_name(spec, context):
                    break
            else:
                break
        if spec.consumed == 0:
            return self._implicit_int_spec(context)
        spec.ctype = self._resolve_spec_type(spec)
        return spec

    def _take_keyword_specifier(self, spec: DeclSpec, context: str) -> bool:
        word = str(self.tok.value)
        if word in STORAGE_CLASSES:
            self._take_storage_class(spec, context, word)
        elif word in QUALIFIERS:
            spec.is_const |= word == "const"
            self.advance()
        elif word == "inline":
            spec.is_inline = True
            self.advance()
        elif word in BASIC_TYPE_WORDS:
            if spec.named_type is not None:
                self._two_types_error()
            spec.words.append(word)
            self.advance()
        elif word in TAG_KEYWORDS:
            if spec.words or spec.named_type is not None:
                self._two_types_error()
            if word == "enum":
                spec.named_type = self._parse_enum_specifier(spec, context)
            else:
                spec.named_type = self._parse_struct_specifier(spec, context)
        elif word in ("_Complex", "_Imaginary"):
            self.error("unsupported_feature", f"'{word}' types are not supported", self.tok.loc,
                       detail="Complex numbers are not part of the C that pycc runs.")
            self.advance()
        else:
            return False
        spec.consumed += 1
        return True

    def _take_storage_class(self, spec: DeclSpec, context: str, word: str) -> None:
        token = self.advance()
        if context in ("member", "typename") or (context == "param" and word != "register"):
            self.error("bad_declarator", f"storage class '{word}' is not allowed here", token.loc,
                       detail=f"'{word}' can only be used when declaring a variable or "
                              "function, not here.")
        elif spec.storage is not None:
            self.error("invalid_type_combination",
                       "multiple storage classes in declaration specifiers", token.loc,
                       friendly="Use only one of static, extern, typedef, register and auto "
                                "in a declaration.")
        else:
            spec.storage = word

    def _two_types_error(self) -> None:
        self.error("invalid_type_combination", "two or more data types in declaration specifiers",
                   self.tok.loc)

    def _take_type_name(self, spec: DeclSpec, context: str) -> bool:
        """A typedef name as the type, or an unknown name that is clearly meant as one."""
        token = self.tok
        name = str(token.value)
        symbol = self.scopes.lookup(name)
        if symbol is not None and symbol.kind == "typedef":
            spec.named_type = symbol.ctype
            spec.is_const |= symbol.is_const
        elif symbol is None and self._looks_like_unknown_type(context):
            self._report_unknown_type(token)
            spec.named_type = INT  # carry on as gcc does, as if it said 'int'
        else:
            return False
        self.advance()
        spec.consumed += 1
        return True

    def _looks_like_unknown_type(self, context: str) -> bool:
        """Is the current (undeclared) identifier being used as a type name?

        'itn x' (two names in a row) can only be a declaration. At file scope, in
        struct members and in parameters, 'Name *p' is too. Inside a block 'a * b' could
        be a multiplication, so we also require the shape of a declaration after it.
        """
        following = self.peek()
        if following.kind is TokenKind.IDENT:
            return context != "typename"
        if not following.is_punct("*"):
            return False
        if context in ("file", "member", "param"):
            return True
        return (context == "block" and self.peek(2).kind is TokenKind.IDENT
                and self.peek(3).is_punct(";", "=", ",", "[")
                and self.scopes.lookup(str(self.peek(2).value)) is None)

    def _report_unknown_type(self, token: Token) -> None:
        name = str(token.value)
        suggestion, hint = unknown_type_hint(name, self.scopes.typedef_names())
        message = f"unknown type name '{name}'"
        if suggestion is not None:
            message += f"; did you mean '{suggestion}'?"
        self.error("unknown_type", message, token.loc, name=name, hint=hint)

    def _implicit_int_spec(self, context: str) -> DeclSpec | None:
        """'main() {' or 'x;' at file scope: old C's implicit int, with a warning."""
        token = self.tok
        if context != "file" or token.kind is not TokenKind.IDENT:
            return None
        name = str(token.value)
        if self.peek().is_punct("("):
            self._implicit_warning = self.warning(
                "missing_return_type", "return type defaults to 'int'", token.loc, name=name)
        else:
            self._implicit_warning = self.warning(
                "implicit_int", f"type defaults to 'int' in declaration of '{name}'", token.loc,
                name=name)
        return DeclSpec(loc=token.loc, ctype=INT)

    def _resolve_spec_type(self, spec: DeclSpec) -> CType:
        if spec.named_type is not None:
            return spec.named_type
        if spec.words:
            return self._resolve_basic_type(spec.words, spec.loc)
        # Only 'static' or 'const' with no type: implicit int, as in old C.
        name = self.tok.value if self.tok.kind is TokenKind.IDENT else ""
        self.warning("implicit_int", f"type defaults to 'int' in declaration of '{name}'",
                     spec.loc, name=name)
        return INT

    def _resolve_basic_type(self, words: list[str], loc: SourceLocation) -> CType:
        """Combine keywords like 'unsigned long long int' into one type."""
        counts = Counter(words)
        bases = [word for word in words if word in ("void", "char", "int", "float", "double",
                                                     "_Bool")]
        problem = _basic_type_problem(counts, bases)
        if problem is not None:
            self.error("invalid_type_combination", problem, loc)
        base = bases[0] if bases else "int"
        unsigned = counts["unsigned"] > 0
        signed = counts["signed"] > 0
        if base == "void":
            return VOID
        if base == "_Bool":
            return BOOL
        if base == "float":
            return FLOAT
        if base == "double":
            return DOUBLE  # 'long double' is treated as double (see docs/SUPPORTED_C.md)
        if base == "char":
            return UCHAR if unsigned else SCHAR if signed else CHAR
        if counts["short"]:
            return USHORT if unsigned else SHORT
        longs = counts["long"]
        if longs >= 2:
            return ULLONG if unsigned else LLONG
        if longs == 1:
            return ULONG if unsigned else LONG
        return UINT if unsigned else INT

    # ----------------------------------------------------------- struct / union
    def _parse_struct_specifier(self, spec: DeclSpec, context: str) -> StructType:
        keyword = self.advance()
        is_union = keyword.value == "union"
        tag: str | None = None
        tag_loc = keyword.loc
        if self.tok.kind is TokenKind.IDENT:
            tag, tag_loc = str(self.tok.value), self.tok.loc
            self.advance()
        if self.at_punct("{"):
            stype = self._struct_for_definition(tag, is_union, tag_loc)
            self._tag_defined_at.setdefault(id(stype), tag_loc)
            self._parse_struct_body(stype)
            self._pending_tags.append(ast.StructDecl(tag_loc, tag, stype, True))
            spec.declares_tag = True
            spec.bodies += 1
            return stype
        if tag is None:
            self.fail("expected_token", f"expected '{{' {self._before()}", self.tok.loc,
                      expected="'{' or a tag name", found=self._found())
        if self.at_punct(";") and spec.consumed == 0 and context in ("file", "block", "member"):
            return self._forward_struct(tag, is_union, tag_loc, spec)
        existing = self.scopes.lookup_tag(tag)
        if existing is None:
            created = StructType(tag, None, is_union)
            self.scopes.declare_tag(tag, created)
            return created
        return self._checked_tag_kind(existing, tag, is_union, tag_loc)

    def _forward_struct(self, tag: str, is_union: bool, loc: SourceLocation,
                        spec: DeclSpec) -> StructType:
        """'struct Node;' declares an incomplete type in the current scope."""
        spec.declares_tag = True
        existing = self.scopes.lookup_tag(tag, current_only=True)
        if existing is not None:
            return self._checked_tag_kind(existing, tag, is_union, loc)
        created = StructType(tag, None, is_union)
        self.scopes.declare_tag(tag, created)
        self._pending_tags.append(ast.StructDecl(loc, tag, created, False))
        return created

    def _struct_for_definition(self, tag: str | None, is_union: bool,
                               loc: SourceLocation) -> StructType:
        """The StructType a '{ ... }' body fills: a new one or an earlier forward one."""
        if tag is None:
            return StructType(None, None, is_union)
        existing = self.scopes.lookup_tag(tag, current_only=True)
        if existing is None:
            created = StructType(tag, None, is_union)
            # Registered before the body is parsed, so 'struct Node *next;' inside works.
            self.scopes.declare_tag(tag, created)
            return created
        checked = self._checked_tag_kind(existing, tag, is_union, loc)
        if checked.is_complete:
            keyword = "union" if is_union else "struct"
            name = f"{keyword} {tag}"
            previous = self._tag_defined_at.get(id(checked))
            self.error("redefinition", f"redefinition of '{name}'", loc,
                       concept=Concept.STRUCTS, name=name,
                       notes=[f"previous definition of '{name}' was here"],
                       friendly=redefinition_friendly(name, previous, loc))
            return StructType(tag, None, is_union)
        return checked

    def _checked_tag_kind(self, existing: CType, tag: str, is_union: bool,
                          loc: SourceLocation) -> StructType:
        if isinstance(existing, StructType) and existing.is_union == is_union:
            return existing
        self._wrong_tag_kind(tag, loc, Concept.STRUCTS)
        return StructType(tag, None, is_union)

    def _wrong_tag_kind(self, tag: str, loc: SourceLocation, concept: Concept) -> None:
        self.error("redefinition", f"'{tag}' defined as wrong kind of tag", loc, concept=concept,
                   friendly=f"'{tag}' was declared earlier with a different keyword. Use the "
                            "same one (struct, union or enum) every time.")

    def _parse_struct_body(self, stype: StructType) -> None:
        opener = self.advance()
        errors_before = self._error_count
        members: list[Member] = []
        seen: set[str] = set()
        while not self.at_punct("}"):
            if self.at_eof:
                self._report_eof_brace(opener)
                break
            try:
                new_members = self._parse_member_declaration()
            except _ParseError:
                self._skip_statement()
                continue
            for member, member_loc in new_members:
                if member.name in seen:
                    # Point at the second name, as gcc does, not at the ';' after it.
                    self.error("redefinition", f"duplicate member '{member.name}'",
                               member_loc, concept=Concept.STRUCTS, name=member.name)
                    continue
                seen.add(member.name)
                members.append(member)
        self.accept("}")
        # An empty body after a member error is that error's fault; do not add to it.
        if not members and self._error_count == errors_before:
            self.warning("empty_struct", f"{stype.keyword} has no members", opener.loc)
        stype.complete(members)

    def _parse_member_declaration(self) -> list[tuple[Member, SourceLocation]]:
        """One member line such as 'int x, y;': each member with where its name is."""
        spec = self._parse_decl_specifiers("member")
        if spec is None:
            self.fail("expected_token", f"expected specifier-qualifier-list {self._before()}",
                      self.tok.loc, expected="a member type such as 'int'", found=self._found())
        if spec.ended_early or self.accept(";"):
            return []
        members: list[tuple[Member, SourceLocation]] = []
        while True:
            declarator = self._parse_declarator("member")
            if self.at_punct(":"):
                self.fail("unsupported_feature", "bit-fields are not supported", self.tok.loc,
                          detail="Use a whole int (or unsigned char) for each field instead.")
            ctype = self._build_type(spec.ctype, declarator)
            members.append((self._checked_member(declarator, ctype), declarator.loc))
            if self.accept(","):
                continue
            if self.accept(";"):
                break
            if self.at_punct("}"):
                self.warning("missing_semicolon", "no semicolon at end of struct or union",
                             self._missing_loc(), line=self._missing_loc().line)
                break
            self.error("missing_semicolon", f"expected ';' {self._before()}", self._missing_loc(),
                       friendly=self._semicolon_friendly())
            break
        return members

    def _checked_member(self, declarator: Declarator, ctype: CType) -> Member:
        name = declarator.name or ""
        if ctype.is_function:
            self.error("bad_declarator", f"field '{name}' declared as a function",
                       declarator.loc, detail="A struct can hold a pointer to a function, "
                                              "written 'int (*name)(int)', but not a function.")
        elif not ctype.is_complete:
            self.error("incomplete_type", f"field '{name}' has incomplete type", declarator.loc,
                       name=name)
        return Member(name, ctype)

    def _tag_body_missing_semicolon(self, context: str) -> bool:
        """After 'struct S { ... }': is the ';' missing before the next declaration?"""
        if context not in ("file", "block", "member"):
            return False
        token = self.tok
        if token.is_punct(";", "*", "(") or token.is_keyword(*QUALIFIERS):
            return False
        if token.kind is TokenKind.IDENT:
            following = self.peek()
            if not (self.scopes.is_typedef(str(token.value))
                    and (following.kind is TokenKind.IDENT or following.is_punct("*"))):
                return False
        self.error("missing_semicolon", f"expected ';', identifier or '(' {self._before()}",
                   self._missing_loc(),
                   friendly=f"A struct or enum definition ends with '}};'. Add ';' after the "
                            f"'}}' on line {self._missing_loc().line}.")
        return True

    # ----------------------------------------------------------- enum
    def _parse_enum_specifier(self, spec: DeclSpec, context: str) -> EnumType:
        keyword = self.advance()
        tag: str | None = None
        tag_loc = keyword.loc
        if self.tok.kind is TokenKind.IDENT:
            tag, tag_loc = str(self.tok.value), self.tok.loc
            self.advance()
        if self.at_punct("{"):
            return self._parse_enum_body(tag, tag_loc, spec)
        if tag is None:
            self.fail("expected_token", f"expected '{{' {self._before()}", self.tok.loc,
                      expected="'{' or a tag name", found=self._found())
        existing = self.scopes.lookup_tag(tag)
        if isinstance(existing, EnumType):
            return existing
        if existing is not None:
            self._wrong_tag_kind(tag, tag_loc, Concept.ENUMS)
        created = EnumType(tag)
        self.scopes.declare_tag(tag, created)
        if self.at_punct(";"):
            spec.declares_tag = True
        return created

    def _parse_enum_body(self, tag: str | None, tag_loc: SourceLocation,
                         spec: DeclSpec) -> EnumType:
        opener = self.advance()
        constants = self._parse_enumerators(opener)
        etype = EnumType(tag, tuple(constants))
        if tag is not None:
            existing = self.scopes.lookup_tag(tag, current_only=True)
            if isinstance(existing, EnumType) and existing.constants:
                self.error("redefinition", f"redefinition of 'enum {tag}'", tag_loc,
                           concept=Concept.ENUMS, name=f"enum {tag}")
            self.scopes.declare_tag(tag, etype)
        self._pending_tags.append(ast.EnumDecl(tag_loc, tag, etype, list(constants)))
        spec.declares_tag = True
        spec.bodies += 1
        return etype

    def _check_enumerator_unique(self, name: str, loc: SourceLocation) -> None:
        """``enum A { X }; enum B { X };`` is an error in C, as in gcc.

        Every enumerator is an ordinary name of the enclosing scope, so two enums of
        one scope cannot share a constant even when the values agree.
        """
        key = (self.scopes.depth, name)
        previous = self._enumerator_at.get(key)
        existing = self.scopes.lookup_current(name)
        if existing is not None and existing.kind == "enum":
            self.error("redefinition", f"redeclaration of enumerator '{name}'", loc,
                       concept=Concept.ENUMS, name=name,
                       notes=[f"previous definition of '{name}' was here"],
                       friendly=redefinition_friendly(name, previous, loc))
        self._enumerator_at[key] = loc

    def _parse_enumerators(self, opener: Token) -> list[tuple[str, int]]:
        constants: list[tuple[str, int]] = []
        next_value = 0
        while not self.at_punct("}"):
            if self.at_eof:
                self._report_eof_brace(opener)
                return constants
            if self.tok.kind is not TokenKind.IDENT:
                self.error("expected_token", f"expected identifier {self._before()}",
                           self.tok.loc, expected="an enum constant name", found=self._found())
                self._skip_to_closing_brace()
                break
            name_token = self.advance()
            name = str(name_token.value)
            if self.accept("="):
                value = self._enumerator_value(name, next_value)
            else:
                value = self._implicit_enumerator_value(name_token, next_value)
            self._check_enumerator_unique(name, name_token.loc)
            self.scopes.declare(name, Symbol("enum", INT, value=value))
            constants.append((name, value))
            next_value = value + 1
            if self.accept(","):
                continue
            if not self.at_punct("}"):
                self.error("expected_token", f"expected ',' or '}}' {self._before()}",
                           self._missing_loc(), expected="',' or '}'", found=self._found())
                self._skip_to_closing_brace()
                break
        self.accept("}")
        return constants

    def _enumerator_value(self, name: str, fallback: int) -> int:
        """The value after '=': a constant expression that should fit in an int."""
        expr = self._parse_conditional()
        value = eval_const_expr(expr, self.scopes)
        if value is None:
            self.error("enum_value_not_const",
                       f"enumerator value for '{name}' is not an integer constant", expr.loc)
            return fallback
        if not INT.min <= value <= INT.max:
            # gcc accepts this with -pedantic warning; we follow, as enum constants are int.
            self.warning("enum_overflow",
                         "ISO C restricts enumerator values to range of 'int'", expr.loc,
                         name=name)
        return value

    def _implicit_enumerator_value(self, name_token: Token, value: int) -> int:
        """Previous value + 1; past INT_MAX that is gcc's 'overflow in enumeration values'."""
        if value > INT.max:
            self.error("enum_overflow", "overflow in enumeration values", name_token.loc,
                       name=str(name_token.value))
        return value

    def _skip_to_closing_brace(self) -> None:
        depth = 0
        while not self.at_eof:
            if self.at_punct("{"):
                depth += 1
            elif self.at_punct("}"):
                if depth == 0:
                    return
                depth -= 1
            self.advance()

    # ----------------------------------------------------------- declarators
    @_nested
    def _parse_declarator(self, context: str) -> Declarator:
        """pointer* direct-declarator. ``context`` as in _parse_decl_specifiers."""
        pointers: list[_DeclOp] = []
        while self.accept("*"):
            is_const = False
            while self.tok.is_keyword(*QUALIFIERS):
                is_const |= self.advance().value == "const"
            pointers.append(_PtrOp(is_const))
        direct = self._parse_direct_declarator(context)
        direct.ops = pointers + direct.ops
        return direct

    def _parse_direct_declarator(self, context: str) -> Declarator:
        abstract_ok = context in ("param", "typename")
        name: str | None = None
        loc = self.tok.loc
        inner_ops: list[_DeclOp] = []
        if self.tok.kind is TokenKind.IDENT and context != "typename":
            name = str(self.advance().value)
        elif self.at_punct("(") and self._paren_starts_declarator(context):
            opener = self.advance()
            inner = self._parse_declarator(context)
            self.expect_close(")", opener)
            name, loc, inner_ops = inner.name, inner.loc, inner.ops
        elif not abstract_ok:
            self.fail("bad_declarator", f"expected identifier or '(' {self._before()}",
                      self.tok.loc, detail="A declaration needs a name after the type, like "
                                           "'int count = 5;'.")
        suffixes: list[_DeclOp] = []
        while True:
            if self.at_punct("["):
                suffixes.append(self._parse_array_suffix(name))
            elif self.at_punct("("):
                suffixes.append(self._parse_parameter_list())
            else:
                break
        # 'a[2][3]' is an array of 2 arrays of 3: the rightmost suffix is applied first.
        return Declarator(name, loc, list(reversed(suffixes)) + inner_ops)

    def _paren_starts_declarator(self, context: str) -> bool:
        """In 'int (*)(int)' the first '(' nests a declarator; in 'int (int)' it lists params."""
        if context not in ("param", "typename"):
            return True
        following = self.peek()
        if following.is_punct("*", "(", "["):
            return True
        return (context == "param" and following.kind is TokenKind.IDENT
                and not self.scopes.is_typedef(str(following.value)))

    def _parse_array_suffix(self, name: str | None) -> _ArrayOp:
        opener = self.advance()
        if self.accept("]"):
            return _ArrayOp(None)
        size_expr = self._parse_assignment()
        self.expect_close("]", opener)
        length = eval_const_expr(size_expr, self.scopes)
        label = name or "type name"
        if length is None:
            self.error("array_size_not_const",
                       f"size of array '{label}' is not an integer constant", size_expr.loc)
        elif length < 0:
            self.error("bad_declarator", f"size of array '{label}' is negative", size_expr.loc,
                       detail="An array needs a size of zero or more elements.")
            length = None
        return _ArrayOp(length)

    @_nested
    def _parse_parameter_list(self) -> _FuncOp:
        opener = self.advance()
        if self.accept(")"):
            return _FuncOp([], False, has_prototype=False)  # 'int f()': old style, no info
        if self.tok.is_keyword("void") and self.peek().is_punct(")"):
            self.advance()
            self.advance()
            return _FuncOp([], False, has_prototype=True)
        depth = self.scopes.depth
        self.scopes.push()  # parameter names shadow typedefs only inside the prototype
        try:
            params, variadic = self._parse_parameters()
        finally:
            self.scopes.truncate(depth)
        self.expect_close(")", opener)
        return _FuncOp(params, variadic, has_prototype=True)

    def _parse_parameters(self) -> tuple[list[_Param], bool]:
        params: list[_Param] = []
        while True:
            if self.at_punct("..."):
                if not params:
                    self.error("expected_token", "ISO C requires a named argument before '...'",
                               self.tok.loc, expected="a named parameter", found="'...'")
                self.advance()
                return params, True
            params.append(self._parse_parameter())
            if not self.accept(","):
                return params, False

    def _parse_parameter(self) -> _Param:
        start = self.tok
        spec = self._parse_decl_specifiers("param")
        if spec is None:
            self.fail("expected_token",
                      f"expected declaration specifiers or '...' {self._before()}", start.loc,
                      friendly="Inside a function's parentheses C expected a parameter type, "
                               "like 'int x'. If this is a call, it must be inside a function "
                               "body.")
        declarator = self._parse_declarator("param")
        ctype = self._build_type(spec.ctype, declarator)
        is_const, _ = _qualifiers(spec.is_const, declarator.ops)
        if ctype.is_void:
            self.error("bad_declarator", "'void' must be the only parameter", start.loc,
                       detail="Write (void) for 'no parameters'; otherwise give each "
                              "parameter a real type.")
        # C adjusts array and function parameters to pointers: 'char *argv[]' is 'char **'.
        if isinstance(ctype, ArrayType):
            ctype = PointerType(ctype.element)
        elif isinstance(ctype, FunctionType):
            ctype = PointerType(ctype)
        if declarator.name:
            self.scopes.declare(declarator.name, Symbol("var", ctype))
        loc = declarator.loc if declarator.name else start.loc
        return _Param(declarator.name, ctype, is_const, loc)

    def _build_type(self, base: CType, declarator: Declarator) -> CType:
        """Apply the declarator's operations to the base type (see Declarator)."""
        ctype = base
        for op in declarator.ops:
            ctype = self._apply_op(ctype, op, declarator)
        return ctype

    def _apply_op(self, ctype: CType, op: _DeclOp, declarator: Declarator) -> CType:
        name = declarator.name or "type name"
        if isinstance(op, _PtrOp):
            return PointerType(ctype)
        if isinstance(op, _ArrayOp):
            if ctype.is_function:
                self._declarator_error(f"declaration of '{name}' as array of functions",
                                       declarator, "Use an array of function pointers instead.")
            elif ctype.is_void:
                self._declarator_error(f"declaration of '{name}' as array of voids", declarator,
                                       "An array needs an element type such as int or char.")
            elif isinstance(ctype, ArrayType) and ctype.length is None:
                self._declarator_error("array type has incomplete element type", declarator,
                                       "Only the first [] of a multi-dimensional array may "
                                       "be empty.")
            return ArrayType(ctype, op.length)
        if ctype.is_array:
            self._declarator_error(f"'{name}' declared as function returning an array",
                                   declarator, "Return a pointer, or a struct holding the array.")
        elif ctype.is_function:
            self._declarator_error(f"'{name}' declared as function returning a function",
                                   declarator, "Return a function pointer instead.")
        return FunctionType(ctype, tuple(param.ctype for param in op.params), op.variadic,
                            op.has_prototype)

    def _declarator_error(self, message: str, declarator: Declarator, detail: str) -> None:
        self.error("bad_declarator", message, declarator.loc, detail=detail)

    # ----------------------------------------------------------- initializers
    @_nested
    def _parse_initializer(self) -> ast.Expr:
        """An expression, or a brace list that may nest: '{1, {2, 3}, }'."""
        if not self.at_punct("{"):
            return self._parse_assignment()
        opener = self.advance()
        items: list[ast.Expr] = []
        while not self.at_punct("}"):
            if self.at_eof:
                self._report_eof_brace(opener)
                break
            if self.at_punct(".", "["):
                # Report it and step over the designator, so the value and the rest of
                # the function still parse: unwinding here would make the following
                # statements look like file-scope junk and bury the one real error.
                self.error("unsupported_feature", "designated initializers are not supported",
                           self.tok.loc, detail="List the values in order instead, like "
                                                "{1, 2, 3}.")
                self._skip_designator()
            items.append(self._parse_initializer())
            if not self.accept(","):
                break
        if not self.at_eof:
            self.expect_close("}", opener)
        return ast.InitList(opener.loc, items)

    def _skip_designator(self) -> None:
        """Step over '.x', '[3]' (or a chain of them) and the '=' after them."""
        while self.at_punct(".", "["):
            if self.advance().is_punct("."):
                if self.tok.kind is TokenKind.IDENT:
                    self.advance()
                continue
            depth = 1
            while depth and not self.at_eof and not self.at_punct(";", "}"):
                token = self.advance()
                depth += 1 if token.is_punct("[") else -1 if token.is_punct("]") else 0
        self.accept("=")

    def _complete_array_type(self, ctype: CType, init: ast.Expr | None,
                             declarator: Declarator, spec: DeclSpec) -> CType:
        """Infer 'int a[] = {1, 2}' -> int[2], 'char s[] = "abc"' -> char[4]; check fits."""
        if not isinstance(ctype, ArrayType):
            return ctype
        name = declarator.name or ""
        if init is None:
            if ctype.length is None and spec.storage != "extern":
                self.error("bad_declarator", f"array size missing in '{name}'", declarator.loc,
                           detail="Give the array a size, like 'int a[10];', or initialize it "
                                  "so C can count: 'int a[] = {1, 2, 3};'.")
            return ctype
        string = _string_initializer(init)
        is_char_array = isinstance(ctype.element, IntType) and ctype.element.size == 1
        if string is not None and is_char_array:
            return self._fit_string(ctype, string, name)
        if isinstance(init, ast.InitList):
            return self._fit_list(ctype, init)
        message = ("array of inappropriate type initialized from string constant"
                   if string is not None else "invalid initializer")
        self.error("bad_initializer", message, init.loc,
                   detail="An array is initialized with a brace list, like {1, 2, 3}"
                          + (" or, for char arrays, a string." if string is None else "."))
        return ctype

    def _fit_string(self, ctype: ArrayType, string: ast.StringLiteral, name: str) -> CType:
        needed = len(string.value) + 1  # the hidden NUL at the end
        if ctype.length is None:
            return ArrayType(ctype.element, needed)
        if needed - 1 > ctype.length:
            self.warning("string_literal_to_char_array",
                         "initializer-string for array of 'char' is too long", string.loc,
                         friendly=f"'{name}' holds {ctype.length} chars but the string needs "
                                  f"{needed} (one extra for the '\\0' at the end).")
        return ctype

    def _fit_list(self, ctype: ArrayType, init: ast.InitList) -> CType:
        count = len(init.items)
        if ctype.length is None:
            return ArrayType(ctype.element, count)
        if count > ctype.length:
            self.warning("bad_initializer", "excess elements in array initializer",
                         init.items[ctype.length].loc,
                         detail=f"The array has room for {ctype.length} values but the list "
                                f"has {count}. The extra ones are dropped.")
        return ctype

    # ================================================================ statements
    @_nested
    def _parse_compound(self, new_scope: bool = True,
                        brace_missing: bool = False) -> ast.CompoundStmt:
        """'{' block items '}'. ``brace_missing``: the '{' was forgotten and reported."""
        opener = self.tok
        if not brace_missing and not self.accept("{"):
            self.fail("expected_token", f"expected '{{' {self._before()}", self.tok.loc,
                      expected="'{'", found=self._found())
        depth = self.scopes.depth
        if new_scope:
            self.scopes.push()
        items: list[ast.Node] = []
        closer: Token | None = None
        try:
            while not self.at_punct("}"):
                if self.at_eof:
                    self._report_eof_brace(opener)
                    break
                item = self._parse_block_item_guarded()
                if item is not None:
                    items.append(item)
            closer = self.accept("}")
        finally:
            self.scopes.truncate(depth)
        return ast.CompoundStmt(opener.loc, items, end_loc=closer.loc if closer else None)

    def _parse_block_item_guarded(self) -> ast.Node | None:
        """One declaration or statement; on a hard error skip it and return None."""
        depth = self.scopes.depth
        start = self.pos
        self._pending_tags = []
        try:
            item = self._parse_block_item()
        except _ParseError:
            self.scopes.truncate(depth)
            self._skip_statement()
            return None
        if self.pos == start:
            self.advance()  # never loop forever on a token nobody consumes
        return item

    def _parse_block_item(self) -> ast.Node:
        if self._starts_declaration():
            spec = self._parse_decl_specifiers("block")
            if spec is None:
                self.fail("unexpected_token", f"expected expression {self._before()}",
                          self.tok.loc, found=self._found())
            return ast.DeclStmt(spec.loc, self._parse_declaration_rest(spec, "block"))
        return self._parse_statement()

    def _starts_declaration(self) -> bool:
        token = self.tok
        if token.kind is TokenKind.KEYWORD:
            return token.value in DECL_START_WORDS
        if token.kind is not TokenKind.IDENT or self.peek().is_punct(":"):
            return False
        symbol = self.scopes.lookup(str(token.value))
        if symbol is not None:
            return symbol.kind == "typedef"
        return self._looks_like_unknown_type("block")

    def _parse_statement(self) -> ast.Stmt:
        """One statement: a block, ';', a keyword statement or an expression with ';'."""
        token = self.tok
        if token.is_punct("{"):
            return self._parse_compound()
        if token.is_punct(";"):
            self.advance()
            return ast.EmptyStmt(token.loc)
        if token.kind is TokenKind.KEYWORD:
            handler = self._STATEMENT_HANDLERS.get(str(token.value))
            if handler is not None:
                return self._parse_keyword_statement(handler)
        if token.kind is TokenKind.IDENT and self.peek().is_punct(":"):
            self.fail("unsupported_feature", "labels and 'goto' are not supported", token.loc,
                      detail="Use loops with break and continue instead; they are easier "
                             "to follow.")
        expr = self._parse_expression()
        self.expect_semicolon()
        return ast.ExprStmt(token.loc, expr)

    @_nested
    def _nested_call(self, parse_operand: Callable[[], ast.Expr]) -> ast.Expr:
        """Parse an operand one nesting level down ('- - - x' and '(int)(char)x' chain)."""
        return parse_operand()

    @_nested
    def _parse_keyword_statement(self, handler: Callable[[Parser], ast.Stmt]) -> ast.Stmt:
        """Run an if/while/for/... handler as one nesting level ('if' bodies can nest)."""
        return handler(self)

    def _parse_condition(self, keyword: str) -> ast.Expr:
        """'(cond)' after if/while/switch. A missing '(' is reported once, not twice."""
        opener = self.tok
        has_open = self.accept("(") is not None
        if not has_open:
            self.error("expected_token", f"expected '(' {self._before()}", self.tok.loc,
                       friendly=f"The condition goes in parentheses: {keyword} (x > 0).")
        cond = self._parse_expression()
        if has_open:
            self.expect_close(")", opener)
        else:
            self.accept(")")
        return cond

    def _parse_if(self) -> ast.Stmt:
        keyword = self.advance()
        cond = self._parse_condition("if")
        then = self._parse_statement()
        otherwise = None
        if self.tok.is_keyword("else"):
            self.advance()
            otherwise = self._parse_statement()
        return ast.IfStmt(keyword.loc, cond, then, otherwise)

    def _parse_else_without_if(self) -> ast.Stmt:
        keyword = self.advance()
        self.error("else_without_if_stmt", "'else' without a previous 'if'", keyword.loc)
        return self._parse_statement()  # keep the else-branch so later errors still show

    def _parse_while(self) -> ast.Stmt:
        keyword = self.advance()
        cond = self._parse_condition("while")
        return ast.WhileStmt(keyword.loc, cond, self._parse_statement())

    def _parse_do(self) -> ast.Stmt:
        keyword = self.advance()
        body = self._parse_statement()
        if not self.tok.is_keyword("while"):
            self.fail("expected_token", f"expected 'while' {self._before()}", self.tok.loc,
                      expected="'while'", found=self._found())
        self.advance()
        cond = self._parse_condition("while")
        self.expect_semicolon()
        return ast.DoWhileStmt(keyword.loc, body, cond)

    def _parse_for(self) -> ast.Stmt:
        keyword = self.advance()
        opener = self.tok
        self.expect_token("(", fatal=True)
        depth = self.scopes.depth
        self.scopes.push()  # 'for (int i = 0; ...)': i lives only inside the loop
        try:
            init = self._parse_for_init()
            cond = None if self.at_punct(";") else self._parse_expression()
            self.expect_semicolon()
            step = None if self.at_punct(")") else self._parse_expression()
            self.expect_close(")", opener)
            body = self._parse_statement()
        finally:
            self.scopes.truncate(depth)
        return ast.ForStmt(keyword.loc, init, cond, step, body)

    def _parse_for_init(self) -> ast.Node | None:
        if self.accept(";"):
            return None
        if self._starts_declaration():
            spec = self._parse_decl_specifiers("block")
            assert spec is not None  # _starts_declaration saw a specifier
            return ast.DeclStmt(spec.loc, self._parse_declaration_rest(spec, "block"))
        init = self._parse_expression()
        self.expect_semicolon()
        return init

    def _parse_switch(self) -> ast.Stmt:
        keyword = self.advance()
        cond = self._parse_condition("switch")
        return ast.SwitchStmt(keyword.loc, cond, self._parse_statement())

    def _parse_case(self) -> ast.Stmt:
        keyword = self.advance()
        value = self._parse_conditional()
        folded = eval_const_expr(value, self.scopes)
        if folded is None:
            self.error("case_not_constant", "case label does not reduce to an integer constant",
                       value.loc)
        self._expect_label_colon()
        return ast.CaseLabel(keyword.loc, value, folded, self._parse_label_body())

    def _parse_default(self) -> ast.Stmt:
        keyword = self.advance()
        self._expect_label_colon()
        return ast.DefaultLabel(keyword.loc, self._parse_label_body())

    def _expect_label_colon(self) -> None:
        if self.accept(":"):
            return
        if self.at_punct(";"):
            # 'case 1;' is a classic slip: report it, then treat the ';' as the ':'.
            self.error("expected_token", "expected ':' or '...' before ';' token",
                       self._missing_loc(), expected="':'", found="';'")
            self.advance()
            return
        self.expect_token(":")

    def _parse_label_body(self) -> ast.Stmt:
        if self.at_punct("}"):
            return ast.EmptyStmt(self.tok.loc)  # 'default: }' is accepted, as gcc does
        if self._starts_declaration():
            self.error("unexpected_token",
                       "a label can only be part of a statement and a declaration is not a "
                       "statement", self.tok.loc,
                       friendly="Put braces around the case body to declare variables in it: "
                                "case 1: { int x = 5; ... break; }")
            spec = self._parse_decl_specifiers("block")
            assert spec is not None
            return ast.DeclStmt(spec.loc, self._parse_declaration_rest(spec, "block"))
        return self._parse_statement()

    def _parse_return(self) -> ast.Stmt:
        keyword = self.advance()
        value = None if self.at_punct(";") else self._parse_expression()
        self.expect_semicolon()
        return ast.ReturnStmt(keyword.loc, value)

    def _parse_break(self) -> ast.Stmt:
        keyword = self.advance()
        self.expect_semicolon()
        return ast.BreakStmt(keyword.loc)

    def _parse_continue(self) -> ast.Stmt:
        keyword = self.advance()
        self.expect_semicolon()
        return ast.ContinueStmt(keyword.loc)

    def _parse_goto(self) -> ast.Stmt:
        self.fail("unsupported_feature", "labels and 'goto' are not supported", self.tok.loc,
                  detail="Use loops with break and continue instead; they are easier to follow.")

    _STATEMENT_HANDLERS = {
        "if": _parse_if, "else": _parse_else_without_if, "while": _parse_while,
        "do": _parse_do, "for": _parse_for, "switch": _parse_switch, "case": _parse_case,
        "default": _parse_default, "return": _parse_return, "break": _parse_break,
        "continue": _parse_continue, "goto": _parse_goto,
    }

    # =============================================================== expressions
    def _parse_expression(self) -> ast.Expr:
        """expression := assignment (',' assignment)*"""
        first = self._parse_assignment()
        if not self.at_punct(","):
            return first
        exprs = [first]
        while self.accept(","):
            exprs.append(self._parse_assignment())
        return ast.Comma(first.loc, exprs)

    def _parse_assignment(self) -> ast.Expr:
        """Right-associative: 'a = b = c' is 'a = (b = c)'."""
        target = self._parse_conditional()
        token = self.tok
        if token.kind is TokenKind.PUNCT and token.value in ASSIGN_OPS:
            self.advance()
            value = self._parse_assignment()
            return ast.Assignment(token.loc, str(token.value), target, value)
        return target

    def _parse_conditional(self) -> ast.Expr:
        cond = self._parse_binary(1)
        question = self.accept("?")
        if question is None:
            return cond
        then = self._parse_expression()
        self.expect_token(":")
        otherwise = self._parse_conditional()
        return ast.Conditional(question.loc, cond, then, otherwise)

    def _parse_binary(self, min_precedence: int) -> ast.Expr:
        """Precedence climbing over BINARY_PRECEDENCE; all of these are left-associative."""
        left = self._parse_cast()
        while True:
            token = self.tok
            precedence = 0
            if token.kind is TokenKind.PUNCT:
                precedence = BINARY_PRECEDENCE.get(str(token.value), 0)
            if precedence < min_precedence:
                return left
            self.advance()
            right = self._parse_binary(precedence + 1)
            left = ast.BinaryOp(token.loc, str(token.value), left, right)

    def _is_type_name_start(self, token: Token) -> bool:
        if token.kind is TokenKind.KEYWORD:
            return token.value in TYPE_START_WORDS
        return token.kind is TokenKind.IDENT and self.scopes.is_typedef(str(token.value))

    def _parse_type_name(self) -> CType:
        """A type in a cast or sizeof: 'unsigned char', 'int (*)(int)', 'Tile *'."""
        spec = self._parse_decl_specifiers("typename")
        if spec is None:
            self.fail("unexpected_token", f"expected expression {self._before()}", self.tok.loc,
                      found=self._found())
        return self._build_type(spec.ctype, self._parse_declarator("typename"))

    def _parse_cast(self) -> ast.Expr:
        if self.at_punct("(") and self._is_type_name_start(self.peek()):
            opener = self.advance()
            ctype = self._parse_type_name()
            self.expect_close(")", opener)
            if self.at_punct("{"):
                self.fail("unsupported_feature", "compound literals are not supported",
                          self.tok.loc, detail="Declare a variable with an initializer first, "
                                               "then use the variable.")
            return ast.Cast(opener.loc, ctype, self._nested_call(self._parse_cast))
        return self._parse_unary()

    def _parse_unary(self) -> ast.Expr:
        token = self.tok
        if token.is_punct("++", "--"):
            self.advance()
            return ast.UnaryOp(token.loc, str(token.value), self._nested_call(self._parse_unary))
        if token.kind is TokenKind.PUNCT and token.value in UNARY_OPS:
            self.advance()
            return ast.UnaryOp(token.loc, str(token.value), self._nested_call(self._parse_cast))
        if token.is_keyword("sizeof"):
            return self._parse_sizeof()
        return self._parse_postfix(self._parse_primary())

    def _parse_sizeof(self) -> ast.Expr:
        keyword = self.advance()
        if self.at_punct("(") and self._is_type_name_start(self.peek()):
            opener = self.advance()
            ctype = self._parse_type_name()
            self.expect_close(")", opener)
            return ast.SizeofType(keyword.loc, ctype)
        return ast.SizeofExpr(keyword.loc, self._parse_unary())

    def _parse_postfix(self, expr: ast.Expr) -> ast.Expr:
        while True:
            token = self.tok
            if token.is_punct("["):
                self.advance()
                index = self._parse_expression()
                self.expect_close("]", token)
                expr = ast.Index(token.loc, expr, index)
            elif token.is_punct("("):
                expr = ast.Call(expr.loc, expr, self._parse_call_args())
            elif token.is_punct(".", "->"):
                self.advance()
                if self.tok.kind is not TokenKind.IDENT:
                    self.fail("expected_token", f"expected identifier {self._before()}",
                              self.tok.loc, expected="a member name", found=self._found())
                name = str(self.advance().value)
                expr = ast.Member(token.loc, expr, name, arrow=token.value == "->")
            elif token.is_punct("++", "--"):
                self.advance()
                expr = ast.PostfixOp(token.loc, str(token.value), expr)
            else:
                return expr

    @_nested
    def _parse_call_args(self) -> list[ast.Expr]:
        opener = self.advance()
        args: list[ast.Expr] = []
        if not self.at_punct(")"):
            args.append(self._parse_assignment())
            while self.accept(","):
                args.append(self._parse_assignment())
        self.expect_close(")", opener)
        return args

    @_nested
    def _parse_primary(self) -> ast.Expr:
        token = self.tok
        if token.kind is TokenKind.IDENT:
            self.advance()
            if token.value == FUNC_NAME and self._function_name is not None:
                return self._function_name_literal(token)
            return ast.Identifier(token.loc, str(token.value))
        if token.kind is TokenKind.INT:
            self.advance()
            return ast.IntLiteral(token.loc, int(token.value), token.ctype or INT, token.text)
        if token.kind is TokenKind.FLOAT:
            self.advance()
            return ast.FloatLiteral(token.loc, float(token.value), token.ctype or DOUBLE,
                                    token.text)
        if token.kind is TokenKind.CHAR:
            self.advance()
            return ast.CharLiteral(token.loc, int(token.value), token.text)
        if token.kind is TokenKind.STRING:
            return self._parse_strings()
        if token.is_punct("("):
            opener = self.advance()
            expr = self._parse_expression()
            self.expect_close(")", opener)
            expr.parenthesized = True
            return expr
        self.fail("unexpected_token", f"expected expression {self._before()}", token.loc,
                  found=self._found())

    def _function_name_literal(self, token: Token) -> ast.StringLiteral:
        """``__func__`` inside a function: C99 says it behaves like
        ``static const char __func__[] = "name";``, which is a string literal's value
        (read-only, NUL-terminated). ``assert`` uses it for its message."""
        name = self._function_name or ""
        return ast.StringLiteral(token.loc, name.encode("utf-8"), f'"{name}"')

    def _parse_strings(self) -> ast.StringLiteral:
        """Adjacent literals join: "py" "cc" is one string, as in C."""
        first = self.tok
        parts: list[bytes] = []
        texts: list[str] = []
        while self.tok.kind is TokenKind.STRING:
            token = self.advance()
            parts.append(bytes(token.value))  # type: ignore[arg-type]
            texts.append(token.text)
        return ast.StringLiteral(first.loc, b"".join(parts), " ".join(texts))


def _string_initializer(init: ast.Expr) -> ast.StringLiteral | None:
    """The string in 'char s[] = "abc"' or 'char s[] = {"abc"}', if that is the form."""
    if isinstance(init, ast.StringLiteral):
        return init
    if isinstance(init, ast.InitList) and len(init.items) == 1:
        only = init.items[0]
        if isinstance(only, ast.StringLiteral):
            return only
    return None


def _basic_type_problem(counts: Counter, bases: list[str]) -> str | None:
    """gcc's complaint about a bad keyword combination, or None if it is fine."""
    if counts["signed"] and counts["unsigned"]:
        return "both 'signed' and 'unsigned' in declaration specifiers"
    if counts["long"] > 2:
        return "'long long long' is too long for GCC"
    if counts["short"] and counts["long"]:
        return "both 'long' and 'short' in declaration specifiers"
    if len(bases) > 1:
        return "two or more data types in declaration specifiers"
    base = bases[0] if bases else "int"
    modifiers = counts["signed"] + counts["unsigned"] + counts["short"] + counts["long"]
    if base in ("void", "_Bool", "float") and modifiers:
        return f"both modifiers and '{base}' in declaration specifiers"
    if base == "double" and (modifiers != counts["long"] or counts["long"] > 1):
        return "invalid modifiers with 'double' in declaration specifiers"
    if base == "char" and (counts["short"] or counts["long"]):
        return "invalid modifiers with 'char' in declaration specifiers"
    return None


def parse(tokens: Iterable[Token], unit_name: str) -> ParseResult:
    """Parse a token list (from ``tokenize``) into a TranslationUnit."""
    _ensure_recursion_headroom()
    parser = Parser(tokens, unit_name)
    unit = parser.parse_translation_unit()
    return ParseResult(unit, parser.diagnostics)


def _ensure_recursion_headroom() -> None:
    """Make sure MAX_NESTING levels fit under Python's recursion limit.

    We only ever raise the limit, never lower it again: lowering it while another
    thread is parsing could make that parse fail. Python 3.12 keeps Python frames on
    the heap, so a higher limit does not risk the C stack.
    """
    needed = MAX_NESTING * _FRAMES_PER_LEVEL + _RECURSION_HEADROOM
    if sys.getrecursionlimit() < needed:
        sys.setrecursionlimit(needed)


def parse_text(text: str, filename: str = "main.c") -> ParseResult:
    """Lex and parse raw text (no preprocessing); diagnostics come in source order."""
    diagnostics: list[Diagnostic] = []
    tokens = tokenize(lines_from_text(text, filename), diagnostics)
    result = parse(tokens, filename)
    combined = diagnostics + result.diagnostics
    return ParseResult(result.ast, sorted(combined, key=_source_order))


def _source_order(diag: Diagnostic) -> tuple[int, int, int]:
    if diag.location is None:
        return (1, 0, 0)  # notes without a place, such as 'too many errors', go last
    return (0, diag.location.line, diag.location.column)


# --- prototypes of library functions ---------------------------------------------------

# The library's own typedefs, so prototypes such as "size_t strlen(const char *s);" or
# "FILE *fopen(const char *path, const char *mode);" parse without their header.
# FILE matches the layout stdio.h declares.
_LIBRARY_FILE = StructType("_CVFILE", [Member("_id", INT)])
LIBRARY_TYPEDEFS: dict[str, CType] = {
    "size_t": ULONG, "ssize_t": LONG, "ptrdiff_t": LONG, "time_t": LONG, "clock_t": LONG,
    "int8_t": SCHAR, "uint8_t": UCHAR, "int16_t": SHORT, "uint16_t": USHORT,
    "int32_t": INT, "uint32_t": UINT, "int64_t": LONG, "uint64_t": ULONG,
    "bool": BOOL, "FILE": _LIBRARY_FILE,
}


def parse_prototype(text: str, typedefs: dict[str, CType] | None = None) -> FunctionType:
    """Parse one C prototype, such as ``"int move(int direction);"``, into its type.

    ``typedefs`` adds type names the prototype may use on top of ``LIBRARY_TYPEDEFS``.
    Raises ``ValueError`` when the text is not exactly one valid function prototype,
    because a broken prototype is a bug in the library, never a user mistake.
    """
    source = text.strip()
    if not source.endswith(";"):
        source += ";"
    diagnostics: list[Diagnostic] = []
    tokens = tokenize(lines_from_text(source, "<prototype>"), diagnostics)
    parser = Parser(tokens, "<prototype>")
    for name, ctype in {**LIBRARY_TYPEDEFS, **(typedefs or {})}.items():
        parser.scopes.declare(name, Symbol("typedef", ctype))
    unit = parser.parse_translation_unit()
    errors = [diag for diag in diagnostics + parser.diagnostics if diag.is_error]
    functions = [decl for decl in unit.decls if isinstance(decl, ast.FunctionDecl)]
    if errors or len(functions) != 1:
        problem = errors[0].message if errors else "expected exactly one function prototype"
        raise ValueError(f"bad prototype {text!r}: {problem}")
    return functions[0].ftype
