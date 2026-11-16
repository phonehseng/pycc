"""Shell parser: tokens -> a small syntax tree.

The grammar is the beginner-sized subset of bash listed in docs/SUPPORTED_SHELL.md:

    list      := and_or ((';' | newline) and_or)*
    and_or    := pipeline (('&&' | '||') pipeline)*
    pipeline  := ['!'] command ('|' command)*
    command   := simple | if | for | while | until | '(' list ')'   [redirections]
    simple    := (NAME=value)* word* with redirections anywhere

Keywords (``if``, ``then``, ``do``...) are only keywords when they are unquoted and
in command position, exactly like bash, so ``echo then`` prints "then".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pycc.shell.lexer import LexError, Token, Word, tokenize

REDIRECT_OPS = {">", ">>", "<", "2>", "2>>", "1>", "&>", "&>>", "2>&1", "1>&2", ">&2"}
# These redirections duplicate a stream and take no file name.
DUP_REDIRECTS = {"2>&1", "1>&2", ">&2"}
RESERVED = {"if", "then", "elif", "else", "fi", "for", "in", "do", "done", "while",
            "until", "!"}
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class ParseError(Exception):
    """A shell syntax error, worded like bash. ``incomplete`` means the input simply
    stopped early (an ``if`` without ``fi``), so more lines could fix it."""

    def __init__(self, message: str, incomplete: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.incomplete = incomplete


# --- syntax tree --------------------------------------------------------------------

@dataclass
class Redirect:
    """One redirection: ``op`` from ``REDIRECT_OPS`` and the file word (if any)."""

    op: str
    target: Word | None = None


@dataclass
class SimpleCommand:
    """``NAME=v ... word word ... > file``."""

    assignments: list[tuple[str, Word]] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)


@dataclass
class Pipeline:
    """Commands joined by ``|``; ``negated`` for a leading ``!``."""

    commands: list["Command"]
    negated: bool = False


@dataclass
class AndOr:
    """``first && second || third``: ``rest`` holds (operator, pipeline) pairs."""

    first: Pipeline
    rest: list[tuple[str, Pipeline]] = field(default_factory=list)


@dataclass
class CommandList:
    """Commands run one after another (``;`` or newlines)."""

    items: list[AndOr] = field(default_factory=list)


@dataclass
class IfCommand:
    """``if``/``elif`` branches as (condition, body) pairs plus an optional ``else``."""

    branches: list[tuple[CommandList, CommandList]]
    else_body: CommandList | None = None
    redirects: list[Redirect] = field(default_factory=list)


@dataclass
class ForCommand:
    """``for NAME in WORDS; do BODY; done`` (no ``in`` means ``"$@"``)."""

    name: str
    words: list[Word] | None
    body: CommandList
    redirects: list[Redirect] = field(default_factory=list)


@dataclass
class WhileCommand:
    """``while COND; do BODY; done``; ``until`` loops set ``until``."""

    condition: CommandList
    body: CommandList
    until: bool = False
    redirects: list[Redirect] = field(default_factory=list)


@dataclass
class Subshell:
    """``( LIST )``: runs in a copy of the shell so ``cd`` inside does not leak."""

    body: CommandList
    redirects: list[Redirect] = field(default_factory=list)


Command = SimpleCommand | IfCommand | ForCommand | WhileCommand | Subshell


# --- parser -----------------------------------------------------------------------

class Parser:
    """Recursive-descent parser over the token list of one source text."""

    def __init__(self, tokens: list[Token]) -> None:
        """Start at the first token."""
        self.tokens = tokens
        self.index = 0

    @property
    def current(self) -> Token:
        """The token being looked at."""
        return self.tokens[self.index]

    def _advance(self) -> Token:
        token = self.current
        if token.kind != "eof":
            self.index += 1
        return token

    def _keyword(self) -> str | None:
        token = self.current
        if token.kind == "word" and token.word is not None and token.word.plain in RESERVED:
            return token.word.plain
        return None

    def _is_op(self, *ops: str) -> bool:
        return self.current.kind == "op" and self.current.text in ops

    def _skip_newlines(self) -> None:
        while self.current.kind == "newline":
            self._advance()

    def _unexpected(self) -> ParseError:
        token = self.current
        if token.kind == "eof":
            return ParseError("syntax error: unexpected end of file", incomplete=True)
        shown = "newline" if token.kind == "newline" else token.text
        return ParseError(f"syntax error near unexpected token `{shown}'")

    def _expect_keyword(self, keyword: str) -> None:
        self._skip_newlines()
        if self._keyword() != keyword:
            raise self._unexpected()
        self._advance()

    # --- lists ------------------------------------------------------------------

    def parse_program(self) -> CommandList:
        """Parse the whole input; anything left over is a syntax error."""
        program = self.parse_list(stop=set())
        if self.current.kind != "eof":
            raise self._unexpected()
        return program

    def parse_list(self, stop: set[str]) -> CommandList:
        """Parse commands until EOF, ``)`` or one of the ``stop`` keywords."""
        result = CommandList()
        while True:
            self._skip_newlines()
            if self.current.kind == "eof" or self._is_op(")") or self._keyword() in stop:
                return result
            result.items.append(self.parse_and_or())
            if self._is_op(";"):
                self._advance()
            elif self.current.kind == "newline":
                self._advance()
            elif self._is_op("&"):
                raise ParseError("syntax error near unexpected token `&' "
                                 "(background jobs are not supported)")
            elif not (self.current.kind == "eof" or self._is_op(")") or self._keyword() in stop):
                raise self._unexpected()

    def parse_and_or(self) -> AndOr:
        """``pipeline && pipeline || pipeline``."""
        node = AndOr(self.parse_pipeline())
        while self._is_op("&&", "||"):
            operator = self._advance().text
            self._skip_newlines()
            node.rest.append((operator, self.parse_pipeline()))
        return node

    def parse_pipeline(self) -> Pipeline:
        """``[!] command | command ...``."""
        negated = False
        if self._keyword() == "!":
            self._advance()
            negated = True
        commands = [self.parse_command()]
        while self._is_op("|"):
            self._advance()
            self._skip_newlines()
            commands.append(self.parse_command())
        return Pipeline(commands, negated)

    # --- commands -----------------------------------------------------------------

    def parse_command(self) -> Command:
        """One simple or compound command (compound ones may carry redirections)."""
        keyword = self._keyword()
        if keyword == "if":
            node: Command = self._parse_if()
        elif keyword == "for":
            node = self._parse_for()
        elif keyword in ("while", "until"):
            node = self._parse_while(until=keyword == "until")
        elif self._is_op("("):
            node = self._parse_subshell()
        elif keyword is not None:
            raise self._unexpected()
        else:
            return self._parse_simple()
        node.redirects = self._parse_trailing_redirects()
        return node

    def _parse_simple(self) -> SimpleCommand:
        command = SimpleCommand()
        while True:
            token = self.current
            if token.kind == "word" and token.word is not None:
                self._advance()
                if not command.words and _ASSIGNMENT.match(token.text):
                    command.assignments.append(_split_assignment(token.word))
                else:
                    command.words.append(token.word)
            elif token.kind == "op" and token.text in REDIRECT_OPS:
                command.redirects.append(self._parse_redirect())
            else:
                break
        if not (command.words or command.assignments or command.redirects):
            raise self._unexpected()
        return command

    def _parse_redirect(self) -> Redirect:
        operator = self._advance().text
        if operator in DUP_REDIRECTS:
            return Redirect(operator)
        if self.current.kind in ("eof", "newline"):
            # A missing file name is a plain error in bash, not a request for more lines.
            raise ParseError("syntax error near unexpected token `newline'")
        if self.current.kind != "word":
            raise self._unexpected()
        return Redirect(operator, self._advance().word)

    def _parse_trailing_redirects(self) -> list[Redirect]:
        redirects: list[Redirect] = []
        while self.current.kind == "op" and self.current.text in REDIRECT_OPS:
            redirects.append(self._parse_redirect())
        return redirects

    def _parse_if(self) -> IfCommand:
        self._advance()                                     # 'if'
        branches = [self._parse_condition_and_body("then", {"elif", "else", "fi"})]
        else_body = None
        while True:
            keyword = self._keyword()
            if keyword == "elif":
                self._advance()
                branches.append(self._parse_condition_and_body("then", {"elif", "else", "fi"}))
            elif keyword == "else":
                self._advance()
                else_body = self.parse_list({"fi"})
            elif keyword == "fi":
                self._advance()
                return IfCommand(branches, else_body)
            else:
                raise self._unexpected()

    def _parse_condition_and_body(self, opener: str, closers: set[str]) -> tuple[CommandList,
                                  CommandList]:
        condition = self.parse_list({opener})
        self._require_non_empty(condition)
        self._expect_keyword(opener)
        body = self.parse_list(closers)
        self._require_non_empty(body)
        return condition, body

    def _require_non_empty(self, body: CommandList) -> None:
        # bash rejects "if; then" and "then fi": an empty block is a syntax error.
        if not body.items:
            raise self._unexpected()

    def _parse_for(self) -> ForCommand:
        self._advance()                                     # 'for'
        token = self.current
        name = token.word.plain if token.kind == "word" and token.word else None
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise self._unexpected()
        self._advance()
        words: list[Word] | None = None
        self._skip_newlines()
        if self._keyword() == "in":
            self._advance()
            words = []
            while self.current.kind == "word" and self.current.word is not None:
                words.append(self._advance().word)
        if self._is_op(";"):
            self._advance()
        self._expect_keyword("do")
        body = self.parse_list({"done"})
        self._require_non_empty(body)
        self._expect_keyword("done")
        return ForCommand(name, words, body)

    def _parse_while(self, until: bool) -> WhileCommand:
        self._advance()                                     # 'while' / 'until'
        condition, body = self._parse_condition_and_body("do", {"done"})
        self._expect_keyword("done")
        return WhileCommand(condition, body, until)

    def _parse_subshell(self) -> Subshell:
        self._advance()                                     # '('
        body = self.parse_list(set())
        self._require_non_empty(body)
        if not self._is_op(")"):
            raise self._unexpected()
        self._advance()
        return Subshell(body)


def _split_assignment(word: Word) -> tuple[str, Word]:
    """Split ``NAME=value`` into the name and a word holding just the value."""
    first = word.parts[0]
    name, _, rest = first.text.partition("=")              # first part is an unquoted Lit
    value_parts = list(word.parts[1:])
    if rest:
        value_parts.insert(0, type(first)(rest, False))
    return name, Word(value_parts, word.raw[len(name) + 1:])


def parse(source: str) -> CommandList:
    """Lex and parse ``source``; raises ``ParseError`` (lexer errors included)."""
    try:
        tokens = tokenize(source)
    except LexError as error:
        raise ParseError(error.message, incomplete=error.incomplete) from None
    return Parser(tokens).parse_program()


def needs_more_input(source: str) -> bool:
    r"""True when ``source`` is unfinished (open quote, ``if`` without ``fi``,
    trailing ``|``, ``&&`` or ``\``), so the terminal should ask for another line."""
    # An odd number of trailing backslashes ends in an escaped newline; an even
    # number is only escaped backslashes, so a line ending in two of them is complete.
    trailing = len(source) - len(source.rstrip("\\"))
    if trailing % 2 == 1:
        return True
    try:
        parse(source)
    except ParseError as error:
        return error.incomplete
    return False
