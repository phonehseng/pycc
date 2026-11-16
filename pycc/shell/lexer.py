"""Shell lexer: turns a command line into words and operators.

A shell word is not just text. ``"$HOME/src"`` and ``$HOME/src`` look alike but behave
differently: quoted text is never split into several words or globbed. So instead of a
plain string each word is a list of *parts* that remember whether they were quoted and
whether they are expansions (``$X``, ``$(...)``) to be filled in later. The lexer only
records what was typed; ``shell/expand.py`` does the expanding when the command runs,
because ``$?`` must reflect the command that ran just before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Longest operators first so ">>" is not read as two ">".
OPERATORS: tuple[str, ...] = (
    "2>&1", "1>&2", "2>>", "&>>", ">&2", "&&", "||", ">>", "2>", "&>", "1>",
    "|", ";", "<", ">", "(", ")", "&",
)
_OPERATOR_START = set("|;&<>()")
_SPECIAL_PARAMS = set("?#@*$0123456789")


class LexError(Exception):
    """Unfinished input, such as a missing closing quote.

    ``incomplete`` is True when more typed lines could finish the input, which lets
    the terminal show a ``>`` continuation prompt instead of an error.
    """

    def __init__(self, message: str, incomplete: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.incomplete = incomplete


@dataclass
class Lit:
    """Literal text. ``quoted`` text is safe from word splitting and globbing."""

    text: str
    quoted: bool = False


@dataclass
class Param:
    """A variable reference such as ``$HOME``, ``${X}``, ``$?`` or ``$1``."""

    name: str
    quoted: bool = False


@dataclass
class CommandSub:
    """A command substitution ``$(...)`` (or backticks) holding its source text."""

    source: str
    quoted: bool = False


@dataclass
class Arith:
    """An arithmetic expansion ``$(( ... ))`` holding the expression text."""

    expression: str
    quoted: bool = False


WordPart = Lit | Param | CommandSub | Arith


@dataclass
class Word:
    """One shell word made of parts; ``raw`` is the text exactly as typed."""

    parts: list[WordPart] = field(default_factory=list)
    raw: str = ""

    @property
    def plain(self) -> str | None:
        """The text if the word is one unquoted literal (keywords, names), else None."""
        if len(self.parts) == 1 and isinstance(self.parts[0], Lit) and not self.parts[0].quoted:
            return self.parts[0].text
        return None


@dataclass
class Token:
    """A lexer token: ``kind`` is ``word``, ``op``, ``newline`` or ``eof``."""

    kind: str
    text: str
    word: Word | None = None


class Lexer:
    """Single-pass scanner over one chunk of shell source."""

    def __init__(self, source: str) -> None:
        """Prepare to scan ``source`` (which may contain several lines)."""
        self.source = source
        self.pos = 0

    # --- character helpers ----------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        index = self.pos + offset
        return self.source[index] if index < len(self.source) else ""

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    # --- top level ------------------------------------------------------------

    def tokens(self) -> list[Token]:
        """Scan everything and return the tokens, ending with an ``eof`` token."""
        result: list[Token] = []
        while True:
            token = self._next_token()
            result.append(token)
            if token.kind == "eof":
                return result

    def _next_token(self) -> Token:
        self._skip_blanks_and_comments()
        if self._at_end():
            return Token("eof", "")
        char = self._peek()
        if char == "\n":
            self.pos += 1
            return Token("newline", "\n")
        operator = self._match_operator()
        if operator:
            self.pos += len(operator)
            return Token("op", operator)
        start = self.pos
        word = self._scan_word()
        word.raw = self.source[start:self.pos]
        return Token("word", word.raw, word)

    def _skip_blanks_and_comments(self) -> None:
        while not self._at_end():
            char = self._peek()
            if char in " \t\r":
                self.pos += 1
            elif char == "\\" and self._peek(1) == "\n":
                self.pos += 2           # line continuation between words
            elif char == "#":
                while not self._at_end() and self._peek() != "\n":
                    self.pos += 1
            else:
                return

    def _match_operator(self) -> str | None:
        char = self._peek()
        if char not in _OPERATOR_START and not (char in "12" and self._peek(1) == ">"):
            return None
        for operator in OPERATORS:
            if self.source.startswith(operator, self.pos):
                return operator
        return None

    # --- words -----------------------------------------------------------------

    def _scan_word(self) -> Word:
        word = Word()
        while not self._at_end():
            char = self._peek()
            if char in " \t\r\n" or char in _OPERATOR_START:
                break
            if char == "\\":
                self._scan_backslash(word)
            elif char == "'":
                self._scan_single_quoted(word)
            elif char == '"':
                self._scan_double_quoted(word)
            elif char == "$":
                self._scan_dollar(word, quoted=False)
            elif char == "`":
                self._scan_backticks(word, quoted=False)
            else:
                _add_text(word, char, quoted=False)
                self.pos += 1
        return word

    def _scan_backslash(self, word: Word) -> None:
        nxt = self._peek(1)
        if nxt == "\n":
            self.pos += 2               # continuation inside a word joins the lines
        elif nxt == "":
            _add_text(word, "\\", quoted=True)
            self.pos += 1
        else:
            _add_text(word, nxt, quoted=True)
            self.pos += 2

    def _scan_single_quoted(self, word: Word) -> None:
        end = self.source.find("'", self.pos + 1)
        if end < 0:
            raise LexError("unexpected EOF while looking for matching `''")
        # '' must still produce a (quoted, empty) part so that '' is an argument.
        word.parts.append(Lit(self.source[self.pos + 1:end], quoted=True))
        self.pos = end + 1

    def _scan_double_quoted(self, word: Word) -> None:
        self.pos += 1
        # A quoted empty part makes "" a real (empty) argument. It must go again when
        # an expansion follows, otherwise "$@" with no arguments would still leave one
        # empty argument behind instead of none.
        placeholder = Lit("", quoted=True)
        word.parts.append(placeholder)
        while True:
            if self._at_end():
                raise LexError("unexpected EOF while looking for matching `\"'")
            char = self._peek()
            if char == '"':
                self.pos += 1
                _drop_unused_placeholder(word, placeholder)
                return
            if char == "\\" and self._peek(1) in '$`"\\\n':
                if self._peek(1) != "\n":
                    _add_text(word, self._peek(1), quoted=True)
                self.pos += 2
            elif char == "$":
                self._scan_dollar(word, quoted=True)
            elif char == "`":
                self._scan_backticks(word, quoted=True)
            else:
                _add_text(word, char, quoted=True)
                self.pos += 1

    def _scan_dollar(self, word: Word, quoted: bool) -> None:
        nxt = self._peek(1)
        if nxt == "(":
            end = _find_closing_paren(self.source, self.pos + 2)
            if end < 0:
                raise LexError("unexpected EOF while looking for matching `)'")
            body = self.source[self.pos + 2:end]
            if _is_arithmetic(body):
                word.parts.append(Arith(body[1:-1], quoted))
            else:
                word.parts.append(CommandSub(body, quoted))
            self.pos = end + 1
        elif nxt == "{":
            end = self.source.find("}", self.pos + 2)
            if end < 0:
                raise LexError("unexpected EOF while looking for matching `}'")
            word.parts.append(Param(self.source[self.pos + 2:end], quoted))
            self.pos = end + 1
        elif nxt in _SPECIAL_PARAMS and nxt:
            word.parts.append(Param(nxt, quoted))
            self.pos += 2
        elif nxt.isalpha() or nxt == "_":
            end = self.pos + 1
            while end < len(self.source) and (self.source[end].isalnum()
                                              or self.source[end] == "_"):
                end += 1
            word.parts.append(Param(self.source[self.pos + 1:end], quoted))
            self.pos = end
        else:
            _add_text(word, "$", quoted)   # a lone $ is just a dollar sign
            self.pos += 1

    def _scan_backticks(self, word: Word, quoted: bool) -> None:
        end = self.source.find("`", self.pos + 1)
        if end < 0:
            raise LexError("unexpected EOF while looking for matching ``'")
        word.parts.append(CommandSub(self.source[self.pos + 1:end], quoted))
        self.pos = end + 1


def _add_text(word: Word, text: str, quoted: bool) -> None:
    """Append text, merging with the previous literal of the same kind."""
    if word.parts and isinstance(word.parts[-1], Lit) and word.parts[-1].quoted == quoted:
        word.parts[-1].text += text
    else:
        word.parts.append(Lit(text, quoted))


def _drop_unused_placeholder(word: Word, placeholder: Lit) -> None:
    """Remove the empty part of a double-quoted string once something else is in it."""
    index = next(i for i, part in enumerate(word.parts) if part is placeholder)
    if not placeholder.text and index < len(word.parts) - 1:
        del word.parts[index]


def _is_arithmetic(body: str) -> bool:
    """``$((1 + 2))``: the body of the ``$(`` is one parenthesised group.

    ``$( (cd src; ls) )`` with spaces is a sub-shell inside a command substitution,
    as in bash, so only a ``(`` straight after ``$(`` that closes at the very end
    counts.
    """
    return (body.startswith("(") and body.endswith(")")
            and _find_closing_paren(body, 1) == len(body) - 1)


def _find_closing_paren(source: str, index: int) -> int:
    """Index of the ``)`` closing a ``$(`` whose body starts at ``index`` (or -1).

    Parentheses inside quotes do not count, so ``$(echo ")")`` works.
    """
    depth = 1
    quote = ""
    while index < len(source):
        char = source[index]
        if quote:
            if char == quote:
                quote = ""
            elif char == "\\" and quote == '"':
                index += 1
        elif char in "'\"":
            quote = char
        elif char == "\\":
            index += 1
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def tokenize(source: str) -> list[Token]:
    """Convenience wrapper: all tokens of ``source``."""
    return Lexer(source).tokens()
