"""Word expansion: what the shell does to every word just before running a command.

The steps, in bash's order:

1. tilde: an unquoted leading ``~`` becomes the home folder;
2. parameters ``$X ${X} $? $# $0..$9 $@ $*``, command substitution ``$(...)`` and
   arithmetic ``$((...))``;
3. word splitting: *unquoted* expansion results are split on blanks, so
   ``X="a b"; ls $X`` passes two arguments while ``ls "$X"`` passes one;
4. globbing: unquoted ``* ? [..]`` are matched against the filesystem; a pattern
   that matches nothing is left as typed (bash's default);
5. quote removal, which happens for free because the lexer already took the quote
   characters out and only remembered *which* text was quoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from pycc.shell.arith import ArithError, evaluate
from pycc.shell.lexer import Arith, CommandSub, Lit, Param, Word
from pycc.vfs.filesystem import HOME, VirtualFS

_BLANKS = " \t\n"


class ExpansionError(Exception):
    """An expansion bash would refuse, e.g. ``${X:-y}`` (not supported here)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ExpansionContext(Protocol):
    """What expansion needs from the running shell."""

    vfs: VirtualFS
    cwd: str

    def lookup(self, name: str) -> str | None:
        """Value of a variable or special parameter, None when unset."""

    def positional_args(self) -> list[str]:
        """``$1``, ``$2``... as a list (for ``$@`` and ``$*``)."""

    def substitute(self, source: str) -> str:
        """Run ``source`` in a sub-shell and return its stdout."""


@dataclass
class _Field:
    """One word under construction: chunks of (text, may_glob)."""

    chunks: list[tuple[str, bool]] = field(default_factory=list)
    # A field that saw quotes survives even when empty: "" is a real argument.
    keep_empty: bool = False

    def add(self, text: str, globbable: bool) -> None:
        """Append a chunk; ``globbable`` chunks may hold live wildcards."""
        self.chunks.append((text, globbable))

    @property
    def text(self) -> str:
        """The field's final text."""
        return "".join(text for text, _ in self.chunks)

    @property
    def is_empty(self) -> bool:
        """True when the field should vanish (empty and never quoted)."""
        return not self.text and not self.keep_empty


class _FieldBuilder:
    """Collects fields while honouring word-splitting rules."""

    def __init__(self) -> None:
        self.fields: list[_Field] = []
        self.current = _Field()

    def add_quoted(self, text: str) -> None:
        """Add quoted text: never split or globbed, and it keeps the field alive."""
        self.current.add(text, False)
        self.current.keep_empty = True

    def add_unquoted_literal(self, text: str) -> None:
        """Add typed, unquoted text (already split by the lexer, may glob)."""
        self.current.add(text, True)

    def add_split(self, value: str) -> None:
        """Add an unquoted expansion result, splitting it on blanks."""
        if not value:
            return
        if value[0] in _BLANKS:
            self.break_field()
        pieces = value.split()
        for index, piece in enumerate(pieces):
            if index:
                self.break_field()
            self.current.add(piece, True)
        if value[-1] in _BLANKS:
            self.break_field()

    def break_field(self) -> None:
        """End the current field and start a new one."""
        if not self.current.is_empty:
            self.fields.append(self.current)
        self.current = _Field()

    def finish(self) -> list[_Field]:
        """Close the last field and return them all."""
        self.break_field()
        return self.fields


def expand_words(words: list[Word], ctx: ExpansionContext) -> list[str]:
    """Fully expand a command's words into its argv."""
    argv: list[str] = []
    for word in words:
        argv.extend(expand_word(word, ctx))
    return argv


def expand_word(word: Word, ctx: ExpansionContext) -> list[str]:
    """Expand one word into zero or more fields (splitting and globbing included)."""
    builder = _FieldBuilder()
    for index, part in enumerate(word.parts):
        _expand_part(part, index == 0, builder, ctx)
    result: list[str] = []
    for fld in builder.finish():
        result.extend(_glob_field(fld, ctx))
    return result


def expand_single(word: Word, ctx: ExpansionContext) -> str:
    """Expand without splitting or globbing (assignment values, ``case``-like uses)."""
    text: list[str] = []
    for index, part in enumerate(word.parts):
        if isinstance(part, Lit):
            at_start = index == 0 and not part.quoted
            text.append(_expand_tilde(part.text, ctx) if at_start else part.text)
        elif isinstance(part, Param):
            text.append(_param_value(part.name, ctx))
        elif isinstance(part, Arith):
            text.append(_arithmetic(part, ctx))
        else:
            text.append(_substitution(part, ctx))
    return "".join(text)


def _expand_part(part: Lit | Param | CommandSub | Arith, first: bool,
                 builder: _FieldBuilder, ctx: ExpansionContext) -> None:
    if isinstance(part, Lit):
        if part.quoted:
            builder.add_quoted(part.text)
        elif first and _tilde_applies(part.text):
            builder.add_quoted(_expand_tilde(part.text, ctx))   # home path is never globbed
        else:
            builder.add_unquoted_literal(part.text)
    elif isinstance(part, Param) and part.name == "@" and part.quoted:
        _expand_quoted_at(builder, ctx)
    elif isinstance(part, Param):
        _add_value(_param_value(part.name, ctx), part.quoted, builder)
    elif isinstance(part, Arith):
        _add_value(_arithmetic(part, ctx), part.quoted, builder)
    else:
        _add_value(_substitution(part, ctx), part.quoted, builder)


def _add_value(value: str, quoted: bool, builder: _FieldBuilder) -> None:
    if quoted:
        builder.add_quoted(value)
    else:
        builder.add_split(value)


def _expand_quoted_at(builder: _FieldBuilder, ctx: ExpansionContext) -> None:
    """``"$@"``: every positional argument becomes its own field, unsplit."""
    args = ctx.positional_args()
    for index, arg in enumerate(args):
        if index:
            builder.break_field()
        builder.add_quoted(arg)


def _tilde_applies(text: str) -> bool:
    return text == "~" or text.startswith("~/")


def _expand_tilde(text: str, ctx: ExpansionContext) -> str:
    if not _tilde_applies(text):
        return text
    return (ctx.lookup("HOME") or HOME) + text[1:]


def _param_value(name: str, ctx: ExpansionContext) -> str:
    if name in ("@", "*"):
        return " ".join(ctx.positional_args())
    if not _valid_parameter(name):
        raise ExpansionError(f"${{{name}}}: bad substitution")
    value = ctx.lookup(name)
    return "" if value is None else value


def _valid_parameter(name: str) -> bool:
    if name in ("?", "#", "$") or name.isdigit():
        return True
    return bool(name) and (name[0].isalpha() or name[0] == "_") and \
        all(ch.isalnum() or ch == "_" for ch in name)


def _arithmetic(part: Arith, ctx: ExpansionContext) -> str:
    """``$((i + 1))``: the value as decimal text (shell/arith.py)."""
    try:
        return str(evaluate(part.expression, ctx.lookup))
    except ArithError as error:
        raise ExpansionError(error.message) from None


def _substitution(part: CommandSub, ctx: ExpansionContext) -> str:
    # Like bash, trailing newlines of the captured output are removed.
    return ctx.substitute(part.source).rstrip("\n")


def _escape_glob(text: str) -> str:
    """Make quoted text match itself literally inside a glob pattern."""
    return "".join(f"[{ch}]" if ch in "*?[" else ch for ch in text)


def _glob_field(fld: _Field, ctx: ExpansionContext) -> list[str]:
    has_wildcard = any(globbable and any(ch in text for ch in "*?[")
                       for text, globbable in fld.chunks)
    if not has_wildcard:
        return [fld.text]
    pattern = "".join(text if globbable else _escape_glob(text) for text, globbable in fld.chunks)
    matches = ctx.vfs.glob(pattern, ctx.cwd)
    return matches or [fld.text]


@dataclass
class SimpleContext:
    """A ready-made ``ExpansionContext`` over plain values (tests and tools)."""

    vfs: VirtualFS
    cwd: str = HOME
    variables: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    runner: Callable[[str], str] | None = None

    def lookup(self, name: str) -> str | None:
        """Look up ``$#`` and positional arguments from ``args``, the rest in ``variables``."""
        if name == "#":
            return str(len(self.args))
        if name.isdigit() and name != "0":
            index = int(name) - 1
            return self.args[index] if index < len(self.args) else None
        return self.variables.get(name)

    def positional_args(self) -> list[str]:
        """The positional arguments."""
        return list(self.args)

    def substitute(self, source: str) -> str:
        """Delegate ``$(...)`` to ``runner`` (empty output without one)."""
        return self.runner(source) if self.runner else ""
