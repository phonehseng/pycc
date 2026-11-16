"""``make``: a small, honest GNU make over the virtual filesystem.

A Makefile says which files are built from which (``util.o: util.c util.h``) and
how (the Tab-indented *recipe* under the rule). ``make`` rebuilds a target only when
it is missing or one of its prerequisites is newer, so after editing one file only the
files that depend on it are compiled again. Users learn exactly that here, with the
same messages GNU make prints, plus a plain-English line under each error.

Supported: variables (``=``, ``:=``, ``+=``, ``?=``, ``$(NAME)``, ``${NAME}``,
``$(SRCS:.c=.o)`` and a few functions such as ``$(wildcard *.c)``), the automatic
variables ``$@ $< $^ $+ $? $* $|``, comments, line continuation, ``.PHONY``, pattern
rules (``%.o: %.c``), order-only prerequisites (``obj/x.o: x.c | obj``), the built-in
rules for ``%.o`` from ``%.c`` and a program from its ``.c`` file, ``-n``, ``-B``,
``-s``, ``-f FILE``, ``-C DIR`` and ``NAME=value`` on the command line. Conditionals
(``ifeq``) and ``include`` are not supported and say so.

Every recipe line runs through the shell's own executor in a child shell, as in real
make (so ``cd`` in one line does not leak), which is why pipes, redirections and
``./bot`` work and ``RunProgram`` events reach the host.
"""
from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, concept_for_code
from pycc.core.events import ShellEvent
from pycc.shell import constructs
from pycc.shell.cc import indented_friendly
from pycc.shell.commands import CommandContext, is_terminal
from pycc.vfs.errors import VfsError
from pycc.vfs.filesystem import VirtualFS

MakeGen = Generator[ShellEvent, Any, Any]

MAKEFILE_NAMES = ("GNUmakefile", "makefile", "Makefile")
ERROR_STATUS = 2                       # GNU make exits with 2 when something failed
AUTOMATIC = "@<^+?*|"
BUILTIN_ORIGIN = "<builtin>"

# GNU make predefines CC, RM and MAKE but not CFLAGS: "CFLAGS ?= -Wall" must take effect,
# while "CC ?= gcc" keeps cc (a classic make surprise, kept on purpose).
DEFAULT_VARIABLES = {"CC": "cc", "RM": "rm -f", "MAKE": "make"}

# Words that start GNU make directives this make does not implement. Without this list
# they would fall through to "missing separator", which sends a student hunting for a
# Tab problem that does not exist.
UNSUPPORTED_DIRECTIVES = frozenset({"include", "-include", "sinclude", "ifeq", "ifneq",
                                    "ifdef", "ifndef", "else", "endif", "define", "endef"})
# "export NAME" / "unexport NAME" without a value only change what recipes inherit;
# every variable already reaches the recipes here, so the lines are accepted and ignored.
_EXPORT_ONLY = re.compile(r"^\s*(?:export|unexport)(?:\s+[A-Za-z0-9_.\-]+)*\s*$")
ORDER_ONLY_MARK = "|"

_ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+|override\s+)?([A-Za-z0-9_.\-]+)\s*(::=|:=|\+=|\?=|=)(.*)$")
_RECIPE_PREFIXES = "@-+"


# =========================================================================== errors

class MakeError(Exception):
    """A fatal make error: printed as ``make: *** <message>.  Stop.`` (or with the
    makefile position instead of ``make:``)."""

    def __init__(self, code: str, message: str, friendly: str,
                 where: str = "make") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.friendly = friendly
        self.where = where

    @property
    def text(self) -> str:
        """The exact line GNU make prints."""
        return f"{self.where}: *** {self.message}.  Stop."


class _RecipeFailed(Exception):
    """A recipe line failed; its ``Error N`` line is already printed."""


# =========================================================================== the model

@dataclass
class Variable:
    """A make variable. ``recursive`` ones (``=``) are expanded every time they are
    used; simple ones (``:=``) were expanded once, when they were assigned."""

    value: str
    recursive: bool = True


@dataclass
class RecipeLine:
    """One line of a recipe and where it was written (0 = a built-in rule)."""

    text: str
    line: int = 0


@dataclass
class Rule:
    """``targets: prerequisites`` plus the recipe written under it."""

    targets: list[str]
    prerequisites: list[str]
    recipe: list[RecipeLine] = field(default_factory=list)
    line: int = 0


@dataclass
class PatternRule:
    """A rule with ``%`` in it, like ``%.o: %.c``: one recipe for many files."""

    target: str
    prerequisites: list[str]
    recipe: list[RecipeLine] = field(default_factory=list)
    line: int = 0
    order_only: list[str] = field(default_factory=list)     # after "|"

    def match(self, name: str) -> str | None:
        """The *stem* ``%`` stands for in ``name`` (``util`` for ``util.o``), or None."""
        prefix, _, suffix = self.target.partition("%")
        long_enough = len(name) >= len(prefix) + len(suffix)
        if long_enough and name.startswith(prefix) and name.endswith(suffix):
            stem = name[len(prefix):len(name) - len(suffix)]
            return stem or None
        return None


@dataclass
class TargetEntry:
    """Everything the makefile says about one target, merged over all its rules."""

    name: str
    prerequisites: list[str] = field(default_factory=list)
    recipe_rule: Rule | None = None
    # Order-only prerequisites (after "|") must exist first but never make the target
    # out of date and are not part of $^ or $<: the classic use is an output folder,
    # whose time changes every time a file is created inside it.
    order_only: list[str] = field(default_factory=list)


@dataclass
class Makefile:
    """A parsed makefile."""

    name: str
    variables: dict[str, Variable] = field(default_factory=dict)
    targets: dict[str, TargetEntry] = field(default_factory=dict)
    patterns: list[PatternRule] = field(default_factory=list)
    phony: set[str] = field(default_factory=set)
    default_goal: str | None = None
    warnings: list[str] = field(default_factory=list)


def builtin_patterns() -> list[PatternRule]:
    """GNU make's built-in rules that matter for C: objects and one-file programs."""
    return [
        PatternRule("%.o", ["%.c"], [RecipeLine("$(CC) $(CFLAGS) -c $< -o $@")]),
        PatternRule("%", ["%.c"], [RecipeLine("$(CC) $(CFLAGS) $^ -o $@")]),
    ]


# =========================================================================== expansion

class Expander:
    """Expands ``$(...)`` references against a variable table.

    ``files`` lets ``$(wildcard ...)`` look at the virtual filesystem; it is a function
    from a glob pattern to the matching names (relative to make's folder).
    """

    def __init__(self, variables: dict[str, Variable],
                 files: Callable[[str], list[str]] | None = None) -> None:
        """Expand against ``variables`` (shared, so later assignments are seen)."""
        self.variables = variables
        self.files = files or (lambda pattern: [])
        self._active: set[str] = set()

    def expand(self, text: str, auto: dict[str, str] | None = None) -> str:
        """Replace every ``$`` reference in ``text``; ``$$`` becomes ``$``."""
        out: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char != "$":
                out.append(char)
                index += 1
                continue
            following = text[index + 1:index + 2]
            if following == "$":
                out.append("$")
                index += 2
            elif following in ("(", "{"):
                end = _matching_close(text, index + 1)
                out.append(self._reference(text[index + 2:end], auto))
                index = end + 1
            else:
                # "$@" or "$X": a one-character name; a lone "$" at the end is dropped.
                out.append(self.lookup(following, auto) if following else "")
                index += 2
        return "".join(out)

    def lookup(self, name: str, auto: dict[str, str] | None = None) -> str:
        """The value of one variable (automatic variables first)."""
        if auto is not None:
            if name in auto:
                return auto[name]
            if len(name) == 2 and name[0] in auto and name[1] in "DF":
                return _dir_or_file(auto[name[0]], name[1])
        variable = self.variables.get(name)
        if variable is None:
            return ""
        if not variable.recursive:
            return variable.value
        if name in self._active:
            raise MakeError(
                "make_circular", f"Recursive variable '{name}' references itself (eventually)",
                f"'{name}' is defined using itself, so expanding it would never end. To add "
                f"to a variable write `{name} += more` instead of `{name} = $({name}) more`.")
        self._active.add(name)
        try:
            return self.expand(variable.value, auto)
        finally:
            self._active.discard(name)

    def _reference(self, body: str, auto: dict[str, str] | None) -> str:
        """``$(body)``: a function call, a substitution reference or a variable."""
        function, _, rest = body.partition(" ")
        if function in _FUNCTIONS and rest:
            arguments = [self.expand(arg, auto) for arg in _split_arguments(rest.lstrip())]
            return _FUNCTIONS[function](self, arguments)
        name, colon, replacement = _split_top_level(body, ":")
        if colon and "=" in replacement:
            old, _, new = replacement.partition("=")
            value = self.lookup(self.expand(name, auto), auto)
            return _substitute_words(value, self.expand(old, auto), self.expand(new, auto))
        return self.lookup(self.expand(body, auto), auto)


def _matching_close(text: str, open_index: int) -> int:
    """Index of the bracket closing the one at ``open_index`` (same kind, nested)."""
    opener = text[open_index]
    closer = ")" if opener == "(" else "}"
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == opener:
            depth += 1
        elif text[index] == closer:
            depth -= 1
            if depth == 0:
                return index
    raise MakeError("make_missing_separator", "unterminated variable reference",
                    f"A '$({opener}' was opened but never closed with '{closer}'.")


def _split_top_level(text: str, separator: str) -> tuple[str, str, str]:
    """``str.partition`` that ignores separators inside ``$(...)``."""
    depth = 0
    for index, char in enumerate(text):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        elif char == separator and depth == 0:
            return text[:index], separator, text[index + 1:]
    return text, "", ""


def _split_arguments(text: str) -> list[str]:
    """Function arguments: split on commas that are not inside ``$(...)``."""
    parts: list[str] = []
    rest = text
    while True:
        head, comma, rest = _split_top_level(rest, ",")
        parts.append(head)
        if not comma:
            return parts


def _pattern_replace(word: str, pattern: str, replacement: str) -> str:
    """``patsubst`` for one word: ``%.c`` -> ``%.o`` keeps the stem."""
    if "%" not in pattern:
        return replacement if word == pattern else word
    stem = PatternRule(pattern, []).match(word)
    if stem is None:
        return word
    return replacement.replace("%", stem, 1) if "%" in replacement else replacement


def _substitute_words(value: str, old: str, new: str) -> str:
    """``$(SRCS:.c=.o)``: change the ending of every word."""
    if "%" in old:
        return " ".join(_pattern_replace(word, old, new) for word in value.split())
    return " ".join(word[:len(word) - len(old)] + new if old and word.endswith(old) else word
                    for word in value.split())


def _dir_or_file(path: str, part: str) -> str:
    """``$(@D)`` and ``$(@F)``: the folder or the file name of a path."""
    folder, name = posixpath.split(path)
    return (folder or ".") if part == "D" else name


def _fn_wildcard(expander: Expander, args: list[str]) -> str:
    return " ".join(name for pattern in args[0].split() for name in expander.files(pattern))


def _fn_patsubst(expander: Expander, args: list[str]) -> str:
    pattern, replacement, text = (args + ["", ""])[:3]
    return " ".join(_pattern_replace(word, pattern.strip(), replacement.strip())
                    for word in text.split())


def _fn_subst(expander: Expander, args: list[str]) -> str:
    old, new, text = (args + ["", ""])[:3]
    return text.replace(old, new) if old else text


def _fn_notdir(expander: Expander, args: list[str]) -> str:
    return " ".join(posixpath.basename(word) for word in args[0].split())


def _fn_addprefix(expander: Expander, args: list[str]) -> str:
    prefix, text = (args + [""])[:2]
    return " ".join(prefix + word for word in text.split())


def _matches_any(word: str, patterns: list[str]) -> bool:
    """True when ``word`` matches one of the (``%``) patterns of ``filter``."""
    return any(_pattern_replace(word, pattern, "\0") == "\0" for pattern in patterns)


def _fn_filter(expander: Expander, args: list[str]) -> str:
    patterns, text = (args + [""])[:2]
    keep = patterns.split()
    return " ".join(word for word in text.split() if _matches_any(word, keep))


def _fn_filter_out(expander: Expander, args: list[str]) -> str:
    patterns, text = (args + [""])[:2]
    drop = patterns.split()
    return " ".join(word for word in text.split() if not _matches_any(word, drop))


def _fn_addsuffix(expander: Expander, args: list[str]) -> str:
    suffix, text = (args + [""])[:2]
    return " ".join(word + suffix for word in text.split())


def _fn_dir(expander: Expander, args: list[str]) -> str:
    # GNU keeps the trailing slash ("src/main.c" -> "src/") and answers "./" for none.
    return " ".join((posixpath.dirname(word) or ".") + "/" for word in args[0].split())


def _fn_basename(expander: Expander, args: list[str]) -> str:
    return " ".join(posixpath.splitext(word)[0] for word in args[0].split())


def _fn_strip(expander: Expander, args: list[str]) -> str:
    return " ".join(",".join(args).split())


def _fn_sort(expander: Expander, args: list[str]) -> str:
    return " ".join(sorted(set(args[0].split())))


_FUNCTIONS: dict[str, Callable[[Expander, list[str]], str]] = {
    "wildcard": _fn_wildcard, "patsubst": _fn_patsubst, "subst": _fn_subst,
    "notdir": _fn_notdir, "addprefix": _fn_addprefix, "filter-out": _fn_filter_out,
    "sort": _fn_sort, "filter": _fn_filter, "addsuffix": _fn_addsuffix, "dir": _fn_dir,
    "basename": _fn_basename, "strip": _fn_strip,
}


# =========================================================================== parsing

def parse_makefile(text: str, name: str = "Makefile",
                   variables: dict[str, Variable] | None = None,
                   overrides: set[str] | None = None,
                   files: Callable[[str], list[str]] | None = None) -> Makefile:
    """Read a makefile; raises ``MakeError`` (``missing separator`` and friends).

    ``variables`` are the starting values (defaults, environment, command line);
    names in ``overrides`` came from the command line and win over the file.
    """
    return _Parser(text, name, variables or {}, overrides or set(), files).parse()


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash-continued lines; each result keeps its first line's number."""
    result: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        continued = _ends_with_continuation(raw)
        body = raw[:-1].rstrip() if continued else raw
        if pending:
            body = body.lstrip()        # the indentation of a continuation is not text
        else:
            start = number              # keep the Tab of a recipe's first line
        pending.append(body)
        if not continued:
            result.append((start, " ".join(pending)))
            pending = []
    if pending:
        result.append((start, " ".join(pending)))
    return result


def _ends_with_continuation(line: str) -> bool:
    """An odd number of trailing backslashes continues the line (``\\\\`` does not)."""
    count = len(line) - len(line.rstrip("\\"))
    return count % 2 == 1


def _strip_comment(line: str) -> str:
    """Drop a ``#`` comment; ``\\#`` is a literal ``#``."""
    out: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and line[index + 1:index + 2] == "#":
            out.append("#")
            index += 2
            continue
        if char == "#":
            break
        out.append(char)
        index += 1
    return "".join(out)


class _Parser:
    """Turns makefile text into a ``Makefile`` (one pass, like GNU make)."""

    def __init__(self, text: str, name: str, variables: dict[str, Variable],
                 overrides: set[str], files: Callable[[str], list[str]] | None) -> None:
        self.text = text
        self.makefile = Makefile(name, dict(variables))
        self.overrides = overrides
        self.expander = Expander(self.makefile.variables, files)
        self.rules: list[Rule] = []
        self.current: list[RecipeLine] | None = None    # recipe of the rule being read

    def parse(self) -> Makefile:
        """Read every line, then attach each recipe to its targets."""
        for number, line in logical_lines(self.text):
            self._line(number, line)
        self._attach_recipes()
        return self.makefile

    def _error(self, number: int, message: str, friendly: str,
               code: str = "make_missing_separator") -> MakeError:
        return MakeError(code, message, friendly, where=f"{self.makefile.name}:{number}")

    def _line(self, number: int, line: str) -> None:
        if line.startswith("\t"):
            self._recipe_line(number, line[1:])
            return
        content = _strip_comment(line)
        if not content.strip():
            return                          # blank lines do not end a recipe
        assignment = _ASSIGNMENT.match(content)
        if assignment is not None:
            self.current = None
            self._assign(*assignment.groups())
            return
        if _EXPORT_ONLY.match(content):
            return
        self._refuse_directive(number, content)
        targets, colon, rest = _split_top_level(content, ":")
        if not colon or not targets.strip():
            raise self._error(number, "missing separator", _separator_friendly(number, line))
        self._rule(number, targets, rest.lstrip(":"))

    def _refuse_directive(self, number: int, content: str) -> None:
        """Stop with a clear message on ``include``/``ifeq``/``define`` lines."""
        word = content.split(None, 1)[0]
        if word not in UNSUPPORTED_DIRECTIVES:
            return
        raise self._error(
            number, f"'{word}' is not supported by this make",
            f"GNU make understands '{word}', but the workshop's make does not. Write the "
            "rules and variables straight into this Makefile instead.",
            code="unsupported_feature")

    def _recipe_line(self, number: int, text: str) -> None:
        if self.current is None:
            if not _strip_comment(text).strip():
                return
            raise self._error(number, "recipe commences before first target",
                              "A line that starts with a Tab is a recipe, and a recipe "
                              "must come right under a 'target: prerequisites' line.")
        if text.strip():
            self.current.append(RecipeLine(text, number))

    def _assign(self, name: str, operator: str, value: str) -> None:
        """One variable assignment (``=``, ``:=``, ``::=``, ``+=``, ``?=``)."""
        if name in self.overrides:
            return                      # "make CC=gcc" beats "CC = cc" in the file
        value = value.strip()
        variables = self.makefile.variables
        existing = variables.get(name)
        if operator in (":=", "::="):
            variables[name] = Variable(self.expander.expand(value), recursive=False)
        elif operator == "?=":
            if existing is None:
                variables[name] = Variable(value)
        elif operator == "+=" and existing is not None:
            added = value if existing.recursive else self.expander.expand(value)
            joined = f"{existing.value} {added}" if existing.value else added
            variables[name] = Variable(joined, existing.recursive)
        else:
            variables[name] = Variable(value)

    def _rule(self, number: int, target_text: str, rest: str) -> None:
        """``targets: prerequisites [; recipe]``."""
        prerequisite_text, _, inline = rest.partition(";")
        targets = self.expander.expand(target_text).split()
        prerequisites, order_only = _split_order_only(
            self.expander.expand(prerequisite_text).split())
        inline = inline.strip()
        recipe = [RecipeLine(inline, number)] if inline else []
        self.current = recipe
        if targets == [".PHONY"]:
            self.makefile.phony.update(prerequisites)
            return
        if all(_is_special_target(t) for t in targets):
            return                          # other special targets (.SUFFIXES...) do nothing
        patterns = [t for t in targets if "%" in t]
        for target in patterns:
            self.makefile.patterns.append(
                PatternRule(target, prerequisites, recipe, number, order_only))
        plain = [t for t in targets if "%" not in t]
        if plain:
            self.rules.append(Rule(plain, prerequisites, recipe, number))
            for target in plain:
                entry = self.makefile.targets.setdefault(target, TargetEntry(target))
                _extend_unique(entry.prerequisites, prerequisites)
                _extend_unique(entry.order_only, order_only)
            if self.makefile.default_goal is None:
                self.makefile.default_goal = next(
                    (t for t in plain if not t.startswith(".") or "/" in t), None)

    def _attach_recipes(self) -> None:
        """Give each target the recipe of its rule (a later recipe wins, as in GNU make)."""
        for rule in self.rules:
            if not rule.recipe:
                continue
            for target in rule.targets:
                entry = self.makefile.targets[target]
                old = entry.recipe_rule
                if old is not None and old is not rule:
                    name = self.makefile.name
                    self.makefile.warnings += [
                        f"{name}:{rule.line}: warning: overriding recipe for target '{target}'",
                        f"{name}:{old.line}: warning: ignoring old recipe for target "
                        f"'{target}'"]
                entry.recipe_rule = rule


def _split_order_only(words: list[str]) -> tuple[list[str], list[str]]:
    """``a b | dir`` -> ``(["a", "b"], ["dir"])``: normal and order-only prerequisites."""
    if ORDER_ONLY_MARK not in words:
        return words, []
    cut = words.index(ORDER_ONLY_MARK)
    return words[:cut], [word for word in words[cut + 1:] if word != ORDER_ONLY_MARK]


def _extend_unique(names: list[str], more: list[str]) -> None:
    """Append the names of ``more`` that ``names`` does not hold yet (order kept)."""
    names.extend(name for name in dict.fromkeys(more) if name not in names)


def _is_special_target(name: str) -> bool:
    """``.SUFFIXES``, ``.SILENT``...: GNU make's special targets (all capitals)."""
    word = name[1:].replace("_", "")
    return name.startswith(".") and word.isalpha() and word.isupper()


def _separator_friendly(number: int, line: str) -> str:
    """Explain "missing separator", which almost always means spaces instead of a Tab."""
    if line[:1] == " ":
        return (f"Line {number} starts with spaces. Recipe lines (the commands under a "
                "rule) must start with one Tab character: delete the spaces and press the "
                "Tab key instead.")
    return (f"make could not understand line {number}. A rule looks like "
            "'target: prerequisites', a variable like 'NAME = value', and every recipe line "
            "starts with a Tab.")


# =========================================================================== options

@dataclass
class MakeOptions:
    """What the ``make`` command line asked for."""

    makefile: str | None = None
    dry_run: bool = False
    always: bool = False
    silent: bool = False
    goals: list[str] = field(default_factory=list)
    overrides: dict[str, str] = field(default_factory=dict)
    directory: str | None = None          # -C DIR: work as if started in DIR


_LONG_FLAGS = {"--dry-run": "dry_run", "--just-print": "dry_run", "--recon": "dry_run",
               "--always-make": "always", "--silent": "silent", "--quiet": "silent"}
_SHORT_FLAGS = {"n": "dry_run", "B": "always", "s": "silent"}


def parse_make_args(args: list[str]) -> MakeOptions:
    """Parse ``make`` arguments; raises ``MakeError`` for an unknown option."""
    options = MakeOptions()
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg in _LONG_FLAGS:
            setattr(options, _LONG_FLAGS[arg], True)
        elif arg.startswith(("--file=", "--makefile=")):
            options.makefile = arg.split("=", 1)[1]
        elif arg.startswith("--directory="):
            options.directory = arg.split("=", 1)[1]
        elif arg.startswith("-") and len(arg) > 1 and not arg.startswith("--"):
            index = _short_options(arg, args, index, options)
        elif arg.startswith("--"):
            raise _bad_option(f"unrecognized option '{arg}'")
        elif "=" in arg and (assignment := _ASSIGNMENT.match(arg)) is not None:
            # The regex, not a split on "=", so that "CC:=gcc" names CC (not "CC:").
            name, _operator, value = assignment.groups()
            options.overrides[name] = value.strip()
        else:
            options.goals.append(arg)
    return options


_VALUED_FLAGS = {"f": "makefile", "C": "directory"}


def _short_options(arg: str, args: list[str], index: int, options: MakeOptions) -> int:
    """One ``-nB`` cluster; ``-f FILE``, ``-C DIR`` and ``-j N`` take a value."""
    for position, letter in enumerate(arg[1:], start=1):
        if letter in _SHORT_FLAGS:
            setattr(options, _SHORT_FLAGS[letter], True)
        elif letter in _VALUED_FLAGS:
            value = arg[position + 1:]
            if not value:
                if index >= len(args):
                    raise _bad_option(f"option requires an argument -- '{letter}'")
                value = args[index]
                index += 1
            setattr(options, _VALUED_FLAGS[letter], value)
            return index
        elif letter == "j":
            if not arg[position + 1:] and index < len(args) and args[index].isdigit():
                index += 1               # "-j 4": jobs are accepted and run one by one
            return index
        elif letter == "k":
            continue                     # -k is accepted; make still stops at an error
        else:
            raise _bad_option(f"invalid option -- '{letter}'")
    return index


def _bad_option(text: str) -> MakeError:
    """An option error, printed as ``make: <text>`` (no ``***``, like GNU make)."""
    return MakeError("bad_option", text,
                     "make understands -n (show commands only), -B (rebuild everything), "
                     "-s (silent), -f FILE and -C FOLDER. Type `man make` for details.",
                     where="")


# =========================================================================== running

@dataclass
class _Plan:
    """How one target is brought up to date."""

    prerequisites: list[str]
    recipe: list[RecipeLine]
    stem: str = ""
    builtin: bool = False
    order_only: list[str] = field(default_factory=list)


@dataclass
class _Outcome:
    """What updating a target did: ``changed`` means it was (or would be) remade."""

    changed: bool


class MakeRun:
    """One ``make`` invocation over a parsed makefile."""

    def __init__(self, ctx: CommandContext, makefile: Makefile, options: MakeOptions) -> None:
        """Prepare to bring the goals of ``options`` up to date in the shell's folder."""
        self.ctx = ctx
        self.makefile = makefile
        self.options = options
        self.expander = Expander(makefile.variables, _globber(ctx))
        self.patterns = makefile.patterns + builtin_patterns()
        self.done: dict[str, _Outcome] = {}
        self.recipes_run = 0

    @property
    def vfs(self) -> VirtualFS:
        """The filesystem make works on."""
        return self.ctx.vfs

    def mtime(self, name: str) -> int | None:
        """The logical modification time of a file, or None if it does not exist."""
        path = self.ctx.resolve(name)
        if not self.vfs.exists(path):
            return None
        return self.vfs.stat(path).mtime

    # --- goals ----------------------------------------------------------------------------
    def run(self) -> MakeGen:
        """Update every goal; returns the exit status (0, or 2 after an error)."""
        goals = self.options.goals or ([self.makefile.default_goal]
                                       if self.makefile.default_goal else [])
        if not goals:
            raise MakeError("make_no_rule", "No targets",
                            "This Makefile has no rules. Add one, such as "
                            "'prog: main.c', with its recipe on a Tab-indented line.")
        for goal in goals:
            before = self.recipes_run
            yield from self.update(goal, None, [])
            if self.recipes_run == before:
                self._nothing_done(goal)
        return 0

    def _nothing_done(self, goal: str) -> None:
        """GNU make's two "nothing happened" messages."""
        plan = self.find_plan(goal)
        if plan is not None and plan.recipe and goal not in self.makefile.phony:
            self.ctx.out(f"make: '{goal}' is up to date.\n")
            return
        report(self.ctx, "make_nothing_to_do", f"make: Nothing to be done for '{goal}'.",
               "Everything is already built and no file changed since the last make, so "
               "there is nothing to do. Edit or `touch` a file to make it rebuild.",
               Severity.NOTE, stream="stdout")

    # --- rules ----------------------------------------------------------------------------
    def find_plan(self, target: str) -> _Plan | None:
        """The rule that builds ``target``: its own recipe, a pattern rule, or none."""
        entry = self.makefile.targets.get(target)
        if entry is not None and entry.recipe_rule is not None:
            return _Plan(list(entry.prerequisites), entry.recipe_rule.recipe,
                         order_only=list(entry.order_only))
        if target not in self.makefile.phony:
            implicit = self._implicit_plan(target, entry)
            if implicit is not None:
                return implicit
        if entry is not None:
            return _Plan(list(entry.prerequisites), [], order_only=list(entry.order_only))
        return None

    def _implicit_plan(self, target: str, entry: TargetEntry | None) -> _Plan | None:
        """The first pattern rule whose prerequisites exist or can be made."""
        explicit = entry.prerequisites if entry is not None else []
        for index, pattern in enumerate(self.patterns):
            stem = pattern.match(target)
            if stem is None or not pattern.recipe:
                continue
            needed = [p.replace("%", stem) for p in pattern.prerequisites]
            if all(self._can_exist(p) for p in needed):
                merged = needed + [p for p in explicit if p not in needed]
                builtin = index >= len(self.makefile.patterns)
                order_only = [p.replace("%", stem) for p in pattern.order_only]
                _extend_unique(order_only, entry.order_only if entry is not None else [])
                return _Plan(merged, pattern.recipe, stem, builtin, order_only)
        return None

    def _can_exist(self, name: str) -> bool:
        """GNU's "ought to exist": the file is there or the makefile names it a target."""
        return name in self.makefile.targets or self.mtime(name) is not None

    # --- updating ---------------------------------------------------------------------------
    def update(self, target: str, parent: str | None, chain: list[str]) -> MakeGen:
        """Bring ``target`` up to date (prerequisites first); returns an ``_Outcome``."""
        if target in self.done:
            return self.done[target]
        plan = self.find_plan(target)
        if plan is None:
            if self.mtime(target) is not None:
                self.done[target] = _Outcome(False)
                return self.done[target]
            raise _no_rule(target, parent)
        changed_prerequisites = yield from self._update_prerequisites(target, plan, chain)
        outcome = yield from self._remake_if_needed(target, plan, changed_prerequisites)
        self.done[target] = outcome
        return outcome

    def _update_prerequisites(self, target: str, plan: _Plan, chain: list[str]) -> MakeGen:
        """Update each prerequisite; returns the set of the normal ones that were remade.

        Order-only prerequisites are brought up to date too (the output folder must
        exist before the compiler writes into it) but never count as changed.
        """
        changed: set[str] = set()
        inner = chain + [target]
        order_only = [p for p in plan.order_only if p not in plan.prerequisites]
        for prerequisite in dict.fromkeys(plan.prerequisites + order_only):
            if prerequisite in inner:
                report(self.ctx, "make_circular",
                       f"make: Circular {target} <- {prerequisite} dependency dropped.",
                       f"'{target}' depends on '{prerequisite}', which (through its own "
                       f"prerequisites) depends on '{target}' again. A file cannot be "
                       "needed to build itself: remove one of the two dependencies.",
                       Severity.WARNING)
                plan.prerequisites = [p for p in plan.prerequisites if p != prerequisite]
                continue
            outcome = yield from self.update(prerequisite, target, inner)
            if outcome.changed and prerequisite not in order_only:
                changed.add(prerequisite)
        return changed

    def _remake_if_needed(self, target: str, plan: _Plan, changed: set[str]) -> MakeGen:
        """Run the recipe when the target is missing, older than a prerequisite, phony,
        or ``-B`` was given; returns the ``_Outcome``."""
        own_time = self.mtime(target)
        newer = [p for p in dict.fromkeys(plan.prerequisites)
                 if own_time is None or p in changed or (self.mtime(p) or 0) > own_time]
        phony = target in self.makefile.phony
        if not (phony or self.options.always or own_time is None or newer):
            return _Outcome(False)
        if not plan.recipe:
            # Nothing to run: the target counts as new when something under it was.
            return _Outcome(bool(changed) or (own_time is None and not phony))
        yield from self._run_recipe(target, plan, newer)
        return _Outcome(True)

    def _run_recipe(self, target: str, plan: _Plan, newer: list[str]) -> MakeGen:
        """Echo and execute each recipe line; the first failure stops make."""
        prerequisites = plan.prerequisites
        auto = {"@": target, "<": prerequisites[0] if prerequisites else "",
                "^": " ".join(dict.fromkeys(prerequisites)), "+": " ".join(prerequisites),
                "?": " ".join(newer), "*": plan.stem,
                "|": " ".join(dict.fromkeys(plan.order_only))}
        self.recipes_run += 1
        for recipe_line in plan.recipe:
            command = self.expander.expand(recipe_line.text, auto)
            silent, ignore, command = _recipe_prefixes(command)
            if not command:
                continue
            if self.options.dry_run or not (silent or self.options.silent):
                self.ctx.out(command + "\n")
            if self.options.dry_run:
                continue
            status = yield from self._execute(command)
            if status != 0:
                self._recipe_failed(target, recipe_line, plan, status, ignore)
        if not self.options.dry_run:
            self.ctx.shell.count_construct(constructs.MAKE_REBUILD)

    def _execute(self, command: str) -> MakeGen:
        """Run one recipe line in a child shell, like make's ``/bin/sh -c``."""
        shell = self.ctx.shell
        child = shell.spawn(shell.positional, shell.script_name)
        return (yield from child.execute_subcommand(command, self.ctx.io))

    def _recipe_failed(self, target: str, recipe_line: RecipeLine, plan: _Plan, status: int,
                       ignore: bool) -> None:
        """``make: *** [Makefile:7: util.o] Error 1`` (or ``(ignored)`` with ``-``)."""
        origin = (BUILTIN_ORIGIN if plan.builtin or not recipe_line.line
                  else f"{self.makefile.name}:{recipe_line.line}")
        if ignore:
            self.ctx.err(f"make: [{origin}: {target}] Error {status} (ignored)\n")
            return
        report(self.ctx, "make_recipe_failed",
               f"make: *** [{origin}: {target}] Error {status}",
               f"The command for '{target}' failed (it ended with status {status}), so make "
               "stopped. Fix the first error printed above, then run make again.")
        raise _RecipeFailed()


def _recipe_prefixes(command: str) -> tuple[bool, bool, str]:
    """Strip ``@`` (do not echo), ``-`` (ignore errors) and ``+`` from a recipe line."""
    silent = ignore = False
    text = command.lstrip()
    while text and text[0] in _RECIPE_PREFIXES:
        silent |= text[0] == "@"
        ignore |= text[0] == "-"
        text = text[1:].lstrip()
    return silent, ignore, text


def _no_rule(target: str, parent: str | None) -> MakeError:
    """``No rule to make target 'x'`` (with ``needed by`` for a prerequisite)."""
    needed = f", needed by '{parent}'" if parent else ""
    if parent:
        friendly = (f"'{parent}' lists '{target}' as a prerequisite, but there is no file "
                    f"called '{target}' and no rule that makes it. Check the spelling in the "
                    "Makefile, or create the file.")
    else:
        friendly = (f"The Makefile has no rule called '{target}' and there is no file with "
                    "that name. Look at the Makefile for the targets you can build.")
    return MakeError("make_no_rule", f"No rule to make target '{target}'{needed}", friendly)


# =========================================================================== reporting

def report(ctx: CommandContext, code: str, message: str, friendly: str,
           severity: Severity = Severity.ERROR, stream: str = "stderr") -> None:
    """Print a make message (friendly line only on a terminal) and record it."""
    sink = ctx.stdout if stream == "stdout" else ctx.stderr
    sink.write(message + "\n")
    if friendly and is_terminal(sink):
        sink.write(indented_friendly(friendly))
    ctx.shell.last_diagnostics.append(Diagnostic(
        severity, code, message, friendly, None,
        concept_for_code(code, Concept.BUILD_MAKE), Phase.MAKE))


def _starting_variables(ctx: CommandContext, options: MakeOptions) -> dict[str, Variable]:
    """Defaults, then the shell's exported environment, then ``NAME=value`` arguments."""
    variables = {name: Variable(value) for name, value in DEFAULT_VARIABLES.items()}
    variables["CURDIR"] = Variable(ctx.cwd, recursive=False)
    for name, value in ctx.shell.exported_environment().items():
        variables[name] = Variable(value, recursive=False)
    for name, value in options.overrides.items():
        variables[name] = Variable(value)
    return variables


def _find_makefile(ctx: CommandContext, options: MakeOptions) -> tuple[str | None, str | None]:
    """``(name, text)`` of the makefile to read; ``(None, None)`` when there is none.

    Raises ``MakeError`` when ``-f FILE`` names a file that does not exist.
    """
    if options.makefile is not None:
        try:
            return options.makefile, ctx.vfs.read_text(ctx.resolve(options.makefile))
        except VfsError as error:
            ctx.err(f"make: {options.makefile}: {error.reason}\n")
            raise MakeError("make_no_makefile",
                            f"No rule to make target '{options.makefile}'",
                            f"There is no file called '{options.makefile}' to read the rules "
                            "from. Check the name after -f with `ls`.") from None
    for name in MAKEFILE_NAMES:
        if ctx.vfs.is_file(ctx.resolve(name)):
            return name, ctx.vfs.read_text(ctx.resolve(name))
    return None, None


def run_make(ctx: CommandContext, options: MakeOptions) -> MakeGen:
    """Read the makefile and update the goals; returns the exit status.

    With ``-C DIR`` everything happens in ``DIR``, announced like GNU make does; the
    shell's own folder is restored afterwards, also when a recipe fails.
    """
    if options.directory is None:
        return (yield from _make_here(ctx, options))
    folder = ctx.resolve(options.directory)
    if not ctx.vfs.is_dir(folder):
        reason = "Not a directory" if ctx.vfs.exists(folder) else "No such file or directory"
        raise MakeError("make_no_makefile", f"{options.directory}: {reason}",
                        f"-C names the folder make should work in, but '{options.directory}' "
                        "is not a folder here. Check the name with `ls`.")
    shell = ctx.shell
    saved = shell.cwd
    ctx.out(f"make: Entering directory '{folder}'\n")
    shell.cwd = folder
    try:
        return (yield from _make_here(ctx, options))
    finally:
        shell.cwd = saved
        ctx.out(f"make: Leaving directory '{folder}'\n")


def _make_here(ctx: CommandContext, options: MakeOptions) -> MakeGen:
    """``run_make`` in the current folder: find, parse and run the makefile."""
    name, text = _find_makefile(ctx, options)
    if name is None and not options.goals:
        raise MakeError("make_no_makefile", "No targets specified and no makefile found",
                        "make reads its rules from a file called Makefile in this folder, "
                        "and there is none here. Create one, or `cd` into the project "
                        "folder first.")
    variables = _starting_variables(ctx, options)
    if name is None or text is None:
        # No makefile but a goal ("make hello"): the built-in rules may still apply.
        makefile = Makefile("Makefile", variables)
    else:
        makefile = parse_makefile(text, name, variables, set(options.overrides),
                                  _globber(ctx))
        for warning in makefile.warnings:
            ctx.err(warning + "\n")
    return (yield from MakeRun(ctx, makefile, options).run())


def _globber(ctx: CommandContext) -> Callable[[str], list[str]]:
    """``$(wildcard ...)`` over the VFS (results keep the pattern's style, so
    ``*.c`` answers names relative to make's folder)."""
    def matches(pattern: str) -> list[str]:
        return ctx.vfs.glob(pattern, ctx.cwd)
    return matches


def cmd_make(ctx: CommandContext, args: list[str]) -> MakeGen:
    """The ``make`` command (registered by ``shell.toolchain_commands``)."""
    try:
        options = parse_make_args(args)
        return (yield from run_make(ctx, options))
    except _RecipeFailed:
        return ERROR_STATUS
    except MakeError as error:
        text = error.text if error.where else f"make: {error.message}"
        report(ctx, error.code, text, error.friendly)
        return ERROR_STATUS


__all__ = ["Expander", "MakeError", "MakeOptions", "MakeRun", "Makefile", "PatternRule",
           "RecipeLine", "Rule", "TargetEntry", "Variable", "builtin_patterns", "cmd_make",
           "logical_lines", "parse_make_args", "parse_makefile", "run_make"]
