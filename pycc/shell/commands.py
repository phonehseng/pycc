"""The command registry and the toolkit every command is written with.

A command is a plain function ``fn(ctx, args) -> int`` (the exit status) or a
generator function that yields shell events and returns the status (only commands
that run other programs need that). Commands never print directly: they write to
``ctx.stdout``/``ctx.stderr``, which the executor wires to the terminal, a pipe or a
file. That is what makes ``ls > files.txt`` and ``ls | wc -l`` work for free.

Other modules plug in by calling ``register_command`` (the toolchain replaces the
``cc``/``gcc``/``make`` placeholders this way).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generator

from pycc.core.concepts import Concept
from pycc.core.diagnostics import CODE_CONCEPTS, Diagnostic, Phase, Severity, concept_for_code
from pycc.core.events import InputSource, OutputSink, ShellEvent
from pycc.shell import manpages
from pycc.shell.ansi import friendly_line
from pycc.vfs.errors import VfsError
from pycc.vfs.filesystem import VirtualFS

if TYPE_CHECKING:          # only for type hints; the executor imports this module
    from pycc.shell.executor import Shell

CommandResult = int | Generator[ShellEvent, Any, int]
CommandFn = Callable[["CommandContext", list[str]], CommandResult]


class CommandError(Exception):
    """Raised inside a command to stop it with a diagnostic.

    The executor catches it, prints ``message`` and the friendly line, records the
    diagnostic and uses ``status`` as the exit code. This keeps commands short: no
    error plumbing in every branch.
    """

    def __init__(self, code: str, message: str, friendly: str, status: int = 1,
                 concept: Concept | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.friendly = friendly
        self.status = status
        self.concept = concept


class ExitRequest(Exception):
    """Raised by ``exit N``; caught where a shell, script or sub-shell ends."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class LoopControl(Exception):
    """Raised by ``break [N]`` / ``continue [N]``; caught by the enclosing loops.

    ``levels`` counts how many loops it still has to leave: ``break 2`` leaves the
    inner loop and then ends the outer one.
    """

    def __init__(self, kind: str, levels: int = 1) -> None:
        super().__init__(kind)
        self.kind = kind
        self.levels = levels


@dataclass
class IO:
    """The three streams a command runs with."""

    stdin: InputSource
    stdout: OutputSink
    stderr: OutputSink


@dataclass
class CommandSpec:
    """A registered command with its man page (markup) and one-line summary."""

    name: str
    fn: CommandFn
    manpage: str
    summary: str


_REGISTRY: dict[str, CommandSpec] = {}


def register_command(name: str, fn: CommandFn, manpage: str | None = None,
                     summary: str | None = None) -> None:
    """Add or replace a command. Missing texts fall back to ``shell.manpages``."""
    _REGISTRY[name] = CommandSpec(
        name, fn,
        manpage if manpage is not None else manpages.MANPAGES.get(name, f"# {name}\n"),
        summary if summary is not None else manpages.SUMMARIES.get(name, ""),
    )


def lookup(name: str) -> CommandSpec | None:
    """The command called ``name``, or None."""
    return _REGISTRY.get(name)


def command_names() -> list[str]:
    """Every registered command name, sorted."""
    return sorted(_REGISTRY)


# --- friendly texts for filesystem errors --------------------------------------------

_VFS_FRIENDLY: dict[str, str] = {
    "no_such_file": "There is nothing called '{path}' here. Run `ls` to see the names "
                    "in this folder, and check the spelling.",
    "is_a_directory": "'{path}' is a folder, not a file. Use `ls {path}` to look inside it.",
    "not_a_directory": "Part of '{path}' is a file, but it is used like a folder.",
    "file_exists": "Something called '{path}' already exists. Pick another name or "
                   "remove the old one first.",
    "permission_denied": "'{path}' is protected: system files and folders are read-only.",
    "directory_not_empty": "The folder '{path}' still has files in it. Use `rm -r` to "
                           "remove it together with everything inside.",
    "invalid_operation": "That operation is not possible on '{path}'.",
    "same_file": "A file cannot be copied onto itself. Give the copy a new name, like "
                 "`cp main.c main_backup.c`.",
}


def vfs_friendly(error: VfsError) -> str:
    """The beginner explanation for a filesystem error."""
    template = _VFS_FRIENDLY.get(error.code, "The file operation on '{path}' failed.")
    return template.format(path=error.path)


def vfs_concept(error: VfsError) -> Concept:
    """The concept a filesystem error teaches (codes core does not list are file work)."""
    return CODE_CONCEPTS.get(error.code, Concept.SHELL_FILES)


def is_terminal(stream: object) -> bool:
    """True when a stream is the user's terminal rather than a pipe or file."""
    return bool(getattr(stream, "is_terminal", False))


def report_error(diagnostics: list[Diagnostic], stderr: OutputSink, code: str, message: str,
                 friendly: str, concept: Concept | None = None) -> None:
    """Print a bash-style error plus its dimmed friendly line and record it.

    The friendly line only goes to a terminal: with ``2> errors.txt`` the file gets
    exactly what bash would write, while the diagnostic is still recorded.
    """
    stderr.write(message + "\n")
    if friendly and is_terminal(stderr):
        stderr.write(friendly_line(friendly))
    diagnostics.append(Diagnostic(
        severity=Severity.ERROR, code=code, message=message, friendly=friendly,
        location=None, concept=concept or concept_for_code(code, Concept.SHELL_NAVIGATION),
        phase=Phase.SHELL,
    ))


# --- the context handed to every command -----------------------------------------------

@dataclass
class CommandContext:
    """Everything a running command may touch: the shell and its three streams."""

    shell: "Shell"
    name: str
    stdin: InputSource
    stdout: OutputSink
    stderr: OutputSink

    @property
    def vfs(self) -> VirtualFS:
        """The filesystem."""
        return self.shell.vfs

    @property
    def io(self) -> IO:
        """The command's streams bundled, for running nested scripts."""
        return IO(self.stdin, self.stdout, self.stderr)

    @property
    def cwd(self) -> str:
        """The shell's working directory."""
        return self.shell.cwd

    def resolve(self, path: str) -> str:
        """Absolute path of ``path`` relative to the working directory."""
        return self.vfs.resolve(path, self.cwd)

    def out(self, text: str) -> None:
        """Write to standard output."""
        self.stdout.write(text)

    def err(self, text: str) -> None:
        """Write to standard error (no friendly line; use ``fail`` for errors)."""
        self.stderr.write(text)

    @property
    def stdout_is_terminal(self) -> bool:
        """True when output goes to the terminal (so colours make sense)."""
        return is_terminal(self.stdout)

    def fail(self, code: str, message: str, friendly: str, status: int = 1,
             concept: Concept | None = None) -> int:
        """Report an error and return ``status`` so a command can ``return ctx.fail(...)``."""
        report_error(self.shell.last_diagnostics, self.stderr, code, message, friendly, concept)
        return status

    def warn_vfs(self, error: VfsError, message: str | None = None) -> None:
        """Report a filesystem error without stopping (commands that loop over files)."""
        self.fail(error.code, message or f"{self.name}: {error}", vfs_friendly(error),
                  concept=vfs_concept(error))

    def read_stdin(self) -> str:
        """Everything left on standard input."""
        lines: list[str] = []
        while (line := self.stdin.readline()) is not None:
            lines.append(line)
        return "".join(lines)

    def read_inputs(self, operands: list[str]) -> list[tuple[str, str | None]]:
        """``(name, text)`` per operand (``-`` or none means stdin); text is None
        when the file could not be read (the error is already reported)."""
        if not operands:
            return [("-", self.read_stdin())]
        result: list[tuple[str, str | None]] = []
        for operand in operands:
            if operand == "-":
                result.append(("-", self.read_stdin()))
                continue
            try:
                result.append((operand, self.vfs.read_text(self.resolve(operand))))
            except VfsError as error:
                error.path = operand
                self.warn_vfs(error, f"{self.name}: {operand}: {error.reason}")
                result.append((operand, None))
        return result

    def emit(self, event: ShellEvent) -> None:
        """Send a non-text event (``ClearScreen``) to the terminal."""
        self.shell.emit(event)


# --- option parsing ---------------------------------------------------------------------

def parse_options(name: str, args: list[str], flags: str, valued: str = "",
                  numeric_shortcut: str = "") -> tuple[dict[str, str | bool], list[str]]:
    """Parse short options the POSIX way.

    ``flags`` lists the letters that are switches (``-l``), ``valued`` the letters that
    take a value (``-n 5`` or ``-n5``). Letters combine (``-la``); ``--`` ends options;
    a lone ``-`` is an operand. With ``numeric_shortcut`` set to a letter, ``-5`` means
    ``-<letter> 5`` (as in ``head -5``). Unknown letters raise ``CommandError``.
    """
    options: dict[str, str | bool] = {}
    operands: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            operands.extend(args[index:])
            break
        if not arg.startswith("-") or arg == "-":
            operands.append(arg)
            continue
        if numeric_shortcut and arg[1:].isdigit():
            options[numeric_shortcut] = arg[1:]
            continue
        index = _parse_cluster(name, arg, args, index, flags, valued, options)
    return options, operands


def _parse_cluster(name: str, arg: str, args: list[str], index: int, flags: str, valued: str,
                   options: dict[str, str | bool]) -> int:
    """Parse one ``-abc`` cluster; returns the index of the next unread argument."""
    if arg.startswith("--"):
        raise bad_option(name, arg, flags + valued, long_option=True)
    for position, letter in enumerate(arg[1:], start=1):
        if letter in valued:
            value = arg[position + 1:]
            if not value:
                if index >= len(args):
                    raise CommandError("missing_operand",
                                       f"{name}: option requires an argument -- '{letter}'",
                                       f"`-{letter}` needs a value after it, like `{name} "
                                       f"-{letter} 5`.", 2)
                value = args[index]
                index += 1
            options[letter] = value
            return index
        if letter not in flags:
            raise bad_option(name, letter, flags + valued)
        options[letter] = True
    return index


def bad_option(name: str, option: str, known: str, long_option: bool = False) -> CommandError:
    """The error for an option a command does not understand."""
    shown = f"unrecognized option '{option}'" if long_option else f"invalid option -- '{option}'"
    listing = ", ".join(f"-{letter}" for letter in known) or "no options"
    return CommandError("bad_option", f"{name}: {shown}",
                        f"{name} understands {listing}. Type `man {name}` for details.", 2)


def missing_operand(name: str, example: str) -> CommandError:
    """The error for a command run without the file names it needs."""
    return CommandError("missing_operand", f"{name}: missing operand",
                        f"{name} needs something to work on, for example `{example}`.", 1)
