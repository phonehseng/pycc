"""Events exchanged between the interpreter, the shell and their hosts.

The interpreter and the shell are generators. Instead of calling their host directly
they *yield* one of the event objects below and wait for the host to resume them. A
debugger or an animated front end can pause between events; the CLI, the test suite
and the headless program runner just loop as fast as they can. Every host follows the
same protocol:

    gen = interpreter.run(argv)
    event = next(gen)
    while True:
        reply = handle(event)          # None unless the event asks for a value
        try:
            event = gen.send(reply)
        except StopIteration as done:
            exit_code = done.value
            break
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union

from pycc.core.diagnostics import Diagnostic, SourceLocation


# --- interpreter events ------------------------------------------------------------

@dataclass
class Statement:
    """About to execute the statement at ``location``. ``depth`` is the call depth."""

    location: SourceLocation
    depth: int = 0


@dataclass
class CallEnter:
    name: str
    location: SourceLocation | None = None


@dataclass
class CallExit:
    name: str


@dataclass
class DeviceCall:
    """A device function call. The host performs it and sends back the result value."""

    name: str
    args: list = field(default_factory=list)


@dataclass
class Output:
    """Text written to ``stdout`` or ``stderr``."""

    stream: str
    text: str


@dataclass
class InputRequest:
    """The program is reading stdin. The host sends back a line (with its newline)
    or ``None`` for end of file."""

    prompt: str = ""


@dataclass
class RunWarning:
    """A non-fatal runtime finding (uninitialized read, signed overflow, leak)."""

    diagnostic: Diagnostic


@dataclass
class Trap:
    """A fatal runtime error."""

    diagnostic: Diagnostic


@dataclass
class Exited:
    code: int


Event = Union[Statement, CallEnter, CallExit, DeviceCall, Output, InputRequest,
              RunWarning, Trap, Exited]


# --- streams shared by the shell and the interpreter -------------------------------

class InputSource(Protocol):
    def readline(self) -> str | None:
        """Return the next line including its newline, or None at end of file."""


class OutputSink(Protocol):
    def write(self, text: str) -> None: ...


class StringSource:
    """An ``InputSource`` over a fixed string (pipes, ``< file`` redirection, tests)."""

    def __init__(self, text: str = "") -> None:
        self._lines = text.splitlines(keepends=True)
        self._index = 0

    def readline(self) -> str | None:
        if self._index >= len(self._lines):
            return None
        line = self._lines[self._index]
        self._index += 1
        return line


class StringSink:
    """An ``OutputSink`` that collects text (pipes, redirection, tests)."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> None:
        self.parts.append(text)

    @property
    def text(self) -> str:
        return "".join(self.parts)


class NullSink:
    def write(self, text: str) -> None:
        return None


# --- shell events --------------------------------------------------------------------

@dataclass
class RunProgram:
    """The shell wants an executable run. The host runs it with the given streams
    and sends back the exit code."""

    executable: bytes
    argv: list[str]
    stdin: InputSource
    stdout: OutputSink
    stderr: OutputSink
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)   # the shell's exported variables


@dataclass
class ShellOutput:
    stream: str
    text: str


@dataclass
class ClearScreen:
    pass


@dataclass
class CommandFinished:
    """Yielded once per top-level ``Shell.execute`` call, before it returns, so hosts
    can show what happened (an editor gutter, a grader)."""

    line: str
    exit_code: int
    diagnostics: list[Diagnostic] = field(default_factory=list)
    build: object | None = None                 # BuildResult of a cc/make in this command
    runs: list["RunSummary"] = field(default_factory=list)
    constructs: dict[str, int] = field(default_factory=dict)


ShellEvent = Union[RunProgram, ShellOutput, ClearScreen, CommandFinished]


# --- summaries ------------------------------------------------------------------------

@dataclass
class RunSummary:
    """What happened during one program run."""

    exit_code: int | None
    trap: Diagnostic | None
    warnings: list[Diagnostic]
    statements: int
    ticks: int
    stdout: str
    stderr: str
    trace_summary: str = ""
    budget_hit: bool = False
    leaks: int = 0
    last_statements: list[str] = field(default_factory=list)
    # Counts of constructs that actually executed, keyed by construct name
    # ("loop_iteration", "function_call", "pointer_deref", "malloc", "free",
    # "struct_member", "array_index", "recursion", "switch", "file_open", ...).
    constructs: dict[str, int] = field(default_factory=dict)
    # sha1 of the normalised token stream of the program, so
    # re-running the same program can be recognised whatever the whitespace.
    program_hash: str = ""

    @property
    def ok(self) -> bool:
        return self.trap is None and self.exit_code == 0
