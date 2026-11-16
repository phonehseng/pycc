"""Shared helpers for the interpreter tests: compile C text, link it, run it.

The helpers go through the real toolchain (preprocessor, lexer, parser, checker,
linker) so every interpreter test also proves that the checker's annotations are
what the interpreter expects. This module has no tests of its own.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pycc.core.events import Event, InputRequest, DeviceCall
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.compiler import build_project
from pycc.interp.hostio import DictSourceFiles, ProgramIO
from pycc.interp.interpreter import CompletedRun, Interpreter, run_to_completion
from pycc.interp.linker import LinkedProgram
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import CompileOptions, Toolchain


def toolchain(registry: BuiltinRegistry | None = None) -> Toolchain:
    """A toolchain over the standard library (or the given registry)."""
    return Toolchain(registry or make_registry(), system_headers(), CompileOptions(wall=True))


def build(source: str | dict[str, str], registry: BuiltinRegistry | None = None) -> LinkedProgram:
    """Compile and link ``source`` (one main.c, or a dict of path -> text).

    Every ``.c`` file in the dict is compiled; headers are found by ``#include``.
    Compile or link errors fail the test with gcc's messages.
    """
    files = {"main.c": source} if isinstance(source, str) else dict(source)
    tools = toolchain(registry)
    sources = sorted(path for path in files if path.endswith(".c"))
    vfs = DictSourceFiles({"/" + path.lstrip("/"): text for path, text in files.items()})
    result = build_project(vfs, sources, "a.out", tools, cwd="/")
    errors = [d.format_gcc() for d in result.diagnostics if d.is_error]
    assert result.ok and result.link_result is not None, "\n".join(errors)
    program = result.link_result.program
    assert program is not None
    return program


def run_c(source: str | dict[str, str], argv: list[str] | None = None, stdin: str = "",
          io: ProgramIO | None = None, budget: int = 1_000_000,
          registry: BuiltinRegistry | None = None,
          device: Callable[[DeviceCall], Any] | None = None,
          on_event: Callable[[Event], None] | None = None) -> CompletedRun:
    """Build and run a program to the end; returns (exit_code, stdout, stderr, summary)."""
    registry = registry or make_registry()
    program = build(source, registry)
    return run_to_completion(program, registry, argv or ["prog"], stdin, io, device=device,
                             on_event=on_event, budget=budget)


def events_of(source: str | dict[str, str], argv: list[str] | None = None,
              stdin_lines: list[str | None] | None = None, budget: int = 1_000_000,
              registry: BuiltinRegistry | None = None) -> tuple[list[Event], int, Interpreter]:
    """Drive a run by hand and return every event, the exit code and the interpreter."""
    registry = registry or make_registry()
    program = build(source, registry)
    interpreter = Interpreter(program, registry, budget=budget)
    replies = list(stdin_lines or [])
    generator = interpreter.run(argv or ["prog"])
    events: list[Event] = []
    reply: Any = None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as done:
            return events, done.value, interpreter
        events.append(event)
        reply = None
        if isinstance(event, InputRequest):
            reply = replies.pop(0) if replies else None


def warning_codes(run: CompletedRun) -> list[str]:
    """The codes of every runtime warning of a run, in order."""
    return [diagnostic.code for diagnostic in run.summary.warnings]


def trap_code(run: CompletedRun) -> str | None:
    """The code of the trap that stopped a run (None if it ended normally)."""
    return run.summary.trap.code if run.summary.trap is not None else None


def program(body: str, includes: tuple[str, ...] = ("stdio.h",), before: str = "") -> str:
    """A whole C file: the includes, top-level code, then ``int main(void) { body }``."""
    head = "".join(f"#include <{name}>\n" for name in includes)
    return f"{head}{before}\nint main(void) {{\n{body}\n}}\n"
