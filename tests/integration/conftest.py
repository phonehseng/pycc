"""A workbench for the end-to-end scenarios.

Every scenario drives a real ``Shell`` over a ``VirtualFS``, compiles with the
standard-library ``Toolchain`` and runs the programs through ``shell/headless.py``.
Nothing is faked: what these tests see is what the terminal shows.

``Terminal`` bundles one such setup. ``Terminal.run`` returns a ``Line`` whose
``stdout``/``stderr`` have the colour codes removed (``raw_stderr`` keeps them, for
the tests that check the explanation line is dimmed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pycc.core.diagnostics import Diagnostic
from pycc.core.events import CommandFinished, RunSummary
from pycc.interp.toolchain import Toolchain
from pycc.shell.ansi import DIM, strip_ansi
from pycc.shell.builtins import register_builtins
from pycc.shell.executor import Shell
from pycc.shell.headless import DEFAULT_BUDGET, make_headless_shell
from pycc.interp.devices import register_device_function
from pycc.interp.stdlib import make_registry, system_headers
from pycc.vfs import VirtualFS, standard_layout

PROJECT = "/home/user/project"


@dataclass
class Line:
    """One command line as the user saw it."""

    exit_code: int
    stdout: str
    stderr: str
    raw_stderr: str
    finished: CommandFinished | None

    @property
    def runs(self) -> list[RunSummary]:
        """The summary of every program that ran on this line."""
        return list(self.finished.runs) if self.finished else []

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """The diagnostics of the commands that failed on this line."""
        return list(self.finished.diagnostics) if self.finished else []

    @property
    def codes(self) -> list[str]:
        return [diag.code for diag in self.diagnostics]


@dataclass
class Terminal:
    """A shell, its filesystem, its toolchain and (optionally) a device for its programs."""

    vfs: VirtualFS
    toolchain: Toolchain
    device: Any = None
    shell: Shell = field(init=False)
    budget: int = DEFAULT_BUDGET

    def __post_init__(self) -> None:
        factory = (lambda: self.device) if self.device is not None else None
        self.shell = make_headless_shell(self.vfs, self.toolchain, device_factory=factory,
                                         budget=self.budget)

    def write(self, files: dict[str, str]) -> None:
        """Save project files (paths relative to the project folder)."""
        for name, text in files.items():
            path = f"{PROJECT}/{name}"
            folder = path.rsplit("/", 1)[0]
            if not self.vfs.is_dir(folder):
                self.vfs.mkdir(folder, parents=True)
            self.vfs.write_file(path, text)

    def read(self, name: str) -> bytes:
        return self.vfs.read_file(f"{PROJECT}/{name}")

    def run(self, line: str) -> Line:
        """Type ``line`` at the prompt and wait for it to finish."""
        result = self.shell.run(line)
        return Line(result.exit_code, strip_ansi(result.stdout), strip_ansi(result.stderr),
                    result.stderr, result.finished)

    def ok(self, line: str) -> Line:
        """Run a line that must succeed."""
        result = self.run(line)
        assert result.exit_code == 0, (line, result.stdout, result.stderr)
        return result


def assert_reported(line: Line, gcc: str, friendly: str) -> None:
    """``gcc`` is printed as a whole stderr line and its friendly line follows it.

    Notes that belong to the diagnostic (``note: previous definition...``) may sit in
    between, as in gcc; the friendly line is the first line after them and it is
    dimmed on the terminal. Compiler, linker, make and crash reports indent it by two
    spaces under the error; bash's own errors (``bash: ./x: Permission denied``) put
    it on the next line without the indent.
    """
    lines = line.stderr.splitlines()
    assert gcc in lines, f"{gcc!r} not in stderr:\n{line.stderr}"
    index = lines.index(gcc) + 1
    while index < len(lines) and ": note: " in lines[index]:
        index += 1
    assert index < len(lines), f"no friendly line after {gcc!r}"
    shown = lines[index]
    if not gcc.startswith("bash: "):
        assert shown.startswith("  "), f"not an indented friendly line: {shown!r}"
    assert friendly in shown, f"{friendly!r} not in {shown!r}"
    assert f"{DIM}{shown}" in line.raw_stderr, "the friendly line is not dimmed"


@pytest.fixture(autouse=True)
def fresh_commands() -> Any:
    """Undo any command a test replaced (the command table is global)."""
    yield
    register_builtins()


def stdlib_toolchain() -> Toolchain:
    """A toolchain with the C standard library and nothing else."""
    return Toolchain(make_registry(), system_headers())


@dataclass
class Reply:
    value: int


class Counter:
    """A tiny device for the device-call scenarios: ``pulse(n)`` adds to a counter."""

    def __init__(self) -> None:
        self.count = 0
        self.tick = 0
        self.log: list[tuple[str, list[int]]] = []

    def perform(self, name: str, args: list[int]) -> Reply:
        self.log.append((name, list(args)))
        self.tick += 1
        if name == "pulse":
            self.count += args[0]
            return Reply(1)
        if name == "read_count":
            return Reply(self.count)
        raise AssertionError(f"unknown device function {name}")


COUNTER_H = """#ifndef COUNTER_H
#define COUNTER_H
int pulse(int amount);
int read_count(void);
#endif
"""


def counter_toolchain() -> Toolchain:
    """The standard library plus ``<counter.h>``, whose functions go to a ``Counter``."""
    registry = make_registry()
    register_device_function(registry, "pulse", "int pulse(int amount);", header="counter.h")
    register_device_function(registry, "read_count", "int read_count(void);",
                             header="counter.h")
    headers = system_headers()
    headers.add("counter.h", COUNTER_H)
    return Toolchain(registry, headers)


@pytest.fixture(scope="session")
def toolchain() -> Toolchain:
    return stdlib_toolchain()


@pytest.fixture
def term(toolchain: Toolchain) -> Terminal:
    """A fresh project folder."""
    return Terminal(standard_layout("project"), toolchain)
