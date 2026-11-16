"""Shared fixtures for the shell tests: a fresh device, a shell and a fake program runner."""
from __future__ import annotations

from typing import Iterator

import pytest

from pycc.shell import EXE_MAGIC, Shell, ShellResult
from pycc.shell.ansi import strip_ansi
from pycc.shell.builtins import register_builtins
from pycc.vfs import VirtualFS, standard_layout

PROJECT = "/home/user/project"


def fake_runner(executable, argv, stdin, stdout, stderr, cwd) -> int:
    """Stands in for the C interpreter: prints the program's payload lines, echoes
    stdin with an ``in:`` prefix and exits with the number after ``exit=`` (if any)."""
    payload = executable[len(EXE_MAGIC):].decode()
    code = 0
    for line in payload.splitlines():
        if line.startswith("exit="):
            code = int(line[5:])
        elif line.startswith("err="):
            stderr.write(line[4:] + "\n")
        else:
            stdout.write(line + "\n")
    while (line := stdin.readline()) is not None:
        stdout.write("in:" + line)
    return code


def make_program(vfs: VirtualFS, path: str, payload: str = "hello",
                 executable: bool = True) -> None:
    """Write a fake compiled program (CVEXE1 magic + payload)."""
    vfs.write_file(path, EXE_MAGIC + payload.encode())
    vfs.set_executable(path, executable)


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[None]:
    """Undo any command replacement a test made."""
    yield
    register_builtins()


@pytest.fixture
def vfs() -> VirtualFS:
    return standard_layout("project", {"stdio.h": "int puts(const char *s);\n"})


@pytest.fixture
def shell(vfs) -> Shell:
    return Shell(vfs, program_runner=fake_runner)


class Runner:
    """Runs lines and gives back plain text (ANSI removed) for easy asserts."""

    def __init__(self, shell: Shell) -> None:
        self.shell = shell

    def __call__(self, line: str) -> ShellResult:
        result = self.shell.run(line)
        result.stdout = strip_ansi(result.stdout)
        result.stderr = strip_ansi(result.stderr)
        return result

    def out(self, line: str) -> str:
        return self(line).stdout


@pytest.fixture
def run(shell) -> Runner:
    return Runner(shell)
