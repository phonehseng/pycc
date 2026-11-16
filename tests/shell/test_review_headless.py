"""Review tests for the headless host, VfsHostFiles, export and layering rules."""
from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pycc.core.events import ShellOutput
from pycc.interp.hostio import EEXIST, EISDIR
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain
from pycc.shell import Shell
from pycc.shell.headless import make_headless_shell
from pycc.shell.hostfiles import VfsHostFiles
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import VirtualFS, standard_layout

READ_C = ('#include <stdio.h>\nint main(void) {\n    int c = getchar();\n'
          '    printf("c=%d\\n", c);\n    c = getchar();\n    printf("again=%d\\n", c);\n'
          '    return 0;\n}\n')
MIXED_C = ('#include <stdio.h>\nint main(void) {\n    printf("out1\\n");\n'
           '    fprintf(stderr, "err1\\n");\n    printf("out2\\n");\n    return 0;\n}\n')
HELLO_C = '#include <stdio.h>\nint main(void) {\n    printf("hi\\n");\n    return 0;\n}\n'


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers())


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    for name, text in {"read.c": READ_C, "mixed.c": MIXED_C, "hello.c": HELLO_C}.items():
        vfs.write_file(f"{PROJECT}/{name}", text)
    return vfs


@pytest.fixture
def term(fs, toolchain) -> Shell:
    return make_headless_shell(fs, toolchain)


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


# --- stdin -------------------------------------------------------------------------------

def test_terminal_stdin_without_lines_is_end_of_file(run) -> None:
    result = run("cc read.c -o read && ./read")
    assert result.exit_code == 0
    assert result.stdout == "c=-1\nagain=-1\n"


def test_terminal_stdin_lines_are_read_then_eof(run, term) -> None:
    run("cc read.c -o read")
    term.terminal_input.lines.append("Z\n")
    assert run("./read").stdout == "c=90\nagain=10\n"


def test_pipe_stdin_beats_the_terminal(run, term) -> None:
    run("cc read.c -o read")
    term.terminal_input.lines.append("Z\n")
    assert run("echo A | ./read").stdout == "c=65\nagain=10\n"
    assert term.terminal_input.lines == ["Z\n"]        # the terminal was not touched


# --- stdout and stderr -----------------------------------------------------------------------

def test_stdout_and_stderr_events_keep_program_order(term) -> None:
    term.run("cc mixed.c -o mixed")
    events = [(e.stream, e.text) for e in term.run("./mixed").events
              if isinstance(e, ShellOutput)]
    assert events == [("stdout", "out1\n"), ("stderr", "err1\n"), ("stdout", "out2\n")]


def test_stderr_skips_the_pipe_and_reaches_the_terminal_first(term) -> None:
    term.run("cc mixed.c -o mixed")
    events = [(e.stream, e.text) for e in term.run("./mixed | cat").events
              if isinstance(e, ShellOutput)]
    assert events == [("stderr", "err1\n"), ("stdout", "out1\nout2\n")]


def test_both_streams_into_a_pipe_with_2_to_1(run) -> None:
    run("cc mixed.c -o mixed")
    result = run("./mixed 2>&1 | cat")
    assert result.stdout == "out1\nerr1\nout2\n"
    assert result.stderr == ""


# --- VfsHostFiles rename --------------------------------------------------------------

def test_rename_folder_onto_an_empty_folder_replaces_it() -> None:
    vfs = standard_layout("project")
    vfs.mkdir("/tmp/old")
    vfs.write_file("/tmp/old/a.txt", "a")
    vfs.mkdir("/tmp/new")
    assert VfsHostFiles(vfs).rename("/tmp/old", "/tmp/new") == 0
    assert vfs.read_text("/tmp/new/a.txt") == "a"
    assert not vfs.exists("/tmp/new/old")
    assert not vfs.exists("/tmp/old")


def test_rename_folder_onto_a_full_folder_fails() -> None:
    vfs = standard_layout("project")
    vfs.mkdir("/tmp/old")
    vfs.mkdir("/tmp/new")
    vfs.write_file("/tmp/new/keep.txt", "k")
    assert VfsHostFiles(vfs).rename("/tmp/old", "/tmp/new") == -EEXIST
    assert vfs.is_dir("/tmp/old")
    assert vfs.read_text("/tmp/new/keep.txt") == "k"


def test_rename_onto_itself_is_a_no_op() -> None:
    vfs = standard_layout("project")
    vfs.write_file("/tmp/a.txt", "a")
    assert VfsHostFiles(vfs).rename("/tmp/a.txt", "/tmp/a.txt") == 0
    assert vfs.read_text("/tmp/a.txt") == "a"


def test_rename_file_onto_folder_is_eisdir() -> None:
    vfs = standard_layout("project")
    vfs.write_file("/tmp/a.txt", "a")
    vfs.mkdir("/tmp/box")
    assert VfsHostFiles(vfs).rename("/tmp/a.txt", "/tmp/box") == -EISDIR


# --- layering and the real disk ---------------------------------------------------------

def test_pycc_imports_only_the_standard_library() -> None:
    code = ("import sys; import pycc.cli, pycc.shell.cc, pycc.shell.make, pycc.shell.headless, "
            "pycc.shell.toolchain_commands, pycc.shell.hostfiles, pycc.shell.sysheaders; "
            "bad = sorted({m.split('.')[0] for m in sys.modules} - set(sys.stdlib_module_names) "
            "- set(sys.builtin_module_names) - {'pycc', '__main__'}); print(bad)")
    root = Path(__file__).resolve().parents[2]
    output = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                            text=True, check=True).stdout
    assert output.strip() == "[]"


def test_build_make_and_run_never_touch_the_real_disk(monkeypatch, run, fs) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError(f"real disk touched: {args!r}")

    fs.write_file(f"{PROJECT}/Makefile", "prog: hello.c\n\tcc hello.c -o prog\n")
    run("cc read.c -o warmup && ./warmup")     # load lazily imported modules first
    monkeypatch.setattr(builtins, "open", refuse)
    for name in ("write_bytes", "write_text", "mkdir", "touch", "unlink"):
        monkeypatch.setattr(Path, name, refuse)
    for name in ("makedirs", "mkdir", "remove", "rename", "replace"):
        monkeypatch.setattr(os, name, refuse)
    result = run("make && ./prog && echo x > f.txt && cat f.txt")
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "cc hello.c -o prog\nhi\nx\n"
