"""Tests for tab completion."""
import pytest

from tests.shell.conftest import PROJECT, make_program
from pycc.vfs import VirtualFS


@pytest.fixture
def project(vfs: VirtualFS) -> VirtualFS:
    vfs.mkdir(f"{PROJECT}/src")
    vfs.write_file(f"{PROJECT}/src/main.c", "")
    vfs.write_file(f"{PROJECT}/src/maths.c", "")
    vfs.write_file(f"{PROJECT}/notes.txt", "")
    vfs.write_file(f"{PROJECT}/.hidden", "")
    make_program(vfs, f"{PROJECT}/prog")
    vfs.write_file(f"{PROJECT}/Makefile",
                   "CC = gcc\nCFLAGS := -Wall\n.PHONY: all clean\nall: prog\n"
                   "prog: main.o farm.o\n\t$(CC) $^ -o $@\n%.o: %.c\n\t$(CC) -c $<\n"
                   "clean:\n\trm -f *.o\n")


def test_command_names(shell) -> None:
    assert shell.complete("gr") == ["grep"]
    assert {"cat", "cc", "cd", "chmod", "clear", "cp"} <= set(shell.complete("c"))


def test_command_after_pipe_and_and(shell) -> None:
    assert shell.complete("ls | gr") == ["grep"]
    assert shell.complete("true && ech") == ["echo"]


def test_paths(project, shell) -> None:
    assert shell.complete("cat no") == ["notes.txt"]
    assert shell.complete("cat src/ma") == ["src/main.c", "src/maths.c"]
    assert shell.complete("ls s") == ["src/"]


def test_hidden_only_with_dot(project, shell) -> None:
    assert ".hidden" not in shell.complete("cat ")
    assert shell.complete("cat .h") == [".hidden"]


def test_cd_completes_folders_only(project, shell) -> None:
    assert shell.complete("cd ") == ["src/"]


def test_dot_slash_completes_runnable(project, shell) -> None:
    assert shell.complete("./p") == ["./prog"]
    assert shell.complete("./n") == []


def test_options(shell) -> None:
    assert shell.complete("cc -W") == ["-Wall", "-Wextra", "-Werror"]
    assert shell.complete("ls -") == ["-1", "-a", "-l"]
    assert shell.complete("make -") == ["-f", "-n"]


def test_make_targets(project, shell) -> None:
    assert shell.complete("make ") == ["all", "clean", "prog"]
    assert shell.complete("make c") == ["clean"]


def test_make_without_makefile_completes_paths(vfs, shell) -> None:
    vfs.write_file(f"{PROJECT}/x.c", "")
    assert shell.complete("make x") == ["x.c"]


def test_cursor_in_the_middle(project, shell) -> None:
    assert shell.complete("cat no | wc", cursor=6) == ["notes.txt"]


def test_tilde_paths(project, shell) -> None:
    assert shell.complete("cd ~/pro") == ["~/project/"]
