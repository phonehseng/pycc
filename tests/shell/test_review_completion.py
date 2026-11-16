"""Review tests for tab completion: keywords, glued redirections, mid-line and spaces."""
import pytest

from tests.shell.conftest import PROJECT
from pycc.vfs import VirtualFS

BACKSLASH = "\\"


@pytest.fixture
def project(vfs: VirtualFS) -> VirtualFS:
    """A project with a source file, a notes file and a name holding a space."""
    vfs.write_file(f"{PROJECT}/main.c", "")
    vfs.write_file(f"{PROJECT}/notes.txt", "")
    vfs.write_file(f"{PROJECT}/my notes.txt", "")
    return vfs


def test_command_after_then_and_do_keywords(shell) -> None:
    assert shell.complete("if true; then ec") == ["echo"]
    assert shell.complete("for x in a; do gr") == ["grep"]
    assert shell.complete("! tr") == ["tr", "tree", "true"]


def test_command_after_an_assignment(shell) -> None:
    assert shell.complete("CROP=corn ech") == ["echo"]


def test_glued_redirection_completes_the_file(project, shell) -> None:
    assert shell.complete("echo hi >ma") == [">main.c"]
    assert shell.complete("sort <no") == ["<notes.txt"]


def test_word_after_redirection_is_a_path_even_for_make(project, shell) -> None:
    project.write_file(f"{PROJECT}/Makefile", "all:\n\techo hi\n")
    assert shell.complete("make > ma") == ["main.c"]


def test_completion_in_the_middle_of_a_line(project, shell) -> None:
    line = "cat ma | wc -l"
    assert shell.complete(line, cursor=len("cat ma")) == ["main.c"]
    assert shell.complete(line, cursor=len("cat ma | w")) == ["wc", "which"]


def test_names_with_spaces_are_escaped(project, shell) -> None:
    assert shell.complete("cat my") == ["my" + BACKSLASH + " notes.txt"]


def test_escaped_space_continues_the_same_word(project, shell) -> None:
    assert shell.complete("cat my" + BACKSLASH + " n") == ["my" + BACKSLASH + " notes.txt"]


def test_completed_escaped_name_runs_as_one_word(project, run, shell) -> None:
    project.write_file(f"{PROJECT}/my notes.txt", "inside\n")
    candidate = shell.complete("cat my")[0]
    assert run.out("cat " + candidate) == "inside\n"
