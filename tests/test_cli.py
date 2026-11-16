"""Tests for the command line (pycc/cli.py): run, build, exec and shell."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from pycc.cli import main, sandbox_from, split_args

HELLO = '#include <stdio.h>\nint main(int argc, char **argv) {\n' \
        '    printf("hi %d %s\\n", argc, argc > 1 ? argv[1] : "-");\n    return 3;\n}\n'
UAF = '#include <stdlib.h>\nint main(void) {\n    int *p = malloc(sizeof *p);\n' \
      '    free(p);\n    return *p;\n}\n'
ASK = '#include <stdio.h>\nint main(void) {\n    int n = 0;\n    printf("n? ");\n' \
      '    if (scanf("%d", &n) != 1) return 1;\n    printf("%d\\n", n * 2);\n    return 0;\n}\n'


@pytest.fixture
def folder(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.c").write_text(HELLO)
    (tmp_path / "uaf.c").write_text(UAF)
    (tmp_path / "ask.c").write_text(ASK)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.c").write_text("int twice(int x) { return 2 * x; }\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return tmp_path


def test_split_args_separates_sources_flags_and_program_args() -> None:
    assert split_args(["a.c", "-Wall", "-I", "inc", "b.c", "--", "x", "-v"]) == (
        ["a.c", "b.c"], ["-Wall", "-I", "inc"], ["x", "-v"])


def test_run_prints_output_and_returns_the_exit_code(folder, capsys) -> None:
    assert main(["run", "hello.c", "--", "world"]) == 3
    assert capsys.readouterr().out == "hi 2 world\n"


def test_run_reports_a_use_after_free_like_a_sanitizer(folder, capsys) -> None:
    assert main(["run", "uaf.c"]) == 139
    err = capsys.readouterr().err
    assert "uaf.c:5:5: error: heap-use-after-free" in err
    assert "Segmentation fault (core dumped)" in err
    assert "\x1b[" not in err                      # no colour codes when not a terminal


def test_run_reads_standard_input(folder, capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("21\n"))
    assert main(["run", "ask.c"]) == 0
    assert capsys.readouterr().out == "n? 42\n"


def test_a_compile_error_stops_before_running(folder, capsys) -> None:
    (folder / "bad.c").write_text("int main(void) {\n    return 0\n}\n")
    assert main(["run", "bad.c"]) == 1
    assert "bad.c:2:13: error: expected ';' before '}' token" in capsys.readouterr().err


def test_build_writes_a_program_that_exec_runs(folder, capsys) -> None:
    assert main(["build", "hello.c", "-o", "hello.prog"]) == 0
    assert (folder / "hello.prog").read_bytes().startswith(b"CVEXE1\n")
    assert main(["exec", "hello.prog", "again"]) == 3
    assert capsys.readouterr().out == "hi 2 again\n"


def test_programs_never_write_to_the_real_disk(folder, capsys) -> None:
    (folder / "w.c").write_text('#include <stdio.h>\nint main(void) {\n'
                                '    FILE *f = fopen("out.txt", "w");\n'
                                '    fprintf(f, "x\\n");\n    fclose(f);\n    return 0;\n}\n')
    assert main(["run", "w.c"]) == 0
    assert not (folder / "out.txt").exists()


def test_shell_command_runs_cc_and_pipes(folder, capsys) -> None:
    assert main(["shell", "-c", "cc hello.c -o h && ./h | wc -l; ls src"]) == 0
    assert capsys.readouterr().out == "1\nutil.c\n"


def test_sandbox_skips_hidden_folders(folder) -> None:
    vfs, root = sandbox_from(folder)
    assert vfs.is_file(f"{root}/src/util.c")
    assert not vfs.exists(f"{root}/.git")
