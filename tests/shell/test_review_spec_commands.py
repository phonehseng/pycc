"""Review fixes for the shell's commands as the spec lists them.

``time`` and ``printf`` exist, ``ls`` sorts its operands, ``mv`` blames the right
path, the command-not-found line points at a close file name, and Tab completes
variables, ``chmod`` modes and more options.
"""
from __future__ import annotations

from pycc.core.events import RunSummary
from pycc.shell.sessioncmds import time_report
from tests.shell.conftest import PROJECT, make_program


def _summary(statements: int, ticks: int = 0) -> RunSummary:
    return RunSummary(exit_code=0, trap=None, warnings=[], statements=statements,
                      ticks=ticks, stdout="", stderr="")


# --- time -----------------------------------------------------------------------------------

def test_time_runs_the_command_and_prints_three_lines(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/a.txt", "x\n")
    result = run("time ls")
    assert result.exit_code == 0 and "a.txt" in result.stdout
    assert result.stderr == "\nreal\t0m0.001s\nuser\t0m0.000s\nsys\t0m0.000s\n"


def test_time_keeps_the_commands_status(run) -> None:
    result = run("time false")
    assert result.exit_code == 1 and "real\t" in result.stderr


def test_time_report_uses_the_steady_clock() -> None:
    report = time_report([_summary(statements=300, ticks=2)])
    assert report == "\nreal\t0m0.205s\nuser\t0m0.003s\nsys\t0m0.001s\n"


def test_time_has_a_man_page(run) -> None:
    assert "time - measure how long a command takes" in run("man time").stdout


def test_time_counts_the_programs_it_ran(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/bot")
    shell.program_runner = _recording_runner(shell)
    result = shell.run("time ./bot")
    assert result.stdout == "hello\n"
    assert "user\t0m0.005s" in result.stderr and "sys\t0m0.001s" in result.stderr


def _recording_runner(shell):
    """A fake runner that also hands the shell a run summary, as the headless host does."""
    def runner(executable, argv, stdin, stdout, stderr, cwd) -> int:
        stdout.write("hello\n")
        shell.record_run(_summary(statements=500))
        return 0
    return runner


# --- printf ---------------------------------------------------------------------------------

def test_printf_formats_like_c(run) -> None:
    assert run.out("printf '%s has %d cats\\n' Ada 3") == "Ada has 3 cats\n"
    assert run.out("printf '%5.2f|%-4s|%03d|%x|%c\\n' 3.14159 ab 7 255 hello") == \
        " 3.14|ab  |007|ff|h\n"


def test_printf_reuses_the_format_for_more_arguments(run) -> None:
    assert run.out("printf '%s=%d\\n' a 1 b 2") == "a=1\nb=2\n"


def test_printf_adds_no_newline_and_keeps_percent(run) -> None:
    assert run.out("printf 'x'") == "x"
    assert run.out("printf '100%%\\n'") == "100%\n"


def test_printf_bad_number_prints_zero_and_fails(run) -> None:
    result = run("printf '%d\\n' abc")
    assert result.exit_code == 1 and result.stdout == "0\n"
    assert "bash: printf: abc: invalid number" in result.stderr


def test_printf_without_a_format_is_a_usage_error(run) -> None:
    result = run("printf")
    assert result.exit_code == 2 and "usage: printf format" in result.stderr


# --- ls, mv, command not found --------------------------------------------------------------

def test_ls_sorts_file_operands_like_gnu_ls(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/notes.txt", "n\n")
    vfs.write_file(f"{PROJECT}/main.c", "m\n")
    assert run.out("ls notes.txt main.c") == "main.c  notes.txt\n"
    long_lines = run.out("ls -l notes.txt main.c").splitlines()
    assert long_lines[0].endswith("main.c") and long_lines[1].endswith("notes.txt")


def test_mv_of_a_system_file_blames_the_source(run) -> None:
    result = run("mv /usr/include/stdio.h .")
    assert "'/usr/include/stdio.h' is protected" in result.stderr
    assert "'.' is protected" not in result.stderr


def test_command_not_found_points_at_a_file_with_another_case(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/notes.txt", "n\n")
    result = run("Notes.txt")
    assert "There is a file called 'notes.txt' here: try `cat notes.txt`." in result.stderr
    assert "Did you mean" not in result.stderr


def test_command_not_found_never_suggests_a_command_for_a_file_name(run) -> None:
    assert "Did you mean" not in run("Notes.txt").stderr


# --- completion -----------------------------------------------------------------------------

def test_completion_of_variables_and_export(shell) -> None:
    assert shell.complete("echo $HO") == ["$HOME"]
    assert shell.complete("echo ${HO") == ["${HOME}"]
    assert shell.complete("export H") == ["HOME"]


def test_completion_of_more_options_and_chmod_modes(shell) -> None:
    assert shell.complete("find . -") == ["-name", "-type"]
    assert shell.complete("history -") == ["-c"]
    assert shell.complete("echo -") == ["-e", "-n"]
    assert shell.complete("chmod ") == ["+x", "-x"]
