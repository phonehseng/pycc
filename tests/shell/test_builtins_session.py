"""Tests for session commands: help man history env export test which date sleep fortune."""
import pytest

from pycc.shell import Shell, command_names
from pycc.shell.manpages import MANPAGES, SUMMARIES, render_manpage
from pycc.shell.sessioncmds import PROVERBS
from tests.shell.conftest import PROJECT
from pycc.vfs import VirtualFS


# --- every command is documented ---------------------------------------------------------

def test_every_registered_command_has_a_manpage_and_summary() -> None:
    for name in command_names():
        assert name in MANPAGES, name
        assert SUMMARIES[name], name


def test_design_command_list_is_registered() -> None:
    expected = ("ls cd pwd cat mkdir touch rm cp mv echo head tail wc grep find diff sort uniq "
                "seq clear history man help chmod env export which true false test [ exit source "
                "sh bash cc gcc make file date sleep tree fortune").split()
    assert set(expected) <= set(command_names())


def test_render_manpage_keeps_usage_on_its_own_line() -> None:
    text = render_manpage(MANPAGES["grep"])
    lines = text.splitlines()
    assert "grep - find lines that match a pattern" in lines[0]
    assert lines[1].startswith("grep [-n]")
    assert any(line.startswith("  - -n") for line in lines)


# --- help and man -------------------------------------------------------------------------

def test_help_lists_groups_with_one_liners(run) -> None:
    output = run.out("help")
    assert "Files:" in output and "Text and pipes:" in output
    assert "  grep     find lines that match a pattern" in output


def test_help_for_one_command(run) -> None:
    assert run.out("help ls").startswith("ls - list the files in a folder")
    assert run("help nothing").exit_code == 1


def test_man(run) -> None:
    assert run.out("man cd").startswith("cd - change the current folder")
    result = run("man nothing")
    assert result.stderr.startswith("No manual entry for nothing") and result.exit_code == 16
    assert "What manual page do you want?" in run("man").stderr


# --- history ------------------------------------------------------------------------------

def test_history_numbers_lines(run) -> None:
    run("echo a")
    run("pwd")
    assert run.out("history") == "    1  echo a\n    2  pwd\n    3  history\n"
    assert run.out("history 1") == "    4  history 1\n"


def test_history_clear(run, shell) -> None:
    run("echo a")
    run("history -c")
    assert shell.history == []


# --- env and export -----------------------------------------------------------------------

def test_env_shows_exported_only(run) -> None:
    run("LOCAL=1")
    output = run.out("env")
    assert "HOME=/home/user\n" in output and "LOCAL" not in output


def test_export(run) -> None:
    run("export CROP=corn")
    assert "CROP=corn\n" in run.out("env")
    run("SEASON=spring; export SEASON")
    assert 'declare -x SEASON="spring"' in run.out("export")


def test_export_invalid_name(run) -> None:
    assert "not a valid identifier" in run("export 1x=2").stderr


def test_exported_variables_reach_scripts(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/show.sh", "echo crop=$CROP\n")
    assert run.out("export CROP=beet; sh show.sh") == "crop=beet\n"


# --- test and [ ---------------------------------------------------------------------------

@pytest.fixture
def files(vfs: VirtualFS) -> None:
    vfs.write_file(f"{PROJECT}/full.txt", "x")
    vfs.write_file(f"{PROJECT}/empty.txt", "")
    vfs.mkdir(f"{PROJECT}/dir")


@pytest.mark.parametrize("expression, status", [
    ("-f full.txt", 0), ("-f dir", 1), ("-d dir", 0), ("-e dir", 0), ("-e nope", 1),
    ("-s full.txt", 0), ("-s empty.txt", 1), ("-x full.txt", 1), ("-w full.txt", 0),
    ("-z ''", 0), ("-n ''", 1), ("-n abc", 0), ("abc", 0), ("''", 1),
    ("a = a", 0), ("a == b", 1), ("a != b", 0),
    ("3 -eq 3", 0), ("3 -ne 3", 1), ("2 -lt 3", 0), ("3 -le 3", 0), ("4 -gt 3", 0), ("2 -ge 3", 1),
    ("! -f nope", 0), ("-f full.txt -a -d dir", 0), ("-f nope -o -d dir", 0),
    (r"\( -f nope \)", 1), (r"! \( -f nope -o -d nope \)", 0),
])
def test_test_expressions(files, run, expression, status) -> None:
    assert run(f"test {expression}").exit_code == status
    assert run(f"[ {expression} ]").exit_code == status


def test_empty_test_is_false(run) -> None:
    assert run("test").exit_code == 1
    assert run("[ ]").exit_code == 1


def test_bracket_needs_closing(run, shell) -> None:
    result = run("[ -f x")
    assert result.exit_code == 2 and "bash: [: missing `]'" in result.stderr
    assert shell.last_diagnostics[0].code == "shell_syntax_error"


def test_integer_expected(run) -> None:
    result = run("[ abc -lt 3 ]")
    assert result.exit_code == 2 and "integer expression expected" in result.stderr


def test_unquoted_variable_with_spaces(run) -> None:
    assert "too many arguments" in run('X="a b"; [ $X = a ]').stderr
    assert run('X="a b"; [ "$X" = "a b" ]').exit_code == 0


def test_true_false(run) -> None:
    assert run("true").exit_code == 0 and run("false").exit_code == 1


# --- which, date, sleep, fortune, placeholders --------------------------------------------

def test_which(vfs, run) -> None:
    assert run.out("which ls grep") == "/bin/ls\n/bin/grep\n"
    result = run("which nothing")
    assert result.exit_code == 1 and "which: no nothing in (/bin:.)" in result.stderr


def test_date_is_fixed(run) -> None:
    assert run.out("date") == run.out("date") == "Mon Mar  4 08:00:00 UTC 2024\n"


def test_sleep(run) -> None:
    assert run("sleep 2").exit_code == 0
    assert "invalid time interval" in run("sleep soon").stderr
    assert "missing operand" in run("sleep").stderr


def test_fortune_prints_a_proverb(run) -> None:
    assert run.out("fortune").strip() in PROVERBS


@pytest.mark.parametrize("tool", ["cc", "gcc", "make"])
def test_build_placeholders(run, shell, tool) -> None:
    result = run(f"{tool} main.c")
    assert result.exit_code == 127 and result.stderr.startswith(f"{tool}: the ")
    assert "not installed in this workshop yet" in result.stderr


def test_which_finds_programs_through_path(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/bot", b"CVEXE1\nx")
    vfs.set_executable(f"{PROJECT}/bot")
    assert run.out("which bot") == "./bot\n"


def test_which_needs_bin_markers(run) -> None:
    bare = Shell(VirtualFS(), "/")
    assert bare.run("which ls").exit_code == 1
