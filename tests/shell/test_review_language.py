"""Review tests for the shell language: quoting, expansion, redirection order, exit codes."""
from pycc.core.events import RunProgram
from pycc.shell.parser import needs_more_input
from tests.shell.conftest import PROJECT, make_program

BACKSLASH = "\\"


# --- quoting and expansion ----------------------------------------------------------------

def test_single_quotes_keep_dollar_double_quotes_expand(run) -> None:
    assert run.out("X=corn; echo '$X' \"$X\" $X") == "$X corn corn\n"


def test_nested_command_substitution(run) -> None:
    assert run.out("echo $(echo outer $(echo inner))") == "outer inner\n"


def test_command_substitution_inside_double_quotes_is_one_word(run) -> None:
    assert run.out('for w in "$(echo a b)"; do echo "[$w]"; done') == "[a b]\n"


def test_backslash_newline_joins_lines(run) -> None:
    assert run.out("echo one" + BACKSLASH + "\ntwo") == "onetwo\n"
    assert run.out("echo one " + BACKSLASH + "\n two") == "one two\n"


def test_variable_with_spaces_splits_only_when_unquoted(run) -> None:
    result = run('X="a   b"; echo $X; echo "$X"; touch $X; ls')
    assert result.stdout.splitlines()[:2] == ["a b", "a   b"]
    assert run.out("ls -1") == "a\nb\n"


def test_quoted_at_with_no_arguments_gives_no_words(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/args.sh", 'for a in "$@"; do echo "[$a]"; done\necho n=$#\n')
    assert run.out("sh args.sh") == "n=0\n"
    assert run.out("sh args.sh 'x y' z") == "[x y]\n[z]\nn=2\n"


def test_quoted_empty_variable_is_still_one_argument(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/count.sh", "echo $#\n")
    assert run.out('E=; sh count.sh "$E" ""') == "2\n"


def test_quoted_glob_is_not_expanded(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/a.c", "")
    assert run.out('echo "*.c" *.c ' + BACKSLASH + "*.c") == "*.c a.c *.c\n"


def test_for_over_glob_without_matches_keeps_the_pattern(run) -> None:
    assert run.out('for f in *.zz; do echo "got $f"; done') == "got *.zz\n"


def test_needs_more_input_counts_trailing_backslashes() -> None:
    assert needs_more_input("echo a " + BACKSLASH)
    assert not needs_more_input("echo a " + BACKSLASH * 2)
    assert needs_more_input("echo a " + BACKSLASH * 3)


# --- redirection order --------------------------------------------------------------------

def test_file_then_dup_sends_both_streams_to_the_file(run) -> None:
    result = run("ls nothing > out.txt 2>&1")
    assert result.stdout == "" and result.stderr == ""
    assert "No such file or directory" in run.out("cat out.txt")


def test_dup_then_file_keeps_errors_on_the_terminal(run) -> None:
    result = run("ls nothing 2>&1 > out.txt")
    assert "No such file or directory" in result.stdout + result.stderr
    assert run.out("cat out.txt") == ""


def test_redirect_to_a_trailing_slash_is_a_directory_error(run, vfs) -> None:
    result = run("echo hi > logs/")
    assert result.exit_code == 1
    assert "logs/: Is a directory" in result.stderr
    assert not vfs.exists(f"{PROJECT}/logs")
    assert result.finished.diagnostics[0].code == "redirect_error"


# --- exit codes ---------------------------------------------------------------------------

def test_pipeline_status_is_the_last_command(run) -> None:
    assert run.out("false | true; echo $?; true | false; echo $?") == "0\n1\n"


def test_if_without_taken_branch_is_zero_and_body_status_wins(run) -> None:
    assert run.out("if false; then echo x; fi; echo $?") == "0\n"
    assert run.out("if true; then false; fi; echo $?") == "1\n"


def test_exit_in_a_pipeline_stage_ends_only_that_stage(run, shell) -> None:
    result = run("exit 3 | cat; echo after")
    assert result.stdout == "after\n"
    assert not shell.exited


def test_exit_inside_if_in_a_script_stops_the_script_only(run, vfs, shell) -> None:
    vfs.write_file(f"{PROJECT}/stop.sh",
                   "if true; then\n  echo first\n  exit 4\nfi\necho never\n")
    result = run("sh stop.sh; echo status=$?")
    assert result.stdout == "first\nstatus=4\n"
    assert not shell.exited


def test_exit_inside_if_at_the_prompt_ends_the_shell(run, shell) -> None:
    result = run("if true; then exit 5; fi; echo never")
    assert result.exit_code == 5 and result.stdout == ""
    assert shell.exited


# --- programs see exported variables -------------------------------------------------------

def _first_run_event(shell, line: str) -> RunProgram:
    generator = shell.execute(line)
    event = next(event for event in generator if isinstance(event, RunProgram))
    generator.close()
    return event


def test_run_program_carries_exported_environment(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/bot")
    shell.run("export CROP=corn; SECRET=1")
    event = _first_run_event(shell, "./bot")
    assert event.env["CROP"] == "corn" and event.env["HOME"] == "/home/user"
    assert "SECRET" not in event.env


def test_prefix_assignment_is_exported_for_that_command_only(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/bot")
    event = _first_run_event(shell, "SEED=7 ./bot")
    assert event.env["SEED"] == "7"
    assert "SEED" not in shell.env and "SEED" not in shell.exported


# --- history ---------------------------------------------------------------------------------

def test_history_skips_empty_and_blank_lines(shell) -> None:
    for line in ("", "   ", "\t", "echo hi", ""):
        shell.run(line)
    assert shell.history == ["echo hi"]
