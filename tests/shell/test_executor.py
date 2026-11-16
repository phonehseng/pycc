"""Tests for the executor: lists, pipes, redirection, variables, scripts, programs."""
from typing import Callable

import pytest

from pycc.core.events import (ClearScreen, CommandFinished, RunProgram, ShellOutput, StringSink,
                         StringSource)
from pycc.shell import (IO, OBJ_MAGIC, CommandContext, Shell, TerminalSink, register_command,
                   run_program_headless)
from pycc.shell.ansi import DIM
from tests.shell.conftest import PROJECT, fake_runner, make_program


def drive(shell: Shell, line: str, handler: Callable[[RunProgram], int]) -> tuple[int, list]:
    """Run ``line`` answering each RunProgram with ``handler(event)``."""
    generator = shell.execute(line)
    events, reply = [], None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as stop:
            return stop.value, events
        events.append(event)
        reply = handler(event) if isinstance(event, RunProgram) else None


# --- basics, lists, exit codes ------------------------------------------------------------

def test_echo_reaches_terminal_as_shell_output(shell) -> None:
    result = shell.run("echo hello")
    assert result.exit_code == 0
    assert result.events[0] == ShellOutput("stdout", "hello\n")
    assert result.events[1] == CommandFinished("echo hello", 0)
    assert len(result.events) == 2


def test_command_finished_carries_diagnostics_and_constructs(vfs, shell) -> None:
    finished = shell.run("cat nope | wc -l > n.txt && echo $?").finished
    assert finished.exit_code == 0 and finished.line == "cat nope | wc -l > n.txt && echo $?"
    assert [d.code for d in finished.diagnostics] == ["no_such_file"]
    assert finished.constructs == {"pipeline": 1, "redirection": 1, "exit_code_test": 2}


def test_command_finished_counts_scripts(vfs, shell) -> None:
    vfs.write_file(f"{PROJECT}/a.sh", "if true; then echo; fi\n")
    constructs = shell.run("sh a.sh").finished.constructs
    assert constructs == {"script_run": 1, "exit_code_test": 1}


def test_record_run_and_build_reach_command_finished(shell) -> None:
    def fake_make(ctx: CommandContext, args: list[str]) -> int:
        ctx.shell.record_build("built")
        ctx.shell.count_construct("make_rebuild", 2)
        return 0

    register_command("make", fake_make)
    generator = shell.execute("make")
    event = next(generator)
    assert isinstance(event, CommandFinished)
    assert event.build == "built" and event.constructs == {"make_rebuild": 2}
    assert shell.last_build == "built"
    shell.record_run("summary")            # hosts add RunSummary objects during a line
    assert shell.run("true").finished.runs == []


def test_ctrl_c_closes_and_sets_130(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    generator = shell.execute("./a; echo never")
    assert isinstance(next(generator), RunProgram)
    generator.close()
    assert shell.last_status == 130
    assert shell.run("echo $?").stdout == "130\n"


def test_execute_subcommand_runs_without_history(shell) -> None:
    sink = StringSink()
    io = IO(StringSource(""), sink, StringSink())
    generator = shell.execute_subcommand("echo recipe; exit 2", io)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        assert stop.value == 2
    assert sink.text == "recipe\n" and shell.history == []


def test_sequence_runs_all(run) -> None:
    assert run.out("echo a; echo b\necho c") == "a\nb\nc\n"


@pytest.mark.parametrize("line, output", [
    ("true && echo yes", "yes\n"),
    ("false && echo yes", ""),
    ("false || echo rescued", "rescued\n"),
    ("true || echo never", ""),
    ("false && echo a || echo b", "b\n"),
    ("false; echo $?", "1\n"),
    ("true; echo $?", "0\n"),
    ("nosuch 2>/dev/null; echo $?", "127\n"),
    ("! false; echo $?", "0\n"),
    ("cat missing 2>/dev/null || echo status=$?", "status=1\n"),
])
def test_exit_code_semantics(run, line, output) -> None:
    assert run.out(line) == output


def test_status_of_last_command_is_returned(run) -> None:
    assert run("true; false").exit_code == 1


def test_exit_sets_status_and_flag(shell, run) -> None:
    result = run("exit 3; echo unreachable")
    assert result.exit_code == 3 and result.stdout == "" and shell.exited


def test_exit_with_bad_argument(run) -> None:
    result = run("exit abc")
    assert result.exit_code == 2 and "numeric argument required" in result.stderr


# --- variables and substitution -----------------------------------------------------------

def test_variables(run) -> None:
    assert run.out("NAME=user; echo hi $NAME ${NAME}s") == "hi user users\n"


def test_prefix_assignment_is_temporary(shell, run) -> None:
    run("X=1 true")
    assert "X" not in shell.env


def test_command_substitution(run) -> None:
    assert run.out('n=$(echo 3); echo "n is $n"; echo $(seq 3)') == "n is 3\n1 2 3\n"


def test_substitution_status(run) -> None:
    assert run.out("x=$(false); echo $?") == "1\n"


def test_substitution_cd_does_not_leak(shell, run) -> None:
    run("x=$(cd /tmp; pwd)")
    assert shell.cwd == PROJECT


def test_substitution_errors_still_show(run) -> None:
    result = run("x=$(cat nope)")
    assert "cat: nope: No such file or directory" in result.stderr


def test_globbing_in_commands(vfs, run) -> None:
    for name in ("b.c", "a.c", "notes.txt"):
        vfs.write_file(f"{PROJECT}/{name}", "")
    assert run.out("echo *.c") == "a.c b.c\n"
    assert run.out("echo '*.c'") == "*.c\n"


def test_quoting_and_escapes(run) -> None:
    assert run.out("echo 'a  b' \"c  d\" e\\ \\ f") == "a  b c  d e  f\n"
    assert run.out("echo \"it's\" 'say \"hi\"'") == "it's say \"hi\"\n"
    assert run.out('echo "\\$HOME is $HOME"') == "$HOME is /home/user\n"


def test_bad_substitution_reports(run) -> None:
    result = run("echo ${X:-a}")
    assert result.exit_code == 1 and "bad substitution" in result.stderr


# --- pipelines ----------------------------------------------------------------------------

def test_pipeline_echo_grep_wc(run) -> None:
    assert run.out("echo -e 'corn\\nwheat\\ncorn cob' | grep corn | wc -l") == "2\n"


def test_pipeline_status_is_last_stage(run) -> None:
    assert run("false | true").exit_code == 0
    assert run("true | false").exit_code == 1


def test_pipeline_stderr_goes_to_terminal(run) -> None:
    result = run("cat nope | wc -l")
    assert result.stdout == "0\n" and "No such file" in result.stderr


# --- redirection --------------------------------------------------------------------------

def test_redirect_create_truncate_append(vfs, run) -> None:
    run("echo one > out.txt")
    run("echo two > out.txt")
    run("echo three >> out.txt")
    assert vfs.read_text(f"{PROJECT}/out.txt") == "two\nthree\n"


def test_redirect_input(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/in.txt", "b\na\n")
    assert run.out("sort < in.txt") == "a\nb\n"


def test_redirect_stderr(vfs, run) -> None:
    result = run("cat nope 2> err.txt")
    assert result.stderr == ""
    # Files get exactly what bash prints: no friendly line, no colour codes.
    assert vfs.read_text(f"{PROJECT}/err.txt") == "cat: nope: No such file or directory\n"


def test_redirect_both_with_2_and_1(vfs, run) -> None:
    run("cat nope > all.txt 2>&1; echo ok >> all.txt")
    assert vfs.read_text(f"{PROJECT}/all.txt") == "cat: nope: No such file or directory\nok\n"


def test_redirect_order_matters(vfs, run) -> None:
    result = run("cat nope 2>&1 > only_out.txt")
    # stderr was pointed at the terminal's stdout *before* stdout moved to the file.
    assert "No such file" in result.stdout
    assert vfs.read_text(f"{PROJECT}/only_out.txt") == ""


def test_redirect_ampersand(vfs, run) -> None:
    run("cat nope &> both.txt")
    assert "No such file" in vfs.read_text(f"{PROJECT}/both.txt")


def test_redirect_stdout_to_stderr(run) -> None:
    result = run("echo warn >&2")
    assert result.stdout == "" and result.stderr == "warn\n"


def test_redirect_missing_input_file(run, shell) -> None:
    result = run("sort < nope.txt")
    assert result.exit_code == 1
    assert "bash: nope.txt: No such file or directory" in result.stderr
    assert shell.last_diagnostics[0].code == "redirect_error"


def test_redirect_into_read_only(run) -> None:
    result = run("echo x > /usr/include/mine.h")
    assert "Permission denied" in result.stderr and result.exit_code == 1


def test_ambiguous_redirect(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/a.txt", "")
    vfs.write_file(f"{PROJECT}/b.txt", "")
    assert "ambiguous redirect" in run("echo x > *.txt").stderr


def test_redirect_on_compound_command(vfs, run) -> None:
    run("for i in 1 2; do echo $i; done > nums.txt")
    assert vfs.read_text(f"{PROJECT}/nums.txt") == "1\n2\n"


def test_redirect_alone_creates_file(vfs, run) -> None:
    run("> empty.txt")
    assert vfs.read_text(f"{PROJECT}/empty.txt") == ""


# --- control flow -------------------------------------------------------------------------

def test_if_with_test_brackets(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/log.txt", "x")
    script = "if [ -f log.txt ]; then echo file; elif [ -d log.txt ]; then echo dir; else echo none; fi"
    assert run.out(script) == "file\n"
    assert run.out(script.replace("log.txt", "nope")) == "none\n"


def test_for_loop_over_glob(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/a.c", "")
    vfs.write_file(f"{PROJECT}/b.c", "")
    assert run.out("for f in *.c; do echo src:$f; done") == "src:a.c\nsrc:b.c\n"


def test_if_inside_for(run) -> None:
    assert run.out("for i in 1 2 3; do if [ $i -ne 2 ]; then echo $i; fi; done") == "1\n3\n"


def test_while_false_never_runs(run) -> None:
    assert run("while false; do echo never; done").stdout == ""


def test_while_with_file_counter(vfs, run) -> None:
    script = ("echo x > c.txt\n"
              "while [ $(wc -l < c.txt) -lt 3 ]; do echo x >> c.txt; done\n"
              "wc -l < c.txt")
    assert run.out(script) == "3\n"


def test_until_loop(vfs, run) -> None:
    assert run.out("until [ -f done.txt ]; do touch done.txt; echo made; done") == "made\n"


def test_runaway_loop_is_stopped(run, shell) -> None:
    result = run("while true; do true; done")
    assert result.exit_code == 1 and "loop stopped" in result.stderr
    assert shell.last_diagnostics[-1].code == "loop_limit"


def test_nested_runaway_loop_stops_the_whole_line(run, shell) -> None:
    result = run("while true; do while true; do true; done; done; echo after")
    assert result.exit_code == 1
    assert result.stderr.count("loop stopped") == 1 and "after" not in result.stdout
    assert [d.code for d in shell.last_diagnostics] == ["loop_limit"]


def test_many_short_nested_loops_share_one_line_budget(run) -> None:
    words = " ".join(["x"] * 400)
    result = run(f"for a in {words}; do for b in {words}; do true; done; done")
    assert result.exit_code == 1 and "loops stopped after 50000 rounds" in result.stderr


def test_line_budget_resets_for_the_next_line(run) -> None:
    run("while true; do true; done")
    assert run.out("for i in 1 2 3; do echo $i; done") == "1\n2\n3\n"


def test_silent_loop_yields_heartbeats_so_ctrl_c_lands(shell) -> None:
    generator = shell.execute("while true; do while true; do true; done; done")
    for _ in range(50):
        event = next(generator)
        assert event == ShellOutput("stdout", "")
    generator.close()
    assert shell.last_status == 130


def test_subshell_keeps_cwd(shell, run) -> None:
    assert run.out("(cd /tmp; pwd); pwd") == f"/tmp\n{PROJECT}\n"


def test_syntax_error_reported(run, shell) -> None:
    result = run("if true; then echo")
    assert result.exit_code == 2
    assert "bash: syntax error: unexpected end of file" in result.stderr
    assert shell.last_diagnostics[0].code == "shell_syntax_error"


# --- errors and friendly lines ------------------------------------------------------------

def test_command_not_found(run, shell) -> None:
    result = run("foo")
    assert result.exit_code == 127
    assert result.stderr.startswith("bash: foo: command not found\n")
    assert "There is no program called 'foo'." in result.stderr
    assert shell.last_diagnostics[0].code == "command_not_found"


def test_command_not_found_suggests_dot_slash(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/harvest", "")
    assert "type './harvest'" in run("harvest").stderr


def test_command_not_found_suggests_close_name(run) -> None:
    assert "Did you mean 'grep'?" in run("grpe x").stderr


def test_friendly_line_is_dimmed(shell) -> None:
    result = shell.run("foo")
    lines = result.stderr.splitlines()
    assert lines[1].startswith(DIM)


def test_diagnostics_have_friendly_and_concept(shell) -> None:
    shell.run("cat nope")
    diagnostic = shell.last_diagnostics[0]
    assert diagnostic.code == "no_such_file" and diagnostic.friendly
    assert diagnostic.concept.value == "shell_files"
    shell.run("true")
    assert shell.last_diagnostics == []


# --- scripts ------------------------------------------------------------------------------

SCRIPT = "#!/bin/sh\necho script $1 $#\nexit 4\n"


def test_script_without_executable_bit(vfs, run, shell) -> None:
    vfs.write_file(f"{PROJECT}/run.sh", SCRIPT)
    result = run("./run.sh")
    assert result.exit_code == 126
    assert result.stderr.startswith("bash: ./run.sh: Permission denied\n")
    assert "Try `chmod +x run.sh`" in result.stderr
    assert shell.last_diagnostics[0].code == "permission_denied"


def test_script_with_executable_bit(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/run.sh", SCRIPT)
    result = run("chmod +x run.sh && ./run.sh corn")
    assert result.stdout == "script corn 1\n" and result.exit_code == 4


@pytest.mark.parametrize("runner", ["sh", "bash"])
def test_sh_runs_without_executable_bit(vfs, run, runner) -> None:
    vfs.write_file(f"{PROJECT}/run.sh", SCRIPT)
    assert run(f"{runner} run.sh a b").stdout == "script a 2\n"


def test_script_variables_do_not_leak_but_source_does(vfs, shell, run) -> None:
    vfs.write_file(f"{PROJECT}/env.sh", "CROP=corn\ncd /tmp\n")
    run("sh env.sh")
    assert "CROP" not in shell.env and shell.cwd == PROJECT
    run("source env.sh")
    assert shell.env["CROP"] == "corn" and shell.cwd == "/tmp"


def test_dot_is_source(vfs, shell, run) -> None:
    vfs.write_file(f"{PROJECT}/env.sh", "SEASON=spring\n")
    run(". ./env.sh")
    assert shell.env["SEASON"] == "spring"


def test_sh_dash_c(run) -> None:
    assert run.out("sh -c 'echo $0 $1' first") == "sh first\n"


def test_any_executable_text_file_is_a_script(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/tool", "echo no shebang needed\n")
    vfs.set_executable(f"{PROJECT}/tool")
    assert run.out("./tool") == "no shebang needed\n"


def test_path_includes_current_folder(vfs, run) -> None:
    make_program(vfs, f"{PROJECT}/prog", "from path")
    assert run.out("prog") == "from path\n"


def test_trap_exit_code_prints_crash_message(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    _code, events = drive(shell, "./a", lambda event: 139)
    assert ShellOutput("stderr", "Segmentation fault (core dumped)\n") in events
    assert shell.last_status == 139


def test_running_a_folder(run) -> None:
    result = run("./")
    assert "Is a directory" in result.stderr and result.exit_code == 126


def test_running_missing_file(run) -> None:
    result = run("./nope")
    assert result.stderr.startswith("bash: ./nope: No such file or directory")
    assert result.exit_code == 127


def test_bin_marker_runs_builtin(run) -> None:
    assert run.out("/bin/echo via marker") == "via marker\n"


def test_script_calling_itself_is_stopped(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/loop.sh", "sh loop.sh\n")
    assert "maximum nesting level" in run("sh loop.sh").stderr


# --- compiled programs --------------------------------------------------------------------

def test_run_program_event_wiring(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/prog")
    seen = []

    def host(event: RunProgram) -> int:
        seen.append(event)
        event.stdout.write("grown\n")
        return 0

    code, events = drive(shell, "./prog north 3", host)
    event = seen[0]
    assert code == 0
    assert event.argv == ["./prog", "north", "3"]
    assert event.executable.startswith(b"CVEXE1\n") and event.cwd == PROJECT
    assert isinstance(event.stdout, TerminalSink)
    assert events[-2] == ShellOutput("stdout", "grown\n")
    assert isinstance(events[-1], CommandFinished)


def test_run_program_exit_code_from_host(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    code, events = drive(shell, "./a; echo $?", lambda event: 5)
    assert ShellOutput("stdout", "5\n") in events and code == 0
    code, _ = drive(shell, "./a", lambda event: 5)
    assert code == 5 and shell.last_status == 5


def test_run_program_in_pipeline(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    seen = []

    def host(event: RunProgram) -> int:
        seen.append(event)
        event.stdout.write("x marks\nnothing\nx again\n")
        return 0

    code, events = drive(shell, "./a | grep x", host)
    assert isinstance(seen[0].stdout, StringSink)          # wired into the pipe
    output = "".join(e.text for e in events if isinstance(e, ShellOutput))
    assert output == "x marks\nx again\n" and code == 0


def test_run_program_reads_pipe_stdin(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    seen = []

    def host(event: RunProgram) -> int:
        seen.append(event.stdin.readline())
        return 0

    drive(shell, "echo seed | ./a", host)
    assert seen == ["seed\n"]


def test_run_program_with_output_redirect(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")

    def host(event: RunProgram) -> int:
        event.stdout.write("to file\n")
        event.stderr.write("to terminal\n")
        return 0

    _code, events = drive(shell, "./a > out.txt", host)
    assert vfs.read_text(f"{PROJECT}/out.txt") == "to file\n"
    assert ShellOutput("stderr", "to terminal\n") in events


def test_run_program_with_input_redirect(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    vfs.write_file(f"{PROJECT}/in.txt", "line\n")
    lines = []
    drive(shell, "./a < in.txt", lambda event: lines.append(event.stdin.readline()) or 0)
    assert lines == ["line\n"]


def test_headless_runner_fallback(vfs, run) -> None:
    make_program(vfs, f"{PROJECT}/a", "first\nexit=3")
    result = run("echo in | ./a")
    assert result.stdout == "first\nin:in\n" and result.exit_code == 3


def test_headless_runner_inside_script(vfs, run) -> None:
    make_program(vfs, f"{PROJECT}/a", "from program")
    vfs.write_file(f"{PROJECT}/go.sh", "./a | wc -l\n")
    assert run.out("sh go.sh") == "1\n"


def test_program_in_command_substitution(vfs, run) -> None:
    make_program(vfs, f"{PROJECT}/a", "42")
    assert run.out("echo answer=$(./a)") == "answer=42\n"


def test_no_runtime_available(vfs) -> None:
    make_program(vfs, f"{PROJECT}/a")
    shell = Shell(vfs)
    result = shell.run("./a")
    assert result.exit_code == 126 and "no runtime available" in result.stderr
    assert shell.last_diagnostics[0].code == "not_executable"


def test_program_without_executable_bit(vfs, run, shell) -> None:
    make_program(vfs, f"{PROJECT}/a", executable=False)
    result = run("./a")
    assert "Permission denied" in result.stderr and "chmod +x a" in result.stderr


def test_object_file_cannot_execute(vfs, run, shell) -> None:
    vfs.write_file(f"{PROJECT}/drone.o", OBJ_MAGIC + b"\x78\x9c junk")
    vfs.set_executable(f"{PROJECT}/drone.o")
    result = run("./drone.o")
    assert result.exit_code == 126
    assert result.stderr.startswith("bash: ./drone.o: cannot execute binary file: Exec format error\n")
    assert "That is an object file; link it with cc into a program first" in result.stderr
    assert shell.last_diagnostics[0].code == "cannot_execute_binary"


def test_other_binary_cannot_execute(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/blob", b"\x00\x01\x02")
    vfs.set_executable(f"{PROJECT}/blob")
    assert "cannot execute binary file" in run("./blob").stderr


def test_sh_refuses_compiled_program(vfs, run) -> None:
    make_program(vfs, f"{PROJECT}/a")
    result = run("sh a")
    assert "cannot execute binary file" in result.stderr and result.exit_code == 126


def test_run_program_headless_helper() -> None:
    out = StringSink()
    event = RunProgram(b"CVEXE1\nhi", ["./a"], StringSource(""), out, StringSink(), "/")
    assert run_program_headless(event, fake_runner) == 0 and out.text == "hi\n"
    err = StringSink()
    event = RunProgram(b"CVEXE1\n", ["./a"], StringSource(""), StringSink(), err, "/")
    assert run_program_headless(event, None) == 126 and "no runtime" in err.text


def test_output_order_before_program(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    _code, events = drive(shell, "echo before; ./a", lambda event: 0)
    assert events[0] == ShellOutput("stdout", "before\n")
    assert isinstance(events[1], RunProgram)


def test_terminal_sink_forward(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/a")
    live = []

    def host(event: RunProgram) -> int:
        event.stdout.forward = live.append      # a live host shows text as it comes
        event.stdout.write("now\n")
        return 0

    _code, events = drive(shell, "./a", host)
    assert live == ["now\n"]
    assert not any(isinstance(e, ShellOutput) for e in events)


# --- registry and misc --------------------------------------------------------------------

def test_register_command_replaces_placeholder(run) -> None:
    assert run("cc main.c").exit_code == 127
    assert "cc: the compiler is not installed in this workshop yet" in run("cc x.c").stderr

    def fake_cc(ctx: CommandContext, args: list[str]) -> int:
        ctx.out("compiled " + " ".join(args) + "\n")
        return 0

    register_command("cc", fake_cc, "# cc - test compiler\n")
    assert run.out("cc main.c") == "compiled main.c\n"


def test_clear_emits_clear_screen(shell) -> None:
    assert ClearScreen() in shell.run("echo a; clear").events


def test_history_records_lines(shell) -> None:
    shell.run("echo a")
    shell.run("   ")
    shell.run("ls")
    assert shell.history == ["echo a", "ls"]


def test_prompt(shell, run) -> None:
    assert shell.prompt == "user@pycc:~/project$ "
    run("cd /tmp")
    assert shell.prompt == "user@pycc:/tmp$ "


def test_constructor_matches_decisions_signature(vfs) -> None:
    toolchain = object()
    shell = Shell(vfs, PROJECT, toolchain, {"CROP": "corn"}, fake_runner)
    assert shell.toolchain is toolchain and shell.program_runner is fake_runner
    assert shell.env["CROP"] == "corn" and shell.env["PATH"] == "/bin:."
    assert shell.env["SHELL"] == "/bin/sh" and shell.env["PWD"] == PROJECT
