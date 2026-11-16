"""Tests for the headless host (shell/headless.py) and VfsHostFiles (shell/hostfiles.py).

Programs are compiled by the real toolchain and run by the real interpreter, through
``run_program_headless`` directly and through a whole ``Shell`` (pipes, redirections,
``$?``, ``&&``/``||``). Machine calls go to a small fake device.
"""
from __future__ import annotations

from typing import Any

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity
from pycc.core.events import (InputRequest, Output, RunProgram, StringSink, StringSource,
                         DeviceCall)
from pycc.interp.compiler import build_project
from pycc.interp.hostio import EACCES, EISDIR, ENOENT, ENOTDIR, DictSourceFiles
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain
from pycc.shell import Shell
from pycc.shell.ansi import DIM
from pycc.shell.headless import (HeadlessRunner, HostActionReply, make_headless_shell,
                            make_program_runner, run_program_headless, device_reply)
from pycc.shell.hostfiles import VfsHostFiles
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import VirtualFS, standard_layout

HELLO_C = '#include <stdio.h>\nint main(void) {\n    printf("hello\\n");\n    return 0;\n}\n'
ECHO_C = ('#include <stdio.h>\n#include <stdlib.h>\nint main(int argc, char **argv) {\n'
          '    char line[64];\n    const char *crop = getenv("CROP");\n'
          '    printf("crop=%s\\n", crop ? crop : "none");\n'
          '    while (fgets(line, sizeof line, stdin)) printf("got %s", line);\n'
          '    return argc > 1 ? atoi(argv[1]) : 0;\n}\n')
CRASH_C = ('#include <stdio.h>\nint main(void) {\n    int *p = 0;\n    printf("before\\n");\n'
           '    return *p;\n}\n')
FILES_C = ('#include <stdio.h>\nint main(void) {\n    FILE *f = fopen("out.txt", "w");\n'
           '    if (f == NULL) { printf("no file\\n"); return 1; }\n'
           '    fprintf(f, "saved\\n");\n    fclose(f);\n    return 0;\n}\n')
LOOP_C = "int main(void) {\n    while (1) { }\n    return 0;\n}\n"
MOVER_C = ('#include <stdio.h>\n#include <machine.h>\nint main(void) {\n'
           '    int steps = 0;\n    while (move(1)) steps++;\n'
           '    printf("steps=%d x=%d\\n", steps, get_x());\n    move(9);\n'
           '    return 0;\n}\n')
MACHINE_H = "int move(int direction);\nint get_x(void);\n"


# --- a tiny machine API and a fake device -------------------------------------------------

def _reply_value(machine: Any, reply: Any) -> Any:
    """What a device function wrapper does with the host's reply."""
    for warning in reply.warnings:
        machine.warn(warning.code, warning.message, warning.friendly, warning.concept)
    if reply.trap is not None:
        machine.trap(reply.trap.code, reply.trap.message, reply.trap.friendly,
                     reply.trap.concept)
    return machine.make_int(int(reply.value or 0))


def _move(machine: Any, args: list) -> Any:
    reply = yield DeviceCall("move", [int(args[0].raw)])
    return _reply_value(machine, reply)


def _get_x(machine: Any, args: list) -> Any:
    reply = yield DeviceCall("get_x", [])
    return _reply_value(machine, reply)


class FakeDeviceTrap(Exception):
    """Shaped like device.rules.DeviceTrap: a code and ``to_diagnostic``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_diagnostic(self) -> Diagnostic:
        return Diagnostic(Severity.ERROR, self.code, self.message, "Use a real direction.",
                          None, Concept.DEVICE_API, Phase.RUNTIME)


class FakeResult:
    def __init__(self, value: int) -> None:
        self.value = value
        self.record = None


class FakeDevice:
    """A corridor three tiles long; each move costs one tick and bonks at the wall."""

    def __init__(self) -> None:
        self.tick = 0
        self.x = 0
        self.warnings: list[Diagnostic] = []
        self.calls: list[tuple[str, list]] = []

    def perform(self, name: str, args: list) -> FakeResult:
        self.calls.append((name, list(args)))
        if name == "get_x":
            return FakeResult(self.x)
        if args[0] not in (1, 2):
            raise FakeDeviceTrap("invalid_direction", f"move(): {args[0]} is not a direction")
        self.tick += 1
        if self.x >= 3:
            self.warnings.append(Diagnostic(Severity.WARNING, "bonk", "bonk", "A wall.",
                                            None, Concept.LOOP_BOUNDS, Phase.RUNTIME))
            return FakeResult(0)
        self.x += 1
        return FakeResult(1)


# --- fixtures ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    registry = make_registry()
    registry.register("move", "int move(int direction);", _move, header="machine.h")
    registry.register("get_x", "int get_x(void);", _get_x, header="machine.h")
    headers = system_headers()
    headers.add("machine.h", MACHINE_H)
    return Toolchain(registry, headers)


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    for name, text in {"hello.c": HELLO_C, "echo.c": ECHO_C, "crash.c": CRASH_C,
                       "files.c": FILES_C, "mover.c": MOVER_C}.items():
        vfs.write_file(f"{PROJECT}/{name}", text)
    vfs.write_file(f"{PROJECT}/in.txt", "one\ntwo\n")
    return vfs


@pytest.fixture
def device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def term(fs, toolchain, device) -> Shell:
    shell = make_headless_shell(fs, toolchain, device_factory=lambda: device)
    for program in ("hello", "echo", "crash", "files", "mover"):
        assert Runner(shell)(f"cc {program}.c -o {program}").exit_code == 0
    return shell


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


def compile_bytes(toolchain: Toolchain, text: str, name: str = "prog.c") -> bytes:
    build = build_project(DictSourceFiles({f"/p/{name}": text}), [name], "prog", toolchain,
                          cwd="/p")
    assert build.ok, [d.format_gcc() for d in build.diagnostics]
    return build.executable.to_bytes()


def event_for(data: bytes, stdin: str = "", env: dict[str, str] | None = None,
              argv: list[str] | None = None) -> RunProgram:
    return RunProgram(data, argv or ["./prog"], StringSource(stdin), StringSink(),
                      StringSink(), PROJECT, dict(env or {}))


# --- run_program_headless ----------------------------------------------------------------

def test_hello_runs_and_is_summarised(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, HELLO_C))
    outcome = run_program_headless(event, toolchain)
    assert outcome.exit_code == 0 and event.stdout.text == "hello\n"
    summary = outcome.summary
    assert summary.ok and summary.stdout == "hello\n" and summary.statements > 0
    assert summary.program_hash and summary.ticks == 0


def test_stdin_lines_are_answered_from_the_event(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, ECHO_C), stdin="a\nb\n")
    run_program_headless(event, toolchain)
    assert event.stdout.text == "crop=none\ngot a\ngot b\n"


def test_environment_reaches_getenv(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, ECHO_C), env={"CROP": "corn"})
    run_program_headless(event, toolchain)
    assert event.stdout.text.startswith("crop=corn\n")


def test_argv_and_exit_code(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, ECHO_C), argv=["./prog", "7"])
    assert run_program_headless(event, toolchain).exit_code == 7


def test_trap_prints_diagnostic_friendly_and_exit_message(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, CRASH_C, "crash.c"))
    outcome = run_program_headless(event, toolchain)
    assert outcome.exit_code == 139
    trap = outcome.summary.trap
    assert trap.code == "null_dereference" and trap.location.file == "crash.c"
    assert event.stderr.text == (f"{trap.format_gcc()}\n{DIM}  {trap.friendly}\x1b[0m\n"
                                 "Segmentation fault (core dumped)\n")
    assert outcome.summary.stdout == "before\n"
    assert "Segmentation fault" in outcome.summary.stderr


def test_shell_hosts_do_not_announce_the_exit_twice(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, CRASH_C))
    run_program_headless(event, toolchain, announce_exit=False)
    assert "Segmentation fault" not in event.stderr.text


def test_budget_exhaustion_is_killed(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, LOOP_C))
    outcome = run_program_headless(event, toolchain, budget=500)
    assert outcome.exit_code == 137 and outcome.summary.budget_hit
    assert event.stderr.text.endswith("Killed\n")


def test_bad_executable_is_refused_with_126(toolchain) -> None:
    event = event_for(b"CVEXE1\n\x09garbage")
    outcome = run_program_headless(event, toolchain)
    assert outcome.exit_code == 126
    assert event.stderr.text.splitlines()[0] == ("bash: ./prog: cannot execute binary file: "
                                                 "Exec format error")
    assert outcome.summary.exit_code == 126


def test_damaged_executable_body_is_refused(toolchain) -> None:
    good = compile_bytes(toolchain, HELLO_C)
    event = event_for(good[:12])
    assert run_program_headless(event, toolchain).exit_code == 126


def test_program_built_for_another_machine_fails_to_link(toolchain) -> None:
    other = Toolchain(toolchain.builtins.copy(), toolchain.headers.copy())
    other.builtins.register("beep", "int beep(void);", lambda machine, args: 1,
                            header="machine.h")
    other.headers.add("beep.h", "int beep(void);\n")
    data = compile_bytes(other, "#include <beep.h>\nint main(void) { return beep(); }\n")
    plain = Toolchain(make_registry(), system_headers())
    event = event_for(data)
    outcome = run_program_headless(event, plain)
    assert outcome.exit_code == 127
    lines = event.stderr.text.splitlines()
    assert lines[0].startswith("./prog: ld: ") and "undefined reference to 'beep'" in lines[0]
    assert lines[-1] == "./prog: note: this program was built by a toolchain with other device functions"


def test_files_go_through_host_files(toolchain) -> None:
    vfs = standard_layout("project")
    event = event_for(compile_bytes(toolchain, FILES_C))
    assert run_program_headless(event, toolchain, VfsHostFiles(vfs)).exit_code == 0
    assert vfs.read_text(f"{PROJECT}/out.txt") == "saved\n"


def test_without_files_fopen_fails(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, FILES_C))
    assert run_program_headless(event, toolchain, None).exit_code == 1
    assert event.stdout.text == "no file\n"


# --- machine calls -----------------------------------------------------------------------

def test_device_calls_are_performed(toolchain, device) -> None:
    event = event_for(compile_bytes(toolchain, MOVER_C))
    outcome = run_program_headless(event, toolchain, device=device)
    assert event.stdout.text == "steps=3 x=3\n"
    assert outcome.exit_code == 134                       # move(9) traps
    assert outcome.summary.trap.code == "invalid_direction"
    assert outcome.summary.ticks == 4                      # four moves took time
    assert outcome.summary.constructs["device_call"] == 6
    assert any(w.code == "bonk" for w in outcome.summary.warnings)


def test_device_call_without_a_device_traps_invalid_argument(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, MOVER_C))
    outcome = run_program_headless(event, toolchain)
    assert outcome.exit_code == 134
    assert outcome.summary.trap.code == "invalid_argument"
    assert "no device connected" in event.stderr.text


def test_device_reply_shapes() -> None:
    device = FakeDevice()
    ok = device_reply(device, DeviceCall("move", [1]))
    assert ok == HostActionReply(1, [], None)
    trapped = device_reply(device, DeviceCall("move", [5]))
    assert trapped.value is None and trapped.trap.code == "invalid_direction"
    assert device_reply(None, DeviceCall("move", [1])).trap.code == "invalid_argument"


def test_device_reply_passes_result_warnings_and_trap() -> None:
    warning = Diagnostic(Severity.WARNING, "unripe_harvest", "unripe", "Wait.", None,
                         Concept.CONDITIONALS, Phase.RUNTIME)

    class Result:
        value, warnings, trap = 0, [warning], None

    class Device:
        def perform(self, name: str, args: list) -> Result:
            return Result()

    reply = device_reply(Device(), DeviceCall("harvest", []))
    assert reply.warnings == [warning] and reply.value == 0


def test_device_reply_reraises_host_bugs() -> None:
    class Broken:
        def perform(self, name: str, args: list) -> None:
            raise KeyError(name)

    with pytest.raises(KeyError):
        device_reply(Broken(), DeviceCall("nope", []))


class ScriptedInterpreter:
    """Stands in for the interpreter to check the host's side of the protocol."""

    instances: list["ScriptedInterpreter"] = []

    def __init__(self, program, builtins, budget, seed, io) -> None:
        self.io = io
        self.budget = budget
        self.replies: list[Any] = []
        self.statements_executed = 2
        ScriptedInterpreter.instances.append(self)

    def run(self, argv):
        self.replies.append((yield InputRequest("name? ")))
        self.replies.append((yield InputRequest("again? ")))
        self.replies.append((yield Output("stderr", "to stderr\n")))
        return 5


def test_host_protocol_with_a_scripted_interpreter(toolchain) -> None:
    event = event_for(compile_bytes(toolchain, HELLO_C), stdin="user\n",
                      argv=["./prog", "x"])
    outcome = run_program_headless(event, toolchain, budget=99,
                                   interpreter_factory=ScriptedInterpreter)
    interpreter = ScriptedInterpreter.instances[-1]
    assert interpreter.replies == ["user\n", None, None]
    assert interpreter.budget == 99 and interpreter.io.argv0 == "./prog"
    assert interpreter.io.cwd == PROJECT
    assert outcome.exit_code == 5 and event.stderr.text == "to stderr\n"
    assert outcome.summary.statements == 2


# --- the shell hook ----------------------------------------------------------------------

def test_make_program_runner_accepts_the_round_one_signature(toolchain) -> None:
    runner = make_program_runner(toolchain, lambda: None)
    out, err = StringSink(), StringSink()
    status = runner(compile_bytes(toolchain, HELLO_C), ["./hello"], StringSource(""), out,
                    err, "/")
    assert status == 0 and out.text == "hello\n"
    assert isinstance(runner, HeadlessRunner) and runner.outcomes[0].summary.ok


def test_runner_reports_summaries(toolchain) -> None:
    seen = []
    runner = make_program_runner(toolchain, lambda: None, on_summary=seen.append)
    runner.run_event(event_for(compile_bytes(toolchain, HELLO_C)))
    assert len(seen) == 1 and seen[0].stdout == "hello\n"


def test_runner_gets_a_fresh_world_per_run(toolchain) -> None:
    devices: list[FakeDevice] = []

    def factory() -> FakeDevice:
        devices.append(FakeDevice())
        return devices[-1]

    runner = make_program_runner(toolchain, lambda: None, device_factory=factory)
    data = compile_bytes(toolchain, MOVER_C)
    runner.run_event(event_for(data))
    runner.run_event(event_for(data))
    assert len(devices) == 2 and all(w.x == 3 for w in devices)


def test_plain_shell_with_runner(fs, toolchain) -> None:
    shell = Shell(fs, program_runner=make_program_runner(toolchain, lambda: VfsHostFiles(fs)))
    fs.write_file(f"{PROJECT}/hello", compile_bytes(toolchain, HELLO_C))
    fs.set_executable(f"{PROJECT}/hello")
    assert Runner(shell)("./hello").stdout == "hello\n"


# --- through the whole shell -------------------------------------------------------------

def test_run_compiled_program(run) -> None:
    result = run("./hello")
    assert (result.exit_code, result.stdout, result.stderr) == (0, "hello\n", "")


def test_program_output_into_a_pipe(run) -> None:
    assert run("./echo < in.txt | grep two").stdout == "got two\n"


def test_program_output_redirected_to_a_file(run, fs) -> None:
    assert run("./hello > out.txt").stdout == ""
    assert fs.read_text(f"{PROJECT}/out.txt") == "hello\n"


def test_program_reads_a_redirected_file(run) -> None:
    assert run("./echo < in.txt").stdout == "crop=none\ngot one\ngot two\n"


def test_program_reads_stdin_from_a_pipe(run) -> None:
    assert run("echo hi | ./echo").stdout == "crop=none\ngot hi\n"


def test_program_between_two_pipes(run) -> None:
    assert run("cat in.txt | ./echo | wc -l").stdout.strip() == "3"


def test_exit_status_after_a_trap(run) -> None:
    result = run("./crash; echo status=$?")
    assert result.stdout == "before\nstatus=139\n"
    assert "Segmentation fault (core dumped)" in result.stderr
    assert result.stderr.count("Segmentation fault") == 1
    assert "crash.c:5" in result.stderr


def test_exit_codes_through_and_and_or(run) -> None:
    assert run("./hello && echo yes").stdout == "hello\nyes\n"
    assert run("./crash && echo yes").stdout == "before\n"
    assert run("./crash || echo rescued").stdout == "before\nrescued\n"
    assert run("./echo 3 < /dev/null; echo $?").stdout == "crop=none\n3\n"


def test_exported_variables_reach_the_program(run) -> None:
    assert run("export CROP=wheat; ./echo < /dev/null").stdout == "crop=wheat\n"
    assert run("CROP=corn ./echo < /dev/null").stdout == "crop=corn\n"


def test_unexported_variables_do_not(run) -> None:
    assert run("SECRET=1; CROP=x; ./echo < /dev/null").stdout == "crop=none\n"


def test_program_writes_files_in_the_vfs(run, fs) -> None:
    assert run("./files && cat out.txt").stdout == "saved\n"


def test_world_program_through_the_shell(run, device) -> None:
    result = run("./mover")
    assert result.stdout == "steps=3 x=3\n" and result.exit_code == 134
    assert "Aborted (core dumped)" in result.stderr
    assert device.calls[0] == ("move", [1])


def test_command_finished_lists_the_runs(run) -> None:
    finished = run("./hello; ./crash").finished
    assert [summary.exit_code for summary in finished.runs] == [0, 139]
    assert finished.runs[1].trap.code == "null_dereference"


def test_script_runs_programs(run, fs) -> None:
    fs.write_file(f"{PROJECT}/go.sh", "./hello\n./hello | wc -l\n")
    assert run("sh go.sh").stdout.split() == ["hello", "1"]


def test_command_substitution_runs_programs(run) -> None:
    assert run('echo "said: $(./hello)"').stdout == "said: hello\n"


def test_cannot_execute_a_damaged_program(run, fs) -> None:
    fs.write_file(f"{PROJECT}/broken", b"CVEXE1\n\x09junk")
    fs.set_executable(f"{PROJECT}/broken")
    result = run("./broken; echo $?")
    assert result.stdout == "126\n"
    assert "cannot execute binary file" in result.stderr


# --- VfsHostFiles ------------------------------------------------------------------------

@pytest.fixture
def host() -> VfsHostFiles:
    return VfsHostFiles(standard_layout("project", {"stdio.h": "x\n"}))


def test_host_files_read_and_write(host) -> None:
    assert host.write(f"{PROJECT}/a.txt", b"one\n", False) == 0
    assert host.write(f"{PROJECT}/a.txt", b"two\n", True) == 0
    assert host.read(f"{PROJECT}/a.txt") == b"one\ntwo\n"
    assert host.exists(f"{PROJECT}/a.txt") and not host.is_dir(f"{PROJECT}/a.txt")


def test_host_files_errors_are_negative_errno(host) -> None:
    assert host.read(f"{PROJECT}/missing") == -ENOENT
    assert host.read(PROJECT) == -EISDIR
    assert host.write("/usr/include/stdio.h", b"x", False) == -EACCES
    assert host.write(f"{PROJECT}/nodir/a.txt", b"x", False) == -ENOENT
    host.write(f"{PROJECT}/f", b"", False)
    assert host.read(f"{PROJECT}/f/x") == -ENOTDIR


def test_host_files_remove(host) -> None:
    host.write(f"{PROJECT}/a", b"", False)
    assert host.remove(f"{PROJECT}/a") == 0 and not host.exists(f"{PROJECT}/a")
    assert host.remove(f"{PROJECT}/a") == -ENOENT
    assert host.remove("/usr/include/stdio.h") == -EACCES


def test_host_files_rename(host) -> None:
    host.write(f"{PROJECT}/a", b"A", False)
    host.write(f"{PROJECT}/b", b"B", False)
    assert host.rename(f"{PROJECT}/a", f"{PROJECT}/b") == 0
    assert host.read(f"{PROJECT}/b") == b"A" and not host.exists(f"{PROJECT}/a")
    assert host.rename(f"{PROJECT}/missing", f"{PROJECT}/c") == -ENOENT
    assert host.rename(f"{PROJECT}/b", "/tmp") == -EISDIR
