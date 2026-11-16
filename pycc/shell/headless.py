"""Running compiled programs: the headless host.

The shell never runs a program itself; it yields ``RunProgram`` and waits for a host
to answer with the exit code. This module is that host for the CLI and the tests
(an embedding application can provide its own, for example one that steps the
program a slice at a time). It does what a real operating system and C runtime do
between ``./prog`` and the shell prompt coming back:

1. load the executable (a damaged file: ``cannot execute binary file``, status 126);
2. link it against this toolchain's library (built with other device functions:
   status 127);
3. run the interpreter, answering its events: printing ``Output``, feeding
   ``InputRequest`` from stdin, performing ``DeviceCall`` on the device, if any;
4. when the program crashes, print the trap like a sanitizer would (``file:line:col:
   error: ...`` and the explanation line); the shell then adds ``Segmentation fault``;
5. summarise the run in a ``RunSummary``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pycc.core.concepts import Concept
from pycc.core.diagnostics import (EXIT_CODE_MESSAGES, Diagnostic, Phase, Severity,
                              exit_code_for_trap)
from pycc.core.events import (InputRequest, InputSource, Output, OutputSink, RunProgram,
                         RunSummary, RunWarning, StringSource, Trap, DeviceCall)
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.compiler import link_executable, load_executable
from pycc.interp.hostio import HostFiles, ProgramIO
from pycc.interp.linker import BadExecutable, Executable, LinkedProgram, program_hash
from pycc.interp.summary import SummaryBuilder
from pycc.interp.toolchain import Toolchain
from pycc.shell.cc import indented_friendly
from pycc.shell.executor import DEFAULT_CWD, Shell
from pycc.shell.hostfiles import VfsHostFiles
from pycc.shell.toolchain_commands import install

DEFAULT_BUDGET = 1_000_000
EXIT_BAD_EXECUTABLE = 126
EXIT_LINK_FAILED = 127
OTHER_MACHINE_NOTE = "this program was built by a toolchain with other device functions"

# Interpreter(program, builtins, budget=..., seed=..., io=...) per.
InterpreterFactory = Callable[..., Any]


@dataclass
class ProgramOutcome:
    """How a headless run ended: the exit status and the summary of what happened."""

    exit_code: int
    summary: RunSummary


@dataclass
class HostActionReply:
    """The reply to a ``DeviceCall``: the value, any warnings, and a trap (or None).

    Device function wrappers (``pycc.interp.devices``) only read these three
    attributes.
    """

    value: Any
    warnings: list[Diagnostic] = field(default_factory=list)
    trap: Diagnostic | None = None


def default_interpreter_factory() -> InterpreterFactory:
    """The interpreter class (imported late: it is the heaviest part of interp)."""
    from pycc.interp.interpreter import Interpreter

    return Interpreter


# =========================================================================== one run

def run_program_headless(event: RunProgram, toolchain: Toolchain,
                         files: HostFiles | None = None, device: Any = None,
                         budget: int = DEFAULT_BUDGET, seed: int = 0,
                         announce_exit: bool = True,
                         interpreter_factory: InterpreterFactory | None = None
                         ) -> ProgramOutcome:
    """Run the program of a ``RunProgram`` event to completion.

    ``files`` is what ``fopen`` & co. see (``VfsHostFiles(vfs)``; None means no files
    at all). ``device`` answers device calls through ``device.perform``; without one,
    every device call traps ``invalid_argument``. ``announce_exit`` adds the line a
    shell prints after a crash ("Segmentation fault (core dumped)"); the shell itself
    passes False because it prints that line on its own.
    """
    name = event.argv[0] if event.argv else "a.out"
    try:
        executable = load_executable(event.executable)
    except BadExecutable as error:
        return _refuse(event, error.diagnostic(name), EXIT_BAD_EXECUTABLE, prefix="bash: ")
    linked = link_executable(executable, toolchain.builtins, name)
    if not linked.ok or linked.program is None:
        return _link_failed(event, name, linked.diagnostics)
    return _ProgramHost(event, executable, linked.program, toolchain.builtins, files, device,
                        budget, seed, announce_exit,
                        interpreter_factory or default_interpreter_factory()).run()


def _refuse(event: RunProgram, diag: Diagnostic, status: int, prefix: str) -> ProgramOutcome:
    """The program could not even start: print why and summarise an empty run."""
    text = f"{prefix}{diag.message}\n" + _friendly(diag.friendly)
    event.stderr.write(text)
    return ProgramOutcome(status, _empty_summary(status, text))


def _link_failed(event: RunProgram, name: str, diagnostics: list[Diagnostic]) -> ProgramOutcome:
    """Linking before the run failed."""
    if any(diag.code == "bad_executable" for diag in diagnostics):
        bad = next(diag for diag in diagnostics if diag.code == "bad_executable")
        return _refuse(event, bad, EXIT_BAD_EXECUTABLE, prefix="bash: ")
    lines = [f"{name}: {diag.message}\n" + _friendly(diag.friendly) for diag in diagnostics]
    lines.append(f"{name}: note: {OTHER_MACHINE_NOTE}\n")
    text = "".join(lines)
    event.stderr.write(text)
    return ProgramOutcome(EXIT_LINK_FAILED, _empty_summary(EXIT_LINK_FAILED, text))


def _friendly(text: str) -> str:
    return indented_friendly(text) if text else ""


def _empty_summary(status: int, stderr: str) -> RunSummary:
    return RunSummary(exit_code=status, trap=None, warnings=[], statements=0, ticks=0,
                      stdout="", stderr=stderr)


class _ProgramHost:
    """Drives one interpreter run and answers its events."""

    def __init__(self, event: RunProgram, executable: Executable, program: LinkedProgram,
                 builtins: BuiltinRegistry, files: HostFiles | None, device: Any, budget: int,
                 seed: int, announce_exit: bool, factory: InterpreterFactory) -> None:
        self.event = event
        self.device = device
        self.files = files
        self.announce_exit = announce_exit
        self.builder = SummaryBuilder(program_hash(executable))
        argv = list(event.argv) or ["a.out"]
        io = ProgramIO(files=files, cwd=event.cwd, argv0=argv[0], env=dict(event.env))
        self.interpreter = factory(program, builtins, budget=budget, seed=seed, io=io)
        self.argv = argv
        self.sync_clock()

    def sync_clock(self) -> None:
        """Tell the program the device's time, so ``time()`` returns the current tick."""
        if self.device is not None and hasattr(self.interpreter, "device_tick"):
            self.interpreter.device_tick = _tick(self.device)

    def run(self) -> ProgramOutcome:
        """Run to the end; returns the outcome."""
        tick_before = _tick(self.device)
        generator = self.interpreter.run(self.argv)
        reply: Any = None
        while True:
            try:
                program_event = generator.send(reply)
            except StopIteration as stop:
                exit_code = int(stop.value if stop.value is not None else 0)
                break
            self.builder.feed(program_event)
            reply = self.answer(program_event)
        summary = self.builder.finish(exit_code, self.interpreter,
                                      _tick(self.device) - tick_before)
        return ProgramOutcome(exit_code, summary)

    def answer(self, program_event: Any) -> Any:
        """The reply to one interpreter event (None for most)."""
        if isinstance(program_event, Output):
            sink = self.event.stderr if program_event.stream == "stderr" else self.event.stdout
            sink.write(program_event.text)
        elif isinstance(program_event, InputRequest):
            return self.event.stdin.readline() if self.event.stdin is not None else None
        elif isinstance(program_event, DeviceCall):
            reply = device_reply(self.device, program_event)
            # The reply's warnings are not added to the summary here: the device
            # function's wrapper hands each one back as a RunWarning event, and
            # adding them twice would show every warning twice in the run report.
            self.sync_clock()
            return reply
        elif isinstance(program_event, RunWarning):
            self.report_warning(program_event.diagnostic)
        elif isinstance(program_event, Trap):
            self.report_trap(program_event.diagnostic)
        return None

    def report_warning(self, diag: Diagnostic) -> None:
        """Print a runtime warning on stderr, the way a sanitizer or valgrind would.

        A leak, an uninitialised read or a device warning does not stop the program, but the
        user should still see it in the terminal: ``leak.c:4:14: warning: 16 bytes
        in 1 blocks are definitely lost...`` and the friendly line under it.
        """
        text = diag.format_gcc() + "\n" + _friendly(diag.friendly)
        self.event.stderr.write(text)
        self.builder.stderr.append(text)

    def report_trap(self, diag: Diagnostic) -> None:
        """Print the crash on stderr, the way the terminal shows it."""
        text = diag.format_gcc() + "\n" + _friendly(diag.friendly)
        if self.announce_exit:
            message = EXIT_CODE_MESSAGES.get(exit_code_for_trap(diag.code))
            if message:
                text += message + "\n"
        self.event.stderr.write(text)
        # The summary's stderr shows what the user saw, crash report included.
        self.builder.stderr.append(text)


def _tick(device: Any) -> int:
    """The device clock (0 without a device)."""
    return int(getattr(device, "tick", 0) or 0) if device is not None else 0


# =========================================================================== device calls

def device_reply(device: Any, action: DeviceCall) -> HostActionReply:
    """Perform a device call and package the answer for the program.

    ``Device.perform`` returns an ``ActionResult`` and raises ``DeviceTrap`` for invalid
    use; both become one ``HostActionReply``. Warnings the device collected during the
    call travel with the reply.
    """
    if device is None:
        return HostActionReply(None, [], _no_device_trap(action.name))
    warnings_before = len(getattr(device, "warnings", None) or [])
    try:
        result = device.perform(action.name, list(action.args))
    except Exception as error:  # noqa: BLE001 - only device traps are turned into replies
        if not hasattr(error, "to_diagnostic"):
            raise
        return HostActionReply(None, _new_warnings(device, warnings_before),
                               error.to_diagnostic())
    warnings = list(getattr(result, "warnings", None) or []) or \
        _new_warnings(device, warnings_before)
    return HostActionReply(getattr(result, "value", result),
                           [_as_diagnostic(w) for w in warnings],
                           getattr(result, "trap", None))


def _new_warnings(device: Any, before: int) -> list[Diagnostic]:
    """Warnings the device appended to ``device.warnings`` since ``before``."""
    found = list(getattr(device, "warnings", None) or [])[before:]
    return [_as_diagnostic(warning) for warning in found]


def _as_diagnostic(warning: Any) -> Diagnostic:
    """Device warnings may be ``DeviceWarning`` objects with ``to_diagnostic``."""
    return warning.to_diagnostic() if hasattr(warning, "to_diagnostic") else warning


def _no_device_trap(name: str) -> Diagnostic:
    """Device calls need a device: without one they trap ``invalid_argument``."""
    return Diagnostic(
        Severity.ERROR, "invalid_argument",
        f"{name}(): there is no device connected to this program",
        f"{name}() is a device function, but this run has no device attached to answer "
        "it. Run the program from a host that provides the device.",
        None, Concept.DEVICE_API, Phase.RUNTIME)


# =========================================================================== the shell hook

FilesFactory = Callable[[], HostFiles | None]
DeviceFactory = Callable[[], Any]


class HeadlessRunner:
    """The shell's ``program_runner``: runs every program headless.

    Built by ``make_program_runner``. The shell calls it with ``(executable, argv,
    stdin, stdout, stderr, cwd)``; when it can, the shell calls ``run_event`` with the
    whole ``RunProgram`` instead, so the program also sees the exported environment.
    """

    def __init__(self, toolchain: Toolchain, files_factory: FilesFactory,
                 device_factory: DeviceFactory | None = None, budget: int = DEFAULT_BUDGET,
                 on_summary: Callable[[RunSummary], None] | None = None,
                 interpreter_factory: InterpreterFactory | None = None) -> None:
        """``files_factory``/``device_factory`` are called once per run, so each program
        gets fresh views (and, if the factory says so, a freshly reset device)."""
        self.toolchain = toolchain
        self.files_factory = files_factory
        self.device_factory = device_factory
        self.budget = budget
        self.on_summary = on_summary
        self.interpreter_factory = interpreter_factory
        self.outcomes: list[ProgramOutcome] = []

    def __call__(self, executable: bytes, argv: list[str], stdin: InputSource,
                 stdout: OutputSink, stderr: OutputSink, cwd: str) -> int:
        """The round-1 ``ProgramRunner`` signature (no environment)."""
        return self.run_event(RunProgram(executable, list(argv), stdin or StringSource(""),
                                         stdout, stderr, cwd))

    def run_event(self, event: RunProgram) -> int:
        """Run one program; returns its exit code and reports the summary."""
        device = self.device_factory() if self.device_factory is not None else None
        outcome = run_program_headless(event, self.toolchain, self.files_factory(), device,
                                       self.budget, announce_exit=False,
                                       interpreter_factory=self.interpreter_factory)
        self.outcomes.append(outcome)
        if self.on_summary is not None:
            self.on_summary(outcome.summary)
        return outcome.exit_code


def make_program_runner(toolchain: Toolchain, files_factory: FilesFactory,
                        device_factory: DeviceFactory | None = None,
                        budget: int = DEFAULT_BUDGET,
                        on_summary: Callable[[RunSummary], None] | None = None,
                        interpreter_factory: InterpreterFactory | None = None
                        ) -> HeadlessRunner:
    """The callable ``Shell(program_runner=...)`` expects, running programs headless."""
    return HeadlessRunner(toolchain, files_factory, device_factory, budget, on_summary,
                          interpreter_factory)


def make_headless_shell(vfs: Any, toolchain: Toolchain, cwd: str | None = None,
                        device_factory: DeviceFactory | None = None,
                        env: dict[str, str] | None = None,
                        budget: int = DEFAULT_BUDGET,
                        interpreter_factory: InterpreterFactory | None = None) -> Shell:
    """A complete headless terminal: a ``Shell`` with real cc/make and this runner.

    Each run's ``RunSummary`` is handed back through ``shell.record_run``, so
    ``CommandFinished.runs`` lists every program of the line.
    """
    shell = Shell(vfs, cwd or DEFAULT_CWD, env=env)
    install(shell, toolchain)
    shell.program_runner = make_program_runner(
        toolchain, lambda: VfsHostFiles(vfs), device_factory, budget,
        on_summary=shell.record_run, interpreter_factory=interpreter_factory)
    return shell


__all__ = ["DEFAULT_BUDGET", "HeadlessRunner", "HostActionReply", "ProgramOutcome",
           "default_interpreter_factory", "make_headless_shell", "make_program_runner",
           "run_program_headless", "device_reply"]
