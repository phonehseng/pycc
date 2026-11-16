"""The shell executor: runs parsed command lines against the virtual filesystem.

``Shell.execute(line)`` is a generator, like the C interpreter. It yields
``ShellOutput`` for text meant for the terminal, ``ClearScreen`` for ``clear``,
``RunProgram`` whenever a compiled program must run (also inside pipelines, scripts
and ``$(...)``) and finally one ``CommandFinished`` summary. The host answers
``RunProgram`` by sending back the exit code. A host that sends ``None`` instead lets
the shell run the program through the injected ``program_runner`` (tests and
``Shell.run`` rely on that); command substitution always does.

Terminal output is queued and released (``_flush``) after every command and before
every ``RunProgram``, so text appears in the order it was produced even though
builtins are ordinary functions.

Pipelines run their stages one after another, each writing into a ``StringSink`` that
becomes the next stage's ``StringSource``. Real shells run stages at the same time;
for the short, finite programs pycc runs the result is the same and it keeps the
executor single-threaded.
"""
from __future__ import annotations

import difflib
import inspect
import posixpath
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from pycc.core.concepts import Concept
from pycc.core.diagnostics import EXIT_CODE_MESSAGES, Diagnostic
from pycc.core.events import (CommandFinished, InputSource, OutputSink, RunProgram, RunSummary,
                         ShellEvent, ShellOutput, StringSink, StringSource)
from pycc.shell import builtins as _builtins  # noqa: F401 - importing registers the commands
from pycc.shell import completion, constructs
from pycc.shell.commands import (IO, CommandContext, CommandError, CommandSpec, ExitRequest,
                            LoopControl,
                            command_names, lookup, report_error, vfs_concept, vfs_friendly)
from pycc.shell.expand import ExpansionError, expand_single, expand_words
from pycc.shell.lexer import CommandSub
from pycc.shell.parser import (AndOr, Command, CommandList, ForCommand, IfCommand, ParseError,
                          Pipeline, Redirect, SimpleCommand, Subshell, WhileCommand, parse)
from pycc.shell.streams import RedirectError, TerminalSink, TerminalSource, apply_redirects
from pycc.vfs.errors import VfsError
from pycc.vfs.filesystem import BUILTIN_MARKER, HOME, VirtualFS

DEFAULT_CWD = "/home/user/project"
EXE_MAGIC = b"CVEXE1\n"
OBJ_MAGIC = b"CVOBJ1\n"
# Guards that keep a runaway script from freezing a single-threaded host.
LOOP_LIMIT = 10_000           # rounds of one loop
LINE_ROUND_LIMIT = 50_000     # rounds of all loops of one typed line, at any depth
# Exceptions a command raises on purpose for the executor to handle (exit, break...).
SHELL_SIGNALS = (ExitRequest, LoopControl, ExpansionError, RedirectError, ParseError)
HEARTBEAT_ROUNDS = 16         # a loop yields an empty event this often, so the host can
                              # end its frame (and see Ctrl+C) even when nothing is printed
NESTING_LIMIT = 40
INTERRUPTED_STATUS = 130

ProgramRunner = Callable[[bytes, list[str], InputSource, OutputSink, OutputSink, str], int]
ShellGen = Generator[ShellEvent, Any, int]


@dataclass
class ShellResult:
    """What ``Shell.run`` returns: status, the terminal text and every event."""

    exit_code: int
    stdout: str
    stderr: str
    events: list[ShellEvent] = field(default_factory=list)

    @property
    def finished(self) -> CommandFinished | None:
        """The ``CommandFinished`` summary of the line."""
        return next((e for e in self.events if isinstance(e, CommandFinished)), None)


def run_program_headless(event: RunProgram, runner: ProgramRunner | None) -> int:
    """The default host for ``RunProgram``: call ``runner`` with the event's wiring.

    A runner with a ``run_event`` method (``shell.headless.HeadlessRunner``) gets the
    whole event, so the program also sees the exported environment.
    """
    if runner is None:
        event.stderr.write(f"bash: {event.argv[0]}: no runtime available\n")
        return 126
    run_event = getattr(runner, "run_event", None)
    if callable(run_event):
        return int(run_event(event))
    return int(runner(event.executable, event.argv, event.stdin, event.stdout,
                      event.stderr, event.cwd))


def default_environment(cwd: str) -> dict[str, str]:
    """The variables a fresh shell starts with."""
    return {"HOME": HOME, "USER": "user", "PATH": "/bin:.", "SHELL": "/bin/sh",
            "PWD": cwd, "TERM": "pycc"}


@dataclass
class _CommandRecord:
    """What one top-level ``execute`` gathers for its ``CommandFinished``.

    Child shells (scripts, sub-shells) share the record of the shell that started
    them, so a ``cc`` run inside ``build.sh`` still counts for the typed line.
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)
    runs: list[RunSummary] = field(default_factory=list)
    constructs: dict[str, int] = field(default_factory=dict)
    build: object | None = None
    loop_rounds: int = 0
    loops_stopped: bool = False   # a runaway loop was stopped: the rest of the line is skipped


class Shell:
    """A bash-like shell over a ``VirtualFS``."""

    def __init__(self, vfs: VirtualFS, cwd: str = DEFAULT_CWD, toolchain: object | None = None,
                 env: dict[str, str] | None = None,
                 program_runner: ProgramRunner | None = None) -> None:
        """Create a shell in ``cwd``.

        ``toolchain`` is kept for the ``cc``/``make`` commands (``None`` keeps the
        placeholders); ``program_runner`` runs compiled programs when the host does not.
        """
        self.vfs = vfs
        self.cwd = vfs.resolve(cwd)
        self.toolchain = toolchain
        self.env: dict[str, str] = default_environment(self.cwd)
        self.env.update(env or {})
        self.exported: set[str] = set(self.env)
        self.program_runner = program_runner
        self.history: list[str] = []
        self.last_status = 0
        self.positional: list[str] = []
        self.script_name = "bash"
        self.exited = False
        self.terminal_input = TerminalSource()
        self.depth = 0
        self.last_build: object | None = None
        self.loop_depth = 0            # loops around the running command (break/continue)
        self._job_stderr: OutputSink | None = None   # where bash reports a crashed job
        self._record = _CommandRecord()
        self._pending: list[ShellEvent] = []

    # --- public interface -------------------------------------------------------------

    def execute(self, line: str) -> ShellGen:
        """Run one line typed at the prompt; ends by yielding ``CommandFinished``.

        Closing the generator (Ctrl+C) stops whatever runs and leaves status 130.
        """
        self._record = _CommandRecord()
        build_before = self.last_build
        if line.strip():
            self.history.append(line.rstrip("\n"))
        try:
            status = yield from self.run_source(line, self.terminal_io())
        except GeneratorExit:
            self.last_status = INTERRUPTED_STATUS
            self._pending.clear()
            raise
        if self._record.build is None and self.last_build is not build_before:
            self._record.build = self.last_build      # a command that only set the attribute
        if self._record.build is not None:
            self.last_build = self._record.build
        yield CommandFinished(line, status, list(self._record.diagnostics), self._record.build,
                              list(self._record.runs), dict(self._record.constructs))
        return status

    def execute_subcommand(self, line: str, io: IO | None = None) -> ShellGen:
        """Run ``line`` inside the current command (``make`` recipes): no history entry
        and no ``CommandFinished``; ``exit`` ends only this line."""
        return (yield from self.run_source(line, io or self.terminal_io()))

    def run(self, line: str) -> ShellResult:
        """Run a line headless and collect its output (tests, graders, scripts)."""
        events: list[ShellEvent] = []
        generator = self.execute(line)
        while True:
            try:
                # Sending None to RunProgram makes the shell use the headless runner.
                events.append(generator.send(None))
            except StopIteration as stop:
                exit_code = stop.value
                break
        texts = {"stdout": "", "stderr": ""}
        for event in events:
            if isinstance(event, ShellOutput):
                texts[event.stream] += event.text
        return ShellResult(exit_code, texts["stdout"], texts["stderr"], events)

    def complete(self, line: str, cursor: int | None = None) -> list[str]:
        """Tab-completion candidates for the word under the cursor."""
        return completion.complete(self, line, len(line) if cursor is None else cursor)

    @property
    def last_diagnostics(self) -> list[Diagnostic]:
        """Diagnostics of the failed commands in the last line."""
        return self._record.diagnostics

    @property
    def recorded_runs(self) -> list[RunSummary]:
        """The program runs of the line so far (``time`` reads them)."""
        return self._record.runs

    def record_run(self, summary: RunSummary) -> None:
        """Hosts hand back what happened in each program run of this line."""
        self._record.runs.append(summary)

    def record_build(self, result: object) -> None:
        """``cc``/``make`` report their ``BuildResult`` here (also from inside scripts)."""
        self.last_build = result
        self._record.build = result

    def count_construct(self, key: str, amount: int = 1) -> None:
        """Count a shell construct (``shell/constructs.py``) used by this line."""
        self._record.constructs[key] = self._record.constructs.get(key, 0) + amount

    @property
    def prompt(self) -> str:
        """The prompt text, e.g. ``user@pycc:~/project$ ``."""
        home = self.env.get("HOME", HOME)
        inside_home = self.cwd == home or self.cwd.startswith(home + "/")
        shown = "~" + self.cwd[len(home):] if inside_home else self.cwd
        return f"user@pycc:{shown}$ "

    def terminal_io(self) -> IO:
        """Streams connected to the user's terminal."""
        return IO(self.terminal_input, TerminalSink("stdout", self._pending),
                  TerminalSink("stderr", self._pending))

    def emit(self, event: ShellEvent) -> None:
        """Queue a non-text event (``ClearScreen``) in order with the output."""
        self._pending.append(event)

    def change_directory(self, path: str) -> None:
        """Make ``path`` the working directory, keeping ``PWD``/``OLDPWD`` right."""
        self.env["OLDPWD"] = self.cwd
        self.cwd = path
        self.env["PWD"] = path

    # --- expansion context (used by shell.expand) -----------------------------------------

    def lookup(self, name: str) -> str | None:
        """Value of a variable or special parameter (``$?``, ``$#``, ``$0``...)."""
        if name == "?":
            self.count_construct(constructs.EXIT_CODE_TEST)
            return str(self.last_status)
        if name == "#":
            return str(len(self.positional))
        if name == "$":
            return "4242"
        if name == "0":
            return self.script_name
        if name.isdigit():
            index = int(name) - 1
            return self.positional[index] if index < len(self.positional) else None
        return self.env.get(name)

    def positional_args(self) -> list[str]:
        """The script's arguments ``$1 $2 ...``."""
        return list(self.positional)

    def substitute(self, source: str) -> str:
        """``$(...)``: run ``source`` in a sub-shell and capture its stdout.

        Expansion is not a generator, so programs inside run through the headless
        runner; their errors are passed on to the terminal queue.
        """
        child = self.spawn(self.positional, self.script_name, share_output=False)
        sink = StringSink()
        generator = child.run_source(source, IO(StringSource(""), sink, child.terminal_io().stderr))
        while True:
            try:
                event = generator.send(None)
            except StopIteration as stop:
                self.last_status = stop.value
                return sink.text
            if isinstance(event, ShellOutput) and not event.text:
                continue                    # a loop heartbeat: nothing to show
            if not isinstance(event, RunProgram):
                self._pending.append(event)

    # --- running sources and trees --------------------------------------------------------

    def run_source(self, source: str, io: IO, catch_exit: bool = True) -> ShellGen:
        """Parse and run shell source with the given streams."""
        try:
            tree = parse(source)
        except ParseError as error:
            self.report(io, "shell_syntax_error", f"bash: {error.message}", _syntax_friendly(error))
            yield from self._flush()
            self.last_status = 2
            return 2
        try:
            status = yield from self.run_list(tree, io)
        except ExitRequest as request:
            if not catch_exit:
                raise
            status = request.code
            self.exited = True
        yield from self._flush()
        self.last_status = status
        return status

    def run_list(self, node: CommandList, io: IO) -> ShellGen:
        """Run commands in sequence; the status is the last one's."""
        status = 0
        for item in node.items:
            if self._record.loops_stopped:
                break
            status = yield from self._run_and_or(item, io)
            self.last_status = status
        return status

    def _run_and_or(self, node: AndOr, io: IO) -> ShellGen:
        status = yield from self._run_pipeline(node.first, io)
        for operator, pipeline in node.rest:
            self.last_status = status
            self.count_construct(constructs.EXIT_CODE_TEST)
            if (operator == "&&") == (status == 0):
                status = yield from self._run_pipeline(pipeline, io)
        return status

    def _run_pipeline(self, node: Pipeline, io: IO) -> ShellGen:
        if len(node.commands) > 1:
            self.count_construct(constructs.PIPELINE)
        stdin = io.stdin
        status = 0
        for index, command in enumerate(node.commands):
            last = index == len(node.commands) - 1
            sink: OutputSink = io.stdout if last else StringSink()
            stage_io = IO(stdin, sink, io.stderr)
            if len(node.commands) > 1:
                status = yield from self._run_stage(command, stage_io)
            else:
                status = yield from self._run_command(command, stage_io)
            if not last:
                stdin = StringSource(sink.text)
        if node.negated:
            status = 0 if status else 1
        self.last_status = status
        return status

    def _run_stage(self, node: Command, io: IO) -> ShellGen:
        """One stage of a real pipeline: like bash, each stage is its own sub-shell,
        so ``exit 3 | cat`` ends only that stage instead of the whole shell."""
        try:
            return (yield from self._run_command(node, io))
        except ExitRequest as request:
            return request.code

    def _run_command(self, node: Command, io: IO) -> ShellGen:
        if isinstance(node, SimpleCommand):
            return (yield from self._run_simple(node, io))
        redirected = self._redirect(node.redirects, io)
        if redirected is None:
            return 1
        if isinstance(node, IfCommand):
            return (yield from self._run_if(node, redirected))
        if isinstance(node, ForCommand):
            return (yield from self._run_for(node, redirected))
        if isinstance(node, WhileCommand):
            return (yield from self._run_while(node, redirected))
        return (yield from self._run_subshell(node, redirected))

    def _run_if(self, node: IfCommand, io: IO) -> ShellGen:
        self.count_construct(constructs.EXIT_CODE_TEST)
        for condition, body in node.branches:
            if (yield from self.run_list(condition, io)) == 0:
                return (yield from self.run_list(body, io))
        if node.else_body is not None:
            return (yield from self.run_list(node.else_body, io))
        return 0

    def _run_for(self, node: ForCommand, io: IO) -> ShellGen:
        try:
            values = self.positional_args() if node.words is None else expand_words(node.words,
                                          self)
        except ExpansionError as error:
            return self.report(io, "shell_syntax_error", f"bash: {error.message}", _BAD_SUB)
        status = 0
        for count, value in enumerate(values):
            if count >= LOOP_LIMIT:
                return self._loop_limit(io)
            if (yield from self._next_round(io)):
                return status or 1
            self.env[node.name] = value
            status, stop = yield from self._run_loop_body(node.body, io)
            if stop:
                return status
        return status

    def _run_while(self, node: WhileCommand, io: IO) -> ShellGen:
        status = 0
        rounds = 0
        while True:
            condition = yield from self.run_list(node.condition, io)
            if (condition == 0) == node.until:
                return status
            rounds += 1
            if rounds > LOOP_LIMIT:
                return self._loop_limit(io)
            if (yield from self._next_round(io)):
                return status or 1
            status, stop = yield from self._run_loop_body(node.body, io)
            if stop:
                return status

    def _run_loop_body(self, body: CommandList, io: IO) -> ShellGen:
        """One round of a loop; returns ``(status, stop)``.

        ``break`` and ``continue`` arrive as ``LoopControl``. ``break``/``continue``
        with a count above 1 belong to an outer loop, so they go on up after this
        loop has been left.
        """
        self.loop_depth += 1
        try:
            return (yield from self.run_list(body, io)), False
        except LoopControl as control:
            if control.levels > 1:
                control.levels -= 1
                raise LoopControl(control.kind, control.levels) from None
            return 0, control.kind == "break"
        finally:
            self.loop_depth -= 1

    def _next_round(self, io: IO) -> Generator[ShellEvent, Any, bool]:
        """Count one loop round for the whole line; True when the loop must stop.

        Every round of every loop (nested ones too) counts against
        ``LINE_ROUND_LIMIT``. Every ``HEARTBEAT_ROUNDS`` rounds an empty
        ``ShellOutput`` is yielded, so a loop that prints nothing still hands
        control back to the host often enough for Ctrl+C to land.
        """
        record = self._record
        if record.loops_stopped:
            return True
        record.loop_rounds += 1
        if record.loop_rounds > LINE_ROUND_LIMIT:
            self._stop_loops(io, f"bash: loops stopped after {LINE_ROUND_LIMIT} rounds "
                                 "in one command line")
            return True
        if record.loop_rounds % HEARTBEAT_ROUNDS == 0:
            yield ShellOutput("stdout", "")
        return False

    def _loop_limit(self, io: IO) -> int:
        return self._stop_loops(io, f"bash: loop stopped after {LOOP_LIMIT} rounds")

    def _stop_loops(self, io: IO, message: str) -> int:
        """Report a runaway loop once and skip the rest of the line (outer loops too)."""
        self._record.loops_stopped = True
        return self.report(io, "loop_limit", message,
                           "This loop never stopped on its own. Check that its condition "
                           "becomes false at some point.", concept=Concept.SHELL_SCRIPTING)

    def _run_subshell(self, node: Subshell, io: IO) -> ShellGen:
        child = self.spawn(self.positional, self.script_name)
        try:
            return (yield from child.run_list(node.body, io))
        except ExitRequest as request:
            return request.code

    # --- simple commands ----------------------------------------------------------------

    def _run_simple(self, node: SimpleCommand, io: IO) -> ShellGen:
        try:
            argv = expand_words(node.words, self)
            assignments = [(name, expand_single(word, self)) for name, word in node.assignments]
        except ExpansionError as error:
            return self.report(io, "shell_syntax_error", f"bash: {error.message}", _BAD_SUB)
        redirected = self._redirect(node.redirects, io)
        if redirected is None:
            return 1
        if not argv:
            # Only assignments (or only redirections): they change the shell itself.
            for name, value in assignments:
                self.env[name] = value
            # bash: the status of "x=$(cmd)" is cmd's status.
            return self.last_status if _has_substitution(node) else 0
        saved = {name: self.env.get(name) for name, _ in assignments}
        self.env.update(dict(assignments))
        # "CROP=corn ./bot" hands CROP to that one program, so it counts as exported
        # while the command runs (and only then).
        borrowed = {name for name, _ in assignments} - self.exported
        self.exported |= borrowed
        # bash prints "Segmentation fault" on its own stderr, not into the program's
        # "2> errors.txt": remember the stream from before this command's redirections.
        outer_stderr, self._job_stderr = self._job_stderr, io.stderr
        try:
            status = yield from self.run_argv(argv, redirected)
        finally:
            self._job_stderr = outer_stderr
            _restore_variables(self.env, saved)
            self.exported -= borrowed
        yield from self._flush()
        return status

    def run_argv(self, argv: list[str], io: IO) -> ShellGen:
        """Run an already-expanded command: a path, a builtin, a file on PATH or
        "command not found"."""
        name = argv[0]
        if "/" in name:
            return (yield from self._run_path(name, self.vfs.resolve(name, self.cwd), argv, io))
        spec = lookup(name)
        if spec is not None:
            return (yield from self._call(spec, argv, io))
        found = self._search_path(name)
        if found is not None:
            return (yield from self._run_path(name, found, argv, io))
        return self.report(io, "command_not_found", f"bash: {name}: command not found",
                           self._not_found_friendly(name), status=127)

    def _search_path(self, name: str) -> str | None:
        """An executable file called ``name`` in a ``PATH`` folder (``.`` is the cwd)."""
        for folder in self.env.get("PATH", "").split(":"):
            path = self.vfs.resolve(f"{folder or '.'}/{name}", self.cwd)
            if self.vfs.is_file(path) and self.vfs.stat(path).executable:
                return path
        return None

    def _call(self, spec: CommandSpec, argv: list[str], io: IO) -> ShellGen:
        ctx = CommandContext(self, argv[0], io.stdin, io.stdout, io.stderr)
        try:
            result = spec.fn(ctx, argv[1:])
            if inspect.isgenerator(result):
                result = yield from result
            return int(result or 0)
        except CommandError as error:
            return ctx.fail(error.code, error.message, error.friendly, error.status, error.concept)
        except VfsError as error:
            return ctx.fail(error.code, f"{argv[0]}: {error}", vfs_friendly(error),
                            concept=vfs_concept(error))
        except SHELL_SIGNALS:
            raise
        except Exception as error:        # one broken command is a shell error, not a crash
            return ctx.fail("internal_error", f"{argv[0]}: {error}",
                            f"The command '{argv[0]}' hit a bug in pycc's shell. Try another "
                            "way for now.")

    def _not_found_friendly(self, name: str) -> str:
        text = f"There is no program called '{name}'."
        if self.vfs.is_file(self.vfs.resolve(name, self.cwd)):
            return f"{text} To run the file '{name}' in this folder type './{name}'."
        twin = self._close_file(name)
        if twin is not None:
            runnable = self.vfs.stat(self.vfs.resolve(twin, self.cwd)).executable
            hint = f"./{twin}" if runnable else f"cat {twin}"
            return f"{text} There is a file called '{twin}' here: try `{hint}`."
        if "." not in name:          # "notes.txt" is a file name, never a mistyped command
            close = difflib.get_close_matches(name, command_names(), n=1, cutoff=0.7)
            if close:
                return f"{text} Did you mean '{close[0]}'?"
        return f"{text} Type `help` to see the commands you can use."

    def _close_file(self, name: str) -> str | None:
        """A file in the working folder whose name is close to ``name`` (a typo or case)."""
        if "/" in name or not self.vfs.is_dir(self.cwd):
            return None
        files = [entry for entry in self.vfs.listdir(self.cwd)
                 if self.vfs.is_file(posixpath.join(self.cwd, entry))]
        same_case = [entry for entry in files if entry.lower() == name.lower()]
        if same_case:
            return same_case[0]
        close = difflib.get_close_matches(name, files, n=1, cutoff=0.8)
        return close[0] if close else None

    # --- running files --------------------------------------------------------------------

    def _run_path(self, shown: str, path: str, argv: list[str], io: IO) -> ShellGen:
        if not self.vfs.exists(path):
            return self.report(io, "no_such_file", f"bash: {shown}: No such file or directory",
                               f"There is no file at '{shown}'. Run `ls` to check the name.",
                               status=127)
        if self.vfs.is_dir(path):
            return self.report(io, "is_a_directory", f"bash: {shown}: Is a directory",
                               f"'{shown}' is a folder. Use `cd {shown}` to go into "
                               "it.", status=126)
        data = self.vfs.read_file(path)
        if data.startswith(BUILTIN_MARKER):
            return (yield from self._run_marker(data, argv, io))
        if not self.vfs.stat(path).executable:
            return self._permission_denied(shown, data, io)
        return (yield from self.exec_file(shown, data, argv, io))

    def _run_marker(self, data: bytes, argv: list[str], io: IO) -> ShellGen:
        name = data[len(BUILTIN_MARKER):].decode("utf-8", "replace").strip()
        spec = lookup(name)
        if spec is None:
            return self.report(io, "command_not_found", f"bash: {argv[0]}: command not found",
                               self._not_found_friendly(name), status=127)
        return (yield from self._call(spec, [name] + argv[1:], io))

    def _permission_denied(self, shown: str, data: bytes, io: IO) -> int:
        base = shown.rsplit("/", 1)[-1]
        if data.startswith(EXE_MAGIC):
            return self.report(io, "not_executable", f"bash: {shown}: Permission denied",
                               f"The program '{base}' is not marked as runnable. "
                               f"Try `chmod +x {base}`.", status=126)
        if data.startswith(OBJ_MAGIC):
            # chmod +x would only trade this error for "cannot execute binary file":
            # an object file is half a program and has to be linked first.
            return self.report(io, "permission_denied", f"bash: {shown}: Permission denied",
                               _object_permission_friendly(shown), status=126)
        return self.report(io, "permission_denied", f"bash: {shown}: Permission denied",
                           f"Try `chmod +x {base}` to mark the file as runnable, then run it "
                           "again.",
                           status=126)

    def exec_file(self, shown: str, data: bytes, argv: list[str], io: IO) -> ShellGen:
        """Run file contents: a compiled program, an object file (refused) or a script.

        Any executable text file is a script, with or without ``#!/bin/sh``.
        """
        if data.startswith(EXE_MAGIC):
            return (yield from self.run_program(data, argv, io))
        if data.startswith(OBJ_MAGIC) or _looks_binary(data):
            return self.report(io, "cannot_execute_binary",
                               f"bash: {shown}: cannot execute binary file: Exec format error",
                               _binary_friendly(shown, data), status=126)
        return (yield from self.run_script(data.decode("utf-8", "replace"), shown, argv[1:], io))

    def run_script(self, text: str, name: str, args: list[str], io: IO) -> ShellGen:
        """Run script text in a child shell (``sh f.sh``, ``bash f.sh``, ``./f.sh``)."""
        if self.depth >= NESTING_LIMIT:
            return self.report(io, "loop_limit", "bash: maximum nesting level exceeded",
                               "A script keeps starting itself again. Check what it runs.",
                               concept=Concept.SHELL_SCRIPTING)
        self.count_construct(constructs.SCRIPT_RUN)
        child = self.spawn(args, name)
        return (yield from child.run_source(text, io))

    def source_file(self, text: str, args: list[str], io: IO) -> ShellGen:
        """``source f`` / ``. f``: run in *this* shell so variables and ``cd`` stick."""
        self.count_construct(constructs.SCRIPT_RUN)
        saved = self.positional
        if args:
            self.positional = list(args)
        try:
            return (yield from self.run_source(text, io, catch_exit=False))
        finally:
            self.positional = saved

    def run_program(self, executable: bytes, argv: list[str], io: IO) -> ShellGen:
        """Yield ``RunProgram`` and return the exit code the host sends back."""
        yield from self._flush()
        event = RunProgram(executable, list(argv), io.stdin, io.stdout, io.stderr, self.cwd,
                           self.exported_environment())
        reply = yield event
        if reply is None:
            if self.program_runner is None:
                return self.report(io, "not_executable", f"bash: {argv[0]}: no runtime available",
                                   "Programs can only run once the shell has a program "
                                   "runner (pycc.shell.headless).", status=126)
            reply = run_program_headless(event, self.program_runner)
        status = int(reply)
        # Like bash after a crashed process: "Segmentation fault (core dumped)".
        if status in EXIT_CODE_MESSAGES and status != INTERRUPTED_STATUS:
            (self._job_stderr or io.stderr).write(EXIT_CODE_MESSAGES[status] + "\n")
        yield from self._flush()
        return status

    def exported_environment(self) -> dict[str, str]:
        """The variables a started program sees (only exported ones, as in bash)."""
        return {name: value for name, value in self.env.items() if name in self.exported}

    def spawn(self, args: list[str], script_name: str, share_output: bool = True) -> "Shell":
        """A child shell: copies of the variables and folder, the same filesystem and
        the same per-line record."""
        child = Shell(self.vfs, self.cwd, self.toolchain, dict(self.env), self.program_runner)
        child.exported = set(self.exported)
        child.positional = list(args)
        child.script_name = script_name
        child.last_status = self.last_status
        child.terminal_input = self.terminal_input
        child.depth = self.depth + 1
        child.history = self.history
        child._record = self._record
        if share_output:
            child._pending = self._pending
        return child

    # --- redirections and output ----------------------------------------------------------

    def _redirect(self, redirects: list[Redirect], io: IO) -> IO | None:
        """Apply redirections; on failure report it and return None (command skipped)."""
        if not redirects:
            return io
        self.count_construct(constructs.REDIRECTION, len(redirects))
        try:
            return apply_redirects(redirects, io, self.vfs, self.cwd, self._redirect_target)
        except RedirectError as error:
            self.report(io, "redirect_error", error.message, error.friendly,
                        concept=Concept.SHELL_REDIRECTION)
            return None

    def _redirect_target(self, redirect: Redirect) -> str:
        assert redirect.target is not None          # the parser guarantees a file word
        try:
            fields = expand_words([redirect.target], self)
        except ExpansionError as error:
            raise RedirectError(f"bash: {error.message}", _BAD_SUB) from None
        if len(fields) != 1 or not fields[0]:
            raise RedirectError(f"bash: {redirect.target.raw}: ambiguous redirect",
                                "A redirection needs exactly one file name; this one expands "
                                "to several (or none). Put it in quotes or pick one file.")
        return fields[0]

    def report(self, io: IO, code: str, message: str, friendly: str, status: int = 1,
               concept: Concept | None = None) -> int:
        """Print an error with its friendly line, record it and return ``status``."""
        report_error(self.last_diagnostics, io.stderr, code, message, friendly, concept)
        return status

    def _flush(self) -> Generator[ShellEvent, Any, None]:
        while self._pending:
            yield self._pending.pop(0)


# --- helpers ------------------------------------------------------------------------------

_BAD_SUB = "Only plain `${NAME}` is supported inside braces here."


def _syntax_friendly(error: ParseError) -> str:
    message = error.message
    if "matching" in message:
        return "A quote or bracket was opened but never closed."
    if "unexpected end of file" in message:
        return "The command stopped too early: every `if` needs a `fi` and every loop a `done`."
    if "background" in message:
        return ("A single `&` runs jobs in the background, which this shell does not do. "
                "Did you mean `&&`?")
    return ("The shell did not expect this word here. Check the order: "
            "`if [ ... ]; then ...; fi` and `for x in a b; do ...; done`.")


def _has_substitution(node: SimpleCommand) -> bool:
    return any(isinstance(part, CommandSub) for _, word in node.assignments for part in word.parts)


def _restore_variables(env: dict[str, str], saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value


def _looks_binary(data: bytes) -> bool:
    if b"\0" in data[:1024]:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _link_example(shown: str) -> str:
    """``cc util.o -o util`` for ``./util.o``: how to turn an object into a program."""
    base = shown.rsplit("/", 1)[-1]
    program = base[:-2] if base.endswith(".o") else "program"
    return f"`cc {base} -o {program}`"


def _binary_friendly(shown: str, data: bytes) -> str:
    if data.startswith(OBJ_MAGIC):
        return (f"That is an object file; link it with cc into a program first "
                f"(for example {_link_example(shown)}).")
    return "That file is not a program or a script, so the shell cannot run it."


def _object_permission_friendly(shown: str) -> str:
    """``./util.o`` straight after ``cc -c``: chmod would not help, linking does."""
    return ("That is an object file (half of a program, made by `cc -c`), so `chmod +x` "
            f"will not help. Link it with cc into a program first (for example "
            f"{_link_example(shown)}), then run the program.")
