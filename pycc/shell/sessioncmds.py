"""Commands about the shell session itself.

Help (``help man history``), the screen (``clear``), variables (``env export``),
conditions (``test [ true false``), leaving and scripting (``exit source . sh bash``),
little extras (``which date sleep fortune``) and the ``cc``/``gcc``/``make``
placeholders that the toolchain module replaces through ``register_command``.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Generator

from pycc.core.concepts import Concept
from pycc.core.events import ClearScreen, RunSummary, ShellEvent
from pycc.shell import commands as registry
from pycc.shell.commands import (CommandContext, CommandError, ExitRequest, LoopControl,
                            missing_operand, parse_options, vfs_friendly)
from pycc.shell.manpages import render_manpage
from pycc.vfs.errors import VfsError

FAKE_DATE = "Mon Mar  4 08:00:00 UTC 2024"
# The steady clock of `time` (docs/SUPPORTED_SHELL.md section 3).
STATEMENT_SECONDS = 0.00001
PROGRAM_START_SECONDS = 0.001
TICK_SECONDS = 0.1
SHELL_SECONDS = 0.001

PROVERBS: tuple[str, ...] = (
    "A pointer without a target is a map to nowhere.",
    "Every malloc deserves its free.",
    "Arrays start at zero; so do good habits.",
    "Today's compiler warning is tomorrow's bug you never had to chase.",
    "A string is only as long as its terminating zero says.",
    "Check the return value, and the return value will look after you.",
    "Initialize before you read, or read whatever the last owner left behind.",
    "Small functions, clear names, quiet nights.",
    "When in doubt, print it out.",
    "Every loop needs a way home.",
    "sizeof is a friend; guessing sizes is not.",
    "The header promises; the source file keeps the promise.",
)

# How help groups the commands, so a newcomer reads six short sections, not one wall.
HELP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Moving around", ("pwd", "ls", "cd", "tree", "find")),
    ("Files", ("cat", "touch", "mkdir", "cp", "mv", "rm", "chmod", "file")),
    ("Text and pipes", ("echo", "printf", "head", "tail", "wc", "grep", "sort", "uniq", "diff",
                        "seq")),
    ("Building", ("cc", "gcc", "make")),
    ("Scripts", ("sh", "bash", "source", "test", "true", "false", "break", "continue", "exit",
                 "export", "env")),
    ("Help and extras", ("help", "man", "history", "clear", "which", "time", "date", "sleep",
                         "fortune")),
)


# --- help, man, history, clear ----------------------------------------------------------

def cmd_help(ctx: CommandContext, args: list[str]) -> int:
    """List the commands with one-line summaries, or describe one command."""
    if args:
        spec = registry.lookup(args[0])
        if spec is None:
            return ctx.fail("command_not_found", f"bash: help: no help topics match `{args[0]}'",
                            "Type `help` on its own to see every command.")
        ctx.out(f"{spec.name} - {spec.summary}\nType `man {spec.name}` for more.\n")
        return 0
    lines = ["Commands of the pycc shell. Type `man NAME` to learn more about one."]
    listed: set[str] = set()
    for title, names in HELP_GROUPS:
        present = [name for name in names if registry.lookup(name)]
        lines.append(f"\n{title}:")
        lines.extend(_help_line(name) for name in present)
        listed.update(present)
    aliases = ("[", ".")          # documented under test and source
    extra = [name for name in registry.command_names()
             if name not in listed and name not in aliases]
    if extra:
        lines.append("\nMore:")
        lines.extend(_help_line(name) for name in extra)
    ctx.out("\n".join(lines) + "\n")
    return 0


def _help_line(name: str) -> str:
    spec = registry.lookup(name)
    return f"  {name:<8} {spec.summary if spec else ''}"


def cmd_man(ctx: CommandContext, args: list[str]) -> int:
    """Show a command's man page."""
    if not args:
        return ctx.fail("missing_operand",
                        "What manual page do you want?\nFor example, try 'man man'.",
                        "Give man the name of a command, like `man ls`.")
    spec = registry.lookup(args[0])
    if spec is None:
        return ctx.fail("command_not_found", f"No manual entry for {args[0]}",
                        "There is no command by that name. Type `help` to list the commands.",
                        16)
    ctx.out(render_manpage(spec.manpage))
    return 0


def cmd_history(ctx: CommandContext, args: list[str]) -> int:
    """List typed commands (``-c`` clears, ``N`` shows the last N)."""
    options, operands = parse_options("history", args, "c")
    if options.get("c"):
        ctx.shell.history.clear()
        return 0
    entries = list(enumerate(ctx.shell.history, start=1))
    if operands:
        if not operands[0].isdigit():
            raise CommandError("bad_option",
                               f"bash: history: {operands[0]}: numeric argument required",
                               "history N shows the last N commands, like `history 5`.")
        entries = entries[-int(operands[0]):] if int(operands[0]) else []
    ctx.out("".join(f"{number:>5}  {line}\n" for number, line in entries))
    return 0


def cmd_clear(ctx: CommandContext, args: list[str]) -> int:
    """Clear the terminal screen."""
    ctx.emit(ClearScreen())
    return 0


# --- variables ----------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def cmd_env(ctx: CommandContext, args: list[str]) -> int:
    """Print the exported variables as ``NAME=value``."""
    shell = ctx.shell
    names = sorted(name for name in shell.exported if name in shell.env)
    ctx.out("".join(f"{name}={shell.env[name]}\n" for name in names))
    return 0


def cmd_export(ctx: CommandContext, args: list[str]) -> int:
    """Mark variables for export, optionally assigning them."""
    shell = ctx.shell
    if not args:
        names = sorted(name for name in shell.exported if name in shell.env)
        ctx.out("".join(f'declare -x {name}="{shell.env[name]}"\n' for name in names))
        return 0
    status = 0
    for arg in args:
        name, has_value, value = arg.partition("=")
        if not _IDENTIFIER.fullmatch(name):
            status = ctx.fail("bad_option", f"bash: export: `{arg}': not a valid identifier",
                              "Variable names use letters, digits and _, and cannot start "
                              "with a digit. No spaces around "
                              "`=`.", concept=Concept.SHELL_SCRIPTING)
            continue
        if has_value:
            shell.env[name] = value
        shell.exported.add(name)
    return status


# --- conditions ---------------------------------------------------------------------------

_UNARY = {"-f", "-d", "-e", "-z", "-n", "-x", "-s", "-r", "-w"}
_BINARY = {"=", "==", "!=", "-eq", "-ne", "-lt", "-le", "-gt", "-ge"}


class _TestParser:
    """Evaluates ``test`` expressions: ``!``, ``-a``/``-o``, ``( )``, unary and binary tests."""

    def __init__(self, ctx: CommandContext, words: list[str]) -> None:
        self.ctx = ctx
        self.words = words
        self.index = 0

    def _error(self, text: str, friendly: str) -> CommandError:
        return CommandError("shell_syntax_error", f"bash: {self.ctx.name}: {text}", friendly, 2,
                            Concept.SHELL_SCRIPTING)

    def _peek(self, offset: int = 0) -> str | None:
        position = self.index + offset
        return self.words[position] if position < len(self.words) else None

    def evaluate(self) -> bool:
        """Evaluate all the words; leftovers mean a malformed test."""
        if not self.words:
            return False
        result = self._or()
        if self.index != len(self.words):
            raise self._error("too many arguments",
                              "Put quotes around values that may contain spaces, like `[ "
                              "\"$name\" = x ]`.")
        return result

    def _or(self) -> bool:
        result = self._and()
        while self._peek() == "-o":
            self.index += 1
            result = self._and() or result
        return result

    def _and(self) -> bool:
        result = self._not()
        while self._peek() == "-a":
            self.index += 1
            result = self._not() and result
        return result

    def _not(self) -> bool:
        if self._peek() == "!" and self._peek(1) is not None:
            self.index += 1
            return not self._not()
        return self._primary()

    def _primary(self) -> bool:
        word = self._peek()
        if word == "(":
            self.index += 1
            result = self._or()
            if self._peek() != ")":
                raise self._error("`)' expected", "Every ( in a test needs a matching ).")
            self.index += 1
            return result
        if self._peek(1) in _BINARY and self._peek(2) is not None:
            self.index += 3
            return self._binary(word or "", self.words[self.index - 2], self.words[self.index - 1])
        if word in _UNARY and self._peek(1) is not None:
            self.index += 2
            return self._unary(word, self.words[self.index - 1])
        self.index += 1
        return bool(word)

    def _unary(self, op: str, operand: str) -> bool:
        if op == "-z":
            return operand == ""
        if op == "-n":
            return operand != ""
        vfs = self.ctx.vfs
        path = self.ctx.resolve(operand) if operand else ""
        if not path or not vfs.exists(path):
            return False
        checks = {
            "-e": True, "-r": True,
            "-f": vfs.is_file(path),
            "-d": vfs.is_dir(path),
            "-x": vfs.stat(path).executable,
            "-s": vfs.stat(path).size > 0 if vfs.is_file(path) else True,
            "-w": not vfs.stat(path).read_only,
        }
        return checks[op]

    def _binary(self, left: str, op: str, right: str) -> bool:
        if op in ("=", "=="):
            return left == right
        if op == "!=":
            return left != right
        a, b = self._integer(left), self._integer(right)
        return {"-eq": a == b, "-ne": a != b, "-lt": a < b, "-le": a <= b,
                "-gt": a > b, "-ge": a >= b}[op]

    def _integer(self, text: str) -> int:
        if not re.fullmatch(r"\s*-?\d+\s*", text):
            raise self._error(f"{text}: integer expression expected",
                              "-eq, -lt and friends compare whole numbers. Use = to compare text.")
        return int(text)


def cmd_test(ctx: CommandContext, args: list[str]) -> int:
    """``test EXPR``: status 0 when true, 1 when false, 2 on a malformed test."""
    return 0 if _TestParser(ctx, args).evaluate() else 1


def cmd_bracket(ctx: CommandContext, args: list[str]) -> int:
    """``[ EXPR ]``: like test, but the last word must be ``]``."""
    if not args or args[-1] != "]":
        raise CommandError("shell_syntax_error", "bash: [: missing `]'",
                           "A test written with [ must end with ] and spaces on both sides: "
                           "`[ -f notes.txt ]`.", 2, Concept.SHELL_SCRIPTING)
    return cmd_test(ctx, args[:-1])


def cmd_true(ctx: CommandContext, args: list[str]) -> int:
    """Succeed."""
    return 0


def cmd_false(ctx: CommandContext, args: list[str]) -> int:
    """Fail."""
    return 1


# --- leaving and scripting ----------------------------------------------------------------

def cmd_exit(ctx: CommandContext, args: list[str]) -> int:
    """Leave the shell or script with a status."""
    if not args:
        raise ExitRequest(ctx.shell.last_status)
    if not re.fullmatch(r"-?\d+", args[0]):
        ctx.fail("bad_option", f"bash: exit: {args[0]}: numeric argument required",
                 "exit takes a number: 0 means success, anything else means failure.",
                 concept=Concept.SHELL_EXIT_CODES)
        raise ExitRequest(2)
    raise ExitRequest(int(args[0]) & 255)


def cmd_break(ctx: CommandContext, args: list[str]) -> int:
    """Leave the innermost ``for``/``while``/``until`` loop (``break N``: N loops)."""
    return _loop_control(ctx, "break", args)


def cmd_continue(ctx: CommandContext, args: list[str]) -> int:
    """Skip to the next round of the innermost loop (``continue N``: of the Nth)."""
    return _loop_control(ctx, "continue", args)


def _loop_control(ctx: CommandContext, kind: str, args: list[str]) -> int:
    """Raise ``LoopControl`` for the loops to catch; outside a loop do nothing."""
    if args and not re.fullmatch(r"[1-9]\d*", args[0]):
        return ctx.fail("bad_option", f"bash: {kind}: {args[0]}: loop count out of range",
                        f"`{kind}` takes how many loops to leave, a whole number from 1 up. "
                        f"Most of the time plain `{kind}` is what you want.",
                        concept=Concept.SHELL_SCRIPTING)
    if ctx.shell.loop_depth == 0:
        # bash only warns here, and the status stays 0.
        ctx.stderr.write(f"bash: {kind}: only meaningful in a `for', `while', or `until' "
                         "loop\n")
        return 0
    raise LoopControl(kind, min(int(args[0]) if args else 1, ctx.shell.loop_depth))


def _read_script(ctx: CommandContext, operand: str) -> str:
    try:
        data = ctx.vfs.read_file(ctx.resolve(operand))
    except VfsError as error:
        missing = error.code == "no_such_file"
        error.path = operand
        friendly = (f"There is no script called '{operand}' here. Run `ls` to check its name."
                    if missing else vfs_friendly(error))
        raise CommandError(error.code, f"{ctx.name}: {operand}: {error.reason}", friendly,
                           127 if missing else 126) from None
    if data.startswith((b"CVEXE1\n", b"CVOBJ1\n")) or b"\0" in data[:1024]:
        raise CommandError("cannot_execute_binary",
                           f"{ctx.name}: {operand}: cannot execute binary file",
                           f"'{operand}' is a compiled file, not a script. Programs run directly: "
                           f"`./{operand.rsplit('/', 1)[-1]}`.", 126)
    return data.decode("utf-8", "replace")


def cmd_source(ctx: CommandContext, args: list[str]) -> Generator[ShellEvent, Any, int]:
    """Run a script in the current shell, so its variables and ``cd`` stay."""
    if not args:
        raise CommandError("missing_operand", f"bash: {ctx.name}: filename argument required",
                           f"Name the script to read, like `{ctx.name} env.sh`.", 2)
    text = _read_script(ctx, args[0])
    return (yield from ctx.shell.source_file(text, args[1:], ctx.io))


def cmd_sh(ctx: CommandContext, args: list[str]) -> Generator[ShellEvent, Any, int]:
    """Run a script (or ``-c COMMANDS``) in a child shell."""
    if args and args[0] == "-c":
        if len(args) < 2:
            raise CommandError("missing_operand", f"{ctx.name}: -c: option requires an argument",
                               f"Put the commands in quotes after -c: `{ctx.name} -c \"ls\"`.", 2)
        return (yield from ctx.shell.run_script(args[1], ctx.name, args[2:], ctx.io))
    if not args:
        raise missing_operand(ctx.name, f"{ctx.name} build.sh")
    text = _read_script(ctx, args[0])
    return (yield from ctx.shell.run_script(text, args[0], args[1:], ctx.io))


# --- small extras -------------------------------------------------------------------------

def cmd_which(ctx: CommandContext, args: list[str]) -> int:
    """Print where each command lives, searching the ``PATH`` folders in order.

    Built-in commands are found through their marker files in ``/bin``, so ``which``
    answers exactly what running the name would do.
    """
    if not args:
        return 1
    path_value = ctx.shell.env.get("PATH", "")
    status = 0
    for name in args:
        found = _which_one(ctx, name, path_value)
        if found:
            ctx.out(found + "\n")
            continue
        status = ctx.fail("command_not_found", f"which: no {name} in ({path_value})",
                          f"There is no command called '{name}'.")
    return status


def _which_one(ctx: CommandContext, name: str, path_value: str) -> str | None:
    candidates = [name] if "/" in name else \
        [f"{folder or '.'}/{name}" for folder in path_value.split(":")]
    for candidate in candidates:
        path = ctx.resolve(candidate)
        if ctx.vfs.is_file(path) and ctx.vfs.stat(path).executable:
            return candidate
    return None


def cmd_date(ctx: CommandContext, args: list[str]) -> int:
    """Print the simulated (fixed) date."""
    ctx.out(FAKE_DATE + "\n")
    return 0


def cmd_sleep(ctx: CommandContext, args: list[str]) -> int:
    """Pretend to wait (returns at once, so scripts stay fast and deterministic)."""
    if not args:
        raise missing_operand("sleep", "sleep 1")
    for arg in args:
        if not re.fullmatch(r"\d+(\.\d+)?[smhd]?", arg):
            raise CommandError("bad_option", f"sleep: invalid time interval '{arg}'",
                               "sleep takes a number of seconds, like `sleep 2`.")
    return 0


def cmd_time(ctx: CommandContext, args: list[str]) -> Generator[ShellEvent, Any, int]:
    """Run a command, then print how long it took on stderr, in bash's three lines.

    pycc keeps a steady clock so the numbers are the same on every computer:
    each C statement a program runs takes ten microseconds (``user``), starting a
    program a millisecond (``sys``) and each device call a tenth of a second.
    """
    before = len(ctx.shell.recorded_runs)
    status = (yield from ctx.shell.run_argv(args, ctx.io)) if args else 0
    runs = ctx.shell.recorded_runs[before:]
    ctx.err(time_report(runs))
    return status


def time_report(runs: list[RunSummary]) -> str:
    """bash's ``real``/``user``/``sys`` lines for the programs in ``runs``."""
    user = sum(run.statements for run in runs) * STATEMENT_SECONDS
    system = len(runs) * PROGRAM_START_SECONDS
    real = user + system + sum(run.ticks for run in runs) * TICK_SECONDS + SHELL_SECONDS
    return "".join(f"{label}\t{_minutes_seconds(value)}\n" for label, value in
                   (("\nreal", real), ("user", user), ("sys", system)))


def _minutes_seconds(seconds: float) -> str:
    """``0m0.004s``, the way bash's ``time`` writes a duration."""
    minutes, rest = divmod(round(seconds, 3), 60)
    return f"{int(minutes)}m{rest:.3f}s"


def cmd_fortune(ctx: CommandContext, args: list[str]) -> int:
    """Print a C proverb (deterministic: it depends on how much you have typed)."""
    ctx.out(PROVERBS[len(ctx.shell.history) % len(PROVERBS)] + "\n")
    return 0


def _placeholder(tool: str) -> Callable[[CommandContext, list[str]], int]:
    """A stand-in for a build tool that another module installs later."""
    what = "build tool" if tool == "make" else "compiler"

    def run(ctx: CommandContext, args: list[str]) -> int:
        """Explain that the tool is not available yet."""
        return ctx.fail("command_not_found",
                        f"{tool}: the {what} is not installed in this workshop yet",
                        f"The workshop's {what} arrives later in the story.", 127,
                        Concept.BUILD_MAKE if tool == "make" else Concept.BUILD_CC)

    run.__doc__ = f"Placeholder for {tool} until the toolchain registers the real one."
    return run


SESSION_COMMANDS = {
    "help": cmd_help, "man": cmd_man, "history": cmd_history, "clear": cmd_clear,
    "env": cmd_env, "export": cmd_export, "test": cmd_test, "[": cmd_bracket,
    "true": cmd_true, "false": cmd_false, "exit": cmd_exit, "source": cmd_source,
    "break": cmd_break, "continue": cmd_continue,
    ".": cmd_source, "sh": cmd_sh, "bash": cmd_sh, "which": cmd_which, "date": cmd_date,
    "sleep": cmd_sleep, "fortune": cmd_fortune, "time": cmd_time,
    "cc": _placeholder("cc"), "gcc": _placeholder("gcc"), "make": _placeholder("make"),
}
