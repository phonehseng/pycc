"""The ``pycc`` command line: compile and run C programs in the simulated machine.

    pycc run main.c util.c [-Wall ...] [-- ARGS...]   compile, link and run
    pycc build main.c util.c -o prog [-Wall ...]      compile and link only
    pycc exec prog [ARGS...]                          run a program built with -o
    pycc shell                                        the Bash-like shell

Programs never touch the real disk. The current folder is copied into an in-memory
filesystem (as ``~/project``), the program runs there, and what it writes stays there.
Only ``pycc build -o`` writes one file to the real disk: the built program.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from pycc import __version__
from pycc.core.events import ClearScreen, RunProgram, ShellOutput
from pycc.interp.stdlib import default_toolchain
from pycc.shell.ansi import strip_ansi
from pycc.shell.executor import Shell
from pycc.shell.headless import DEFAULT_BUDGET, make_headless_shell
from pycc.shell.streams import TerminalSink
from pycc.vfs import VirtualFS, standard_layout

PROJECT_NAME = "project"
SKIPPED_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
                ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
MAX_FILE_BYTES = 1 << 20          # larger files are not copied into the sandbox
MAX_FILES = 2000
DEFAULT_CFLAGS = ["-Wall", "-std=c99"]


class RealStdin:
    """The shell's keyboard: lines from the real standard input (None at end of file)."""

    is_terminal = True

    def readline(self) -> str | None:
        line = sys.stdin.readline()
        return line if line else None


def _colour_ok(target: Any) -> bool:
    """Keep the ANSI colours only for a real terminal (and never with NO_COLOR set)."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):
        return False


def _write(stream: str, text: str) -> None:
    target = sys.stderr if stream == "stderr" else sys.stdout
    target.write(text if _colour_ok(target) else strip_ansi(text))
    target.flush()


# --- the sandbox ---------------------------------------------------------------------------

def sandbox_from(folder: Path, project: str = PROJECT_NAME) -> tuple[VirtualFS, str]:
    """A fresh VFS with ``folder``'s files copied into ``~/<project>``."""
    vfs = standard_layout(project)
    root = f"/home/user/{project}"
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith("."))
        rel = Path(dirpath).relative_to(folder).as_posix()
        target = root if rel == "." else f"{root}/{rel}"
        if not vfs.is_dir(target):
            vfs.mkdir(target, parents=True)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            vfs.write_file(f"{target}/{name}", data)
            count += 1
            if count >= MAX_FILES:
                return vfs, root
    return vfs, root


def make_shell(folder: Path, budget: int) -> Shell:
    vfs, root = sandbox_from(folder)
    shell = make_headless_shell(vfs, default_toolchain(), cwd=root, budget=budget)
    shell.terminal_input = RealStdin()
    return shell


def drive(shell: Shell, line: str) -> int:
    """Run one command line, printing output as it happens; returns its exit status."""
    generator = shell.execute(line)
    reply: Any = None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as stop:
            return int(stop.value or 0)
        except KeyboardInterrupt:
            generator.close()
            _write("stderr", "^C\n")
            return 130
        reply = None
        if isinstance(event, ShellOutput):
            _write(event.stream, event.text)
        elif isinstance(event, ClearScreen):
            _write("stdout", "\x1b[2J\x1b[H")
        elif isinstance(event, RunProgram):
            # Show a running program's output live (prompts before scanf, for example).
            for sink in (event.stdout, event.stderr):
                if isinstance(sink, TerminalSink):
                    sink.forward = (lambda text, s=sink.stream: _write(s, text))


def _sandbox_path(path: str) -> str:
    """A command-line path as the sandbox sees it (relative to ~/project)."""
    real = Path(path)
    if real.is_absolute():
        try:
            return real.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            raise SystemExit(f"pycc: {path}: only files inside the current folder can be used")
    return real.as_posix()


# --- commands ----------------------------------------------------------------------------------

def split_args(items: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Sources, cc flags and program arguments (after ``--``)."""
    program_args: list[str] = []
    if "--" in items:
        cut = items.index("--")
        items, program_args = items[:cut], items[cut + 1:]
    sources: list[str] = []
    flags: list[str] = []
    takes_value = {"-I", "-D", "-U", "-o", "-std"}
    it = iter(items)
    for item in it:
        if item in takes_value:
            flags += [item, next(it, "")]
        elif item.startswith("-"):
            flags.append(item)
        else:
            sources.append(item)
    return sources, flags, program_args


def cmd_run(args: argparse.Namespace) -> int:
    sources, flags, program_args = split_args(args.items)
    if not sources:
        raise SystemExit("pycc run: give at least one .c file")
    shell = make_shell(Path.cwd(), args.budget)
    cflags = flags or DEFAULT_CFLAGS
    words = ["cc", *cflags, *(_sandbox_path(s) for s in sources), "-o", "/tmp/a.out"]
    status = drive(shell, shlex.join(words))
    if status != 0:
        return status
    return drive(shell, shlex.join(["/tmp/a.out", *program_args]))


def cmd_build(args: argparse.Namespace) -> int:
    sources, flags, _ = split_args(args.items)
    if not sources:
        raise SystemExit("pycc build: give at least one .c file")
    output = None
    if "-o" in flags:
        index = flags.index("-o")
        output = flags[index + 1]
        del flags[index:index + 2]
    shell = make_shell(Path.cwd(), args.budget)
    cflags = flags or DEFAULT_CFLAGS
    words = ["cc", *cflags, *(_sandbox_path(s) for s in sources), "-o", "/tmp/a.out"]
    status = drive(shell, shlex.join(words))
    if status == 0 and output:
        Path(output).write_bytes(shell.vfs.read_file("/tmp/a.out"))
    return status


def cmd_exec(args: argparse.Namespace) -> int:
    program = Path(args.program)
    try:
        data = program.read_bytes()
    except OSError as error:
        raise SystemExit(f"pycc exec: {args.program}: {error.strerror}")
    shell = make_shell(Path.cwd(), args.budget)
    shell.vfs.write_file("/tmp/prog", data)
    shell.vfs.set_executable("/tmp/prog")
    return drive(shell, shlex.join(["/tmp/prog", *args.args]))


def cmd_shell(args: argparse.Namespace) -> int:
    shell = make_shell(Path.cwd(), args.budget)
    if args.command is not None:
        return drive(shell, args.command)
    print("pycc shell: a Bash-like shell with cc and make. Your folder is ~/project "
          "(in memory; nothing is written back). Type 'help', or 'exit' to leave.")
    status = 0
    while True:
        try:
            line = input(shell.prompt)
        except EOFError:
            print()
            return status
        except KeyboardInterrupt:
            print("^C")
            continue
        status = drive(shell, line)
        if shell.exited:
            return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycc",
        description="A C99 compiler and an LP64 interpreter with memory-error detection.")
    parser.add_argument("--version", action="version", version=f"pycc {__version__}")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="statements a program may run before it is stopped "
                             f"(default {DEFAULT_BUDGET:,})")
    sub = parser.add_subparsers(dest="command_name", required=True)

    run = sub.add_parser("run", help="compile, link and run C files",
                         usage="pycc run FILE.c [FILE.c ...] [CFLAGS] [-- ARGS ...]")
    run.add_argument("items", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    build = sub.add_parser("build", help="compile and link; -o writes the program",
                           usage="pycc build FILE.c [FILE.c ...] [CFLAGS] [-o PROGRAM]")
    build.add_argument("items", nargs=argparse.REMAINDER)
    build.set_defaults(func=cmd_build)

    exe = sub.add_parser("exec", help="run a program written by 'pycc build -o'")
    exe.add_argument("program")
    exe.add_argument("args", nargs=argparse.REMAINDER)
    exe.set_defaults(func=cmd_exec)

    sh = sub.add_parser("shell", help="start the Bash-like shell (cc, make, pipes...)")
    sh.add_argument("-c", dest="command", help="run one command line and exit")
    sh.set_defaults(func=cmd_shell)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.name == "nt":
        os.system("")        # turns on ANSI colour handling in the Windows console
    return args.func(args)


__all__ = ["main", "sandbox_from"]
