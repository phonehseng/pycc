"""Where a command's input comes from and its output goes to.

Three kinds of stream exist besides the plain ``StringSource``/``StringSink`` of
``core.events`` (used for pipes):

- ``TerminalSink``/``TerminalSource``: the user's terminal. They carry
  ``is_terminal = True`` so commands can colour output and print friendly lines only
  when a person is reading;
- ``FileSink``: output redirected into a VFS file (``> out.txt``);
- ``/dev/null``, which swallows output and reads as empty.

``apply_redirects`` turns a command's redirections into a new set of streams, in
order, exactly like bash: ``> f 2>&1`` sends both streams to ``f`` but ``2>&1 > f``
leaves errors on the terminal.
"""
from __future__ import annotations

from typing import Callable

from pycc.core.events import InputSource, NullSink, OutputSink, ShellEvent, ShellOutput, StringSource
from pycc.shell.commands import IO
from pycc.shell.parser import Redirect
from pycc.vfs.errors import VfsError
from pycc.vfs.filesystem import VirtualFS

DEV_NULL = "/dev/null"


class TerminalSink:
    """Output going to the terminal: queued as ``ShellOutput`` events.

    A host that shows a running program's output live can set ``forward`` to a
    callback; text then goes straight to it instead of being queued.
    """

    is_terminal = True

    def __init__(self, stream: str, pending: list[ShellEvent]) -> None:
        """``stream`` is ``stdout`` or ``stderr``; ``pending`` is the shell's queue."""
        self.stream = stream
        self.pending = pending
        self.forward: Callable[[str], None] | None = None

    def write(self, text: str) -> None:
        """Queue (or forward) ``text``; consecutive writes merge into one event."""
        if not text:
            return
        if self.forward is not None:
            self.forward(text)
            return
        last = self.pending[-1] if self.pending else None
        if isinstance(last, ShellOutput) and last.stream == self.stream:
            last.text += text
        else:
            self.pending.append(ShellOutput(self.stream, text))


class TerminalSource:
    """Keyboard input. Headless it holds queued lines (EOF when empty); an interactive
    host sees ``is_terminal`` and asks the user instead."""

    is_terminal = True

    def __init__(self, lines: list[str] | None = None) -> None:
        """Start with optional queued lines (each should end with a newline)."""
        self.lines = list(lines or [])

    def readline(self) -> str | None:
        """Next queued line or None (end of file)."""
        return self.lines.pop(0) if self.lines else None


class FileSink:
    """Output redirected into a VFS file; every write lands in the file at once."""

    def __init__(self, vfs: VirtualFS, path: str, append: bool) -> None:
        """Open like bash: ``>`` truncates (creating the file), ``>>`` appends."""
        self.vfs = vfs
        self.path = path
        if append and vfs.is_file(path):
            vfs.write_file(path, b"", append=True)    # checks write permission now
        else:
            vfs.write_file(path, b"")

    def write(self, text: str) -> None:
        """Append ``text`` to the file."""
        if text:
            self.vfs.write_file(self.path, text, append=True)


class RedirectError(Exception):
    """A redirection that cannot be set up; the command does not run."""

    def __init__(self, message: str, friendly: str) -> None:
        """``message`` is what bash prints, ``friendly`` the beginner explanation."""
        super().__init__(message)
        self.message = message
        self.friendly = friendly


def apply_redirects(redirects: list[Redirect], io: IO, vfs: VirtualFS, cwd: str,
                    target_of: Callable[[Redirect], str]) -> IO:
    """New streams after applying ``redirects`` left to right.

    ``target_of`` expands a redirection's file word (the executor owns expansion).
    Raises ``RedirectError``.
    """
    stdin, stdout, stderr = io.stdin, io.stdout, io.stderr
    for redirect in redirects:
        op = redirect.op
        if op == "2>&1":
            stderr = stdout
        elif op in ("1>&2", ">&2"):
            stdout = stderr
        elif op == "<":
            stdin = open_input(vfs, cwd, target_of(redirect))
        else:
            sink = open_output(vfs, cwd, target_of(redirect), append=op.endswith(">>"))
            if op.startswith("2"):
                stderr = sink
            elif op.startswith("&"):
                stdout = stderr = sink
            else:
                stdout = sink
    return IO(stdin, stdout, stderr)


def open_input(vfs: VirtualFS, cwd: str, target: str) -> InputSource:
    """``< target``: the whole file as an input stream."""
    if target == DEV_NULL:
        return StringSource("")
    try:
        return StringSource(vfs.read_text(vfs.resolve(target, cwd)))
    except VfsError as error:
        raise RedirectError(f"bash: {target}: {error.reason}",
                            f"`<` reads a file into the command, but '{target}' cannot be "
                            f"read: {error.reason.lower()}.") from None


def open_output(vfs: VirtualFS, cwd: str, target: str, append: bool) -> OutputSink:
    """``> target`` / ``>> target``: a sink writing into the file."""
    if target == DEV_NULL:
        return NullSink()
    if target.endswith("/"):
        # "> logs/" names a folder; resolving would drop the slash and create a file.
        raise RedirectError(f"bash: {target}: Is a directory",
                            f"'{target}' ends with / so it names a folder. Output can only go "
                            "into a file, like `> logs/out.txt`.")
    try:
        return FileSink(vfs, vfs.resolve(target, cwd), append)
    except VfsError as error:
        raise RedirectError(f"bash: {target}: {error.reason}",
                            f"The output could not go into '{target}': {error.reason.lower()}. "
                            "Check that the folder exists and is yours.") from None
