"""``<stdio.h>``: the terminal, files, and printf/scanf.

How the pieces fit together:

* **Streams.** A ``FILE *`` points at a 4-byte ``struct _CVFILE`` in the data segment
  (so an open file is never reported as a memory leak). Python keeps the real state
  in a table keyed by that address: a ``StreamState`` with the path, mode, position,
  a mirror of the file's bytes and the end-of-file and error flags. ``stdin``,
  ``stdout`` and ``stderr`` are three such objects, created on first use.
* **Output.** Text for ``stdout``/``stderr`` is yielded as an ``Output`` event on every
  call, so it appears in the terminal at once. File writes go straight through to
  ``HostFiles`` on every call ("write-through"): if the program crashes a moment
  later, the half-written file is there to ``cat``, just like a real crash after
  ``fflush``.
* **Input.** One ``StdinBuffer`` per run holds what the user typed. Every stdin
  reader takes bytes from it; when it runs dry the reader yields ``InputRequest``
  with the current prompt (the ``stdout`` text since the last newline) and appends
  the host's reply. A reply of ``None`` is end of file, for good.
* **Mistakes.** NULL ``FILE *`` traps ``null_dereference``; a closed or made-up one
  traps ``invalid_stream``. Files still open when the program ends are reported by
  ``finalize`` as ``file_not_closed`` warnings.

All the per-run state lives in ``machine.state["stdio"]``, so two runs never share a
stdin buffer or a FILE table.
"""
from __future__ import annotations

import codecs
import math
import random
import re
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Protocol

from pycc.core.concepts import Concept
from pycc.core.events import InputRequest, Output
from pycc.interp import constructs
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.c_types import (
    DOUBLE, FLOAT, INT, LONG, SCHAR, SHORT, SIZE_T, UCHAR, UINT, ULONG, USHORT, VOID, CType,
    FunctionType, Member, PointerType, StructType, encode, format_type, function_address,
    wrap_int,
)
from pycc.interp.hostio import EBADF, EINVAL, EISDIR, ENOENT, join_path
from pycc.interp.stdlib.errno_ import c_strerror
from pycc.interp.stdlib.headers import LibraryFunction, register_functions
from pycc.interp.stdlib.string_ import char_pointer, read_c_string, require_pointer
from pycc.interp.values import Value
from pycc.core.rng import mix_seed

HEADER = "stdio.h"
STATE_KEY = "stdio"
EOF = -1

# The layout promised by stdio.h: typedef struct _CVFILE { int _id; } FILE;
FILE_TYPE = StructType("_CVFILE", [Member("_id", INT)])
FILE_PTR = PointerType(FILE_TYPE)

STANDARD_STREAMS: dict[str, tuple[int, str]] = {"stdin": (0, "r"), "stdout": (1, "w"),
                                                "stderr": (2, "w")}
VALID_MODES = frozenset({"r", "w", "a", "r+", "w+", "a+"})
SPACE_BYTES = frozenset(b" \t\n\v\f\r")
# Sets, not bytes literals: ``EOF in b"+-"`` would raise, ``EOF in SIGNS`` is False.
SIGNS = frozenset(b"+-")
HEX_MARKS = frozenset(b"xX")
EXPONENT_MARKS = frozenset(b"eE")
DECIMAL_DIGITS = frozenset(b"0123456789")


_PROMPT_LIMIT = 200          # characters of the current output line kept as the prompt
_STDIN_COMPACT_AT = 4096     # drop consumed stdin bytes once this many have piled up
_UNLIMITED = 1 << 62
# How many closed FILE objects wait before their memory is reused. Reusing them keeps
# a program that opens and closes a log file every tick from filling the data
# segment; the wait keeps a FILE * used just after fclose trapping invalid_stream.
_FILE_QUARANTINE = 16

Events = Generator[Any, Any, Any]


# ==================================================================== per-run state

@dataclass
class StdinBuffer:
    """Everything typed into the program so far, and how much of it was read."""

    data: bytearray = field(default_factory=bytearray)
    pos: int = 0
    at_eof: bool = False
    requests: int = 0

    def fill(self, prompt: str) -> Generator[Any, Any, bool]:
        """Ask the host for input until there is an unread byte or end of file."""
        while self.pos >= len(self.data) and not self.at_eof:
            self.requests += 1
            reply = yield InputRequest(prompt)
            if reply is None:
                self.at_eof = True      # end of file is permanent, as for a closed pipe
            else:
                self._append(reply)
        return self.pos < len(self.data)

    def _append(self, reply: str | bytes) -> None:
        if self.pos > _STDIN_COMPACT_AT:
            del self.data[:self.pos]
            self.pos = 0
        self.data += reply.encode("utf-8") if isinstance(reply, str) else bytes(reply)

    def getc(self, prompt: str) -> Generator[Any, Any, int]:
        """Take one byte (asking the host if needed); ``EOF`` at end of input."""
        if not (yield from self.fill(prompt)):
            return EOF
        byte = self.data[self.pos]
        self.pos += 1
        return byte

    def unget(self, byte: int) -> None:
        """Push one byte back so the next read sees it again (``ungetc``)."""
        if self.pos > 0 and self.data[self.pos - 1] == byte:
            self.pos -= 1
        else:
            self.data.insert(self.pos, byte)


@dataclass
class StreamState:
    """What a ``FILE *`` really is: D7's StreamState(path, mode, pos, buffer, eof, error)."""

    id: int
    addr: int
    path: str
    mode: str
    kind: str = "file"                      # file | stdin | stdout | stderr
    pos: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    eof: bool = False
    error: bool = False
    closed: bool = False
    pushback: list[int] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        """True for modes that allow reading (r, r+, w+, a+)."""
        return self.mode.startswith("r") or "+" in self.mode

    @property
    def writable(self) -> bool:
        """True for modes that allow writing (w, a and every + mode)."""
        return self.mode[0] in "wa" or "+" in self.mode

    @property
    def appending(self) -> bool:
        """True for a and a+: every write goes to the end of the file."""
        return self.mode.startswith("a")

    @property
    def is_console(self) -> bool:
        """True for stdout and stderr, whose text becomes Output events."""
        return self.kind in ("stdout", "stderr")

    @property
    def label(self) -> str:
        """How messages name the stream: its path, or stdin/stdout/stderr."""
        return self.path if self.kind == "file" else self.kind


@dataclass
class StdioState:
    """All of stdio's bookkeeping for one run (stored in ``machine.state``)."""

    streams: dict[int, StreamState] = field(default_factory=dict)   # FILE address -> stream
    standard: dict[str, int] = field(default_factory=dict)          # "stdin" -> FILE address
    stdin: StdinBuffer = field(default_factory=StdinBuffer)
    prompt: str = ""                       # stdout text since the last newline
    decoders: dict[str, codecs.IncrementalDecoder] = field(default_factory=dict)
    next_id: int = 3
    closed_files: deque[int] = field(default_factory=deque)        # FILE addresses, oldest first

    def note_stdout(self, text: str) -> None:
        """Remember the unfinished output line: it is the prompt for the next read."""
        if "\n" in text:
            self.prompt = text.rsplit("\n", 1)[1]
        else:
            self.prompt += text
        self.prompt = self.prompt[-_PROMPT_LIMIT:]


def stdio_state(machine: Machine) -> StdioState:
    """This run's stdio state, created on first use."""
    state = machine.state.get(STATE_KEY)
    if state is None:
        state = machine.state[STATE_KEY] = StdioState()
    return state


def standard_stream(machine: Machine, name: str) -> StreamState:
    """``stdin``, ``stdout`` or ``stderr``, creating its FILE object on first use.

    Both the library variable's initialiser and functions such as ``getchar`` come
    here, so they always agree on the address.
    """
    state = stdio_state(machine)
    addr = state.standard.get(name)
    if addr is not None:
        return state.streams[addr]
    stream_id, mode = STANDARD_STREAMS[name]
    addr = machine.memory.alloc_global(FILE_TYPE.size, FILE_TYPE.align, name, FILE_TYPE)
    machine.memory.store(addr, INT, stream_id)
    stream = StreamState(stream_id, addr, name, mode, kind=name)
    state.streams[addr] = stream
    state.standard[name] = addr
    return stream


def stdin_buffer(machine: Machine) -> StdinBuffer:
    """The run's stdin buffer."""
    return stdio_state(machine).stdin


def _files(machine: Machine) -> Any:
    """The ``HostFiles`` of this run, or None (no filesystem: every fopen fails)."""
    return getattr(machine.io, "files", None)


def _resolve(machine: Machine, path: str) -> str:
    """A program path made absolute against the program's working directory."""
    return join_path(getattr(machine.io, "cwd", "/") or "/", path)


# ==================================================================== FILE * arguments

def stream_arg(machine: Machine, value: Value, function: str) -> StreamState:
    """Turn a ``FILE *`` argument into its stream, trapping on NULL, closed or garbage."""
    addr = value.raw
    if addr == 0:
        machine.trap(
            "null_dereference",
            f"{function}(): the FILE * argument is a NULL pointer",
            f"{function}() was given a NULL FILE pointer. fopen returns NULL when it cannot "
            "open a file, so check it straight after opening: if (f == NULL) { "
            "perror(\"fopen\"); return 1; }",
            Concept.ERROR_HANDLING, function=function)
    stream = stdio_state(machine).streams.get(addr)
    if stream is None:
        machine.trap(
            "invalid_stream",
            f"{function}(): {addr:#x} is not an open FILE",
            f"{function}() needs a FILE pointer that came from fopen (or stdin, stdout, "
            "stderr). This pointer does not point at any FILE; it may be uninitialised.",
            Concept.FILE_IO, function=function, address=addr)
    if stream.closed:
        machine.trap(
            "invalid_stream",
            f"{function}(): the FILE for '{stream.label}' was already closed",
            f"'{stream.label}' was closed with fclose, so this FILE pointer is no longer "
            "valid. Use files only between fopen and fclose, and set the pointer to NULL "
            "after closing it.",
            Concept.FILE_IO, function=function, path=stream.label)
    return stream


def _count_read(machine: Machine, stream: StreamState) -> None:
    """Reading the terminal trains scanf-style input; reading a file trains file I/O."""
    machine.count_construct(constructs.SCANF if stream.kind == "stdin" else constructs.FILE_READ)


# ==================================================================== low-level I/O

def write_stream(machine: Machine, stream: StreamState, data: bytes) -> Generator[Any, Any, bool]:
    """Write bytes to a stream; True on success. Console text is yielded as Output."""
    if not stream.writable:
        stream.error = True
        machine.set_errno(EBADF)
        return False
    if stream.is_console:
        yield from _emit_console(machine, stream, data)
        return True
    return _write_file(machine, stream, data)


def _emit_console(machine: Machine, stream: StreamState, data: bytes) -> Events:
    """Yield ``Output`` for stdout/stderr text.

    An incremental UTF-8 decoder per stream keeps a character that a program prints
    one byte at a time (``putchar``) from turning into two replacement marks.
    """
    state = stdio_state(machine)
    decoder = state.decoders.get(stream.kind)
    if decoder is None:
        decoder = state.decoders[stream.kind] = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(data)
    machine.count_construct(constructs.PRINTF)
    if text:
        if stream.kind == "stdout":
            state.note_stdout(text)
        yield Output(stream.kind, text)


def _write_file(machine: Machine, stream: StreamState, data: bytes) -> bool:
    """Update the stream's mirror of the file and write through to the host."""
    if stream.appending:
        stream.pos = len(stream.buffer)
    at_end = stream.pos == len(stream.buffer)
    stream.buffer[stream.pos:stream.pos + len(data)] = data
    stream.pos += len(data)
    machine.count_construct(constructs.FILE_WRITE)
    files = _files(machine)
    # Appending is cheap; overwriting the middle (r+) rewrites the whole file.
    if at_end:
        result = files.write(stream.path, bytes(data), True)
    else:
        result = files.write(stream.path, bytes(stream.buffer), False)
    if result < 0:
        stream.error = True
        machine.set_errno(-result)
        return False
    return True


def read_byte(machine: Machine, stream: StreamState) -> Generator[Any, Any, int]:
    """Read one byte from a stream (stdin may ask the host); ``EOF`` at the end."""
    if not stream.readable:
        stream.error = True
        machine.set_errno(EBADF)
        return EOF
    if stream.pushback:
        return stream.pushback.pop()
    if stream.kind == "stdin":
        state = stdio_state(machine)
        byte = yield from state.stdin.getc(state.prompt)
    elif stream.pos < len(stream.buffer):
        byte = stream.buffer[stream.pos]
        stream.pos += 1
    else:
        byte = EOF
    if byte == EOF:
        stream.eof = True
    return byte


def unread_byte(machine: Machine, stream: StreamState, byte: int) -> None:
    """Push a byte back; stdin's pushback lives in the shared StdinBuffer."""
    if stream.kind == "stdin":
        stdin_buffer(machine).unget(byte)
    else:
        stream.pushback.append(byte)
    stream.eof = False


def write_console(machine: Machine, name: str, text: str) -> Events:
    """Write text to ``stdout`` or ``stderr`` (used by perror and assert)."""
    return (yield from write_stream(machine, standard_stream(machine, name), text.encode("utf-8")))


# ==================================================================== printf formatting

_PRINTF_SPEC = re.compile(
    rb"%(?P<flags>[-+ 0#]*)(?P<width>\*|\d+)?(?:\.(?P<precision>\*|\d*))?"
    rb"(?P<length>hh|h|ll|l|z|j|t|L)?(?P<conversion>[diouxXcspfFeEgGn%])")

_SIGNED_TYPES: dict[str, CType] = {"hh": SCHAR, "h": SHORT, "": INT}
_UNSIGNED_TYPES: dict[str, CType] = {"hh": UCHAR, "h": USHORT, "": UINT}


def _int_type(length: str, signed: bool) -> CType:
    """The integer type a conversion reads: ``%hhd`` is a signed char, ``%lu`` an
    unsigned long. ``l``, ``ll``, ``z``, ``j`` and ``t`` are all 64 bits here."""
    table = _SIGNED_TYPES if signed else _UNSIGNED_TYPES
    return table.get(length, LONG if signed else ULONG)


@dataclass
class FormatSpec:
    """One parsed conversion such as ``%-08.3lf``."""

    flags: str
    width: int | None
    precision: int | None
    length: str
    conversion: str

    @property
    def text(self) -> str:
        """The conversion as the user wrote it, for messages."""
        precision = "" if self.precision is None else f".{self.precision}"
        width = "" if self.width is None else str(self.width)
        return f"%{self.flags}{width}{precision}{self.length}{self.conversion}"


class _Arguments:
    """Hands out the variadic arguments one by one, warning when they run out."""

    def __init__(self, machine: Machine, values: list[Value], function: str,
                 first_number: int) -> None:
        self.machine = machine
        self.values = values
        self.function = function
        self.first_number = first_number
        self.index = 0

    @property
    def number(self) -> int:
        """The 1-based argument number of the value handed out last (for messages)."""
        return self.first_number + self.index - 1

    def next(self, spec_text: str) -> Value | None:
        """The next variadic argument, or None (with a warning) when there is none."""
        if self.index >= len(self.values):
            self.index += 1
            self.machine.warn(
                "format_argument_type",
                f"{self.function}(): format '{spec_text}' expects a matching argument, but "
                "there are not enough arguments",
                f"The format string has more % conversions than there are values after it. "
                f"'{spec_text}' has nothing to print, so a real program would print garbage. "
                "Pass one value per conversion.",
                Concept.STDIO)
            return None
        value = self.values[self.index]
        self.index += 1
        return value


def _kind_of(ctype: CType) -> str:
    """'int', 'float' or 'pointer': the categories a format mismatch is about."""
    if ctype.is_float:
        return "float"
    if ctype.is_pointer or isinstance(ctype, FunctionType):
        return "pointer"
    return "int"


_EXPECTED_NAMES = {"int": "int", "float": "double", "pointer": "a pointer"}


def _crosses_registers(value: Value, wanted: str) -> bool:
    """True when the value travels in the other register file than ``wanted`` reads.

    On a real machine variadic doubles go in floating-point registers and ints and
    pointers in integer registers. ``printf("%d", 4.5)`` therefore reads an integer
    register nobody wrote, so it prints whatever was there, never ``4``. An int passed
    for a pointer (or the reverse) shares the integer registers, so that mix-up keeps
    its value.
    """
    actual = _kind_of(value.ctype)
    return (actual == "float") != (wanted == "float")


def _garbage_rng(machine: Machine) -> random.Random:
    """The next seeded stream for a mismatched conversion (D: garbage is reproducible).

    The stream depends on the memory seed and on how many mismatches this run has
    printed, so the same program with the same seed prints the same garbage, while two
    mismatches in a loop still print different values like a real stale register.
    """
    state = machine.state.setdefault(STATE_KEY + ":garbage", {"count": 0})
    state["count"] += 1
    return random.Random(mix_seed(machine.memory.seed, f"printf-garbage:{state['count']}"))


def garbage_int(machine: Machine) -> int:
    """A seeded stand-in for the stale integer register ``%d`` reads after a double."""
    return _garbage_rng(machine).randint(-2**31, 2**31 - 1)


def garbage_double(machine: Machine) -> float:
    """A seeded stand-in for the stale floating register ``%f`` reads after an int.

    The value is kept to a readable size (real garbage can be 300 digits long), and it
    always has a fractional part, so it can never equal the int that was meant.
    """
    rng = _garbage_rng(machine)
    return rng.randint(-9_999_999, 9_999_999) + rng.randint(1, 999_999) / 1_000_000


def _integer_register(machine: Machine, value: Value) -> int:
    """What a conversion reading an integer register (%c, %s, %p) finds for ``value``.

    A double went into a floating register, so the integer register holds seeded
    garbage; ``printf("%c", 65.0)`` must not print ``A`` by accident.
    """
    if _crosses_registers(value, "int"):
        return garbage_int(machine)
    return int(value.raw) if math.isfinite(value.raw) else 0


def _as_pointer_register(machine: Machine, value: Value) -> Value:
    """``value`` as the address %s or %p reads; a double becomes a garbage address."""
    if not _crosses_registers(value, "pointer"):
        return value
    # The garbage is an unsigned 32-bit pattern, never NULL by accident, so %s with
    # a double cannot print "(null)" and look like a sensible answer.
    address = (garbage_int(machine) & 0xFFFF_FFFF) or 1
    return Value(PointerType(UCHAR), address)


def _check_kind(args: _Arguments, value: Value, spec: FormatSpec, wanted: str) -> None:
    """Warn (gcc's -Wformat wording) when an argument's category does not fit."""
    actual = _kind_of(value.ctype)
    if actual == wanted:
        return
    args.machine.warn(
        "format_argument_type",
        f"format '{spec.text}' expects argument of type '{_EXPECTED_NAMES[wanted]}', but "
        f"argument {args.number} has type '{format_type(value.ctype)}'",
        f"'{spec.text}' prints {_EXPECTED_NAMES[wanted]} values, but argument {args.number} "
        f"is a {format_type(value.ctype)}. A real program prints a wrong or garbage value "
        "here. Use the conversion that matches the type (%d int, %f double, %s string, "
        "%p pointer).",
        Concept.STDIO)


def _pad(text: bytes, width: int | None, left: bool) -> bytes:
    """Pad to ``width`` with spaces, on the right for the '-' flag."""
    if width is None or len(text) >= width:
        return text
    fill = b" " * (width - len(text))
    return text + fill if left else fill + text


def format_integer(value: int, spec: FormatSpec) -> bytes:
    """Render an already-wrapped integer for d i u o x X with C's flag rules."""
    conversion = spec.conversion
    negative = value < 0
    magnitude = -value if negative else value
    digits = {"o": f"{magnitude:o}", "x": f"{magnitude:x}", "X": f"{magnitude:X}"}.get(
        conversion, str(magnitude))
    if spec.precision is not None:
        # An explicit precision is a minimum digit count; '%.0d' of 0 prints nothing.
        digits = "" if spec.precision == 0 and magnitude == 0 else digits.zfill(spec.precision)
    prefix = ""
    if "#" in spec.flags:
        if conversion == "o" and not digits.startswith("0"):
            digits = "0" + digits
        elif conversion in "xX" and magnitude != 0:
            prefix = "0" + conversion
    if conversion in "di":
        sign = "-" if negative else "+" if "+" in spec.flags else " " if " " in spec.flags else ""
    else:
        sign = ""
    head = sign + prefix
    width = spec.width or 0
    if "-" not in spec.flags and "0" in spec.flags and spec.precision is None:
        digits = digits.rjust(width - len(head), "0")
    return _pad((head + digits).encode("ascii"), spec.width, "-" in spec.flags)


def format_float(value: float, spec: FormatSpec) -> bytes:
    """Render a double for f F e E g G.

    Python's ``%`` operator follows C's rules for finite numbers; infinities and NaN
    are spelled by hand because C writes ``-nan`` and never zero-pads them.
    """
    if not math.isfinite(value):
        negative = math.copysign(1.0, value) < 0
        sign = "-" if negative else "+" if "+" in spec.flags else " " if " " in spec.flags else ""
        text = sign + ("nan" if math.isnan(value) else "inf")
        if spec.conversion.isupper():
            text = text.upper()
        return _pad(text.encode("ascii"), spec.width, "-" in spec.flags)
    precision = 6 if spec.precision is None else spec.precision
    width = "" if spec.width is None else str(spec.width)
    python_format = f"%{spec.flags}{width}.{precision}{spec.conversion}"
    return (python_format % value).encode("ascii")


def _format_pointer(value: Value, spec: FormatSpec) -> bytes:
    """``%p``: ``0x`` and lower-case hex, or ``(nil)`` for NULL, as glibc prints."""
    addr = int(value.raw)
    target = value.ctype.target if isinstance(value.ctype, PointerType) else None
    if isinstance(value.ctype, FunctionType) or isinstance(target, FunctionType):
        addr = function_address(addr)
    text = b"(nil)" if addr == 0 else f"0x{addr:x}".encode("ascii")
    return _pad(text, spec.width, "-" in spec.flags)


def _format_string(machine: Machine, value: Value | None, spec: FormatSpec,
                   args: _Arguments) -> bytes:
    """``%s``: the C string, cut to the precision; ``(null)`` for NULL like glibc."""
    addr = int(value.raw) if value is not None else 0
    if addr == 0:
        text = b"(null)" if spec.precision is None or spec.precision >= 6 else b""
    else:
        text = read_c_string(machine, addr, args.function, f"argument {args.number} (%s)",
                             limit=spec.precision)
    return _pad(text, spec.width, "-" in spec.flags)


def _parse_spec(match: re.Match[bytes], args: _Arguments) -> FormatSpec:
    """Build a FormatSpec, fetching ``*`` width and precision from the arguments."""
    flags = match["flags"].decode("ascii")
    width: int | None = None
    if match["width"] == b"*":
        value = args.next("*")
        width = int(value.raw) if value is not None else 0
        if width < 0:            # a negative * width means "left-justify"
            flags += "-"
            width = -width
    elif match["width"] is not None:
        width = int(match["width"])
    precision: int | None = None
    if match["precision"] == b"*":
        value = args.next(".*")
        precision = int(value.raw) if value is not None else 0
        if precision < 0:        # a negative * precision counts as none at all
            precision = None
    elif match["precision"] is not None:
        precision = int(match["precision"] or b"0")
    length = (match["length"] or b"").decode("ascii")
    return FormatSpec(flags, width, precision, length, match["conversion"].decode("ascii"))


def _convert(machine: Machine, spec: FormatSpec, args: _Arguments, written: int) -> bytes:
    """Produce the bytes for one conversion."""
    conversion = spec.conversion
    if conversion == "%":
        return b"%"
    value = args.next(spec.text)
    if conversion in "diouxX":
        raw = 0
        if value is not None:
            _check_kind(args, value, spec, "int")
            if _crosses_registers(value, "int"):
                raw = garbage_int(machine)
            else:
                raw = int(value.raw) if math.isfinite(value.raw) else 0
        return format_integer(wrap_int(raw, _int_type(spec.length, conversion in "di")), spec)
    if conversion in "fFeEgG":
        number = 0.0
        if value is not None:
            _check_kind(args, value, spec, "float")
            if _crosses_registers(value, "float"):
                number = garbage_double(machine)
            else:
                number = float(value.raw)
        return format_float(number, spec)
    if conversion == "c":
        byte = 0
        if value is not None:
            _check_kind(args, value, spec, "int")
            byte = _integer_register(machine, value) & 0xFF
        return _pad(bytes([byte]), spec.width, "-" in spec.flags)
    if conversion == "s":
        if value is not None:
            _check_kind(args, value, spec, "pointer")
            value = _as_pointer_register(machine, value)
        return _format_string(machine, value, spec, args)
    if conversion == "p":
        if value is None:
            return b"(nil)"
        if _crosses_registers(value, "pointer"):
            # Only a double is reported here: an int for %p shares the register and
            # is a common, harmless habit (printf("%p", 0)) the checker already flags.
            _check_kind(args, value, spec, "pointer")
        return _format_pointer(_as_pointer_register(machine, value), spec)
    # %n stores how many bytes have been produced so far and prints nothing.
    if value is not None:
        require_pointer(machine, int(value.raw), args.function, f"argument {args.number} (%n)")
        machine.memory.store(int(value.raw), _int_type(spec.length, True), written)
    return b""


def format_printf(machine: Machine, fmt: bytes, values: list[Value], function: str,
                  first_number: int) -> bytes:
    """Expand a printf format string with its arguments into the output bytes.

    ``first_number`` is the 1-based position of the first variadic argument in the
    call (2 for printf), so warnings can say "argument 3" like gcc does.
    """
    args = _Arguments(machine, values, function, first_number)
    out = bytearray()
    pos = 0
    while True:
        percent = fmt.find(b"%", pos)
        if percent < 0:
            out += fmt[pos:]
            return bytes(out)
        out += fmt[pos:percent]
        match = _PRINTF_SPEC.match(fmt, percent)
        if match is None:
            out += b"%"          # glibc prints an unknown conversion as written
            pos = percent + 1
            continue
        spec = _parse_spec(match, args)
        out += _convert(machine, spec, args, len(out))
        pos = match.end()


def _format_from_args(machine: Machine, args: list[Value], index: int, function: str) -> bytes:
    """Read the format string at ``args[index]`` and expand it with what follows."""
    fmt = read_c_string(machine, args[index].raw, function, "the format string")
    return format_printf(machine, fmt, args[index + 1:], function, index + 2)


# ==================================================================== scanf parsing

class _Reader(Protocol):
    """Where scanf takes its characters from: a stream or a string."""

    consumed: int

    def peek(self) -> Generator[Any, Any, int]:
        """The next byte without consuming it (``EOF`` at the end)."""
        ...

    def take(self) -> Generator[Any, Any, int]:
        """Consume and return the next byte (``EOF`` at the end)."""
        ...


class _StreamReader:
    """Characters from a FILE (stdin may ask the host for more)."""

    def __init__(self, machine: Machine, stream: StreamState) -> None:
        self.machine = machine
        self.stream = stream
        self.consumed = 0

    def peek(self) -> Generator[Any, Any, int]:
        """The next byte without consuming it (``EOF`` at the end)."""
        byte = yield from read_byte(self.machine, self.stream)
        if byte != EOF:
            unread_byte(self.machine, self.stream, byte)
        return byte

    def take(self) -> Generator[Any, Any, int]:
        """Consume and return the next byte (``EOF`` at the end)."""
        byte = yield from read_byte(self.machine, self.stream)
        if byte != EOF:
            self.consumed += 1
        return byte


class _BytesReader:
    """Characters from a C string (sscanf). It never asks the host for anything."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.consumed = 0

    def peek(self) -> Generator[Any, Any, int]:
        """The next byte without consuming it (``EOF`` at the end)."""
        yield from ()            # a generator like the stream reader, without events
        return self.data[self.consumed] if self.consumed < len(self.data) else EOF

    def take(self) -> Generator[Any, Any, int]:
        """Consume and return the next byte (``EOF`` at the end)."""
        byte = yield from self.peek()
        if byte != EOF:
            self.consumed += 1
        return byte


_SCAN_SPEC = re.compile(
    rb"%(?P<suppress>\*)?(?P<width>\d+)?(?P<length>hh|h|ll|l|z|j|t|L)?"
    rb"(?P<conversion>[diouxXcsfeEgGpn]|\[\^?\]?[^\]]*\])")


def _skip_space(reader: _Reader) -> Events:
    """A space in the format (and most conversions) skips any amount of white space."""
    while (yield from reader.peek()) in SPACE_BYTES:
        yield from reader.take()


def _digit_value(byte: int) -> int:
    """0-35 for 0-9 a-z A-Z, 99 for anything else."""
    char = chr(byte) if byte >= 0 else ""
    if char.isdigit():
        return ord(char) - ord("0")
    if "a" <= char.lower() <= "z":
        return ord(char.lower()) - ord("a") + 10
    return 99


def _scan_integer(reader: _Reader, base: int, width: int | None) -> Generator[Any, Any, int | None]:
    """Read an integer like strtol, never more than ``width`` characters."""
    limit = width or _UNLIMITED
    taken = 0
    negative = False
    byte = yield from reader.peek()
    if byte in SIGNS and taken < limit:
        negative = byte == ord("-")
        yield from reader.take()
        taken += 1
        byte = yield from reader.peek()
    seen_digit = False
    if base in (0, 16) and byte == ord("0") and taken < limit:
        yield from reader.take()
        taken += 1
        seen_digit = True
        byte = yield from reader.peek()
        if byte in HEX_MARKS and taken < limit:
            yield from reader.take()
            taken += 1
            base = 16
            byte = yield from reader.peek()
        elif base == 0:
            base = 8
    if base == 0:
        base = 10
    value = 0
    while taken < limit and byte != EOF and _digit_value(byte) < base:
        value = value * base + _digit_value(byte)
        seen_digit = True
        yield from reader.take()
        taken += 1
        byte = yield from reader.peek()
    if not seen_digit:
        return None
    return -value if negative else value


def _scan_float(reader: _Reader, width: int | None) -> Generator[Any, Any, float | None]:
    """Read a decimal floating-point number (sign, digits, point, exponent)."""
    limit = width or _UNLIMITED
    text = bytearray()

    def take_if(allowed: frozenset[int]) -> Generator[Any, Any, bool]:
        if len(text) < limit and (yield from reader.peek()) in allowed:
            text.append((yield from reader.take()))
            return True
        return False

    def take_digits() -> Generator[Any, Any, int]:
        count = 0
        while (yield from take_if(DECIMAL_DIGITS)):
            count += 1
        return count

    yield from take_if(SIGNS)
    digits = yield from take_digits()
    if (yield from take_if(frozenset(b"."))):
        digits += yield from take_digits()
    if digits == 0:
        return None
    if (yield from take_if(EXPONENT_MARKS)):
        yield from take_if(SIGNS)
        yield from take_digits()
    # An exponent marker without digits ("1e") is not part of the number.
    cleaned = bytes(text).rstrip(b"+-").rstrip(b"eE")
    return float(cleaned)


def _parse_scanset(spec: bytes) -> tuple[frozenset[int], bool]:
    """``[abc]``, ``[^\\n]``, ``[a-z]``: the member bytes and whether it is negated."""
    body = spec[1:-1]
    negated = body.startswith(b"^")
    if negated:
        body = body[1:]
    members: set[int] = set()
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == ord("-"):
            members.update(range(body[index], body[index + 2] + 1))
            index += 3
        else:
            members.add(body[index])
            index += 1
    return frozenset(members), negated


def _scan_chars(reader: _Reader, width: int | None, accept: Any) -> Generator[Any, Any, bytes]:
    """Take characters while ``accept(byte)`` holds, at most ``width`` of them."""
    limit = width or _UNLIMITED
    data = bytearray()
    while len(data) < limit:
        byte = yield from reader.peek()
        if byte == EOF or not accept(byte):
            break
        data.append((yield from reader.take()))
    return bytes(data)


def _scan_one(reader: _Reader, conversion: str, width: int | None) -> Generator[Any, Any, Any]:
    """Read one conversion's value; None means a matching failure."""
    if conversion in "dui":
        return (yield from _scan_integer(reader, 0 if conversion == "i" else 10, width))
    if conversion == "o":
        return (yield from _scan_integer(reader, 8, width))
    if conversion in "xXp":
        return (yield from _scan_integer(reader, 16, width))
    if conversion in "feEgG":
        return (yield from _scan_float(reader, width))
    if conversion == "c":
        wanted = width or 1
        data = yield from _scan_chars(reader, wanted, lambda byte: True)
        return data if len(data) == wanted else None
    if conversion == "s":
        data = yield from _scan_chars(reader, width, lambda byte: byte not in SPACE_BYTES)
        return data or None
    members, negated = _parse_scanset(conversion.encode("latin-1"))
    data = yield from _scan_chars(reader, width, lambda byte: (byte in members) != negated)
    return data or None


def _store_scanned(machine: Machine, target: Value, conversion: str, length: str,
                   value: Any, function: str, number: int) -> None:
    """Store a converted value through the pointer argument, typed by the conversion."""
    addr = int(target.raw)
    require_pointer(machine, addr, function, f"argument {number}")
    if conversion in "di":
        machine.memory.store(addr, _int_type(length, True), value)
    elif conversion in "uoxX":
        machine.memory.store(addr, _int_type(length, False), value)
    elif conversion == "p":
        machine.memory.store(addr, PointerType(VOID), value)
    elif conversion in "feEgG":
        machine.memory.store(addr, DOUBLE if length in ("l", "L") else FLOAT, value)
    elif conversion == "c":
        machine.write_bytes(addr, value)
    else:                         # %s and %[...] add the terminating '\0'
        machine.write_bytes(addr, value + b"\0")


# The outcomes of one step of a scanf format.
_STORED = "stored"            # a value was converted and stored
_SKIPPED = "skipped"          # matched, but nothing counted (literal, %*d, %n)
_MISMATCH = "mismatch"        # the input does not fit: stop, keep the count
_INPUT_END = "input_end"      # the input ran out: stop, EOF if nothing was stored


def _match_literal(reader: _Reader, byte: int) -> Generator[Any, Any, str]:
    """An ordinary format character must appear in the input."""
    got = yield from reader.peek()
    if got == EOF:
        return _INPUT_END
    if got != byte:
        return _MISMATCH
    yield from reader.take()
    return _SKIPPED


class _ScanTargets:
    """The pointer arguments of a scanf call, handed out in order."""

    def __init__(self, machine: Machine, targets: list[Value], function: str,
                 first_number: int) -> None:
        self.machine = machine
        self.targets = targets
        self.function = function
        self.first_number = first_number
        self.index = 0

    def store(self, conversion: str, length: str, value: Any) -> bool:
        """Store through the next pointer; False (with a warning) if there is none."""
        if self.index >= len(self.targets):
            self.machine.warn(
                "format_argument_type",
                f"{self.function}(): format '%{conversion}' has no matching pointer argument",
                "The format string has more % conversions than there are pointers after "
                "it, so the value has nowhere to go. Pass one &variable per conversion.",
                Concept.STDIO)
            return False
        _store_scanned(self.machine, self.targets[self.index], conversion, length, value,
                       self.function, self.first_number + self.index)
        self.index += 1
        return True


def _scan_conversion(reader: _Reader, match: re.Match[bytes],
                     targets: _ScanTargets) -> Generator[Any, Any, str]:
    """Carry out one ``%`` conversion of a scanf format."""
    conversion = match["conversion"].decode("latin-1")
    length = (match["length"] or b"").decode("ascii")
    width = int(match["width"]) if match["width"] else None
    suppress = match["suppress"] is not None
    if conversion == "n":
        # %n reports how many characters were consumed; it is never counted.
        if not suppress:
            targets.store("d", length, reader.consumed)
        return _SKIPPED
    if conversion != "c" and not conversion.startswith("["):
        yield from _skip_space(reader)
    if (yield from reader.peek()) == EOF:
        return _INPUT_END
    value = yield from _scan_one(reader, conversion, width)
    if value is None:
        return _MISMATCH
    if suppress:
        return _SKIPPED
    return _STORED if targets.store(conversion, length, value) else _MISMATCH


def scan_format(machine: Machine, reader: _Reader, fmt: bytes, targets: list[Value],
                function: str, first_number: int) -> Generator[Any, Any, int]:
    """Run a scanf format over ``reader``; returns the number of values stored.

    Returns ``EOF`` when the input ends before the first conversion. A character that
    does not match stays unread, which is why ``scanf("%d", &n)`` on "abc" returns 0
    and leaves "abc" waiting for the next read.
    """
    pointers = _ScanTargets(machine, targets, function, first_number)
    stored = 0
    pos = 0
    while pos < len(fmt):
        byte = fmt[pos]
        if byte in SPACE_BYTES:
            # Any white space in the format skips any amount (even none) in the input.
            yield from _skip_space(reader)
            pos += 1
            continue
        if fmt.startswith(b"%%", pos):
            yield from _skip_space(reader)
            outcome = yield from _match_literal(reader, byte)
            pos += 2
        elif byte != ord("%"):
            outcome = yield from _match_literal(reader, byte)
            pos += 1
        else:
            match = _SCAN_SPEC.match(fmt, pos)
            if match is None:
                return stored
            pos = match.end()
            outcome = yield from _scan_conversion(reader, match, pointers)
        if outcome == _STORED:
            stored += 1
        elif outcome == _MISMATCH:
            return stored
        elif outcome == _INPUT_END:
            return stored if stored else EOF
    return stored


# ==================================================================== formatted output

def c_printf(machine: Machine, args: list[Value]) -> Events:
    """``int printf(const char *format, ...)``: format to stdout; returns bytes written."""
    data = _format_from_args(machine, args, 0, "printf")
    ok = yield from write_stream(machine, standard_stream(machine, "stdout"), data)
    return machine.make_int(len(data) if ok else EOF)


def c_fprintf(machine: Machine, args: list[Value]) -> Events:
    """``int fprintf(FILE *stream, const char *format, ...)``."""
    stream = stream_arg(machine, args[0], "fprintf")
    data = _format_from_args(machine, args, 1, "fprintf")
    ok = yield from write_stream(machine, stream, data)
    return machine.make_int(len(data) if ok else EOF)


def c_sprintf(machine: Machine, args: list[Value]) -> Value:
    """``int sprintf(char *str, const char *format, ...)``: no size check, like C.

    The memory model still traps the overflow, naming the buffer that was too small.
    """
    machine.count_construct(constructs.PRINTF)
    data = _format_from_args(machine, args, 1, "sprintf")
    require_pointer(machine, args[0].raw, "sprintf", "the destination buffer")
    machine.write_bytes(args[0].raw, data + b"\0")
    return machine.make_int(len(data))


def c_snprintf(machine: Machine, args: list[Value]) -> Value:
    """``int snprintf(char *str, size_t size, const char *format, ...)``.

    Writes at most ``size - 1`` bytes plus ``'\\0'`` and returns the length the whole
    text *would* have had, so ``if (n >= size)`` detects truncation.
    """
    machine.count_construct(constructs.PRINTF)
    data = _format_from_args(machine, args, 2, "snprintf")
    size = args[1].raw
    if size > 0:
        require_pointer(machine, args[0].raw, "snprintf", "the destination buffer")
        machine.write_bytes(args[0].raw, data[:size - 1] + b"\0")
    return machine.make_int(len(data))


# ==================================================================== character output

def c_puts(machine: Machine, args: list[Value]) -> Events:
    """``int puts(const char *s)``: the string and a newline to stdout."""
    text = read_c_string(machine, args[0].raw, "puts", "the string")
    ok = yield from write_stream(machine, standard_stream(machine, "stdout"), text + b"\n")
    return machine.make_int(len(text) + 1 if ok else EOF)


def c_putchar(machine: Machine, args: list[Value]) -> Events:
    """``int putchar(int c)``: one byte to stdout; returns it as unsigned char."""
    byte = args[0].raw & 0xFF
    ok = yield from write_stream(machine, standard_stream(machine, "stdout"), bytes([byte]))
    return machine.make_int(byte if ok else EOF)


def c_fputs(machine: Machine, args: list[Value]) -> Events:
    """``int fputs(const char *s, FILE *stream)``: no newline added; 1 on success."""
    text = read_c_string(machine, args[0].raw, "fputs", "the string")
    stream = stream_arg(machine, args[1], "fputs")
    ok = yield from write_stream(machine, stream, text)
    return machine.make_int(1 if ok else EOF)


def _fputc(machine: Machine, args: list[Value], function: str) -> Events:
    byte = args[0].raw & 0xFF
    stream = stream_arg(machine, args[1], function)
    ok = yield from write_stream(machine, stream, bytes([byte]))
    return machine.make_int(byte if ok else EOF)


def c_fputc(machine: Machine, args: list[Value]) -> Events:
    """``int fputc(int c, FILE *stream)``."""
    return (yield from _fputc(machine, args, "fputc"))


def c_putc(machine: Machine, args: list[Value]) -> Events:
    """``int putc(int c, FILE *stream)``: the same as fputc."""
    return (yield from _fputc(machine, args, "putc"))


def c_perror(machine: Machine, args: list[Value]) -> Events:
    """``void perror(const char *s)``: "s: <strerror(errno)>" on stderr."""
    message = c_strerror(machine.get_errno())
    prefix = b""
    if args[0].raw != 0:
        prefix = read_c_string(machine, args[0].raw, "perror", "the message")
    text = f"{prefix.decode('utf-8', 'replace')}: {message}\n" if prefix else f"{message}\n"
    yield from write_console(machine, "stderr", text)
    return None


# ==================================================================== formatted input

def c_scanf(machine: Machine, args: list[Value]) -> Events:
    """``int scanf(const char *format, ...)``: read from stdin."""
    fmt = read_c_string(machine, args[0].raw, "scanf", "the format string")
    machine.count_construct(constructs.SCANF)
    reader = _StreamReader(machine, standard_stream(machine, "stdin"))
    count = yield from scan_format(machine, reader, fmt, args[1:], "scanf", 2)
    return machine.make_int(count)


def c_fscanf(machine: Machine, args: list[Value]) -> Events:
    """``int fscanf(FILE *stream, const char *format, ...)``."""
    stream = stream_arg(machine, args[0], "fscanf")
    fmt = read_c_string(machine, args[1].raw, "fscanf", "the format string")
    machine.count_construct(constructs.SCANF)
    if stream.kind == "file":
        machine.count_construct(constructs.FILE_READ)
    count = yield from scan_format(machine, _StreamReader(machine, stream), fmt, args[2:],
                                   "fscanf", 3)
    return machine.make_int(count)


def c_sscanf(machine: Machine, args: list[Value]) -> Events:
    """``int sscanf(const char *str, const char *format, ...)``: read from a string."""
    data = read_c_string(machine, args[0].raw, "sscanf", "the input string")
    fmt = read_c_string(machine, args[1].raw, "sscanf", "the format string")
    machine.count_construct(constructs.SCANF)
    count = yield from scan_format(machine, _BytesReader(data), fmt, args[2:], "sscanf", 3)
    return machine.make_int(count)


# ==================================================================== character input

def c_fgets(machine: Machine, args: list[Value]) -> Events:
    """``char *fgets(char *s, int size, FILE *stream)``: one line, newline included.

    Reads at most ``size - 1`` bytes, stops after a newline, always adds ``'\\0'``.
    Returns NULL when end of file comes before any byte.
    """
    buf, size = args[0].raw, args[1].raw
    stream = stream_arg(machine, args[2], "fgets")
    require_pointer(machine, buf, "fgets", "the buffer")
    _count_read(machine, stream)
    if size <= 0:
        return char_pointer(machine, 0)
    line = bytearray()
    while len(line) < size - 1:
        byte = yield from read_byte(machine, stream)
        if byte == EOF:
            break
        line.append(byte)
        if byte == ord("\n"):
            break
    if not line and size > 1:
        return char_pointer(machine, 0)
    machine.write_bytes(buf, bytes(line) + b"\0")
    return char_pointer(machine, buf)


def _fgetc(machine: Machine, stream: StreamState) -> Events:
    _count_read(machine, stream)
    byte = yield from read_byte(machine, stream)
    return machine.make_int(byte)


def c_fgetc(machine: Machine, args: list[Value]) -> Events:
    """``int fgetc(FILE *stream)``: the next byte as unsigned char, or EOF."""
    return (yield from _fgetc(machine, stream_arg(machine, args[0], "fgetc")))


def c_getc(machine: Machine, args: list[Value]) -> Events:
    """``int getc(FILE *stream)``: the same as fgetc."""
    return (yield from _fgetc(machine, stream_arg(machine, args[0], "getc")))


def c_getchar(machine: Machine, args: list[Value]) -> Events:
    """``int getchar(void)``: the next byte of stdin, or EOF."""
    return (yield from _fgetc(machine, standard_stream(machine, "stdin")))


def c_ungetc(machine: Machine, args: list[Value]) -> Value:
    """``int ungetc(int c, FILE *stream)``: push one byte back to be read again."""
    stream = stream_arg(machine, args[1], "ungetc")
    if args[0].raw == EOF:
        return machine.make_int(EOF)
    byte = args[0].raw & 0xFF
    unread_byte(machine, stream, byte)
    return machine.make_int(byte)


# ==================================================================== files

def _parse_mode(text: str) -> str | None:
    """"rb+" -> "r+"; None for a mode fopen does not accept."""
    mode = text.replace("b", "")
    return mode if mode in VALID_MODES else None


def _open_contents(files: Any, path: str, mode: str) -> bytes | int:
    """The file's current bytes after opening it in ``mode``, or a negative errno.

    "r" needs an existing file; "w" truncates or creates; "a" creates but keeps the
    contents. Writing an empty append probes write permission without changing data.
    """
    if files.is_dir(path):
        return -EISDIR
    if mode.startswith("r"):
        data = files.read(path)
        if isinstance(data, int) or mode == "r":
            return data
        probe = files.write(path, b"", True)
        return probe if probe < 0 else data
    if mode.startswith("w"):
        result = files.write(path, b"", False)
        return result if result < 0 else b""
    data = files.read(path)
    if isinstance(data, int):
        if data != -ENOENT:
            return data
        data = b""
    result = files.write(path, b"", True)
    return result if result < 0 else data


def c_fopen(machine: Machine, args: list[Value]) -> Value:
    """``FILE *fopen(const char *pathname, const char *mode)``.

    On failure returns NULL and sets errno (ENOENT, EACCES, EISDIR, EINVAL), which is
    what ``perror("fopen")`` then explains.
    """
    machine.count_construct(constructs.FILE_OPEN)
    name = read_c_string(machine, args[0].raw, "fopen", "the path").decode("utf-8", "replace")
    mode = _parse_mode(read_c_string(machine, args[1].raw, "fopen", "the mode").decode(
        "latin-1"))
    if mode is None:
        machine.set_errno(EINVAL)
        return machine.make_pointer(0, FILE_TYPE)
    files = _files(machine)
    if files is None:
        machine.set_errno(ENOENT)
        return machine.make_pointer(0, FILE_TYPE)
    path = _resolve(machine, name)
    contents = _open_contents(files, path, mode)
    if isinstance(contents, int):
        machine.set_errno(-contents)
        return machine.make_pointer(0, FILE_TYPE)
    state = stdio_state(machine)
    addr = _file_object(machine, state, name)
    machine.memory.store(addr, INT, state.next_id)
    state.streams[addr] = StreamState(state.next_id, addr, path, mode, buffer=bytearray(contents))
    state.next_id += 1
    return machine.make_pointer(addr, FILE_TYPE)


def _file_object(machine: Machine, state: StdioState, name: str) -> int:
    """The address for a new FILE: a long-closed one if there is one, else a new one.

    glibc reuses the memory of closed FILEs too; without reuse every fopen would
    take another slice of the data segment until a long-running program trapped.
    """
    if len(state.closed_files) > _FILE_QUARANTINE:
        return state.closed_files.popleft()
    return machine.memory.alloc_global(FILE_TYPE.size, FILE_TYPE.align, f"FILE {name}",
                                       FILE_TYPE)


def c_fclose(machine: Machine, args: list[Value]) -> Value:
    """``int fclose(FILE *stream)``: the FILE pointer is invalid afterwards."""
    stream = stream_arg(machine, args[0], "fclose")
    stream.closed = True
    if stream.kind == "file":
        stdio_state(machine).closed_files.append(stream.addr)
    return machine.make_int(0)


def c_fread(machine: Machine, args: list[Value]) -> Events:
    """``size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)``: returns
    the number of whole items read."""
    ptr, size, count = args[0].raw, args[1].raw, args[2].raw
    stream = stream_arg(machine, args[3], "fread")
    _count_read(machine, stream)
    total = size * count
    if total == 0:
        return machine.make_int(0, SIZE_T)
    require_pointer(machine, ptr, "fread", "the buffer")
    data = bytearray()
    while len(data) < total:
        byte = yield from read_byte(machine, stream)
        if byte == EOF:
            break
        data.append(byte)
    machine.write_bytes(ptr, bytes(data))
    return machine.make_int(len(data) // size, SIZE_T)


def c_fwrite(machine: Machine, args: list[Value]) -> Events:
    """``size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)``."""
    ptr, size, count = args[0].raw, args[1].raw, args[2].raw
    stream = stream_arg(machine, args[3], "fwrite")
    total = size * count
    if total == 0:
        return machine.make_int(0, SIZE_T)
    require_pointer(machine, ptr, "fwrite", "the data")
    ok = yield from write_stream(machine, stream, machine.read_bytes(ptr, total))
    return machine.make_int(count if ok else 0, SIZE_T)


def c_feof(machine: Machine, args: list[Value]) -> Value:
    """``int feof(FILE *stream)``: non-zero once a read has hit the end."""
    return machine.make_int(1 if stream_arg(machine, args[0], "feof").eof else 0)


def c_ferror(machine: Machine, args: list[Value]) -> Value:
    """``int ferror(FILE *stream)``: non-zero once a read or write has failed."""
    return machine.make_int(1 if stream_arg(machine, args[0], "ferror").error else 0)


def c_clearerr(machine: Machine, args: list[Value]) -> None:
    """``void clearerr(FILE *stream)``: reset the end-of-file and error flags."""
    stream = stream_arg(machine, args[0], "clearerr")
    stream.eof = stream.error = False


def c_fflush(machine: Machine, args: list[Value]) -> Value:
    """``int fflush(FILE *stream)``: writes are already write-through, so this only
    validates the stream. ``fflush(NULL)`` is allowed and flushes everything."""
    if args[0].raw != 0:
        stream_arg(machine, args[0], "fflush")
    return machine.make_int(0)


def _path_call(machine: Machine, result: int) -> Value:
    """0 on success; -1 and errno for a negative HostFiles result."""
    if result < 0:
        machine.set_errno(-result)
        return machine.make_int(-1)
    return machine.make_int(0)


def c_remove(machine: Machine, args: list[Value]) -> Value:
    """``int remove(const char *pathname)``: delete a file."""
    name = read_c_string(machine, args[0].raw, "remove", "the path").decode("utf-8", "replace")
    files = _files(machine)
    if files is None:
        return _path_call(machine, -ENOENT)
    return _path_call(machine, files.remove(_resolve(machine, name)))


def c_rename(machine: Machine, args: list[Value]) -> Value:
    """``int rename(const char *oldpath, const char *newpath)``: move a file."""
    old = read_c_string(machine, args[0].raw, "rename", "the old path").decode("utf-8", "replace")
    new = read_c_string(machine, args[1].raw, "rename", "the new path").decode("utf-8", "replace")
    files = _files(machine)
    if files is None:
        return _path_call(machine, -ENOENT)
    return _path_call(machine, files.rename(_resolve(machine, old), _resolve(machine, new)))


# ==================================================================== end of run

def finalize(machine: Machine) -> None:
    """Called by the interpreter when the program ends: warn about files left open."""
    state = machine.state.get(STATE_KEY)
    if state is None:
        return
    for stream in state.streams.values():
        if stream.kind == "file" and not stream.closed:
            machine.warn(
                "file_not_closed",
                f"file '{stream.path}' was opened with fopen() but never closed",
                f"'{stream.path}' was still open when the program ended. Every fopen needs "
                "a matching fclose; on a real system unclosed files can lose their last "
                "writes and the program can run out of file handles.",
                Concept.FILE_IO)


# ==================================================================== registration

def _standard_pointer_bytes(name: str) -> Any:
    """The initialiser of ``stdin``/``stdout``/``stderr``: a pointer to its FILE."""
    def initial_bytes(machine: Machine) -> bytes:
        return encode(standard_stream(machine, name).addr, FILE_PTR)
    return initial_bytes


def _io(name: str, prototype: str, impl: Any) -> LibraryFunction:
    return LibraryFunction(name, prototype, impl, Concept.FILE_IO)


def _out(name: str, prototype: str, impl: Any, format_arg: int | None = None,
         format_kind: str | None = None) -> LibraryFunction:
    return LibraryFunction(name, prototype, impl, Concept.STDIO, format_arg, format_kind)


FUNCTIONS: tuple[LibraryFunction, ...] = (
    _out("printf", "int printf(const char *format, ...);", c_printf, 0, "printf"),
    _out("fprintf", "int fprintf(FILE *stream, const char *format, ...);", c_fprintf, 1,
         "printf"),
    _out("sprintf", "int sprintf(char *str, const char *format, ...);", c_sprintf, 1, "printf"),
    _out("snprintf", "int snprintf(char *str, size_t size, const char *format, ...);",
         c_snprintf, 2, "printf"),
    _out("puts", "int puts(const char *s);", c_puts),
    _out("putchar", "int putchar(int c);", c_putchar),
    _io("fputs", "int fputs(const char *s, FILE *stream);", c_fputs),
    _io("fputc", "int fputc(int c, FILE *stream);", c_fputc),
    _io("putc", "int putc(int c, FILE *stream);", c_putc),
    _out("scanf", "int scanf(const char *format, ...);", c_scanf, 0, "scanf"),
    _out("fscanf", "int fscanf(FILE *stream, const char *format, ...);", c_fscanf, 1, "scanf"),
    _out("sscanf", "int sscanf(const char *str, const char *format, ...);", c_sscanf, 1,
         "scanf"),
    _io("fgets", "char *fgets(char *s, int size, FILE *stream);", c_fgets),
    _io("fgetc", "int fgetc(FILE *stream);", c_fgetc),
    _io("getc", "int getc(FILE *stream);", c_getc),
    _out("getchar", "int getchar(void);", c_getchar),
    _io("ungetc", "int ungetc(int c, FILE *stream);", c_ungetc),
    _io("fopen", "FILE *fopen(const char *pathname, const char *mode);", c_fopen),
    _io("fclose", "int fclose(FILE *stream);", c_fclose),
    _io("fread", "size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);", c_fread),
    _io("fwrite", "size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);",
        c_fwrite),
    _io("feof", "int feof(FILE *stream);", c_feof),
    _io("ferror", "int ferror(FILE *stream);", c_ferror),
    _io("clearerr", "void clearerr(FILE *stream);", c_clearerr),
    _io("fflush", "int fflush(FILE *stream);", c_fflush),
    _io("remove", "int remove(const char *pathname);", c_remove),
    _io("rename", "int rename(const char *oldpath, const char *newpath);", c_rename),
    LibraryFunction("perror", "void perror(const char *s);", c_perror, Concept.ERROR_HANDLING),
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<stdio.h>`` function and the three standard streams."""
    register_functions(registry, HEADER, FUNCTIONS)
    for name in STANDARD_STREAMS:
        registry.register_variable(name, f"FILE *{name};", _standard_pointer_bytes(name),
                                   header=HEADER)
