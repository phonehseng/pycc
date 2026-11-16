"""Adversarial review tests for <stdio.h>: printf corners, scanf, fgets and files."""
from __future__ import annotations

import pytest

from pycc.interp.c_types import CHAR, INT, LONG
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.traps import TrapError
from pycc.interp.values import Value


def _arg(machine: FakeMachine, arg: object) -> Value:
    """Python test values -> C argument values (str becomes a string literal)."""
    if isinstance(arg, Value):
        return arg
    if isinstance(arg, int):
        return machine.int_(arg)
    if isinstance(arg, float):
        return machine.double(arg)
    if isinstance(arg, str):
        return machine.ptr(machine.literal(arg))
    raise TypeError(arg)


def printf(fmt: str, *args: object) -> str:
    """What printf(fmt, args...) prints on stdout."""
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal(fmt)),
                 *(_arg(machine, a) for a in args))
    return machine.stdout


# ---------------------------------------------------------------- printf corners

@pytest.mark.parametrize(("fmt", "args", "expected"), [
    ("[%-5d]", (42,), "[42   ]"),
    ("[%05.1f]", (3.14159,), "[003.1]"),
    ("[%05.1f]", (-3.14159,), "[-03.1]"),
    ("[%.3s]", ("abcdef",), "[abc]"),
    ("[%x]", (-1,), "[ffffffff]"),
    ("[%X]", (-255,), "[FFFFFF01]"),
    ("[%u]", (-1,), "[4294967295]"),
    ("[%%]", (), "[%]"),
    ("100%%", (), "100%"),
    ("[%y]", (), "[%y]"),
    ("[%-+5d]", (7,), "[+7   ]"),
    ("[%08.3d]", (5,), "[     005]"),
    ("[% d]", (5,), "[ 5]"),
])
def test_printf_corner_cases(fmt: str, args: tuple, expected: str) -> None:
    assert printf(fmt, *args) == expected


def test_printf_ld_prints_a_full_long() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("%ld %lu")),
                 machine.long_(-9_000_000_000), machine.long_(-1))
    assert machine.stdout == "-9000000000 18446744073709551615"


def test_printf_c_with_zero_writes_a_nul_byte() -> None:
    machine = FakeMachine()
    result = machine.call("printf", machine.ptr(machine.literal("a%cb")), machine.int_(0))
    assert result.raw == 3
    assert machine.stdout == "a\0b"


def test_printf_precision_string_need_not_be_terminated() -> None:
    """%.3s of a char[3] with no '\\0' is valid C: precision bounds the read."""
    machine = FakeMachine()
    exact = machine.buffer(3)
    machine.write_bytes(exact, b"abc")
    machine.call("printf", machine.ptr(machine.literal("[%.3s]")), machine.ptr(exact))
    assert machine.stdout == "[abc]"


def test_printf_unterminated_string_without_precision_traps() -> None:
    machine = FakeMachine()
    exact = machine.buffer(3)
    machine.write_bytes(exact, b"abc")
    with pytest.raises(TrapError) as caught:
        machine.call("printf", machine.ptr(machine.literal("%s")), machine.ptr(exact))
    assert caught.value.code == "string_not_terminated"


def test_snprintf_returns_untruncated_length() -> None:
    machine = FakeMachine()
    buf = machine.buffer(4)
    result = machine.call("snprintf", machine.ptr(buf), machine.size(4),
                          machine.ptr(machine.literal("%d")), machine.int_(123456))
    assert result.raw == 6
    assert machine.text_at(buf) == "123"


# ---------------------------------------------------------------- scanf

def test_scanf_comma_separated_integers() -> None:
    machine = FakeMachine(stdin="3, 4\n")
    a, b = machine.buffer(4), machine.buffer(4)
    count = machine.call("scanf", machine.ptr(machine.literal("%d,%d")),
                         machine.ptr(a, INT), machine.ptr(b, INT))
    assert count.raw == 2
    assert (machine.memory.load(a, INT), machine.memory.load(b, INT)) == (3, 4)


def test_scanf_separator_mismatch_stops_after_first() -> None:
    machine = FakeMachine(stdin="3;4\n")
    a, b = machine.buffer(4), machine.buffer(4)
    count = machine.call("scanf", machine.ptr(machine.literal("%d,%d")),
                         machine.ptr(a, INT), machine.ptr(b, INT))
    assert count.raw == 1


def test_scanf_c_reads_whitespace_unless_format_has_space() -> None:
    machine = FakeMachine(stdin="5 x\n")
    n, c = machine.buffer(4), machine.buffer(1)
    machine.call("scanf", machine.ptr(machine.literal("%d%c")), machine.ptr(n, INT),
                 machine.ptr(c))
    assert machine.read_bytes(c, 1) == b" "
    machine2 = FakeMachine(stdin="5 x\n")
    n2, c2 = machine2.buffer(4), machine2.buffer(1)
    machine2.call("scanf", machine2.ptr(machine2.literal("%d %c")), machine2.ptr(n2, INT),
                  machine2.ptr(c2))
    assert machine2.read_bytes(c2, 1) == b"x"


def test_scanf_at_eof_returns_eof() -> None:
    machine = FakeMachine(stdin=[None])
    n = machine.buffer(4)
    assert machine.call("scanf", machine.ptr(machine.literal("%d")),
                        machine.ptr(n, INT)).raw == -1


# ---------------------------------------------------------------- fgets and EOF

def test_fgets_small_buffer_splits_a_long_line() -> None:
    machine = FakeMachine(stdin="abcdefg\n")
    buf = machine.buffer(4)
    stdin = machine.stream("stdin")
    parts = []
    while machine.call("fgets", machine.ptr(buf), machine.int_(4), stdin).raw:
        parts.append(machine.text_at(buf))
    assert parts == ["abc", "def", "g\n"]


def test_reading_past_eof_repeatedly_asks_the_host_once() -> None:
    machine = FakeMachine(stdin=["x", None])
    stdin = machine.stream("stdin")
    results = [machine.call("fgetc", stdin).raw for _ in range(5)]
    assert results == [ord("x"), -1, -1, -1, -1]
    assert len(machine.prompts) == 2
    assert machine.call("feof", stdin).raw == 1


def test_file_read_past_eof_keeps_returning_eof() -> None:
    machine = FakeMachine(files={"/in.txt": b"z"})
    f = machine.call("fopen", machine.ptr(machine.literal("/in.txt")),
                     machine.ptr(machine.literal("r")))
    assert [machine.call("fgetc", f).raw for _ in range(3)] == [ord("z"), -1, -1]
    buf = machine.buffer(8)
    assert machine.call("fgets", machine.ptr(buf), machine.int_(8), f).raw == 0


# ---------------------------------------------------------------- files

def _open(machine: FakeMachine, path: str, mode: str) -> Value:
    return machine.call("fopen", machine.ptr(machine.literal(path)),
                        machine.ptr(machine.literal(mode)))


def test_fopen_append_keeps_old_contents() -> None:
    machine = FakeMachine(files={"/log.txt": b"one\n"})
    f = _open(machine, "/log.txt", "a")
    machine.call("fputs", machine.ptr(machine.literal("two\n")), f)
    machine.call("fclose", f)
    assert machine.files.read("/log.txt") == b"one\ntwo\n"


def test_fopen_append_creates_missing_file() -> None:
    machine = FakeMachine()
    f = _open(machine, "/new.txt", "a")
    assert f.raw != 0
    machine.call("fputc", machine.int_(ord("!")), f)
    assert machine.files.read("/new.txt") == b"!"


def test_fwrite_keeps_nul_bytes() -> None:
    machine = FakeMachine()
    data = machine.buffer(4)
    machine.write_bytes(data, b"a\0b\0")
    f = _open(machine, "/bin.dat", "wb")
    written = machine.call("fwrite", machine.ptr(data), machine.size(1), machine.size(4), f)
    assert written.raw == 4
    assert machine.files.read("/bin.dat") == b"a\0b\0"


def test_fclose_twice_traps_invalid_stream() -> None:
    machine = FakeMachine()
    f = _open(machine, "/x.txt", "w")
    machine.call("fclose", f)
    with pytest.raises(TrapError) as caught:
        machine.call("fclose", f)
    assert caught.value.code == "invalid_stream"


def test_stdio_on_garbage_file_pointer_traps_invalid_stream() -> None:
    machine = FakeMachine()
    junk = machine.buffer(8)
    with pytest.raises(TrapError) as caught:
        machine.call("fgetc", machine.ptr(junk, CHAR))
    assert caught.value.code == "invalid_stream"


def test_unclosed_file_is_reported_by_finalize() -> None:
    from pycc.interp.stdlib import finalize

    machine = FakeMachine()
    _open(machine, "/keep.txt", "w")
    finalize(machine)
    assert machine.warning_codes() == ["file_not_closed"]


def test_open_close_loop_does_not_fill_the_data_segment() -> None:
    """A program that appends to a log every tick opens thousands of FILEs."""
    machine = FakeMachine()
    for _ in range(6000):
        f = _open(machine, "/log.txt", "a")
        machine.call("fputc", machine.int_(ord(".")), f)
        assert machine.call("fclose", f).raw == 0
    assert machine.files.read("/log.txt") == b"." * 6000


def test_file_pointer_used_right_after_fclose_still_traps() -> None:
    machine = FakeMachine()
    stale = _open(machine, "/a.txt", "w")
    machine.call("fclose", stale)
    fresh = _open(machine, "/b.txt", "w")
    assert fresh.raw != stale.raw
    with pytest.raises(TrapError) as caught:
        machine.call("fputc", machine.int_(ord("x")), stale)
    assert caught.value.code == "invalid_stream"


def test_printf_returns_byte_count_and_warns_on_missing_argument() -> None:
    machine = FakeMachine()
    result = machine.call("printf", machine.ptr(machine.literal("%d %d")), machine.int_(1))
    assert result.raw == len(machine.stdout)
    assert "format_argument_type" in machine.warning_codes()


def test_scanf_integer_overflow_wraps_like_the_store() -> None:
    machine = FakeMachine()
    n = machine.buffer(4)
    count = machine.call("sscanf", machine.ptr(machine.literal("4294967297")),
                         machine.ptr(machine.literal("%d")), machine.ptr(n, INT))
    assert count.raw == 1
    assert machine.memory.load(n, INT) == 1
