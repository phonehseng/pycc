"""printf and friends: format parsing, flags, widths, lengths and return values."""
from __future__ import annotations

import math

import pytest

from pycc.core.events import Output
from pycc.interp import constructs
from pycc.interp.c_types import INT, LONG, SIZE_T, FunctionType, PointerType
from pycc.interp.memory import Memory
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.values import Value, make_int


def _value(machine: FakeMachine, arg: object) -> Value:
    """Python test values -> C argument values (str becomes a string literal)."""
    if isinstance(arg, Value):
        return arg
    if isinstance(arg, bool):
        return machine.int_(int(arg))
    if isinstance(arg, int):
        return machine.int_(arg)
    if isinstance(arg, float):
        return machine.double(arg)
    if isinstance(arg, str):
        return machine.ptr(machine.literal(arg))
    raise TypeError(arg)


def printf(fmt: str, *args: object) -> str:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal(fmt)), *(_value(machine, a) for a in args))
    return machine.stdout


NAN = float("nan")
NEG_NAN = math.copysign(NAN, -1.0)
INF = float("inf")

CASES = [
    ("plain text", (), "plain text"),
    ("%d", (42,), "42"),
    ("%d", (-42,), "-42"),
    ("%i", (-1,), "-1"),
    ("%5d", (42,), "   42"),
    ("%-5d|", (42,), "42   |"),
    ("%05d", (42,), "00042"),
    ("%05d", (-42,), "-0042"),
    ("%+d", (42,), "+42"),
    ("% d", (42,), " 42"),
    ("%+d", (-42,), "-42"),
    ("%-+6d|", (7,), "+7    |"),
    ("%.3d", (7,), "007"),
    ("%8.3d", (7,), "     007"),
    ("%08.3d", (7,), "     007"),
    ("%.0d", (0,), ""),
    ("%u", (-1,), "4294967295"),
    ("%x", (255,), "ff"),
    ("%X", (255,), "FF"),
    ("%#x", (255,), "0xff"),
    ("%#X", (255,), "0XFF"),
    ("%#x", (0,), "0"),
    ("%#08x", (255,), "0x0000ff"),
    ("%x", (-1,), "ffffffff"),
    ("%o", (8,), "10"),
    ("%#o", (8,), "010"),
    ("%#o", (0,), "0"),
    ("%hhd", (300,), "44"),
    ("%hhu", (-1,), "255"),
    ("%hd", (70000,), "4464"),
    ("%ld", (make_int(1 << 40, LONG),), "1099511627776"),
    ("%lld", (make_int(-(1 << 40), LONG),), "-1099511627776"),
    ("%lu", (make_int(-1, LONG),), "18446744073709551615"),
    ("%lx", (make_int(-1, LONG),), "ffffffffffffffff"),
    ("%zu", (make_int(12, SIZE_T),), "12"),
    ("%*d", (5, 42), "   42"),
    ("%-*d|", (5, 42), "42   |"),
    ("%*d|", (-5, 42), "42   |"),
    ("%.*f", (2, 3.14159), "3.14"),
    ("%.*f", (-1, 3.5), "3.500000"),
    ("%f", (3.14159,), "3.141590"),
    ("%.0f", (2.5,), "2"),
    ("%.0f", (3.5,), "4"),
    ("%10.3f", (3.14159,), "     3.142"),
    ("%-10.3f|", (3.14159,), "3.142     |"),
    ("%010.2f", (-3.14159,), "-000003.14"),
    ("%+.1f", (2.0,), "+2.0"),
    ("%#.0f", (3.0,), "3."),
    ("%e", (12345.678,), "1.234568e+04"),
    ("%E", (12345.678,), "1.234568E+04"),
    ("%.2e", (0.0,), "0.00e+00"),
    ("%g", (0.0001,), "0.0001"),
    ("%g", (1e-5,), "1e-05"),
    ("%g", (123456789.0,), "1.23457e+08"),
    ("%g", (100000.0,), "100000"),
    ("%G", (1e-5,), "1E-05"),
    ("%#g", (1.0,), "1.00000"),
    ("%F", (INF,), "INF"),
    ("%f", (-INF,), "-inf"),
    ("%+f", (INF,), "+inf"),
    ("%5f", (NAN,), "  nan"),
    ("%f", (NEG_NAN,), "-nan"),
    ("%05f", (INF,), "  inf"),
    ("%-6f|", (INF,), "inf   |"),
    ("%c", (65,), "A"),
    ("%3c", (65,), "  A"),
    ("%-3c|", (65,), "A  |"),
    ("%c", (321,), "A"),
    ("%s", ("hi",), "hi"),
    ("%5s", ("hi",), "   hi"),
    ("%-5s|", ("hi",), "hi   |"),
    ("%.2s", ("hello",), "he"),
    ("%.0s|", ("hello",), "|"),
    ("%%", (), "%"),
    ("100%%", (), "100%"),
    ("%q", (), "%q"),
    ("%5.1f%%", (99.5,), " 99.5%"),
    ("%d %s %c", (1, "x", 66), "1 x B"),
    ("[%3d][%-3d]", (1, 2), "[  1][2  ]"),
]


@pytest.mark.parametrize("fmt,args,expected", CASES, ids=[c[0] for c in CASES])
def test_printf_formats(fmt: str, args: tuple, expected: str) -> None:
    assert printf(fmt, *args) == expected


@pytest.mark.parametrize("fmt,value", [
    ("%d", 123), ("%5d", -7), ("%-8.3d|", 42), ("%x", 48879), ("%08X", 48879),
    ("%.3f", 2.0 / 3.0), ("%12.4e", 6.02e23), ("%g", 0.5), ("%-+8.2f|", 1.005),
])
def test_printf_agrees_with_python_percent_formatting(fmt: str, value: object) -> None:
    # Python's % operator implements C's rules for these conversions.
    assert printf(fmt, value) == fmt % value


def test_null_string_prints_null_in_parentheses() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("[%s]")), machine.null())
    assert machine.stdout == "[(null)]"


def test_pointer_prints_hex_with_0x() -> None:
    assert printf("%p", FakeMachine.ptr(0x1234)) == "0x1234"


def test_null_pointer_prints_nil() -> None:
    assert printf("%p", FakeMachine.ptr(0)) == "(nil)"


def test_function_pointer_prints_code_address() -> None:
    designator = Value(FunctionType(INT, ()), 3)
    assert printf("%p", designator) == "0x400030"
    pointer = Value(PointerType(FunctionType(INT, ())), 1)
    assert printf("%p", pointer) == "0x400010"


def test_printf_returns_number_of_bytes() -> None:
    machine = FakeMachine()
    result = machine.call("printf", machine.ptr(machine.literal("%d apples\n")), machine.int_(12))
    assert result.raw == len("12 apples\n")


def test_printf_yields_one_output_event_per_call() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("a")))
    machine.call("printf", machine.ptr(machine.literal("b\n")))
    outputs = [e for e in machine.events if isinstance(e, Output)]
    assert [(e.stream, e.text) for e in outputs] == [("stdout", "a"), ("stdout", "b\n")]


def test_printf_counts_construct() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("x")))
    assert machine.constructs[constructs.PRINTF] == 1


def test_utf8_text_survives_byte_by_byte_output() -> None:
    machine = FakeMachine()
    for byte in "é".encode("utf-8"):
        machine.call("putchar", machine.int_(byte))
    assert machine.stdout == "é"


def test_double_for_int_conversion_warns() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("%d")), machine.double(2.0))
    assert machine.warning_codes() == ["format_argument_type"]
    assert "expects argument of type 'int'" in machine.warnings[0].message
    assert "argument 2 has type 'double'" in machine.warnings[0].message


def test_int_for_float_conversion_warns_and_prints_garbage() -> None:
    # the int sits in an integer register, %f reads a floating one.
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("%f")), machine.int_(2))
    assert machine.warning_codes() == ["format_argument_type"]
    assert machine.stdout != "2.000000"
    assert float(machine.stdout) != 2.0


def test_double_for_int_conversion_never_prints_the_truncated_value() -> None:
    # printf("%d", 4.5) must not print 4, or a wrong format could pass.
    outputs = set()
    for seed in range(20):
        machine = FakeMachine(memory=Memory(seed=seed))
        machine.call("printf", machine.ptr(machine.literal("%d")), machine.double(4.5))
        outputs.add(machine.stdout)
    assert "4" not in outputs
    assert len(outputs) > 1                # the garbage follows the seed


def test_mismatch_garbage_is_reproducible_for_one_seed() -> None:
    def run_once() -> str:
        machine = FakeMachine(memory=Memory(seed=7))
        machine.call("printf", machine.ptr(machine.literal("%d %d")), machine.double(1.5),
                     machine.double(1.5))
        return machine.stdout

    first = run_once()
    assert first == run_once()
    left, right = first.split()
    assert left != right                   # two stale registers, two different values


def test_int_for_pointer_conversion_keeps_the_value() -> None:
    # Pointers and ints share the integer registers, so no garbage is invented.
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("%x")), machine.ptr(0x40))
    assert machine.stdout == "40"


def test_missing_argument_warns_and_prints_zero() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("%d and %d")), machine.int_(1))
    assert machine.stdout == "1 and 0"
    assert machine.warning_codes() == ["format_argument_type"]


def test_percent_n_stores_bytes_so_far() -> None:
    machine = FakeMachine()
    count = machine.buffer(4)
    machine.call("printf", machine.ptr(machine.literal("hello%n!")), machine.ptr(count))
    assert machine.memory.load(count, INT) == 5
    assert machine.stdout == "hello!"


def test_fprintf_to_stderr() -> None:
    machine = FakeMachine()
    machine.call("fprintf", machine.stream("stderr"), machine.ptr(machine.literal("oops %d\n")),
                 machine.int_(3))
    assert machine.stderr == "oops 3\n"
    assert machine.stdout == ""


def test_fprintf_to_stdout_matches_printf() -> None:
    machine = FakeMachine()
    machine.call("fprintf", machine.stream("stdout"), machine.ptr(machine.literal("%s")),
                 machine.ptr(machine.literal("ok")))
    assert machine.stdout == "ok"


# --- sprintf / snprintf ------------------------------------------------------------------

def test_sprintf_writes_terminated_string_and_returns_length() -> None:
    machine = FakeMachine()
    buf = machine.buffer(32, b"X")
    result = machine.call("sprintf", machine.ptr(buf), machine.ptr(machine.literal("%d-%s")),
                          machine.int_(7), machine.ptr(machine.literal("up")))
    assert machine.text_at(buf) == "7-up"
    assert result.raw == 4
    assert machine.stdout == ""


def test_sprintf_overflow_traps_out_of_bounds() -> None:
    from pycc.interp.traps import TrapError
    machine = FakeMachine()
    buf = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        machine.call("sprintf", machine.ptr(buf), machine.ptr(machine.literal("%s")),
                     machine.ptr(machine.literal("too long")))
    assert caught.value.code == "out_of_bounds"


def test_snprintf_truncates_and_returns_full_length() -> None:
    machine = FakeMachine()
    buf = machine.buffer(8, b"X")
    result = machine.call("snprintf", machine.ptr(buf), machine.size(5),
                          machine.ptr(machine.literal("%s")),
                          machine.ptr(machine.literal("abcdefgh")))
    assert machine.text_at(buf) == "abcd"
    assert result.raw == 8
    assert machine.memory.read_bytes(buf + 5, 1) == b"X"   # nothing past size touched


def test_snprintf_with_room_writes_everything() -> None:
    machine = FakeMachine()
    buf = machine.buffer(16)
    result = machine.call("snprintf", machine.ptr(buf), machine.size(16),
                          machine.ptr(machine.literal("n=%03d")), machine.int_(9))
    assert machine.text_at(buf) == "n=009"
    assert result.raw == 5


def test_snprintf_size_zero_writes_nothing_and_accepts_null() -> None:
    machine = FakeMachine()
    result = machine.call("snprintf", machine.null(), machine.size(0),
                          machine.ptr(machine.literal("%d")), machine.int_(12345))
    assert result.raw == 5


def test_snprintf_size_one_writes_only_terminator() -> None:
    machine = FakeMachine()
    buf = machine.buffer(4, b"X")
    machine.call("snprintf", machine.ptr(buf), machine.size(1), machine.ptr(machine.literal("abc")))
    assert machine.memory.read_bytes(buf, 2) == b"\0X"


# --- puts / putchar / fputs / fputc -----------------------------------------------------------

def test_puts_adds_newline_and_returns_length_plus_one() -> None:
    machine = FakeMachine()
    result = machine.call("puts", machine.ptr(machine.literal("hello")))
    assert machine.stdout == "hello\n"
    assert result.raw == 6


def test_putchar_returns_the_byte() -> None:
    machine = FakeMachine()
    result = machine.call("putchar", machine.int_(ord("z")))
    assert machine.stdout == "z"
    assert result.raw == ord("z")


def test_fputs_adds_no_newline() -> None:
    machine = FakeMachine()
    result = machine.call("fputs", machine.ptr(machine.literal("abc")), machine.stream("stdout"))
    assert machine.stdout == "abc"
    assert result.raw >= 0


def test_fputc_to_stderr() -> None:
    machine = FakeMachine()
    machine.call("fputc", machine.int_(ord("!")), machine.stream("stderr"))
    assert machine.stderr == "!"


def test_printf_with_null_format_traps() -> None:
    from pycc.interp.traps import TrapError
    machine = FakeMachine()
    with pytest.raises(TrapError) as caught:
        machine.call("printf", machine.null())
    assert caught.value.code == "null_dereference"
    assert "printf()" in caught.value.message
