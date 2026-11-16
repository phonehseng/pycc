"""scanf, sscanf, fscanf and the stdin readers: conversions, EOF, the StdinBuffer."""
from __future__ import annotations

import pytest

from pycc.core.events import InputRequest
from pycc.interp import constructs
from pycc.interp.c_types import CHAR, DOUBLE, FLOAT, INT, LONG, SHORT, UINT, ULONG
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.stdlib.stdio import EOF, stdin_buffer


def sscanf(machine: FakeMachine, text: str, fmt: str, *targets: int) -> int:
    result = machine.call("sscanf", machine.ptr(machine.literal(text)),
                          machine.ptr(machine.literal(fmt)), *(machine.ptr(t) for t in targets))
    return result.raw


# --- sscanf conversions ----------------------------------------------------------------------

@pytest.mark.parametrize("text,fmt,ctype,expected", [
    ("42", "%d", INT, 42),
    ("  -17", "%d", INT, -17),
    ("+5", "%d", INT, 5),
    ("0x1f", "%i", INT, 31),
    ("017", "%i", INT, 15),
    ("99", "%i", INT, 99),
    ("ff", "%x", UINT, 255),
    ("0XFF", "%x", UINT, 255),
    ("17", "%o", UINT, 15),
    ("4000000000", "%u", UINT, 4000000000),
    ("123456789012", "%ld", LONG, 123456789012),
    ("18446744073709551615", "%lu", ULONG, 18446744073709551615),
    ("7", "%zu", ULONG, 7),
    ("70000", "%hd", SHORT, 4464),
    ("12345", "%3d", INT, 123),
])
def test_integer_conversions(text: str, fmt: str, ctype, expected: int) -> None:
    machine = FakeMachine()
    target = machine.buffer(8)
    assert sscanf(machine, text, fmt, target) == 1
    assert machine.memory.load(target, ctype) == expected


@pytest.mark.parametrize("text,fmt,ctype,expected", [
    ("3.5", "%f", FLOAT, 3.5),
    ("-2.25", "%lf", DOUBLE, -2.25),
    ("1e3", "%lf", DOUBLE, 1000.0),
    ("6.02E+2", "%lf", DOUBLE, 602.0),
    (".5", "%lf", DOUBLE, 0.5),
    ("7", "%lf", DOUBLE, 7.0),
    ("2.5e", "%lf", DOUBLE, 2.5),
])
def test_float_conversions(text: str, fmt: str, ctype, expected: float) -> None:
    machine = FakeMachine()
    target = machine.buffer(8)
    assert sscanf(machine, text, fmt, target) == 1
    assert machine.memory.load(target, ctype) == pytest.approx(expected)


def test_string_conversion_stops_at_space_and_terminates() -> None:
    machine = FakeMachine()
    word = machine.buffer(16, b"X")
    assert sscanf(machine, "  hello device", "%s", word) == 1
    assert machine.text_at(word) == "hello"


def test_string_width_limits_characters() -> None:
    machine = FakeMachine()
    word = machine.buffer(16)
    assert sscanf(machine, "abcdefgh", "%3s", word) == 1
    assert machine.text_at(word) == "abc"


def test_char_conversion_does_not_skip_space_or_terminate() -> None:
    machine = FakeMachine()
    chars = machine.buffer(4, b"X")
    assert sscanf(machine, " ab", "%c", chars) == 1
    assert machine.memory.read_bytes(chars, 2) == b" X"


def test_char_conversion_with_width() -> None:
    machine = FakeMachine()
    chars = machine.buffer(4, b"X")
    assert sscanf(machine, "abc", "%2c", chars) == 1
    assert machine.memory.read_bytes(chars, 3) == b"abX"


def test_space_before_char_conversion_skips_whitespace() -> None:
    machine = FakeMachine()
    chars = machine.buffer(4)
    assert sscanf(machine, "   z", " %c", chars) == 1
    assert machine.memory.read_bytes(chars, 1) == b"z"


def test_scanset_reads_until_newline() -> None:
    machine = FakeMachine()
    line = machine.buffer(32)
    assert sscanf(machine, "two words\nnext", "%[^\n]", line) == 1
    assert machine.text_at(line) == "two words"


def test_scanset_with_range() -> None:
    machine = FakeMachine()
    word = machine.buffer(32)
    assert sscanf(machine, "abc123", "%[a-z]", word) == 1
    assert machine.text_at(word) == "abc"


def test_several_conversions_and_literals() -> None:
    machine = FakeMachine()
    day, month, year = machine.buffer(4), machine.buffer(4), machine.buffer(4)
    assert sscanf(machine, "05/11/2026", "%d/%d/%d", day, month, year) == 3
    assert [machine.memory.load(a, INT) for a in (day, month, year)] == [5, 11, 2026]


def test_literal_percent_is_matched() -> None:
    machine = FakeMachine()
    value = machine.buffer(4)
    assert sscanf(machine, "50%", "%d%%", value) == 1
    assert sscanf(machine, "50 %", "%d%%", value) == 1


def test_suppressed_conversion_is_not_counted() -> None:
    machine = FakeMachine()
    value = machine.buffer(4)
    assert sscanf(machine, "skip 9", "%*s %d", value) == 1
    assert machine.memory.load(value, INT) == 9


def test_percent_n_records_consumed_characters() -> None:
    machine = FakeMachine()
    value, used = machine.buffer(4), machine.buffer(4)
    assert sscanf(machine, "  123abc", "%d%n", value, used) == 1
    assert machine.memory.load(used, INT) == 5


def test_partial_match_returns_count_so_far() -> None:
    machine = FakeMachine()
    first, second = machine.buffer(4), machine.buffer(4, b"\x07")
    assert sscanf(machine, "12 apples", "%d %d", first, second) == 1
    assert machine.memory.load(first, INT) == 12
    assert machine.memory.read_bytes(second, 1) == b"\x07"     # untouched


def test_matching_failure_on_first_conversion_returns_zero() -> None:
    machine = FakeMachine()
    value = machine.buffer(4)
    assert sscanf(machine, "abc", "%d", value) == 0


def test_literal_mismatch_stops() -> None:
    machine = FakeMachine()
    a, b = machine.buffer(4), machine.buffer(4)
    assert sscanf(machine, "1-2", "%d,%d", a, b) == 1


def test_empty_input_returns_eof() -> None:
    machine = FakeMachine()
    value = machine.buffer(4)
    assert sscanf(machine, "", "%d", value) == EOF
    assert sscanf(machine, "   ", "%d", value) == EOF


def test_eof_after_a_conversion_returns_count() -> None:
    machine = FakeMachine()
    a, b = machine.buffer(4), machine.buffer(4)
    assert sscanf(machine, "4", "%d %d", a, b) == 1


def test_too_few_pointers_warns() -> None:
    machine = FakeMachine()
    a = machine.buffer(4)
    assert sscanf(machine, "1 2", "%d %d", a) == 1
    assert machine.warning_codes() == ["format_argument_type"]


def test_null_target_traps() -> None:
    from pycc.interp.traps import TrapError
    machine = FakeMachine()
    with pytest.raises(TrapError) as caught:
        machine.call("sscanf", machine.ptr(machine.literal("5")),
                     machine.ptr(machine.literal("%d")),
                     machine.null())
    assert caught.value.code == "null_dereference"


def test_string_overflow_traps_out_of_bounds() -> None:
    from pycc.interp.traps import TrapError
    machine = FakeMachine()
    small = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        sscanf(machine, "abcdefgh", "%s", small)
    assert caught.value.code == "out_of_bounds"


# --- scanf on stdin ------------------------------------------------------------------------------

def scanf(machine: FakeMachine, fmt: str, *targets: int) -> int:
    return machine.call("scanf", machine.ptr(machine.literal(fmt)),
                        *(machine.ptr(t) for t in targets)).raw


def test_scanf_requests_input_when_buffer_is_empty() -> None:
    machine = FakeMachine(stdin="21\n")
    value = machine.buffer(4)
    assert scanf(machine, "%d", value) == 1
    assert machine.memory.load(value, INT) == 21
    assert sum(isinstance(e, InputRequest) for e in machine.events) == 1


def test_scanf_prompt_is_unfinished_output_line() -> None:
    machine = FakeMachine(stdin="5\n")
    machine.call("printf", machine.ptr(machine.literal("Intro\nHow many? ")))
    scanf(machine, "%d", machine.buffer(4))
    assert machine.prompts == ["How many? "]


def test_scanf_reads_values_across_lines() -> None:
    machine = FakeMachine(stdin=["1\n", "2\n"])
    a, b = machine.buffer(4), machine.buffer(4)
    assert scanf(machine, "%d %d", a, b) == 2
    assert machine.memory.load(b, INT) == 2


def test_scanf_leaves_rest_of_line_buffered() -> None:
    machine = FakeMachine(stdin="3 4\n")
    a, b = machine.buffer(4), machine.buffer(4)
    scanf(machine, "%d", a)
    scanf(machine, "%d", b)
    assert machine.memory.load(b, INT) == 4
    assert len(machine.prompts) == 1          # the second scanf needed no new input


def test_scanf_letters_stay_unread_after_failure() -> None:
    machine = FakeMachine(stdin="abc\n")
    value, word = machine.buffer(4), machine.buffer(8)
    assert scanf(machine, "%d", value) == 0
    assert scanf(machine, "%s", word) == 1
    assert machine.text_at(word) == "abc"


def test_scanf_at_end_of_input_returns_eof() -> None:
    machine = FakeMachine(stdin=[None])
    assert scanf(machine, "%d", machine.buffer(4)) == EOF


def test_eof_is_permanent() -> None:
    machine = FakeMachine(stdin=[None, "late\n"])
    assert machine.call("getchar").raw == EOF
    assert machine.call("getchar").raw == EOF
    assert len(machine.prompts) == 1


def test_scanf_counts_construct() -> None:
    machine = FakeMachine(stdin="1\n")
    scanf(machine, "%d", machine.buffer(4))
    assert machine.constructs[constructs.SCANF] == 1


# --- getchar, ungetc, fgets on stdin ---------------------------------------------------------

def test_getchar_reads_bytes_then_eof() -> None:
    machine = FakeMachine(stdin="hi\n")
    got = [machine.call("getchar").raw for _ in range(4)]
    assert got == [ord("h"), ord("i"), ord("\n"), EOF]


def test_ungetc_pushes_back_into_stdin_buffer() -> None:
    machine = FakeMachine(stdin="xy\n")
    first = machine.call("getchar").raw
    machine.call("ungetc", machine.int_(first), machine.stream("stdin"))
    assert machine.call("getchar").raw == ord("x")
    assert stdin_buffer(machine).pos == 1


def test_ungetc_of_a_different_byte() -> None:
    machine = FakeMachine(stdin="b\n")
    machine.call("ungetc", machine.int_(ord("a")), machine.stream("stdin"))
    assert machine.call("getchar").raw == ord("a")
    assert machine.call("getchar").raw == ord("b")


def test_ungetc_eof_returns_eof() -> None:
    machine = FakeMachine()
    assert machine.call("ungetc", machine.int_(EOF), machine.stream("stdin")).raw == EOF


def test_fgets_keeps_newline() -> None:
    machine = FakeMachine(stdin="hello\n")
    buf = machine.buffer(32)
    result = machine.call("fgets", machine.ptr(buf), machine.int_(32), machine.stream("stdin"))
    assert result.raw == buf
    assert machine.text_at(buf) == "hello\n"


def test_fgets_stops_at_size_minus_one() -> None:
    machine = FakeMachine(stdin="abcdefgh\n")
    buf = machine.buffer(8)
    machine.call("fgets", machine.ptr(buf), machine.int_(4), machine.stream("stdin"))
    assert machine.text_at(buf) == "abc"
    machine.call("fgets", machine.ptr(buf), machine.int_(8), machine.stream("stdin"))
    assert machine.text_at(buf) == "defgh\n"


def test_fgets_last_line_without_newline() -> None:
    machine = FakeMachine(stdin=["tail", None])
    buf = machine.buffer(16)
    result = machine.call("fgets", machine.ptr(buf), machine.int_(16), machine.stream("stdin"))
    assert result.raw == buf
    assert machine.text_at(buf) == "tail"


def test_fgets_at_eof_returns_null_and_sets_feof() -> None:
    machine = FakeMachine(stdin=[None])
    buf = machine.buffer(16)
    assert machine.call("fgets", machine.ptr(buf), machine.int_(16),
                        machine.stream("stdin")).raw == 0
    assert machine.call("feof", machine.stream("stdin")).raw == 1


def test_fgets_size_one_writes_empty_string() -> None:
    machine = FakeMachine(stdin="abc\n")
    buf = machine.buffer(4, b"X")
    result = machine.call("fgets", machine.ptr(buf), machine.int_(1), machine.stream("stdin"))
    assert result.raw == buf
    assert machine.text_at(buf) == ""


def test_fgets_into_too_small_buffer_traps() -> None:
    from pycc.interp.traps import TrapError
    machine = FakeMachine(stdin="a long line of text\n")
    buf = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        machine.call("fgets", machine.ptr(buf), machine.int_(100), machine.stream("stdin"))
    assert caught.value.code == "out_of_bounds"


def test_fscanf_on_stdin_shares_the_buffer() -> None:
    machine = FakeMachine(stdin="8 9\n")
    a, b = machine.buffer(4), machine.buffer(4)
    machine.call("fscanf", machine.stream("stdin"), machine.ptr(machine.literal("%d")),
                 machine.ptr(a))
    scanf(machine, "%d", b)
    assert machine.memory.load(b, INT) == 9


def test_reading_char_variable_through_scanf() -> None:
    machine = FakeMachine(stdin="Q\n")
    target = machine.buffer(1)
    assert scanf(machine, " %c", target) == 1
    assert machine.memory.load(target, CHAR) == ord("Q")
