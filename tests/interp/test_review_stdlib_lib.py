"""Adversarial review tests for <stdlib.h>, <string.h>, <math.h>, headers and FakeMachine."""
from __future__ import annotations

import math

import pytest

from pycc.interp.c_types import CHAR, INT, LONG, PointerType
from pycc.interp.stdlib import STANDARD_HEADERS
from pycc.interp.stdlib.headers import header_text
from pycc.interp.stdlib.errno_ import ERANGE
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.traps import TrapError


# ---------------------------------------------------------------- number parsing

def test_strtol_overflow_sets_erange_and_clamps() -> None:
    machine = FakeMachine()
    end = machine.buffer(8)
    text = machine.literal("99999999999999999999x")
    result = machine.call("strtol", machine.ptr(text), machine.ptr(end, PointerType(CHAR)),
                          machine.int_(10))
    assert result.raw == LONG.max
    assert machine.errno == ERANGE
    assert machine.memory.load(end, PointerType(CHAR)) == text + 20


def test_strtol_negative_overflow_clamps_to_long_min() -> None:
    machine = FakeMachine()
    result = machine.call("strtol", machine.ptr(machine.literal("-99999999999999999999")),
                          machine.null(), machine.int_(10))
    assert result.raw == LONG.min
    assert machine.errno == ERANGE


@pytest.mark.parametrize("text", ["garbage", "", "  ", "-", "+x", "x12"])
def test_atoi_of_garbage_is_zero(text: str) -> None:
    machine = FakeMachine()
    assert machine.call("atoi", machine.ptr(machine.literal(text))).raw == 0


def test_atoi_stops_at_first_non_digit() -> None:
    machine = FakeMachine()
    assert machine.call("atoi", machine.ptr(machine.literal("  -12abc"))).raw == -12


def test_strtod_underflow_sets_erange() -> None:
    machine = FakeMachine()
    result = machine.call("strtod", machine.ptr(machine.literal("1e-400")), machine.null())
    assert result.raw == 0.0
    assert machine.errno == ERANGE


def test_strtod_of_plain_zero_leaves_errno_alone() -> None:
    machine = FakeMachine()
    machine.call("strtod", machine.ptr(machine.literal("0.000e-400")), machine.null())
    assert machine.errno == 0


# ---------------------------------------------------------------- strings

@pytest.mark.parametrize(("left", "right", "sign"), [
    ("abc", "abd", -1), ("abd", "abc", 1), ("abc", "abc", 0), ("ab", "abc", -1),
    ("abc", "ab", 1), ("", "", 0), ("\xff", "a", 1),
])
def test_strcmp_sign_conventions(left: str, right: str, sign: int) -> None:
    machine = FakeMachine()
    raw = machine.call("strcmp", machine.ptr(machine.literal(left.encode("latin-1"))),
                       machine.ptr(machine.literal(right.encode("latin-1")))).raw
    assert (raw > 0) - (raw < 0) == sign


def test_memcpy_of_zero_bytes_accepts_null() -> None:
    machine = FakeMachine()
    result = machine.call("memcpy", machine.null(), machine.null(), machine.size(0))
    assert result.raw == 0


def test_strncpy_from_exactly_n_unterminated_bytes() -> None:
    machine = FakeMachine()
    src = machine.buffer(3)
    machine.write_bytes(src, b"xyz")
    dest = machine.buffer(3)
    machine.call("strncpy", machine.ptr(dest), machine.ptr(src), machine.size(3))
    assert machine.read_bytes(dest, 3) == b"xyz"


def test_strncmp_of_unterminated_arrays_within_n() -> None:
    machine = FakeMachine()
    a, b = machine.buffer(2), machine.buffer(2)
    machine.write_bytes(a, b"hi")
    machine.write_bytes(b, b"hi")
    assert machine.call("strncmp", machine.ptr(a), machine.ptr(b), machine.size(2)).raw == 0


def test_memset_with_huge_count_traps_instead_of_exhausting_python() -> None:
    machine = FakeMachine()
    buf = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        machine.call("memset", machine.ptr(buf), machine.int_(0), machine.size(2 ** 64 - 1))
    assert caught.value.code == "out_of_bounds"


def test_strncpy_with_huge_count_traps_instead_of_exhausting_python() -> None:
    machine = FakeMachine()
    dest = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        machine.call("strncpy", machine.ptr(dest), machine.ptr(machine.literal("a")),
                     machine.size(2 ** 64 - 1))
    assert caught.value.code == "out_of_bounds"


def test_qsort_with_huge_count_traps_out_of_bounds() -> None:
    machine = FakeMachine()
    arr = machine.buffer(8)
    fid = machine.add_callback(lambda m, args: 0)
    with pytest.raises(TrapError) as caught:
        machine.call("qsort", machine.ptr(arr), machine.size(2 ** 63), machine.size(4),
                     machine.int_(fid))
    assert caught.value.code == "out_of_bounds"


# ---------------------------------------------------------------- heap

def test_realloc_null_is_malloc() -> None:
    machine = FakeMachine()
    block = machine.call("realloc", machine.null(), machine.size(16))
    assert block.raw != 0
    machine.write_bytes(block.raw, b"x" * 16)
    machine.call("free", block)
    assert machine.memory.leaks() == []


def test_free_null_does_nothing() -> None:
    machine = FakeMachine()
    machine.call("free", machine.null())
    assert machine.memory.leaks() == []


def test_double_free_traps() -> None:
    machine = FakeMachine()
    block = machine.call("malloc", machine.size(8))
    machine.call("free", block)
    with pytest.raises(TrapError) as caught:
        machine.call("free", block)
    assert caught.value.code == "double_free"


# ---------------------------------------------------------------- qsort callbacks

def _int_array(machine: FakeMachine, values: list[int]) -> int:
    addr = machine.buffer(4 * len(values))
    for index, value in enumerate(values):
        machine.memory.store(addr + 4 * index, INT, value)
    return addr


def test_qsort_sorts_with_comparator() -> None:
    machine = FakeMachine()
    arr = _int_array(machine, [5, 1, 4, 2, 3])

    def compare(m: FakeMachine, args: list) -> int:
        return m.memory.load(args[0].raw, INT) - m.memory.load(args[1].raw, INT)

    fid = machine.add_callback(compare)
    machine.call("qsort", machine.ptr(arr), machine.size(5), machine.size(4),
                 machine.int_(fid))
    assert [machine.memory.load(arr + 4 * i, INT) for i in range(5)] == [1, 2, 3, 4, 5]


def test_qsort_comparator_trap_propagates_and_leaves_array_untouched() -> None:
    machine = FakeMachine()
    arr = _int_array(machine, [3, 2, 1])

    def compare(m: FakeMachine, args: list) -> int:
        return m.memory.load(0, INT)        # dereferences NULL: a memory trap

    fid = machine.add_callback(compare)
    with pytest.raises(TrapError) as caught:
        machine.call("qsort", machine.ptr(arr), machine.size(3), machine.size(4),
                     machine.int_(fid))
    assert caught.value.code == "null_dereference"
    assert [machine.memory.load(arr + 4 * i, INT) for i in range(3)] == [3, 2, 1]


def test_qsort_bad_function_pointer_traps() -> None:
    machine = FakeMachine()
    arr = _int_array(machine, [2, 1])
    with pytest.raises(TrapError) as caught:
        machine.call("qsort", machine.ptr(arr), machine.size(2), machine.size(4),
                     machine.int_(99))
    assert caught.value.code == "bad_function_pointer"


def test_qsort_only_promises_order_not_stability() -> None:
    """C makes no stability promise; equal keys must merely end up adjacent and sorted."""
    machine = FakeMachine()
    values = [(2, 0), (1, 1), (2, 2), (1, 3)]
    arr = _int_array(machine, [key * 10 + tag for key, tag in values])

    def compare(m: FakeMachine, args: list) -> int:
        return m.memory.load(args[0].raw, INT) // 10 - m.memory.load(args[1].raw, INT) // 10

    machine.call("qsort", machine.ptr(arr), machine.size(4), machine.size(4),
                 machine.int_(machine.add_callback(compare)))
    keys = [machine.memory.load(arr + 4 * i, INT) // 10 for i in range(4)]
    assert keys == [1, 1, 2, 2]


# ---------------------------------------------------------------- math

def test_pow_of_one_and_nan_is_one() -> None:
    machine = FakeMachine()
    assert machine.call("pow", machine.double(1.0), machine.double(math.nan)).raw == 1.0
    assert machine.call("pow", machine.double(math.nan), machine.double(0.0)).raw == 1.0


# ---------------------------------------------------------------- headers

@pytest.mark.parametrize("name", STANDARD_HEADERS)
def test_header_text_is_pure_ascii(name: str) -> None:
    assert header_text(name).isascii()


# ---------------------------------------------------------------- FakeMachine honesty

def test_fake_machine_reads_through_memory_traps() -> None:
    machine = FakeMachine()
    with pytest.raises(TrapError) as caught:
        machine.read_bytes(0, 4)
    assert caught.value.code == "null_dereference"


def test_fake_machine_write_to_literal_traps() -> None:
    machine = FakeMachine()
    lit = machine.literal("const")
    with pytest.raises(TrapError) as caught:
        machine.call("strcpy", machine.ptr(lit), machine.ptr(machine.literal("x")))
    assert caught.value.code == "write_to_literal"


def test_fake_machine_use_after_free_traps() -> None:
    machine = FakeMachine()
    block = machine.call("malloc", machine.size(8))
    machine.call("free", block)
    with pytest.raises(TrapError) as caught:
        machine.call("strlen", block)
    assert caught.value.code == "use_after_free"


def test_fake_machine_write_cstring_overflow_traps() -> None:
    machine = FakeMachine()
    buf = machine.buffer(2)
    with pytest.raises(TrapError) as caught:
        machine.write_cstring(buf, "too long")
    assert caught.value.code == "out_of_bounds"
