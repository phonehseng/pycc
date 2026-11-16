"""<stdlib.h>: heap, number parsing, rand, qsort/bsearch, exit/abort, getenv."""
from __future__ import annotations

import pytest

from pycc.core.events import Output
from pycc.interp import constructs
from pycc.interp.c_types import CHAR, INT, LONG, PointerType
from pycc.interp.stdlib.errno_ import EINVAL, ERANGE
from pycc.interp.stdlib.fake_machine import FakeMachine, FakeProgramExit
from pycc.interp.stdlib.stdlib_ import next_rand, parse_c_long
from pycc.interp.traps import TrapError


@pytest.fixture
def m() -> FakeMachine:
    return FakeMachine()


def lit(machine: FakeMachine, text: str):
    return machine.ptr(machine.literal(text))


# --- heap ------------------------------------------------------------------------------------

def test_malloc_and_free(m: FakeMachine) -> None:
    addr = m.call("malloc", m.size(16)).raw
    assert m.memory.object_at(addr).kind == "heap"
    m.call("free", m.ptr(addr))
    assert m.memory.leaks() == []
    assert m.constructs[constructs.MALLOC] == 1
    assert m.constructs[constructs.FREE] == 1


def test_calloc_zeroes(m: FakeMachine) -> None:
    addr = m.call("calloc", m.size(4), m.size(4)).raw
    assert m.memory.read_bytes(addr, 16) == bytes(16)


def test_realloc_keeps_contents_and_moves(m: FakeMachine) -> None:
    addr = m.call("malloc", m.size(4)).raw
    m.memory.store(addr, INT, 77)
    bigger = m.call("realloc", m.ptr(addr), m.size(64)).raw
    assert bigger != addr
    assert m.memory.load(bigger, INT) == 77
    assert m.constructs[constructs.REALLOC] == 1


def test_double_free_traps(m: FakeMachine) -> None:
    addr = m.call("malloc", m.size(8)).raw
    m.call("free", m.ptr(addr))
    with pytest.raises(TrapError) as caught:
        m.call("free", m.ptr(addr))
    assert caught.value.code == "double_free"


def test_free_of_non_heap_traps(m: FakeMachine) -> None:
    with pytest.raises(TrapError) as caught:
        m.call("free", m.ptr(m.buffer(8)))
    assert caught.value.code == "invalid_free"


def test_free_null_is_fine(m: FakeMachine) -> None:
    m.call("free", m.null())


def test_huge_malloc_returns_null_with_warning(m: FakeMachine) -> None:
    assert m.call("malloc", m.size(1 << 40)).raw == 0
    assert m.warning_codes() == ["heap_exhausted"]


# --- numbers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("42", 42), ("  -17abc", -17), ("+8", 8), ("abc", 0), ("", 0), ("2147483648", -2147483648),
])
def test_atoi(m: FakeMachine, text: str, value: int) -> None:
    assert m.call("atoi", lit(m, text)).raw == value


def test_atol_and_atof(m: FakeMachine) -> None:
    assert m.call("atol", lit(m, "9000000000")).raw == 9000000000
    assert m.call("atof", lit(m, " 2.5e2xyz")).raw == 250.0
    assert m.call("atof", lit(m, "nope")).raw == 0.0


@pytest.mark.parametrize("text,base,value,consumed", [
    ("123", 10, 123, 3), ("  -42x", 10, -42, 5), ("0x1F", 16, 31, 4), ("1F", 16, 31, 2),
    ("0x1F", 0, 31, 4), ("017", 0, 15, 3), ("17", 0, 17, 2), ("777", 8, 511, 3),
    ("z", 36, 35, 1), ("101", 2, 5, 3), ("0x", 16, 0, 1), ("xyz", 10, 0, 0),
])
def test_parse_c_long(text: str, base: int, value: int, consumed: int) -> None:
    assert parse_c_long(text.encode(), base)[:2] == (value, consumed)


def test_strtol_sets_endptr(m: FakeMachine) -> None:
    text = m.literal("  250 rest")
    end = m.buffer(8)
    value = m.call("strtol", m.ptr(text), m.ptr(end, PointerType(CHAR)), m.int_(10)).raw
    assert value == 250
    assert m.memory.load(end, PointerType(CHAR)) == text + 5


def test_strtol_without_digits_points_endptr_at_start(m: FakeMachine) -> None:
    text = m.literal("abc")
    end = m.buffer(8)
    assert m.call("strtol", m.ptr(text), m.ptr(end), m.int_(0)).raw == 0
    assert m.memory.load(end, PointerType(CHAR)) == text


def test_strtol_overflow_clamps_and_sets_erange(m: FakeMachine) -> None:
    value = m.call("strtol", lit(m, "99999999999999999999"), m.null(), m.int_(10)).raw
    assert value == LONG.max
    assert m.errno == ERANGE
    assert m.call("strtol", lit(m, "-99999999999999999999"), m.null(), m.int_(10)).raw == LONG.min


def test_strtol_bad_base_sets_einval(m: FakeMachine) -> None:
    assert m.call("strtol", lit(m, "10"), m.null(), m.int_(1)).raw == 0
    assert m.errno == EINVAL


def test_strtod(m: FakeMachine) -> None:
    text = m.literal("3.25kg")
    end = m.buffer(8)
    assert m.call("strtod", m.ptr(text), m.ptr(end)).raw == 3.25
    assert m.memory.load(end, PointerType(CHAR)) == text + 4


def test_strtod_overflow_sets_erange(m: FakeMachine) -> None:
    assert m.call("strtod", lit(m, "1e999"), m.null()).raw == float("inf")
    assert m.errno == ERANGE


def test_abs_and_labs(m: FakeMachine) -> None:
    assert m.call("abs", m.int_(-5)).raw == 5
    assert m.call("abs", m.int_(INT.min)).raw == INT.min      # wraps, as on hardware
    assert m.call("labs", m.long_(-(1 << 40))).raw == 1 << 40


# --- rand ------------------------------------------------------------------------------------

def test_rand_after_srand_1_matches_the_c_standard(m: FakeMachine) -> None:
    m.call("srand", m.uint(1))
    assert [m.call("rand").raw for _ in range(5)] == [16838, 5758, 10113, 17515, 31051]


def test_rand_without_srand_uses_seed_1(m: FakeMachine) -> None:
    assert m.call("rand").raw == 16838


def test_srand_restarts_sequence(m: FakeMachine) -> None:
    m.call("srand", m.uint(7))
    first = [m.call("rand").raw for _ in range(3)]
    m.call("srand", m.uint(7))
    assert [m.call("rand").raw for _ in range(3)] == first


def test_rand_stays_in_range() -> None:
    state = 12345
    for _ in range(1000):
        state, value = next_rand(state)
        assert 0 <= value <= 32767


# --- qsort / bsearch ----------------------------------------------------------------------

def _int_compare(machine: FakeMachine, args):
    a = machine.memory.load(args[0].raw, INT)
    b = machine.memory.load(args[1].raw, INT)
    return (a > b) - (a < b)


def _int_array(machine: FakeMachine, values: list[int]) -> int:
    base = machine.buffer(4 * len(values))
    for index, value in enumerate(values):
        machine.memory.store(base + 4 * index, INT, value)
    return base


def _read_ints(machine: FakeMachine, base: int, count: int) -> list[int]:
    return [machine.memory.load(base + 4 * i, INT) for i in range(count)]


def test_qsort_sorts_through_callback(m: FakeMachine) -> None:
    values = [5, -2, 9, 0, 3, 3, 8, 1]
    base = _int_array(m, values)
    fid = m.add_callback(_int_compare)
    m.call("qsort", m.ptr(base), m.size(len(values)), m.size(4), m.int_(fid))
    assert _read_ints(m, base, len(values)) == sorted(values)


def test_qsort_is_stable(m: FakeMachine) -> None:
    # Sort pairs (key, tag) by key only: equal keys must keep their order.
    pairs = [(2, 0), (1, 1), (2, 2), (1, 3), (0, 4)]
    base = m.buffer(8 * len(pairs))
    for index, (key, tag) in enumerate(pairs):
        m.memory.store(base + 8 * index, INT, key)
        m.memory.store(base + 8 * index + 4, INT, tag)

    def by_key(machine: FakeMachine, args):
        return machine.memory.load(args[0].raw, INT) - machine.memory.load(args[1].raw, INT)

    m.call("qsort", m.ptr(base), m.size(len(pairs)), m.size(8), m.int_(m.add_callback(by_key)))
    tags = [m.memory.load(base + 8 * i + 4, INT) for i in range(len(pairs))]
    assert tags == [4, 1, 3, 0, 2]


def test_qsort_passes_events_from_comparator(m: FakeMachine) -> None:
    def chatty(machine: FakeMachine, args):
        yield Output("stdout", "?")
        return _int_compare(machine, args)

    base = _int_array(m, [2, 1])
    m.call("qsort", m.ptr(base), m.size(2), m.size(4), m.int_(m.add_callback(chatty)))
    assert m.stdout == "?"
    assert _read_ints(m, base, 2) == [1, 2]


def test_qsort_pointers_point_into_the_array(m: FakeMachine) -> None:
    base = _int_array(m, [3, 1, 2])
    seen: list[int] = []

    def spy(machine: FakeMachine, args):
        seen.extend(arg.raw for arg in args)
        return _int_compare(machine, args)

    m.call("qsort", m.ptr(base), m.size(3), m.size(4), m.int_(m.add_callback(spy)))
    assert all(base <= addr < base + 12 for addr in seen)


def test_qsort_null_comparator_traps(m: FakeMachine) -> None:
    base = _int_array(m, [2, 1])
    with pytest.raises(TrapError) as caught:
        m.call("qsort", m.ptr(base), m.size(2), m.size(4), m.int_(0))
    assert caught.value.code == "null_dereference"


def test_bsearch_finds_and_misses(m: FakeMachine) -> None:
    base = _int_array(m, [1, 3, 5, 7, 9, 11])
    fid = m.add_callback(_int_compare)
    key = _int_array(m, [7])
    found = m.call("bsearch", m.ptr(key), m.ptr(base), m.size(6), m.size(4), m.int_(fid)).raw
    assert found == base + 12
    missing = _int_array(m, [4])
    result = m.call("bsearch", m.ptr(missing), m.ptr(base), m.size(6), m.size(4), m.int_(fid))
    assert result.raw == 0


# --- exit, abort, getenv, system ----------------------------------------------------------

def test_exit_raises_program_exit(m: FakeMachine) -> None:
    with pytest.raises(FakeProgramExit) as caught:
        m.call("exit", m.int_(3))
    assert caught.value.code == 3


def test_abort_traps(m: FakeMachine) -> None:
    with pytest.raises(TrapError) as caught:
        m.call("abort")
    assert caught.value.code == "abort_called"
    assert caught.value.friendly


def test_getenv_reads_io_env() -> None:
    m = FakeMachine(env={"HOME": "/home/user"})
    home = m.call("getenv", lit(m, "HOME")).raw
    assert m.text_at(home) == "/home/user"
    assert m.call("getenv", lit(m, "NOPE")).raw == 0


def test_system_returns_minus_one(m: FakeMachine) -> None:
    assert m.call("system", lit(m, "ls")).raw == -1
