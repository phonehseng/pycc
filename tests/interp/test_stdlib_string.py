"""<string.h>: lengths, copies, comparisons, searching, strtok and raw memory."""
from __future__ import annotations

import pytest

from pycc.interp import constructs
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.traps import TrapError


@pytest.fixture
def m() -> FakeMachine:
    return FakeMachine()


def lit(machine: FakeMachine, text: str):
    return machine.ptr(machine.literal(text))


def test_strlen(m: FakeMachine) -> None:
    assert m.call("strlen", lit(m, "hello")).raw == 5
    assert m.call("strlen", lit(m, "")).raw == 0


def test_strlen_counts_construct(m: FakeMachine) -> None:
    m.call("strlen", lit(m, "x"))
    assert m.constructs[constructs.STRING_CALL] == 1


def test_strlen_null_traps_with_function_name(m: FakeMachine) -> None:
    with pytest.raises(TrapError) as caught:
        m.call("strlen", m.null())
    assert caught.value.code == "null_dereference"
    assert caught.value.message.startswith("strlen():")


def test_strlen_unterminated_traps(m: FakeMachine) -> None:
    buf = m.buffer(4, b"x")
    with pytest.raises(TrapError) as caught:
        m.call("strlen", m.ptr(buf))
    assert caught.value.code == "string_not_terminated"


def test_strcpy_copies_and_returns_dest(m: FakeMachine) -> None:
    dest = m.buffer(16, b"X")
    result = m.call("strcpy", m.ptr(dest), lit(m, "seed"))
    assert result.raw == dest
    assert m.text_at(dest) == "seed"


def test_strcpy_overflow_traps(m: FakeMachine) -> None:
    dest = m.buffer(4)
    with pytest.raises(TrapError) as caught:
        m.call("strcpy", m.ptr(dest), lit(m, "pumpkin"))
    assert caught.value.code == "out_of_bounds"


def test_strcpy_into_literal_traps(m: FakeMachine) -> None:
    with pytest.raises(TrapError) as caught:
        m.call("strcpy", lit(m, "abc"), lit(m, "xyz"))
    assert caught.value.code == "write_to_literal"


def test_strncpy_pads_with_zeros(m: FakeMachine) -> None:
    dest = m.buffer(8, b"X")
    m.call("strncpy", m.ptr(dest), lit(m, "ab"), m.size(6))
    assert m.memory.read_bytes(dest, 8) == b"ab\0\0\0\0XX"


def test_strncpy_does_not_terminate_long_source(m: FakeMachine) -> None:
    dest = m.buffer(8, b"X")
    m.call("strncpy", m.ptr(dest), lit(m, "abcdef"), m.size(3))
    assert m.memory.read_bytes(dest, 4) == b"abcX"


def test_strncpy_zero_count_touches_nothing(m: FakeMachine) -> None:
    dest = m.buffer(4, b"X")
    m.call("strncpy", m.ptr(dest), m.null(), m.size(0))
    assert m.memory.read_bytes(dest, 1) == b"X"


def test_strcat_appends(m: FakeMachine) -> None:
    dest = m.string("corn", size=16)
    m.call("strcat", m.ptr(dest), lit(m, "field"))
    assert m.text_at(dest) == "cornfield"


def test_strncat_appends_at_most_n_and_terminates(m: FakeMachine) -> None:
    dest = m.string("ab", size=16)
    m.call("strncat", m.ptr(dest), lit(m, "cdefg"), m.size(2))
    assert m.text_at(dest) == "abcd"


def test_strdup_allocates_copy(m: FakeMachine) -> None:
    copy = m.call("strdup", lit(m, "wheat")).raw
    assert m.text_at(copy) == "wheat"
    assert m.memory.object_at(copy).kind == "heap"
    assert m.constructs[constructs.MALLOC] == 1
    m.call("free", m.ptr(copy))
    assert m.memory.leaks() == []


@pytest.mark.parametrize("left,right,sign", [
    ("abc", "abc", 0), ("abc", "abd", -1), ("abd", "abc", 1), ("ab", "abc", -1),
    ("abc", "ab", 1), ("", "", 0), ("B", "a", -1),
])
def test_strcmp_sign(m: FakeMachine, left: str, right: str, sign: int) -> None:
    result = m.call("strcmp", lit(m, left), lit(m, right)).raw
    assert (result > 0) - (result < 0) == sign


def test_strcmp_returns_byte_difference(m: FakeMachine) -> None:
    assert m.call("strcmp", lit(m, "a"), lit(m, "c")).raw == -2


def test_strcmp_compares_as_unsigned_char(m: FakeMachine) -> None:
    high = m.string(b"\xe9")
    assert m.call("strcmp", m.ptr(high), lit(m, "a")).raw > 0


def test_strncmp_limits_comparison(m: FakeMachine) -> None:
    assert m.call("strncmp", lit(m, "apple"), lit(m, "apricot"), m.size(2)).raw == 0
    assert m.call("strncmp", lit(m, "apple"), lit(m, "apricot"), m.size(3)).raw < 0
    assert m.call("strncmp", lit(m, "ab"), lit(m, "abc"), m.size(5)).raw < 0


def test_strchr_and_strrchr(m: FakeMachine) -> None:
    text = m.literal("banana")
    assert m.call("strchr", m.ptr(text), m.int_(ord("a"))).raw == text + 1
    assert m.call("strrchr", m.ptr(text), m.int_(ord("a"))).raw == text + 5
    assert m.call("strchr", m.ptr(text), m.int_(ord("z"))).raw == 0


def test_strchr_finds_terminator(m: FakeMachine) -> None:
    text = m.literal("abc")
    assert m.call("strchr", m.ptr(text), m.int_(0)).raw == text + 3


def test_strstr(m: FakeMachine) -> None:
    text = m.literal("haystack needle stack")
    assert m.call("strstr", m.ptr(text), lit(m, "needle")).raw == text + 9
    assert m.call("strstr", m.ptr(text), lit(m, "pin")).raw == 0
    assert m.call("strstr", m.ptr(text), lit(m, "")).raw == text


def _tokens(m: FakeMachine, text: str, delims: str) -> list[str]:
    buf = m.string(text)
    found = []
    token = m.call("strtok", m.ptr(buf), lit(m, delims)).raw
    while token:
        found.append(m.text_at(token))
        token = m.call("strtok", m.null(), lit(m, delims)).raw
    return found


def test_strtok_sequence(m: FakeMachine) -> None:
    assert _tokens(m, "wheat,corn,,beet", ",") == ["wheat", "corn", "beet"]


def test_strtok_skips_leading_and_trailing_delimiters(m: FakeMachine) -> None:
    assert _tokens(m, "  a b  c ", " ") == ["a", "b", "c"]


def test_strtok_several_delimiters(m: FakeMachine) -> None:
    assert _tokens(m, "x=1;y=2", "=;") == ["x", "1", "y", "2"]


def test_strtok_only_delimiters_returns_null(m: FakeMachine) -> None:
    assert _tokens(m, ",,,", ",") == []


def test_strtok_writes_terminators_into_the_string(m: FakeMachine) -> None:
    buf = m.string("a,b")
    m.call("strtok", m.ptr(buf), lit(m, ","))
    assert m.memory.read_bytes(buf, 4) == b"a\0b\0"


def test_strtok_null_without_previous_string_returns_null(m: FakeMachine) -> None:
    assert m.call("strtok", m.null(), lit(m, ",")).raw == 0


def test_strtok_on_literal_traps(m: FakeMachine) -> None:
    with pytest.raises(TrapError) as caught:
        m.call("strtok", lit(m, "a,b"), lit(m, ","))
    assert caught.value.code == "write_to_literal"


def test_memset(m: FakeMachine) -> None:
    buf = m.buffer(8)
    result = m.call("memset", m.ptr(buf), m.int_(0x41 + 256), m.size(3))
    assert result.raw == buf
    assert m.memory.read_bytes(buf, 4) == b"AAA\0"


def test_memcpy(m: FakeMachine) -> None:
    src, dest = m.string("12345"), m.buffer(8)
    m.call("memcpy", m.ptr(dest), m.ptr(src), m.size(6))
    assert m.text_at(dest) == "12345"


def test_memmove_overlap_forward(m: FakeMachine) -> None:
    buf = m.string("abcdef")
    m.call("memmove", m.ptr(buf + 2), m.ptr(buf), m.size(4))
    assert m.text_at(buf) == "ababcd"


def test_memmove_overlap_backward(m: FakeMachine) -> None:
    buf = m.string("abcdef")
    m.call("memmove", m.ptr(buf), m.ptr(buf + 2), m.size(4))
    assert m.text_at(buf) == "cdefef"


def test_memcpy_keeps_uninitialised_state(m: FakeMachine) -> None:
    src = m.memory.malloc(4)            # never written: garbage
    dest = m.buffer(4)
    m.call("memcpy", m.ptr(dest), m.ptr(src), m.size(4))
    assert m.memory.init_mask(dest, 4) == b"\0\0\0\0"


def test_memcmp(m: FakeMachine) -> None:
    a, b = m.string(b"ab\x01"), m.string(b"ab\x05")
    assert m.call("memcmp", m.ptr(a), m.ptr(b), m.size(3)).raw == -4
    assert m.call("memcmp", m.ptr(a), m.ptr(b), m.size(2)).raw == 0


def test_strerror_returns_glibc_text(m: FakeMachine) -> None:
    text = m.call("strerror", m.int_(2)).raw
    assert m.text_at(text) == "No such file or directory"
    edom = m.call("strerror", m.int_(33)).raw
    assert m.text_at(edom) == "Numerical argument out of domain"


def test_strerror_text_is_read_only(m: FakeMachine) -> None:
    text = m.call("strerror", m.int_(2)).raw
    assert m.memory.object_at(text).kind == "literal"
