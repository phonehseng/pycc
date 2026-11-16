"""``<stdlib.h>``: heap memory, number parsing, rand, sorting, searching, exit.

Heap functions go straight to the simulated memory, so its traps (``double_free``,
``invalid_free``, ``use_after_free``) and its ``heap_exhausted`` warning reach the
user unchanged. ``qsort`` and ``bsearch`` call the program's own comparison function
back through ``Machine.call_function``; any events that function yields (output, device
actions) pass through, so a ``printf`` inside a comparator shows up in order.

``rand`` is the sample generator printed in the C standard itself, so ``srand(1)`` gives the same numbers as the textbook: 16838, 5758, 10113...
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Generator
from typing import Any

from pycc.core.concepts import Concept
from pycc.interp import constructs
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.c_types import CHAR, INT, LONG, VOID, PointerType, wrap_int
from pycc.interp.stdlib.errno_ import EINVAL, ERANGE
from pycc.interp.stdlib.headers import LibraryFunction, counted, register_functions
from pycc.interp.stdlib.string_ import (
    char_pointer, read_c_string, require_pointer, require_room, void_pointer,
)
from pycc.interp.values import Value

HEADER = "stdlib.h"
STATE_RAND = "stdlib.rand_next"

RAND_MAX = 32767
RAND_MULTIPLIER = 1103515245
RAND_INCREMENT = 12345

_SPACE = frozenset(b" \t\n\v\f\r")
_FLOAT_PREFIX = re.compile(
    r"[ \t\n\v\f\r]*[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
    re.IGNORECASE)

Events = Generator[Any, Any, Any]


# --- number parsing ----------------------------------------------------------------------

def _digit(byte: int) -> int:
    """The value of a digit in bases up to 36 (99 if it is not a digit)."""
    if 48 <= byte <= 57:
        return byte - 48
    if 97 <= (byte | 0x20) <= 122:
        return (byte | 0x20) - 97 + 10
    return 99


def parse_c_long(data: bytes, base: int) -> tuple[int, int, bool]:
    """Parse like ``strtol``: (value, bytes consumed, overflowed).

    Leading white space and a sign are allowed; base 0 means "decide from the prefix"
    (``0x`` hex, ``0`` octal, else decimal) and base 16 accepts an optional ``0x``.
    ``consumed`` is 0 when there were no digits at all. On overflow the value is
    clamped to LONG_MIN or LONG_MAX, as C requires.
    """
    index = 0
    while index < len(data) and data[index] in _SPACE:
        index += 1
    negative = False
    if index < len(data) and data[index] in b"+-":
        negative = data[index] == ord("-")
        index += 1
    has_hex_prefix = (data[index:index + 2].lower() == b"0x" and index + 2 < len(data)
                      and _digit(data[index + 2]) < 16)
    if base in (0, 16) and has_hex_prefix:
        base = 16
        index += 2
    elif base == 0:
        base = 8 if data[index:index + 1] == b"0" else 10
    start = index
    value = 0
    while index < len(data) and _digit(data[index]) < base:
        value = value * base + _digit(data[index])
        index += 1
    if index == start:
        return 0, 0, False
    value = -value if negative else value
    if value > LONG.max:
        return LONG.max, index, True
    if value < LONG.min:
        return LONG.min, index, True
    return value, index, False


def parse_c_double(data: bytes) -> tuple[float, int, bool]:
    """Parse like ``strtod``: (value, bytes consumed, out of range).

    "Out of range" covers overflow ("1e999" becomes infinity) and underflow
    ("1e-999" becomes 0.0 although its digits are not all zero); glibc sets
    ``errno = ERANGE`` for both.
    """
    match = _FLOAT_PREFIX.match(data.decode("latin-1"))
    if match is None:
        return 0.0, 0, False
    text = match.group(0)
    value = float(text)
    mantissa = text.lower().split("e")[0]
    overflowed = math.isinf(value) and "inf" not in text.lower()
    underflowed = value == 0.0 and any(digit in mantissa for digit in "123456789")
    return value, match.end(), overflowed or underflowed


def _store_end(machine: Machine, endptr: int, addr: int) -> None:
    """``*endptr = addr`` when the caller asked for the end position."""
    if endptr != 0:
        machine.memory.store(endptr, PointerType(CHAR), addr)


def c_atoi(machine: Machine, args: list[Value]) -> Value:
    """``int atoi(const char *nptr)``: the leading integer, 0 if there is none."""
    value, _, _ = parse_c_long(read_c_string(machine, args[0].raw, "atoi", "the string"), 10)
    return machine.make_int(wrap_int(value, INT), INT)


def c_atol(machine: Machine, args: list[Value]) -> Value:
    """``long atol(const char *nptr)``."""
    value, _, _ = parse_c_long(read_c_string(machine, args[0].raw, "atol", "the string"), 10)
    return machine.make_int(value, LONG)


def c_atof(machine: Machine, args: list[Value]) -> Value:
    """``double atof(const char *nptr)``."""
    value, _, _ = parse_c_double(read_c_string(machine, args[0].raw, "atof", "the string"))
    return machine.make_double(value)


def c_strtol(machine: Machine, args: list[Value]) -> Value:
    """``long strtol(const char *nptr, char **endptr, int base)``.

    ``*endptr`` gets the first unparsed character (``nptr`` itself if there were no
    digits); overflow returns LONG_MAX/LONG_MIN with ``errno = ERANGE``; a base other
    than 0 or 2..36 sets ``errno = EINVAL``.
    """
    nptr, endptr, base = args[0].raw, args[1].raw, args[2].raw
    data = read_c_string(machine, nptr, "strtol", "the string")
    if base != 0 and not 2 <= base <= 36:
        machine.set_errno(EINVAL)
        _store_end(machine, endptr, nptr)
        return machine.make_int(0, LONG)
    value, consumed, overflowed = parse_c_long(data, base)
    if overflowed:
        machine.set_errno(ERANGE)
    _store_end(machine, endptr, nptr + consumed)
    return machine.make_int(value, LONG)


def c_strtod(machine: Machine, args: list[Value]) -> Value:
    """``double strtod(const char *nptr, char **endptr)``: ERANGE on overflow or
    underflow."""
    nptr, endptr = args[0].raw, args[1].raw
    value, consumed, overflowed = parse_c_double(
        read_c_string(machine, nptr, "strtod", "the string"))
    if overflowed:
        machine.set_errno(ERANGE)
    _store_end(machine, endptr, nptr + consumed)
    return machine.make_double(value)


def c_abs(machine: Machine, args: list[Value]) -> Value:
    """``int abs(int j)``. ``abs(INT_MIN)`` wraps back to INT_MIN, as on real hardware."""
    return machine.make_int(wrap_int(abs(args[0].raw), INT), INT)


def c_labs(machine: Machine, args: list[Value]) -> Value:
    """``long labs(long j)``."""
    return machine.make_int(wrap_int(abs(args[0].raw), LONG), LONG)


# --- random numbers ----------------------------------------------------------------------

def next_rand(state: int) -> tuple[int, int]:
    """One step of the C standard's sample generator: (new state, rand() result)."""
    state = (state * RAND_MULTIPLIER + RAND_INCREMENT) & 0xFFFFFFFF
    return state, (state // 65536) % (RAND_MAX + 1)


def c_rand(machine: Machine, args: list[Value]) -> Value:
    """``int rand(void)``: 0..RAND_MAX; without srand the seed is 1."""
    state, result = next_rand(machine.state.get(STATE_RAND, 1))
    machine.state[STATE_RAND] = state
    return machine.make_int(result)


def c_srand(machine: Machine, args: list[Value]) -> None:
    """``void srand(unsigned int seed)``: restart the sequence from ``seed``."""
    machine.state[STATE_RAND] = args[0].raw & 0xFFFFFFFF


# --- heap memory ---------------------------------------------------------------------------

@counted(constructs.MALLOC)
def c_malloc(machine: Machine, args: list[Value]) -> Value:
    """``void *malloc(size_t size)``: uninitialised bytes, or NULL when the heap is full."""
    return void_pointer(machine, machine.malloc(args[0].raw))


@counted(constructs.MALLOC)
def c_calloc(machine: Machine, args: list[Value]) -> Value:
    """``void *calloc(size_t nmemb, size_t size)``: zeroed bytes, or NULL."""
    return void_pointer(machine, machine.memory.calloc(args[0].raw, args[1].raw))


@counted(constructs.REALLOC)
def c_realloc(machine: Machine, args: list[Value]) -> Value:
    """``void *realloc(void *ptr, size_t size)``: the block always moves, so a stale
    copy of the old pointer traps ``use_after_free`` when used."""
    return void_pointer(machine, machine.memory.realloc(args[0].raw, args[1].raw))


@counted(constructs.FREE)
def c_free(machine: Machine, args: list[Value]) -> None:
    """``void free(void *ptr)``: ``free(NULL)`` is allowed and does nothing."""
    machine.free(args[0].raw)


# --- ending the program -----------------------------------------------------------------

def c_exit(machine: Machine, args: list[Value]) -> None:
    """``void exit(int status)``: end the program now with this exit code."""
    machine.exit(args[0].raw)


def c_abort(machine: Machine, args: list[Value]) -> None:
    """``void abort(void)``: stop at once, like a crash (exit code 134)."""
    machine.trap(
        "abort_called",
        "abort() was called",
        "abort() stops the program immediately, without cleaning up, and the shell "
        "reports 'Aborted'. Programs call it when they find something impossible; if "
        "you only want to stop, return from main or call exit() instead.",
        Concept.ERROR_HANDLING)


# --- sorting and searching ---------------------------------------------------------------

def _comparator(machine: Machine, value: Value, function: str) -> Callable[[int, int], Events]:
    """A generator function comparing two element addresses via the C callback."""
    fid = int(value.raw)
    if fid == 0:
        machine.trap(
            "null_dereference",
            f"{function}(): the comparison function is a NULL pointer",
            f"{function}() needs the name of a function that compares two elements, such "
            "as int compare(const void *a, const void *b). It was given NULL.",
            Concept.POINTERS, function=function)

    def compare(left: int, right: int) -> Events:
        result = yield from machine.call_function(
            fid, [machine.make_pointer(left, VOID), machine.make_pointer(right, VOID)])
        raw = result.raw if isinstance(result, Value) else result
        return wrap_int(int(raw), INT)
    return compare


def _merge(left: list[int], right: list[int], compare: Callable[[int, int], Events],
           addrs: list[int]) -> Generator[Any, Any, list[int]]:
    """Merge two sorted runs of element indexes; ties keep their order (stable)."""
    merged: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if (yield from compare(addrs[left[i]], addrs[right[j]])) <= 0:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    return merged + left[i:] + right[j:]


def _merge_sort(count: int, compare: Callable[[int, int], Events],
                addrs: list[int]) -> Generator[Any, Any, list[int]]:
    """Bottom-up merge sort of the indexes 0..count-1.

    Elements stay where they are while sorting, so every pointer handed to the
    comparator points into the user's own array; only the final order is written.
    glibc's qsort is also a merge sort, so equal elements keep their order there too.
    """
    order = list(range(count))
    width = 1
    while width < count:
        runs: list[int] = []
        for low in range(0, count, 2 * width):
            runs += yield from _merge(order[low:low + width], order[low + width:low + 2 * width],
                                      compare, addrs)
        order = runs
        width *= 2
    return order


def c_qsort(machine: Machine, args: list[Value]) -> Events:
    """``void qsort(void *base, size_t nmemb, size_t size, int (*compar)(...))``."""
    base, count, size = args[0].raw, args[1].raw, args[2].raw
    compare = _comparator(machine, args[3], "qsort")
    if count <= 1 or size == 0:
        return None
    require_pointer(machine, base, "qsort", "the array")
    # Check the whole array first: a bad nmemb must trap, not build a huge list.
    require_room(machine, base, count * size, "write")
    addrs = [base + index * size for index in range(count)]
    order = yield from _merge_sort(count, compare, addrs)
    elements = [machine.read_bytes(addrs[index], size) for index in order]
    for addr, data in zip(addrs, elements):
        machine.write_bytes(addr, data)
    return None


def c_bsearch(machine: Machine, args: list[Value]) -> Events:
    """``void *bsearch(const void *key, const void *base, size_t nmemb, size_t size,
    int (*compar)(...))``: binary search of a sorted array; NULL if not found."""
    key, base, count, size = args[0].raw, args[1].raw, args[2].raw, args[3].raw
    compare = _comparator(machine, args[4], "bsearch")
    low, high = 0, count
    while low < high:
        middle = (low + high) // 2
        element = base + middle * size
        result = yield from compare(key, element)
        if result < 0:
            high = middle
        elif result > 0:
            low = middle + 1
        else:
            return void_pointer(machine, element)
    return void_pointer(machine, 0)


# --- the environment --------------------------------------------------------------------

def c_getenv(machine: Machine, args: list[Value]) -> Value:
    """``char *getenv(const char *name)``: the variable's value or NULL (reads io.env)."""
    name = read_c_string(machine, args[0].raw, "getenv", "the name").decode("utf-8", "replace")
    env = getattr(machine.io, "env", None) or {}
    value = env.get(name)
    if value is None:
        return char_pointer(machine, 0)
    # Read-only data, deduplicated: the same value always comes back at one address.
    return char_pointer(machine, machine.memory.intern_string(value.encode("utf-8")))


def c_system(machine: Machine, args: list[Value]) -> Value:
    """``int system(const char *command)``: programs cannot start other programs in
    pycc, so this always fails with -1."""
    return machine.make_int(-1)


FUNCTIONS: tuple[LibraryFunction, ...] = (
    LibraryFunction("malloc", "void *malloc(size_t size);", c_malloc, Concept.DYNAMIC_MEMORY),
    LibraryFunction("calloc", "void *calloc(size_t nmemb, size_t size);", c_calloc,
                    Concept.DYNAMIC_MEMORY),
    LibraryFunction("realloc", "void *realloc(void *ptr, size_t size);", c_realloc,
                    Concept.DYNAMIC_MEMORY),
    LibraryFunction("free", "void free(void *ptr);", c_free, Concept.DYNAMIC_MEMORY),
    LibraryFunction("atoi", "int atoi(const char *nptr);", c_atoi, Concept.STRINGS),
    LibraryFunction("atol", "long atol(const char *nptr);", c_atol, Concept.STRINGS),
    LibraryFunction("atof", "double atof(const char *nptr);", c_atof, Concept.STRINGS),
    LibraryFunction("strtol", "long strtol(const char *nptr, char **endptr, int base);",
                    c_strtol, Concept.ERROR_HANDLING),
    LibraryFunction("strtod", "double strtod(const char *nptr, char **endptr);", c_strtod,
                    Concept.ERROR_HANDLING),
    LibraryFunction("abs", "int abs(int j);", c_abs, Concept.OPERATORS),
    LibraryFunction("labs", "long labs(long j);", c_labs, Concept.OPERATORS),
    LibraryFunction("rand", "int rand(void);", c_rand, Concept.FUNCTIONS),
    LibraryFunction("srand", "void srand(unsigned int seed);", c_srand, Concept.FUNCTIONS),
    LibraryFunction("exit", "void exit(int status);", c_exit, Concept.ERROR_HANDLING),
    LibraryFunction("abort", "void abort(void);", c_abort, Concept.ERROR_HANDLING),
    LibraryFunction("qsort", "void qsort(void *base, size_t nmemb, size_t size, "
                    "int (*compar)(const void *, const void *));", c_qsort, Concept.POINTERS),
    LibraryFunction("bsearch", "void *bsearch(const void *key, const void *base, size_t nmemb, "
                    "size_t size, int (*compar)(const void *, const void *));", c_bsearch,
                    Concept.POINTERS),
    LibraryFunction("getenv", "char *getenv(const char *name);", c_getenv,
                    Concept.PROGRAM_ARGS),
    LibraryFunction("system", "int system(const char *command);", c_system,
                    Concept.SHELL_SCRIPTING),
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<stdlib.h>`` function."""
    register_functions(registry, HEADER, FUNCTIONS)
