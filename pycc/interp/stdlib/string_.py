"""``<string.h>``: C strings and raw memory, plus the string helpers the library shares.

Every function works on the simulated memory through checked reads and writes, so the
classic mistakes are caught where they happen: ``strcpy`` into a buffer that is too
small traps ``out_of_bounds`` naming the buffer, ``strtok`` on a string literal traps
``write_to_literal``, and a missing ``'\\0'`` traps ``string_not_terminated``.

The helpers at the top (``require_pointer``, ``read_c_string``, ``char_pointer``) are
used by the other stdlib modules too: they give every "NULL where a string was
expected" the same clear message.
"""
from __future__ import annotations

from pycc.core.concepts import Concept
from pycc.interp import constructs
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.c_types import CHAR, INT, SIZE_T, VOID
from pycc.interp.hostio import ENOMEM
from pycc.interp.stdlib.errno_ import c_strerror
from pycc.interp.stdlib.headers import LibraryFunction, counted, register_functions
from pycc.interp.values import Value

HEADER = "string.h"
STATE_STRTOK = "string.strtok_next"


# --- helpers shared by the whole library ---------------------------------------------------

def require_pointer(machine: Machine, addr: int, function: str, what: str) -> None:
    """Trap ``null_dereference`` with a message naming the function and argument."""
    if addr == 0:
        machine.trap(
            "null_dereference",
            f"{function}(): {what} is a NULL pointer",
            f"{function}() was given NULL for {what}. NULL points at nothing, so there is "
            "nothing there to read or write. Check the pointer first (fopen, malloc, "
            "strchr and getenv all return NULL when they fail).",
            Concept.POINTERS, function=function)


def read_c_string(machine: Machine, addr: int, function: str, what: str,
                  limit: int | None = None) -> bytes:
    """The bytes of the C string at ``addr`` (without its ``'\\0'``).

    ``limit`` stops early without needing a terminator (``strncpy``, ``%.3s``).
    Without a limit the string may be as long as memory itself: the memory model
    still traps if it runs off the end of its object.
    """
    require_pointer(machine, addr, function, what)
    if limit is None:
        return machine.memory.read_cstring(addr, machine.memory.size)
    if limit <= 0:
        return b""
    room = machine.memory.check_access(addr, 1, "read").end - addr
    if limit < room:
        return machine.memory.read_cstring(addr, limit)
    # The limit reaches the end of the object. Memory.read_cstring would call a
    # missing '\0' there "unterminated", but C's bounded readers (%.3s, strncpy,
    # strncmp) never look past the limit, so a char[3] holding "abc" is fine.
    data = machine.memory.read_bytes(addr, room)
    if 0 in data or limit > room:
        return machine.memory.read_cstring(addr, limit)   # finds the '\0', or traps
    return data


def require_room(machine: Machine, addr: int, count: int, mode: str) -> None:
    """Check that ``count`` bytes at ``addr`` exist before building them in Python.

    A wrong ``size_t`` such as ``(size_t)-1`` would otherwise make Python allocate
    exabytes before the memory model could object; checking first turns it into the
    ``out_of_bounds`` trap the user should see.
    """
    machine.memory.check_access(addr, count, mode)


def char_pointer(machine: Machine, addr: int) -> Value:
    """A ``char *`` result (``NULL`` when ``addr`` is 0)."""
    return machine.make_pointer(addr, CHAR)


def void_pointer(machine: Machine, addr: int) -> Value:
    """A ``void *`` result."""
    return machine.make_pointer(addr, VOID)


def compare_bytes(left: bytes, right: bytes) -> int:
    """C's comparison result: the difference of the first differing bytes (as glibc).

    A sentinel ``'\\0'`` is appended to both so that "abc" vs "ab" compares 'c' with
    the shorter string's terminator, exactly like walking two C strings.
    """
    for a, b in zip(left + b"\0", right + b"\0"):
        if a != b:
            return a - b
    return 0


# --- length, copying, joining ----------------------------------------------------------------

@counted(constructs.STRING_CALL)
def c_strlen(machine: Machine, args: list[Value]) -> Value:
    """``size_t strlen(const char *s)``: bytes before the ``'\\0'``."""
    text = read_c_string(machine, args[0].raw, "strlen", "the string")
    return machine.make_int(len(text), SIZE_T)


@counted(constructs.STRING_CALL)
def c_strcpy(machine: Machine, args: list[Value]) -> Value:
    """``char *strcpy(char *dest, const char *src)``: copy including the ``'\\0'``."""
    dest, src = args[0].raw, args[1].raw
    text = read_c_string(machine, src, "strcpy", "the source string")
    require_pointer(machine, dest, "strcpy", "the destination")
    machine.write_bytes(dest, text + b"\0")
    return char_pointer(machine, dest)


@counted(constructs.STRING_CALL)
def c_strncpy(machine: Machine, args: list[Value]) -> Value:
    """``char *strncpy(char *dest, const char *src, size_t n)``.

    Writes exactly ``n`` bytes: the source, then ``'\\0'`` padding. When the source has
    ``n`` or more characters no terminator is written at all, the classic surprise.
    """
    dest, src, count = args[0].raw, args[1].raw, args[2].raw
    if count == 0:
        return char_pointer(machine, dest)
    text = read_c_string(machine, src, "strncpy", "the source string", limit=count)
    require_pointer(machine, dest, "strncpy", "the destination")
    require_room(machine, dest, count, "write")
    machine.write_bytes(dest, text + bytes(count - len(text)))
    return char_pointer(machine, dest)


@counted(constructs.STRING_CALL)
def c_strcat(machine: Machine, args: list[Value]) -> Value:
    """``char *strcat(char *dest, const char *src)``: append after dest's text."""
    dest, src = args[0].raw, args[1].raw
    start = len(read_c_string(machine, dest, "strcat", "the destination string"))
    text = read_c_string(machine, src, "strcat", "the source string")
    machine.write_bytes(dest + start, text + b"\0")
    return char_pointer(machine, dest)


@counted(constructs.STRING_CALL)
def c_strncat(machine: Machine, args: list[Value]) -> Value:
    """``char *strncat(char *dest, const char *src, size_t n)``: append at most n
    characters and always a ``'\\0'``."""
    dest, src, count = args[0].raw, args[1].raw, args[2].raw
    start = len(read_c_string(machine, dest, "strncat", "the destination string"))
    text = read_c_string(machine, src, "strncat", "the source string", limit=count)
    machine.write_bytes(dest + start, text + b"\0")
    return char_pointer(machine, dest)


@counted(constructs.STRING_CALL)
def c_strdup(machine: Machine, args: list[Value]) -> Value:
    """``char *strdup(const char *s)``: a malloc'd copy (the caller must free it)."""
    text = read_c_string(machine, args[0].raw, "strdup", "the string")
    machine.count_construct(constructs.MALLOC)
    addr = machine.malloc(len(text) + 1)
    if addr == 0:
        machine.set_errno(ENOMEM)
        return char_pointer(machine, 0)
    machine.write_bytes(addr, text + b"\0")
    return char_pointer(machine, addr)


# --- comparing and searching ------------------------------------------------------------

@counted(constructs.STRING_CALL)
def c_strcmp(machine: Machine, args: list[Value]) -> Value:
    """``int strcmp(const char *s1, const char *s2)``: <0, 0 or >0."""
    left = read_c_string(machine, args[0].raw, "strcmp", "the first string")
    right = read_c_string(machine, args[1].raw, "strcmp", "the second string")
    return machine.make_int(compare_bytes(left, right), INT)


@counted(constructs.STRING_CALL)
def c_strncmp(machine: Machine, args: list[Value]) -> Value:
    """``int strncmp(const char *s1, const char *s2, size_t n)``: compare at most n."""
    count = args[2].raw
    if count == 0:
        return machine.make_int(0, INT)
    left = read_c_string(machine, args[0].raw, "strncmp", "the first string", limit=count)
    right = read_c_string(machine, args[1].raw, "strncmp", "the second string", limit=count)
    return machine.make_int(compare_bytes(left[:count], right[:count]), INT)


@counted(constructs.STRING_CALL)
def c_strchr(machine: Machine, args: list[Value]) -> Value:
    """``char *strchr(const char *s, int c)``: first ``c``, or NULL.

    Searching for ``'\\0'`` finds the terminator, as C specifies.
    """
    addr = args[0].raw
    text = read_c_string(machine, addr, "strchr", "the string")
    index = (text + b"\0").find(args[1].raw & 0xFF)
    return char_pointer(machine, addr + index if index >= 0 else 0)


@counted(constructs.STRING_CALL)
def c_strrchr(machine: Machine, args: list[Value]) -> Value:
    """``char *strrchr(const char *s, int c)``: last ``c``, or NULL."""
    addr = args[0].raw
    text = read_c_string(machine, addr, "strrchr", "the string")
    index = (text + b"\0").rfind(args[1].raw & 0xFF)
    return char_pointer(machine, addr + index if index >= 0 else 0)


@counted(constructs.STRING_CALL)
def c_strstr(machine: Machine, args: list[Value]) -> Value:
    """``char *strstr(const char *haystack, const char *needle)``: first match or NULL."""
    addr = args[0].raw
    haystack = read_c_string(machine, addr, "strstr", "the haystack")
    needle = read_c_string(machine, args[1].raw, "strstr", "the needle")
    index = haystack.find(needle)
    return char_pointer(machine, addr + index if index >= 0 else 0)


def _span_length(text: bytes, accept: bytes, inside: bool) -> int:
    """How many leading bytes of ``text`` are (``inside``) or are not in ``accept``."""
    for index, byte in enumerate(text):
        if (byte in accept) != inside:
            return index
    return len(text)


@counted(constructs.STRING_CALL)
def c_strspn(machine: Machine, args: list[Value]) -> Value:
    """``size_t strspn(const char *s, const char *accept)``: length of the leading run
    made only of bytes from ``accept``."""
    text = read_c_string(machine, args[0].raw, "strspn", "the string")
    accept = read_c_string(machine, args[1].raw, "strspn", "the accepted characters")
    return machine.make_int(_span_length(text, accept, inside=True), SIZE_T)


@counted(constructs.STRING_CALL)
def c_strcspn(machine: Machine, args: list[Value]) -> Value:
    """``size_t strcspn(const char *s, const char *reject)``: length of the leading run
    with no byte from ``reject``.

    ``line[strcspn(line, "\\n")] = '\\0';`` is the classic way to cut the newline
    that ``fgets`` keeps; it works whether or not the line has one.
    """
    text = read_c_string(machine, args[0].raw, "strcspn", "the string")
    reject = read_c_string(machine, args[1].raw, "strcspn", "the rejected characters")
    return machine.make_int(_span_length(text, reject, inside=False), SIZE_T)


@counted(constructs.STRING_CALL)
def c_strtok(machine: Machine, args: list[Value]) -> Value:
    """``char *strtok(char *str, const char *delim)``: split a string in place.

    The first call passes the string; later calls pass NULL and continue where the
    last one stopped. strtok writes ``'\\0'`` over each delimiter it ends a token at,
    which is why it cannot be used on a string literal. The position between calls
    lives in ``machine.state`` (C keeps it in a hidden static variable).
    """
    start = args[0].raw or machine.state.get(STATE_STRTOK, 0)
    delims = read_c_string(machine, args[1].raw, "strtok", "the delimiter string")
    if start == 0:
        return char_pointer(machine, 0)
    text = machine.memory.read_cstring(start, machine.memory.size)
    first = 0
    while first < len(text) and text[first] in delims:
        first += 1
    if first == len(text):
        machine.state[STATE_STRTOK] = 0
        return char_pointer(machine, 0)
    end = first
    while end < len(text) and text[end] not in delims:
        end += 1
    if end < len(text):
        machine.write_bytes(start + end, b"\0")
        machine.state[STATE_STRTOK] = start + end + 1
    else:
        # The token runs to the end: the next call starts on the terminator and
        # returns NULL.
        machine.state[STATE_STRTOK] = start + end
    return char_pointer(machine, start + first)


# --- raw memory -----------------------------------------------------------------------------

@counted(constructs.STRING_CALL)
def c_memset(machine: Machine, args: list[Value]) -> Value:
    """``void *memset(void *s, int c, size_t n)``: fill n bytes with ``(unsigned char)c``."""
    addr, byte, count = args[0].raw, args[1].raw & 0xFF, args[2].raw
    if count:
        require_pointer(machine, addr, "memset", "the destination")
        require_room(machine, addr, count, "write")
        machine.write_bytes(addr, bytes([byte]) * count)
    return void_pointer(machine, addr)


def _copy(machine: Machine, args: list[Value], function: str) -> Value:
    """Shared body of memcpy and memmove: a checked copy that keeps init state."""
    dest, src, count = args[0].raw, args[1].raw, args[2].raw
    if count:
        require_pointer(machine, dest, function, "the destination")
        require_pointer(machine, src, function, "the source")
        machine.memory.copy_bytes(dest, src, count)
    return void_pointer(machine, dest)


@counted(constructs.STRING_CALL)
def c_memcpy(machine: Machine, args: list[Value]) -> Value:
    """``void *memcpy(void *dest, const void *src, size_t n)``.

    Overlapping regions are undefined in C; here they copy like memmove rather than
    corrupting data in a way that depends on the CPU.
    """
    return _copy(machine, args, "memcpy")


@counted(constructs.STRING_CALL)
def c_memmove(machine: Machine, args: list[Value]) -> Value:
    """``void *memmove(void *dest, const void *src, size_t n)``: overlap-safe copy."""
    return _copy(machine, args, "memmove")


@counted(constructs.STRING_CALL)
def c_memcmp(machine: Machine, args: list[Value]) -> Value:
    """``int memcmp(const void *s1, const void *s2, size_t n)``: compare raw bytes."""
    count = args[2].raw
    if count == 0:
        return machine.make_int(0, INT)
    require_pointer(machine, args[0].raw, "memcmp", "the first block")
    require_pointer(machine, args[1].raw, "memcmp", "the second block")
    left = machine.read_bytes(args[0].raw, count)
    right = machine.read_bytes(args[1].raw, count)
    for a, b in zip(left, right):
        if a != b:
            return machine.make_int(a - b, INT)
    return machine.make_int(0, INT)


@counted(constructs.STRING_CALL)
def c_strerror_impl(machine: Machine, args: list[Value]) -> Value:
    """``char *strerror(int errnum)``: the message for an errno value.

    The text lives in read-only data, like glibc's table, so it never leaks.
    """
    text = c_strerror(args[0].raw).encode("utf-8")
    return char_pointer(machine, machine.memory.intern_string(text))


FUNCTIONS: tuple[LibraryFunction, ...] = (
    LibraryFunction("strlen", "size_t strlen(const char *s);", c_strlen, Concept.STRINGS),
    LibraryFunction("strcpy", "char *strcpy(char *dest, const char *src);", c_strcpy,
                    Concept.STRINGS),
    LibraryFunction("strncpy", "char *strncpy(char *dest, const char *src, size_t n);",
                    c_strncpy, Concept.STRINGS),
    LibraryFunction("strcat", "char *strcat(char *dest, const char *src);", c_strcat,
                    Concept.STRINGS),
    LibraryFunction("strncat", "char *strncat(char *dest, const char *src, size_t n);",
                    c_strncat, Concept.STRINGS),
    LibraryFunction("strdup", "char *strdup(const char *s);", c_strdup, Concept.DYNAMIC_MEMORY),
    LibraryFunction("strcmp", "int strcmp(const char *s1, const char *s2);", c_strcmp,
                    Concept.STRINGS),
    LibraryFunction("strncmp", "int strncmp(const char *s1, const char *s2, size_t n);",
                    c_strncmp, Concept.STRINGS),
    LibraryFunction("strchr", "char *strchr(const char *s, int c);", c_strchr, Concept.STRINGS),
    LibraryFunction("strrchr", "char *strrchr(const char *s, int c);", c_strrchr,
                    Concept.STRINGS),
    LibraryFunction("strstr", "char *strstr(const char *haystack, const char *needle);",
                    c_strstr, Concept.STRINGS),
    LibraryFunction("strspn", "size_t strspn(const char *s, const char *accept);", c_strspn,
                    Concept.STRINGS),
    LibraryFunction("strcspn", "size_t strcspn(const char *s, const char *reject);",
                    c_strcspn, Concept.STRINGS),
    LibraryFunction("strtok", "char *strtok(char *str, const char *delim);", c_strtok,
                    Concept.STRINGS),
    LibraryFunction("memset", "void *memset(void *s, int c, size_t n);", c_memset,
                    Concept.POINTERS),
    LibraryFunction("memcpy", "void *memcpy(void *dest, const void *src, size_t n);",
                    c_memcpy, Concept.POINTERS),
    LibraryFunction("memmove", "void *memmove(void *dest, const void *src, size_t n);",
                    c_memmove, Concept.POINTERS),
    LibraryFunction("memcmp", "int memcmp(const void *s1, const void *s2, size_t n);",
                    c_memcmp, Concept.POINTERS),
    LibraryFunction("strerror", "char *strerror(int errnum);", c_strerror_impl,
                    Concept.ERROR_HANDLING),
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<string.h>`` function."""
    register_functions(registry, HEADER, FUNCTIONS)
