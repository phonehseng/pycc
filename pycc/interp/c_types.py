"""The C type model: sizes, alignment, layouts and conversions.

We model the LP64 data model that gcc uses on Linux: ``int`` is 4 bytes, ``long`` and
pointers are 8, ``char`` is signed. Every other part of the toolchain (parser, checker,
memory, interpreter, stdlib) asks this module two kinds of questions:

* *shape*: how big is this type, how is it aligned, where is member ``x``?
* *arithmetic*: what type does ``a + b`` have, and what happens to the bits when a
  value is converted?

Keeping those answers in one place means the static checker and the dynamic
interpreter can never disagree about them.
"""
from __future__ import annotations

import builtins
import math
import struct
from dataclasses import dataclass, field
from typing import Protocol


class CType:
    """Base class for all C types. Every subclass provides ``size`` and ``align``
    (as dataclass fields or properties); the base deliberately defines neither, so
    dataclass subclasses do not inherit a default that would break field ordering."""

    size: int
    align: int

    # --- classification helpers (cheap, used everywhere) -------------------------
    @property
    def is_integer(self) -> bool:
        return isinstance(self, (IntType, EnumType))

    @property
    def is_float(self) -> bool:
        return isinstance(self, FloatType)

    @property
    def is_arithmetic(self) -> bool:
        return self.is_integer or self.is_float

    @property
    def is_pointer(self) -> bool:
        return isinstance(self, PointerType)

    @property
    def is_array(self) -> bool:
        return isinstance(self, ArrayType)

    @property
    def is_struct(self) -> bool:
        return isinstance(self, StructType)

    @property
    def is_function(self) -> bool:
        return isinstance(self, FunctionType)

    @property
    def is_void(self) -> bool:
        return isinstance(self, VoidType)

    @property
    def is_scalar(self) -> bool:
        return self.is_arithmetic or self.is_pointer

    @property
    def is_complete(self) -> bool:
        return True

    def __str__(self) -> str:
        return format_type(self)


@dataclass(frozen=True)
class IntType(CType):
    name: str
    size: int
    signed: bool

    @property
    def align(self) -> int:  # type: ignore[override]
        return self.size

    @property
    def bits(self) -> int:
        return self.size * 8

    @property
    def min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def rank(self) -> int:
        """Integer conversion rank: char < short < int < long < long long."""
        return {1: 1, 2: 2, 4: 3, 8: 4 if "long long" not in self.name else 5}[self.size]


@dataclass(frozen=True)
class FloatType(CType):
    name: str
    size: int

    @property
    def align(self) -> int:  # type: ignore[override]
        return self.size


@dataclass(frozen=True)
class VoidType(CType):
    size: int = 0
    align: int = 1

    @property
    def is_complete(self) -> bool:
        return False


@dataclass(frozen=True)
class PointerType(CType):
    target: CType
    size: int = 8
    align: int = 8


@dataclass(frozen=True)
class ArrayType(CType):
    element: CType
    length: int | None = None     # None = incomplete (``int a[]`` in a parameter or extern)

    @property
    def size(self) -> int:  # type: ignore[override]
        return self.element.size * (self.length or 0)

    @property
    def align(self) -> int:  # type: ignore[override]
        return self.element.align

    @property
    def is_complete(self) -> bool:
        return self.length is not None


@dataclass
class Member:
    name: str
    ctype: CType
    offset: int = 0


class StructType(CType):
    """A struct or union. Identity based: each declaration is its own type.

    The type may be *incomplete* (members unknown) after ``struct Node;`` or inside
    its own definition (``struct Node *next;``); ``complete()`` fills it in later.
    """

    def __init__(self, tag: str | None, members: list[Member] | None = None,
                 is_union: bool = False) -> None:
        self.tag = tag
        self.is_union = is_union
        self.members: list[Member] | None = None
        self.size = 0
        self.align = 1
        if members is not None:
            self.complete(members)

    def complete(self, members: list[Member]) -> None:
        """Lay out the members with natural alignment, exactly like gcc."""
        offset = 0
        align = 1
        for member in members:
            malign = member.ctype.align or 1
            align = max(align, malign)
            if self.is_union:
                member.offset = 0
                offset = max(offset, member.ctype.size)
            else:
                offset = _round_up(offset, malign)
                member.offset = offset
                offset += member.ctype.size
        self.members = members
        self.align = align
        self.size = _round_up(offset, align) if members else 0

    @property
    def is_complete(self) -> bool:
        return self.members is not None

    def member(self, name: str) -> Member | None:
        if self.members is None:
            return None
        for member in self.members:
            if member.name == name:
                return member
        return None

    @property
    def keyword(self) -> str:
        return "union" if self.is_union else "struct"

    def __repr__(self) -> str:
        return f"StructType({self.keyword} {self.tag or '<anonymous>'})"


@dataclass(frozen=True)
class EnumType(CType):
    tag: str | None
    constants: tuple[tuple[str, int], ...] = ()
    size: int = 4
    align: int = 4

    @property
    def signed(self) -> bool:
        return True

    @property
    def bits(self) -> int:
        return 32

    @property
    def min(self) -> int:
        return -(1 << 31)

    @property
    def max(self) -> int:
        return (1 << 31) - 1

    @property
    def rank(self) -> int:
        return 3


@dataclass(frozen=True)
class FunctionType(CType):
    ret: CType
    params: tuple[CType, ...] = ()
    variadic: bool = False
    has_prototype: bool = True   # False for ``int f()`` (old style, unspecified params)
    size: int = 1
    align: int = 1


# --- the basic types -------------------------------------------------------------------

BOOL = IntType("_Bool", 1, False)
CHAR = IntType("char", 1, True)
SCHAR = IntType("signed char", 1, True)
UCHAR = IntType("unsigned char", 1, False)
SHORT = IntType("short", 2, True)
USHORT = IntType("unsigned short", 2, False)
INT = IntType("int", 4, True)
UINT = IntType("unsigned int", 4, False)
LONG = IntType("long", 8, True)
ULONG = IntType("unsigned long", 8, False)
LLONG = IntType("long long", 8, True)
ULLONG = IntType("unsigned long long", 8, False)
FLOAT = FloatType("float", 4)
DOUBLE = FloatType("double", 8)
VOID = VoidType()
SIZE_T = ULONG
PTRDIFF_T = LONG
CHAR_PTR = PointerType(CHAR)
VOID_PTR = PointerType(VOID)

INT_TYPES_BY_NAME: dict[str, IntType] = {
    t.name: t for t in (BOOL, CHAR, SCHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG, LLONG, ULLONG)
}


def pointer_to(target: CType) -> PointerType:
    return PointerType(target)


def array_of(element: CType, length: int | None) -> ArrayType:
    return ArrayType(element, length)


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


# --- integer arithmetic -----------------------------------------------------------------

def wrap_int(value: int, ctype: CType) -> int:
    """Reduce a Python int to the two's-complement range of ``ctype``.

    This is what makes ``unsigned char c = 255; c++`` give 0 and
    ``int x = 2147483647; x++`` give -2147483648, exactly as on a real machine.
    """
    bits = ctype.bits  # type: ignore[attr-defined]
    value &= (1 << bits) - 1
    if ctype.signed and value >= (1 << (bits - 1)):  # type: ignore[attr-defined]
        value -= 1 << bits
    return value


def fits(value: int, ctype: CType) -> bool:
    return ctype.min <= value <= ctype.max  # type: ignore[attr-defined]


def integer_promote(ctype: CType) -> CType:
    """Types narrower than int are promoted to int before arithmetic (C99 6.3.1.1)."""
    if isinstance(ctype, EnumType):
        return INT
    if isinstance(ctype, IntType) and ctype.size < 4:
        return INT
    if ctype is BOOL:
        return INT
    return ctype


def usual_arithmetic_conversions(a: CType, b: CType) -> CType:
    """The common type of two arithmetic operands (C99 6.3.1.8)."""
    if a.is_float or b.is_float:
        if DOUBLE in (a, b) or (a.is_float and a.size == 8) or (b.is_float and b.size == 8):
            return DOUBLE
        return FLOAT
    a = integer_promote(a)
    b = integer_promote(b)
    if a == b:
        return a
    assert isinstance(a, IntType) and isinstance(b, IntType)
    if a.signed == b.signed:
        return a if a.rank >= b.rank else b
    unsigned, signed = (a, b) if not a.signed else (b, a)
    if unsigned.rank >= signed.rank:
        return unsigned
    if signed.size > unsigned.size:
        return signed
    return IntType(f"unsigned {signed.name}" if not signed.name.startswith("unsigned") else signed.name,
                   signed.size, False)


def convert_value(value: int | float, src: CType, dst: CType) -> int | float:
    """Convert a raw value between scalar types with C semantics.

    int -> int: two's-complement truncation; float -> int: toward zero (out-of-range
    wraps, which is undefined in C but what we promise the user); int -> float:
    nearest representable; pointers are integers.
    """
    value = _cross_function_pointer(value, src, dst)
    if dst.is_float:
        result = float(value)
        if dst.size == 4:
            # Round through IEEE single precision so float arithmetic is honest.
            result = struct.unpack("<f", struct.pack("<f", _clamp_float32(result)))[0]
        return result
    if dst.is_pointer:
        return int(value) & ((1 << 64) - 1) if isinstance(value, int) else int(value)
    if dst is BOOL:
        return 1 if value else 0
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return dst.min if dst.signed else 0  # type: ignore[attr-defined]
        value = int(value)  # truncates toward zero
    return wrap_int(int(value), dst)


def _is_function_pointer(ctype: CType) -> bool:
    """True for a function designator or a pointer to a function."""
    return isinstance(ctype, FunctionType) or (
        isinstance(ctype, PointerType) and isinstance(ctype.target, FunctionType))


def _cross_function_pointer(value: int | float, src: CType, dst: CType) -> int | float:
    """Translate between a function pointer's id and the address it shows.

    Inside the interpreter a function pointer holds a small function id, but the user
    sees it as ``0x400000 + id*16``. ``(void *)fp`` or ``(long)fp`` must keep showing
    that address (printf("%p", (void *)main) is the portable way to print one), and
    casting the address back must find the function again.
    """
    leaving = _is_function_pointer(src) and not _is_function_pointer(dst)
    if leaving and isinstance(value, int):
        return function_address(value)
    entering = _is_function_pointer(dst) and not _is_function_pointer(src)
    if entering and isinstance(value, int) and value >= FUNCTION_ADDRESS_BASE:
        offset = value - FUNCTION_ADDRESS_BASE
        if offset % FUNCTION_ADDRESS_STEP == 0:
            return offset // FUNCTION_ADDRESS_STEP
    return value


def _clamp_float32(value: float) -> float:
    limit = 3.4028234663852886e38
    if value > limit:
        return math.inf
    if value < -limit:
        return -math.inf
    return value


def decay(ctype: CType) -> CType:
    """Arrays become pointers to their first element; functions become function pointers."""
    if isinstance(ctype, ArrayType):
        return PointerType(ctype.element)
    if isinstance(ctype, FunctionType):
        return PointerType(ctype)
    return ctype


def strip_qualifiers(ctype: CType) -> CType:
    # Qualifiers (const, volatile) are not modelled as separate types; parsers record
    # them on declarations. This hook keeps call sites readable.
    return ctype


def is_compatible(a: CType, b: CType) -> bool:
    """Loose compatibility used for prototype/definition and pointer checks."""
    if a is b or a == b:
        return True
    if isinstance(a, EnumType) and b == INT or isinstance(b, EnumType) and a == INT:
        return True
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        if a.target.is_void or b.target.is_void:
            return True
        return is_compatible(a.target, b.target)
    if isinstance(a, ArrayType) and isinstance(b, ArrayType):
        return is_compatible(a.element, b.element) and (a.length is None or b.length is None or a.length == b.length)
    if isinstance(a, ArrayType) and isinstance(b, PointerType):
        return is_compatible(a.element, b.target)
    if isinstance(a, PointerType) and isinstance(b, ArrayType):
        return is_compatible(a.target, b.element)
    if isinstance(a, FunctionType) and isinstance(b, FunctionType):
        if not is_compatible(a.ret, b.ret):
            return False
        if not a.has_prototype or not b.has_prototype:
            return True
        if len(a.params) != len(b.params) or a.variadic != b.variadic:
            return False
        return all(is_compatible(decay(x), decay(y)) for x, y in zip(a.params, b.params))
    if isinstance(a, StructType) and isinstance(b, StructType):
        return a is b or (a.tag is not None and a.tag == b.tag and a.is_union == b.is_union)
    return False


# --- encoding to and from memory bytes ---------------------------------------------------

_INT_FORMATS = {(1, True): "<b", (1, False): "<B", (2, True): "<h", (2, False): "<H",
                (4, True): "<i", (4, False): "<I", (8, True): "<q", (8, False): "<Q"}


def encode(value: int | float, ctype: CType) -> bytes:
    """Little-endian bytes for a scalar value of ``ctype`` (pointers are 8-byte ints)."""
    if isinstance(ctype, FloatType):
        return struct.pack("<f" if ctype.size == 4 else "<d", float(value))
    if isinstance(ctype, PointerType):
        return struct.pack("<Q", int(value) & ((1 << 64) - 1))
    signed = getattr(ctype, "signed", True)
    return struct.pack(_INT_FORMATS[(ctype.size, signed)], wrap_int(int(value), ctype))


def decode(data: bytes, ctype: CType) -> int | float:
    if isinstance(ctype, FloatType):
        return struct.unpack("<f" if ctype.size == 4 else "<d", data[:ctype.size])[0]
    if isinstance(ctype, PointerType):
        return struct.unpack("<Q", data[:8])[0]
    signed = getattr(ctype, "signed", True)
    return struct.unpack(_INT_FORMATS[(ctype.size, signed)], data[:ctype.size])[0]


# --- printing ------------------------------------------------------------------------------

def format_type(ctype: CType) -> str:
    """Render a type the way gcc prints it in diagnostics: ``int *``, ``struct Tile[8]``."""
    if isinstance(ctype, (IntType, FloatType)):
        return ctype.name
    if isinstance(ctype, VoidType):
        return "void"
    if isinstance(ctype, EnumType):
        return f"enum {ctype.tag}" if ctype.tag else "enum <anonymous>"
    if isinstance(ctype, StructType):
        return f"{ctype.keyword} {ctype.tag}" if ctype.tag else f"{ctype.keyword} <anonymous>"
    if isinstance(ctype, PointerType):
        inner = ctype.target
        if isinstance(inner, FunctionType):
            params = ", ".join(format_type(p) for p in inner.params) or ("" if inner.has_prototype else "")
            if inner.has_prototype and not inner.params:
                params = "void"
            if inner.variadic:
                params += ", ..."
            return f"{format_type(inner.ret)} (*)({params})"
        if isinstance(inner, ArrayType):
            return f"{format_type(inner.element)} (*)[{inner.length if inner.length is not None else ''}]"
        base = format_type(inner)
        return base + ("*" if base.endswith("*") else " *")
    if isinstance(ctype, ArrayType):
        length = "" if ctype.length is None else str(ctype.length)
        return f"{format_type(ctype.element)}[{length}]"
    if isinstance(ctype, FunctionType):
        params = ", ".join(format_type(p) for p in ctype.params)
        if ctype.has_prototype and not ctype.params:
            params = "void"
        if ctype.variadic:
            params += ", ..."
        return f"{format_type(ctype.ret)}({params})"
    return "<type>"


# --- values for the debugger ----------------------------------------------

# How far a function pointer's address is from 0: D3 makes function id N show as
# 0x400000 + N*16, which looks like code addresses on a real Linux machine. printf's
# %p and the variables panel both use it, so the two always show the same number.
FUNCTION_ADDRESS_BASE = 0x400000
FUNCTION_ADDRESS_STEP = 16


def function_address(function_id: int) -> int:
    """The address a function pointer holding ``function_id`` shows (0 stays NULL)."""
    return FUNCTION_ADDRESS_BASE + function_id * FUNCTION_ADDRESS_STEP if function_id else 0


class ValueMemory(Protocol):
    """What ``format_value`` reads: ``interp.memory.Memory`` or a snapshot-like object."""

    # builtins.bytes, because inside this class the name 'bytes' is the method itself.
    def bytes(self, start: int, n: int) -> builtins.bytes:
        """Raw bytes, unchecked (garbage included)."""
        ...

    def init_mask(self, start: int, n: int) -> builtins.bytes:
        """One byte per address: 1 if the program wrote it, 0 if it is garbage."""
        ...


_CHAR_NAMES = frozenset({"char", "signed char", "unsigned char"})
_CHAR_ESCAPES = {0: "\\0", 7: "\\a", 8: "\\b", 9: "\\t", 10: "\\n", 13: "\\r", 39: "\\'",
                 92: "\\\\"}


def format_value(memory: ValueMemory, addr: int, ctype: CType, max_items: int = 16) -> str:
    """The value at ``addr`` as the variables panel shows it.

    Pointers are ``0x..`` (``NULL`` for zero), chars ``'c'``, a char array holding a
    finished string ``"text"``, other arrays and structs ``{..}`` with at most
    ``max_items`` entries, and any value with an uninitialised byte ``?``: a beginner
    must see that the variable holds garbage, not a plausible-looking number.
    """
    if isinstance(ctype, ArrayType):
        return _format_array(memory, addr, ctype, max_items)
    if isinstance(ctype, StructType):
        return _format_struct(memory, addr, ctype, max_items)
    return _format_scalar(memory, addr, ctype)


def _is_initialised(memory: ValueMemory, addr: int, size: int) -> bool:
    """True when every byte of ``[addr, addr + size)`` was written by the program."""
    mask = memory.init_mask(addr, size)
    return len(mask) == size and all(mask)


def _format_scalar(memory: ValueMemory, addr: int, ctype: CType) -> str:
    """One number, character, pointer or enum value."""
    if isinstance(ctype, (VoidType, FunctionType)) or ctype.size <= 0:
        return "?"
    if not _is_initialised(memory, addr, ctype.size):
        return "?"
    value = decode(memory.bytes(addr, ctype.size), ctype)
    if isinstance(ctype, PointerType):
        if isinstance(ctype.target, FunctionType):
            value = function_address(int(value))   # the number printf("%p") shows too
        return "NULL" if value == 0 else f"0x{int(value):x}"
    if isinstance(ctype, FloatType):
        return f"{value:g}"
    if isinstance(ctype, EnumType):
        names = [name for name, number in ctype.constants if number == value]
        return f"{names[0]} ({value})" if names else str(value)
    if isinstance(ctype, IntType) and ctype.name in _CHAR_NAMES:
        return _char_literal(int(value))
    return str(value)


def _char_literal(code: int) -> str:
    """``'a'``, ``'\\n'`` or ``'\\x07'``: a char the way C source would write it."""
    code &= 0xFF
    if code in _CHAR_ESCAPES:
        return f"'{_CHAR_ESCAPES[code]}'"
    return f"'{chr(code)}'" if 32 <= code < 127 else f"'\\x{code:02x}'"


def _format_array(memory: ValueMemory, addr: int, ctype: ArrayType, max_items: int) -> str:
    """A string literal for a terminated char array, otherwise ``{a, b, ...}``."""
    count = ctype.length or 0
    element = ctype.element
    if isinstance(element, IntType) and element.name in _CHAR_NAMES:
        text = _string_literal(memory, addr, count)
        if text is not None:
            return text
    items = [format_value(memory, addr + index * element.size, element, max_items)
             for index in range(min(count, max_items))]
    more = ", ..." if count > max_items else ""
    return "{" + ", ".join(items) + more + "}"


def _string_literal(memory: ValueMemory, addr: int, length: int) -> str | None:
    """``"text"`` when the array holds a written, printable, terminated string."""
    data = memory.bytes(addr, length)
    end = data.find(b"\0")
    if end < 0 or not _is_initialised(memory, addr, end + 1):
        return None          # no '\0' yet: show the characters one by one instead
    text = data[:end].decode("latin-1")
    if not all(32 <= ord(ch) < 127 for ch in text):
        return None
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_struct(memory: ValueMemory, addr: int, ctype: StructType, max_items: int) -> str:
    """``{x = 1, y = 2}``; ``{...}`` for a struct whose members are not known yet."""
    if ctype.members is None:
        return "{...}"
    parts = [f"{member.name} = {format_value(memory, addr + member.offset, member.ctype, max_items)}"
             for member in ctype.members[:max_items]]
    more = ", ..." if len(ctype.members) > max_items else ""
    return "{" + ", ".join(parts) + more + "}"
