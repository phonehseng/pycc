"""Runtime values: what an evaluated C expression produces.

The interpreter needs three kinds of results:

* ``Value(ctype, raw)``: an rvalue of scalar type. ``raw`` is a Python ``int`` for
  integers, enums and pointers (a pointer is just an address) and a ``float`` for
  floating types. A function designator is ``Value(FunctionType, fid)`` where ``fid``
  indexes the program's function table, which is what makes function pointers and
  ``qsort`` callbacks work.
* ``LValue(ctype, addr)``: a place in memory that can be read or assigned (``x``,
  ``a[i]``, ``*p``, ``s.field``). Arrays stay lvalues until they decay.
* ``StructValue(ctype, data)``: a whole struct copied by value (assignment, passing,
  returning). The bytes are a snapshot, so later changes to the source do not leak in.

Values are immutable: arithmetic always makes a new value, just like the machine.
"""
from __future__ import annotations

from dataclasses import dataclass

from pycc.interp.c_types import (
    BOOL,
    DOUBLE,
    INT,
    VOID_PTR,
    CType,
    FunctionType,
    PointerType,
    StructType,
    convert_value,
    wrap_int,
)


@dataclass(frozen=True)
class Value:
    """A scalar rvalue: an integer, a float, a pointer or a function designator."""

    ctype: CType
    raw: int | float

    @property
    def is_function(self) -> bool:
        """True for a function designator (``raw`` is a function table index)."""
        return isinstance(self.ctype, FunctionType)

    @property
    def is_null(self) -> bool:
        """True for a pointer whose address is 0."""
        return self.ctype.is_pointer and self.raw == 0


@dataclass(frozen=True)
class LValue:
    """A designated object in memory: something with an address."""

    ctype: CType
    addr: int


@dataclass(frozen=True)
class StructValue:
    """A struct or union held by value as its raw bytes (padding included)."""

    ctype: StructType
    data: bytes

    def __post_init__(self) -> None:
        # A wrong-sized byte string would silently corrupt the destination on store.
        if len(self.data) != self.ctype.size:
            raise ValueError(
                f"StructValue for {self.ctype} needs {self.ctype.size} bytes, got {len(self.data)}")


AnyValue = Value | StructValue

# NULL is a void pointer to address 0; it converts to every other pointer type.
NULL = Value(VOID_PTR, 0)


def make_int(raw: int, ctype: CType = INT) -> Value:
    """Make an integer value, wrapping ``raw`` to the width of ``ctype``."""
    # Compare by value, not identity: an equal _Bool type built elsewhere must still
    # normalise to 0/1 instead of wrapping (2 would wrap to 0, which is wrong).
    if ctype == BOOL:
        return Value(BOOL, 1 if raw else 0)
    return Value(ctype, wrap_int(int(raw), ctype))


def make_double(raw: float) -> Value:
    """Make a ``double`` value."""
    return Value(DOUBLE, float(raw))


def make_float(raw: float, ctype: CType) -> Value:
    """Make a floating value of ``ctype``, rounding through ``float`` precision if needed."""
    return Value(ctype, convert_value(float(raw), DOUBLE, ctype))


def make_pointer(addr: int, ctype: CType) -> Value:
    """Make a pointer value. ``ctype`` may be the pointer type or the pointed-to type."""
    ptype = ctype if isinstance(ctype, PointerType) else PointerType(ctype)
    return Value(ptype, int(addr) & ((1 << 64) - 1))


def make_function(ftype: FunctionType, fid: int) -> Value:
    """Make a function designator for entry ``fid`` of the function table."""
    return Value(ftype, fid)


def truthy(value: AnyValue) -> bool:
    """C truth: any non-zero scalar (including a non-NULL pointer) is true."""
    if isinstance(value, StructValue):
        raise TypeError(f"{value.ctype} used where a scalar is required")
    if value.is_function:
        # A function designator decays to a non-NULL pointer, which is always true.
        return True
    return value.raw != 0


def as_python(value: AnyValue) -> int | float | bytes:
    """Unwrap to plain Python: the raw number, or the struct's bytes."""
    if isinstance(value, StructValue):
        return value.data
    return value.raw
