"""Tests for runtime values and the trap/warning types."""
from __future__ import annotations

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Phase, Severity, SourceLocation
from pycc.interp.c_types import (
    BOOL,
    CHAR,
    DOUBLE,
    FLOAT,
    INT,
    UCHAR,
    VOID_PTR,
    FunctionType,
    Member,
    PointerType,
    StructType,
)
from pycc.interp.traps import DEFAULT_FRIENDLY, MemoryTrap, PendingWarning, TrapError
from pycc.interp.values import (
    NULL,
    LValue,
    StructValue,
    Value,
    as_python,
    make_double,
    make_float,
    make_function,
    make_int,
    make_pointer,
    truthy,
)

POINT = StructType("Point", [Member("x", INT), Member("y", INT)])


# --- values ------------------------------------------------------------------------------

def test_make_int_wraps_to_width() -> None:
    assert make_int(300, UCHAR).raw == 44
    assert make_int(2**31).raw == -(2**31)
    assert make_int(5).ctype == INT


def test_make_int_bool_normalises() -> None:
    assert make_int(7, BOOL).raw == 1 and make_int(0, BOOL).raw == 0


def test_make_double_and_float() -> None:
    assert make_double(1).raw == 1.0 and make_double(1).ctype == DOUBLE
    single = make_float(0.1, FLOAT)
    assert single.raw != 0.1 and single.raw == pytest.approx(0.1, rel=1e-6)


def test_make_pointer_accepts_target_or_pointer_type() -> None:
    assert make_pointer(4096, INT).ctype == PointerType(INT)
    assert make_pointer(4096, PointerType(CHAR)).ctype == PointerType(CHAR)
    assert make_pointer(-1, INT).raw == (1 << 64) - 1


def test_null_is_a_void_pointer_to_zero() -> None:
    assert NULL.ctype == VOID_PTR and NULL.raw == 0 and NULL.is_null
    assert not make_pointer(4096, INT).is_null


def test_truthy_scalars() -> None:
    assert truthy(make_int(-1)) and not truthy(make_int(0))
    assert truthy(make_double(0.5)) and not truthy(make_double(0.0))
    assert truthy(make_double(float("nan")))        # NaN != 0 in C too
    assert not truthy(NULL)


def test_function_designator() -> None:
    ftype = FunctionType(INT, (INT,))
    designator = make_function(ftype, 3)
    assert designator == Value(ftype, 3)
    assert designator.is_function and truthy(designator)


def test_struct_value_holds_bytes() -> None:
    value = StructValue(POINT, b"\x01\x00\x00\x00\x02\x00\x00\x00")
    assert as_python(value) == value.data
    with pytest.raises(TypeError):
        truthy(value)


def test_struct_value_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        StructValue(POINT, b"\x00")


def test_lvalue_and_as_python() -> None:
    place = LValue(INT, 8192)
    assert place.addr == 8192 and place.ctype == INT
    assert as_python(make_int(9)) == 9


def test_values_are_immutable() -> None:
    with pytest.raises(AttributeError):
        make_int(1).raw = 2  # type: ignore[misc]


# --- traps -------------------------------------------------------------------------------

def test_trap_error_carries_fields() -> None:
    trap = TrapError("division_by_zero", "division by zero", extra={"line": 3})
    assert trap.code == "division_by_zero" and str(trap) == "division by zero"
    assert trap.friendly == DEFAULT_FRIENDLY["division_by_zero"]
    assert trap.concept is Concept.OPERATORS
    assert trap.extra == {"line": 3}


def test_trap_to_diagnostic_is_runtime_error() -> None:
    location = SourceLocation("main.c", 7, 3)
    trap = TrapError("out_of_bounds", "index 8 is past the end", "friendly")
    diagnostic = trap.to_diagnostic(location)
    assert diagnostic.phase is Phase.RUNTIME and diagnostic.severity is Severity.ERROR
    assert diagnostic.location == location and diagnostic.friendly == "friendly"
    assert diagnostic.format_gcc() == "main.c:7:3: error: index 8 is past the end"


def test_explicit_concept_wins_and_unknown_code_defaults() -> None:
    assert TrapError("custom", "m", "f", concept=Concept.ARRAYS).concept is Concept.ARRAYS
    assert TrapError("custom_unknown", "m", "f").concept is Concept.POINTERS


def test_memory_trap_is_a_trap_error() -> None:
    assert issubclass(MemoryTrap, TrapError)
    with pytest.raises(TrapError):
        raise MemoryTrap("wild_pointer", "segfault")


def test_every_default_friendly_text_is_filled() -> None:
    assert all(text.strip() for text in DEFAULT_FRIENDLY.values())


def test_pending_warning_to_diagnostic() -> None:
    warning = PendingWarning("uninitialized_read", "read of x", "x was never set",
                             Concept.UNINITIALIZED, {"object": "x"})
    diagnostic = warning.to_diagnostic(None)
    assert diagnostic.severity is Severity.WARNING and diagnostic.code == "uninitialized_read"
    assert diagnostic.concept is Concept.UNINITIALIZED
