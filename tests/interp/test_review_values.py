"""Review tests for runtime values and traps: bool normalisation, pointer wrapping and
complete diagnostics for every trap code the memory raises."""
from __future__ import annotations

from pycc.core.diagnostics import CODE_CONCEPTS
from pycc.interp.c_types import INT, IntType
from pycc.interp.traps import DEFAULT_FRIENDLY, TrapError
from pycc.interp.values import make_int, make_pointer


def test_make_int_normalises_an_equal_bool_type() -> None:
    other_bool = IntType("_Bool", 1, False)    # equal to BOOL but a different object
    assert make_int(2, other_bool).raw == 1
    assert make_int(256, other_bool).raw == 1


def test_make_pointer_wraps_negative_addresses() -> None:
    assert make_pointer(-1, INT).raw == (1 << 64) - 1


def test_every_default_friendly_code_has_a_concept() -> None:
    assert all(code in CODE_CONCEPTS for code in DEFAULT_FRIENDLY)


def test_trap_without_friendly_uses_default() -> None:
    trap = TrapError("double_free", "free(): double free detected")
    assert trap.friendly == DEFAULT_FRIENDLY["double_free"]
    assert trap.concept == CODE_CONCEPTS["double_free"]
