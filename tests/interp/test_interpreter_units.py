"""Unit tests of the interpreter's pure helpers (arithmetic, frames, switch tables)."""
from __future__ import annotations

import math

import pytest

from pycc.core.diagnostics import SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.c_types import CHAR, INT, VOID, ArrayType, FunctionType, PointerType, StructType
from pycc.interp.call import (
    BREAK, CONTINUE, CallFrame, Place, Returned, Slot, build_switch_table, index_of, is_lvalue,
    strip_implicit,
)
from pycc.interp.interpreter import c_mod, exact_int_result, float_result, pointer_step, trunc_div
from pycc.interp.memory import Frame
from pycc.interp.symbols import SymbolRef
from pycc.interp.values import Value

LOC = SourceLocation("main.c", 1, 1)


# --- arithmetic ----------------------------------------------------------------------------

@pytest.mark.parametrize(("a", "b", "quotient", "remainder"), [
    (7, 2, 3, 1), (-7, 2, -3, -1), (7, -2, -3, 1), (-7, -2, 3, -1), (0, 5, 0, 0),
])
def test_division_truncates_toward_zero(a: int, b: int, quotient: int, remainder: int) -> None:
    assert trunc_div(a, b) == quotient
    assert c_mod(a, b) == remainder
    assert trunc_div(a, b) * b + c_mod(a, b) == a


@pytest.mark.parametrize(("op", "expected"), [
    ("+", 13), ("-", 7), ("*", 30), ("/", 3), ("%", 1), ("&", 2), ("|", 11), ("^", 9),
])
def test_exact_int_results(op: str, expected: int) -> None:
    assert exact_int_result(op, 10, 3) == expected


@pytest.mark.parametrize("op", ["/", "%"])
def test_integer_division_by_zero_raises(op: str) -> None:
    with pytest.raises(ZeroDivisionError):
        exact_int_result(op, 1, 0)


def test_unknown_operator_is_a_bug() -> None:
    with pytest.raises(ValueError):
        exact_int_result("<<", 1, 1)
    with pytest.raises(ValueError):
        float_result("%", 1.0, 1.0)


def test_float_division_by_zero_is_ieee() -> None:
    assert float_result("/", 1.0, 0.0) == math.inf
    assert float_result("/", -1.0, 0.0) == -math.inf
    assert float_result("/", 1.0, -0.0) == -math.inf
    assert math.isnan(float_result("/", 0.0, 0.0))
    assert float_result("*", 1.5, 2.0) == 3.0


def test_pointer_step_is_the_target_size() -> None:
    assert pointer_step(PointerType(INT)) == 4
    assert pointer_step(PointerType(CHAR)) == 1
    assert pointer_step(PointerType(VOID)) == 1          # gcc's void * arithmetic
    assert pointer_step(PointerType(ArrayType(INT, 3))) == 12
    assert pointer_step(PointerType(FunctionType(INT))) == 1


# --- frames and control flow ------------------------------------------------------------------

def make_frame() -> CallFrame:
    function = ast.FunctionDef(LOC, "main", FunctionType(INT), [], ast.CompoundStmt(LOC))
    return CallFrame(function, Frame(1, "main", LOC, 0), None)


def test_frame_lookup_finds_the_innermost_name() -> None:
    frame = make_frame()
    frame.bind(Slot("x", INT, 100))
    frame.scopes.append({})
    frame.bind(Slot("x", INT, 200))
    assert frame.lookup("x").addr == 200
    frame.scopes.pop()
    assert frame.lookup("x").addr == 100
    assert frame.lookup("y") is None


def test_frame_info_lists_visible_slots() -> None:
    frame = make_frame()
    frame.bind(Slot("a", INT, 8))
    info = frame.info()
    assert (info.id, info.name, [v.name for v in info.locals]) == (1, "main", ["a"])


def test_flow_signals() -> None:
    assert BREAK is not CONTINUE
    assert repr(BREAK) == "BREAK"
    assert Returned(Value(INT, 3)).value == Value(INT, 3)
    assert Place(INT, 4096).origin is None


# --- AST helpers --------------------------------------------------------------------------------

def case(value: int, body: ast.Stmt) -> ast.CaseLabel:
    return ast.CaseLabel(LOC, ast.IntLiteral(LOC, value, INT), value, body)


def test_switch_table_finds_nested_and_stacked_labels() -> None:
    target = ast.ExprStmt(LOC, ast.IntLiteral(LOC, 0, INT))
    inner = case(2, target)
    outer = case(1, inner)
    block = ast.CompoundStmt(LOC, [outer, ast.CompoundStmt(LOC, [ast.DefaultLabel(LOC, target)])])
    table = build_switch_table(ast.SwitchStmt(LOC, ast.IntLiteral(LOC, 0, INT), block))
    assert table.path_for(1) == [block, outer]
    assert table.path_for(2) == [block, outer, inner]
    assert table.path_for(99)[-1].__class__ is ast.DefaultLabel


def test_switch_table_skips_nested_switches() -> None:
    nested = ast.SwitchStmt(LOC, ast.IntLiteral(LOC, 0, INT), case(5, ast.EmptyStmt(LOC)))
    body = ast.CompoundStmt(LOC, [nested])
    table = build_switch_table(ast.SwitchStmt(LOC, ast.IntLiteral(LOC, 0, INT), body))
    assert table.path_for(5) is None


def test_index_of_uses_identity() -> None:
    first, second = ast.EmptyStmt(LOC), ast.EmptyStmt(LOC)
    assert first == second                       # dataclass equality would be ambiguous
    assert index_of([first, second], second) == 1
    with pytest.raises(ValueError):
        index_of([first], second)


def test_strip_implicit_and_lvalues() -> None:
    name = ast.Identifier(LOC, "x")
    name.ref = SymbolRef("local", "x", INT)  # type: ignore[attr-defined]
    wrapped = ast.ImplicitCast(LOC, name, INT)
    assert strip_implicit(wrapped) is name
    assert is_lvalue(name) and not is_lvalue(wrapped)
    function = ast.Identifier(LOC, "f")
    function.ref = SymbolRef("function", "f", FunctionType(INT))  # type: ignore[attr-defined]
    assert not is_lvalue(function)
    struct = StructType("P", [])
    member = ast.Member(LOC, ast.Identifier(LOC, "p"), "x", arrow=True)
    assert is_lvalue(member) and struct.size == 0
    assert is_lvalue(ast.UnaryOp(LOC, "*", name)) and not is_lvalue(ast.UnaryOp(LOC, "-", name))
