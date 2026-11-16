"""Tests for interp.constexpr: folding integer constant expressions with C semantics."""
from __future__ import annotations

import pytest

from pycc.core.diagnostics import SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.c_types import INT
from pycc.interp.constexpr import DictScope, eval_const_expr
from pycc.interp.parser import parse_text


def fold(text: str, constants: dict[str, int] | None = None) -> int | None:
    """Parse ``text`` as an initializer expression and fold it."""
    result = parse_text(f"int probe = {text};")
    assert result.ok, [d.format_gcc() for d in result.diagnostics]
    init = result.ast.decls[-1].init
    return eval_const_expr(init, DictScope(constants))


@pytest.mark.parametrize("text, value", [
    ("42", 42), ("'A'", 65), ("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("10 - 2 - 3", 5),
    ("7 / 2", 3), ("-7 / 2", -3), ("7 % 3", 1), ("-7 % 3", -1), ("7 % -3", 1),
    ("1 << 4", 16), ("256 >> 2", 64), ("6 & 3", 2), ("6 | 3", 7), ("6 ^ 3", 5),
    ("~0", -1), ("-5", -5), ("+5", 5), ("!0", 1), ("!7", 0),
    ("3 < 4", 1), ("3 > 4", 0), ("3 <= 3", 1), ("3 >= 4", 0), ("2 == 2", 1), ("2 != 2", 0),
    ("1 && 2", 1), ("1 && 0", 0), ("0 || 3", 1), ("0 || 0", 0),
    ("1 ? 10 : 20", 10), ("0 ? 10 : 20", 20),
])
def test_basic_operators(text: str, value: int) -> None:
    assert fold(text) == value


def test_sizeof_types() -> None:
    assert fold("sizeof(int)") == 4
    assert fold("sizeof(char)") == 1
    assert fold("sizeof(long)") == 8
    assert fold("sizeof(double) * 2") == 16
    assert fold("sizeof(int *)") == 8
    assert fold("sizeof(int[10])") == 40


def test_sizeof_of_struct_uses_layout() -> None:
    result = parse_text("struct T { char c; int i; }; unsigned long n = sizeof(struct T);")
    assert eval_const_expr(result.ast.decls[-1].init) == 8


def test_sizeof_of_literals() -> None:
    assert fold('sizeof "abc"') == 4  # three letters plus the NUL
    assert fold("sizeof 'a'") == 4    # a character constant is an int
    assert fold("sizeof 1.5") == 8
    assert fold("sizeof 1L") == 8


def test_unsigned_arithmetic_follows_c() -> None:
    assert fold("-1u > 0") == 1             # -1u is 4294967295
    assert fold("-1 < 0u") == 0             # -1 converts to unsigned first
    assert fold("0u - 1") == 4294967295
    assert fold("2147483647 + 1") == -2147483648  # int wraps (as our machines do)


def test_casts() -> None:
    assert fold("(unsigned char)300") == 44
    assert fold("(char)200") == -56
    assert fold("(int)2.9") == 2
    assert fold("(int)-2.9") == -2
    assert fold("(_Bool)5") == 1
    assert fold("(long)1 << 40") == 1 << 40


def test_enum_constants_through_scope() -> None:
    assert fold("ROWS * COLS", {"ROWS": 4, "COLS": 5}) == 20


def test_enum_constants_from_parser_scope() -> None:
    result = parse_text("enum { A = 2, B = A * 3, C }; int grid[C];")
    assert result.ast.decls[0].constants == [("A", 2), ("B", 6), ("C", 7)]
    assert result.ast.decls[-1].ctype.length == 7


def test_short_circuit_allows_non_constant_other_side() -> None:
    assert fold("0 && x") == 0
    assert fold("1 || x") == 1
    assert fold("1 && x") is None


@pytest.mark.parametrize("text", [
    "x", "x + 1", "1 / 0", "5 % 0", "f()", "1.5", "1 << 40", "1 << -1", "&x", "y = 2",
    "1 ? x : 2", "(1, 2)",
])
def test_not_constant(text: str) -> None:
    source = f"int x, y; int f(void); int probe = {text};"
    result = parse_text(source)
    init = result.ast.decls[-1].init
    assert eval_const_expr(init, DictScope()) is None


def test_unknown_identifier_without_scope() -> None:
    node = ast.Identifier(SourceLocation("t.c", 1, 1), "N")
    assert eval_const_expr(node) is None
    assert eval_const_expr(node, DictScope({"N": 3})) == 3


def test_handwritten_tree() -> None:
    loc = SourceLocation("t.c", 1, 1)
    tree = ast.BinaryOp(loc, "*", ast.IntLiteral(loc, 6, INT), ast.IntLiteral(loc, 7, INT))
    assert eval_const_expr(tree) == 42


def test_sizeof_incomplete_struct_is_not_constant() -> None:
    result = parse_text("struct Later; unsigned long n = sizeof(struct Later);")
    assert eval_const_expr(result.ast.decls[-1].init) is None
