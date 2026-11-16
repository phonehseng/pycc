"""Review tests for constant folding: enum constants and sizeof of variables."""
from __future__ import annotations

from pycc.interp import astnodes as ast
from pycc.interp.c_types import ArrayType, INT
from pycc.interp.constexpr import DictScope, eval_const_expr
from pycc.interp.parser import ParseResult, parse_text


def array_length(source: str, name: str) -> int | None:
    """Parse ``source`` and return the length of array variable ``name``."""
    result = parse_text(source)
    assert result.ok, [d.message for d in result.diagnostics]
    return _find_var(result, name).ctype.length


def _find_var(result: ParseResult, name: str) -> ast.VarDecl:
    for node in ast.walk(result.ast):
        if isinstance(node, ast.VarDecl) and node.name == name:
            return node
    raise AssertionError(f"no variable {name}")


def test_sizeof_array_over_sizeof_element() -> None:
    assert array_length("int a[10]; int b[sizeof a / sizeof a[0]];", "b") == 10


def test_sizeof_parenthesized_variable() -> None:
    assert array_length("double d[3]; char c[sizeof(d)];", "c") == 24


def test_sizeof_array_completed_by_initializer() -> None:
    assert array_length("int a[] = {1, 2, 3}; int c[sizeof a];", "c") == 12


def test_sizeof_string_variable_counts_nul() -> None:
    assert array_length('char s[] = "abc"; int z[sizeof s];', "z") == 4


def test_sizeof_struct_members_through_dot_and_arrow() -> None:
    source = ("struct P { int x; double d; }; struct P ps[4]; struct P *pp;"
              "int n[sizeof ps / sizeof ps[0]]; int m[sizeof ps[1].d + sizeof pp->x];")
    assert array_length(source, "n") == 4
    assert array_length(source, "m") == 12


def test_array_parameter_is_a_pointer_for_sizeof() -> None:
    source = "void f(int v[10]) { char c[sizeof v]; }"
    assert array_length(source, "c") == 8


def test_enum_constants_and_sizeof_mix() -> None:
    source = "enum { A = 3, B = A * 2 }; enum E { X = sizeof(long) * B }; int z[X + A];"
    assert array_length(source, "z") == 51


def test_variable_shadowing_enum_is_not_constant() -> None:
    result = parse_text("enum { N = 3 }; void f(void) { int N = 5; int a[N]; }")
    assert [d.code for d in result.errors] == ["array_size_not_const"]


def test_dict_scope_can_supply_variable_types() -> None:
    expr = ast.SizeofExpr(None, ast.Identifier(None, "grid"))  # type: ignore[arg-type]
    scope = DictScope(types={"grid": ArrayType(INT, 5)})
    assert eval_const_expr(expr, scope) == 20
    assert eval_const_expr(expr, DictScope()) is None


def test_sizeof_unknown_name_is_not_constant() -> None:
    expr = ast.SizeofExpr(None, ast.Identifier(None, "x"))  # type: ignore[arg-type]
    assert eval_const_expr(expr) is None
