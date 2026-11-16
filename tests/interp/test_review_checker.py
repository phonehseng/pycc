"""Review tests for the semantic checker: annotations, implicit casts, refs and flow.

These pin down the AST contract on tricky programs (every
expression typed, every identifier resolved, conversions explicit) and the bugs the
review found: 'return x; break;' in a switch, reads in loops that assign later, sizeof
of an out-of-range element, and non-constant initializers of globals and statics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pycc.interp import astnodes as ast
from pycc.interp.c_types import (
    DOUBLE, INT, LONG, PTRDIFF_T, SIZE_T, ArrayType, EnumType, PointerType, format_type,
)
from pycc.interp.checker import can_fall_through
from tests.interp.test_checker import Build, body, build, nodes

STDDEF_H = (Path(__file__).resolve().parents[2] / "pycc" / "interp" / "stdlib" / "headers"
            / "stddef.h").read_text(encoding="utf-8")


def unannotated(built: Build) -> list[str]:
    """Every expression without a ctype and every identifier without a ref."""
    missing = []
    for node in ast.walk(built.unit):
        if isinstance(node, ast.Expr) and getattr(node, "ctype", None) is None:
            missing.append(ast.describe(node))
        if isinstance(node, ast.Identifier) and getattr(node, "ref", None) is None:
            missing.append(f"ref of {node.name}")
    return missing


def first(built: Build, kind: type, fn: str = "main", **fields: object) -> ast.Node:
    """The first node of ``kind`` in function ``fn`` whose fields match."""
    for node in nodes(built.function(fn), kind):
        if all(getattr(node, key) == value for key, value in fields.items()):
            return node
    raise LookupError(kind)


# ============================================================ annotation completeness

ANNOTATION_PROGRAMS = [
    "struct P { int x; double y; };\nstruct P g = {1, 2};\n"
    "int main(void) { struct P p = {3, 4.5f}; struct P *q = &p; q->x = p.x + (int)q->y;\n"
    "struct P arr[2] = {{1, 2}, {3, 4}}; return arr[1].x; }\n",
    "int add(int a, int b) { return a + b; }\n"
    "int main(void) { int (*fp)(int, int) = add; return fp(1, 2) + (*fp)(3, 4); }\n",
    "enum { N = 3 };\nint t[N] = {N, N + 1};\n"
    "int main(void) { switch (t[0]) { case N: return 1; case N + 1: return 2; } return 0; }\n",
    "int main(void) { int m[2][3] = {1, 2, 3, 4, 5, 6}; int (*row)[3] = m;\n"
    "return row[1][2] + m[0][1] + (int)sizeof m[0] + (m[1] - m[0]); }\n",
    "int main(void) { int x = 1; x++ = 3; &5; return (x, y, z) ? q : r; }\n",
    "struct P { int x; };\nint main(void) { int n = 3; return n.x + n->y + nope.z + n(1, k); }\n",
]


@pytest.mark.parametrize("source", ANNOTATION_PROGRAMS)
def test_review_every_expression_and_identifier_is_annotated(source: str) -> None:
    assert unannotated(build(source)) == []


# ============================================================ implicit casts and types

def test_review_compound_assignment_converts_the_value_to_the_common_type() -> None:
    built = build("int main(void) { int a = 1; long l = 2; a += 1.5; l += a; return a; }")
    plus = [n for n in nodes(built.function("main"), ast.Assignment) if n.op == "+="]
    assert plus[0].ctype == INT and plus[0].value.ctype == DOUBLE   # 1.5 is double already
    assert isinstance(plus[1].value, ast.ImplicitCast) and plus[1].value.to == LONG
    assert plus[1].ctype == LONG


def test_review_ternary_with_int_and_double_casts_the_int_branch() -> None:
    built = build(body("int x = 1; double r = x ? 1 : 2.0; return (int)r;"))
    cond = first(built, ast.Conditional)
    assert cond.ctype == DOUBLE
    assert isinstance(cond.then, ast.ImplicitCast) and cond.then.to == DOUBLE
    assert not isinstance(cond.otherwise, ast.ImplicitCast)


def test_review_comparisons_and_logic_yield_int() -> None:
    built = build(body("double d = 2; int *p = 0; return (d < 3) + (p == 0) + (d && p) + !d;"))
    for node in nodes(built.function("main"), ast.BinaryOp):
        if node.op in ("<", "==", "&&"):
            assert node.ctype == INT, node.op
    assert first(built, ast.UnaryOp, op="!").ctype == INT


def test_review_pointer_difference_is_ptrdiff_t_and_sizeof_is_size_t() -> None:
    built = build(body("int a[4]; int *p = a; int *q = a + 3; "
                       "return (int)(q - p) + (int)sizeof(int) + (int)sizeof a;"))
    minus = first(built, ast.BinaryOp, op="-")
    assert minus.ctype == PTRDIFF_T and format_type(minus.ctype) == "long"
    assert ast.expression_type(first(built, ast.SizeofType)) == SIZE_T
    assert first(built, ast.SizeofExpr).ctype == SIZE_T
    assert format_type(SIZE_T) == "unsigned long"


def test_review_string_literal_is_a_char_array_counting_the_nul() -> None:
    built = build(body("const char *s = \"hey\"; return s[0];"))
    literal = first(built, ast.StringLiteral)
    assert literal.ctype == ArrayType(literal.ctype.element, 4)
    init = nodes(built.function("main"), ast.VarDecl)[0].init
    assert isinstance(init, ast.ImplicitCast) and isinstance(init.to, PointerType)


def test_review_null_from_stddef_is_a_null_pointer_constant() -> None:
    built = build("#include \"stddef.h\"\nint main(void) { int *p = NULL; char *s = 0;\n"
                  "if (p == NULL || s != NULL) return 1; return 0; }\n",
                  files={"stddef.h": STDDEF_H})
    assert built.diagnostics == []
    assert unannotated(built) == []


def test_review_enum_arithmetic_never_casts_between_enum_and_int() -> None:
    built = build("enum Color { RED, GREEN, BLUE };\n"
                  "int main(void) { enum Color c = GREEN; c = c + 1; int x = BLUE * 2 - c;\n"
                  "return x + -c; }\n")
    assert built.diagnostics == []
    casts = nodes(built.function("main"), ast.ImplicitCast)
    assert all(not isinstance(cast.to, EnumType) and not isinstance(_type(cast.expr), EnumType)
               for cast in casts)
    green = first(built, ast.Identifier, name="GREEN")
    assert (green.ref.kind, green.ref.key, green.ref.value) == ("enum_const", "1", 1)


def _type(expr: ast.Expr) -> object:
    return ast.expression_type(expr)


def test_review_unprototyped_call_promotes_char_and_float() -> None:
    built = build("int g();\nint main(void) { char c = 'a'; float f = 1; return g(c, f); }\n"
                  "int g(int a, double b) { return a + (int)b; }\n")
    call = first(built, ast.Call)
    assert [arg.to for arg in call.args] == [INT, DOUBLE]


def test_review_prototype_with_unnamed_parameters_converts_arguments() -> None:
    built = build("int f(int, double);\nint main(void) { return f('a', 2); }\n"
                  "int f(int a, double b) { return a + (int)b; }\n")
    assert built.result.ok
    call = first(built, ast.Call)
    assert not isinstance(call.args[0], ast.ImplicitCast)   # 'a' is an int already
    assert call.args[1].to == DOUBLE


# ============================================================ references and keys

def test_review_shadowing_in_nested_blocks_resolves_to_each_declaration() -> None:
    built = build("int x = 9;\nint main(void) {\n int x = 1;\n {\n  int x = 2;\n  x++;\n }\n"
                  " return x;\n}\n")
    refs = [(n.loc.line, n.ref.kind) for n in nodes(built.function("main"), ast.Identifier)]
    assert refs == [(6, "local"), (8, "local")]
    assert [d.storage_key for d in nodes(built.function("main"), ast.VarDecl)] == ["x", "x"]
    assert [d.code for d in built.diagnostics] == []


def test_review_recursion_refers_to_the_function_being_defined() -> None:
    built = build("int fact(int n);\nint fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }\n"
                  "int main(void) { return fact(3); }\n")
    inner = first(built, ast.Call, fn="fact")
    assert inner.func.ref.kind == "function" and inner.func.ref.key == "fact"
    assert built.diagnostics == []


def test_review_static_local_keys_are_stable_across_two_checks() -> None:
    source = ("int tick(void) { static int n = 0; return ++n; }\n"
              "int main(void) { return tick(); }\n")
    keys = [[d.storage_key for d in nodes(build(source).unit, ast.VarDecl)] for _ in range(2)]
    assert keys[0] == keys[1] == ["main.c::tick::n@1:29"]


# ============================================================ uninitialized_use

@pytest.mark.parametrize("statements", [
    "int x; for (int i = 0; i < 3; i++) { x = i; } return x;",
    "int prev; int d = 0; for (int i = 0; i < 3; i++) { if (i > 0) d += i - prev; prev = i; }"
    " return d;",
    "int last; int i = 0; while (i < 3) { if (i) i += last; last = i; i++; } return i;",
    "int step; int t = 0; for (int i = 0; i < 9; i += step) { step = 2; t++; } return t;",
    "int x; scanf(\"%d\", &x); return x;",
    "int x; int *p = &x; *p = 4; return x;",
])
def test_review_no_false_uninitialized_warning(statements: str) -> None:
    assert "uninitialized_use" not in build(body(statements)).codes()


def test_review_uninitialized_read_before_any_loop_still_warns() -> None:
    built = build(body("int x; while (x < 3) { x++; } return x;"))
    assert built.codes().count("uninitialized_use") == 1


def test_review_loop_that_never_assigns_still_warns() -> None:
    built = build(body("int x; int t = 0; for (int i = 0; i < 3; i++) t += x; return t;"))
    assert "uninitialized_use" in built.codes()


# ============================================================ control_reaches_end

@pytest.mark.parametrize("statements, falls", [
    ("switch (n) { case 1: return 1; case 2: return 2; default: return 0; }", False),
    ("switch (n) { case 1: return 1; break; default: return 0; break; }", False),
    ("switch (n) { case 1: return 1; default: break; }", True),
    ("while (1) { if (n) return 1; }", False),
    ("while (1) { if (n) { return 1; break; } }", False),
    ("while (1) { if (n) break; }", True),
    ("for (;;) { switch (n) { case 1: break; default: return 2; } }", False),
    ("do { if (n) return 1; } while (1);", False),
    ("do { if (n) continue; return 1; } while (n > 3);", True),
    ("for (;;) { switch (n) { default: continue; } }", False),
])
def test_review_flow_analysis(statements: str, falls: bool) -> None:
    built = build(f"int f(int n) {{ {statements} }}\nint main(void) {{ return f(1); }}\n")
    assert can_fall_through(built.function("f").body) is falls


def test_review_return_then_break_in_switch_does_not_warn() -> None:
    built = build("int f(int n);\nint f(int n) { switch (n) { case 1: return 1; break;\n"
                  "default: return 0; break; } }\nint main(void) { return f(1); }\n")
    assert "control_reaches_end" not in built.codes()


# ============================================================ array bounds under sizeof

def test_review_sizeof_of_out_of_range_element_is_not_an_array_bounds_warning() -> None:
    built = build(body("int a[4]; return (int)sizeof a[10];"))
    assert "array_bounds" not in built.codes()


def test_review_out_of_range_read_still_warns() -> None:
    built = build(body("int a[4] = {0}; return a[4];"))
    assert built.codes() == ["array_bounds"]


# ============================================================ constant initializers

@pytest.mark.parametrize("source", [
    "int x = 5;\nint y = x;\nint main(void) { return y; }\n",
    "int f(void);\nint g = f();\nint main(void) { return g; }\nint f(void) { return 1; }\n",
    "int f(void);\nint main(void) { static int s = f(); return s; }\n"
    "int f(void) { return 1; }\n",
    "int a = 1;\nint b[2] = {1, a};\nint main(void) { return b[0]; }\n",
    "int main(void) { int local = 3; static int *p = &local; return *p; }\n",
    "int main(void) { const int k = 3; static int s = k; return s; }\n",
])
def test_review_non_constant_static_initializer_is_an_error(source: str) -> None:
    built = build(source)
    diag = built.only("bad_initializer")
    assert diag.message == "initializer element is not constant" and diag.is_error


@pytest.mark.parametrize("source", [
    "int g = 4;\nint *p = &g;\nint arr[3];\nint *q = arr;\nint *r = &arr[2];\n"
    "int main(void) { return *p + *q + *r; }\n",
    "enum { N = 3 };\ndouble third = 1.0 / N;\nunsigned long n = sizeof(int) * N;\n"
    "int main(void) { return (int)(third * n); }\n",
    "struct P { int x; int y; };\nstruct P origin = {0, -1};\nint *py = &origin.y;\n"
    "const char *names[] = {\"a\", \"b\"};\nint main(void) { return *py + names[0][0]; }\n",
    "int add(int a, int b);\nint (*op)(int, int) = add;\nint main(void) { return op(1, 2); }\n"
    "int add(int a, int b) { return a + b; }\n",
    "const int N = 5;\nint m = N * 2;\nint main(void) { static int count = 0; "
    "static char *s = \"hi\"; return m + count + s[0]; }\n",
])
def test_review_constant_static_initializers_are_accepted(source: str) -> None:
    assert build(source).result.ok
