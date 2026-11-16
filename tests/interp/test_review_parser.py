"""Review tests for the parser: declarators, scopes, error positions, recovery, limits."""
from __future__ import annotations

import os
import time

from pycc.interp import astnodes as ast
from pycc.interp.c_types import ArrayType, FunctionType, PointerType, StructType
from pycc.interp.parser import MAX_NESTING, ParseResult, parse_text


def first_lines(result: ParseResult) -> list[str]:
    """The first gcc line of every diagnostic, for compact asserts."""
    return [diag.format_gcc().splitlines()[0] for diag in result.diagnostics]


def var(result: ParseResult, name: str) -> ast.VarDecl:
    """The VarDecl called ``name`` anywhere in the tree."""
    for node in ast.walk(result.ast):
        if isinstance(node, ast.VarDecl) and node.name == name:
            return node
    raise AssertionError(f"no variable {name}")


# --- declarators ---------------------------------------------------------------------

def test_array_of_pointers_to_functions_returning_pointer() -> None:
    ctype = parse_text("int *(*fp[3])(void);").ast.decls[0].ctype
    assert isinstance(ctype, ArrayType) and ctype.length == 3
    function = ctype.element.target
    assert isinstance(function, FunctionType)
    assert function.params == () and function.has_prototype
    assert isinstance(function.ret, PointerType)


def test_pointer_to_two_dimensional_array() -> None:
    ctype = parse_text("int (*a)[8][8];").ast.decls[0].ctype
    assert isinstance(ctype, PointerType)
    assert ctype.target.length == 8 and ctype.target.element.length == 8
    assert ctype.size == 8 and ctype.target.size == 256


def test_function_returning_pointer_to_function() -> None:
    decl = parse_text("int *(*g(int))(char);").ast.decls[0]
    assert isinstance(decl, ast.FunctionDecl)
    returned = decl.ftype.ret
    assert isinstance(returned, PointerType) and isinstance(returned.target, FunctionType)
    assert isinstance(returned.target.ret, PointerType)


def test_function_returning_pointer_to_array() -> None:
    decl = parse_text("int (*f(void))[3];").ast.decls[0]
    assert isinstance(decl, ast.FunctionDecl)
    assert decl.ftype.ret.target.length == 3


# --- scopes ---------------------------------------------------------------------------

def test_typedef_shadowed_in_block_then_visible_again() -> None:
    source = "typedef int T; void f(void) { { int T = 2; T * 3; } T x; x = 1; }"
    result = parse_text(source)
    assert result.ok, first_lines(result)
    inner_block = result.ast.decls[1].body.items[0]
    assert isinstance(inner_block.items[1], ast.ExprStmt)  # 'T * 3' is a multiplication
    assert isinstance(result.ast.decls[1].body.items[1], ast.DeclStmt)


def test_struct_tag_and_typedef_name_share_one_type() -> None:
    result = parse_text("typedef struct Tile Tile; struct Tile { int x; }; Tile t;")
    assert result.ok
    tile = var(result, "t").ctype
    assert isinstance(tile, StructType) and tile.is_complete and tile.size == 4


def test_incomplete_struct_completed_later_is_the_same_type() -> None:
    result = parse_text("struct S; struct S *p; struct S { int a; double b; };")
    assert result.ok
    assert var(result, "p").ctype.target.size == 16


def test_struct_and_union_with_one_tag_is_an_error() -> None:
    result = parse_text("struct S { int a; } s; union S u;")
    assert [d.message for d in result.errors] == ["'S' defined as wrong kind of tag"]


# --- error positions -------------------------------------------------------------------

def test_missing_semicolon_at_end_of_line_points_after_last_token() -> None:
    result = parse_text("int main(void) {\n  int x = 5\n  return x;\n}")
    assert first_lines(result) == ["main.c:2:12: error: expected ',' or ';' before 'return'"]


def test_missing_semicolon_before_brace_on_next_line() -> None:
    result = parse_text("int main(void) {\n  int x;\n  x = 6\n}")
    assert first_lines(result) == ["main.c:3:8: error: expected ';' before '}' token"]


# --- recovery --------------------------------------------------------------------------

def test_missing_paren_in_call_is_one_error() -> None:
    result = parse_text('int main(void) {\n  printf("%d" x);\n  return 0;\n}')
    assert first_lines(result) == ["main.c:2:14: error: expected ')' before 'x'"]


def test_missing_paren_before_next_line_statement_is_one_error() -> None:
    result = parse_text("int main(void) {\n  foo(1, 2\n  bar();\n  return 0;\n}")
    assert len(result.errors) == 1


def test_missing_function_body_brace_is_one_error() -> None:
    result = parse_text("int main(void)\n  return 0;\n}")
    assert first_lines(result) == ["main.c:1:15: error: expected '{' before 'return'"]
    assert isinstance(result.ast.decls[0], ast.FunctionDef)


def test_missing_semicolon_after_prototype_keeps_usual_error() -> None:
    result = parse_text("int f(void)\nint main(void) { return 0; }")
    assert len(result.errors) == 1
    assert result.errors[0].code == "missing_semicolon"


def test_several_independent_mistakes_each_reported_once() -> None:
    source = ("int main(void) {\n int x = 5\n int y = 6;\n printf(\"%d\" x);\n"
              " return 0\n}")
    assert len(parse_text(source).errors) == 3


def test_enum_overflow_is_reported() -> None:
    result = parse_text("enum { A = 2147483647, B };")
    assert [d.message for d in result.errors] == ["overflow in enumeration values"]


def test_enum_value_outside_int_warns() -> None:
    result = parse_text("enum { A = 4294967296 };")
    assert result.ok and result.diagnostics[0].code == "enum_overflow"


# --- limits ----------------------------------------------------------------------------

def test_two_hundred_nested_parens_parse() -> None:
    result = parse_text("int x = " + "(" * 200 + "1" + ")" * 200 + ";")
    assert result.ok
    assert var(result, "x").init.parenthesized


def test_two_hundred_nested_blocks_parse() -> None:
    result = parse_text("void f(void) " + "{" * 200 + "}" * 200)
    assert result.ok


def test_absurd_nesting_is_one_diagnostic_not_a_crash() -> None:
    depth = MAX_NESTING * 4
    result = parse_text("int a = 1; int x = " + "(" * depth + "1" + ")" * depth
                        + "; int z = 2;")
    assert [d.code for d in result.errors] == ["nesting_too_deep"]
    assert [d.name for d in result.ast.decls] == ["a", "z"]  # parsing went on


def test_long_unary_chain_is_a_diagnostic_not_a_crash() -> None:
    result = parse_text("int x = " + "!" * 9000 + "1; int z = 2;")
    assert [d.code for d in result.errors] == ["nesting_too_deep"]
    assert result.ast.decls[-1].name == "z"


def test_long_flat_expression_can_be_walked() -> None:
    result = parse_text("int main(void) { int a = 1; return " + " + ".join(["a"] * 3000)
                        + "; }")
    assert result.ok
    assert sum(1 for _ in ast.walk(result.ast)) > 6000
    assert ast.dump(result.ast).count("Identifier a") == 3000


def test_two_thousand_line_file_parses_under_a_second() -> None:
    lines = [f"int g{i}(int a, int b) {{ int c = a * b + {i}; if (c > 3) {{ c -= 1; }} "
             f"return c; }}" for i in range(2000)]
    start = time.perf_counter()
    result = parse_text("\n".join(lines))
    assert time.perf_counter() - start < (4.0 if os.environ.get("CI") else 1.0)   # CI machines are slower
    assert result.ok and len(result.ast.decls) == 2000


def test_duplicate_member_points_at_second_name() -> None:
    result = parse_text("struct S {\n  int a;\n  char b, a;\n};")
    assert first_lines(result) == ["main.c:3:11: error: duplicate member 'a'"]


def test_deep_if_chain_parses() -> None:
    result = parse_text("void f(int x) { " + "if (x) " * 200 + "x = 1; }")
    assert result.ok


def test_nesting_counter_recovers_after_an_error() -> None:
    depth = MAX_NESTING + 10
    bad = "int x = " + "(" * depth + "1" + ")" * depth + ";\n"
    good = "int y = " + "(" * 200 + "1" + ")" * 200 + ";\n"
    result = parse_text(bad + good)
    assert [d.code for d in result.errors] == ["nesting_too_deep"]
    assert result.ast.decls[-1].name == "y"
