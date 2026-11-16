"""Tests for interp.parser on valid programs: precedence, declarators, types, statements."""
from __future__ import annotations

import json

import pytest

from pycc.interp import astnodes as ast
from pycc.interp.c_types import (
    BOOL, CHAR, CHAR_PTR, DOUBLE, INT, LLONG, LONG, SCHAR, SHORT, UCHAR, UINT, ULLONG, ULONG,
    USHORT, VOID, ArrayType, EnumType, FunctionType, PointerType, StructType, format_type,
)
from pycc.interp.lexer import tokenize_text
from pycc.interp.parser import ParseResult, parse, parse_text


def parse_ok(source: str) -> ast.TranslationUnit:
    """Parse and assert that there was no diagnostic at all."""
    result = parse_text(source)
    assert result.diagnostics == [], [d.format_gcc() for d in result.diagnostics]
    return result.ast


def decl(source: str, index: int = -1) -> ast.Decl:
    """The ``index``-th top-level declaration of ``source``."""
    return parse_ok(source).decls[index]


def var_type(source: str) -> str:
    """The gcc-style type of the last declared variable."""
    node = decl(source)
    assert isinstance(node, ast.VarDecl)
    return format_type(node.ctype)


def expr(text: str) -> ast.Expr:
    """Parse ``text`` as an expression statement inside a function."""
    unit = parse_ok(f"void f(void) {{ {text}; }}")
    function = unit.decls[0]
    assert isinstance(function, ast.FunctionDef)
    statement = function.body.items[-1]
    assert isinstance(statement, ast.ExprStmt)
    return statement.expr


def shape(node: ast.Expr) -> str:
    """A compact fully-parenthesised rendering, to check precedence."""
    if isinstance(node, ast.BinaryOp):
        return f"({shape(node.left)} {node.op} {shape(node.right)})"
    if isinstance(node, ast.Assignment):
        return f"({shape(node.target)} {node.op} {shape(node.value)})"
    if isinstance(node, ast.UnaryOp):
        return f"({node.op}{shape(node.operand)})"
    if isinstance(node, ast.PostfixOp):
        return f"({shape(node.operand)}{node.op})"
    if isinstance(node, ast.Conditional):
        return f"({shape(node.cond)} ? {shape(node.then)} : {shape(node.otherwise)})"
    if isinstance(node, ast.Cast):
        return f"(({format_type(node.ctype)}){shape(node.expr)})"
    if isinstance(node, ast.Identifier):
        return node.name
    if isinstance(node, (ast.IntLiteral, ast.CharLiteral)):
        return str(node.value)
    if isinstance(node, ast.Index):
        return f"{shape(node.base)}[{shape(node.index)}]"
    if isinstance(node, ast.Call):
        return f"{shape(node.func)}({', '.join(shape(a) for a in node.args)})"
    if isinstance(node, ast.Member):
        return f"{shape(node.base)}{'->' if node.arrow else '.'}{node.name}"
    if isinstance(node, ast.SizeofType):
        return f"sizeof({format_type(node.ctype)})"
    if isinstance(node, ast.SizeofExpr):
        return f"sizeof {shape(node.expr)}"
    if isinstance(node, ast.Comma):
        return "(" + ", ".join(shape(e) for e in node.exprs) + ")"
    return type(node).__name__


# --- expression precedence --------------------------------------------------------------

@pytest.mark.parametrize("source, expected", [
    ("a = b ? c : d", "(a = (b ? c : d))"),
    ("*p++", "(*(p++))"),
    ("-x * y", "((-x) * y)"),
    ("a & b == c", "(a & (b == c))"),
    ("a || b && c", "(a || (b && c))"),
    ("a + b * c - d", "((a + (b * c)) - d)"),
    ("a - b - c", "((a - b) - c)"),
    ("a = b = c", "(a = (b = c))"),
    ("a += b << 2", "(a += (b << 2))"),
    ("a < b == c > d", "((a < b) == (c > d))"),
    ("a | b ^ c & d", "(a | (b ^ (c & d)))"),
    ("!a && ~b", "((!a) && (~b))"),
    ("&x[3]", "(&x[3])"),
    ("p->next->value", "p->next->value"),
    ("f(a, b)(c)", "f(a, b)(c)"),
    ("++*p", "(++(*p))"),
    ("a ? b : c ? d : e", "(a ? b : (c ? d : e))"),
    ("x = a, b", "((x = a), b)"),
    ("s.items[i].count++", "(s.items[i].count++)"),
    ("a % b / c", "((a % b) / c)"),
])
def test_precedence(source: str, expected: str) -> None:
    assert shape(expr(f"int a, b, c, d, e, x, y, *p, s; {source}")) == expected


def test_parentheses_override_precedence_and_are_recorded() -> None:
    node = expr("int a, b, c; (a + b) * c")
    assert shape(node) == "((a + b) * c)"
    assert isinstance(node, ast.BinaryOp)
    assert node.left.parenthesized and not node.parenthesized


def test_cast_versus_parenthesised_expression() -> None:
    assert shape(expr("int x; (int)x + 1")) == "(((int)x) + 1)"
    assert shape(expr("int x, y; (x) + y")) == "(x + y)"
    # With a typedef name the same spelling is a cast of the unary +y.
    unit = parse_ok("typedef int T; void f(void) { int y; (T) + y; }")
    statement = unit.decls[1].body.items[-1]
    assert isinstance(statement.expr, ast.Cast)
    assert isinstance(statement.expr.expr, ast.UnaryOp)


def test_cast_to_pointer_and_function_pointer_types() -> None:
    node = expr("void *v; (const char *)v")
    assert isinstance(node, ast.Cast) and node.ctype == CHAR_PTR
    node = expr("void *v; (int (*)(int))v")
    assert format_type(node.ctype) == "int (*)(int)"


def test_typedef_shadowed_by_variable_is_not_a_type() -> None:
    unit = parse_ok("typedef int T; void f(void) { int T = 2, y = 3; (T) * y; }")
    statement = unit.decls[1].body.items[-1]
    assert shape(statement.expr) == "(T * y)"


@pytest.mark.parametrize("source, expected", [
    ("sizeof(int)", "sizeof(int)"),
    ("sizeof x", "sizeof x"),
    ("sizeof(x)", "sizeof x"),
    ("sizeof x + 1", "(sizeof x + 1)"),
    ("sizeof(int) * 2", "(sizeof(int) * 2)"),
    ("sizeof(char *)", "sizeof(char *)"),
    ("sizeof arr[0]", "sizeof arr[0]"),
    ("sizeof(struct P)", "sizeof(struct P)"),
])
def test_sizeof_forms(source: str, expected: str) -> None:
    node = expr(f"struct P {{ int a; }}; int x, arr[3]; {source}")
    assert shape(node) == expected


def test_sizeof_of_typedef_name_is_sizeof_type() -> None:
    unit = parse_ok("typedef struct { int a, b; } Pair; unsigned long n = sizeof(Pair);")
    init = unit.decls[-1].init
    assert isinstance(init, ast.SizeofType)
    assert init.ctype.size == 8


def test_comma_expression_in_for() -> None:
    unit = parse_ok("void f(void) { int i, j; for (i = 0, j = 9; i < j; i++, j--) ; }")
    loop = unit.decls[0].body.items[-1]
    assert isinstance(loop.init, ast.Comma) and isinstance(loop.step, ast.Comma)
    assert isinstance(loop.body, ast.EmptyStmt)


def test_string_literal_concatenation() -> None:
    node = decl('char *s = "py" "cc" "\\n";').init
    assert isinstance(node, ast.StringLiteral)
    assert node.value == b"pycc\n"


def test_literal_nodes_keep_types_and_text() -> None:
    node = decl("long a = 2147483648;").init
    assert isinstance(node, ast.IntLiteral) and node.ctype == LONG and node.text == "2147483648"
    assert decl("unsigned b = 0xFFFFFFFF;").init.ctype == UINT
    assert decl("unsigned c = 10u;").init.ctype == UINT
    assert decl("long d = 5L;").init.ctype == LONG
    assert decl("double e = 1.5;").init.ctype == DOUBLE
    char = decl("char f = '\\n';").init
    assert isinstance(char, ast.CharLiteral) and char.value == 10


# --- declaration specifiers ---------------------------------------------------------------

@pytest.mark.parametrize("spelling, ctype", [
    ("char", CHAR), ("signed char", SCHAR), ("unsigned char", UCHAR),
    ("short", SHORT), ("short int", SHORT), ("signed short", SHORT),
    ("unsigned short int", USHORT), ("int", INT), ("signed", INT), ("signed int", INT),
    ("unsigned", UINT), ("unsigned int", UINT), ("long", LONG), ("long int", LONG),
    ("unsigned long", ULONG), ("long unsigned int", ULONG), ("long long", LLONG),
    ("long long int", LLONG), ("unsigned long long", ULLONG), ("long double", DOUBLE),
    ("double", DOUBLE), ("_Bool", BOOL),
])
def test_basic_type_combinations(spelling: str, ctype: object) -> None:
    assert decl(f"{spelling} x;").ctype == ctype


def test_storage_classes() -> None:
    unit = parse_ok("static int a; extern int b; void f(void) { register int r; auto int q; }")
    a, b, f = unit.decls
    assert a.storage == "static" and not a.is_extern
    assert b.storage == "extern" and b.is_extern
    locals_ = [stmt.decls[0] for stmt in f.body.items]
    assert [v.storage for v in locals_] == ["register", "auto"]


def test_volatile_and_inline_are_parsed_and_ignored() -> None:
    unit = parse_ok("volatile int v; static inline int sq(int x) { return x * x; }")
    assert unit.decls[0].ctype == INT
    function = unit.decls[1]
    assert isinstance(function, ast.FunctionDef) and function.is_inline
    assert function.storage == "static"


def test_const_flags() -> None:
    s = decl("const char *s;")
    assert format_type(s.ctype) == "char *"
    assert not s.is_const and s.points_to_const
    p = decl("char * const p = 0;")
    assert p.is_const and not p.points_to_const
    n = decl("const int n = 3;")
    assert n.is_const
    table = decl("const int table[2] = {1, 2};")
    assert table.is_const


# --- declarators --------------------------------------------------------------------------

@pytest.mark.parametrize("source, expected", [
    ("int *a[8];", "int *[8]"),
    ("int (*a)[8];", "int (*)[8]"),
    ("int **pp;", "int **"),
    ("int (*fp)(int, char *);", "int (*)(int, char *)"),
    ("int (*ops[4])(int);", "int (*)(int)[4]"),
    ("unsigned long long x;", "unsigned long long"),
    ("char *(*getter)(void);", "char * (*)(void)"),
    ("int ((x));", "int"),
    ("void (*handler)(int, ...);", "void (*)(int, ...)"),
])
def test_declarator_types(source: str, expected: str) -> None:
    assert var_type(source) == expected


def lengths(ctype: object) -> list[int | None]:
    """Array lengths from the outside in: int[3][4] gives [3, 4]."""
    found = []
    while isinstance(ctype, ArrayType):
        found.append(ctype.length)
        ctype = ctype.element
    return found


def test_multi_dimensional_array_is_array_of_arrays() -> None:
    grid = decl("int grid[3][4];").ctype
    assert lengths(grid) == [3, 4]
    assert grid.element.element == INT
    assert grid.size == 48


def test_array_of_pointers_versus_pointer_to_array_structure() -> None:
    arr = decl("int *a[8];").ctype
    assert isinstance(arr, ArrayType) and arr.length == 8 and isinstance(arr.element, PointerType)
    ptr = decl("int (*a)[8];").ctype
    assert isinstance(ptr, PointerType) and isinstance(ptr.target, ArrayType)
    assert ptr.target.length == 8


def test_function_prototypes() -> None:
    f = decl("static int f(void);")
    assert isinstance(f, ast.FunctionDecl) and f.storage == "static"
    assert f.ftype == FunctionType(INT, (), False, True)
    g = decl("int g();")
    assert g.ftype.has_prototype is False and g.ftype.params == ()
    h = decl("int h(int a, char *b);")
    assert h.param_names == ["a", "b"]
    assert h.ftype.params == (INT, CHAR_PTR)
    printf = decl("int printf(const char *fmt, ...);")
    assert printf.ftype.variadic and printf.ftype.params == (CHAR_PTR,)


def test_unnamed_parameters_in_prototype() -> None:
    f = decl("int f(int, char *, int (*)(int));")
    assert f.param_names == [None, None, None]
    assert format_type(f.ftype) == "int(int, char *, int (*)(int))"


def test_array_and_function_parameters_adjust_to_pointers() -> None:
    main = decl("int main(int argc, char *argv[]) { return 0; }")
    assert isinstance(main, ast.FunctionDef)
    assert main.param_names == ["argc", "argv"]
    assert format_type(main.ftype.params[1]) == "char **"
    apply = decl("int apply(int f(int), int v);")
    assert format_type(apply.ftype.params[0]) == "int (*)(int)"


def test_function_definition_details() -> None:
    f = decl("static int add(const int a, int b) { return a + b; }")
    assert isinstance(f, ast.FunctionDef)
    assert f.name == "add" and f.storage == "static"
    assert f.param_is_const == [True, False]
    assert [loc.column for loc in f.param_locs] == [26, 33]
    assert isinstance(f.body.items[0], ast.ReturnStmt)


def test_function_returning_function_pointer() -> None:
    f = decl("int (*pick(int which))(int) { return 0; }")
    assert isinstance(f, ast.FunctionDef) and f.param_names == ["which"]
    assert format_type(f.ftype.ret) == "int (*)(int)"


def test_multiple_declarators_share_specifiers() -> None:
    unit = parse_ok("int a, *b, c[2] = {1, 2};")
    assert [format_type(d.ctype) for d in unit.decls] == ["int", "int *", "int[2]"]


# --- array sizes and initializers -----------------------------------------------------------

def test_array_size_constant_folding() -> None:
    assert var_type("int a[2 * 3 + 1];") == "int[7]"
    assert var_type("int b[sizeof(int) * 2];") == "int[8]"
    assert var_type("enum { N = 4 }; int c[N + 1];") == "int[5]"
    assert var_type("int d[(1 << 3) - 1];") == "int[7]"


def test_array_length_inferred_from_initializers() -> None:
    assert var_type("int a[] = {1, 2, 3};") == "int[3]"
    assert var_type('char s[] = "abc";') == "char[4]"
    assert var_type('char t[] = {"hi"};') == "char[3]"
    assert var_type("char u[] = {'a', 'b'};") == "char[2]"
    assert var_type('char e[] = "";') == "char[1]"
    assert var_type('char z[] = "a\\0b";') == "char[4]"
    assert lengths(decl("int m[][2] = {{1, 2}, {3, 4}, {5, 6}};").ctype) == [3, 2]


def test_nested_brace_initializers() -> None:
    init = decl("int m[2][2] = {{1, 2}, {3, 4},};").init
    assert isinstance(init, ast.InitList) and len(init.items) == 2
    assert all(isinstance(item, ast.InitList) for item in init.items)
    assert init.items[1].items[0].value == 3


def test_extern_array_may_have_no_size() -> None:
    assert var_type("extern int table[];") == "int[]"


def test_exact_fit_char_array_is_allowed() -> None:
    assert var_type('char s[3] = "abc";') == "char[3]"


# --- structs, unions, enums, typedefs ------------------------------------------------------

def test_struct_definition_with_layout() -> None:
    unit = parse_ok("struct Tile { char kind; int crop; double water; };")
    struct_decl = unit.decls[0]
    assert isinstance(struct_decl, ast.StructDecl) and struct_decl.tag == "Tile"
    stype = struct_decl.ctype
    assert [m.name for m in stype.members] == ["kind", "crop", "water"]
    assert [m.offset for m in stype.members] == [0, 4, 8]
    assert stype.size == 16


def test_self_referential_struct() -> None:
    unit = parse_ok("struct Node { int value; struct Node *next; }; struct Node head;")
    stype = unit.decls[0].ctype
    next_type = stype.member("next").ctype
    assert isinstance(next_type, PointerType) and next_type.target is stype
    assert unit.decls[1].ctype is stype
    assert stype.size == 16


def test_nested_struct_definition() -> None:
    unit = parse_ok("struct Outer { struct Inner { int a; char b; } in; int c; }; "
                    "struct Inner loose;")
    kinds = [type(d).__name__ for d in unit.decls]
    assert kinds == ["StructDecl", "StructDecl", "VarDecl"]
    inner, outer, loose = unit.decls
    assert inner.tag == "Inner" and outer.tag == "Outer"
    assert outer.ctype.member("in").ctype is inner.ctype
    assert loose.ctype is inner.ctype
    assert outer.ctype.size == 12


def test_forward_declaration_completed_later() -> None:
    unit = parse_ok("struct Crate; struct Crate *spare; struct Crate { int count; };")
    forward, spare, definition = unit.decls
    assert isinstance(forward, ast.StructDecl) and not forward.is_definition
    assert spare.ctype.target is definition.ctype
    assert definition.ctype.is_complete


def test_union() -> None:
    node = decl("union Value { int i; double d; char c[3]; };")
    assert isinstance(node, ast.StructDecl)
    assert node.ctype.is_union and node.ctype.size == 8
    assert all(m.offset == 0 for m in node.ctype.members)


def test_struct_definition_with_variables() -> None:
    unit = parse_ok("struct P { int x, y; } origin, *cursor;")
    struct_decl, origin, cursor = unit.decls
    assert origin.ctype is struct_decl.ctype
    assert cursor.ctype.target is struct_decl.ctype


def test_array_of_structs() -> None:
    unit = parse_ok("struct Tile { int crop; }; struct Tile field[4][5];")
    field_type = unit.decls[1].ctype
    assert lengths(field_type) == [4, 5]
    assert field_type.element.element is unit.decls[0].ctype
    assert field_type.size == 80


def test_enum_with_explicit_values_and_folding() -> None:
    node = decl("enum Dir { NORTH, EAST = 4, SOUTH, WEST = EAST * 2 + 1, LAST = -1 };")
    assert isinstance(node, ast.EnumDecl) and node.tag == "Dir"
    assert node.constants == [("NORTH", 0), ("EAST", 4), ("SOUTH", 5), ("WEST", 9), ("LAST", -1)]
    assert isinstance(node.ctype, EnumType) and node.ctype.constants == tuple(node.constants)


def test_enum_constants_stay_identifiers_in_expressions() -> None:
    unit = parse_ok("enum { RED, GREEN }; int c = GREEN;")
    init = unit.decls[-1].init
    assert isinstance(init, ast.Identifier) and init.name == "GREEN"


def test_enum_variable_and_trailing_comma() -> None:
    unit = parse_ok("enum Crop { WHEAT, CORN, }; enum Crop planted = CORN;")
    assert unit.decls[1].ctype is unit.decls[0].ctype


def test_typedef_chains() -> None:
    unit = parse_ok("typedef struct { int x, y; } Tile; typedef Tile *TileP; "
                    "typedef TileP Row[3]; Row row; TileP p;")
    struct_decl, tile, tilep, row_t, row, p = unit.decls
    assert isinstance(tile, ast.TypedefDecl) and tile.ctype is struct_decl.ctype
    assert isinstance(tilep.ctype, PointerType) and tilep.ctype.target is struct_decl.ctype
    assert isinstance(row.ctype, ArrayType) and row.ctype.length == 3
    assert row.ctype.element.target is struct_decl.ctype
    assert p.ctype == tilep.ctype


def test_typedef_of_basic_and_function_types() -> None:
    unit = parse_ok("typedef unsigned char byte; typedef int Fn(int); byte b; Fn *call;")
    assert unit.decls[2].ctype == UCHAR
    assert format_type(unit.decls[3].ctype) == "int (*)(int)"


def test_typedef_const_carries_to_variables() -> None:
    unit = parse_ok("typedef const int CInt; CInt limit = 3;")
    assert unit.decls[0].is_const and unit.decls[1].is_const


def test_block_scoped_typedef_and_tag() -> None:
    unit = parse_ok("void f(void) { typedef int Count; struct S { int a; }; Count c; "
                    "struct S s; }")
    items = unit.decls[0].body.items
    assert all(isinstance(item, ast.DeclStmt) for item in items)
    assert isinstance(items[0].decls[0], ast.TypedefDecl)
    assert isinstance(items[1].decls[0], ast.StructDecl)


def test_inner_struct_tag_shadows_outer() -> None:
    unit = parse_ok("struct S { int a; }; void f(void) { struct S { char c; } inner; } "
                    "struct S outer;")
    outer_struct = unit.decls[0].ctype
    inner_decl = unit.decls[1].body.items[0].decls[1]
    assert inner_decl.ctype is not outer_struct
    assert unit.decls[2].ctype is outer_struct


# --- statements ---------------------------------------------------------------------------

STATEMENTS = """
int main(void) {
    int i = 0, total = 0;
    ;
    { int inner = 1; total += inner; }
    if (i < 3) total++; else total--;
    if (i) { total = 0; }
    while (i < 10) i++;
    do { i--; } while (i > 0);
    for (int k = 0; k < 3; k++) continue;
    for (;;) break;
    switch (total) {
        case 1:
        case 2: total = 5; break;
        case 'a' + 1: { int z = 2; total = z; } break;
        default: total = 0;
    }
    return total;
}
"""


def statement_kinds() -> list[str]:
    """Node names of the statements in STATEMENTS' main body."""
    function = parse_ok(STATEMENTS).decls[0]
    return [type(item).__name__ for item in function.body.items]


def test_every_statement_kind() -> None:
    assert statement_kinds() == [
        "DeclStmt", "EmptyStmt", "CompoundStmt", "IfStmt", "IfStmt", "WhileStmt",
        "DoWhileStmt", "ForStmt", "ForStmt", "SwitchStmt", "ReturnStmt"]


def test_statement_details() -> None:
    items = parse_ok(STATEMENTS).decls[0].body.items
    if_else, if_only = items[3], items[4]
    assert isinstance(if_else.otherwise, ast.ExprStmt) and if_only.otherwise is None
    for_decl, for_ever = items[7], items[8]
    assert isinstance(for_decl.init, ast.DeclStmt) and isinstance(for_decl.body, ast.ContinueStmt)
    assert for_ever.init is None and for_ever.cond is None and for_ever.step is None
    assert isinstance(for_ever.body, ast.BreakStmt)
    assert isinstance(items[-1].value, ast.Identifier)


def test_switch_labels_nest_and_fold() -> None:
    switch = parse_ok(STATEMENTS).decls[0].body.items[9]
    body = switch.body.items
    first = body[0]
    assert isinstance(first, ast.CaseLabel) and first.const_value == 1
    assert isinstance(first.body, ast.CaseLabel) and first.body.const_value == 2
    assert isinstance(body[2], ast.CaseLabel) and body[2].const_value == 98
    assert isinstance(body[-1], ast.DefaultLabel)


def test_case_label_with_enum_constant() -> None:
    unit = parse_ok("enum { A, B = 7 }; int f(int v) { switch (v) { case B: return 1; } "
                    "return 0; }")
    case = unit.decls[1].body.items[0].body.items[0]
    assert case.const_value == 7 and isinstance(case.value, ast.Identifier)


def test_dangling_else_binds_to_nearest_if() -> None:
    function = parse_ok("void f(int a, int b) { if (a) if (b) a = 1; else a = 2; }").decls[0]
    outer = function.body.items[0]
    assert outer.otherwise is None and outer.then.otherwise is not None


def test_return_without_value() -> None:
    function = parse_ok("void f(void) { return; }").decls[0]
    assert function.body.items[0].value is None


def test_mixed_declarations_and_code() -> None:
    items = parse_ok("void f(void) { int a = 1; a++; int b = a; }").decls[0].body.items
    assert [type(i).__name__ for i in items] == ["DeclStmt", "ExprStmt", "DeclStmt"]


def test_local_function_prototype() -> None:
    items = parse_ok("void f(void) { int helper(int); helper(2); }").decls[0].body.items
    assert isinstance(items[0].decls[0], ast.FunctionDecl)


# --- locations, API and tree helpers ---------------------------------------------------------

def test_node_locations() -> None:
    unit = parse_ok("int main(void) {\n    int x = 1 + 2;\n    return x;\n}")
    main = unit.decls[0]
    assert (main.loc.line, main.loc.column) == (1, 5)
    var = main.body.items[0].decls[0]
    assert (var.loc.line, var.loc.column) == (2, 9)
    assert (var.init.loc.line, var.init.loc.column) == (2, 15)  # the '+' operator
    assert (main.body.items[1].loc.line, main.body.items[1].loc.column) == (3, 5)


def test_parse_accepts_tokens_and_unit_name() -> None:
    result = parse(tokenize_text("int x;", "farm.c"), "farm.c")
    assert isinstance(result, ParseResult) and result.ok
    assert result.ast.name == "farm.c"
    assert result.ast.decls[0].loc.file == "farm.c"


def test_parse_adds_missing_eof() -> None:
    tokens = tokenize_text("int x;", "a.c")[:-1]
    assert parse(tokens, "a.c").ok


def test_to_dict_is_json_serialisable() -> None:
    unit = parse_ok('struct P { int x; }; int main(void) { char *s = "hi"; return 0; }')
    data = unit.to_dict()
    text = json.dumps(data)
    assert data["node"] == "TranslationUnit"
    assert data["decls"][0]["ctype"] == "struct P"
    local = data["decls"][1]["body"]["items"][0]["decls"][0]
    assert local["ctype"] == "char *"
    assert local["init"]["value"] == "hi"
    assert json.loads(text) == data


def test_walk_visits_every_node() -> None:
    unit = parse_ok("int f(int a) { if (a) return a * 2; return 0; }")
    names = [type(node).__name__ for node in ast.walk(unit)]
    assert names[0] == "TranslationUnit"
    assert names.count("ReturnStmt") == 2
    assert "BinaryOp" in names and "Identifier" in names


def test_dump_outline() -> None:
    text = ast.dump(parse_ok("int main(void) { return 0; }"))
    assert text.splitlines() == [
        "TranslationUnit 'main.c'",
        "  FunctionDef main: int(void)",
        "    CompoundStmt",
        "      ReturnStmt",
        "        IntLiteral 0 (int)",
    ]


def test_void_function_and_struct_return() -> None:
    unit = parse_ok("struct P { int x; }; struct P make(void); void run(void) {}")
    assert unit.decls[1].ftype.ret is unit.decls[0].ctype
    assert unit.decls[2].ftype.ret == VOID


def test_struct_type_identity_is_preserved() -> None:
    stype = decl("struct Q { int a; };").ctype
    assert isinstance(stype, StructType) and stype.tag == "Q"
