"""Tests for the semantic checker: errors, AST annotations, symbols and storage keys.

Warnings and their groups live in test_checker_warnings.py. Programs are built the way
the compiler driver builds them: preprocess (with small fake system headers), lex,
parse, then ``check``.
"""
from __future__ import annotations

import time

import pytest

from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.c_types import (
    CHAR, DOUBLE, INT, SIZE_T, ArrayType, FunctionType, PointerType, StructType,
)
from pycc.interp.checker import ERROR_TYPE, CheckResult, can_fall_through, check
from pycc.interp.constexpr import eval_const_expr
from pycc.interp.lexer import tokenize
from pycc.interp.parser import parse
from pycc.interp.preprocessor import DictResolver, preprocess
from pycc.interp.toolchain import CompileOptions

STDIO_H = """#ifndef _STDIO_H
#define _STDIO_H
#define NULL ((void *)0)
int printf(const char *format, ...);
int scanf(const char *format, ...);
int puts(const char *s);
#endif
"""
STDLIB_H = """#ifndef _STDLIB_H
#define _STDLIB_H
void exit(int status);
void *malloc(unsigned long size);
void free(void *p);
#endif
"""
FARM_H = """#ifndef CV_FARM_H
#define CV_FARM_H
int move(int direction);
#endif
"""
SYSTEM_HEADERS = {"stdio.h": STDIO_H, "stdlib.h": STDLIB_H, "farm.h": FARM_H}


def _noop(machine: object, args: list) -> None:
    return None


def make_registry() -> BuiltinRegistry:
    """A registry like the real one, with one device function."""
    registry = BuiltinRegistry()
    registry.register("printf", "int printf(const char *format, ...);", _noop,
                      header="stdio.h", format_arg=0, format_kind="printf")
    registry.register("scanf", "int scanf(const char *format, ...);", _noop,
                      header="stdio.h", format_arg=0, format_kind="scanf")
    registry.register("puts", "int puts(const char *s);", _noop, header="stdio.h")
    registry.register("exit", "void exit(int status);", _noop, header="stdlib.h")
    registry.register("malloc", "void *malloc(unsigned long size);", _noop, header="stdlib.h")
    registry.register("free", "void free(void *p);", _noop, header="stdlib.h")
    registry.register("move", "int move(int direction);", _noop, header="farm.h")
    return registry


@pytest.fixture(scope="module")
def registry() -> BuiltinRegistry:
    return make_registry()


class Build:
    """One compiled unit: the AST, the checker's result and the front end's diagnostics."""

    def __init__(self, unit: ast.TranslationUnit, result: CheckResult,
                 front: list[Diagnostic]) -> None:
        self.unit = unit
        self.result = result
        self.front = front

    @property
    def diagnostics(self) -> list[Diagnostic]:
        return self.result.diagnostics

    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]

    def only(self, code: str) -> Diagnostic:
        found = [d for d in self.diagnostics if d.code == code]
        assert len(found) == 1, [d.format_gcc() for d in self.diagnostics]
        return found[0]

    def function(self, name: str) -> ast.FunctionDef:
        for decl in self.unit.decls:
            if isinstance(decl, ast.FunctionDef) and decl.name == name:
                return decl
        raise KeyError(name)


def build(source: str, *, files: dict[str, str] | None = None,
          options: CompileOptions | None = None, name: str = "main.c",
          reg: BuiltinRegistry | None = None) -> Build:
    """Preprocess, lex, parse and check ``source`` as the file ``name``."""
    resolver = DictResolver({name: source, **(files or {})}, SYSTEM_HEADERS)
    pre = preprocess(source, name, resolver)
    front = list(pre.diagnostics)
    tokens = tokenize(pre.lines, front)
    parsed = parse(tokens, name)
    front.extend(parsed.diagnostics)
    result = check(parsed.ast, reg or make_registry(),
                   options or CompileOptions(wall=True, wextra=True))
    return Build(parsed.ast, result, front)


def where(diag: Diagnostic) -> str:
    return str(diag.location)


def body(src: str) -> str:
    """Wrap statements in 'int main(void) { ... }' on line 2 (line 1 includes stdio)."""
    return f"#include <stdio.h>\nint main(void) {{ {src} }}\n"


def nodes(root: ast.Node, kind: type) -> list:
    return [n for n in ast.walk(root) if isinstance(n, kind)]


# ============================================================================ basics

def test_clean_program_has_no_diagnostics() -> None:
    result = build("#include <stdio.h>\nint main(void) {\n    printf(\"hi\\n\");\n"
                   "    return 0;\n}\n").result
    assert result.diagnostics == [] and result.ok


def test_check_accepts_default_registry_and_options() -> None:
    parsed = parse(tokenize(preprocess("int main(void) { return 0; }", "m.c",
                                       DictResolver()).lines), "m.c")
    result = check(parsed.ast)
    assert result.ok and result.symbols.source_path == "m.c"


def test_every_diagnostic_is_a_check_phase_diagnostic_with_friendly_text() -> None:
    found = build(body("int y = x; struct S *p = 0; p->q = 1;")).diagnostics
    assert found
    assert all(d.phase is Phase.CHECK and d.friendly and d.concept for d in found)


# ============================================================================ errors

def test_undeclared_identifier_with_did_you_mean() -> None:
    diag = build(body("int count = 1; return cuont;")).only("undeclared_identifier")
    assert diag.message == "'cuont' undeclared (first use in this function); did you mean 'count'?"
    assert where(diag) == "main.c:2:40"
    assert diag.severity is Severity.ERROR
    assert "Did you mean 'count'?" in diag.friendly
    assert diag.notes and "only once" in diag.notes[0]


def test_undeclared_identifier_differing_only_in_case_teaches_case() -> None:
    source = "enum Dir { NORTH, EAST };\nint main(void) { return north; }\n"
    diag = build(source).only("undeclared_identifier")
    assert diag.message.endswith("did you mean 'NORTH'?")
    assert "C is case-sensitive" in diag.friendly
    assert "int north" not in diag.friendly


def test_undeclared_with_suggestion_does_not_suggest_a_declaration() -> None:
    diag = build(body("int count = 1; return cuont;")).only("undeclared_identifier")
    assert "int cuont" not in diag.friendly


def test_implicit_declaration_differing_only_in_case() -> None:
    source = "int move(int d);\nint main(void) { Move(1); return 0; }\n"
    diag = build(source).only("implicit_declaration")
    assert diag.message.endswith("did you mean 'move'?")
    assert "C is case-sensitive" in diag.friendly


def test_undeclared_identifier_is_reported_once_per_function() -> None:
    found = build(body("int a = zz; int b = zz; return a + b;")).diagnostics
    assert [d.code for d in found] == ["undeclared_identifier"]


def test_undeclared_identifier_at_file_scope() -> None:
    diag = build("int x = y;\n").only("undeclared_identifier")
    assert diag.message == "'y' undeclared here (not in a function)"


def test_undeclared_library_variable_suggests_header() -> None:
    reg = make_registry()
    reg.register_variable("stdin", "FILE *stdin;", lambda m: b"", header="stdio.h")
    diag = build("int main(void) { return stdin == 0; }", reg=reg).only("undeclared_identifier")
    assert "#include <stdio.h>" in diag.friendly


def test_implicit_declaration_suggests_the_header() -> None:
    diag = build("int main(void) {\n    printf(\"hi\");\n    return 0;\n}\n").only(
        "implicit_declaration")
    assert diag.message == "implicit declaration of function 'printf'"
    assert where(diag) == "main.c:2:5"
    assert diag.severity is Severity.ERROR
    assert "Did you forget #include <stdio.h>?" in diag.friendly


def test_implicit_declaration_is_an_error_even_without_wall() -> None:
    found = build("int main(void) { helper(); return 0; }", options=CompileOptions()).diagnostics
    assert [(d.code, d.severity) for d in found] == [("implicit_declaration", Severity.ERROR)]


def test_implicit_declaration_did_you_mean() -> None:
    diag = build(body("prinft(\"x\"); return 0;")).only("implicit_declaration")
    assert diag.message.endswith("did you mean 'printf'?")
    assert "Did you mean 'printf'?" in diag.friendly


def test_implicit_declaration_reported_once_then_declared() -> None:
    found = build("int main(void) { helper(); helper(); return 0; }").diagnostics
    assert [d.code for d in found] == ["implicit_declaration"]

def test_conflicting_types_prototype_versus_definition() -> None:
    diag = build("int area(int w, int h);\nint area(int w) { return w; }\n").only(
        "conflicting_types")
    assert diag.message == "conflicting types for 'area'"
    assert where(diag) == "main.c:2:5"
    assert diag.notes == ["previous declaration of 'area' was here"]
    assert "int area(int, int)" in diag.friendly and "int area(int)" in diag.friendly


def test_conflicting_types_two_prototypes() -> None:
    diag = build("int f(int a);\ndouble f(int a);\n").only("conflicting_types")
    assert where(diag) == "main.c:2:8"


def test_conflicting_types_variable_versus_function() -> None:
    diag = build("int total;\nint total(void) { return 1; }\n").only("conflicting_types")
    assert diag.message == "'total' redeclared as different kind of symbol"
    assert where(diag) == "main.c:2:5"


def test_conflicting_types_for_variables() -> None:
    diag = build("int x;\ndouble x;\n").only("conflicting_types")
    assert diag.message == "conflicting types for 'x'"


def test_old_style_prototype_then_definition_is_fine() -> None:
    assert build("int f();\nint f(int a) { return a; }\n"
                 "int main(void) { return f(1); }\n",
                 options=CompileOptions()).diagnostics == []


def test_redefinition_of_function() -> None:
    diag = build("int one(void) { return 1; }\nint one(void) { return 1; }\n").only(
        "redefinition")
    assert diag.message == "redefinition of 'one'"
    assert where(diag) == "main.c:2:5"
    assert diag.notes == ["previous definition of 'one' was here"]
    assert "main.c:1:5" in diag.friendly


def test_redefinition_of_variable() -> None:
    diag = build("int x = 1;\nint x = 2;\n").only("redefinition")
    assert (diag.message, where(diag)) == ("redefinition of 'x'", "main.c:2:5")
    assert diag.notes == ["previous definition of 'x' was here"]


def test_redefinition_from_double_include_explains_include_guards() -> None:
    header = "int counter = 0;\n"
    source = '#include "count.h"\n#include "count.h"\nint main(void) { return counter; }\n'
    diag = build(source, files={"count.h": header}).only("redefinition")
    assert where(diag) == "count.h:1:5"
    assert "include guard" in diag.friendly and "#ifndef COUNT_H" in diag.friendly


def test_struct_redefined_by_double_include_explains_include_guards() -> None:
    header = "struct Point { int x; int y; };\n"
    source = '#include "point.h"\n#include "point.h"\nint main(void) { return 0; }\n'
    built = build(source, files={"point.h": header})
    found = [d for d in built.front if d.code == "redefinition"]
    assert len(found) == 1
    diag = found[0]
    assert diag.message == "redefinition of 'struct Point'"
    assert where(diag) == "point.h:1:8"
    assert diag.notes == ["previous definition of 'struct Point' was here"]
    assert "point.h was included more than once" in diag.friendly
    assert "#ifndef POINT_H" in diag.friendly


def test_struct_with_include_guard_is_fine() -> None:
    header = "#ifndef POINT_H\n#define POINT_H\nstruct Point { int x; };\n#endif\n"
    source = '#include "point.h"\n#include "point.h"\nint main(void) { return 0; }\n'
    built = build(source, files={"point.h": header})
    assert built.front == [] and built.diagnostics == []


def test_local_redeclaration() -> None:
    diag = build(body("int a = 1; int a = 2; return a;")).only("redefinition")
    assert diag.message == "redeclaration of 'a' with no linkage"
    assert diag.notes == ["previous declaration of 'a' was here"]


def test_shadowing_in_inner_block_is_allowed() -> None:
    assert build(body("int a = 1; { int a = 2; a++; } return a;")).diagnostics == []


def test_duplicate_parameter_name() -> None:
    diag = build("int f(int a, int a) { return a; }\n", options=CompileOptions()).only(
        "redefinition")
    assert diag.message == "redefinition of parameter 'a'"


def test_too_few_arguments() -> None:
    diag = build("int add(int a, int b) { return a + b; }\n"
                 "int main(void) { return add(1); }\n").only("too_few_arguments")
    assert diag.message == "too few arguments to function 'add'"
    assert where(diag) == "main.c:2:25"
    assert diag.notes == ["declared here"]
    assert "int add(int, int)" in diag.friendly


def test_too_many_arguments() -> None:
    diag = build("int one(void) { return 1; }\n"
                 "int main(void) { return one(1, 2); }\n").only("too_many_arguments")
    assert diag.message == "too many arguments to function 'one'"
    assert "takes 0 arguments but this call passes 2" in diag.friendly


def test_variadic_builtin_accepts_extra_arguments() -> None:
    assert build(body('printf("%d %d", 1, 2); return 0;')).diagnostics == []


def test_not_a_struct_dot_on_int() -> None:
    diag = build(body("int n = 0; return n.x;")).only("not_a_struct")
    assert diag.message == "request for member 'x' in something not a structure or union"
    assert where(diag) == "main.c:2:37"


def test_not_a_struct_dot_on_pointer_suggests_arrow() -> None:
    diag = build("struct P { int x; };\nint get(struct P *p) { return p.x; }\n").only(
        "not_a_struct")
    assert diag.message == "'p' is a pointer; did you mean to use '->'?"
    assert "p->x" in diag.friendly


def test_not_a_struct_arrow_on_struct_suggests_dot() -> None:
    diag = build("struct P { int x; };\nint get(struct P p) { return p->x; }\n").only(
        "not_a_struct")
    assert diag.message == "invalid type argument of '->' (have 'struct P')"
    assert "p.x" in diag.friendly


def test_no_member_with_did_you_mean() -> None:
    diag = build("struct P { int x; int count; };\n"
                 "int get(struct P p) { return p.cuont; }\n").only("no_member")
    assert diag.message == "'struct P' has no member named 'cuont'; did you mean 'count'?"
    assert where(diag) == "main.c:2:31"


def test_invalid_subscript_on_int() -> None:
    diag = build(body("int n = 0; return n[1];")).only("invalid_subscript")
    assert diag.message == "subscripted value is neither array nor pointer nor vector"
    assert where(diag) == "main.c:2:37"


def test_invalid_subscript_with_double_index() -> None:
    diag = build(body("int a[3] = {0}; double d = 1; return a[d];")).only("invalid_subscript")
    assert diag.message == "array subscript is not an integer"


def test_lvalue_required_for_assignment() -> None:
    diag = build(body("int a = 1; a + 1 = 3; return a;")).only("lvalue_required")
    assert diag.message == "lvalue required as left operand of assignment"
    assert where(diag) == "main.c:2:29"


def test_lvalue_required_for_increment_and_address() -> None:
    built = build(body("int a = 1; (a + 1)++; int *p = &5; return a + *p;"))
    messages = [d.message for d in built.diagnostics if d.code == "lvalue_required"]
    assert messages == ["lvalue required as increment operand",
                        "lvalue required as unary '&' operand"]


def test_void_value() -> None:
    diag = build("void beep(void) {}\nint main(void) { int x = beep(); return x; }\n").only(
        "void_value")
    assert diag.message == "void value not ignored as it ought to be"
    assert where(diag) == "main.c:2:26"


def test_return_value_in_void() -> None:
    diag = build("void f(void) {\n    return 1;\n}\n").only("return_value_in_void")
    assert diag.message == "'return' with a value, in function returning void"
    assert where(diag) == "main.c:2:5"


def test_return_void_call_in_void_function_is_fine() -> None:
    assert build("static void g(void) {}\nstatic void f(void) { return g(); }\n"
                 "int main(void) { f(); return 0; }\n").diagnostics == []


def test_return_missing_value() -> None:
    diag = build("int f(void);\nint f(void) {\n    return;\n}\n").only("return_missing_value")
    assert diag.message == "'return' with no value, in function returning non-void"
    assert where(diag) == "main.c:3:5"
    assert "return 0;" in diag.friendly


def test_array_size_not_const_is_reported_by_the_front_end() -> None:
    built = build(body("int n = 3; int a[n]; return 0;"))
    assert [d.code for d in built.front] == ["array_size_not_const"]


def test_incompatible_types_struct_to_int() -> None:
    diag = build("struct P { int x; };\nint f(struct P p) { int n = p; return n; }\n").only(
        "incompatible_types")
    assert diag.message == "incompatible types when initializing type 'int' using type 'struct P'"
    assert where(diag) == "main.c:2:29"
    assert "member" in diag.friendly


def test_incompatible_types_assignment_return_and_argument() -> None:
    built = build("struct P { int x; };\nstatic int take(struct P p) { return p.x; }\n"
                  "static struct P make(int v) { return v; }\n"
                  "int main(void) { struct P p = {1}; int n = 0; p = n; return take(n); }\n")
    messages = [d.message for d in built.diagnostics if d.code == "incompatible_types"]
    assert messages == [
        "incompatible types when returning type 'int' but 'struct P' was expected",
        "incompatible types when assigning to type 'struct P' from type 'int'",
        "incompatible type for argument 1 of 'take'",
    ]


def test_struct_assignment_of_same_type_is_fine() -> None:
    assert build("struct P { int x; };\nint main(void) { struct P a = {1}; struct P b = a; "
                 "a = b; return a.x; }\n").diagnostics == []


def test_call_non_function() -> None:
    diag = build(body("int speed = 3; return speed(2);")).only("call_non_function")
    assert diag.message == "called object 'speed' is not a function or function pointer"
    assert where(diag) == "main.c:2:40"


def test_break_and_continue_outside_loop() -> None:
    built = build(body("break; continue; return 0;"))
    assert [(d.code, d.message) for d in built.diagnostics] == [
        ("break_outside_loop", "break statement not within loop or switch"),
        ("break_outside_loop", "continue statement not within a loop"),
    ]
    assert where(built.diagnostics[0]) == "main.c:2:18"


def test_break_inside_switch_is_fine_but_continue_needs_a_loop() -> None:
    built = build(body("int v = 1; switch (v) { case 1: break; default: continue; } return 0;"))
    assert [d.message for d in built.diagnostics] == ["continue statement not within a loop"]


def test_case_and_default_outside_switch() -> None:
    built = build(body("case 1: ; default: ; return 0;"))
    assert [(d.code, d.message) for d in built.diagnostics] == [
        ("case_outside_switch", "case label not within a switch statement"),
        ("case_outside_switch", "'default' label not within a switch statement"),
    ]


def test_duplicate_case() -> None:
    diag = build(body("int v = 1; switch (v) { case 1: break; case 1: break; } return 0;")).only(
        "duplicate_case")
    assert diag.message == "duplicate case value"
    assert where(diag) == "main.c:2:57"
    assert diag.notes == ["previously used here"]


def test_duplicate_case_through_enum_constants() -> None:
    built = build("enum Dir { NORTH, SOUTH = 0 };\nint f(int d) { switch (d) { case NORTH: "
                  "return 1; case SOUTH: return 2; } return 0; }\n",
                  options=CompileOptions())
    assert built.codes() == ["duplicate_case"]


def test_multiple_default_labels() -> None:
    diag = build(body("int v = 1; switch (v) { default: break; default: break; } "
                      "return 0;")).only("duplicate_case")
    assert diag.message == "multiple default labels in one switch"


def test_invalid_operands_struct_plus_int() -> None:
    diag = build("struct P { int x; };\nint f(struct P p) { return p + 1; }\n").only(
        "invalid_operands")
    assert diag.message == "invalid operands to binary + (have 'struct P' and 'int')"
    assert where(diag) == "main.c:2:30"


def test_invalid_operands_pointer_times_pointer() -> None:
    diag = build(body("int a = 1; int *p = &a; int *q = &a; return p * q;")).only(
        "invalid_operands")
    assert diag.message == "invalid operands to binary * (have 'int *' and 'int *')"
    assert "Dereference" in diag.friendly


def test_invalid_operands_dereferencing_an_int() -> None:
    diag = build(body("int a = 1; return *a;")).only("invalid_operands")
    assert diag.message == "invalid type argument of unary '*' (have 'int')"


def test_pointer_difference_is_fine() -> None:
    built = build(body("int a[4] = {0}; int *p = a; int *q = &a[3]; long n = q - p; "
                       "return (int)n;"))
    assert built.diagnostics == []


def test_incomplete_type_sizeof() -> None:
    diag = build("struct Node;\nunsigned long f(void) { return sizeof(struct Node); }\n",
                 options=CompileOptions()).only("incomplete_type")
    assert diag.message == "invalid application of 'sizeof' to incomplete type 'struct Node'"
    assert where(diag) == "main.c:2:32"


def test_incomplete_type_member_access() -> None:
    diag = build("struct Node;\nint f(struct Node *n) { return n->value; }\n",
                 options=CompileOptions()).only("incomplete_type")
    assert diag.message == "invalid use of undefined type 'struct Node'"


def test_incomplete_type_variable() -> None:
    diag = build("struct Node;\nint main(void) { struct Node n; return 0; }\n",
                 options=CompileOptions()).only("incomplete_type")
    assert diag.message == "storage size of 'n' isn't known"
    assert where(diag) == "main.c:2:30"


def test_too_many_initializers_struct() -> None:
    diag = build("struct P { int x; int y; };\nstruct P p = {1, 2, 3};\n").only(
        "too_many_initializers")
    assert diag.message == "excess elements in struct initializer"
    assert where(diag) == "main.c:2:21"


def test_too_many_initializers_scalar_and_nested_array() -> None:
    built = build("int x = {1, 2};\nint g[2][2] = {{1, 2, 3}, {4}};\n")
    assert [(d.code, d.message) for d in built.diagnostics] == [
        ("too_many_initializers", "excess elements in scalar initializer"),
        ("too_many_initializers", "excess elements in array initializer"),
    ]


def test_brace_elision_fills_nested_aggregates() -> None:
    built = build("struct P { int x; double y; };\nstruct P ps[2] = {1, 2, 3, 4};\n"
                  "int g[2][2] = {1, 2, 3, 4};\n")
    assert built.diagnostics == []
    items = built.unit.decls[1].init.items
    assert isinstance(items[1], ast.ImplicitCast) and items[1].to == DOUBLE
    assert isinstance(items[3], ast.ImplicitCast) and items[3].to == DOUBLE


def test_assign_to_array() -> None:
    diag = build(body('char name[8]; name = "Ada"; return 0;')).only("assign_to_array")
    assert diag.message == "assignment to expression with array type"
    assert where(diag) == "main.c:2:32"
    assert "strcpy" in diag.friendly


def test_assign_to_const_variable() -> None:
    diag = build(body("const int limit = 3; limit = 4; return limit;")).only("assign_to_const")
    assert diag.message == "assignment of read-only variable 'limit'"
    assert where(diag) == "main.c:2:39"


def test_assign_to_const_location_and_increment() -> None:
    built = build('int main(void) { const char *s = "hi"; *s = \'x\'; const int n = 1; n++; '
                  "return 0; }\n")
    assert [d.message for d in built.diagnostics if d.code == "assign_to_const"] == [
        "assignment of read-only location '*s'", "increment of read-only variable 'n'"]


def test_assign_to_const_member() -> None:
    diag = build("struct P { int x; };\nint main(void) { const struct P p = {1}; p.x = 2; "
                 "return 0; }\n").only("assign_to_const")
    assert diag.message == "assignment of member 'x' in read-only object"


def test_const_pointer_variable_may_change_its_target() -> None:
    assert build(body("int a = 1; int *const p = &a; *p = 2; return a;")).diagnostics == []


def test_unsupported_variadic_definition() -> None:
    diag = build("int sum(int n, ...) { return n; }\n", options=CompileOptions()).only(
        "unsupported_feature")
    assert diag.message == "variadic function definitions are not supported"
    assert "'sum'" in diag.friendly and "array and a count" in diag.friendly


def test_unsupported_goto_comes_from_the_parser() -> None:
    built = build(body("goto end; return 0;"))
    assert [d.code for d in built.front] == ["unsupported_feature"]


def test_errors_do_not_cascade_from_one_bad_name() -> None:
    built = build(body("int a = missing + 1; int b = a * missing; return b;"))
    assert built.codes() == ["undeclared_identifier"]


# ============================================================================ annotations

ANNOTATED = """#include <stdio.h>
enum Dir { NORTH, EAST = 5 };
typedef struct Node { int value; struct Node *next; } Node;
static int counter;
int total;
static double half(double v) { return v / 2; }
static int sum(int *xs, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += xs[i];
    return s;
}
int main(void) {
    int a[3] = {1, 2, 3};
    char c = 'a';
    double d = 2;
    float f = 1.5f;
    Node node = {1, NULL};
    Node *p = &node;
    int (*fp)(int *, int) = sum;
    static int calls = 0;
    int r = c + 1;
    int q = r + 1;
    d = half(3);
    p->next = p;
    p->next->value = fp(a, 3) + EAST;
    printf("%d %c %f %f\\n", node.value, c, d, f);
    calls++;
    counter = total + calls + q;
    return sizeof a + sizeof(Node) > 0 ? NORTH : r;
}
"""


@pytest.fixture(scope="module")
def annotated() -> Build:
    return build(ANNOTATED)


def test_annotated_program_is_clean(annotated: Build) -> None:
    assert annotated.front == [] and annotated.diagnostics == []


def test_every_expression_has_a_ctype(annotated: Build) -> None:
    for function in nodes(annotated.unit, ast.FunctionDef):
        for expr in nodes(function, ast.Expr):
            assert ast.expression_type(expr) is not None, ast.describe(expr)
            assert expr.ctype is not None and expr.ctype is not ERROR_TYPE, ast.describe(expr)


def test_every_identifier_has_a_ref(annotated: Build) -> None:
    for ident in nodes(annotated.unit, ast.Identifier):
        assert ident.ref is not None, ident.name


def _ref(built: Build, function: str, name: str) -> ast.Identifier:
    for ident in nodes(built.function(function), ast.Identifier):
        if ident.name == name:
            return ident
    raise KeyError(name)


@pytest.mark.parametrize("function, name, kind, key", [
    ("main", "a", "local", "a"),
    ("sum", "xs", "param", "xs"),
    ("main", "total", "global", "total"),
    ("main", "counter", "unit_static", "main.c::counter"),
    ("main", "half", "function", "main.c::half"),
    ("main", "printf", "builtin", "printf"),
    ("main", "EAST", "enum_const", "5"),
])
def test_identifier_ref_kinds(annotated: Build, function: str, name: str, kind: str,
                              key: str) -> None:
    ref = _ref(annotated, function, name).ref
    assert (ref.kind, ref.key) == (kind, key)


def test_static_local_ref_and_key(annotated: Build) -> None:
    ref = _ref(annotated, "main", "calls").ref
    assert ref.kind == "static_local"
    assert ref.key == "main.c::main::calls@20:16"
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "calls")
    assert decl.storage_key == ref.key
    assert ref.key in annotated.result.symbols.static_locals


def test_enum_constant_ref_carries_value(annotated: Build) -> None:
    ref = _ref(annotated, "main", "EAST").ref
    assert ref.value == 5 and ref.ctype == INT


def test_storage_keys_on_declarations(annotated: Build) -> None:
    decls = {getattr(d, "name", None): d for d in annotated.unit.decls}
    assert decls["counter"].storage_key == "main.c::counter"
    assert decls["total"].storage_key == "total"
    assert decls["half"].storage_key == "main.c::half"
    assert decls["main"].storage_key == "main"
    assert decls["printf"].storage_key == "printf"


def test_int_plus_int_has_no_implicit_cast(annotated: Build) -> None:
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "q")
    assert isinstance(decl.init, ast.BinaryOp)
    assert not nodes(decl.init, ast.ImplicitCast)
    assert decl.init.ctype == INT


def test_char_plus_int_promotes_the_char(annotated: Build) -> None:
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "r")
    left = decl.init.left
    assert isinstance(left, ast.ImplicitCast) and left.to == INT
    assert isinstance(left.expr, ast.Identifier) and left.expr.ctype == CHAR


def test_double_from_int_initialisation_converts(annotated: Build) -> None:
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "d")
    assert isinstance(decl.init, ast.ImplicitCast) and decl.init.to == DOUBLE
    assert isinstance(decl.init.expr, ast.IntLiteral)


def test_argument_converted_to_double_parameter(annotated: Build) -> None:
    call = next(c for c in nodes(annotated.function("main"), ast.Call)
                if isinstance(c.func, ast.Identifier) and c.func.name == "half")
    arg = call.args[0]
    assert isinstance(arg, ast.ImplicitCast) and arg.to == DOUBLE and arg.ctype == DOUBLE


def test_variadic_arguments_get_default_promotions(annotated: Build) -> None:
    call = next(c for c in nodes(annotated.function("main"), ast.Call)
                if isinstance(c.func, ast.Identifier) and c.func.name == "printf")
    fmt, value, char, double, flt = call.args
    assert isinstance(fmt, ast.ImplicitCast) and fmt.to == PointerType(CHAR)
    assert isinstance(fmt.expr, ast.StringLiteral)
    assert isinstance(value, ast.Member) and value.ctype == INT
    assert isinstance(char, ast.ImplicitCast) and char.to == INT
    assert isinstance(double, ast.Identifier)
    assert isinstance(flt, ast.ImplicitCast) and flt.to == DOUBLE


def test_arrays_passed_to_functions_decay(annotated: Build) -> None:
    call = next(c for c in nodes(annotated.function("main"), ast.Call)
                if isinstance(c.func, ast.Identifier) and c.func.name == "fp")
    arg = call.args[0]
    assert isinstance(arg, ast.ImplicitCast) and arg.to == PointerType(INT)
    assert arg.expr.ctype == ArrayType(INT, 3)


def test_function_pointer_initialised_from_function_name(annotated: Build) -> None:
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "fp")
    assert isinstance(decl.init, ast.ImplicitCast)
    assert isinstance(decl.init.to, PointerType) and decl.init.to.target.is_function
    assert decl.init.expr.ctype.is_function


def test_function_pointer_call_has_return_type(annotated: Build) -> None:
    call = next(c for c in nodes(annotated.function("main"), ast.Call)
                if isinstance(c.func, ast.Identifier) and c.func.name == "fp")
    assert call.ctype == INT
    assert call.func.ref.kind == "local"


def test_typedef_struct_pointer_member_access(annotated: Build) -> None:
    members = [m for m in nodes(annotated.function("main"), ast.Member) if m.name == "next"]
    assert members and all(isinstance(m.ctype, PointerType) for m in members)
    value = next(m for m in nodes(annotated.function("main"), ast.Member) if m.name == "value"
                 and m.arrow)
    assert value.ctype == INT
    assert isinstance(value.base.ctype.target, StructType)


def test_null_initialiser_becomes_pointer_cast(annotated: Build) -> None:
    decl = next(d for d in nodes(annotated.function("main"), ast.VarDecl) if d.name == "node")
    second = decl.init.items[1]
    assert isinstance(second, ast.ImplicitCast) and second.to.is_pointer
    assert decl.init.ctype.is_struct


def test_array_name_keeps_array_type_under_sizeof(annotated: Build) -> None:
    sizeofs = nodes(annotated.function("main"), ast.SizeofExpr)
    assert sizeofs[0].expr.ctype == ArrayType(INT, 3)
    assert sizeofs[0].ctype == SIZE_T
    sized = nodes(annotated.function("main"), ast.SizeofType)[0]
    assert ast.expression_type(sized) == SIZE_T and sized.ctype.is_struct


def test_string_literal_type_includes_nul(annotated: Build) -> None:
    literal = nodes(annotated.function("main"), ast.StringLiteral)[0]
    assert literal.ctype == ArrayType(CHAR, len(literal.value) + 1)


def test_index_base_decays_to_pointer() -> None:
    built = build(body("int a[4] = {0}; return a[1];"))
    index = nodes(built.function("main"), ast.Index)[0]
    assert isinstance(index.base, ast.ImplicitCast) and index.base.to == PointerType(INT)
    assert index.ctype == INT


def test_reversed_subscript_is_normalised() -> None:
    built = build(body("int a[4] = {0}; return 2[a];"))
    index = nodes(built.function("main"), ast.Index)[0]
    assert index.base.ctype == PointerType(INT) and index.index.ctype == INT


def test_return_value_is_converted() -> None:
    built = build("double f(int x) { return x; }\n", options=CompileOptions())
    ret = nodes(built.function("f"), ast.ReturnStmt)[0]
    assert isinstance(ret.value, ast.ImplicitCast) and ret.value.to == DOUBLE


def test_usual_arithmetic_conversions_int_and_double() -> None:
    built = build(body("int i = 2; double d = 1; return (int)(i * d);"))
    mul = next(b for b in nodes(built.function("main"), ast.BinaryOp) if b.op == "*")
    assert isinstance(mul.left, ast.ImplicitCast) and mul.left.to == DOUBLE
    assert mul.ctype == DOUBLE


def test_compound_assignment_converts_value_to_common_type() -> None:
    built = build(body("double d = 1; int i = 2; d += i; i += d; return i;"))
    first, second = [a for a in nodes(built.function("main"), ast.Assignment)]
    assert isinstance(first.value, ast.ImplicitCast) and first.value.to == DOUBLE
    assert isinstance(second.value, ast.Identifier) and second.ctype == INT


def test_comparison_results_are_int_and_pointer_null_compare_casts() -> None:
    built = build(body("int a = 1; int *p = &a; return p != 0 && a < 2;"))
    compare = next(b for b in nodes(built.function("main"), ast.BinaryOp) if b.op == "!=")
    assert compare.ctype == INT
    assert isinstance(compare.right, ast.ImplicitCast) and compare.right.to == PointerType(INT)


def test_pointer_arithmetic_keeps_pointer_type() -> None:
    built = build(body("int a[4] = {0}; int *p = a + 1; return *p;"))
    plus = next(b for b in nodes(built.function("main"), ast.BinaryOp) if b.op == "+")
    assert plus.ctype == PointerType(INT)
    assert not isinstance(plus.right, ast.ImplicitCast)


def test_switch_on_char_promotes_condition() -> None:
    built = build(body("char c = 'a'; switch (c) { case 'a': return 1; } return 0;"))
    switch = nodes(built.function("main"), ast.SwitchStmt)[0]
    assert isinstance(switch.cond, ast.ImplicitCast) and switch.cond.to == INT


def test_conditional_arithmetic_branches_join() -> None:
    built = build(body("int i = 1; double d = (i > 0) ? i : 2.5; return (int)d;"))
    cond = nodes(built.function("main"), ast.Conditional)[0]
    assert cond.ctype == DOUBLE and isinstance(cond.then, ast.ImplicitCast)


def test_implicit_cast_folds_as_a_constant() -> None:
    loc = SourceLocation("main.c", 1, 1)
    cast = ast.ImplicitCast(loc, ast.IntLiteral(loc, 300, INT), CHAR)
    assert cast.ctype == CHAR
    assert eval_const_expr(cast) == 44
    assert ast.dump(cast).startswith("ImplicitCast -> char")


# ============================================================================ symbols

def test_unit_symbols_definitions_and_declarations() -> None:
    built = build("#include <stdio.h>\nextern int shared;\nint count;\nint limit = 3;\n"
                  "static int hidden;\nint helper(int x);\n"
                  "int main(void) { printf(\"%d\", helper(count + shared + hidden + limit));"
                  " return 0; }\n")
    symbols = built.result.symbols
    assert symbols.source_path == "main.c"
    assert symbols.defined["count"].tentative is True
    assert symbols.defined["limit"].tentative is False
    assert symbols.defined["hidden"].linkage == "internal"
    assert symbols.defined["hidden"].key == "main.c::hidden"
    assert symbols.defined["main"].kind == "function"
    assert symbols.declared["shared"].defined is False
    assert symbols.declared["helper"].kind == "function"
    assert "printf" in symbols.declared
    assert set(symbols.undefined_references()) == {"printf", "helper", "shared"}
    assert "hidden" not in symbols.referenced
    assert str(symbols.referenced["helper"][0]) == "main.c:7:31"


def test_tentative_definition_followed_by_real_one() -> None:
    symbols = build("int x;\nint x = 5;\nint x;\n").result.symbols
    assert symbols.defined["x"].tentative is False
    assert str(symbols.defined["x"].loc) == "main.c:2:5"


def test_two_tentative_definitions_are_fine() -> None:
    built = build("int x;\nint x;\n")
    assert built.diagnostics == [] and built.result.symbols.defined["x"].tentative


def test_static_function_symbol_and_key() -> None:
    built = build("static int helper(void) { return 1; }\nint main(void) { return helper(); }\n")
    symbol = built.result.symbols.defined["helper"]
    assert (symbol.linkage, symbol.key) == ("internal", "main.c::helper")
    assert "helper" not in built.result.symbols.referenced
    assert built.function("helper").storage_key == "main.c::helper"


def test_static_local_keys_differ_per_function() -> None:
    built = build("static int a(void) { static int n = 0; return ++n; }\n"
                  "static int b(void) { static int n = 0; return ++n; }\n"
                  "int main(void) { return a() + b(); }\n")
    assert sorted(built.result.symbols.static_locals) == [
        "main.c::a::n@1:33", "main.c::b::n@2:33"]


def test_library_function_defined_in_unit_is_a_function_ref() -> None:
    built = build("int move(int d) { return d; }\nint main(void) { return move(1); }\n",
                  options=CompileOptions())
    call = nodes(built.function("main"), ast.Call)[0]
    assert call.func.ref.kind == "function"


def test_builtin_variable_ref_kind() -> None:
    reg = make_registry()
    reg.register_variable("errno", "int errno;", lambda m: b"", header="errno.h")
    built = build("extern int errno;\nint main(void) { errno = 0; return errno; }\n", reg=reg)
    ident = nodes(built.function("main"), ast.Identifier)[0]
    assert ident.ref.kind == "builtin" and built.diagnostics == []


def test_block_scope_extern_refers_to_global() -> None:
    built = build("int g = 1;\nint main(void) { extern int g; return g; }\n")
    ident = nodes(built.function("main"), ast.Identifier)[0]
    assert ident.ref.kind == "global" and built.diagnostics == []


def test_implicitly_declared_function_is_referenced() -> None:
    built = build("int main(void) { helper(); return 0; }")
    assert "helper" in built.result.symbols.referenced


# ============================================================================ flow helper

@pytest.mark.parametrize("statements, falls", [
    ("return 1;", False),
    ("if (x) return 1; else return 2;", False),
    ("if (x) return 1;", True),
    ("while (1) { x++; }", False),
    ("while (1) { if (x) break; }", True),
    ("for (;;) { }", False),
    ("do { return 1; } while (x);", False),
    ("exit(1);", False),
    ("switch (x) { case 1: return 1; default: return 2; }", False),
    ("switch (x) { case 1: return 1; }", True),
    ("switch (x) { case 1: break; default: return 2; }", True),
    ("x++;", True),
])
def test_can_fall_through(statements: str, falls: bool) -> None:
    built = build(f"#include <stdlib.h>\nint f(int x) {{ {statements} }}\n",
                  options=CompileOptions())
    assert can_fall_through(built.function("f").body) is falls


# ============================================================================ robustness

def test_deeply_nested_expression_does_not_crash() -> None:
    expr = " + ".join(["1"] * 3000)
    built = build(f"int main(void) {{ return {expr}; }}")
    assert built.result.ok


def test_two_thousand_line_unit_checks_quickly() -> None:
    lines = ["#include <stdio.h>"]
    for index in range(185):
        lines += [
            f"static int f{index}(int a, int b) {{",
            "    int total = 0;",
            "    for (int i = 0; i < a; i++) {",
            "        if (i % 2 == 0) total += i * b;",
            "        else total -= b;",
            "    }",
            '    printf("%d\\n", total);',
            "    return total;",
            "}",
            "",
        ]
    lines.append("int main(void) {")
    lines += [f"    f{index}({index}, 2);" for index in range(185)]
    lines.append("    return 0;\n}")
    source = "\n".join(lines)
    assert source.count("\n") >= 2000
    pre = preprocess(source, "big.c", DictResolver({}, SYSTEM_HEADERS))
    parsed = parse(tokenize(pre.lines), "big.c")
    start = time.perf_counter()
    result = check(parsed.ast, make_registry(), CompileOptions(wall=True, wextra=True))
    elapsed = time.perf_counter() - start
    assert result.ok
    assert elapsed < 0.5


def test_function_type_from_implicit_declaration_is_old_style() -> None:
    built = build("int main(void) { return later(1, 2); }\nint later(int a, int b) "
                  "{ return a + b; }\n", options=CompileOptions())
    assert built.codes() == ["implicit_declaration"]
    call = nodes(built.function("main"), ast.Call)[0]
    assert isinstance(call.func.ctype, FunctionType)
