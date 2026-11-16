"""Tests for parser diagnostics: gcc wording, exact positions, friendly text, recovery."""
from __future__ import annotations

import random
import re

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import CODE_CONCEPTS, Diagnostic, Severity
from pycc.interp import astnodes as ast
from pycc.interp.c_types import INT
from pycc.interp.friendly import EXTRA_CONCEPTS, did_you_mean, friendly_text
from pycc.interp.parser import MAX_ERRORS, parse_text


def diagnostics(source: str) -> list[Diagnostic]:
    """Every diagnostic (lexer and parser) for ``source``."""
    return parse_text(source).diagnostics


def single(source: str) -> Diagnostic:
    """The one and only diagnostic of ``source``."""
    found = diagnostics(source)
    assert len(found) == 1, [d.format_gcc() for d in found]
    return found[0]


def where(diag: Diagnostic) -> str:
    """'file:line:column' of a diagnostic."""
    return str(diag.location)


# --- the headline errors, with exact file:line:column --------------------------------------

def test_missing_semicolon_after_expression_points_after_previous_token() -> None:
    diag = single("int main(void) {\n    int x;\n    x = 5\n}\n")
    assert diag.code == "missing_semicolon"
    assert diag.format_gcc() == "main.c:3:10: error: expected ';' before '}' token"
    assert "line 3" in diag.friendly
    assert diag.concept is Concept.SYNTAX


def test_missing_semicolon_after_declaration_uses_gcc_wording() -> None:
    diag = single("int main(void) {\n    int x = 5\n    return x;\n}\n")
    assert diag.format_gcc() == "main.c:2:14: error: expected ',' or ';' before 'return'"
    assert diag.code == "missing_semicolon"


def test_missing_semicolon_without_initializer_mentions_equals() -> None:
    diag = single("int count\nint main(void) { return 0; }")
    assert diag.format_gcc() == "main.c:1:10: error: expected '=', ',' or ';' before 'int'"


def test_missing_semicolon_at_end_of_input() -> None:
    diag = single("int x")
    assert diag.message == "expected '=', ',' or ';' at end of input"
    assert where(diag) == "main.c:1:6"


def test_missing_semicolon_keeps_parsing_the_next_statement() -> None:
    result = parse_text("int main(void) {\n    int x = 5\n    return x;\n}\n")
    body = result.ast.decls[0].body.items
    assert [type(item).__name__ for item in body] == ["DeclStmt", "ReturnStmt"]


def test_misspelled_keyword_gets_a_hint() -> None:
    diag = single("int main(void) {\n    retrun 0;\n}")
    assert diag.format_gcc() == "main.c:2:11: error: expected ';' before numeric constant"
    assert "Did you mean 'return'?" in diag.friendly


def test_missing_closing_brace() -> None:
    diag = single("int main(void) {\n    return 0;\n")
    assert diag.code == "unbalanced_brace"
    assert diag.format_gcc().splitlines() == [
        "main.c:2:14: error: expected '}' at end of input",
        "main.c:2:14: note: to match this '{' at line 1, column 16",
    ]
    assert "line 1" in diag.friendly


def test_missing_closing_brace_still_returns_the_function() -> None:
    result = parse_text("int main(void) {\n    if (1) {\n        return 0;\n")
    assert len(result.errors) == 1  # reported once, not once per open block
    assert isinstance(result.ast.decls[0], ast.FunctionDef)


def test_missing_closing_paren_in_condition() -> None:
    diag = single("int main(void) {\n    int x = 0;\n    if (x > 0 {\n        x = 1;\n    }\n}\n")
    assert diag.code == "unbalanced_paren"
    assert diag.format_gcc().splitlines()[0] == "main.c:3:14: error: expected ')' before '{' token"
    assert diag.notes == ["to match this '(' at line 3, column 8"]
    assert "line 3" in diag.friendly


def test_missing_closing_paren_in_call() -> None:
    diag = single('int main(void) {\n    printf("hi";\n}\n')
    assert diag.message == "expected ')' before ';' token"
    assert where(diag) == "main.c:2:16"


def test_missing_paren_recovers_as_if_inserted() -> None:
    result = parse_text("int main(void) { int x = (1 + 2; return x; }")
    assert len(result.errors) == 1
    items = result.ast.decls[0].body.items
    assert isinstance(items[0].decls[0].init, ast.BinaryOp)
    assert isinstance(items[1], ast.ReturnStmt)


def test_unknown_type_name_with_suggestion() -> None:
    diag = single("int main(void) {\n    itn x = 5;\n    return x;\n}\n")
    assert diag.code == "unknown_type"
    assert diag.format_gcc() == "main.c:2:5: error: unknown type name 'itn'; did you mean 'int'?"
    assert "Did you mean 'int'?" in diag.friendly
    assert diag.concept is Concept.VARIABLES


def test_unknown_type_is_treated_as_int_and_parsing_goes_on() -> None:
    result = parse_text("int main(void) {\n    itn x = 5;\n    return x;\n}\n")
    var = result.ast.decls[0].body.items[0].decls[0]
    assert var.name == "x" and var.ctype == INT


@pytest.mark.parametrize("source, location, suggestion", [
    ("voi main(void) { }", "main.c:1:1", "void"),
    ("int f(itn x);", "main.c:1:7", "int"),
    ("struct S { cahr c; };", "main.c:1:12", "char"),
    ("typedef int Count;\nCoutn c;", "main.c:2:1", "Count"),
    ("void f(void) { doubel *p = 0; }", "main.c:1:16", "double"),
])
def test_unknown_type_names_everywhere(source: str, location: str, suggestion: str) -> None:
    diag = single(source)
    assert diag.code == "unknown_type"
    assert where(diag) == location
    assert diag.message.endswith(f"did you mean '{suggestion}'?")


def test_unknown_type_from_a_missing_header() -> None:
    diag = single("bool ready = 1;")
    assert diag.message == "unknown type name 'bool'"
    assert "#include <stdbool.h>" in diag.friendly


def test_unknown_type_without_any_close_match() -> None:
    diag = single("Zebra z;")
    assert diag.message == "unknown type name 'Zebra'"
    assert "typedef" in diag.friendly


def test_else_without_if() -> None:
    diag = single("int main(void) {\n    int y;\n    if (y) y = 1;;\n    else y = 2;\n}\n")
    assert diag.code == "else_without_if_stmt"
    assert diag.format_gcc() == "main.c:4:5: error: 'else' without a previous 'if'"
    assert "stray ';'" in diag.friendly
    assert diag.concept is Concept.CONDITIONALS


def test_stray_character_without_cascade() -> None:
    diag = single("int main(void) {\n    int x = 5 @ 3;\n}\n")
    assert diag.format_gcc() == "main.c:2:15: error: stray '@' in program"
    assert diag.code == "stray_character"


def test_stray_backslash() -> None:
    diag = single("int main(void) {\n    int x = 1;\n    x = \\ 2;\n}\n")
    assert diag.format_gcc() == "main.c:3:9: error: stray '\\' in program"


def test_unterminated_string_is_one_error() -> None:
    diag = single('int main(void) {\n    printf("hello);\n    return 0;\n}\n')
    assert diag.code == "unterminated_string"
    assert diag.format_gcc() == 'main.c:2:12: error: missing terminating " character'
    assert "closing" in diag.friendly


def test_error_then_recovery_reports_both_errors() -> None:
    found = diagnostics("int main(void) {\n    int x = ;\n    int y = 3\n    return y;\n}\n")
    assert [d.format_gcc() for d in found] == [
        "main.c:2:13: error: expected expression before ';' token",
        "main.c:3:14: error: expected ',' or ';' before 'return'",
    ]
    assert [d.code for d in found] == ["unexpected_token", "missing_semicolon"]


def test_errors_in_two_functions_are_both_reported() -> None:
    found = diagnostics("void a(void) { x = ; }\nvoid b(void) { y = (1; }\n")
    assert [where(d) for d in found] == ["main.c:1:20", "main.c:2:22"]
    assert [d.code for d in found] == ["unexpected_token", "unbalanced_paren"]


# --- implicit int and missing return type -------------------------------------------------

def test_missing_return_type_is_a_warning_and_implies_int() -> None:
    result = parse_text("main() {\n    return 0;\n}\n")
    diag = result.diagnostics[0]
    assert len(result.diagnostics) == 1 and result.ok
    assert diag.severity is Severity.WARNING
    assert diag.code == "missing_return_type"
    assert diag.format_gcc() == "main.c:1:1: warning: return type defaults to 'int'"
    assert "int main" in diag.friendly
    main = result.ast.decls[0]
    assert isinstance(main, ast.FunctionDef) and main.ftype.ret == INT


def test_implicit_int_variable() -> None:
    diag = single("counter;")
    assert diag.code == "implicit_int" and diag.severity is Severity.WARNING
    assert diag.message == "type defaults to 'int' in declaration of 'counter'"


def test_implicit_int_warning_is_dropped_when_the_line_has_a_real_error() -> None:
    diag = single('printf("hi");\nint main(void) { return 0; }')
    assert diag.message == "expected declaration specifiers or '...' before string constant"
    assert where(diag) == "main.c:1:8"
    assert "inside a function body" in diag.friendly


# --- more gcc messages ------------------------------------------------------------------------

@pytest.mark.parametrize("source, code, message, location", [
    ("int main(void);\n{\n    return 0;\n}", "bad_declarator",
     "expected identifier or '(' before '{' token", "main.c:2:1"),
    ("int = 5;", "bad_declarator", "expected identifier or '(' before '=' token", "main.c:1:5"),
    ("}", "bad_declarator", "expected identifier or '(' before '}' token", "main.c:1:1"),
    ("struct P { int x; }\nint main(void) { return 0; }", "missing_semicolon",
     "expected ';', identifier or '(' before 'int'", "main.c:1:20"),
    ("int main(void) { foo(1, ); }", "unexpected_token",
     "expected expression before ')' token", "main.c:1:25"),
    ("int main(void) { int x = 0; if x > 0 { x = 1; } return 0; }", "expected_token",
     "expected '(' before 'x'", "main.c:1:32"),
    ("int main(void) {\n    for (int i = 0; i < 3) { }\n}", "missing_semicolon",
     "expected ';' before ')' token", "main.c:2:26"),
    ("int main(void) { int a, b, c; a = b ? c; }", "expected_token",
     "expected ':' before ';' token", "main.c:1:40"),
    ("int main(void) { struct { int f; } s; s.; }", "expected_token",
     "expected identifier before ';' token", "main.c:1:41"),
    ("int main(void) { int x; do x++; until (x); }", "expected_token",
     "expected 'while' before 'until'", "main.c:1:33"),
    ("int f(5);", "expected_token",
     "expected declaration specifiers or '...' before numeric constant", "main.c:1:7"),
    ("struct S { 5 };", "expected_token",
     "expected specifier-qualifier-list before numeric constant", "main.c:1:12"),
    ("int f(int) { return 0; }", "bad_declarator", "parameter name omitted", "main.c:1:7"),
    ("int f(...);", "expected_token", "ISO C requires a named argument before '...'",
     "main.c:1:7"),
    ("int a[3] = 5;", "bad_initializer", "invalid initializer", "main.c:1:12"),
    ("int b[] = \"ab\";", "bad_initializer",
     "array of inappropriate type initialized from string constant", "main.c:1:11"),
    ("int c[];", "bad_declarator", "array size missing in 'c'", "main.c:1:5"),
    ("void v;", "bad_declarator", "variable or field 'v' declared void", "main.c:1:6"),
    ("int a[-1];", "bad_declarator", "size of array 'a' is negative", "main.c:1:7"),
    ("int a[2][];", "bad_declarator", "array type has incomplete element type", "main.c:1:5"),
    ("int g(void)[2];", "bad_declarator", "'g' declared as function returning an array",
     "main.c:1:5"),
    ("int f(void) = 3;", "bad_initializer", "function 'f' is initialized like a variable",
     "main.c:1:13"),
    ("typedef int T = 4;", "bad_initializer", "typedef 'T' is initialized", "main.c:1:15"),
    ("int char x;", "invalid_type_combination",
     "two or more data types in declaration specifiers", "main.c:1:1"),
    ("signed unsigned x;", "invalid_type_combination",
     "both 'signed' and 'unsigned' in declaration specifiers", "main.c:1:1"),
    ("short long y;", "invalid_type_combination",
     "both 'long' and 'short' in declaration specifiers", "main.c:1:1"),
    ("long long long z;", "invalid_type_combination", "'long long long' is too long for GCC",
     "main.c:1:1"),
    ("static extern int q;", "invalid_type_combination",
     "multiple storage classes in declaration specifiers", "main.c:1:8"),
    ("int f(static int a);", "bad_declarator", "storage class 'static' is not allowed here",
     "main.c:1:7"),
    ("struct S { int a; };\nstruct S { int b; };", "redefinition",
     "redefinition of 'struct S'", "main.c:2:8"),
    ("struct S { int a; int a; };", "redefinition", "duplicate member 'a'", "main.c:1:23"),
    ("struct A { int a; }; union A u;", "redefinition", "'A' defined as wrong kind of tag",
     "main.c:1:28"),
    ("struct S { struct S inner; };", "incomplete_type", "field 'inner' has incomplete type",
     "main.c:1:21"),
    ("int n = 2;\nenum { A = n };", "enum_value_not_const",
     "enumerator value for 'A' is not an integer constant", "main.c:2:12"),
    ("void f(void) { int n = 2; int a[n]; }", "array_size_not_const",
     "size of array 'a' is not an integer constant", "main.c:1:33"),
    ("int f(int v) { switch (v) { case v: return 1; } return 0; }", "case_not_constant",
     "case label does not reduce to an integer constant", "main.c:1:34"),
    ("int f(int v) { switch (v) { case 1; return 1; } return 0; }", "expected_token",
     "expected ':' or '...' before ';' token", "main.c:1:35"),
    ("void f(int v) { switch (v) { case 1: int x = 2; break; } }", "unexpected_token",
     "a label can only be part of a statement and a declaration is not a statement",
     "main.c:1:38"),
    ("void f(void) { goto end; }", "unsupported_feature", "labels and 'goto' are not supported",
     "main.c:1:16"),
    ("struct S { int a : 3; };", "unsupported_feature", "bit-fields are not supported",
     "main.c:1:18"),
    ("int a[3] = { .x = 1 };", "unsupported_feature",
     "designated initializers are not supported", "main.c:1:14"),
    ("int *p = (int *){0};", "unsupported_feature", "compound literals are not supported",
     "main.c:1:17"),
])
def test_error_messages(source: str, code: str, message: str, location: str) -> None:
    diag = single(source)
    assert (diag.code, diag.message, where(diag)) == (code, message, location)
    assert diag.severity is Severity.ERROR


@pytest.mark.parametrize("source, code, message", [
    ("struct E {};", "empty_struct", "struct has no members"),
    ("int;", "useless_declaration", "useless type name in empty declaration"),
    ("struct S { int a }; int x;", "missing_semicolon", "no semicolon at end of struct or union"),
    ("int b[2] = {1, 2, 3};", "bad_initializer", "excess elements in array initializer"),
    ('char d[2] = "abc";', "string_literal_to_char_array",
     "initializer-string for array of 'char' is too long"),
    ("const x = 3;", "implicit_int", "type defaults to 'int' in declaration of 'x'"),
])
def test_warnings(source: str, code: str, message: str) -> None:
    diag = single(source)
    assert (diag.code, diag.message, diag.severity) == (code, message, Severity.WARNING)


def test_semicolon_after_function_header_has_a_specific_hint() -> None:
    diag = single("int main(void);\n{\n    return 0;\n}")
    assert "Remove it" in diag.friendly


def test_nested_function_means_a_missing_brace() -> None:
    found = diagnostics("int main(void) {\n    int x = 1;\nint helper(void) {\n"
                        "    return 2;\n}\n")
    assert [(d.code, where(d)) for d in found] == [
        ("nested_function", "main.c:3:5"), ("unbalanced_brace", "main.c:5:2")]
    assert found[0].message == "ISO C forbids nested functions"
    assert "'}' is probably missing" in found[0].friendly


def test_bit_field_error_does_not_add_an_empty_struct_warning() -> None:
    assert [d.code for d in diagnostics("struct S { int a : 3; };")] == ["unsupported_feature"]


# --- limits and robustness -------------------------------------------------------------------

def test_error_cap_stops_after_max_errors_with_a_note() -> None:
    result = parse_text("int main(void) {\n" + "    x = ;\n" * 40 + "}\n")
    assert len(result.errors) == MAX_ERRORS == 20
    note = result.diagnostics[-1]
    assert note.code == "too_many_errors" and note.severity is Severity.NOTE
    assert note.message == "compilation terminated due to -fmax-errors=20."
    assert isinstance(result.ast, ast.TranslationUnit)


def test_one_error_per_token() -> None:
    # 'int x = = 5' could produce several complaints about the second '='.
    assert len(parse_text("int x = = 5;").errors) == 1


@pytest.mark.parametrize("source", [
    "", "}}}{{", "(((", "int", "struct", "struct {", "enum { A = }", "int f(", "int a[",
    "if (x) else", "x ? : ;", "int main(void) { for (", "typedef", "int (*)(", "'", '"',
    "sizeof(", "int main(void) { switch (x) { case", "int x = {1, {2,", "#include <x.h>",
])
def test_garbage_never_raises_and_returns_an_ast(source: str) -> None:
    result = parse_text(source)
    assert isinstance(result.ast, ast.TranslationUnit)


def test_random_token_soup_never_raises() -> None:
    words = ["int", "x", "(", ")", "{", "}", ";", ",", "*", "=", "1", "struct", "S", "[",
             "]", "if", "else", "return", "'a'", '"s"', "?", ":", "typedef", "enum", "->",
             "sizeof", "case", "for", "while", "do", "+", "&&", "...", "char", "@"]
    rng = random.Random(1234)
    for _ in range(300):
        source = " ".join(rng.choice(words) for _ in range(rng.randint(1, 40)))
        result = parse_text(source)
        assert isinstance(result.ast, ast.TranslationUnit)
        assert len(result.errors) <= MAX_ERRORS


# --- the diagnostic contract -------------------------------------------------------------------

BROKEN_PROGRAMS = [
    "int main(void) {\n    int x = 5\n    return x;\n}\n",
    "int main(void) {\n    return 0;\n",
    "int main(void) { if (x > 0 { } }",
    "itn main(void) { }",
    "int main(void) { else; }",
    "int x = 5 @ 3;",
    'char *s = "open;',
    "main() { }",
    "int a[3] = 5; struct E {}; int;",
    "struct S { int a : 1; }; enum { A = x };",
    "int f(int v) { switch (v) { case v; } }",
]


@pytest.mark.parametrize("source", BROKEN_PROGRAMS)
def test_every_diagnostic_honours_the_contract(source: str) -> None:
    found = diagnostics(source)
    assert found
    for diag in found:
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", diag.code)
        assert diag.code in CODE_CONCEPTS or diag.code in EXTRA_CONCEPTS
        assert diag.message and diag.friendly
        assert not re.search(r"\{[a-z_]+\}", diag.friendly)  # no unfilled placeholder
        assert isinstance(diag.concept, Concept)
        assert diag.location is not None and diag.location.line >= 1
        assert diag.location.column >= 1
        assert diag.format_gcc().startswith(f"main.c:{diag.location.line}:")


def test_diagnostics_are_in_source_order() -> None:
    found = diagnostics("int x = 1 @;\nint y = ;\nint z = 'ab';")
    lines = [d.location.line for d in found]
    assert lines == sorted(lines)


# --- friendly helpers ---------------------------------------------------------------------------

def test_did_you_mean() -> None:
    assert did_you_mean("itn", ["int", "char", "long"]) == "int"
    assert did_you_mean("Tlie", ["Tile", "Row"]) == "Tile"
    assert did_you_mean("zzz", ["int", "char"]) is None
    assert did_you_mean("int", ["int"]) is None  # the name itself is never a suggestion


def test_friendly_text_fills_placeholders_and_keeps_unknown_ones() -> None:
    assert "line 7" in friendly_text("missing_semicolon", line=7)
    assert "{line}" in friendly_text("missing_semicolon")
    assert friendly_text("not_a_real_code", detail="Plain words.") == "Plain words."
