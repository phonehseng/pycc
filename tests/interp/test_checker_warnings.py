"""Tests for the checker's warnings, warning groups, -Werror and format string checks."""
from __future__ import annotations

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation
from pycc.interp.c_types import CHAR, DOUBLE, FLOAT, INT, LONG, PointerType, UINT
from pycc.interp.checker import WARNING_GROUPS, apply_warning_options, warning_enabled, warning_group
from pycc.interp.format_check import (
    PRINTF, SCANF, check_format, parse_format, suggest_conversion, type_phrase,
)
from pycc.interp.toolchain import CompileOptions
from tests.interp.test_checker import Build, build, where

WALL = CompileOptions(wall=True)
WEXTRA = CompileOptions(wextra=True)
ALL = CompileOptions(wall=True, wextra=True)


def warns(source: str, options: CompileOptions = ALL) -> Build:
    return build(source, options=options)


def main(src: str) -> str:
    """Statements inside main on line 2; line 1 includes stdio."""
    return f"#include <stdio.h>\nint main(void) {{ {src} }}\n"


# ============================================================================ groups

def test_warning_groups_match_the_decisions() -> None:
    assert WARNING_GROUPS["default"] == {
        "implicit_declaration", "incompatible_pointer", "int_from_pointer", "pointer_from_int",
        "integer_overflow_constant", "string_literal_to_char_array", "address_compare"}
    assert WARNING_GROUPS["wall"] == {
        "unused_variable", "control_reaches_end", "assignment_as_condition",
        "uninitialized_use", "array_bounds", "format_mismatch", "unused_value",
        "pointer_int_comparison", "implicit_int", "missing_return_type"}
    assert WARNING_GROUPS["wextra"] == {"sign_compare", "missing_prototype"}


@pytest.mark.parametrize("code, group", [
    ("unused_variable", "wall"), ("sign_compare", "wextra"), ("pointer_from_int", "default"),
    ("macro_redefined", "default"),
])
def test_warning_group_lookup(code: str, group: str) -> None:
    assert warning_group(code) == group


def test_warning_enabled_follows_options() -> None:
    assert warning_enabled("pointer_from_int", CompileOptions())
    assert not warning_enabled("unused_variable", CompileOptions())
    assert warning_enabled("unused_variable", WALL)
    assert not warning_enabled("sign_compare", WALL)
    assert warning_enabled("sign_compare", WEXTRA)


def test_wall_warnings_need_wall() -> None:
    source = main("int unused; return 0;")
    assert warns(source, CompileOptions()).diagnostics == []
    assert warns(source, WALL).only("unused_variable").severity is Severity.WARNING


def test_default_warnings_are_always_on() -> None:
    found = warns(main("int *p = 5; return *p;"), CompileOptions()).diagnostics
    assert [d.code for d in found] == ["pointer_from_int"]


def test_wextra_warnings_need_wextra() -> None:
    source = main("unsigned u = 3; int i = -1; return i < u;")
    assert "sign_compare" not in warns(source, WALL).codes()
    assert "sign_compare" in warns(source, WEXTRA).codes()


def test_werror_turns_warnings_into_errors() -> None:
    options = CompileOptions(wall=True, werror=True)
    built = warns(main("int unused; return 0;"), options)
    diag = built.only("unused_variable")
    assert diag.severity is Severity.ERROR
    assert diag.message == "unused variable 'unused' [-Werror]"
    assert not built.result.ok


def test_werror_leaves_disabled_warnings_out() -> None:
    assert warns(main("int unused; return 0;"), CompileOptions(werror=True)).diagnostics == []


def test_apply_warning_options_filters_front_end_warnings() -> None:
    loc = SourceLocation("main.c", 1, 1)
    implicit = Diagnostic(Severity.WARNING, "implicit_int", "type defaults to 'int'", "x", loc,
                          Concept.VARIABLES, Phase.PARSE)
    other = Diagnostic(Severity.WARNING, "macro_redefined", "'X' redefined", "x", loc,
                       Concept.PREPROCESSOR, Phase.PREPROCESS)
    error = Diagnostic(Severity.ERROR, "missing_semicolon", "expected ';'", "x", loc,
                       Concept.SYNTAX, Phase.PARSE)
    assert apply_warning_options([implicit, other, error], None) == [other, error]
    kept = apply_warning_options([implicit], CompileOptions(wall=True, werror=True))
    assert kept[0].severity is Severity.ERROR and kept[0].message.endswith(" [-Werror]")
    assert implicit.severity is Severity.WARNING   # the original is untouched


# ============================================================================ each warning

def test_unused_variable() -> None:
    diag = warns(main("int spare = 1; return 0;")).only("unused_variable")
    assert diag.message == "unused variable 'spare'"
    assert where(diag) == "main.c:2:22"
    assert "never used" in diag.friendly


def test_used_variables_and_parameters_are_not_reported() -> None:
    built = warns("static int f(int unused_param) { int x; x = 3; return x; }\n"
                  "int main(void) { return f(1); }\n")
    assert built.diagnostics == []


def test_unused_static_local() -> None:
    assert warns(main("static int calls; return 0;")).codes() == ["unused_variable"]


def test_control_reaches_end_points_at_closing_brace() -> None:
    source = "int sign(int x);\nint sign(int x) {\n    if (x > 0)\n        return 1;\n}\n"
    diag = warns(source).only("control_reaches_end")
    assert diag.message == "control reaches end of non-void function"
    assert where(diag) == "main.c:5:1"
    assert "'sign'" in diag.friendly


@pytest.mark.parametrize("body_text", [
    "if (x) return 1; else return 0;",
    "while (1) { }",
    "for (;;) { if (x) return 1; }",
    "exit(1);",
    "switch (x) { case 0: return 1; default: return 2; }",
    "do { return 1; } while (0);",
])
def test_control_reaches_end_not_reported_when_every_path_stops(body_text: str) -> None:
    source = f"#include <stdlib.h>\nint f(int x);\nint f(int x) {{ {body_text} }}\n"
    assert "control_reaches_end" not in warns(source).codes()


def test_main_may_fall_off_the_end() -> None:
    assert warns("int main(void) { }\n").diagnostics == []


def test_void_main_warns() -> None:
    diag = warns("void main(void) { }\n").only("missing_return_type")
    assert diag.message == "return type of 'main' is not 'int'"


def test_implicit_return_type_warns_through_the_front_end() -> None:
    built = warns("main(void) { return 0; }\n")
    assert [d.code for d in apply_warning_options(built.front, WALL)] == ["missing_return_type"]
    assert apply_warning_options(built.front, CompileOptions()) == []


def test_implicit_int_warns_through_the_front_end() -> None:
    built = warns("static x = 3;\nint main(void) { return x; }\n")
    assert [d.code for d in apply_warning_options(built.front, WALL)] == ["implicit_int"]


def test_assignment_as_condition() -> None:
    diag = warns(main("int x = 0; if (x = 5) return 1; return 0;")).only(
        "assignment_as_condition")
    assert diag.message == "suggest parentheses around assignment used as truth value"
    assert where(diag) == "main.c:2:33"
    assert "'x == ...'" in diag.friendly


def test_assignment_as_condition_silenced_by_parentheses() -> None:
    built = warns(main("int x = 0; while ((x = x + 1)) { if (x > 3) break; } return 0;"))
    assert "assignment_as_condition" not in built.codes()


def test_pointer_int_comparison() -> None:
    diag = warns(main("int a = 1; int *p = &a; return p == 5;")).only("pointer_int_comparison")
    assert diag.message == "comparison between pointer and integer"
    assert where(diag) == "main.c:2:51"


def test_comparison_with_null_is_fine() -> None:
    assert warns(main("int a = 1; int *p = &a; return p == NULL || p != 0;")).diagnostics == []


def test_int_from_pointer() -> None:
    diag = warns(main("int a = 1; int n = &a; return n;")).only("int_from_pointer")
    assert diag.message == ("initialization of 'int' from 'int *' makes integer from pointer "
                            "without a cast")
    assert where(diag) == "main.c:2:37"
    assert "'*'" in diag.friendly


def test_int_from_string_literal_hint() -> None:
    diag = warns(main('char c = "x"; return c;')).only("int_from_pointer")
    assert "single quotes" in diag.friendly


def test_pointer_from_int_in_assignment_and_argument() -> None:
    built = warns("static int deref(int *p) { return *p; }\n"
                  "int main(void) { int *p; p = 7; return deref(42); }\n")
    messages = [d.message for d in built.diagnostics if d.code == "pointer_from_int"]
    assert messages == [
        "assignment to 'int *' from 'int' makes pointer from integer without a cast",
        "passing argument 1 of 'deref' makes pointer from integer without a cast",
    ]


def test_pointer_from_int_on_return() -> None:
    diag = warns("static int *get(void) { return 1; }\nint main(void) { return *get(); }\n").only(
        "pointer_from_int")
    assert diag.message == ("returning 'int' from a function with return type 'int *' makes "
                            "pointer from integer without a cast")


def test_null_pointer_constants_are_fine() -> None:
    assert warns(main("int *p = 0; char *s = NULL; p = 0; return p == s;")).codes() == [
        "incompatible_pointer"]


def test_incompatible_pointer() -> None:
    diag = warns(main("double d = 1; int *p = &d; return *p;")).only("incompatible_pointer")
    assert diag.message == "initialization of 'int *' from incompatible pointer type 'double *'"
    assert where(diag) == "main.c:2:41"


def test_incompatible_pointer_argument() -> None:
    diag = warns("static int first(int *xs) { return xs[0]; }\n"
                 "int main(void) { char s[4] = \"abc\"; return first(s); }\n").only(
        "incompatible_pointer")
    assert diag.message == "passing argument 1 of 'first' from incompatible pointer type"


def test_void_pointer_converts_silently() -> None:
    built = warns("#include <stdlib.h>\nint main(void) { int *p = malloc(4); free(p); "
                  "return 0; }\n")
    assert built.diagnostics == []


def test_char_signedness_difference_is_not_reported() -> None:
    assert warns(main('unsigned char *u = "hi"; return u[0];')).diagnostics == []


def test_uninitialized_use() -> None:
    diag = warns(main("int total; total += 1; return total;")).only("uninitialized_use")
    assert diag.message == "'total' is used uninitialized"
    assert where(diag) == "main.c:2:29"
    assert "int total = 0;" in diag.friendly


@pytest.mark.parametrize("statements", [
    "int x; x = 1; return x;",
    'int x; scanf("%d", &x); return x;',
    "int x = 0; return x;",
    "int a[3]; a[0] = 1; return a[0];",
    "int x; return sizeof x;",
    "static int x; return x;",
])
def test_uninitialized_use_not_reported(statements: str) -> None:
    assert "uninitialized_use" not in warns(main(statements)).codes()


def test_uninitialized_use_reported_once() -> None:
    built = warns(main("int x; int y = x + x; return y + x;"))
    assert built.codes().count("uninitialized_use") == 1


def test_array_bounds() -> None:
    diag = warns(main("int field[8]; field[8] = 1; return field[0];")).only("array_bounds")
    assert diag.message == "array subscript 8 is above array bounds of 'int[8]'"
    assert where(diag) == "main.c:2:37"
    assert "numbered 0 to 7" in diag.friendly


def test_array_bounds_negative_and_address_of_end() -> None:
    built = warns(main("int a[4] = {0}; int *end = &a[4]; return a[-1] + (end != 0);"))
    messages = [d.message for d in built.diagnostics if d.code == "array_bounds"]
    assert messages == ["array subscript -1 is below array bounds of 'int[4]'"]


def test_array_bounds_through_enum_constant() -> None:
    built = warns("enum { SIZE = 3 };\nint main(void) { int a[SIZE] = {0}; return a[SIZE]; }\n")
    assert built.codes() == ["array_bounds"]


@pytest.mark.parametrize("condition", ['crop == "wheat"', '"wheat" != crop',
                                       'name == "Ada"'])
def test_comparing_with_a_string_literal_warns_by_default(condition: str) -> None:
    # gcc's -Waddress, always on, concept STRINGS.
    source = main(f'char crop[8] = "oat"; char *name = crop; if ({condition}) return 1; '
                  'return 0;')
    diag = warns(source, CompileOptions()).only("address_compare")
    assert diag.message == "comparison with string literal results in unspecified behavior"
    assert diag.severity is Severity.WARNING
    assert diag.concept is Concept.STRINGS
    assert diag.friendly.startswith("Use strcmp to compare text")


def test_string_literal_compare_is_not_flagged_for_other_comparisons() -> None:
    built = warns(main('char *name = "Ada"; if (name == NULL) return 1; '
                       'if (name[0] == \'A\') return 2; return 0;'), CompileOptions())
    assert "address_compare" not in built.codes()


def test_format_mismatch_float_with_int() -> None:
    diag = warns(main('printf("%f\\n", 3); return 0;')).only("format_mismatch")
    assert diag.message == ("format '%f' expects argument of type 'double', but argument 2 "
                            "has type 'int'")
    assert where(diag) == "main.c:2:26"
    assert "Use %d" in diag.friendly


def test_format_mismatch_int_with_pointer_and_string_with_int() -> None:
    built = warns(main('char *name = "Ada"; int n = 3; printf("%d %s", name, n); return 0;'))
    messages = [d.message for d in built.diagnostics if d.code == "format_mismatch"]
    assert messages == [
        "format '%d' expects argument of type 'int', but argument 2 has type 'char *'",
        "format '%s' expects argument of type 'char *', but argument 3 has type 'int'",
    ]


def test_format_mismatch_too_few_and_too_many_arguments() -> None:
    built = warns(main('printf("%d %d\\n", 1); printf("%d\\n", 1, 2); return 0;'))
    messages = [d.message for d in built.diagnostics if d.code == "format_mismatch"]
    assert messages == ["format '%d' expects a matching 'int' argument",
                        "too many arguments for format"]


def test_format_scanf_without_ampersand() -> None:
    diag = warns(main('int age = 0; scanf("%d", age); return age;')).only("format_mismatch")
    assert diag.message == ("format '%d' expects argument of type 'int *', but argument 2 has "
                            "type 'int'")
    assert "'&age'" in diag.friendly


def test_format_ok_cases() -> None:
    source = main('char c = \'a\'; float f = 1; double d = 2; long l = 3; char s[8] = "x";'
                  ' int n; printf("%c %f %f %ld %s %p %5.2f %%\\n", c, f, d, l, s, s, d);'
                  ' scanf("%d %lf %s", &n, &d, s); return n;')
    assert warns(source).diagnostics == []


def test_format_string_in_variable_is_not_checked() -> None:
    assert warns(main('const char *fmt = "%d"; printf(fmt, "x"); return 0;')).diagnostics == []


def test_unused_value() -> None:
    diag = warns(main("int x = 1; x == 5; return x;")).only("unused_value")
    assert diag.message == "statement with no effect"
    assert where(diag) == "main.c:2:29"
    assert "'='" in diag.friendly


@pytest.mark.parametrize("statement", ["x;", "x + 1;", "-x;", "5;"])
def test_unused_value_variants(statement: str) -> None:
    assert warns(main(f"int x = 1; {statement} return x;")).codes() == ["unused_value"]


@pytest.mark.parametrize("statement", ["x++;", "x = 2;", "(void)x;", 'puts("hi");', "x += 1;"])
def test_statements_with_effects_are_quiet(statement: str) -> None:
    assert warns(main(f"int x = 1; {statement} return x;")).diagnostics == []


def test_sign_compare() -> None:
    diag = warns(main("unsigned u = 3; int i = -1; return i < u;"), WEXTRA).only("sign_compare")
    assert diag.message == ("comparison of integer expressions of different signedness: 'int' "
                            "and 'unsigned int'")
    assert where(diag) == "main.c:2:55"


def test_sign_compare_not_for_nonnegative_constants_or_promoted_types() -> None:
    built = warns(main("unsigned u = 3; unsigned char c = 1; int i = 2; "
                       "return (u < 10) + (c < i);"), WEXTRA)
    assert built.diagnostics == []


def test_missing_prototype() -> None:
    diag = warns("int helper(void) { return 1; }\nint main(void) { return helper(); }\n",
                 WEXTRA).only("missing_prototype")
    assert diag.message == "no previous prototype for 'helper'"
    assert where(diag) == "main.c:1:5"
    assert "int helper(void);" in diag.friendly


def test_missing_prototype_skips_static_main_and_prototyped() -> None:
    source = ("int shown(int x);\nint shown(int x) { return x; }\n"
              "static int hidden(void) { return 2; }\nint main(void) { return shown(1) + "
              "hidden(); }\n")
    assert warns(source, WEXTRA).diagnostics == []


def test_string_literal_too_long_for_nested_char_array() -> None:
    diag = warns('char names[2][3] = {"ab", "abcd"};\n').only("string_literal_to_char_array")
    assert diag.message == "initializer-string for array of 'char' is too long"
    assert where(diag) == "main.c:1:27"


def test_string_literal_too_long_at_top_level_comes_from_the_parser() -> None:
    built = warns('char s[3] = "abcd";\n')
    assert [d.code for d in built.front] == ["string_literal_to_char_array"]
    assert built.diagnostics == []


def test_char_pointer_from_string_is_fine() -> None:
    assert warns(main('char *s = "x"; return s[0];')).diagnostics == []


def test_integer_overflow_constant() -> None:
    diag = warns(main("char c = 300; return c;"), CompileOptions()).only(
        "integer_overflow_constant")
    assert diag.message == ("overflow in conversion from 'int' to 'char' changes value from "
                            "'300' to '44'")
    assert where(diag) == "main.c:2:27"
    assert "-128 to 127" in diag.friendly


def test_integer_overflow_constant_unsigned() -> None:
    diag = warns(main("unsigned char b = -1; return b;")).only("integer_overflow_constant")
    assert diag.message == ("unsigned conversion from 'int' to 'unsigned char' changes value "
                            "from '-1' to '255'")


@pytest.mark.parametrize("decl", ["unsigned u = -1;", "char c = 127;", "unsigned char b = 255;",
                                  "_Bool flag = 5;", "short s = 'a' + 1;"])
def test_integer_constants_that_fit_are_quiet(decl: str) -> None:
    assert warns(main(f"{decl} return 0;"), CompileOptions()).diagnostics == []


def test_integer_overflow_in_argument() -> None:
    built = warns("static int show(char c) { return c; }\nint main(void) { return show(1000); }\n")
    assert built.codes() == ["integer_overflow_constant"]


def test_incompatible_pointer_in_conditional() -> None:
    built = warns(main("int a = 1; double d = 2; int *p = &a; double *q = &d; "
                       "return *(a ? p : q);"))
    assert "incompatible_pointer" in built.codes()


def test_warnings_carry_their_concepts() -> None:
    diag = warns(main("int unused; return 0;")).only("unused_variable")
    assert diag.concept is Concept.VARIABLES and diag.phase is Phase.CHECK


# ============================================================================ format_check

def test_parse_format_reads_conversions() -> None:
    parsed = parse_format("x=%-5d y=%.2f %% %lu %*d", PRINTF)
    assert [c.text for c in parsed.conversions] == ["%-5d", "%.2f", "%lu", "%*d"]
    assert parsed.conversions[2].length == "l"
    assert parsed.conversions[3].star_width
    assert parsed.conversions[0].offset == 2
    assert parsed.problems == []


def test_parse_format_scanf_suppression_and_scanset() -> None:
    parsed = parse_format("%*d %[^\n] %3s", SCANF)
    assert [(c.char, c.suppressed) for c in parsed.conversions] == [
        ("d", True), ("[", False), ("s", False)]


def test_parse_format_problems() -> None:
    assert parse_format("%y", PRINTF).problems[0][1] == (
        "unknown conversion type character 'y' in format")
    assert parse_format("50%", PRINTF).problems[0][1] == "spurious trailing '%' in format"


def test_check_format_star_width_needs_int() -> None:
    issues = check_format(PRINTF, "%*d", [DOUBLE, INT], 2)
    assert issues[0].message == ("field width specifier '*' expects argument of type 'int', "
                                 "but argument 2 has type 'double'")


def test_check_format_suppressed_scanf_takes_no_argument() -> None:
    assert check_format(SCANF, "%*d %d", [PointerType(INT)], 2) == []


def test_check_format_long_and_size_lengths() -> None:
    assert check_format(PRINTF, "%ld %zu", [LONG, LONG], 2) == []
    issue = check_format(PRINTF, "%ld", [INT], 2)[0]
    assert issue.message == ("format '%ld' expects argument of type 'long int', but argument 2 "
                             "has type 'int'")


def test_check_format_scanf_float_versus_double() -> None:
    assert check_format(SCANF, "%f %lf", [PointerType(FLOAT), PointerType(DOUBLE)], 2) == []
    assert check_format(SCANF, "%f", [PointerType(DOUBLE)], 2)[0].message == (
        "format '%f' expects argument of type 'float *', but argument 2 has type 'double *'")


def test_check_format_reports_argument_index() -> None:
    issue = check_format(PRINTF, "%d %s", [INT, INT], 2)[0]
    assert (issue.argument, issue.offset) == (1, 3)


@pytest.mark.parametrize("ctype, expected", [
    (INT, "%d"), (UINT, "%u"), (LONG, "%ld"), (DOUBLE, "%f"), (PointerType(CHAR), "%s"),
    (PointerType(INT), "%p"),
])
def test_suggest_conversion(ctype, expected: str) -> None:
    assert suggest_conversion(ctype) == expected


def test_type_phrase_articles() -> None:
    assert type_phrase("int") == "an 'int'"
    assert type_phrase("double") == "a 'double'"
    assert type_phrase("unsigned int", capital=True) == "An 'unsigned int'"
