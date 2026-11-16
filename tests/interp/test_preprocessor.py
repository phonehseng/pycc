"""Tests for the C preprocessor (interp/preprocessor.py)."""
from __future__ import annotations

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Phase, Severity
from pycc.interp.preprocessor import (
    FIXED_DATE,
    MAX_INCLUDE_DEPTH,
    DictResolver,
    Macro,
    PreprocessResult,
    preprocess,
)
from pycc.interp.source import PPLine

MAIN = "/home/user/farm/main.c"
SRC_MAIN = "/home/user/farm/src/main.c"


def run(text: str, files: dict[str, str] | None = None, system: dict[str, str] | None = None,
        include_dirs: list[str] | None = None, defines: dict[str, str | None] | None = None,
        filename: str = MAIN) -> PreprocessResult:
    """Preprocess ``text`` as ``filename`` with an in-memory resolver."""
    resolver = DictResolver(files or {}, system or {}, include_dirs or [])
    return preprocess(text, filename, resolver, defines)


def texts(result: PreprocessResult) -> list[str]:
    return [line.text.strip() for line in result.lines]


def codes(result: PreprocessResult) -> list[str]:
    return [diagnostic.code for diagnostic in result.diagnostics]


def only(result: PreprocessResult, code: str):
    found = [d for d in result.diagnostics if d.code == code]
    assert len(found) == 1, result.diagnostics
    return found[0]


# --- plain text, splicing and comments ---------------------------------------------------

def test_plain_lines_keep_text_and_numbers() -> None:
    result = run("int a;\n\nint b;\n")
    assert result.lines == [PPLine("int a;", MAIN, 1), PPLine("int b;", MAIN, 3)]
    assert result.ok


def test_columns_preserved_when_no_macro_used() -> None:
    result = run("    int  x = 1;")
    assert result.lines[0].text == "    int  x = 1;"


def test_backslash_continuation_joins_and_keeps_later_numbers() -> None:
    result = run("int a = 1 + \\\n2;\nint b;")
    assert result.lines == [PPLine("int a = 1 + 2;", MAIN, 1), PPLine("int b;", MAIN, 3)]


def test_backslash_continuation_in_define() -> None:
    result = run("#define SUM(a, b) \\\n  ((a) + (b))\nint x = SUM(1, 2);")
    assert texts(result) == ["int x = ((1) + (2));"]
    assert result.lines[0].line == 3


def test_line_comment_removed() -> None:
    assert texts(run("int a; // the answer")) == ["int a;"]


def test_block_comment_replaced_by_one_space() -> None:
    assert run("int/*x*/a;").lines[0].text == "int a;"


def test_multiline_block_comment_keeps_later_line_numbers() -> None:
    result = run("int a; /* starts\n still\n ends */ int b;\nint c;")
    assert [(line.text.strip(), line.line) for line in result.lines] == [
        ("int a;", 1), ("int b;", 3), ("int c;", 4),
    ]


def test_comment_markers_inside_strings_untouched() -> None:
    source = 'char *u = "http://x/*y*/"; char c = \'/\';'
    assert run(source).lines[0].text == source


def test_escaped_quote_inside_string() -> None:
    source = 'char *s = "a\\"//b"; // gone'
    assert texts(run(source)) == ['char *s = "a\\"//b";']


def test_unterminated_comment_reported_at_its_start() -> None:
    result = run("int a;\n/* never\nclosed\n")
    diagnostic = only(result, "unterminated_comment")
    assert diagnostic.location.line == 2
    assert diagnostic.message == "unterminated comment"
    assert texts(result) == ["int a;"]


def test_comment_inside_directive_spanning_lines() -> None:
    result = run("#define N 3 /* a\n b */ + 1\nint x = N;")
    assert texts(result) == ["int x = 3 + 1;"]


# --- object-like and function-like macros ------------------------------------------------

def test_object_like_macro() -> None:
    assert texts(run("#define WIDTH 10\nint w = WIDTH;")) == ["int w = 10;"]


def test_macro_not_expanded_inside_string_or_char() -> None:
    result = run("#define N 5\nchar *s = \"N\"; char c = 'N'; int n = N;")
    assert texts(result) == ["char *s = \"N\"; char c = 'N'; int n = 5;"]


def test_macro_name_part_of_longer_identifier_untouched() -> None:
    assert texts(run("#define N 5\nint NN = N;")) == ["int NN = 5;"]


def test_function_like_macro() -> None:
    assert texts(run("#define SQ(x) ((x)*(x))\nint a = SQ(3);")) == ["int a = ((3)*(3));"]


def test_nested_macro_calls() -> None:
    result = run("#define SQ(x) ((x)*(x))\n#define TWICE(y) (2*(y))\nint a = TWICE(SQ(SQ(2)));")
    assert texts(result) == ["int a = (2*(((((2)*(2)))*(((2)*(2))))));"]


def test_macro_body_uses_other_macro_rescanned() -> None:
    result = run("#define SIZE (ROWS * COLS)\n#define ROWS 3\n#define COLS 4\nint n = SIZE;")
    assert texts(result) == ["int n = (3 * 4);"]


def test_args_with_commas_inside_parentheses() -> None:
    result = run("#define FIRST(a, b) (a)\nint x = FIRST(f(1, 2), 3);")
    assert texts(result) == ["int x = (f(1, 2));"]


def test_multiline_macro_invocation() -> None:
    result = run("#define ADD(a, b) ((a) + (b))\nint x = ADD(1,\n          2);\nint y;")
    assert texts(result) == ["int x = ((1) + (2));", "int y;"]
    assert [line.line for line in result.lines] == [2, 4]


def test_macro_name_at_line_end_then_paren_on_next_line() -> None:
    result = run("#define F(x) [x]\nint a = F\n(7);")
    assert texts(result) == ["int a = [7];"]


def test_function_like_name_without_call_stays_word() -> None:
    result = run("#define F(x) [x]\nint F;\nint (*p) = F\n;")
    assert texts(result) == ["int F;", "int (*p) = F", ";"]


def test_recursion_guard_self_reference() -> None:
    assert texts(run("#define X X+1\nint a = X;")) == ["int a = X+1;"]


def test_recursion_guard_mutual() -> None:
    assert texts(run("#define A B\n#define B A\nint v = A;")) == ["int v = A;"]


def test_recursive_function_like_macro_not_reexpanded() -> None:
    assert texts(run("#define f(x) f(x + 1)\nint v = f(2);")) == ["int v = f(2 + 1);"]


def test_expansion_does_not_paste_tokens() -> None:
    assert texts(run("#define NEG -1\nint a = -NEG;")) == ["int a = - -1;"]


def test_zero_parameter_macro() -> None:
    assert texts(run("#define NOW() 42\nint t = NOW();")) == ["int t = 42;"]


def test_macro_args_mismatch_too_many() -> None:
    result = run("#define SQ(x) ((x)*(x))\nint a = SQ(1, 2);")
    diagnostic = only(result, "macro_args_mismatch")
    assert diagnostic.message == 'macro "SQ" passed 2 arguments, but takes just 1'
    assert diagnostic.location.line == 2
    assert diagnostic.concept == Concept.PREPROCESSOR


def test_macro_args_mismatch_too_few() -> None:
    result = run("#define ADD(a, b) a+b\nint a = ADD(1);")
    assert only(result, "macro_args_mismatch").message == 'macro "ADD" requires 2 arguments, but only 1 given'


def test_unterminated_macro_call_at_end_of_file() -> None:
    result = run("#define F(x) x\nint a = F(1,\n")
    assert "unterminated argument list" in only(result, "macro_args_mismatch").message


def test_undef_forgets_macro() -> None:
    result = run("#define N 1\nint a = N;\n#undef N\nint b = N;")
    assert texts(result) == ["int a = 1;", "int b = N;"]
    assert "N" not in result.macros


def test_macros_reported_in_result() -> None:
    result = run("#define SQ(x) ((x) * (x))")
    assert result.macros["SQ"].params == ["x"]
    assert result.macros["SQ"].body == "((x) * (x))"
    assert result.macros["SQ"].line == 1
    assert isinstance(result.macros["__PYCC__"], Macro)


def test_macro_redefined_differently_warns() -> None:
    result = run("#define N 1\n#define N 2\nint a = N;")
    diagnostic = only(result, "macro_redefined")
    assert diagnostic.severity == Severity.WARNING
    assert diagnostic.location.line == 2
    assert texts(result) == ["int a = 2;"]
    assert result.ok


def test_identical_redefinition_is_silent() -> None:
    assert run("#define N  (1 + 2)\n#define N (1   +   2)").diagnostics == []


def test_stringizing_rejected() -> None:
    result = run("#define S(x) #x")
    assert "not supported" in only(result, "invalid_directive").message


def test_token_pasting_rejected() -> None:
    assert "##" in only(run("#define CAT(a, b) a ## b"), "invalid_directive").message


def test_variadic_rejected() -> None:
    assert "variadic" in only(run("#define LOG(...) 0"), "invalid_directive").message


def test_define_without_name() -> None:
    assert only(run("#define"), "invalid_directive").message == "no macro name given in #define directive"


def test_command_line_defines() -> None:
    result = run("int a = DEBUG; int b = LEVEL; int c = SQ(2);",
                 defines={"DEBUG": None, "LEVEL": "3", "SQ(x)": "((x)*(x))"})
    assert texts(result) == ["int a = 1; int b = 3; int c = ((2)*(2));"]


# --- built-in macros ---------------------------------------------------------------------

def test_line_and_file_builtins() -> None:
    result = run("int l = __LINE__;\nchar *f = __FILE__;")
    assert texts(result) == ["int l = 1;", f'char *f = "{MAIN}";']


def test_line_and_file_inside_included_file() -> None:
    files = {"/home/user/farm/where.h": "\n\nint hl = __LINE__; char *hf = __FILE__;"}
    result = run('#include "where.h"\nint ml = __LINE__;', files)
    assert texts(result) == ['int hl = 3; char *hf = "/home/user/farm/where.h";', "int ml = 2;"]


def test_pycc_date_time() -> None:
    result = run("int v = __PYCC__;\nchar *d = __DATE__; char *t = __TIME__;")
    assert texts(result)[0] == "int v = 1;"
    assert FIXED_DATE in texts(result)[1]


def test_line_macro_through_another_macro() -> None:
    assert texts(run("#define HERE __LINE__\n\nint h = HERE;")) == ["int h = 3;"]


# --- conditionals ------------------------------------------------------------------------

def test_ifdef_and_ifndef() -> None:
    source = "#define A\n#ifdef A\nyes1\n#endif\n#ifndef A\nno1\n#endif\n#ifdef B\nno2\n#else\nyes2\n#endif"
    assert texts(run(source)) == ["yes1", "yes2"]


def test_if_elif_else_chain() -> None:
    source = ("#define LEVEL 2\n#if LEVEL == 1\none\n#elif LEVEL == 2\ntwo\n"
              "#elif LEVEL == 2\nagain\n#else\nother\n#endif")
    assert texts(run(source)) == ["two"]


def test_if_defined_forms() -> None:
    source = "#define A 0\n#if defined(A) && !defined B\nok\n#endif\n#if defined A || defined(C)\nok2\n#endif"
    assert texts(run(source)) == ["ok", "ok2"]


def test_if_arithmetic_and_operators() -> None:
    cases = {
        "1 + 2 * 3 == 7": True, "(1 + 2) * 3 == 9": True, "0x10 == 16": True,
        "'A' == 65": True, "'\\n' == 10": True, "7 / 2 == 3": True, "-7 / 2 == -3": True,
        "-7 % 2 == -1": True, "1 << 4 == 16": True, "~0 == -1": True, "(5 & 3) == 1": True,
        "(5 | 3) == 7": True, "(5 ^ 3) == 6": True, "3 > 2 && 2 >= 2 && 1 < 2 && 1 <= 1": True,
        "1 ? 0 : 1": False, "0 ? 0 : 5": True, "!1": False, "010 == 8": True,
        "UNKNOWN_NAME": False, "1 != 1": False, "10L == 10": True,
    }
    for expression, expected in cases.items():
        result = run(f"#if {expression}\nyes\n#endif")
        assert result.ok, (expression, result.diagnostics)
        assert texts(result) == (["yes"] if expected else []), expression


def test_if_uses_macro_values() -> None:
    assert texts(run("#define W 4\n#define H 3\n#if W * H > 10\nbig\n#endif")) == ["big"]


def test_if_division_by_zero_is_diagnostic() -> None:
    result = run("#if 1 / 0\nx\n#endif")
    diagnostic = only(result, "division_by_zero")
    assert diagnostic.phase == Phase.PREPROCESS
    assert diagnostic.location.line == 1
    assert texts(result) == []


def test_if_short_circuit_skips_division_by_zero() -> None:
    result = run("#if 0 && 1 / 0\nx\n#elif 1 || 1 % 0\ny\n#endif")
    assert result.ok
    assert texts(result) == ["y"]


def test_if_bad_expression() -> None:
    assert only(run("#if 1 +\n#endif"), "invalid_directive").location.line == 1
    assert "no expression" in only(run("#if\n#endif"), "invalid_directive").message


def test_nested_conditionals_in_skipped_block() -> None:
    source = "#if 0\n#if 1\nhidden\n#else\nhidden2\n#endif\n#bogus\n#endif\nshown"
    result = run(source)
    assert texts(result) == ["shown"]
    assert result.ok


def test_unterminated_ifndef() -> None:
    result = run("int a;\n#ifndef FARM_H\nint b;\n")
    diagnostic = only(result, "unterminated_if")
    assert diagnostic.message == "unterminated #ifndef"
    assert diagnostic.location.line == 2


def test_else_without_if() -> None:
    assert only(run("#else\n"), "else_without_if").message == "#else without #if"
    assert only(run("#elif 1\n"), "else_without_if").message == "#elif without #if"


def test_endif_without_if() -> None:
    diagnostic = only(run("int a;\n#endif"), "endif_without_if")
    assert diagnostic.message == "#endif without #if"
    assert diagnostic.location.line == 2


def test_else_after_else() -> None:
    assert only(run("#if 1\n#else\n#else\n#endif"), "invalid_directive").message == "#else after #else"


def test_conditionals_do_not_cross_files() -> None:
    files = {"/home/user/farm/open.h": "#if 1\nint a;\n"}
    result = run('#include "open.h"\n#endif', files)
    assert set(codes(result)) == {"unterminated_if", "endif_without_if"}
    assert only(result, "unterminated_if").location.file == "/home/user/farm/open.h"


# --- other directives --------------------------------------------------------------------

def test_error_directive() -> None:
    result = run('int a;\n#error "need FARM_SIZE"')
    diagnostic = only(result, "error_directive")
    assert diagnostic.message == '#error "need FARM_SIZE"'
    assert diagnostic.severity == Severity.ERROR
    assert diagnostic.location.line == 2
    assert not result.ok


def test_error_in_skipped_block_ignored() -> None:
    assert run("#ifdef NOPE\n#error no\n#endif").diagnostics == []


def test_warning_directive() -> None:
    result = run("#warning slow path")
    diagnostic = only(result, "warning_directive")
    assert diagnostic.severity == Severity.WARNING
    assert diagnostic.concept == Concept.PREPROCESSOR
    assert result.ok


def test_line_and_unknown_pragma_ignored() -> None:
    result = run('#line 100 "x.c"\n#pragma pack(1)\n#\nint a;')
    assert result.diagnostics == []
    assert result.lines == [PPLine("int a;", MAIN, 4)]


def test_invalid_directive() -> None:
    result = run("#inclde <stdio.h>")
    diagnostic = only(result, "invalid_directive")
    assert diagnostic.message == "invalid preprocessing directive #inclde"


def test_directive_lines_produce_no_output() -> None:
    assert run("#define A 1\n#undef A\n#pragma once\n").lines == []


# --- includes ----------------------------------------------------------------------------

FARM_H = "#ifndef FARM_H\n#define FARM_H\nvoid plant(void);\n#endif\n"
FARM_H_PATH = "/home/user/farm/farm.h"


def test_include_guard_double_include_yields_content_once() -> None:
    result = run('#include "farm.h"\n#include "farm.h"\nint main;', {FARM_H_PATH: FARM_H})
    assert texts(result) == ["void plant(void);", "int main;"]
    assert result.guards == {FARM_H_PATH: True}
    assert result.includes == [FARM_H_PATH]


def test_unguarded_header_included_twice_yields_content_twice() -> None:
    result = run('#include "plain.h"\n#include "plain.h"', {"/home/user/farm/plain.h": "int shared;\n"})
    assert texts(result) == ["int shared;", "int shared;"]
    assert result.guards == {"/home/user/farm/plain.h": False}


def test_guard_must_wrap_whole_file() -> None:
    header = "#ifndef A_H\n#define A_H\n#endif\nint outside;\n"
    result = run('#include "a.h"', {"/home/user/farm/a.h": header})
    assert result.guards["/home/user/farm/a.h"] is False


def test_guard_with_if_not_defined_and_comments() -> None:
    header = "/* tools */\n#if !defined(T_H)\n#define T_H 1\nint t;\n#endif // T_H\n"
    result = run('#include "t.h"\n#include "t.h"', {"/home/user/farm/t.h": header})
    assert result.guards["/home/user/farm/t.h"] is True
    assert texts(result) == ["int t;"]


def test_pragma_once() -> None:
    files = {"/home/user/farm/once.h": "#pragma once\nint once;\n"}
    result = run('#include "once.h"\n#include "once.h"', files)
    assert texts(result) == ["int once;"]
    assert result.guards["/home/user/farm/once.h"] is True


def test_included_lines_map_to_their_own_file_and_line() -> None:
    files = {"/home/user/farm/farm.h": "// header\n\nvoid plant(void);\n"}
    result = run("int before;\n#include \"farm.h\"\nint after;", files)
    assert result.lines == [
        PPLine("int before;", MAIN, 1),
        PPLine("void plant(void);", "/home/user/farm/farm.h", 3),
        PPLine("int after;", MAIN, 3),
    ]


def test_quoted_include_searches_including_files_directory_first() -> None:
    files = {
        "/home/user/farm/src/util.h": "int from_src;",
        "/home/user/farm/include/util.h": "int from_include;",
    }
    result = run('#include "util.h"', files, include_dirs=["/home/user/farm/include"], filename=SRC_MAIN)
    assert texts(result) == ["int from_src;"]


def test_quoted_include_falls_back_to_include_dirs_in_order() -> None:
    files = {"/a/x.h": "int a;", "/b/x.h": "int b;", "/b/y.h": "int y;"}
    result = run('#include "x.h"\n#include "y.h"', files, include_dirs=["/a", "/b"], filename=SRC_MAIN)
    assert texts(result) == ["int a;", "int y;"]


def test_quoted_include_falls_back_to_system_headers() -> None:
    result = run('#include "stdio.h"', system={"stdio.h": "int printf(const char *f);"})
    assert result.lines[0].file == "/usr/include/stdio.h"


def test_angled_include_prefers_system_then_include_dirs() -> None:
    files = {
        "/home/user/farm/include/stdio.h": "int fake;",
        "/home/user/farm/include/grid.h": "int grid;",
    }
    result = run("#include <stdio.h>\n#include <grid.h>", files,
                 system={"stdio.h": "int real;"}, include_dirs=["/home/user/farm/include"])
    assert texts(result) == ["int real;", "int grid;"]


def test_angled_include_does_not_search_current_directory() -> None:
    result = run("#include <farm.h>", {FARM_H_PATH: FARM_H})
    diagnostic = only(result, "include_not_found")
    assert '#include "farm.h"' in diagnostic.friendly


def test_nested_includes_resolve_relative_to_each_header() -> None:
    files = {
        "/home/user/farm/include/zone.h": '#include "detail/tiles.h"\nint zone;',
        "/home/user/farm/include/detail/tiles.h": '#include "../common.h"\nint tiles;',
        "/home/user/farm/include/common.h": "int common;",
    }
    result = run('#include "zone.h"', files, include_dirs=["/home/user/farm/include"], filename=SRC_MAIN)
    assert texts(result) == ["int common;", "int tiles;", "int zone;"]
    assert result.includes == [
        "/home/user/farm/include/zone.h",
        "/home/user/farm/include/detail/tiles.h",
        "/home/user/farm/include/common.h",
    ]
    assert result.lines[0] == PPLine("int common;", "/home/user/farm/include/common.h", 1)


def test_relative_main_file_and_path_normalisation() -> None:
    result = run('#include "../inc/a.h"', {"inc/a.h": "int a;"}, filename="src/main.c")
    assert result.lines == [PPLine("int a;", "inc/a.h", 1)]


def test_macros_from_header_visible_after_include() -> None:
    files = {"/home/user/farm/size.h": "#define FIELD 8\n"}
    assert texts(run('#include "size.h"\nint f = FIELD;', files)) == ["int f = 8;"]


def test_include_through_macro_name() -> None:
    files = {"/home/user/farm/farm.h": FARM_H}
    assert texts(run('#define HEADER "farm.h"\n#include HEADER', files)) == ["void plant(void);"]


def test_include_not_found_is_fatal_with_friendly_text() -> None:
    result = run('int a;\n#include "farmm.h"\nint b;', filename=SRC_MAIN)
    diagnostic = only(result, "include_not_found")
    assert diagnostic.message == "farmm.h: No such file or directory"
    assert diagnostic.severity == Severity.FATAL
    assert diagnostic.location.file == SRC_MAIN
    assert diagnostic.location.line == 2
    assert "/home/user/farm/src" in diagnostic.friendly
    assert "-I" in diagnostic.friendly
    assert diagnostic.concept == Concept.HEADERS
    assert texts(result) == ["int a;"]  # like gcc, preprocessing stops here
    # Like gcc, the error points at the header name: column 10 of '#include "farmm.h"'.
    assert diagnostic.format_gcc().startswith(f"{SRC_MAIN}:2:10: ")


def test_include_not_found_inside_header_reports_header_location() -> None:
    files = {"/home/user/farm/a.h": "\n#include <missing.h>\n"}
    diagnostic = only(run('#include "a.h"', files), "include_not_found")
    assert (diagnostic.location.file, diagnostic.location.line) == ("/home/user/farm/a.h", 2)
    assert "system header" in diagnostic.friendly


def test_include_without_filename() -> None:
    assert only(run("#include"), "invalid_directive").message.startswith("#include expects")


def test_include_depth_limit() -> None:
    files = {"/home/user/farm/loop.h": '#include "loop.h"\nint x;\n'}
    result = run('#include "loop.h"', files)
    diagnostic = only(result, "include_depth")
    expected = f"#include nested depth {MAX_INCLUDE_DEPTH + 1} exceeds maximum of {MAX_INCLUDE_DEPTH}"
    assert diagnostic.message == expected
    assert len(result.lines) == MAX_INCLUDE_DEPTH
    assert result.guards["/home/user/farm/loop.h"] is False


def test_all_diagnostics_carry_preprocess_phase_and_friendly_text() -> None:
    result = run("#else\n#error x\n#define\n#if 1/0\n#endif\n#ifndef Q\n", {})
    assert len(result.diagnostics) >= 5
    for diagnostic in result.diagnostics:
        assert diagnostic.phase == Phase.PREPROCESS
        assert diagnostic.friendly
        assert diagnostic.location is not None and diagnostic.location.file == MAIN


def test_every_line_maps_to_origin_in_mixed_program() -> None:
    files = {
        "/home/user/farm/farm.h": FARM_H,
        "/home/user/farm/grid.h": "#pragma once\n/* grid\n   api */\nint grid_w(void);\n",
    }
    source = (
        '#include "farm.h"\n'          # 1
        "#define STEPS 3\n"            # 2
        '#include "grid.h"\n'          # 3
        "int main(void) {\n"           # 4
        "#ifdef STEPS\n"               # 5
        "    plant(); // go\n"         # 6
        "#endif\n"                     # 7
        "    return STEPS;\n"          # 8
        "}\n"                          # 9
    )
    result = run(source, files)
    assert [(line.file, line.line, line.text.strip()) for line in result.lines] == [
        ("/home/user/farm/farm.h", 3, "void plant(void);"),
        ("/home/user/farm/grid.h", 4, "int grid_w(void);"),
        (MAIN, 4, "int main(void) {"),
        (MAIN, 6, "plant();"),
        (MAIN, 8, "return 3;"),
        (MAIN, 9, "}"),
    ]
    assert result.ok


# --- resolver on its own -----------------------------------------------------------------

def test_dict_resolver_returns_none_when_missing() -> None:
    assert DictResolver({}, {}, []).resolve("x.h", True, "main.c") is None


def test_dict_resolver_absolute_name() -> None:
    resolver = DictResolver({"/opt/lib/x.h": "int x;"}, {}, [])
    assert resolver.resolve("/opt/lib/./x.h", True, "main.c") == ("/opt/lib/x.h", "int x;")


# --- review: unsupported macros stop the file; POSIX headers explain themselves ----------

def test_unsupported_macro_operator_stops_the_file() -> None:
    result = run("#define S(x) #x\nint a = S(hi);\nint b = 2;")
    diagnostic = only(result, "invalid_directive")
    assert diagnostic.severity == Severity.FATAL
    assert "Compilation stops here." in diagnostic.friendly
    assert texts(result) == []               # nothing after the define reaches the parser


def test_token_pasting_also_stops_the_file() -> None:
    result = run("#define CAT(a, b) a ## b\nint CAT(v, 1) = 5;")
    assert only(result, "invalid_directive").severity == Severity.FATAL
    assert "int" not in " ".join(texts(result))


def test_posix_header_gets_a_friendly_explanation() -> None:
    diagnostic = only(run("#include <unistd.h>\nint main(void) { return 0; }"),
                      "include_not_found")
    assert diagnostic.message == "unistd.h: No such file or directory"
    assert "POSIX, not standard C" in diagnostic.friendly
    assert "parse argv by hand" in diagnostic.friendly
