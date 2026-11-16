"""Review tests for the C preprocessor: tricky cases found while auditing the module.

Each test names the behaviour it protects. Several of them failed before the review
fixes (multi-character operators split apart, ``#if 08`` crashing, the standard's
``f(2)(9)`` rescanning example, a forgotten ``)`` being quadratically slow).
"""
from __future__ import annotations

import time

from pycc.core.diagnostics import Phase
from pycc.interp.preprocessor import DictResolver, PreprocessResult, preprocess
from pycc.interp.source import PPLine

MAIN = "/home/user/farm/main.c"


def run(text: str, files: dict[str, str] | None = None, filename: str = MAIN) -> PreprocessResult:
    """Preprocess ``text`` with an in-memory resolver over ``files``."""
    return preprocess(text, filename, DictResolver(files or {}))


def numbered(result: PreprocessResult) -> list[tuple[str, int]]:
    """Each output line as ``(stripped text, line number)``."""
    return [(line.text.strip(), line.line) for line in result.lines]


def codes(result: PreprocessResult) -> list[str]:
    """The diagnostic codes, in order."""
    return [diagnostic.code for diagnostic in result.diagnostics]


# --- multi-character operators inside macro bodies ---------------------------------------

def test_increment_in_macro_body_stays_one_operator() -> None:
    assert run("#define INC(x) x++\nINC(a);").text == "a++;"


def test_comparison_and_arrow_in_macro_body_stay_whole() -> None:
    source = "#define EQ(a, b) a==b\n#define NEXT(p) p->next\nEQ(1, 2) NEXT(q)"
    assert run(source).text == "1==2 q->next"


def test_compound_assignment_and_shift_in_object_macro() -> None:
    assert run("#define BUMP total += 1 << 2\nBUMP;").text == "total += 1 << 2;"


def test_negative_macro_after_minus_still_separated() -> None:
    assert run("#define NEG -1\nint x = -NEG;").text == "int x = - -1;"


def test_token_paste_operator_still_rejected() -> None:
    result = run("#define CAT(a, b) a##b\n")
    assert codes(result) == ["invalid_directive"]
    assert "##" in result.diagnostics[0].message


# --- #if arithmetic -----------------------------------------------------------------------

def test_invalid_octal_digit_in_if_is_a_diagnostic_not_a_crash() -> None:
    result = run("#if 08\nint x;\n#endif\nint y;")
    assert codes(result) == ["invalid_directive"]
    assert result.diagnostics[0].message == 'invalid digit "8" in octal constant'
    assert result.diagnostics[0].phase is Phase.PREPROCESS
    assert result.diagnostics[0].friendly
    assert numbered(result) == [("int y;", 4)]


def test_if_nested_parentheses_and_unary_minus() -> None:
    source = "#if ((-(-2)) * (-(3))) == -6 && -(1 - 2) == 1\nyes\n#endif"
    assert run(source).text == "yes"


def test_if_truncating_division_and_remainder_of_negatives() -> None:
    source = "#if -9 / 2 == -4 && -9 % 2 == -1 && ~0 == -1\nyes\n#endif"
    assert run(source).text == "yes"


def test_elif_after_taken_branch_is_not_evaluated() -> None:
    source = "#define A 1\n#if A\none\n#elif 1/0\ntwo\n#elif 1\nthree\n#else\nfour\n#endif"
    result = run(source)
    assert numbered(result) == [("one", 3)]
    assert result.ok


def test_ifdef_inside_skipped_block_still_counts_nesting() -> None:
    source = "#if 0\n#ifdef X\n#else\nbad\n#endif\nbad2\n#elif 1\ngood\n#endif"
    result = run(source)
    assert numbered(result) == [("good", 8)]
    assert result.ok


def test_garbage_conditions_in_skipped_block_are_ignored() -> None:
    source = "#if 0\n#if garbage (\n#elif 1/0\n#endif\n#endif\nafter"
    result = run(source)
    assert result.text == "after"
    assert result.ok


# --- macro expansion order -----------------------------------------------------------------

def test_standard_rescanning_example_f2_9() -> None:
    # C99 6.10.3.5: the ')' comes from outside f's expansion, so f may expand again.
    source = "#define f(a) a*g\n#define g(a) f(a)\nf(2)(9)"
    assert run(source).text == "2*9*g"


def test_self_reference_through_two_macros_stops() -> None:
    assert run("#define A B\n#define B A\nA B").text == "A B"


def test_function_like_name_without_parenthesis_is_a_plain_word() -> None:
    result = run("#define F(x) (x)\nint F;\nF\n+1;")
    assert numbered(result) == [("int F;", 2), ("F", 3), ("+1;", 4)]


def test_object_like_macro_used_like_a_function() -> None:
    assert run("#define N 5\nint y = N(3);").text == "int y = 5(3);"


def test_empty_macro_bodies_vanish() -> None:
    result = run("#define EMPTY\n#define NOTHING(x)\nEMPTY int x NOTHING(1);")
    assert result.text.split() == ["int", "x", ";"]


def test_macro_name_then_blank_line_then_parenthesis_is_a_call() -> None:
    result = run("#define F(x) [x]\nint y = F\n\n(3);\nz;")
    assert numbered(result) == [("int y = [3];", 2), ("z;", 5)]


def test_macro_call_over_three_lines_keeps_following_numbers() -> None:
    result = run("#define ADD(a, b) a+b\nint s = ADD(1,\n\n 2);\nnext;")
    assert numbered(result) == [("int s = 1+2;", 2), ("next;", 5)]


# --- line numbers --------------------------------------------------------------------------

def test_code_between_multiline_comments_keeps_real_lines() -> None:
    result = run("a /* 1\n2 */ b /* 3\n4 */ c\nd")
    assert numbered(result) == [("a", 1), ("b", 2), ("c", 3), ("d", 4)]


def test_crlf_continuation_and_line_macro() -> None:
    result = run("x = __LINE__ + \\\r\n __LINE__;\r\ny = __LINE__;\r\n")
    assert numbered(result) == [("x = 1 +  1;", 1), ("y = 3;", 3)]


def test_backslash_followed_by_trailing_spaces_still_splices() -> None:
    result = run("#define TWO 1 + \\   \n 1\nint t = TWO;\nend;")
    assert numbered(result) == [("int t = 1 + 1;", 3), ("end;", 4)]


def test_tabs_around_directive_parts() -> None:
    result = run("\t#\tdefine\tWIDTH\t7\n\tint w = WIDTH;")
    assert result.lines == [PPLine("\tint w = 7;", MAIN, 2)]


# --- includes ------------------------------------------------------------------------------

def test_header_in_subdirectory_includes_sibling_with_dotdot() -> None:
    files = {
        "/home/user/farm/inc/sub/a.h": '#include "../b.h"\nint a;',
        "/home/user/farm/inc/b.h": "int b;",
    }
    result = run('#include "inc/sub/a.h"\nint main;', files)
    assert result.text.split("\n") == ["int b;", "int a;", "int main;"]
    assert result.includes == ["/home/user/farm/inc/sub/a.h", "/home/user/farm/inc/b.h"]
    assert result.lines[0].file == "/home/user/farm/inc/b.h"


def test_mutual_includes_with_guards_terminate() -> None:
    files = {
        "/home/user/farm/a.h": '#ifndef A_H\n#define A_H\n#include "b.h"\nint a;\n#endif',
        "/home/user/farm/b.h": '#ifndef B_H\n#define B_H\n#include "a.h"\nint b;\n#endif',
    }
    result = run('#include "a.h"\n#include "b.h"\nint main;', files)
    assert result.text.split("\n") == ["int b;", "int a;", "int main;"]
    assert result.guards == {"/home/user/farm/a.h": True, "/home/user/farm/b.h": True}
    assert result.ok


def test_mutual_includes_without_guards_hit_depth_limit_once() -> None:
    files = {
        "/home/user/farm/a.h": '#include "b.h"\n',
        "/home/user/farm/b.h": '#include "a.h"\n',
    }
    result = run('#include "a.h"\n', files)
    assert codes(result) == ["include_depth"]


# --- performance ---------------------------------------------------------------------------

def test_five_thousand_line_file_is_fast() -> None:
    body = "".join(f"int v{i} = SQ({i}) + 1; /* note */\n" for i in range(5000))
    started = time.perf_counter()
    result = run("#define SQ(x) ((x)*(x))\n" + body)
    assert len(result.lines) == 5000
    assert result.lines[-1] == PPLine("int v4999 = ((4999)*(4999)) + 1;", MAIN, 5001)
    assert time.perf_counter() - started < 5


def test_forgotten_parenthesis_in_long_file_is_fast() -> None:
    body = "".join(f'printf("{i}");\n' for i in range(5000))
    started = time.perf_counter()
    result = run("#define SQ(x) ((x)*(x))\nint y = SQ(1;\n" + body)
    assert codes(result) == ["macro_args_mismatch"]
    assert result.diagnostics[0].location is not None
    assert result.diagnostics[0].location.line == 2
    assert time.perf_counter() - started < 5
