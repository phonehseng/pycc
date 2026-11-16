"""Corners of expression, initializer and statement evaluation the interpreter must get right."""
from __future__ import annotations

import pytest

from tests.interp.test_interpreter_support import program, run_c, trap_code, warning_codes

CASES: list[tuple[str, str, str]] = [
    # --- operators -------------------------------------------------------------------------
    ("compound_shift_and_bitwise", 'int x = 3;\nx <<= 2;\nx |= 1;\nx ^= 4;\nx &= 0xD;\nx %= 7;\n'
     'printf("%d\\n", x);', "2\n"),
    ("pointer_compound_assignment", 'int a[4] = {1, 2, 3, 4};\nint *p = a;\np += 3;\np -= 1;\n'
     'printf("%d\\n", *p);', "3\n"),
    ("float_increment", 'double d = 1.5;\nd++;\nfloat f = 0.5f;\n--f;\n'
     'printf("%.1f %.1f\\n", d, f);', "2.5 -0.5\n"),
    ("pointer_decrement", 'int a[3] = {7, 8, 9};\nint *p = &a[2];\np--;\nprintf("%d\\n", *p--);',
     "8\n"),
    ("complement_of_unsigned_char_is_int", 'unsigned char c = 0;\nprintf("%d\\n", ~c);', "-1\n"),
    ("plain_char_is_signed", 'char c = 200;\nprintf("%d\\n", c);', "-56\n"),
    ("not_on_pointers", 'int x;\nint *p = &x;\nint *q = 0;\nprintf("%d %d\\n", !p, !q);', "0 1\n"),
    ("pointer_comparisons", 'int a[3];\nprintf("%d %d %d\\n", &a[0] < &a[2], &a[1] == a + 1, '
     '&a[2] >= &a[1]);', "1 1 1\n"),
    ("address_of_dereference", 'int x = 4;\nint *p = &x;\nprintf("%d\\n", &*p == p);', "1\n"),
    ("sizeof_does_not_evaluate", 'int n = 1;\nsize_t s = sizeof(n++);\nprintf("%zu %d\\n", s, n);',
     "4 1\n"),
    ("ternary_evaluates_one_side", 'int a = 0, b = 0;\nint r = 1 ? a++ : b++;\n'
     'printf("%d %d %d\\n", r, a, b);', "0 1 0\n"),
    ("assignment_is_an_expression", 'int a, b, c;\na = b = c = 7;\nprintf("%d\\n", a + b + c);',
     "21\n"),
    ("logical_results_are_zero_or_one", 'printf("%d %d %d\\n", 5 && 3, 0 || -2, !!42);', "1 1 1\n"),
    ("function_pointer_equality", 'int (*f)(void) = one;\nprintf("%d %d\\n", f == one, f());',
     "1 1\n"),
    ("unary_minus_of_unsigned_wraps", 'unsigned int u = 1;\nprintf("%u\\n", -u);', "4294967295\n"),
    # --- initializers ------------------------------------------------------------------------
    ("scalar_in_braces", 'int x = {5};\nprintf("%d\\n", x);', "5\n"),
    ("brace_elision_for_struct_array", 'struct P ps[2] = {1, 2, 3, 4};\n'
     'printf("%d %d\\n", ps[1].x, ps[1].y);', "3 4\n"),
    ("two_dimensional_partial_rows", 'int m[2][3] = {{1}, {2, 3}};\n'
     'printf("%d %d %d %d\\n", m[0][0], m[0][2], m[1][1], m[1][2]);', "1 0 3 0\n"),
    ("string_member_initializer", 'struct Tag t = {"oak", 3};\nprintf("%s %d %d\\n", t.name, '
     't.n, t.name[4]);', "oak 3 0\n"),
    ("union_initializes_its_first_member", 'union U u = {65};\nprintf("%d %c\\n", u.i, u.c);',
     "65 A\n"),
    ("struct_initialised_from_struct", 'struct P a = {1, 2};\nstruct P b = a;\n'
     'struct P list[2] = {a, b};\nprintf("%d\\n", list[1].y);', "2\n"),
    ("char_array_from_brace_string", 'char s[6] = {"wheat"};\nprintf("%s\\n", s);', "wheat\n"),
    ("pointer_array_of_literals", 'const char *crops[] = {"wheat", "corn"};\n'
     'printf("%s %zu\\n", crops[1], sizeof crops / sizeof crops[0]);', "corn 2\n"),
    # --- statements ------------------------------------------------------------------------------
    ("switch_jumps_past_a_declaration", 'int k = 2;\nswitch (k) {\ncase 1:;\n  int v = 10;\n'
     'case 2:\n  v = 5;\n  printf("%d\\n", v);\n}', "5\n"),
    ("switch_on_unsigned_with_negative_case", 'unsigned char c = 255;\nswitch (c) {\n'
     'case 255: printf("max\\n"); break;\ndefault: printf("other\\n");\n}', "max\n"),
    ("switch_on_char", 'char c = \'b\';\nswitch (c) { case \'a\': printf("A\\n"); break; '
     'case \'b\': printf("B\\n"); break; }', "B\n"),
    ("continue_in_do_while", 'int i = 0, n = 0;\ndo {\n  i++;\n  if (i % 2) continue;\n  n++;\n'
     '} while (i < 6);\nprintf("%d\\n", n);', "3\n"),
    ("return_from_inside_a_loop", 'printf("%d\\n", first_even());',
     "4\n"),
    ("struct_member_of_returned_struct_array", 'printf("%d\\n", make().v[1]);', "20\n"),
    ("static_local_initialised_once", 'for (int i = 0; i < 3; i++) visits();\n'
     'printf("%d\\n", visits());', "4\n"),
    ("empty_for_header", 'int n = 0;\nfor (;;) { if (++n == 3) break; }\nprintf("%d\\n", n);',
     "3\n"),
]

BEFORE = """struct P { int x, y; };
struct Tag { char name[8]; int n; };
union U { int i; char c; };
struct Box { int v[2]; };
int one(void) { return 1; }
int first_even(void) { for (int i = 1; ; i++) { if (i % 4 == 0) return i; } }
struct Box make(void) { struct Box b = {{10, 20}}; return b; }
int visits(void) { static int count = 0; return ++count; }
"""


@pytest.mark.parametrize(("body", "stdout"), [case[1:] for case in CASES],
                         ids=[case[0] for case in CASES])
def test_expression(body: str, stdout: str) -> None:
    run = run_c(program(body + "\nreturn 0;", ("stdio.h", "stddef.h"), BEFORE))
    assert run.summary.trap is None, run.summary.trap.format_gcc()
    assert run.stdout == stdout


def test_negating_int_min_overflows() -> None:
    run = run_c(program('int m = -2147483647 - 1;\nint n = -m;\nprintf("%d\\n", n);\nreturn 0;'))
    assert run.stdout == "-2147483648\n"
    assert warning_codes(run) == ["signed_overflow"]


def test_int_max_increment_overflows_but_char_does_not() -> None:
    run = run_c(program('int i = 2147483647;\ni++;\nchar c = 127;\nc++;\nreturn 0;'))
    assert warning_codes(run) == ["signed_overflow"]


def test_left_shift_into_sign_bit_warns() -> None:
    run = run_c(program('int one = 1;\nint x = one << 31;\nprintf("%d\\n", x);\nreturn 0;'))
    assert run.stdout == "-2147483648\n"
    assert warning_codes(run) == ["signed_overflow"]


def test_negative_shift_count_warns() -> None:
    run = run_c(program('int n = -1;\nint x = 8 >> n;\nreturn x;'))
    assert "shift_overflow" in warning_codes(run)


def test_struct_copy_keeps_garbage_uninitialised() -> None:
    run = run_c(program('struct P { int x, y; } a, b;\na.x = 1;\nb = a;\nreturn b.y > 0;'))
    assert "uninitialized_read" in warning_codes(run)


def test_dangling_pointer_to_a_dead_block_local_traps() -> None:
    run = run_c(program('int *p;\n{\n  int inner = 5;\n  p = &inner;\n}\nreturn *p;'))
    assert trap_code(run) == "wild_pointer"
    assert "out of scope" in run.summary.trap.message


def test_pointer_returned_from_a_function_local_traps() -> None:
    run = run_c(program('int *p = dangle();\nreturn *p;',
                        before='int *dangle(void) { int local = 3; return &local; }'))
    assert trap_code(run) == "wild_pointer"


def test_stepping_past_an_array_into_its_neighbour_is_caught() -> None:
    run = run_c(program('int a[2] = {1, 2};\nint b[2] = {3, 4};\nint *p = a;\n'
                        'return *(p + 5) + b[0];'))
    assert trap_code(run) == "out_of_bounds"
    assert "'a'" in run.summary.trap.message


def test_bad_function_pointer_traps() -> None:
    run = run_c(program('int (*f)(void) = (int (*)(void))12345;\nreturn f();'))
    assert trap_code(run) == "bad_function_pointer"
    assert run.exit_code == 139


def test_function_pointer_prints_as_a_code_address() -> None:
    # The pointer's raw value is its function-table id; printf renders a
    # function-typed value as a code address (0x400000 + id * 16, docs D3).
    run = run_c(program('int (*f)(void) = one;\nprintf("%p\\n", f);\nreturn 0;', before=BEFORE))
    assert run.stdout.startswith("0x4000")
