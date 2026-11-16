"""The "tricky C semantics" suite: small real C programs and exactly what they print.

Each case is a complete program compiled by the real toolchain and run by the
interpreter; the expected output is what gcc on 64-bit Linux prints. The table at
the top holds the programs whose stdout and exit code say it all; the tests below it
check the runs where the warnings, traps and friendly texts matter too.
"""
from __future__ import annotations

import pytest

from pycc.interp.hostio import DictHostFiles, ProgramIO
from tests.interp.test_interpreter_support import program, run_c, trap_code, warning_codes

STD = ("stdio.h", "stdlib.h", "string.h")

# (id, source, expected stdout, expected exit code)
CASES: list[tuple[str, str, str, int]] = [
    # --- integers: promotion, wraparound, division ------------------------------------
    ("char_plus_char_is_int", program(
        'char a = 100, b = 100;\nprintf("%d %zu\\n", a + b, sizeof(a + b));\nreturn 0;'),
     "200 4\n", 0),
    ("unsigned_char_wraps", program(
        'unsigned char c = 255;\nc = c + 1;\nunsigned char d = 255;\nd++;\n'
        'printf("%d %d\\n", c, d);\nreturn 0;'), "0 0\n", 0),
    ("unsigned_int_wraps_below_zero", program(
        'unsigned int u = 0;\nu--;\nprintf("%u\\n", u);\nreturn 0;'), "4294967295\n", 0),
    ("division_truncates_toward_zero", program(
        'printf("%d %d %d %d\\n", 7 / 2, -7 / 2, 7 / -2, -7 / -2);\nreturn 0;'),
     "3 -3 -3 3\n", 0),
    ("modulo_takes_sign_of_dividend", program(
        'printf("%d %d %d %d\\n", 7 % 3, -7 % 3, 7 % -3, -7 % -3);\nreturn 0;'),
     "1 -1 1 -1\n", 0),
    ("signed_unsigned_comparison", program(
        'int neg = -1;\nunsigned int zero = 0;\nprintf("%d\\n", neg < zero);\nreturn 0;'),
     "0\n", 0),
    ("char_overflow_by_conversion", program(
        'char c = 127;\nc++;\nchar d = 100;\nd += 100;\nprintf("%d %d\\n", c, d);\nreturn 0;'),
     "-128 -56\n", 0),
    ("bool_normalises", program(
        '_Bool b = 5;\n_Bool z = 0;\nz--;\nprintf("%d %d\\n", b, z);\nreturn 0;'),
     "1 1\n", 0),
    ("long_literals_and_shifts", program(
        'long big = 1L << 40;\nprintf("%ld %zu %zu\\n", big, sizeof(2147483648), sizeof(1));'
        '\nreturn 0;'), "1099511627776 8 4\n", 0),
    ("hex_and_octal_literals", program(
        'printf("%d %d %x %X\\n", 0x1F, 017, 255, 3054);\nreturn 0;'), "31 15 ff BEE\n", 0),
    ("bitwise_operators", program(
        'printf("%d %d %d %d\\n", 0xF0 & 0x3C, 0xF0 | 0x0F, 0xFF ^ 0x0F, ~0);\nreturn 0;'),
     "48 255 240 -1\n", 0),
    ("shifts_signed_and_unsigned", program(
        'int n = -16;\nunsigned int u = 0x80000000u;\n'
        'printf("%d %d %u\\n", 1 << 4, n >> 2, u >> 31);\nreturn 0;'), "16 -4 1\n", 0),
    # --- floating point -----------------------------------------------------------------
    ("float_versus_double_literal", program(
        'float f = 0.1f;\ndouble d = 0.1;\nprintf("%d %.10f %.10f\\n", f == d, f, d);\nreturn 0;'),
     "0 0.1000000015 0.1000000000\n", 0),
    ("float_stays_32_bit", program(
        'float f = 16777216.0f;\nf = f + 1.0f;\nprintf("%.1f\\n", f);\nreturn 0;'),
     "16777216.0\n", 0),
    ("casts_truncate", program(
        'printf("%d %d %d %d\\n", (int)3.99, (int)-3.99, (char)300, (unsigned char)-1);'
        '\nreturn 0;'), "3 -3 44 255\n", 0),
    ("int_division_versus_float_division", program(
        'int a = 7, b = 2;\nprintf("%d %.1f %.1f\\n", a / b, a / 2.0, (double)a / b);\nreturn 0;'),
     "3 3.5 3.5\n", 0),
    ("compound_assignment_converts", program(
        'int i = 10;\ni *= 2.5;\nunsigned int u = 5;\nu -= 10;\ndouble d = 1;\nd /= 4;\n'
        'printf("%d %u %.2f\\n", i, u, d);\nreturn 0;'), "25 4294967291 0.25\n", 0),
    ("ternary_takes_common_type", program(
        'int yes = 1;\nprintf("%f %zu\\n", yes ? 1 : 2.0, sizeof(yes ? \'a\' : \'b\'));\n'
        'return 0;'),
     "1.000000 4\n", 0),
    # --- sizeof, arrays, pointers --------------------------------------------------------
    ("sizeof_array_pointer_struct", program(
        'int a[10];\nint *p = a;\nstruct S { char c; int i; };\n'
        'struct T { char c; double d; char e; };\n'
        'printf("%zu %zu %zu %zu\\n", sizeof a, sizeof p, sizeof(struct S), sizeof(struct T));'
        '\nreturn 0;'), "40 8 8 24\n", 0),
    ("array_parameter_is_a_pointer", program(
        'int a[10] = {0};\nreturn report(a);',
        before='int report(int arr[]) { printf("%zu\\n", sizeof(arr)); return arr[0]; }'),
     "8\n", 0),
    ("pointer_arithmetic_and_subtraction", program(
        'int a[5] = {10, 20, 30, 40, 50};\nint *p = &a[1];\nint *q = &a[4];\n'
        'printf("%ld %d %d %ld\\n", q - p, *(p + 2), p[-1], (char *)q - (char *)p);\nreturn 0;'),
     "3 40 10 12\n", 0),
    ("pre_and_post_increment_on_pointers", program(
        'int a[3] = {10, 20, 30};\nint *p = a;\nint first = *p++;\nint second = *p;\n'
        'int third = *++p;\nprintf("%d %d %d %ld\\n", first, second, third, p - a);\nreturn 0;'),
     "10 20 30 2\n", 0),
    ("array_decays_in_calls", program(
        'int a[4] = {1, 2, 3, 4};\nprintf("%d\\n", sum(a, 4));\nreturn 0;',
        before='int sum(int *v, int n) { int t = 0; for (int i = 0; i < n; i++) t += v[i]; '
               'return t; }'), "10\n", 0),
    ("two_dimensional_arrays", program(
        'int m[2][3] = {{1, 2, 3}, {4, 5, 6}};\nint total = 0;\n'
        'for (int r = 0; r < 2; r++)\n  for (int c = 0; c < 3; c++)\n    total += m[r][c];\n'
        'printf("%d %d %zu %zu\\n", total, m[1][2], sizeof m, sizeof m[0]);\nreturn 0;'),
     "21 6 24 12\n", 0),
    ("partial_initialisers_zero_fill", program(
        'int a[5] = {1};\nstruct P { int x, y, z; } p = {7};\n'
        'printf("%d %d %d %d %d %d\\n", a[0], a[1], a[4], p.x, p.y, p.z);\nreturn 0;'),
     "1 0 0 7 0 0\n", 0),
    ("pointer_to_pointer", program(
        'int x = 5;\nint *p = &x;\nint **pp = &p;\n**pp = 9;\nprintf("%d\\n", x);\nreturn 0;'),
     "9\n", 0),
    # --- strings ----------------------------------------------------------------------------
    ("strings_and_nul", program(
        'char s[] = "hello";\nprintf("%zu %zu ", sizeof s, strlen(s));\ns[2] = \'\\0\';\n'
        'printf("%s\\n", s);\nreturn 0;', STD), "6 5 he\n", 0),
    ("char_array_padded_with_zeros", program(
        'char s[8] = "hi";\nprintf("%d %d %s\\n", s[2], s[7], s);\nreturn 0;'), "0 0 hi\n", 0),
    ("string_functions", program(
        'char buf[16];\nstrcpy(buf, "hello");\nstrcat(buf, " world");\n'
        'printf("%s %zu %d\\n", buf, strlen(buf), strcmp("a", "b") < 0);\nreturn 0;', STD),
     "hello world 11 1\n", 0),
    ("char_arithmetic", program(
        'char c = \'a\';\nc = c + 1;\nprintf("%c %d %c\\n", c, \'A\', \'z\' - 1);\nreturn 0;'),
     "b 65 y\n", 0),
    ("sprintf_into_buffer", program(
        'char buf[32];\nint n = sprintf(buf, "%d-%s", 42, "x");\nprintf("%s %d\\n", buf, n);'
        '\nreturn 0;'), "42-x 4\n", 0),
    # --- structs, unions, enums, typedefs -----------------------------------------------------
    ("struct_copies_are_independent", program(
        'struct P { int x, y; };\nstruct P a = {1, 2};\nstruct P b = a;\nb.x = 9;\n'
        'struct P c;\nc = b;\nc.y = 7;\nprintf("%d %d %d %d %d\\n", a.x, b.x, b.y, c.x, c.y);'
        '\nreturn 0;'), "1 9 2 9 7\n", 0),
    ("struct_returned_by_value", program(
        'struct P p = make(3, 4);\nprintf("%d %d %d\\n", p.x, p.y, make(5, 6).y);\nreturn 0;',
        before='struct P { int x, y; };\nstruct P make(int x, int y) { struct P p = {x, y}; '
               'return p; }'), "3 4 6\n", 0),
    ("struct_passed_by_value", program(
        'struct P p = {1, 2};\nbump(p);\nprintf("%d\\n", p.x);\nreturn 0;',
        before='struct P { int x, y; };\nvoid bump(struct P p) { p.x = 100; }'), "1\n", 0),
    ("struct_through_pointer", program(
        'struct P p = {1, 2};\nbump(&p);\nprintf("%d %d\\n", p.x, p.y);\nreturn 0;',
        before='struct P { int x, y; };\nvoid bump(struct P *p) { p->x = 100; (*p).y += 1; }'),
     "100 3\n", 0),
    ("nested_structs", program(
        'struct Line l = {{1, 2}, {3, 4}};\nstruct Line m = l;\nm.b.y = 40;\n'
        'printf("%d %d %d %zu\\n", l.a.x + l.b.x, l.b.y, m.b.y, sizeof l);\nreturn 0;',
        before='struct Pt { int x, y; };\nstruct Line { struct Pt a, b; };'),
     "4 4 40 16\n", 0),
    ("array_of_structs", program(
        'struct P ps[3] = {{1, 2}, {3, 4}};\nprintf("%d %d %d\\n", ps[1].x, ps[0].y, ps[2].x);'
        '\nreturn 0;', before='struct P { int x, y; };'), "3 2 0\n", 0),
    ("struct_with_array_returned", program(
        'struct Box b = fill();\nprintf("%d %d %s\\n", b.v[0], b.v[2], b.name);\nreturn 0;',
        before='struct Box { int v[3]; char name[8]; };\n'
               'struct Box fill(void) { struct Box b = {{7, 8, 9}, "crate"}; return b; }'),
     "7 9 crate\n", 0),
    ("unions_overlap", program(
        'union U { int i; unsigned char b[4]; } u;\nu.i = 0x01020304;\n'
        'printf("%d %d %zu\\n", u.b[0], u.b[3], sizeof u);\nreturn 0;'), "4 1 4\n", 0),
    ("enums", program(
        'enum Color c = BLUE;\nprintf("%d %d %d\\n", RED, GREEN, c);\nreturn c == BLUE ? 0 : 1;',
        before='enum Color { RED, GREEN = 5, BLUE };'), "0 5 6\n", 0),
    ("typedefs", program(
        'Point p = {2, 3};\nBinOp op = add;\nprintf("%d\\n", op(p.x, p.y));\nreturn 0;',
        before='typedef struct { int x, y; } Point;\ntypedef int (*BinOp)(int, int);\n'
               'int add(int a, int b) { return a + b; }'), "5\n", 0),
    # --- functions and control flow --------------------------------------------------------
    ("function_pointers_table", program(
        'int (*ops[3])(int, int) = {add, sub, mul};\nfor (int i = 0; i < 3; i++)\n'
        '  printf("%d ", ops[i](6, 3));\nprintf("%d\\n", (*ops[0])(1, 1));\nreturn 0;',
        before='int add(int a, int b) { return a + b; }\nint sub(int a, int b) { return a - b; }\n'
               'int mul(int a, int b) { return a * b; }'), "9 3 18 2\n", 0),
    ("callback_parameter", program(
        'printf("%d\\n", apply(twice, 21));\nreturn 0;',
        before='int twice(int x) { return 2 * x; }\n'
               'int apply(int (*f)(int), int v) { return f(v); }'), "42\n", 0),
    ("qsort_with_comparator", program(
        'int v[6] = {5, 3, 9, 1, 7, 3};\nqsort(v, 6, sizeof v[0], cmp);\n'
        'for (int i = 0; i < 6; i++) printf("%d ", v[i]);\nprintf("\\n");\nreturn 0;', STD,
        before='int cmp(const void *a, const void *b) { return *(const int *)a - '
               '*(const int *)b; }'), "1 3 3 5 7 9 \n", 0),
    ("recursion_factorial_and_fib", program(
        'printf("%d %d\\n", fact(10), fib(15));\nreturn 0;',
        before='int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }\n'
               'int fib(int n) { if (n < 2) return n; return fib(n - 1) + fib(n - 2); }'),
     "3628800 610\n", 0),
    ("mutual_recursion", program(
        'printf("%d %d\\n", is_even(10), is_even(7));\nreturn 0;',
        before='int is_odd(int n);\nint is_even(int n) { return n == 0 ? 1 : is_odd(n - 1); }\n'
               'int is_odd(int n) { return n == 0 ? 0 : is_even(n - 1); }'), "1 0\n", 0),
    ("switch_fallthrough", program(
        'for (int i = 0; i < 4; i++) {\n  switch (i) {\n  case 0: printf("zero ");\n'
        '  case 1: printf("one ");\n    break;\n  case 2: { printf("two "); }\n'
        '  default: printf("dflt ");\n  }\n}\nprintf("\\n");\nreturn 0;'),
     "zero one one two dflt dflt \n", 0),
    ("switch_without_match_does_nothing", program(
        'int x = 9;\nswitch (x) { case 1: printf("one\\n"); }\nprintf("done\\n");\nreturn 0;'),
     "done\n", 0),
    ("short_circuit_side_effects", program(
        'int a = 0 && touch();\nint b = 1 || touch();\nint c = 1 && touch();\n'
        'printf("%d %d %d %d\\n", a, b, c, calls);\nreturn 0;',
        before='int calls = 0;\nint touch(void) { calls++; return 1; }'), "0 1 1 1\n", 0),
    ("comma_operator", program(
        'int a = 1;\nint x = (a++, a + 10);\nint i, j;\nfor (i = 0, j = 10; i < j; i++, j--);\n'
        'printf("%d %d %d\\n", x, a, i);\nreturn 0;'), "12 2 5\n", 0),
    ("break_and_continue_in_nested_loops", program(
        'int count = 0;\nfor (int i = 0; i < 5; i++) {\n  if (i == 1) continue;\n'
        '  for (int j = 0; j < 5; j++) {\n    if (j == 3) break;\n    count++;\n  }\n'
        '  if (i == 3) break;\n}\nprintf("%d\\n", count);\nreturn 0;'), "9\n", 0),
    ("do_while_runs_once", program(
        'int n = 0;\ndo { n++; } while (n > 5);\nwhile (n < 0) n--;\nprintf("%d\\n", n);\n'
        'return 0;'),
     "1\n", 0),
    ("global_versus_local_shadowing", program(
        'int x = 2;\n{\n  int x = 3;\n  printf("%d ", x);\n}\nprintf("%d ", x);\nshow();\n'
        'return 0;', before='int x = 1;\nvoid show(void) { printf("%d\\n", x); }'),
     "3 2 1\n", 0),
    ("static_locals_persist", program(
        'counter();\ncounter();\nprintf("%d\\n", counter());\nreturn 0;',
        before='int counter(void) { static int n = 0; n++; return n; }'), "3\n", 0),
    ("globals_and_their_initialisers", program(
        'printf("%d %d %s %d %d\\n", g, *gp, msg, table[2], zero);\nreturn 0;',
        before='int g = 5;\nint *gp = &g;\nchar *msg = "hi";\nint table[3] = {1, 2, 3};\n'
               'int zero;'), "5 5 hi 3 0\n", 0),
    ("main_without_return_gives_zero", "#include <stdio.h>\nint main(void) { printf(\"x\\n\"); }\n",
     "x\n", 0),
    ("main_return_value_is_masked", program('return 258;'), "", 2),
    ("exit_mid_program", program(
        'printf("before\\n");\nfinish();\nprintf("after\\n");\nreturn 0;', STD,
        before='void finish(void) { exit(7); }'), "before\n", 7),
    # --- heap -----------------------------------------------------------------------------------
    ("malloc_free_and_realloc", program(
        'int *p = malloc(2 * sizeof(int));\np[0] = 1;\np[1] = 2;\n'
        'p = realloc(p, 4 * sizeof(int));\n'
        'p[3] = 4;\nprintf("%d %d %d\\n", p[0], p[1], p[3]);\nfree(p);\nreturn 0;', STD),
     "1 2 4\n", 0),
    ("calloc_is_zeroed", program(
        'int *p = calloc(4, sizeof(int));\nprintf("%d %d\\n", p[0], p[3]);\nfree(p);\nreturn 0;',
        STD), "0 0\n", 0),
    ("linked_list_on_the_heap", program(
        'struct Node *head = NULL;\nfor (int i = 1; i <= 3; i++) {\n'
        '  struct Node *n = malloc(sizeof *n);\n  n->value = i;\n  n->next = head;\n'
        '  head = n;\n}\nint total = 0;\nwhile (head != NULL) {\n'
        '  struct Node *next = head->next;\n'
        '  total = total * 10 + head->value;\n  free(head);\n  head = next;\n}\n'
        'printf("%d\\n", total);\nreturn 0;', STD,
        before='struct Node { int value; struct Node *next; };'), "321\n", 0),
    # --- printf -----------------------------------------------------------------------------------
    ("printf_integer_formats", program(
        'printf("%d|%i|%u|%x|%X|%o|%c|%%\\n", -5, 42, 7u, 255, 255, 8, \'Q\');\n'
        'printf("[%5d][%-5d][%05d][%ld][%lu][%zu]\\n", 42, 42, 42, -9L, 9UL, sizeof(int));'
        '\nreturn 0;'), "-5|42|7|ff|FF|10|Q|%\n[   42][42   ][00042][-9][9][4]\n", 0),
    ("printf_float_and_string_formats", program(
        'printf("%f|%.2f|%8.3f|%e|%g|%s|%.3s|%5s|\\n", 3.14159, 2.5, 1.0, 1234.5, 0.5, "ok", '
        '"abcdef", "x");\nreturn 0;'),
     "3.141590|2.50|   1.000|1.234500e+03|0.5|ok|abc|    x|\n", 0),
]


@pytest.mark.parametrize(("source", "stdout", "exit_code"),
                         [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_program_output(source: str, stdout: str, exit_code: int) -> None:
    run = run_c(source)
    assert run.summary.trap is None, run.summary.trap.format_gcc()
    assert run.stdout == stdout
    assert run.exit_code == exit_code


# =========================================================================== warnings

def test_signed_overflow_wraps_with_warning() -> None:
    run = run_c(program('int x = 2147483647;\nx = x + 1;\nprintf("%d\\n", x);\nreturn 0;'))
    assert run.stdout == "-2147483648\n"
    assert warning_codes(run) == ["signed_overflow"]
    warning = run.summary.warnings[0]
    assert "cannot be represented in type 'int'" in warning.message
    assert warning.location is not None and warning.location.line == 5


def test_unsigned_wraparound_has_no_warning() -> None:
    run = run_c(program('unsigned int u = 4294967295u;\nu = u + 1;\nprintf("%u\\n", u);\n'
                        'return 0;'))
    assert run.stdout == "0\n"
    assert warning_codes(run) == []


def test_shift_by_too_much_warns() -> None:
    run = run_c(program('int n = 33;\nprintf("%d\\n", 1 << n);\nreturn 0;'))
    assert "shift_overflow" in warning_codes(run)
    assert run.stdout == "2\n"        # x86 uses the count modulo 32


def test_uninitialized_read_warns_with_honest_garbage() -> None:
    source = program('int x;\nint y = x + 0;\nprintf("%d\\n", y);\nreturn 0;')
    first, second = run_c(source), run_c(source)
    assert warning_codes(first) == ["uninitialized_read"]
    assert "'x'" in first.summary.warnings[0].message
    assert first.stdout == second.stdout      # seeded garbage: the same every run
    assert first.exit_code == 0


def test_mismatched_printf_conversion_prints_seeded_garbage() -> None:
    # a wrong format must never pass an output check by accident.
    source = program('printf("%d\\n", 4.5);\nreturn 0;')
    first, second = run_c(source), run_c(source)
    assert first.stdout != "4\n"
    assert first.stdout == second.stdout      # seeded: the same every run
    assert "format_argument_type" in warning_codes(first)


def test_memory_leak_warnings_one_per_block() -> None:
    run = run_c(program('malloc(4);\nmalloc(8);\nvoid *kept = malloc(16);\nfree(kept);\n'
                        'return 0;', STD))
    assert run.exit_code == 0
    assert warning_codes(run) == ["memory_leak", "memory_leak"]
    assert run.summary.leaks == 2
    assert "4 bytes" in run.summary.warnings[0].message


def test_memory_leak_warnings_are_capped_at_ten() -> None:
    run = run_c(program('for (int i = 0; i < 12; i++) malloc(4);\nreturn 0;', STD))
    assert warning_codes(run).count("memory_leak") == 10
    assert run.summary.leaks == 12


# =========================================================================== traps

def test_string_literal_write_traps() -> None:
    run = run_c(program('char *s = "hello";\ns[0] = \'H\';\nreturn 0;'))
    assert trap_code(run) == "write_to_literal"
    assert run.exit_code == 139


def test_out_of_bounds_trap_has_friendly_message() -> None:
    run = run_c(program('int field[4];\nfor (int i = 0; i <= 4; i++)\n  field[i] = i;\nreturn 0;'))
    trap = run.summary.trap
    assert trap is not None and trap.code == "out_of_bounds"
    assert trap.message == "index 4 is past the end of 'field' (4 elements)"
    assert "valid indexes are 0 to 3" in trap.friendly
    assert trap.location is not None and trap.location.line == 6
    assert run.exit_code == 139


def test_negative_index_traps() -> None:
    run = run_c(program('int a[3] = {0};\nint i = -1;\nreturn a[i];'))
    assert trap_code(run) == "out_of_bounds"


def test_heap_overrun_traps() -> None:
    run = run_c(program('int *p = malloc(3 * sizeof(int));\np[3] = 1;\nreturn 0;', STD))
    assert trap_code(run) == "out_of_bounds"


def test_null_dereference_traps() -> None:
    run = run_c(program('int *p = NULL;\n*p = 3;\nreturn 0;', STD))
    assert trap_code(run) == "null_dereference"
    assert run.exit_code == 139


def test_use_after_free_traps() -> None:
    run = run_c(program('int *p = malloc(sizeof(int));\n*p = 1;\nfree(p);\nreturn *p;', STD))
    assert trap_code(run) == "use_after_free"
    assert run.exit_code == 139


def test_double_free_aborts() -> None:
    run = run_c(program('int *p = malloc(4);\nfree(p);\nfree(p);\nreturn 0;', STD))
    assert trap_code(run) == "double_free"
    assert run.exit_code == 134


def test_stack_overflow_from_runaway_recursion() -> None:
    run = run_c(program('return down(0);', before='int down(int n) { return down(n + 1); }'))
    assert trap_code(run) == "stack_overflow"
    assert run.exit_code == 139


def test_instruction_budget_stops_infinite_loop() -> None:
    run = run_c(program('int n = 0;\nwhile (1) n++;\nreturn n;'), budget=2_000)
    assert trap_code(run) == "instruction_budget"
    assert run.exit_code == 137
    assert run.summary.budget_hit


def test_division_by_zero_traps() -> None:
    run = run_c(program('int zero = 0;\nprintf("%d\\n", 10 % zero);\nreturn 0;'))
    assert trap_code(run) == "division_by_zero"
    assert run.exit_code == 136


def test_output_before_a_trap_is_kept() -> None:
    run = run_c(program('printf("partial\\n");\nint *p = NULL;\nreturn *p;', STD))
    assert run.stdout == "partial\n"
    assert run.exit_code == 139


@pytest.mark.parametrize(("body", "code", "status"), [
    ('int *p = NULL;\nreturn *p;', "null_dereference", 139),
    ('abort();\nreturn 0;', "abort_called", 134),
    ('int *p = malloc(4);\nfree(p);\nfree(p);\nreturn 0;', "double_free", 134),
    ('int d = 0;\nreturn 1 / d;', "division_by_zero", 136),
    ('for (;;) {}\nreturn 0;', "instruction_budget", 137),
])
def test_exit_code_table(body: str, code: str, status: int) -> None:
    run = run_c(program(body, STD), budget=5_000)
    assert trap_code(run) == code
    assert run.exit_code == status


# =========================================================================== the outside device

def test_argc_and_argv() -> None:
    source = (
        '#include <stdio.h>\nint main(int argc, char *argv[]) {\n'
        '  for (int i = 0; i < argc; i++) printf("[%s]", argv[i]);\n'
        '  printf(" %d %d\\n", argc, argv[argc] == NULL);\n  return argc;\n}\n')
    run = run_c(source, argv=["./prog", "seed", "42"])
    assert run.stdout == "[./prog][seed][42] 3 1\n"
    assert run.exit_code == 3


def test_scanf_from_stdin() -> None:
    run = run_c(program('int a, b;\nif (scanf("%d %d", &a, &b) != 2) return 1;\n'
                        'printf("%d\\n", a * b);\nreturn 0;'), stdin="6 7\n")
    assert run.stdout == "42\n"


def test_fgets_from_stdin() -> None:
    run = run_c(program('char line[32];\nwhile (fgets(line, sizeof line, stdin) != NULL)\n'
                        '  printf("> %s", line);\nreturn 0;'), stdin="first\nsecond\n")
    assert run.stdout == "> first\n> second\n"


def test_file_round_trip_on_host_files() -> None:
    files = DictHostFiles()
    source = program(
        'FILE *out = fopen("notes.txt", "w");\nif (out == NULL) return 1;\n'
        'fprintf(out, "wheat %d\\n", 12);\nfclose(out);\n'
        'FILE *in = fopen("notes.txt", "r");\nchar line[32];\nfgets(line, sizeof line, in);\n'
        'fclose(in);\nprintf("read: %s", line);\nreturn 0;')
    run = run_c(source, io=ProgramIO(files, cwd="/home/user"))
    assert run.stdout == "read: wheat 12\n"
    assert files.files["/home/user/notes.txt"] == b"wheat 12\n"


def test_fopen_without_files_returns_null() -> None:
    run = run_c(program('FILE *f = fopen("x.txt", "r");\nprintf("%d\\n", f == NULL);\nreturn 0;'))
    assert run.stdout == "1\n"


# =========================================================================== several files

def test_static_functions_are_private_to_their_unit() -> None:
    files = {
        "main.c": '#include <stdio.h>\nint other(void);\nstatic int helper(void) { return 1; }\n'
                  'int main(void) { printf("%d %d\\n", helper(), other()); return 0; }\n',
        "other.c": 'static int helper(void) { return 2; }\nint other(void) { return helper(); }\n',
    }
    assert run_c(files).stdout == "1 2\n"


def test_static_globals_are_separate_per_unit() -> None:
    files = {
        "main.c": '#include <stdio.h>\nstatic int count = 10;\nint bump(void);\n'
                  'int main(void) {\n  bump();\n  bump();\n'
                  '  printf("%d %d\\n", count, bump());\n  return 0;\n}\n',
        "other.c": 'static int count = 0;\nint bump(void) { return ++count; }\n',
    }
    assert run_c(files).stdout == "10 3\n"


def test_extern_globals_across_units() -> None:
    files = {
        "main.c": '#include <stdio.h>\nextern int harvest;\nvoid add(int n);\n'
                  'int main(void) { add(3); harvest += 1; printf("%d\\n", harvest); return 0; }\n',
        "store.c": 'int harvest = 5;\nvoid add(int n) { harvest += n; }\n',
    }
    assert run_c(files).stdout == "9\n"


def test_three_file_program_with_a_header() -> None:
    files = {
        "field.h": '#ifndef FIELD_H\n#define FIELD_H\nstruct Field { int w, h; };\n'
                   'int field_area(struct Field f);\nextern int fields_made;\n'
                   'struct Field make_field(int w, int h);\n#endif\n',
        "field.c": '#include "field.h"\nint fields_made = 0;\n'
                   'struct Field make_field(int w, int h) { struct Field f = {w, h}; '
                   'fields_made++; return f; }\n',
        "area.c": '#include "field.h"\nint field_area(struct Field f) { return f.w * f.h; }\n',
        "main.c": '#include <stdio.h>\n#include "field.h"\nint main(void) {\n'
                  '  struct Field a = make_field(3, 4), b = make_field(2, 5);\n'
                  '  printf("%d %d %d\\n", field_area(a), field_area(b), fields_made);\n'
                  '  return 0;\n}\n',
    }
    run = run_c(files)
    assert run.stdout == "12 10 2\n"
    assert run.exit_code == 0
