"""Review tests: tricky C semantics, scopes, warnings and speed of the interpreter.

Every expected output is what gcc on 64-bit Linux prints for the same program. The
last tests guard two things a user would notice at once: warnings pointing at the
wrong line, and a long loop that takes ages.
"""
from __future__ import annotations

import os
import time

import pytest

from pycc.core.events import RunWarning, Statement
from pycc.interp.interpreter import Interpreter, run_to_completion
from pycc.interp.stdlib import make_registry
from tests.interp.test_interpreter_support import (
    build, events_of, program, run_c, trap_code, warning_codes,
)

STD = ("stdio.h", "stdlib.h", "string.h")


def output_of(body: str, before: str = "", includes: tuple[str, ...] = STD) -> str:
    """Run ``main`` with ``body``; the run must end normally. Returns stdout."""
    run = run_c(program(body, includes, before))
    assert trap_code(run) is None, run.summary.trap
    return run.stdout


# --- values: structs, arrays of structs, pointers to pointers ------------------------------

POINT = "struct P { int x; int y; };\n"


def test_modifying_a_struct_copy_leaves_the_original_alone() -> None:
    body = ('struct P p = {1, 2};\nstruct P q = p;\nq.x = 9;\n'
            'printf("%d %d\\n", p.x, q.x);\nreturn 0;')
    assert output_of(body, POINT) == "1 9\n"


def test_a_struct_parameter_is_a_copy() -> None:
    before = POINT + "void bump(struct P s) { s.x = 100; }\n"
    body = 'struct P p = {1, 2};\nbump(p);\nprintf("%d\\n", p.x);\nreturn 0;'
    assert output_of(body, before) == "1\n"


def test_a_returned_struct_is_a_fresh_copy() -> None:
    before = POINT + "struct P keep;\nstruct P get(void) { return keep; }\n"
    body = ('keep.x = 5;\nstruct P r = get();\nr.x = 6;\n'
            'printf("%d %d\\n", keep.x, r.x);\nreturn 0;')
    assert output_of(body, before) == "5 6\n"


def test_array_member_of_a_struct_in_an_array_is_an_lvalue() -> None:
    before = "struct Row { int id; int m[3]; };\n"
    body = ('struct Row rows[2];\n'
            'for (int i = 0; i < 2; i++)\n  for (int j = 0; j < 3; j++)\n'
            '    rows[i].m[j] = i * 10 + j;\n'
            'rows[1].m[2] += 100;\n'
            'printf("%d %d %d\\n", rows[0].m[1], rows[1].m[0], rows[1].m[2]);\nreturn 0;')
    assert output_of(body, before) == "1 10 112\n"


def test_writing_through_a_pointer_to_a_pointer() -> None:
    body = ('int x = 5, y = 8;\nint *p = &x;\nint **pp = &p;\n**pp = 7;\n*pp = &y;\n'
            '**pp += 1;\nprintf("%d %d %d\\n", x, y, *p);\nreturn 0;')
    assert output_of(body) == "7 9 9\n"


# --- integers: signedness, shifts, comparisons, conversions -----------------------------------

def test_plain_char_is_signed_and_promotes_to_int() -> None:
    body = ('char c = 200;\nunsigned char u = 200;\nsigned char s = -1;\n'
            'unsigned int w = s;\n'
            'printf("%d %d %d %d %u\\n", c, u, c + 1, (unsigned char)s, w);\nreturn 0;')
    assert output_of(body) == "-56 200 -55 255 4294967295\n"


def test_shifting_negative_values_and_bad_counts() -> None:
    body = 'int k = -8;\nint n = -1;\nprintf("%d %d %d\\n", k >> 1, n << 1, -1 >> 31);\nreturn 0;'
    assert output_of(body) == "-4 -2 -1\n"
    run = run_c(program('int n = -1;\nint x = 1 << n;\nreturn x == 0;', STD))
    assert "shift_overflow" in warning_codes(run)


def test_comparing_unsigned_with_negative_numbers() -> None:
    body = ('unsigned int u = 1;\nint n = -1;\nlong l = -1;\n'
            'printf("%d %d %d %d\\n", n < u, -1 < 1u, l < u, (unsigned char)-1 == 255);\n'
            'return 0;')
    assert output_of(body) == "0 0 1 1\n"


def test_float_to_int_conversions_truncate_and_wrap() -> None:
    body = ('double d = -3.9;\nint t = (int)d;\nunsigned char c = (unsigned char)300.7;\n'
            'long big = (long)1e18;\nprintf("%d %d %ld\\n", t, c, big);\nreturn 0;')
    assert output_of(body) == "-3 44 1000000000000000000\n"


def test_switch_on_unsigned_matches_a_negative_case_after_wrapping() -> None:
    body = ('unsigned int u = 4294967295u;\nchar c = -1;\n'
            'switch (u) { case -1: printf("u "); break; default: printf("no "); }\n'
            'switch (c) { case -1: printf("c\\n"); break; default: printf("no\\n"); }\n'
            'return 0;')
    assert output_of(body) == "u c\n"


def test_value_of_a_struct_assignment_and_conditional() -> None:
    body = ('struct P a = {1, 2}, b = {3, 4};\nint pick = 0;\n'
            'printf("%d ", (a = b).y);\nstruct P r = pick ? a : b;\n'
            'printf("%d\\n", r.x);\nreturn 0;')
    assert output_of(body, POINT) == "4 3\n"


def test_exit_inside_a_qsort_comparator_ends_the_program_cleanly() -> None:
    before = 'int cmp(const void *a, const void *b) {\n  exit(4);\n}\n'
    run = run_c(program('int a[3] = {3, 1, 2};\nqsort(a, 3, sizeof(int), cmp);\nreturn 0;',
                        STD, before))
    assert run.exit_code == 4 and trap_code(run) is None


def test_compound_division_by_zero_traps_at_its_statement() -> None:
    run = run_c(program('int x = 5;\nint zero = 0;\nx /= zero;\nreturn x;', STD))
    assert trap_code(run) == "division_by_zero" and run.exit_code == 136
    assert run.summary.trap.location.line == 8           # x /= zero;


# --- program shape: argv, string literals ------------------------------------------------

def test_main_with_char_star_star_argv_ends_with_null() -> None:
    source = ('#include <stdio.h>\nint main(int argc, char **argv) {\n'
              '  argv[1][0] = \'H\';\n'
              '  printf("%d %s %s %d\\n", argc, argv[0], argv[1], argv[argc] == 0);\n'
              '  return argc;\n}\n')
    run = run_c(source, argv=["prog", "hi"])
    assert run.stdout == "2 prog Hi 1\n" and run.exit_code == 2


def test_main_with_char_star_array_argv() -> None:
    source = ('#include <stdio.h>\nint main(int argc, char *argv[]) {\n'
              '  for (int i = 0; i < argc; i++) printf("[%s]", argv[i]);\n'
              '  printf("\\n");\n  return 0;\n}\n')
    assert run_c(source, argv=["prog", "a", "bc"]).stdout == "[prog][a][bc]\n"


def test_identical_string_literals_share_one_address_across_units() -> None:
    files = {
        "main.c": ('#include <stdio.h>\nconst char *other(void);\n'
                   'int main(void) {\n  const char *mine = "hey";\n'
                   '  printf("%d %s\\n", mine == other(), other());\n  return 0;\n}\n'),
        "other.c": 'const char *other(void) { return "hey"; }\n',
    }
    assert run_c(files).stdout == "1 hey\n"


# --- scopes: no local outlives its block -------------------------------------------------

def test_break_continue_and_return_release_nested_block_locals() -> None:
    before = ('int pick(int n) {\n'
              '  for (int i = 0; ; i++) {\n'
              '    int big[200];\n    big[0] = n;\n'
              '    { int inner[200]; inner[0] = 1; if (i == 0) return big[0] & inner[0]; }\n'
              '  }\n}\n')
    body = ('int s = 0;\n'
            'for (int i = 0; i < 3000; i++) {\n'
            '  int a[200];\n  a[0] = i;\n'
            '  if (i % 3 == 0) continue;\n'
            '  { int b[200]; b[0] = 1; if (i == 2999) break; }\n'
            '  s += pick(a[0]);\n'
            '}\n'
            'int after = 1;\n'
            'printf("%d %d\\n", s, after);\nreturn 0;')
    assert output_of(body, before) == "999 1\n"


def test_only_live_locals_are_visible_after_leaving_blocks() -> None:
    source = ('int main(void) {\n'
              '  int keep = 1;\n'
              '  for (int i = 0; i < 2; i++) { int inner = i; if (inner) break; }\n'
              '  while (1) { int w = 0; { int deep = w; break; } }\n'
              '  return keep;\n'
              '}\n')
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    generator = interpreter.run(["prog"])
    event = next(generator)
    while not (isinstance(event, Statement) and event.location.line == 5):
        event = next(generator)
    frame = interpreter.stack_frames()[-1]
    assert [var.name for var in frame.locals] == ["keep"]
    alive = [obj.name for obj in interpreter.memory.frames[-1].objects if obj.alive]
    assert alive == ["keep"]
    generator.close()


def test_struct_temporaries_in_a_loop_condition_do_not_fill_the_stack() -> None:
    before = ("struct Box { int a[8]; };\n"
              "struct Box make(int v) { struct Box b = {{0}}; b.a[0] = v; return b; }\n")
    body = ('int n = 0;\nwhile (make(n).a[0] < 20000) n++;\n'
            'printf("%d\\n", n);\nreturn 0;')
    assert output_of(body, before) == "20000\n"


# --- warnings: the right line, once -------------------------------------------------------

def test_overflow_in_a_callee_points_at_the_callee_line() -> None:
    before = 'int add(int a, int b) {\n  return a + b;\n}\n'
    run = run_c(program('int x = add(2147483647, 1);\nprintf("%d\\n", x);\nreturn 0;',
                        STD, before))
    warnings = [d for d in run.summary.warnings if d.code == "signed_overflow"]
    assert len(warnings) == 1
    assert warnings[0].location.line == 5           # 'return a + b;' (after 3 includes)
    assert run.stdout == "-2147483648\n"


def test_uninitialised_reads_in_a_loop_warn_once() -> None:
    source = ('int main(void) {\n  int s = 0;\n'
              '  for (int i = 0; i < 5; i++) {\n    int x;\n    s = x;\n  }\n'
              '  return 0;\n}\n')
    events, _, _ = events_of(source)
    codes = [event.diagnostic.code for event in events if isinstance(event, RunWarning)]
    assert codes == ["uninitialized_read"]


# --- speed -----------------------------------------------------------------------------------

@pytest.mark.parametrize("loop", [
    "for (int i = 0; i < 100000; i++) {\n  s += i;\n}",
    "int i = 0;\nwhile (i < 100000) i += 1;",
])
def test_a_200000_statement_loop_runs_in_under_two_seconds(loop: str) -> None:
    registry = make_registry()
    compiled = build(program(f"long s = 0;\n{loop}\nreturn (int)(s & 1);", ()), registry)
    # CPU time of this process only, so other programs running on the machine do
    # not make the test flaky; compiling is not part of the measurement.
    start = time.process_time()
    run = run_to_completion(compiled, registry)
    elapsed = time.process_time() - start
    assert run.summary.statements >= 200_000
    limit = 12.0 if os.environ.get("CI") else 2.0    # hosted CI machines are shared and slower
    assert elapsed < limit, f"{run.summary.statements} statements took {elapsed:.2f}s"
