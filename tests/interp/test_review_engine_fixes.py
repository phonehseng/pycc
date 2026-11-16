"""Review of the engine fixes through the real toolchain.

Each test is a seam the first round left open: a conversion that still printed the
"right" answer by accident, or two views of one pointer that disagreed.
"""
from __future__ import annotations

import pytest

from pycc.interp.c_types import (
    FUNCTION_ADDRESS_BASE, INT, FunctionType, PointerType, convert_value, format_value,
    function_address,
)
from pycc.interp.memory import Memory
from pycc.interp.toolchain import CompileOptions
from tests.interp.test_checker import build as check_source
from tests.interp.test_interpreter_support import program, run_c, trap_code, warning_codes


# --- N3: every conversion that reads the other register file prints garbage --------------

def test_char_conversion_with_a_double_never_prints_the_letter() -> None:
    # printf("%c", 65.0) reads an integer register nobody wrote; 'A' would pass a check.
    run = run_c(program('printf("%c\\n", 65.0);\nreturn 0;'))
    assert run.stdout != "A\n"
    assert warning_codes(run) == ["format_argument_type"]


def test_char_conversion_with_an_int_still_prints_the_letter() -> None:
    run = run_c(program('printf("%c%c\\n", 65, \'B\');\nreturn 0;'))
    assert run.stdout == "AB\n"
    assert warning_codes(run) == []


def test_pointer_conversion_with_a_double_prints_garbage_and_warns() -> None:
    run = run_c(program('printf("%p\\n", 4.5);\nreturn 0;'))
    assert run.stdout not in ("0x4\n", "(nil)\n")
    assert warning_codes(run) == ["format_argument_type"]


def test_pointer_conversion_with_an_int_keeps_the_value_quietly() -> None:
    # Ints and pointers share the integer registers (the checker already flags it).
    run = run_c(program('printf("%p\\n", 64);\nreturn 0;'))
    assert run.stdout == "0x40\n"
    assert warning_codes(run) == []


def test_string_conversion_with_a_double_is_a_wild_pointer_not_null() -> None:
    # 0.0 used to become NULL and print "(null)", a plausible-looking answer.
    run = run_c(program('printf("%s\\n", 0.0);\nreturn 0;'))
    assert "(null)" not in run.stdout
    assert trap_code(run) == "wild_pointer"
    assert warning_codes(run) == ["format_argument_type"]


def test_char_garbage_is_the_same_for_the_same_seed() -> None:
    source = program('printf("%c|%p\\n", 65.0, 1.5);\nreturn 0;')
    assert run_c(source).stdout == run_c(source).stdout


# --- D3/D9: one function pointer, one address, everywhere ------------------------------

def test_function_pointer_cast_to_void_pointer_prints_its_code_address() -> None:
    # printf("%p", (void *)main) is the portable spelling; it must match printf("%p", fp).
    run = run_c(program('int (*fp)(void) = main;\n'
                        'printf("%p %p\\n", (void *)fp, fp);\nreturn 0;'))
    left, right = run.stdout.split()
    assert left == right
    assert int(left, 16) >= FUNCTION_ADDRESS_BASE


def test_function_pointer_survives_a_round_trip_through_void_pointer() -> None:
    source = program('void *kept = (void *)twice;\n'
                     'int (*back)(int) = (int (*)(int))kept;\n'
                     'printf("%d\\n", back(21));\nreturn 0;',
                     before="int twice(int x) { return 2 * x; }\n")
    run = run_c(source)
    assert run.stdout == "42\n" and trap_code(run) is None


def test_function_pointer_cast_to_long_shows_the_same_address() -> None:
    source = program('long n = (long)twice;\nprintf("%lx %p\\n", n, twice);\nreturn 0;',
                     before="int twice(int x) { return 2 * x; }\n")
    left, right = run_c(source).stdout.split()
    assert "0x" + left == right


def test_variables_panel_shows_the_address_printf_prints() -> None:
    memory = Memory(seed=1)
    frame = memory.push_frame("main")
    fp_type = PointerType(FunctionType(INT, []))
    slot = memory.alloc_local(frame, 8, 8, "fp", fp_type)
    memory.store(slot, fp_type, 3)                     # function id 3 inside the machine
    assert format_value(memory, slot, fp_type) == f"0x{function_address(3):x}"


def test_function_address_conversion_keeps_null_and_unrelated_numbers() -> None:
    fp_type = PointerType(FunctionType(INT, []))
    void_ptr = PointerType(INT)
    assert convert_value(0, fp_type, void_ptr) == 0               # NULL stays NULL
    assert convert_value(0, void_ptr, fp_type) == 0
    assert convert_value(function_address(5), void_ptr, fp_type) == 5
    assert convert_value(0x1234, void_ptr, fp_type) == 0x1234     # not a code address


# --- N4: address_compare in the shapes beginners write --------------------------------

HEAD = "#include <stdio.h>\n"


@pytest.mark.parametrize("body", [
    '#define IS(x) ((x) == "a")\nint main(void) { char *s = "a"; return IS(s); }',
    'int main(void) { char *s = "a"; return s == "a" ? 1 : 0; }',
    'int f(const char *s) { return s == ("x"); }\nint main(void) { return f("x"); }',
    'char *g = "x";\nint main(void) { while (g != "x") { break; } return 0; }',
])
def test_string_literal_compare_warns_in_every_shape(body: str) -> None:
    codes = check_source(HEAD + body, options=CompileOptions()).codes()
    assert codes.count("address_compare") == 1


def test_ordering_a_string_literal_is_not_address_compare() -> None:
    # gcc's -Waddress is about == and != only.
    codes = check_source(HEAD + 'int main(void) { char *s = "a"; return s < "b"; }',
                         options=CompileOptions()).codes()
    assert "address_compare" not in codes
