"""Review tests for format string checking: the common conversions, stars and %n."""
from __future__ import annotations

import pytest

from pycc.interp.c_types import CHAR, DOUBLE, INT, LONG, SIZE_T, VOID_PTR, PointerType
from pycc.interp.format_check import PRINTF, SCANF, check_format, parse_format
from tests.interp.test_checker import body, build

STRING = PointerType(CHAR)


def messages(kind: str, text: str, *types: object) -> list[str]:
    return [issue.message for issue in check_format(kind, text, list(types), 2)]


def test_review_common_printf_conversions_accept_their_types() -> None:
    text = "%% %5.2f %-10s %ld %zu %p %c\n"
    assert messages(PRINTF, text, DOUBLE, STRING, LONG, SIZE_T, VOID_PTR, INT) == []


def test_review_percent_percent_consumes_no_argument() -> None:
    assert parse_format("100%% done %d", PRINTF).conversions[0].char == "d"
    assert messages(PRINTF, "100%%\n", INT) == ["too many arguments for format"]


def test_review_star_width_and_precision_take_int_arguments_first() -> None:
    assert messages(PRINTF, "%*d %.*f %-*.*s", INT, INT, INT, DOUBLE,
                    INT, INT, STRING) == []
    assert messages(PRINTF, "%*d", SIZE_T, INT) == [
        "field width specifier '*' expects argument of type 'int', but argument 2 has type "
        "'unsigned long'"]


def test_review_star_without_its_argument_is_reported() -> None:
    assert messages(PRINTF, "%.*f") == [
        "field precision specifier '*' expects a matching 'int' argument",
        "format '%.*f' expects a matching 'double' argument"]


@pytest.mark.parametrize("text, given, expected", [
    ("%ld", INT, "long int"),
    ("%zu", INT, "size_t"),
    ("%c", STRING, "int"),
    ("%5.2f", INT, "double"),
    ("%-10s", INT, "char *"),
    ("%p", INT, "void *"),
])
def test_review_mismatches_name_the_expected_type(text: str, given: object,
                                                  expected: str) -> None:
    [message] = messages(PRINTF, text, given)
    assert f"expects argument of type '{expected}'" in message


def test_review_printf_n_follows_its_length_modifier() -> None:
    assert messages(PRINTF, "%ln", PointerType(LONG)) == []
    assert messages(PRINTF, "%n", PointerType(LONG)) == [
        "format '%n' expects argument of type 'int *', but argument 2 has type 'long *'"]


def test_review_format_ending_inside_a_conversion_says_it_lacks_a_type() -> None:
    assert parse_format("%l", PRINTF).problems[0][1] == "conversion lacks type at end of format"
    assert parse_format("50%", PRINTF).problems[0][1] == "spurious trailing '%' in format"


def test_review_scanf_suppressed_conversion_takes_no_argument() -> None:
    assert messages(SCANF, "%*d %d", PointerType(INT)) == []


def test_review_printf_call_with_every_common_conversion_is_clean() -> None:
    built = build(body("int w = 5; char c = 'x'; float f = 1.5f;"
                       "printf(\"%% %5.2f %-10s %ld %zu %p %c %*d\\n\", f, \"a\", 3L,"
                       " sizeof(int), (void *)0, c, w, w); return 0;"))
    assert built.diagnostics == []
