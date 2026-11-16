"""<ctype.h> classifiers and conversions, <math.h> results and errno behaviour."""
from __future__ import annotations

import math

import pytest

from pycc.interp.stdlib.errno_ import EDOM, ERANGE
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.stdlib.stdio import EOF


@pytest.fixture
def m() -> FakeMachine:
    return FakeMachine()


# --- ctype -------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,true_chars,false_chars", [
    ("isalpha", "aZ", "1 _"),
    ("isdigit", "09", "a "),
    ("isalnum", "a9Z", "_ -"),
    ("isspace", " \t\n\v\f\r", "a0"),
    ("isupper", "AZ", "a1"),
    ("islower", "az", "A1"),
    ("ispunct", "!.,_~", "a1 "),
    ("isxdigit", "09afAF", "gG "),
    ("isprint", " a~", "\n\x7f"),
])
def test_classifiers(m: FakeMachine, name: str, true_chars: str, false_chars: str) -> None:
    for char in true_chars:
        assert m.call(name, m.int_(ord(char))).raw == 1, char
    for char in false_chars:
        assert m.call(name, m.int_(ord(char))).raw == 0, char


@pytest.mark.parametrize("name", ["isalpha", "isdigit", "isspace", "isprint"])
def test_classifiers_reject_eof_and_high_bytes(m: FakeMachine, name: str) -> None:
    assert m.call(name, m.int_(EOF)).raw == 0
    assert m.call(name, m.int_(0xE9)).raw == 0


def test_toupper_tolower(m: FakeMachine) -> None:
    assert m.call("toupper", m.int_(ord("q"))).raw == ord("Q")
    assert m.call("toupper", m.int_(ord("7"))).raw == ord("7")
    assert m.call("tolower", m.int_(ord("Q"))).raw == ord("q")
    assert m.call("tolower", m.int_(EOF)).raw == EOF


# --- math ----------------------------------------------------------------------------------------

def call(m: FakeMachine, name: str, *values: float) -> float:
    return m.call(name, *(m.double(v) for v in values)).raw


@pytest.mark.parametrize("name,args,expected", [
    ("sqrt", (16.0,), 4.0),
    ("pow", (2.0, 10.0), 1024.0),
    ("pow", (-2.0, 3.0), -8.0),
    ("fabs", (-3.5,), 3.5),
    ("floor", (2.7,), 2.0),
    ("floor", (-2.1,), -3.0),
    ("ceil", (2.1,), 3.0),
    ("ceil", (-2.7,), -2.0),
    ("round", (2.5,), 3.0),
    ("round", (-2.5,), -3.0),
    ("round", (0.49999999999999994,), 0.0),
    ("round", (1.4,), 1.0),
    ("sin", (0.0,), 0.0),
    ("cos", (0.0,), 1.0),
    ("tan", (0.0,), 0.0),
    ("atan2", (1.0, 1.0), math.pi / 4),
    ("fmod", (7.5, 2.0), 1.5),
    ("fmod", (-7.5, 2.0), -1.5),
    ("exp", (1.0,), math.e),
    ("log", (math.e,), 1.0),
])
def test_math_values(m: FakeMachine, name: str, args: tuple, expected: float) -> None:
    assert call(m, name, *args) == pytest.approx(expected)
    assert m.errno == 0


def test_floor_and_ceil_keep_negative_zero(m: FakeMachine) -> None:
    assert math.copysign(1.0, call(m, "ceil", -0.5)) == -1.0


def test_floor_of_infinity_is_infinity(m: FakeMachine) -> None:
    assert call(m, "floor", math.inf) == math.inf


@pytest.mark.parametrize("name,args", [
    ("sqrt", (-1.0,)), ("log", (-1.0,)), ("fmod", (1.0, 0.0)), ("pow", (-8.0, 1.0 / 3.0)),
    ("sin", (math.inf,)),
])
def test_domain_errors_return_nan_and_set_edom(m: FakeMachine, name: str, args: tuple) -> None:
    assert math.isnan(call(m, name, *args))
    assert m.errno == EDOM


def test_sqrt_of_negative_prints_minus_nan(m: FakeMachine) -> None:
    value = m.call("sqrt", m.double(-1.0))
    m.call("printf", m.ptr(m.literal("%f")), value)
    assert m.stdout == "-nan"


@pytest.mark.parametrize("name,args,expected", [
    ("log", (0.0,), -math.inf), ("exp", (1000.0,), math.inf), ("pow", (10.0, 400.0), math.inf),
    ("pow", (0.0, -1.0), math.inf),
])
def test_range_errors_return_infinity_and_set_erange(m: FakeMachine, name: str, args: tuple,
                                                     expected: float) -> None:
    assert call(m, name, *args) == expected
    assert m.errno == ERANGE


def test_nan_argument_passes_through_without_errno(m: FakeMachine) -> None:
    assert math.isnan(call(m, "sqrt", math.nan))
    assert m.errno == 0
