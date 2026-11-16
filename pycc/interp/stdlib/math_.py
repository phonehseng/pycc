"""``<math.h>``: floating-point functions with C's (glibc's) error behaviour.

Python's ``math`` raises exceptions where C quietly returns a special value and sets
``errno``. The user must see C's behaviour, so each function here catches the
problem and answers the way glibc does:

* a *domain error* (``sqrt(-1.0)``, ``log(-1.0)``, ``fmod(x, 0.0)``) returns NaN
  and sets ``errno = EDOM``; glibc's NaN from these is negative, so printf shows
  ``-nan``;
* a *pole or range error* (``log(0.0)``, ``exp(1000.0)``, ``pow(10.0, 400.0)``)
  returns plus or minus infinity and sets ``errno = ERANGE``.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from pycc.core.concepts import Concept
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.stdlib.errno_ import EDOM, ERANGE
from pycc.interp.stdlib.headers import LibraryFunction, register_functions
from pycc.interp.values import Value

HEADER = "math.h"

NEG_NAN = math.copysign(math.nan, -1.0)


def _domain_error(machine: Machine) -> float:
    """errno = EDOM and the negative NaN glibc returns."""
    machine.set_errno(EDOM)
    return NEG_NAN


def _range_error(machine: Machine, value: float) -> float:
    """errno = ERANGE and the infinity (or zero) that stands in for the result."""
    machine.set_errno(ERANGE)
    return value


def _sqrt(machine: Machine, x: float) -> float:
    if x < 0:
        return _domain_error(machine)
    return math.sqrt(x)


def _is_odd_integer(y: float) -> bool:
    return math.isfinite(y) and y == int(y) and int(y) % 2 == 1


def _pow(machine: Machine, x: float, y: float) -> float:
    if x == 0.0 and y < 0:
        # A pole error: 0 to a negative power is "1 / 0". The sign survives only
        # for odd integer powers, as in pow(-0.0, -1.0) == -inf.
        return _range_error(machine, math.copysign(math.inf, x) if _is_odd_integer(y)
                            else math.inf)
    if x < 0 and math.isfinite(y) and y != int(y):
        return _domain_error(machine)
    try:
        return math.pow(x, y)
    except OverflowError:
        negative = x < 0 and _is_odd_integer(y)
        return _range_error(machine, -math.inf if negative else math.inf)


def _round(machine: Machine, x: float) -> float:
    """Round half away from zero (Python's round() rounds half to even)."""
    if not math.isfinite(x):
        return x
    whole = math.floor(abs(x))
    if abs(x) - whole >= 0.5:
        whole += 1
    return math.copysign(float(whole), x)


def _whole(function: Callable[[float], int]) -> Callable[[Machine, float], float]:
    """floor/ceil: Python returns an int and fails on inf/nan; C returns a double."""
    def apply(machine: Machine, x: float) -> float:
        if not math.isfinite(x):
            return x
        result = float(function(x))
        # C keeps the sign of zero: floor(-0.0) and ceil(-0.5) are both -0.0.
        return math.copysign(0.0, x) if result == 0 else result
    return apply


def _trig(function: Callable[[float], float]) -> Callable[[Machine, float], float]:
    """sin/cos/tan: infinity has no angle, which is a domain error."""
    def apply(machine: Machine, x: float) -> float:
        if math.isinf(x):
            return _domain_error(machine)
        return function(x)
    return apply


def _fmod(machine: Machine, x: float, y: float) -> float:
    if y == 0.0 or math.isinf(x):
        return _domain_error(machine)
    return math.fmod(x, y)


def _exp(machine: Machine, x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return _range_error(machine, math.inf)


def _log(machine: Machine, x: float) -> float:
    if x == 0.0:
        return _range_error(machine, -math.inf)
    if x < 0:
        return _domain_error(machine)
    return math.log(x)


def _fabs(machine: Machine, x: float) -> float:
    return math.fabs(x)


def _atan2(machine: Machine, y: float, x: float) -> float:
    return math.atan2(y, x)


def _wrap(body: Callable[..., float],
          nan_in_nan_out: bool = True) -> Callable[[Machine, list[Value]], Value]:
    """Turn ``body(machine, *floats) -> float`` into a builtin implementation.

    NaN arguments pass straight through (C returns NaN without touching errno),
    except where ``nan_in_nan_out`` is False: C's ``pow(1.0, NAN)`` and
    ``pow(NAN, 0.0)`` are 1.0, because those results do not depend on the NaN.
    """
    def impl(machine: Machine, args: list[Value]) -> Value:
        values = [float(arg.raw) for arg in args]
        if nan_in_nan_out and any(math.isnan(v) for v in values):
            return machine.make_double(math.nan)
        return machine.make_double(body(machine, *values))
    impl.__doc__ = body.__doc__
    return impl


_BODIES: dict[str, tuple[str, Callable[..., float]]] = {
    "sqrt": ("double sqrt(double x);", _sqrt),
    "pow": ("double pow(double x, double y);", _pow),
    "fabs": ("double fabs(double x);", _fabs),
    "floor": ("double floor(double x);", _whole(math.floor)),
    "ceil": ("double ceil(double x);", _whole(math.ceil)),
    "round": ("double round(double x);", _round),
    "sin": ("double sin(double x);", _trig(math.sin)),
    "cos": ("double cos(double x);", _trig(math.cos)),
    "tan": ("double tan(double x);", _trig(math.tan)),
    "atan2": ("double atan2(double y, double x);", _atan2),
    "fmod": ("double fmod(double x, double y);", _fmod),
    "exp": ("double exp(double x);", _exp),
    "log": ("double log(double x);", _log),
}

FUNCTIONS: tuple[LibraryFunction, ...] = tuple(
    LibraryFunction(name, prototype, _wrap(body, nan_in_nan_out=name != "pow"),
                    Concept.FLOATING)
    for name, (prototype, body) in _BODIES.items()
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<math.h>`` function."""
    register_functions(registry, HEADER, FUNCTIONS)
