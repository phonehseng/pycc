"""``<ctype.h>``: classifying and converting single characters (the "C" locale).

Each function takes an ``int`` holding an ``unsigned char`` value or ``EOF``. The
classifiers answer 1 for true and 0 for false. C only promises "non-zero" for true
(glibc answers numbers such as 1024); 1 keeps printed results easy to read and
programs that test ``if (isdigit(c))`` behave identically.
"""
from __future__ import annotations

from collections.abc import Callable

from pycc.core.concepts import Concept
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.stdlib.headers import LibraryFunction, register_functions
from pycc.interp.values import Value

HEADER = "ctype.h"

_SPACE = frozenset(b" \t\n\v\f\r")
_DIGITS = frozenset(b"0123456789")
_UPPER = frozenset(range(ord("A"), ord("Z") + 1))
_LOWER = frozenset(range(ord("a"), ord("z") + 1))
_ALPHA = _UPPER | _LOWER
_ALNUM = _ALPHA | _DIGITS
_XDIGIT = _DIGITS | frozenset(b"abcdefABCDEF")
_PRINT = frozenset(range(32, 127))
_PUNCT = _PRINT - _ALNUM - {ord(" ")}

# name -> the set of character codes for which it answers true
CLASSES: dict[str, frozenset[int]] = {
    "isalpha": _ALPHA,
    "isdigit": _DIGITS,
    "isalnum": _ALNUM,
    "isspace": _SPACE,
    "isupper": _UPPER,
    "islower": _LOWER,
    "ispunct": _PUNCT,
    "isxdigit": _XDIGIT,
    "isprint": _PRINT,
}


def _classifier(members: frozenset[int]) -> Callable[[Machine, list[Value]], Value]:
    """Build the implementation of one ``isxxx`` function."""
    def classify(machine: Machine, args: list[Value]) -> Value:
        return machine.make_int(1 if args[0].raw in members else 0)
    return classify


def c_toupper(machine: Machine, args: list[Value]) -> Value:
    """``int toupper(int c)``: 'a'..'z' become 'A'..'Z'; anything else is unchanged."""
    code = args[0].raw
    return machine.make_int(code - 32 if code in _LOWER else code)


def c_tolower(machine: Machine, args: list[Value]) -> Value:
    """``int tolower(int c)``: 'A'..'Z' become 'a'..'z'; anything else is unchanged."""
    code = args[0].raw
    return machine.make_int(code + 32 if code in _UPPER else code)


FUNCTIONS: tuple[LibraryFunction, ...] = tuple(
    LibraryFunction(name, f"int {name}(int c);", _classifier(members), Concept.STRINGS)
    for name, members in CLASSES.items()
) + (
    LibraryFunction("toupper", "int toupper(int c);", c_toupper, Concept.STRINGS),
    LibraryFunction("tolower", "int tolower(int c);", c_tolower, Concept.STRINGS),
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<ctype.h>`` function."""
    register_functions(registry, HEADER, FUNCTIONS)
