"""``<assert.h>``: the function behind ``assert(e)``.

C's ``assert`` is a macro that stringizes its argument, and pycc's preprocessor
has no ``#`` operator. Instead, when ``<assert.h>`` defines ``__CV_ASSERT__`` the
preprocessor rewrites ``assert(e)`` into ``__cv_assert((e), "e", __FILE__, __LINE__,
__func__)``. This function then does what glibc's assert does: print

    ./a.out: main.c:12: main: Assertion `count > 0' failed.

on stderr and stop the program (trap ``assert_failed``, exit code 134 "Aborted").
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pycc.core.concepts import Concept
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.stdlib.headers import LibraryFunction, register_functions
from pycc.interp.stdlib.stdio import write_console
from pycc.interp.values import Value

HEADER = "assert.h"


def _text(machine: Machine, addr: int, fallback: str) -> str:
    """A string argument, or ``fallback`` if the rewritten call passed NULL."""
    if addr == 0:
        return fallback
    return machine.memory.read_cstring(addr, machine.memory.size).decode("utf-8", "replace")


def c_cv_assert(machine: Machine, args: list[Value]) -> Generator[Any, Any, None]:
    """``void __cv_assert(int ok, const char *expr, const char *file, int line,
    const char *func)``: nothing when ``ok`` is true, else message and trap."""
    if args[0].raw:
        return None
    expr = _text(machine, args[1].raw, "?")
    file = _text(machine, args[2].raw, "?")
    line = args[3].raw
    func = _text(machine, args[4].raw, "?")
    program = getattr(machine.io, "argv0", None) or "a.out"
    where = f"{file}:{line}: {func}: Assertion `{expr}' failed."
    yield from write_console(machine, "stderr", f"{program}: {where}\n")
    machine.trap(
        "assert_failed",
        where,
        f"assert({expr}) states something the program relies on, and it was false at "
        f"{file}:{line} in {func}(), so the program stopped on purpose. Find out which "
        "value is not what you expected, for example by printing it just before the "
        "assert.",
        Concept.ERROR_HANDLING, expression=expr, file=file, line=line, function=func)
    return None


FUNCTIONS: tuple[LibraryFunction, ...] = (
    LibraryFunction("__cv_assert", "void __cv_assert(int ok, const char *expr, const char *file, "
                    "int line, const char *func);", c_cv_assert, Concept.ERROR_HANDLING),
)


def register(registry: BuiltinRegistry) -> None:
    """Register ``__cv_assert``."""
    register_functions(registry, HEADER, FUNCTIONS)
