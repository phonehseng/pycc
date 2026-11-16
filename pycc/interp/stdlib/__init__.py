"""pycc's simulated C standard library.

Three entry points:

* ``install(registry)`` registers every library function and variable (``stdin``,
  ``stdout``, ``stderr``, ``errno``) with its prototype, header and concept.
* ``system_headers()`` returns the texts behind ``#include <stdio.h>`` and friends.
  Each prototype in them is exactly the one registered here (the tests check it).
* ``finalize(machine)`` is called by the interpreter when a program ends, so the
  library can report what the program left behind (files never closed).

Each header has its own module (``stdio``, ``stdlib_``, ``string_``...; the trailing
underscore keeps a module from shadowing Python's own ``string`` or ``math``).
Implementations only talk to the ``Machine`` facade of ``interp.builtins``, so the
whole library is tested against ``fake_machine.FakeMachine`` without an interpreter.
"""
from __future__ import annotations

from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.stdlib import assert_, ctype_, errno_, math_, stdio, stdlib_, string_, time_
from pycc.interp.stdlib.headers import STANDARD_HEADERS, load_headers, system_headers

__all__ = ["STANDARD_HEADERS", "default_toolchain", "finalize", "install", "load_headers",
           "make_registry", "system_headers"]

_MODULES = (stdio, stdlib_, string_, ctype_, math_, errno_, assert_, time_)


def install(registry: BuiltinRegistry) -> None:
    """Register the whole standard library. Calling it twice is harmless."""
    for module in _MODULES:
        module.register(registry)


def make_registry() -> BuiltinRegistry:
    """A new registry holding just the standard library (hosts may add device functions to it)."""
    registry = BuiltinRegistry()
    install(registry)
    return registry


def default_toolchain():
    """A ``Toolchain`` with the whole C standard library and its headers, and no devices."""
    from pycc.interp.toolchain import Toolchain

    return Toolchain(make_registry(), system_headers())


def finalize(machine: Machine) -> None:
    """End-of-run checks: one ``file_not_closed`` warning per file left open."""
    stdio.finalize(machine)
