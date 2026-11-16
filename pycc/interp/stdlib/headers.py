"""The standard headers: their texts and the declarations they promise.

The C files in ``interp/stdlib/headers/`` are hand-written so a user who types
``cat /usr/include/stdio.h`` reads a short, commented, honest header. The Python side
must agree with them exactly: every function a header declares is registered with the
very same prototype text, and every registered function appears in its header. The
test suite checks both directions, so a header and its implementation cannot drift.

This module also defines ``LibraryFunction``, the one record every stdlib module uses
to describe a function (name, prototype, implementation, concept), so registration is
a table, not a page of repeated ``registry.register(...)`` calls, and ``counted``, the
decorator that counts a construct (``string_call``, ``malloc``...) on every call.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pycc.core.concepts import Concept
from pycc.interp.builtins import BuiltinRegistry, Impl
from pycc.interp.toolchain import SystemHeaders

HEADER_DIR = Path(__file__).with_name("headers")

# Every header the simulated C library ships, in a fixed order.
STANDARD_HEADERS: tuple[str, ...] = (
    "stdio.h", "stdlib.h", "string.h", "ctype.h", "math.h", "errno.h", "stdbool.h",
    "stdint.h", "limits.h", "stddef.h", "assert.h", "time.h",
)


@dataclass(frozen=True)
class LibraryFunction:
    """One library function as its module describes it.

    ``format_arg``/``format_kind`` mark printf- and scanf-like functions so the checker
    can compare the format string with the arguments (``-Wformat``).
    """

    name: str
    prototype: str
    impl: Impl
    concept: Concept
    format_arg: int | None = None
    format_kind: str | None = None
    doc: str = ""


def counted(kind: str) -> Callable[[Impl], Impl]:
    """Decorate an implementation so every call counts construct ``kind``.

    The wrapper keeps the generator-ness of the function it wraps: the interpreter
    looks at ``inspect.isgeneratorfunction`` to decide whether to ``yield from`` an
    implementation, so a plain wrapper around a generator would break it.
    """
    def decorate(impl: Impl) -> Impl:
        if inspect.isgeneratorfunction(impl):
            @functools.wraps(impl)
            def generator_wrapper(machine: Any, args: list) -> Any:
                machine.count_construct(kind)
                return (yield from impl(machine, args))
            return generator_wrapper

        @functools.wraps(impl)
        def wrapper(machine: Any, args: list) -> Any:
            machine.count_construct(kind)
            return impl(machine, args)
        return wrapper
    return decorate


def register_functions(registry: BuiltinRegistry, header: str,
                       functions: tuple[LibraryFunction, ...]) -> None:
    """Register every function of one header with its concept and format info."""
    for function in functions:
        registry.register(function.name, function.prototype, function.impl,
                          concept=function.concept, header=header,
                          format_arg=function.format_arg, format_kind=function.format_kind,
                          doc=function.doc)


@lru_cache(maxsize=None)
def header_text(name: str) -> str:
    """The text of one standard header (``KeyError`` for a header we do not ship)."""
    if name not in STANDARD_HEADERS:
        raise KeyError(name)
    # Text mode turns a Windows checkout's CRLF back into the LF the preprocessor and
    # the interactive `cat` expect.
    return (HEADER_DIR / name).read_text(encoding="utf-8")


def load_headers() -> dict[str, str]:
    """All standard headers, keyed by bare name (``"stdio.h"``)."""
    return {name: header_text(name) for name in STANDARD_HEADERS}


def system_headers() -> SystemHeaders:
    """A fresh ``SystemHeaders`` holding the standard headers.

    A host may add its own device headers to the result; the copy keeps one
    toolchain's additions from leaking into another.
    """
    return SystemHeaders(load_headers())
