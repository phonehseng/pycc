"""Which system header would have declared a name: "did you forget #include <stdio.h>?"

The checker already knows the header of every *function* and library *variable*
(``BuiltinRegistry.header_for``). Constants are different: ``EOF``, ``NULL`` and
``INT_MAX`` are macros (or enumerators) that exist only in header text, so a user who
forgets ``#include <stdio.h>`` gets "'EOF' undeclared" with a generic hint to declare
a variable called ``EOF`` - exactly the wrong advice.

This module looks the name up in the toolchain's system headers and, when one of
them defines it, rewrites the friendly text of the ``undeclared_identifier``
diagnostic to point at the header. The gcc message itself is never changed. When a
header is only meant to be reached through another one (``<motor.h>`` includes
``<motor_defs.h>``), the public header is suggested.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from pycc.core.diagnostics import Diagnostic
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.toolchain import SYSTEM_INCLUDE_DIR, SystemHeaders

UNDECLARED = "undeclared_identifier"
IMPLICIT = "implicit_declaration"
# Umbrella headers that include everything are never the one to suggest. A host that
# ships such a header can add its name here.
CATCH_ALL_HEADERS: tuple[str, ...] = ()
# When several standard headers define a name (NULL), suggest the familiar one.
PREFERRED_HEADERS = ("stdio.h", "stdlib.h", "string.h")

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_ENUM_BODY = re.compile(r"\benum\b[^{;]*\{([^}]*)\}")
_ENUMERATOR = re.compile(r"^\s*([A-Za-z_]\w*)")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', re.MULTILINE)
_QUOTED_NAME = re.compile(r"^'([^']+)'")
_IMPLICIT_NAME = re.compile(r"implicit declaration of function '([^']+)'")


def names_defined_in(text: str) -> set[str]:
    """The macros and enumerators a header defines (comments ignored)."""
    code = _COMMENT.sub(" ", text)
    names = set(_DEFINE.findall(code))
    for body in _ENUM_BODY.findall(code):
        for item in body.split(","):
            match = _ENUMERATOR.match(item)
            if match:
                names.add(match.group(1))
    return names


def defining_headers(name: str, headers: SystemHeaders) -> list[str]:
    """Every system header whose own text defines ``name``, best suggestion first."""
    found = [header for header in headers.names()
             if header not in CATCH_ALL_HEADERS
             and name in names_defined_in(headers.get(header) or "")]
    return sorted(found, key=_preference)


def header_to_include(header: str, headers: SystemHeaders) -> str:
    """The header a user should write for something defined in ``header``.

    A public header (``motor.h``) that includes ``header`` (``motor_defs.h``) is the
    one to write, so it wins; otherwise ``header`` itself.
    """
    for candidate in headers.names():
        if candidate in CATCH_ALL_HEADERS or candidate == header:
            continue
        if header in _INCLUDE.findall(headers.get(candidate) or ""):
            return candidate
    return header


def undeclared_friendly(name: str, header: str, via: str) -> str:
    """The friendly text for a constant the user forgot to include."""
    where = f"<{header}>" if via == header else f"<{header}> (which <{via}> includes)"
    return (f"'{name}' is not a variable of yours: it is defined in {where}. Did you "
            f"forget #include <{via}>? Add it at the top of the file.")


def implicit_friendly(name: str, header: str, via: str) -> str:
    """The friendly text for a function whose public header was not included."""
    return (f"'{name}' is declared in <{header}> (which <{via}> includes). Did you forget "
            f"#include <{via}>? Add it at the top of the file.")


def point_at_headers(diagnostics: Iterable[Diagnostic], headers: SystemHeaders,
                     included: Iterable[str] = (),
                     builtins: BuiltinRegistry | None = None) -> None:
    """Point "undeclared" and "implicit declaration" errors at the header to include.

    Constants (``EOF``) are looked up in the header texts. Functions already name
    their header, but one declared in an internal header (``<motor_defs.h>``) should
    point at the public header that includes it; with ``builtins`` that text is
    improved too. ``included`` lists the files the unit already included
    (``/usr/include/stdio.h``); a name from a header that *was* included is a
    different mistake (a typo), so its friendly text is left alone.
    """
    already = {path[len(SYSTEM_INCLUDE_DIR) + 1:] for path in included
               if path.startswith(SYSTEM_INCLUDE_DIR + "/")}
    for diag in diagnostics:
        if diag.code == UNDECLARED:
            _hint_constant(diag, headers, already)
        elif diag.code == IMPLICIT and builtins is not None:
            _hint_function(diag, headers, already, builtins)


def _hint_constant(diag: Diagnostic, headers: SystemHeaders, already: set[str]) -> None:
    match = _QUOTED_NAME.match(diag.message)
    if match is None:
        return
    name = match.group(1)
    found = defining_headers(name, headers)
    if not found or found[0] in already:
        return
    via = header_to_include(found[0], headers)
    if via not in already:
        diag.friendly = undeclared_friendly(name, found[0], via)


def _hint_function(diag: Diagnostic, headers: SystemHeaders, already: set[str],
                   builtins: BuiltinRegistry) -> None:
    match = _IMPLICIT_NAME.search(diag.message)
    if match is None:
        return
    name = match.group(1)
    header = builtins.header_for(name)
    if header is None:
        _hint_macro(diag, name, headers, already)
        return
    if header in already:
        return
    via = header_to_include(header, headers)
    if via != header and via not in already:
        diag.friendly = implicit_friendly(name, header, via)


def _hint_macro(diag: Diagnostic, name: str, headers: SystemHeaders,
                already: set[str]) -> None:
    """A function-like library macro (``assert``) is not in the registry at all."""
    found = defining_headers(name, headers)
    if not found or found[0] in already:
        return
    via = header_to_include(found[0], headers)
    if via not in already:
        diag.friendly = macro_friendly(name, via)


def macro_friendly(name: str, header: str) -> str:
    """The friendly text for a library macro used like a function (``assert``)."""
    return (f"'{name}' comes from <{header}>. Did you forget #include <{header}>? Add it "
            "at the top of the file.")


def _preference(header: str) -> tuple[int, str]:
    rank = PREFERRED_HEADERS.index(header) if header in PREFERRED_HEADERS else len(
        PREFERRED_HEADERS)
    return rank, header


__all__ = ["defining_headers", "header_to_include", "implicit_friendly", "macro_friendly",
           "names_defined_in", "point_at_headers", "undeclared_friendly"]
