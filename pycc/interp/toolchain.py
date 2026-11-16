"""The toolchain bundle: options, system headers and the builtin registry.

One ``Toolchain`` (standard library, system headers and any device functions) is
handed to ``cc``, ``make`` and the headless runner, so every path compiles and links
the same way. ``pycc.interp.stdlib.default_toolchain()`` builds the standard one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pycc.interp.builtins import BuiltinRegistry

SYSTEM_INCLUDE_DIR = "/usr/include"


@dataclass
class CompileOptions:
    include_dirs: list[str] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)
    wall: bool = False
    wextra: bool = False
    werror: bool = False

    def copy(self) -> "CompileOptions":
        return CompileOptions(list(self.include_dirs), dict(self.defines), self.wall, self.wextra, self.werror)


class SystemHeaders:
    """The texts behind ``#include <...>``. Keyed by bare name ("stdio.h", "device.h")."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers: dict[str, str] = dict(headers or {})

    def get(self, name: str) -> str | None:
        return self._headers.get(name)

    def add(self, name: str, text: str) -> None:
        self._headers[name] = text

    def names(self) -> list[str]:
        return sorted(self._headers)

    def path_of(self, name: str) -> str:
        """The path a diagnostic shows for a system header."""
        return f"{SYSTEM_INCLUDE_DIR}/{name}"

    def as_dict(self) -> dict[str, str]:
        return dict(self._headers)

    def copy(self) -> "SystemHeaders":
        return SystemHeaders(self._headers)


@dataclass
class Toolchain:
    builtins: BuiltinRegistry
    headers: SystemHeaders
    default_options: CompileOptions = field(default_factory=CompileOptions)
    zone_id: str | None = None
