"""Symbol tables shared by the checker, the linker and the interpreter.

The checker decides what every name means inside one translation unit and writes the
answer onto the AST (``Identifier.ref``). The linker joins the units' tables into one
program. The interpreter trusts both and never re-derives a name. Storage keys are chosen
so that two files may each have a ``static int count`` without
colliding, and so a key survives re-parsing an object file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pycc.core.diagnostics import SourceLocation
from pycc.interp.c_types import CType

RefKind = Literal["local", "param", "global", "unit_static", "static_local",
                  "function", "builtin", "enum_const"]
Linkage = Literal["external", "internal"]


@dataclass
class SymbolRef:
    """What an identifier resolves to. ``key`` is the storage key for globals,
    statics and functions, the local slot name for locals and params, and the
    constant's value as text for enum constants."""

    kind: RefKind
    key: str
    ctype: CType
    value: int | None = None      # enum constants only


@dataclass
class SymbolDef:
    """One definition or declaration visible at file scope of a unit."""

    name: str
    kind: Literal["function", "variable"]
    linkage: Linkage
    ctype: CType
    loc: SourceLocation
    tentative: bool = False       # ``int count;`` at file scope without an initializer
    defined: bool = True          # False for prototypes and ``extern`` declarations
    key: str = ""                 # storage key (see below)


@dataclass
class UnitSymbols:
    """The file-scope symbols of one translation unit after checking."""

    source_path: str
    defined: dict[str, SymbolDef] = field(default_factory=dict)       # name -> definition
    declared: dict[str, SymbolDef] = field(default_factory=dict)      # name -> prototype / extern
    referenced: dict[str, list[SourceLocation]] = field(default_factory=dict)
    static_locals: dict[str, SymbolDef] = field(default_factory=dict) # key -> definition

    def external_definitions(self) -> dict[str, SymbolDef]:
        return {n: d for n, d in self.defined.items() if d.linkage == "external"}

    def undefined_references(self) -> dict[str, list[SourceLocation]]:
        """Names used in this unit that no definition in this unit provides."""
        return {n: locs for n, locs in self.referenced.items() if n not in self.defined}


def external_key(name: str) -> str:
    return name


def unit_static_key(source_path: str, name: str) -> str:
    return f"{source_path}::{name}"


def static_local_key(source_path: str, function: str, name: str, loc: SourceLocation) -> str:
    return f"{source_path}::{function}::{name}@{loc.line}:{loc.column}"


def is_static_key(key: str) -> bool:
    return "::" in key
