"""Library functions written in Python: the registry and the Machine facade.

Every function a C program can call without defining it (``printf``, ``malloc``,
``move``...) is registered here with its C prototype. The checker reads the prototypes
through the headers, the linker treats registered names as provided, and the interpreter
calls the Python implementation when the program calls the function.

An implementation receives the ``Machine`` facade and the argument values. It may be a
plain function or a generator: a generator may ``yield`` events (``Output``,
``InputRequest``, ``DeviceCall``) and receive the host's reply, exactly like the
interpreter itself (core/events.py). See.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pycc.core.concepts import Concept
from pycc.core.diagnostics import SourceLocation
from pycc.interp.c_types import INT, CType, FunctionType
from pycc.interp.values import Value

Impl = Callable[..., Any]


@dataclass
class BuiltinSpec:
    name: str
    prototype: str                      # C text, e.g. "int printf(const char *format, ...);"
    impl: Impl
    concept: Concept | None = None
    header: str | None = None           # "stdio.h", "device.h", ...
    format_arg: int | None = None       # index of the format-string argument
    format_kind: str | None = None      # "printf" | "scanf"
    doc: str = ""
    ftype: FunctionType | None = None   # parsed lazily from ``prototype``

    @property
    def is_generator(self) -> bool:
        return inspect.isgeneratorfunction(self.impl)


@dataclass
class BuiltinVariable:
    """A library object with an address: ``stdin``, ``stdout``, ``stderr``, ``errno``."""

    name: str
    ctype_decl: str                     # C declaration text, e.g. "FILE *stdin;"
    init: Callable[["Machine"], bytes]  # initial bytes, given the machine
    header: str | None = None
    ctype: CType | None = None          # resolved lazily


class BuiltinRegistry:
    """All library functions and variables a program may link against."""

    def __init__(self) -> None:
        self.functions: dict[str, BuiltinSpec] = {}
        self.variables: dict[str, BuiltinVariable] = {}

    # --- registration -------------------------------------------------------------
    def register(self, name: str, prototype: str, impl: Impl, concept: Concept | None = None,
                 header: str | None = None, format_arg: int | None = None,
                 format_kind: str | None = None, doc: str = "") -> BuiltinSpec:
        """Register a function. An identical re-registration is ignored; a different
        prototype under the same name raises ``ValueError``."""
        normalized = _normalize_prototype(prototype)
        existing = self.functions.get(name)
        if existing is not None:
            if _normalize_prototype(existing.prototype) != normalized:
                raise ValueError(f"builtin {name!r} already registered with a different prototype")
            return existing
        spec = BuiltinSpec(name, prototype.strip(), impl, concept, header, format_arg, format_kind,
                           doc)
        self.functions[name] = spec
        return spec

    def register_variable(self, name: str, ctype_decl: str, init: Callable[["Machine"], bytes],
                          header: str | None = None) -> BuiltinVariable:
        var = BuiltinVariable(name, ctype_decl, init, header)
        self.variables[name] = var
        return var

    # --- queries --------------------------------------------------------------------
    def get(self, name: str) -> BuiltinSpec | None:
        return self.functions.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.functions or name in self.variables

    def names(self) -> list[str]:
        return sorted(self.functions)

    def header_for(self, name: str) -> str | None:
        spec = self.functions.get(name)
        if spec is not None:
            return spec.header
        var = self.variables.get(name)
        return var.header if var is not None else None

    def format_info(self, name: str) -> tuple[int, str] | None:
        spec = self.functions.get(name)
        if spec is None or spec.format_arg is None or spec.format_kind is None:
            return None
        return (spec.format_arg, spec.format_kind)

    def prototypes_for_header(self, header: str) -> list[str]:
        return [s.prototype for s in self.functions.values() if s.header == header]

    def function_type(self, name: str) -> FunctionType:
        """The parsed ``FunctionType`` of a builtin (parsed once, lazily)."""
        spec = self.functions[name]
        if spec.ftype is None:
            from pycc.interp.parser import parse_prototype  # lazy: parser is a heavier import

            spec.ftype = parse_prototype(spec.prototype)
        return spec.ftype

    def copy(self) -> "BuiltinRegistry":
        other = BuiltinRegistry()
        other.functions = {n: BuiltinSpec(**{**s.__dict__}) for n, s in self.functions.items()}
        other.variables = dict(self.variables)
        return other


def _normalize_prototype(text: str) -> str:
    return " ".join(text.replace(";", " ").split())


class Machine(Protocol):
    """The facade a builtin sees. The interpreter implements it.

    Builtins must not touch the interpreter directly; everything they need is here so
    the standard library can be tested with a small fake.
    """

    memory: Any                         # interp.memory.Memory
    io: Any                             # interp.hostio.ProgramIO
    state: dict[str, Any]               # per-run scratch storage for libraries (FILE table, stdin buffer...)
    statements_executed: int
    device_tick: int
    current_location: SourceLocation | None

    def read_cstring(self, addr: int, limit: int = 4096) -> str: ...
    def write_cstring(self, addr: int, text: str, limit: int | None = None) -> int: ...
    def read_bytes(self, addr: int, count: int) -> bytes: ...
    def write_bytes(self, addr: int, data: bytes) -> None: ...
    def malloc(self, size: int) -> int: ...
    def free(self, addr: int) -> None: ...
    def make_int(self, value: int, ctype: CType = INT) -> Value: ...
    def make_double(self, value: float) -> Value: ...
    def make_pointer(self, addr: int, target: CType) -> Value: ...
    def get_errno(self) -> int: ...
    def set_errno(self, code: int) -> None: ...
    def exit(self, code: int) -> None: ...                       # raises ProgramExit
    def trap(self, code: str, message: str, friendly: str, concept: Concept | None = None,
             **extra: Any) -> None: ...                           # raises TrapError
    def warn(self, code: str, message: str, friendly: str, concept: Concept | None = None) -> None: ...
    def call_function(self, fid: int, args: list[Value]) -> Any: ...   # generator: yields events, returns Value
    def count_construct(self, kind: str, amount: int = 1) -> None: ...
