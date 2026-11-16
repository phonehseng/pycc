"""Object files, executables and the linker.

``cc -c util.c`` produces an *object file*: one translation unit after compiling, with
a table of the names it defines and the names it needs from elsewhere. The *linker*
joins object files into one program and checks that every name somebody needs is
defined exactly once - by one of the units or by the library (the builtins). This is
where beginners meet "undefined reference" and "multiple definition", so every error
here gets a message copied from ``ld`` and a friendly text that says which file to fix.

On disk (in the virtual filesystem) an object file is ``b"CVOBJ1\\n"`` followed by
zlib-compressed JSON of its preprocessed lines and build options. The AST is not stored:
loading re-runs the lexer, parser and checker on the stored lines, which is
deterministic, so the annotations (``Identifier.ref``, ``ctype``) come back exactly as
the compiler left them. An executable is ``b"CVEXE1\\n"``, a
version byte and zlib-compressed JSON of its object files; running it links again.

Name resolution per unit follows C linkage: the unit's own ``static`` names first,
then the program's external names, then the library.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation, concept_for_code
from pycc.interp import astnodes as ast
from pycc.interp.builtins import BuiltinRegistry, BuiltinSpec
from pycc.interp.friendly import did_you_mean
from pycc.interp.lexer import TokenKind, tokenize
from pycc.interp.source import PPLine
from pycc.interp.symbols import SymbolDef, UnitSymbols, is_static_key, static_local_key, unit_static_key
from pycc.interp.toolchain import CompileOptions, Toolchain

OBJECT_MAGIC = b"CVOBJ1\n"
OBJECT_VERSION = 1
EXECUTABLE_MAGIC = b"CVEXE1\n"
EXECUTABLE_VERSION = 1

# Real programs start in the C runtime's _start, which calls main. ld blames that
# startup object when main is missing, and so do we.
_STARTUP_OBJECT = "crt1.o"
_STARTUP_FUNCTION = "_start"


# =========================================================================== errors

class BadObjectFile(Exception):
    """The bytes are not an object file this toolchain can read (wrong magic or version)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def diagnostic(self, path: str) -> Diagnostic:
        """The ``bad_object_file`` diagnostic ``cc`` prints for ``path``."""
        return _diagnostic(
            "bad_object_file", f"{path}: file not recognized: file format not recognized",
            f"{path} is not an object file made by this cc ({self.reason}). Object files "
            "(.o) come from 'cc -c file.c'; rebuild it from its .c file.",
            None,
        )


class BadExecutable(Exception):
    """The bytes are not an executable this toolchain can run (wrong magic or version)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def diagnostic(self, name: str) -> Diagnostic:
        """The ``bad_executable`` diagnostic the host prints when running ``name``."""
        return _diagnostic(
            "bad_executable", f"{name}: cannot execute binary file: Exec format error",
            f"{name} is not a program this machine can run ({self.reason}). Build it "
            "again with cc.",
            None, phase=Phase.RUNTIME,
        )


# =========================================================================== serialising

def options_to_dict(options: CompileOptions) -> dict[str, Any]:
    """CompileOptions as plain JSON data."""
    return {"include_dirs": list(options.include_dirs), "defines": dict(options.defines),
            "wall": options.wall, "wextra": options.wextra, "werror": options.werror}


def options_from_dict(data: dict[str, Any]) -> CompileOptions:
    """The inverse of ``options_to_dict``."""
    return CompileOptions(list(data.get("include_dirs", [])), dict(data.get("defines", {})),
                          bool(data.get("wall")), bool(data.get("wextra")),
                          bool(data.get("werror")))


def _lines_to_json(lines: Iterable[PPLine]) -> list[list[Any]]:
    return [[line.text, line.file, line.line,
             list(line.col_map) if line.col_map is not None else None] for line in lines]


def _lines_from_json(data: list[list[Any]]) -> list[PPLine]:
    return [PPLine(str(text), str(file), int(number),
                   tuple(int(c) for c in col_map) if col_map is not None else None)
            for text, file, number, col_map in data]


def _pack(payload: Any) -> bytes:
    # sort_keys keeps the bytes identical for identical inputs, so builds are reproducible.
    return zlib.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _unpack(data: bytes) -> Any:
    return json.loads(zlib.decompress(data).decode("utf-8"))


# =========================================================================== object files

@dataclass
class ObjectFile:
    """One compiled translation unit.

    ``unit_name`` is the source path exactly as given to ``cc`` ("src/util.c"); storage
    keys of statics are built from it. ``source_path`` is that file's absolute path in
    the virtual filesystem. ``ast`` and ``symbols`` are ``None`` while the object is
    only *decoded* (read from bytes but not re-checked); ``load`` fills them in.
    """

    unit_name: str
    source_path: str
    ast: ast.TranslationUnit | None
    symbols: UnitSymbols | None
    pp_lines: list[PPLine]
    options: CompileOptions = field(default_factory=CompileOptions)
    cwd: str = "/"

    @property
    def object_name(self) -> str:
        """The name ld shows: ``src/util.c`` -> ``util.o``."""
        return object_name_for(self.unit_name)

    @property
    def is_loaded(self) -> bool:
        """True when the annotated AST and the symbol table are present."""
        return self.ast is not None and self.symbols is not None

    # --- bytes ------------------------------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        """The JSON data stored in a ``.o`` file (and inside executables)."""
        return {"version": OBJECT_VERSION, "unit_name": self.unit_name,
                "source_path": self.source_path, "cwd": self.cwd,
                "options": options_to_dict(self.options),
                "pp_lines": _lines_to_json(self.pp_lines)}

    def to_bytes(self) -> bytes:
        """The exact bytes of the ``.o`` file."""
        return OBJECT_MAGIC + _pack(self.to_payload())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ObjectFile:
        """Rebuild an unloaded object from ``to_payload`` data; raises BadObjectFile."""
        if not isinstance(payload, dict) or payload.get("version") != OBJECT_VERSION:
            raise BadObjectFile("unsupported object file version")
        try:
            return cls(str(payload["unit_name"]), str(payload["source_path"]), None, None,
                       _lines_from_json(payload["pp_lines"]),
                       options_from_dict(payload.get("options", {})),
                       str(payload.get("cwd", "/")))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise BadObjectFile(f"damaged contents: {error}") from error

    @classmethod
    def decode(cls, data: bytes) -> ObjectFile:
        """Read the bytes without re-checking; ``load`` must run before linking."""
        if not data.startswith(OBJECT_MAGIC):
            raise BadObjectFile("wrong magic number")
        try:
            payload = _unpack(data[len(OBJECT_MAGIC):])
        except (zlib.error, ValueError, UnicodeDecodeError) as error:
            raise BadObjectFile(f"damaged contents: {error}") from error
        return cls.from_payload(payload)

    @classmethod
    def from_bytes(cls, data: bytes, toolchain: Toolchain | BuiltinRegistry) -> ObjectFile:
        """Read a ``.o`` file and restore its annotated AST; raises BadObjectFile."""
        registry = toolchain.builtins if isinstance(toolchain, Toolchain) else toolchain
        return cls.decode(data).load(registry)

    def load(self, builtins: BuiltinRegistry) -> ObjectFile:
        """Re-run lex, parse and check on ``pp_lines`` (deterministic; diagnostics are
        discarded, they were shown when the object was compiled). Returns self."""
        if self.is_loaded:
            return self
        # Imported here because the compile driver itself imports this module.
        from pycc.interp.compiler import run_front_end

        front = run_front_end(self.pp_lines, self.unit_name, builtins, self.options)
        if not front.ok or front.ast is None:
            # The interpreter runs only fully checked trees; a half-annotated one
            # would crash mid-run instead of failing here with a clear message.
            raise BadObjectFile("its code no longer compiles")
        self.ast, self.symbols = front.ast, front.symbols
        return self


def object_name_for(source: str) -> str:
    """``src/util.c`` -> ``util.o``; ``util.o`` stays ``util.o``."""
    stem = posixpath.splitext(posixpath.basename(source))[0]
    return f"{stem}.o"


# =========================================================================== executables

@dataclass
class Executable:
    """What ``cc -o prog`` writes: the program's object files, linked on each run."""

    objects: list[ObjectFile]

    def to_bytes(self) -> bytes:
        """The exact bytes of the executable file."""
        payload = [obj.to_payload() for obj in self.objects]
        return EXECUTABLE_MAGIC + bytes([EXECUTABLE_VERSION]) + _pack(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> Executable:
        """Decode an executable; its objects still need ``load``. Raises BadExecutable."""
        if not data.startswith(EXECUTABLE_MAGIC):
            raise BadExecutable("wrong magic number")
        body = data[len(EXECUTABLE_MAGIC):]
        if not body or body[0] != EXECUTABLE_VERSION:
            raise BadExecutable("unsupported executable version")
        try:
            payload = _unpack(body[1:])
            if not isinstance(payload, list):
                raise ValueError("expected a list of object files")
            return cls([ObjectFile.from_payload(item) for item in payload])
        except (zlib.error, ValueError, UnicodeDecodeError, BadObjectFile) as error:
            raise BadExecutable(f"damaged contents: {error}") from error


# =========================================================================== program hash

def program_hash(executable: Executable | Iterable[ObjectFile]) -> str:
    """sha1 of the normalised token stream.

    Whitespace and comments never reach the tokens, and identifiers are renamed in
    order of first appearance, so re-indenting or renaming variables gives the same
    hash while any real change to the code gives a different one. Units are sorted by
    name so the order of files on the cc command line does not matter.
    """
    objects = executable.objects if isinstance(executable, Executable) else list(executable)
    renamed: dict[str, str] = {}
    parts: list[str] = []
    for obj in sorted(objects, key=lambda o: o.unit_name):
        for token in tokenize(obj.pp_lines, []):
            if token.kind is TokenKind.EOF:
                continue
            if token.kind is TokenKind.IDENT:
                name = renamed.setdefault(str(token.value), f"id{len(renamed)}")
                parts.append(f"I:{name}")
            else:
                parts.append(f"{token.kind.value}:{token.text}")
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


# =========================================================================== link results

# What a key can mean: a FunctionDef, a VarDecl, a BuiltinSpec or a BuiltinVariable.
Definition = object


@dataclass
class LinkedProgram:
    """The whole program after linking, ready for the interpreter.

    ``functions`` and ``globals`` are keyed by storage key (bare name for external
    names, ``path::name`` for statics, the static-local key for static locals).
    ``function_table[id]`` is the function with that id; id 0 is never used so a NULL
    function pointer can never name a function. ``definitions`` maps every key any
    unit refers to onto what it means, builtins included.
    """

    units: list[ObjectFile]
    functions: dict[str, ast.FunctionDef]
    globals: dict[str, ast.VarDecl]
    function_table: list[ast.FunctionDef | BuiltinSpec | None]
    entry: str
    builtins: BuiltinRegistry
    definitions: dict[str, Definition] = field(default_factory=dict)
    function_ids: dict[str, int] = field(default_factory=dict)
    unit_of: dict[str, str] = field(default_factory=dict)          # key -> defining unit
    unit_statics: dict[str, dict[str, str]] = field(default_factory=dict)  # unit -> name -> key

    def resolve(self, unit_name: str, name: str) -> str | None:
        """The key ``name`` means inside ``unit_name``: unit statics, externals, builtins."""
        static_key = self.unit_statics.get(unit_name, {}).get(name)
        if static_key is not None:
            return static_key
        return name if name in self.definitions else None

    def function_id(self, key: str) -> int:
        """The function-pointer id of a function or builtin (0 when it has none)."""
        return self.function_ids.get(key, 0)

    def function_at(self, fid: int) -> ast.FunctionDef | BuiltinSpec | None:
        """The function a pointer id names, or None for an invalid id."""
        if 0 < fid < len(self.function_table):
            return self.function_table[fid]
        return None

    @property
    def entry_function(self) -> ast.FunctionDef:
        """The FunctionDef execution starts in (``main``)."""
        return self.functions[self.entry]


@dataclass
class LinkResult:
    """The program (when linking worked) and every link diagnostic."""

    ok: bool
    program: LinkedProgram | None
    diagnostics: list[Diagnostic] = field(default_factory=list)


def link(objects: Iterable[ObjectFile], builtins: BuiltinRegistry, entry: str = "main",
         known_sources: Iterable[str] = ()) -> LinkResult:
    """Link loaded object files against the library.

    ``known_sources`` lists source or object files that exist in the project but were
    not linked; they let "undefined reference" ask "did you pass util.o to the
    linker?" instead of only saying the name is missing.
    """
    units = list(objects)
    for unit in units:
        if not unit.is_loaded:
            raise ValueError(f"object {unit.unit_name!r} must be loaded before linking")
    return _Linker(units, builtins, entry, list(known_sources)).run()


# =========================================================================== the linker

@dataclass
class _Placed:
    """Where an external definition came from."""

    unit: ObjectFile
    sym: SymbolDef


class _UnitIndex:
    """The AST nodes of one unit that the linker needs, found once."""

    def __init__(self, unit: ObjectFile) -> None:
        tree = unit.ast
        assert tree is not None
        self.functions: dict[str, ast.FunctionDef] = {}
        self.variables: dict[str, ast.VarDecl] = {}
        self.static_locals: list[tuple[ast.FunctionDef, ast.VarDecl]] = []
        # (name, loc) -> the function using it there; None marks a use inside a global
        # initializer, which ld reports as a use from the data section, not a function.
        self.sites: dict[tuple[str, SourceLocation], str | None] = {}
        for decl in tree.decls:
            if isinstance(decl, ast.FunctionDef):
                self.functions[decl.name] = decl
                self._index_body(decl)
            elif isinstance(decl, ast.VarDecl):
                self._index_global(decl)

    def _index_global(self, decl: ast.VarDecl) -> None:
        """Record a file-scope variable and the names its initializer uses."""
        if decl.init is not None:
            self._index_sites(decl.init, None)
        if decl.is_extern:
            return
        # 'int x; int x = 3;' in one file is one variable: keep the initialized one.
        if decl.name not in self.variables or decl.init is not None:
            self.variables[decl.name] = decl

    def _index_body(self, function: ast.FunctionDef) -> None:
        """Record the names a function uses and its static locals."""
        self._index_sites(function.body, function.name)
        for node in ast.walk(function.body):
            if isinstance(node, ast.VarDecl) and node.storage == "static":
                self.static_locals.append((function, node))

    def _index_sites(self, tree: ast.Node, function: str | None) -> None:
        """Remember, for every identifier below ``tree``, which function it is in."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Identifier):
                self.sites.setdefault((node.name, node.loc), function)

    def function_containing(self, name: str, loc: SourceLocation) -> str | None:
        """The function whose body uses ``name`` at ``loc`` (None: a global initializer)."""
        if (name, loc) in self.sites:
            return self.sites[(name, loc)]
        # A checker that records another position than the identifier's own: fall back
        # to the nearest function starting above the use, in the same file.
        above = [f for f in self.functions.values()
                 if f.loc.file == loc.file and f.loc.line <= loc.line]
        return max(above, key=lambda f: f.loc.line).name if above else None


class _Linker:
    """One link run: collect definitions, resolve references, build the program."""

    def __init__(self, units: list[ObjectFile], builtins: BuiltinRegistry, entry: str,
                 known_sources: list[str]) -> None:
        self.units = units
        self.builtins = builtins
        self.entry = entry
        self.known_sources = known_sources
        self.indexes = {id(unit): _UnitIndex(unit) for unit in units}
        self.externals: dict[str, _Placed] = {}
        self.diagnostics: list[Diagnostic] = []

    def run(self) -> LinkResult:
        """Do the link and package the result."""
        self._collect_externals()
        self._check_references()
        self._check_entry()
        if any(diag.is_error for diag in self.diagnostics):
            return LinkResult(False, None, self.diagnostics)
        return LinkResult(True, self._build_program(), self.diagnostics)

    # --- definitions --------------------------------------------------------------------
    def _collect_externals(self) -> None:
        seen_units: dict[str, ObjectFile] = {}
        for unit in self.units:
            if unit.unit_name in seen_units:
                self._report_duplicate_unit(seen_units[unit.unit_name], unit)
                continue
            seen_units[unit.unit_name] = unit
            for name, sym in _symbols(unit).external_definitions().items():
                first = self.externals.get(name)
                if first is None:
                    self.externals[name] = _Placed(unit, sym)
                else:
                    self.diagnostics.append(_multiple_definition(name, first, _Placed(unit, sym)))

    def _report_duplicate_unit(self, first: ObjectFile, again: ObjectFile) -> None:
        externals = _symbols(again).external_definitions()
        name = next(iter(externals), first.unit_name)
        self.diagnostics.append(_diagnostic(
            "multiple_definition",
            f"ld: {again.object_name}: multiple definition of '{name}'; "
            f"{first.object_name}: first defined here",
            f"{first.unit_name} was given to the linker twice, so everything it defines is "
            "defined two times. List each file only once on the cc command line.",
            externals[name].loc if name in externals else None,
        ))

    # --- references ---------------------------------------------------------------------
    def _check_references(self) -> None:
        for unit in self.units:
            symbols = _symbols(unit)
            reported: set[tuple[str | None, str]] = set()
            for name, locations in symbols.referenced.items():
                if self._resolves(unit, name):
                    continue
                for loc in locations:
                    function = self.indexes[id(unit)].function_containing(name, loc)
                    if (function, name) in reported:
                        continue
                    reported.add((function, name))
                    self.diagnostics.append(self._unresolved(unit, name, loc, function))

    def _resolves(self, unit: ObjectFile, name: str) -> bool:
        """Can ``name`` be found from ``unit`` ?"""
        own = _symbols(unit).defined.get(name)
        if own is not None:
            return True
        declared = _symbols(unit).declared.get(name)
        if declared is not None and declared.linkage == "internal":
            return False    # a static prototype must be defined in this very file
        if name in self.externals:
            return True
        return name in self.builtins

    def _unresolved(self, unit: ObjectFile, name: str, loc: SourceLocation,
                    function: str | None) -> Diagnostic:
        """``undefined_reference`` for one use of ``name``."""
        where = (f"ld: {unit.object_name}: in function '{function}': " if function
                 else f"ld: {unit.object_name}:(.data+0x0): ")
        message = f"{where}undefined reference to '{name}'"
        return _diagnostic("undefined_reference", message,
                           self._undefined_friendly(unit, name), loc)

    def _undefined_friendly(self, unit: ObjectFile, name: str) -> str:
        """Pick the explanation that matches why ``name`` is missing."""
        is_function = _looks_like_function(unit, name)
        hidden = self._static_elsewhere(unit, name)
        if hidden is not None:
            return _static_friendly(name, is_function, hidden)
        missing = self._unlinked_source_for(unit, name)
        if missing is not None:
            return self._not_given_friendly(unit, name, missing)
        return _not_defined_friendly(name, is_function, did_you_mean(name, self.externals))

    def _static_elsewhere(self, unit: ObjectFile, name: str) -> ObjectFile | None:
        """Another unit that defines ``name`` as static (private to that file)."""
        for other in self.units:
            sym = _symbols(other).defined.get(name)
            if other is not unit and sym is not None and sym.linkage == "internal":
                return other
        return None

    def _unlinked_source_for(self, unit: ObjectFile, name: str) -> str | None:
        """A known but unlinked source whose name matches the header declaring ``name``."""
        declared = _symbols(unit).declared.get(name)
        if declared is None or declared.loc.file == unit.unit_name:
            return None
        header_stem = _stem(declared.loc.file)
        linked = {_stem(u.unit_name) for u in self.units}
        for source in self.known_sources:
            if _stem(source) == header_stem and _stem(source) not in linked:
                return source
        return None

    def _not_given_friendly(self, unit: ObjectFile, name: str, source: str) -> str:
        header = _symbols(unit).declared[name].loc.file
        command = " ".join([u.unit_name for u in self.units] + [_as_source(source)])
        return (f"Did you pass {object_name_for(source)} to the linker? {header} declares "
                f"'{name}', and its definition is probably in {_as_source(source)}, which "
                f"was not part of this build. Give cc every file of the program: "
                f"cc {command} -o <program>")

    # --- main -------------------------------------------------------------------------
    def _check_entry(self) -> None:
        placed = self.externals.get(self.entry)
        if placed is not None and placed.sym.kind == "function":
            return
        hidden = next((u for u in self.units
                       if (s := _symbols(u).defined.get(self.entry)) is not None
                       and s.linkage == "internal"), None)
        if hidden is not None:
            friendly = (f"{self.entry}() in {hidden.unit_name} is marked static, which hides "
                        "it from the linker, so the program has no starting point. Remove "
                        f"'static' from {self.entry}().")
        else:
            friendly = (f"Every C program starts in {self.entry}(), but none of the files "
                        f"you linked defines it. If this file is a helper, build it with "
                        f"'cc -c' and link it together with the file that has {self.entry}().")
        self.diagnostics.append(_diagnostic(
            "no_main",
            f"ld: {_STARTUP_OBJECT}: in function '{_STARTUP_FUNCTION}': "
            f"undefined reference to '{self.entry}'",
            friendly, None))

    # --- the program ----------------------------------------------------------------------
    def _build_program(self) -> LinkedProgram:
        program = LinkedProgram(self.units, {}, {}, [None], self.entry, self.builtins)
        for unit in self.units:
            self._place_unit(program, unit)
        self._place_builtins(program)
        for unit in self.units:
            _annotate_refs(unit, program)
        return program

    def _place_unit(self, program: LinkedProgram, unit: ObjectFile) -> None:
        """Add the unit's functions, globals and statics under their storage keys."""
        symbols, index = _symbols(unit), self.indexes[id(unit)]
        statics = program.unit_statics.setdefault(unit.unit_name, {})
        for name, sym in symbols.defined.items():
            key = sym.key or (unit_static_key(unit.unit_name, name)
                              if sym.linkage == "internal" else name)
            if sym.linkage == "internal":
                statics[name] = key
            nodes = index.functions if sym.kind == "function" else index.variables
            node = nodes.get(name)
            if node is None:
                continue
            program.unit_of[key] = unit.unit_name
            program.definitions[key] = node
            if isinstance(node, ast.FunctionDef):
                program.functions[key] = node
                program.function_ids[key] = len(program.function_table)
                program.function_table.append(node)
            else:
                program.globals[key] = node
        for function, decl in index.static_locals:
            key = static_local_key(unit.unit_name, function.name, decl.name, decl.loc)
            program.globals[key] = decl
            program.definitions[key] = decl
            program.unit_of[key] = unit.unit_name

    def _place_builtins(self, program: LinkedProgram) -> None:
        """Library names the units use: builtin functions also get pointer ids."""
        for unit in self.units:
            for name in _symbols(unit).referenced:
                if program.resolve(unit.unit_name, name) is not None:
                    continue
                spec = self.builtins.get(name)
                if spec is not None:
                    program.definitions[name] = spec
                    program.function_ids[name] = len(program.function_table)
                    program.function_table.append(spec)
                elif name in self.builtins.variables:
                    program.definitions[name] = self.builtins.variables[name]


_EXTERNAL_REF_KINDS = ("global", "function", "builtin")


def _annotate_refs(unit: ObjectFile, program: LinkedProgram) -> None:
    """Write the link-time answer into every ``Identifier.ref`` of external names.

    The checker can only guess whether a declared-but-not-defined name is another
    unit's function or a library builtin; after linking we know for sure.
    """
    assert unit.ast is not None
    for node in ast.walk(unit.ast):
        ref = getattr(node, "ref", None) if isinstance(node, ast.Identifier) else None
        if ref is None or ref.kind not in _EXTERNAL_REF_KINDS or is_static_key(ref.key):
            continue
        if ref.key in program.functions:
            ref.kind = "function"
        elif ref.key in program.globals:
            ref.kind = "global"
        elif ref.key in program.definitions:
            ref.kind = "builtin"


# =========================================================================== helpers

def _symbols(unit: ObjectFile) -> UnitSymbols:
    assert unit.symbols is not None
    return unit.symbols


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0]


def _as_source(path: str) -> str:
    """``util.o`` -> ``util.c`` (what the user should put on the command line)."""
    return posixpath.splitext(path)[0] + ".c" if path.endswith(".o") else path


def _looks_like_function(unit: ObjectFile, name: str) -> bool:
    declared = _symbols(unit).declared.get(name)
    if declared is not None:
        return declared.kind == "function"
    return True     # an undeclared name only reaches the linker through a call


def _diagnostic(code: str, message: str, friendly: str, location: SourceLocation | None,
                phase: Phase = Phase.LINK) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, message, friendly, location,
                      concept_for_code(code), phase)


def _multiple_definition(name: str, first: _Placed, again: _Placed) -> Diagnostic:
    """``multiple_definition`` with ld's "first defined here"."""
    in_function = f"in function '{name}': " if again.sym.kind == "function" else ""
    message = (f"ld: {again.unit.object_name}: {in_function}multiple definition of '{name}'; "
               f"{first.unit.object_name}: first defined here")
    a, b = first.sym.loc, again.sym.loc
    if _same_header(first, again):
        friendly = _header_definition_friendly(name, first, again)
    elif first.sym.kind == "function" and again.sym.kind == "function":
        friendly = (f"'{name}' has a body in both {a.file} (line {a.line}) and {b.file} "
                    f"(line {b.line}). A program may define each function only once: keep "
                    "one definition and put just the prototype in a header both files "
                    "include. If each file needs its own private copy, mark them static.")
    elif first.sym.tentative or again.sym.tentative:
        friendly = (f"Both {a.file} and {b.file} define the global variable '{name}'. Even "
                    f"a declaration without a value, like 'int {name};', is a definition, "
                    "so the linker finds two variables with one name. Declare it with "
                    f"'extern' in a header and define it (without extern) in exactly one "
                    ".c file.")
    else:
        friendly = (f"'{name}' is defined in both {a.file} (line {a.line}) and {b.file} "
                    f"(line {b.line}). Each global name may be defined only once in a "
                    "program: keep one definition and use an 'extern' declaration in the "
                    "other file, or mark both static if they are meant to be separate.")
    return _diagnostic("multiple_definition", message, friendly, b)


def _same_header(first: _Placed, again: _Placed) -> bool:
    """Both definitions come from one header that two .c files include.

    That is the classic "global defined in a header" mistake; it needs its own
    explanation, because "defined in globals.h and globals.h" would only confuse.
    """
    a, b = first.sym.loc, again.sym.loc
    if a is None or b is None or a.file != b.file:
        return False
    return a.file not in (first.unit.unit_name, again.unit.unit_name)


def _header_definition_friendly(name: str, first: _Placed, again: _Placed) -> str:
    """Why a definition in a header breaks as soon as two files include it."""
    header = first.sym.loc.file
    units = f"{first.unit.unit_name} and {again.unit.unit_name}"
    if again.sym.kind == "function":
        fix = (f"Keep only the prototype of '{name}' in {header} and move its body into "
               "exactly one .c file.")
    else:
        fix = (f"In {header} write only a declaration, 'extern' in front and no '= value', "
               f"and define '{name}' (with its value) in exactly one .c file.")
    return (f"'{name}' is defined in the header {header}, and both {units} include it, so "
            "each object file gets its own copy and the linker finds two. Include guards "
            "do not help here: they only stop a header being read twice inside one .c "
            f"file. {fix}")


def _not_defined_friendly(name: str, is_function: bool, suggestion: str | None) -> str:
    if is_function:
        text = (f"Your code calls '{name}', and its prototype told the compiler it exists, "
                "but none of the files you linked contains its body (its definition). "
                "Write the function, or add the .c file that defines it to the cc command.")
    else:
        text = (f"Your code uses the variable '{name}', which is declared 'extern', but no "
                f"file defines it. Define it (without extern) in exactly one .c file.")
    if suggestion:
        text += f" Did you mean '{suggestion}'?"
    return text


def _static_friendly(name: str, is_function: bool, owner: ObjectFile) -> str:
    what = "function" if is_function else "variable"
    return (f"The {what} '{name}' is defined in {owner.unit_name}, but it is marked static "
            f"there, so it is private to that file and other files cannot use it. Remove "
            f"'static' from its definition in {owner.unit_name} if other files need it.")


