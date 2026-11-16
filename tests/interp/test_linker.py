"""Tests for interp.linker: object files, executables, linking and program hashes.

The link layer only relies on the checker's contract: every
Identifier gets a ``ref`` and each unit gets ``UnitSymbols``. These tests use a small
stand-in checker with exactly that contract (``StubChecker`` below), so the linker is
tested on its own; tests/interp/test_compiler.py also runs the real checker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pycc.core.diagnostics import Diagnostic, Phase, Severity, concept_for_code
from pycc.interp import astnodes as ast
from pycc.interp import compiler
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.c_types import INT, CType
from pycc.interp.compiler import build_project, compile_source, link_executable
from pycc.interp.hostio import DictSourceFiles
from pycc.interp.linker import (
    EXECUTABLE_MAGIC, OBJECT_MAGIC, BadExecutable, BadObjectFile, Executable, ObjectFile,
    _pack, link, object_name_for, program_hash,
)
from pycc.interp.symbols import SymbolDef, SymbolRef, UnitSymbols, static_local_key, unit_static_key
from pycc.interp.toolchain import CompileOptions, SystemHeaders, Toolchain

PROJECT = "/home/user/project"

MACHINE_H = """#ifndef CV_MACHINE_H
#define CV_MACHINE_H
int move(int direction);
void say(const char *message);
#endif
"""

MAIN_C = """#include <machine.h>
#include "drone.h"

int main(void) {
    irrigate(3);
    say("done");
    return 0;
}
"""

DRONE_H = """#ifndef DRONE_H
#define DRONE_H
void irrigate(int rows);
extern int water_used;
#endif
"""

DRONE_C = """#include <machine.h>
#include "drone.h"

int water_used = 0;

static int helper(int n) {
    return n * 2;
}

void irrigate(int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        move(1);
        water_used = water_used + helper(1);
    }
}
"""


# =========================================================================== a stand-in checker

@dataclass
class StubCheckResult:
    """Same shape as interp.checker.CheckResult."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    symbols: UnitSymbols | None = None


class StubChecker:
    """Annotates refs and builds UnitSymbols per the D3 contract; records options.

    Scopes are flattened per function (enough for these tests). With ``-Wall`` it warns
    about locals that are never used, so option propagation can be observed.
    """

    def __init__(self) -> None:
        self.calls: list[CompileOptions] = []

    def __call__(self, unit: ast.TranslationUnit, registry: BuiltinRegistry,
                 options: CompileOptions) -> StubCheckResult:
        self.calls.append(options)
        return _StubRun(unit, registry, options).run()


class _StubRun:
    def __init__(self, unit: ast.TranslationUnit, registry: BuiltinRegistry,
                 options: CompileOptions) -> None:
        self.unit, self.registry, self.options = unit, registry, options
        self.symbols = UnitSymbols(unit.name)
        self.file_scope: dict[str, SymbolDef] = {}
        self.enums: dict[str, int] = {}
        self.diagnostics: list[Diagnostic] = []

    def run(self) -> StubCheckResult:
        for decl in self.unit.decls:
            self._declare(decl)
        for decl in self.unit.decls:
            if isinstance(decl, ast.VarDecl) and decl.init is not None:
                self._annotate(decl.init, {})
            elif isinstance(decl, ast.FunctionDef):
                self._function(decl)
        return StubCheckResult(self.diagnostics, self.symbols)

    def _key(self, name: str, linkage: str) -> str:
        return unit_static_key(self.unit.name, name) if linkage == "internal" else name

    def _declare(self, decl: ast.Decl) -> None:
        if isinstance(decl, ast.EnumDecl):
            self.enums.update(dict(decl.constants))
            return
        if isinstance(decl, (ast.FunctionDef, ast.FunctionDecl)):
            kind, ctype, tentative = "function", decl.ftype, False
            is_def = isinstance(decl, ast.FunctionDef)
        elif isinstance(decl, ast.VarDecl):
            kind, ctype, tentative = "variable", decl.ctype, decl.init is None
            is_def = not decl.is_extern
        else:
            return
        linkage = "internal" if decl.storage == "static" else "external"
        sym = SymbolDef(decl.name, kind, linkage, ctype, decl.loc, tentative and is_def, is_def,
                        self._key(decl.name, linkage))
        self.file_scope.setdefault(decl.name, sym)
        if is_def:
            old = self.symbols.defined.get(decl.name)
            if old is None or (old.tentative and not tentative):
                self.symbols.defined[decl.name] = sym
        else:
            self.symbols.declared.setdefault(decl.name, sym)

    def _function(self, func: ast.FunctionDef) -> None:
        local: dict[str, SymbolRef] = {
            name: SymbolRef("param", name, ctype)
            for name, ctype in zip(func.param_names, func.ftype.params) if name}
        declared_locals: dict[str, ast.VarDecl] = {}
        for node in ast.walk(func.body):
            if isinstance(node, ast.VarDecl):
                if node.storage == "static":
                    key = static_local_key(self.unit.name, func.name, node.name, node.loc)
                    self.symbols.static_locals[key] = SymbolDef(
                        node.name, "variable", "internal", node.ctype, node.loc, key=key)
                    local[node.name] = SymbolRef("static_local", key, node.ctype)
                else:
                    local[node.name] = SymbolRef("local", node.name, node.ctype)
                    declared_locals[node.name] = node
        used = self._annotate(func.body, local)
        if self.options.wall:
            for name, decl in declared_locals.items():
                if name not in used:
                    self.diagnostics.append(Diagnostic(
                        Severity.WARNING, "unused_variable", f"unused variable '{name}'",
                        f"'{name}' is never used.", decl.loc,
                        concept_for_code("unused_variable"), Phase.CHECK))

    def _annotate(self, root: ast.Node, local: dict[str, SymbolRef]) -> set[str]:
        used: set[str] = set()
        for node in ast.walk(root):
            if isinstance(node, ast.Identifier):
                node.ref = self._resolve(node, local)  # type: ignore[attr-defined]
                used.add(node.name)
        return used

    def _resolve(self, node: ast.Identifier, local: dict[str, SymbolRef]) -> SymbolRef:
        name = node.name
        if name in local:
            return local[name]
        if name in self.enums:
            return SymbolRef("enum_const", str(self.enums[name]), INT, self.enums[name])
        self.symbols.referenced.setdefault(name, []).append(node.loc)
        sym = self.file_scope.get(name)
        if sym is not None:
            kind = "function" if sym.kind == "function" else "global"
            return SymbolRef(kind, sym.key, sym.ctype)
        ctype: CType = INT
        return SymbolRef("builtin" if name in self.registry else "function", name, ctype)


# =========================================================================== fixtures

def make_registry() -> BuiltinRegistry:
    """A tiny device library: move, say and water."""
    registry = BuiltinRegistry()
    registry.register("move", "int move(int direction);", lambda m, a: 1, header="machine.h")
    registry.register("say", "void say(const char *message);", lambda m, a: None,
                      header="machine.h")
    registry.register("water", "int water(void);", lambda m, a: 1, header="machine.h")
    return registry


def make_toolchain(registry: BuiltinRegistry | None = None) -> Toolchain:
    return Toolchain(registry or make_registry(), SystemHeaders({"machine.h": MACHINE_H}))


def project_files(**files: str) -> DictSourceFiles:
    """DictSourceFiles under the project folder; keyword names use '_' for '.'."""
    return DictSourceFiles({f"{PROJECT}/{name}": text for name, text in files.items()})


@pytest.fixture
def stub_checker(monkeypatch: pytest.MonkeyPatch) -> StubChecker:
    """Route the compile driver (and object loading) through the stand-in checker."""
    checker = StubChecker()
    monkeypatch.setattr(compiler, "load_checker", lambda: checker)
    return checker


def compile_unit(path: str, files: DictSourceFiles, toolchain: Toolchain | None = None,
                 options: CompileOptions | None = None) -> ObjectFile:
    result = compile_source(files, path, toolchain or make_toolchain(), options, cwd=PROJECT)
    assert result.ok, [d.format_gcc() for d in result.diagnostics]
    assert result.object is not None
    return result.object


def compile_texts(sources: dict[str, str], toolchain: Toolchain | None = None) -> list[ObjectFile]:
    files = DictSourceFiles({f"{PROJECT}/{name}": text for name, text in sources.items()})
    files.files.setdefault(f"{PROJECT}/drone.h", DRONE_H)
    return [compile_unit(name, files, toolchain) for name in sources]


def two_file_objects(toolchain: Toolchain | None = None) -> list[ObjectFile]:
    return compile_texts({"main.c": MAIN_C, "drone.c": DRONE_C}, toolchain)


def identifiers(obj: ObjectFile) -> list[ast.Identifier]:
    assert obj.ast is not None
    return [node for node in ast.walk(obj.ast) if isinstance(node, ast.Identifier)]


def only_error(result) -> Diagnostic:
    errors = [d for d in result.diagnostics if d.is_error]
    assert len(errors) == 1, [d.message for d in errors]
    return errors[0]


# =========================================================================== linking a project

def test_two_file_project_links(stub_checker: StubChecker) -> None:
    result = link(two_file_objects(), make_registry())
    assert result.ok and result.diagnostics == []
    program = result.program
    assert program is not None
    assert set(program.functions) == {"main", "irrigate", "drone.c::helper"}
    assert set(program.globals) == {"water_used"}
    assert program.entry == "main" and program.entry_function.name == "main"


def test_program_units_keep_their_order(stub_checker: StubChecker) -> None:
    program = link(two_file_objects(), make_registry()).program
    assert [unit.unit_name for unit in program.units] == ["main.c", "drone.c"]


def test_resolution_order_statics_externals_builtins(stub_checker: StubChecker) -> None:
    program = link(two_file_objects(), make_registry()).program
    assert program.resolve("drone.c", "helper") == "drone.c::helper"
    assert program.resolve("main.c", "helper") is None
    assert program.resolve("main.c", "irrigate") == "irrigate"
    assert program.resolve("drone.c", "move") == "move"


def test_definitions_table_covers_every_key(stub_checker: StubChecker) -> None:
    program = link(two_file_objects(), make_registry()).program
    assert isinstance(program.definitions["irrigate"], ast.FunctionDef)
    assert isinstance(program.definitions["water_used"], ast.VarDecl)
    assert program.definitions["say"] is program.builtins.get("say")
    assert program.unit_of["drone.c::helper"] == "drone.c"


def test_function_table_starts_at_one(stub_checker: StubChecker) -> None:
    program = link(two_file_objects(), make_registry()).program
    assert program.function_table[0] is None
    assert program.function_at(0) is None
    assert program.function_at(len(program.function_table)) is None
    main_id = program.function_id("main")
    assert main_id >= 1 and program.function_at(main_id).name == "main"


def test_referenced_builtins_get_function_ids(stub_checker: StubChecker) -> None:
    program = link(two_file_objects(), make_registry()).program
    say_id = program.function_id("say")
    assert say_id > 0 and program.function_at(say_id) is program.builtins.get("say")
    assert program.function_id("water") == 0     # never referenced


def test_link_marks_refs_with_the_final_kind(stub_checker: StubChecker) -> None:
    main, drone = two_file_objects()
    link([main, drone], make_registry())
    by_name = {node.name: node.ref for node in identifiers(main)}
    assert by_name["irrigate"].kind == "function"
    assert by_name["say"].kind == "builtin"
    helper = next(n.ref for n in identifiers(drone) if n.name == "helper")
    assert helper.key == "drone.c::helper"


def test_static_locals_are_program_globals(stub_checker: StubChecker) -> None:
    source = "int tick(void) {\n    static int count = 0;\n    count++;\n    return count;\n}\n" \
             "int main(void) { return tick(); }\n"
    program = link(compile_texts({"main.c": source}), make_registry()).program
    keys = [key for key in program.globals if "::tick::count@" in key]
    assert keys == ["main.c::tick::count@2:16"]


def test_link_requires_loaded_objects(stub_checker: StubChecker) -> None:
    obj = ObjectFile.decode(two_file_objects()[0].to_bytes())
    with pytest.raises(ValueError):
        link([obj], make_registry())


def test_object_name_for() -> None:
    assert object_name_for("src/drone.c") == "drone.o"
    assert object_name_for("drone.o") == "drone.o"


# =========================================================================== undefined references

def test_undefined_reference_message_is_ld_style(stub_checker: StubChecker) -> None:
    objects = compile_texts({"main.c": MAIN_C})
    diag = only_error(link(objects, make_registry()))
    assert diag.code == "undefined_reference"
    assert diag.message == "ld: main.o: in function 'main': undefined reference to 'irrigate'"
    assert diag.phase is Phase.LINK and diag.severity is Severity.ERROR
    assert diag.location is not None and diag.location.file == "main.c"
    assert diag.location.line == 5


def test_undefined_reference_friendly_not_defined_anywhere(stub_checker: StubChecker) -> None:
    diag = only_error(link(compile_texts({"main.c": MAIN_C}), make_registry()))
    assert diag.friendly == (
        "Your code calls 'irrigate', and its prototype told the compiler it exists, but "
        "none of the files you linked contains its body (its definition). Write the "
        "function, or add the .c file that defines it to the cc command.")
    assert diag.concept.value == "linking"


def test_undefined_reference_suggests_a_close_name(stub_checker: StubChecker) -> None:
    other = "void irigate(int rows) { }\n"
    diag = only_error(link(compile_texts({"main.c": MAIN_C, "pump.c": other}), make_registry()))
    assert diag.friendly.endswith("Did you mean 'irigate'?")


def test_undefined_reference_object_not_given(stub_checker: StubChecker) -> None:
    objects = compile_texts({"main.c": MAIN_C})
    diag = only_error(link(objects, make_registry(), known_sources=["drone.c", "notes.c"]))
    assert diag.code == "undefined_reference"
    assert diag.friendly.startswith("Did you pass drone.o to the linker? drone.h declares "
                                    "'irrigate', and its definition is probably in drone.c")
    assert diag.friendly.endswith("cc main.c drone.c -o <program>")


def test_object_not_given_hint_needs_a_matching_name(stub_checker: StubChecker) -> None:
    objects = compile_texts({"main.c": MAIN_C})
    diag = only_error(link(objects, make_registry(), known_sources=["pump.c"]))
    assert "Did you pass" not in diag.friendly


def test_undefined_reference_static_in_other_file(stub_checker: StubChecker) -> None:
    hidden = "static void irrigate(int rows) { }\n"
    diag = only_error(link(compile_texts({"main.c": MAIN_C, "pump.c": hidden}), make_registry()))
    assert diag.code == "undefined_reference"
    assert diag.message == "ld: main.o: in function 'main': undefined reference to 'irrigate'"
    assert diag.friendly == (
        "The function 'irrigate' is defined in pump.c, but it is marked static there, so it "
        "is private to that file and other files cannot use it. Remove 'static' from its "
        "definition in pump.c if other files need it.")


def test_undefined_extern_variable(stub_checker: StubChecker) -> None:
    source = '#include "drone.h"\nint main(void) {\n    return water_used;\n}\n'
    diag = only_error(link(compile_texts({"main.c": source}), make_registry()))
    assert diag.message == "ld: main.o: in function 'main': undefined reference to 'water_used'"
    assert diag.friendly.startswith("Your code uses the variable 'water_used', which is "
                                    "declared 'extern', but no file defines it.")


def test_undefined_reference_from_a_global_initializer(stub_checker: StubChecker) -> None:
    source = "extern int total;\nint *where = &total;\nint main(void) { return 0; }\n"
    diag = only_error(link(compile_texts({"main.c": source}), make_registry()))
    assert diag.message == "ld: main.o:(.data+0x0): undefined reference to 'total'"


def test_one_error_per_function_and_name(stub_checker: StubChecker) -> None:
    source = ("void irrigate(int rows);\nint main(void) {\n    irrigate(1);\n    irrigate(2);\n"
              "    return 0;\n}\nvoid later(void) { irrigate(3); }\n")
    result = link(compile_texts({"main.c": source}), make_registry())
    messages = [d.message for d in result.diagnostics]
    assert messages == [
        "ld: main.o: in function 'main': undefined reference to 'irrigate'",
        "ld: main.o: in function 'later': undefined reference to 'irrigate'",
    ]

def test_own_definition_beats_a_builtin(stub_checker: StubChecker) -> None:
    source = "int water(void) { return 7; }\nint main(void) { return water(); }\n"
    result = link(compile_texts({"main.c": source}), make_registry())
    assert result.ok and "water" in result.program.functions


# =========================================================================== multiple definitions

def test_multiple_definition_of_a_function(stub_checker: StubChecker) -> None:
    again = '#include "drone.h"\nvoid irrigate(int rows) {\n}\n'
    objects = compile_texts({"main.c": MAIN_C, "drone.c": DRONE_C, "pump.c": again})
    diag = only_error(link(objects, make_registry()))
    assert diag.code == "multiple_definition"
    assert diag.message == ("ld: pump.o: in function 'irrigate': multiple definition of "
                            "'irrigate'; drone.o: first defined here")
    assert diag.location.file == "pump.c" and diag.location.line == 2
    assert diag.friendly.startswith("'irrigate' has a body in both drone.c (line 10) and "
                                    "pump.c (line 2).")
    assert diag.concept.value == "linkage"


def test_multiple_definition_of_tentative_globals(stub_checker: StubChecker) -> None:
    objects = compile_texts({"a.c": "int count;\nint main(void) { return count; }\n",
                             "b.c": "int count;\n"})
    diag = only_error(link(objects, make_registry()))
    assert diag.message == "ld: b.o: multiple definition of 'count'; a.o: first defined here"
    assert "Even a declaration without a value, like 'int count;', is a definition" in diag.friendly


def test_multiple_definition_of_initialized_globals(stub_checker: StubChecker) -> None:
    objects = compile_texts({"a.c": "int count = 1;\nint main(void) { return count; }\n",
                             "b.c": "int count = 2;\n"})
    diag = only_error(link(objects, make_registry()))
    assert diag.message == "ld: b.o: multiple definition of 'count'; a.o: first defined here"
    assert diag.friendly.startswith("'count' is defined in both a.c (line 1) and b.c (line 1).")


def test_tentative_and_extern_link_fine(stub_checker: StubChecker) -> None:
    objects = compile_texts({"a.c": "int count;\nint main(void) { return count; }\n",
                             "b.c": "extern int count;\nvoid bump(void) { count++; }\n"})
    assert link(objects, make_registry()).ok


def test_two_statics_with_one_name_do_not_collide(stub_checker: StubChecker) -> None:
    objects = compile_texts({"a.c": "static int count;\nint main(void) { return count; }\n",
                             "b.c": "static int count;\n"})
    program = link(objects, make_registry()).program
    assert {"a.c::count", "b.c::count"} <= set(program.globals)


def test_same_source_linked_twice(stub_checker: StubChecker) -> None:
    main, drone = two_file_objects()
    diag = only_error(link([main, drone, drone], make_registry()))
    assert diag.code == "multiple_definition"
    assert diag.message == ("ld: drone.o: multiple definition of 'water_used'; drone.o: "
                            "first defined here")
    assert diag.friendly.startswith("drone.c was given to the linker twice")


# =========================================================================== main

def test_no_main(stub_checker: StubChecker) -> None:
    objects = compile_texts({"drone.c": DRONE_C})
    diag = only_error(link(objects, make_registry()))
    assert diag.code == "no_main"
    assert diag.message == "ld: crt1.o: in function '_start': undefined reference to 'main'"
    assert diag.location is None
    assert diag.friendly.startswith("Every C program starts in main(), but none of the files")


def test_static_main_is_not_an_entry(stub_checker: StubChecker) -> None:
    objects = compile_texts({"main.c": "static int main(void) { return 0; }\n"})
    diag = only_error(link(objects, make_registry()))
    assert diag.code == "no_main"
    assert "main() in main.c is marked static" in diag.friendly


def test_other_entry_point(stub_checker: StubChecker) -> None:
    objects = compile_texts({"t.c": "int test_case(void) { return 0; }\n"})
    assert link(objects, make_registry(), entry="test_case").ok


# =========================================================================== object files

def test_object_bytes_start_with_magic(stub_checker: StubChecker) -> None:
    data = two_file_objects()[0].to_bytes()
    assert data.startswith(OBJECT_MAGIC)


def test_object_bytes_are_deterministic(stub_checker: StubChecker) -> None:
    assert two_file_objects()[1].to_bytes() == two_file_objects()[1].to_bytes()


def test_object_round_trip_keeps_fields(stub_checker: StubChecker) -> None:
    original = compile_unit("drone.c", project_files(**{"drone.c": DRONE_C, "drone.h": DRONE_H}),
                            options=CompileOptions(["inc"], {"X": "1"}, wall=True))
    restored = ObjectFile.from_bytes(original.to_bytes(), make_toolchain())
    assert restored.unit_name == "drone.c"
    assert restored.source_path == f"{PROJECT}/drone.c"
    assert restored.cwd == PROJECT
    assert restored.pp_lines == original.pp_lines
    assert restored.options == original.options


def test_from_bytes_restores_every_annotation(stub_checker: StubChecker) -> None:
    for original in two_file_objects():
        restored = ObjectFile.from_bytes(original.to_bytes(), make_toolchain())
        assert restored.is_loaded
        names = identifiers(restored)
        assert names and all(getattr(node, "ref", None) is not None for node in names)
        assert restored.symbols.defined.keys() == original.symbols.defined.keys()


def test_from_bytes_accepts_a_registry(stub_checker: StubChecker) -> None:
    obj = two_file_objects()[0]
    assert ObjectFile.from_bytes(obj.to_bytes(), make_registry()).is_loaded


def test_restored_objects_link(stub_checker: StubChecker) -> None:
    restored = [ObjectFile.from_bytes(o.to_bytes(), make_toolchain()) for o in two_file_objects()]
    assert link(restored, make_registry()).ok


def test_bad_object_magic() -> None:
    with pytest.raises(BadObjectFile) as caught:
        ObjectFile.from_bytes(b"hello device\n", make_toolchain())
    diag = caught.value.diagnostic("drone.o")
    assert diag.code == "bad_object_file"
    assert diag.message == "drone.o: file not recognized: file format not recognized"
    assert diag.concept.value == "build_cc"


def test_bad_object_version() -> None:
    data = OBJECT_MAGIC + _pack({"version": 99, "unit_name": "a.c"})
    with pytest.raises(BadObjectFile, match="version"):
        ObjectFile.decode(data)


def test_damaged_object_body() -> None:
    with pytest.raises(BadObjectFile):
        ObjectFile.decode(OBJECT_MAGIC + b"not zlib at all")


def test_executable_cannot_be_read_as_object(stub_checker: StubChecker) -> None:
    exe = Executable(two_file_objects())
    with pytest.raises(BadObjectFile):
        ObjectFile.decode(exe.to_bytes())


# =========================================================================== executables

def test_executable_round_trip(stub_checker: StubChecker) -> None:
    exe = Executable(two_file_objects())
    data = exe.to_bytes()
    assert data.startswith(EXECUTABLE_MAGIC)
    loaded = Executable.from_bytes(data)
    assert [o.unit_name for o in loaded.objects] == ["main.c", "drone.c"]
    assert not any(o.is_loaded for o in loaded.objects)
    result = link_executable(loaded, make_registry())
    assert result.ok and set(result.program.functions) >= {"main", "irrigate"}


def test_executable_bad_magic() -> None:
    with pytest.raises(BadExecutable) as caught:
        Executable.from_bytes(b"#!/bin/sh\necho hi\n")
    diag = caught.value.diagnostic("./farm")
    assert diag.code == "bad_executable"
    assert diag.message == "./farm: cannot execute binary file: Exec format error"


def test_executable_bad_version(stub_checker: StubChecker) -> None:
    data = bytearray(Executable(two_file_objects()).to_bytes())
    data[len(EXECUTABLE_MAGIC)] = 99
    with pytest.raises(BadExecutable, match="version"):
        Executable.from_bytes(bytes(data))


def test_executable_damaged_body() -> None:
    with pytest.raises(BadExecutable):
        Executable.from_bytes(EXECUTABLE_MAGIC + bytes([1]) + b"garbage")


def test_link_executable_on_another_machine(stub_checker: StubChecker) -> None:
    exe = Executable.from_bytes(Executable(two_file_objects()).to_bytes())
    smaller = BuiltinRegistry()
    smaller.register("say", "void say(const char *message);", lambda m, a: None)
    result = link_executable(exe, smaller)
    assert not result.ok
    assert [d.message for d in result.diagnostics] == [
        "ld: drone.o: in function 'irrigate': undefined reference to 'move'"]


# =========================================================================== program hash

HASH_BASE = """int main(void) {
    int total = 0;
    for (int i = 0; i < 3; i++) {
        total += i;
    }
    return total;
}
"""


def hash_of(source: str) -> str:
    return program_hash(compile_texts({"main.c": source}))


def test_hash_is_a_sha1(stub_checker: StubChecker) -> None:
    digest = hash_of(HASH_BASE)
    assert len(digest) == 40 and int(digest, 16) >= 0


def test_hash_ignores_whitespace_and_comments(stub_checker: StubChecker) -> None:
    variant = ("/* counting */\nint main(void){int total=0; // start\n"
               "for(int i=0;i<3;i++){total+=i;}\n\n\nreturn total;}\n")
    assert hash_of(variant) == hash_of(HASH_BASE)


def test_hash_ignores_identifier_names(stub_checker: StubChecker) -> None:
    variant = ("int main(void) {\n    int sum = 0;\n    for (int k = 0; k < 3; k++) {\n"
               "        sum += k;\n    }\n    return sum;\n}\n")
    assert hash_of(variant) == hash_of(HASH_BASE)


def test_hash_changes_with_real_changes(stub_checker: StubChecker) -> None:
    assert hash_of(HASH_BASE.replace("i < 3", "i < 4")) != hash_of(HASH_BASE)
    assert hash_of(HASH_BASE.replace("total += i", "total -= i")) != hash_of(HASH_BASE)


def test_hash_tells_apart_two_different_variables(stub_checker: StubChecker) -> None:
    one = "int main(void) { int a = 1; int b = 2; return a; }\n"
    two = "int main(void) { int a = 1; int b = 2; return b; }\n"
    assert hash_of(one) != hash_of(two)


def test_hash_of_executable_equals_hash_of_objects(stub_checker: StubChecker) -> None:
    objects = two_file_objects()
    assert program_hash(Executable(objects)) == program_hash(objects)
    assert program_hash(list(reversed(objects))) == program_hash(objects)


# =========================================================================== build + link together

def test_build_project_reports_link_errors(stub_checker: StubChecker) -> None:
    files = project_files(**{"main.c": MAIN_C, "drone.h": DRONE_H, "drone.c": DRONE_C})
    build = build_project(files, ["main.c"], "farm", make_toolchain(), cwd=PROJECT,
                          known_sources=["main.c", "drone.c"])
    assert not build.ok and build.executable is None
    assert build.diagnostics[-1].friendly.startswith("Did you pass drone.o to the linker?")
