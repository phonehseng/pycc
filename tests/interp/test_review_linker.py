"""Review tests for interp.linker, run with the real checker and C library.

They cover what a multi-file project does: reloading executables, mixing
objects built with different -I folders, private (static) names in several files, and
running a program built by a toolchain with other device functions.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from pycc.interp import astnodes as ast
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.compiler import (
    ANOTHER_MACHINE_NOTE, build_project, compile_source, compile_text, link_executable,
    load_executable,
)
from pycc.interp.hostio import DictSourceFiles
from pycc.interp.linker import (
    OBJECT_MAGIC, BadExecutable, BadObjectFile, Executable, ObjectFile, _pack, link,
    program_hash,
)
from pycc.interp.source import PPLine
from pycc.interp.stdlib import make_registry
from pycc.interp.stdlib.headers import system_headers
from pycc.interp.toolchain import CompileOptions, Toolchain

PROJECT = "/home/user/project"
MACHINE_H = "#ifndef CV_MACHINE_H\n#define CV_MACHINE_H\nint move(int direction);\n#endif\n"


def zone_toolchain(with_move: bool = True) -> Toolchain:
    """The C library plus (optionally) a one-function device library."""
    registry = make_registry()
    headers = system_headers()
    if with_move:
        registry.register("move", "int move(int direction);", lambda m, a: 1,
                          header="machine.h")
        headers.add("machine.h", MACHINE_H)
    return Toolchain(registry, headers, CompileOptions())


def project(**files: str) -> DictSourceFiles:
    """Project files; keyword names use '__' for '/' and '_' for the last '.'."""
    def path(name: str) -> str:
        stem, _, ext = name.replace("__", "/").rpartition("_")
        return f"{PROJECT}/{stem}.{ext}"
    return DictSourceFiles({path(name): text for name, text in files.items()})


def compile_ok(files: DictSourceFiles, path: str, toolchain: Toolchain,
               options: CompileOptions | None = None) -> ObjectFile:
    result = compile_source(files, path, toolchain, options, cwd=PROJECT)
    assert result.ok, [d.format_gcc() for d in result.diagnostics]
    assert result.object is not None
    return result.object


def ref_keys(objects: list[ObjectFile]) -> list[tuple[str, str, int, int, str, str]]:
    """Every identifier's (unit, name, line, column, ref kind, ref key)."""
    found = []
    for obj in objects:
        assert obj.ast is not None
        for node in ast.walk(obj.ast):
            if isinstance(node, ast.Identifier) and node.ref is not None:
                found.append((obj.unit_name, node.name, node.loc.line, node.loc.column,
                              node.ref.kind, node.ref.key))
    return found


MAIN_C = """#include <stdio.h>
#include <machine.h>
#include "drone.h"

static int helper(void) {
    static int calls;
    return ++calls;
}

int main(void) {
    move(1);
    printf("%d\\n", helper() + drive() + total);
    return 0;
}
"""

DRONE_C = """#include "drone.h"

extern int total;
int total = 3;

static int helper(void) {
    static int base = 4;
    return base;
}

int drive(void) {
    return helper();
}
"""

DRONE_H = "int drive(void);\nextern int total;\n"


def two_units(toolchain: Toolchain | None = None) -> list[ObjectFile]:
    """main.c (no -I) and src/drone.c (built with -Iinclude)."""
    tc = toolchain or zone_toolchain()
    files = project(main_c=MAIN_C, drone_h=DRONE_H, src__drone_c=DRONE_C,
                    include__drone_h=DRONE_H)
    return [compile_ok(files, "main.c", tc),
            compile_ok(files, "src/drone.c", tc, CompileOptions(include_dirs=["include"]))]


# =========================================================================== reload determinism

def test_from_bytes_restores_the_same_keys() -> None:
    objects = two_units()
    restored = [ObjectFile.from_bytes(o.to_bytes(), zone_toolchain()) for o in objects]
    assert ref_keys(restored) == ref_keys(objects)


def test_reloaded_executable_links_to_the_same_program() -> None:
    objects = two_units()
    first = link(objects, zone_toolchain().builtins)
    again = link_executable(load_executable(Executable(objects).to_bytes()),
                            zone_toolchain().builtins)
    assert first.ok and again.ok
    assert first.program is not None and again.program is not None
    assert list(again.program.functions) == list(first.program.functions)
    assert list(again.program.globals) == list(first.program.globals)
    assert again.program.function_ids == first.program.function_ids


def test_executable_bytes_survive_a_round_trip_unchanged() -> None:
    data = Executable(two_units()).to_bytes()
    assert Executable(load_executable(data).objects).to_bytes() == data


def test_objects_with_different_include_dirs_link() -> None:
    main, drone = two_units()
    assert main.options.include_dirs == [] and drone.options.include_dirs == ["include"]
    assert link([main, drone], zone_toolchain().builtins).ok


def test_include_dirs_survive_reloading() -> None:
    drone = two_units()[1]
    assert ObjectFile.decode(drone.to_bytes()).options.include_dirs == ["include"]


# =========================================================================== names

def test_extern_declaration_and_definition_in_one_unit() -> None:
    program = link(two_units(), zone_toolchain().builtins).program
    assert program is not None
    assert program.unit_of["total"] == "src/drone.c"
    assert program.globals["total"].init is not None


def test_static_function_reused_in_two_units() -> None:
    program = link(two_units(), zone_toolchain().builtins).program
    assert program is not None
    assert program.resolve("main.c", "helper") == "main.c::helper"
    assert program.resolve("src/drone.c", "helper") == "src/drone.c::helper"
    ids = {program.function_id("main.c::helper"), program.function_id("src/drone.c::helper")}
    assert len(ids) == 2 and 0 not in ids


def test_static_locals_of_same_named_functions_stay_apart() -> None:
    program = link(two_units(), zone_toolchain().builtins).program
    assert program is not None
    statics = [key for key in program.globals if "::helper::" in key]
    assert sorted(statics) == ["main.c::helper::calls@6:16", "src/drone.c::helper::base@7:16"]


def test_global_initializer_reference_is_not_blamed_on_a_function() -> None:
    source = ("extern int missing;\nint main(void) { return 0; }\n"
              "int *where = &missing;\n")
    result = compile_text(source, "main.c", zone_toolchain())
    assert result.object is not None
    linked = link([result.object], zone_toolchain().builtins)
    assert [d.message for d in linked.diagnostics] == [
        "ld: main.o:(.data+0x0): undefined reference to 'missing'"]


def test_undefined_reference_in_a_subdirectory_unit() -> None:
    files = project(src__util_c="int helper(void);\nint util(void) { return helper(); }\n",
                    main_c="int util(void);\nint main(void) { return util(); }\n")
    tc = zone_toolchain()
    linked = link([compile_ok(files, "main.c", tc), compile_ok(files, "src/util.c", tc)],
                  tc.builtins)
    (diag,) = linked.diagnostics
    assert diag.message == "ld: util.o: in function 'util': undefined reference to 'helper'"
    assert diag.location is not None and diag.location.file == "src/util.c"


# =========================================================================== another toolchain

def built_executable() -> bytes:
    return Executable(two_units()).to_bytes()


def test_another_toolchain_without_the_function_explains_why() -> None:
    result = link_executable(load_executable(built_executable()),
                             zone_toolchain(with_move=False).builtins, "prog")
    assert not result.ok
    (diag,) = result.diagnostics
    assert diag.code == "undefined_reference"
    assert ANOTHER_MACHINE_NOTE in diag.notes
    assert "toolchain" in diag.friendly and "Write the function" not in diag.friendly
    assert "'move'" in diag.friendly

def test_same_toolchain_link_has_no_note() -> None:
    result = link_executable(load_executable(built_executable()), zone_toolchain().builtins)
    assert result.ok and result.diagnostics == []


def test_note_is_added_only_once_when_linking_again() -> None:
    exe = load_executable(built_executable())
    registry = zone_toolchain(with_move=False).builtins
    first = link_executable(exe, registry).diagnostics[0]
    assert first.notes.count(ANOTHER_MACHINE_NOTE) == 1


# =========================================================================== damaged files

def test_object_payload_with_bad_options_is_a_bad_object() -> None:
    payload = ObjectFile.decode(two_units()[0].to_bytes()).to_payload()
    payload["options"] = ["not", "a", "dict"]
    with pytest.raises(BadObjectFile):
        ObjectFile.decode(OBJECT_MAGIC + _pack(payload))


def test_object_whose_code_no_longer_checks_is_rejected() -> None:
    obj = ObjectFile.decode(compile_text("int main(void) { return 0; }\n", "main.c",
                                         zone_toolchain()).object.to_bytes())
    obj.pp_lines.append(PPLine("int main(void) { return undefined_name; }", "main.c", 2))
    with pytest.raises(BadObjectFile):
        obj.load(zone_toolchain().builtins)


def test_executable_with_broken_code_is_a_bad_executable() -> None:
    exe = load_executable(built_executable())
    exe.objects[0].pp_lines.append(PPLine("int x = y;", "main.c", 99))
    result = link_executable(exe, zone_toolchain().builtins, "prog")
    assert [d.code for d in result.diagnostics] == ["bad_executable"]


# =========================================================================== hashes and size

def test_program_hash_ignores_command_line_order() -> None:
    main, drone = two_units()
    assert program_hash([main, drone]) == program_hash([drone, main])


def test_program_hash_is_a_fixed_value_for_fixed_code() -> None:
    # A literal expected value: if dict ordering, set iteration or anything else that
    # varies between machines leaked into the hash, this would stop matching somewhere.
    obj = compile_text("int main(void) { int a = 1; return a; }\n", "main.c",
                       Toolchain(BuiltinRegistry(), system_headers(), CompileOptions())).object
    assert obj is not None
    assert program_hash([obj]) == "5c0d1e679b3567adf768f74909ecd21df9be3563"


def test_program_hash_survives_reloading() -> None:
    objects = two_units()
    assert program_hash(objects) == program_hash(Executable.from_bytes(
        Executable(objects).to_bytes()))


def test_large_preprocessed_units_compress_well() -> None:
    body = "".join(f"    total += SQUARE(i + {k});\n" for k in range(500))
    source = ("#include <stdio.h>\n#define SQUARE(x) ((x) * (x))\nint main(void) {\n"
              "    int total = 0;\n    int i = 3;\n" + body + "    return total;\n}\n")
    obj = compile_text(source, "main.c", zone_toolchain()).object
    assert obj is not None
    text_size = sum(len(line.text) for line in obj.pp_lines)
    assert len(obj.to_bytes()) < text_size / 3


@pytest.mark.parametrize("make", [lambda: b"", lambda: b"CVEXE1\n"])
def test_empty_or_truncated_executable(make: Callable[[], bytes]) -> None:
    with pytest.raises(BadExecutable):
        load_executable(make())
