"""Review tests for interp.compiler and interp.includes with the real checker.

Focus: paths shown the way cc shows them for files in subfolders, header lookup from
subfolders, -Werror, and the cc command line a build reports.
"""
from __future__ import annotations

from pycc.core.diagnostics import Severity
from pycc.interp.compiler import (
    WERROR_SUFFIX, build_project, compile_source, format_cc_command,
)
from pycc.interp.hostio import DictSourceFiles
from pycc.interp.linker import object_name_for
from pycc.interp.stdlib import make_registry
from pycc.interp.stdlib.headers import system_headers
from pycc.interp.toolchain import CompileOptions, Toolchain

PROJECT = "/home/user/project"


def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers(), CompileOptions())


def files(**named: str) -> DictSourceFiles:
    """Project files from a dict of relative paths (passed as **{"src/a.c": ...})."""
    return DictSourceFiles({f"{PROJECT}/{path}": text for path, text in named.items()})


def first_error(result) -> str:
    return next(d for d in result.diagnostics if d.is_error).format_gcc()


def test_error_in_a_subfolder_source_shows_the_path_as_given() -> None:
    project = files(**{"src/bad.c": "int f(void) {\n    return nope;\n}\n"})
    result = compile_source(project, "src/bad.c", toolchain(), cwd=PROJECT)
    assert not result.ok
    assert first_error(result).startswith("src/bad.c:2:")


def test_dot_slash_path_is_kept_as_written() -> None:
    project = files(**{"main.c": "int main(void) {\n    return nope;\n}\n"})
    result = compile_source(project, "./main.c", toolchain(), cwd=PROJECT)
    assert first_error(result).startswith("./main.c:2:")
    assert result.unit_name == "./main.c" and result.cwd == PROJECT


def test_parent_folder_source_is_found_and_shown_as_given() -> None:
    project = files(**{"lib/util.c": "int util(void) {\n    return nope;\n}\n"})
    result = compile_source(project, "../lib/util.c", toolchain(), cwd=f"{PROJECT}/src")
    assert first_error(result).startswith("../lib/util.c:2:")


def test_quoted_include_next_to_a_subfolder_source() -> None:
    project = files(**{"src/gear.h": "int gear(void)\n", "src/gear.c": '#include "gear.h"\n'})
    result = compile_source(project, "src/gear.c", toolchain(), cwd=PROJECT)
    # The missing ';' is reported inside the header, under the path cc resolved.
    assert first_error(result).startswith("src/gear.h:")


def test_object_built_from_subfolder_source_remembers_both_paths() -> None:
    project = files(**{"src/ok.c": "int ok(void) { return 1; }\n"})
    result = compile_source(project, "src/ok.c", toolchain(), cwd=PROJECT)
    assert result.object is not None
    assert result.object.unit_name == "src/ok.c"
    assert result.object.source_path == f"{PROJECT}/src/ok.c"
    assert result.object.object_name == object_name_for("src/ok.c") == "ok.o"


def test_missing_subfolder_source() -> None:
    result = compile_source(files(), "src/none.c", toolchain(), cwd=PROJECT)
    assert result.diagnostics[0].message == "cc: error: src/none.c: No such file or directory"


def test_build_from_subfolders_with_include_dir() -> None:
    project = files(**{
        "include/parts.h": "int part_count(void);\n",
        "src/parts.c": '#include "parts.h"\nint part_count(void) { return 4; }\n',
        "src/main.c": '#include "parts.h"\nint main(void) { return part_count(); }\n',
    })
    build = build_project(project, ["src/main.c", "src/parts.c"], "bin/foundry", toolchain(),
                          CompileOptions(include_dirs=["include"]), cwd=PROJECT)
    assert build.ok, [d.format_gcc() for d in build.diagnostics]
    assert build.commands == ["cc -Iinclude src/main.c src/parts.c -o bin/foundry"]


def test_format_cc_command_quotes_odd_paths() -> None:
    command = format_cc_command(["my farm/main.c"], "farm bot", CompileOptions(wall=True))
    assert command == "cc -Wall 'my farm/main.c' -o 'farm bot'"


def test_werror_from_the_real_checker_stops_the_object() -> None:
    project = files(**{"main.c": "int main(void) {\n    int unused;\n    return 0;\n}\n"})
    result = compile_source(project, "main.c", toolchain(),
                            CompileOptions(wall=True, werror=True), cwd=PROJECT)
    assert not result.ok and result.object is None
    (diag,) = [d for d in result.diagnostics if d.code == "unused_variable"]
    assert diag.severity is Severity.ERROR
    assert diag.message.endswith(WERROR_SUFFIX)
    assert not diag.message.endswith(WERROR_SUFFIX * 2)


def test_build_options_are_not_shared_between_objects() -> None:
    project = files(**{"a.c": "int a(void) { return 1; }\n",
                       "main.c": "int a(void);\nint main(void) { return a(); }\n"})
    options = CompileOptions(include_dirs=["include"])
    build = build_project(project, ["main.c", "a.c"], "prog", toolchain(), options, cwd=PROJECT)
    first, second = build.objects
    first.options.include_dirs.append("changed")
    assert second.options.include_dirs == ["include"]
    assert options.include_dirs == ["include"]
