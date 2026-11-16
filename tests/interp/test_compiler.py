"""Tests for the compile driver (interp.compiler), include resolution (interp.includes)
and the prototype helper (interp.parser.parse_prototype).

Most tests route the checker through the stand-in from test_linker so they test the
driver alone; the tests at the end run the real checker when it is available.
"""
from __future__ import annotations

import pytest

from pycc.core.diagnostics import Severity
from pycc.interp import astnodes as ast
from pycc.interp.c_types import INT, LONG, ULONG, FunctionType, PointerType, StructType, VOID
from pycc.interp.compiler import (
    WERROR_SUFFIX, build_project, compile_source, compile_text, format_cc_command,
    link_executable, load_executable,
)
from pycc.interp.hostio import DictSourceFiles
from pycc.interp.includes import IncludeResolverImpl
from pycc.interp.linker import BadExecutable, Executable, ObjectFile
from pycc.interp.parser import parse_prototype
from pycc.interp.source import PPLine
from pycc.interp.toolchain import CompileOptions, SystemHeaders
from tests.interp.test_linker import (
    DRONE_C, DRONE_H, MAIN_C, PROJECT, StubChecker, make_registry, make_toolchain,
    project_files, stub_checker,  # noqa: F401 - stub_checker is a fixture used by name
)


def resolver(files: dict[str, str], include_dirs: list[str] | None = None,
             system: dict[str, str] | None = None) -> IncludeResolverImpl:
    sources = DictSourceFiles({f"{PROJECT}/{path}": text for path, text in files.items()})
    return IncludeResolverImpl(sources, PROJECT, include_dirs or [],
                               SystemHeaders(system if system is not None else {"stdio.h": "S"}))


# =========================================================================== include resolution

def test_quoted_include_looks_next_to_the_including_file_first() -> None:
    found = resolver({"src/util.h": "SRC", "inc/util.h": "INC"}, ["inc"]).resolve(
        "util.h", True, "src/main.c")
    assert found == ("src/util.h", "SRC")


def test_quoted_include_then_tries_the_I_folders_in_order() -> None:
    found = resolver({"a/util.h": "A", "b/util.h": "B"}, ["a", "b"]).resolve(
        "util.h", True, "main.c")
    assert found == ("a/util.h", "A")


def test_quoted_include_falls_back_to_system_headers() -> None:
    assert resolver({}).resolve("stdio.h", True, "main.c") == ("/usr/include/stdio.h", "S")


def test_quoted_include_prefers_the_project_over_system() -> None:
    found = resolver({"stdio.h": "MINE"}).resolve("stdio.h", True, "main.c")
    assert found == ("stdio.h", "MINE")


def test_angled_include_prefers_system_headers_over_I_folders() -> None:
    found = resolver({"inc/stdio.h": "MINE"}, ["inc"]).resolve("stdio.h", False, "main.c")
    assert found == ("/usr/include/stdio.h", "S")


def test_angled_include_then_tries_I_folders() -> None:
    found = resolver({"include/parts.h": "P"}, ["include"]).resolve("parts.h", False, "main.c")
    assert found == ("include/parts.h", "P")


def test_angled_include_never_looks_next_to_the_file() -> None:
    assert resolver({"parts.h": "P"}).resolve("parts.h", False, "main.c") is None


def test_include_paths_are_normalised() -> None:
    found = resolver({"include/parts.h": "P"}, ["./include/"]).resolve("parts.h", True, "main.c")
    assert found == ("include/parts.h", "P")


def test_include_with_a_relative_folder() -> None:
    found = resolver({"common/defs.h": "D"}).resolve("../common/defs.h", True, "src/main.c")
    assert found == ("common/defs.h", "D")


def test_absolute_include_is_shown_as_written() -> None:
    found = resolver({"x.h": "X"}).resolve(f"{PROJECT}/x.h", True, "main.c")
    assert found == (f"{PROJECT}/x.h", "X")


def test_system_header_including_a_sibling_with_quotes() -> None:
    found = resolver({}, system={"farm.h": "F", "machine.h": "M"}).resolve(
        "machine.h", True, "/usr/include/farm.h")
    assert found == ("/usr/include/machine.h", "M")


def test_missing_include_resolves_to_none() -> None:
    assert resolver({}).resolve("nope.h", True, "main.c") is None


def test_unreadable_file_counts_as_missing() -> None:
    class Broken(DictSourceFiles):
        def read_text(self, path: str) -> str:
            raise OSError("disk on fire")

    broken = IncludeResolverImpl(Broken({f"{PROJECT}/a.h": "A"}), PROJECT, [], SystemHeaders())
    assert broken.resolve("a.h", True, "main.c") is None


# =========================================================================== diagnostic paths

def test_error_in_a_header_points_at_the_header(stub_checker: StubChecker) -> None:
    files = project_files(**{"main.c": '#include "drone.h"\nint main(void) { return 0; }\n',
                             "drone.h": "int broken(void)\n"})
    result = compile_source(files, "main.c", make_toolchain(), cwd=PROJECT)
    assert not result.ok and result.object is None
    assert result.diagnostics[0].location.file == "drone.h"


def test_error_in_an_I_header_uses_the_resolved_path(stub_checker: StubChecker) -> None:
    files = project_files(**{"main.c": '#include "parts.h"\nint main(void) { return 0; }\n',
                             "include/parts.h": "int x = ;\n"})
    result = compile_source(files, "main.c", make_toolchain(),
                            CompileOptions(include_dirs=["include"]), cwd=PROJECT)
    assert result.diagnostics[0].location.file == "include/parts.h"


def test_source_paths_are_shown_as_given(stub_checker: StubChecker) -> None:
    files = project_files(**{"src/main.c": "int main(void) { return 0 }\n"})
    result = compile_source(files, "src/main.c", make_toolchain(), cwd=PROJECT)
    assert result.unit_name == "src/main.c" and result.cwd == PROJECT
    assert result.diagnostics[0].location.file == "src/main.c"


def test_system_header_lines_use_usr_include(stub_checker: StubChecker) -> None:
    files = project_files(**{"main.c": "#include <machine.h>\nint main(void) { return 0; }\n"})
    result = compile_source(files, "main.c", make_toolchain(), cwd=PROJECT)
    assert result.ok
    assert "/usr/include/machine.h" in {line.file for line in result.object.pp_lines}
    assert result.pp_result.includes == ["/usr/include/machine.h"]


def test_missing_header_is_fatal_and_stops_the_unit(stub_checker: StubChecker) -> None:
    files = project_files(**{"main.c": '#include "drone.h"\nint main(void) { return 0 }\n'})
    result = compile_source(files, "main.c", make_toolchain(), cwd=PROJECT)
    assert [d.code for d in result.diagnostics] == ["include_not_found"]
    diag = result.diagnostics[0]
    assert diag.severity is Severity.FATAL
    assert diag.message == "drone.h: No such file or directory"
    assert diag.location.file == "main.c" and diag.location.line == 1
    assert stub_checker.calls == []


def test_missing_source_file(stub_checker: StubChecker) -> None:
    result = compile_source(project_files(), "ghost.c", make_toolchain(), cwd=PROJECT)
    assert not result.ok and result.pp_result is None
    assert result.diagnostics[0].code == "no_such_file"
    assert result.diagnostics[0].message == "cc: error: ghost.c: No such file or directory"


def test_parse_errors_skip_the_checker(stub_checker: StubChecker) -> None:
    result = compile_text("int main(void) { return 0 }\n", "main.c", make_toolchain())
    assert not result.ok
    assert result.diagnostics[0].code == "missing_semicolon"
    assert stub_checker.calls == []


# =========================================================================== options

def test_command_line_defines_reach_the_preprocessor(stub_checker: StubChecker) -> None:
    source = ("#ifdef DEBUG\nint debug_level = LEVEL;\n#endif\n"
              "int main(void) { return 0; }\n")
    options = CompileOptions(defines={"DEBUG": "1", "LEVEL": "3"})
    result = compile_text(source, "main.c", make_toolchain(), options)
    assert result.ok
    assert "int debug_level = 3;" in result.pp_result.text
    without = compile_text(source, "main.c", make_toolchain())
    assert "debug_level" not in without.pp_result.text


def test_toolchain_default_options_are_used(stub_checker: StubChecker) -> None:
    toolchain = make_toolchain()
    toolchain.default_options = CompileOptions(wall=True)
    compile_text("int main(void) { return 0; }\n", "main.c", toolchain)
    assert stub_checker.calls[-1].wall is True


def test_wall_reaches_the_checker(stub_checker: StubChecker) -> None:
    source = "int main(void) {\n    int unused;\n    return 0;\n}\n"
    quiet = compile_text(source, "main.c", make_toolchain())
    loud = compile_text(source, "main.c", make_toolchain(), CompileOptions(wall=True))
    assert [c.wall for c in stub_checker.calls] == [False, True]
    assert quiet.diagnostics == []
    assert [d.code for d in loud.diagnostics] == ["unused_variable"]
    assert loud.ok and loud.object is not None


def test_options_are_copied_into_the_object(stub_checker: StubChecker) -> None:
    options = CompileOptions(wall=True)
    result = compile_text("int main(void) { return 0; }\n", "main.c", make_toolchain(), options)
    assert result.object.options == options and result.object.options is not options


def test_werror_turns_warnings_into_errors(stub_checker: StubChecker) -> None:
    source = "int main(void) {\n    int unused;\n    return 0;\n}\n"
    result = compile_text(source, "main.c", make_toolchain(),
                          CompileOptions(wall=True, werror=True))
    assert not result.ok and result.object is None
    diag = result.diagnostics[0]
    assert diag.severity is Severity.ERROR
    assert diag.message == "unused variable 'unused'" + WERROR_SUFFIX


def test_werror_also_covers_preprocessor_warnings(stub_checker: StubChecker) -> None:
    source = "#warning check me\nint main(void) { return 0; }\n"
    assert compile_text(source, "main.c", make_toolchain()).ok
    strict = compile_text(source, "main.c", make_toolchain(), CompileOptions(werror=True))
    assert not strict.ok and strict.diagnostics[0].message.endswith(WERROR_SUFFIX)


def test_werror_never_doubles_the_suffix(stub_checker: StubChecker) -> None:
    source = "#warning once\nint main(void) { return 0; }\n"
    result = compile_text(source, "main.c", make_toolchain(), CompileOptions(werror=True))
    assert result.diagnostics[0].message.count(WERROR_SUFFIX) == 1


def test_format_cc_command() -> None:
    options = CompileOptions(["include"], {"DEBUG": "", "LEVEL": "3"}, wall=True, wextra=True)
    assert format_cc_command(["main.c", "drone.c"], "farm", options) == \
        "cc -Wall -Wextra -Iinclude -DDEBUG -DLEVEL=3 main.c drone.c -o farm"
    assert format_cc_command(["my file.c"], "", CompileOptions()) == "cc 'my file.c'"


# =========================================================================== projects

def project() -> DictSourceFiles:
    return project_files(**{"main.c": MAIN_C, "drone.h": DRONE_H, "drone.c": DRONE_C})


def test_build_project_compiles_and_links(stub_checker: StubChecker) -> None:
    build = build_project(project(), ["main.c", "drone.c"], "farm", make_toolchain(),
                          CompileOptions(wall=True), cwd=PROJECT)
    assert build.ok, [d.format_gcc() for d in build.diagnostics]
    assert build.commands == ["cc -Wall main.c drone.c -o farm"]
    assert build.cwd == PROJECT and build.output_path == "farm"
    assert [o.unit_name for o in build.objects] == ["main.c", "drone.c"]
    assert build.executable is not None and build.link_result.ok


def test_shared_header_is_preprocessed_into_both_units(stub_checker: StubChecker) -> None:
    build = build_project(project(), ["main.c", "drone.c"], "farm", make_toolchain(), cwd=PROJECT)
    for obj in build.objects:
        assert "drone.h" in {line.file for line in obj.pp_lines}


def test_build_project_skips_linking_after_a_compile_error(stub_checker: StubChecker) -> None:
    files = project()
    files.files[f"{PROJECT}/drone.c"] = "void irrigate(int rows) { rows++ }\n"
    build = build_project(files, ["main.c", "drone.c"], "farm", make_toolchain(), cwd=PROJECT)
    assert not build.ok and build.executable is None and build.link_result is None
    assert [d.code for d in build.diagnostics] == ["missing_semicolon"]
    assert [o.unit_name for o in build.objects] == ["main.c"]


def test_build_project_without_sources(stub_checker: StubChecker) -> None:
    build = build_project(project(), [], "farm", make_toolchain(), cwd=PROJECT)
    assert not build.ok
    assert build.diagnostics[0].code == "cc_no_input"
    assert build.diagnostics[0].message == "cc: fatal error: no input files"


def test_built_executable_round_trips(stub_checker: StubChecker) -> None:
    build = build_project(project(), ["main.c", "drone.c"], "farm", make_toolchain(), cwd=PROJECT)
    loaded = load_executable(build.executable.to_bytes())
    result = link_executable(loaded, make_registry())
    assert result.ok and set(result.program.functions) >= {"main", "irrigate", "drone.c::helper"}


def test_load_executable_rejects_other_files() -> None:
    with pytest.raises(BadExecutable):
        load_executable(b"\x7fELF...")


def test_link_executable_reports_damaged_objects(stub_checker: StubChecker) -> None:
    lines = [PPLine("int main( {", "main.c", 1)]   # no longer compiles
    broken = ObjectFile("main.c", f"{PROJECT}/main.c", None, None, lines, CompileOptions())
    result = link_executable(Executable([broken]), make_registry(), program_name="./farm")
    assert not result.ok and result.diagnostics[0].code == "bad_executable"
    assert result.diagnostics[0].message.startswith("./farm: ")


def test_compile_text_compiles_in_the_root_folder(stub_checker: StubChecker) -> None:
    result = compile_text("int main(void) { return 0; }\n", "prog.c", make_toolchain())
    assert result.ok and result.object.source_path == "/prog.c" and result.cwd == "/"


# =========================================================================== parse_prototype

def test_parse_prototype_simple() -> None:
    ftype = parse_prototype("int move(int direction);")
    assert ftype == FunctionType(INT, (INT,))


def test_parse_prototype_void_and_without_semicolon() -> None:
    assert parse_prototype("void wait_tick(void)") == FunctionType(VOID, ())


def test_parse_prototype_variadic() -> None:
    ftype = parse_prototype("int printf(const char *format, ...);")
    assert ftype.variadic and ftype.params[0].is_pointer


def test_parse_prototype_library_typedefs() -> None:
    assert parse_prototype("size_t strlen(const char *s);").ret == ULONG
    fopen = parse_prototype("FILE *fopen(const char *path, const char *mode);")
    assert isinstance(fopen.ret, PointerType) and isinstance(fopen.ret.target, StructType)


def test_parse_prototype_extra_typedefs() -> None:
    assert parse_prototype("tick_t now(void);", {"tick_t": LONG}).ret == LONG


def test_parse_prototype_rejects_non_prototypes() -> None:
    for text in ("int x;", "int f(int a) { return a; }", "int (;", "int f(void); int g(void);"):
        with pytest.raises(ValueError):
            parse_prototype(text)


def test_registry_uses_parse_prototype() -> None:
    assert make_registry().function_type("move") == FunctionType(INT, (INT,))


# =========================================================================== with the real checker

def real_checker_available() -> bool:
    try:
        from pycc.interp.checker import check  # noqa: F401
    except ImportError:
        return False
    return True


needs_checker = pytest.mark.skipif(not real_checker_available(),
                                   reason="interp.checker is not available yet")


@needs_checker
def test_real_checker_two_file_project() -> None:
    build = build_project(project(), ["main.c", "drone.c"], "farm", make_toolchain(),
                          CompileOptions(wall=True), cwd=PROJECT)
    assert build.ok, [d.format_gcc() for d in build.diagnostics]
    restored = [ObjectFile.from_bytes(o.to_bytes(), make_toolchain()) for o in build.objects]
    for obj in restored:
        names = [n for n in ast.walk(obj.ast) if isinstance(n, ast.Identifier)]
        assert names and all(getattr(n, "ref", None) is not None for n in names)


@needs_checker
def test_real_checker_wall_warning() -> None:
    source = "int main(void) {\n    int unused;\n    return 0;\n}\n"
    loud = compile_text(source, "main.c", make_toolchain(), CompileOptions(wall=True))
    quiet = compile_text(source, "main.c", make_toolchain())
    assert "unused_variable" in [d.code for d in loud.diagnostics]
    assert "unused_variable" not in [d.code for d in quiet.diagnostics]

