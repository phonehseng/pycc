"""Tests for ``cc``/``gcc`` (shell/cc.py), their registration and the /usr/include copy.

Every test drives a real ``Shell`` with the real toolchain (standard library, compiler,
linker) and, where a program must run, the headless runner.
"""
from __future__ import annotations

import pytest

from pycc.core.diagnostics import Severity
from pycc.interp.compiler import BuildResult
from pycc.interp.linker import EXECUTABLE_MAGIC, OBJECT_MAGIC
from pycc.interp.source import PPLine
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import CompileOptions, SystemHeaders, Toolchain
from pycc.shell import Shell
from pycc.shell.ansi import DIM, RESET
from pycc.shell.cc import CcUsageError, indented_friendly, parse_cc_args, render_preprocessed
from pycc.shell.headless import make_headless_shell
from pycc.shell.sysheaders import install_system_headers
from pycc.shell.toolchain_commands import install
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import PermissionDenied, VirtualFS, standard_layout

DRONE_H = "#ifndef DRONE_H\n#define DRONE_H\nint lift(int x);\n#endif\n"
DRONE_C = '#include "drone.h"\nint lift(int x) { return x + 1; }\n'
MAIN_C = ('#include <stdio.h>\n#include "drone.h"\nint main(void) {\n'
          '    printf("%d\\n", lift(41));\n    return 0;\n}\n')
HELLO_C = '#include <stdio.h>\nint main(void) {\n    printf("hello\\n");\n    return 0;\n}\n'
BAD_C = "int main(void) {\n    int x = 1\n    return x;\n}\n"
UNUSED_C = "int main(void) {\n    int unused;\n    return 0;\n}\n"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers())


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    for name, text in {"drone.h": DRONE_H, "drone.c": DRONE_C, "main.c": MAIN_C,
                       "hello.c": HELLO_C, "bad.c": BAD_C, "unused.c": UNUSED_C}.items():
        vfs.write_file(f"{PROJECT}/{name}", text)
    return vfs


@pytest.fixture
def term(fs, toolchain) -> Shell:
    return make_headless_shell(fs, toolchain)


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


def stderr_lines(result) -> list[str]:
    return result.stderr.splitlines()


# --- building programs -------------------------------------------------------------------

def test_single_file_builds_an_executable(run, fs) -> None:
    result = run("cc hello.c -o hello")
    assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")
    assert fs.read_file(f"{PROJECT}/hello").startswith(EXECUTABLE_MAGIC)
    assert fs.stat(f"{PROJECT}/hello").executable


def test_default_output_is_a_out(run, fs) -> None:
    assert run("cc hello.c").exit_code == 0
    assert fs.stat(f"{PROJECT}/a.out").executable


def test_built_program_runs(run) -> None:
    assert run("cc hello.c -o hello && ./hello").stdout == "hello\n"


def test_gcc_is_the_same_compiler(run) -> None:
    assert run("gcc hello.c -o hello && ./hello").stdout == "hello\n"


def test_multi_file_project_links(run) -> None:
    assert run("cc main.c drone.c -o bot && ./bot").stdout == "42\n"


def test_dash_c_writes_object_file_in_cwd(run, fs) -> None:
    assert run("cc -c drone.c").exit_code == 0
    data = fs.read_file(f"{PROJECT}/drone.o")
    assert data.startswith(OBJECT_MAGIC)
    assert not fs.stat(f"{PROJECT}/drone.o").executable


def test_dash_c_of_a_subfolder_file_writes_into_cwd(run, fs) -> None:
    fs.mkdir(f"{PROJECT}/src")
    fs.write_file(f"{PROJECT}/src/util.c", "int util(void) { return 7; }\n")
    assert run("cc -c src/util.c").exit_code == 0
    assert fs.is_file(f"{PROJECT}/util.o") and not fs.exists(f"{PROJECT}/src/util.o")


def test_dash_c_with_dash_o_names_the_object(run, fs) -> None:
    assert run("cc -c drone.c -o parts.o").exit_code == 0
    assert fs.is_file(f"{PROJECT}/parts.o") and not fs.exists(f"{PROJECT}/drone.o")


def test_linking_object_files(run) -> None:
    assert run("cc -c drone.c && cc -c main.c").exit_code == 0
    assert run("cc main.o drone.o -o bot && ./bot").stdout == "42\n"


def test_mixing_c_and_o_inputs(run) -> None:
    run("cc -c drone.c")
    assert run("cc main.c drone.o -o bot && ./bot").stdout == "42\n"


def test_cwd_is_used_for_relative_paths(run, fs) -> None:
    fs.mkdir(f"{PROJECT}/sub")
    fs.write_file(f"{PROJECT}/sub/hello.c", HELLO_C)
    assert run("cd sub && cc hello.c -o hi && ./hi").stdout == "hello\n"
    assert fs.is_file(f"{PROJECT}/sub/hi")


def test_cc_with_dash_c_and_o_inputs_warns_that_they_are_unused(run) -> None:
    run("cc -c drone.c")
    result = run("cc -c main.c drone.o")
    assert result.exit_code == 0
    assert "cc: warning: drone.o: linker input file unused because linking not done" \
        in result.stderr


# --- diagnostics -------------------------------------------------------------------------

def test_syntax_error_prints_gcc_line_and_exits_1(run, fs, term) -> None:
    result = run("cc bad.c -o bad")
    assert result.exit_code == 1
    diag = result.finished.diagnostics[0]
    assert diag.location.file == "bad.c" and diag.location.line == 2
    assert stderr_lines(result)[0] == diag.format_gcc()
    assert stderr_lines(result)[0].startswith("bad.c:2:")
    assert not fs.exists(f"{PROJECT}/bad")


def test_friendly_line_is_dim_and_indented(term) -> None:
    result = term.run("cc bad.c")
    diag = result.finished.diagnostics[0]
    assert f"{diag.format_gcc()}\n{DIM}  {diag.friendly}{RESET}\n" in result.stderr


def test_indented_friendly_format() -> None:
    assert indented_friendly("Add a ';'.") == f"{DIM}  Add a ';'.{RESET}\n"


def test_redirected_errors_hold_exactly_what_gcc_prints(run, fs) -> None:
    run("cc bad.c 2> errors.txt")
    text = fs.read_text(f"{PROJECT}/errors.txt")
    assert text.startswith("bad.c:2:") and DIM not in text
    assert len(text.splitlines()) == 1


def test_diagnostics_are_recorded_for_the_line(run) -> None:
    finished = run("cc bad.c").finished
    assert finished.exit_code == 1
    assert finished.diagnostics and all(d.location.file == "bad.c"
                                        for d in finished.diagnostics)


def test_last_diagnostics_holds_the_compile_errors(term) -> None:
    term.run("cc bad.c")
    assert term.last_diagnostics and term.last_diagnostics[0].is_error


def test_warning_with_wall_still_builds(run, fs) -> None:
    result = run("cc -Wall unused.c -o u")
    assert result.exit_code == 0
    assert "warning: unused variable 'unused'" in result.stderr
    assert fs.is_file(f"{PROJECT}/u")


def test_no_warning_without_wall(run) -> None:
    assert run("cc unused.c -o u").stderr == ""


def test_werror_turns_warnings_into_errors(run, fs) -> None:
    result = run("cc -Wall -Werror unused.c -o u")
    assert result.exit_code == 1
    assert "[-Werror]" in result.stderr
    assert stderr_lines(result)[-1] == "cc1: all warnings being treated as errors"
    assert not fs.exists(f"{PROJECT}/u")


def test_missing_header_is_fatal_with_trailer(run, fs) -> None:
    fs.write_file(f"{PROJECT}/inc.c", '#include "nope.h"\nint main(void) { return 0; }\n')
    result = run("cc inc.c")
    lines = stderr_lines(result)
    assert result.exit_code == 1
    assert lines[0].startswith("inc.c:1") and "fatal error: nope.h" in lines[0]
    assert lines[-1] == "compilation terminated."


def test_undefined_reference_reports_ld_and_collect2(run, fs) -> None:
    result = run("cc main.c -o bot")
    lines = stderr_lines(result)
    assert result.exit_code == 1
    assert lines[0] == "ld: main.o: in function 'main': undefined reference to 'lift'"
    assert lines[-1] == "collect2: error: ld returned 1 exit status"
    assert [d.code for d in result.finished.diagnostics] == ["undefined_reference"]
    assert "drone.c" in result.finished.diagnostics[0].friendly
    assert not fs.exists(f"{PROJECT}/bot")


def test_no_input_files(run) -> None:
    result = run("cc")
    assert result.exit_code == 1
    assert stderr_lines(result)[0] == "cc: fatal error: no input files"
    assert stderr_lines(result)[-1] == "compilation terminated."
    assert result.finished.diagnostics[0].code == "cc_no_input"


def test_gcc_names_itself_in_messages(run) -> None:
    assert stderr_lines(run("gcc"))[0] == "gcc: fatal error: no input files"


def test_missing_input_file(run) -> None:
    result = run("cc nope.c")
    lines = stderr_lines(result)
    assert result.exit_code == 1
    assert lines[0] == "cc: error: nope.c: No such file or directory"
    assert "cc: fatal error: no input files" in lines


def test_missing_input_stops_linking_but_compiles_the_rest(run, fs) -> None:
    result = run("cc hello.c nope.c -o hello")
    assert result.exit_code == 1
    assert "cc: error: nope.c: No such file or directory" in result.stderr
    assert "no input files" not in result.stderr
    assert not fs.exists(f"{PROJECT}/hello")


def test_same_input_output(run, fs) -> None:
    result = run("cc hello.c -o hello.c")
    assert result.exit_code == 1
    assert stderr_lines(result)[0] == ("cc: fatal error: input file 'hello.c' is the same "
                                       "as output file")
    assert result.finished.diagnostics[0].code == "same_input_output"
    assert fs.read_text(f"{PROJECT}/hello.c") == HELLO_C


def test_same_input_output_through_another_path(run) -> None:
    result = run("cc hello.c -o ./hello.c")
    assert result.finished.diagnostics[0].code == "same_input_output"


def test_unknown_file_type(run, fs) -> None:
    fs.write_file(f"{PROJECT}/notes.txt", "hello\n")
    result = run("cc notes.txt")
    assert result.exit_code == 1
    assert result.finished.diagnostics[0].code == "cc_unknown_file_type"
    assert "notes.txt: file not recognized" in result.stderr


def test_header_on_the_command_line_is_explained(run) -> None:
    result = run("cc main.c drone.h")
    diag = result.finished.diagnostics[0]
    assert diag.code == "cc_unknown_file_type" and "header" in diag.friendly


def test_bad_object_file(run, fs) -> None:
    fs.write_file(f"{PROJECT}/fake.o", "not an object\n")
    result = run("cc main.c fake.o -o bot")
    lines = stderr_lines(result)
    assert result.exit_code == 1
    assert lines[0] == "ld: fake.o: file not recognized: file format not recognized"
    assert lines[-1] == "collect2: error: ld returned 1 exit status"
    assert result.finished.diagnostics[0].code == "bad_object_file"


def test_unknown_option(run) -> None:
    result = run("cc -x hello.c")
    assert result.exit_code == 1
    assert stderr_lines(result)[0] == "cc: error: unrecognized command-line option '-x'"
    assert result.finished.diagnostics[0].code == "cc_bad_option"


def test_option_missing_its_value(run) -> None:
    result = run("cc hello.c -o")
    assert stderr_lines(result)[0] == "cc: error: missing filename after '-o'"


@pytest.mark.parametrize("flags", ["-g", "-O2", "-O0", "-std=c99", "-lm", "-pedantic",
                                   "-Wno-unused", "-fno-common"])
def test_harmless_flags_are_accepted(run, flags) -> None:
    assert run(f"cc {flags} hello.c -o hello").exit_code == 0


def test_dash_c_dash_o_with_several_files_is_refused(run) -> None:
    result = run("cc -c main.c drone.c -o both.o")
    assert result.exit_code == 1
    assert "cannot specify '-o' with '-c'" in result.stderr


# --- -I, -D and -E -----------------------------------------------------------------------

def test_include_dir_attached_and_separate(run, fs) -> None:
    fs.mkdir(f"{PROJECT}/include")
    fs.move(f"{PROJECT}/drone.h", f"{PROJECT}/include/drone.h")
    assert run("cc main.c drone.c -o bot").exit_code == 1
    assert run("cc -Iinclude main.c drone.c -o bot && ./bot").stdout == "42\n"
    assert run("cc -I include main.c drone.c -o bot2 && ./bot2").stdout == "42\n"


def test_define_flag(run, fs) -> None:
    fs.write_file(f"{PROJECT}/dbg.c", '#include <stdio.h>\nint main(void) {\n'
                  '#ifdef DEBUG\n    printf("debug %d\\n", LEVEL);\n#endif\n'
                  '    return 0;\n}\n')
    assert run("cc -DDEBUG -D LEVEL=3 dbg.c -o dbg && ./dbg").stdout == "debug 3\n"
    assert run("cc -DLEVEL=1 dbg.c -o dbg && ./dbg").stdout == ""


def test_dash_e_prints_preprocessed_text(run, fs) -> None:
    fs.write_file(f"{PROJECT}/m.c", "#define TWICE(x) ((x) * 2)\nint y = TWICE(4);\n")
    result = run("cc -E m.c")
    assert result.exit_code == 0
    assert "int y = ((4) * 2);" in result.stdout
    assert '"m.c"' in result.stdout


def test_dash_e_with_output_file(run, fs) -> None:
    fs.write_file(f"{PROJECT}/m.c", "#define N 5\nint n = N;\n")
    assert run("cc -E m.c -o m.i").stdout == ""
    assert "int n = 5;" in fs.read_text(f"{PROJECT}/m.i")


def test_render_preprocessed_marks_file_changes() -> None:
    lines = [PPLine("int a;", "a.h", 1), PPLine("int b;", "m.c", 2), PPLine("int c;", "m.c", 3)]
    assert render_preprocessed(lines) == '# 1 "a.h"\nint a;\n# 2 "m.c"\nint b;\nint c;\n'


# --- argument parsing --------------------------------------------------------------------

def test_parse_collects_inputs_in_order() -> None:
    request = parse_cc_args("cc", ["-Wall", "b.c", "a.o", "-o", "prog", "-c"])
    assert request.inputs == ["b.c", "a.o"] and request.sources == ["b.c"]
    assert request.output == "prog" and request.compile_only and request.options.wall


def test_parse_define_without_value_is_one() -> None:
    request = parse_cc_args("cc", ["-DDEBUG", "-D", "N=4", "-Iinc", "-I", "lib", "x.c"])
    assert request.options.defines == {"DEBUG": "1", "N": "4"}
    assert request.options.include_dirs == ["inc", "lib"]


def test_parse_starts_from_toolchain_defaults_without_changing_them() -> None:
    defaults = CompileOptions(include_dirs=["base"], wall=True)
    request = parse_cc_args("cc", ["-Iextra", "x.c"], defaults)
    assert request.options.include_dirs == ["base", "extra"] and request.options.wall
    assert defaults.include_dirs == ["base"]


def test_parse_rejects_unknown_option() -> None:
    with pytest.raises(CcUsageError) as caught:
        parse_cc_args("gcc", ["--frobnicate"])
    assert caught.value.code == "cc_bad_option"
    assert caught.value.message == "gcc: error: unrecognized command-line option '--frobnicate'"


# --- records -----------------------------------------------------------------------------

def test_last_build_is_a_build_result(term) -> None:
    finished = term.run("cc main.c drone.c -o bot").finished
    build = term.last_build
    assert isinstance(build, BuildResult) and build.ok
    assert build.output_path == "bot" and build.cwd == PROJECT
    assert build.executable is not None and len(build.objects) == 2
    assert build.commands == ["cc main.c drone.c -o bot"]
    assert finished.build is build


def test_failed_build_is_recorded_too(term) -> None:
    term.run("cc bad.c -o bad")
    assert isinstance(term.last_build, BuildResult) and not term.last_build.ok
    assert term.last_build.diagnostics[0].severity is Severity.ERROR


def test_constructs_count_compiles_and_links(run) -> None:
    assert run("cc main.c drone.c -o bot").finished.constructs == {"cc_compile": 2, "link": 1}
    assert run("cc -c drone.c").finished.constructs == {"cc_compile": 1}
    assert run("cc bad.c").finished.constructs == {}


# --- registration and system headers -----------------------------------------------------

def test_install_stores_the_toolchain(fs, toolchain) -> None:
    shell = Shell(fs)
    install(shell, toolchain)
    assert shell.toolchain is toolchain
    assert Runner(shell)("cc -c drone.c").exit_code == 0


def test_shell_without_toolchain_keeps_the_placeholder_answer(fs, toolchain) -> None:
    install(Shell(fs), toolchain)          # the real command is registered...
    result = Runner(Shell(fs))("cc hello.c")   # ...but this shell has no toolchain
    assert result.exit_code == 127
    assert "not installed in this workshop yet" in result.stderr


def test_install_copies_headers_read_only(fs, toolchain) -> None:
    install(Shell(fs), toolchain)
    assert fs.read_text("/usr/include/stdio.h") == toolchain.headers.get("stdio.h")
    assert fs.stat("/usr/include/stdio.h").read_only


def test_install_system_headers_replaces_old_texts(toolchain) -> None:
    vfs = standard_layout("project", {"farm.h": "old\n"})
    headers = toolchain.headers.copy()
    headers.add("farm.h", "int water(void);\n")
    written = install_system_headers(vfs, headers)
    assert "/usr/include/farm.h" in written
    assert vfs.read_text("/usr/include/farm.h") == "int water(void);\n"
    with pytest.raises(PermissionDenied):
        vfs.write_file("/usr/include/farm.h", "mine")
    with pytest.raises(PermissionDenied):
        vfs.write_file("/usr/include/new.h", "mine")


def test_install_system_headers_makes_missing_folders() -> None:
    vfs = VirtualFS()
    install_system_headers(vfs, SystemHeaders({"sys/types.h": "typedef int pid_t;\n"}))
    assert vfs.read_text("/usr/include/sys/types.h") == "typedef int pid_t;\n"
    assert vfs.stat("/usr/include").read_only


def test_headers_can_be_read_with_cat(run, toolchain) -> None:
    assert run("cat /usr/include/stdlib.h").stdout == toolchain.headers.get("stdlib.h")


def test_folder_as_input_is_explained(run, fs) -> None:
    fs.mkdir(f"{PROJECT}/src")
    result = run("cc src hello.c")
    assert result.exit_code == 1
    assert "cc: error: src: Is a directory" in result.stderr
    assert result.finished.diagnostics[0].code == "is_a_directory"
