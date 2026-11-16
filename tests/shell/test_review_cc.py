"""Review tests for ``cc``: argument edge cases, relative paths and the cwd."""
from __future__ import annotations

import pytest

from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain
from pycc.shell import Shell
from pycc.shell.cc import CcUsageError, parse_cc_args
from pycc.shell.headless import make_headless_shell
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import VirtualFS, standard_layout

HELLO_C = '#include <stdio.h>\nint main(void) {\n    printf("hi\\n");\n    return 0;\n}\n'
MACROS_C = ('#include <stdio.h>\n#include "config.h"\nint main(void) {\n'
            '#ifdef LOUD\n    printf("loud ");\n#endif\n'
            '    printf("%d\\n", SIZE);\n    return 0;\n}\n')
CONFIG_H = "#ifndef SIZE\n#define SIZE 3\n#endif\n"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers())


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    vfs.mkdir(f"{PROJECT}/src")
    vfs.mkdir(f"{PROJECT}/include")
    vfs.write_file(f"{PROJECT}/src/hello.c", HELLO_C)
    vfs.write_file(f"{PROJECT}/macros.c", MACROS_C)
    vfs.write_file(f"{PROJECT}/include/config.h", CONFIG_H)
    return vfs


@pytest.fixture
def term(fs, toolchain) -> Shell:
    return make_headless_shell(fs, toolchain)


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


# --- option parsing ---------------------------------------------------------------------

def test_output_before_the_inputs() -> None:
    request = parse_cc_args("cc", ["-o", "bot", "main.c", "drone.c"])
    assert request.output == "bot"
    assert request.inputs == ["main.c", "drone.c"]


def test_include_dir_attached_and_separate() -> None:
    request = parse_cc_args("cc", ["-Iinclude", "-I", "lib", "main.c"])
    assert request.options.include_dirs == ["include", "lib"]
    assert request.inputs == ["main.c"]


def test_bare_define_is_one_and_value_define_keeps_text() -> None:
    request = parse_cc_args("cc", ["-DDEBUG", "-D", "SIZE=10", "-DEMPTY=", "main.c"])
    assert request.options.defines == {"DEBUG": "1", "SIZE": "10", "EMPTY": ""}


def test_repeated_flags_are_harmless_and_last_output_wins() -> None:
    request = parse_cc_args("cc", ["-Wall", "-Wall", "-o", "a", "-o", "b", "main.c"])
    assert request.options.wall is True
    assert request.output == "b"


def test_undefine_applies_left_to_right() -> None:
    request = parse_cc_args("cc", ["-DX", "-UX", "-DY", "-U", "Z", "main.c"])
    assert request.options.defines == {"Y": "1"}


def test_define_needs_an_identifier() -> None:
    with pytest.raises(CcUsageError) as caught:
        parse_cc_args("cc", ["-D=3", "main.c"])
    assert caught.value.message == "<command-line>: error: macro names must be identifiers"
    assert caught.value.code == "cc_bad_option"


def test_bad_define_reports_like_gcc_through_the_shell(run) -> None:
    result = run("cc -D1X macros.c")
    assert result.exit_code == 1
    assert result.stderr.splitlines()[0] == ("<command-line>: error: macro names must be "
                                             "identifiers")


def test_missing_value_after_undefine() -> None:
    with pytest.raises(CcUsageError) as caught:
        parse_cc_args("cc", ["main.c", "-U"])
    assert "macro name missing after '-U'" in caught.value.message


# --- relative paths and the working directory --------------------------------------------

def test_subfolder_source_from_project_root_writes_object_here(run, fs) -> None:
    assert run("cc -c src/hello.c").exit_code == 0
    assert fs.is_file(f"{PROJECT}/hello.o")
    assert not fs.exists(f"{PROJECT}/src/hello.o")


def test_cd_into_subfolder_then_compile(run, fs) -> None:
    result = run("cd src && cc hello.c && ./a.out")
    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    assert fs.is_file(f"{PROJECT}/src/a.out")
    assert not fs.exists(f"{PROJECT}/a.out")


def test_object_only_link_from_another_folder(run, fs) -> None:
    assert run("cc -c src/hello.c -o src/hello.o").exit_code == 0
    result = run("cc src/hello.o -o greeter && ./greeter")
    assert result.exit_code == 0
    assert result.stdout == "hi\n"


def test_include_dir_is_relative_to_cwd_with_both_spellings(run) -> None:
    assert run("cc -Iinclude -DSIZE=7 macros.c -o m1 && ./m1").stdout == "7\n"
    assert run("cc -I include -D LOUD macros.c -o m2 && ./m2").stdout == "loud 3\n"


def test_undefine_on_the_command_line_reaches_the_preprocessor(run) -> None:
    assert run("cc -Iinclude -DLOUD -ULOUD macros.c -o m && ./m").stdout == "3\n"


def test_preprocess_missing_header_says_compilation_terminated(run, fs) -> None:
    fs.write_file(f"{PROJECT}/bad.c", "#include <nothere.h>\nint main(void) { return 0; }\n")
    result = run("cc -E bad.c")
    assert result.exit_code == 1
    lines = [line for line in result.stderr.splitlines() if not line.startswith("  ")]
    assert lines[-1] == "compilation terminated."
    assert "fatal error: nothere.h" in lines[0]
