"""Review tests for ``make``: timestamps, order-only folders, -C, ordering, defaults."""
from __future__ import annotations

import pytest

from pycc.core.events import ShellOutput
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain
from pycc.shell import Shell
from pycc.shell.headless import make_headless_shell
from pycc.shell.make import Expander, parse_make_args, parse_makefile
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import VirtualFS, standard_layout

HELLO_C = '#include <stdio.h>\nint main(void) {\n    printf("hi\\n");\n    return 0;\n}\n'

OBJ_DIR_MAKEFILE = """\
prog: build/main.o
\t$(CC) $^ -o $@

build/main.o: main.c | build
\t$(CC) -c $< -o $@

build:
\tmkdir -p build
"""

PATTERN_OBJ_DIR_MAKEFILE = """\
prog: build/main.o
\t$(CC) $^ -o $@

build/%.o: %.c | build
\t$(CC) -c $< -o $@

build:
\tmkdir -p build
"""


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers())


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    vfs.write_file(f"{PROJECT}/main.c", HELLO_C)
    return vfs


@pytest.fixture
def term(fs, toolchain) -> Shell:
    return make_headless_shell(fs, toolchain)


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


def write(fs: VirtualFS, name: str, text: str) -> None:
    fs.mkdir(f"{PROJECT}/{name}".rsplit("/", 1)[0], parents=True)
    fs.write_file(f"{PROJECT}/{name}", text)


def commands(text: str) -> list[str]:
    """The tool lines of an output, without dimmed friendly explanations."""
    return [line for line in text.splitlines() if not line.startswith("  ")]


# --- order-only prerequisites --------------------------------------------------------------

def test_order_only_prerequisites_are_parsed_apart() -> None:
    makefile = parse_makefile("a.o: a.c b.h | build obj\n\tcc -c a.c\n")
    entry = makefile.targets["a.o"]
    assert entry.prerequisites == ["a.c", "b.h"]
    assert entry.order_only == ["build", "obj"]


def test_order_only_folder_is_made_first_but_not_passed_to_the_compiler(run, fs) -> None:
    write(fs, "Makefile", OBJ_DIR_MAKEFILE)
    result = run("make")
    assert result.exit_code == 0, result.stderr
    assert commands(result.stdout) == ["mkdir -p build", "cc -c main.c -o build/main.o",
                                       "cc build/main.o -o prog"]
    assert run("./prog").stdout == "hi\n"


def test_order_only_folder_never_makes_targets_out_of_date(run, fs) -> None:
    write(fs, "Makefile", OBJ_DIR_MAKEFILE)
    run("make")
    fs.touch(f"{PROJECT}/build")            # the folder gets newer than build/main.o
    assert commands(run("make").stdout) == ["make: 'prog' is up to date."]


def test_order_only_in_a_pattern_rule(run, fs) -> None:
    write(fs, "Makefile", PATTERN_OBJ_DIR_MAKEFILE)
    result = run("make")
    assert result.exit_code == 0, result.stderr
    assert commands(result.stdout)[:2] == ["mkdir -p build", "cc -c main.c -o build/main.o"]


def test_dollar_bar_lists_order_only_prerequisites(run, fs) -> None:
    write(fs, "Makefile", "all: main.c | out\n\t@echo [$^] [$|]\nout:\n\t@mkdir out\n")
    assert run("make").stdout == "[main.c] [out]\n"


# --- timestamps on the logical clock ---------------------------------------------------------

def test_touch_bumps_the_clock_so_only_the_dependent_rebuilds(run, fs) -> None:
    write(fs, "util.c", "int util(void) { return 1; }\n")
    write(fs, "Makefile", "all: main.o util.o\nmain.o: main.c\n\tcc -c main.c\n"
                          "util.o: util.c\n\tcc -c util.c\n")
    run("make")
    fs.touch(f"{PROJECT}/util.c")
    assert commands(run("make").stdout) == ["cc -c util.c"]


def test_directory_prerequisite_uses_the_folder_time(run, fs) -> None:
    write(fs, "assets/a.txt", "a\n")
    write(fs, "Makefile", "list.txt: assets\n\tls assets > list.txt\n")
    run("make")
    assert commands(run("make").stdout) == ["make: 'list.txt' is up to date."]
    fs.write_file(f"{PROJECT}/assets/b.txt", "b\n")    # a new file changes the folder
    assert commands(run("make").stdout) == ["ls assets > list.txt"]
    assert fs.read_text(f"{PROJECT}/list.txt") == "a.txt\nb.txt\n"


# --- defaults, echo and output ordering ------------------------------------------------------

def test_unset_cc_defaults_to_cc_in_the_builtin_rule(run, fs) -> None:
    result = run("make main")
    assert result.exit_code == 0
    assert commands(result.stdout) == ["cc  main.c -o main"]


def test_at_echo_prints_only_the_text(run, fs) -> None:
    write(fs, "Makefile", "all:\n\t@echo building now\n\techo loud\n")
    assert run("make").stdout == "building now\necho loud\nloud\n"


def test_echo_comes_before_the_recipe_output_and_errors_in_order(term, fs) -> None:
    write(fs, "Makefile", "all:\n\techo one\n\t@echo two >&2\n\t@echo three\n")
    events = [(e.stream, e.text) for e in term.run("make").events
              if isinstance(e, ShellOutput)]
    assert events == [("stdout", "echo one\none\n"), ("stderr", "two\n"),
                      ("stdout", "three\n")]


def test_program_output_follows_its_echo(run, fs) -> None:
    write(fs, "Makefile", "run: prog\n\t./prog\nprog: main.c\n\tcc main.c -o prog\n")
    assert run("make run").stdout == "cc main.c -o prog\n./prog\nhi\n"


def test_make_output_into_a_pipe_keeps_order(run, fs) -> None:
    write(fs, "Makefile", "all:\n\techo a\n\t@echo b\n")
    assert run("make | cat").stdout == "echo a\na\nb\n"


# --- subdirectories -------------------------------------------------------------------------

def test_make_inside_a_subdirectory_uses_that_folder(run, fs) -> None:
    write(fs, "lib/main.c", HELLO_C)
    write(fs, "lib/Makefile", "prog: main.c\n\tcc main.c -o prog\n")
    assert run("cd lib && make && ./prog").stdout == "cc main.c -o prog\nhi\n"
    assert fs.is_file(f"{PROJECT}/lib/prog")
    assert not fs.exists(f"{PROJECT}/prog")


def test_dash_c_runs_in_the_folder_and_returns(run, fs, term) -> None:
    write(fs, "lib/main.c", HELLO_C)
    write(fs, "lib/Makefile", "prog: main.c\n\tcc main.c -o prog\n")
    result = run("make -C lib")
    assert result.exit_code == 0, result.stderr
    assert result.stdout == (f"make: Entering directory '{PROJECT}/lib'\ncc main.c -o prog\n"
                             f"make: Leaving directory '{PROJECT}/lib'\n")
    assert fs.is_file(f"{PROJECT}/lib/prog")
    assert term.cwd == PROJECT


def test_dash_c_restores_the_folder_after_a_failure(run, fs, term) -> None:
    write(fs, "lib/Makefile", "all:\n\tfalse\n")
    assert run("make -C lib").exit_code == 2
    assert term.cwd == PROJECT


def test_dash_c_missing_folder(run) -> None:
    result = run("make -C nowhere")
    assert result.exit_code == 2
    assert commands(result.stderr) == ["make: *** nowhere: No such file or directory.  Stop."]


def test_recursive_make_through_dollar_make(run, fs) -> None:
    write(fs, "lib/main.c", HELLO_C)
    write(fs, "lib/Makefile", "prog: main.c\n\t@cc main.c -o prog\n")
    write(fs, "Makefile", ".PHONY: all\nall:\n\t@$(MAKE) -C lib\n")
    assert run("make").exit_code == 0
    assert fs.is_file(f"{PROJECT}/lib/prog")


def test_curdir_is_the_folder_make_runs_in(run, fs) -> None:
    write(fs, "Makefile", "all:\n\t@echo $(CURDIR)\n")
    assert run("make").stdout == f"{PROJECT}\n"


# --- command line and directives --------------------------------------------------------------

def test_colon_equals_override_names_the_variable() -> None:
    options = parse_make_args(["CC:=gcc", "CFLAGS=-Wall -g"])
    assert options.overrides == {"CC": "gcc", "CFLAGS": "-Wall -g"}


def test_dash_c_option_forms() -> None:
    assert parse_make_args(["-C", "lib"]).directory == "lib"
    assert parse_make_args(["-Clib", "all"]).directory == "lib"
    assert parse_make_args(["--directory=lib"]).directory == "lib"


def test_include_is_refused_with_a_clear_message(run, fs) -> None:
    write(fs, "Makefile", "include config.mk\nall:\n\techo hi\n")
    result = run("make")
    assert result.exit_code == 2
    assert commands(result.stderr) == [
        "Makefile:1: *** 'include' is not supported by this make.  Stop."]
    assert result.finished.diagnostics[-1].code == "unsupported_feature"


def test_export_without_value_is_accepted(run, fs) -> None:
    write(fs, "Makefile", "export CC\nall:\n\t@echo ok\n")
    assert run("make").stdout == "ok\n"


def test_more_text_functions() -> None:
    makefile = parse_makefile(
        "SRCS = src/a.c b.c c.h\nC = $(filter %.c,$(SRCS))\n"
        "O = $(addsuffix .o,$(basename $(C)))\nD = $(dir $(SRCS))\n"
        "S = $(strip   x    y  )\nall:\n")
    expander = Expander(makefile.variables)
    assert expander.expand("$(C)") == "src/a.c b.c"
    assert expander.expand("$(O)") == "src/a.o b.o"
    assert expander.expand("$(D)") == "src/ ./ ./"
    assert expander.expand("$(S)") == "x y"
