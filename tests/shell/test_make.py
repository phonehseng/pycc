"""Tests for the simulated GNU make (shell/make.py), driven through a real Shell."""
from __future__ import annotations

import pytest

from pycc.core.events import RunProgram
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain
from pycc.shell import Shell
from pycc.shell.ansi import DIM
from pycc.shell.headless import make_headless_shell
from pycc.shell.make import (Expander, MakeError, PatternRule, Variable, logical_lines,
                        parse_make_args, parse_makefile)
from tests.shell.conftest import PROJECT, Runner
from pycc.vfs import VirtualFS, standard_layout

DRONE_H = "int lift(int x);\n"
DRONE_C = '#include "drone.h"\nint lift(int x) { return x + 1; }\n'
MAIN_C = ('#include <stdio.h>\n#include "drone.h"\nint main(void) {\n'
          '    printf("%d\\n", lift(41));\n    return 0;\n}\n')
MOTOR_C = "int motor(void) { return 2; }\n"

MAKEFILE = """\
# The drone project
CC = cc
CFLAGS = -Wall
OBJS = main.o drone.o motor.o

bot: $(OBJS)
\t$(CC) $(CFLAGS) $(OBJS) -o $@

main.o: main.c drone.h
\t$(CC) $(CFLAGS) -c main.c -o main.o
drone.o: drone.c drone.h
\t$(CC) $(CFLAGS) -c drone.c -o drone.o
motor.o: motor.c
\t$(CC) $(CFLAGS) -c motor.c -o motor.o

.PHONY: clean
clean:
\trm -f bot $(OBJS)
"""

BUILD_LINES = ["cc -Wall -c main.c -o main.o", "cc -Wall -c drone.c -o drone.o",
               "cc -Wall -c motor.c -o motor.o", "cc -Wall main.o drone.o motor.o -o bot"]


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain(make_registry(), system_headers())


@pytest.fixture
def fs() -> VirtualFS:
    vfs = standard_layout("project")
    for name, text in {"drone.h": DRONE_H, "drone.c": DRONE_C, "main.c": MAIN_C,
                       "motor.c": MOTOR_C, "Makefile": MAKEFILE}.items():
        vfs.write_file(f"{PROJECT}/{name}", text)
    return vfs


@pytest.fixture
def term(fs, toolchain) -> Shell:
    return make_headless_shell(fs, toolchain)


@pytest.fixture
def run(term) -> Runner:
    return Runner(term)


def messages(text: str) -> list[str]:
    """The tool's own lines, without the indented friendly explanations."""
    return [line for line in text.splitlines() if not line.startswith("  ")]


def write_makefile(fs: VirtualFS, text: str) -> None:
    fs.write_file(f"{PROJECT}/Makefile", text)


# --- building a project ------------------------------------------------------------------

def test_three_file_project_builds(run, fs) -> None:
    result = run("make")
    assert result.exit_code == 0
    assert result.stdout.splitlines() == BUILD_LINES
    assert run("./bot").stdout == "42\n"


def test_second_make_has_nothing_to_do(run) -> None:
    run("make")
    result = run("make")
    assert result.exit_code == 0
    assert result.stdout == "make: 'bot' is up to date.\n"


def test_nothing_to_be_done_for_a_target_without_recipe(run, fs) -> None:
    write_makefile(fs, "all: bot\n\n" + MAKEFILE)
    run("make")
    result = run("make")
    assert result.stdout.splitlines()[0] == "make: Nothing to be done for 'all'."
    assert result.finished.diagnostics[0].code == "make_nothing_to_do"


def test_nothing_to_be_done_has_a_friendly_line_on_the_terminal(term, fs) -> None:
    write_makefile(fs, "all: main.c\n")
    assert f"{DIM}  Everything is already built" in term.run("make").stdout


def test_touching_a_header_rebuilds_its_dependents(run) -> None:
    run("make")
    result = run("touch drone.h && make")
    assert result.stdout.splitlines() == [BUILD_LINES[0], BUILD_LINES[1], BUILD_LINES[3]]


def test_editing_one_source_rebuilds_only_it(run, fs) -> None:
    run("make")
    fs.write_file(f"{PROJECT}/motor.c", "int motor(void) { return 3; }\n")
    assert run("make").stdout.splitlines() == [BUILD_LINES[2], BUILD_LINES[3]]


def test_missing_target_file_is_rebuilt(run) -> None:
    run("make")
    assert run("rm bot && make").stdout.splitlines() == [BUILD_LINES[3]]


def test_make_named_target(run, fs) -> None:
    assert run("make drone.o").stdout == BUILD_LINES[1] + "\n"
    assert fs.is_file(f"{PROJECT}/drone.o") and not fs.exists(f"{PROJECT}/bot")


def test_make_several_targets(run) -> None:
    assert run("make motor.o drone.o").stdout.splitlines() == [BUILD_LINES[2], BUILD_LINES[1]]


def test_make_clean_removes_files(run, fs) -> None:
    run("make")
    result = run("make clean")
    assert result.stdout == "rm -f bot main.o drone.o motor.o\n"
    assert not any(fs.exists(f"{PROJECT}/{n}") for n in ("bot", "main.o", "drone.o", "motor.o"))


def test_phony_target_runs_even_if_a_file_has_its_name(run, fs) -> None:
    fs.write_file(f"{PROJECT}/clean", "")
    assert run("make clean").stdout.startswith("rm -f")


def test_dry_run_prints_without_running(run, fs) -> None:
    result = run("make -n")
    assert result.stdout.splitlines() == BUILD_LINES
    assert not fs.exists(f"{PROJECT}/main.o")
    assert result.finished.constructs == {}


def test_always_make_rebuilds_everything(run) -> None:
    run("make")
    assert run("make -B").stdout.splitlines() == BUILD_LINES


def test_make_rebuild_is_counted_per_target(run) -> None:
    constructs = run("make").finished.constructs
    assert constructs["make_rebuild"] == 4
    assert constructs["cc_compile"] == 3 and constructs["link"] == 1


def test_build_result_reaches_the_shell(run, term) -> None:
    finished = run("make").finished
    assert finished.build is term.last_build and term.last_build.ok


def test_dash_f_reads_another_file(run, fs) -> None:
    fs.write_file(f"{PROJECT}/build.mk", "hi:\n\techo hello from build.mk\n")
    assert run("make -f build.mk").stdout == "echo hello from build.mk\nhello from build.mk\n"


def test_dash_f_missing_file(run) -> None:
    result = run("make -f nope.mk")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[0] == "make: nope.mk: No such file or directory"


def test_lowercase_makefile_is_found(run, fs) -> None:
    fs.remove(f"{PROJECT}/Makefile")
    fs.write_file(f"{PROJECT}/makefile", "go:\n\techo lower\n")
    assert run("make").stdout == "echo lower\nlower\n"


# --- errors ------------------------------------------------------------------------------

def test_no_makefile(run, fs) -> None:
    fs.remove(f"{PROJECT}/Makefile")
    result = run("make")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[0] == ("make: *** No targets specified and no makefile "
                                             "found.  Stop.")
    assert result.finished.diagnostics[0].code == "make_no_makefile"


def test_missing_separator_reports_the_line(run, fs) -> None:
    write_makefile(fs, "CC = cc\n\nbot: main.c\n    $(CC) main.c -o bot\n")
    result = run("make")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[0] == "Makefile:4: *** missing separator.  Stop."
    diag = result.finished.diagnostics[0]
    assert diag.code == "make_missing_separator" and "Tab" in diag.friendly


def test_missing_separator_is_detected_before_anything_runs(run, fs) -> None:
    write_makefile(fs, "a:\n\techo first\nb\n")
    result = run("make")
    assert result.stdout == "" and "Makefile:3: *** missing separator" in result.stderr


def test_recipe_before_first_target(run, fs) -> None:
    write_makefile(fs, "\techo lost\n")
    assert run("make").stderr.startswith(
        "Makefile:1: *** recipe commences before first target.  Stop.")


def test_no_rule_for_goal(run) -> None:
    result = run("make nope")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[0] == "make: *** No rule to make target 'nope'.  Stop."
    assert result.finished.diagnostics[0].code == "make_no_rule"


def test_no_rule_for_prerequisite_says_needed_by(run, fs) -> None:
    write_makefile(fs, "bot: main.o wheels.o\n\tcc main.o wheels.o -o bot\n")
    result = run("make")
    assert ("make: *** No rule to make target 'wheels.c', needed by 'wheels.o'.  Stop."
            not in result.stderr)
    assert messages(result.stderr)[-1] == ("make: *** No rule to make target 'wheels.o', "
                                              "needed by 'bot'.  Stop.")


def test_circular_dependency_is_dropped(run, fs) -> None:
    write_makefile(fs, "a: b\n\techo a\nb: a\n\techo b\n")
    result = run("make")
    assert "make: Circular b <- a dependency dropped." in result.stderr
    assert result.stdout == "echo b\nb\necho a\na\n"
    assert any(d.code == "make_circular" for d in result.finished.diagnostics)


def test_self_dependency(run, fs) -> None:
    write_makefile(fs, "loop: loop\n\techo once\n")
    result = run("make")
    assert "make: Circular loop <- loop dependency dropped." in result.stderr
    assert result.exit_code == 0


def test_recipe_failure_stops_with_the_exact_message(run, fs) -> None:
    fs.write_file(f"{PROJECT}/drone.c", "int lift(int x) { return x + 1 }\n")
    result = run("make")
    assert result.exit_code == 2
    assert messages(result.stderr)[-1] == "make: *** [Makefile:12: drone.o] Error 1"
    assert result.stdout.splitlines() == BUILD_LINES[:2]       # motor.o is never tried
    codes = [d.code for d in result.finished.diagnostics]
    assert codes[-1] == "make_recipe_failed" and len(codes) >= 2


def test_ignored_failure_with_dash_prefix(run, fs) -> None:
    write_makefile(fs, "go:\n\t-false\n\techo after\n")
    result = run("make")
    assert result.exit_code == 0
    assert "make: [Makefile:2: go] Error 1 (ignored)" in result.stderr
    assert result.stdout == "false\necho after\nafter\n"


def test_unknown_option(run) -> None:
    result = run("make -q")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[0] == "make: invalid option -- 'q'"


def test_recursive_variable_referencing_itself(run, fs) -> None:
    write_makefile(fs, "CFLAGS = $(CFLAGS) -Wall\nall:\n\techo $(CFLAGS)\n")
    result = run("make")
    assert result.exit_code == 2
    assert "Recursive variable 'CFLAGS' references itself (eventually)" in result.stderr


# --- recipes run through the shell -------------------------------------------------------

def test_at_prefix_hides_the_echo(run, fs) -> None:
    write_makefile(fs, "hi:\n\t@echo quiet\n")
    assert run("make").stdout == "quiet\n"


def test_dry_run_shows_silent_lines_too(run, fs) -> None:
    write_makefile(fs, "hi:\n\t@echo quiet\n")
    assert run("make -n").stdout == "echo quiet\n"


def test_silent_flag(run, fs) -> None:
    write_makefile(fs, "hi:\n\techo loud\n")
    assert run("make -s").stdout == "loud\n"


def test_recipe_pipes_and_redirections(run, fs) -> None:
    write_makefile(fs, "list:\n\t@ls | grep '\\.c$$' > sources.txt\n")
    assert run("make").exit_code == 0
    assert fs.read_text(f"{PROJECT}/sources.txt") == "drone.c\nmain.c\nmotor.c\n"


def test_recipe_runs_programs(run, fs) -> None:
    write_makefile(fs, MAKEFILE + "run: bot\n\t./bot\n")
    result = run("make run")
    assert result.stdout.splitlines()[-2:] == ["./bot", "42"]


def test_run_program_events_reach_the_host(fs, toolchain) -> None:
    shell = make_headless_shell(fs, toolchain)
    write_makefile(fs, "run:\n\t@./bot\n")
    fs.write_file(f"{PROJECT}/bot", b"CVEXE1\nfake")
    fs.set_executable(f"{PROJECT}/bot")
    generator = shell.execute("make run")
    seen: list[RunProgram] = []
    reply = None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as stop:
            status = stop.value
            break
        reply = None
        if isinstance(event, RunProgram):
            seen.append(event)
            reply = 0
    assert status == 0 and [e.argv for e in seen] == [["./bot"]]


def test_failing_program_in_recipe_stops_make(run, fs) -> None:
    fs.write_file(f"{PROJECT}/fail.c", "int main(void) { return 3; }\n")
    write_makefile(fs, "check: fail\n\t./fail\n\techo never\n")
    result = run("make check")
    assert messages(result.stderr)[-1] == "make: *** [Makefile:2: check] Error 3"
    assert "never" not in result.stdout


def test_cd_in_a_recipe_does_not_move_the_shell(run, term, fs) -> None:
    fs.mkdir(f"{PROJECT}/sub")
    write_makefile(fs, "go:\n\tcd sub && pwd\n")
    assert run("make").stdout.splitlines()[-1] == f"{PROJECT}/sub"
    assert term.cwd == PROJECT


def test_make_output_can_be_redirected(run, fs) -> None:
    write_makefile(fs, "hi:\n\techo hello\n")
    run("make > log.txt")
    assert fs.read_text(f"{PROJECT}/log.txt") == "echo hello\nhello\n"


# --- variables, pattern rules and implicit rules -----------------------------------------

def test_pattern_rule_and_automatic_variables(run, fs) -> None:
    write_makefile(fs, "CC = gcc\nCFLAGS = -Wall\nOBJS = main.o drone.o\n\n"
                       "bot: $(OBJS)\n\t$(CC) $^ -o $@\n\n"
                       "%.o: %.c drone.h\n\t$(CC) $(CFLAGS) -c $< -o $@\n")
    result = run("make")
    assert result.stdout.splitlines() == ["gcc -Wall -c main.c -o main.o",
                                          "gcc -Wall -c drone.c -o drone.o",
                                          "gcc main.o drone.o -o bot"]


def test_newer_prerequisites_variable(run, fs) -> None:
    write_makefile(fs, "stamp: main.c drone.c\n\t@echo changed: $?\n\t@touch stamp\n")
    run("make")
    assert run("touch drone.c && make").stdout == "changed: drone.c\n"


def test_stem_variable(run, fs) -> None:
    write_makefile(fs, "%.txt: %.c\n\t@echo stem=$* first=$<\n")
    assert run("make drone.txt").stdout == "stem=drone first=drone.c\n"


def test_implicit_rule_for_objects(run, fs) -> None:
    write_makefile(fs, "CFLAGS = -Wall\nbot: main.o drone.o\n\tcc $^ -o $@\n")
    result = run("make")
    assert result.stdout.splitlines()[:2] == ["cc -Wall -c main.c -o main.o",
                                              "cc -Wall -c drone.c -o drone.o"]


def test_implicit_rule_failure_names_builtin(run, fs) -> None:
    fs.write_file(f"{PROJECT}/broken.c", "int x = ;\n")
    write_makefile(fs, "all: broken.o\n")
    assert messages(run("make").stderr)[-1] == "make: *** [<builtin>: broken.o] Error 1"


def test_make_without_makefile_uses_builtin_rules(run, fs) -> None:
    fs.remove(f"{PROJECT}/Makefile")
    fs.write_file(f"{PROJECT}/hello.c", '#include <stdio.h>\nint main(void) { '
                  'printf("hi\\n"); return 0; }\n')
    result = run("make hello && ./hello")
    assert result.stdout == "cc  hello.c -o hello\nhi\n"


def test_command_line_variable_overrides_the_file(run) -> None:
    result = run("make -n CFLAGS=-O2")
    assert result.stdout.splitlines()[0] == "cc -O2 -c main.c -o main.o"


def test_plus_equals_and_simple_assignment(run, fs) -> None:
    write_makefile(fs, "A = one\nA += two\nB := $(A)\nA = three\nshow:\n\t@echo $(A) / $(B)\n")
    assert run("make").stdout == "three / one two\n"


def test_conditional_assignment(run, fs) -> None:
    write_makefile(fs, "CC ?= tcc\nX ?= set\nshow:\n\t@echo $(CC) $(X)\n")
    assert run("make").stdout == "cc set\n"


def test_braces_comments_and_continuation(run, fs) -> None:
    write_makefile(fs, "# comment line\nNAMES = a \\\n        b   # trailing comment\n"
                       "show:\n\t@echo ${NAMES}\n")
    assert run("make").stdout == "a b\n"


def test_dollar_dollar_reaches_the_shell(run, fs) -> None:
    write_makefile(fs, "show:\n\t@echo $$HOME\n")
    assert run("make").stdout == "/home/user\n"


def test_substitution_reference_and_wildcard(run, fs) -> None:
    write_makefile(fs, "SRCS = $(wildcard *.c)\nOBJS = $(SRCS:.c=.o)\n"
                       "show:\n\t@echo $(OBJS)\n")
    assert run("make").stdout == "drone.o main.o motor.o\n"


def test_patsubst_and_friends(run, fs) -> None:
    write_makefile(fs, "S = src/a.c src/b.c\nshow:\n"
                       "\t@echo $(patsubst src/%.c,build/%.o,$(S)) $(notdir $(S))\n")
    assert run("make").stdout == "build/a.o build/b.o a.c b.c\n"


def test_later_recipe_overrides_with_a_warning(run, fs) -> None:
    write_makefile(fs, "go:\n\t@echo old\ngo:\n\t@echo new\n")
    result = run("make")
    assert result.stdout == "new\n"
    assert "Makefile:3: warning: overriding recipe for target 'go'" in result.stderr


def test_inline_recipe_after_semicolon(run, fs) -> None:
    write_makefile(fs, "go: ; @echo inline\n")
    assert run("make").stdout == "inline\n"


# --- unit tests of the parts -------------------------------------------------------------

def test_logical_lines_join_continuations() -> None:
    assert logical_lines("A = 1 \\\n    2\nB = 3\n") == [(1, "A = 1 2"), (3, "B = 3")]


def test_parse_makefile_collects_targets_and_default_goal() -> None:
    makefile = parse_makefile(".PHONY: all\nall: bot\nbot: a.o\n\tcc a.o -o bot\n")
    assert makefile.default_goal == "all" and makefile.phony == {"all"}
    assert makefile.targets["bot"].recipe_rule.recipe[0].line == 4


def test_parse_makefile_raises_missing_separator() -> None:
    with pytest.raises(MakeError) as caught:
        parse_makefile("all:\n  echo spaces\n", "Makefile")
    assert caught.value.text == "Makefile:2: *** missing separator.  Stop."


def test_expander_handles_nested_references() -> None:
    variables = {"WHICH": Variable("CC"), "CC": Variable("gcc")}
    assert Expander(variables).expand("$($(WHICH)) $$x $(MISSING)") == "gcc $x "


def test_pattern_rule_matching() -> None:
    rule = PatternRule("lib%.o", ["%.c"])
    assert rule.match("libfoo.o") == "foo" and rule.match("foo.o") is None


def test_parse_make_args() -> None:
    options = parse_make_args(["-nB", "-f", "x.mk", "CC=gcc", "all", "-j4"])
    assert options.dry_run and options.always and options.makefile == "x.mk"
    assert options.overrides == {"CC": "gcc"} and options.goals == ["all"]
