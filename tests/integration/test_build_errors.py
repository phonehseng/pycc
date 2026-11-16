"""Scenario 3: every compile and link error a beginner meets, as the terminal shows it.

Each test types the command a user would type and checks the exact gcc / ld / make
text, that the friendly line follows it (indented and dimmed), the diagnostic code that
is recorded, and that no output file was written.
"""
from __future__ import annotations

import pytest

from pycc.interp.toolchain import Toolchain
from tests.integration.conftest import PROJECT, Terminal, assert_reported
from pycc.vfs import standard_layout

def no_file(term: Terminal, name: str) -> bool:
    return not term.vfs.exists(f"{PROJECT}/{name}")


# --- the compiler ------------------------------------------------------------------------

def test_missing_semicolon(term: Terminal) -> None:
    term.write({"main.c": "#include <stdio.h>\nint main(void) {\n    int x = 1\n"
                          "    printf(\"%d\\n\", x);\n    return 0;\n}\n"})
    line = term.run("cc -Wall main.c -o bot")
    assert line.exit_code == 1 and line.codes == ["missing_semicolon"]
    assert_reported(line, "main.c:3:14: error: expected ',' or ';' before 'printf'",
                    "Every statement and declaration in C ends with ';'.")
    assert no_file(term, "bot")


def test_undeclared_identifier_with_a_suggestion(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    int count = 0;\n    return cuont;\n}\n"})
    line = term.run("cc main.c -o bot")
    assert line.exit_code == 1 and "undeclared_identifier" in line.codes
    assert_reported(line, "main.c:3:12: error: 'cuont' undeclared (first use in this "
                          "function); did you mean 'count'?", "Did you mean 'count'?")
    assert ("main.c:3:12: note: each undeclared identifier is reported only once for each "
            "function it appears in") in line.stderr

def test_redefinition_from_a_header_without_include_guard(term: Terminal) -> None:
    term.write({"plan.h": "struct Plan { int steps; };\n",
                "main.c": '#include "plan.h"\n#include "plan.h"\nint main(void) { return 0; }\n'})
    line = term.run("cc -Wall main.c -o bot")
    assert line.exit_code == 1 and line.codes == ["redefinition"]
    assert_reported(line, "plan.h:1:8: error: redefinition of 'struct Plan'",
                    "plan.h was included more than once and has no include guard")
    assert "plan.h:1:8: note: previous definition of 'struct Plan' was here" in line.stderr


def test_a_warning_alone_still_builds(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    int unused;\n    return 0;\n}\n"})
    line = term.run("cc -Wall main.c -o bot")
    assert line.exit_code == 0
    assert_reported(line, "main.c:2:9: warning: unused variable 'unused'",
                    "'unused' is declared but never used.")
    assert term.read("bot").startswith(b"CVEXE1\n")


def test_werror_turns_the_warning_into_an_error(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    int unused;\n    return 0;\n}\n"})
    line = term.run("cc -Wall -Werror main.c -o bot")
    assert line.exit_code == 1
    assert "main.c:2:9: error: unused variable 'unused' [-Werror]" in line.stderr
    assert no_file(term, "bot")


# --- the linker --------------------------------------------------------------------------

HELPER_USER = "int helper(void);\nint main(void) {\n    return helper();\n}\n"


def test_undefined_reference_when_an_object_was_not_passed(term: Terminal) -> None:
    term.write({"helper.h": "int helper(void);\n",
                "helper.c": '#include "helper.h"\nint helper(void) { return 4; }\n',
                "main.c": '#include "helper.h"\nint main(void) {\n    return helper();\n}\n'})
    term.ok("cc -c helper.c && cc -c main.c")
    line = term.run("cc main.o -o bot")
    assert line.exit_code == 1 and line.codes == ["undefined_reference"]
    assert_reported(line, "ld: main.o: in function 'main': undefined reference to 'helper'",
                    "Did you pass helper.o to the linker?")


def test_undefined_reference_when_the_function_is_written_nowhere(term: Terminal) -> None:
    term.write({"main.c": HELPER_USER})
    line = term.run("cc main.c -o bot")
    assert_reported(line, "ld: main.o: in function 'main': undefined reference to 'helper'",
                    "none of the files you linked contains its body")
    assert line.stderr.splitlines()[-1] == "collect2: error: ld returned 1 exit status"
    assert no_file(term, "bot")


def test_undefined_reference_when_the_function_is_static_elsewhere(term: Terminal) -> None:
    term.write({"helper.c": "static int helper(void) { return 4; }\n", "main.c": HELPER_USER})
    line = term.run("cc -Wall main.c helper.c -o bot")
    assert line.exit_code == 1 and line.codes == ["undefined_reference"]
    assert_reported(line, "ld: main.o: in function 'main': undefined reference to 'helper'",
                    "it is marked static there, so it is private to that file")


def test_multiple_definition_of_a_global_defined_in_a_header(term: Terminal) -> None:
    term.write({"totals.h": "#ifndef TOTALS_H\n#define TOTALS_H\nint harvest_total = 0;\n"
                            "#endif\n",
                "main.c": '#include "totals.h"\nint bump(void);\n'
                          "int main(void) { return bump() + harvest_total; }\n",
                "count.c": '#include "totals.h"\nint bump(void) { return ++harvest_total; }\n'})
    line = term.run("cc -Wall main.c count.c -o bot")
    assert line.exit_code == 1 and line.codes == ["multiple_definition"]
    assert_reported(line, "ld: count.o: multiple definition of 'harvest_total'; main.o: "
                          "first defined here",
                    "is defined in the header totals.h, and both main.c and count.c include it")
    assert "Include guards do not help here" in line.stderr

# --- make --------------------------------------------------------------------------------

def test_missing_separator_in_a_makefile(term: Terminal) -> None:
    term.write({"main.c": "int main(void) { return 0; }\n",
                "Makefile": "bot: main.c\n    cc main.c -o bot\n"})
    line = term.run("make")
    assert line.exit_code == 2 and line.codes == ["make_missing_separator"]
    assert_reported(line, "Makefile:2: *** missing separator.  Stop.",
                    "Recipe lines (the commands under a rule) must start with one Tab")
    assert no_file(term, "bot")


def test_no_rule_to_make_target(term: Terminal) -> None:
    term.write({"Makefile": "bot: main.c\n\tcc main.c -o bot\n"})
    line = term.run("make")
    assert line.exit_code == 2
    assert line.stderr.startswith(
        "make: *** No rule to make target 'main.c', needed by 'bot'.  Stop.\n")


@pytest.mark.parametrize("command, code", [
    ("cc", "cc_no_input"),
    ("cc missing.c", "no_such_file"),
])
def test_driver_errors(term: Terminal, command: str, code: str) -> None:
    line = term.run(command)
    assert line.exit_code == 1 and code in line.codes
