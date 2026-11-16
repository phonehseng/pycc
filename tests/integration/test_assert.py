"""``assert`` and ``__func__``: found missing by the scenarios, now wired end to end.

``<assert.h>`` promised that the preprocessor rewrites
``assert(e)`` into ``__cv_assert((e), "e", __FILE__, __LINE__, __func__)`` and that
the parser knows ``__func__``; neither was there, so every ``assert`` was an
"implicit declaration". These tests use them the way a user does.
"""
from __future__ import annotations

from tests.integration.conftest import Terminal

CHECKED_C = """\
#include <assert.h>
#include <stdio.h>

static int share(int crops, int baskets) {
    assert(baskets > 0 && "no baskets");
    return crops / baskets;
}

int main(void) {
    printf("%d each, from %s\\n", share(8, 2), __func__);
    share(8, 0);
    printf("never printed\\n");
    return 0;
}
"""


def build(term: Terminal, source: str) -> None:
    term.write({"prog.c": source})
    term.ok("cc -Wall prog.c -o prog")


def test_a_true_assert_lets_the_program_go_on(term: Terminal) -> None:
    build(term, CHECKED_C)
    line = term.run("./prog")
    assert line.stdout == "4 each, from main\n"      # printed before the failing call


def test_a_false_assert_aborts_with_the_glibc_message(term: Terminal) -> None:
    build(term, CHECKED_C)
    line = term.run("./prog; echo \"status $?\"")
    assert line.stdout.endswith("status 134\n") and "never printed" not in line.stdout
    lines = line.stderr.splitlines()
    assert lines[0] == "./prog: prog.c:5: share: Assertion `baskets > 0 && \"no baskets\"' failed."
    assert lines[-1] == "Aborted (core dumped)"
    assert line.runs[0].trap.code == "assert_failed"


def test_ndebug_turns_asserts_off(term: Terminal) -> None:
    build(term, "#define NDEBUG\n#include <assert.h>\nint main(void) {\n"
                "    assert(1 == 2);\n    return 0;\n}\n")
    assert term.run("./prog").exit_code == 0


def test_assert_takes_exactly_one_argument(term: Terminal) -> None:
    term.write({"prog.c": "#include <assert.h>\nint main(void) {\n    assert(1, 2);\n"
                          "    return 0;\n}\n"})
    line = term.run("cc prog.c")
    assert 'prog.c:3: error: macro "assert" passed 2 arguments, but takes just 1' in line.stderr
    assert "__cv_assert" not in line.stderr         # library insides are never suggested


def test_assert_without_the_header_is_an_implicit_declaration(term: Terminal) -> None:
    term.write({"prog.c": "int main(void) {\n    assert(1);\n    return 0;\n}\n"})
    line = term.run("cc prog.c")
    assert "prog.c:2:5: error: implicit declaration of function 'assert'" in line.stderr
    assert "Did you forget #include <assert.h>?" in line.stderr


def test_func_names_the_enclosing_function(term: Terminal) -> None:
    build(term, '#include <stdio.h>\nstatic void where(void) { printf("%s\\n", __func__); }\n'
                "int main(void) {\n    where();\n    printf(\"%s %zu\\n\", __func__, "
                "sizeof __func__);\n    return 0;\n}\n")
    assert term.ok("./prog").stdout == "where\nmain 5\n"


def test_a_variable_called_assert_is_left_alone(term: Terminal) -> None:
    build(term, "#include <assert.h>\nint main(void) {\n    int assert = 3;\n"
                "    return assert - 3;\n}\n")
    assert term.run("./prog").exit_code == 0
