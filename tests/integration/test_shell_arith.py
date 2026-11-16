"""``$(( ))``, ``break`` and ``continue``: found missing when a script counted retries.

Before, ``i=$((i + 1))`` ran ``(i + 1)`` as a sub-shell command and printed "i+1:
command not found", so no ``while`` loop in a script could count, and ``break`` was
"command not found", so a retry loop could only stop at the 10 000-round guard.
"""
from __future__ import annotations

import pytest

from pycc.shell.arith import ArithError, evaluate
from tests.integration.conftest import Terminal

VARIABLES = {"i": "4", "n": "-3", "word": "wheat", "?": "1"}


@pytest.mark.parametrize("expression, value", [
    ("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("7 / 2", 3), ("-7 / 2", -3), ("-7 % 2", -1),
    ("i + 1", 5), ("$i * $i", 16), ("n - 1", -4), ("word + 1", 1), ("missing", 0),
    ("i > 3 && i < 5", 1), ("!(i == 4)", 0), ("$? + 1", 2), ("", 0), ("  - - 2", 2),
])
def test_evaluate(expression: str, value: int) -> None:
    assert evaluate(expression, VARIABLES.get) == value


@pytest.mark.parametrize("expression, message", [
    ("1 / 0", "1 / 0: division by 0 (error token is \"0\")"),
    ("i +", "i +: syntax error: operand expected (error token is \"\")"),
    ("i += 1", "i += 1: syntax error: invalid arithmetic operator (error token is \"= 1\")"),
    ("(1 + 2", "(1 + 2: syntax error: operand expected (error token is \"\")"),
])
def test_errors_use_bash_wording(expression: str, message: str) -> None:
    with pytest.raises(ArithError) as caught:
        evaluate(expression, VARIABLES.get)
    assert caught.value.message == message


def test_a_counting_while_loop(term: Terminal) -> None:
    line = term.ok("i=0; while [ $i -lt 3 ]; do echo \"try $i\"; i=$((i + 1)); done")
    assert line.stdout == "try 0\ntry 1\ntry 2\n"


def test_a_script_retries_a_failing_program(term: Terminal) -> None:
    term.write({"flaky.c": "int main(void) {\n    return 1;\n}\n",
                "retry.sh": "cc flaky.c -o flaky\ntries=0\n"
                            "until ./flaky; do\n    tries=$((tries + 1))\n"
                            "    if [ $tries -ge 3 ]; then\n        break\n    fi\ndone\n"
                            "echo \"gave up after $tries tries\"\n"})
    assert term.run("sh retry.sh").stdout.endswith("gave up after 3 tries\n")


def test_division_by_zero_is_a_shell_error(term: Terminal) -> None:
    line = term.run("echo $((5 / 0))")
    assert line.exit_code == 1
    assert "division by 0" in line.stderr


def test_a_subshell_inside_command_substitution_is_not_arithmetic(term: Terminal) -> None:
    assert term.ok("echo $( (echo inner) )").stdout == "inner\n"


def test_break_and_continue_in_loops(term: Terminal) -> None:
    line = term.ok("for i in 1 2 3 4 5; do if [ $i -eq 2 ]; then continue; fi; "
                   "if [ $i -eq 4 ]; then break; fi; echo $i; done")
    assert line.stdout == "1\n3\n"


def test_break_two_leaves_both_loops(term: Terminal) -> None:
    line = term.ok("for a in x y; do for b in 1 2; do echo $a$b; break 2; done; done; "
                   "echo after")
    assert line.stdout == "x1\nafter\n"


def test_break_outside_a_loop_only_warns(term: Terminal) -> None:
    line = term.run("break")
    assert line.exit_code == 0
    assert line.stderr == "bash: break: only meaningful in a `for', `while', or `until' loop\n"


def test_break_needs_a_positive_count(term: Terminal) -> None:
    line = term.run("for i in 1; do break 0; done")
    assert line.exit_code == 1 and "loop count out of range" in line.stderr
