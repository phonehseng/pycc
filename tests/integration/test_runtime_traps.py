"""Scenario 4: programs that crash, and programs that only deserve a warning.

Through the terminal, as the user sees it: the trap is printed like a sanitizer
report (``file:line:col: error: ...`` and the friendly line), the shell adds the line
bash prints after a crash (``Segmentation fault (core dumped)``) and ``$?`` holds the
status of a killed process (139, 134, 136, 137). A leak or an uninitialised read is a
warning: the program still finishes with its own exit code.
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import Line, Terminal, assert_reported

OUT_OF_BOUNDS = """\
#include <stdio.h>
int main(void) {
    int field[8];
    int i;
    for (i = 0; i <= 8; i++) {
        field[i] = i;
    }
    printf("never printed\\n");
    return 0;
}
"""
NULL_DEREF = """\
#include <stdio.h>
int main(void) {
    int *basket = NULL;
    printf("about to look in the basket\\n");
    return *basket;
}
"""
ENDLESS = """\
int main(void) {
    int rounds = 0;
    while (1) {
        rounds++;
    }
    return 0;
}
"""
RECURSION = """\
int dig(int depth) {
    return dig(depth + 1) + 1;
}
int main(void) {
    return dig(0);
}
"""
DOUBLE_FREE = """\
#include <stdlib.h>
int main(void) {
    int *rows = malloc(4 * sizeof(int));
    free(rows);
    free(rows);
    return 0;
}
"""
DIVISION = """\
int main(void) {
    int plots = 0;
    return 12 / plots;
}
"""
LEAK = """\
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int *seeds = malloc(16);
    seeds[0] = 3;
    printf("planted %d\\n", seeds[0]);
    return 0;
}
"""
UNINITIALISED = """\
#include <stdio.h>
int main(void) {
    int total;
    int crops[2];
    crops[0] = 1;
    total = crops[1];
    printf("finished\\n");
    return 0;
}
"""


def crash(term: Terminal, source: str) -> Line:
    """Build ``prog.c`` and run it, then print the status the way a user checks it."""
    term.write({"prog.c": source})
    return term.run("cc prog.c -o prog && ./prog; echo \"status $?\"")


@pytest.mark.parametrize("source, status, banner", [
    (OUT_OF_BOUNDS, 139, "Segmentation fault (core dumped)"),
    (NULL_DEREF, 139, "Segmentation fault (core dumped)"),
    (RECURSION, 139, "Segmentation fault (core dumped)"),
    (DOUBLE_FREE, 134, "Aborted (core dumped)"),
    (DIVISION, 136, "Floating point exception (core dumped)"),
    (ENDLESS, 137, "Killed"),
], ids=["out_of_bounds", "null", "stack_overflow", "double_free", "division", "budget"])
def test_status_and_banner(term: Terminal, source: str, status: int, banner: str) -> None:
    line = crash(term, source)
    assert line.stdout.endswith(f"status {status}\n")
    assert line.stderr.splitlines()[-1] == banner
    assert line.exit_code == 0          # the echo after ';' ran and succeeded


def test_out_of_bounds_names_the_array_and_the_index(term: Terminal) -> None:
    line = crash(term, OUT_OF_BOUNDS)
    assert_reported(line, "prog.c:6:9: error: index 8 is past the end of 'field' (8 elements)",
                    "Check the loop condition: < length, not <= length.")
    assert "never printed" not in line.stdout
    [summary] = line.runs
    assert summary.trap.code == "out_of_bounds" and summary.exit_code == 139


def test_null_dereference_after_some_output(term: Terminal) -> None:
    line = crash(term, NULL_DEREF)
    assert line.stdout == "about to look in the basket\nstatus 139\n"
    assert_reported(line, "prog.c:5:5: error: segmentation fault: read from address 0x0 "
                          "(NULL pointer dereference)", "The program followed a NULL pointer.")


def test_endless_loop_hits_the_instruction_budget(term: Terminal) -> None:
    line = crash(term, ENDLESS)
    assert_reported(line, "prog.c:4:9: error: killed: the program ran 1000000 statements "
                          "without finishing (instruction budget exhausted)",
                    "A loop probably never reaches its exit condition")
    summary = line.runs[0]
    assert summary.budget_hit and summary.trap.code == "instruction_budget"


def test_stack_overflow_from_endless_recursion(term: Terminal) -> None:
    line = crash(term, RECURSION)
    assert_reported(line, "prog.c:2:5: error: stack overflow in 'dig': more than 512 "
                          "nested calls", "recursion without a base case")


def test_double_free(term: Terminal) -> None:
    line = crash(term, DOUBLE_FREE)
    assert line.runs[0].trap.code == "double_free"
    assert "prog.c:5:5: error: free(): double free detected" in line.stderr
    assert "Each malloc needs exactly one free" in line.stderr


def test_division_by_zero(term: Terminal) -> None:
    line = crash(term, DIVISION)
    assert_reported(line, "prog.c:3:5: error: integer division by zero (12 / 0)",
                    "Dividing a whole number by zero has no answer")


def test_the_status_survives_for_an_if(term: Terminal) -> None:
    term.write({"prog.c": DIVISION})
    term.ok("cc prog.c -o prog")
    line = term.run("if ./prog; then echo fine; else echo \"crashed with $?\"; fi")
    assert line.stdout == "crashed with 136\n"


def test_a_leak_is_a_warning_not_a_trap(term: Terminal) -> None:
    line = crash(term, LEAK)
    assert line.stdout == "planted 3\nstatus 0\n"
    assert_reported(line, "prog.c:4:5: warning: 16 bytes in 1 blocks are definitely lost: "
                          "malloc(16) was never freed", "Every malloc needs exactly one free")
    [summary] = line.runs
    assert summary.trap is None and summary.exit_code == 0 and summary.leaks == 1
    assert [w.code for w in summary.warnings] == ["memory_leak"]


def test_an_uninitialised_read_warns_and_the_program_finishes(term: Terminal) -> None:
    line = crash(term, UNINITIALISED)
    assert line.stdout == "finished\nstatus 0\n"
    [summary] = line.runs
    assert summary.trap is None and summary.exit_code == 0
    assert [w.code for w in summary.warnings] == ["uninitialized_read"]
    warning = summary.warnings[0]
    assert (warning.location.file, warning.location.line) == ("prog.c", 6)
    assert f"{warning.location}: warning: read of uninitialised value" in line.stderr


def test_a_crash_inside_a_pipeline_still_reports_on_the_terminal(term: Terminal) -> None:
    term.write({"prog.c": NULL_DEREF})
    term.ok("cc prog.c -o prog")
    line = term.run("./prog | wc -l")
    assert line.stdout.strip() == "1"
    assert "Segmentation fault (core dumped)" in line.stderr


def test_the_crash_banner_is_the_shells_not_the_programs(term: Terminal) -> None:
    # bash reports a crashed job on its own stderr, so "2> crash.txt" catches the
    # program's report while "Segmentation fault" still reaches the terminal.
    term.write({"prog.c": NULL_DEREF})
    term.ok("cc prog.c -o prog")
    line = term.run("./prog 2> crash.txt")
    assert line.stderr == "Segmentation fault (core dumped)\n"
    saved = term.ok("cat crash.txt").stdout
    assert "error: segmentation fault: read from address 0x0" in saved
    assert "Segmentation fault (core dumped)" not in saved


def test_a_script_with_redirected_stderr_keeps_the_banner_in_its_log(term: Terminal) -> None:
    term.write({"prog.c": NULL_DEREF, "run.sh": "cc prog.c -o prog\n./prog\n"})
    line = term.run("sh run.sh 2> log.txt")
    assert line.stderr == ""
    assert term.ok("cat log.txt").stdout.endswith("Segmentation fault (core dumped)\n")
