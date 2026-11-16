"""Scenario 2: a project of several files, built by hand and then with make.

``plan_types.h`` is shared by everything, ``worker.h`` declares the worker functions,
``worker.c`` defines them and ``main.c`` uses them. Every header has an include guard,
so ``main.c`` may include ``plan_types.h`` twice (directly and through ``worker.h``).
``route.h`` is used by ``worker.c`` only, which lets the make tests check that touching
a header rebuilds exactly the files that depend on it.
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import Terminal

FILES = {
    "plan_types.h": """\
#ifndef PLAN_TYPES_H
#define PLAN_TYPES_H

struct Plan {
    int steps;
    int value;
};

#endif
""",
    "route.h": """\
#ifndef ROUTE_H
#define ROUTE_H

#define STRIDE 1

#endif
""",
    "worker.h": """\
#ifndef WORKER_H
#define WORKER_H

#include "plan_types.h"

int run_plan(struct Plan plan);

#endif
""",
    "worker.c": """\
#include <stdio.h>
#include "worker.h"
#include "route.h"

int run_plan(struct Plan plan) {
    int collected = 0;
    int i;
    for (i = 0; i < plan.steps; i++) {
        if (i % STRIDE == 0) {
            collected += plan.value;
        }
    }
    return collected;
}
""",
    "main.c": """\
#include <stdio.h>
#include "plan_types.h"
#include "worker.h"

int main(void) {
    struct Plan plan = { 1, 1 };
    printf("collected %d\\n", run_plan(plan));
    return 0;
}
""",
}

MAKEFILE = """\
CC = gcc
CFLAGS = -Wall -std=c99
OBJS = main.o worker.o

bot: $(OBJS)
\t$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@

main.o: main.c worker.h plan_types.h
worker.o: worker.c worker.h plan_types.h route.h

.PHONY: clean
clean:
\trm -f $(OBJS) bot
"""

COMPILE_MAIN = "gcc -Wall -std=c99 -c main.c -o main.o"
COMPILE_WORKER = "gcc -Wall -std=c99 -c worker.c -o worker.o"
LINK = "gcc -Wall -std=c99 main.o worker.o -o bot"


@pytest.fixture
def project(term: Terminal) -> Terminal:
    term.write(FILES)
    return term


def made(term: Terminal) -> list[str]:
    """The commands one ``make`` echoed."""
    return term.ok("make").stdout.splitlines()


# --- by hand: separate compile steps and a link step ------------------------------------

def test_compile_steps_write_object_files(project: Terminal) -> None:
    project.ok("cc -Wall -c worker.c")
    project.ok("cc -Wall -c main.c")
    assert project.read("worker.o").startswith(b"CVOBJ1\n")
    assert project.read("main.o").startswith(b"CVOBJ1\n")


def test_link_step_builds_a_program_that_runs_both_files(project: Terminal) -> None:
    project.ok("cc -Wall -c worker.c")
    project.ok("cc -Wall -c main.c")
    project.ok("cc worker.o main.o -o bot")
    run = project.ok("./bot")
    assert run.stdout == "collected 1\n"


def test_the_link_step_counts_as_a_link(project: Terminal) -> None:
    project.ok("cc -c worker.c && cc -c main.c")
    line = project.ok("cc worker.o main.o -o bot")
    assert line.finished.constructs.get("link") == 1
    assert "cc_compile" not in line.finished.constructs


def test_one_cc_line_does_the_same_as_the_three_steps(project: Terminal) -> None:
    run = project.ok("cc -Wall main.c worker.c -o bot && ./bot")
    assert run.stdout == "collected 1\n" and run.stderr == ""


def test_include_guards_let_a_header_be_included_twice(project: Terminal) -> None:
    # main.c includes plan_types.h directly and again through worker.h.
    line = project.run("cc -Wall -c main.c")
    assert line.exit_code == 0 and line.stderr == ""


# --- with a Makefile --------------------------------------------------------------------

@pytest.fixture
def make_project(project: Terminal) -> Terminal:
    project.write({"Makefile": MAKEFILE})
    return project


def test_make_builds_everything_in_dependency_order(make_project: Terminal) -> None:
    assert made(make_project) == [COMPILE_MAIN, COMPILE_WORKER, LINK]
    assert make_project.ok("./bot").stdout == "collected 1\n"


def test_make_again_has_nothing_to_do(make_project: Terminal) -> None:
    made(make_project)
    assert made(make_project) == ["make: 'bot' is up to date."]


def test_touching_a_shared_header_rebuilds_both_objects(make_project: Terminal) -> None:
    made(make_project)
    make_project.ok("touch plan_types.h")
    assert made(make_project) == [COMPILE_MAIN, COMPILE_WORKER, LINK]


def test_touching_a_header_rebuilds_exactly_its_dependents(make_project: Terminal) -> None:
    made(make_project)
    make_project.ok("touch route.h")
    assert made(make_project) == [COMPILE_WORKER, LINK]


def test_editing_main_rebuilds_main_only(make_project: Terminal) -> None:
    made(make_project)
    make_project.ok("touch main.c")
    assert made(make_project) == [COMPILE_MAIN, LINK]


def test_make_counts_rebuilds_and_compiles(make_project: Terminal) -> None:
    line = make_project.ok("make")
    constructs = line.finished.constructs
    assert constructs.get("cc_compile") == 2 and constructs.get("link") == 1
    assert constructs.get("make_rebuild", 0) >= 1


def test_make_dry_run_prints_without_building(make_project: Terminal) -> None:
    line = make_project.ok("make -n")
    assert line.stdout.splitlines() == [COMPILE_MAIN, COMPILE_WORKER, LINK]
    assert not make_project.vfs.exists("/home/user/project/bot")


def test_make_clean_then_make_rebuilds_all(make_project: Terminal) -> None:
    made(make_project)
    assert make_project.ok("make clean").stdout == "rm -f main.o worker.o bot\n"
    assert not make_project.vfs.exists("/home/user/project/main.o")
    assert made(make_project) == [COMPILE_MAIN, COMPILE_WORKER, LINK]


def test_a_failed_compile_stops_make_with_the_gcc_error(make_project: Terminal) -> None:
    make_project.write({"worker.c": FILES["worker.c"].replace("collected = 0;", "collected = 0")})
    line = make_project.run("make")
    assert line.exit_code == 2
    assert "worker.c:6:22: error: expected ',' or ';' before 'int'" in line.stderr
    assert "make: *** [Makefile:9: worker.o] Error 1" in line.stderr
    assert not make_project.vfs.exists("/home/user/project/bot")


def test_make_and_run_like_the_run_button(make_project: Terminal) -> None:
    run = make_project.ok("make && ./bot")
    assert run.stdout.endswith("collected 1\n")
    assert len(run.runs) == 1 and run.runs[0].exit_code == 0
