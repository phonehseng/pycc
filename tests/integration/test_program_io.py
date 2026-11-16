"""Scenario 6: a program's own input and output - arguments, files and errors.

``argc``/``argv`` parsed with ``atoi``, a CSV report written with ``fopen`` and
``fprintf`` that ``cat`` shows afterwards (files are written through to the virtual
filesystem), and the conventional failure path: ``fopen`` returns NULL, ``perror``
explains why and the program exits with status 2.
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import Terminal

REPORT_C = """\
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rows;
    int i;
    FILE *out;
    if (argc < 2) {
        fprintf(stderr, "usage: %s ROWS [CROP]\\n", argv[0]);
        return 1;
    }
    rows = atoi(argv[1]);
    out = fopen("report.csv", "w");
    if (out == NULL) {
        perror("report.csv");
        return 2;
    }
    fprintf(out, "row,crop\\n");
    for (i = 0; i < rows; i++) {
        fprintf(out, "%d,%s\\n", i + 1, argc > 2 ? argv[2] : "wheat");
    }
    fclose(out);
    printf("wrote %d rows to report.csv\\n", rows);
    return 0;
}
"""
READ_PLAN_C = """\
#include <stdio.h>

int main(int argc, char *argv[]) {
    char line[64];
    int count = 0;
    FILE *in = fopen(argc > 1 ? argv[1] : "plan.txt", "r");
    if (in == NULL) {
        perror(argc > 1 ? argv[1] : "plan.txt");
        return 2;
    }
    while (fgets(line, sizeof line, in) != NULL) {
        count++;
    }
    fclose(in);
    printf("%d lines\\n", count);
    return 0;
}
"""


@pytest.fixture
def io_project(term: Terminal) -> Terminal:
    term.write({"report.c": REPORT_C, "read_plan.c": READ_PLAN_C,
                "plan.txt": "till\nplant\nwater\n"})
    term.ok("cc -Wall report.c -o report")
    term.ok("cc -Wall read_plan.c -o read_plan")
    return term


def test_missing_argument_prints_usage_with_argv0(io_project: Terminal) -> None:
    line = io_project.run("./report")
    assert line.exit_code == 1 and line.stderr == "usage: ./report ROWS [CROP]\n"


def test_arguments_are_parsed_and_the_report_is_written(io_project: Terminal) -> None:
    line = io_project.ok("./report 3 carrot")
    assert line.stdout == "wrote 3 rows to report.csv\n"
    assert io_project.ok("cat report.csv").stdout == "row,crop\n1,carrot\n2,carrot\n3,carrot\n"


def test_the_report_is_an_ordinary_file_for_the_shell(io_project: Terminal) -> None:
    io_project.ok("./report 2")
    assert io_project.ok("ls").stdout.split().count("report.csv") == 1
    assert io_project.ok("grep -c wheat report.csv").stdout == "2\n"
    assert io_project.ok("wc -l report.csv").stdout.split()[0] == "3"


def test_atoi_of_a_word_is_zero(io_project: Terminal) -> None:
    io_project.ok("./report many")
    assert io_project.ok("cat report.csv").stdout == "row,crop\n"


def test_quoted_arguments_arrive_as_one_argv_entry(io_project: Terminal) -> None:
    io_project.ok("./report 1 'sweet corn'")
    assert io_project.ok("cat report.csv").stdout == "row,crop\n1,sweet corn\n"


def test_the_run_counts_file_constructs(io_project: Terminal) -> None:
    constructs = io_project.ok("./report 2").runs[0].constructs
    assert constructs.get("file_open") == 1 and constructs.get("file_write", 0) >= 3


def test_reading_a_file_the_shell_wrote(io_project: Terminal) -> None:
    assert io_project.ok("./read_plan").stdout == "3 lines\n"
    io_project.ok("echo harvest >> plan.txt")
    assert io_project.ok("./read_plan").stdout == "4 lines\n"


def test_fopen_failure_reports_with_perror_and_exits_2(io_project: Terminal) -> None:
    line = io_project.run("./read_plan seeds/missing.txt")
    assert line.exit_code == 2
    assert line.stderr == "seeds/missing.txt: No such file or directory\n"
    assert line.stdout == ""
    assert io_project.run("./read_plan seeds/missing.txt; echo $?").stdout == "2\n"


def test_fopen_of_a_folder_for_writing_fails(io_project: Terminal) -> None:
    io_project.ok("mkdir report.csv")
    line = io_project.run("./report 1")
    assert line.exit_code == 2 and line.stderr == "report.csv: Is a directory\n"


def test_writing_into_a_read_only_place_fails(io_project: Terminal) -> None:
    io_project.ok("cd /usr/include")
    line = io_project.run("/home/user/project/report 1")
    assert line.exit_code == 2 and line.stderr == "report.csv: Permission denied\n"
