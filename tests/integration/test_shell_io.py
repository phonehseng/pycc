"""Scenario 5: compiled programs inside the shell's plumbing.

Pipes, output and error redirections, input from a file, a build script that reacts
to the exit status with ``if``/``$?``, and environment variables set with ``export``
that the program reads with ``getenv``.
"""
from __future__ import annotations

import pytest

from pycc.core.events import RunProgram
from tests.integration.conftest import Terminal

LOGGER_C = """\
#include <stdio.h>
int main(void) {
    printf("INFO drone awake\\n");
    printf("WARN soil is dry at (2, 3)\\n");
    printf("INFO harvested 4 wheat\\n");
    printf("WARN rock ahead\\n");
    fprintf(stderr, "error: basket full\\n");
    return 0;
}
"""
PLAN_READER_C = """\
#include <stdio.h>
#include <string.h>
int main(void) {
    int steps = 0;
    char crop[16];
    char line[64];
    if (scanf("%d %15s", &steps, crop) != 2) {
        printf("bad plan\\n");
        return 1;
    }
    printf("plan: %d steps of %s\\n", steps, crop);
    fgets(line, sizeof line, stdin);            /* the rest of the first line */
    while (fgets(line, sizeof line, stdin) != NULL) {
        line[strcspn(line, "\\n")] = '\\0';
        printf("note: %s\\n", line);
    }
    return steps;
}
"""
PLAN_TXT = "3 wheat\nwater first\nmind the pond\n"
ENV_C = """\
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const char *crop = getenv("CROP");
    if (crop == NULL) {
        printf("no CROP set\\n");
        return 1;
    }
    printf("planting %s\\n", crop);
    return 0;
}
"""
RUN_SH = """\
cc -Wall reader.c -o reader
./reader < plan.txt > plan.log
if [ $? -eq 3 ]; then
    echo "three steps planned"
else
    echo "unexpected plan"
fi
"""


@pytest.fixture
def shell_project(term: Terminal) -> Terminal:
    term.write({"logger.c": LOGGER_C, "reader.c": PLAN_READER_C, "plan.txt": PLAN_TXT,
                "env.c": ENV_C, "run.sh": RUN_SH})
    term.ok("cc -Wall logger.c -o bot")
    return term


def test_pipe_through_grep_and_wc(shell_project: Terminal) -> None:
    line = shell_project.ok("./bot | grep WARN | wc -l")
    assert line.stdout.strip() == "2"
    assert line.stderr == "error: basket full\n"       # stderr skips the pipe
    assert line.finished.constructs.get("pipeline") == 1


def test_grep_sees_the_program_output_lines(shell_project: Terminal) -> None:
    line = shell_project.ok("./bot | grep WARN")
    assert line.stdout == "WARN soil is dry at (2, 3)\nWARN rock ahead\n"


def test_redirect_stdout_into_a_folder(shell_project: Terminal) -> None:
    shell_project.ok("mkdir logs")
    line = shell_project.ok("./bot > logs/run.txt")
    assert line.stdout == "" and line.stderr == "error: basket full\n"
    assert shell_project.ok("cat logs/run.txt").stdout.splitlines() == [
        "INFO drone awake", "WARN soil is dry at (2, 3)", "INFO harvested 4 wheat",
        "WARN rock ahead"]


def test_append_redirect_adds_a_second_run(shell_project: Terminal) -> None:
    shell_project.ok("./bot > run.txt")
    shell_project.ok("./bot >> run.txt")
    assert shell_project.ok("wc -l run.txt").stdout.split()[0] == "8"


def test_redirect_stderr_into_a_file(shell_project: Terminal) -> None:
    line = shell_project.ok("./bot 2> errors.txt")
    assert line.stderr == "" and "WARN rock ahead" in line.stdout
    assert shell_project.ok("cat errors.txt").stdout == "error: basket full\n"


def test_both_streams_into_one_file(shell_project: Terminal) -> None:
    shell_project.ok("./bot > all.txt 2>&1")
    text = shell_project.ok("cat all.txt").stdout
    assert "error: basket full\n" in text and "INFO drone awake\n" in text


def test_stdin_from_a_file_feeds_scanf_and_fgets(shell_project: Terminal) -> None:
    shell_project.ok("cc -Wall reader.c -o reader")
    line = shell_project.run("./reader < plan.txt")
    assert line.exit_code == 3
    assert line.stdout == ("plan: 3 steps of wheat\nnote: water first\n"
                           "note: mind the pond\n")


def test_stdin_from_a_pipe(shell_project: Terminal) -> None:
    shell_project.ok("cc -Wall reader.c -o reader")
    line = shell_project.run("echo 5 carrot | ./reader")
    assert line.exit_code == 5 and line.stdout == "plan: 5 steps of carrot\n"


def test_empty_stdin_is_end_of_file(shell_project: Terminal) -> None:
    shell_project.ok("cc -Wall reader.c -o reader")
    line = shell_project.run("./reader < /dev/null")
    assert line.exit_code == 1 and line.stdout == "bad plan\n"


def test_a_script_builds_and_reacts_to_the_exit_status(shell_project: Terminal) -> None:
    line = shell_project.ok("sh run.sh")
    assert line.stdout == "three steps planned\n"
    assert shell_project.ok("cat plan.log").stdout.startswith("plan: 3 steps of wheat\n")
    assert line.finished.constructs.get("script_run") == 1
    assert line.finished.constructs.get("exit_code_test", 0) >= 1


def test_a_script_needs_chmod_before_dot_slash(shell_project: Terminal) -> None:
    refused = shell_project.run("./run.sh")
    assert refused.exit_code == 126 and "Permission denied" in refused.stderr
    shell_project.ok("chmod +x run.sh")
    assert shell_project.ok("./run.sh").stdout == "three steps planned\n"


def test_and_or_lists_follow_the_program_status(shell_project: Terminal) -> None:
    shell_project.ok("cc env.c -o env")
    line = shell_project.ok("./env && echo yes || echo no")
    assert line.stdout == "no CROP set\nno\n"


def test_getenv_sees_exported_variables(shell_project: Terminal) -> None:
    shell_project.ok("cc env.c -o env")
    assert shell_project.run("./env").stdout == "no CROP set\n"
    shell_project.ok("export CROP=pumpkin")
    assert shell_project.ok("./env").stdout == "planting pumpkin\n"


def test_an_unexported_variable_stays_in_the_shell(shell_project: Terminal) -> None:
    shell_project.ok("cc env.c -o env")
    shell_project.ok("CROP=carrot")
    assert shell_project.run("./env").stdout == "no CROP set\n"


def test_a_one_command_assignment_reaches_the_program(shell_project: Terminal) -> None:
    shell_project.ok("cc env.c -o env")
    assert shell_project.ok("CROP=carrot ./env").stdout == "planting carrot\n"
    assert shell_project.run("./env").stdout == "no CROP set\n"


def test_command_substitution_captures_program_output(shell_project: Terminal) -> None:
    line = shell_project.ok('count=$(./bot | grep -c INFO); echo "info lines: $count"')
    assert line.stdout == "info lines: 2\n"


def test_ctrl_c_during_a_run_leaves_status_130(shell_project: Terminal) -> None:
    shell_project.write({"loop.c": "int main(void) {\n    while (1) { }\n    return 0;\n}\n"})
    shell_project.ok("cc loop.c -o loop")
    generator = shell_project.shell.execute("./loop; echo after")
    event = next(generator)
    while not isinstance(event, RunProgram):     # the host is about to start the program
        event = generator.send(None)
    generator.close()                            # the user pressed Ctrl+C
    assert shell_project.shell.last_status == 130
    assert shell_project.ok("echo $?").stdout == "130\n"
