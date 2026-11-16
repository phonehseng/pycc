"""Scenario 8: object files and executables are real files in the virtual filesystem.

``cc -c`` writes the bytes of an ``ObjectFile`` and ``cc -o`` those of an
``Executable``; the shell treats them like a real system would: ``cat`` shows
gibberish, ``file`` tells them apart, an object cannot be run, a program can be
copied, moved and still run, and the bytes load back into the toolchain.
"""
from __future__ import annotations

import pytest

from pycc.interp.compiler import link_executable, load_executable
from pycc.interp.linker import ObjectFile
from tests.integration.conftest import PROJECT, Terminal, assert_reported

DRONE_C = "int steps(void) {\n    return 3;\n}\n"
MAIN_C = ("#include <stdio.h>\nint steps(void);\nint main(void) {\n"
          "    printf(\"%d steps\\n\", steps());\n    return 0;\n}\n")


@pytest.fixture
def built(term: Terminal) -> Terminal:
    term.write({"drone.c": DRONE_C, "main.c": MAIN_C})
    term.ok("cc -Wall -c drone.c")
    term.ok("cc -Wall -c main.c")
    term.ok("cc main.o drone.o -o bot")
    return term


def test_cat_of_an_object_file_is_gibberish(built: Terminal) -> None:
    text = built.ok("cat drone.o").stdout
    assert text.startswith("CVOBJ1\n")
    assert "steps" not in text          # compressed: the source is not readable
    assert "�" in text             # bytes that are not text show as replacement marks


def test_file_tells_objects_programs_and_sources_apart(built: Terminal) -> None:
    out = built.ok("file drone.o bot main.c").stdout.splitlines()
    assert out[0] == "drone.o: pycc relocatable object file (link it with cc)"
    assert out[1] == "bot: pycc executable, runnable program"
    assert out[2] == "main.c: C source, ASCII text"


def test_an_object_file_is_not_runnable_and_chmod_is_not_the_fix(built: Terminal) -> None:
    line = built.run("./drone.o")
    assert line.exit_code == 126
    assert_reported(line, "bash: ./drone.o: Permission denied",
                    "chmod +x` will not help. Link it with cc into a program first")


def test_an_executable_object_still_cannot_run(built: Terminal) -> None:
    built.ok("chmod +x drone.o")
    line = built.run("./drone.o")
    assert line.exit_code == 126 and line.codes == ["cannot_execute_binary"]
    assert line.stderr.startswith("bash: ./drone.o: cannot execute binary file: Exec format "
                                  "error\n")
    assert "`cc drone.o -o drone`" in line.stderr


def test_the_object_bytes_load_back(built: Terminal, toolchain) -> None:
    obj = ObjectFile.from_bytes(built.read("drone.o"), toolchain)
    assert obj.unit_name == "drone.c" and obj.is_loaded
    assert "steps" in obj.symbols.defined


def test_the_executable_bytes_load_and_link(built: Terminal, toolchain) -> None:
    executable = load_executable(built.read("bot"))
    assert sorted(obj.unit_name for obj in executable.objects) == ["drone.c", "main.c"]
    assert link_executable(executable, toolchain.builtins).ok


def test_a_copied_program_runs_from_another_folder(built: Terminal) -> None:
    built.ok("mkdir -p tools && cp bot tools/bot2")
    assert built.ok("./tools/bot2").stdout == "3 steps\n"
    built.ok("cd tools")
    assert built.ok("./bot2").stdout == "3 steps\n"


def test_a_moved_program_keeps_working(built: Terminal) -> None:
    built.ok("mv bot prog")
    assert built.ok("./prog").stdout == "3 steps\n"
    assert built.run("./bot").exit_code == 127


def test_the_project_folder_is_on_the_path(built: Terminal) -> None:
    # The default PATH is /bin:., so a bare name finds ./bot.
    assert built.ok("bot").stdout == "3 steps\n"


def test_a_misspelt_program_name(built: Terminal) -> None:
    line = built.run("bto")
    assert line.exit_code == 127 and "bash: bto: command not found" in line.stderr
    assert built.run("./bto").stderr.startswith("bash: ./bto: No such file or directory\n")


def test_rebuilding_after_deleting_objects(built: Terminal) -> None:
    built.ok("rm drone.o main.o")
    line = built.run("cc main.o drone.o -o bot")
    assert line.exit_code == 1
    assert line.stderr.startswith("cc: error: main.o: No such file or directory\n")


def test_a_text_file_with_the_object_name_is_not_an_object(built: Terminal) -> None:
    built.vfs.write_file(f"{PROJECT}/fake.o", "int x;\n")
    line = built.run("cc fake.o -o fake")
    assert line.exit_code == 1
    assert "fake.o" in line.stderr
