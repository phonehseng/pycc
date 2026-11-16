"""Focused tests for the seams the end-to-end scenarios found and fixed.

Each block names the seam. They run the real toolchain, so a regression shows up here
with a small program before it shows up in a long scenario.
"""
from __future__ import annotations

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity
from pycc.interp.header_hints import (defining_headers, header_to_include, names_defined_in,
                                 point_at_headers)
from pycc.interp.toolchain import SystemHeaders, Toolchain
from pycc.shell.sysheaders import install_system_headers
from pycc.vfs import VirtualFS, standard_layout
from tests.integration.conftest import Counter, Terminal, counter_toolchain

# --- forgotten #include: constants and functions -----------------------------------------

def undeclared(name: str) -> Diagnostic:
    return Diagnostic(Severity.ERROR, "undeclared_identifier",
                      f"'{name}' undeclared (first use in this function)", "generic",
                      None, Concept.VARIABLES, Phase.CHECK)


def test_names_defined_in_reads_defines_and_enumerators_but_not_comments() -> None:
    text = ("/* enum Colour { RED = 1 }; */\n#define EOF (-1)\n"
            "enum Direction { NORTH = 0, EAST, SOUTH = 2 };\n")
    assert names_defined_in(text) == {"EOF", "NORTH", "EAST", "SOUTH"}


def test_null_is_suggested_from_stdio(toolchain: Toolchain) -> None:
    assert defining_headers("NULL", toolchain.headers)[0] == "stdio.h"


def test_names_are_suggested_through_the_public_header() -> None:
    headers = SystemHeaders({"motor_defs.h": "enum { FORWARD = 1, BACK = 2 };\n",
                             "motor.h": '#include "motor_defs.h"\nint drive(int way);\n',
                             "stdio.h": "#define EOF (-1)\n"})
    assert defining_headers("FORWARD", headers) == ["motor_defs.h"]
    assert header_to_include("motor_defs.h", headers) == "motor.h"
    assert header_to_include("stdio.h", headers) == "stdio.h"


def test_an_unknown_name_keeps_its_generic_text(toolchain: Toolchain) -> None:
    diag = undeclared("cuont")
    point_at_headers([diag], toolchain.headers)
    assert diag.friendly == "generic"


def test_a_name_from_an_included_header_keeps_its_text(toolchain: Toolchain) -> None:
    diag = undeclared("EOF")
    point_at_headers([diag], toolchain.headers, ["/usr/include/stdio.h"])
    assert diag.friendly == "generic"


def test_eof_without_stdio(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    return EOF;\n}\n"})
    line = term.run("cc main.c -o bot")
    assert "main.c:2:12: error: 'EOF' undeclared (first use in this function)" in line.stderr
    assert "it is defined in <stdio.h>. Did you forget #include <stdio.h>?" in line.stderr


def test_a_header_with_its_own_function_keeps_the_direct_hint() -> None:
    term = Terminal(standard_layout("project"), counter_toolchain())
    term.write({"main.c": "int main(void) {\n    return read_count();\n}\n"})
    assert "'read_count' is declared in <counter.h>." in term.run("cc main.c").stderr


# --- strspn / strcspn -------------------------------------------------------------------

def test_strcspn_and_strspn(term: Terminal) -> None:
    term.write({"main.c": '#include <stdio.h>\n#include <string.h>\nint main(void) {\n'
                          '    char line[] = "wheat\\n";\n'
                          '    printf("%zu %zu ", strcspn(line, "\\n"), strcspn("abc", "x"));\n'
                          '    printf("%zu %zu\\n", strspn("aab!", "ab"), strspn("", "a"));\n'
                          '    line[strcspn(line, "\\n")] = 0;\n'
                          '    printf("[%s]\\n", line);\n    return 0;\n}\n'})
    assert term.ok("cc -Wall main.c -o s && ./s").stdout == "5 3 3 0\n[wheat]\n"


def test_string_h_declares_the_span_functions(term: Terminal) -> None:
    text = term.ok("cat /usr/include/string.h").stdout
    assert "size_t strcspn(const char *s, const char *reject);" in text
    assert "size_t strspn(const char *s, const char *accept);" in text


# --- system headers from a plain mapping --------------------------------------------------

def test_install_system_headers_accepts_a_plain_mapping(toolchain: Toolchain) -> None:
    vfs = VirtualFS()
    written = install_system_headers(vfs, toolchain.headers.as_dict())
    assert "/usr/include/stdio.h" in written
    assert vfs.read_text("/usr/include/stdio.h") == toolchain.headers.get("stdio.h")


def test_install_system_headers_still_takes_system_headers() -> None:
    vfs = VirtualFS()
    assert install_system_headers(vfs, SystemHeaders({"a.h": "int a;\n"})) == [
        "/usr/include/a.h"]


# --- missing header: gcc's column --------------------------------------------------------

def test_a_missing_header_points_at_its_name(term: Terminal) -> None:
    term.write({"main.c": '#include <stdio.h>\n#include "dron.h"\nint main(void) { return 0; }\n'})
    line = term.run("cc main.c")
    assert line.stderr.startswith("main.c:2:10: fatal error: dron.h: No such file or directory\n")
    assert line.stderr.splitlines()[-1] == "compilation terminated."


# --- runtime warnings: printed once, counted once -----------------------------------------

PULSE_C = ("#include <counter.h>\nint main(void) {\n    pulse(1);\n    pulse(2);\n"
           "    return read_count() == 3 ? 0 : 1;\n}\n")


def test_runtime_warnings_go_to_the_programs_stderr(term: Terminal) -> None:
    term.write({"main.c": "#include <stdlib.h>\nint main(void) {\n    malloc(8);\n"
                          "    return 0;\n}\n"})
    term.ok("cc main.c -o bot")
    line = term.ok("./bot 2> err.txt")
    assert line.stderr == ""
    assert "warning: 8 bytes in 1 blocks are definitely lost" in term.ok("cat err.txt").stdout


def test_device_calls_are_counted_once_per_call() -> None:
    term = Terminal(standard_layout("project"), counter_toolchain(), Counter())
    term.write({"main.c": PULSE_C})
    constructs = term.ok("cc main.c -o bot && ./bot").runs[0].constructs
    assert constructs["device_call"] == 3
    assert term.device.count == 3
