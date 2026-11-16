"""assert, time, errno, registration details and the FakeMachine itself."""
from __future__ import annotations

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import CODE_CONCEPTS
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.c_types import INT, LONG
from pycc.interp.stdlib import finalize, install, make_registry
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.stdlib.headers import STANDARD_HEADERS
from pycc.interp.traps import TrapError


def lit(machine: FakeMachine, text: str):
    return machine.ptr(machine.literal(text))


# --- assert ------------------------------------------------------------------------------

def _assert(machine: FakeMachine, ok: int) -> None:
    machine.call("__cv_assert", machine.int_(ok), lit(machine, "count > 0"), lit(machine, "main.c"),
                 machine.int_(12), lit(machine, "main"))


def test_true_assert_does_nothing() -> None:
    machine = FakeMachine()
    _assert(machine, 1)
    assert machine.stderr == ""


def test_false_assert_prints_glibc_message_and_traps() -> None:
    machine = FakeMachine(argv0="./harvest")
    with pytest.raises(TrapError) as caught:
        _assert(machine, 0)
    assert machine.stderr == "./harvest: main.c:12: main: Assertion `count > 0' failed.\n"
    assert caught.value.code == "assert_failed"
    assert caught.value.concept is Concept.ERROR_HANDLING
    assert "count > 0" in caught.value.friendly


# --- time --------------------------------------------------------------------------------

def test_time_returns_device_tick_and_stores_it() -> None:
    machine = FakeMachine(device_tick=37)
    slot = machine.buffer(8)
    assert machine.call("time", machine.ptr(slot)).raw == 37
    assert machine.memory.load(slot, LONG) == 37
    assert machine.call("time", machine.null()).raw == 37


def test_time_is_zero_headless() -> None:
    assert FakeMachine().call("time", FakeMachine.null()).raw == 0


def test_clock_counts_statements() -> None:
    machine = FakeMachine()
    machine.statements_executed = 1234
    assert machine.call("clock").raw == 1234


# --- errno -------------------------------------------------------------------------------

def test_errno_is_a_real_int_in_memory() -> None:
    machine = FakeMachine()
    addr = machine.globals["errno"]
    assert machine.memory.load(addr, INT) == 0
    machine.set_errno(34)
    assert machine.memory.load(addr, INT) == 34
    assert machine.memory.object_at(addr).kind == "global"


# --- registration ------------------------------------------------------------------------

REQUIRED = {
    "printf", "fprintf", "sprintf", "snprintf", "puts", "putchar", "fputs", "fputc", "scanf",
    "fscanf", "sscanf", "fgets", "fgetc", "getchar", "ungetc", "fopen", "fclose", "fread",
    "fwrite", "feof", "ferror", "fflush", "remove", "rename", "perror", "malloc", "calloc",
    "realloc", "free", "atoi", "atol", "atof", "strtol", "strtod", "abs", "labs", "rand", "srand",
    "exit", "abort", "qsort", "bsearch", "getenv", "system", "strlen", "strcpy", "strncpy",
    "strcat", "strncat", "strcmp", "strncmp", "strchr", "strrchr", "strstr", "strtok", "memset",
    "memcpy", "memmove", "memcmp", "strdup", "strerror", "isalpha", "isdigit", "isalnum",
    "isspace", "isupper", "islower", "ispunct", "toupper", "tolower", "sqrt", "pow", "fabs",
    "floor", "ceil", "round", "sin", "cos", "tan", "atan2", "fmod", "exp", "log", "__cv_assert",
    "time", "clock",
}


def test_every_required_function_is_registered() -> None:
    assert REQUIRED <= set(make_registry().names())


def test_variables_are_registered_with_headers() -> None:
    registry = make_registry()
    assert {name: registry.header_for(name) for name in ("stdin", "stdout", "stderr", "errno")} == {
        "stdin": "stdio.h", "stdout": "stdio.h", "stderr": "stdio.h", "errno": "errno.h"}


def test_every_function_has_concept_and_known_header() -> None:
    for spec in make_registry().functions.values():
        assert isinstance(spec.concept, Concept), spec.name
        assert spec.header in STANDARD_HEADERS, spec.name


@pytest.mark.parametrize("name,info", [
    ("printf", (0, "printf")), ("fprintf", (1, "printf")), ("sprintf", (1, "printf")),
    ("snprintf", (2, "printf")), ("scanf", (0, "scanf")), ("fscanf", (1, "scanf")),
    ("sscanf", (1, "scanf")),
])
def test_format_info(name: str, info: tuple) -> None:
    assert make_registry().format_info(name) == info


def test_non_format_functions_have_no_format_info() -> None:
    assert make_registry().format_info("puts") is None


def test_install_twice_is_harmless() -> None:
    registry = BuiltinRegistry()
    install(registry)
    install(registry)
    assert len(registry.functions) == len(make_registry().functions)


@pytest.mark.parametrize("name", [
    "printf", "scanf", "fgets", "getchar", "qsort", "bsearch", "__cv_assert", "sscanf", "puts",
])
def test_event_producing_functions_are_generators(name: str) -> None:
    assert make_registry().get(name).is_generator


@pytest.mark.parametrize("name", ["strlen", "malloc", "sprintf", "fopen", "rand", "sqrt"])
def test_plain_functions_are_not_generators(name: str) -> None:
    assert not make_registry().get(name).is_generator


def test_diagnostic_codes_used_are_known() -> None:
    for code in ("null_dereference", "invalid_stream", "file_not_closed", "assert_failed",
                 "abort_called", "format_argument_type", "out_of_bounds"):
        assert code in CODE_CONCEPTS


# --- per-run state -----------------------------------------------------------------------

def test_two_machines_do_not_share_state() -> None:
    first, second = FakeMachine(), FakeMachine()
    first.call("srand", first.uint(99))
    assert second.call("rand").raw == 16838


def test_finalize_without_stdio_use_is_quiet() -> None:
    machine = FakeMachine()
    machine.state.clear()
    finalize(machine)
    assert machine.warning_codes() == []


def test_fake_machine_call_function_rejects_unknown_id() -> None:
    machine = FakeMachine()
    with pytest.raises(TrapError):
        machine.drive(machine.call_function(99, []))
