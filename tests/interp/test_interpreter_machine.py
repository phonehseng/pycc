"""The Machine facade as builtins see it inside a real run.

Each test registers a few tiny builtins next to the standard library; the C program
declares their prototypes itself, exactly as a device header would.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from pycc.core import diagnostics
from pycc.core.events import RunWarning, Trap, DeviceCall
from pycc.interp import interpreter as interpreter_module
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.interpreter import (
    TRAP_EXIT_CODES, Interpreter, ProgramExit, UnexpectedDeviceCall, run_to_completion,
)
from pycc.interp.machine import InterpreterMachine
from pycc.interp.stdlib import make_registry
from pycc.interp.values import Value
from tests.interp.test_interpreter_support import build, run_c, trap_code, warning_codes


def scan_ahead(machine: Any, args: list[Value]) -> Generator[Any, Any, Value]:
    """A machine API builtin: asks the device and returns its answer."""
    reply = yield DeviceCall("scan", [args[0].raw])
    return machine.make_int(int(reply))


def apply_each(machine: Any, args: list[Value]) -> Generator[Any, Any, Value]:
    """Call a C function pointer once per number and add up the results."""
    total = 0
    for number in range(int(args[1].raw)):
        result = yield from machine.call_function(int(args[0].raw), [machine.make_int(number)])
        total += int(result.raw)
    return machine.make_int(total)


def registry_with_extras() -> BuiltinRegistry:
    registry = make_registry()
    registry.register("ping", "int ping(int x);", lambda m, a: m.make_int(a[0].raw + 1))
    registry.register("scan_ahead", "int scan_ahead(int d);", scan_ahead)
    registry.register("apply_each", "int apply_each(int (*f)(int), int n);", apply_each)
    registry.register("explode", "void explode(void);", lambda m, a: m.trap(
        "invalid_direction", "move(): 7 is not a direction", "Use NORTH, EAST, SOUTH or WEST."))
    registry.register("grumble", "void grumble(void);", lambda m, a: m.warn(
        "bonk", "bonk: the bot walked into a fence", "Check can_move first."))
    registry.register("set_err", "void set_err(int e);", lambda m, a: m.set_errno(a[0].raw))
    registry.register("get_err", "int get_err(void);", lambda m, a: m.make_int(m.get_errno()))
    registry.register("tick", "int tick(void);", lambda m, a: m.make_int(m.device_tick))
    registry.register("stop", "void stop(int code);", lambda m, a: m.exit(a[0].raw))
    registry.register("plain", "int plain(void);", lambda m, a: 7)
    registry.register("halve", "double halve(double x);", lambda m, a: a[0].raw / 2)
    registry.register("label", "int label(char *buf, int size);",
                      lambda m, a: m.make_int(m.write_cstring(a[0].raw, "crate", a[1].raw)))
    registry.register("shout", "int shout(const char *s);",
                      lambda m, a: m.make_int(len(m.read_cstring(a[0].raw))))
    registry.register("where", "int where(void);",
                      lambda m, a: m.make_int(m.current_location.line))
    registry.register("steps", "int steps(void);",
                      lambda m, a: m.make_int(m.statements_executed))
    registry.register("bonus", "void bonus(void);", lambda m, a: m.count_construct("printf", 5))
    return registry


PROTOTYPES = """#include <stdio.h>
#include <errno.h>
int ping(int x); int scan_ahead(int d); int apply_each(int (*f)(int), int n);
void explode(void); void grumble(void); void set_err(int e); int get_err(void);
int tick(void); void stop(int code); int plain(void); double halve(double x);
int label(char *buf, int size); int shout(const char *s); int where(void);
int steps(void); void bonus(void);
"""


def run_extra(body: str, before: str = "", **kwargs: Any):
    source = f"{PROTOTYPES}{before}\nint main(void) {{\n{body}\n}}\n"
    return run_c(source, registry=registry_with_extras(), **kwargs)


# --- plain and generator builtins -----------------------------------------------------------

def test_plain_builtin_returns_a_value() -> None:
    assert run_extra('printf("%d\\n", ping(41));\nreturn 0;').stdout == "42\n"


def test_builtin_may_return_python_numbers() -> None:
    run = run_extra('printf("%d %.2f\\n", plain(), halve(5));\nreturn 0;')
    assert run.stdout == "7 2.50\n"


def test_device_call_goes_to_the_host_and_back() -> None:
    seen: list[DeviceCall] = []

    def device(action: DeviceCall) -> int:
        seen.append(action)
        return action.args[0] * 10

    run = run_extra('printf("%d\\n", scan_ahead(3) + scan_ahead(4));\nreturn 0;', device=device)
    assert run.stdout == "70\n"
    assert [(a.name, a.args) for a in seen] == [("scan", [3]), ("scan", [4])]
    assert run.summary.constructs["device_call"] == 2


def test_device_call_without_a_world_raises() -> None:
    with pytest.raises(UnexpectedDeviceCall):
        run_extra('return scan_ahead(1);')


def test_device_call_inside_an_expression_pauses_the_program() -> None:
    registry = registry_with_extras()
    program = build(f"{PROTOTYPES}int main(void) {{ return 1 + scan_ahead(2) * 3; }}\n", registry)
    generator = Interpreter(program, registry).run(["prog"])
    event = next(generator)
    while not isinstance(event, DeviceCall):
        event = generator.send(None)
    event = generator.send(5)
    while True:
        try:
            event = generator.send(None)
        except StopIteration as done:
            assert done.value == 16
            break


# --- traps, warnings, exit -------------------------------------------------------------------

def test_machine_trap_becomes_a_trap_event_at_the_call() -> None:
    run = run_extra('printf("a\\n");\nexplode();\nprintf("b\\n");\nreturn 0;')
    assert run.stdout == "a\n"
    assert trap_code(run) == "invalid_direction"
    assert run.summary.trap.location.line == 11
    assert run.exit_code == 134


def test_machine_warn_becomes_a_run_warning() -> None:
    run = run_extra('grumble();\nreturn 0;')
    assert warning_codes(run) == ["bonk"]
    assert run.summary.warnings[0].location.line == 10


def test_machine_exit_ends_cleanly() -> None:
    run = run_extra('printf("x\\n");\nstop(4);\nprintf("y\\n");\nreturn 0;')
    assert (run.exit_code, run.stdout, run.summary.trap) == (4, "x\n", None)


def test_program_exit_is_the_interpreters_exception() -> None:
    assert interpreter_module.ProgramExit is ProgramExit
    assert ProgramExit(3).code == 3


def test_trap_exit_codes_are_the_core_table() -> None:
    assert TRAP_EXIT_CODES is diagnostics.TRAP_EXIT_CODES


# --- errno, strings, clock ------------------------------------------------------------------------

def test_errno_lives_in_program_memory() -> None:
    run = run_extra('set_err(ENOENT);\nprintf("%d %d\\n", errno, get_err());\nerrno = 0;\n'
                    'return get_err();')
    assert run.stdout == "2 2\n"
    assert run.exit_code == 0


def test_errno_without_a_library_variable_falls_back() -> None:
    registry = BuiltinRegistry()
    registry.register("set_err", "void set_err(int e);", lambda m, a: m.set_errno(a[0].raw))
    registry.register("get_err", "int get_err(void);", lambda m, a: m.make_int(m.get_errno()))
    source = ("void set_err(int e); int get_err(void);\n"
              "int main(void) { set_err(5); return get_err(); }\n")
    program = build(source, registry)
    assert run_to_completion(program, registry, ["p"]).exit_code == 5


def test_write_and_read_cstrings() -> None:
    run = run_extra('char buf[4];\nint n = label(buf, sizeof buf);\n'
                    'printf("%s %d %d\\n", buf, n, shout("hello"));\nreturn 0;')
    assert run.stdout == "cra 4 5\n"


def test_machine_sees_the_current_statement_and_clock() -> None:
    run = run_extra('int a = 0;\na = where();\nreturn a * 100 + steps();')
    assert run.exit_code == (11 * 100 + 3) & 0xFF   # where() on line 11, 3rd statement


def test_device_tick_is_set_by_the_host() -> None:
    registry = registry_with_extras()
    interpreter = Interpreter(build(f"{PROTOTYPES}int main(void) {{ return tick(); }}\n",
                                    registry), registry)
    interpreter.device_tick = 12
    events = list(interpreter.run(["prog"]))
    assert events[-1].code == 12
    assert interpreter.machine.device_tick == 12


def test_builtins_can_count_constructs() -> None:
    assert run_extra('bonus();\nreturn 0;').summary.constructs["printf"] == 5


# --- calling back into C --------------------------------------------------------------------------

def test_call_function_runs_a_c_callback() -> None:
    run = run_extra('printf("%d\\n", apply_each(square, 4));\nreturn 0;',
                    before='int square(int x) { return x * x; }')
    assert run.stdout == "14\n"
    assert run.summary.constructs["function_call"] == 4


def test_callback_output_appears_in_order() -> None:
    run = run_extra('apply_each(show, 3);\nprintf("done\\n");\nreturn 0;',
                    before='int show(int x) { printf("%d ", x); return 0; }')
    assert run.stdout == "0 1 2 done\n"


def test_callback_through_a_null_pointer_traps() -> None:
    run = run_extra('int (*f)(int) = 0;\nreturn apply_each(f, 1);')
    assert trap_code(run) == "bad_function_pointer"
    assert run.exit_code == 139


def test_trap_inside_a_callback_keeps_the_stack() -> None:
    registry = registry_with_extras()
    source = (f"{PROTOTYPES}int boom(int x) {{ int *p = 0; return *p + x; }}\n"
              "int main(void) { return apply_each(boom, 2); }\n")
    interpreter = Interpreter(build(source, registry), registry)
    events = list(interpreter.run(["prog"]))
    assert isinstance(events[-1], Trap)
    assert [frame.name for frame in interpreter.stack_frames()] == ["main", "boom"]


def test_machine_rng_is_seeded() -> None:
    registry = make_registry()
    program = build("int main(void) { return 0; }\n", registry)
    first = Interpreter(program, registry, seed=7).machine.rng.random()
    second = Interpreter(program, registry, seed=7).machine.rng.random()
    assert first == second


def test_machine_is_the_concrete_facade() -> None:
    registry = make_registry()
    interpreter = Interpreter(build("int main(void) { return 0; }\n", registry), registry)
    machine = interpreter.machine
    assert isinstance(machine, InterpreterMachine)
    assert machine.memory is interpreter.memory
    assert machine.io is interpreter.io
    addr = machine.malloc(8)
    machine.write_bytes(addr, b"abc\0")
    assert machine.read_cstring(addr) == "abc"
    assert machine.read_bytes(addr, 2) == b"ab"
    machine.free(addr)
    assert interpreter.memory.leaks() == []


def test_warnings_from_builtins_are_events() -> None:
    registry = registry_with_extras()
    program = build(f"{PROTOTYPES}int main(void) {{\n  grumble();\n  return 0;\n}}\n", registry)
    events = list(Interpreter(program, registry).run(["p"]))
    assert any(isinstance(e, RunWarning) and e.diagnostic.code == "bonk" for e in events)
