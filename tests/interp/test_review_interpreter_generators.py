"""Review tests: generator discipline, traps crossing generators, budget, host control.

The interpreter is a stack of generators. These tests drive it by hand, the way a
stepping debugger does, and check that no event is lost when a builtin pauses
deep inside a condition, that a trap anywhere becomes exactly one ``Trap`` event,
and that closing the run (Ctrl+C) really stops everything underneath it.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from pycc.core.events import CallEnter, Event, InputRequest, Statement, Trap, DeviceCall
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.interpreter import Interpreter
from pycc.interp.stdlib import make_registry
from pycc.interp.values import Value
from tests.interp.test_interpreter_machine import PROTOTYPES, registry_with_extras
from tests.interp.test_interpreter_support import build, events_of, run_c, trap_code


def drive(interpreter: Interpreter, device: Any = None,
          stdin: list[str | None] | None = None) -> tuple[list[Event], int]:
    """Run to the end, answering DeviceCall with ``device(action)`` and stdin by line."""
    lines = list(stdin or [])
    generator = interpreter.run(["prog"])
    events: list[Event] = []
    reply: Any = None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as done:
            return events, done.value
        events.append(event)
        reply = None
        if isinstance(event, DeviceCall):
            reply = device(event)
        elif isinstance(event, InputRequest):
            reply = lines.pop(0) if lines else None


def extras_interpreter(body: str, before: str = "",
                       registry: BuiltinRegistry | None = None) -> Interpreter:
    """An interpreter for ``main`` over the extra test builtins of the machine tests."""
    registry = registry or registry_with_extras()
    source = f"{PROTOTYPES}{before}\nint main(void) {{\n{body}\n}}\n"
    return Interpreter(build(source, registry), registry)


# --- no event lost ------------------------------------------------------------------------------

def test_device_call_in_a_while_condition_is_asked_every_test() -> None:
    answers = iter([3, 2, 0])
    interpreter = extras_interpreter(
        'int rounds = 0;\nwhile (scan_ahead(rounds) != 0) rounds++;\nreturn rounds;')
    events, code = drive(interpreter, device=lambda action: next(answers))
    actions = [event.args for event in events if isinstance(event, DeviceCall)]
    assert actions == [[0], [1], [2]]
    assert code == 2


def test_nested_device_calls_inside_an_if_condition_keep_their_order() -> None:
    seen: list[int] = []

    def device(action: DeviceCall) -> int:
        seen.append(action.args[0])
        return action.args[0] * 2

    interpreter = extras_interpreter(
        'if (scan_ahead(1) + ping(scan_ahead(2)) > 6) return 7;\nreturn 1;')
    _, code = drive(interpreter, device=device)
    assert seen == [1, 2]
    assert code == 7                        # 2 + (4 + 1) == 7 > 6


def test_input_request_inside_a_loop_condition_reaches_the_host() -> None:
    source = ('#include <stdio.h>\nint main(void) {\n  int c, n = 0;\n'
              '  while ((c = getchar()) != EOF) n++;\n  return n;\n}\n')
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    events, code = drive(interpreter, stdin=["ab\n", None])
    assert sum(isinstance(event, InputRequest) for event in events) == 2
    assert code == 3


# --- traps crossing generator boundaries ---------------------------------------------------------

def test_trap_from_a_builtin_inside_a_condition_is_one_trap_then_the_end() -> None:
    interpreter = extras_interpreter('int x = 0;\nwhile (x < 3 && (explode(), 1)) x++;\nreturn 0;')
    events, code = drive(interpreter)
    traps = [event for event in events if isinstance(event, Trap)]
    assert len(traps) == 1 and events[-1] is traps[0]
    assert traps[0].diagnostic.code == "invalid_direction"
    assert code == 134


def test_trap_inside_a_qsort_comparator_is_one_trap_with_the_stack_intact() -> None:
    source = ('#include <stdlib.h>\n'
              'int cmp(const void *a, const void *b) {\n'
              '  int *p = 0;\n'
              '  return *p;\n'
              '}\n'
              'int main(void) {\n'
              '  int a[3] = {3, 1, 2};\n'
              '  qsort(a, 3, sizeof(int), cmp);\n'
              '  return 0;\n'
              '}\n')
    events, code, interpreter = events_of(source)
    traps = [event for event in events if isinstance(event, Trap)]
    assert len(traps) == 1 and events[-1] is traps[0]
    assert traps[0].diagnostic.code == "null_dereference"
    assert traps[0].diagnostic.location.line == 4
    assert code == 139
    assert [frame.name for frame in interpreter.stack_frames()] == ["main", "cmp"]


def test_statements_of_a_comparator_report_depth_one() -> None:
    source = ('#include <stdlib.h>\n'
              'int cmp(const void *a, const void *b) {\n'
              '  return *(const int *)a - *(const int *)b;\n'
              '}\n'
              'int main(void) {\n'
              '  int a[2] = {2, 1};\n'
              '  qsort(a, 2, sizeof(int), cmp);\n'
              '  return a[0];\n'
              '}\n')
    events, code, _ = events_of(source)
    depths = {event.location.line: event.depth for event in events if isinstance(event, Statement)}
    assert depths[3] == 1 and depths[7] == 0
    assert code == 1


# --- host control: closing the run ------------------------------------------------------------

def closing_registry(log: list[str]) -> BuiltinRegistry:
    """A registry whose ``each`` builtin calls back into C and logs its own clean-up."""
    def each(machine: Any, args: list[Value]) -> Generator[Any, Any, Value]:
        try:
            for number in range(int(args[1].raw)):
                yield from machine.call_function(int(args[0].raw), [machine.make_int(number)])
            return machine.make_int(0)
        finally:
            log.append("builtin cleaned up")

    registry = make_registry()
    registry.register("each", "int each(int (*f)(int), int n);", each)
    return registry


def test_closing_during_a_callback_closes_the_builtin_too() -> None:
    log: list[str] = []
    registry = closing_registry(log)
    source = ('int each(int (*f)(int), int n);\n'
              'int body(int n) {\n  while (1) { }\n  return n;\n}\n'
              'int main(void) {\n  return each(body, 3);\n}\n')
    generator = Interpreter(build(source, registry), registry).run(["prog"])
    for _ in range(40):
        next(generator)
    generator.close()                         # Ctrl+C while the callback loops
    assert log == ["builtin cleaned up"]
    with pytest.raises(StopIteration):
        next(generator)


def test_closing_while_waiting_for_input_ends_cleanly() -> None:
    registry = make_registry()
    source = '#include <stdio.h>\nint main(void) {\n  return getchar();\n}\n'
    generator = Interpreter(build(source, registry), registry).run(["prog"])
    event = next(generator)
    while not isinstance(event, InputRequest):
        event = next(generator)
    generator.close()                         # must not raise "ignored GeneratorExit"
    with pytest.raises(StopIteration):
        next(generator)


def test_call_enter_is_yielded_for_the_callback() -> None:
    log: list[str] = []
    registry = closing_registry(log)
    source = ('int each(int (*f)(int), int n);\n'
              'int body(int n) {\n  return n;\n}\n'
              'int main(void) {\n  return each(body, 2);\n}\n')
    interpreter = Interpreter(build(source, registry), registry)
    events, code = drive(interpreter)
    assert [event.name for event in events if isinstance(event, CallEnter)] == \
        ["main", "body", "body"]
    assert code == 0 and log == ["builtin cleaned up"]


# --- budget accounting ---------------------------------------------------------------------

def test_every_for_condition_test_is_charged() -> None:
    source = ('int main(void) {\n  int s = 0;\n  for (int i = 0; i < 3; i++)\n'
              '    s += i;\n  return s;\n}\n')
    events, code, interpreter = events_of(source)
    lines = [event.location.line for event in events if isinstance(event, Statement)]
    # decl, header (init + test), then body / header (step + test) three times, return.
    assert lines == [2, 3, 4, 3, 4, 3, 4, 3, 5]
    assert interpreter.statements_executed == len(lines)
    assert code == 3


def test_budget_runs_out_in_a_loop_condition_with_an_exact_count() -> None:
    source = 'int main(void) {\n  int i = 0;\n  while (i < 1000000) i++;\n  return 0;\n}\n'
    events, code, interpreter = events_of(source, budget=10)
    assert sum(isinstance(event, Statement) for event in events) == 10
    assert isinstance(events[-1], Trap)
    assert events[-1].diagnostic.code == "instruction_budget"
    assert interpreter.statements_executed == 11
    assert code == 137


def test_do_while_tests_cost_a_step_each() -> None:
    source = ('int main(void) {\n  int i = 0;\n  do {\n    i++;\n  } while (i < 3);\n'
              '  return i;\n}\n')
    events, code, _ = events_of(source)
    lines = [event.location.line for event in events if isinstance(event, Statement)]
    assert lines == [2, 4, 5, 4, 5, 4, 5, 6]
    assert code == 3


def test_continue_in_a_for_loop_still_runs_the_step() -> None:
    run = run_c('#include <stdio.h>\nint main(void) {\n  int n = 0;\n'
                '  for (int i = 0; i < 6; i++) {\n    if (i % 2) continue;\n    n++;\n  }\n'
                '  printf("%d\\n", n);\n  return 0;\n}\n')
    assert run.stdout == "3\n" and trap_code(run) is None


# --- the machine facade ------------------------------------------------------------------------

def test_write_cstring_with_no_room_writes_nothing() -> None:
    run = run_c(f"{PROTOTYPES}int main(void) {{\n  char buf[2] = \"x\";\n"
                "  int n = label(buf, 0);\n  return n * 10 + (buf[0] == 'x');\n}\n",
                registry=registry_with_extras())
    assert run.exit_code == 1 and trap_code(run) is None


def test_write_cstring_truncates_to_the_limit() -> None:
    run = run_c(f"{PROTOTYPES}int main(void) {{\n  char buf[8];\n"
                "  int n = label(buf, 3);\n  printf(\"%d %s\\n\", n, buf);\n  return 0;\n}\n",
                registry=registry_with_extras())
    assert run.stdout == "3 cr\n"
