"""The event protocol of a run: stepping, calls, input, warnings and the one terminal event."""
from __future__ import annotations

import pytest

from pycc.core.events import (
    CallEnter, CallExit, Exited, InputRequest, Output, RunWarning, Statement, Trap,
)
from pycc.interp.interpreter import Interpreter
from pycc.interp.stdlib import make_registry
from tests.interp.test_interpreter_support import build, events_of, run_c

LOOP = """int main(void) {
    int total = 0;
    for (int i = 0; i < 2; i++)
        total += i;
    return total;
}
"""


def statement_lines(events: list) -> list[int]:
    return [event.location.line for event in events if isinstance(event, Statement)]


def terminal_events(events: list) -> list:
    return [event for event in events if isinstance(event, (Exited, Trap))]


# --- stepping ----------------------------------------------------------------------------

def test_for_loop_steps_header_and_body_in_turn() -> None:
    events, code, _ = events_of(LOOP)
    assert statement_lines(events) == [2, 3, 4, 3, 4, 3, 5]
    assert code == 1


def test_while_loop_steps_its_condition_every_round() -> None:
    source = "int main(void) {\n  int n = 2;\n  while (n > 0)\n    n--;\n  return n;\n}\n"
    events, _, _ = events_of(source)
    assert statement_lines(events) == [2, 3, 4, 3, 4, 3, 5]


def test_do_while_steps_the_condition_after_the_body() -> None:
    source = "int main(void) {\n  int n = 0;\n  do\n    n++;\n  while (n < 2);\n  return n;\n}\n"
    events, code, _ = events_of(source)
    assert statement_lines(events) == [2, 4, 5, 4, 5, 6]
    assert code == 2


def test_if_condition_is_its_own_step() -> None:
    source = ("int main(void) {\n  int x = 1;\n  if (x)\n    x = 5;\n  else\n    x = 6;\n"
              "  return x;\n}\n")
    events, code, _ = events_of(source)
    assert statement_lines(events) == [2, 3, 4, 7]
    assert code == 5


def test_statement_depth_counts_calls() -> None:
    source = "int f(void) {\n  return 1;\n}\nint main(void) {\n  return f();\n}\n"
    events, _, _ = events_of(source)
    depths = [(e.location.line, e.depth) for e in events if isinstance(e, Statement)]
    assert depths == [(5, 0), (2, 1)]


def test_statement_count_matches_the_interpreter() -> None:
    events, _, interpreter = events_of(LOOP)
    assert interpreter.statements_executed == len(statement_lines(events)) == 7


# --- calls ----------------------------------------------------------------------------------

def test_calls_are_bracketed_by_enter_and_exit() -> None:
    source = "int sq(int v) { return v * v; }\nint main(void) { return sq(sq(2)); }\n"
    events, code, _ = events_of(source)
    calls = [(type(e).__name__, e.name) for e in events if isinstance(e, (CallEnter, CallExit))]
    assert calls == [("CallEnter", "main"), ("CallEnter", "sq"), ("CallExit", "sq"),
                     ("CallEnter", "sq"), ("CallExit", "sq"), ("CallExit", "main")]
    assert code == 16


def test_call_enter_carries_the_call_site() -> None:
    source = "void f(void) { }\nint main(void) {\n  f();\n  return 0;\n}\n"
    events, _, _ = events_of(source)
    enter = [e for e in events if isinstance(e, CallEnter) and e.name == "f"][0]
    assert enter.location is not None and enter.location.line == 3


# --- terminal events ---------------------------------------------------------------------------

def test_exactly_one_exited_event_last() -> None:
    events, code, _ = events_of("int main(void) { return 3; }\n")
    assert terminal_events(events) == [Exited(3)]
    assert events[-1] == Exited(3)
    assert code == 3


def test_exactly_one_trap_event_last() -> None:
    events, code, _ = events_of("int main(void) { int *p = 0; *p = 1; return 0; }\n")
    terminals = terminal_events(events)
    assert len(terminals) == 1 and isinstance(terminals[0], Trap)
    assert events[-1] is terminals[0]
    assert terminals[0].diagnostic.code == "null_dereference"
    assert code == 139


def test_trap_diagnostic_is_a_runtime_error_at_the_statement() -> None:
    source = "int main(void) {\n  int zero = 0;\n  return 4 / zero;\n}\n"
    events, _, _ = events_of(source)
    trap = events[-1]
    assert isinstance(trap, Trap)
    assert trap.diagnostic.code == "division_by_zero"
    assert trap.diagnostic.phase.value == "runtime"
    assert trap.diagnostic.location is not None and trap.diagnostic.location.line == 3


def test_exit_status_is_masked_to_a_byte() -> None:
    events, code, _ = events_of("#include <stdlib.h>\nint main(void) { exit(-1); }\n")
    assert events[-1] == Exited(255)
    assert code == 255


def test_exit_still_reports_leaks_before_exited() -> None:
    source = "#include <stdlib.h>\nint main(void) { malloc(4); exit(0); }\n"
    events, _, _ = events_of(source)
    assert isinstance(events[-2], RunWarning)
    assert events[-2].diagnostic.code == "memory_leak"
    assert isinstance(events[-1], Exited)


def test_files_left_open_are_reported_at_exit() -> None:
    from pycc.interp.hostio import DictHostFiles, ProgramIO
    source = ('#include <stdio.h>\nint main(void) {\n  fopen("a.txt", "w");\n'
              '  fopen("b.txt", "w");\n  return 0;\n}\n')
    run = run_c(source, io=ProgramIO(DictHostFiles()))
    codes = [w.code for w in run.summary.warnings]
    assert codes.count("file_not_closed") == 2


# --- input and output ----------------------------------------------------------------------------

def test_input_request_carries_the_prompt() -> None:
    source = ('#include <stdio.h>\nint main(void) {\n  int age;\n  printf("Age: ");\n'
              '  scanf("%d", &age);\n  printf("next year %d\\n", age + 1);\n  return 0;\n}\n')
    events, code, _ = events_of(source, stdin_lines=["41\n"])
    requests = [e for e in events if isinstance(e, InputRequest)]
    assert requests == [InputRequest("Age: ")]
    outputs = "".join(e.text for e in events if isinstance(e, Output))
    assert outputs == "Age: next year 42\n"
    assert code == 0


def test_end_of_input_reply() -> None:
    source = ('#include <stdio.h>\nint main(void) {\n  int c = getchar();\n'
              '  return c == EOF ? 9 : 0;\n}\n')
    events, code, _ = events_of(source, stdin_lines=[None])
    assert code == 9
    assert any(isinstance(e, InputRequest) for e in events)


# --- warnings -------------------------------------------------------------------------------------

def test_warning_is_reported_once_per_code_and_place() -> None:
    source = ("int main(void) {\n  int x = 2147483647;\n  for (int i = 0; i < 3; i++)\n"
              "    x = x + 1;\n  return 0;\n}\n")
    run = run_c(source)
    assert [w.code for w in run.summary.warnings] == ["signed_overflow"]


def test_warnings_arrive_before_the_next_statement() -> None:
    source = "int main(void) {\n  int x;\n  int y = x;\n  return 0;\n}\n"
    events, _, _ = events_of(source)
    index = next(i for i, e in enumerate(events) if isinstance(e, RunWarning))
    assert isinstance(events[index + 1], Statement)
    assert events[index + 1].location.line == 4
    assert events[index].diagnostic.location.line == 3


def test_pending_warnings_come_out_before_a_trap() -> None:
    source = "int main(void) {\n  int x;\n  int y = x / 0;\n  return y;\n}\n"
    events, _, _ = events_of(source)
    kinds = [type(e).__name__ for e in events[-2:]]
    assert kinds == ["RunWarning", "Trap"]


# --- host control ---------------------------------------------------------------------------------

def test_closing_the_generator_stops_the_run() -> None:
    registry = make_registry()
    interpreter = Interpreter(build("int main(void) { while (1) { } }\n", registry), registry)
    generator = interpreter.run(["prog"])
    for _ in range(50):
        next(generator)
    generator.close()                       # what the host does for Ctrl+C
    with pytest.raises(StopIteration):
        next(generator)
    assert interpreter.statements_executed > 0


def test_an_interpreter_runs_once() -> None:
    registry = make_registry()
    interpreter = Interpreter(build("int main(void) { return 0; }\n", registry), registry)
    list(interpreter.run(["prog"]))
    with pytest.raises(RuntimeError):
        next(interpreter.run(["prog"]))


def test_budget_is_counted_in_statements() -> None:
    source = "int main(void) {\n  int n = 0;\n  while (1)\n    n++;\n}\n"
    events, code, interpreter = events_of(source, budget=10)
    assert code == 137
    assert interpreter.statements_executed == 11
    assert len(statement_lines(events)) == 10


def test_a_local_struct_definition_is_not_a_step() -> None:
    source = "int main(void) {\n  struct P { int x; };\n  struct P p = {4};\n  return p.x;\n}\n"
    events, code, _ = events_of(source)
    assert statement_lines(events) == [3, 4]
    assert code == 4
