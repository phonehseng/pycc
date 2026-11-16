"""What a debugger's variables panel and memory view can see of a running program."""
from __future__ import annotations

from typing import Any

from pycc.core.events import CallEnter, CallExit, Statement, Trap
from pycc.interp.c_types import INT
from pycc.interp.interpreter import CALL_TRACE_LIMIT, Interpreter
from pycc.interp.stdlib import make_registry
from tests.interp.test_interpreter_support import build, run_c

NESTED = """int g = 7;
static int s;
int inner(int a) {
    int b = a * 2;
    return b;
}
int main(void) {
    int x = 3;
    { int y = 4; inner(x + y); }
    static int calls = 1;
    return calls;
}
"""


def run_until(source: str, line: int, depth: int | None = None) -> tuple[Interpreter, Any]:
    """Start a run and stop at the first Statement on ``line``; returns it paused."""
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    generator = interpreter.run(["prog"])
    for event in generator:
        if isinstance(event, Statement) and event.location.line == line and (
                depth is None or event.depth == depth):
            return interpreter, generator
    raise AssertionError(f"line {line} never ran")


def value_of(interpreter: Interpreter, frame_name: str, var: str) -> int:
    for frame in interpreter.stack_frames():
        if frame.name == frame_name:
            for info in frame.locals:
                if info.name == var:
                    return int(interpreter.memory.load(info.addr, INT))
    raise KeyError(var)


# --- stack frames -------------------------------------------------------------------------

def test_stack_frames_while_paused_in_a_callee() -> None:
    interpreter, _ = run_until(NESTED, line=5)
    frames = interpreter.stack_frames()
    assert [frame.name for frame in frames] == ["main", "inner"]
    assert [info.name for info in frames[0].locals] == ["x", "y"]
    assert [info.name for info in frames[1].locals] == ["a", "b"]
    assert value_of(interpreter, "inner", "a") == 7
    assert value_of(interpreter, "inner", "b") == 14
    assert value_of(interpreter, "main", "y") == 4


def test_frame_locations_are_the_running_statements() -> None:
    interpreter, _ = run_until(NESTED, line=5)
    main, inner = interpreter.stack_frames()
    assert main.loc.line == 9          # the call site in main
    assert inner.loc.line == 5
    assert interpreter.current_location.line == 5
    assert main.id != inner.id


def test_block_locals_disappear_after_their_block() -> None:
    interpreter, _ = run_until(NESTED, line=11)
    (main,) = interpreter.stack_frames()
    names = [info.name for info in main.locals]
    assert "y" not in names and "x" in names
    assert "calls" in names            # static locals show up where they are declared


def test_shadowed_variables_are_both_visible() -> None:
    source = "int main(void) {\n  int x = 1;\n  {\n    int x = 2;\n    x++;\n  }\n  return 0;\n}\n"
    interpreter, _ = run_until(source, line=5)
    names = [info.name for info in interpreter.stack_frames()[0].locals]
    assert names == ["x", "x"]


def test_globals_lists_every_static_object() -> None:
    interpreter, _ = run_until(NESTED, line=8)
    names = sorted(info.name for info in interpreter.globals())
    assert names == ["calls", "g", "s"]
    g = next(info for info in interpreter.globals() if info.name == "g")
    assert interpreter.memory.load(g.addr, INT) == 7


def test_stack_is_intact_at_a_trap() -> None:
    source = ("int crash(int depth) {\n  int *p = 0;\n  if (depth == 0) return *p;\n"
              "  return crash(depth - 1);\n}\nint main(void) { return crash(2); }\n")
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    events = list(interpreter.run(["prog"]))
    assert isinstance(events[-1], Trap)
    assert [frame.name for frame in interpreter.stack_frames()] == ["main"] + ["crash"] * 3


def test_frames_are_empty_after_a_normal_exit() -> None:
    registry = make_registry()
    interpreter = Interpreter(build(NESTED, registry), registry)
    list(interpreter.run(["prog"]))
    assert interpreter.stack_frames() == []
    assert interpreter.depth == 0


# --- call trace and counters ------------------------------------------------------------------

def test_call_trace_is_a_bounded_ring() -> None:
    source = ("void tick(void) { }\n"
              "int main(void) { for (int i = 0; i < 100; i++) tick(); return 0; }\n")
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    list(interpreter.run(["prog"]))
    trace = list(interpreter.call_trace)
    assert len(trace) == CALL_TRACE_LIMIT
    assert isinstance(trace[-1], CallExit) and trace[-1].name == "main"
    assert all(isinstance(e, (CallEnter, CallExit)) for e in trace)


def test_summary_trace_comes_from_call_events() -> None:
    run = run_c("int f(void) { return 1; }\nint main(void) { return f() + f(); }\n")
    assert run.summary.trace_summary == "main -> f x2"


def test_construct_counts() -> None:
    source = """#include <stdio.h>
struct P { int x; };
int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int main(void) {
    int a[3] = {1, 2, 3};
    int *p = a;
    struct P s = {1};
    struct P *ps = &s;
    for (int i = 0; i < 3; i++) a[i]++;
    p++;
    p = p + 1;
    switch (*p) { default: break; }
    ps->x = s.x;
    return fact(3) - 6;
}
"""
    counts = run_c(source).summary.constructs
    assert counts["loop_iteration"] == 3
    assert counts["array_index"] == 3
    assert counts["pointer_arith"] == 2
    assert counts["switch"] == 1
    assert counts["struct_member"] == 2
    assert counts["pointer_deref"] == 2           # *p and ps->x
    assert counts["function_call"] == 3           # fact(3), fact(2), fact(1); main is not counted
    assert counts["recursion"] == 2


def test_summary_statements_and_last_lines() -> None:
    run = run_c("int main(void) {\n  int a = 1;\n  a++;\n  return a;\n}\n")
    assert run.summary.statements == 3
    assert run.summary.last_statements[-1] == "main.c:4: return a;"


def test_leak_warning_points_at_the_allocation() -> None:
    source = "#include <stdlib.h>\nint main(void) {\n  void *p = malloc(4);\n  return 0;\n}\n"
    warning = run_c(source).summary.warnings[0]
    assert warning.code == "memory_leak"
    assert warning.location is not None and warning.location.line == 3


def test_exit_from_a_callee_unwinds_every_frame() -> None:
    source = "#include <stdlib.h>\nvoid quit(void) { exit(2); }\nint main(void) { quit(); }\n"
    registry = make_registry()
    interpreter = Interpreter(build(source, registry), registry)
    events = list(interpreter.run(["prog"]))
    assert events[-1].code == 2
    assert interpreter.stack_frames() == []
