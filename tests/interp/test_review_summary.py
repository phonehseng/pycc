"""Review tests for interp.summary: the call trace under recursion, and edge cases."""
from __future__ import annotations

from pycc.core.diagnostics import SourceLocation
from pycc.core.events import CallEnter, CallExit, Statement
from pycc.interp.summary import LAST_STATEMENTS, TRACE_EVENTS, SummaryBuilder


def recursion(depth: int, unwind: bool) -> SummaryBuilder:
    """main calls fact, which calls itself ``depth`` times; optionally all return."""
    builder = SummaryBuilder()
    builder.feed(CallEnter("main"))
    for _ in range(depth):
        builder.feed(CallEnter("fact"))
    if unwind:
        for _ in range(depth):
            builder.feed(CallExit("fact"))
    return builder


def test_shallow_recursion_is_counted() -> None:
    assert recursion(5, unwind=True).trace_summary() == "main -> fact x5"


def test_deep_recursion_still_trapped_shows_the_recursing_function() -> None:
    # A stack_overflow trap: the window holds only calls into fact.
    assert recursion(500, unwind=False).trace_summary() == f"... -> fact x{TRACE_EVENTS}"


def test_deep_recursion_that_unwound_is_not_an_empty_trace() -> None:
    # The window holds only returns from fact; the trace must still name it.
    assert recursion(500, unwind=True).trace_summary() == f"... -> fact x{TRACE_EVENTS}"


def test_mutual_recursion_keeps_first_call_order() -> None:
    builder = SummaryBuilder()
    for _ in range(3):
        builder.feed(CallEnter("is_even"))
        builder.feed(CallEnter("is_odd"))
    assert builder.trace_summary() == "is_even x3 -> is_odd x3"


def test_unwound_short_recursion_counts_the_calls() -> None:
    builder = recursion(10, unwind=True)
    assert "main" in builder.trace_summary()
    assert builder.trace_summary().endswith("fact x10")


def test_statement_window_keeps_only_the_last_ones() -> None:
    builder = SummaryBuilder()
    for line in range(1, 30):
        builder.feed(Statement(SourceLocation("main.c", line, 1)))
    rendered = builder.finish(0).last_statements
    assert len(rendered) == LAST_STATEMENTS
    assert rendered[0] == f"main.c:{30 - LAST_STATEMENTS}:"


def test_header_statements_use_the_header_path() -> None:
    builder = SummaryBuilder()
    builder.feed(Statement(SourceLocation("include/parts.h", 4, 5)))
    lines = {("include/parts.h", 4): "return 4;"}
    assert builder.last_statements(lines) == ["include/parts.h:4: return 4;"]


def test_statement_count_without_interpreter_counts_events() -> None:
    builder = SummaryBuilder()
    for line in range(7):
        builder.feed(Statement(SourceLocation("main.c", line + 1, 1)))
    assert builder.finish(0).statements == 7
