"""Tests for interp.summary.SummaryBuilder, fed with scripted event sequences."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation
from pycc.core.events import (
    CallEnter, CallExit, Exited, InputRequest, Output, RunWarning, Statement, Trap, DeviceCall,
)
from pycc.interp.source import PPLine
from pycc.interp.summary import LAST_STATEMENTS, TRACE_EVENTS, SummaryBuilder


def diag(code: str, severity: Severity = Severity.WARNING) -> Diagnostic:
    return Diagnostic(severity, code, f"{code} happened", "plain words", None,
                      Concept.LOOP_BOUNDS, Phase.RUNTIME)


@dataclass
class FakeMemory:
    leaked: int = 0

    def leaks(self) -> list[object]:
        return [object()] * self.leaked


@dataclass
class FakeInterpreter:
    """The attributes SummaryBuilder reads from the interpreter."""

    statements_executed: int = 0
    constructs: Counter = field(default_factory=Counter)
    memory: FakeMemory = field(default_factory=FakeMemory)
    program: object | None = None


FARM_LINES = [
    PPLine("int main(void) {", "main.c", 1),
    PPLine("    for (int i = 0; i < 8; i++) {", "main.c", 2),
    PPLine("        harvest_row(i);", "main.c", 3),
    PPLine("    }", "main.c", 4),
    PPLine("    return 0;", "main.c", 5),
    PPLine("void harvest_row(int y) {", "rows.c", 1),
    PPLine("    move(EAST);", "rows.c", 2),
]


def farm_interpreter(**extra: object) -> FakeInterpreter:
    program = SimpleNamespace(units=[SimpleNamespace(pp_lines=FARM_LINES)])
    return FakeInterpreter(program=program, **extra)


def at(file: str, line: int) -> Statement:
    return Statement(SourceLocation(file, line, 5))


def scripted_farm_run() -> SummaryBuilder:
    """main calls harvest_row 8 times; each harvest_row moves 8 times."""
    builder = SummaryBuilder(program_hash="abc123")
    builder.feed(CallEnter("main", SourceLocation("main.c", 1)))
    for row in range(8):
        builder.feed(at("main.c", 2))
        builder.feed(at("main.c", 3))
        builder.feed(CallEnter("harvest_row"))
        for _ in range(8):
            builder.feed(at("rows.c", 2))
            builder.feed(CallEnter("move"))
            builder.feed(DeviceCall("move", [1]))
            builder.feed(CallExit("move"))
        builder.feed(CallExit("harvest_row"))
        builder.feed(Output("stdout", f"row {row}\n"))
    builder.feed(at("main.c", 5))
    builder.feed(CallExit("main"))
    builder.feed(Exited(0))
    return builder


# =========================================================================== output and counts

def test_stdout_and_stderr_are_collected_separately() -> None:
    builder = SummaryBuilder()
    for event in (Output("stdout", "a"), Output("stderr", "oops\n"), Output("stdout", "b\n")):
        builder.feed(event)
    summary = builder.finish(0)
    assert summary.stdout == "ab\n" and summary.stderr == "oops\n"


def test_unrelated_events_are_ignored() -> None:
    builder = SummaryBuilder()
    for event in (InputRequest("> "), DeviceCall("move", [1]), Exited(0), "noise", None):
        builder.feed(event)
    summary = builder.finish(0)
    assert summary.stdout == "" and summary.statements == 0 and summary.warnings == []


def test_statements_come_from_the_interpreter_when_given() -> None:
    builder = scripted_farm_run()
    assert builder.finish(0).statements == 8 * 2 + 64 + 1
    assert builder.finish(0, FakeInterpreter(statements_executed=500)).statements == 500


def test_exit_code_ticks_and_hash() -> None:
    summary = scripted_farm_run().finish(3, farm_interpreter(), device_tick_delta=72)
    assert summary.exit_code == 3 and summary.ticks == 72
    assert summary.program_hash == "abc123"
    assert summary.trap is None and not summary.ok


def test_ok_run() -> None:
    summary = scripted_farm_run().finish(0, farm_interpreter())
    assert summary.ok and summary.stdout.count("row") == 8


# =========================================================================== warnings and traps

def test_run_warnings_are_collected_in_order() -> None:
    builder = SummaryBuilder()
    builder.feed(RunWarning(diag("bonk")))
    builder.feed(RunWarning(diag("signed_overflow")))
    builder.add_warnings([diag("unripe_harvest")])
    assert [w.code for w in builder.finish(0).warnings] == ["bonk", "signed_overflow",
                                                            "unripe_harvest"]


def test_trap_is_recorded() -> None:
    builder = SummaryBuilder()
    builder.feed(Trap(diag("null_dereference", Severity.ERROR)))
    summary = builder.finish(139)
    assert summary.trap.code == "null_dereference"
    assert summary.budget_hit is False and not summary.ok


def test_budget_hit() -> None:
    builder = SummaryBuilder()
    builder.feed(Trap(diag("instruction_budget", Severity.ERROR)))
    assert builder.finish(137).budget_hit is True


# =========================================================================== interpreter facts

def test_leaks_come_from_memory() -> None:
    summary = SummaryBuilder().finish(0, FakeInterpreter(memory=FakeMemory(leaked=3)))
    assert summary.leaks == 3


def test_constructs_come_from_the_interpreter() -> None:
    interpreter = FakeInterpreter(constructs=Counter({"loop_iteration": 8, "device_call": 64}))
    summary = SummaryBuilder().finish(0, interpreter)
    assert summary.constructs == {"loop_iteration": 8, "device_call": 64}
    assert isinstance(summary.constructs, dict) and not isinstance(summary.constructs, Counter)


def test_finish_without_an_interpreter() -> None:
    summary = SummaryBuilder().finish(None)
    assert summary.leaks == 0 and summary.constructs == {} and summary.exit_code is None


# =========================================================================== the call trace

def test_trace_summary_counts_calls_in_order_of_first_call() -> None:
    builder = SummaryBuilder()
    builder.feed(CallEnter("main"))
    for _ in range(3):
        builder.feed(CallEnter("harvest_row"))
        builder.feed(CallEnter("move"))
        builder.feed(CallExit("move"))
        builder.feed(CallEnter("move"))
        builder.feed(CallExit("move"))
        builder.feed(CallExit("harvest_row"))
    assert builder.finish(0).trace_summary == "main -> harvest_row x3 -> move x6"


def test_trace_summary_keeps_only_the_last_events() -> None:
    summary = scripted_farm_run().finish(0)
    # 64 events = the last 32 calls entered and left; main entered long before them.
    assert summary.trace_summary.startswith("... -> ")
    window = summary.trace_summary.removeprefix("... -> ")
    assert "main" not in window and "move x" in window


def test_trace_window_size() -> None:
    builder = SummaryBuilder()
    for index in range(TRACE_EVENTS):
        builder.feed(CallEnter(f"f{index}"))
    assert not builder.trace_summary().startswith("...")
    builder.feed(CallEnter("last"))
    assert builder.trace_summary().startswith("... -> f1 -> ")


def test_empty_trace() -> None:
    assert SummaryBuilder().finish(0).trace_summary == ""


# =========================================================================== last statements

def test_last_statements_show_file_line_and_text() -> None:
    summary = scripted_farm_run().finish(0, farm_interpreter())
    assert len(summary.last_statements) == LAST_STATEMENTS
    assert summary.last_statements[-1] == "main.c:5: return 0;"
    assert summary.last_statements[-2] == "rows.c:2: move(EAST);"


def test_last_statements_without_program_text() -> None:
    builder = SummaryBuilder()
    builder.feed(at("main.c", 9))
    assert builder.finish(0).last_statements == ["main.c:9:"]


def test_fewer_statements_than_the_window() -> None:
    builder = SummaryBuilder()
    builder.feed(at("main.c", 3))
    summary = builder.finish(0, farm_interpreter())
    assert summary.last_statements == ["main.c:3: harvest_row(i);"]
