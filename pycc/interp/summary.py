"""Turning the events of one program run into a ``RunSummary``.

Graders, editors and tutoring tools all want to know the same things about a run:
what it printed, how it ended, which warnings it raised and what it was doing just
before the end. ``SummaryBuilder`` is the one place that works this out. A host (the
headless runner, or a debugger that steps the program) feeds it every interpreter
event as it happens and calls ``finish`` once at the end.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any

from pycc.core.diagnostics import Diagnostic, SourceLocation
from pycc.core.events import CallEnter, CallExit, Output, RunSummary, RunWarning, Statement, Trap

TRACE_EVENTS = 64        # a tutoring tool sees the last 64 call events, like interpreter.call_trace
LAST_STATEMENTS = 10     # and the last 10 statements before the end
BUDGET_TRAP = "instruction_budget"


class SummaryBuilder:
    """Collects one run's events; ``finish`` returns the ``RunSummary``."""

    def __init__(self, program_hash: str = "") -> None:
        self.program_hash = program_hash
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.warnings: list[Diagnostic] = []
        self.trap: Diagnostic | None = None
        self.statement_count = 0
        self.calls: deque[CallEnter | CallExit] = deque(maxlen=TRACE_EVENTS)
        self.calls_seen = 0
        self.statements: deque[SourceLocation] = deque(maxlen=LAST_STATEMENTS)

    # --- feeding -----------------------------------------------------------------------
    def feed(self, event: Any) -> None:
        """Record one event; events that do not matter for the summary are ignored."""
        if isinstance(event, Statement):
            self.statement_count += 1
            self.statements.append(event.location)
        elif isinstance(event, (CallEnter, CallExit)):
            self.calls.append(event)
            self.calls_seen += 1
        elif isinstance(event, Output):
            (self.stderr if event.stream == "stderr" else self.stdout).append(event.text)
        elif isinstance(event, RunWarning):
            self.warnings.append(event.diagnostic)
        elif isinstance(event, Trap):
            self.trap = event.diagnostic

    def add_warnings(self, warnings: list[Diagnostic]) -> None:
        """Warnings that arrived outside the event stream (device action replies)."""
        self.warnings.extend(warnings)

    # --- finishing -----------------------------------------------------------------------
    def finish(self, exit_code: int | None, interpreter: Any = None,
               device_tick_delta: int = 0) -> RunSummary:
        """Build the summary. ``interpreter`` may be None (for example a failed link)."""
        constructs = getattr(interpreter, "constructs", None) or Counter()
        statements = getattr(interpreter, "statements_executed", None)
        return RunSummary(
            exit_code=exit_code,
            trap=self.trap,
            warnings=list(self.warnings),
            statements=self.statement_count if statements is None else int(statements),
            ticks=device_tick_delta,
            stdout="".join(self.stdout),
            stderr="".join(self.stderr),
            trace_summary=self.trace_summary(),
            budget_hit=self.trap is not None and self.trap.code == BUDGET_TRAP,
            leaks=_leak_count(interpreter),
            last_statements=self.last_statements(_program_lines(interpreter)),
            constructs={key: int(count) for key, count in dict(constructs).items()},
            program_hash=self.program_hash,
        )

    def trace_summary(self) -> str:
        """The recent calls as "main -> visit_row x8 -> move x64".

        Each function appears once, in the order it was first called within the window,
        with how many times it was entered; a window that lost older calls starts with
        "...". When the window holds only returns - a deep recursion unwinding, like
        ``fact`` returning 100 times - the returns are counted instead, so a tutoring tool
        still sees "... -> fact x64" rather than nothing.
        """
        counts: Counter[str] = Counter(e.name for e in self.calls if isinstance(e, CallEnter))
        if not counts:
            counts = Counter(e.name for e in self.calls)
        if not counts:
            return ""
        parts = [name if count == 1 else f"{name} x{count}" for name, count in counts.items()]
        prefix = "... -> " if self.calls_seen > len(self.calls) else ""
        return prefix + " -> ".join(parts)

    def last_statements(self, lines: dict[tuple[str, int], str] | None = None) -> list[str]:
        """The last statements as "file:line: <line text>" (the text when known)."""
        lines = lines or {}
        rendered = []
        for loc in self.statements:
            text = lines.get((loc.file, loc.line), "")
            rendered.append(f"{loc.file}:{loc.line}: {text}".rstrip())
        return rendered


def _leak_count(interpreter: Any) -> int:
    memory = getattr(interpreter, "memory", None)
    leaks = getattr(memory, "leaks", None)
    return len(leaks()) if callable(leaks) else 0


def _program_lines(interpreter: Any) -> dict[tuple[str, int], str]:
    """(file, line) -> source text, from the preprocessed lines of every unit.

    The first preprocessed line for a position wins: that is the text the user
    wrote, before any continuation lines were joined onto it.
    """
    program = getattr(interpreter, "program", None)
    lines: dict[tuple[str, int], str] = {}
    for unit in getattr(program, "units", None) or []:
        for line in getattr(unit, "pp_lines", None) or []:
            lines.setdefault((line.file, line.line), line.text.strip())
    return lines
