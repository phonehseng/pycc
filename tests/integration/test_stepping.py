"""Scenario 10: stepping through a two-file program, as the debugger does.

The editor highlights the line of every ``Statement`` event in the tab of the file
that event names, so the file and line must be right when control crosses from
``src/main.c`` into ``src/worker.c`` and back. The program is built with ``cc`` in the
terminal; the test then loads the executable and drives the interpreter generator by
hand, answering device calls from a small ``Counter`` device, one event at a time.
"""
from __future__ import annotations

from typing import Any

import pytest

from pycc.core.events import CallEnter, CallExit, Exited, Statement, DeviceCall
from pycc.interp.compiler import link_executable, load_executable
from pycc.interp.hostio import ProgramIO
from pycc.interp.interpreter import Interpreter
from pycc.shell.headless import device_reply
from pycc.shell.hostfiles import VfsHostFiles
from pycc.vfs import standard_layout
from tests.integration.conftest import PROJECT, Counter, Terminal, counter_toolchain

MAIN_C = """\
#include <stdio.h>
#include "worker.h"

int main(void) {
    int got = 0;
    got = work(1);
    printf("%d\\n", got);
    return 0;
}
"""
WORKER_H = """\
#ifndef WORKER_H
#define WORKER_H
int work(int steps);
#endif
"""
WORKER_C = """\
#include <counter.h>
#include "worker.h"

int work(int steps) {
    int i;
    int total = 0;
    for (i = 0; i < steps; i++) {
        pulse(2);
        total += read_count();
    }
    return total;
}
"""


class Stepper:
    """Collects what a debugger would see while single-stepping a run."""

    def __init__(self, term: Terminal) -> None:
        executable = load_executable(term.read("bot"))
        linked = link_executable(executable, term.toolchain.builtins, "bot")
        assert linked.ok
        io = ProgramIO(files=VfsHostFiles(term.vfs), cwd=PROJECT, argv0="./bot")
        self.interpreter = Interpreter(linked.program, term.toolchain.builtins, io=io)
        self.device = term.device
        self.events: list[Any] = []
        self.frames_at: dict[tuple[str, int], list[str]] = {}

    def run(self) -> int:
        """Step until the program ends, one ``send`` per event; returns the exit code."""
        generator = self.interpreter.run(["./bot"])
        reply: Any = None
        while True:
            try:
                event = generator.send(reply)
            except StopIteration as stop:
                return int(stop.value)
            self.events.append(event)
            reply = self.answer(event)

    def answer(self, event: Any) -> Any:
        if isinstance(event, Statement):
            where = (event.location.file, event.location.line)
            self.frames_at.setdefault(where, [f.name for f in self.interpreter.stack_frames()])
        if isinstance(event, DeviceCall):
            return device_reply(self.device, event)
        return None

    @property
    def lines(self) -> list[tuple[str, int]]:
        return [(e.location.file, e.location.line) for e in self.events
                if isinstance(e, Statement)]


@pytest.fixture
def term() -> Terminal:
    return Terminal(standard_layout("project"), counter_toolchain(), Counter())


@pytest.fixture
def stepper(term: Terminal) -> Stepper:
    term.write({"src/main.c": MAIN_C, "src/worker.h": WORKER_H, "src/worker.c": WORKER_C})
    term.ok("cc -Wall src/main.c src/worker.c -o bot")
    return Stepper(term)


def test_statements_cross_files_with_the_right_lines(stepper: Stepper) -> None:
    assert stepper.run() == 0
    assert stepper.lines == [
        ("src/main.c", 5), ("src/main.c", 6),
        ("src/worker.c", 5), ("src/worker.c", 6), ("src/worker.c", 7),
        ("src/worker.c", 8), ("src/worker.c", 9),
        ("src/worker.c", 7),                 # the for condition is tested again: i < steps
        ("src/worker.c", 11),
        ("src/main.c", 7), ("src/main.c", 8),
    ]


def test_the_file_names_resolve_to_the_editor_tabs(term: Terminal, stepper: Stepper) -> None:
    stepper.run()
    tabs = {term.vfs.resolve(file, PROJECT) for file, _ in stepper.lines}
    assert tabs == {f"{PROJECT}/src/main.c", f"{PROJECT}/src/worker.c"}
    for path in tabs:
        assert term.vfs.is_file(path)


def test_depth_and_the_call_stack_follow_the_call(stepper: Stepper) -> None:
    stepper.run()
    depths = {(e.location.file, e.location.line): e.depth for e in stepper.events
              if isinstance(e, Statement)}
    assert depths[("src/worker.c", 8)] == depths[("src/main.c", 6)] + 1
    assert stepper.frames_at[("src/worker.c", 8)] == ["main", "work"]
    assert stepper.frames_at[("src/main.c", 7)] == ["main"]


def test_call_events_frame_the_visit_to_the_other_file(stepper: Stepper) -> None:
    stepper.run()
    kinds = [(type(e).__name__, getattr(e, "name", "")) for e in stepper.events
             if isinstance(e, (CallEnter, CallExit))]
    assert kinds == [("CallEnter", "main"), ("CallEnter", "work"), ("CallExit", "work"),
                     ("CallExit", "main")]
    enter = next(e for e in stepper.events if isinstance(e, CallEnter) and e.name == "work")
    assert (enter.location.file, enter.location.line) == ("src/main.c", 6)


def test_device_calls_pause_the_program_at_the_right_line(stepper: Stepper) -> None:
    stepper.run()
    last_line = None
    seen: list[tuple[str, tuple[str, int]]] = []
    for event in stepper.events:
        if isinstance(event, Statement):
            last_line = (event.location.file, event.location.line)
        elif isinstance(event, DeviceCall):
            seen.append((event.name, last_line))
    assert seen == [("pulse", ("src/worker.c", 8)), ("read_count", ("src/worker.c", 9))]


def test_the_device_changed_and_the_program_exited_cleanly(stepper: Stepper) -> None:
    assert stepper.run() == 0
    assert isinstance(stepper.events[-1], Exited) and stepper.events[-1].code == 0
    assert stepper.device.count == 2


def test_stopping_half_way_leaves_no_half_finished_device_call(term: Terminal,
                                                             stepper: Stepper) -> None:
    generator = stepper.interpreter.run(["./bot"])
    event = next(generator)
    while not isinstance(event, DeviceCall):
        event = generator.send(None)
    generator.close()                                   # the user pressed Stop
    assert term.device.log == []                          # pulse() was never performed
