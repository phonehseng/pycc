"""``<time.h>``: time measured in deterministic, simulated units.

A program must behave the same on every run, so there is no wall clock: ``time()``
returns the device tick (0 when no device is attached) and ``clock()`` the number of
statements executed so far. ``clock() / CLOCKS_PER_SEC`` therefore reads as "seconds"
of a machine that runs one million statements per second.
"""
from __future__ import annotations

from pycc.core.concepts import Concept
from pycc.interp.builtins import BuiltinRegistry, Machine
from pycc.interp.c_types import LONG
from pycc.interp.stdlib.headers import LibraryFunction, register_functions
from pycc.interp.values import Value

HEADER = "time.h"


def c_time(machine: Machine, args: list[Value]) -> Value:
    """``time_t time(time_t *tloc)``: the device tick, also stored at ``tloc`` if given."""
    now = int(machine.device_tick)
    if args[0].raw != 0:
        machine.memory.store(args[0].raw, LONG, now)
    return machine.make_int(now, LONG)


def c_clock(machine: Machine, args: list[Value]) -> Value:
    """``clock_t clock(void)``: statements executed since the program started."""
    return machine.make_int(int(machine.statements_executed), LONG)


FUNCTIONS: tuple[LibraryFunction, ...] = (
    LibraryFunction("time", "time_t time(time_t *tloc);", c_time, Concept.FUNCTIONS),
    LibraryFunction("clock", "clock_t clock(void);", c_clock, Concept.FUNCTIONS),
)


def register(registry: BuiltinRegistry) -> None:
    """Register every ``<time.h>`` function."""
    register_functions(registry, HEADER, FUNCTIONS)
