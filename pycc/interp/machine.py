"""The concrete ``Machine``: what a library function sees of a running program.

``interp.builtins.Machine`` is the facade every builtin (``printf``, ``malloc``,
``move``...) talks to. The standard library is tested against a
small fake; this module is the real thing the interpreter hands out. It owns nothing
interesting itself: memory, the clock and the call machinery all belong to the
interpreter, and the facade only forwards, so a builtin can never reach further into
the interpreter than the protocol allows.

Two exceptions leave a builtin and stop the program:

* ``ProgramExit(code)`` from ``exit()``: a clean end with an exit status;
* ``TrapError`` from ``trap()``: a crash, turned into a ``Trap`` event by the
  interpreter with the stack still intact.

Warnings never stop anything: ``warn()`` queues them in ``memory.pending_warnings``
and the interpreter reports them before the next statement.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, Protocol

from pycc.core.concepts import Concept
from pycc.core.diagnostics import SourceLocation, concept_for_code
from pycc.interp.c_types import INT, CType
from pycc.interp.hostio import ProgramIO
from pycc.interp.memory import Memory
from pycc.interp.traps import PendingWarning, TrapError
from pycc.interp.values import Value, make_double, make_int, make_pointer
from pycc.core.rng import SeededRandom

# Where errno lives when no library registered an ``errno`` variable (a bare
# registry in a test): the value is kept in Python instead of program memory.
_ERRNO_FALLBACK_KEY = "errno"


class ProgramExit(Exception):
    """Raised by ``exit()``: the program ends now with ``code`` as its exit status."""

    def __init__(self, code: int) -> None:
        super().__init__(f"exit({code})")
        self.code = int(code)


class MachineHost(Protocol):
    """What the facade needs from the interpreter (kept small on purpose)."""

    memory: Memory
    statements_executed: int
    current_location: SourceLocation | None

    def count_construct(self, kind: str, amount: int = 1) -> None:
        """Count a construct that ran."""

    def builtin_variable_address(self, name: str) -> int | None:
        """Address of a library variable such as ``errno``, if installed."""

    def call_function_id(self, fid: int, args: list[Value]) -> Generator[Any, Any, Any]:
        """Call a function by pointer id (a generator returning its value)."""

    def first_report(self, code: str, location: SourceLocation | None) -> bool:
        """True the first time warning ``code`` is reported at ``location`` this run."""


class InterpreterMachine:
    """The ``Machine`` protocol implemented over a running interpreter.

    ``state`` is per-run scratch space for libraries (stdio keeps its FILE table and
    stdin buffer there). ``device_tick`` is set by the host that owns a device, so
    ``time()`` can report it; it stays 0 headless.
    """

    def __init__(self, host: MachineHost, io: ProgramIO | None = None, seed: int = 0) -> None:
        self._host = host
        self.io = io if io is not None else ProgramIO()
        self.state: dict[str, Any] = {}
        self.device_tick = 0
        self.rng = SeededRandom(seed).child("machine")

    # --- live views of the interpreter ------------------------------------------------
    @property
    def memory(self) -> Memory:
        """The program's simulated memory."""
        return self._host.memory

    @property
    def statements_executed(self) -> int:
        """How many statements the program has run so far (``clock()`` reads this)."""
        return self._host.statements_executed

    @property
    def current_location(self) -> SourceLocation | None:
        """The statement being executed when the builtin was called."""
        return self._host.current_location

    # --- strings and bytes ----------------------------------------------------------------
    def read_cstring(self, addr: int, limit: int = 4096) -> str:
        """The NUL-terminated string at ``addr`` as text (bad pointers trap)."""
        return self.memory.read_cstring(addr, limit).decode("utf-8", "replace")

    def write_cstring(self, addr: int, text: str, limit: int | None = None) -> int:
        """Store ``text`` plus ``'\\0'`` (at most ``limit`` bytes in all); bytes written."""
        data = text.encode("utf-8")
        if limit is not None:
            if limit <= 0:
                return 0        # like snprintf(buf, 0, ...): no room, not even for '\0'
            data = data[:limit - 1]
        self.memory.write_bytes(addr, data + b"\0")
        return len(data) + 1

    def read_bytes(self, addr: int, count: int) -> bytes:
        """``count`` bytes from ``addr`` (checked like any C access)."""
        return self.memory.read_bytes(addr, count)

    def write_bytes(self, addr: int, data: bytes) -> None:
        """Store ``data`` at ``addr`` (checked like any C access)."""
        self.memory.write_bytes(addr, data)

    # --- heap ---------------------------------------------------------------------------------
    def malloc(self, size: int) -> int:
        """A new heap block, or 0 when the heap is full."""
        return self.memory.malloc(size)

    def free(self, addr: int) -> None:
        """Give a heap block back (bad frees trap)."""
        self.memory.free(addr)

    # --- values ---------------------------------------------------------------------------------
    def make_int(self, value: int, ctype: CType = INT) -> Value:
        """An integer value wrapped to ``ctype``."""
        return make_int(value, ctype)

    def make_double(self, value: float) -> Value:
        """A ``double`` value."""
        return make_double(value)

    def make_pointer(self, addr: int, target: CType) -> Value:
        """A pointer to ``target`` holding ``addr``."""
        return make_pointer(addr, target)

    # --- errno ----------------------------------------------------------------------------------
    def get_errno(self) -> int:
        """The program's ``errno`` (read from its real address in the data segment)."""
        addr = self._host.builtin_variable_address("errno")
        if addr is None:
            return int(self.state.get(_ERRNO_FALLBACK_KEY, 0))
        return int(self.memory.load(addr, INT))

    def set_errno(self, code: int) -> None:
        """Set ``errno``; the program sees it exactly as C would."""
        addr = self._host.builtin_variable_address("errno")
        if addr is None:
            self.state[_ERRNO_FALLBACK_KEY] = int(code)
            return
        self.memory.store(addr, INT, code)

    # --- ending, crashing, warning ----------------------------------------------------------------
    def exit(self, code: int) -> None:
        """End the program now (``exit(code)``)."""
        raise ProgramExit(code)

    def trap(self, code: str, message: str, friendly: str, concept: Concept | None = None,
             **extra: Any) -> None:
        """Stop the program with a runtime trap (the interpreter yields ``Trap``)."""
        raise TrapError(code, message, friendly, concept, extra)

    def warn(self, code: str, message: str, friendly: str, concept: Concept | None = None) -> None:
        """Queue a runtime warning; it is reported before the next statement."""
        self.memory.pending_warnings.append(PendingWarning(
            code, message, friendly, concept or concept_for_code(code, Concept.STDIO)))

    def first_report(self, code: str, location: SourceLocation | None) -> bool:
        """True if a warning the device sent back should be shown.

        Queued warnings are filtered by the interpreter itself; warnings that come back
        in a device call's reply pass through here, so both follow one rule: each
        (code, place) once per run.
        """
        return self._host.first_report(code, location)

    # --- calling back into C ----------------------------------------------------------------------
    def call_function(self, fid: int, args: list[Value]) -> Generator[Any, Any, Any]:
        """Call the C function with pointer id ``fid`` (``qsort`` comparators).

        A generator: the callee's statements and events flow through the builtin
        that called it, so a ``printf`` inside a comparator appears in order.
        """
        return (yield from self._host.call_function_id(fid, list(args)))

    def count_construct(self, kind: str, amount: int = 1) -> None:
        """Count a construct for the run summary (keys in interp/constructs.py)."""
        self._host.count_construct(kind, amount)
