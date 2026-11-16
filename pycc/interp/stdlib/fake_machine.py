"""A small stand-in for the interpreter, so the standard library is tested for real.

``FakeMachine`` implements the ``Machine`` protocol of ``interp.builtins`` over a real
``interp.memory.Memory`` and a real ``ProgramIO`` (usually with ``DictHostFiles``).
It lays out the library variables (``stdin``, ``errno``...) the way the interpreter
does, and ``call(name, *args)`` runs a library function exactly as a C call would:
it drives generator implementations, collects ``Output`` text, answers
``InputRequest`` from a queue of lines and records every other event.

Tests also get helpers to put strings and buffers into memory and to build argument
values, so a test reads like the C it stands for::

    m = FakeMachine(stdin="42\\n")
    n = m.buffer(4)
    m.call("scanf", m.literal("%d"), m.ptr(n))
    assert m.memory.load(n, INT) == 42
"""
from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from pycc.core.concepts import Concept
from pycc.core.diagnostics import SourceLocation, concept_for_code
from pycc.core.events import InputRequest, Output
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.c_types import CHAR, DOUBLE, INT, LONG, SIZE_T, UINT, CType, PointerType, decode
from pycc.interp.hostio import DictHostFiles, HostFiles, ProgramIO
from pycc.interp.memory import Memory
from pycc.interp.traps import PendingWarning, TrapError
from pycc.interp.values import Value, make_double, make_int, make_pointer

Callback = Callable[["FakeMachine", list[Value]], Any]


class FakeProgramExit(Exception):
    """What ``exit()`` raises here (the interpreter raises its own ``ProgramExit``)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"exit({code})")
        self.code = code


class FakeMachine:
    """The ``Machine`` facade without an interpreter. See the module docstring."""

    def __init__(self, *, files: HostFiles | dict[str, bytes] | None = None, cwd: str = "/",
                 env: dict[str, str] | None = None, argv0: str = "a.out",
                 stdin: str | Iterable[str | None] | None = None,
                 registry: BuiltinRegistry | None = None, memory: Memory | None = None,
                 device_tick: int = 0, no_files: bool = False) -> None:
        """Build a machine. ``stdin`` is either one string (split into lines) or the
        exact replies to send, where ``None`` means end of file. ``no_files`` gives
        the program no filesystem at all (``ProgramIO.files is None``)."""
        if registry is None:
            from pycc.interp.stdlib import make_registry  # lazy: avoids a cycle at import

            registry = make_registry()
        self.registry = registry
        self.memory = memory if memory is not None else Memory()
        if isinstance(files, dict) or (files is None and not no_files):
            files = DictHostFiles(files or {})
        self.files = files
        self.io = ProgramIO(files, cwd, argv0, dict(env or {}))
        self.state: dict[str, Any] = {}
        self.statements_executed = 0
        self.device_tick = device_tick
        self.current_location: SourceLocation | None = SourceLocation("main.c", 1, 1)
        self.constructs: Counter[str] = Counter()
        self.events: list[Any] = []
        self.prompts: list[str] = []
        self.stdout = ""
        self.stderr = ""
        self._replies = self._initial_replies(stdin)
        self._callbacks: dict[int, Callback] = {}
        self.globals: dict[str, int] = {}
        self._lay_out_variables()

    @staticmethod
    def _initial_replies(stdin: str | Iterable[str | None] | None) -> list[str | None]:
        if stdin is None:
            return []
        if isinstance(stdin, str):
            return list(stdin.splitlines(keepends=True))
        return list(stdin)

    def _lay_out_variables(self) -> None:
        """Place every library variable in the data segment, as the linker would."""
        for name, variable in self.registry.variables.items():
            data = variable.init(self)
            addr = self.memory.alloc_global(len(data), min(len(data), 8) or 1, name)
            self.memory.write_bytes(addr, data)
            self.globals[name] = addr

    # ================================================================ Machine protocol

    def read_cstring(self, addr: int, limit: int = 4096) -> str:
        """The C string at ``addr`` as text."""
        return self.memory.read_cstring(addr, limit).decode("utf-8", "replace")

    def write_cstring(self, addr: int, text: str, limit: int | None = None) -> int:
        """Store ``text`` plus ``'\\0'`` (at most ``limit`` bytes in all); bytes written."""
        data = text.encode("utf-8")
        if limit is not None:
            data = data[:max(limit - 1, 0)]
        self.memory.write_bytes(addr, data + b"\0")
        return len(data) + 1

    def read_bytes(self, addr: int, count: int) -> bytes:
        """Checked raw read through the real memory model."""
        return self.memory.read_bytes(addr, count)

    def write_bytes(self, addr: int, data: bytes) -> None:
        """Checked raw write through the real memory model."""
        self.memory.write_bytes(addr, data)

    def malloc(self, size: int) -> int:
        """Heap allocation (0 when the heap is full), as the interpreter does it."""
        return self.memory.malloc(size)

    def free(self, addr: int) -> None:
        """Give a heap block back; bad frees trap exactly as in a real run."""
        self.memory.free(addr)

    def make_int(self, value: int, ctype: CType = INT) -> Value:
        """An integer value of ``ctype``."""
        return make_int(value, ctype)

    def make_double(self, value: float) -> Value:
        """A ``double`` value."""
        return make_double(value)

    def make_pointer(self, addr: int, target: CType) -> Value:
        """A pointer value to ``target``."""
        return make_pointer(addr, target)

    def get_errno(self) -> int:
        """The current value of the program's ``errno`` variable."""
        return int(self.memory.load(self.globals["errno"], INT))

    def set_errno(self, code: int) -> None:
        """Store ``code`` in the program's ``errno`` variable."""
        self.memory.store(self.globals["errno"], INT, code)

    def exit(self, code: int) -> None:
        """End the program: raises ``FakeProgramExit``."""
        raise FakeProgramExit(code)

    def trap(self, code: str, message: str, friendly: str, concept: Concept | None = None,
             **extra: Any) -> None:
        """Stop the program with a runtime error: raises ``TrapError``."""
        raise TrapError(code, message, friendly, concept, extra)

    def warn(self, code: str, message: str, friendly: str, concept: Concept | None = None) -> None:
        """Queue a non-fatal warning, as the interpreter's memory queue does.

        Without an explicit concept the code's registered one is used, so a test sees
        the same concept the user would.
        """
        concept = concept or concept_for_code(code, Concept.STDIO)
        self.memory.pending_warnings.append(PendingWarning(code, message, friendly, concept))

    def call_function(self, fid: int, args: list[Value]) -> Any:
        """Call a Python stand-in for a C function (see ``add_callback``)."""
        callback = self._callbacks.get(fid)
        if callback is None:
            raise TrapError("bad_function_pointer", f"call through bad function pointer {fid}")
        result = callback(self, args)
        if inspect.isgenerator(result):
            result = yield from result
        return result if isinstance(result, Value) else make_int(int(result))

    def count_construct(self, kind: str, amount: int = 1) -> None:
        """Record that the program used construct ``kind``."""
        self.constructs[kind] += amount

    # ================================================================ running builtins

    def call(self, name: str, *args: Value) -> Value | None:
        """Call library function ``name`` with argument values and return its result."""
        spec = self.registry.get(name)
        if spec is None:
            raise KeyError(name)
        result = spec.impl(self, list(args))
        if inspect.isgenerator(result):
            return self.drive(result)
        return result

    def drive(self, generator: Any) -> Any:
        """Run a generator implementation to completion, acting as the host."""
        reply: Any = None
        while True:
            try:
                event = generator.send(reply)
            except StopIteration as done:
                return done.value
            reply = self._handle(event)

    def _handle(self, event: Any) -> Any:
        self.events.append(event)
        if isinstance(event, Output):
            if event.stream == "stdout":
                self.stdout += event.text
            else:
                self.stderr += event.text
            return None
        if isinstance(event, InputRequest):
            self.prompts.append(event.prompt)
            return self._replies.pop(0) if self._replies else None
        return None

    def feed(self, *lines: str | None) -> None:
        """Queue more replies for future ``InputRequest`` events."""
        self._replies.extend(lines)

    def add_callback(self, function: Callback) -> int:
        """Register a Python function as a C function; returns its function id."""
        fid = len(self._callbacks) + 1
        self._callbacks[fid] = function
        return fid

    @property
    def warnings(self) -> list[PendingWarning]:
        """Warnings queued so far (not drained)."""
        return list(self.memory.pending_warnings)

    def warning_codes(self) -> list[str]:
        """The codes of the warnings queued so far, in order."""
        return [warning.code for warning in self.memory.pending_warnings]

    @property
    def errno(self) -> int:
        """The program's ``errno`` (for assertions)."""
        return self.get_errno()

    # ================================================================ memory helpers

    def literal(self, text: str | bytes) -> int:
        """A string literal in read-only data; returns its address."""
        data = text.encode("utf-8") if isinstance(text, str) else text
        return self.memory.intern_string(data)

    def string(self, text: str | bytes, size: int | None = None) -> int:
        """A writable ``char[size]`` holding ``text`` (default size: just enough)."""
        data = text.encode("utf-8") if isinstance(text, str) else text
        size = len(data) + 1 if size is None else size
        addr = self.memory.alloc_global(size, 1, f"buf{len(self.memory.objects())}")
        self.memory.write_bytes(addr, data + b"\0")
        return addr

    def buffer(self, size: int, fill: bytes = b"") -> int:
        """A zeroed writable buffer of ``size`` bytes (optionally pre-filled)."""
        addr = self.memory.alloc_global(size, 8, f"buf{len(self.memory.objects())}")
        if fill:
            self.memory.write_bytes(addr, (fill * size)[:size])
        return addr

    def text_at(self, addr: int) -> str:
        """The C string at ``addr`` (for assertions)."""
        return self.read_cstring(addr, self.memory.size)

    def stream(self, name: str) -> Value:
        """The value of ``stdin``/``stdout``/``stderr`` as the program would see it."""
        addr = decode(self.memory.read_bytes(self.globals[name], 8), PointerType(CHAR))
        return make_pointer(int(addr), CHAR)

    # ================================================================ value helpers

    @staticmethod
    def int_(value: int) -> Value:
        """An ``int`` argument value."""
        return make_int(value, INT)

    @staticmethod
    def uint(value: int) -> Value:
        """An ``unsigned int`` argument value."""
        return make_int(value, UINT)

    @staticmethod
    def long_(value: int) -> Value:
        """A ``long`` argument value."""
        return make_int(value, LONG)

    @staticmethod
    def size(value: int) -> Value:
        """A ``size_t`` argument value."""
        return make_int(value, SIZE_T)

    @staticmethod
    def double(value: float) -> Value:
        """A ``double`` argument value."""
        return Value(DOUBLE, float(value))

    @staticmethod
    def ptr(addr: int, target: CType = CHAR) -> Value:
        """A pointer argument value (``char *`` unless ``target`` says otherwise)."""
        return make_pointer(addr, target)

    @staticmethod
    def null() -> Value:
        """A NULL ``char *`` argument value."""
        return make_pointer(0, CHAR)
