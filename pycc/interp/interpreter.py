"""The steppable C interpreter.

The interpreter runs a linked, checked program by walking its AST. It never decides
what a name or a type means: the checker already wrote that on every node
(``expr.ctype``, ``Identifier.ref``, ``ImplicitCast``), and the interpreter trusts it.
What the interpreter adds is *time*: values, memory, calls and loops.

Everything is a generator. ``run(argv)`` yields events (``Statement`` before each
statement, ``CallEnter``/``CallExit``, ``Output`` and ``InputRequest`` from the
library, ``DeviceCall`` from device functions, ``RunWarning``) and finally exactly
one ``Exited`` or ``Trap``; its return value is the exit code. Because every
``exec_*`` and ``eval_*`` method is a generator too (``yield from`` all the way
down), a device call deep inside an expression pauses the whole program until the
host has performed it and sent back the result.

How the C semantics are kept honest:

* integers live as Python ints wrapped to their C width after every operation; a
  signed operation that wraps queues a ``signed_overflow`` warning;
* every load and store goes through ``interp.memory.Memory``, which checks that the
  address lies inside one live object, so out-of-bounds, NULL, dangling and
  read-only accesses stop at the statement that made them;
* locals are allocated on the simulated stack when their declaration runs and die
  when their block ends; statics and globals live in the data segment;
* structs move by value as bytes (``StructValue``); arrays decay to pointers only
  where the checker inserted the conversion.

A trap anywhere (memory, arithmetic, a library function) becomes one ``Trap`` event
whose location is the statement that was running, with the call stack still intact
for the Variables panel. ``exit()`` raises ``ProgramExit`` and ends cleanly.
"""
from __future__ import annotations

import inspect
import math
import operator
import sys
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterable
from typing import Any, NamedTuple

from pycc.core.diagnostics import (
    TRAP_EXIT_CODES,
    Diagnostic,
    Phase,
    Severity,
    SourceLocation,
    concept_for_code,
    exit_code_for_trap,
)
from pycc.core.events import (
    CallEnter,
    CallExit,
    Event,
    Exited,
    InputRequest,
    RunSummary,
    RunWarning,
    Statement,
    StringSource,
    Trap,
    DeviceCall,
)
from pycc.interp import astnodes as ast
from pycc.interp import constructs, stdlib
from pycc.interp.builtins import BuiltinRegistry, BuiltinSpec
from pycc.interp.c_types import (
    CHAR,
    DOUBLE,
    INT,
    LONG,
    SIZE_T,
    ArrayType,
    CType,
    FunctionType,
    IntType,
    PointerType,
    StructType,
    convert_value,
    decode,
    format_type,
    integer_promote,
    usual_arithmetic_conversions,
    wrap_int,
)
from pycc.interp.call import (
    BREAK,
    CONTINUE,
    CallFrame,
    Flow,
    FrameInfo,
    Place,
    Returned,
    Slot,
    SwitchTable,
    VarInfo,
    build_switch_table,
    index_of,
    is_lvalue,
    strip_implicit,
)
from pycc.interp.hostio import ProgramIO
from pycc.interp.linker import LinkedProgram, program_hash
from pycc.interp.machine import InterpreterMachine, ProgramExit
from pycc.interp.memory import Memory, MemoryObject
from pycc.interp.summary import SummaryBuilder
from pycc.interp.traps import PendingWarning, TrapError
from pycc.interp.values import StructValue, Value, truthy

__all__ = [
    "CALL_TRACE_LIMIT", "CompletedRun", "FrameInfo", "Interpreter", "LEAK_WARNING_LIMIT",
    "MAX_CALL_DEPTH", "ProgramExit", "TRAP_EXIT_CODES", "UnexpectedDeviceCall", "VarInfo",
    "run_to_completion",
]

MAX_CALL_DEPTH = 512          # nested calls before stack_overflow
CALL_TRACE_LIMIT = 64         # call events kept for the trace summary
LEAK_WARNING_LIMIT = 10       # memory_leak warnings at exit (docs D5)
POINTER_MASK = (1 << 64) - 1  # pointers are 64-bit addresses

# Each C call costs a few dozen Python frames (statement, expression and call
# generators delegating to each other). Python's default limit of 1000 would stop a
# recursive C program long before our own 512-frame cap, so the limit is raised to
# leave room for the full C stack plus deep expressions.
_PYTHON_FRAMES_PER_CALL = 64
_RECURSION_LIMIT = MAX_CALL_DEPTH * _PYTHON_FRAMES_PER_CALL + 10_000

_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
}

# Operators whose exact result Python computes directly (a table beats a chain of ifs
# on the hot path of every integer expression).
_EXACT_INT_OPERATORS: dict[str, Callable[[int, int], int]] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "&": operator.and_, "|": operator.or_, "^": operator.xor,
}

# The two results every comparison and logical operator produces (Values are
# immutable, so sharing them is safe and saves an allocation per test).
_FALSE = Value(INT, 0)
_TRUE = Value(INT, 1)

_BUDGET_FRIENDLY = (
    "The program ran for too long without finishing, so it was stopped. "
    "A loop probably never reaches its exit condition: check that the loop variable "
    "changes on every round and that the condition can become false.")


# =========================================================================== pure arithmetic

def trunc_div(a: int, b: int) -> int:
    """Integer division that truncates toward zero, as C does (-7 / 2 == -3)."""
    quotient = abs(a) // abs(b)
    return quotient if (a >= 0) == (b >= 0) else -quotient


def c_mod(a: int, b: int) -> int:
    """C's remainder: it takes the sign of the dividend (-7 % 2 == -1)."""
    return a - b * trunc_div(a, b)


def exact_int_result(op: str, a: int, b: int) -> int:
    """The mathematically exact result of an integer operator (before wrapping).

    Raises ``ZeroDivisionError`` for ``/`` and ``%`` by zero; the caller turns that
    into the ``division_by_zero`` trap.
    """
    simple = _EXACT_INT_OPERATORS.get(op)
    if simple is not None:
        return simple(a, b)
    if op in ("/", "%"):
        if b == 0:
            raise ZeroDivisionError
        return trunc_div(a, b) if op == "/" else c_mod(a, b)
    raise ValueError(f"not an integer operator: {op}")


def float_result(op: str, a: float, b: float) -> float:
    """IEEE arithmetic: dividing by zero gives inf or nan, never an error."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0.0:
            if a == 0.0 or math.isnan(a):
                return math.nan
            return math.copysign(math.inf, a) * math.copysign(1.0, b)
        return a / b
    raise ValueError(f"not a floating operator: {op}")


def pointer_step(ptype: CType) -> int:
    """Bytes one step of a pointer moves: the target's size (void and functions: 1)."""
    target = ptype.target if isinstance(ptype, PointerType) else None
    if target is None or target.is_void or target.is_function or target.size <= 0:
        return 1
    return target.size


def _is_char(ctype: CType) -> bool:
    """Is ``ctype`` one of the three char types (the ones a string can initialise)?"""
    return isinstance(ctype, IntType) and ctype.size == 1


class _InitCursor:
    """Walks the items of a brace initializer while brace elision consumes them.

    Mirrors the checker's cursor exactly, so ``{1, 2, 3, 4}`` for ``int[2][2]`` fills
    the same elements here as the checker type-checked.
    """

    def __init__(self, items: list[ast.Expr]) -> None:
        self.items = items
        self.pos = 0

    @property
    def done(self) -> bool:
        """True when every item has been used."""
        return self.pos >= len(self.items)

    def take(self) -> ast.Expr:
        """The next item; the cursor moves past it."""
        item = self.items[self.pos]
        self.pos += 1
        return item


# =========================================================================== the interpreter

Gen = Generator[Event, Any, Any]


class Interpreter:
    """Runs one ``LinkedProgram``. Create one per run and call ``run(argv)`` once."""

    def __init__(self, program: LinkedProgram, builtins: BuiltinRegistry,
                 memory: Memory | None = None, budget: int = 1_000_000, seed: int = 0,
                 io: ProgramIO | None = None) -> None:
        self.program = program
        self.builtins = builtins
        self.memory = memory if memory is not None else Memory(seed=seed)
        self.budget = budget
        self.seed = seed
        self.io = io if io is not None else ProgramIO()
        self.machine = InterpreterMachine(self, self.io, seed)
        self.statements_executed = 0
        self.current_location: SourceLocation | None = None
        self.constructs: Counter[str] = Counter()
        self.call_trace: deque[CallEnter | CallExit] = deque(maxlen=CALL_TRACE_LIMIT)
        self._frames: list[CallFrame] = []
        self._global_addrs: dict[str, int] = {}
        self._global_info: list[VarInfo] = []
        self._builtin_vars: dict[str, int] = {}
        self._switch_tables: dict[int, SwitchTable] = {}
        self._block_declares: dict[int, bool] = {}
        self._constants: dict[int, Value] = {}
        self._reported: set[tuple[str, SourceLocation | None]] = set()
        self._alloc_sites: dict[int, SourceLocation | None] = {}
        self._started = False

    # ================================================================ inspection
    def stack_frames(self) -> list[FrameInfo]:
        """The active calls, outermost (main) first, with their visible locals."""
        return [frame.info() for frame in self._frames]

    def globals(self) -> list[VarInfo]:
        """Every global, file-static and static local variable with its address."""
        return list(self._global_info)

    @property
    def depth(self) -> int:
        """Call depth of the running statement (main is 0)."""
        return len(self._frames) - 1 if self._frames else 0

    @property
    def device_tick(self) -> int:
        """The device's clock as the host last set it (``time()`` reads it)."""
        return self.machine.device_tick

    @device_tick.setter
    def device_tick(self, tick: int) -> None:
        """Set by the host that owns a device, before or during the run."""
        self.machine.device_tick = tick

    def builtin_variable_address(self, name: str) -> int | None:
        """Address of a library variable (``errno``, ``stdin``...), if installed."""
        return self._builtin_vars.get(name)

    def count_construct(self, kind: str, amount: int = 1) -> None:
        """Count a construct that ran (keys from interp/constructs.py)."""
        self.constructs[kind] += amount

    # ================================================================ running
    def run(self, argv: list[str] | None = None) -> Generator[Event, Any, int]:
        """Run the program; yields events and returns the exit code (docs D5)."""
        if self._started:
            raise RuntimeError("an Interpreter runs its program only once")
        self._started = True
        if sys.getrecursionlimit() < _RECURSION_LIMIT:
            sys.setrecursionlimit(_RECURSION_LIMIT)
        try:
            yield from self._lay_out_data()
            code = yield from self._run_main(list(argv or [self.io.argv0]))
            # Inside the try: a library check at exit that traps still ends the run
            # with exactly one terminal event instead of an exception in the host.
            yield from self._finish()
        except TrapError as trap:
            return (yield from self._report_trap(trap))
        except RecursionError:
            return (yield from self._report_trap(self._python_stack_trap()))
        status = code & 0xFF
        yield Exited(status)
        return status

    def _run_main(self, argv: list[str]) -> Generator[Event, Any, int]:
        """main's return value, or the status given to ``exit()`` wherever it was called."""
        try:
            return (yield from self._call_main(argv))
        except ProgramExit as stop:
            # exit() unwinds every call at once, as the real process would end.
            self._frames.clear()
            return stop.code

    def _report_trap(self, trap: TrapError) -> Generator[Event, Any, int]:
        """Yield pending warnings, then the one ``Trap``; returns its exit code."""
        yield from self._drain_warnings()
        diagnostic = trap.to_diagnostic(self.current_location)
        yield Trap(diagnostic)
        return exit_code_for_trap(trap.code)

    def _python_stack_trap(self) -> TrapError:
        """The trap for a program nested too deeply for the host's own stack."""
        return TrapError(
            "stack_overflow",
            f"stack overflow: calls and expressions nested {len(self._frames)} deep "
            "exhausted the machine's stack",
            "The program nested so many calls (or one giant expression) that the stack ran "
            "out. Recursion needs a base case that is always reached.")

    # --- start-up ---------------------------------------------------------------
    def _lay_out_data(self) -> Gen:
        """Library variables, globals, statics and string literals, then initialisers."""
        for name, variable in self.builtins.variables.items():
            data = variable.init(self.machine)
            addr = self.memory.alloc_global(len(data), min(len(data), 8) or 1, name)
            self.memory.write_bytes(addr, data)
            self._builtin_vars[name] = addr
        for key, decl in self.program.globals.items():
            ctype = decl.ctype
            addr = self.memory.alloc_global(ctype.size, ctype.align or 1, decl.name, ctype)
            self._global_addrs[key] = addr
            self._global_info.append(VarInfo(decl.name, ctype, addr))
        self._intern_literals()
        for key, decl in self.program.globals.items():
            if decl.init is not None:
                self.current_location = decl.loc
                yield from self._init_object(self._global_addrs[key], decl.ctype, decl.init)

    def _intern_literals(self) -> None:
        """Place every string literal in read-only data before main starts."""
        for unit in self.program.units:
            if unit.ast is None:
                continue
            for node in ast.walk(unit.ast):
                if isinstance(node, ast.StringLiteral):
                    self.memory.intern_string(node.value)

    def _call_main(self, argv: list[str]) -> Gen:
        """Call main with (), (argc, argv) or (argc, argv, envp); returns its status."""
        main = self.program.entry_function
        params = main.ftype.params
        args: list[Value] = []
        if len(params) >= 2:
            args = [Value(INT, len(argv)), Value(params[1], self._string_vector("argv", argv))]
        if len(params) >= 3:
            env = [f"{name}={value}" for name, value in self.io.env.items()]
            args.append(Value(params[2], self._string_vector("envp", env)))
        result = yield from self._call_user(main, args, main.loc, count=False)
        if not isinstance(result, Value):
            return 0          # 'void main' (warned at compile time) still ends with 0
        return int(result.raw)

    def _string_vector(self, name: str, texts: list[str]) -> int:
        """A NULL-terminated ``char *[]`` of writable strings, like argv (argv[argc] == NULL)."""
        pointers = []
        for index, text in enumerate(texts):
            data = text.encode("utf-8") + b"\0"
            addr = self.memory.alloc_global(len(data), 1, f"{name}[{index}]",
                                            ArrayType(CHAR, len(data)))
            self.memory.write_bytes(addr, data)
            pointers.append(addr)
        vector_type = ArrayType(PointerType(CHAR), len(texts) + 1)
        vector = self.memory.alloc_global(vector_type.size, 8, name, vector_type)
        for index, pointer in enumerate(pointers + [0]):
            self.memory.store(vector + index * 8, PointerType(CHAR), pointer)
        return vector

    # --- the end ---------------------------------------------------------------------
    def _finish(self) -> Gen:
        """Normal end: pending warnings, the library's checks, then the leak report."""
        yield from self._drain_warnings()
        stdlib.finalize(self.machine)
        for warning in self.memory.drain_warnings():
            yield RunWarning(warning.to_diagnostic(None))
        for block in self.memory.leaks()[:LEAK_WARNING_LIMIT]:
            yield RunWarning(self._leak_diagnostic(block))

    def _leak_diagnostic(self, block: MemoryObject) -> Diagnostic:
        """The valgrind-style ``memory_leak`` warning for one block never freed."""
        location = self._alloc_sites.get(block.id)
        return Diagnostic(
            Severity.WARNING, "memory_leak",
            f"{block.size} bytes in 1 blocks are definitely lost: {block.name} was never freed",
            f"The block from {block.name} was still allocated when the program ended. Every "
            "malloc needs exactly one free once you are done with the memory; otherwise a "
            "long-running program slowly fills the heap with junk.",
            location, concept_for_code("memory_leak"), Phase.RUNTIME)

    # ================================================================ statements
    def exec(self, stmt: ast.Node) -> Generator[Event, Any, Flow]:
        """Execute one statement; returns None, BREAK, CONTINUE or a Returned."""
        handler = _EXEC.get(type(stmt))
        if handler is None:
            return _ready(None)    # struct/typedef declarations: nothing happens at run time
        return handler(self, stmt)

    def _statement(self, loc: SourceLocation) -> Iterable[Event]:
        """The events that start a statement; callers ``yield from`` the result.

        Normally that is just the one ``Statement``, returned as a tuple: this runs
        before every statement, and a tuple is much cheaper to create and to pass
        through than a generator. Queued warnings of the previous step take the
        slower path that reports them first.
        """
        if self.memory.pending_warnings:
            return self._statement_after_warnings(loc)
        return (self._step(loc),)

    def _statement_after_warnings(self, loc: SourceLocation) -> Gen:
        """Report the previous step's warnings, then start the statement at ``loc``."""
        yield from self._drain_warnings()
        yield self._step(loc)

    def _step(self, loc: SourceLocation) -> Statement:
        """Charge one statement to the budget and make the ``Statement`` event."""
        self.statements_executed += 1
        self.current_location = loc
        if self._frames:
            self._frames[-1].location = loc
        if self.statements_executed > self.budget:
            raise TrapError(
                "instruction_budget",
                f"killed: the program ran {self.budget} statements without finishing "
                "(instruction budget exhausted)", _BUDGET_FRIENDLY,
                extra={"budget": self.budget})
        return Statement(loc, self.depth)

    def _exec_items(self, items: list[ast.Node]) -> Generator[Event, Any, Flow]:
        """Run statements in order until one of them breaks, continues or returns."""
        for item in items:
            flow = yield from self.exec(item)
            if flow is not None:
                return flow
        return None

    def _release_temporaries(self, mark: int) -> None:
        """Free the stack slots allocated since ``mark`` in the running function.

        A struct returned by a call has no variable, so ``mk().a[0]`` gives it a
        stack slot (``_temporary``) that lives until its block ends. That is
        harmless for code that runs once per block, but a loop's condition, step
        and unbraced body run again and again in the same block: every loop calls
        this before each round, or ``while (mk().a[0] < n)`` would use up the stack.
        """
        # The stack grows down: anything allocated since the mark sits below it.
        if self.memory.stack_pointer < mark:
            self.memory.scope_release(self._frames[-1].memory_frame, mark)

    def _open_scope(self) -> int:
        """Start a block scope; returns the stack mark that ``_close_scope`` needs."""
        frame = self._frames[-1]
        mark = self.memory.scope_mark(frame.memory_frame)
        frame.scopes.append({})
        return mark

    def _close_scope(self, mark: int) -> None:
        """End a block scope: its names disappear and its locals die."""
        frame = self._frames[-1]
        frame.scopes.pop()
        self.memory.scope_release(frame.memory_frame, mark)

    def exec_compound(self, stmt: ast.CompoundStmt) -> Generator[Event, Any, Flow]:
        """A block: its locals live from their declaration to the closing brace.

        A ``break``, ``continue`` or ``return`` leaving the block still closes it,
        so no local outlives its block whichever way control leaves. A block that
        declares nothing (most loop bodies) has nothing to scope and skips that work.
        """
        scoped = self._declares_variables(stmt)
        mark = self._open_scope() if scoped else 0
        flow = None
        # The loop is written out instead of calling _exec_items: one generator
        # fewer between every statement and the host makes long loops faster.
        for item in stmt.items:
            flow = yield from self.exec(item)
            if flow is not None:
                break
        if scoped:
            self._close_scope(mark)
        return flow

    def _declares_variables(self, block: ast.CompoundStmt) -> bool:
        """Does ``block`` itself declare anything? (Cached: a loop body runs often.)"""
        known = self._block_declares.get(id(block))
        if known is None:
            known = any(isinstance(_unlabelled(item), ast.DeclStmt) for item in block.items)
            self._block_declares[id(block)] = known
        return known

    def exec_expr(self, stmt: ast.ExprStmt) -> Generator[Event, Any, Flow]:
        """An expression statement: evaluate and throw the value away."""
        yield from self._statement(stmt.loc)
        yield from self.eval(stmt.expr)
        return None

    def exec_empty(self, stmt: ast.EmptyStmt) -> Generator[Event, Any, Flow]:
        """A lone ';' still costs a step (so 'while (1);' runs out of budget)."""
        yield from self._statement(stmt.loc)
        return None

    def exec_decl(self, stmt: ast.DeclStmt) -> Generator[Event, Any, Flow]:
        """Declarations: allocate each local on the stack and run its initialiser.

        A declaration that makes no variable (a local struct or typedef) does nothing
        at run time, so it is not a step either.
        """
        variables = [decl for decl in stmt.decls if isinstance(decl, ast.VarDecl)]
        if not variables:
            return None
        yield from self._statement(stmt.loc)
        for decl in variables:
            yield from self._declare(decl)
        return None

    def exec_if(self, stmt: ast.IfStmt) -> Generator[Event, Any, Flow]:
        """``if``: the condition is a step of its own at the ``if`` line."""
        yield from self._statement(stmt.loc)
        cond = yield from self.eval(stmt.cond)
        if _truth(cond):
            return (yield from self.exec(stmt.then))
        if stmt.otherwise is not None:
            return (yield from self.exec(stmt.otherwise))
        return None

    def exec_while(self, stmt: ast.WhileStmt) -> Generator[Event, Any, Flow]:
        """``while``: one step for every test of the condition."""
        mark = self.memory.stack_pointer
        while True:
            self._release_temporaries(mark)
            yield from self._statement(stmt.loc)
            if not _truth((yield from self.eval(stmt.cond))):
                return None
            self.constructs[constructs.LOOP_ITERATION] += 1
            flow = yield from self.exec(stmt.body)
            if flow is BREAK:
                return None
            if isinstance(flow, Returned):
                return flow

    def exec_do_while(self, stmt: ast.DoWhileStmt) -> Generator[Event, Any, Flow]:
        """``do ... while``: the body first, then a step for the test at its line."""
        mark = self.memory.stack_pointer
        while True:
            self._release_temporaries(mark)
            self.constructs[constructs.LOOP_ITERATION] += 1
            flow = yield from self.exec(stmt.body)
            if flow is BREAK:
                return None
            if isinstance(flow, Returned):
                return flow
            yield from self._statement(stmt.cond.loc)
            if not _truth((yield from self.eval(stmt.cond))):
                return None

    def exec_for(self, stmt: ast.ForStmt) -> Generator[Event, Any, Flow]:
        """``for``: the header line is a step before the first test and before each
        later step-and-test, so stepping shows header, body, header, body...

        ``for (int i = ...)`` gives ``i`` a scope of its own around the loop; every
        way out (the test failing, ``break``, ``return``) closes that scope.
        """
        scope = self._open_scope()
        yield from self._statement(stmt.loc)
        yield from self._for_init(stmt.init)
        mark = self.memory.stack_pointer
        flow: Flow = None
        while True:
            self._release_temporaries(mark)
            if stmt.cond is not None and not _truth((yield from self.eval(stmt.cond))):
                break
            self.constructs[constructs.LOOP_ITERATION] += 1
            flow = yield from self.exec(stmt.body)
            if flow is BREAK or isinstance(flow, Returned):
                break
            flow = None                       # 'continue' goes on to the step
            yield from self._statement(stmt.loc)
            if stmt.step is not None:
                yield from self.eval(stmt.step)
        self._close_scope(scope)
        return None if flow is BREAK else flow

    def _for_init(self, init: ast.Node | None) -> Gen:
        """The first clause of a ``for``: declarations or an expression (or nothing)."""
        if isinstance(init, ast.DeclStmt):
            for decl in init.decls:
                if isinstance(decl, ast.VarDecl):
                    yield from self._declare(decl)
        elif isinstance(init, ast.Expr):
            yield from self.eval(init)

    def exec_return(self, stmt: ast.ReturnStmt) -> Generator[Event, Any, Flow]:
        """``return``: the checker already converted the value to the return type."""
        yield from self._statement(stmt.loc)
        value = None
        if stmt.value is not None:
            value = yield from self.eval(stmt.value)
        return Returned(value)

    def exec_break(self, stmt: ast.BreakStmt) -> Generator[Event, Any, Flow]:
        """``break``: leave the innermost loop or switch."""
        yield from self._statement(stmt.loc)
        return BREAK

    def exec_continue(self, stmt: ast.ContinueStmt) -> Generator[Event, Any, Flow]:
        """``continue``: skip to the next round of the innermost loop."""
        yield from self._statement(stmt.loc)
        return CONTINUE

    def exec_label(self, stmt: ast.CaseLabel | ast.DefaultLabel) -> Generator[Event, Any, Flow]:
        """Reaching a label by falling through just runs what follows it."""
        return (yield from self.exec(stmt.body))

    def exec_switch(self, stmt: ast.SwitchStmt) -> Generator[Event, Any, Flow]:
        """``switch``: jump to the matching label and fall through until ``break``."""
        yield from self._statement(stmt.loc)
        value = yield from self.eval(stmt.cond)
        self.count_construct(constructs.SWITCH)
        path = self._switch_table(stmt).path_for(int(value.raw))
        if path is None:
            return None
        flow = yield from self._exec_path(path, 0)
        return None if flow is BREAK else flow

    def _switch_table(self, stmt: ast.SwitchStmt) -> SwitchTable:
        """Where each label of ``stmt`` is (worked out once, then cached)."""
        table = self._switch_tables.get(id(stmt))
        if table is None:
            table = build_switch_table(stmt)
            ctype = ast.expression_type(stmt.cond)
            if ctype is not None and ctype.is_integer:
                # 'case -1' in a switch on an unsigned value matches the wrapped value.
                table.cases = {wrap_int(key, ctype): path for key, path in table.cases.items()}
            self._switch_tables[id(stmt)] = table
        return table

    def _exec_path(self, path: list[ast.Node], index: int) -> Generator[Event, Any, Flow]:
        """Enter ``path[index]`` heading for the label at the end of ``path``."""
        node = path[index]
        if index == len(path) - 1:
            return (yield from self.exec(node))
        child = path[index + 1]
        if not isinstance(node, ast.CompoundStmt):
            return (yield from self._exec_path(path, index + 1))   # a label around the label
        mark = self._open_scope()
        position = index_of(node.items, child)
        # Jumping past a declaration is legal C: the variable exists, uninitialised.
        for skipped in node.items[:position]:
            if isinstance(skipped, ast.DeclStmt):
                for decl in skipped.decls:
                    if isinstance(decl, ast.VarDecl):
                        yield from self._declare(decl, initialize=False)
        flow = yield from self._exec_path(path, index + 1)
        if flow is None:
            flow = yield from self._exec_items(node.items[position + 1:])
        self._close_scope(mark)
        return flow

    # ================================================================ declarations
    def _declare(self, decl: ast.VarDecl, initialize: bool = True) -> Gen:
        """Bring a block-scope variable to life."""
        if decl.is_extern or decl.storage == "extern":
            return
        frame = self._frames[-1]
        if decl.storage == "static":
            # Initialised once at start-up; bound here only so the Variables panel
            # can show it among the function's locals.
            key = getattr(decl, "storage_key", None)
            if key in self._global_addrs:
                frame.bind(Slot(decl.name, decl.ctype, self._global_addrs[key]))
            return
        ctype = decl.ctype
        addr = self.memory.alloc_local(frame.memory_frame, ctype.size, ctype.align or 1,
                                       decl.name, ctype)
        frame.bind(Slot(decl.name, ctype, addr))
        if decl.init is not None and initialize:
            yield from self._init_object(addr, ctype, decl.init)

    def _init_object(self, addr: int, ctype: CType, init: ast.Expr) -> Gen:
        """Run the initialiser of the object of ``ctype`` at ``addr``."""
        if isinstance(init, ast.InitList):
            # A brace list sets every byte: members it does not name become zero.
            self.memory.write_bytes(addr, bytes(ctype.size))
            yield from self._fill_braced(addr, ctype, init)
            return
        if isinstance(ctype, ArrayType):
            literal = strip_implicit(init)
            if isinstance(literal, ast.StringLiteral):
                self._write_string(addr, ctype, literal)
            return          # any other array initialiser was rejected by the parser
        if isinstance(ctype, StructType) and is_lvalue(init):
            yield from self._copy_struct(Place(ctype, addr), init)
            return
        value = yield from self.eval(init)
        self._store(Place(ctype, addr), value)

    def _write_string(self, addr: int, array: ArrayType, literal: ast.StringLiteral) -> None:
        """``char s[N] = "..."``: copy the characters, then the NUL if it fits, then zeros."""
        length = array.length if array.length is not None else len(literal.value) + 1
        data = (literal.value + b"\0")[:length]
        self.memory.write_bytes(addr, data + bytes(length - len(data)))

    def _fill_braced(self, addr: int, target: CType, init: ast.InitList) -> Gen:
        """Fill an already-zeroed object from one brace list."""
        items = init.items
        if (isinstance(target, ArrayType) and _is_char(target.element) and len(items) == 1
                and isinstance(items[0], ast.StringLiteral)):
            self._write_string(addr, target, items[0])      # char s[8] = {"abc"}
            return
        cursor = _InitCursor(items)
        if isinstance(target, (ArrayType, StructType)):
            yield from self._fill(addr, target, cursor)
        elif not cursor.done:
            yield from self._fill_scalar(addr, target, cursor)    # int x = {5};
        # Excess items were reported at compile time and are ignored, as gcc does.

    def _fill(self, addr: int, target: ArrayType | StructType, cursor: _InitCursor) -> Gen:
        """Consume items for an aggregate, entering sub-aggregates without braces."""
        if isinstance(target, ArrayType):
            size = target.element.size
            count = 0
            while not cursor.done and (target.length is None or count < target.length):
                yield from self._fill_one(addr + count * size, target.element, cursor)
                count += 1
            return
        members = target.members or []
        if target.is_union:
            members = members[:1]       # a union's brace list initialises its first member
        for member in members:
            if cursor.done:
                return
            yield from self._fill_one(addr + member.offset, member.ctype, cursor)

    def _fill_one(self, addr: int, sub: CType, cursor: _InitCursor) -> Gen:
        """Initialise one element or member of type ``sub`` from the cursor."""
        item = cursor.items[cursor.pos]
        if isinstance(item, ast.InitList):
            cursor.take()
            yield from self._fill_braced(addr, sub, item)
        elif isinstance(sub, ArrayType) and _is_char(sub.element) and \
                isinstance(strip_implicit(item), ast.StringLiteral):
            cursor.take()
            literal = strip_implicit(item)
            assert isinstance(literal, ast.StringLiteral)
            self._write_string(addr, sub, literal)
        elif isinstance(sub, ArrayType):
            yield from self._fill(addr, sub, cursor)
        elif isinstance(sub, StructType):
            if isinstance(ast.expression_type(item), StructType):
                cursor.take()
                value = yield from self.eval(item)
                self._store(Place(sub, addr), value)
            else:
                yield from self._fill(addr, sub, cursor)
        else:
            yield from self._fill_scalar(addr, sub, cursor)

    def _fill_scalar(self, addr: int, sub: CType, cursor: _InitCursor) -> Gen:
        """Initialise one scalar from the next item (``{5}`` around it is allowed)."""
        item = cursor.take()
        if isinstance(item, ast.InitList):
            yield from self._fill_braced(addr, sub, item)
            return
        value = yield from self.eval(item)
        self._store(Place(sub, addr), value)

    # ================================================================ memory access
    def _load(self, place: Place) -> Value | StructValue:
        """Read the object at ``place`` as a value."""
        ctype = place.ctype
        if type(ctype) is IntType:          # by far the most common case, tested first
            return Value(ctype, self.memory.load(place.addr, ctype, place.origin))
        if isinstance(ctype, StructType):
            return StructValue(ctype, self.memory.read_bytes(place.addr, ctype.size, place.origin))
        if isinstance(ctype, ArrayType):
            return Value(PointerType(ctype.element), place.addr)   # an array used as a value
        return Value(ctype, self.memory.load(place.addr, ctype, place.origin))

    def _store(self, place: Place, value: Value | StructValue | None) -> None:
        """Write ``value`` into the object at ``place``."""
        if isinstance(value, StructValue):
            self.memory.write_bytes(place.addr, value.data, place.origin)
        elif value is not None:
            self.memory.store(place.addr, place.ctype, value.raw, place.origin)

    def _copy_struct(self, dst: Place, source: ast.Expr) -> Gen:
        """Struct assignment from an lvalue: bytes *and* their initialised state move."""
        src = yield from self.place(source)
        self.memory.copy_bytes(dst.addr, src.addr, dst.ctype.size)

    def _temporary(self, value: Value | StructValue) -> Place:
        """Give an rvalue (a struct returned by a call) an address on the stack."""
        ctype = value.ctype
        frame = self._frames[-1]
        addr = self.memory.alloc_local(frame.memory_frame, ctype.size, ctype.align or 1,
                                       "<temporary>", ctype)
        self._store(Place(ctype, addr), value)
        return Place(ctype, addr)

    def _static_address(self, key: str) -> int:
        """Address of a global, static or library variable by storage key."""
        addr = self._global_addrs.get(key)
        if addr is None:
            addr = self._builtin_vars.get(key)
        if addr is None:
            raise RuntimeError(f"variable {key!r} has no storage (was the program linked?)")
        return addr

    # ================================================================ places (lvalues)
    def place(self, expr: ast.Expr) -> Generator[Event, Any, Place]:
        """Evaluate ``expr`` as an lvalue: where the object lives, without reading it."""
        handler = _PLACE.get(type(expr))
        if handler is None:
            return self._place_of_rvalue(expr)
        return handler(self, expr)

    def _place_of_rvalue(self, expr: ast.Expr) -> Generator[Event, Any, Place]:
        """Evaluate a value that has no address (a returned struct) into a temporary."""
        value = yield from self.eval(expr)
        return self._temporary(value)

    def place_identifier(self, expr: ast.Identifier) -> Generator[Event, Any, Place]:
        """A variable's storage, found the way the checker resolved the name."""
        return _ready(self._variable_place(expr))

    def _variable_place(self, expr: ast.Identifier) -> Place:
        """Where a named variable lives (a plain function: finding it never pauses)."""
        ref = expr.ref  # type: ignore[attr-defined]
        ctype = expr.ctype or ref.ctype  # type: ignore[attr-defined]
        if ref.kind in ("local", "param"):
            slot = self._frames[-1].lookup(ref.key)
            if slot is None:
                raise RuntimeError(f"local {ref.key!r} is not in scope")
            return Place(ctype, slot.addr)
        return Place(ctype, self._static_address(ref.key))

    def place_string(self, expr: ast.StringLiteral) -> Generator[Event, Any, Place]:
        """A string literal lives in read-only data (writing to it traps)."""
        addr = self.memory.intern_string(expr.value)
        return _ready(Place(ArrayType(CHAR, len(expr.value) + 1), addr))

    def place_index(self, expr: ast.Index) -> Generator[Event, Any, Place]:
        """``base[index]``: ``*(base + index)``, remembering the base as provenance."""
        base = yield from self.eval(expr.base)
        index = yield from self.eval(expr.index)
        element = expr.ctype  # type: ignore[attr-defined]
        step = element.size if element.size > 0 else 1
        self.count_construct(constructs.ARRAY_INDEX)
        addr = (int(base.raw) + int(index.raw) * step) & POINTER_MASK
        return Place(element, addr, int(base.raw))

    def place_member(self, expr: ast.Member) -> Generator[Event, Any, Place]:
        """``s.x`` and ``p->x``: the member's offset from the struct's address."""
        if expr.arrow:
            pointer = yield from self.eval(expr.base)
            holder_type = ast.expression_type(expr.base).target  # type: ignore[union-attr]
            base_addr, origin = int(pointer.raw), int(pointer.raw)
            self.count_construct(constructs.POINTER_DEREF)
        else:
            holder = yield from self.place(expr.base)
            holder_type, base_addr, origin = holder.ctype, holder.addr, holder.origin
        member = holder_type.member(expr.name)
        assert member is not None, "the checker guarantees the member exists"
        self.count_construct(constructs.STRUCT_MEMBER)
        return Place(member.ctype, (base_addr + member.offset) & POINTER_MASK, origin)

    def place_unary(self, expr: ast.UnaryOp) -> Generator[Event, Any, Place]:
        """``*p``: the object the pointer points at."""
        if expr.op != "*":
            return (yield from self._place_of_rvalue(expr))
        pointer, origin = yield from self._pointer_with_origin(expr.operand)
        self.count_construct(constructs.POINTER_DEREF)
        return Place(expr.ctype, int(pointer.raw), origin)  # type: ignore[attr-defined]

    def _pointer_with_origin(
            self, expr: ast.Expr) -> Generator[Event, Any, tuple[Value, int | None]]:
        """Evaluate a pointer; for ``p + n`` also return ``p`` as the provenance."""
        inner = strip_implicit(expr)
        if isinstance(inner, ast.BinaryOp) and inner.op in ("+", "-") and _is_pointer_offset(inner):
            return (yield from self._pointer_offset(inner))
        value = yield from self.eval(expr)
        return value, None

    # ================================================================ expressions (rvalues)
    def eval(self, expr: ast.Expr) -> Generator[Event, Any, Any]:
        """Evaluate ``expr`` to a ``Value``, a ``StructValue`` or None (void)."""
        return _EVAL[type(expr)](self, expr)

    def eval_int_literal(self, expr: ast.IntLiteral) -> Gen:
        """An integer constant, already typed by the parser (2147483648 is a long)."""
        return _ready(self._constant(expr))

    def eval_float_literal(self, expr: ast.FloatLiteral) -> Gen:
        """A floating constant; an ``f`` suffix makes it a 32-bit float."""
        # 0.1f really is the float nearest 0.1, not the double.
        return _ready(Value(expr.ctype, convert_value(expr.value, DOUBLE, expr.ctype)))

    def eval_char_literal(self, expr: ast.CharLiteral) -> Gen:
        """A character constant such as ``'a'``."""
        return _ready(Value(INT, expr.value))      # 'a' has type int in C

    def eval_string(self, expr: ast.StringLiteral) -> Gen:
        """A string literal used as a value: the address of its first char."""
        return _ready(Value(PointerType(CHAR), self.memory.intern_string(expr.value)))

    def eval_identifier(self, expr: ast.Identifier) -> Gen:
        """Read a variable; function names give their pointer id; enum constants their value."""
        return _ready(self._identifier_value(expr))

    def _identifier_value(self, expr: ast.Identifier) -> Value | StructValue:
        """The value a name stands for (a plain function: reading a name never pauses)."""
        ref = expr.ref  # type: ignore[attr-defined]
        if ref.kind == "enum_const":
            return Value(INT, ref.value if ref.value is not None else int(ref.key))
        if isinstance(ref.ctype, FunctionType):
            return Value(ref.ctype, self.program.function_id(ref.key))
        return self._load(self._variable_place(expr))

    def _quick_value(self, expr: ast.Expr) -> Value | StructValue | None:
        """The value of a name or a constant (maybe converted), or None for the rest.

        Operands are mostly plain variables and constants, which can never pause the
        program. Hot paths ask this first and only build a generator (``eval``) for
        anything else, and an ``int``-like local is read here directly: those few
        saved calls per operand are a large share of the cost of ``s += i;``.
        """
        kind = type(expr)
        if kind is ast.Identifier:
            ref = expr.ref  # type: ignore[attr-defined]
            ctype = expr.ctype  # type: ignore[attr-defined]
            if type(ctype) is IntType and (ref.kind == "local" or ref.kind == "param"):
                slot = self._frames[-1].lookup(ref.key)
                if slot is not None:
                    return Value(ctype, self.memory.load(slot.addr, ctype))
            return self._identifier_value(expr)  # type: ignore[arg-type]
        if kind is ast.IntLiteral:
            return self._constant(expr)  # type: ignore[arg-type]
        if kind is ast.CharLiteral:
            return Value(INT, expr.value)  # type: ignore[attr-defined]
        if kind is ast.ImplicitCast:
            # An array name reads as the address of its first element (see _load),
            # so decay needs no special case here.
            inner = self._quick_value(expr.expr)  # type: ignore[attr-defined]
            if inner is None:
                return None
            return self._convert(inner, expr.to)  # type: ignore[attr-defined]
        return None

    def _constant(self, expr: ast.IntLiteral) -> Value:
        """The value of an integer constant, made once per literal and then reused.

        Values are immutable, so one object can stand for ``100000`` in every round
        of a loop instead of building (and wrapping) a new one each time.
        """
        value = self._constants.get(id(expr))
        if value is None:
            value = Value(expr.ctype, wrap_int(expr.value, expr.ctype))  # type: ignore[arg-type]
            self._constants[id(expr)] = value
        return value

    def _quick_place(self, expr: ast.Expr) -> Place | None:
        """Where a plain variable lives, or None when finding the object takes ``place``."""
        if type(expr) is ast.Identifier:
            return self._variable_place(expr)  # type: ignore[arg-type]
        return None

    def eval_place(self, expr: ast.Expr) -> Gen:
        """Read an lvalue expression (``a[i]``, ``s.x``, ``*p``)."""
        place = yield from self.place(expr)
        return self._load(place)

    def eval_member(self, expr: ast.Member) -> Gen:
        """Member read; a member of a returned struct is read from its bytes."""
        if expr.arrow or is_lvalue(expr.base):
            return (yield from self.eval_place(expr))
        holder = yield from self.eval(expr.base)
        member = holder.ctype.member(expr.name)
        self.count_construct(constructs.STRUCT_MEMBER)
        if isinstance(member.ctype, ArrayType):
            place = self._temporary(holder)        # an array member needs an address
            return Value(PointerType(member.ctype.element), place.addr + member.offset)
        data = holder.data[member.offset:member.offset + member.ctype.size]
        if isinstance(member.ctype, StructType):
            return StructValue(member.ctype, data)
        return Value(member.ctype, decode(data, member.ctype))

    def eval_implicit_cast(self, expr: ast.ImplicitCast) -> Gen:
        """A conversion the checker made visible: decay, promotion or assignment."""
        source = ast.expression_type(expr.expr)
        if isinstance(source, ArrayType):
            place = yield from self.place(expr.expr)      # the array's address
            return Value(expr.to, place.addr)
        value = self._quick_value(expr.expr)
        if value is None:
            value = yield from self.eval(expr.expr)
        return self._convert(value, expr.to)

    def eval_cast(self, expr: ast.Cast) -> Gen:
        """``(type) expr``; a cast to void evaluates and discards."""
        value = yield from self.eval(expr.expr)
        if expr.ctype.is_void:
            return None
        return self._convert(value, expr.ctype)

    @staticmethod
    def _convert(value: Value | StructValue | None, to: CType) -> Value | StructValue | None:
        """Convert a value to ``to`` (void discards it; structs pass unchanged)."""
        if isinstance(value, Value) and value.ctype is to:
            return value        # the common case 'int to int': nothing to do
        if value is None or isinstance(value, StructValue) or to.is_void:
            return None if to.is_void else value
        if isinstance(to, StructType):
            return value
        return Value(to, convert_value(value.raw, value.ctype, to))

    def eval_sizeof_type(self, expr: ast.SizeofType) -> Gen:
        """``sizeof(type)``: known without running anything."""
        return _ready(Value(SIZE_T, expr.ctype.size))

    def eval_sizeof_expr(self, expr: ast.SizeofExpr) -> Gen:
        """``sizeof expr`` never evaluates expr: arrays give their full size."""
        ctype = ast.expression_type(expr.expr)
        return _ready(Value(SIZE_T, ctype.size if ctype is not None else 0))

    def eval_conditional(self, expr: ast.Conditional) -> Gen:
        """``c ? a : b``: only the chosen side is evaluated."""
        cond = yield from self.eval(expr.cond)
        return (yield from self.eval(expr.then if truthy(cond) else expr.otherwise))

    def eval_comma(self, expr: ast.Comma) -> Gen:
        """``a, b``: evaluate left to right; the value is the last one."""
        result = None
        for item in expr.exprs:
            result = yield from self.eval(item)
        return result

    def eval_init_list(self, expr: ast.InitList) -> Gen:
        """Brace lists are handled by the initializer code, never evaluated."""
        raise RuntimeError("a brace list can only appear in a declaration")

    # --- operators ------------------------------------------------------------------
    def eval_binary(self, expr: ast.BinaryOp) -> Gen:
        """Binary operators; operands were already converted by the checker."""
        op = expr.op
        if op in ("&&", "||"):
            return (yield from self._logical(expr))
        if op in ("+", "-") and _involves_pointer(expr):
            if _is_pointer_offset(expr):
                value, _ = yield from self._pointer_offset(expr)
                return value
            return (yield from self._pointer_difference(expr))
        left = self._quick_value(expr.left)
        if left is None:
            left = yield from self.eval(expr.left)
        right = self._quick_value(expr.right)
        if right is None:
            right = yield from self.eval(expr.right)
        if op in _COMPARISONS:
            return _TRUE if _COMPARISONS[op](left.raw, right.raw) else _FALSE
        if op in ("<<", ">>"):
            return self._shift(op, left, right, expr.ctype)  # type: ignore[attr-defined]
        return self._arithmetic(op, left, right, expr.ctype)  # type: ignore[attr-defined]

    def _logical(self, expr: ast.BinaryOp) -> Gen:
        """``&&`` and ``||`` stop as soon as the answer is known (short circuit)."""
        left = _truth((yield from self.eval(expr.left)))
        if expr.op == "&&" and not left:
            return _FALSE
        if expr.op == "||" and left:
            return _TRUE
        right = _truth((yield from self.eval(expr.right)))
        return _TRUE if right else _FALSE

    def _pointer_offset(self, expr: ast.BinaryOp) -> Generator[Event, Any, tuple[Value, int]]:
        """``p + n``, ``n + p``, ``p - n``: move by n elements; returns (result, p)."""
        left = yield from self.eval(expr.left)
        right = yield from self.eval(expr.right)
        pointer, count = (left, right) if left.ctype.is_pointer else (right, left)
        sign = -1 if expr.op == "-" else 1
        step = pointer_step(pointer.ctype)
        self.count_construct(constructs.POINTER_ARITH)
        raw = (int(pointer.raw) + sign * int(count.raw) * step) & POINTER_MASK
        return Value(expr.ctype, raw), int(pointer.raw)  # type: ignore[attr-defined]

    def _pointer_difference(self, expr: ast.BinaryOp) -> Gen:
        """``p - q``: how many elements apart two pointers are."""
        left = yield from self.eval(expr.left)
        right = yield from self.eval(expr.right)
        self.count_construct(constructs.POINTER_ARITH)
        distance = trunc_div(int(left.raw) - int(right.raw), pointer_step(left.ctype))
        ctype = expr.ctype or LONG  # type: ignore[attr-defined]
        return Value(ctype, wrap_int(distance, ctype))

    def _arithmetic(self, op: str, left: Value, right: Value, ctype: CType) -> Value:
        """``+ - * / % & | ^`` in the common type ``ctype``."""
        if type(ctype) is not IntType and ctype.is_float:
            raw = float_result(op, float(left.raw), float(right.raw))
            return Value(ctype, convert_value(raw, DOUBLE, ctype))
        a, b = int(left.raw), int(right.raw)
        try:
            exact = exact_int_result(op, a, b)
        except ZeroDivisionError:
            raise self._division_trap(op, a) from None
        result = wrap_int(exact, ctype)
        if result != exact and getattr(ctype, "signed", False):
            self._overflow_warning(f"{a} {op} {b}", exact, result, ctype)
        return Value(ctype, result)

    def _shift(self, op: str, left: Value, right: Value, ctype: CType) -> Value:
        """``<<`` and ``>>``: out-of-range counts are masked the way x86 does."""
        a, count = int(left.raw), int(right.raw)
        bits = ctype.size * 8
        if count < 0 or count >= bits:
            self._shift_warning(count, ctype)
            count &= bits - 1
        if op == ">>":
            return Value(ctype, a >> count)       # arithmetic for signed values
        exact = a << count
        result = wrap_int(exact, ctype)
        if result != exact and getattr(ctype, "signed", False):
            self._overflow_warning(f"{a} << {count}", exact, result, ctype)
        return Value(ctype, result)

    def eval_unary(self, expr: ast.UnaryOp) -> Gen:
        """Prefix operators."""
        op = expr.op
        if op in ("++", "--"):
            return (yield from self._increment(expr.operand, op, prefix=True))
        if op == "&":
            return (yield from self._address_of(expr))
        if op == "*":
            if isinstance(expr.ctype, FunctionType):  # type: ignore[attr-defined]
                pointer = yield from self.eval(expr.operand)    # (*fp)(x) calls fp
                return Value(expr.ctype, pointer.raw)  # type: ignore[attr-defined]
            place = yield from self.place_unary(expr)
            if place.ctype.is_void:
                return None
            return self._load(place)
        value = yield from self.eval(expr.operand)
        if op == "!":
            return _FALSE if truthy(value) else _TRUE
        ctype = expr.ctype  # type: ignore[attr-defined]
        if op == "+":
            return value
        if op == "~":
            return Value(ctype, wrap_int(~int(value.raw), ctype))
        if ctype.is_float:                                 # unary minus
            return Value(ctype, -value.raw)
        exact = -int(value.raw)
        result = wrap_int(exact, ctype)
        if result != exact and getattr(ctype, "signed", False):
            self._overflow_warning(f"-({value.raw})", exact, result, ctype)
        return Value(ctype, result)

    def _address_of(self, expr: ast.UnaryOp) -> Gen:
        """``&x``: the address, never a read (``&a[10]`` of a 10-array is fine)."""
        operand = expr.operand
        if isinstance(ast.expression_type(operand), FunctionType):
            function = yield from self.eval(operand)
            return Value(expr.ctype, function.raw)  # type: ignore[attr-defined]
        place = yield from self.place(operand)
        return Value(expr.ctype, place.addr)  # type: ignore[attr-defined]

    def eval_postfix(self, expr: ast.PostfixOp) -> Gen:
        """``x++`` and ``x--``: the value before the change."""
        # Hands back _increment's own generator rather than wrapping it in another.
        return self._increment(expr.operand, expr.op, prefix=False)

    def _increment(self, operand: ast.Expr, op: str, prefix: bool) -> Gen:
        """++ and -- on integers, floats and pointers (pointers move one element)."""
        place = self._quick_place(operand)
        if place is None:
            place = yield from self.place(operand)
        old = self._load(place)
        ctype = place.ctype
        delta = 1 if op == "++" else -1
        if ctype.is_pointer:
            self.count_construct(constructs.POINTER_ARITH)
            new_raw: int | float = (int(old.raw) + delta * pointer_step(ctype)) & POINTER_MASK
        elif ctype.is_float:
            new_raw = convert_value(old.raw + delta, DOUBLE, ctype)
        else:
            # C computes x + 1 in the promoted type, then converts back: char 127 + 1
            # becomes -128 by conversion, which is not an overflow; int max + 1 is.
            promoted = integer_promote(ctype)
            exact = int(old.raw) + delta
            wrapped = wrap_int(exact, promoted)
            if wrapped != exact and getattr(promoted, "signed", False):
                self._overflow_warning(f"{old.raw} {'+' if delta > 0 else '-'} 1", exact,
                                       wrapped, promoted)
            new_raw = wrapped if promoted is ctype else convert_value(exact, promoted, ctype)
        new = Value(ctype, new_raw)
        self._store(place, new)
        return new if prefix else old

    def eval_assignment(self, expr: ast.Assignment) -> Gen:
        """``=`` and the compound assignments (``+=``, ``<<=``...)."""
        target_type = ast.expression_type(expr.target)
        if expr.op == "=" and isinstance(target_type, StructType) and is_lvalue(expr.value):
            place = yield from self.place(expr.target)
            yield from self._copy_struct(place, expr.value)
            return self._load(place)
        place = self._quick_place(expr.target)
        if place is None:
            place = yield from self.place(expr.target)
        if expr.op == "=":
            value = self._quick_value(expr.value)
            if value is None:
                value = yield from self.eval(expr.value)
            self._store(place, value)
            return value
        old = self._load(place)
        value = self._quick_value(expr.value)
        if value is None:
            value = yield from self.eval(expr.value)
        new = self._compound(expr.op[:-1], old, value, place.ctype)
        self._store(place, new)
        return new

    def _compound(self, op: str, old: Value, value: Value, ctype: CType) -> Value:
        """``x op= v``: compute in the common type, then convert back to x's type."""
        if ctype.is_pointer:
            self.count_construct(constructs.POINTER_ARITH)
            sign = -1 if op == "-" else 1
            raw = (int(old.raw) + sign * int(value.raw) * pointer_step(ctype)) & POINTER_MASK
            return Value(ctype, raw)
        if op in ("<<", ">>"):
            promoted = integer_promote(ctype)
            result = self._shift(op, Value(promoted, old.raw), value, promoted)
        else:
            # 's += i' with both sides already of s's (int-sized) type is the common
            # case; then the common type is that type and nothing needs converting.
            same = value.ctype is ctype and type(ctype) is IntType and ctype.size >= 4
            common = ctype if same else usual_arithmetic_conversions(ctype, value.ctype)
            left = self._convert(old, common)
            right = self._convert(value, common)
            result = self._arithmetic(op, left, right, common)  # type: ignore[arg-type]
        return self._convert(result, ctype)  # type: ignore[return-value]

    # --- calls ----------------------------------------------------------------------------
    def eval_call(self, expr: ast.Call) -> Gen:
        """A call through a name or a function pointer; arguments left to right."""
        callee = yield from self.eval(expr.func)
        args = []
        for arg in expr.args:
            args.append((yield from self.eval(arg)))
        return (yield from self.call_function_id(int(callee.raw), args, expr.loc,
                                                 ast.expression_type(expr)))

    def call_function_id(self, fid: int, args: list[Any],
                         call_site: SourceLocation | None = None,
                         result_type: CType | None = None) -> Gen:
        """Call the function with pointer id ``fid`` (user code or a builtin).

        ``result_type`` is the call's checked type; a builtin's result is converted to
        it (without one, the builtin's own prototype decides).
        """
        target = self.program.function_at(fid)
        if target is None:
            raise self._bad_pointer_trap(fid)
        saved = self.current_location
        if isinstance(target, BuiltinSpec):
            result = yield from self._call_builtin(target, args, result_type)
        else:
            result = yield from self._call_user(target, args, call_site or saved)
        self.current_location = saved
        return result

    def _call_user(self, function: ast.FunctionDef, args: list[Any],
                   call_site: SourceLocation | None, count: bool = True) -> Gen:
        """Push a frame, bind the parameters, run the body, pop the frame."""
        if len(self._frames) >= MAX_CALL_DEPTH:
            raise self._stack_overflow_trap(function.name)
        recursive = any(frame.function is function for frame in self._frames)
        memory_frame = self.memory.push_frame(function.name, call_site)
        frame = CallFrame(function, memory_frame, call_site, location=function.loc)
        self._frames.append(frame)
        if count:
            self.count_construct(constructs.FUNCTION_CALL)
            if recursive:
                self.count_construct(constructs.RECURSION)
        enter = CallEnter(function.name, call_site)
        self.call_trace.append(enter)
        yield enter
        self._bind_parameters(frame, args)
        flow = yield from self._exec_items(function.body.items)
        result = flow.value if isinstance(flow, Returned) else self._fell_off(function)
        # Warnings from the callee's last statement ('return a + b;' overflowing) belong
        # to that statement: report them now, before the caller's location comes back.
        yield from self._drain_warnings()
        self._frames.pop()
        self.memory.pop_frame()
        leave = CallExit(function.name)
        self.call_trace.append(leave)
        yield leave
        return result

    def _bind_parameters(self, frame: CallFrame, args: list[Any]) -> None:
        """Each parameter is a fresh local holding a copy of its argument."""
        function = frame.function
        for index, (name, ctype) in enumerate(zip(function.param_names, function.ftype.params)):
            if isinstance(ctype, ArrayType):
                ctype = PointerType(ctype.element)     # 'int a[]' is 'int *a'
            addr = self.memory.alloc_local(frame.memory_frame, ctype.size, ctype.align or 1,
                                           name or f"<param {index + 1}>", ctype)
            if name:
                frame.bind(Slot(name, ctype, addr))
            if index < len(args):   # a missing argument (old-style call) stays garbage
                self._store(Place(ctype, addr), self._convert(args[index], ctype))

    @staticmethod
    def _fell_off(function: ast.FunctionDef) -> Value | StructValue | None:
        """The result when control reaches the closing brace (main: 0, C99)."""
        ret = function.ftype.ret
        if ret.is_void:
            return None
        if isinstance(ret, StructType):
            return StructValue(ret, bytes(ret.size))
        return Value(ret, 0)

    def _call_builtin(self, spec: BuiltinSpec, args: list[Any],
                      result_type: CType | None) -> Gen:
        """Run a library function; generator implementations may yield events."""
        result = spec.impl(self.machine, args)
        if inspect.isgenerator(result):
            result = yield from self._drive(result)
        value = self._builtin_result(result, result_type or self._builtin_type(spec).ret)
        if isinstance(value, Value) and value.ctype.is_pointer:
            self._note_allocation(int(value.raw))
        return value

    def _drive(self, generator: Generator[Any, Any, Any]) -> Gen:
        """Pass a builtin's events to the host and the host's replies back.

        ``yield from`` would do the forwarding too, but each ``DeviceCall`` must be
        counted on its way out. If the host abandons the run (Ctrl+C closes the
        interpreter's generator while ``qsort`` waits on a comparator), the
        ``finally`` closes the builtin as well, so its own clean-up runs now and
        not at some random garbage collection later.
        """
        reply: Any = None
        try:
            while True:
                try:
                    event = generator.send(reply)
                except StopIteration as done:
                    return done.value
                if isinstance(event, DeviceCall):
                    self.count_construct(constructs.DEVICE_CALL)
                reply = yield event
        finally:
            generator.close()

    @staticmethod
    def _builtin_result(result: Any, ret: CType) -> Value | StructValue | None:
        """Coerce what an implementation returned to its C return type ``ret``."""
        if ret.is_void:
            return None
        if isinstance(result, StructValue):
            return result
        if result is None:
            return Value(ret, 0)
        if isinstance(result, Value):
            return Value(ret, convert_value(result.raw, result.ctype, ret))
        source = DOUBLE if isinstance(result, float) else LONG
        return Value(ret, convert_value(result, source, ret))

    def _builtin_type(self, spec: BuiltinSpec) -> FunctionType:
        """The builtin's parsed prototype (only needed for calls from library code)."""
        if spec.ftype is None:
            from pycc.interp.parser import parse_prototype  # lazy: only for unregistered specs

            spec.ftype = (self.builtins.function_type(spec.name) if spec.name in
                          self.builtins.functions else parse_prototype(spec.prototype))
        return spec.ftype

    def _note_allocation(self, addr: int) -> None:
        """Remember where a heap block was born, for the leak report."""
        block = self.memory.object_at(addr) if addr else None
        if block is not None and block.kind == "heap" and block.id not in self._alloc_sites:
            self._alloc_sites[block.id] = self.current_location

    # ================================================================ warnings and traps
    def _warn(self, code: str, message: str, friendly: str) -> None:
        """Queue a runtime warning; it is reported before the next statement."""
        self.memory.pending_warnings.append(
            PendingWarning(code, message, friendly, concept_for_code(code)))

    def first_report(self, code: str, location: SourceLocation | None) -> bool:
        """True the first time a warning ``code`` is reported at ``location`` this run.

        Every runtime warning, queued here or sent back by the device (for example inside a
        loop), is shown at most once per (code, place), so a loop that repeats the
        same mistake cannot bury the terminal under identical lines.
        """
        key = (code, location)
        if key in self._reported:
            return False
        self._reported.add(key)
        return True

    def _drain_warnings(self) -> Gen:
        """Yield queued warnings at the current statement, each (code, place) once."""
        for warning in self.memory.drain_warnings():
            if self.first_report(warning.code, self.current_location):
                yield RunWarning(warning.to_diagnostic(self.current_location))

    def _overflow_warning(self, operation: str, exact: int, result: int, ctype: CType) -> None:
        """Queue ``signed_overflow``: ``operation`` gave ``exact``, which wrapped to ``result``."""
        name = format_type(ctype)
        self._warn(
            "signed_overflow",
            f"signed integer overflow: {operation} cannot be represented in type '{name}'",
            f"'{name}' holds values from {ctype.min} to {ctype.max}. "  # type: ignore[attr-defined]
            f"{operation} = {exact} does not fit, so it wrapped around to {result}. Use a "
            "wider type such as long, or keep the numbers smaller.")

    def _shift_warning(self, count: int, ctype: CType) -> None:
        """Queue ``shift_overflow`` for a negative or too-large shift count."""
        bits = ctype.size * 8
        name = format_type(ctype)
        if count < 0:
            message = f"shift exponent {count} is negative"
        else:
            message = f"shift exponent {count} is too large for {bits}-bit type '{name}'"
        self._warn("shift_overflow", message,
                   f"A shift count must be between 0 and {bits - 1} for '{name}'. The machine "
                   f"only looks at the low bits of {count}, so the result is not what the "
                   "code seems to say.")

    def _division_trap(self, op: str, dividend: int) -> TrapError:
        """The ``division_by_zero`` trap for ``dividend / 0`` or ``dividend % 0``."""
        what = "division" if op == "/" else "modulo"
        sign = "/" if op == "/" else "%"
        return TrapError(
            "division_by_zero", f"integer {what} by zero ({dividend} {sign} 0)",
            f"The right side of '{sign}' was 0. Dividing a whole number by zero has no "
            "answer, so the machine stops. Check the divisor first, for example "
            "'if (count != 0)'.")

    def _bad_pointer_trap(self, fid: int) -> TrapError:
        """The ``bad_function_pointer`` trap for a call through NULL or garbage."""
        if fid == 0:
            message = "call through a NULL function pointer"
            friendly = ("This function pointer is NULL, so there is no function to call. "
                        "Assign it a function's name before calling through it.")
        else:
            message = f"call through invalid function pointer {fid:#x}"
            friendly = ("This function pointer does not point at a real function: it may be "
                        "uninitialised garbage or a number cast to a pointer.")
        return TrapError("bad_function_pointer", message, friendly, extra={"id": fid})

    def _stack_overflow_trap(self, name: str) -> TrapError:
        """The ``stack_overflow`` trap for one call too many."""
        return TrapError(
            "stack_overflow",
            f"stack overflow in '{name}': more than {MAX_CALL_DEPTH} nested calls",
            "Each function call takes a slice of the stack. Too many calls at once (often "
            "recursion without a base case that is always reached) used it all up.",
            extra={"function": name, "depth": len(self._frames)})


# =========================================================================== dispatch tables

def _ready(result: Any) -> Generator[Event, Any, Any]:
    """A generator that yields nothing and returns ``result``.

    Every eval/place/exec handler must hand back a generator (callers ``yield from``
    it); handlers that never pause, such as literals, wrap their answer with this.
    """
    return result
    yield  # unreachable, but it makes this function a generator


def _unlabelled(stmt: ast.Node) -> ast.Node:
    """The statement under any ``case``/``default`` labels in front of it."""
    while isinstance(stmt, (ast.CaseLabel, ast.DefaultLabel)):
        stmt = stmt.body
    return stmt


def _truth(value: Value | StructValue) -> bool:
    """C truth of a condition; plain integers (nearly every condition) skip ``truthy``."""
    if type(value) is Value and type(value.ctype) is IntType:
        return value.raw != 0
    return truthy(value)


def _involves_pointer(expr: ast.BinaryOp) -> bool:
    """Is either operand of ``expr`` a pointer (so ``+``/``-`` mean pointer arithmetic)?"""
    left, right = ast.expression_type(expr.left), ast.expression_type(expr.right)
    return bool(left and left.is_pointer) or bool(right and right.is_pointer)


def _is_pointer_offset(expr: ast.BinaryOp) -> bool:
    """``p + n``, ``n + p`` or ``p - n`` (not ``p - q``)."""
    left, right = ast.expression_type(expr.left), ast.expression_type(expr.right)
    if left is None or right is None:
        return False
    if left.is_pointer and right.is_integer:
        return True
    return expr.op == "+" and left.is_integer and right.is_pointer


_EXEC: dict[type, Callable[[Interpreter, Any], Generator[Event, Any, Flow]]] = {
    ast.CompoundStmt: Interpreter.exec_compound,
    ast.ExprStmt: Interpreter.exec_expr,
    ast.DeclStmt: Interpreter.exec_decl,
    ast.EmptyStmt: Interpreter.exec_empty,
    ast.IfStmt: Interpreter.exec_if,
    ast.WhileStmt: Interpreter.exec_while,
    ast.DoWhileStmt: Interpreter.exec_do_while,
    ast.ForStmt: Interpreter.exec_for,
    ast.ReturnStmt: Interpreter.exec_return,
    ast.BreakStmt: Interpreter.exec_break,
    ast.ContinueStmt: Interpreter.exec_continue,
    ast.SwitchStmt: Interpreter.exec_switch,
    ast.CaseLabel: Interpreter.exec_label,
    ast.DefaultLabel: Interpreter.exec_label,
}

_PLACE: dict[type, Callable[[Interpreter, Any], Generator[Event, Any, Place]]] = {
    ast.Identifier: Interpreter.place_identifier,
    ast.StringLiteral: Interpreter.place_string,
    ast.Index: Interpreter.place_index,
    ast.Member: Interpreter.place_member,
    ast.UnaryOp: Interpreter.place_unary,
}

_EVAL: dict[type, Callable[[Interpreter, Any], Gen]] = {
    ast.IntLiteral: Interpreter.eval_int_literal,
    ast.FloatLiteral: Interpreter.eval_float_literal,
    ast.CharLiteral: Interpreter.eval_char_literal,
    ast.StringLiteral: Interpreter.eval_string,
    ast.Identifier: Interpreter.eval_identifier,
    ast.BinaryOp: Interpreter.eval_binary,
    ast.UnaryOp: Interpreter.eval_unary,
    ast.PostfixOp: Interpreter.eval_postfix,
    ast.Assignment: Interpreter.eval_assignment,
    ast.Call: Interpreter.eval_call,
    ast.Index: Interpreter.eval_place,
    ast.Member: Interpreter.eval_member,
    ast.Cast: Interpreter.eval_cast,
    ast.SizeofType: Interpreter.eval_sizeof_type,
    ast.SizeofExpr: Interpreter.eval_sizeof_expr,
    ast.Conditional: Interpreter.eval_conditional,
    ast.Comma: Interpreter.eval_comma,
    ast.InitList: Interpreter.eval_init_list,
    ast.ImplicitCast: Interpreter.eval_implicit_cast,
}


# =========================================================================== test host

class UnexpectedDeviceCall(RuntimeError):
    """``run_to_completion`` met a device call but was given no device."""


class CompletedRun(NamedTuple):
    """What ``run_to_completion`` returns (unpacks as a 4-tuple)."""

    exit_code: int
    stdout: str
    stderr: str
    summary: RunSummary


def run_to_completion(program: LinkedProgram, builtins: BuiltinRegistry,
                      argv: list[str] | None = None, stdin_text: str = "",
                      io: ProgramIO | None = None, *,
                      device: Callable[[DeviceCall], Any] | None = None,
                      on_event: Callable[[Event], None] | None = None,
                      budget: int = 1_000_000, seed: int = 0) -> CompletedRun:
    """Drive a whole run as a headless host does (tests and tools).

    ``InputRequest`` is answered line by line from ``stdin_text`` (then end of file),
    ``Output`` is collected, ``DeviceCall`` goes to ``device`` (without one it raises
    ``UnexpectedDeviceCall``) and ``on_event`` sees every event.
    """
    interpreter = Interpreter(program, builtins, budget=budget, seed=seed, io=io)
    builder = SummaryBuilder(program_hash(program.units))
    stdin = StringSource(stdin_text)
    generator = interpreter.run(list(argv) if argv else None)
    reply: Any = None
    while True:
        try:
            event = generator.send(reply)
        except StopIteration as done:
            exit_code = int(done.value)
            break
        builder.feed(event)
        if on_event is not None:
            on_event(event)
        reply = None
        if isinstance(event, InputRequest):
            reply = stdin.readline()
        elif isinstance(event, DeviceCall):
            if device is None:
                generator.close()
                raise UnexpectedDeviceCall(f"the program called {event.name}() but no device "
                                            "was given to run_to_completion")
            reply = device(event)
    summary = builder.finish(exit_code, interpreter)
    return CompletedRun(exit_code, summary.stdout, summary.stderr, summary)

