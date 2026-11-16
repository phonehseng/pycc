"""Call frames, places and control flow: the bookkeeping behind the interpreter.

The interpreter (interp/interpreter.py) walks the checked AST. This module holds the
small pieces of state it walks with, kept apart so each one can be read on its own:

* ``Place``: where an lvalue lives (its type, its address and, for ``a[i]`` and
  ``p->x``, the address the pointer started from, so an overrun that jumps clean
  over the gap between two variables is still blamed on the right array);
* ``CallFrame``: one active C function, with a stack of block scopes mapping names to
  ``Slot``s (C allows ``int x`` in an inner block to hide an outer ``x``);
* ``BREAK``, ``CONTINUE`` and ``Returned``: how a statement tells the loop or the
  function around it that it did not simply finish;
* ``SwitchTable``: where each ``case`` label of a switch sits, so execution can jump
  straight to it and fall through from there;
* ``FrameInfo`` and ``VarInfo``: the read-only picture of the stack the Variables
  panel and the memory view of a debugger ask for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from pycc.core.diagnostics import SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.c_types import CType
from pycc.interp.memory import Frame
from pycc.interp.values import StructValue, Value

# --- inspection records ---------------------------------------------------------------


@dataclass(frozen=True)
class VarInfo:
    """A variable the user can look at: its name, type and address."""

    name: str
    ctype: CType
    addr: int


@dataclass(frozen=True)
class FrameInfo:
    """One active function call. ``id`` is the memory ``Frame.id`` (docs D9)."""

    id: int
    name: str
    loc: SourceLocation | None
    locals: list[VarInfo]


# --- places -----------------------------------------------------------------------------


class Place(NamedTuple):
    """An lvalue: an object of ``ctype`` at ``addr``.

    ``origin`` is the pointer the address was computed from (``a`` in ``a[i]``), or
    None when it is not known; memory checks use it as provenance. A named tuple
    rather than a frozen dataclass because the interpreter makes one for nearly
    every variable it touches, and tuples are several times cheaper to create.
    """

    ctype: CType
    addr: int
    origin: int | None = None


# --- frames -------------------------------------------------------------------------------


@dataclass
class Slot:
    """A named variable in a block scope."""

    name: str
    ctype: CType
    addr: int


@dataclass
class CallFrame:
    """One active C function call."""

    function: ast.FunctionDef
    memory_frame: Frame
    call_site: SourceLocation | None
    scopes: list[dict[str, Slot]] = field(default_factory=lambda: [{}])
    location: SourceLocation | None = None      # the statement this frame is running

    @property
    def name(self) -> str:
        """The function's name."""
        return self.function.name

    def bind(self, slot: Slot) -> None:
        """Declare ``slot`` in the innermost block scope."""
        self.scopes[-1][slot.name] = slot

    def lookup(self, name: str) -> Slot | None:
        """The innermost visible variable called ``name``."""
        for scope in reversed(self.scopes):
            slot = scope.get(name)
            if slot is not None:
                return slot
        return None

    def visible_slots(self) -> list[Slot]:
        """Every variable in scope, outer blocks first (shadowed ones included)."""
        return [slot for scope in self.scopes for slot in scope.values()]

    def info(self) -> FrameInfo:
        """The read-only picture of this frame."""
        locals_ = [VarInfo(s.name, s.ctype, s.addr) for s in self.visible_slots()]
        return FrameInfo(self.memory_frame.id, self.name, self.location, locals_)


# --- control flow ----------------------------------------------------------------------------


class _Jump:
    """A ``break`` or ``continue`` travelling outwards to its loop or switch."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


BREAK = _Jump("BREAK")
CONTINUE = _Jump("CONTINUE")


@dataclass(frozen=True)
class Returned:
    """A ``return`` travelling outwards to its function, carrying the value."""

    value: Value | StructValue | None


# A statement finishes with None (carry on), BREAK, CONTINUE or a Returned.
Flow = _Jump | Returned | None


# --- switch tables ------------------------------------------------------------------------------


@dataclass
class SwitchTable:
    """Paths from a switch body down to each of its labels.

    A path lists the nodes from the body to the label: ``case`` labels may sit
    inside nested blocks (``{ case 1: ... }``) or inside other labels
    (``case 1: case 2: x++;`` nests the second label in the first).
    """

    cases: dict[int, list[ast.Node]] = field(default_factory=dict)
    default: list[ast.Node] | None = None

    def path_for(self, value: int) -> list[ast.Node] | None:
        """The path to the label that ``value`` selects (``default`` when no case does)."""
        return self.cases.get(value, self.default)


def build_switch_table(switch: ast.SwitchStmt) -> SwitchTable:
    """Find every label of ``switch`` (not those of switches nested inside it)."""
    table = SwitchTable()
    stack: list[tuple[ast.Node, list[ast.Node]]] = [(switch.body, [switch.body])]
    while stack:
        node, path = stack.pop()
        if isinstance(node, ast.CaseLabel):
            if node.const_value is not None:
                table.cases.setdefault(node.const_value, path)
            stack.append((node.body, path + [node.body]))
        elif isinstance(node, ast.DefaultLabel):
            if table.default is None:
                table.default = path
            stack.append((node.body, path + [node.body]))
        elif isinstance(node, ast.CompoundStmt):
            for item in reversed(node.items):
                stack.append((item, path + [item]))
        # Labels inside if/while/for bodies (Duff's device) are not searched: the
        # supported subset does not need them.
    return table


def index_of(items: list[ast.Node], child: ast.Node) -> int:
    """Position of ``child`` in ``items`` by identity (dataclass ``==`` compares deeply)."""
    for position, item in enumerate(items):
        if item is child:
            return position
    raise ValueError("node is not an item of this block")


def strip_implicit(expr: ast.Expr) -> ast.Expr:
    """The expression under any checker-inserted ``ImplicitCast`` wrappers."""
    while isinstance(expr, ast.ImplicitCast):
        expr = expr.expr
    return expr


def is_lvalue(expr: ast.Expr) -> bool:
    """Does ``expr`` designate an object in memory (same rule as the checker)?"""
    if isinstance(expr, ast.Identifier):
        ref = getattr(expr, "ref", None)
        if ref is None or ref.kind in ("function", "enum_const"):
            return False
        return not ref.ctype.is_function
    if isinstance(expr, (ast.Index, ast.StringLiteral)):
        return True
    if isinstance(expr, ast.UnaryOp):
        return expr.op == "*"
    if isinstance(expr, ast.Member):
        return expr.arrow or is_lvalue(expr.base)
    return False
