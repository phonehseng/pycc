"""The abstract syntax tree (AST) the parser builds and every later phase reads.

Each node is a small dataclass with a ``loc`` (where it starts in the user's file).
Declarations carry *resolved* C types from ``interp.c_types``: the parser already knows
that ``Tile *grid[4]`` is "array of 4 pointers to struct Tile", so the checker and the
interpreter never re-parse type syntax. Expressions carry no types; the checker and
the interpreter compute those.

Design notes a reviewer may wonder about:

* Parentheses do not get a node. ``(a = b)`` is just an Assignment whose
  ``parenthesized`` flag is set; the checker needs that flag to stay quiet about
  ``if ((c = next()))`` the way gcc does.
* ``case``/``default`` follow the C grammar: a label owns the statement right after it
  (``case 1: case 2: x++;`` nests). Fall-through is the job of the interpreter.
* Enum constants stay ``Identifier`` nodes; consumers look them up in
  ``EnumDecl.constants`` (after checking, ``Identifier.ref`` carries the value too).

Annotations added by the checker (interp/checker.py). They are
plain attributes, not constructor fields, so the parser never has to fill them:

* ``Expr.ctype``: the expression's C type after lvalue conversion and before
  array/function decay (``int[8]`` for an array name). ``IntLiteral``, ``FloatLiteral``
  and ``Cast`` already hold it as a field. ``SizeofType.ctype`` is the exception: it is
  the type being measured, and the expression itself is a ``size_t``; use
  ``expression_type()`` when you need the value type of any expression.
* ``Identifier.ref``: an ``interp.symbols.SymbolRef`` saying what the name means.
* ``ImplicitCast`` nodes wrap operands wherever C converts silently (assignment,
  arguments, return, arithmetic conversions, array/function decay).
* ``FunctionDef.storage_key``, ``FunctionDecl.storage_key`` and ``VarDecl.storage_key``:
  the storage key from interp/symbols.py.

Helpers: ``to_dict()`` on every node (for tools and saves), ``walk(node)`` to visit a
tree and ``dump(node)`` to print it as an indented outline.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields

from pycc.core.diagnostics import SourceLocation
from pycc.interp.c_types import SIZE_T, CType, EnumType, FunctionType, StructType, format_type


@dataclass
class Node:
    """Base class of every AST node."""

    loc: SourceLocation

    def children(self) -> list[Node]:
        """Direct child nodes, in source order."""
        found: list[Node] = []
        for item in fields(self):
            _collect_nodes(getattr(self, item.name), found)
        return found

    def to_dict(self) -> dict:
        """A JSON-friendly dictionary: node name, location and every field."""
        result: dict = {"node": type(self).__name__}
        for item in fields(self):
            result[item.name] = _to_plain(getattr(self, item.name))
        return result


def _collect_nodes(value: object, found: list[Node]) -> None:
    if isinstance(value, Node):
        found.append(value)
    elif isinstance(value, (list, tuple)):
        for element in value:
            _collect_nodes(element, found)


def _to_plain(value: object) -> object:
    """Turn node field values into plain JSON-style data."""
    if isinstance(value, Node):
        return value.to_dict()
    if isinstance(value, SourceLocation):
        return {"file": value.file, "line": value.line, "column": value.column}
    if isinstance(value, CType):
        return format_type(value)
    if isinstance(value, bytes):
        return value.decode("latin-1")
    if isinstance(value, (list, tuple)):
        return [_to_plain(element) for element in value]
    return value


# --- expressions --------------------------------------------------------------------

@dataclass
class Expr(Node):
    """Base class of expressions. ``parenthesized`` records '( ... )' around it."""

    parenthesized: bool = field(default=False, kw_only=True)


@dataclass
class IntLiteral(Expr):
    """An integer constant; ``ctype`` is the type C gives it (2147483648 is a long)."""

    value: int
    ctype: CType
    text: str = ""


@dataclass
class FloatLiteral(Expr):
    """A floating constant; ``ctype`` is double, or float with an 'f' suffix."""

    value: float
    ctype: CType
    text: str = ""


@dataclass
class CharLiteral(Expr):
    """A character constant such as 'a'; in C its type is int."""

    value: int
    text: str = ""


@dataclass
class StringLiteral(Expr):
    """A string literal; adjacent literals are already joined. No final NUL in ``value``."""

    value: bytes
    text: str = ""


@dataclass
class Identifier(Expr):
    """A name: a variable, a function or an enum constant."""

    name: str


@dataclass
class BinaryOp(Expr):
    """``left op right`` for arithmetic, comparison, bitwise and logical operators."""

    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    """Prefix operators: - + ! ~ * & and prefix ++/--."""

    op: str
    operand: Expr


@dataclass
class PostfixOp(Expr):
    """Postfix ++ and --."""

    op: str
    operand: Expr


@dataclass
class Assignment(Expr):
    """``target op value`` where op is '=' or a compound one like '+='."""

    op: str
    target: Expr
    value: Expr


@dataclass
class Call(Expr):
    """A function call ``func(args)``."""

    func: Expr
    args: list[Expr] = field(default_factory=list)


@dataclass
class Index(Expr):
    """Array subscript ``base[index]``."""

    base: Expr
    index: Expr


@dataclass
class Member(Expr):
    """Member access: ``base.name`` or, with ``arrow``, ``base->name``."""

    base: Expr
    name: str
    arrow: bool = False


@dataclass
class Cast(Expr):
    """``(ctype) expr``."""

    ctype: CType
    expr: Expr


@dataclass
class SizeofType(Expr):
    """``sizeof(type)``."""

    ctype: CType


@dataclass
class SizeofExpr(Expr):
    """``sizeof expr``; the expression is never evaluated."""

    expr: Expr


@dataclass
class Conditional(Expr):
    """``cond ? then : otherwise``."""

    cond: Expr
    then: Expr
    otherwise: Expr


@dataclass
class Comma(Expr):
    """``a, b, c``: evaluate in order, the value is the last one."""

    exprs: list[Expr] = field(default_factory=list)


@dataclass
class InitList(Expr):
    """A brace initializer ``{1, 2, {3, 4}}``; items may be nested InitLists."""

    items: list[Expr] = field(default_factory=list)


@dataclass
class ImplicitCast(Expr):
    """A conversion C performs without being asked, made visible by the checker.

    ``expr`` is converted to ``to``: ``double d = 3`` stores ``ImplicitCast(3, double)``,
    ``char + int`` promotes the char, and an array name passed to a function becomes
    ``ImplicitCast(name, pointer to element)``. The parser never creates these.
    """

    expr: Expr
    to: CType
    ctype: CType | None = None

    def __post_init__(self) -> None:
        if self.ctype is None:
            self.ctype = self.to


# Checker annotations (see the module docstring). Assigned after the dataclasses above
# are built, on purpose: as class attributes they give every node a default of None
# without becoming constructor fields of each expression class.
Expr.ctype = None  # type: ignore[attr-defined]
Identifier.ref = None  # type: ignore[attr-defined]


def expression_type(expr: Expr) -> CType | None:
    """The type of the value ``expr`` produces (None before checking).

    Only ``sizeof(type)`` needs care: its ``ctype`` field is the measured type.
    """
    if isinstance(expr, SizeofType):
        return SIZE_T
    return getattr(expr, "ctype", None)


# --- statements ---------------------------------------------------------------------

@dataclass
class Stmt(Node):
    """Base class of statements."""


@dataclass
class CompoundStmt(Stmt):
    """A block ``{ ... }``: declarations and statements in order.

    ``end_loc`` is where the closing '}' is (None if it was missing); gcc reports
    "control reaches end of non-void function" there.
    """

    items: list[Node] = field(default_factory=list)
    end_loc: SourceLocation | None = field(default=None, kw_only=True)


@dataclass
class ExprStmt(Stmt):
    """An expression followed by ';'."""

    expr: Expr


@dataclass
class DeclStmt(Stmt):
    """Declarations inside a block (VarDecl, StructDecl, TypedefDecl, ...)."""

    decls: list[Node] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    """``if (cond) then else otherwise``; ``otherwise`` is None without an else."""

    cond: Expr
    then: Stmt
    otherwise: Stmt | None = None


@dataclass
class WhileStmt(Stmt):
    """``while (cond) body``."""

    cond: Expr
    body: Stmt


@dataclass
class DoWhileStmt(Stmt):
    """``do body while (cond);``."""

    body: Stmt
    cond: Expr


@dataclass
class ForStmt(Stmt):
    """``for (init; cond; step) body``. ``init`` is a DeclStmt, an Expr or None."""

    init: Node | None
    cond: Expr | None
    step: Expr | None
    body: Stmt


@dataclass
class ReturnStmt(Stmt):
    """``return;`` or ``return value;``."""

    value: Expr | None = None


@dataclass
class BreakStmt(Stmt):
    """``break;``."""


@dataclass
class ContinueStmt(Stmt):
    """``continue;``."""


@dataclass
class SwitchStmt(Stmt):
    """``switch (cond) body``."""

    cond: Expr
    body: Stmt


@dataclass
class CaseLabel(Stmt):
    """``case value: body``. ``const_value`` is the folded integer (None if not constant)."""

    value: Expr
    const_value: int | None
    body: Stmt


@dataclass
class DefaultLabel(Stmt):
    """``default: body``."""

    body: Stmt


@dataclass
class EmptyStmt(Stmt):
    """A lone ';'."""


# --- declarations -------------------------------------------------------------------

@dataclass
class Decl(Node):
    """Base class of declarations."""


@dataclass
class VarDecl(Decl):
    """A variable. ``storage`` is 'static', 'extern', 'register', 'auto' or None.

    C qualifiers are not part of the type model, so they live here: ``is_const`` means
    the variable itself (or, for an array, its elements) may not be assigned, and
    ``points_to_const`` means it is a pointer to const data (``const char *s``).
    """

    name: str
    ctype: CType
    init: Expr | None = None
    storage: str | None = None
    is_extern: bool = False
    is_const: bool = False
    points_to_const: bool = False


@dataclass
class FunctionDecl(Decl):
    """A prototype without a body, such as ``int add(int a, int b);``."""

    name: str
    ftype: FunctionType
    storage: str | None = None
    param_names: list[str | None] = field(default_factory=list)
    is_inline: bool = False


@dataclass
class FunctionDef(Decl):
    """A function with its body. ``param_names[i]`` names ``ftype.params[i]``."""

    name: str
    ftype: FunctionType
    param_names: list[str | None]
    body: CompoundStmt
    storage: str | None = None
    is_inline: bool = False
    param_is_const: list[bool] = field(default_factory=list)
    param_locs: list[SourceLocation] = field(default_factory=list)


@dataclass
class StructDecl(Decl):
    """A struct or union definition (``is_definition``) or a forward ``struct S;``."""

    tag: str | None
    ctype: StructType
    is_definition: bool = True


@dataclass
class EnumDecl(Decl):
    """An enum definition with its folded ``(name, value)`` constants."""

    tag: str | None
    ctype: EnumType
    constants: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class TypedefDecl(Decl):
    """``typedef ctype name;``."""

    name: str
    ctype: CType
    is_const: bool = False


# The storage key the checker gives functions and variables (interp/symbols.py).
VarDecl.storage_key = None  # type: ignore[attr-defined]
FunctionDecl.storage_key = None  # type: ignore[attr-defined]
FunctionDef.storage_key = None  # type: ignore[attr-defined]


@dataclass
class TranslationUnit(Node):
    """One source file after preprocessing: its top-level declarations."""

    name: str
    decls: list[Decl] = field(default_factory=list)


# --- tree helpers -------------------------------------------------------------------

def walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every node below it, parents before children.

    An explicit stack instead of recursion: 'a + a + ... + a' with a few thousand
    terms is a tree thousands of levels deep, and nested generators would hit
    Python's recursion limit.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children()))  # reversed: leftmost child first


def describe(node: Node) -> str:
    """One line describing a node (its name plus its most telling fields)."""
    name = type(node).__name__
    label = _NODE_LABELS.get(type(node))
    return f"{name} {label(node)}".rstrip() if label else name


def dump(node: Node, indent: int = 0) -> str:
    """An indented outline of the tree, one node per line; handy when debugging."""
    lines: list[str] = []
    stack = [(node, indent)]  # an explicit stack, for the same reason as in walk()
    while stack:
        current, depth = stack.pop()
        lines.append("  " * depth + describe(current))
        stack.extend((child, depth + 1) for child in reversed(current.children()))
    return "\n".join(lines)


def _type_text(ctype: CType) -> str:
    return format_type(ctype)


_NODE_LABELS = {
    TranslationUnit: lambda n: repr(n.name),
    IntLiteral: lambda n: f"{n.value} ({_type_text(n.ctype)})",
    FloatLiteral: lambda n: f"{n.value} ({_type_text(n.ctype)})",
    CharLiteral: lambda n: n.text or str(n.value),
    StringLiteral: lambda n: repr(n.value),
    Identifier: lambda n: n.name,
    BinaryOp: lambda n: n.op,
    UnaryOp: lambda n: n.op,
    PostfixOp: lambda n: n.op,
    Assignment: lambda n: n.op,
    Member: lambda n: ("->" if n.arrow else ".") + n.name,
    Cast: lambda n: f"({_type_text(n.ctype)})",
    ImplicitCast: lambda n: f"-> {_type_text(n.to)}",
    SizeofType: lambda n: f"({_type_text(n.ctype)})",
    CaseLabel: lambda n: str(n.const_value),
    VarDecl: lambda n: f"{n.name}: {_type_text(n.ctype)}"
                       + (f" [{n.storage}]" if n.storage else ""),
    FunctionDecl: lambda n: f"{n.name}: {_type_text(n.ftype)}",
    FunctionDef: lambda n: f"{n.name}: {_type_text(n.ftype)}",
    StructDecl: lambda n: _type_text(n.ctype),
    EnumDecl: lambda n: f"{_type_text(n.ctype)} {n.constants}",
    TypedefDecl: lambda n: f"{n.name} = {_type_text(n.ctype)}",
}
