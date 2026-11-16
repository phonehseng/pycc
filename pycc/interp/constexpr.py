"""Folding integer constant expressions at compile time.

C needs some numbers while compiling: array sizes (``int grid[ROWS * COLS]``), enum
values (``B = A + 1``) and case labels. ``eval_const_expr`` computes them from the
AST, or returns None when the expression is not a compile-time constant (it mentions a
variable, calls a function, divides by zero, ...).

We evaluate with real C typing, carrying a (value, type) pair, so ``-1u > 0`` is 1 and
``(unsigned char)300`` is 44, exactly what gcc would compute.

``sizeof`` of an expression is folded when its type follows from declarations alone:
``sizeof grid / sizeof grid[0]`` works if the scope can tell us the type of ``grid``
(the optional ``object_type`` method; the parser's scopes have it).
"""
from __future__ import annotations

from typing import Protocol

from pycc.interp import astnodes as ast
from pycc.interp.c_types import (
    CHAR, DOUBLE, INT, ULONG, ArrayType, CType, PointerType, StructType, convert_value,
    integer_promote, usual_arithmetic_conversions, wrap_int,
)

_Typed = tuple[int, CType]


class ConstScope(Protocol):
    """Anything that can tell us the value of an enum constant (the parser's scopes).

    A scope may also offer ``object_type(name) -> CType | None``, the declared type of
    a variable. It is optional (looked up with getattr) so that simple scopes that only
    know enum constants keep working.
    """

    def enum_constant(self, name: str) -> int | None:
        """The value of enum constant ``name``, or None if it is not one."""
        ...


class DictScope:
    """A ConstScope backed by plain dictionaries (handy in tests and tools)."""

    def __init__(self, constants: dict[str, int] | None = None,
                 types: dict[str, CType] | None = None) -> None:
        self.constants = dict(constants or {})
        self.types = dict(types or {})

    def enum_constant(self, name: str) -> int | None:
        """Look the name up in the constants dictionary."""
        return self.constants.get(name)

    def object_type(self, name: str) -> CType | None:
        """Look the name up in the variable types dictionary."""
        return self.types.get(name)


def eval_const_expr(expr: ast.Expr, scope: ConstScope | None = None) -> int | None:
    """The integer value of ``expr`` if it is a constant expression, else None."""
    typed = _eval(expr, scope)
    return None if typed is None else typed[0]


def _eval(expr: ast.Expr, scope: ConstScope | None) -> _Typed | None:
    handler = _HANDLERS.get(type(expr))
    return handler(expr, scope) if handler is not None else None


def _literal(expr: ast.IntLiteral, scope: ConstScope | None) -> _Typed:
    return expr.value, expr.ctype


def _char(expr: ast.CharLiteral, scope: ConstScope | None) -> _Typed:
    return expr.value, INT


def _identifier(expr: ast.Identifier, scope: ConstScope | None) -> _Typed | None:
    if scope is None:
        return None
    value = scope.enum_constant(expr.name)
    return None if value is None else (value, INT)


def _sizeof_type(expr: ast.SizeofType, scope: ConstScope | None) -> _Typed | None:
    if not expr.ctype.is_complete and not expr.ctype.is_function:
        return None  # sizeof(struct Unknown) is an error the checker reports
    return expr.ctype.size, ULONG


def _sizeof_expr(expr: ast.SizeofExpr, scope: ConstScope | None) -> _Typed | None:
    """sizeof of an expression whose type is clear without the checker."""
    ctype = _static_type(expr.expr, scope)
    if ctype is None or ctype.is_function or not ctype.is_complete:
        return None
    return ctype.size, ULONG


def _static_type(expr: ast.Expr, scope: ConstScope | None) -> CType | None:
    """The type of ``expr`` when declarations alone decide it, else None.

    This is a deliberately small subset of the checker's typing: names, literals,
    casts, a[i], *p, &x, s.m and p->m. sizeof never evaluates its operand, so knowing
    the type is all we need.
    """
    if isinstance(expr, ast.StringLiteral):
        return ArrayType(CHAR, len(expr.value) + 1)  # the hidden NUL counts
    if isinstance(expr, (ast.IntLiteral, ast.FloatLiteral)):
        return expr.ctype
    if isinstance(expr, ast.CharLiteral):
        return INT  # 'a' is an int in C, so sizeof 'a' is 4
    if isinstance(expr, (ast.Cast, ast.SizeofType)):
        return expr.ctype if isinstance(expr, ast.Cast) else ULONG
    if isinstance(expr, ast.SizeofExpr):
        return ULONG
    if isinstance(expr, ast.ImplicitCast):
        return expr.to
    if isinstance(expr, ast.Identifier):
        lookup = getattr(scope, "object_type", None)
        return lookup(expr.name) if lookup is not None else None
    return _static_derived_type(expr, scope)


def _static_derived_type(expr: ast.Expr, scope: ConstScope | None) -> CType | None:
    """Types reached through a[i], *p, &x, s.m and p->m."""
    if isinstance(expr, ast.Index):
        return _pointee(_static_type(expr.base, scope))
    if isinstance(expr, ast.UnaryOp) and expr.op == "*":
        return _pointee(_static_type(expr.operand, scope))
    if isinstance(expr, ast.UnaryOp) and expr.op == "&":
        inner = _static_type(expr.operand, scope)
        return None if inner is None else PointerType(inner)
    if isinstance(expr, ast.Member):
        base = _static_type(expr.base, scope)
        holder = _pointee(base) if expr.arrow else base
        if isinstance(holder, StructType):
            member = holder.member(expr.name)
            return None if member is None else member.ctype
    return None


def _pointee(ctype: CType | None) -> CType | None:
    """What an array or pointer holds: int[8] -> int, char * -> char."""
    if isinstance(ctype, ArrayType):
        return ctype.element
    if isinstance(ctype, PointerType):
        return ctype.target
    return None


def _cast(expr: ast.Cast, scope: ConstScope | None) -> _Typed | None:
    if not expr.ctype.is_integer:
        return None
    floating = _float_constant(expr.expr)
    if floating is not None:
        # (int)2.5 is allowed in a constant expression: an immediate float cast.
        return int(convert_value(floating, DOUBLE, expr.ctype)), expr.ctype
    inner = _eval(expr.expr, scope)
    if inner is None:
        return None
    return int(convert_value(inner[0], inner[1], expr.ctype)), expr.ctype


def _implicit_cast(expr: ast.ImplicitCast, scope: ConstScope | None) -> _Typed | None:
    """A conversion the checker made explicit folds like the cast it stands for."""
    if not expr.to.is_integer:
        return None
    inner = _eval(expr.expr, scope)
    if inner is None:
        return None
    return int(convert_value(inner[0], inner[1], expr.to)), expr.to


def _float_constant(expr: ast.Expr) -> float | None:
    """The value of a float literal, possibly signed (-2.5), or None."""
    if isinstance(expr, ast.FloatLiteral):
        return expr.value
    if isinstance(expr, ast.UnaryOp) and expr.op in ("-", "+"):
        inner = _float_constant(expr.operand)
        if inner is not None:
            return -inner if expr.op == "-" else inner
    return None


def _unary(expr: ast.UnaryOp, scope: ConstScope | None) -> _Typed | None:
    inner = _eval(expr.operand, scope)
    if inner is None:
        return None
    value, ctype = inner
    if expr.op == "!":
        return int(value == 0), INT
    ctype = integer_promote(ctype)
    if expr.op == "-":
        return wrap_int(-value, ctype), ctype
    if expr.op == "+":
        return value, ctype
    if expr.op == "~":
        return wrap_int(~value, ctype), ctype
    return None  # & * ++ -- are never constant integers


def _binary(expr: ast.BinaryOp, scope: ConstScope | None) -> _Typed | None:
    if expr.op in ("&&", "||"):
        return _logical(expr, scope)
    left = _eval(expr.left, scope)
    right = _eval(expr.right, scope)
    if left is None or right is None:
        return None
    if expr.op in ("<<", ">>"):
        return _shift(expr.op, left, right)
    common = usual_arithmetic_conversions(left[1], right[1])
    a = wrap_int(left[0], common)
    b = wrap_int(right[0], common)
    if expr.op in _COMPARISONS:
        return int(_COMPARISONS[expr.op](a, b)), INT
    result = _arithmetic(expr.op, a, b)
    return None if result is None else (wrap_int(result, common), common)


def _logical(expr: ast.BinaryOp, scope: ConstScope | None) -> _Typed | None:
    left = _eval(expr.left, scope)
    if left is None:
        return None
    # Short-circuit like C: '0 && x' is constant even if x is not.
    if expr.op == "&&" and left[0] == 0:
        return 0, INT
    if expr.op == "||" and left[0] != 0:
        return 1, INT
    right = _eval(expr.right, scope)
    return None if right is None else (int(right[0] != 0), INT)


def _shift(op: str, left: _Typed, right: _Typed) -> _Typed | None:
    ctype = integer_promote(left[1])
    count = right[0]
    if count < 0 or count >= ctype.bits:  # type: ignore[attr-defined]
        return None  # undefined behaviour: refuse to fold it
    value = left[0] << count if op == "<<" else left[0] >> count
    return wrap_int(value, ctype), ctype


def _arithmetic(op: str, a: int, b: int) -> int | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op in ("/", "%"):
        if b == 0:
            return None
        # C division truncates toward zero; Python's // floors, so fix the sign.
        quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
        return quotient if op == "/" else a - quotient * b
    if op == "&":
        return a & b
    if op == "|":
        return a | b
    if op == "^":
        return a ^ b
    return None


def _conditional(expr: ast.Conditional, scope: ConstScope | None) -> _Typed | None:
    cond = _eval(expr.cond, scope)
    if cond is None:
        return None
    then = _eval(expr.then, scope)
    otherwise = _eval(expr.otherwise, scope)
    if then is None or otherwise is None:
        return None
    common = usual_arithmetic_conversions(then[1], otherwise[1])
    chosen = then if cond[0] else otherwise
    return wrap_int(chosen[0], common), common


_COMPARISONS = {
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
}

_HANDLERS = {
    ast.IntLiteral: _literal,
    ast.CharLiteral: _char,
    ast.Identifier: _identifier,
    ast.SizeofType: _sizeof_type,
    ast.SizeofExpr: _sizeof_expr,
    ast.Cast: _cast,
    ast.ImplicitCast: _implicit_cast,
    ast.UnaryOp: _unary,
    ast.BinaryOp: _binary,
    ast.Conditional: _conditional,
}
