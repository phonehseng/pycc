"""Arithmetic expansion: ``$(( expression ))``, the shell's integer calculator.

Scripts count with it (``i=$((i + 1))`` in a ``while`` loop), so a build script can
retry three times or number its log lines. Supported, as in bash, on whole numbers:

* numbers (``42``), variable names (``i``, with or without ``$``) and ``$?``/``$#``/
  ``$1``...; an unset or non-numeric variable counts as 0;
* ``+ - * / %`` (division truncates toward zero, like C), unary ``- + !``;
* comparisons ``< <= > >= == !=`` and ``&& ||``, which give 1 or 0;
* parentheses.

Assignment operators (``$((i += 1))``), ``++``, bit operators and ``**`` are not
supported; using one is a clear syntax error rather than a wrong number.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import NoReturn

Lookup = Callable[[str], str | None]

_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+)|(?P<name>\$?[A-Za-z_]\w*|\$[?#0-9])"
    r"|(?P<op>&&|\|\||<=|>=|==|!=|[-+*/%()<>!]))")
# Binary operators by precedence, loosest first (C and bash agree on this order).
_LEVELS: tuple[tuple[str, ...], ...] = (
    ("||",), ("&&",), ("==", "!="), ("<", "<=", ">", ">="), ("+", "-"), ("*", "/", "%"))


class ArithError(Exception):
    """A malformed expression or a division by zero; ``message`` is bash's wording."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def evaluate(expression: str, lookup: Lookup) -> int:
    """The value of ``expression``; raises ``ArithError`` like bash would fail."""
    tokens = _tokenize(expression)
    if not tokens:
        return 0                                   # $(( )) is 0 in bash
    parser = _Parser(expression, tokens, lookup)
    value = parser.expression(0)
    if parser.pos != len(tokens):
        parser.fail()
    return value


def _tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    text = expression.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None or match.end() == pos:
            bad = text[pos:].strip()
            raise ArithError(f"{expression.strip()}: syntax error: invalid arithmetic "
                             f"operator (error token is \"{bad}\")")
        tokens.append(match.group(match.lastgroup or "op"))
        pos = match.end()
    return tokens


class _Parser:
    """Precedence climbing over the token list."""

    def __init__(self, expression: str, tokens: list[str], lookup: Lookup) -> None:
        self.expression_text = expression.strip()
        self.tokens = tokens
        self.lookup = lookup
        self.pos = 0

    def fail(self) -> NoReturn:
        """Stop at the current token, quoting the rest the way bash does."""
        rest = " ".join(self.tokens[self.pos:])
        if not rest:
            raise ArithError(f"{self.expression_text}: syntax error: operand expected "
                             "(error token is \"\")")
        raise ArithError(f"{self.expression_text}: syntax error in expression "
                         f"(error token is \"{rest}\")")

    def peek(self) -> str:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def expression(self, level: int) -> int:
        """Parse operators of precedence ``level`` and tighter."""
        if level == len(_LEVELS):
            return self.unary()
        value = self.expression(level + 1)
        while self.peek() in _LEVELS[level]:
            operator = self.tokens[self.pos]
            self.pos += 1
            value = self.apply(operator, value, self.expression(level + 1))
        return value

    def unary(self) -> int:
        token = self.peek()
        if token in ("-", "+", "!"):
            self.pos += 1
            value = self.unary()
            return {"-": -value, "+": value, "!": int(not value)}[token]
        return self.primary()

    def primary(self) -> int:
        token = self.peek()
        if token == "(":
            self.pos += 1
            value = self.expression(0)
            if self.peek() != ")":
                self.fail()
            self.pos += 1
            return value
        if token.isdigit():
            self.pos += 1
            return int(token)
        if token and (token[0] == "$" or token[0].isalpha() or token[0] == "_"):
            self.pos += 1
            return _as_number(self.lookup(token.lstrip("$")))
        self.fail()

    def apply(self, operator: str, left: int, right: int) -> int:
        if operator in ("/", "%") and right == 0:
            raise ArithError(f"{self.expression_text}: division by 0 "
                                   f"(error token is \"{right}\")")
        results: dict[str, Callable[[], int]] = {
            "+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right,
            "/": lambda: _c_divide(left, right),
            "%": lambda: left - right * _c_divide(left, right),
            "<": lambda: int(left < right), "<=": lambda: int(left <= right),
            ">": lambda: int(left > right), ">=": lambda: int(left >= right),
            "==": lambda: int(left == right), "!=": lambda: int(left != right),
            "&&": lambda: int(bool(left and right)), "||": lambda: int(bool(left or right)),
        }
        return results[operator]()


def _c_divide(left: int, right: int) -> int:
    """Integer division toward zero, as in C and bash (Python's ``//`` floors)."""
    quotient = abs(left) // abs(right)
    return quotient if (left >= 0) == (right > 0) else -quotient


def _as_number(value: str | None) -> int:
    """A variable's value as a number (unset, empty or not a number: 0)."""
    text = (value or "").strip()
    sign = -1 if text.startswith("-") else 1
    digits = text.lstrip("+-")
    return sign * int(digits) if digits.isdigit() else 0


__all__ = ["ArithError", "evaluate"]
