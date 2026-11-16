"""Static checks of printf- and scanf-style format strings (gcc's -Wformat).

``printf("%d\\n", name)`` compiles in C, then prints garbage, because nothing forces
the conversion (``%d``) to match the argument (a ``char *``). gcc catches it at compile
time when the format is a string literal, and so do we: the checker hands every call to
a registered format function (``BuiltinRegistry.format_info``) to ``check_format``.

The module knows nothing about the AST. It takes the format text and the types of the
arguments after the default argument promotions (a ``char`` arrives as an ``int``, a
``float`` as a ``double``) and returns ``FormatIssue`` records; the checker turns them
into ``format_mismatch`` warnings with locations.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pycc.interp.c_types import CType, IntType, PointerType, format_type

PRINTF = "printf"
SCANF = "scanf"

# Characters that may appear between '%' and the conversion character.
_FLAGS = "-+ #0"
_LENGTHS = ("hh", "ll", "h", "l", "L", "z", "j", "t")
_PRINTF_CONVERSIONS = frozenset("diouxXcspfFeEgGaAn%")
_SCANF_CONVERSIONS = frozenset("diouxXcspfFeEgGaAn%[")


def type_phrase(type_text: str, capital: bool = False) -> str:
    """"an 'int'" / "a 'double'": a quoted type name with the right article."""
    article = "an" if type_text[:1].lower() in "aeiou" else "a"
    if capital:
        article = article.capitalize()
    return f"{article} '{type_text}'"


@dataclass(frozen=True)
class Conversion:
    """One conversion specification such as ``%-5ld``.

    ``offset`` is the index of its '%' in the format text, ``char`` the conversion
    character, ``length`` the length modifier ('' when there is none).
    """

    text: str
    offset: int
    char: str
    length: str = ""
    star_width: bool = False
    star_precision: bool = False
    suppressed: bool = False        # scanf's '%*d' reads but stores nothing


@dataclass
class ParsedFormat:
    """The conversions of a format string plus problems with the string itself."""

    conversions: list[Conversion] = field(default_factory=list)
    # (offset of the '%', gcc message, friendly text) for each problem
    problems: list[tuple[int, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class FormatIssue:
    """One mismatch between a format string and the call's arguments.

    ``offset`` is the position of the offending conversion in the format text (or
    None for problems with the call as a whole); ``argument`` is the 0-based index of
    the variadic argument concerned, if any.
    """

    message: str
    friendly: str
    offset: int | None = None
    argument: int | None = None


def parse_format(text: str, kind: str = PRINTF) -> ParsedFormat:
    """Split a format string into its conversions (``%%`` is skipped)."""
    result = ParsedFormat()
    known = _SCANF_CONVERSIONS if kind == SCANF else _PRINTF_CONVERSIONS
    index = 0
    while True:
        start = text.find("%", index)
        if start < 0:
            return result
        conversion, index = _read_conversion(text, start, kind)
        if conversion is None:
            result.problems.append(_unfinished_problem(text, start))
            return result
        if conversion.char == "%":
            continue
        if conversion.char not in known:
            result.problems.append((start, f"unknown conversion type character "
                                           f"'{conversion.char}' in format",
                                    _unknown_friendly(conversion.char, kind)))
            continue
        result.conversions.append(conversion)


def _unfinished_problem(text: str, start: int) -> tuple[int, str, str]:
    """The problem for a '%' that the string ends inside of (gcc words the two cases apart)."""
    friendly = ("A format string cannot end with a lone '%'. Write '%%' to print a percent "
                "sign.")
    if start == len(text) - 1:
        return start, "spurious trailing '%' in format", friendly
    return (start, "conversion lacks type at end of format",
            f"'{text[start:]}' at the end of the format has no conversion letter, such as "
            "the 'd' in %ld. Finish it, or write '%%' to print a percent sign.")


def _read_conversion(text: str, start: int, kind: str) -> tuple[Conversion | None, int]:
    """Read the conversion that starts at ``text[start] == '%'``."""
    index = start + 1
    suppressed = star_width = star_precision = False
    if kind == SCANF and index < len(text) and text[index] == "*":
        suppressed = True
        index += 1
    while index < len(text) and text[index] in _FLAGS and kind == PRINTF:
        index += 1
    if index < len(text) and text[index] == "*" and kind == PRINTF:
        star_width = True
        index += 1
    while index < len(text) and text[index].isdigit():
        index += 1
    if index < len(text) and text[index] == "." and kind == PRINTF:
        index += 1
        if index < len(text) and text[index] == "*":
            star_precision = True
            index += 1
        while index < len(text) and text[index].isdigit():
            index += 1
    length = ""
    for candidate in _LENGTHS:
        if text.startswith(candidate, index):
            length = candidate
            index += len(candidate)
            break
    if index >= len(text):
        return None, index
    char = text[index]
    index += 1
    if char == "[":
        index = _skip_scanset(text, index)
    return Conversion(text[start:index], start, char, length, star_width, star_precision,
                      suppressed), index


def _skip_scanset(text: str, index: int) -> int:
    """Skip a scanf scanset such as ``[^\\n]``; ']' right after '[' or '[^' is literal."""
    if index < len(text) and text[index] == "^":
        index += 1
    if index < len(text) and text[index] == "]":
        index += 1
    closing = text.find("]", index)
    return len(text) if closing < 0 else closing + 1


def _unknown_friendly(char: str, kind: str) -> str:
    common = "%d, %f, %c and %s" if kind == SCANF else "%d, %f, %c, %s and %p"
    return (f"'%{char}' is not a conversion {kind} knows. Common ones are {common}; "
            "write '%%' for a percent sign.")


# --- what each conversion expects ---------------------------------------------------------

@dataclass(frozen=True)
class _Expectation:
    """What an argument must look like for one conversion."""

    description: str       # gcc's spelling: 'int', 'long unsigned int', 'char *'
    family: str            # int | float | string | pointer | int_pointer | float_pointer...
    size: int = 0          # the integer (or pointed-to) size that matches


_PRINTF_INT_NAMES = {"": "int", "hh": "int", "h": "int", "l": "long int",
                     "ll": "long long int", "z": "size_t", "j": "intmax_t", "t": "ptrdiff_t"}
_PRINTF_UINT_NAMES = {"": "unsigned int", "hh": "int", "h": "int", "l": "long unsigned int",
                      "ll": "long long unsigned int", "z": "size_t", "j": "uintmax_t",
                      "t": "ptrdiff_t"}
_SCANF_INT_NAMES = {"": "int", "hh": "signed char", "h": "short int", "l": "long int",
                    "ll": "long long int", "z": "size_t", "j": "intmax_t", "t": "ptrdiff_t"}
_SCANF_UINT_NAMES = {"": "unsigned int", "hh": "unsigned char", "h": "short unsigned int",
                     "l": "long unsigned int", "ll": "long long unsigned int", "z": "size_t",
                     "j": "uintmax_t", "t": "ptrdiff_t"}
_LENGTH_SIZES = {"": 4, "hh": 1, "h": 2, "l": 8, "ll": 8, "z": 8, "j": 8, "t": 8}


def expectation(conversion: Conversion, kind: str) -> _Expectation:
    """The argument type ``conversion`` expects, for printf or scanf."""
    if kind == SCANF:
        return _scanf_expectation(conversion)
    return _printf_expectation(conversion)


def _printf_expectation(conversion: Conversion) -> _Expectation:
    char, length = conversion.char, conversion.length
    if char in "di":
        # Narrow lengths (%hd, %hhd) still receive a promoted int.
        size = 4 if length in ("", "h", "hh") else 8
        return _Expectation(_PRINTF_INT_NAMES.get(length, "int"), "int", size)
    if char in "ouxX":
        size = 4 if length in ("", "h", "hh") else 8
        return _Expectation(_PRINTF_UINT_NAMES.get(length, "unsigned int"), "int", size)
    if char == "c":
        return _Expectation("int", "int", 4)
    if char == "s":
        return _Expectation("char *", "string", 1)
    if char == "p":
        return _Expectation("void *", "pointer")
    if char == "n":
        # %n stores the count so far; its length modifier sizes the int it points to.
        return _Expectation(_SCANF_INT_NAMES.get(length, "int") + " *", "int_pointer",
                            _LENGTH_SIZES.get(length, 4))
    long_double = length == "L"
    return _Expectation("long double" if long_double else "double", "float")


def _scanf_expectation(conversion: Conversion) -> _Expectation:
    char, length = conversion.char, conversion.length
    size = _LENGTH_SIZES.get(length, 4)
    if char in "din":
        return _Expectation(_SCANF_INT_NAMES.get(length, "int") + " *", "int_pointer", size)
    if char in "ouxX":
        return _Expectation(_SCANF_UINT_NAMES.get(length, "unsigned int") + " *", "int_pointer",
                            size)
    if char in "cs[":
        return _Expectation("char *", "string", 1)
    if char == "p":
        return _Expectation("void **", "pointer_pointer")
    if length == "l":
        return _Expectation("double *", "float_pointer", 8)
    if length == "L":
        return _Expectation("long double *", "float_pointer", 8)
    return _Expectation("float *", "float_pointer", 4)


def matches(expected: _Expectation, actual: CType) -> bool:
    """Does an argument of type ``actual`` (already promoted) fit ``expected``?"""
    family = expected.family
    if family == "int":
        return actual.is_integer and actual.size == expected.size
    if family == "float":
        return actual.is_float
    if family == "pointer":
        return actual.is_pointer
    if not isinstance(actual, PointerType):
        return False
    target = actual.target
    if family == "string":
        return isinstance(target, IntType) and target.size == 1
    if family == "int_pointer":
        return target.is_integer and target.size == expected.size
    if family == "float_pointer":
        return target.is_float and target.size == expected.size
    if family == "pointer_pointer":
        return target.is_pointer
    return False


# --- the check ----------------------------------------------------------------------------

def check_format(kind: str, text: str, arguments: Sequence[CType], first_number: int,
                 argument_names: Sequence[str | None] | None = None) -> list[FormatIssue]:
    """Compare a format string with the types of the arguments that follow it.

    ``arguments`` are the variadic arguments' types after default promotions;
    ``first_number`` is the 1-based position of the first of them in the call (gcc
    counts the format string itself, so for ``printf`` it is 2). ``argument_names``
    optionally gives the source text of each argument (a variable name), used to make
    the scanf "forgot the &" hint concrete.
    """
    parsed = parse_format(text, kind)
    issues = [FormatIssue(message, friendly, offset)
              for offset, message, friendly in parsed.problems]
    names = list(argument_names or [])
    position = 0
    for conversion in parsed.conversions:
        for star in _star_arguments(conversion):
            issue = _check_star(star, conversion, arguments, position, first_number)
            if issue is not None:
                issues.append(issue)
            position += 1
        if conversion.suppressed:
            continue
        expected = expectation(conversion, kind)
        if position >= len(arguments):
            issues.append(FormatIssue(
                f"format '{conversion.text}' expects a matching '{expected.description}' "
                "argument",
                f"The format has more conversions than arguments: {conversion.text} has no "
                "value to go with it. Add an argument or remove the conversion.",
                conversion.offset))
            position += 1
            continue
        actual = arguments[position]
        if not matches(expected, actual):
            name = names[position] if position < len(names) else None
            issues.append(FormatIssue(
                f"format '{conversion.text}' expects argument of type '{expected.description}', "
                f"but argument {first_number + position} has type '{format_type(actual)}'",
                _mismatch_friendly(kind, conversion, expected, actual, first_number + position,
                                   name),
                conversion.offset, position))
        position += 1
    if position < len(arguments):
        issues.append(FormatIssue(
            "too many arguments for format",
            "There are more arguments than conversions in the format, so the extra ones are "
            "ignored. Add a conversion such as %d for each value, or remove the extra "
            "arguments.", None, position))
    return issues


def _star_arguments(conversion: Conversion) -> list[str]:
    """'%*.*f' takes two extra int arguments (width, then precision) before its value."""
    stars = []
    if conversion.star_width:
        stars.append("field width")
    if conversion.star_precision:
        stars.append("field precision")
    return stars


def _check_star(star: str, conversion: Conversion, arguments: Sequence[CType], position: int,
                first_number: int) -> FormatIssue | None:
    if position >= len(arguments):
        return FormatIssue(f"{star} specifier '*' expects a matching 'int' argument",
                           f"The '*' in {conversion.text} takes the {star} from an extra int "
                           "argument, but there is none. Add it before the value.",
                           conversion.offset)
    actual = arguments[position]
    if actual.is_integer and actual.size == 4:
        return None
    return FormatIssue(f"{star} specifier '*' expects argument of type 'int', but argument "
                       f"{first_number + position} has type '{format_type(actual)}'",
                       f"The '*' in {conversion.text} reads the {star} from an int argument.",
                       conversion.offset, position)


def suggest_conversion(actual: CType, kind: str = PRINTF) -> str | None:
    """The conversion that fits a value of type ``actual`` (for printf), if obvious."""
    if kind == SCANF:
        return None
    if isinstance(actual, PointerType):
        target = actual.target
        if isinstance(target, IntType) and target.size == 1:
            return "%s"
        return "%p"
    if actual.is_float:
        return "%f"
    if actual.is_integer:
        signed = getattr(actual, "signed", True)
        if actual.size == 8:
            return "%ld" if signed else "%lu"
        return "%d" if signed else "%u"
    return None


def _mismatch_friendly(kind: str, conversion: Conversion, expected: _Expectation,
                       actual: CType, number: int, name: str | None) -> str:
    actual_text = format_type(actual)
    if kind == SCANF and expected.family != "pointer" and not actual.is_pointer:
        target = f"&{name}" if name else "&variable"
        plain = name or "variable"
        return (f"scanf needs to know where to store what it reads, so it takes addresses. "
                f"Pass '{target}' instead of '{plain}' (argument {number}).")
    if kind == SCANF:
        return (f"{conversion.text} stores into {type_phrase(expected.description)}, but "
                f"argument {number} is {type_phrase(actual_text)}. Make the conversion match "
                "the variable's type.")
    text = (f"{conversion.text} prints {type_phrase(expected.description)}, but argument "
            f"{number} is {type_phrase(actual_text)}.")
    suggestion = suggest_conversion(actual)
    if suggestion is not None and suggestion != conversion.text:
        text += f" Use {suggestion} for {type_phrase(actual_text)}, or change the argument."
    if expected.family == "float" and actual.is_integer:
        text += (" printf does not turn an int into a double: write the number with a "
                 "decimal point (3.0) or cast it with (double).")
    return text
