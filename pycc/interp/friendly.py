"""Friendly explanations shared by the lexer, the parser and (later) the checker.

Every diagnostic in pycc has two texts. ``message`` copies gcc word for word, so
a user who later meets a real compiler recognises it. ``friendly`` is one or two short
sentences in plain English for a beginner. The friendly texts live here, in one table,
so that the tone stays consistent and the checker can reuse the same helpers.

We keep each friendly text short on purpose: a user who just broke their program
needs one clear hint, not a lecture.
"""
from __future__ import annotations

import difflib
from collections.abc import Iterable

from pycc.core.concepts import Concept
from pycc.core.diagnostics import (
    CODE_CONCEPTS,
    Diagnostic,
    Phase,
    Severity,
    SourceLocation,
)

# Codes the front end uses that are not (yet) in core's table. Each one names its
# concept explicitly, as every code outside CODE_CONCEPTS must.
EXTRA_CONCEPTS: dict[str, Concept] = {
    "empty_char_constant": Concept.STRINGS,
    "multichar_constant": Concept.STRINGS,
    "unsupported_feature": Concept.SYNTAX,
    "invalid_type_combination": Concept.VARIABLES,
    "enum_value_not_const": Concept.ENUMS,
    "case_not_constant": Concept.CONDITIONALS,
    "too_many_errors": Concept.SYNTAX,
    "useless_declaration": Concept.VARIABLES,
    "nesting_too_deep": Concept.SYNTAX,
    "enum_overflow": Concept.ENUMS,
}

# Plain-English templates, keyed by diagnostic code. Placeholders in braces are filled
# from keyword arguments; a missing placeholder is left visible rather than crashing,
# because a slightly odd hint is better than no compile result at all.
FRIENDLY_TEMPLATES: dict[str, str] = {
    # --- lexer ---------------------------------------------------------------
    "unterminated_string": "This string starts with '\"' but never ends. Add a closing '\"' "
                           "on the same line.",
    "unterminated_char": "This character constant starts with a single quote but never ends. "
                         "Add a closing ' right after the character.",
    "empty_char_constant": "Single quotes need exactly one character between them, like 'a'.",
    "multichar_constant": "Single quotes hold one character. For text with several characters, "
                          "use double quotes: \"{text}\".",
    "invalid_suffix": "The letters after the number ('{suffix}') are not a valid suffix. "
                      "Integers may end in u, l or ll; decimals in f.",
    "invalid_number": "{detail}",
    "stray_character": "The character {char} has no meaning in C here. Delete it.",
    "invalid_escape": "'\\{char}' is not an escape sequence C knows. Common ones are \\n, "
                      "\\t, \\\\ and \\0.",
    "unterminated_comment": "This comment starts with /* but never ends with */.",
    "integer_overflow_constant": "This number is too big for a signed type, so it becomes "
                                 "unsigned. Add a 'u' suffix if that is what you want.",
    # --- parser --------------------------------------------------------------
    "missing_semicolon": "Every statement and declaration in C ends with ';'. "
                         "Add one at the end of line {line}.",
    "expected_token": "C expected {expected} here, but found {found}.",
    "unexpected_token": "C expected a value or expression here, but found {found}.",
    "unknown_type": "'{name}' is not a type C knows.{hint}",
    "unbalanced_brace": "The '{{' on line {line} is never closed. Add a matching '}}' where the "
                        "block should end.",
    "unbalanced_paren": "A '(' on line {line} is never closed. Add the missing ')'.",
    "else_without_if_stmt": "This 'else' has no 'if' right before it. Check for a stray ';' "
                            "after the if's condition, or missing braces.",
    "missing_return_type": "Functions should say what they return. Write 'int {name}(...)' "
                           "instead of just '{name}(...)'.",
    "implicit_int": "This declaration has no type. Write the type first, for example "
                    "'int {name};'.",
    "bad_declarator": "{detail}",
    "bad_initializer": "{detail}",
    "nested_function": "'{name}' starts inside another function. A '}}' is probably missing "
                       "before it, to close the function above.",
    "empty_struct": "A struct needs at least one member inside its braces.",
    "unsupported_feature": "{detail}",
    "invalid_type_combination": "These type words do not combine. Pick one type, for example "
                                "'int' or 'unsigned long'.",
    "enum_value_not_const": "Enum values must be numbers known while compiling, like 3 or "
                            "A + 1.",
    "case_not_constant": "A case label must be a constant, like 3, 'x' or an enum name, "
                         "not a variable.",
    "array_size_not_const": "Array sizes must be constants known while compiling, like 8 or "
                            "a #define. Use malloc for sizes known only at run time.",
    "redefinition": "'{name}' is defined twice. Keep one definition.",
    "incomplete_type": "The full definition of '{name}' is not known here. Define it before "
                       "using it, or use a pointer to it.",
    "useless_declaration": "This line names a type but declares nothing. Add a name, "
                           "like 'int count;'.",
    "too_many_errors": "Fix the first errors first: later ones are often caused by them.",
    "enum_overflow": "Enum constants are ints, so '{name}' must be at most 2147483647. "
                     "Use smaller values or plain #define constants.",
    "nesting_too_deep": "This code nests brackets or blocks hundreds of levels deep. Split "
                        "it into smaller expressions or helper functions.",
}

# Words worth suggesting when a type name is misspelled.
TYPE_WORDS: tuple[str, ...] = (
    "int", "char", "short", "long", "float", "double", "void", "signed", "unsigned",
    "_Bool", "struct", "union", "enum", "const",
)

# Standard type names that only exist after an #include. A user who writes 'bool'
# needs a header, not a spelling correction.
HEADER_FOR_TYPE: dict[str, str] = {
    "bool": "stdbool.h",
    "size_t": "stddef.h",
    "FILE": "stdio.h",
    "int8_t": "stdint.h", "int16_t": "stdint.h", "int32_t": "stdint.h", "int64_t": "stdint.h",
    "uint8_t": "stdint.h", "uint16_t": "stdint.h", "uint32_t": "stdint.h",
    "uint64_t": "stdint.h",
}


class _KeepMissing(dict):
    """A format_map dictionary that leaves unknown ``{placeholders}`` untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def did_you_mean(name: str, candidates: Iterable[str], cutoff: float = 0.6) -> str | None:
    """Return the candidate closest to ``name``, or None if nothing is close enough.

    difflib's ratio is forgiving of swapped letters ('itn' vs 'int'), which are the
    typos beginners make most. A name that differs only in upper/lower case
    ('north' for 'NORTH') wins first: difflib does not see those as close at all.
    Names that start with two underscores belong to the library's insides
    (``__cv_assert``), so they are never suggested to a user unless the user
    wrote such a name too.
    """
    pool = set(candidates) - {name}
    if not name.startswith("__"):
        pool = {candidate for candidate in pool if not candidate.startswith("__")}
    unique = sorted(pool)
    twins = [candidate for candidate in unique if candidate.lower() == name.lower()]
    if twins:
        return twins[0]
    matches = difflib.get_close_matches(name, unique, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def friendly_text(code: str, **details: object) -> str:
    """Fill in the friendly template for ``code`` with the given details."""
    template = FRIENDLY_TEMPLATES.get(code, "{detail}")
    values = _KeepMissing({key: str(value) for key, value in details.items()})
    values.setdefault("detail", "There is a problem with the code here.")
    return template.format_map(values)


def concept_of(code: str) -> Concept:
    """The concept a code teaches: core's table first, then the front end's extras."""
    if code in CODE_CONCEPTS:
        return CODE_CONCEPTS[code]
    return EXTRA_CONCEPTS.get(code, Concept.SYNTAX)


def make_diagnostic(code: str, message: str, location: SourceLocation | None, *,
                    friendly: str | None = None, severity: Severity = Severity.ERROR,
                    phase: Phase = Phase.PARSE, concept: Concept | None = None,
                    notes: list[str] | None = None, **details: object) -> Diagnostic:
    """Build a Diagnostic, filling the friendly text and concept from the tables."""
    return Diagnostic(
        severity=severity,
        code=code,
        message=message,
        friendly=friendly if friendly is not None else friendly_text(code, **details),
        location=location,
        concept=concept if concept is not None else concept_of(code),
        phase=phase,
        notes=list(notes or []),
    )


def unknown_type_hint(name: str, known_types: Iterable[str]) -> tuple[str | None, str]:
    """Suggestion and friendly hint for an unknown type name.

    Returns ``(suggestion, hint)``: the suggestion goes into the gcc message ("did you
    mean 'int'?"), the hint into the friendly text.
    """
    header = HEADER_FOR_TYPE.get(name)
    if header is not None:
        return None, f" It is defined in <{header}>: add '#include <{header}>' at the top."
    suggestion = did_you_mean(name, [*TYPE_WORDS, *known_types])
    if suggestion is not None:
        return suggestion, f" Did you mean '{suggestion}'?"
    return None, " Check the spelling, or define it with typedef or struct first."


def include_guard_name(path: str) -> str:
    """The conventional guard macro for a header path: 'include/point.h' -> 'POINT_H'."""
    base = path.rsplit("/", 1)[-1]
    return "".join(char if char.isalnum() else "_" for char in base).upper()


def redefinition_friendly(name: str, previous: SourceLocation | None,
                          current: SourceLocation | None) -> str:
    """Friendly text for "redefinition of 'X'".

    When both definitions come from the same header, the header was included twice
    without an include guard; that is the fix worth explaining, not "delete one".
    """
    if (previous is not None and current is not None and previous.file == current.file
            and previous.file.endswith(".h")):
        guard = include_guard_name(previous.file)
        return (f"'{name}' is defined twice because {previous.file} was included more than "
                f"once and has no include guard. Wrap the header in '#ifndef {guard}', "
                f"'#define {guard}' at the top and '#endif' at the bottom, so its contents "
                "are only read once.")
    if previous is not None:
        return (f"'{name}' is defined twice: first at {previous}, and again here. "
                "Keep one definition.")
    return friendly_text("redefinition", name=name)
