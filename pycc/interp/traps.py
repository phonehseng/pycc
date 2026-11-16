"""Runtime traps and pending warnings: how the simulated machine says "that went wrong".

A *trap* is a fatal runtime error (a NULL dereference, a double free, dividing by
zero). The interpreter, the memory model and device functions all raise ``TrapError``
so that one ``except`` clause in the interpreter can turn any of them into a ``Trap``
event.

A *pending warning* is a non-fatal finding (reading an uninitialised variable, running
out of heap). The memory model cannot yield events itself, so it queues them in
``Memory.pending_warnings``; the interpreter drains the queue after each step and
yields ``RunWarning`` events, attaching the source location it knows about.

Both carry the same four things every diagnostic needs: a stable
snake_case ``code``, a tool-style ``message``, a plain-English ``friendly`` text and a
``Concept`` tag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pycc.core.concepts import Concept
from pycc.core.diagnostics import (
    Diagnostic,
    Phase,
    Severity,
    SourceLocation,
    concept_for_code,
)

# Plain-English fallbacks for runtime codes, used when the raiser has nothing more
# specific to say. Specific situations (which array, which index) should pass their own
# friendly text; these keep a bare ``TrapError("division_by_zero", msg)`` student-ready.
DEFAULT_FRIENDLY: dict[str, str] = {
    "null_dereference": "The program followed a NULL pointer. NULL means 'points at "
                        "nothing', so there is no value there to read or write.",
    "out_of_bounds": "The program reached outside a variable or array, into memory that "
                     "belongs to something else.",
    "wild_pointer": "This pointer does not point at any variable or allocation; it may be "
                    "garbage or past the end of an array.",
    "use_after_free": "This memory was already given back with free(). Once freed, it "
                      "must not be used again.",
    "double_free": "This block was already freed. Each malloc needs exactly one free.",
    "invalid_free": "free() only accepts the exact address that malloc, calloc or "
                    "realloc returned (or NULL).",
    "memory_leak": "Some malloc'd blocks were never freed before the program ended.",
    "heap_exhausted": "The heap is full, so malloc returned NULL. Check malloc's result "
                      "and free blocks you no longer need.",
    "stack_overflow": "Each function call takes a slice of the stack. Too many calls at "
                      "once (often recursion without a base case) used it all up.",
    "division_by_zero": "Dividing an integer by zero has no answer, so the machine stops.",
    "instruction_budget": "The program ran for too long. A loop may never reach its exit "
                          "condition.",
    "write_to_literal": "String literals like \"hello\" are read-only. Copy the text into "
                        "a char array first if you want to change it.",
    "assert_failed": "An assert() check was false, so the program stopped on purpose.",
    "bad_function_pointer": "This function pointer does not point at a real function.",
    "string_not_terminated": "A C string must end with a '\\0' byte. This one runs off the "
                             "end of its memory without one.",
    "abort_called": "The program called abort() and stopped immediately.",
    "uninitialized_read": "This variable was read before it was given a value, so it holds "
                          "leftover garbage.",
    "signed_overflow": "A signed integer went past its largest or smallest value and "
                       "wrapped around.",
}


def default_friendly(code: str) -> str:
    """Return the generic plain-English text for ``code`` (empty if none is known)."""
    return DEFAULT_FRIENDLY.get(code, "")


class TrapError(Exception):
    """A fatal runtime error raised by the interpreter, the memory model or the device.

    ``extra`` holds structured details (object name, index, length...) so a front end
    can talk about specifics without parsing the message.
    """

    def __init__(self, code: str, message: str, friendly: str | None = None,
                 concept: Concept | None = None, extra: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.friendly = friendly if friendly is not None else default_friendly(code)
        # Unknown codes fall back to POINTERS: almost every unlisted runtime trap is
        # about touching memory the program should not.
        self.concept = concept if concept is not None else concept_for_code(code, Concept.POINTERS)
        self.extra: dict = dict(extra or {})

    def to_diagnostic(self, location: SourceLocation | None) -> Diagnostic:
        """Convert to a runtime ``Diagnostic`` at the statement that trapped."""
        return Diagnostic(
            severity=Severity.ERROR,
            code=self.code,
            message=self.message,
            friendly=self.friendly,
            location=location,
            concept=self.concept,
            phase=Phase.RUNTIME,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code!r}, {self.message!r})"


class MemoryTrap(TrapError):
    """A trap raised by the simulated memory (bad pointer, bad free, overflow...)."""


@dataclass
class PendingWarning:
    """A non-fatal runtime finding waiting for the interpreter to report it."""

    code: str
    message: str
    friendly: str
    concept: Concept
    extra: dict = field(default_factory=dict)

    def to_diagnostic(self, location: SourceLocation | None) -> Diagnostic:
        """Convert to a runtime warning ``Diagnostic`` (wrapped in a RunWarning event)."""
        return Diagnostic(
            severity=Severity.WARNING,
            code=self.code,
            message=self.message,
            friendly=self.friendly,
            location=location,
            concept=self.concept,
            phase=Phase.RUNTIME,
        )
