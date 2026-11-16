"""Diagnostics: the one shape every error, warning and trap takes.

The toolchain speaks gcc ("expected ';' before '}' token"), the shell speaks bash
("command not found") and the runtime speaks in traps ("out of bounds"). All of them
produce a ``Diagnostic`` so that a terminal, an editor or a grader can treat them the
same way. Each diagnostic has two texts: ``message`` is what the real tool would print,
``friendly`` is a plain-English explanation written for a beginner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pycc.core.concepts import Concept


@dataclass(frozen=True)
class SourceLocation:
    """A place in a source file. Lines and columns are 1-based; 0 = unknown."""

    file: str
    line: int
    column: int = 0

    def __str__(self) -> str:
        if self.column:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.file}:{self.line}"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    FATAL = "fatal"


class Phase(StrEnum):
    """Which tool produced the diagnostic. The terminal prefixes output by phase."""

    PREPROCESS = "preprocess"
    PARSE = "parse"
    CHECK = "check"
    LINK = "link"
    RUNTIME = "runtime"
    SHELL = "shell"
    MAKE = "make"


@dataclass
class Diagnostic:
    severity: Severity
    code: str
    message: str
    friendly: str
    location: SourceLocation | None
    concept: Concept
    phase: Phase
    notes: list[str] = field(default_factory=list)

    def format_gcc(self) -> str:
        """Render like gcc: ``file:line:col: error: message`` (a fatal one says ``fatal error``)."""
        prefix = f"{self.location}: " if self.location else ""
        label = "fatal error" if self.severity is Severity.FATAL else self.severity.value
        text = f"{prefix}{label}: {self.message}"
        for note in self.notes:
            text += f"\n{prefix}note: {note}"
        return text

    @property
    def is_error(self) -> bool:
        return self.severity in (Severity.ERROR, Severity.FATAL)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "friendly": self.friendly,
            "file": self.location.file if self.location else None,
            "line": self.location.line if self.location else None,
            "column": self.location.column if self.location else None,
            "concept": self.concept.value,
            "phase": self.phase.value,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Diagnostic":
        location = None
        if data.get("file") is not None and data.get("line") is not None:
            location = SourceLocation(data["file"], int(data["line"]), int(data.get("column") or 0))
        return cls(
            severity=Severity(data["severity"]),
            code=data["code"],
            message=data["message"],
            friendly=data.get("friendly", ""),
            location=location,
            concept=Concept(data.get("concept", Concept.SYNTAX.value)),
            phase=Phase(data.get("phase", Phase.CHECK.value)),
            notes=list(data.get("notes", [])),
        )


def concept_for_code(code: str, default: Concept = Concept.SYNTAX) -> Concept:
    """Look up the default concept of a known diagnostic code."""
    return CODE_CONCEPTS.get(code, default)


# Known diagnostic codes and the concept each one teaches. Modules may use codes that
# are not listed here as long as they pass the concept explicitly.
CODE_CONCEPTS: dict[str, Concept] = {
    # --- preprocessor -------------------------------------------------------------
    "include_not_found": Concept.HEADERS,
    "unterminated_comment": Concept.SYNTAX,
    "unterminated_if": Concept.PREPROCESSOR,
    "else_without_if": Concept.PREPROCESSOR,
    "endif_without_if": Concept.PREPROCESSOR,
    "macro_args_mismatch": Concept.PREPROCESSOR,
    "error_directive": Concept.PREPROCESSOR,
    "include_depth": Concept.HEADERS,
    "invalid_directive": Concept.PREPROCESSOR,
    "macro_redefined": Concept.PREPROCESSOR,
    # --- lexer --------------------------------------------------------------------
    "unterminated_string": Concept.STRINGS,
    "unterminated_char": Concept.STRINGS,
    "invalid_suffix": Concept.INTEGER_TYPES,
    "stray_character": Concept.SYNTAX,
    "invalid_escape": Concept.STRINGS,
    "invalid_number": Concept.VARIABLES,
    # --- parser -------------------------------------------------------------------
    "missing_semicolon": Concept.SYNTAX,
    "expected_token": Concept.SYNTAX,
    "unexpected_token": Concept.SYNTAX,
    "unknown_type": Concept.VARIABLES,
    "unbalanced_brace": Concept.SYNTAX,
    "unbalanced_paren": Concept.SYNTAX,
    "else_without_if_stmt": Concept.CONDITIONALS,
    "missing_return_type": Concept.FUNCTION_DESIGN,
    "bad_declarator": Concept.VARIABLES,
    "bad_initializer": Concept.ARRAYS,
    "empty_struct": Concept.STRUCTS,
    "nested_function": Concept.FUNCTIONS,
    # --- checker: errors -------------------------------------------------------------
    "undeclared_identifier": Concept.VARIABLES,
    "implicit_declaration": Concept.FUNCTION_DESIGN,
    "conflicting_types": Concept.FUNCTION_DESIGN,
    "redefinition": Concept.HEADERS,
    "too_few_arguments": Concept.FUNCTION_DESIGN,
    "too_many_arguments": Concept.FUNCTION_DESIGN,
    "not_a_struct": Concept.STRUCTS,
    "no_member": Concept.STRUCTS,
    "invalid_subscript": Concept.ARRAYS,
    "lvalue_required": Concept.OPERATORS,
    "void_value": Concept.FUNCTION_DESIGN,
    "return_value_in_void": Concept.FUNCTION_DESIGN,
    "return_missing_value": Concept.FUNCTION_DESIGN,
    "array_size_not_const": Concept.ARRAYS,
    "incompatible_types": Concept.VARIABLES,
    "call_non_function": Concept.FUNCTIONS,
    "break_outside_loop": Concept.LOOPS,
    "case_outside_switch": Concept.CONDITIONALS,
    "duplicate_case": Concept.CONDITIONALS,
    "invalid_operands": Concept.OPERATORS,
    "incomplete_type": Concept.STRUCTS,
    "assign_to_array": Concept.ARRAYS,
    "assign_to_const": Concept.VARIABLES,
    # --- checker: warnings -----------------------------------------------------------
    "unused_variable": Concept.VARIABLES,
    "control_reaches_end": Concept.FUNCTION_DESIGN,
    "implicit_int": Concept.VARIABLES,
    "assignment_as_condition": Concept.OPERATORS,
    "pointer_int_comparison": Concept.POINTERS,
    "int_from_pointer": Concept.POINTERS,
    "pointer_from_int": Concept.POINTERS,
    "incompatible_pointer": Concept.POINTERS,
    "uninitialized_use": Concept.UNINITIALIZED,
    "array_bounds": Concept.ARRAYS,
    "format_mismatch": Concept.STDIO,
    "unused_value": Concept.OPERATORS,
    "sign_compare": Concept.INTEGER_TYPES,
    "missing_prototype": Concept.HEADERS,
    "string_literal_to_char_array": Concept.STRINGS,
    "integer_overflow_constant": Concept.INTEGER_TYPES,
    "address_compare": Concept.STRINGS,           # name == "yes"
    # --- linker -----------------------------------------------------------------------
    "undefined_reference": Concept.LINKING,
    "multiple_definition": Concept.LINKAGE,
    "no_main": Concept.LINKING,
    "static_not_visible": Concept.LINKAGE,
    "bad_object_file": Concept.BUILD_CC,
    # --- runtime traps ---------------------------------------------------------------
    "null_dereference": Concept.POINTERS,
    "out_of_bounds": Concept.ARRAYS,
    "wild_pointer": Concept.POINTERS,
    "use_after_free": Concept.DYNAMIC_MEMORY,
    "double_free": Concept.DYNAMIC_MEMORY,
    "invalid_free": Concept.DYNAMIC_MEMORY,
    "memory_leak": Concept.DYNAMIC_MEMORY,
    "heap_exhausted": Concept.DYNAMIC_MEMORY,
    "stack_overflow": Concept.RECURSION,
    "division_by_zero": Concept.OPERATORS,
    "instruction_budget": Concept.LOOP_BOUNDS,
    "write_to_literal": Concept.STRINGS,
    "assert_failed": Concept.ERROR_HANDLING,
    "bad_function_pointer": Concept.POINTERS,
    "invalid_argument": Concept.DEVICE_API,
    "string_not_terminated": Concept.STRINGS,
    "abort_called": Concept.ERROR_HANDLING,
    # --- runtime warnings ------------------------------------------------------------
    "uninitialized_read": Concept.UNINITIALIZED,
    "signed_overflow": Concept.INTEGER_TYPES,
    "shift_overflow": Concept.INTEGER_TYPES,
    "format_argument_type": Concept.STDIO,
    "file_not_closed": Concept.FILE_IO,
    "fopen_failed": Concept.ERROR_HANDLING,
    "unchecked_null": Concept.ERROR_HANDLING,
    # --- shell ----------------------------------------------------------------------
    "command_not_found": Concept.SHELL_NAVIGATION,
    "no_such_file": Concept.SHELL_FILES,
    "permission_denied": Concept.SHELL_SCRIPTING,
    "is_a_directory": Concept.SHELL_FILES,
    "not_a_directory": Concept.SHELL_NAVIGATION,
    "not_executable": Concept.SHELL_EXIT_CODES,
    "cannot_execute_binary": Concept.BUILD_CC,
    "shell_syntax_error": Concept.SHELL_SCRIPTING,
    "redirect_error": Concept.SHELL_REDIRECTION,
    "pipe_error": Concept.SHELL_PIPES,
    "directory_not_empty": Concept.SHELL_FILES,
    "file_exists": Concept.SHELL_FILES,
    "bad_option": Concept.SHELL_FILES,
    "missing_operand": Concept.SHELL_FILES,
    "exit_nonzero": Concept.SHELL_EXIT_CODES,
    # --- cc / make -------------------------------------------------------------------
    "cc_no_input": Concept.BUILD_CC,
    "cc_bad_option": Concept.BUILD_CC,
    "cc_unknown_file_type": Concept.BUILD_CC,
    "make_no_makefile": Concept.BUILD_MAKE,
    "make_no_rule": Concept.BUILD_MAKE,
    "make_missing_separator": Concept.BUILD_MAKE,
    "make_recipe_failed": Concept.BUILD_MAKE,
    "make_circular": Concept.BUILD_MAKE,
    "make_nothing_to_do": Concept.BUILD_MAKE,
    # --- more toolchain and runtime codes ----------------------------------------------
    "unsupported_feature": Concept.SYNTAX,        # goto, VLAs, bit-fields, designated inits...
    "invalid_stream": Concept.FILE_IO,            # closed or garbage FILE *
    "bad_executable": Concept.BUILD_CC,           # wrong magic or version in a CVEXE file
    "same_input_output": Concept.BUILD_CC,        # cc main.c -o main.c
    "too_many_initializers": Concept.ARRAYS,      # excess elements in an initializer
    # Front-end codes added by the checker, lexer and parser.
    "case_not_constant": Concept.CONDITIONALS,    # case label is not an integer constant
    "empty_char_constant": Concept.STRINGS,       # ''
    "multichar_constant": Concept.STRINGS,        # 'ab'
    "enum_overflow": Concept.ENUMS,               # enumerator value outside int
    "enum_value_not_const": Concept.ENUMS,        # enumerator initialised with a variable
    "invalid_type_combination": Concept.VARIABLES,  # long char, short double
    "nesting_too_deep": Concept.SYNTAX,           # expression nested beyond the parser limit
    "too_many_errors": Concept.SYNTAX,            # compilation stopped at -fmax-errors
    "useless_declaration": Concept.VARIABLES,     # 'int;' declares nothing
    "warning_directive": Concept.PREPROCESSOR,    # #warning text
    # Shell and virtual filesystem codes.
    "loop_limit": Concept.SHELL_SCRIPTING,        # a script loop ran past the round guard
    "vfs_error": Concept.SHELL_FILES,             # generic filesystem failure
    "invalid_operation": Concept.SHELL_FILES,     # rm of the root, mv into itself...
    "same_file": Concept.SHELL_FILES,             # cp a a
}

# The toolchain's own failure: a bug in pycc, not in the program being run.
CODE_CONCEPTS["internal_error"] = Concept.DEVICE_API

# Exit code a program ends with when a trap stops it, mirroring what a shell reports
# for a real crashed process. Memory and pointer faults look like
# a segmentation fault, bad frees and asserts like an abort, division by zero like a
# floating point exception, and a runaway loop like a process killed by the system.
TRAP_EXIT_CODES: dict[str, int] = {
    "null_dereference": 139,
    "out_of_bounds": 139,
    "wild_pointer": 139,
    "use_after_free": 139,
    "write_to_literal": 139,
    "stack_overflow": 139,
    "bad_function_pointer": 139,
    "string_not_terminated": 139,
    "double_free": 134,
    "invalid_free": 134,
    "assert_failed": 134,
    "abort_called": 134,
    "invalid_argument": 134,
    "invalid_stream": 134,
    "division_by_zero": 136,
    "instruction_budget": 137,
    "heap_exhausted": 134,
}

EXIT_CODE_MESSAGES: dict[int, str] = {
    139: "Segmentation fault (core dumped)",
    134: "Aborted (core dumped)",
    136: "Floating point exception (core dumped)",
    137: "Killed",
    130: "^C",
}


def exit_code_for_trap(code: str) -> int:
    return TRAP_EXIT_CODES.get(code, 134)
