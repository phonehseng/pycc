"""The concept taxonomy: what part of C (or the shell) a diagnostic is about.

The preprocessor, compiler, linker, runtime, shell and make all tag their
diagnostics with one of these values, so a tool built on pycc (an editor, a grader,
a tutoring system) can group errors by topic without parsing messages.
"""
from __future__ import annotations

from enum import StrEnum


class Concept(StrEnum):
    """A topic of C or the shell. Every diagnostic carries one."""

    FUNCTIONS = "functions"              # calling functions, statements, semicolons
    SYNTAX = "syntax"                    # general parse errors, braces, parentheses
    VARIABLES = "variables"              # declarations, types, sizeof, casting
    INTEGER_TYPES = "integer_types"      # overflow, wraparound, signed/unsigned, truncation
    FLOATING = "floating"                # float/double, integer vs float division
    OPERATORS = "operators"              # arithmetic, comparison, logical, precedence, = vs ==
    CONDITIONALS = "conditionals"        # if/else, switch
    LOOPS = "loops"                      # for/while/do, break/continue
    LOOP_BOUNDS = "loop_bounds"          # off-by-one, infinite loops
    FUNCTION_DESIGN = "function_design"  # parameters, return values, prototypes
    SCOPE = "scope"                      # block scope, shadowing, globals, static locals
    RECURSION = "recursion"              # base cases, stack depth
    ARRAYS = "arrays"                    # indexing, bounds, 2D arrays, decay
    STRINGS = "strings"                  # char arrays, the NUL terminator, string.h
    POINTERS = "pointers"                # & and *, NULL, dangling pointers
    POINTER_ARITHMETIC = "pointer_arithmetic"
    STRUCTS = "structs"                  # struct, ., ->, typedef, padding
    ENUMS = "enums"
    DYNAMIC_MEMORY = "dynamic_memory"    # malloc/free, leaks, use-after-free, double free
    UNINITIALIZED = "uninitialized"      # reading variables before assigning them
    PREPROCESSOR = "preprocessor"        # #define, #include, macros, #ifdef
    HEADERS = "headers"                  # prototypes in headers, include guards
    LINKAGE = "linkage"                  # extern, static, declaration vs definition
    LINKING = "linking"                  # undefined reference, object files, link order
    SHELL_NAVIGATION = "shell_navigation"    # ls, cd, pwd
    SHELL_FILES = "shell_files"              # cat, mkdir, touch, rm, cp, mv
    SHELL_PIPES = "shell_pipes"              # |, grep, wc, head, tail
    SHELL_REDIRECTION = "shell_redirection"  # >, >>, <, 2>
    SHELL_EXIT_CODES = "shell_exit_codes"    # $?, &&, ||
    SHELL_SCRIPTING = "shell_scripting"      # variables, for, if, .sh scripts
    BUILD_CC = "build_cc"                # cc flags: -c, -o, -Wall, -I
    BUILD_MAKE = "build_make"            # Makefiles, targets, dependencies
    PROGRAM_ARGS = "program_args"        # argc / argv
    STDIO = "stdio"                      # printf, scanf, fgets, stdout vs stderr
    FILE_IO = "file_io"                  # fopen, fclose, fgets, fprintf
    ERROR_HANDLING = "error_handling"    # return codes, errno, perror, exit codes
    DEVICE_API = "device_api"            # functions provided by the embedding host


CONCEPT_TITLES: dict[Concept, str] = {
    Concept.FUNCTIONS: "Functions and statements",
    Concept.SYNTAX: "Syntax",
    Concept.VARIABLES: "Variables and types",
    Concept.INTEGER_TYPES: "Integer types and overflow",
    Concept.FLOATING: "Floating point",
    Concept.OPERATORS: "Operators",
    Concept.CONDITIONALS: "If, else and switch",
    Concept.LOOPS: "Loops",
    Concept.LOOP_BOUNDS: "Loop bounds",
    Concept.FUNCTION_DESIGN: "Parameters and return values",
    Concept.SCOPE: "Scope",
    Concept.RECURSION: "Recursion",
    Concept.ARRAYS: "Arrays",
    Concept.STRINGS: "Strings",
    Concept.POINTERS: "Pointers",
    Concept.POINTER_ARITHMETIC: "Pointer arithmetic",
    Concept.STRUCTS: "Structs and typedef",
    Concept.ENUMS: "Enums",
    Concept.DYNAMIC_MEMORY: "Dynamic memory",
    Concept.UNINITIALIZED: "Uninitialized variables",
    Concept.PREPROCESSOR: "The preprocessor",
    Concept.HEADERS: "Headers and include guards",
    Concept.LINKAGE: "extern and static",
    Concept.LINKING: "Linking",
    Concept.SHELL_NAVIGATION: "Moving around the shell",
    Concept.SHELL_FILES: "Files in the shell",
    Concept.SHELL_PIPES: "Pipes",
    Concept.SHELL_REDIRECTION: "Redirection",
    Concept.SHELL_EXIT_CODES: "Exit codes",
    Concept.SHELL_SCRIPTING: "Shell scripts",
    Concept.BUILD_CC: "Compiling with cc",
    Concept.BUILD_MAKE: "Makefiles",
    Concept.PROGRAM_ARGS: "Program arguments",
    Concept.STDIO: "Standard input and output",
    Concept.FILE_IO: "File input and output",
    Concept.ERROR_HANDLING: "Error handling",
    Concept.DEVICE_API: "Device functions",
}

