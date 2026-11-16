"""The closed list of construct keys the interpreter counts during a run.

``RunSummary.constructs`` maps these keys to counts, so a tool can tell which parts
of C a program actually exercised (loops, pointers, malloc...). Keep the list closed:
a new key must be added here.
"""
from __future__ import annotations

LOOP_ITERATION = "loop_iteration"
FUNCTION_CALL = "function_call"
RECURSION = "recursion"             # a call whose callee is already on the stack
POINTER_DEREF = "pointer_deref"
POINTER_ARITH = "pointer_arith"
ARRAY_INDEX = "array_index"
STRUCT_MEMBER = "struct_member"
SWITCH = "switch"
MALLOC = "malloc"
FREE = "free"
REALLOC = "realloc"
STRING_CALL = "string_call"         # any string.h function
FILE_OPEN = "file_open"
FILE_WRITE = "file_write"
FILE_READ = "file_read"
SCANF = "scanf"
PRINTF = "printf"
DEVICE_CALL = "device_call"

ALL_KEYS: tuple[str, ...] = (
    LOOP_ITERATION, FUNCTION_CALL, RECURSION, POINTER_DEREF, POINTER_ARITH, ARRAY_INDEX,
    STRUCT_MEMBER, SWITCH, MALLOC, FREE, REALLOC, STRING_CALL, FILE_OPEN, FILE_WRITE,
    FILE_READ, SCANF, PRINTF, DEVICE_CALL,
)
