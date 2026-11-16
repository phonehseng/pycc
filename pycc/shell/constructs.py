"""The closed list of construct keys the shell counts.

They travel in ``CommandFinished.constructs``.
"""
from __future__ import annotations

PIPELINE = "pipeline"
REDIRECTION = "redirection"
CC_COMPILE = "cc_compile"
LINK = "link"
MAKE_REBUILD = "make_rebuild"       # per target actually rebuilt
SCRIPT_RUN = "script_run"
EXIT_CODE_TEST = "exit_code_test"   # $?, &&, ||, if ./prog

ALL_KEYS: tuple[str, ...] = (PIPELINE, REDIRECTION, CC_COMPILE, LINK, MAKE_REBUILD, SCRIPT_RUN,
                             EXIT_CODE_TEST)
