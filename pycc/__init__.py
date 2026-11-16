"""pycc: a C99 toolchain and an LP64 interpreter with memory-error detection.

See README.md. The command line lives in ``pycc.cli``; the pieces are
``pycc.interp`` (preprocessor, compiler, linker, interpreter, C library),
``pycc.shell`` (a Bash-like shell with cc and make) and ``pycc.vfs`` (the
in-memory filesystem programs run in).
"""

__version__ = "0.1.0"
