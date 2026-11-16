"""``<errno.h>``: the ``errno`` variable, its codes and their texts.

``errno`` is a real ``int`` in the program's data segment (registered as a library
variable), so ``printf("%d", errno)`` and ``&errno`` behave as in C. Library functions
set it through ``Machine.set_errno``. The numbers are Linux's, which is what a user
meets later with a real compiler.

``hostio`` owns the codes the file layer can produce. The maths functions also need
``EDOM`` (an argument outside a function's domain, as in ``sqrt(-1.0)``), which only
this module adds, together with its ``strerror`` text.
"""
from __future__ import annotations

from pycc.interp import hostio
from pycc.interp.builtins import BuiltinRegistry, Machine

HEADER = "errno.h"

EDOM = 33
ERANGE = hostio.ERANGE
EINVAL = hostio.EINVAL

# Every macro errno.h defines, name -> value. The header test checks the C text
# against this table so the two cannot disagree.
ERRNO_MACROS: dict[str, int] = {
    "ENOENT": hostio.ENOENT,
    "EIO": hostio.EIO,
    "EBADF": hostio.EBADF,
    "ENOMEM": hostio.ENOMEM,
    "EACCES": hostio.EACCES,
    "EEXIST": hostio.EEXIST,
    "ENOTDIR": hostio.ENOTDIR,
    "EISDIR": hostio.EISDIR,
    "EINVAL": hostio.EINVAL,
    "EDOM": EDOM,
    "ERANGE": hostio.ERANGE,
}

_EXTRA_STRERROR: dict[int, str] = {EDOM: "Numerical argument out of domain"}


def c_strerror(code: int) -> str:
    """The glibc text for an errno value (``"No such file or directory"``)."""
    if code in _EXTRA_STRERROR:
        return _EXTRA_STRERROR[code]
    return hostio.strerror(code)


def _errno_initial_bytes(machine: Machine) -> bytes:
    """``errno`` starts at 0, like every C program's."""
    return bytes(4)


def register(registry: BuiltinRegistry) -> None:
    """Register ``extern int errno`` (D2: a library variable with an address)."""
    registry.register_variable("errno", "int errno;", _errno_initial_bytes, header=HEADER)
