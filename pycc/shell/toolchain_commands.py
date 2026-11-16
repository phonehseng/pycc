"""Connect a shell to the C toolchain: the real ``cc``, ``gcc`` and ``make``.

The shell registers placeholders for the build tools so it can exist without a
compiler. ``install`` replaces them with the commands of ``shell.cc`` and
``shell.make`` and hands the shell its ``Toolchain``. The
commands read ``ctx.shell.toolchain`` on every call, so a shell without a toolchain
still answers "not installed yet", and switching toolchains only means storing another
one.
"""
from __future__ import annotations

from pycc.interp.toolchain import Toolchain
from pycc.shell.cc import cmd_cc
from pycc.shell.commands import register_command
from pycc.shell.executor import Shell
from pycc.shell.make import cmd_make
from pycc.shell.sysheaders import install_system_headers

BUILD_COMMANDS = {"cc": cmd_cc, "gcc": cmd_cc, "make": cmd_make}


def register_toolchain_commands() -> None:
    """Register the real build commands (replacing the placeholders); idempotent."""
    for name, command in BUILD_COMMANDS.items():
        register_command(name, command)


def install(shell: Shell, toolchain: Toolchain, headers_into_vfs: bool = True) -> None:
    """Give ``shell`` the real build tools and ``toolchain``.

    With ``headers_into_vfs`` the system headers are also copied (read-only) into the
    shell's ``/usr/include``, so ``cat /usr/include/stdio.h`` shows what cc uses.
    Child shells (scripts, make recipes) inherit ``shell.toolchain`` when spawned.
    """
    register_toolchain_commands()
    shell.toolchain = toolchain
    if headers_into_vfs:
        install_system_headers(shell.vfs, toolchain.headers)


__all__ = ["BUILD_COMMANDS", "install", "register_toolchain_commands"]
