"""Simulated bash-like shell with builtins, scripts and the RunProgram hook."""
from pycc.shell.commands import (IO, CommandContext, CommandError, CommandSpec, ExitRequest,
                            command_names, lookup, register_command)
from pycc.shell.executor import (DEFAULT_CWD, EXE_MAGIC, OBJ_MAGIC, Shell, ShellResult,
                            run_program_headless)
from pycc.shell.parser import needs_more_input
from pycc.shell.streams import FileSink, TerminalSink, TerminalSource

__all__ = [
    "CommandContext", "CommandError", "CommandSpec", "DEFAULT_CWD", "EXE_MAGIC", "ExitRequest",
    "FileSink", "IO", "OBJ_MAGIC", "Shell", "ShellResult", "TerminalSink", "TerminalSource",
    "command_names", "lookup", "needs_more_input", "register_command", "run_program_headless",
]
