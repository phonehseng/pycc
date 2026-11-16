"""The small ANSI SGR subset the interactive terminal understands.

Real terminals colour ``ls`` output and many tools print escape sequences. An interactive
front end only needs to parse these codes (reset, bold, dim, and the eight basic
foreground colours), so everything that colours text goes through this module.
A line that starts with ``DIM`` is a *friendly* explanation under an error; the
terminal widget draws it in the soft notebook style.
"""
from __future__ import annotations

import re

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"

# ls colours, matching the common Linux defaults so the habit transfers.
DIRECTORY = "\x1b[1;34m"
EXECUTABLE = "\x1b[1;32m"

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def paint(text: str, style: str) -> str:
    """Wrap ``text`` in ``style`` and reset afterwards."""
    return f"{style}{text}{RESET}"


def friendly_line(text: str) -> str:
    """Format a plain-English explanation as one dimmed terminal line."""
    return f"{DIM}{text}{RESET}\n"


def strip_ansi(text: str) -> str:
    """Remove every SGR escape (used for widths, plain logs and tests)."""
    return _SGR.sub("", text)
