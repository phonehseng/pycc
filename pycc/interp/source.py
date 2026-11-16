"""Preprocessed source lines.

The preprocessor turns the user's files into a flat list of ``PPLine`` records. Each
record remembers which file and line it came from, so the lexer, parser, checker and
interpreter can always point back at the exact line the user wrote, even for code
that arrived through ``#include`` or a macro.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPLine:
    text: str     # the line after macro expansion, comments removed
    file: str     # VFS path of the originating file (as the user wrote it)
    line: int     # 1-based line number in that file
    # Optional: col_map[i] is the 1-based original column of text[i] (or the column of
    # the macro use for expanded text), so diagnostics inside expanded lines still point
    # at what the user typed. None means the text is unchanged from the source line.
    col_map: tuple[int, ...] | None = None

    def original_column(self, index: int) -> int:
        """Map a 0-based index into ``text`` back to a 1-based source column."""
        if self.col_map is None or index >= len(self.col_map):
            return index + 1
        return self.col_map[index]


def lines_from_text(text: str, filename: str) -> list[PPLine]:
    """Wrap raw text as PPLines without preprocessing (for tests and tools)."""
    return [PPLine(line, filename, number) for number, line in enumerate(text.splitlines(), start=1)]
