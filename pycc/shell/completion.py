"""Tab completion for the interactive terminal.

``complete(shell, line, cursor)`` looks at the word under the cursor and returns full
replacement words, sorted: command names in command position, options after the
commands users use most (``cc``, ``gcc``, ``make``, ``ls``...), ``chmod`` modes,
variable names after ``$`` and ``export``, ``make`` targets read from the Makefile,
and otherwise paths relative to the working directory (folders end
in ``/`` so the next Tab continues inside them). The terminal replaces the current
word with the candidate when there is exactly one, and lists them otherwise.
"""
from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING

from pycc.shell.commands import command_names

if TYPE_CHECKING:
    from pycc.shell.executor import Shell

OPTION_HINTS: dict[str, tuple[str, ...]] = {
    "cc": ("-Wall", "-Wextra", "-Werror", "-c", "-o", "-g", "-I", "-D", "-O2", "-std=c99"),
    "ls": ("-1", "-a", "-l"),
    "make": ("-f", "-n"),
    "grep": ("-E", "-c", "-i", "-n", "-r", "-v"),
    "rm": ("-f", "-r"),
    "head": ("-n",),
    "tail": ("-n",),
    "wc": ("-c", "-l", "-w"),
    "sort": ("-n", "-r"),
    "cat": ("-n",),
    "mkdir": ("-p",),
    "cp": ("-r",),
    "find": ("-name", "-type"),
    "history": ("-c",),
    "echo": ("-e", "-n"),
}
CHMOD_MODES = ("+x", "-x")
_VARIABLE = re.compile(r"^(\$\{?)([A-Za-z0-9_]*)$")
OPTION_HINTS["gcc"] = OPTION_HINTS["cc"]

# A new command starts after these, so the next word is completed as a command.
_COMMAND_BREAK = re.compile(r"(\|\||&&|[|;(])")
_MAKE_TARGET = re.compile(r"^([A-Za-z0-9_./-][A-Za-z0-9_./ -]*?)\s*::?(?!=)")


# After these keywords the next word is a command again: "if true; then ec<Tab>".
_COMMAND_KEYWORDS = {"if", "then", "elif", "else", "while", "until", "do", "!"}
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# A redirection glued to its file name, as in ">out.txt" or "2>>log.txt".
_GLUED_REDIRECT = re.compile(r"^(&>>|&>|2>>|2>|1>|>>|>|<)(.*)$")
_REDIRECT_WORDS = {"&>>", "&>", "2>>", "2>", "1>", ">>", ">", "<"}
# A blank that is not escaped with a backslash separates words.
_WORD_GAP = re.compile(r"(?<!\\)\s+")


def complete(shell: "Shell", line: str, cursor: int) -> list[str]:
    """Candidates for the word that ends at ``cursor`` (anywhere in the line)."""
    segment = _COMMAND_BREAK.split(line[:cursor])[-1]
    words = [word for word in _WORD_GAP.split(segment) if word]
    current = "" if _ends_between_words(segment) else words[-1]
    previous = _command_words(words if current == "" else words[:-1])
    glued = _GLUED_REDIRECT.match(current)
    if glued:
        operator, rest = glued.groups()
        return [operator + candidate for candidate in complete_path(shell, rest)]
    variable = _VARIABLE.match(current)
    if variable:
        return _complete_variable(shell, *variable.groups())
    if not previous:
        return _complete_command(shell, current)
    if previous[-1] in _REDIRECT_WORDS:
        return complete_path(shell, current)          # a file name, whatever the command
    command = previous[0]
    if command == "export":
        return _complete_variable(shell, "", current)
    if command == "chmod" and len(previous) == 1:
        return [mode for mode in CHMOD_MODES if mode.startswith(current)]
    if current.startswith("-") and command in OPTION_HINTS:
        return [option for option in OPTION_HINTS[command] if option.startswith(current)]
    if command == "make":
        targets = make_targets(shell)
        if targets:
            return [target for target in targets if target.startswith(current)]
    return complete_path(shell, current, folders_only=command == "cd")


def _ends_between_words(segment: str) -> bool:
    """True when the cursor sits after a blank (an escaped blank is part of a word)."""
    return not segment or (segment[-1].isspace() and not segment.endswith("\\" + segment[-1]))


def _command_words(words: list[str]) -> list[str]:
    """The words of the current command without leading keywords and ``NAME=value``."""
    index = 0
    while index < len(words) and (words[index] in _COMMAND_KEYWORDS
                                  or _ASSIGNMENT.match(words[index])):
        index += 1
    return words[index:]


def _complete_command(shell: "Shell", current: str) -> list[str]:
    if "/" in current:
        return complete_path(shell, current, runnable_only=True)
    return [name for name in command_names() if name.startswith(current) and name != "."]


def _complete_variable(shell: "Shell", lead: str, prefix: str) -> list[str]:
    """``$HO`` -> ``$HOME``; ``lead`` is what comes before the name (``$``, ``${`` or "")."""
    close = "}" if lead == "${" else ""
    return sorted(f"{lead}{name}{close}" for name in shell.env if name.startswith(prefix))


def _escape(text: str) -> str:
    """Backslash the characters the shell would split or expand (``my\\ notes.txt``)."""
    return re.sub(r"([\s\\'\"$`*?\[\]|&;<>()#])", r"\\\1", text)


def _unescape(text: str) -> str:
    """Undo ``_escape`` on what the user typed so far."""
    return re.sub(r"\\(.)", r"\1", text)


def complete_path(shell: "Shell", current: str, folders_only: bool = False,
                  runnable_only: bool = False) -> list[str]:
    """Paths starting with ``current``; folders get a trailing ``/``.

    Candidates are escaped so that inserting one gives a single shell word even when
    a name holds a space.
    """
    typed = _unescape(current)
    folder_part, _, prefix = typed.rpartition("/")
    folder_text = folder_part + "/" if "/" in typed else ""
    folder = shell.vfs.resolve(folder_text or ".", shell.cwd)
    if not shell.vfs.is_dir(folder):
        return []
    results: list[str] = []
    for name in shell.vfs.listdir(folder):
        if not name.startswith(prefix) or (name.startswith(".") and not prefix.startswith(".")):
            continue
        info = shell.vfs.stat(posixpath.join(folder, name))
        if info.is_dir:
            results.append(_escape(folder_text + name) + "/")
        elif not folders_only and (info.executable or not runnable_only):
            results.append(_escape(folder_text + name))
    return results


def make_targets(shell: "Shell") -> list[str]:
    """Explicit targets of the Makefile in the working directory (no patterns)."""
    for name in ("Makefile", "makefile"):
        path = shell.vfs.resolve(name, shell.cwd)
        if shell.vfs.is_file(path):
            return _targets_in(shell.vfs.read_text(path))
    return []


def _targets_in(text: str) -> list[str]:
    targets: list[str] = []
    for line in text.splitlines():
        if line.startswith("\t"):
            continue                        # recipe lines
        match = _MAKE_TARGET.match(line)
        if not match:
            continue
        for target in match.group(1).split():
            if "%" not in target and not target.startswith(".") and "$" not in target \
                    and target not in targets:
                targets.append(target)
    return sorted(targets)
