"""The built-in commands (docs/SUPPORTED_SHELL.md), registered on import.

This module holds the file and folder commands (``ls cd pwd cat mkdir touch rm cp mv
chmod find tree file``) and registers the rest from two neighbours:
``shell/textcmds.py`` (echo and the text filters used in pipes) and
``shell/sessioncmds.py`` (help, variables, tests, scripts and the placeholders for
``cc``/``gcc``/``make``).

Every command follows the same recipe: parse options with ``parse_options``, do the
work through ``ctx.vfs``, write results to ``ctx.stdout``, report problems with
``ctx.fail``/``ctx.warn_vfs`` and return the exit status.
"""
from __future__ import annotations

import fnmatch
import posixpath
from datetime import datetime, timedelta

from pycc.core.concepts import Concept
from pycc.shell import sessioncmds, textcmds
from pycc.shell.ansi import DIRECTORY, EXECUTABLE, paint
from pycc.shell.commands import (CommandContext, CommandError, missing_operand, parse_options,
                            register_command, vfs_friendly)
from pycc.vfs.errors import VfsError

# ls -l shows a pretend clock: a fixed morning plus one minute per logical tick,
# so newer files really do show later times (which is what make compares).
_CLOCK_START = datetime(2024, 3, 4, 8, 0)


# --- ls ----------------------------------------------------------------------------------

def cmd_ls(ctx: CommandContext, args: list[str]) -> int:
    """List folder contents (``-l`` long, ``-a`` all, ``-1`` one per line)."""
    options, operands = parse_options("ls", args, "la1")
    targets = operands or ["."]
    status = 0
    files: list[str] = []
    folders: list[str] = []
    for target in targets:
        path = ctx.resolve(target)
        if not ctx.vfs.exists(path):
            status = ctx.fail("no_such_file",
                              f"ls: cannot access '{target}': No such file or directory",
                              f"There is nothing called '{target}' here. Run `ls` on its own "
                              "to see the names in this folder.", 2)
        elif ctx.vfs.is_dir(path):
            folders.append(target)
        else:
            files.append(target)
    files.sort()                   # GNU ls sorts its operands: files first, then folders
    folders.sort()
    if files:
        _print_entries(ctx, [(name, ctx.resolve(name)) for name in files], options)
    for index, folder in enumerate(folders):
        if len(targets) > 1:
            ctx.out(("\n" if files or index else "") + f"{folder}:\n")
        _print_entries(ctx, _folder_entries(ctx, folder, bool(options.get("a"))), options)
    return status


def _folder_entries(ctx: CommandContext, folder: str, show_all: bool) -> list[tuple[str, str]]:
    path = ctx.resolve(folder)
    names = ctx.vfs.listdir(path)
    if not show_all:
        names = [name for name in names if not name.startswith(".")]
    entries = [(name, posixpath.join(path, name)) for name in names]
    if show_all:
        entries = [(".", path), ("..", ctx.vfs.resolve(path + "/.."))] + entries
    return entries


def _print_entries(ctx: CommandContext, entries: list[tuple[str, str]],
                   options: dict[str, str | bool]) -> None:
    if not entries:
        return
    if options.get("l"):
        ctx.out("".join(_long_line(ctx, name, path) + "\n" for name, path in entries))
        return
    shown = [_coloured_name(ctx, name, path) for name, path in entries]
    # Like real ls: names side by side on a terminal, one per line into a pipe or file.
    if options.get("1") or not ctx.stdout_is_terminal:
        ctx.out("\n".join(shown) + "\n")
    else:
        ctx.out("  ".join(shown) + "\n")


def _coloured_name(ctx: CommandContext, name: str, path: str) -> str:
    if not ctx.stdout_is_terminal:
        return name
    info = ctx.vfs.stat(path)
    if info.is_dir:
        return paint(name, DIRECTORY)
    if info.executable:
        return paint(name, EXECUTABLE)
    return name


def _long_line(ctx: CommandContext, name: str, path: str) -> str:
    info = ctx.vfs.stat(path)
    size = 4096 if info.is_dir else info.size
    stamp = _CLOCK_START + timedelta(minutes=info.mtime)
    when = f"{stamp:%b} {stamp.day:>2} {stamp:%H:%M}"
    return f"{_mode_string(info.is_dir, info.executable, info.read_only)} 1 user user " \
           f"{size:>5} {when} {_coloured_name(ctx, name, path)}"


def _mode_string(is_dir: bool, executable: bool, read_only: bool) -> str:
    """``drwxr-xr-x`` style permissions from our three flags."""
    run = "x" if executable or is_dir else "-"
    owner = ("r-" if read_only else "rw") + run
    others = "r-" + run
    return ("d" if is_dir else "-") + owner + others + others


# --- moving around ------------------------------------------------------------------------

def cmd_cd(ctx: CommandContext, args: list[str]) -> int:
    """Change the working directory."""
    if len(args) > 1:
        return ctx.fail("bad_option", "bash: cd: too many arguments",
                        "cd takes one folder name. Put quotes around names with spaces.")
    target = args[0] if args else ctx.shell.env.get("HOME", "/")
    if target == "-":
        if "OLDPWD" not in ctx.shell.env:
            return ctx.fail("no_such_file", "bash: cd: OLDPWD not set",
                            "`cd -` goes back to the previous folder, but you have not been "
                            "anywhere else yet.", concept=Concept.SHELL_NAVIGATION)
        target = ctx.shell.env["OLDPWD"]
        ctx.out(target + "\n")
    path = ctx.resolve(target)
    if not ctx.vfs.exists(path):
        return ctx.fail("no_such_file", f"bash: cd: {target}: No such file or directory",
                        f"There is no folder called '{target}' here. Run `ls` to see the "
                        "folders you can go into.")
    if not ctx.vfs.is_dir(path):
        return ctx.fail("not_a_directory", f"bash: cd: {target}: Not a directory",
                        f"'{target}' is a file, so you cannot go into it. Try `cat {target}`.")
    ctx.shell.change_directory(path)
    return 0


def cmd_pwd(ctx: CommandContext, args: list[str]) -> int:
    """Print the working directory."""
    ctx.out(ctx.cwd + "\n")
    return 0


# --- reading and creating files -----------------------------------------------------------

def cmd_cat(ctx: CommandContext, args: list[str]) -> int:
    """Concatenate files to standard output (``-n`` numbers lines)."""
    options, operands = parse_options("cat", args, "n")
    status = 0
    line_number = 0
    for _name, text in ctx.read_inputs(operands):
        if text is None:
            status = 1
            continue
        if options.get("n"):
            numbered = []
            for line in text.splitlines(keepends=True):
                line_number += 1
                numbered.append(f"{line_number:>6}\t{line}")
            text = "".join(numbered)
        ctx.out(text)
    return status


def cmd_mkdir(ctx: CommandContext, args: list[str]) -> int:
    """Create folders (``-p`` parents too)."""
    options, operands = parse_options("mkdir", args, "p")
    if not operands:
        raise missing_operand("mkdir", "mkdir src")
    status = 0
    for operand in operands:
        try:
            ctx.vfs.mkdir(ctx.resolve(operand), parents=bool(options.get("p")))
        except VfsError as error:
            missing_parent = error.code == "no_such_file"
            hint = " Use `mkdir -p` to create the missing folders too." if missing_parent else ""
            error.path = operand
            status = ctx.fail(error.code,
                              f"mkdir: cannot create directory '{operand}': {error.reason}",
                              vfs_friendly(error) + hint)
    return status


def cmd_touch(ctx: CommandContext, args: list[str]) -> int:
    """Create empty files or bump their modification time."""
    _options, operands = parse_options("touch", args, "")
    if not operands:
        raise missing_operand("touch", "touch notes.txt")
    status = 0
    for operand in operands:
        try:
            ctx.vfs.touch(ctx.resolve(operand))
        except VfsError as error:
            error.path = operand
            status = ctx.fail(error.code, f"touch: cannot touch '{operand}': {error.reason}",
                              vfs_friendly(error))
    return status


# --- removing, copying, moving ------------------------------------------------------------

def cmd_rm(ctx: CommandContext, args: list[str]) -> int:
    """Remove files (``-r`` folders, ``-f`` quietly)."""
    options, operands = parse_options("rm", args, "rfR")
    recursive = bool(options.get("r") or options.get("R"))
    force = bool(options.get("f"))
    if not operands and not force:
        raise missing_operand("rm", "rm old.txt")
    status = 0
    for operand in operands:
        if posixpath.basename(operand.rstrip("/")) in (".", ".."):
            status = ctx.fail("bad_option",
                              f"rm: refusing to remove '.' or '..' directory: skipping '{operand}'",
                              "rm will not delete the folder you are standing in or its parent.")
            continue
        if recursive and ctx.resolve(operand) == "/":
            # GNU rm's --preserve-root: one typo must never wipe the whole filesystem.
            status = ctx.fail("permission_denied",
                              "rm: it is dangerous to operate recursively on '/'",
                              "'/' is the top of the whole filesystem. rm refuses to delete "
                              "everything; name the folder you really mean instead.")
            continue
        status = _remove_one(ctx, operand, recursive, force) or status
    return status


def _remove_one(ctx: CommandContext, operand: str, recursive: bool, force: bool) -> int:
    path = ctx.resolve(operand)
    if not ctx.vfs.exists(path):
        if force:
            return 0
        return ctx.fail("no_such_file", f"rm: cannot remove '{operand}': No such file or directory",
                        f"There is nothing called '{operand}' here, so nothing was removed.")
    if ctx.vfs.is_dir(path) and not recursive:
        return ctx.fail("is_a_directory", f"rm: cannot remove '{operand}': Is a directory",
                        f"'{operand}' is a folder. Use `rm -r {operand}` to remove it with "
                        "everything inside.")
    try:
        ctx.vfs.remove(path, recursive=recursive)
    except VfsError as error:
        error.path = operand
        return ctx.fail(error.code, f"rm: cannot remove '{operand}': {error.reason}",
                        vfs_friendly(error))
    return 0


def _sources_and_destination(name: str, operands: list[str], example: str) -> tuple[list[str], str]:
    if not operands:
        raise missing_operand(name, example)
    if len(operands) == 1:
        raise CommandError("missing_operand",
                           f"{name}: missing destination file operand after '{operands[0]}'",
                           f"{name} needs two names: what to {name} and where to, like "
                           f"`{example}`.")
    return operands[:-1], operands[-1]


def _check_multi_target(ctx: CommandContext, name: str, sources: list[str],
                        destination: str) -> None:
    if len(sources) > 1 and not ctx.vfs.is_dir(ctx.resolve(destination)):
        raise CommandError("not_a_directory", f"{name}: target '{destination}' is not a directory",
                           "With several files, the last name must be an existing folder to put "
                           "them in.")


def _resolve_destination(ctx: CommandContext, destination: str) -> str:
    """Absolute destination that keeps a typed trailing ``/`` (it means "a folder")."""
    path = ctx.resolve(destination)
    return path + "/" if destination.endswith("/") and path != "/" else path


def cmd_cp(ctx: CommandContext, args: list[str]) -> int:
    """Copy files (``-r`` folders)."""
    options, operands = parse_options("cp", args, "rR")
    recursive = bool(options.get("r") or options.get("R"))
    sources, destination = _sources_and_destination("cp", operands, "cp main.c backup.c")
    _check_multi_target(ctx, "cp", sources, destination)
    status = 0
    for source in sources:
        path = ctx.resolve(source)
        if ctx.vfs.is_dir(path) and not recursive:
            status = ctx.fail("is_a_directory",
                              f"cp: -r not specified; omitting directory '{source}'",
                              f"'{source}' is a folder. Use `cp -r` to copy folders.")
            continue
        try:
            ctx.vfs.copy(path, _resolve_destination(ctx, destination), recursive=recursive)
        except VfsError as error:
            error.path = source if error.code == "no_such_file" else destination
            message = f"cp: cannot copy '{source}': {error.reason}"
            if error.code == "same_file":
                message = f"cp: '{source}' and '{destination}' are the same file"
            status = ctx.fail(error.code, message, vfs_friendly(error))
    return status


def cmd_mv(ctx: CommandContext, args: list[str]) -> int:
    """Move or rename files and folders."""
    _options, operands = parse_options("mv", args, "")
    sources, destination = _sources_and_destination("mv", operands, "mv old.c new.c")
    _check_multi_target(ctx, "mv", sources, destination)
    status = 0
    for source in sources:
        source_path, destination_path = ctx.resolve(source), _resolve_destination(ctx, destination)
        try:
            ctx.vfs.move(source_path, destination_path)
        except VfsError as error:
            error.path = _blamed_operand(error, source, source_path, destination)
            status = ctx.fail(error.code,
                              f"mv: cannot move '{source}' to '{destination}': {error.reason}",
                              vfs_friendly(error))
    return status


def _blamed_operand(error: VfsError, source: str, source_path: str, destination: str) -> str:
    """The operand, as the user typed it, that a failed ``mv`` is about.

    A missing file is always the source. Otherwise the filesystem says which path it
    refused: a read-only system file is the source, a protected folder the destination.
    """
    if error.code == "no_such_file":
        return source
    refused = getattr(error, "path", None)
    if refused in (source, source_path):
        return source
    return destination


# --- permissions --------------------------------------------------------------------------

def _parse_mode(mode: str) -> bool | None:
    """True for "make executable", False for "not executable", None if unknown."""
    if mode.isdigit() and len(mode) in (3, 4) and all(ch in "01234567" for ch in mode):
        return bool(int(mode[-3]) & 1)
    who, _, perms = mode.partition("+") if "+" in mode else mode.partition("-")
    if perms == "x" and set(who) <= set("uga"):
        return "+" in mode
    return None


def cmd_chmod(ctx: CommandContext, args: list[str]) -> int:
    """Turn the executable bit on (``+x``) or off (``-x``)."""
    # "-x" looks like an option, so take the mode before option parsing.
    if not args:
        raise missing_operand("chmod", "chmod +x run.sh")
    mode, operands = args[0], args[1:]
    executable = _parse_mode(mode)
    if executable is None:
        raise CommandError("bad_option", f"chmod: invalid mode: '{mode}'",
                           "Use `chmod +x FILE` to make a file runnable or `chmod -x FILE` to "
                           "undo it.")
    if not operands:
        raise CommandError("missing_operand", f"chmod: missing operand after '{mode}'",
                           f"Say which file to change, like `chmod {mode} run.sh`.")
    status = 0
    for operand in operands:
        try:
            ctx.vfs.set_executable(ctx.resolve(operand), executable)
        except VfsError as error:
            error.path = operand
            protected = error.code == "permission_denied"
            reason = "Operation not permitted" if protected else error.reason
            verb = "cannot access" if error.code == "no_such_file" else "changing permissions of"
            status = ctx.fail(error.code, f"chmod: {verb} '{operand}': {reason}",
                              vfs_friendly(error))
    return status


# --- searching and describing -------------------------------------------------------------

def cmd_find(ctx: CommandContext, args: list[str]) -> int:
    """Walk folders printing paths that match ``-name`` and ``-type``."""
    starts: list[str] = []
    index = 0
    while index < len(args) and not args[index].startswith("-"):
        starts.append(args[index])
        index += 1
    name_pattern, kind = _find_predicates(args[index:])
    status = 0
    for start in starts or ["."]:
        path = ctx.resolve(start)
        if not ctx.vfs.exists(path):
            status = ctx.fail("no_such_file", f"find: '{start}': No such file or directory",
                              f"There is no folder called '{start}' to search in.")
            continue
        for shown, is_dir in _find_walk(ctx, start, path):
            if _find_matches(shown, is_dir, name_pattern, kind):
                ctx.out(shown + "\n")
    return status


def _find_predicates(expression: list[str]) -> tuple[str | None, str | None]:
    name_pattern: str | None = None
    kind: str | None = None
    index = 0
    while index < len(expression):
        predicate = expression[index]
        if predicate not in ("-name", "-type"):
            raise CommandError("bad_option", f"find: unknown predicate '{predicate}'",
                               "find understands `-name PATTERN` and `-type f` or `-type d`.")
        if index + 1 >= len(expression):
            example = '"*.c"' if predicate == "-name" else "f"
            raise CommandError("missing_operand", f"find: missing argument to '{predicate}'",
                               f"Put a value after {predicate}, like `find . {predicate} "
                               f"{example}`.")
        value = expression[index + 1]
        if predicate == "-type" and value not in ("f", "d"):
            raise CommandError("bad_option", f"find: Unknown argument to -type: {value}",
                               "Use `-type f` for files or `-type d` for folders.")
        if predicate == "-name":
            name_pattern = value
        else:
            kind = value
        index += 2
    return name_pattern, kind


def _find_walk(ctx: CommandContext, start: str, path: str) -> list[tuple[str, bool]]:
    if not ctx.vfs.is_dir(path):
        return [(start, False)]
    found = [(start, True)]
    for folder, dirs, files in ctx.vfs.walk(path):
        relative = posixpath.relpath(folder, path)
        base = start if relative == "." else posixpath.join(start, relative)
        found.extend((posixpath.join(base, name), True) for name in dirs)
        found.extend((posixpath.join(base, name), False) for name in files)
    # walk is top-down per folder; sort so a folder's contents follow it directly.
    return sorted(found, key=lambda item: item[0].split("/"))


def _find_matches(shown: str, is_dir: bool, name_pattern: str | None, kind: str | None) -> bool:
    if kind == "f" and is_dir or kind == "d" and not is_dir:
        return False
    if name_pattern is None:
        return True
    return fnmatch.fnmatchcase(posixpath.basename(shown.rstrip("/")) or shown, name_pattern)


def cmd_tree(ctx: CommandContext, args: list[str]) -> int:
    """Draw a folder and everything below it."""
    _options, operands = parse_options("tree", args, "")
    target = operands[0] if operands else "."
    path = ctx.resolve(target)
    if not ctx.vfs.is_dir(path):
        return ctx.fail("not_a_directory" if ctx.vfs.exists(path) else "no_such_file",
                        f"{target} [error opening dir]",
                        f"tree draws folders; '{target}' is not a folder here.")
    counts = [0, 0]                    # folders, files
    lines = [_coloured_name(ctx, target, path)]
    _tree_lines(ctx, path, "", lines, counts)
    ctx.out("\n".join(lines) + f"\n\n{counts[0]} directories, {counts[1]} files\n")
    return 0


def _tree_lines(ctx: CommandContext, path: str, prefix: str, lines: list[str],
                counts: list[int]) -> None:
    names = [name for name in ctx.vfs.listdir(path) if not name.startswith(".")]
    for index, name in enumerate(names):
        last = index == len(names) - 1
        child = posixpath.join(path, name)
        lines.append(prefix + ("└── " if last else "├── ") + _coloured_name(ctx, name, child))
        if ctx.vfs.is_dir(child):
            counts[0] += 1
            _tree_lines(ctx, child, prefix + ("    " if last else "│   "), lines, counts)
        else:
            counts[1] += 1


def describe_file(data: bytes, name: str, executable: bool) -> str:
    """What ``file`` says about some bytes."""
    if data.startswith(b"CVEXE1\n"):
        return "pycc executable, runnable program"
    if data.startswith(b"CVOBJ1\n"):
        return "pycc relocatable object file (link it with cc)"
    if data.startswith(b"CVBIN1\n"):
        return "pycc built-in command"
    if not data:
        return "empty"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "data"
    if text.startswith("#!") or name.endswith(".sh"):
        return "shell script, ASCII text" + (" executable" if executable else "")
    if name.endswith((".c", ".h")):
        return "C source, ASCII text"
    if name in ("Makefile", "makefile"):
        return "makefile script, ASCII text"
    return "ASCII text"


def cmd_file(ctx: CommandContext, args: list[str]) -> int:
    """Guess the kind of each file."""
    _options, operands = parse_options("file", args, "")
    if not operands:
        raise missing_operand("file", "file prog")
    for operand in operands:
        path = ctx.resolve(operand)
        if not ctx.vfs.exists(path):
            kind = f"cannot open `{operand}' (No such file or directory)"
        elif ctx.vfs.is_dir(path):
            kind = "directory"
        else:
            kind = describe_file(ctx.vfs.read_file(path), posixpath.basename(path),
                                 ctx.vfs.stat(path).executable)
        ctx.out(f"{operand}: {kind}\n")
    return 0


# --- registration -------------------------------------------------------------------------

FILE_COMMANDS = {
    "ls": cmd_ls, "cd": cmd_cd, "pwd": cmd_pwd, "cat": cmd_cat, "mkdir": cmd_mkdir,
    "touch": cmd_touch, "rm": cmd_rm, "cp": cmd_cp, "mv": cmd_mv, "chmod": cmd_chmod,
    "find": cmd_find, "tree": cmd_tree, "file": cmd_file,
}


def register_builtins() -> None:
    """Register every built-in command (safe to call again to reset replacements)."""
    for name, fn in FILE_COMMANDS.items():
        register_command(name, fn)
    for name, fn in textcmds.TEXT_COMMANDS.items():
        register_command(name, fn)
    for name, fn in sessioncmds.SESSION_COMMANDS.items():
        register_command(name, fn)


register_builtins()
