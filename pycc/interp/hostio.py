"""How the toolchain sees files and the outside world without importing the VFS.

``interp`` and ``vfs`` live on the same layer and never import each other. The compiler reads sources through ``SourceFiles`` and the C
standard library reads and writes through ``HostFiles``; ``vfs.VirtualFS`` satisfies
``SourceFiles`` structurally and ``shell/hostfiles.py`` adapts it to ``HostFiles``.
Errors are returned as negative errno values, never raised, because that is what a C
program expects to see in ``errno``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# errno values (Linux numbering, which is what the user will meet with real gcc).
ENOENT = 2
EIO = 5
EBADF = 9
ENOMEM = 12
EACCES = 13
EEXIST = 17
ENOTDIR = 20
EISDIR = 21
EINVAL = 22
ERANGE = 34

ERRNO_NAMES: dict[int, str] = {
    ENOENT: "ENOENT", EIO: "EIO", EBADF: "EBADF", ENOMEM: "ENOMEM", EACCES: "EACCES",
    EEXIST: "EEXIST", ENOTDIR: "ENOTDIR", EISDIR: "EISDIR", EINVAL: "EINVAL", ERANGE: "ERANGE",
}

STRERROR: dict[int, str] = {
    0: "Success",
    ENOENT: "No such file or directory",
    EIO: "Input/output error",
    EBADF: "Bad file descriptor",
    ENOMEM: "Cannot allocate memory",
    EACCES: "Permission denied",
    EEXIST: "File exists",
    ENOTDIR: "Not a directory",
    EISDIR: "Is a directory",
    EINVAL: "Invalid argument",
    ERANGE: "Numerical result out of range",
}


def strerror(code: int) -> str:
    return STRERROR.get(code, f"Unknown error {code}")


class SourceFiles(Protocol):
    """What the compiler needs: read text files and ask whether they exist."""

    def read_text(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...


class HostFiles(Protocol):
    """What the C standard library needs for fopen & co. Absolute paths only.

    Methods return negative errno values on failure (for example ``-ENOENT``).
    """

    def read(self, path: str) -> bytes | int: ...
    def write(self, path: str, data: bytes, append: bool) -> int: ...
    def exists(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...
    def remove(self, path: str) -> int: ...
    def rename(self, old: str, new: str) -> int: ...


@dataclass
class ProgramIO:
    """Everything a running program may touch outside its own memory."""

    files: HostFiles | None = None
    cwd: str = "/"
    argv0: str = "a.out"
    env: dict[str, str] = field(default_factory=dict)


def join_path(cwd: str, path: str) -> str:
    """Resolve ``path`` against ``cwd`` the POSIX way (no filesystem access)."""
    if not path.startswith("/"):
        path = cwd.rstrip("/") + "/" + path
    parts: list[str] = []
    for piece in path.split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/" + "/".join(parts)


class DictSourceFiles:
    """An in-memory ``SourceFiles`` for tests and tools."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})

    def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def exists(self, path: str) -> bool:
        return path in self.files or any(p.startswith(path.rstrip("/") + "/") for p in self.files)

    def is_file(self, path: str) -> bool:
        return path in self.files


class DictHostFiles:
    """An in-memory ``HostFiles`` for tests: a dict of path -> bytes."""

    def __init__(self, files: dict[str, bytes] | None = None, read_only: set[str] | None = None) -> None:
        self.files: dict[str, bytes] = dict(files or {})
        self.read_only: set[str] = set(read_only or ())
        self.dirs: set[str] = set()

    def read(self, path: str) -> bytes | int:
        if path in self.dirs:
            return -EISDIR
        if path not in self.files:
            return -ENOENT
        return self.files[path]

    def write(self, path: str, data: bytes, append: bool) -> int:
        if path in self.dirs:
            return -EISDIR
        if path in self.read_only:
            return -EACCES
        if append and path in self.files:
            self.files[path] += data
        else:
            self.files[path] = bytes(data)
        return 0

    def exists(self, path: str) -> bool:
        return path in self.files or path in self.dirs

    def is_dir(self, path: str) -> bool:
        return path in self.dirs

    def remove(self, path: str) -> int:
        if path in self.read_only:
            return -EACCES
        if path not in self.files:
            return -ENOENT
        del self.files[path]
        return 0

    def rename(self, old: str, new: str) -> int:
        if old not in self.files:
            return -ENOENT
        self.files[new] = self.files.pop(old)
        return 0
