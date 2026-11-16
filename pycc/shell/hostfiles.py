"""``VfsHostFiles``: the virtual filesystem as a running C program sees it.

The C standard library (``fopen``, ``remove``, ``rename``...) talks to files through the
``interp.hostio.HostFiles`` protocol, because ``interp`` and ``vfs`` live on the same
layer and must not import each other. This adapter sits one
layer up, in the shell, and translates in both directions:

* calls go straight to ``VirtualFS``;
* every ``VfsError`` becomes the negative errno a real C library would report
  (``-ENOENT`` for a missing file, ``-EACCES`` for a read-only system header...),
  because a C program never sees Python exceptions, only ``errno``.
"""
from __future__ import annotations

from pycc.interp.hostio import EACCES, EEXIST, EINVAL, EIO, EISDIR, ENOENT, ENOTDIR
from pycc.vfs.errors import (AlreadyExists, DirectoryNotEmpty, InvalidOperation, IsADirectory,
                        NotADirectory, NotFound, PermissionDenied, VfsError)
from pycc.vfs.filesystem import VirtualFS

# Most specific class first: SameFile subclasses InvalidOperation, so order matters.
_ERRNO_FOR: tuple[tuple[type[VfsError], int], ...] = (
    (NotFound, ENOENT),
    (IsADirectory, EISDIR),
    (NotADirectory, ENOTDIR),
    (PermissionDenied, EACCES),
    (AlreadyExists, EEXIST),
    # POSIX lets rename()/rmdir() report EEXIST for a non-empty folder (Linux says
    # ENOTEMPTY, which the simulated library does not define).
    (DirectoryNotEmpty, EEXIST),
    (InvalidOperation, EINVAL),
)


def errno_for(error: VfsError) -> int:
    """The positive errno value that matches a filesystem error."""
    for error_type, code in _ERRNO_FOR:
        if isinstance(error, error_type):
            return code
    return EIO


class VfsHostFiles:
    """``interp.hostio.HostFiles`` over a ``VirtualFS`` (paths are absolute)."""

    def __init__(self, vfs: VirtualFS) -> None:
        """Wrap ``vfs``; the program's writes land in it immediately (write-through)."""
        self.vfs = vfs

    def read(self, path: str) -> bytes | int:
        """The whole file, or ``-errno`` (``-EISDIR`` for a folder)."""
        try:
            return self.vfs.read_file(path)
        except VfsError as error:
            return -errno_for(error)

    def write(self, path: str, data: bytes, append: bool) -> int:
        """Create, overwrite or extend a file; 0 on success or ``-errno``."""
        try:
            self.vfs.write_file(path, bytes(data), append=append)
        except VfsError as error:
            return -errno_for(error)
        return 0

    def exists(self, path: str) -> bool:
        """True if a file or folder lives at ``path``."""
        return self.vfs.exists(path)

    def is_dir(self, path: str) -> bool:
        """True if ``path`` is a folder."""
        return self.vfs.is_dir(path)

    def remove(self, path: str) -> int:
        """C ``remove()``: delete a file or an empty folder; 0 or ``-errno``."""
        try:
            self.vfs.remove(path)
        except VfsError as error:
            return -errno_for(error)
        return 0

    def rename(self, old: str, new: str) -> int:
        """C ``rename()``: move ``old`` to exactly ``new``, replacing a file there.

        ``VirtualFS.move`` behaves like ``mv`` and would drop a file *into* a folder
        called ``new``; C's rename never does that, it fails with ``EISDIR``.
        """
        if not self.vfs.exists(old):
            return -ENOENT
        if self.vfs.resolve(old) == self.vfs.resolve(new):
            return 0                      # renaming a file onto itself changes nothing
        if self.vfs.is_dir(new):
            if not self.vfs.is_dir(old):
                return -EISDIR
            # Folder onto folder: POSIX replaces an empty one (and refuses a full one)
            # instead of moving the old folder inside, as mv would.
            status = self._clear_empty_folder(new)
            if status:
                return status
        try:
            self.vfs.move(old, new)
        except VfsError as error:
            return -errno_for(error)
        return 0

    def _clear_empty_folder(self, path: str) -> int:
        """Remove the empty folder ``path`` so a folder can be renamed onto it."""
        if self.vfs.listdir(path):
            return -EEXIST
        try:
            self.vfs.remove(path)
        except VfsError as error:
            return -errno_for(error)
        return 0


__all__ = ["VfsHostFiles", "errno_for"]
