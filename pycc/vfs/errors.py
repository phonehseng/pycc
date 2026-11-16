"""Errors raised by the virtual filesystem.

Every error carries a stable snake_case ``code`` (the same codes the shell turns into
diagnostics) and formats itself the way bash and coreutils would print it, so a shell
command can show ``cat: notes.txt: No such file or directory`` by simply prefixing
its own name to ``str(error)``.
"""
from __future__ import annotations


class VfsError(Exception):
    """Base class of every filesystem error.

    ``path`` is the path the user typed (or the absolute path when that is all we
    know) and ``reason`` is the strerror-style text such as "No such file or
    directory".
    """

    code = "vfs_error"
    default_reason = "Input/output error"

    def __init__(self, path: str, reason: str | None = None) -> None:
        self.path = path
        self.reason = reason or self.default_reason
        super().__init__(f"{path}: {self.reason}")


class NotFound(VfsError):
    """The path, or one of its parent folders, does not exist."""

    code = "no_such_file"
    default_reason = "No such file or directory"


class IsADirectory(VfsError):
    """A file operation was asked to work on a folder."""

    code = "is_a_directory"
    default_reason = "Is a directory"


class NotADirectory(VfsError):
    """A folder operation met a plain file, e.g. ``cd notes.txt``."""

    code = "not_a_directory"
    default_reason = "Not a directory"


class AlreadyExists(VfsError):
    """Creating something whose name is already taken."""

    code = "file_exists"
    default_reason = "File exists"


class PermissionDenied(VfsError):
    """Writing to a read-only system file or folder, or escaping an export folder."""

    code = "permission_denied"
    default_reason = "Permission denied"


class DirectoryNotEmpty(VfsError):
    """Removing a folder that still has things inside without ``recursive``."""

    code = "directory_not_empty"
    default_reason = "Directory not empty"


class InvalidOperation(VfsError):
    """Something no real filesystem allows, e.g. moving a folder into itself."""

    code = "invalid_operation"
    default_reason = "Invalid argument"


class SameFile(InvalidOperation):
    """Copying a file onto itself, which would destroy what is being read."""

    code = "same_file"
    default_reason = "are the same file"
