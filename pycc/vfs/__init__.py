"""Virtual filesystem: an in-memory POSIX tree for the user's files."""
from pycc.vfs.errors import (AlreadyExists, DirectoryNotEmpty, InvalidOperation, IsADirectory,
                        NotADirectory, NotFound, PermissionDenied, SameFile, VfsError)
from pycc.vfs.filesystem import (BIN_COMMANDS, BUILTIN_MARKER, HOME, VDir, VFile, VirtualFS,
                            VStat, standard_layout)

__all__ = [
    "AlreadyExists", "BIN_COMMANDS", "BUILTIN_MARKER", "DirectoryNotEmpty", "HOME",
    "InvalidOperation", "IsADirectory", "NotADirectory", "NotFound", "PermissionDenied",
    "SameFile",
    "VDir", "VFile", "VStat", "VfsError", "VirtualFS", "standard_layout",
]
