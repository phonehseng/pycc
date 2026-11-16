"""The virtual filesystem every program, command and editor tab in pycc uses.

The user's files never touch the real disk (except through ``export_to_disk``): they
live in a tree of ``VDir`` and ``VFile`` nodes held in memory. Keeping the tree in
memory makes saves a plain JSON dump, makes ``snapshot``/``restore`` cheap (a deep
copy) and keeps tests independent of the machine they run on.

Time is *logical*: a counter that goes up by one on every write, create, move or
touch. ``make`` compares these numbers to decide what to rebuild, so builds are
deterministic no matter how fast the user types.
"""
from __future__ import annotations

import base64
import copy
import fnmatch
import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from pycc.vfs.errors import (AlreadyExists, DirectoryNotEmpty, InvalidOperation, IsADirectory,
                        NotADirectory, NotFound, PermissionDenied, SameFile)

HOME = "/home/user"

# The first bytes of a /bin marker file. The shell sees this prefix and runs the
# built-in command named on the rest of the line, so ``/bin/ls`` works like ``ls``.
BUILTIN_MARKER = b"CVBIN1\n"

# The built-in commands that get a marker in /bin in a fresh layout.
BIN_COMMANDS: tuple[str, ...] = (
    "ls", "cd", "pwd", "cat", "mkdir", "touch", "rm", "cp", "mv", "echo", "head",
    "tail", "wc", "grep", "find", "diff", "sort", "uniq", "seq", "clear", "history",
    "man", "help", "chmod", "env", "export", "which", "true", "false", "test", "[",
    "exit", "source", "sh", "bash", "cc", "gcc", "make", "file", "date", "sleep",
    "tree", "fortune",
)

ChangeCallback = Callable[[str, str], None]


@dataclass
class VFile:
    """A file: raw bytes plus the metadata ``ls -l`` and ``make`` care about."""

    data: bytes = b""
    mtime: int = 0
    executable: bool = False
    read_only: bool = False


@dataclass
class VDir:
    """A folder: named children plus metadata. Folders are always 'executable'
    (searchable) like on a real system, so only the read-only flag matters."""

    children: dict[str, "VFile | VDir"] = field(default_factory=dict)
    mtime: int = 0
    read_only: bool = False


Node = VFile | VDir


@dataclass(frozen=True)
class VStat:
    """What ``stat`` reports about one path."""

    size: int
    mtime: int
    is_dir: bool
    executable: bool
    read_only: bool


def _has_magic(component: str) -> bool:
    """True when a path component contains a glob wildcard."""
    return any(ch in component for ch in "*?[")


class VirtualFS:
    """An in-memory POSIX-like filesystem.

    Paths given to every method may be absolute or relative; relative paths are taken
    from ``/`` unless a ``cwd`` is given to ``resolve``/``glob``. The shell always
    resolves paths against its own working directory before calling in.
    """

    def __init__(self) -> None:
        """Create an empty filesystem containing only ``/``."""
        self.root = VDir()
        self._clock = 0
        self._listeners: list[ChangeCallback] = []

    # --- time and change notification ---------------------------------------------

    @property
    def clock(self) -> int:
        """The current logical time (the mtime the next change will get)."""
        return self._clock

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def on_change(self, callback: ChangeCallback) -> Callable[[], None]:
        """Call ``callback(event, path)`` after every change; returns an unsubscribe.

        Events: ``create`` (new file), ``write`` (existing file changed), ``mkdir``,
        ``remove``, ``attr`` (permission bits), and ``reset`` (whole tree replaced by
        ``restore``/``from_dict``). A move is reported as ``remove`` of the old path
        followed by ``create``/``mkdir`` of the new one, so a file tree only needs to
        understand the basic events.
        """
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback) if callback in self._listeners else None

    def _notify(self, event: str, path: str) -> None:
        # Copy the list: a listener may unsubscribe itself while being notified.
        for listener in list(self._listeners):
            listener(event, path)

    # --- paths -----------------------------------------------------------------------

    def resolve(self, path: str, cwd: str = "/") -> str:
        """Turn ``path`` into an absolute, normalised POSIX path.

        ``~`` means the user's home folder; ``.`` and ``..`` are folded away and
        ``..`` at the root stays at the root, as on Linux.
        """
        if path == "~" or path.startswith("~/"):
            path = HOME + path[1:]
        if not path.startswith("/"):
            path = posixpath.join(cwd or "/", path)
        parts: list[str] = []
        for part in path.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return "/" + "/".join(parts)

    @staticmethod
    def _split(abs_path: str) -> list[str]:
        return [part for part in abs_path.split("/") if part]

    def _lookup(self, path: str) -> Node:
        """Find the node at ``path`` or raise the error bash would report."""
        abs_path = self.resolve(path)
        node: Node = self.root
        for part in self._split(abs_path):
            if isinstance(node, VFile):
                raise NotADirectory(path)
            child = node.children.get(part)
            if child is None:
                raise NotFound(path)
            node = child
        return node

    def _parent_dir(self, path: str) -> tuple[VDir, str]:
        """Return the folder that holds ``path`` and the final name in it."""
        abs_path = self.resolve(path)
        if abs_path == "/":
            raise InvalidOperation(path, "Invalid argument")
        parent_path, name = posixpath.split(abs_path)
        try:
            parent = self._lookup(parent_path)
        except NotFound:
            raise NotFound(path) from None
        except NotADirectory:
            raise NotADirectory(path) from None
        if isinstance(parent, VFile):
            raise NotADirectory(path)
        return parent, name

    def _dir_at(self, path: str) -> VDir:
        node = self._lookup(path)
        if isinstance(node, VFile):
            raise NotADirectory(path)
        return node

    def _file_at(self, path: str) -> VFile:
        node = self._lookup(path)
        if isinstance(node, VDir):
            raise IsADirectory(path)
        return node

    # --- queries ---------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """True if anything (file or folder) lives at ``path``."""
        try:
            self._lookup(path)
        except (NotFound, NotADirectory):
            return False
        return True

    def is_dir(self, path: str) -> bool:
        """True if ``path`` is an existing folder."""
        try:
            return isinstance(self._lookup(path), VDir)
        except (NotFound, NotADirectory):
            return False

    def is_file(self, path: str) -> bool:
        """True if ``path`` is an existing file."""
        try:
            return isinstance(self._lookup(path), VFile)
        except (NotFound, NotADirectory):
            return False

    def read_file(self, path: str) -> bytes:
        """Return the bytes of a file."""
        return self._file_at(path).data

    def read_text(self, path: str) -> str:
        """Return a file decoded as UTF-8 (undecodable bytes become U+FFFD)."""
        return self.read_file(path).decode("utf-8", errors="replace")

    def listdir(self, path: str = "/") -> list[str]:
        """Names inside a folder, sorted so listings are stable."""
        return sorted(self._dir_at(path).children)

    def stat(self, path: str) -> VStat:
        """Size, logical mtime and flags of a path."""
        node = self._lookup(path)
        if isinstance(node, VDir):
            return VStat(len(node.children), node.mtime, True, True, node.read_only)
        return VStat(len(node.data), node.mtime, False, node.executable, node.read_only)

    def walk(self, path: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        """Like ``os.walk``: yields ``(folder, subfolder names, file names)`` top-down."""
        top = self.resolve(path)
        directory = self._dir_at(top)
        dirs = sorted(n for n, c in directory.children.items() if isinstance(c, VDir))
        files = sorted(n for n, c in directory.children.items() if isinstance(c, VFile))
        yield top, dirs, files
        for name in dirs:
            yield from self.walk(posixpath.join(top, name))

    # --- creating and changing -----------------------------------------------------

    def mkdir(self, path: str, parents: bool = False) -> None:
        """Create a folder; with ``parents`` create missing parents and accept an
        existing folder (like ``mkdir -p``)."""
        abs_path = self.resolve(path)
        if parents:
            self._mkdir_parents(abs_path, path)
            return
        parent, name = self._parent_dir(path)
        if name in parent.children:
            raise AlreadyExists(path)
        self._check_writable_dir(parent, path)
        parent.children[name] = VDir(mtime=self._tick())
        parent.mtime = self._clock
        self._notify("mkdir", abs_path)

    def _mkdir_parents(self, abs_path: str, shown: str) -> None:
        node = self.root
        walked = ""
        for part in self._split(abs_path):
            walked += "/" + part
            child = node.children.get(part)
            if child is None:
                self._check_writable_dir(node, shown)
                child = VDir(mtime=self._tick())
                node.children[part] = child
                node.mtime = self._clock
                self._notify("mkdir", walked)
            elif isinstance(child, VFile):
                # mkdir -p a/b where a is a file: bash says "File exists" for a.
                raise AlreadyExists(shown) if walked == abs_path else NotADirectory(shown)
            node = child

    @staticmethod
    def _check_writable_dir(directory: VDir, shown: str) -> None:
        if directory.read_only:
            raise PermissionDenied(shown)

    def write_file(self, path: str, data: bytes | str, append: bool = False) -> None:
        """Create or overwrite a file (or add to its end with ``append``).

        Strings are stored as UTF-8. Writing to a folder, into a missing folder or
        over a read-only system file raises the matching ``VfsError``.
        """
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        parent, name = self._parent_dir(path)
        existing = parent.children.get(name)
        if isinstance(existing, VDir):
            raise IsADirectory(path)
        if existing is None:
            self._check_writable_dir(parent, path)
            parent.children[name] = VFile(payload, self._tick())
            parent.mtime = self._clock
            self._notify("create", self.resolve(path))
            return
        if existing.read_only:
            raise PermissionDenied(path)
        existing.data = existing.data + payload if append else payload
        existing.mtime = self._tick()
        self._notify("write", self.resolve(path))

    def touch(self, path: str) -> None:
        """Create an empty file, or bump the mtime of an existing path (``touch``)."""
        if not self.exists(path):
            self.write_file(path, b"")
            return
        node = self._lookup(path)
        if node.read_only:
            raise PermissionDenied(path)
        node.mtime = self._tick()
        self._notify("write", self.resolve(path))

    def remove(self, path: str, recursive: bool = False) -> None:
        """Delete a file or folder. A non-empty folder needs ``recursive``."""
        parent, name = self._parent_dir(path)
        node = parent.children.get(name)
        if node is None:
            raise NotFound(path)
        if parent.read_only or node.read_only:
            raise PermissionDenied(path)
        if isinstance(node, VDir) and node.children:
            if not recursive:
                raise DirectoryNotEmpty(path)
            self._check_tree_removable(node, path)
        del parent.children[name]
        parent.mtime = self._tick()
        self._notify("remove", self.resolve(path))

    def _check_tree_removable(self, directory: VDir, shown: str) -> None:
        # Refuse up front so a failed rm -r never leaves a half-deleted tree.
        for name, child in directory.children.items():
            child_path = posixpath.join(shown, name)
            if child.read_only:
                raise PermissionDenied(child_path)
            if isinstance(child, VDir):
                self._check_tree_removable(child, child_path)

    def _target_path(self, src: str, dst: str) -> str:
        """Where ``src`` lands: inside ``dst`` if it is a folder, else at ``dst``.

        A trailing ``/`` promises that ``dst`` is a folder, so ``mv notes.txt docs/``
        must fail when ``docs`` is missing instead of renaming the file to ``docs``.
        """
        abs_dst = self.resolve(dst)
        if self.is_dir(abs_dst):
            return posixpath.join(abs_dst, posixpath.basename(self.resolve(src)))
        if dst.endswith("/") and not self.is_dir(src):
            raise NotADirectory(dst)
        return abs_dst

    def move(self, src: str, dst: str) -> str:
        """Rename or move ``src``; returns the absolute path it ended up at."""
        abs_src = self.resolve(src)
        node = self._lookup(abs_src)
        target = self._target_path(abs_src, dst)
        if target == abs_src:
            return target
        if target.startswith(abs_src + "/"):
            raise InvalidOperation(dst, f"cannot move '{src}' to a subdirectory of itself")
        src_parent, src_name = self._parent_dir(abs_src)
        if src_parent.read_only or node.read_only:
            raise PermissionDenied(src)
        dst_parent, dst_name = self._parent_dir(target)
        self._check_writable_dir(dst_parent, dst)
        existing = dst_parent.children.get(dst_name)
        if existing is not None and existing.read_only:
            # Moving over a protected file would replace it just like writing to it.
            raise PermissionDenied(dst)
        if isinstance(existing, VDir):
            raise IsADirectory(dst) if isinstance(node, VFile) else AlreadyExists(dst)
        if isinstance(node, VDir) and isinstance(existing, VFile):
            raise NotADirectory(dst)
        del src_parent.children[src_name]
        node.mtime = self._tick()
        dst_parent.children[dst_name] = node
        src_parent.mtime = dst_parent.mtime = self._clock
        self._notify("remove", abs_src)
        self._notify("mkdir" if isinstance(node, VDir) else "create", target)
        return target

    def copy(self, src: str, dst: str, recursive: bool = False) -> str:
        """Copy a file (or a folder with ``recursive``); returns the new path."""
        abs_src = self.resolve(src)
        node = self._lookup(abs_src)
        target = self._target_path(abs_src, dst)
        if isinstance(node, VDir):
            if not recursive:
                raise IsADirectory(src)
            if target == abs_src or target.startswith(abs_src + "/"):
                raise InvalidOperation(dst, f"cannot copy a directory, '{src}', into itself")
            self._copy_tree(abs_src, target)
            return target
        if target == abs_src:
            # cp refuses this: a real cp would truncate the file it is reading from.
            raise SameFile(src)
        self.write_file(target, node.data)
        self._file_at(target).executable = node.executable
        return target

    def _copy_tree(self, abs_src: str, target: str) -> None:
        if not self.is_dir(target):
            self.mkdir(target)
        for name in self.listdir(abs_src):
            self.copy(posixpath.join(abs_src, name), posixpath.join(target, name), True)

    def set_executable(self, path: str, executable: bool = True) -> None:
        """Turn the executable bit of a file on or off (``chmod +x``/``-x``)."""
        node = self._lookup(path)
        if node.read_only:
            raise PermissionDenied(path)
        if isinstance(node, VFile):
            node.executable = executable
            node.mtime = self._tick()
            self._notify("attr", self.resolve(path))

    def set_read_only(self, path: str, read_only: bool = True) -> None:
        """Mark a system file or folder as read-only (or writable again)."""
        self._lookup(path).read_only = read_only
        self._notify("attr", self.resolve(path))

    # --- globbing ----------------------------------------------------------------------

    def glob(self, pattern: str, cwd: str = "/") -> list[str]:
        """Expand ``* ? [..]`` in each path component, like the shell does.

        Results keep the pattern's style: ``src/*.c`` gives ``src/main.c`` and
        ``/usr/include/*.h`` gives absolute paths. Hidden names only match when the
        pattern component itself starts with a dot. No match gives an empty list.
        """
        absolute = pattern.startswith("/")
        components = [c for c in pattern.split("/") if c]
        start = "/" if absolute else ""
        matches = self._glob_from(start, components, cwd)
        return sorted(matches)

    def _glob_from(self, prefix: str, components: list[str], cwd: str) -> list[str]:
        if not components:
            return [prefix.rstrip("/") or "/"] if prefix else []
        head, rest = components[0], components[1:]
        base = self.resolve(prefix or ".", cwd)
        if not _has_magic(head):
            candidate = prefix + head
            if not self.exists(self.resolve(candidate, cwd)):
                return []
            if rest and not self.is_dir(self.resolve(candidate, cwd)):
                return []
            return self._glob_from(candidate + "/", rest, cwd) if rest else [candidate]
        if not self.is_dir(base):
            return []
        results: list[str] = []
        for name in self.listdir(base):
            if name.startswith(".") and not head.startswith("."):
                continue
            if not fnmatch.fnmatchcase(name, head):
                continue
            if rest:
                if self.is_dir(posixpath.join(base, name)):
                    results.extend(self._glob_from(prefix + name + "/", rest, cwd))
            else:
                results.append(prefix + name)
        return results

    # --- persistence --------------------------------------------------------------------

    def to_dict(self) -> dict:
        """A JSON-safe dump of the whole tree (text files as text, binary as base64)."""
        return {"version": 1, "clock": self._clock, "root": _node_to_dict(self.root)}

    @classmethod
    def from_dict(cls, data: dict) -> "VirtualFS":
        """Rebuild a filesystem from ``to_dict`` output."""
        vfs = cls()
        vfs.load_dict(data)
        return vfs

    def load_dict(self, data: dict) -> None:
        """Replace this tree in place with ``to_dict`` output (keeps listeners)."""
        root = _node_from_dict(data["root"])
        if not isinstance(root, VDir):
            raise ValueError("the root of a saved filesystem must be a folder")
        self.root = root
        self._clock = int(data.get("clock", 0))
        self._notify("reset", "/")

    def snapshot(self) -> dict:
        """An independent deep copy of the current state, for undo and replays."""
        return {"root": copy.deepcopy(self.root), "clock": self._clock}

    def restore(self, snap: dict) -> None:
        """Go back to a ``snapshot`` (the snapshot itself stays reusable)."""
        self.root = copy.deepcopy(snap["root"])
        self._clock = snap["clock"]
        self._notify("reset", "/")

    def export_to_disk(self, real_dir: str | Path, source: str = HOME) -> list[Path]:
        """Copy the folder ``source`` onto the real disk under ``real_dir``.

        This is the only method that writes to the real disk. Every destination is
        checked to stay inside ``real_dir``; a name that would escape it (or that
        Windows would read as a drive or a separator) raises ``PermissionDenied``
        before anything is written. Returns the files written.
        """
        target_root = Path(real_dir).resolve()
        plan = self._export_plan(self.resolve(source), target_root)
        written: list[Path] = []
        for real_path, data in plan:
            if data is None:
                real_path.mkdir(parents=True, exist_ok=True)
                continue
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.write_bytes(data)
            written.append(real_path)
        return written

    def _export_plan(self, source: str, target_root: Path) -> list[tuple[Path, bytes | None]]:
        # Build the whole plan first so an unsafe name aborts before any write.
        plan: list[tuple[Path, bytes | None]] = [(target_root, None)]
        for folder, dirs, files in self.walk(source):
            relative = posixpath.relpath(folder, source)
            for name in dirs + files:
                _check_export_name(name, posixpath.join(folder, name))
                rel = name if relative == "." else posixpath.join(relative, name)
                real_path = (target_root / rel).resolve()
                if target_root != real_path and target_root not in real_path.parents:
                    raise PermissionDenied(posixpath.join(folder, name),
                                           "escapes the export folder")
                is_dir = name in dirs
                data = None if is_dir else self.read_file(posixpath.join(folder, name))
                plan.append((real_path, data))
        return plan


# Names Windows maps to devices whatever the extension ("NUL.txt" is still NUL).
_WINDOWS_DEVICES = frozenset({"CON", "PRN", "AUX", "NUL",
                              *(f"COM{n}" for n in range(1, 10)),
                              *(f"LPT{n}" for n in range(1, 10))})
_UNSAFE_EXPORT_CHARS = frozenset('\\/:*?"<>|\0')


def _check_export_name(name: str, shown: str) -> None:
    """Reject names that mean something special on a real disk.

    Separators and drive colons could point outside the export folder, device names
    write to hardware instead of a file, and Windows silently drops trailing dots and
    spaces (so two different files could land on the same real name).
    """
    unsafe = (name in ("", ".", "..") or any(ch in _UNSAFE_EXPORT_CHARS for ch in name)
              or name.strip() != name or name.endswith(".")
              or name.split(".")[0].upper() in _WINDOWS_DEVICES)
    if unsafe:
        raise PermissionDenied(shown, "unsafe name for export")


def _check_saved_name(name: str) -> None:
    """Refuse a saved child name that is not one plain path component.

    A hand-edited or corrupted save could hold "..", "a/b" or "" as a name; such a
    node could never be reached by a path and could escape the export folder, so
    loading fails loudly instead.
    """
    if name in ("", ".", "..") or "/" in name or "\0" in name:
        raise ValueError(f"invalid name in saved filesystem: {name!r}")


def _node_to_dict(node: Node) -> dict:
    if isinstance(node, VDir):
        return {"type": "dir", "mtime": node.mtime, "read_only": node.read_only,
                "children": {name: _node_to_dict(child) for name, child in node.children.items()}}
    entry: dict = {"type": "file", "mtime": node.mtime, "executable": node.executable,
                   "read_only": node.read_only}
    try:
        entry["text"] = node.data.decode("utf-8")
    except UnicodeDecodeError:
        entry["base64"] = base64.b64encode(node.data).decode("ascii")
    return entry


def _node_from_dict(data: dict) -> Node:
    if data["type"] == "dir":
        for name in data["children"]:
            _check_saved_name(name)
        children = {name: _node_from_dict(child) for name, child in data["children"].items()}
        return VDir(children, int(data.get("mtime", 0)), bool(data.get("read_only", False)))
    if "base64" in data:
        payload = base64.b64decode(data["base64"])
    else:
        payload = data.get("text", "").encode("utf-8")
    return VFile(payload, int(data.get("mtime", 0)), bool(data.get("executable", False)),
                 bool(data.get("read_only", False)))


def standard_layout(project_name: str = "project", headers: dict[str, str] | None = None,
                    commands: Iterable[str] = BIN_COMMANDS) -> VirtualFS:
    """A fresh device: the user's project folder, read-only headers, /bin and /tmp.

    ``headers`` maps a header name (``"stdio.h"``, ``"device.h"``) to its text; the
    toolchain provides them because only it knows the prototypes. ``commands`` lists
    the /bin marker files to create (the shell can pass its registry).
    """
    vfs = VirtualFS()
    vfs.mkdir(f"{HOME}/{project_name}", parents=True)
    vfs.mkdir("/tmp")
    vfs.mkdir("/usr/include", parents=True)
    for name, text in sorted((headers or {}).items()):
        if "/" in name:
            vfs.mkdir(posixpath.dirname(f"/usr/include/{name}"), parents=True)
        vfs.write_file(f"/usr/include/{name}", text)
        vfs.set_read_only(f"/usr/include/{name}")
    vfs.set_read_only("/usr/include")
    vfs.mkdir("/bin")
    for command in commands:
        path = f"/bin/{command}"
        vfs.write_file(path, BUILTIN_MARKER + command.encode("utf-8") + b"\n")
        vfs.set_executable(path)
        vfs.set_read_only(path)
    vfs.set_read_only("/bin")
    return vfs
