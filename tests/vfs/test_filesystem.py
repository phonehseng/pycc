"""Tests for the virtual filesystem: operations, errors, time, events, persistence."""
import json

import pytest

from pycc.vfs import (AlreadyExists, DirectoryNotEmpty, InvalidOperation, IsADirectory,
                 NotADirectory, NotFound, PermissionDenied, VfsError, VirtualFS,
                 standard_layout)
from pycc.vfs.filesystem import BUILTIN_MARKER


@pytest.fixture
def fs() -> VirtualFS:
    vfs = VirtualFS()
    vfs.mkdir("/home/user/project", parents=True)
    return vfs


# --- resolve ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, cwd, expected", [
    ("a.txt", "/home/user", "/home/user/a.txt"),
    ("./src/../main.c", "/p", "/p/main.c"),
    ("..", "/home/user", "/home"),
    ("../../../..", "/home/user", "/"),
    ("~", "/tmp", "/home/user"),
    ("~/project", "/tmp", "/home/user/project"),
    ("/usr//include/", "/", "/usr/include"),
    ("", "/tmp", "/tmp"),
])
def test_resolve(fs, path, cwd, expected) -> None:
    assert fs.resolve(path, cwd) == expected


# --- files ----------------------------------------------------------------------------

def test_write_and_read_text(fs) -> None:
    fs.write_file("/home/user/project/a.txt", "hi\n")
    assert fs.read_file("/home/user/project/a.txt") == b"hi\n"
    assert fs.read_text("/home/user/project/a.txt") == "hi\n"


def test_write_bytes_and_append(fs) -> None:
    fs.write_file("/f", b"ab")
    fs.write_file("/f", "cd", append=True)
    assert fs.read_file("/f") == b"abcd"


def test_overwrite_replaces(fs) -> None:
    fs.write_file("/f", "one")
    fs.write_file("/f", "two")
    assert fs.read_text("/f") == "two"


def test_exists_is_dir_is_file(fs) -> None:
    fs.write_file("/f", "x")
    assert fs.exists("/f") and fs.is_file("/f") and not fs.is_dir("/f")
    assert fs.is_dir("/home") and not fs.is_file("/home")
    assert not fs.exists("/nope") and not fs.exists("/f/inside")


def test_listdir_is_sorted(fs) -> None:
    for name in ("zeta", "alpha", "Mid"):
        fs.write_file(f"/tmp_{name}", "")
    assert fs.listdir("/") == ["home", "tmp_Mid", "tmp_alpha", "tmp_zeta"]


def test_stat_reports_flags(fs) -> None:
    fs.write_file("/run.sh", "echo hi\n")
    fs.set_executable("/run.sh")
    info = fs.stat("/run.sh")
    assert (info.size, info.is_dir, info.executable, info.read_only) == (8, False, True, False)
    assert fs.stat("/home").is_dir


def test_walk_top_down(fs) -> None:
    fs.write_file("/home/user/project/main.c", "")
    fs.mkdir("/home/user/project/src")
    fs.write_file("/home/user/project/src/a.c", "")
    walked = list(fs.walk("/home/user/project"))
    assert walked[0] == ("/home/user/project", ["src"], ["main.c"])
    assert walked[1] == ("/home/user/project/src", [], ["a.c"])


# --- errors -------------------------------------------------------------------------------

def test_read_missing_raises_not_found(fs) -> None:
    with pytest.raises(NotFound) as info:
        fs.read_file("/nope.txt")
    assert info.value.code == "no_such_file"
    assert str(info.value) == "/nope.txt: No such file or directory"


def test_read_directory_raises_is_a_directory(fs) -> None:
    with pytest.raises(IsADirectory) as info:
        fs.read_file("/home")
    assert info.value.code == "is_a_directory"


def test_path_through_file_raises_not_a_directory(fs) -> None:
    fs.write_file("/f", "")
    with pytest.raises(NotADirectory):
        fs.write_file("/f/g", "")
    with pytest.raises(NotADirectory):
        fs.listdir("/f")


def test_write_into_missing_folder(fs) -> None:
    with pytest.raises(NotFound):
        fs.write_file("/missing/f.txt", "")


def test_write_over_folder(fs) -> None:
    with pytest.raises(IsADirectory):
        fs.write_file("/home", "")


def test_mkdir_existing_raises(fs) -> None:
    with pytest.raises(AlreadyExists) as info:
        fs.mkdir("/home")
    assert info.value.code == "file_exists"


def test_mkdir_parents_accepts_existing(fs) -> None:
    fs.mkdir("/home/user/project/a/b/c", parents=True)
    fs.mkdir("/home/user/project/a/b", parents=True)
    assert fs.is_dir("/home/user/project/a/b/c")


def test_mkdir_without_parent(fs) -> None:
    with pytest.raises(NotFound):
        fs.mkdir("/x/y")


def test_remove_non_empty_needs_recursive(fs) -> None:
    fs.write_file("/home/user/project/f", "")
    with pytest.raises(DirectoryNotEmpty) as info:
        fs.remove("/home/user/project")
    assert info.value.code == "directory_not_empty"
    fs.remove("/home/user/project", recursive=True)
    assert not fs.exists("/home/user/project")


def test_remove_empty_folder_and_file(fs) -> None:
    fs.mkdir("/empty")
    fs.remove("/empty")
    fs.write_file("/f", "")
    fs.remove("/f")
    assert not fs.exists("/empty") and not fs.exists("/f")


def test_remove_missing(fs) -> None:
    with pytest.raises(NotFound):
        fs.remove("/nope")


def test_read_only_file_and_folder(fs) -> None:
    fs.mkdir("/sys")
    fs.write_file("/sys/h.h", "x")
    fs.set_read_only("/sys/h.h")
    fs.set_read_only("/sys")
    for action in (lambda: fs.write_file("/sys/h.h", "y"),
                   lambda: fs.write_file("/sys/new.h", "y"),
                   lambda: fs.remove("/sys/h.h"),
                   lambda: fs.set_executable("/sys/h.h")):
        with pytest.raises(PermissionDenied) as info:
            action()
        assert info.value.code == "permission_denied"


def test_recursive_remove_refuses_read_only_inside(fs) -> None:
    fs.mkdir("/a/b", parents=True)
    fs.write_file("/a/b/locked", "")
    fs.set_read_only("/a/b/locked")
    with pytest.raises(PermissionDenied):
        fs.remove("/a", recursive=True)
    assert fs.exists("/a/b/locked")          # nothing half-deleted


def test_all_errors_are_vfs_errors() -> None:
    for cls in (NotFound, IsADirectory, NotADirectory, AlreadyExists, PermissionDenied,
                DirectoryNotEmpty, InvalidOperation):
        assert issubclass(cls, VfsError)


# --- move and copy ------------------------------------------------------------------------

def test_move_rename(fs) -> None:
    fs.write_file("/a", "1")
    assert fs.move("/a", "/b") == "/b"
    assert fs.read_text("/b") == "1" and not fs.exists("/a")


def test_move_into_folder(fs) -> None:
    fs.write_file("/a", "1")
    assert fs.move("/a", "/home") == "/home/a"


def test_move_folder_into_itself(fs) -> None:
    with pytest.raises(InvalidOperation):
        fs.move("/home", "/home/user")


def test_copy_file_keeps_executable(fs) -> None:
    fs.write_file("/a", "1")
    fs.set_executable("/a")
    fs.copy("/a", "/b")
    assert fs.read_text("/b") == "1" and fs.stat("/b").executable and fs.exists("/a")


def test_copy_folder_needs_recursive(fs) -> None:
    fs.write_file("/home/user/project/f", "x")
    with pytest.raises(IsADirectory):
        fs.copy("/home/user/project", "/backup")
    fs.copy("/home/user/project", "/backup", recursive=True)
    assert fs.read_text("/backup/f") == "x"


# --- time and events ----------------------------------------------------------------------

def test_mtime_counter_increases(fs) -> None:
    fs.write_file("/a", "1")
    first = fs.stat("/a").mtime
    fs.write_file("/b", "2")
    assert fs.stat("/b").mtime > first
    fs.write_file("/a", "3")
    assert fs.stat("/a").mtime > fs.stat("/b").mtime


def test_touch_creates_and_bumps(fs) -> None:
    fs.touch("/t")
    before = fs.stat("/t").mtime
    fs.touch("/t")
    assert fs.read_file("/t") == b"" and fs.stat("/t").mtime > before


def test_move_bumps_mtime(fs) -> None:
    fs.write_file("/a", "")
    before = fs.clock
    fs.move("/a", "/b")
    assert fs.stat("/b").mtime > before


def test_on_change_events_and_unsubscribe(fs) -> None:
    seen = []
    unsubscribe = fs.on_change(lambda event, path: seen.append((event, path)))
    fs.write_file("/a", "1")
    fs.write_file("/a", "2")
    fs.mkdir("/d")
    fs.move("/a", "/d")
    fs.remove("/d", recursive=True)
    unsubscribe()
    fs.write_file("/ignored", "")
    assert seen == [("create", "/a"), ("write", "/a"), ("mkdir", "/d"), ("remove", "/a"),
                    ("create", "/d/a"), ("remove", "/d")]


# --- glob ---------------------------------------------------------------------------------

@pytest.fixture
def tree(fs) -> VirtualFS:
    for path in ("main.c", "farm.c", "farm.h", ".hidden.c", "src/a.c", "src/b.c", "src/x.h",
                 "lib/z.c", "Makefile"):
        full = f"/home/user/project/{path}"
        fs.mkdir(full.rsplit("/", 1)[0], parents=True)
        fs.write_file(full, "")
    return fs


CWD = "/home/user/project"


@pytest.mark.parametrize("pattern, expected", [
    ("*.c", ["farm.c", "main.c"]),
    ("?a*.c", ["farm.c", "main.c"]),
    ("f[a-z]rm.*", ["farm.c", "farm.h"]),
    ("src/*.c", ["src/a.c", "src/b.c"]),
    ("*/*.c", ["lib/z.c", "src/a.c", "src/b.c"]),
    (".*.c", [".hidden.c"]),
    ("*.py", []),
    ("Makefile", ["Makefile"]),
])
def test_glob_relative(tree, pattern, expected) -> None:
    assert tree.glob(pattern, CWD) == expected


def test_glob_absolute(tree) -> None:
    assert tree.glob("/home/user/project/src/*.h") == ["/home/user/project/src/x.h"]


def test_glob_escaped_brackets_match_literally(fs) -> None:
    fs.write_file("/a*b", "")
    fs.write_file("/axb", "")
    assert fs.glob("/a[*]b") == ["/a*b"]


# --- persistence --------------------------------------------------------------------------

def test_to_dict_round_trip_is_json_safe(tree) -> None:
    tree.write_file("/bin.dat", b"\x00\xff\x10")
    tree.set_executable("/home/user/project/main.c")
    data = json.loads(json.dumps(tree.to_dict()))
    copy = VirtualFS.from_dict(data)
    assert copy.read_file("/bin.dat") == b"\x00\xff\x10"
    assert copy.stat("/home/user/project/main.c").executable
    assert copy.glob("src/*.c", CWD) == ["src/a.c", "src/b.c"]
    assert copy.clock == tree.clock
    assert copy.to_dict() == tree.to_dict()


def test_snapshot_restore_is_deep(fs) -> None:
    fs.write_file("/a", "before")
    snap = fs.snapshot()
    fs.write_file("/a", "after")
    fs.write_file("/b", "new")
    fs.restore(snap)
    assert fs.read_text("/a") == "before" and not fs.exists("/b")
    fs.write_file("/a", "again")
    fs.restore(snap)                          # the snapshot was not modified
    assert fs.read_text("/a") == "before"


def test_restore_notifies_reset(fs) -> None:
    seen = []
    snap = fs.snapshot()
    fs.on_change(lambda event, path: seen.append(event))
    fs.restore(snap)
    assert seen == ["reset"]


# --- export -------------------------------------------------------------------------------

def test_export_to_disk(tree, tmp_path) -> None:
    tree.write_file("/home/user/project/src/a.c", "int main(void) { return 0; }\n")
    written = tree.export_to_disk(tmp_path / "out", "/home/user/project")
    assert (tmp_path / "out" / "src" / "a.c").read_text() == "int main(void) { return 0; }\n"
    assert (tmp_path / "out" / "Makefile").exists()
    assert len(written) == 9


def test_export_refuses_unsafe_names(fs, tmp_path) -> None:
    fs.write_file("/home/user/project/..\\evil", "x")
    with pytest.raises(PermissionDenied):
        fs.export_to_disk(tmp_path / "out", "/home/user/project")
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "out").exists()


def test_export_refuses_drive_names(fs, tmp_path) -> None:
    fs.write_file("/home/user/project/C:", "x")
    with pytest.raises(PermissionDenied):
        fs.export_to_disk(tmp_path / "out", "/home/user/project")


# --- standard layout ----------------------------------------------------------------------

def test_standard_layout() -> None:
    vfs = standard_layout("farm", {"stdio.h": "int puts(const char *s);\n"})
    assert vfs.is_dir("/home/user/farm") and vfs.is_dir("/tmp")
    assert vfs.read_text("/usr/include/stdio.h").startswith("int puts")
    assert vfs.stat("/usr/include/stdio.h").read_only
    with pytest.raises(PermissionDenied):
        vfs.write_file("/usr/include/new.h", "")
    marker = vfs.stat("/bin/ls")
    assert marker.executable and marker.read_only
    assert vfs.read_file("/bin/ls").startswith(BUILTIN_MARKER)
