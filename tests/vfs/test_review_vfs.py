"""Review tests for the virtual filesystem: copy/move edge cases, saves and export safety."""
from pathlib import Path

import pytest

from pycc.vfs import (HOME, InvalidOperation, NotADirectory, PermissionDenied, SameFile, VirtualFS,
                 standard_layout)


@pytest.fixture
def fs() -> VirtualFS:
    """A small tree: /a/x.txt, an empty /b and a home folder."""
    vfs = VirtualFS()
    vfs.mkdir("/a")
    vfs.mkdir("/b")
    vfs.write_file("/a/x.txt", "hello")
    vfs.mkdir(HOME, parents=True)
    return vfs


def test_resolve_folds_dot_dot_and_stays_at_root(fs: VirtualFS) -> None:
    assert fs.resolve("../../..", "/a") == "/"
    assert fs.resolve("src/../lib/./x.c", "/home") == "/home/lib/x.c"
    assert fs.resolve("~/../user/p") == f"{HOME}/p"


def test_copy_file_onto_itself_is_refused_and_keeps_data(fs: VirtualFS) -> None:
    with pytest.raises(SameFile):
        fs.copy("/a/x.txt", "/a/x.txt")
    with pytest.raises(SameFile):
        fs.copy("/a/x.txt", "/a")            # lands on itself inside its own folder
    assert fs.read_text("/a/x.txt") == "hello"


def test_same_file_is_an_invalid_operation_with_stable_code() -> None:
    error = SameFile("x")
    assert isinstance(error, InvalidOperation)
    assert error.code == "same_file"


def test_copy_directory_onto_existing_directory_nests_it(fs: VirtualFS) -> None:
    assert fs.copy("/a", "/b", recursive=True) == "/b/a"
    assert fs.read_text("/b/a/x.txt") == "hello"
    assert fs.is_file("/a/x.txt")


def test_copy_directory_into_itself_is_refused(fs: VirtualFS) -> None:
    with pytest.raises(InvalidOperation):
        fs.copy("/a", "/a", recursive=True)


def test_move_across_directories_with_rename(fs: VirtualFS) -> None:
    assert fs.move("/a/x.txt", "/b/y.txt") == "/b/y.txt"
    assert not fs.exists("/a/x.txt")
    assert fs.read_text("/b/y.txt") == "hello"


def test_move_onto_read_only_file_is_refused(fs: VirtualFS) -> None:
    fs.write_file("/b/locked.txt", "keep me")
    fs.set_read_only("/b/locked.txt")
    with pytest.raises(PermissionDenied):
        fs.move("/a/x.txt", "/b/locked.txt")
    assert fs.read_text("/b/locked.txt") == "keep me"
    assert fs.exists("/a/x.txt")


def test_trailing_slash_destination_must_be_a_folder(fs: VirtualFS) -> None:
    with pytest.raises(NotADirectory):
        fs.move("/a/x.txt", "/nowhere/")
    with pytest.raises(NotADirectory):
        fs.copy("/a/x.txt", "/nowhere/")
    assert fs.exists("/a/x.txt") and not fs.exists("/nowhere")


def test_trailing_slash_folder_rename_still_works(fs: VirtualFS) -> None:
    assert fs.move("/a", "/renamed/") == "/renamed"
    assert fs.is_file("/renamed/x.txt")


def test_remove_root_is_refused(fs: VirtualFS) -> None:
    with pytest.raises(InvalidOperation):
        fs.remove("/", recursive=True)
    assert fs.exists("/a/x.txt")


@pytest.mark.parametrize("bad_name", ["..", ".", "", "a/b", "x" + chr(0)])
def test_from_dict_rejects_names_that_are_not_one_component(bad_name: str) -> None:
    data = {"version": 1, "clock": 1, "root": {"type": "dir", "children": {
        bad_name: {"type": "file", "text": "boom"}}}}
    with pytest.raises(ValueError):
        VirtualFS.from_dict(data)


def test_round_trip_keeps_binary_and_flags() -> None:
    vfs = standard_layout("project")
    vfs.write_file(f"{HOME}/project/bot", bytes([0, 255, 1]))
    vfs.set_executable(f"{HOME}/project/bot")
    copy = VirtualFS.from_dict(vfs.to_dict())
    assert copy.read_file(f"{HOME}/project/bot") == bytes([0, 255, 1])
    assert copy.stat(f"{HOME}/project/bot").executable
    assert copy.stat("/bin/ls").read_only


@pytest.mark.parametrize("bad_name", ["CON", "nul.txt", "Com1.c", "notes.", "a:b", "a*b",
                                      "trailing "])
def test_export_refuses_names_unsafe_on_a_real_disk(tmp_path: Path, bad_name: str) -> None:
    vfs = VirtualFS()
    vfs.mkdir(f"{HOME}/project", parents=True)
    vfs.write_file(f"{HOME}/project/ok.c", "int x;\n")
    vfs.write_file(f"{HOME}/project/{bad_name}", "x")
    with pytest.raises(PermissionDenied):
        vfs.export_to_disk(tmp_path / "out")
    # The plan is checked before any write, so nothing at all reached the disk.
    assert not (tmp_path / "out").exists()


def test_export_never_writes_outside_the_target(tmp_path: Path) -> None:
    vfs = VirtualFS()
    vfs.mkdir(f"{HOME}/project/src", parents=True)
    vfs.write_file(f"{HOME}/project/src/main.c", "int main(void) { return 0; }\n")
    written = vfs.export_to_disk(tmp_path / "out", f"{HOME}/project")
    target = (tmp_path / "out").resolve()
    assert written and all(target in path.parents for path in written)
    assert (tmp_path / "out" / "src" / "main.c").read_text() == "int main(void) { return 0; }\n"


def test_export_through_a_symlinked_folder_cannot_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    export_root = tmp_path / "out"
    export_root.mkdir()
    try:
        (export_root / "src").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this machine does not allow creating symlinks")
    vfs = VirtualFS()
    vfs.mkdir(f"{HOME}/src", parents=True)
    vfs.write_file(f"{HOME}/src/evil.c", "x")
    with pytest.raises(PermissionDenied):
        vfs.export_to_disk(export_root)
    assert not (outside / "evil.c").exists()
