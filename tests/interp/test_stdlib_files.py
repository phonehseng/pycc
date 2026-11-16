"""fopen and friends: modes, errno on failure, write-through, invalid streams, cleanup."""
from __future__ import annotations

import pytest

from pycc.interp import constructs
from pycc.interp.c_types import INT
from pycc.interp.hostio import EACCES, EINVAL, EISDIR, ENOENT, DictHostFiles
from pycc.interp.stdlib import finalize
from pycc.interp.stdlib.fake_machine import FakeMachine
from pycc.interp.stdlib.stdio import EOF, FILE_TYPE
from pycc.interp.traps import TrapError
from pycc.interp.values import Value


def fopen(machine: FakeMachine, path: str, mode: str) -> Value:
    return machine.call("fopen", machine.ptr(machine.literal(path)),
                        machine.ptr(machine.literal(mode)))


def fprintf(machine: FakeMachine, stream: Value, text: str) -> int:
    return machine.call("fprintf", stream, machine.ptr(machine.literal("%s")),
                        machine.ptr(machine.literal(text))).raw


def test_fopen_missing_file_returns_null_and_sets_enoent() -> None:
    machine = FakeMachine()
    assert fopen(machine, "missing.txt", "r").raw == 0
    assert machine.errno == ENOENT


def test_fopen_without_filesystem_sets_enoent() -> None:
    machine = FakeMachine(no_files=True)
    assert fopen(machine, "x.txt", "w").raw == 0
    assert machine.errno == ENOENT


def test_fopen_bad_mode_sets_einval() -> None:
    machine = FakeMachine()
    assert fopen(machine, "x.txt", "q").raw == 0
    assert machine.errno == EINVAL


def test_fopen_directory_sets_eisdir() -> None:
    files = DictHostFiles()
    files.dirs.add("/data")
    machine = FakeMachine(files=files)
    assert fopen(machine, "/data", "r").raw == 0
    assert machine.errno == EISDIR


def test_fopen_read_only_file_for_writing_sets_eacces() -> None:
    files = DictHostFiles({"/notes.txt": b"x"}, read_only={"/notes.txt"})
    machine = FakeMachine(files=files)
    assert fopen(machine, "/notes.txt", "w").raw == 0
    assert machine.errno == EACCES
    assert fopen(machine, "/notes.txt", "r+").raw == 0
    assert fopen(machine, "/notes.txt", "r").raw != 0


def test_fopen_counts_construct_and_returns_file_pointer() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "out.txt", "w")
    assert stream.raw != 0
    assert stream.ctype.target is FILE_TYPE
    assert machine.constructs[constructs.FILE_OPEN] == 1


def test_file_object_lives_in_data_segment_not_heap() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "out.txt", "w")
    assert machine.memory.object_at(stream.raw).kind == "global"
    assert machine.memory.leaks() == []


def test_relative_path_joined_with_cwd() -> None:
    machine = FakeMachine(cwd="/home/user/project")
    stream = fopen(machine, "log.txt", "w")
    fprintf(machine, stream, "hi")
    assert machine.files.files["/home/user/project/log.txt"] == b"hi"


def test_write_through_is_visible_before_fclose() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/out.txt", "w")
    fprintf(machine, stream, "line 1\n")
    assert machine.files.files["/out.txt"] == b"line 1\n"
    fprintf(machine, stream, "line 2\n")
    assert machine.files.files["/out.txt"] == b"line 1\nline 2\n"


def test_mode_w_truncates_existing_file() -> None:
    machine = FakeMachine(files={"/f.txt": b"old contents"})
    fopen(machine, "/f.txt", "w")
    assert machine.files.files["/f.txt"] == b""


def test_mode_a_appends() -> None:
    machine = FakeMachine(files={"/f.txt": b"one\n"})
    stream = fopen(machine, "/f.txt", "a")
    fprintf(machine, stream, "two\n")
    assert machine.files.files["/f.txt"] == b"one\ntwo\n"


def test_mode_a_creates_missing_file() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/new.txt", "a")
    assert stream.raw != 0
    assert machine.files.files["/new.txt"] == b""


def test_mode_r_plus_overwrites_in_place() -> None:
    machine = FakeMachine(files={"/f.txt": b"abcdef"})
    stream = fopen(machine, "/f.txt", "r+")
    fprintf(machine, stream, "XY")
    assert machine.files.files["/f.txt"] == b"XYcdef"


def test_mode_w_plus_reads_back_nothing_new() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/f.txt", "w+")
    assert machine.call("fgetc", stream).raw == EOF


def test_mode_a_plus_reads_from_start() -> None:
    machine = FakeMachine(files={"/f.txt": b"abc"})
    stream = fopen(machine, "/f.txt", "a+")
    assert machine.call("fgetc", stream).raw == ord("a")


def test_binary_flag_is_accepted() -> None:
    machine = FakeMachine(files={"/f.bin": b"\x01\x02"})
    assert fopen(machine, "/f.bin", "rb").raw != 0
    assert fopen(machine, "/f.bin", "rb+").raw != 0


def test_writing_a_read_only_stream_fails_and_sets_ferror() -> None:
    machine = FakeMachine(files={"/f.txt": b"x"})
    stream = fopen(machine, "/f.txt", "r")
    assert fprintf(machine, stream, "no") == EOF
    assert machine.call("ferror", stream).raw == 1
    machine.call("clearerr", stream)
    assert machine.call("ferror", stream).raw == 0


def test_reading_a_write_only_stream_returns_eof() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/f.txt", "w")
    assert machine.call("fgetc", stream).raw == EOF
    assert machine.call("ferror", stream).raw == 1


def test_fgets_reads_file_line_by_line() -> None:
    machine = FakeMachine(files={"/crops.txt": b"wheat\ncorn\n"})
    stream = fopen(machine, "/crops.txt", "r")
    buf = machine.buffer(32)
    lines = []
    while machine.call("fgets", machine.ptr(buf), machine.int_(32), stream).raw != 0:
        lines.append(machine.text_at(buf))
    assert lines == ["wheat\n", "corn\n"]
    assert machine.call("feof", stream).raw == 1


def test_fgetc_and_ungetc_on_file() -> None:
    machine = FakeMachine(files={"/f.txt": b"ab"})
    stream = fopen(machine, "/f.txt", "r")
    first = machine.call("fgetc", stream).raw
    machine.call("ungetc", machine.int_(first), stream)
    assert machine.call("getc", stream).raw == ord("a")
    assert machine.call("fgetc", stream).raw == ord("b")
    assert machine.call("fgetc", stream).raw == EOF


def test_fscanf_reads_numbers_from_file() -> None:
    machine = FakeMachine(files={"/n.txt": b"3 4 5"})
    stream = fopen(machine, "/n.txt", "r")
    total = 0
    value = machine.buffer(4)
    while machine.call("fscanf", stream, machine.ptr(machine.literal("%d")),
                       machine.ptr(value)).raw == 1:
        total += machine.memory.load(value, INT)
    assert total == 12
    assert machine.constructs[constructs.FILE_READ] >= 1


def test_fwrite_and_fread_round_trip() -> None:
    machine = FakeMachine()
    data = machine.buffer(8)
    for index in range(2):
        machine.memory.store(data + 4 * index, INT, 1000 + index)
    out = fopen(machine, "/nums.bin", "w")
    assert machine.call("fwrite", machine.ptr(data), machine.size(4), machine.size(2), out).raw == 2
    machine.call("fclose", out)
    back = machine.buffer(8)
    stream = fopen(machine, "/nums.bin", "r")
    read = machine.call("fread", machine.ptr(back), machine.size(4), machine.size(5), stream)
    assert read.raw == 2
    assert machine.memory.load(back + 4, INT) == 1001


def test_fwrite_to_stdout_outputs_text() -> None:
    machine = FakeMachine()
    text = machine.string("abc")
    machine.call("fwrite", machine.ptr(text), machine.size(1), machine.size(3),
                 machine.stream("stdout"))
    assert machine.stdout == "abc"


def test_fputs_and_fputc_write_through() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/f.txt", "w")
    machine.call("fputs", machine.ptr(machine.literal("ab")), stream)
    machine.call("fputc", machine.int_(ord("c")), stream)
    machine.call("putc", machine.int_(ord("d")), stream)
    assert machine.files.files["/f.txt"] == b"abcd"
    assert machine.constructs[constructs.FILE_WRITE] == 3


def test_fclose_then_use_traps_invalid_stream() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/f.txt", "w")
    assert machine.call("fclose", stream).raw == 0
    with pytest.raises(TrapError) as caught:
        fprintf(machine, stream, "late")
    assert caught.value.code == "invalid_stream"
    assert "already closed" in caught.value.message


def test_double_fclose_traps_invalid_stream() -> None:
    machine = FakeMachine()
    stream = fopen(machine, "/f.txt", "w")
    machine.call("fclose", stream)
    with pytest.raises(TrapError) as caught:
        machine.call("fclose", stream)
    assert caught.value.code == "invalid_stream"


def test_garbage_file_pointer_traps_invalid_stream() -> None:
    machine = FakeMachine()
    garbage = machine.buffer(4)
    with pytest.raises(TrapError) as caught:
        machine.call("fgetc", machine.ptr(garbage, FILE_TYPE))
    assert caught.value.code == "invalid_stream"
    assert caught.value.friendly


def test_null_file_pointer_traps_null_dereference() -> None:
    machine = FakeMachine()
    with pytest.raises(TrapError) as caught:
        machine.call("fgets", machine.ptr(machine.buffer(8)), machine.int_(8), machine.null())
    assert caught.value.code == "null_dereference"
    assert "fopen returns NULL" in caught.value.friendly


def test_fflush_accepts_null_and_open_streams() -> None:
    machine = FakeMachine()
    assert machine.call("fflush", machine.null()).raw == 0
    assert machine.call("fflush", machine.stream("stdout")).raw == 0


def test_remove_and_rename() -> None:
    machine = FakeMachine(files={"/a.txt": b"x"}, cwd="/")
    assert machine.call("rename", machine.ptr(machine.literal("a.txt")),
                        machine.ptr(machine.literal("b.txt"))).raw == 0
    assert "/b.txt" in machine.files.files
    assert machine.call("remove", machine.ptr(machine.literal("b.txt"))).raw == 0
    assert machine.files.files == {}


def test_remove_missing_sets_errno() -> None:
    machine = FakeMachine()
    assert machine.call("remove", machine.ptr(machine.literal("nope"))).raw == -1
    assert machine.errno == ENOENT


def test_perror_prints_message_and_strerror() -> None:
    machine = FakeMachine()
    fopen(machine, "missing.txt", "r")
    machine.call("perror", machine.ptr(machine.literal("fopen")))
    assert machine.stderr == "fopen: No such file or directory\n"


def test_perror_with_null_prints_only_strerror() -> None:
    machine = FakeMachine()
    machine.set_errno(EACCES)
    machine.call("perror", machine.null())
    assert machine.stderr == "Permission denied\n"


def test_finalize_warns_about_open_files() -> None:
    machine = FakeMachine()
    fopen(machine, "/kept.txt", "w")
    closed = fopen(machine, "/closed.txt", "w")
    machine.call("fclose", closed)
    finalize(machine)
    assert machine.warning_codes() == ["file_not_closed"]
    assert "/kept.txt" in machine.warnings[0].message


def test_finalize_ignores_standard_streams() -> None:
    machine = FakeMachine()
    machine.call("printf", machine.ptr(machine.literal("x")))
    finalize(machine)
    assert machine.warning_codes() == []


def test_standard_streams_are_distinct_file_objects() -> None:
    machine = FakeMachine()
    addresses = {machine.stream(name).raw for name in ("stdin", "stdout", "stderr")}
    assert len(addresses) == 3
    assert all(machine.memory.load(addr, INT) in (0, 1, 2) for addr in addresses)


def test_writing_to_stdin_fails() -> None:
    machine = FakeMachine()
    assert fprintf(machine, machine.stream("stdin"), "x") == EOF
