"""interp.c_types.format_value: values as a debugger's variables panel shows them."""
from __future__ import annotations

from pycc.interp.c_types import (
    CHAR, DOUBLE, INT, ArrayType, EnumType, Member, PointerType, StructType, format_value,
)
from pycc.interp.memory import Memory


def _memory() -> tuple[Memory, object]:
    """A fresh memory and main's frame."""
    memory = Memory(seed=1)
    return memory, memory.push_frame("main")


def test_scalars_pointers_and_chars() -> None:
    memory, frame = _memory()
    x = memory.alloc_local(frame, 4, 4, "x", INT)
    memory.store(x, INT, -3)
    d = memory.alloc_local(frame, 8, 8, "d", DOUBLE)
    memory.store(d, DOUBLE, 2.5)
    c = memory.alloc_local(frame, 1, 1, "c", CHAR)
    memory.store(c, CHAR, ord("\n"))
    p = memory.alloc_local(frame, 8, 8, "p", PointerType(INT))
    memory.store(p, PointerType(INT), x)
    q = memory.alloc_local(frame, 8, 8, "q", PointerType(INT))
    memory.store(q, PointerType(INT), 0)
    assert format_value(memory, x, INT) == "-3"
    assert format_value(memory, d, DOUBLE) == "2.5"
    assert format_value(memory, c, CHAR) == "'\\n'"
    assert format_value(memory, p, PointerType(INT)) == f"0x{x:x}"
    assert format_value(memory, q, PointerType(INT)) == "NULL"


def test_uninitialised_bytes_show_a_question_mark() -> None:
    memory, frame = _memory()
    y = memory.alloc_local(frame, 4, 4, "y", INT)
    assert format_value(memory, y, INT) == "?"


def test_arrays_strings_and_the_item_limit() -> None:
    memory, frame = _memory()
    name_type = ArrayType(CHAR, 6)
    name = memory.alloc_local(frame, 6, 1, "name", name_type)
    memory.write_bytes(name, b"oat\0\0\0")
    row_type = ArrayType(INT, 20)
    row = memory.alloc_local(frame, 80, 4, "row", row_type)
    for index in range(20):
        memory.store(row + 4 * index, INT, index)
    half_type = ArrayType(INT, 3)
    half = memory.alloc_local(frame, 12, 4, "half", half_type)
    memory.store(half, INT, 9)
    assert format_value(memory, name, name_type) == '"oat"'
    assert format_value(memory, row, row_type, max_items=4) == "{0, 1, 2, 3, ...}"
    assert format_value(memory, half, half_type) == "{9, ?, ?}"


def test_an_unterminated_char_array_shows_its_characters() -> None:
    memory, frame = _memory()
    crop_type = ArrayType(CHAR, 3)
    crop = memory.alloc_local(frame, 3, 1, "crop", crop_type)
    memory.write_bytes(crop, b"abc")
    assert format_value(memory, crop, crop_type) == "{'a', 'b', 'c'}"


def test_structs_and_enums() -> None:
    memory, frame = _memory()
    point = StructType("Point", [Member("x", INT), Member("y", INT)])
    where = memory.alloc_local(frame, point.size, point.align, "where", point)
    memory.store(where, INT, 4)
    memory.store(where + 4, INT, 2)
    crop = EnumType("Crop", (("WHEAT", 0), ("CORN", 1)))
    kind = memory.alloc_local(frame, 4, 4, "kind", crop)
    memory.store(kind, crop, 1)
    assert format_value(memory, where, point) == "{x = 4, y = 2}"
    assert format_value(memory, where, StructType("Later")) == "{...}"
    assert format_value(memory, kind, crop) == "CORN (1)"
