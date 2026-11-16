"""Tests for the simulated memory: segments, traps, garbage, heap, stack and snapshots."""
from __future__ import annotations

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Phase, SourceLocation
from pycc.interp.c_types import (
    BOOL,
    CHAR,
    DOUBLE,
    FLOAT,
    INT,
    LONG,
    UCHAR,
    ArrayType,
    Member,
    PointerType,
    StructType,
)
from pycc.interp.memory import (
    HEAP_ALIGN,
    NULL_GUARD,
    Memory,
    MemorySnapshot,
)
from pycc.interp.traps import MemoryTrap

INT8 = ArrayType(INT, 8)


@pytest.fixture
def mem() -> Memory:
    """A default-sized memory with a fixed seed."""
    return Memory(seed=1)


def local(mem: Memory, ctype, name: str = "x", frame_name: str = "main") -> int:
    """Allocate one local in a fresh (or the current) frame and return its address."""
    frame = mem.frames[-1] if mem.frames else mem.push_frame(frame_name)
    return mem.alloc_local(frame, ctype.size, ctype.align, name, ctype)


def trap_of(func, *args) -> MemoryTrap:
    """Run ``func`` expecting a MemoryTrap and return it."""
    with pytest.raises(MemoryTrap) as info:
        func(*args)
    return info.value


# --- segments and layout ----------------------------------------------------------------

def test_segments_are_ordered(mem: Memory) -> None:
    assert NULL_GUARD == mem.data_start < mem.data_limit <= mem.heap_start
    assert mem.heap_start < mem.heap_end < mem.stack_low < mem.size
    assert mem.size - mem.stack_low == 256 << 10


def test_too_small_memory_is_rejected() -> None:
    with pytest.raises(ValueError):
        Memory(size=8192, stack_size=4096)


# --- null and wild pointers -------------------------------------------------------------

def test_null_dereference_trap(mem: Memory) -> None:
    trap = trap_of(mem.load, 0, INT)
    assert trap.code == "null_dereference"
    assert "NULL" in trap.friendly
    assert trap.concept is Concept.POINTERS


def test_null_plus_offset_is_still_null(mem: Memory) -> None:
    assert trap_of(mem.store, 16, INT, 1).code == "null_dereference"


def test_wild_pointer_trap(mem: Memory) -> None:
    trap = trap_of(mem.load, mem.heap_start + 50_000, INT)
    assert trap.code == "wild_pointer"
    assert trap.friendly == ("This pointer does not point at any variable or allocation; "
                             "it may be garbage or past the end of an array.")


def test_address_beyond_memory_is_wild(mem: Memory) -> None:
    assert trap_of(mem.load, mem.size + 100, INT).code == "wild_pointer"


def test_far_from_any_object_is_wild_not_out_of_bounds(mem: Memory) -> None:
    addr = mem.alloc_global(INT8.size, 4, "field", INT8)
    assert trap_of(mem.load, addr + 400, INT).code == "wild_pointer"


# --- out of bounds -----------------------------------------------------------------------

def test_global_array_index_past_end(mem: Memory) -> None:
    addr = mem.alloc_global(INT8.size, 4, "field", INT8)
    trap = trap_of(mem.load, addr + 8 * 4, INT)
    assert trap.code == "out_of_bounds"
    assert trap.message == "index 8 is past the end of 'field' (8 elements)"
    assert trap.extra["index"] == 8 and trap.extra["length"] == 8
    assert "0 to 7" in trap.friendly


def test_local_array_index_past_end(mem: Memory) -> None:
    local(mem, INT, "before")
    addr = local(mem, INT8, "field")
    trap = trap_of(mem.store, addr + 32, INT, 5)
    assert trap.message == "index 8 is past the end of 'field' (8 elements)"


def test_negative_index_is_before_start(mem: Memory) -> None:
    addr = local(mem, INT8, "field")
    trap = trap_of(mem.load, addr - 4, INT)
    assert trap.message == "index -1 is before the start of 'field' (8 elements)"


def test_origin_catches_overrun_into_a_neighbour(mem: Memory) -> None:
    field = mem.alloc_global(INT8.size, 4, "field", INT8)
    other = mem.alloc_global(INT8.size, 4, "other", INT8)
    mem.load(other, INT)                                  # fine on its own
    trap = trap_of(mem.load, other, INT, field)           # but not when derived from field
    assert trap.code == "out_of_bounds"
    assert "'field'" in trap.message


def test_access_crossing_end_of_scalar(mem: Memory) -> None:
    addr = local(mem, INT, "count")
    trap = trap_of(mem.load, addr, LONG)
    assert trap.code == "out_of_bounds"
    assert "'count' (4 bytes)" in trap.message


def test_heap_overrun_counts_in_access_units(mem: Memory) -> None:
    block = mem.malloc(16)
    trap = trap_of(mem.load, block + 16, INT)
    assert trap.message == "index 4 is past the end of 'malloc(16)' (4 elements)"


# --- literals and strings ----------------------------------------------------------------

def test_string_interning_dedups(mem: Memory) -> None:
    first = mem.intern_string(b"hello")
    assert mem.intern_string(b"hello") == first
    assert mem.intern_string(b"other") != first
    assert mem.read_cstring(first) == b"hello"


def test_write_to_literal_traps(mem: Memory) -> None:
    addr = mem.intern_string(b"stone")
    trap = trap_of(mem.store, addr, CHAR, ord("S"))
    assert trap.code == "write_to_literal"
    assert "read-only" in trap.friendly
    assert trap.concept is Concept.STRINGS


def test_literal_object_is_char_array_with_nul(mem: Memory) -> None:
    obj = mem.object_at(mem.intern_string(b"abc"))
    assert obj.kind == "literal" and obj.size == 4 and obj.name == '"abc"'


def test_read_cstring_without_terminator_traps(mem: Memory) -> None:
    ctype = ArrayType(CHAR, 3)
    addr = mem.alloc_global(3, 1, "word", ctype)
    mem.write_bytes(addr, b"abc")
    trap = trap_of(mem.read_cstring, addr)
    assert trap.code == "string_not_terminated"
    assert "'word'" in trap.message


def test_read_cstring_respects_limit(mem: Memory) -> None:
    addr = mem.intern_string(b"abcdef")
    assert mem.read_cstring(addr, limit=3) == b"abc"


def test_write_cstring_overflow_traps(mem: Memory) -> None:
    addr = local(mem, ArrayType(CHAR, 4), "name")
    mem.write_cstring(addr, b"abc")
    assert mem.read_cstring(addr) == b"abc"
    assert trap_of(mem.write_cstring, addr, b"abcd").code == "out_of_bounds"


# --- uninitialised reads and garbage ---------------------------------------------------------

def test_uninitialized_local_read_warns(mem: Memory) -> None:
    addr = local(mem, INT, "count")
    mem.load(addr, INT)
    [warning] = mem.drain_warnings()
    assert warning.code == "uninitialized_read"
    assert warning.extra["object"] == "count"
    assert "'count'" in warning.message
    assert warning.concept is Concept.UNINITIALIZED


def test_uninitialized_warning_once_per_object(mem: Memory) -> None:
    addr = local(mem, INT, "count")
    mem.load(addr, INT)
    mem.load(addr, INT)
    assert len(mem.pending_warnings) == 1


def test_initialized_read_does_not_warn(mem: Memory) -> None:
    addr = local(mem, INT, "count")
    mem.store(addr, INT, 7)
    assert mem.load(addr, INT) == 7
    assert mem.pending_warnings == []


def test_garbage_is_deterministic_per_seed() -> None:
    def garbage(seed: int) -> int:
        memory = Memory(seed=seed)
        return memory.load(local(memory, LONG, "g"), LONG)

    assert garbage(7) == garbage(7)
    assert garbage(7) != garbage(8)


def test_warning_converts_to_runtime_diagnostic(mem: Memory) -> None:
    mem.load(local(mem, INT, "n"), INT)
    diagnostic = mem.pending_warnings[0].to_diagnostic(SourceLocation("main.c", 3, 5))
    assert diagnostic.severity.value == "warning" and diagnostic.phase is Phase.RUNTIME


def test_globals_are_zero_and_initialized(mem: Memory) -> None:
    addr = mem.alloc_global(4, 4, "total", INT)
    assert mem.load(addr, INT) == 0
    assert mem.pending_warnings == []


def test_malloc_is_garbage_and_calloc_is_zero(mem: Memory) -> None:
    mem.load(mem.malloc(8), INT)
    assert [w.code for w in mem.drain_warnings()] == ["uninitialized_read"]
    block = mem.calloc(4, 4)
    assert [mem.load(block + i * 4, INT) for i in range(4)] == [0, 0, 0, 0]
    assert mem.pending_warnings == []


def test_read_cstring_of_uninitialized_buffer_warns(mem: Memory) -> None:
    addr = local(mem, ArrayType(CHAR, 64), "buf")
    mem.store(addr + 63, CHAR, 0)          # guarantee a terminator somewhere
    mem.read_cstring(addr)
    assert mem.pending_warnings[0].code == "uninitialized_read"


# --- scalar wraparound and structs -----------------------------------------------------------

def test_unsigned_char_wraps(mem: Memory) -> None:
    addr = local(mem, UCHAR, "c")
    mem.store(addr, UCHAR, 300)
    assert mem.load(addr, UCHAR) == 44


def test_int_wraps_and_char_is_signed(mem: Memory) -> None:
    addr = local(mem, INT, "big")
    mem.store(addr, INT, 2**31)
    assert mem.load(addr, INT) == -(2**31)
    mem.store(addr, CHAR, 200)
    assert mem.load(addr, CHAR) == -56


def test_bool_float_and_pointer_round_trip(mem: Memory) -> None:
    flag, ratio, ptr = local(mem, BOOL, "f"), local(mem, FLOAT, "r"), local(mem, PointerType(INT), "p")
    mem.store(flag, BOOL, 300)
    mem.store(ratio, FLOAT, 0.1)
    mem.store(ptr, PointerType(INT), 0x1234)
    assert mem.load(flag, BOOL) == 1
    assert mem.load(ratio, FLOAT) == pytest.approx(0.1, rel=1e-6) and mem.load(ratio, FLOAT) != 0.1
    assert mem.load(ptr, PointerType(INT)) == 0x1234


def test_float_overflow_becomes_infinity(mem: Memory) -> None:
    addr = local(mem, FLOAT, "huge")
    mem.store(addr, FLOAT, 1e300)
    assert mem.load(addr, FLOAT) == float("inf")


def test_struct_members_by_offset(mem: Memory) -> None:
    tile = StructType("Tile", [Member("x", INT), Member("kind", CHAR), Member("water", DOUBLE)])
    addr = local(mem, tile, "t")
    for member, value in zip(tile.members, (3, 7, 2.5)):
        mem.store(addr + member.offset, member.ctype, value)
    assert [m.offset for m in tile.members] == [0, 4, 8]
    assert mem.load(addr + 8, DOUBLE) == 2.5 and mem.load(addr + 4, CHAR) == 7


def test_load_rejects_aggregate_types(mem: Memory) -> None:
    with pytest.raises(TypeError):
        mem.load(local(mem, INT8, "a"), INT8)


def test_copy_bytes_preserves_init_state(mem: Memory) -> None:
    src = local(mem, ArrayType(INT, 2), "src")
    dst = local(mem, ArrayType(INT, 2), "dst")
    mem.store(src, INT, 9)
    mem.copy_bytes(dst, src, 8)
    assert mem.load(dst, INT) == 9 and mem.pending_warnings == []
    mem.load(dst + 4, INT)
    assert mem.pending_warnings[0].extra["object"] == "dst"


# --- heap ----------------------------------------------------------------------------------

def test_heap_blocks_are_16_byte_aligned_and_separated(mem: Memory) -> None:
    first, second = mem.malloc(1), mem.malloc(1)
    assert first % HEAP_ALIGN == 0 and second % HEAP_ALIGN == 0
    assert second - first >= 32          # 16 header + 16 minimum block


def test_freed_block_is_reused(mem: Memory) -> None:
    first = mem.malloc(24)
    mem.free(first)
    assert mem.malloc(24) == first


def test_fragmentation_and_coalescing(mem: Memory) -> None:
    a, b, c, wall = mem.malloc(32), mem.malloc(32), mem.malloc(32), mem.malloc(32)
    mem.free(a)
    mem.free(c)
    big = mem.malloc(64)
    assert big > wall                    # neither 48-byte hole fits 64 + header
    mem.free(b)                          # a, b and c merge into one hole
    assert mem.malloc(96) == a


def test_heap_exhaustion_returns_null_and_warns() -> None:
    memory = Memory(size=1 << 16, stack_size=8192)
    assert memory.malloc(memory.heap_capacity) == 0
    [warning] = memory.drain_warnings()
    assert warning.code == "heap_exhausted"
    while memory.malloc(1024):
        pass
    assert memory.heap_used <= memory.heap_capacity


def test_realloc_moves_and_preserves_data(mem: Memory) -> None:
    old = mem.malloc(8)
    mem.store(old, INT, 41)
    new = mem.realloc(old, 64)
    assert new != old and mem.load(new, INT) == 41
    assert mem.pending_warnings == []
    mem.load(new + 4, INT)
    assert mem.pending_warnings[0].code == "uninitialized_read"
    assert trap_of(mem.load, old, INT).code == "use_after_free"


def test_realloc_null_zero_and_failure(mem: Memory) -> None:
    block = mem.realloc(0, 16)
    assert block and mem.realloc(block, 0) == 0
    keep = mem.malloc(16)
    assert mem.realloc(keep, mem.heap_capacity) == 0
    mem.store(keep, INT, 1)              # the original survives a failed realloc


def test_use_after_free(mem: Memory) -> None:
    block = mem.malloc(16)
    mem.free(block)
    trap = trap_of(mem.load, block, INT)
    assert trap.code == "use_after_free"
    assert "free()" in trap.friendly
    assert trap.concept is Concept.DYNAMIC_MEMORY


def test_double_free(mem: Memory) -> None:
    block = mem.malloc(16)
    mem.free(block)
    trap = trap_of(mem.free, block)
    assert trap.code == "double_free"
    assert "exactly one free" in trap.friendly


def test_invalid_free_of_interior_pointer(mem: Memory) -> None:
    block = mem.malloc(32)
    trap = trap_of(mem.free, block + 8)
    assert trap.code == "invalid_free"
    assert "8 bytes into the block" in trap.friendly


def test_invalid_free_of_local_and_wild(mem: Memory) -> None:
    assert "stack variable" in trap_of(mem.free, local(mem, INT, "n")).friendly
    assert trap_of(mem.free, mem.heap_start + 90_000).code == "invalid_free"


def test_free_null_is_a_no_op(mem: Memory) -> None:
    mem.free(0)
    assert mem.leaks() == []


def test_realloc_of_freed_block_traps(mem: Memory) -> None:
    block = mem.malloc(16)
    mem.free(block)
    assert trap_of(mem.realloc, block, 32).code == "use_after_free"


def test_leaks_lists_live_blocks(mem: Memory) -> None:
    kept, freed = mem.malloc(10), mem.malloc(20)
    mem.free(freed)
    assert [obj.start for obj in mem.leaks()] == [kept]
    assert mem.leaks()[0].name == "malloc(10)"


def test_heap_used_tracks_blocks(mem: Memory) -> None:
    block = mem.malloc(20)
    assert mem.heap_used == 16 + 32
    mem.free(block)
    assert mem.heap_used == 0


# --- stack ---------------------------------------------------------------------------------

def test_locals_are_8_byte_aligned(mem: Memory) -> None:
    frame = mem.push_frame("main")
    addrs = [mem.alloc_local(frame, size, 1, f"v{size}", CHAR) for size in (1, 3, 5)]
    assert all(addr % 8 == 0 for addr in addrs)
    assert mem.alloc_local(frame, 16, 16, "wide", None) % 16 == 0


def test_stack_grows_down(mem: Memory) -> None:
    first, second = local(mem, INT, "a"), local(mem, INT, "b")
    assert second < first < mem.size


def test_pop_frame_kills_locals_and_dangling_pointer_traps(mem: Memory) -> None:
    mem.push_frame("main")
    helper = mem.push_frame("helper", SourceLocation("farm.c", 4))
    addr = mem.alloc_local(helper, 4, 4, "result", INT)
    mem.store(addr, INT, 5)
    mem.pop_frame()
    trap = trap_of(mem.load, addr, INT)
    assert trap.code == "wild_pointer"
    assert "points at a local variable of a function that has already returned" in trap.friendly


def test_scope_release_kills_block_locals(mem: Memory) -> None:
    frame = mem.push_frame("main")
    outer = mem.alloc_local(frame, 4, 4, "outer", INT)
    mark = mem.scope_mark(frame)
    inner = mem.alloc_local(frame, 4, 4, "inner", INT)
    mem.scope_release(frame, mark)
    assert mem.stack_pointer == mark
    mem.store(outer, INT, 1)
    trap = trap_of(mem.load, inner, INT)
    assert trap.code == "wild_pointer" and "block" in trap.friendly
    assert [obj.name for obj in frame.locals] == ["outer"]


def test_released_slot_is_reused_with_old_bytes(mem: Memory) -> None:
    frame = mem.push_frame("main")
    mark = mem.scope_mark(frame)
    first = mem.alloc_local(frame, 4, 4, "i", INT)
    mem.store(first, INT, 99)
    mem.scope_release(frame, mark)
    second = mem.alloc_local(frame, 4, 4, "j", INT)
    assert second == first
    assert mem.load(second, INT) == 99          # honest garbage: the old value
    assert mem.pending_warnings[0].extra["object"] == "j"


def test_stack_overflow_by_size() -> None:
    memory = Memory(stack_size=4096)
    with pytest.raises(MemoryTrap) as info:
        for depth in range(1000):
            frame = memory.push_frame("recurse")
            memory.alloc_local(frame, 64, 8, "buf", None)
    assert info.value.code == "stack_overflow"
    assert info.value.concept is Concept.RECURSION
    assert 40 < depth < 64                      # 4096 bytes / (64 + red zone) per call


def test_stack_overflow_by_depth() -> None:
    memory = Memory(max_frames=10)
    for _ in range(10):
        memory.push_frame("recurse")
    trap = trap_of(memory.push_frame, "recurse")
    assert trap.code == "stack_overflow" and "10 nested calls" in trap.message


def test_alloc_local_requires_innermost_frame(mem: Memory) -> None:
    outer = mem.push_frame("main")
    mem.push_frame("helper")
    with pytest.raises(ValueError):
        mem.alloc_local(outer, 4, 4, "x", INT)


def test_pop_without_frame_is_an_error(mem: Memory) -> None:
    with pytest.raises(IndexError):
        mem.pop_frame()


# --- inspection -------------------------------------------------------------------------------

def test_snapshot_content(mem: Memory) -> None:
    tile = StructType("Tile", [Member("x", INT), Member("y", INT)])
    gaddr = mem.alloc_global(INT8.size, 4, "field", INT8)
    mem.store(gaddr + 4, INT, 6)
    frame = mem.push_frame("main", SourceLocation("main.c", 1))
    taddr = mem.alloc_local(frame, tile.size, tile.align, "t", tile)
    mem.store(taddr + 4, INT, 2)
    mem.malloc(20)
    snap = mem.snapshot()
    assert isinstance(snap, MemorySnapshot)
    by_name = {obj["name"]: obj for obj in snap.objects}
    assert by_name["field"]["preview"] == [0, 6, 0, 0, 0, 0, 0, 0]
    assert by_name["field"]["type"] == "int[8]" and by_name["field"]["kind"] == "global"
    assert by_name["t"]["preview"]["y"] == 2 and by_name["t"]["initialized"] is False
    assert by_name["malloc(20)"]["preview"] is None
    assert snap.frames[0]["name"] == "main" and snap.frames[0]["locals"][0]["name"] == "t"
    assert snap.frames[0]["loc"] == "main.c:1"
    assert [block["free"] for block in snap.heap_blocks] == [False, True]
    assert {seg["name"] for seg in snap.segments} == {"null", "data", "heap", "guard", "stack"}


def test_snapshot_preview_is_capped(mem: Memory) -> None:
    big = ArrayType(CHAR, 200)
    mem.alloc_global(big.size, 1, "big", big)
    assert len(mem.snapshot().objects[0]["preview"]) == 64


def test_bytes_and_init_mask_for_hex_grid(mem: Memory) -> None:
    addr = mem.alloc_global(4, 4, "n", INT)
    mem.store(addr, INT, 0x01020304)
    assert mem.bytes(addr, 4) == b"\x04\x03\x02\x01"
    assert mem.init_mask(addr, 4) == b"\x01" * 4
    assert len(mem.bytes(mem.size - 2, 10)) == 2


def test_object_at_and_reset(mem: Memory) -> None:
    addr = mem.alloc_global(4, 4, "n", INT)
    assert mem.object_at(addr + 2).name == "n"
    mem.reset()
    assert mem.object_at(addr) is None and mem.frames == [] and mem.leaks() == []
