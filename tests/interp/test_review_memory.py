"""Review tests for the simulated memory: boundaries, neighbours, odd allocations,
realloc, scopes, partial initialisation, snapshot cost and determinism."""
from __future__ import annotations

import time

import pytest

from pycc.interp.c_types import CHAR, INT, LONG, ArrayType, Member, StructType
from pycc.interp.memory import NULL_GUARD, PREVIEW_BUDGET, Memory
from pycc.interp.traps import MemoryTrap

INT8 = ArrayType(INT, 8)
PAIR = StructType("Pair", [Member("x", INT), Member("y", INT)])


@pytest.fixture
def mem() -> Memory:
    """A default-sized memory with a fixed seed."""
    return Memory(seed=3)


def trap_of(func, *args) -> MemoryTrap:
    """Run ``func`` expecting a MemoryTrap and return it."""
    with pytest.raises(MemoryTrap) as info:
        func(*args)
    return info.value


# --- object boundaries ------------------------------------------------------------------

def test_access_ending_exactly_at_end_is_fine(mem: Memory) -> None:
    a = mem.alloc_global(INT8.size, 4, "a", INT8)
    mem.store(a + 28, INT, 7)
    assert mem.load(a + 28, INT) == 7


def test_one_past_the_end_traps(mem: Memory) -> None:
    a = mem.alloc_global(INT8.size, 4, "a", INT8)
    trap = trap_of(mem.load, a + 32, INT)
    assert trap.code == "out_of_bounds"
    assert trap.extra["index"] == 8 and trap.extra["object"] == "a"


def test_access_straddling_two_adjacent_globals_traps(mem: Memory) -> None:
    a = mem.alloc_global(4, 4, "a", INT)
    mem.alloc_global(4, 4, "b", INT)
    trap = trap_of(mem.load, a, LONG)          # 8 bytes from a 4-byte int
    assert trap.code == "out_of_bounds" and trap.extra["object"] == "a"


def test_heap_write_crossing_block_end_names_that_block(mem: Memory) -> None:
    p = mem.malloc(16)
    mem.malloc(16)
    trap = trap_of(mem.store, p + 12, LONG, 1)
    assert trap.code == "out_of_bounds" and trap.extra["object"] == "malloc(16)"
    assert trap.message.startswith("writing")  # not "reading" for a store


def test_heap_access_into_header_gap_is_overrun_of_previous(mem: Memory) -> None:
    p = mem.malloc(16)
    mem.malloc(16)
    trap = trap_of(mem.load, p + 16, INT)
    assert trap.code == "out_of_bounds" and trap.extra["object"] == "malloc(16)"


def test_origin_overrun_into_live_neighbour_is_blamed_on_base(mem: Memory) -> None:
    a = mem.alloc_global(INT8.size, 4, "a", INT8)
    b = mem.alloc_global(INT8.size, 4, "b", INT8)
    trap = trap_of(mem.load, b, INT, a)       # a[k] that lands inside b
    assert trap.extra["object"] == "a"


# --- zero-size and odd allocations ------------------------------------------------------

def test_malloc_zero_is_unique_and_non_null(mem: Memory) -> None:
    p, q = mem.malloc(0), mem.malloc(0)
    assert p and q and p != q
    assert trap_of(mem.load, p, CHAR).code == "out_of_bounds"
    mem.free(p)
    mem.free(q)
    assert mem.leaks() == []


def test_freed_malloc_zero_pointer_is_use_after_free(mem: Memory) -> None:
    p = mem.malloc(0)
    mem.free(p)
    assert trap_of(mem.load, p, CHAR).code == "use_after_free"
    assert trap_of(mem.free, p).code == "double_free"


def test_calloc_with_two_negative_arguments_fails(mem: Memory) -> None:
    assert mem.calloc(-2, -8) == 0
    assert mem.drain_warnings()[0].code == "heap_exhausted"


# --- realloc ----------------------------------------------------------------------------

def test_realloc_null_behaves_like_malloc(mem: Memory) -> None:
    p = mem.realloc(0, 24)
    assert p and mem.object_at(p).size == 24 and mem.object_at(p).alive


def test_realloc_to_zero_frees_and_returns_null(mem: Memory) -> None:
    p = mem.malloc(8)
    assert mem.realloc(p, 0) == 0
    assert mem.leaks() == []
    assert trap_of(mem.load, p, CHAR).code == "use_after_free"


def test_free_of_old_pointer_after_realloc_moved_is_double_free(mem: Memory) -> None:
    p = mem.malloc(8)
    q = mem.realloc(p, 64)
    assert q != p
    trap = trap_of(mem.free, p)
    assert trap.code == "double_free"
    assert "realloc" in trap.friendly and f"{q:#x}" in trap.friendly
    mem.free(q)


def test_load_through_old_pointer_after_realloc_mentions_realloc(mem: Memory) -> None:
    p = mem.malloc(8)
    mem.store(p, INT, 5)
    q = mem.realloc(p, 32)
    assert mem.load(q, INT) == 5
    trap = trap_of(mem.load, p, INT)
    assert trap.code == "use_after_free" and "realloc" in trap.friendly


def test_plain_free_has_no_realloc_hint(mem: Memory) -> None:
    p = mem.malloc(8)
    mem.free(p)
    assert "realloc moved" not in trap_of(mem.load, p, INT).friendly


# --- partial initialisation -------------------------------------------------------------

def test_partially_initialised_struct_warns_only_on_unset_member(mem: Memory) -> None:
    frame = mem.push_frame("main")
    s = mem.alloc_local(frame, PAIR.size, PAIR.align, "s", PAIR)
    mem.store(s, INT, 1)                        # s.x = 1
    assert mem.load(s, INT) == 1
    assert mem.drain_warnings() == []
    mem.load(s + 4, INT)                        # s.y never set
    warnings = mem.drain_warnings()
    assert [w.code for w in warnings] == ["uninitialized_read"]
    assert warnings[0].extra["object"] == "s"


def test_copy_of_partial_struct_keeps_garbage_flag(mem: Memory) -> None:
    frame = mem.push_frame("main")
    s = mem.alloc_local(frame, PAIR.size, PAIR.align, "s", PAIR)
    t = mem.alloc_local(frame, PAIR.size, PAIR.align, "t", PAIR)
    mem.store(s, INT, 1)
    mem.copy_bytes(t, s, PAIR.size)
    assert mem.init_mask(t, 8) == b"\x01" * 4 + b"\x00" * 4


# --- scopes -----------------------------------------------------------------------------

def test_scope_release_kills_only_objects_after_the_mark(mem: Memory) -> None:
    frame = mem.push_frame("main")
    outer = mem.alloc_local(frame, 4, 4, "outer", INT)
    mark = mem.scope_mark(frame)
    inner = mem.alloc_local(frame, 4, 4, "inner", INT)
    mem.scope_release(frame, mark)
    assert mem.object_at(outer).alive
    assert not mem.object_at(inner).alive
    assert [o.name for o in frame.locals] == ["outer"]
    assert mem.stack_pointer == mark
    trap = trap_of(mem.load, inner, INT)
    assert "out of scope" in trap.message


def test_nested_scopes_release_inner_first(mem: Memory) -> None:
    frame = mem.push_frame("main")
    outer_mark = mem.scope_mark(frame)
    a = mem.alloc_local(frame, 4, 4, "a", INT)
    inner_mark = mem.scope_mark(frame)
    b = mem.alloc_local(frame, 4, 4, "b", INT)
    mem.scope_release(frame, inner_mark)
    assert mem.object_at(a).alive and not mem.object_at(b).alive
    mem.scope_release(frame, outer_mark)
    assert not mem.object_at(a).alive and frame.locals == []


# --- snapshot cost and determinism -------------------------------------------------------

def test_snapshot_is_fast_with_big_objects() -> None:
    mem = Memory(seed=1)
    cube = ArrayType(ArrayType(ArrayType(CHAR, 64), 64), 15)
    mem.alloc_global(cube.size, 1, "cube", cube)
    for _ in range(300):
        mem.malloc(200)
    frame = mem.push_frame("main")
    big = ArrayType(INT, 40000)
    mem.alloc_local(frame, big.size, 4, "big", big)
    mem.snapshot()                              # warm up
    start = time.perf_counter()
    mem.snapshot()
    assert time.perf_counter() - start < 0.05


def test_snapshot_preview_has_a_total_budget(mem: Memory) -> None:
    cube = ArrayType(ArrayType(ArrayType(CHAR, 64), 64), 4)
    mem.alloc_global(cube.size, 1, "cube", cube)
    preview = mem.snapshot().objects[0]["preview"]
    flat = [v for plane in preview for row in plane for v in row]
    assert sum(v is not None for v in flat) == PREVIEW_BUDGET


def test_snapshot_initialized_flag_for_partial_object(mem: Memory) -> None:
    frame = mem.push_frame("main")
    s = mem.alloc_local(frame, PAIR.size, PAIR.align, "s", PAIR)
    mem.store(s, INT, 1)
    entry = next(o for o in mem.snapshot().objects if o["name"] == "s")
    assert entry["initialized"] is False
    mem.store(s + 4, INT, 2)
    entry = next(o for o in mem.snapshot().objects if o["name"] == "s")
    assert entry["initialized"] is True


def test_garbage_is_identical_across_runs_with_same_seed() -> None:
    def garbage(seed: int) -> tuple[bytes, bytes]:
        mem = Memory(seed=seed)
        p = mem.malloc(32)
        frame = mem.push_frame("main")
        x = mem.alloc_local(frame, 32, 8, "x", None)
        return mem.bytes(p, 32), mem.bytes(x, 32)

    assert garbage(11) == garbage(11)
    assert garbage(11) != garbage(12)


def test_reset_restores_the_same_garbage(mem: Memory) -> None:
    p = mem.malloc(16)
    before = mem.bytes(p, 16)
    mem.store(p, INT, 0)
    mem.reset()
    assert mem.malloc(16) == p and mem.bytes(p, 16) == before


def test_null_guard_and_data_segment_start_zeroed(mem: Memory) -> None:
    assert mem.bytes(0, NULL_GUARD) == bytes(NULL_GUARD)


def test_explicit_data_size_is_respected() -> None:
    mem = Memory(data_size=8192)
    assert mem.data_limit == NULL_GUARD + 8192
