"""The honest machine: a simulated address space with checked loads and stores.

Real C lets a program scribble anywhere; the bug shows up much later, somewhere else.
pycc keeps the *behaviour* of real memory (addresses are plain numbers, the
stack grows down, freed memory still holds old bytes, uninitialised variables hold
garbage) but checks every access against a map of the objects that exist, so a bad
access stops at the exact statement that made it, with a message that names the
variable.

Layout of the ``bytearray`` (addresses are offsets into it)::

    [0, 4096)                  null guard: any access is a NULL dereference
    [4096, data_limit)         data: globals, static locals, string literals (grows up)
    [heap_start, heap_end)     heap: first-fit free list, 16-byte aligned blocks
    [heap_end, stack_low)      guard between heap and stack
    [stack_low, size)          stack: grows down from the top

Every global, literal, local and heap block is a ``MemoryObject`` in an object map
sorted by start address. Objects are separated by small gaps (red zones, heap
headers) that belong to nobody, so stepping just past the end of an array lands in a
gap and is reported as ``out_of_bounds`` for that array instead of silently reading
its neighbour. Dead objects (freed blocks, locals of returned functions) stay in the
map until their memory is reused, which is how dangling pointers are caught.

A shadow map records which bytes have been written. Reading an unwritten byte
returns the real (seeded, deterministic) garbage and queues an
``uninitialized_read`` warning instead of stopping, so a beginner sees both the
garbage value and why it appeared.
"""
from __future__ import annotations

import builtins
import random
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field

from pycc.core.concepts import Concept
from pycc.core.diagnostics import SourceLocation, concept_for_code
from pycc.interp.c_types import (
    BOOL,
    CHAR,
    DOUBLE,
    ArrayType,
    CType,
    EnumType,
    FloatType,
    IntType,
    PointerType,
    StructType,
    convert_value,
    decode,
    encode,
    format_type,
)
from pycc.interp.traps import MemoryTrap, PendingWarning

NULL_GUARD = 4096          # addresses below this are "NULL plus a small offset"
SEGMENT_GUARD = 4096       # unused gap between the heap and the stack
HEAP_ALIGN = 16            # like glibc malloc on 64-bit Linux
HEAP_HEADER = 16           # bytes reserved before each block (never part of an object)
MIN_BLOCK = 16             # smallest user region a block can have
LOCAL_ALIGN = 8            # every local starts on an 8-byte boundary
RED_ZONE = 8               # minimum gap after each global and above each local
NEAR_MISS = 64             # a stray access this close to an object counts as its overrun
PREVIEW_LIMIT = 64         # array elements shown per level in the memory view
PREVIEW_BUDGET = 4096      # scalars decoded per object, so a big 3D array stays cheap
DEFAULT_MAX_FRAMES = 512   # call depth cap, matches the interpreter

OBJECT_KINDS = ("global", "literal", "stack", "heap")


@dataclass
class MemoryObject:
    """One thing that lives in memory: a global, a string literal, a local or a block."""

    start: int
    size: int
    kind: str                       # one of OBJECT_KINDS
    name: str
    ctype: CType | None
    frame_id: int | None = None     # owning frame for stack objects
    alive: bool = True
    id: int = 0

    @property
    def end(self) -> int:
        """One past the last byte of the object."""
        return self.start + self.size

    def contains(self, addr: int) -> bool:
        """True if ``addr`` is one of the object's bytes."""
        return self.start <= addr < self.end


@dataclass
class Frame:
    """A function's slice of the stack. ``mark`` is the stack pointer at entry."""

    id: int
    name: str
    loc: SourceLocation | None
    mark: int
    objects: list[MemoryObject] = field(default_factory=list)

    @property
    def locals(self) -> list[MemoryObject]:
        """The frame's variables that are still in scope."""
        return [obj for obj in self.objects if obj.alive]


@dataclass
class HeapBlock:
    """Bookkeeping for one allocated block. The header lives here, in Python, and the
    16 bytes at ``header`` in memory are just a gap, so overruns cannot corrupt it."""

    header: int
    user: int
    requested: int
    capacity: int

    @property
    def footprint(self) -> int:
        """Bytes of heap the block occupies, header included."""
        return HEAP_HEADER + self.capacity


@dataclass
class MemorySnapshot:
    """Plain-data picture of memory for the memory view (no live references)."""

    segments: list[dict]
    objects: list[dict]
    frames: list[dict]
    heap_blocks: list[dict]
    stack_pointer: int
    heap_used: int
    heap_capacity: int


def _round_up(value: int, align: int) -> int:
    """Smallest multiple of ``align`` that is >= ``value``."""
    return (value + align - 1) // align * align


def _round_down(value: int, align: int) -> int:
    """Largest multiple of ``align`` that is <= ``value``."""
    return value // align * align


def _literal_name(data: bytes) -> str:
    """Spell a string literal the way the user wrote it, shortened for messages."""
    text = data.decode("latin-1").encode("unicode_escape").decode("ascii").replace('"', '\\"')
    if len(text) > 24:
        text = text[:21] + "..."
    return f'"{text}"'


class Memory:
    """The simulated address space. See the module docstring for the layout."""

    def __init__(self, size: int = 1 << 20, stack_size: int = 256 << 10, seed: int = 0,
                 *, data_size: int | None = None, max_frames: int = DEFAULT_MAX_FRAMES) -> None:
        """Lay out the segments (see the module docstring) and fill memory."""
        self.size = size
        self.stack_size = stack_size
        self.seed = seed
        self.max_frames = max_frames
        self.data_start = NULL_GUARD
        if data_size is None:
            data_size = max(4096, size // 16)
        self.data_limit = _round_up(self.data_start + data_size, HEAP_ALIGN)
        self.heap_start = self.data_limit
        self.stack_low = size - stack_size
        self.heap_end = _round_down(self.stack_low - SEGMENT_GUARD, HEAP_ALIGN)
        if self.heap_end - self.heap_start < HEAP_HEADER + MIN_BLOCK:
            raise ValueError(f"memory of {size} bytes is too small for a {stack_size}-byte stack")
        self.reset()

    # --- lifecycle ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget everything and refill memory with this seed's garbage."""
        # Filling all of memory up front makes garbage honest: a reused stack slot or
        # heap block shows whatever was there before, exactly like real hardware.
        self._buf = bytearray(random.Random(self.seed).randbytes(self.size))
        self._buf[0:self.heap_start] = bytes(self.heap_start)   # null guard and data start zeroed
        self._init = bytearray(self.size)
        self._starts: list[int] = []
        self._objects: list[MemoryObject] = []
        self._next_object_id = 1
        self._data_ptr = self.data_start
        self._strings: dict[bytes, int] = {}
        self._free_ranges: list[list[int]] = [[self.heap_start, self.heap_end]]
        self._blocks: dict[int, HeapBlock] = {}
        self._frames: list[Frame] = []
        self._next_frame_id = 1
        self._sp = self.size
        self._warned_ids: set[int] = set()
        # Object id of a block realloc moved away from -> where it went, so a stale
        # pointer's trap can say "use the pointer realloc returned".
        self._moved_to: dict[int, int] = {}
        self.pending_warnings: list[PendingWarning] = []

    # --- object map -------------------------------------------------------------------

    def _new_object(self, start: int, size: int, kind: str, name: str, ctype: CType | None,
                    frame_id: int | None = None) -> MemoryObject:
        """Register a live object, evicting dead objects whose memory it reuses."""
        obj = MemoryObject(start, size, kind, name, ctype, frame_id, True, self._next_object_id)
        self._next_object_id += 1
        self._remove_dead_overlapping(start, max(start + size, start + 1))
        index = bisect_right(self._starts, start)
        self._starts.insert(index, start)
        self._objects.insert(index, obj)
        return obj

    def _remove_dead_overlapping(self, start: int, end: int) -> None:
        """Drop dead objects that overlap ``[start, end)``."""
        # Memory being reused: the dead objects that used to live here are gone for
        # good. Keeping the map overlap-free makes every lookup a single bisect.
        index = max(0, bisect_left(self._starts, start) - 1)
        while index < len(self._objects) and self._objects[index].start < end:
            obj = self._objects[index]
            overlaps = obj.start < end and max(obj.end, obj.start + 1) > start
            if overlaps and not obj.alive:
                del self._objects[index]
                del self._starts[index]
            else:
                index += 1

    def _find(self, addr: int) -> MemoryObject | None:
        """The object (alive or dead) whose bytes include ``addr``, if any."""
        index = bisect_right(self._starts, addr) - 1
        if index >= 0 and self._objects[index].contains(addr):
            return self._objects[index]
        return None

    def _object_starting_at(self, addr: int) -> MemoryObject | None:
        """The object whose first address is ``addr`` (zero-size ones too)."""
        index = bisect_left(self._starts, addr)
        if index < len(self._starts) and self._starts[index] == addr:
            return self._objects[index]
        return None

    def object_at(self, addr: int) -> MemoryObject | None:
        """The object (alive or dead) containing ``addr``, or starting there if empty."""
        return self._find(addr) or self._object_starting_at(addr)

    def objects(self) -> list[MemoryObject]:
        """All objects in the map, in address order."""
        return list(self._objects)

    # --- access checking --------------------------------------------------------------

    def check_access(self, addr: int, size: int, mode: str = "read",
                     origin: int | None = None) -> MemoryObject:
        """Return the live object holding ``[addr, addr+size)`` or raise ``MemoryTrap``.

        ``mode`` is ``"read"`` or ``"write"``. ``origin`` is optional provenance: the
        address the pointer was computed from (an array's base for ``a[i]``). When it
        is given the access must stay inside *that* object, which catches overruns
        that jump clean over the red zone into a neighbour.
        """
        size = max(size, 1)
        if origin is not None:
            base = self._find(origin)
            if base is not None:
                return self._check_within(base, addr, size, mode)
        if 0 <= addr < NULL_GUARD:
            raise self._null_trap(addr, mode)
        obj = self._find(addr) if 0 <= addr < self.size else None
        if obj is None:
            obj = self._dead_empty_object_at(addr)
        if obj is None:
            neighbour = self._near_miss(addr, size)
            if neighbour is not None:
                raise self._out_of_bounds_trap(neighbour, addr, size, mode)
            raise self._wild_trap(addr)
        return self._check_within(obj, addr, size, mode)

    def _dead_empty_object_at(self, addr: int) -> MemoryObject | None:
        """A freed zero-size block (``malloc(0)``) starting at ``addr``, if any.

        A zero-size object contains no address, so ``_find`` never returns it; without
        this a dangling ``malloc(0)`` pointer would be reported as a wild pointer
        instead of the use-after-free it really is.
        """
        obj = self._object_starting_at(addr)
        if obj is not None and obj.size == 0 and not obj.alive:
            return obj
        return None

    def _check_within(self, obj: MemoryObject, addr: int, size: int, mode: str) -> MemoryObject:
        """Trap unless ``obj`` is alive, holds the whole access and allows ``mode``."""
        if not obj.alive:
            raise self._dead_trap(obj, addr)
        if addr < obj.start or addr + size > obj.end:
            raise self._out_of_bounds_trap(obj, addr, size, mode)
        if mode == "write" and obj.kind == "literal":
            raise self._literal_trap(obj)
        return obj

    def _near_miss(self, addr: int, size: int) -> MemoryObject | None:
        """The live object an address just outside of probably belonged to, if any."""
        index = bisect_right(self._starts, addr) - 1
        candidates: list[tuple[int, int, MemoryObject]] = []
        if index >= 0:
            before = self._objects[index]
            if before.alive and 0 <= addr - before.end < NEAR_MISS:
                candidates.append((addr - before.end, 0, before))     # ran off the end
        if index + 1 < len(self._objects):
            after = self._objects[index + 1]
            gap = max(0, after.start - (addr + size))
            if after.alive and gap < NEAR_MISS:
                candidates.append((gap, 1, after))                     # ran off the front
        # Nearest wins; on a tie we blame running off the end, the far commoner bug.
        return min(candidates, key=lambda c: (c[0], c[1]))[2] if candidates else None

    # --- trap construction --------------------------------------------------------------

    def _null_trap(self, addr: int, mode: str) -> MemoryTrap:
        """Trap for touching the null guard page (address 0 and the few bytes after it).

        Address 0 is NULL itself. A small non-zero address is NULL plus an offset (a
        field reached through a NULL struct pointer) or a plain number stored in a
        pointer (``int *p = flow[2];``), so the text says both instead of "NULL".
        """
        verb = "write to" if mode == "write" else "read from"
        if addr == 0:
            return MemoryTrap(
                "null_dereference",
                f"segmentation fault: {verb} address {addr:#x} (NULL pointer dereference)",
                "The program followed a NULL pointer. NULL means 'points at nothing', so there "
                "is no value there. Check that the pointer was set (and that malloc did not "
                "return NULL) before using it.",
                extra={"address": addr})
        return MemoryTrap(
            "null_dereference",
            f"segmentation fault: {verb} address {addr:#x} (an address next to NULL)",
            f"The program followed a pointer to address {addr}, which is not where any variable "
            "lives. Either the pointer was NULL and a field was read through it, or a plain "
            "number was stored in the pointer (did you forget '&'?).",
            extra={"address": addr})

    def _wild_trap(self, addr: int) -> MemoryTrap:
        """Trap for an address that belongs to no object at all."""
        return MemoryTrap(
            "wild_pointer",
            f"segmentation fault: address {addr:#x} is not inside any variable or allocation",
            "This pointer does not point at any variable or allocation; it may be garbage "
            "or past the end of an array.",
            extra={"address": addr})

    def _dead_trap(self, obj: MemoryObject, addr: int) -> MemoryTrap:
        """Trap for touching a freed block or an out-of-scope local."""
        if obj.kind == "heap":
            return MemoryTrap(
                "use_after_free",
                f"heap-use-after-free: address {addr:#x} is inside the block from "
                f"{obj.name}, which was already freed",
                "This memory was already given back with free(). After free the block "
                "belongs to the heap again, so the pointer must not be used. Set it to "
                "NULL after freeing so the mistake is easy to spot." + self._realloc_hint(obj),
                extra={"address": addr, "object": obj.name})
        if self._frame_by_id(obj.frame_id) is not None:
            friendly = (f"This pointer points at '{obj.name}', a variable whose block {{ }} "
                        "has already ended. A variable only exists inside the braces where "
                        "it was declared.")
            message = f"stack-use-after-scope: '{obj.name}' at {addr:#x} has gone out of scope"
        else:
            friendly = (f"This pointer points at a local variable of a function that has "
                        f"already returned ('{obj.name}'). Locals disappear when their "
                        "function returns; use malloc or a caller's variable instead.")
            message = f"stack-use-after-return: '{obj.name}' at {addr:#x} no longer exists"
        return MemoryTrap("wild_pointer", message, friendly,
                          extra={"address": addr, "object": obj.name})

    def _realloc_hint(self, obj: MemoryObject) -> str:
        """Extra advice when the block died because realloc moved it, else ''."""
        new_addr = self._moved_to.get(obj.id)
        if new_addr is None:
            return ""
        return (f" realloc moved this block to {new_addr:#x} and freed the old one, so "
                "only the pointer realloc returned is valid now: write p = realloc(p, n).")

    def _literal_trap(self, obj: MemoryObject) -> MemoryTrap:
        """Trap for writing into a read-only string literal."""
        return MemoryTrap(
            "write_to_literal",
            f"segmentation fault: write to read-only string literal {obj.name}",
            "String literals like \"hello\" live in read-only memory. To change the text, "
            "copy it into an array first: char name[] = \"hello\";",
            extra={"object": obj.name})

    def _out_of_bounds_trap(self, obj: MemoryObject, addr: int, size: int, mode: str) -> MemoryTrap:
        """Trap for an access that leaves ``obj``, phrased as an index if possible."""
        element = self._element_size(obj, size)
        offset = addr - obj.start
        extra = {"object": obj.name, "kind": obj.kind, "offset": offset, "size": size}
        if element is None:
            verb = "write" if mode == "write" else "read"
            message = (f"{verb} of {size} bytes at offset {offset} is outside '{obj.name}' "
                       f"({obj.size} bytes)")
            friendly = (f"'{obj.name}' is only {obj.size} bytes long, but the program reached "
                        f"{size} bytes at offset {offset}. That memory belongs to something else.")
            return MemoryTrap("out_of_bounds", message, friendly, extra=extra)
        index, length = offset // element, obj.size // element
        extra.update(index=index, length=length)
        message = self._index_message(obj.name, index, length, size, mode)
        friendly = self._index_friendly(obj.name, index, length, size // element)
        return MemoryTrap("out_of_bounds", message, friendly, extra=extra)

    @staticmethod
    def _element_size(obj: MemoryObject, access_size: int) -> int | None:
        """Bytes per element when the object is array-like, else ``None``."""
        if isinstance(obj.ctype, ArrayType) and obj.ctype.element.size > 0:
            return obj.ctype.element.size
        if obj.kind == "heap":
            # A malloc'd block has no declared type: count in units of the access.
            return access_size
        return None

    @staticmethod
    def _index_message(name: str, index: int, length: int, size: int, mode: str) -> str:
        """gcc/ASan-style one-liner for an array-like overrun."""
        if index < 0:
            return f"index {index} is before the start of '{name}' ({length} elements)"
        if index < length:
            verb = "writing" if mode == "write" else "reading"
            return (f"{verb} {size} bytes at index {index} runs past the end of '{name}' "
                    f"({length} elements)")
        return f"index {index} is past the end of '{name}' ({length} elements)"

    @staticmethod
    def _index_friendly(name: str, index: int, length: int, count: int = 1) -> str:
        """Plain-English explanation of an array-like overrun, with the valid range.

        ``count`` is how many elements the access covers. ``strcpy(name, long_text)``
        starts at index 0, which is valid, so the text must say where it ends instead
        of blaming index 0.
        """
        valid = f"the valid indexes are 0 to {length - 1}" if length else "it has no valid indexes"
        if 0 <= index < length:
            last = index + count - 1
            return (f"'{name}' has {length} elements, so {valid}. This access starts at "
                    f"index {index} but covers {count} elements, up to index {last}, in "
                    "memory that belongs to something else. Make the array bigger, or copy "
                    "less (strncpy and snprintf take the size of the destination).")
        return (f"'{name}' has {length} elements, so {valid}. Index {index} is outside it, in "
                "memory that belongs to something else. Check the "
                "loop condition: < length, not <= length.")

    # --- data segment ---------------------------------------------------------------------

    def _alloc_data(self, size: int, align: int, name: str, ctype: CType | None, kind: str) -> int:
        """Bump-allocate a zeroed object in the data segment."""
        start = _round_up(self._data_ptr, max(align, 1))
        if start + size > self.data_limit:
            raise MemoryTrap(
                "data_segment_full",
                f"cannot place '{name}': the data segment ({self.data_limit - self.data_start} "
                "bytes) is full",
                "There are too many (or too large) global variables and string literals. "
                "Use smaller arrays or malloc for big tables.",
                concept=Concept.VARIABLES, extra={"object": name, "size": size})
        self._data_ptr = start + size + RED_ZONE
        self._buf[start:start + size] = bytes(size)
        self._init[start:start + size] = b"\x01" * size
        self._new_object(start, size, kind, name, ctype)
        return start

    def alloc_global(self, size: int, align: int = 8, name: str = "", ctype: CType | None = None) -> int:
        """Reserve a zeroed, initialised global (or static local) and return its address."""
        return self._alloc_data(size, align, name, ctype, "global")

    def intern_string(self, data: bytes) -> int:
        """Place a string literal (``data`` without its NUL) in read-only data, once.

        Identical literals share one address, as gcc does, which is also why writing
        to one would be a disaster and is trapped.
        """
        data = bytes(data)
        if data in self._strings:
            return self._strings[data]
        addr = self._alloc_data(len(data) + 1, 1, _literal_name(data),
                                ArrayType(CHAR, len(data) + 1), "literal")
        self._buf[addr:addr + len(data)] = data
        self._strings[data] = addr
        return addr

    # --- stack ------------------------------------------------------------------------------

    @property
    def frames(self) -> list[Frame]:
        """Active frames, outermost first."""
        return list(self._frames)

    @property
    def stack_pointer(self) -> int:
        """Lowest address currently in use by the stack."""
        return self._sp

    def _frame_by_id(self, frame_id: int | None) -> Frame | None:
        """The active frame with this id, or ``None`` if it has returned."""
        return next((f for f in self._frames if f.id == frame_id), None)

    def _require_top(self, frame: Frame) -> None:
        """Raise ``ValueError`` unless ``frame`` is the innermost frame."""
        # Only the innermost function can grow the stack; anything else is a bug in
        # the interpreter, not in the student's program, so it is not a trap.
        if not self._frames or self._frames[-1] is not frame:
            raise ValueError(f"frame '{frame.name}' is not the innermost frame")

    def push_frame(self, name: str, loc: SourceLocation | None = None) -> Frame:
        """Enter a function: start a new frame at the current stack pointer."""
        if len(self._frames) >= self.max_frames:
            raise self._stack_overflow_trap(name, f"more than {self.max_frames} nested calls")
        frame = Frame(self._next_frame_id, name, loc, self._sp)
        self._next_frame_id += 1
        self._frames.append(frame)
        return frame

    def pop_frame(self) -> Frame:
        """Leave the innermost function: its locals die and its stack space is freed."""
        if not self._frames:
            raise IndexError("pop_frame() with no active frame")
        frame = self._frames.pop()
        for obj in frame.objects:
            obj.alive = False
        self._sp = frame.mark
        return frame

    def alloc_local(self, frame: Frame, size: int, align: int = 8, name: str = "",
                    ctype: CType | None = None) -> int:
        """Reserve an uninitialised local in ``frame`` (the innermost) and return its address."""
        self._require_top(frame)
        start = _round_down(self._sp - RED_ZONE - size, max(align, LOCAL_ALIGN))
        if start < self.stack_low:
            raise self._stack_overflow_trap(
                frame.name, f"'{name}' needs {size} bytes but the {self.stack_size}-byte stack is full")
        # The old bytes stay: that is the garbage an uninitialised local shows.
        self._init[start:start + size] = bytes(size)
        obj = self._new_object(start, size, "stack", name, ctype, frame.id)
        frame.objects.append(obj)
        self._sp = start
        return start

    def scope_mark(self, frame: Frame) -> int:
        """Remember the stack position at the start of a block ``{``."""
        self._require_top(frame)
        return self._sp

    def scope_release(self, frame: Frame, mark: int) -> None:
        """Leave a block ``}``: kill the locals declared since ``mark``."""
        self._require_top(frame)
        for obj in frame.objects:
            if obj.start < mark:
                obj.alive = False
        frame.objects = [obj for obj in frame.objects if obj.alive]
        self._sp = mark

    def _stack_overflow_trap(self, function: str, detail: str) -> MemoryTrap:
        """Trap for running out of stack (too deep or too big)."""
        return MemoryTrap(
            "stack_overflow",
            f"stack overflow in '{function}': {detail}",
            "Each function call takes a slice of the stack for its local variables. Too many "
            "calls at once (often recursion without a base case) or a huge local array used "
            "it all up.",
            extra={"function": function, "depth": len(self._frames)})

    # --- heap -------------------------------------------------------------------------------

    @property
    def heap_capacity(self) -> int:
        """Total bytes of the heap segment."""
        return self.heap_end - self.heap_start

    @property
    def heap_used(self) -> int:
        """Bytes of heap occupied by live blocks, headers and rounding included."""
        return sum(block.footprint for block in self._blocks.values())

    def heap_blocks(self) -> list[HeapBlock]:
        """Live blocks in address order."""
        return sorted(self._blocks.values(), key=lambda block: block.user)

    def _take_range(self, total: int) -> tuple[int, int] | None:
        """First fit: carve ``total`` bytes from the lowest free range big enough."""
        for index, (start, end) in enumerate(self._free_ranges):
            if end - start < total:
                continue
            if end - start - total < HEAP_HEADER + MIN_BLOCK:
                # The leftover could not hold even the smallest block: hand it over too.
                del self._free_ranges[index]
                return start, end - start
            self._free_ranges[index][0] = start + total
            return start, total
        return None

    def _release_range(self, start: int, end: int) -> None:
        """Return a range to the free list, merging with free neighbours."""
        insort(self._free_ranges, [start, end])
        merged: list[list[int]] = []
        for rng in self._free_ranges:
            if merged and merged[-1][1] == rng[0]:
                merged[-1][1] = rng[1]
            else:
                merged.append(rng)
        self._free_ranges = merged

    def _allocate(self, size: int, name: str) -> int:
        """Shared body of malloc/calloc/realloc: carve a block or return 0."""
        capacity = max(MIN_BLOCK, _round_up(max(size, 0), HEAP_ALIGN))
        taken = self._take_range(HEAP_HEADER + capacity) if size >= 0 else None
        if taken is None:
            self._warn_exhausted(size, name)
            return 0
        header, total = taken
        block = HeapBlock(header, header + HEAP_HEADER, size, total - HEAP_HEADER)
        self._blocks[block.user] = block
        self._init[block.user:block.user + block.capacity] = bytes(block.capacity)
        self._new_object(block.user, size, "heap", name, None)
        return block.user

    def malloc(self, size: int) -> int:
        """Allocate ``size`` uninitialised bytes. Returns 0 (NULL) when the heap is full.

        ``malloc(0)`` returns a unique non-NULL pointer to an empty block (glibc does
        the same): it must still be freed, and any read or write through it traps.
        """
        return self._allocate(size, f"malloc({size})")

    def calloc(self, count: int, size: int) -> int:
        """Allocate ``count * size`` zeroed bytes. Returns 0 when the heap is full."""
        # Two negative arguments would multiply to a plausible positive size; in C both
        # are size_t, so a negative one can only mean a huge request that must fail.
        total = count * size if count >= 0 and size >= 0 else -1
        addr = self._allocate(total, f"calloc({count}, {size})")
        if addr:
            self._buf[addr:addr + total] = bytes(total)
            self._init[addr:addr + total] = b"\x01" * total
        return addr

    def realloc(self, addr: int, size: int) -> int:
        """Resize a block. It always moves (so stale pointers to the old block trap);
        the bytes and their initialised state are copied. On failure returns 0 and
        the old block is untouched, as C requires.

        ``realloc(NULL, n)`` is ``malloc(n)``; ``realloc(p, 0)`` frees ``p`` and
        returns NULL (glibc's choice; C23 leaves it undefined).
        """
        if addr == 0:
            return self._allocate(size, f"realloc(NULL, {size})")
        old = self._blocks.get(addr)
        if old is None:
            raise self._bad_free_trap(addr, "realloc")
        if size == 0:
            self.free(addr)
            return 0
        new = self._allocate(size, f"realloc({size})")
        if new == 0:
            return 0
        self._copy_raw(new, addr, min(old.requested, size))
        old_obj = self._object_starting_at(addr)
        if old_obj is not None:
            self._moved_to[old_obj.id] = new
        self.free(addr)
        return new

    def free(self, addr: int) -> None:
        """Give a block back. ``free(NULL)`` does nothing; anything else odd traps."""
        if addr == 0:
            return
        block = self._blocks.pop(addr, None)
        if block is None:
            raise self._bad_free_trap(addr, "free")
        obj = self._object_starting_at(addr)
        if obj is not None:
            obj.alive = False
        self._release_range(block.header, block.header + block.footprint)

    def _bad_free_trap(self, addr: int, function: str) -> MemoryTrap:
        """Trap for free/realloc of something that is not a live block."""
        start_obj = self._object_starting_at(addr)
        if start_obj is not None and start_obj.kind == "heap" and not start_obj.alive:
            if function == "free":
                return MemoryTrap(
                    "double_free",
                    f"free(): double free detected: the block from {start_obj.name} at "
                    f"{addr:#x} was already freed",
                    "This block was already freed. Each malloc needs exactly one free; set "
                    "the pointer to NULL after freeing so a second free does nothing."
                    + self._realloc_hint(start_obj),
                    extra={"address": addr, "object": start_obj.name})
            return MemoryTrap(
                "use_after_free",
                f"{function}(): the block from {start_obj.name} at {addr:#x} was already freed",
                "This block was already given back with free(), so it cannot be resized. "
                "Keep using the pointer that realloc returned, not the old one.",
                extra={"address": addr, "object": start_obj.name})
        return MemoryTrap("invalid_free", f"{function}(): invalid pointer {addr:#x}",
                          self._invalid_free_friendly(addr, function),
                          extra={"address": addr})

    def _invalid_free_friendly(self, addr: int, function: str) -> str:
        """Explain what the bad pointer handed to free/realloc points at."""
        obj = self.object_at(addr)
        rule = (f"{function}() only accepts the exact address that malloc, calloc or realloc "
                "returned (or NULL).")
        if obj is None:
            return f"{rule} This pointer does not point at any allocation."
        if obj.kind == "heap":
            return f"{rule} This pointer is {addr - obj.start} bytes into the block from {obj.name}."
        return f"{rule} '{obj.name}' is a {obj.kind} variable, not a malloc'd block."

    def leaks(self) -> list[MemoryObject]:
        """Heap blocks that are still allocated (at exit these are memory leaks)."""
        return [obj for obj in self._objects if obj.kind == "heap" and obj.alive]

    def _warn_exhausted(self, size: int, name: str) -> None:
        """Queue a ``heap_exhausted`` warning for a failed allocation."""
        largest = max((end - start for start, end in self._free_ranges), default=0)
        self.pending_warnings.append(PendingWarning(
            "heap_exhausted",
            f"{name}: out of memory (largest free gap is {largest} bytes of "
            f"{self.heap_capacity})",
            "The heap is full, so this call returned NULL. Always check the result of "
            "malloc before using it, and free blocks you no longer need.",
            concept_for_code("heap_exhausted"),
            {"requested": size, "largest_free": largest}))

    # --- loads and stores -------------------------------------------------------------------

    @staticmethod
    def _require_scalar(ctype: CType) -> None:
        """Raise ``TypeError`` for aggregate types (they move as bytes)."""
        if not isinstance(ctype, (IntType, EnumType, FloatType, PointerType)):
            raise TypeError(f"load/store needs a scalar type, not {format_type(ctype)}; "
                            "use read_bytes/write_bytes for aggregates")

    @staticmethod
    def _encode(ctype: CType, value: int | float) -> bytes:
        """Bytes for ``value`` stored as ``ctype`` (wrapping like the hardware)."""
        if ctype == BOOL:
            return b"\x01" if value else b"\x00"
        if isinstance(ctype, FloatType):
            # Converting through convert_value turns out-of-range doubles into inf
            # instead of crashing the float packer.
            return encode(convert_value(float(value), DOUBLE, ctype), ctype)
        return encode(int(value), ctype)

    def load(self, addr: int, ctype: CType, origin: int | None = None) -> int | float:
        """Read a scalar. Uninitialised bytes give real garbage plus a pending warning."""
        self._require_scalar(ctype)
        obj = self.check_access(addr, ctype.size, "read", origin)
        value = decode(builtins.bytes(self._buf[addr:addr + ctype.size]), ctype)
        if self._init.find(0, addr, addr + ctype.size) != -1:
            self._warn_uninitialized(obj, value)
        return value

    def store(self, addr: int, ctype: CType, value: int | float, origin: int | None = None) -> None:
        """Write a scalar, wrapped to the type's width (300 in an unsigned char is 44)."""
        self._require_scalar(ctype)
        self.check_access(addr, ctype.size, "write", origin)
        self._write_raw(addr, self._encode(ctype, value))

    def read_bytes(self, addr: int, n: int, origin: int | None = None) -> builtins.bytes:
        """Checked raw read (no uninitialised warning: copying garbage is legal)."""
        if n <= 0:
            return b""
        self.check_access(addr, n, "read", origin)
        return builtins.bytes(self._buf[addr:addr + n])

    def write_bytes(self, addr: int, data: builtins.bytes, origin: int | None = None) -> None:
        """Checked raw write; the bytes become initialised."""
        if not data:
            return
        self.check_access(addr, len(data), "write", origin)
        self._write_raw(addr, data)

    def copy_bytes(self, dst: int, src: int, n: int) -> None:
        """Checked ``memmove``: copies bytes *and* their initialised state, so copying
        a half-filled struct does not launder its garbage into "initialised" data."""
        if n <= 0:
            return
        self.check_access(src, n, "read")
        self.check_access(dst, n, "write")
        self._copy_raw(dst, src, n)

    def read_cstring(self, addr: int, limit: int = 4096) -> builtins.bytes:
        """Read a NUL-terminated string (without the NUL), never past its object."""
        obj = self.check_access(addr, 1, "read")
        stop = min(obj.end, addr + limit)
        length = self._buf.find(0, addr, stop) - addr
        if length < 0:
            if stop == obj.end:
                raise self._unterminated_trap(obj, addr)
            length = stop - addr            # hit the caller's limit: truncate quietly
        scanned_end = min(addr + length + 1, obj.end)
        if self._init.find(0, addr, scanned_end) != -1:
            self._warn_uninitialized(obj, None)
        return builtins.bytes(self._buf[addr:addr + length])

    def write_cstring(self, addr: int, data: builtins.bytes, origin: int | None = None) -> None:
        """Write ``data`` plus its NUL terminator (an overflow traps, like strcpy should)."""
        self.write_bytes(addr, builtins.bytes(data) + b"\0", origin)

    def _unterminated_trap(self, obj: MemoryObject, addr: int) -> MemoryTrap:
        """Trap for a string with no '\\0' inside its object."""
        return MemoryTrap(
            "string_not_terminated",
            f"string at {addr:#x} runs off the end of '{obj.name}' ({obj.size} bytes) "
            "without a terminating '\\0'",
            f"A C string must end with a '\\0' byte. The text in '{obj.name}' has none, so "
            "anything reading it would keep going into other memory. Leave room for the "
            "'\\0' (an array one longer than the text).",
            extra={"object": obj.name, "size": obj.size})

    def _write_raw(self, addr: int, data: builtins.bytes) -> None:
        """Unchecked write that marks the bytes initialised."""
        self._buf[addr:addr + len(data)] = data
        self._init[addr:addr + len(data)] = b"\x01" * len(data)

    def _copy_raw(self, dst: int, src: int, n: int) -> None:
        """Unchecked memmove of bytes and their initialised state."""
        self._buf[dst:dst + n] = self._buf[src:src + n]
        self._init[dst:dst + n] = self._init[src:src + n]

    def _warn_uninitialized(self, obj: MemoryObject, value: int | float | None) -> None:
        """Queue an ``uninitialized_read`` warning (once per object)."""
        # One warning per object: a loop reading the same garbage a thousand times
        # should show one warning, not bury the user in them.
        if obj.id in self._warned_ids:
            return
        self._warned_ids.add(obj.id)
        shown = "" if value is None else f" (garbage value {value})"
        if obj.kind == "heap":
            friendly = (f"The memory from {obj.name} was read before anything was stored in it, "
                        f"so it holds leftover bytes{shown}. malloc does not clear memory; use "
                        "calloc or set the values first.")
        else:
            friendly = (f"'{obj.name}' was read before it was given a value, so it holds "
                        f"whatever was left in memory{shown}. Give it a value first, for "
                        "example when you declare it.")
        self.pending_warnings.append(PendingWarning(
            "uninitialized_read",
            f"read of uninitialised value in '{obj.name}'{shown}",
            friendly, concept_for_code("uninitialized_read"),
            {"object": obj.name, "value": value}))

    def drain_warnings(self) -> list[PendingWarning]:
        """Return and clear the pending warnings (the interpreter calls this each step)."""
        warnings, self.pending_warnings = self.pending_warnings, []
        return warnings

    # --- inspection -------------------------------------------------------------------------

    def bytes(self, start: int, n: int) -> builtins.bytes:
        """Raw bytes for the hex grid, clipped to memory, unchecked and without warnings."""
        start = max(0, start)
        return builtins.bytes(self._buf[start:min(self.size, start + max(n, 0))])

    def init_mask(self, start: int, n: int) -> builtins.bytes:
        """One byte per address: 1 if initialised, 0 if garbage (greyed in the hex grid)."""
        start = max(0, start)
        return builtins.bytes(self._init[start:min(self.size, start + max(n, 0))])

    def snapshot(self) -> MemorySnapshot:
        """Everything the memory view draws, as plain dicts and lists."""
        return MemorySnapshot(
            segments=self._segment_dicts(),
            objects=[self._object_dict(obj) for obj in self._objects],
            frames=[self._frame_dict(frame) for frame in self._frames],
            heap_blocks=self._heap_block_dicts(),
            stack_pointer=self._sp,
            heap_used=self.heap_used,
            heap_capacity=self.heap_capacity,
        )

    def _segment_dicts(self) -> list[dict]:
        """The five segments as plain dicts for the memory view."""
        return [
            {"name": "null", "start": 0, "end": NULL_GUARD},
            {"name": "data", "start": self.data_start, "end": self.data_limit, "used": self._data_ptr},
            {"name": "heap", "start": self.heap_start, "end": self.heap_end},
            {"name": "guard", "start": self.heap_end, "end": self.stack_low},
            {"name": "stack", "start": self.stack_low, "end": self.size, "used": self._sp},
        ]

    def _object_dict(self, obj: MemoryObject) -> dict:
        """One object as a plain dict for the memory view."""
        return {
            "id": obj.id, "start": obj.start, "size": obj.size, "kind": obj.kind,
            "name": obj.name, "type": format_type(obj.ctype) if obj.ctype else None,
            "alive": obj.alive, "frame": obj.frame_id,
            # find() scans in place; slicing would copy a big block on every refresh.
            "initialized": self._init.find(0, obj.start, obj.end) == -1,
            "preview": self._object_preview(obj) if obj.alive else None,
        }

    def _frame_dict(self, frame: Frame) -> dict:
        """One frame and its in-scope locals as a plain dict."""
        return {
            "id": frame.id, "name": frame.name,
            "loc": str(frame.loc) if frame.loc else None,
            "locals": [{"name": obj.name, "addr": obj.start, "size": obj.size,
                        "type": format_type(obj.ctype) if obj.ctype else None}
                       for obj in frame.locals],
        }

    def _heap_block_dicts(self) -> list[dict]:
        """Used blocks and free holes in address order."""
        used = [{"start": b.user, "header": b.header, "size": b.requested,
                 "capacity": b.capacity, "free": False} for b in self.heap_blocks()]
        holes = [{"start": start, "header": None, "size": end - start,
                  "capacity": end - start, "free": True} for start, end in self._free_ranges]
        return sorted(used + holes, key=lambda block: block["start"])

    def _object_preview(self, obj: MemoryObject) -> object:
        """Decoded value of a live object for the memory view, if typed."""
        ctype = obj.ctype
        if isinstance(ctype, ArrayType) and ctype.length is None and ctype.element.size:
            ctype = ArrayType(ctype.element, obj.size // ctype.element.size)
        if ctype is None:
            return None
        # One shared countdown per object: PREVIEW_LIMIT caps each level, but nested
        # arrays multiply (64 x 64 x 64), so the total needs its own ceiling.
        budget = [PREVIEW_BUDGET]
        return self._preview(ctype, obj.start, budget)

    def _preview(self, ctype: CType, addr: int, budget: list[int]) -> object:
        """Decode a value for display: numbers, lists for arrays, dicts for structs.

        ``budget`` is a one-item list (a mutable counter) of scalars still allowed;
        once it runs out, remaining scalars show as ``None`` ("not decoded").
        """
        if isinstance(ctype, (IntType, EnumType, FloatType, PointerType)):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            return decode(builtins.bytes(self._buf[addr:addr + ctype.size]), ctype)
        if isinstance(ctype, ArrayType):
            count = min(ctype.length or 0, PREVIEW_LIMIT)
            step = ctype.element.size
            return [self._preview(ctype.element, addr + i * step, budget) for i in range(count)]
        if isinstance(ctype, StructType) and ctype.members is not None:
            return {m.name: self._preview(m.ctype, addr + m.offset, budget)
                    for m in ctype.members}
        return None
