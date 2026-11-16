"""Seeded randomness for everything that must be reproducible.

The garbage that fills uninitialised memory and ``rand()``'s default seed come from
here, so a program behaves the same on every run. Seeds are 32-bit so they fit on
screen as eight hex digits that users can share. This module imports only the
standard library so the lowest layers can use it.
"""
from __future__ import annotations

import random
import zlib

SEED_MASK = 0xFFFFFFFF


class SeededRandom(random.Random):
    """A ``random.Random`` that remembers its seed and can derive child streams.

    Deriving named children (``rng.child("layout")``) keeps independent systems from
    disturbing each other's sequences: adding a particle effect that draws random
    numbers in one place must never change the numbers drawn somewhere else.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed_value = int(seed) & SEED_MASK
        super().__init__(self.seed_value)

    def child(self, name: str) -> "SeededRandom":
        """Return a new stream whose seed depends on this seed and ``name``."""
        return SeededRandom(mix_seed(self.seed_value, name))

    def __repr__(self) -> str:
        return f"SeededRandom({format_seed(self.seed_value)})"


def mix_seed(seed: int, salt: str) -> int:
    """Combine a seed with a text salt into a new 32-bit seed (deterministic)."""
    return zlib.crc32(salt.encode("utf-8"), int(seed) & SEED_MASK) & SEED_MASK


def seed_from_text(text: str) -> int:
    """Turn any text (a date, a user name, a shared code) into a 32-bit seed."""
    return zlib.crc32(text.encode("utf-8")) & SEED_MASK


def format_seed(seed: int) -> str:
    """Eight upper-case hex digits, the way seeds are shown to users."""
    return f"{int(seed) & SEED_MASK:08X}"


def parse_seed(text: str) -> int | None:
    """Parse a seed typed by the user.

    Seeds are shown as hex, so hex is assumed; a number longer than eight digits is
    read as decimal (and wrapped to 32 bits).
    """
    cleaned = text.strip().lower().replace("0x", "")
    if not cleaned:
        return None
    try:
        if all(c in "0123456789abcdef" for c in cleaned) and len(cleaned) <= 8:
            return int(cleaned, 16) & SEED_MASK
        return int(cleaned, 10) & SEED_MASK
    except ValueError:
        return None


def random_seed() -> int:
    """A fresh seed for a new run (the only non-deterministic function here)."""
    return random.SystemRandom().getrandbits(32)
