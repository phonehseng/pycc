"""Every diagnostic code pycc can emit must carry a concept tag.

Tools built on pycc look a code up in ``core.diagnostics.CODE_CONCEPTS`` to tell what
a diagnostic is about. A code that is missing falls back to ``Concept.SYNTAX``, which
is silently wrong. This test scans the source tree for codes and fails on any new one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pycc.core.concepts import Concept
from pycc.core.diagnostics import CODE_CONCEPTS, TRAP_EXIT_CODES

ROOT = Path(__file__).resolve().parents[2] / "pycc"
SCANNED_PACKAGES = ("core", "interp", "shell", "vfs")

# ``code="x"`` keyword arguments anywhere, and the first positional string of the calls
# that build diagnostics, warnings and traps. The look-behind keeps ``perror("fopen")``
# (a C function in a docstring) from matching ``error(``.
KEYWORD_CODE = re.compile(r'''code\s*=\s*["']([a-z_][a-z0-9_]*)["']''')
POSITIONAL_CODE = re.compile(
    r'''(?<![A-Za-z0-9_])(?:DeviceWarning|DeviceTrap|Diagnostic|trap|warn|warning|error|fail|report)'''
    r'''\(\s*["']([a-z_][a-z0-9_]*)["']'''
)


def referenced_codes() -> dict[str, str]:
    """Return every code found in the scanned packages, mapped to one file that uses it."""
    found: dict[str, str] = {}
    for package in SCANNED_PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in (KEYWORD_CODE, POSITIONAL_CODE):
                for match in pattern.finditer(text):
                    found.setdefault(match.group(1), str(path.relative_to(ROOT)))
    return found


def test_every_referenced_code_has_a_concept() -> None:
    missing = {code: where for code, where in referenced_codes().items() if code not in CODE_CONCEPTS}
    assert not missing, f"codes without a concept tag (add them to core/diagnostics.py): {missing}"


def test_trap_tables_only_name_known_codes() -> None:
    for code in TRAP_EXIT_CODES:
        assert code in CODE_CONCEPTS, code


@pytest.mark.parametrize("code", ["out_of_bounds", "use_after_free", "loop_limit", "same_file"])
def test_a_sample_of_codes_is_present(code: str) -> None:
    assert code in CODE_CONCEPTS
