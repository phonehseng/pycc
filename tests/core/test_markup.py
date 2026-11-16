"""Inline markup: single-star emphasis (help pages once printed the stars as text)."""
from __future__ import annotations

from pycc.core.markup import parse_spans


def test_single_star_emphasis_is_drawn_bold_not_printed_with_its_stars():
    spans = parse_spans("give the *no* answer its own braces")
    assert [(s.text, s.style) for s in spans] == [("give the ", "plain"), ("no", "bold"),
                                                  (" answer its own braces", "plain")]
    for text in ("5 * 3 * 2", "int *p = &x; and *q", "`a * b`"):
        assert all(s.style != "bold" for s in parse_spans(text)), text
