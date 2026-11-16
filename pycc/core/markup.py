"""A tiny markup language for the man pages and other help texts.

Supported:

    # Heading            (also ## and ###)
    - bullet item
    ```c                 fenced code block (optional language tag)
    code
    ```
    plain paragraphs with **bold**, *emphasis* (drawn bold), `inline code` and [[concept]] links

Why not Markdown: a small, predictable grammar that produces simple blocks is easier
to render in a terminal (or any other front end) and to test than a full Markdown
parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Span:
    text: str
    style: str = "plain"   # plain | bold | code | concept


@dataclass
class Block:
    kind: str                       # heading | paragraph | bullet | code
    spans: list[Span] = field(default_factory=list)
    level: int = 0                  # heading level
    code: str = ""                  # code block text
    language: str = ""              # code block language tag

    @property
    def text(self) -> str:
        return self.code if self.kind == "code" else "".join(s.text for s in self.spans)


# *word* (single stars) is emphasis. The text has no italic face, so it is drawn bold;
# before this the stars were printed as they are ("give the *no* answer"). The inside must
# start with a letter and end with a letter or stop, so "5 * 3 * 2" and "*p" stay as text.
_INLINE = re.compile(r"(\*\*(.+?)\*\*|`([^`]+)`|\[\[([a-z_]+)\]\]"
                     r"|(?<![\w*])\*([A-Za-z](?:[^*\n`]*?[A-Za-z.!?])?)\*(?![\w*]))")


def parse_spans(text: str) -> list[Span]:
    """Split one line into plain / bold / code / concept spans."""
    spans: list[Span] = []
    pos = 0
    for match in _INLINE.finditer(text):
        if match.start() > pos:
            spans.append(Span(text[pos:match.start()]))
        if match.group(2) is not None:
            spans.append(Span(match.group(2), "bold"))
        elif match.group(5) is not None:
            spans.append(Span(match.group(5), "bold"))
        elif match.group(3) is not None:
            spans.append(Span(match.group(3), "code"))
        else:
            spans.append(Span(match.group(4), "concept"))
        pos = match.end()
    if pos < len(text):
        spans.append(Span(text[pos:]))
    return spans


def parse(text: str) -> list[Block]:
    """Parse markup text into blocks. Blank lines separate paragraphs."""
    blocks: list[Block] = []
    paragraph: list[str] = []
    lines = text.splitlines()
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(Block("paragraph", parse_spans(" ".join(paragraph))))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append(Block("code", code="\n".join(code_lines), language=language))
            i += 1
            continue
        if not stripped:
            flush_paragraph()
        elif stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(Block("heading", parse_spans(stripped[level:].strip()), level=min(level, 3)))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            blocks.append(Block("bullet", parse_spans(stripped[2:])))
        else:
            paragraph.append(stripped)
        i += 1
    flush_paragraph()
    return blocks


def to_plain_text(text: str) -> str:
    """Strip markup for places that can only show plain text (logs, tests)."""
    out: list[str] = []
    for block in parse(text):
        if block.kind == "code":
            out.append(block.code)
        elif block.kind == "bullet":
            out.append("- " + block.text)
        else:
            out.append(block.text)
    return "\n".join(out)
