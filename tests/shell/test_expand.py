"""Tests for word expansion: quoting, variables, splitting, substitution, globbing."""
import pytest

from pycc.shell.expand import ExpansionError, SimpleContext, expand_single, expand_word
from pycc.shell.lexer import tokenize
from tests.shell.conftest import PROJECT
from pycc.vfs import VirtualFS


@pytest.fixture
def ctx(vfs: VirtualFS) -> SimpleContext:
    for name in ("main.c", "farm.c", "notes.txt", "my file.c"):
        vfs.write_file(f"{PROJECT}/{name}", "")
    return SimpleContext(vfs, PROJECT, {"X": "a  b", "EMPTY": "", "HOME": "/home/user"},
                         ["one", "two words"], runner=lambda src: f"<{src}>\n\n")


def expand(source: str, ctx: SimpleContext) -> list[str]:
    return expand_word(tokenize(source)[0].word, ctx)


@pytest.mark.parametrize("source, expected", [
    ("plain", ["plain"]),
    ("'single $X'", ["single $X"]),
    ('"double $X"', ["double a  b"]),
    ("$X", ["a", "b"]),
    ("pre${X}post", ["prea", "bpost"]),
    ('""', [""]),
    ("$EMPTY", []),
    ('"$EMPTY"', [""]),
    ("$UNSET", []),
    ("$#", ["2"]),
    ("$1", ["one"]),
    ('"$2"', ["two words"]),
    ("$2", ["two", "words"]),
    ('"$@"', ["one", "two words"]),
    ("$@", ["one", "two", "words"]),
    ('"$*"', ["one two words"]),
    ("~", ["/home/user"]),
    ("~/src", ["/home/user/src"]),
    ("'~'", ["~"]),
    ("a\\ b", ["a b"]),
    ("$(cmd)", ["<cmd>"]),
    ('"$(cmd) x"', ["<cmd> x"]),
])
def test_expansions(ctx, source, expected) -> None:
    assert expand(source, ctx) == expected


def test_glob_matches_sorted(ctx) -> None:
    assert expand("*.c", ctx) == ["farm.c", "main.c", "my file.c"]


def test_quoted_glob_is_literal(ctx) -> None:
    assert expand('"*.c"', ctx) == ["*.c"]


def test_glob_without_match_stays(ctx) -> None:
    assert expand("*.py", ctx) == ["*.py"]


def test_glob_from_variable(ctx) -> None:
    ctx.variables["PAT"] = "*.txt"
    assert expand("$PAT", ctx) == ["notes.txt"]


def test_partially_quoted_glob(ctx) -> None:
    assert expand('"my "*', ctx) == ["my file.c"]


def test_expand_single_does_not_split(ctx) -> None:
    assert expand_single(tokenize("$X*")[0].word, ctx) == "a  b*"


def test_bad_substitution(ctx) -> None:
    with pytest.raises(ExpansionError):
        expand("${X:-y}", ctx)
