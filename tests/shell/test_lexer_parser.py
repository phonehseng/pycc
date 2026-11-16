"""Tests for the shell lexer and parser."""
import pytest

from pycc.shell.lexer import CommandSub, LexError, Lit, Param, tokenize
from pycc.shell.parser import (ForCommand, IfCommand, ParseError, SimpleCommand, Subshell,
                          WhileCommand, needs_more_input, parse)


def kinds(source: str) -> list[tuple[str, str]]:
    return [(t.kind, t.text) for t in tokenize(source)]


def test_words_and_operators() -> None:
    assert kinds("ls -l | wc -l && echo ok") == [
        ("word", "ls"), ("word", "-l"), ("op", "|"), ("word", "wc"), ("word", "-l"),
        ("op", "&&"), ("word", "echo"), ("word", "ok"), ("eof", "")]


@pytest.mark.parametrize("source, op", [
    ("a > f", ">"), ("a >> f", ">>"), ("a < f", "<"), ("a 2> f", "2>"),
    ("a 2>&1", "2>&1"), ("a &> f", "&>"), ("a || b", "||"), ("a ; b", ";"), ("a >&2", ">&2"),
])
def test_operators(source, op) -> None:
    assert ("op", op) in kinds(source)


def test_redirect_glued_to_word() -> None:
    assert kinds("echo hi>out.txt")[:4] == [("word", "echo"), ("word", "hi"), ("op", ">"),
                                            ("word", "out.txt")]


def test_two_inside_word_is_not_redirect() -> None:
    assert ("word", "file2") in kinds("cat file2")


def test_single_quotes_are_literal() -> None:
    word = tokenize("'$HOME *'")[0].word
    assert word.parts == [Lit("$HOME *", quoted=True)]


def test_double_quotes_keep_variables() -> None:
    word = tokenize('"a $X b"')[0].word
    assert word.parts == [Lit("a ", True), Param("X", True), Lit(" b", True)]


def test_backslash_escapes() -> None:
    word = tokenize(r"a\ b\$c")[0].word
    assert "".join(p.text for p in word.parts) == "a b$c"
    assert word.parts[0] == Lit("a", False)


def test_backslash_in_double_quotes() -> None:
    word = tokenize(r'"say \"hi\" \n"')[0].word
    assert word.parts[0].text == 'say "hi" \\n'


def test_dollar_forms() -> None:
    parts = tokenize("$? ${NAME} $1 $@ $# $(echo hi) `pwd` $")
    found = [t.word.parts[0] for t in parts if t.kind == "word"]
    assert found == [Param("?"), Param("NAME"), Param("1"), Param("@"), Param("#"),
                     CommandSub("echo hi"), CommandSub("pwd"), Lit("$")]


def test_nested_command_substitution() -> None:
    word = tokenize('$(echo $(echo ")"))')[0].word
    assert word.parts == [CommandSub('echo $(echo ")")')]


def test_comments_and_continuation() -> None:
    assert kinds("echo a # ignored\necho \\\nb") == [
        ("word", "echo"), ("word", "a"), ("newline", "\n"), ("word", "echo"), ("word", "b"),
        ("eof", "")]


def test_hash_inside_word_is_not_comment() -> None:
    assert ("word", "a#b") in kinds("echo a#b")


def test_unterminated_quote() -> None:
    with pytest.raises(LexError):
        tokenize("echo 'oops")


# --- parser ----------------------------------------------------------------------------

def test_simple_command_with_assignment_and_redirect() -> None:
    command = parse("X=1 echo hi > out.txt").items[0].first.commands[0]
    assert isinstance(command, SimpleCommand)
    assert command.assignments[0][0] == "X"
    assert [w.raw for w in command.words] == ["echo", "hi"]
    assert command.redirects[0].op == ">" and command.redirects[0].target.raw == "out.txt"


def test_and_or_and_pipeline() -> None:
    item = parse("a | b && c || d").items[0]
    assert len(item.first.commands) == 2
    assert [op for op, _ in item.rest] == ["&&", "||"]


def test_if_elif_else() -> None:
    node = parse("if a; then b; elif c; then d; else e; fi").items[0].first.commands[0]
    assert isinstance(node, IfCommand)
    assert len(node.branches) == 2 and node.else_body is not None


def test_for_and_while_multiline() -> None:
    program = parse("for x in 1 2 3\ndo\n  echo $x\ndone\nwhile false; do echo; done")
    assert isinstance(program.items[0].first.commands[0], ForCommand)
    assert isinstance(program.items[1].first.commands[0], WhileCommand)


def test_for_words_and_redirect_after_done() -> None:
    node = parse("for f in *.c; do echo $f; done > list.txt").items[0].first.commands[0]
    assert [w.raw for w in node.words] == ["*.c"]
    assert node.redirects[0].op == ">"


def test_subshell_and_negation() -> None:
    pipeline = parse("! (cd src; ls)").items[0].first
    assert pipeline.negated and isinstance(pipeline.commands[0], Subshell)


def test_keyword_as_argument() -> None:
    command = parse("echo then fi").items[0].first.commands[0]
    assert [w.raw for w in command.words] == ["echo", "then", "fi"]


@pytest.mark.parametrize("source, message", [
    ("then echo", "syntax error near unexpected token `then'"),
    ("echo a |", "syntax error: unexpected end of file"),
    ("if true; then echo", "syntax error: unexpected end of file"),
    ("echo >", "syntax error near unexpected token `newline'"),
    ("| ls", "syntax error near unexpected token `|'"),
    ("sleep 1 &", "background"),
])
def test_syntax_errors(source, message) -> None:
    with pytest.raises(ParseError) as info:
        parse(source)
    assert message in info.value.message


@pytest.mark.parametrize("source, expected", [
    ("if true; then", True), ("echo 'open", True), ("echo a |", True), ("echo \\", True),
    ("echo done", False), ("fi", False),
])
def test_needs_more_input(source, expected) -> None:
    assert needs_more_input(source) is expected
