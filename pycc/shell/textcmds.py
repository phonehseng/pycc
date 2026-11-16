"""Text commands: ``echo`` and the filters that make pipes useful.

``head tail wc grep diff sort uniq seq`` all read files named on the command line or,
when none are given, standard input, so that ``./prog | grep corn | wc -l``
works. They are written against plain strings: the executor has already turned the
previous pipeline stage's output into ``ctx.stdin``.
"""
from __future__ import annotations

import difflib
import posixpath
import re

from pycc.core.concepts import Concept
from pycc.shell.ansi import GREEN, RED, paint
from pycc.shell.commands import CommandContext, CommandError, missing_operand, parse_options

# --- echo ---------------------------------------------------------------------------------

_ECHO_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "a": "\a", "b": "\b", "r": "\r",
                 "e": "\x1b", "v": "\v", "f": "\f"}


def cmd_echo(ctx: CommandContext, args: list[str]) -> int:
    """Print the arguments separated by spaces (``-n`` no newline, ``-e`` escapes)."""
    newline, escapes = True, False
    # echo only treats leading words made purely of n/e/E letters as options.
    while args and re.fullmatch(r"-[neE]+", args[0]):
        newline = newline and "n" not in args[0]
        escapes = ("e" in args[0]) or (escapes and "E" not in args[0])
        args = args[1:]
    text = " ".join(args)
    if escapes:
        text, stop = interpret_escapes(text)
        if stop:
            newline = False
    ctx.out(text + ("\n" if newline else ""))
    return 0


def interpret_escapes(text: str) -> tuple[str, bool]:
    """Apply ``echo -e`` escapes; the flag is True when ``\\c`` stopped the output."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\" or index + 1 >= len(text):
            out.append(char)
            index += 1
            continue
        code = text[index + 1]
        if code == "c":
            return "".join(out), True
        if code in _ECHO_ESCAPES:
            out.append(_ECHO_ESCAPES[code])
            index += 2
        elif code == "0":
            digits = re.match(r"[0-7]{0,3}", text[index + 2:]).group(0)
            out.append(chr(int(digits or "0", 8)))
            index += 2 + len(digits)
        elif code == "x" and re.match(r"[0-9a-fA-F]{1,2}", text[index + 2:]):
            digits = re.match(r"[0-9a-fA-F]{1,2}", text[index + 2:]).group(0)
            out.append(chr(int(digits, 16)))
            index += 2 + len(digits)
        else:
            out.append("\\" + code)
            index += 2
    return "".join(out), False


# --- head and tail ------------------------------------------------------------------------

def _line_count(name: str, options: dict[str, str | bool]) -> int:
    value = str(options.get("n", "10"))
    if not value.isdigit():
        raise CommandError("bad_option", f"{name}: invalid number of lines: '{value}'",
                           f"-n needs a whole number, like `{name} -n 5`.")
    return int(value)


def _head_or_tail(ctx: CommandContext, args: list[str], from_end: bool) -> int:
    name = "tail" if from_end else "head"
    options, operands = parse_options(name, args, "", valued="n", numeric_shortcut="n")
    count = _line_count(name, options)
    inputs = ctx.read_inputs(operands)
    status = 0
    for index, (source, text) in enumerate(inputs):
        if text is None:
            status = 1
            continue
        if len(inputs) > 1:
            ctx.out(("\n" if index else "") + f"==> {source} <==\n")
        lines = text.splitlines(keepends=True)
        chosen = (lines[-count:] if count else []) if from_end else lines[:count]
        ctx.out("".join(chosen))
    return status


def cmd_head(ctx: CommandContext, args: list[str]) -> int:
    """Print the first lines (``-n N``, default 10)."""
    return _head_or_tail(ctx, args, from_end=False)


def cmd_tail(ctx: CommandContext, args: list[str]) -> int:
    """Print the last lines (``-n N``, default 10)."""
    return _head_or_tail(ctx, args, from_end=True)


# --- wc -----------------------------------------------------------------------------------

def cmd_wc(ctx: CommandContext, args: list[str]) -> int:
    """Count lines, words and bytes (``-l -w -c`` pick columns)."""
    options, operands = parse_options("wc", args, "lwc")
    columns = [letter for letter in "lwc" if options.get(letter)] or ["l", "w", "c"]
    rows: list[tuple[list[int], str]] = []
    status = 0
    for source, text in ctx.read_inputs(operands):
        if text is None:
            status = 1
            continue
        counts = {"l": text.count("\n"), "w": len(text.split()), "c": len(text.encode("utf-8"))}
        rows.append(([counts[letter] for letter in columns], "" if source == "-" else source))
    if len(rows) > 1:
        totals = [sum(row[0][i] for row in rows) for i in range(len(columns))]
        rows.append((totals, "total"))
    ctx.out(_format_wc(rows, columns, from_stdin=not operands))
    return status


def _format_wc(rows: list[tuple[list[int], str]], columns: list[str], from_stdin: bool) -> str:
    if not rows:
        return ""
    largest = max(max(numbers) for numbers, _ in rows)
    # Like GNU wc: a single number from a pipe is printed bare; columns are aligned.
    width = 7 if from_stdin and len(columns) > 1 else len(str(largest))
    lines = []
    for numbers, label in rows:
        cells = " ".join(f"{number:>{width}}" for number in numbers)
        lines.append(f"{cells} {label}".rstrip() if label else cells)
    return "\n".join(lines) + "\n"


# --- grep ---------------------------------------------------------------------------------

def basic_to_python_regex(pattern: str) -> str:
    """Translate a POSIX *basic* regex into Python syntax.

    In basic regexes (grep without -E) the characters ``+ ? ( ) { } |`` are ordinary
    and become special only after a backslash; Python is the other way round, so the
    escaping is swapped for exactly these characters.
    """
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            nxt = pattern[index + 1]
            out.append(nxt if nxt in "+?(){}|" else "\\" + nxt)
            index += 2
            continue
        out.append("\\" + char if char in "+?(){}|" else char)
        index += 1
    return "".join(out)


def _compile_grep(pattern: str, extended: bool, ignore_case: bool,
                  fixed: bool = False) -> re.Pattern[str]:
    """The pattern as a Python regex (``fixed`` means ``-F``: plain text, no symbols)."""
    if fixed:
        source = re.escape(pattern)
    else:
        source = pattern if extended else basic_to_python_regex(pattern)
    try:
        return re.compile(source, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        raise CommandError("bad_regex", f"grep: invalid regular expression: {error.msg}",
                           "The pattern has an unmatched bracket or a misplaced symbol. "
                           "Put it in single quotes, and check every ( has a ).", 2,
                           Concept.SHELL_PIPES) from None


def _grep_sources(ctx: CommandContext, operands: list[str], recursive: bool) -> list[str]:
    """File names to search, with folders expanded under ``-r``.

    ``grep -r x`` with no folder searches the current one and prints plain relative
    names (``src/main.c``), while ``grep -r x .`` prints ``./src/main.c``, like GNU grep.
    """
    if not recursive:
        return operands
    names: list[str] = []
    for operand in operands or [""]:
        path = ctx.resolve(operand or ".")
        if not ctx.vfs.is_dir(path):
            names.append(operand)
            continue
        for folder, _dirs, files in ctx.vfs.walk(path):
            relative = posixpath.relpath(folder, path)
            base = operand.rstrip("/") if relative == "." else posixpath.join(operand, relative)
            names.extend(posixpath.join(base, name) if base else name for name in files)
    return names


def cmd_grep(ctx: CommandContext, args: list[str]) -> int:
    """Print lines matching a pattern (``-n -i -c -v -r -E -F -o``)."""
    options, operands = parse_options("grep", args, "nicvrEFo")
    if not operands:
        raise CommandError("missing_operand", "Usage: grep [OPTION]... PATTERNS [FILE]...",
                           "grep needs a pattern to look for, like `grep main main.c`.", 2)
    regex = _compile_grep(operands[0], bool(options.get("E")), bool(options.get("i")),
                          bool(options.get("F")))
    recursive = bool(options.get("r"))
    sources = _grep_sources(ctx, operands[1:], recursive)
    show_names = len(sources) > 1 or recursive
    matched_any, failed = False, False
    for source, text in _grep_inputs(ctx, sources):
        if text is None:
            failed = True
            continue
        label = source if show_names else ""
        if looks_binary(text):
            matched = _grep_binary(ctx, regex, options, label, source, text)
        else:
            matched = _grep_one(ctx, regex, options, label, text)
        matched_any = matched or matched_any
    # GNU grep: an error wins over a match (2), then match (0), then no match (1).
    if failed:
        return 2
    return 0 if matched_any else 1


# Our compiled files start with these magic lines (see shell/executor.py).
_BINARY_MAGICS = ("CVEXE1\n", "CVOBJ1\n")


def looks_binary(text: str) -> bool:
    """True for file contents grep should not print line by line.

    Like GNU grep: a NUL byte, or bytes that were not valid UTF-8 (they decode to
    U+FFFD), mark a binary file. Printing such lines would fill the terminal with
    garbage, so grep only says whether the file matches.
    """
    return text.startswith(_BINARY_MAGICS) or "\0" in text or "\ufffd" in text


def _grep_binary(ctx: CommandContext, regex: re.Pattern[str], options: dict[str, str | bool],
                 label: str, source: str, text: str) -> bool:
    """Search a binary file: report a match in one line instead of printing lines."""
    invert = bool(options.get("v"))
    hits = sum(1 for line in text.splitlines() if bool(regex.search(line)) != invert)
    if options.get("c"):
        ctx.out(f"{label + ':' if label else ''}{hits}\n")
    elif hits:
        shown = "(standard input)" if source == "-" else source
        ctx.out(f"grep: {shown}: binary file matches\n")
    return hits > 0


def _grep_inputs(ctx: CommandContext, sources: list[str]) -> list[tuple[str, str | None]]:
    if not sources:
        return ctx.read_inputs([])
    result: list[tuple[str, str | None]] = []
    for source in sources:
        if source != "-" and ctx.vfs.is_dir(ctx.resolve(source)):
            ctx.fail("is_a_directory", f"grep: {source}: Is a directory",
                     f"'{source}' is a folder. Use `grep -r` to search inside folders.")
            result.append((source, None))
        else:
            result.extend(ctx.read_inputs([source]))
    return result


def _grep_one(ctx: CommandContext, regex: re.Pattern[str], options: dict[str, str | bool],
              label: str, text: str) -> bool:
    invert = bool(options.get("v"))
    prefix = f"{label}:" if label else ""
    hits = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if bool(regex.search(line)) == invert:
            continue
        hits += 1
        if options.get("c"):
            continue
        line_number = f"{number}:" if options.get("n") else ""
        if not options.get("o"):
            ctx.out(f"{prefix}{line_number}{line}\n")
        elif not invert:                          # -o: each match on its own line
            for match in regex.finditer(line):
                if match.group(0):
                    ctx.out(f"{prefix}{line_number}{match.group(0)}\n")
    if options.get("c"):
        ctx.out(f"{prefix}{hits}\n")
    return hits > 0


# --- diff ---------------------------------------------------------------------------------

def cmd_diff(ctx: CommandContext, args: list[str]) -> int:
    """Compare two files (normal format, or ``-u`` unified)."""
    options, operands = parse_options("diff", args, "u")
    if len(operands) != 2:
        raise CommandError("missing_operand",
                           f"diff: missing operand after '{operands[-1] if operands else 'diff'}'",
                           "diff compares two files, like `diff expected.txt out.txt`.", 2)
    texts = ctx.read_inputs(operands)
    if any(text is None for _, text in texts):
        return 2
    old, new = (text.splitlines() for _, text in texts)   # type: ignore[union-attr]
    if old == new:
        return 0
    if options.get("u"):
        lines = list(difflib.unified_diff(old, new, operands[0], operands[1], lineterm=""))
    else:
        lines = _normal_diff(old, new)
    ctx.out("".join(_colour_diff_line(ctx, line) + "\n" for line in lines))
    return 1


def _range(start: int, end: int) -> str:
    """diff's 1-based line range: ``3`` or ``3,5``."""
    return str(start + 1) if end - start == 1 else f"{start + 1},{end}"


def _normal_diff(old: list[str], new: list[str]) -> list[str]:
    lines: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            lines.append(f"{_range(i1, i2)}c{_range(j1, j2)}")
            lines += ["< " + line for line in old[i1:i2]] + ["---"]
            lines += ["> " + line for line in new[j1:j2]]
        elif tag == "delete":
            lines.append(f"{_range(i1, i2)}d{j1}")
            lines += ["< " + line for line in old[i1:i2]]
        else:
            lines.append(f"{i1}a{_range(j1, j2)}")
            lines += ["> " + line for line in new[j1:j2]]
    return lines


def _colour_diff_line(ctx: CommandContext, line: str) -> str:
    if not ctx.stdout_is_terminal or line.startswith(("---", "+++")):
        return line
    if line.startswith(("-", "<")):
        return paint(line, RED)
    if line.startswith(("+", ">")):
        return paint(line, GREEN)
    return line


# --- sort, uniq, seq ----------------------------------------------------------------------

_LEADING_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


def _numeric_key(line: str) -> tuple[float, str]:
    match = _LEADING_NUMBER.match(line)
    return (float(match.group(1)) if match else 0.0, line)


def cmd_sort(ctx: CommandContext, args: list[str]) -> int:
    """Sort lines (``-r`` reverse, ``-n`` numeric)."""
    options, operands = parse_options("sort", args, "rn")
    lines: list[str] = []
    status = 0
    for _source, text in ctx.read_inputs(operands):
        if text is None:
            status = 2
            continue
        lines.extend(text.splitlines())
    key = _numeric_key if options.get("n") else None
    lines.sort(key=key, reverse=bool(options.get("r")))
    ctx.out("".join(line + "\n" for line in lines))
    return status


def cmd_uniq(ctx: CommandContext, args: list[str]) -> int:
    """Collapse repeated neighbouring lines (``-c`` counts them)."""
    options, operands = parse_options("uniq", args, "c")
    (_source, text), = ctx.read_inputs(operands[:1])
    if text is None:
        return 1
    groups: list[list] = []
    for line in text.splitlines():
        if groups and groups[-1][0] == line:
            groups[-1][1] += 1
        else:
            groups.append([line, 1])
    if options.get("c"):
        ctx.out("".join(f"{count:>7} {line}\n" for line, count in groups))
    else:
        ctx.out("".join(f"{line}\n" for line, _count in groups))
    return 0


# seq runs inside a single-threaded host, so a huge range must be refused.
SEQ_LIMIT = 100_000


def _seq_count(first: float, step: float, last: float) -> int:
    """How many numbers seq prints (0 when the step points away from ``last``)."""
    span = (last - first) / step
    # The tiny epsilon absorbs float error such as (1.0 - 0.0) / 0.1 == 9.999999...
    return int(span + 1e-9) + 1 if span >= 0 else 0


def _seq_number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise CommandError("bad_option", f"seq: invalid floating point argument: '{text}'",
                           "seq needs numbers, like `seq 1 10` or `seq 0 2 10`.") from None


def cmd_seq(ctx: CommandContext, args: list[str]) -> int:
    """Print numbers: ``seq LAST``, ``seq FIRST LAST`` or ``seq FIRST STEP LAST``."""
    if not args:
        raise missing_operand("seq", "seq 1 10")
    if len(args) > 3:
        raise CommandError("bad_option", f"seq: extra operand '{args[3]}'",
                           "seq takes at most three numbers: first, step and last.")
    numbers = [_seq_number(arg) for arg in args]
    if len(numbers) == 1:
        first, step, last = 1.0, 1.0, numbers[0]
    elif len(numbers) == 2:
        first, step, last = numbers[0], 1.0, numbers[1]
    else:
        first, step, last = numbers
    if step == 0:
        raise CommandError("bad_option", "seq: invalid Zero increment value: '0'",
                           "A step of 0 would never reach the end. Use a positive or negative "
                           "step.")
    count = _seq_count(first, step, last)
    if count > SEQ_LIMIT:
        raise CommandError("bad_option", f"seq: too many numbers ({count}); the limit is "
                           f"{SEQ_LIMIT}",
                           "That would print so many lines the terminal would freeze. Pick a "
                           "smaller range.")
    whole = all(float(number).is_integer() for number in (first, step, last))
    # first + i * step instead of adding step again and again: repeated float
    # additions drift (0.1 + 0.1 + ...), which could drop or add the last number.
    values = (first + index * step for index in range(count))
    ctx.out("".join((str(int(value)) if whole else f"{value:g}") + "\n" for value in values))
    return 0


# --- printf -------------------------------------------------------------------------------

_PRINTF_SPEC = re.compile(r"%(?P<flags>[-+ 0#]*)(?P<width>\d*)(?:\.(?P<precision>\d*))?"
                          r"(?P<conv>[sdicxXouefgEG%])")
_INTEGER_CONVERSIONS = "dixXou"
_FLOAT_CONVERSIONS = "efgEG"


def cmd_printf(ctx: CommandContext, args: list[str]) -> int:
    """Print FORMAT with the arguments filled in, like C's printf.

    ``%s %d %i %c %x %o %u %f %e %g %%`` with flags, width and precision, and the
    escapes of ``echo -e`` (``\\n``, ``\\t``...). As in bash, the format is used again
    until every argument has been printed, and a missing argument counts as empty
    (or 0). No newline is added: write ``\\n`` yourself.
    """
    if not args:
        raise CommandError("missing_operand", "printf: usage: printf format [arguments]",
                           "printf needs a format, like `printf \"%s has %d cats\\n\" Ada 3`.",
                           2)
    form, values = args[0], args[1:]
    pieces: list[str] = []
    bad: list[str] = []
    while True:
        text, used, stop = _printf_once(form, values, bad)
        pieces.append(text)
        values = values[used:]
        if stop or used == 0 or not values:
            break
    ctx.out("".join(pieces))
    for value in bad[1:]:
        ctx.err(f"bash: printf: {value}: invalid number\n")
    if bad:
        return ctx.fail("bad_option", f"bash: printf: {bad[0]}: invalid number",
                        "%d and friends need a whole number, like 3 or -12; it printed 0.")
    return 0


def _printf_once(form: str, values: list[str], bad: list[str]) -> tuple[str, int, bool]:
    """One pass over the format: (text, arguments used, stopped by ``\\c``)."""
    out: list[str] = []
    used = 0
    position = 0
    for match in _PRINTF_SPEC.finditer(form):
        literal, stop = interpret_escapes(form[position:match.start()])
        out.append(literal)
        if stop:
            return "".join(out), used, True
        position = match.end()
        if match.group("conv") == "%":
            out.append("%")
            continue
        value = values[used] if used < len(values) else None
        used += 1 if value is not None else 0
        out.append(_printf_convert(match, value, bad))
    literal, stop = interpret_escapes(form[position:])
    out.append(literal)
    return "".join(out), used, stop


def _printf_convert(match: re.Match[str], value: str | None, bad: list[str]) -> str:
    """Format one argument for one ``%`` conversion."""
    conv = match.group("conv")
    precision = match.group("precision")
    spec = "%" + match.group("flags") + match.group("width")
    spec += f".{precision or 0}" if precision is not None else ""
    if conv in _INTEGER_CONVERSIONS:
        return (spec + ("d" if conv in "iu" else conv)) % _printf_number(value, int, bad)
    if conv in _FLOAT_CONVERSIONS:
        return (spec + conv) % _printf_number(value, float, bad)
    text = value or ""
    return (spec + "s") % (text[:1] if conv == "c" else text)


def _printf_number(value: str | None, kind: type, bad: list[str]) -> int | float:
    """An argument as a number; ``'a`` is the character code, as in bash."""
    if not value:
        return kind(0)
    if value[0] in "'\"" and len(value) > 1:
        return kind(ord(value[1]))
    try:
        return kind(value) if kind is float else _parse_integer(value)
    except ValueError:
        bad.append(value)
        return kind(0)


def _parse_integer(text: str) -> int:
    """A decimal or ``0x`` hexadecimal whole number (raises ``ValueError``)."""
    lowered = text.strip().lower()
    if lowered.lstrip("+-").startswith("0x"):
        return int(lowered, 16)
    return int(lowered, 10)


# --- tr ----------------------------------------------------------------------------------

_TR_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "r": "\r"}


def expand_tr_set(text: str) -> str:
    r"""A ``tr`` set as its characters: ``a-z`` ranges and ``\n \t \\`` escapes."""
    chars: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            chars.append(_TR_ESCAPES.get(text[index + 1], text[index + 1]))
            index += 2
            continue
        if index + 2 < len(text) and text[index + 1] == "-" and ord(text[index + 2]) >= ord(char):
            chars.extend(chr(code) for code in range(ord(char), ord(text[index + 2]) + 1))
            index += 3
            continue
        chars.append(char)
        index += 1
    return "".join(chars)


def cmd_tr(ctx: CommandContext, args: list[str]) -> int:
    """Translate characters of stdin (``tr abc xyz``) or delete them (``tr -d SET``)."""
    options, operands = parse_options("tr", args, "d")
    wanted = 1 if options.get("d") else 2
    if len(operands) != wanted:
        usage = "tr -d SET" if options.get("d") else "tr SET1 SET2"
        raise CommandError("missing_operand", f"tr: missing operand (usage: {usage})",
                           "tr reads stdin: `echo hello | tr a-z A-Z` or `tr -d '\\r' < notes.txt`.", 1)
    text = ctx.read_stdin()
    source = expand_tr_set(operands[0])
    if options.get("d"):
        ctx.out(text.translate({ord(char): None for char in source}))
        return 0
    target = expand_tr_set(operands[1])
    if not target:
        raise CommandError("bad_operand", "tr: when not truncating set1, string2 must be non-empty",
                           "Give tr a second set of characters to change into.", 1)
    target = target + target[-1] * max(0, len(source) - len(target))   # GNU pads with the last
    ctx.out(text.translate({ord(a): b for a, b in zip(source, target)}))
    return 0


TEXT_COMMANDS = {
    "echo": cmd_echo, "head": cmd_head, "tail": cmd_tail, "wc": cmd_wc, "grep": cmd_grep,
    "diff": cmd_diff, "sort": cmd_sort, "uniq": cmd_uniq, "seq": cmd_seq,
    "printf": cmd_printf, "tr": cmd_tr,
}
