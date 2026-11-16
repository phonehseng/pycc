"""Short beginner man pages, one per command, written in ``core.markup``.

Real man pages are long and dense; these are deliberately tiny so a newcomer is not
overwhelmed: one heading line ``# name - what it does`` (which doubles as the
one-line summary shown by ``help``), the usage, the few options that matter and one
example. ``render_manpage`` turns a page into terminal text.
"""
from __future__ import annotations

import textwrap

from pycc.core.markup import parse
from pycc.shell.ansi import BOLD, RESET

MANPAGES: dict[str, str] = {
    "ls": """# ls - list the files in a folder
`ls [-l] [-a] [-1] [PATH...]`
Shows what is inside a folder, the current one if you name none. Folders are blue,
runnable programs green.
- `-l` long form: permissions, size and time
- `-a` also show hidden names that start with `.`
- `-1` one name per line
Example: `ls -l src`
""",
    "cd": """# cd - change the current folder
`cd [FOLDER]`
Moves you into FOLDER. `cd` alone goes home, `cd ..` goes up one level, `cd -` goes back.
Example: `cd src`
""",
    "pwd": """# pwd - print the current folder
`pwd`
Prints the full path of the folder you are in.
""",
    "cat": """# cat - print files
`cat [-n] [FILE...]`
Prints the files one after another (or what comes in through a pipe).
- `-n` number the lines
Example: `cat -n main.c`
""",
    "mkdir": """# mkdir - make folders
`mkdir [-p] FOLDER...`
- `-p` also make missing parent folders, and do not complain if it exists
Example: `mkdir -p src/include`
""",
    "touch": """# touch - create an empty file or mark it as changed
`touch FILE...`
Creates FILE if it is missing, otherwise updates its time (so `make` rebuilds it).
""",
    "rm": """# rm - remove files
`rm [-r] [-f] PATH...`
There is no bin: removed files are gone.
- `-r` remove folders and everything inside them
- `-f` do not complain about missing files
Example: `rm *.o`
""",
    "cp": """# cp - copy files
`cp [-r] SOURCE... DEST`
Copies SOURCE to DEST, or into DEST when it is a folder.
- `-r` copy folders too
Example: `cp main.c backup.c`
""",
    "mv": """# mv - move or rename
`mv SOURCE... DEST`
Renames SOURCE to DEST, or moves it into the folder DEST.
Example: `mv old.c new.c`
""",
    "echo": """# echo - print text
`echo [-n] [-e] TEXT...`
- `-n` no newline at the end
- `-e` understand escapes such as `\\n` and `\\t`
Example: `echo "hello" > greeting.txt`
""",
    "head": """# head - print the first lines
`head [-n N] [FILE...]`
- `-n N` print N lines instead of 10
Example: `head -n 3 log.txt`
""",
    "tail": """# tail - print the last lines
`tail [-n N] [FILE...]`
- `-n N` print N lines instead of 10
Example: `tail -n 5 log.txt`
""",
    "wc": """# wc - count lines, words and bytes
`wc [-l] [-w] [-c] [FILE...]`
- `-l` lines only
- `-w` words only
- `-c` bytes only
Example: `grep printf *.c | wc -l`
""",
    "grep": """# grep - find lines that match a pattern
`grep [-n] [-i] [-c] [-v] [-r] [-E] [-F] [-o] PATTERN [FILE...]`
- `-n` show line numbers
- `-i` ignore upper/lower case
- `-c` only count matching lines
- `-v` show the lines that do NOT match
- `-r` search folders too
- `-E` extended patterns: `+ ? | ( )` work without backslashes
- `-F` plain text: `.` and `*` mean themselves
- `-o` print only the matching part, one per line
Example: `grep -n main *.c`
""",
    "tr": """# tr - change or delete characters
`tr SET1 SET2` or `tr -d SET` (reads standard input only)
- `tr a-z A-Z` changes each letter of SET1 into the one at the same place in SET2
- `-d` deletes every character of SET
- ranges such as `a-z` and the escapes `\\n` `\\t` work
Example: `echo hello | tr a-z A-Z`
""",
    "find": """# find - search for files by name or kind
`find [PATH...] [-name PATTERN] [-type f|d]`
- `-name` match the name, with `*` and `?` (quote the pattern)
- `-type f` files only, `-type d` folders only
Example: `find . -name "*.c"`
""",
    "diff": """# diff - compare two files line by line
`diff [-u] FILE1 FILE2`
Prints nothing and succeeds when the files are the same.
- `-u` unified form: `-` lines removed, `+` lines added
Example: `diff -u expected.txt out.txt`
""",
    "sort": """# sort - sort lines
`sort [-r] [-n] [FILE...]`
- `-r` reverse order
- `-n` compare as numbers
Example: `sort -n scores.txt`
""",
    "uniq": """# uniq - squeeze repeated neighbouring lines
`uniq [-c] [FILE]`
Only neighbours count, so `sort` first.
- `-c` prefix each line with how many times it appeared
Example: `sort words.txt | uniq -c`
""",
    "seq": """# seq - print a sequence of numbers
`seq [FIRST [STEP]] LAST`
Example: `seq 1 5`
""",
    "clear": """# clear - clear the terminal
`clear`
""",
    "history": """# history - list the commands you typed
`history [-c] [N]`
- `-c` forget the history
- `N` show only the last N commands
""",
    "man": """# man - show the manual page of a command
`man COMMAND`
Example: `man grep`
""",
    "help": """# help - list the commands
`help [COMMAND]`
With a name, shows that command's summary and usage.
""",
    "chmod": """# chmod - change who may run a file
`chmod +x|-x FILE...`
- `+x` mark the file as runnable, `-x` take that away
Example: `chmod +x run.sh`
""",
    "env": """# env - print the exported variables
`env`
""",
    "export": """# export - give a variable to the programs you run
`export NAME[=VALUE]...`
Example: `export GREETING=hello`
""",
    "which": """# which - show where a command lives
`which COMMAND...`
""",
    "true": """# true - do nothing, successfully
`true`
Exits with status 0. Handy in `while true`.
""",
    "false": """# false - do nothing, unsuccessfully
`false`
Exits with status 1.
""",
    "test": """# test - check a condition (also written `[ ... ]`)
`test EXPRESSION` or `[ EXPRESSION ]`
Succeeds (status 0) when the expression is true.
- `-f FILE` is a file, `-d DIR` is a folder, `-e PATH` exists
- `-z TEXT` is empty, `-n TEXT` is not empty
- `A = B`, `A != B` compare text
- `-eq -ne -lt -le -gt -ge` compare numbers
- `!` not, `-a` and, `-o` or
Example: `if [ -f out.txt ]; then cat out.txt; fi`
""",
    "[": """# [ - check a condition (same as `test`)
`[ EXPRESSION ]`
The spaces matter and the last word must be `]`. See `man test`.
""",
    "break": """# break - leave a loop early
`break [N]`
Ends the innermost for, while or until loop at once; `break 2` also ends the loop around it.
Outside a loop it does nothing.
""",
    "continue": """# continue - skip to the next round of a loop
`continue [N]`
Stops this round of the innermost loop and starts the next one (`continue 2`: of the outer loop).
""",
    "exit": """# exit - leave the shell or the script
`exit [N]`
Ends with status N (default: the last command's status).
""",
    "source": """# source - run a script in this shell
`source FILE [ARGS...]` (also written `. FILE`)
Unlike `sh FILE`, variables and `cd` inside FILE stay set afterwards.
""",
    ".": """# . - run a script in this shell (same as `source`)
`. FILE`
""",
    "sh": """# sh - run a shell script
`sh FILE [ARGS...]` or `sh -c "COMMANDS"`
The script gets ARGS as `$1`, `$2`... It does not need `chmod +x`.
Example: `sh build.sh`
""",
    "bash": """# bash - run a shell script (same as `sh`)
`bash FILE [ARGS...]` or `bash -c "COMMANDS"`
""",
    "cc": """# cc - the C compiler
`cc [-Wall] [-Wextra] [-Werror] [-c] [-E] [-I DIR] [-D NAME[=VALUE]] [-o OUT] FILE.c FILE.o...`
- `-c` compile to an object file `.o` only (no linking)
- `-o OUT` name the result (default `a.out`, or `FILE.o` with `-c`)
- `-Wall` turn on the useful warnings; `-Wextra` a few more
- `-Werror` treat every warning as an error
- `-I DIR` also look for `#include "x.h"` headers in DIR
- `-D NAME` / `-D NAME=VALUE` define a macro; `-U NAME` undefine it
- `-E` print the preprocessed text instead of compiling
- `-g`, `-O2`, `-std=c99`, `-lm` are accepted and change nothing here
Example: `cc -Wall main.c util.c -o bot`
""",
    "gcc": """# gcc - the C compiler (same as `cc`)
`gcc [-Wall] [-c] [-o OUT] FILE.c...`
Every option of `cc` works the same way; see `man cc`.
""",
    "make": """# make - rebuild what changed, using a Makefile
`make [-n] [-B] [-s] [-f FILE] [-C DIR] [NAME=VALUE] [TARGET...]`
- `-n` show the commands without running them
- `-B` rebuild everything, even what is up to date
- `-s` do not echo the commands
- `-f FILE` read FILE instead of `Makefile`
- `-C DIR` change into DIR first
Recipe lines must start with a Tab. Example: `make clean`
""",
    "file": """# file - guess what kind of file something is
`file PATH...`
Example: `file prog`
""",
    "date": """# date - print the date
`date`
The simulated clock is always the same day.
""",
    "sleep": """# sleep - wait a moment
`sleep SECONDS`
Time flies here: it returns at once.
""",
    "time": """# time - measure how long a command takes
`time COMMAND [ARGS...]`
Runs the command, then prints `real`, `user` and `sys` on stderr, as bash does.
pycc keeps a steady clock, the same on every computer: each C statement takes ten microseconds, starting a program a millisecond and each device call a tenth of a second.
Example: `time ./bot`
""",
    "printf": """# printf - print text with values filled in
`printf FORMAT [ARGS...]`
- `%s` text, `%d` a whole number, `%c` one character, `%x` hexadecimal, `%f` a decimal number, `%%` a percent sign
- width and precision work as in C: `%5d`, `%-8s`, `%.2f`
- escapes such as `\\n` and `\\t`; no newline is added unless you write one
The format is used again until every argument is printed.
Example: `printf "%s has %d cats\\n" Ada 3`
""",
    "tree": """# tree - draw a folder as a tree
`tree [FOLDER]`
""",
    "fortune": """# fortune - a proverb of the C masters
`fortune`
""",
}


def _summary(page: str) -> str:
    """The text after ``# name - `` on the first line."""
    first = page.strip().splitlines()[0].lstrip("#").strip()
    return first.split(" - ", 1)[1] if " - " in first else first


SUMMARIES: dict[str, str] = {name: _summary(page) for name, page in MANPAGES.items()}


def render_manpage(page: str, width: int = 100) -> str:
    """Turn a markup page into terminal text: bold heading, wrapped bullets.

    Each source line is rendered on its own (the pages are written line by line, so
    the usage line must not be merged into the following sentence).
    """
    lines: list[str] = []
    for source_line in page.strip().splitlines():
        for block in parse(source_line):
            lines.extend(_render_block(block.kind, block.text, width))
    return "\n".join(lines) + "\n"


def _render_block(kind: str, text: str, width: int) -> list[str]:
    if kind == "heading":
        return [f"{BOLD}{text}{RESET}"]
    if kind == "bullet":
        return textwrap.wrap(text, width, initial_indent="  - ", subsequent_indent="    ")
    if kind == "code":
        return ["    " + line for line in text.splitlines()]
    return textwrap.wrap(text, width) or [""]
