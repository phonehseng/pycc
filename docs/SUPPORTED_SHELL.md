# The pycc terminal: shell, commands, cc and make

The pycc shell (`pycc shell`) is a small bash. Everything here behaves like bash, gcc and GNU
make on Linux, with the same messages, so what you learn works on a real computer.
Under every error the terminal adds one dimmed, plain-English line that explains it.
For shell, `cc` and `make` errors that line goes to the screen only, so `2> errors.txt`
gets exactly what bash or gcc would write.

This page lists exactly what is supported. Users: skim the tables. Developers: the
code is in `shell/` (`lexer.py`, `parser.py`, `expand.py`, `arith.py`, `executor.py`,
`builtins.py`, `textcmds.py`, `sessioncmds.py`, `cc.py`, `make.py`) . The end-to-end scenarios in `tests/integration/`
exercise all of it through a real `Shell`.

---

## 1. Where you are

- Your project lives in `/home/user/project`; the prompt shows it as `~/project`.
- `/usr/include` holds read-only copies of the headers the compiler uses
  (`cat /usr/include/stdio.h`), `/bin` has one entry per command (so `which ls` prints
  `/bin/ls`) and `/tmp` is free to use. `/dev/null` reads as empty.
- Starting variables: `HOME=/home/user`, `USER=user`, `PATH=/bin:.`,
  `SHELL=/bin/sh`, `PWD`, `TERM`. Because `PATH` ends with `.`, a program in the
  current folder also runs without `./` (but `./prog` is the habit to learn).
- The filesystem lives in memory. `pycc shell` copies the current folder into `~/project`; nothing is written back to the real disk.

---

## 2. The language

### Supported

| feature | examples |
|---|---|
| words and quoting | `'single'` (literal), `"double $HOME"` (expands), `\ ` escapes |
| comments | `# to the end of the line` |
| variables | `NAME=value`, `$NAME`, `${NAME}`, `export NAME=value`, `export NAME` |
| one-command variables | `MODE=fast ./prog` (only that program sees it) |
| special parameters | `$?` (last status), `$#`, `$0`, `$1`..`$9`, `$@`, `"$@"`, `$*`, `$$` |
| command substitution | `$(ls)`, `` `pwd` `` (trailing newlines removed) |
| arithmetic | `$((i + 1))`, `$(( (a + b) * 2 ))`, `$((n % 3 == 0))` |
| tilde | `~`, `~/project` |
| globbing | `*.c`, `util?.c`, `[abc].txt` (a pattern that matches nothing stays as typed) |
| pipelines | `./prog \| grep WARN \| wc -l`, `! cmd` (negates the status) |
| lists | `a ; b`, `a && b`, `a \|\| b`, newlines |
| redirections | `> f`, `>> f`, `< f`, `2> f`, `2>> f`, `1> f`, `&> f`, `&>> f`, `2>&1`, `1>&2`, `>&2` |
| `if` | `if cmd; then ...; elif cmd; then ...; else ...; fi` |
| loops | `for x in a b c; do ...; done`, `for x; do ...; done` (over `$@`), `while cmd; do ...; done`, `until cmd; do ...; done` |
| loop control | `break`, `break 2`, `continue`, `continue 2` |
| sub-shells | `( cd src; ls )` (the `cd` does not leak out) |
| scripts | `sh run.sh`, `bash run.sh`, `./run.sh` after `chmod +x run.sh`, `source f` / `. f` (runs in this shell) |
| line continuation | `\` at the end of a line inside a script |

Arithmetic works on whole numbers: `+ - * / %` (division truncates toward zero, as in
C), unary `- + !`, `< <= > >= == !=`, `&& ||`, parentheses; names may be written with
or without `$`, and an unset or non-numeric variable counts as 0. `$((5 / 0))` is
`bash: 5 / 0: division by 0 (error token is "0")`.

### Not supported

Job control and `&` (background jobs; the error suggests `&&`), here-documents
(`<<EOF`), shell functions (`f() { ...; }`), `case`, arrays, `set`, `read`,
parameter operators such as `${X:-default}` (`bad substitution`), `[[ ... ]]`,
assignment operators inside `$(( ))` (`i += 1`, `i++`) and `**`. A loop stops after
10 000 rounds, all the loops of one command line together (nested ones too) after
50 000 rounds, and scripts may nest 40 deep, so a runaway script cannot freeze the
host. When a loop is stopped, the rest of the line is skipped. A loop that prints
nothing still hands control back to the host every few rounds, so Ctrl+C always works.

---

## 3. Commands

| command | options and forms |
|---|---|
| `ls` | `-l` (long), `-a` (hidden files), `-1` (one per line), several paths |
| `cd` | `cd DIR`, `cd`, `cd ~`, `cd ..`, `cd -` |
| `pwd` | |
| `cat` | `-n` (number lines), several files, stdin when no file |
| `mkdir` | `-p` (parents, no error if it exists) |
| `touch` | creates the file or makes it newer (what `make` looks at) |
| `rm` | `-r`/`-R` (folders), `-f` (no error when missing) |
| `cp` | `-r`/`-R` |
| `mv` | rename or move into a folder |
| `chmod` | `+x`, `-x` |
| `find` | `find [DIR] -name PATTERN -type f\|d` |
| `tree` | `tree [DIR]` |
| `file` | tells sources, scripts, object files (`relocatable object file`) and programs (`executable`) apart |
| `echo` | `-n` (no newline), `-e` (escapes such as `\t`, `\n`) |
| `printf` | `printf FORMAT ARGS...`: `%s %d %i %c %x %X %o %u %f %e %g %%` with flags, width and precision, and the escapes of `echo -e`; the format repeats until every argument is used; no newline unless you write `\n` |
| `head`, `tail` | `-n N` (default 10) |
| `wc` | `-l`, `-w`, `-c` |
| `grep` | `-n`, `-i`, `-c`, `-v`, `-r`, `-E` (extended regular expressions), `-F` (plain text, no pattern symbols), `-o` (print only the matching parts) |
| `tr` | `tr SET1 SET2` (`a-z` ranges, `
` `	` escapes), `tr -d SET`; reads stdin only |
| `sort` | `-r`, `-n` |
| `uniq` | `-c` |
| `diff` | `-u` |
| `seq` | `seq LAST`, `seq FIRST LAST`, `seq FIRST STEP LAST` |
| `test`, `[ ... ]` | files: `-f -d -e -s -r -w -x`; strings: `-z -n = == !=`; numbers: `-eq -ne -lt -le -gt -ge` |
| `true`, `false`, `exit [N]` | |
| `export`, `env` | `export NAME=value`, `env` lists the variables programs see |
| `sh`, `bash`, `source`, `.` | `bash -c "COMMANDS"` too |
| `break`, `continue` | `[N]`; outside a loop they only warn, like bash |
| `which` | `which cc` -> `/bin/cc` |
| `history` | `-c` clears it |
| `help`, `man NAME` | short beginner pages for every command |
| `clear`, `date`, `sleep` | `sleep` returns at once; the simulated date never changes |
| `time` | `time COMMAND ARGS...` runs one command, then prints bash's `real`/`user`/`sys` lines on stderr. The simulated clock is steady: 10 microseconds per C statement, 1 ms per program start, 0.1 s per device call. Unlike bash's keyword it times one command, not a whole pipeline |
| `fortune` | a proverb of the C masters |
| `cc`, `gcc` | see section 4 |
| `make` | see section 5 |

Errors use bash's words: `bash: foo: command not found`, `bash: ./x: No such file or
directory`, `bash: ./run.sh: Permission denied` (+ "Try `chmod +x run.sh`"),
`bash: ./util.o: cannot execute binary file: Exec format error`, `bash: cd: src: Not
a directory`, `rm: cannot remove 'x': Is a directory`, `bash: syntax error near
unexpected token ...`. A mistyped file name in command position (`Notes.txt`) gets
"There is a file called 'notes.txt' here: try `cat notes.txt`" rather than a command
suggestion.

Tab completes command names, file and folder names, `make` targets, `$NAME` and
`${NAME}` variables (and the names after `export`), `chmod` modes (`+x`, `-x`) and
the options of `cc gcc ls make grep rm head tail wc sort cat mkdir cp find history
echo`.

### Running programs

- `./prog ARGS...` runs a compiled program; it gets `argv`, the exported variables,
  stdin from the terminal, a pipe or a `<` file, and the project's files for `fopen`.
- Each stage of a pipeline runs to completion before the next one starts (the output
  is the same as in a real shell for the short, finite programs pycc runs); stderr skips the pipe
  unless you write `2>&1`.
- After a crash the program's report goes to its stderr (so `./prog 2> crash.txt`
  captures it) and the shell adds bash's own line on the terminal: `Segmentation fault
  (core dumped)` (status 139), `Aborted (core dumped)` (134), `Floating point exception
  (core dumped)` (136), `Killed` (137). Ctrl+C stops a program with `^C` and status 130.
- A `.o` file cannot run: without `chmod +x` it is `Permission denied` (and the friendly
  line explains that `chmod` will not help, linking will); with it, `cannot execute
  binary file`. Any other executable text file runs as a shell script.

---

## 4. `cc` / `gcc`

```
cc [-Wall] [-Wextra] [-Werror] [-c] [-E] [-I DIR] [-D NAME[=VALUE]] [-U NAME] [-o OUT] FILES...
```

| option | meaning |
|---|---|
| `-c` | compile only: `cc -c src/util.c` writes `util.o` in the current folder |
| `-o OUT` | name the output (default `a.out`; with `-c` the `.o` name). Naming an input is `fatal error: input file 'main.c' is the same as output file` |
| `-Wall`, `-Wextra` | turn on warning groups (see `docs/SUPPORTED_C.md`) |
| `-Werror` | every warning becomes an error, marked ` [-Werror]` |
| `-I DIR`, `-IDIR` | search DIR for `#include "..."` and `<...>` headers |
| `-D NAME`, `-D NAME=VALUE`, `-U NAME` | define or undefine a macro |
| `-E` | print the preprocessed text (with `# line "file"` markers) |
| accepted, no effect | `-g...`, `-O...`, `-std=...`, `-l...`, `-L...`, `-pedantic`, `-march=...`, `-f...`, other `-W...`, `-ansi`, `-w`, `-pipe`, `-s`, `-v` |

Inputs are `.c` files (compiled) and `.o` files (linked as they are), in any mix:
`cc main.c util.o -o prog`. Messages are gcc's: `main.c:3:14: error: expected ',' or
';' before 'printf'`, `cc: error: main.o: No such file or directory`, `cc: fatal error:
no input files`, `cc: error: unrecognized command-line option '--version'`; linker
errors end with `collect2: error: ld returned 1 exit status`. cc exits with 1 on any
error and writes nothing then. Not supported: `-S`, `-include`, `-M`/`-MM`
(dependency files), `--version`.

---

## 5. `make`

```
make [-n] [-B] [-s] [-f FILE] [-C DIR] [NAME=VALUE...] [TARGET...]
```

Supported Makefile features:

- rules `target: prerequisites` with Tab-indented recipes; several targets per rule;
  rules without a recipe that only add prerequisites (`main.o: main.c util.h`);
- the first target is the default; `make TARGET`, `make clean`;
- `.PHONY`; order-only prerequisites (`obj/x.o: x.c | obj`);
- variables with `=`, `:=`, `::=`, `+=`, `?=`, used as `$(NAME)` or `${NAME}`;
  `NAME=value` on the command line overrides them; `CC` (`cc`), `RM` (`rm -f`) and
  `MAKE` have defaults, as in GNU make (so `CC ?= gcc` keeps `cc`), while `CFLAGS`
  starts undefined (so `CFLAGS ?= -Wall` takes effect) and an undefined variable is
  empty;
- substitution references `$(SRCS:.c=.o)` and the functions `wildcard`, `patsubst`,
  `subst`, `notdir`, `dir`, `basename`, `addprefix`, `addsuffix`, `filter`,
  `filter-out`, `strip`, `sort`;
- automatic variables `$@ $< $^ $+ $? $* $|`;
- pattern rules (`%.o: %.c`) and the built-in rules (`%.o` from `%.c`, a program from
  its `.c` file);
- recipe prefixes `@` (do not echo), `-` (ignore failure), `+`; comments; `\` line
  continuation; `export NAME` lines (accepted; every variable reaches the recipes);
- up-to-date checks on file times: after `touch util.h` exactly the targets that
  depend on it are rebuilt.

Recipes run through the same shell, so pipes, redirections and `./prog` work in them,
and each line runs in its own sub-shell (a `cd` does not carry to the next line).

Messages are GNU make's:

| situation | message |
|---|---|
| nothing changed | `make: 'prog' is up to date.` / `make: Nothing to be done for 'all'.` |
| spaces instead of a Tab | `Makefile:2: *** missing separator.  Stop.` |
| unknown target | `make: *** No rule to make target 'nothing'.  Stop.` |
| missing prerequisite | `make: *** No rule to make target 'main.c', needed by 'prog'.  Stop.` |
| a recipe failed | `make: *** [Makefile:9: util.o] Error 1` (with `-`: `make: [Makefile:2: all] Error 1 (ignored)`) |
| no Makefile | `make: *** No targets specified and no makefile found.  Stop.` |
| a cycle | `make: Circular b <- a dependency dropped.` |

make exits with 2 when something failed. Not supported (they say so instead of
pretending): conditionals (`ifeq`, `ifneq`, `ifdef`, `ifndef`, `else`, `endif`),
`include`, `define`/`endef`. `-j N` and `-k` are accepted and change nothing: recipes
always run one at a time, in order.

---

## 6. Diagnostic codes of the shell

`command_not_found`, `no_such_file`, `permission_denied`, `not_executable`,
`is_a_directory`, `not_a_directory`, `cannot_execute_binary`, `shell_syntax_error`,
`redirect_error`, `pipe_error`, `directory_not_empty`, `file_exists`, `bad_option`,
`missing_operand`, `exit_nonzero`, `loop_limit`; cc: `cc_no_input`, `cc_bad_option`,
`cc_unknown_file_type`, `no_such_file`, `same_input_output`, `bad_object_file`; make:
`make_no_makefile`, `make_no_rule`, `make_missing_separator`, `make_recipe_failed`,
`make_circular`, `make_nothing_to_do`. A line's diagnostics travel in its
`CommandFinished` event, so a front end or a grader sees exactly what the user saw.

## 7. For developers

- `Shell.execute(line)` is a generator: it yields `ShellOutput`, `ClearScreen`,
  `RunProgram` (answer with the exit code, or `None` to let `program_runner` run it)
  and finally one `CommandFinished(line, exit_code, diagnostics, build, runs,
  constructs)`.
- `shell/headless.py::make_headless_shell(vfs, toolchain, device_factory=...)` is a
  complete terminal for the CLI, tests and automation: real `cc`/`make`, programs run
  by the interpreter, device calls answered by the device the factory returns.
- Shell construct keys (counted in `CommandFinished.constructs`): `pipeline`, `redirection`, `cc_compile`, `link`,
  `make_rebuild`, `script_run`, `exit_code_test`.
