# pycc

A C99 toolchain and an LP64 interpreter, written in Python, that catches memory errors
the moment they happen.

pycc preprocesses, parses, type-checks and links real C, then runs it in a simulated
64-bit machine whose type sizes, struct layout and alignment match gcc on x86-64 Linux.
Every load and store is checked, so instead of silent corruption you get a
sanitizer-style report that points at the line:

```
$ pycc run examples/use_after_free.c
use_after_free.c:15:5: error: heap-use-after-free: address 0x11010 is inside the block from malloc(16), which was already freed
  This memory was already given back with free(). After free the block belongs to the heap again, so the pointer must not be used. Set it to NULL after freeing so the mistake is easy to spot.
Segmentation fault (core dumped)
```

Every message is worded like gcc, ld, bash or AddressSanitizer, so what you learn
carries over to a real toolchain. Under it comes one plain-English line for beginners.

## What it detects

At run time (the program stops, with the exit status a real crash would give):

- null pointer dereference, and wild pointers that point at no object
- out-of-bounds reads and writes, naming the array and its length
- use after free, double free, and freeing a pointer malloc never returned
- stack overflow from runaway recursion
- writes to string literals, and strings missing their `'\0'`
- division by zero, and runaway loops (a statement budget)

As warnings (the program keeps going):

- reads of uninitialized memory
- memory leaks at exit (`16 bytes in 1 blocks are definitely lost`)
- signed overflow, oversized shifts, `printf` arguments that don't match the format,
  files never closed, unchecked `NULL` returns

At compile time: the usual gcc diagnostics, with `-Wall`, `-Wextra` and `-Werror`
groups (implicit declarations, format mismatches, unused variables, `if (x = 1)`,
comparing a string with `==`, missing returns, sign comparisons and more). The linker
reports undefined references, multiple definitions and `static` symbols used from
another file, as ld does.

## Quick start

pycc needs Python 3.12 and has no dependencies.

```bash
git clone https://github.com/phonehseng/pycc
cd pycc
python -m pycc run examples/hello.c -- one two
```

Or install it to get a `pycc` command:

```bash
pip install .
pycc run examples/off_by_one.c
```

| command | what it does |
|---|---|
| `pycc run main.c util.c [-Wall ...] [-- ARGS]` | compile, link and run |
| `pycc build main.c util.c -o prog` | compile and link; writes the program |
| `pycc exec prog [ARGS]` | run a program written by `build -o` |
| `pycc shell` | a Bash-like shell with `cc`, `make`, pipes and redirection |
| `pycc shell -c 'make && ./prog \| wc -l'` | run one shell line |

Programs run in an in-memory filesystem. pycc copies the current folder into it as
`~/project`. What a program writes with `fopen` stays in memory, so it never touches
your real files.

## What is in it

```
pycc/
  interp/        the C toolchain and runtime
    preprocessor.py  #include, #define (object- and function-like), #if/#ifdef, #pragma once
    lexer.py         tokens with exact line and column, for gcc-style carets
    parser.py        recursive descent: declarators, structs, unions, enums, typedefs,
                     function pointers, initializers
    checker.py       type checking, implicit conversions, constant folding, warnings
    compiler.py      one translation unit to an object file; whole projects
    linker.py        object files and executables, symbol resolution with gcc's rules
                     (-fno-common), static linkage, link-time diagnostics
    interpreter.py   a steppable interpreter: every statement and call is an event
    memory.py        byte-addressed memory with a shadow map, heap, stack and red zones
    c_types.py       LP64 sizes, alignment, struct layout, integer promotions
    stdlib/          stdio, stdlib, string, ctype, math, errno, assert, time + headers
    devices.py       let a host provide C functions of its own (see below)
  shell/         a small bash: parser, expansion, pipes, redirection, scripts,
                 about 45 built-in commands (ls, grep, sort, diff, find...), cc,
                 and a GNU-make-compatible make
  vfs/           the in-memory filesystem programs and the shell use
  core/          diagnostics, events and the concept taxonomy
  cli.py         the pycc command
docs/
  SUPPORTED_C.md      exactly which C is accepted, refused, and every diagnostic
  SUPPORTED_SHELL.md  the shell, its commands, cc and make
```

`cc -c` really produces separate object files, and `cc a.o b.o -o prog` really links
them. An object file holds the checked translation unit and its symbol table (the
`CVOBJ1` format); the interpreter then runs the linked, type-annotated AST. There is
no native code generation: the point is a precise, inspectable machine, not speed.

### Design notes

- **Deterministic.** Uninitialized memory is filled with seeded garbage, `rand()` is
  the C standard's sample generator, and the clock is simulated, so a program prints
  the same thing on every run and on every computer.
- **Generators all the way down.** The interpreter and the shell yield events
  (`Statement`, `CallEnter`, `Output`, `InputRequest`, `Trap`...) and the host resumes
  them. That makes a step debugger, a memory view or an animated front end a matter of
  handling events. The CLI just loops over them as fast as it can.
- **gcc as the reference.** Sizes, struct padding, `printf` formatting, integer
  conversions and diagnostic wording follow gcc. The integration tests check documented
  behaviour against outputs worked out from the C standard.

## Embedding: device functions

A host can give C programs functions of its own, for example to drive a simulated
robot or to stub out hardware in a course exercise:

```python
from pycc.interp.devices import register_device_function
from pycc.interp.stdlib import make_registry, system_headers
from pycc.interp.toolchain import Toolchain

registry = make_registry()
register_device_function(registry, "pulse", "int pulse(int amount);", header="counter.h")
headers = system_headers()
headers.add("counter.h", "int pulse(int amount);\n")
toolchain = Toolchain(registry, headers)
```

Each call to `pulse()` pauses the program with a `DeviceCall` event. The host performs
it (any object with a `perform(name, args)` method will do; see
`pycc.shell.headless.make_headless_shell(..., device_factory=...)`) and the program
continues with the value it gets back.

## Limits

pycc supports the C99 that beginners and most course projects use. Not supported:
`goto`, variable-length arrays, bit-fields, designated initializers, variadic function
definitions (calling `printf` is fine), `##` token pasting, `#` stringizing in user
macros, and threads. Each is refused with a clear error rather than silently
mis-compiled. A run is limited to one million statements, 512 nested calls and 1 MiB of
memory. See [docs/SUPPORTED_C.md](docs/SUPPORTED_C.md) for the full contract.

## Tests

```bash
pip install pytest
python -m pytest
```

About 2,900 tests: unit tests for every stage, and end-to-end scenarios that type
commands into a real shell, build multi-file projects with `make` and check the exact
terminal output.

## Origin

pycc started as the engine of a game that teaches C, and was pulled out into its own
project. The beginner-friendly second line under each error comes from there.

## License

MIT. See [LICENSE](LICENSE).
