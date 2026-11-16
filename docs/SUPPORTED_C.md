# The C that pycc runs

pycc compiles and runs real C: a program that works in pycc also builds with
`gcc -Wall -std=c99` on a real computer and prints the same output. This page lists
precisely what the compiler, linker and runtime accept, what they refuse, and every
diagnostic they print. It is written for two readers:

- **users**: what you can use, and what each error means;
- **developers**: the contracts behind it (module names in `code` style).

Everything below is checked by the end-to-end scenarios in `tests/integration/`.

---

## 1. The machine

The simulated machine is a 64-bit Linux box with gcc's sizes (LP64):

| type | size | notes |
|---|---|---|
| `char` | 1 | signed |
| `short` | 2 | |
| `int`, `enum` | 4 | enum constants are `int` |
| `long`, `long long` | 8 | |
| `float` | 4 | |
| `double` | 8 | `long double` is accepted and treated as `double` |
| pointers | 8 | `size_t` is `unsigned long`, `ptrdiff_t` is `long` |
| `_Bool` / `bool` | 1 | `bool`, `true`, `false` come from `<stdbool.h>` |

Structs use natural alignment, so `sizeof` gives the same numbers as gcc.

Limits of one run (each one stops the program with a clear message, never pycc itself):

| limit | value | what happens |
|---|---|---|
| statements per run | 1 000 000 | trap `instruction_budget`, status 137 "Killed" |
| nested calls | 512 | trap `stack_overflow`, status 139 |
| stack | 256 KiB | trap `stack_overflow` |
| whole memory | 1 MiB | `malloc` returns NULL when the heap is full |
| leak reports at exit | 10 | one `memory_leak` warning per block, at most 10 |

Addresses `0` to `4095` belong to nobody: any access there is a NULL dereference.

---

## 2. The language

### Supported

- **Types**: all integer types with `signed`/`unsigned`, `float`, `double`, `_Bool`,
  `void`, pointers (to anything, including pointers to pointers and to functions),
  arrays (also two-dimensional, also `int a[] = {...}` sized by its initializer),
  `struct`, `union`, `enum`, `typedef` (also of anonymous structs), `const`.
- **Declarations**: globals with initializers, `static` globals and functions (private
  to their file), `extern`, `static` local variables, block scoping, declarations
  anywhere in a block (C99), `for (int i = 0; ...)`.
- **Initializers**: scalars, brace lists for arrays and structs (nested), strings for
  `char` arrays; missing elements are zero.
- **Statements**: `if`/`else`, `while`, `do ... while`, `for`, `switch` with `case`,
  `default` and fall-through, `break`, `continue`, `return`, blocks, empty statements.
- **Expressions**: every C operator - arithmetic, `%`, shifts, bitwise, comparisons,
  `&&`/`||` with short-circuit, `!`, `?:`, comma, all compound assignments, pre/post
  `++`/`--`, `&` and `*`, `[]`, `.` and `->`, casts, `sizeof` (of a type and of an
  expression).
- **Arithmetic** follows C exactly: integer promotions and the usual arithmetic
  conversions, every integer operation wraps at its width (a signed overflow also
  warns `signed_overflow`), integer division truncates toward zero, pointer arithmetic
  is scaled by the pointee size.
- **Functions**: prototypes, recursion, structs passed and returned by value,
  function pointers (also as `qsort`/`bsearch` callbacks), `main(void)` and
  `main(int argc, char *argv[])` (or `char **argv`); falling off the end of `main`
  returns 0.
- **Literals**: decimal, octal (`017`), hex (`0x1F`) integers with `u`, `l`, `ul`,
  `ll`, `ull` suffixes; floating literals with exponents; character constants with
  the escapes `\n \t \r \a \b \f \v \\ \' \" \?`, octal (`\101`) and hex (`\x41`);
  string literals (adjacent ones are joined; they live in read-only memory).
- **`__func__`**: the name of the enclosing function, as a string.
- **Inline functions**: `inline` is accepted.

### Not supported (each gives a clear `unsupported_feature` or `array_size_not_const` error)

| feature | message | instead |
|---|---|---|
| `goto` and labels | `labels and 'goto' are not supported` | loops with `break`/`continue` |
| variable-length arrays `int a[n];` | `size of array 'a' is not an integer constant` | `malloc` |
| bit-fields `int x : 3;` | `bit-fields are not supported` | a whole `int` per field |
| designated initializers `{ [1] = 2 }`, `{ .x = 1 }` | `designated initializers are not supported` | values in order |
| compound literals `(struct P){1, 2}` | `compound literals are not supported` | a named variable |
| your own variadic functions `int f(int n, ...)` | `variadic function definitions are not supported` | an array and a count |
| `_Complex`, `_Imaginary` | `'_Complex' types are not supported` | two `double`s |

Also outside the subset: `<stdarg.h>`, `setjmp`/`longjmp`, signals, threads, `volatile`
and `restrict` semantics (both keywords are accepted and ignored), `long double`
precision, K&R-style parameter lists. The order in which function arguments are
evaluated is left to right (C leaves it unspecified; gcc often goes right to left).

---

## 3. The preprocessor

- `#include "x.h"` searches the folder of the including file, then each `-I` folder,
  then the system headers; `#include <x.h>` searches the system headers, then the `-I`
  folders. Includes nest up to 32 deep.
- `#define` object-like and function-like macros (arguments may span lines),
  `#undef`, `#ifdef`, `#ifndef`, `#if`/`#elif`/`#else`/`#endif` with integer constant
  expressions and `defined(X)`, `#pragma once`, `#error`, `#warning`, `#line`
  (accepted and ignored). Unknown pragmas are ignored, as gcc does.
- Built-in macros: `__FILE__`, `__LINE__`, `__DATE__` and `__TIME__` (fixed, so builds
  are reproducible) and `__PYCC__` (1).
- `assert(e)` from `<assert.h>` is rewritten into `__cv_assert((e), "e", __FILE__,
  __LINE__, __func__)` so the failure message quotes the condition as written.
  `#define NDEBUG` before the include turns asserts off.
- **Not supported**: the `#` (stringizing) and `##` (token pasting) operators and
  variadic macros (`...`, `__VA_ARGS__`); a `#define` using them is a fatal error that
  says so and stops the file, so no confusing follow-on errors appear. `#if`
  arithmetic is signed 64-bit only.
- Only the standard C library exists. A POSIX header (`unistd.h`, `getopt.h`,
  `strings.h`, `sys/types.h`, `sys/stat.h`, `fcntl.h`, `dirent.h`, `pthread.h`,
  `sys/time.h`) is gcc's `fatal error: unistd.h: No such file or directory`, and the
  friendly line says it is POSIX and what to use instead (for `getopt`: parse `argv`
  by hand).

---

## 4. Headers and library functions

The headers below are served by the toolchain (`interp/stdlib/headers/`); the same
texts are copied read-only into `/usr/include`, so `cat /usr/include/stdio.h` shows
exactly what the compiler reads.

### `<stdio.h>`
`FILE`, `stdin`, `stdout`, `stderr`, `EOF`, `NULL`, `BUFSIZ`, `size_t`.
`printf fprintf sprintf snprintf puts putchar fputs fputc putc scanf fscanf sscanf
fgets fgetc getc getchar ungetc fopen fclose fread fwrite feof ferror clearerr fflush
remove rename perror`.

- **printf conversions**: `%d %i %u %o %x %X %c %s %p %f %F %e %E %g %G %n %%`; flags
  `- + space 0 #`; width and precision as numbers or `*`; length modifiers `hh h l ll
  z j t L`. NULL for `%p` prints `(nil)`, as glibc does. A mismatch between the format
  and the arguments is a compile warning (`format_mismatch`, with `-Wall`) and a runtime
  warning (`format_argument_type`). A double printed with an integer conversion, or an
  integer printed with a floating one, prints a seeded garbage value (the same for the
  same seed), as the stale register would on a real machine: `printf("%d", 4.5)` never
  prints `4`. This includes `%c`, `%p` and `%s`: `printf("%c", 65.0)` never prints `A`,
  and `%s` with a double reads a garbage address and traps `wild_pointer`. A function
  pointer always shows its code address (`0x400000` + 16 per function), with or without
  a `(void *)` cast and in the variables panel. An int printed with `%p` or a pointer
  with `%x` keeps its value.
- **scanf conversions**: `%d %i %u %o %x %X %c %s %f %e %E %g %G %p %n` and scan sets
  `%[a-z]`, `%[^,]`; `*` (assign nothing), widths and the same length modifiers. A space
  in the format skips any white space. The return value counts the assignments made,
  or `EOF`.
- **fopen modes**: `r w a r+ w+ a+`, each optionally with `b`. Files are the
  project's virtual files: whatever `fprintf` writes is visible to `cat` at once
  (write-through), even if the program crashes later. A failing `fopen` returns NULL
  and sets `errno` (`ENOENT`, `EACCES`, `EISDIR`...), so `perror("plan.txt")` prints
  `plan.txt: No such file or directory`.
- **stdin**: every reader (`getchar`, `fgetc(stdin)`, `fgets`, `scanf`, `ungetc`) shares
  one buffer. Input comes from the terminal, a pipe (`echo 3 | ./prog`) or a file
  (`./prog < plan.txt`); an empty source is end of file.

### `<stdlib.h>`
`EXIT_SUCCESS`, `EXIT_FAILURE`, `RAND_MAX` (32767), `NULL`, `size_t`.
`malloc calloc realloc free atoi atol atof strtol strtod abs labs rand srand exit
abort qsort bsearch getenv system`.
`rand` is the C standard's sample generator (seed 1 until `srand`), so the numbers are
the same on every run. `getenv` reads the variables the shell exported. `system`
always returns -1: programs cannot start other programs in pycc.

### `<string.h>`
`strlen strcpy strncpy strcat strncat strdup strcmp strncmp strchr strrchr strstr
strspn strcspn strtok memset memcpy memmove memcmp strerror`.
Writing past the end of the destination traps `out_of_bounds` naming the buffer; a
string without its `'\0'` traps `string_not_terminated`.

### The others
- `<ctype.h>`: `isalpha isdigit isalnum isspace isupper islower ispunct isxdigit
  isprint toupper tolower`.
- `<math.h>`: `sqrt pow fabs floor ceil round sin cos tan atan2 fmod exp log`, `M_PI`,
  `M_E` (no `-lm` needed).
- `<errno.h>`: `errno`, `ENOENT EIO EBADF ENOMEM EACCES EEXIST ENOTDIR EISDIR EINVAL
  EDOM ERANGE`.
- `<assert.h>`: `assert` (see section 3).
- `<time.h>`: `time_t`, `clock_t`, `CLOCKS_PER_SEC`, `time` (the attached device's
  tick; 0 without a device), `clock` (statements run so far).
- `<stdbool.h>`, `<stddef.h>` (`size_t ptrdiff_t NULL offsetof`), `<stdint.h>` (`int8_t`
  ... `uint64_t`, `intptr_t`, `uintptr_t`, `intmax_t`, their limits and `SIZE_MAX`),
  `<limits.h>` (`CHAR_BIT`, `CHAR/SCHAR/UCHAR/SHRT/USHRT/INT/UINT/LONG/ULONG/LLONG/ULLONG
  _MIN/_MAX`).

Any other standard header (`<stdarg.h>`, `<signal.h>`, `<unistd.h>`...) is
`fatal error: x.h: No such file or directory`.

### Device headers
A program embedding pycc can add its own headers and *device functions*: C functions
that the host answers (a simulated robot, an LED panel, a test double). See
`pycc/interp/devices.py`: `register_device_function(registry, "pulse", "int
pulse(int amount);", header="counter.h")`. Each call pauses the program with a
`DeviceCall` event; the host performs it and sends back the value. Without a device
attached, a call traps `invalid_argument`.

---

## 5. Building: compile, link, run

- `cc -c util.c` writes `util.o`: the preprocessed source with its symbols, in the
  `CVOBJ1` format (`cat util.o` shows gibberish, `file util.o` says
  "relocatable object file"). No `.o` is written when there is an error.
- `cc main.o util.o -o prog` links: every function and global used must be defined
  exactly once (`-fno-common`: `int total;` in two files is two definitions). The
  result is a `CVEXE1` executable, marked runnable.
- `./prog` loads the executable and links it again against the current toolchain. A
  program built by a toolchain with device functions this one lacks cannot start
  (status 127, "this program was built by a toolchain with other device functions").
- Runtime crashes print the error like a sanitizer, `file:line:col: error: ...` plus
  the friendly line, then the shell adds bash's line (`Segmentation fault (core
  dumped)`); runtime warnings (`uninitialized_read`, `memory_leak`...) are
  printed the same way with `warning:` and the program carries on.

Exit statuses: the program's own `exit`/`return` value (`& 255`); 139 memory and
pointer traps ("Segmentation fault (core dumped)"); 134 `free`/`assert`/`abort`/
invalid device arguments ("Aborted (core dumped)"); 136 division by zero
("Floating point exception (core dumped)"); 137 instruction budget
("Killed"); 126 cannot execute; 127 built by another toolchain or command not found;
130 interrupted with Ctrl+C.

---

## 6. Diagnostics

Every diagnostic has a stable `snake_case` code (tools built on pycc can group and
count errors by it), a gcc/ld-style message, a plain-English friendly line and a concept. The
friendly line is printed under the message, indented two spaces and dimmed.

### Preprocessor
| code | example message |
|---|---|
| `include_not_found` (fatal) | `main.c:1:10: fatal error: util.h: No such file or directory` |
| `include_depth` | `#include nested depth 33 exceeds maximum of 32` |
| `unterminated_comment` | `unterminated comment` |
| `unterminated_if` | `unterminated #ifndef` |
| `else_without_if`, `endif_without_if` | `#else without #if`, `#endif without #if` |
| `macro_args_mismatch` | `macro "assert" passed 2 arguments, but takes just 1` |
| `macro_redefined` (warning) | `"N" redefined` (+ the note with the first definition) |
| `error_directive` (fatal) | `#error ...` |
| `warning_directive` (warning) | `#warning ...` |
| `invalid_directive` | `the '#' operator is not supported`, `invalid preprocessing directive #inclde` |

### Lexer and parser
`unterminated_string`, `unterminated_char`, `empty_char_constant`, `multichar_constant`
(warning), `invalid_suffix`, `invalid_number`, `invalid_escape`, `stray_character`,
`missing_semicolon` (`expected ',' or ';' before 'printf'`), `expected_token`,
`unexpected_token` (`expected expression before ';' token`), `unknown_type` (`unknown
type name 'itn'; did you mean 'int'?`), `unbalanced_brace`, `unbalanced_paren`,
`else_without_if_stmt`, `bad_declarator`, `bad_initializer`, `empty_struct`,
`nested_function` (`ISO C forbids nested functions`), `invalid_type_combination`,
`case_not_constant`, `enum_value_not_const`, `enum_overflow`, `array_size_not_const`,
`too_many_initializers`, `nesting_too_deep`, `too_many_errors`, `unsupported_feature`.

### Checker: errors
| code | example message |
|---|---|
| `undeclared_identifier` | `'cuont' undeclared (first use in this function); did you mean 'count'?` |
| `implicit_declaration` | `implicit declaration of function 'helper'` |
| `conflicting_types` | `conflicting types for 'move'` |
| `redefinition` | `redefinition of 'struct Plan'` (+ `note: previous definition of 'struct Plan' was here`) |
| `redefinition` (enumerators) | `redeclaration of enumerator 'BEAN'` (two enums of one scope share a constant) |
| `too_few_arguments`, `too_many_arguments` | `too few arguments to function 'scan_at'` |
| `not_a_struct`, `no_member` | `'struct Tile' has no member named 'crop'` |
| `invalid_subscript` | `subscripted value is neither array nor pointer nor vector` |
| `lvalue_required` | `lvalue required as left operand of assignment` |
| `assign_to_array`, `assign_to_const` | `assignment to expression with array type` |
| `void_value` | `void value not ignored as it ought to be` |
| `return_value_in_void`, `return_missing_value` | `'return' with no value, in function returning non-void` |
| `incompatible_types` | `incompatible types when assigning to type 'int' from type 'struct Plan'` |
| `call_non_function` | `called object 'x' is not a function or function pointer` |
| `break_outside_loop`, `case_outside_switch`, `duplicate_case` | `break statement not within loop or switch` |
| `invalid_operands`, `incomplete_type` | `invalid operands to binary + (have 'struct P' and 'int')` |

When the name comes from a header you did not include, the friendly line says which
one: `'EOF' is not a variable of yours: it is defined in <stdio.h>. Did you forget
#include <stdio.h>? Add it at the top of the file.`

### Checker: warnings
| group | codes |
|---|---|
| always on | `implicit_declaration` (an error), `incompatible_pointer`, `int_from_pointer`, `pointer_from_int`, `integer_overflow_constant`, `string_literal_to_char_array`, `address_compare` (`comparison with string literal results in unspecified behavior` for `==` or `!=` with a string literal; gcc files it under `-Waddress`; the friendly line says "Use strcmp to compare text") |
| `-Wall` | `unused_variable`, `control_reaches_end`, `assignment_as_condition`, `uninitialized_use`, `array_bounds`, `format_mismatch`, `unused_value`, `pointer_int_comparison`, `implicit_int`, `missing_return_type` |
| `-Wextra` | `sign_compare`, `missing_prototype` (gcc files this one under `-Wmissing-prototypes`) |

`-Werror` turns every enabled warning into an error and appends ` [-Werror]`.

### Linker (`ld`)
| code | example message |
|---|---|
| `undefined_reference` | `ld: main.o: in function 'main': undefined reference to 'helper'` then `collect2: error: ld returned 1 exit status` |
| `multiple_definition` | `ld: count.o: multiple definition of 'total'; main.o: first defined here` |
| `no_main` | `ld: crt1.o: in function '_start': undefined reference to 'main'` |
| `static_not_visible`, `bad_object_file`, `bad_executable` | a `static` name used from another file; a damaged `.o` or program |

The friendly line tells the three undefined-reference cases apart: not written
anywhere; written in a file that was not linked ("Did you pass helper.o to the
linker?"); `static` in another file. A multiple definition that comes from a header
explains that include guards do not help and that the header should hold an `extern`
declaration.

### Runtime traps (the program stops)
`null_dereference`, `out_of_bounds` (`index 8 is past the end of 'field' (8 elements)`),
`wild_pointer`, `use_after_free`, `double_free`, `invalid_free`, `write_to_literal`,
`string_not_terminated`, `stack_overflow`, `division_by_zero`, `instruction_budget`,
`assert_failed` (`./prog: prog.c:5: share: Assertion `baskets > 0' failed.`),
`abort_called`, `bad_function_pointer`, `heap_exhausted`, `invalid_stream`, and
`invalid_argument` (a device call with no device, or one the device refused).

### Runtime warnings (the program goes on)
`uninitialized_read`, `memory_leak` (`16 bytes in 1 blocks are definitely lost:
malloc(16) was never freed`), `signed_overflow`, `shift_overflow`,
`format_argument_type`, `file_not_closed`, `fopen_failed`, `unchecked_null`, and any
warnings a device sends back. Each is reported at most once per place per run.

---

## 7. For developers

- Pipeline: `interp/preprocessor.py` -> `interp/lexer.py` -> `interp/parser.py` ->
  `interp/checker.py` -> `interp/linker.py` (`ObjectFile`, `Executable`, `link`) ->
  `interp/interpreter.py`. `interp/compiler.py` drives one unit (`compile_source`) or a
  project (`build_project`); `shell/cc.py` is its only caller in the shell.
- `interp/header_hints.py` rewrites the friendly text of `undeclared_identifier` and
  `implicit_declaration` when a system header defines the name.
- The checker annotates the AST; the interpreter runs only checked trees and never
  re-derives types.
- Device functions are registered with `interp/devices.py::register_device_function`;
  each call yields one `DeviceCall`, which the host answers with a `HostActionReply`
  (value, warnings, trap). `RunSummary.constructs` counts one `device_call` per device
  call and one `function_call` per call of a user function (`main` itself is not
  counted).
- Hosts: `shell/headless.py::run_program_headless` (the CLI, the tests and the
  headless terminal) prints traps and runtime warnings on the program's stderr in the
  format above; a stepping debugger can drive `Interpreter.run` itself and receives the
  same events.
- `interp/c_types.py::format_value(memory, addr, ctype, max_items=16)` renders a value
  for a debugger: `0x..` pointers (`NULL` for zero), `'c'` chars, `"text"` for a
  terminated char array, `{..}` arrays and structs cut after `max_items` entries with
  `...`, `ENUM_NAME (n)` enums and `?` for any uninitialised byte.
