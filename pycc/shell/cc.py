"""``cc`` / ``gcc``: the compiler driver the user types in the terminal.

Real gcc is a *driver*: it reads the command line, runs the compiler proper on every
``.c`` file, and hands the resulting object files to the linker. This module does the
same with the simulated toolchain:

* ``cc -c util.c``            compiles one file into ``util.o`` (in the current folder);
* ``cc main.c util.o -o bot`` compiles ``main.c`` and links it with an existing object;
* ``cc -E main.c``             prints the preprocessed text;
* every diagnostic is printed exactly like gcc (``main.c:4:5: error: ...``), followed
  on the terminal by a dimmed, indented plain-English line.

Only this module calls ``compile_source`` and ``link``. What happened is stored in
``shell.last_build`` (a ``BuildResult``) and in the line's diagnostics, so an editor
gutter or a grader sees the same build the user saw.
"""
from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass, field

from pycc.core.concepts import CONCEPT_TITLES, Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, concept_for_code
from pycc.interp.compiler import BuildResult, CompileResult, WERROR_SUFFIX, compile_source
from pycc.interp.includes import IncludeResolverImpl
from pycc.interp.linker import BadObjectFile, Executable, LinkResult, ObjectFile, link, object_name_for
from pycc.interp.preprocessor import preprocess
from pycc.interp.source import PPLine
from pycc.interp.toolchain import CompileOptions, Toolchain
from pycc.shell import constructs
from pycc.shell.ansi import DIM, RESET
from pycc.shell.commands import CommandContext, is_terminal
from pycc.vfs.errors import VfsError

SOURCE_SUFFIX = ".c"
OBJECT_SUFFIX = ".o"
DEFAULT_OUTPUT = "a.out"

# Flags gcc understands that change nothing in the simulation: debug info,
# optimisation, the language standard and libraries (the whole C library is built in).
_IGNORED_PREFIXES = ("-g", "-O", "-std=", "-l", "-L", "-pedantic", "-march=", "-f", "-W")
_IGNORED_EXACT = ("-ansi", "-w", "-pipe", "-s", "-v")
# Options whose value is attached (-Iinc) or the next argument (-I inc).
_VALUED = ("-o", "-I", "-D", "-U")
_MACRO_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class CcUsageError(Exception):
    """A command line cc cannot work with (an unknown flag, a missing value...)."""

    def __init__(self, code: str, message: str, friendly: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.friendly = friendly


@dataclass
class CcRequest:
    """What one cc command line asks for, after option parsing."""

    program: str
    inputs: list[str] = field(default_factory=list)       # in command-line order
    output: str | None = None
    compile_only: bool = False                             # -c
    preprocess_only: bool = False                          # -E
    options: CompileOptions = field(default_factory=CompileOptions)

    @property
    def sources(self) -> list[str]:
        """The ``.c`` inputs."""
        return [name for name in self.inputs if name.endswith(SOURCE_SUFFIX)]


# =========================================================================== parsing

def parse_cc_args(program: str, args: list[str],
                  defaults: CompileOptions | None = None) -> CcRequest:
    """Turn gcc-style arguments into a ``CcRequest``; raises ``CcUsageError``.

    Values may be attached or separate, as with gcc: ``-Iinclude`` and ``-I include``,
    ``-DDEBUG=1`` and ``-D DEBUG=1``, ``-oprog`` and ``-o prog``. ``-D`` and ``-U``
    apply left to right, so ``-DX -UX`` leaves ``X`` undefined.
    """
    request = CcRequest(program, options=defaults.copy() if defaults else CompileOptions())
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if not arg.startswith("-") or arg == "-":
            request.inputs.append(arg)
            continue
        if arg[:2] in _VALUED:
            value, index = _option_value(program, arg, args, index)
            _apply_valued(request, arg[:2], value)
        elif not _apply_flag(request, arg):
            raise CcUsageError(
                "cc_bad_option", f"{program}: error: unrecognized command-line option '{arg}'",
                f"cc does not know the option '{arg}'. The usual ones are -Wall, -Wextra, "
                "-c, -o NAME, -I FOLDER and -D NAME. Type `man cc` for the full list.")
    return request


def _option_value(program: str, arg: str, args: list[str], index: int) -> tuple[str, int]:
    """The value of ``-o``/``-I``/``-D``/``-U``: attached (``-Iinc``) or the next one."""
    if len(arg) > 2:
        return arg[2:], index
    if index < len(args):
        return args[index], index + 1
    messages = {"-o": "missing filename after '-o'", "-I": "missing path after '-I'",
                "-D": "macro name missing after '-D'", "-U": "macro name missing after '-U'"}
    examples = {"-o": "cc main.c -o prog", "-I": "cc -I include main.c",
                "-D": "cc -D DEBUG main.c", "-U": "cc -U DEBUG main.c"}
    raise CcUsageError("cc_bad_option", f"{program}: error: {messages[arg]}",
                       f"'{arg}' needs a value right after it, for example "
                       f"`{examples[arg]}`.")


def _apply_valued(request: CcRequest, flag: str, value: str) -> None:
    """Store the value of one valued option."""
    if flag == "-o":
        request.output = value
    elif flag == "-I":
        request.options.include_dirs.append(value)
    else:
        name, has_value, text = value.partition("=")
        _check_macro_name(name, flag)
        if flag == "-U":
            request.options.defines.pop(name, None)
        else:
            # gcc defines a bare -DNAME as 1, so "#if NAME" is true.
            request.options.defines[name] = text if has_value else "1"


def _check_macro_name(name: str, flag: str) -> None:
    """``-D=3`` or ``-D 1X``: gcc refuses names that are not C identifiers."""
    if _MACRO_NAME.fullmatch(name):
        return
    raise CcUsageError(
        "cc_bad_option", "<command-line>: error: macro names must be identifiers",
        f"{flag} needs a macro name made of letters, digits and _ (not starting with a "
        f"digit), like `{flag}DEBUG` or `-DWIDTH=10`.")


def _apply_flag(request: CcRequest, arg: str) -> bool:
    """Apply a switch; False when cc does not know it."""
    switches = {"-c": "compile_only", "-E": "preprocess_only"}
    warnings = {"-Wall": "wall", "-Wextra": "wextra", "-Werror": "werror"}
    if arg in switches:
        setattr(request, switches[arg], True)
    elif arg in warnings:
        setattr(request.options, warnings[arg], True)
    elif arg in _IGNORED_EXACT or arg.startswith(_IGNORED_PREFIXES):
        pass
    else:
        return False
    return True


# =========================================================================== printing

def indented_friendly(text: str) -> str:
    """A friendly explanation as the dimmed, two-space-indented line under an error."""
    return f"{DIM}  {text}{RESET}\n"


def gcc_text(diag: Diagnostic) -> str:
    """The exact text gcc (or ld) prints for a diagnostic, without a final newline.

    Messages of the driver itself are printed as they are; linker messages carry no
    source position on the line (``ld: util.o: in function 'main': undefined reference
    to 'irrigate'``), and fatal errors say ``fatal error:`` like gcc instead of the
    bare severity name.
    """
    if diag.location is None and diag.phase is not Phase.LINK:
        # The driver's own messages are complete already ("cc: fatal error: ...").
        return "\n".join([diag.message] + diag.notes)
    if diag.phase is Phase.LINK:
        text = diag.message if diag.message.startswith("ld:") else f"ld: {diag.message}"
        return "\n".join([text] + [f"ld: note: {note}" for note in diag.notes])
    if diag.severity is Severity.FATAL:
        prefix = f"{diag.location}: " if diag.location else ""
        text = f"{prefix}fatal error: {diag.message}"
        return "\n".join([text] + [f"{prefix}note: {note}" for note in diag.notes])
    return diag.format_gcc()


def concept_line(diag: Diagnostic) -> str:
    """The dim ``Topic: ...`` line naming what an error is about, like the trap card."""
    concept = getattr(diag, "concept", None)
    if not isinstance(concept, Concept):
        return ""
    return indented_friendly("Topic: " + CONCEPT_TITLES.get(concept, concept.value.replace("_", " ")))


def print_diagnostic(ctx: CommandContext, diag: Diagnostic) -> None:
    """Print one diagnostic on stderr; the friendly lines only reach a terminal, so
    ``cc main.c 2> errors.txt`` stores exactly what gcc would have written."""
    ctx.stderr.write(gcc_text(diag) + "\n")
    if diag.friendly and is_terminal(ctx.stderr):
        ctx.stderr.write(indented_friendly(diag.friendly))
        ctx.stderr.write(concept_line(diag))


def _record(ctx: CommandContext, diagnostics: list[Diagnostic]) -> None:
    """Print diagnostics and add them to the line's record (for an editor gutter or a grader)."""
    for diag in diagnostics:
        print_diagnostic(ctx, diag)
    ctx.shell.last_diagnostics.extend(diagnostics)


def driver_error(code: str, message: str, friendly: str,
                 severity: Severity = Severity.ERROR) -> Diagnostic:
    """A diagnostic from the driver itself: no file position, phase of the compiler."""
    return Diagnostic(severity, code, message, friendly, None,
                      concept_for_code(code, Concept.BUILD_CC), Phase.PREPROCESS)


# =========================================================================== the command

class CcRun:
    """One cc invocation: checks the inputs, compiles, links and reports."""

    def __init__(self, ctx: CommandContext, request: CcRequest, toolchain: Toolchain,
                 command_line: str) -> None:
        """Prepare a run of ``request`` in the shell's working directory."""
        self.ctx = ctx
        self.request = request
        self.toolchain = toolchain
        self.command_line = command_line
        self.diagnostics: list[Diagnostic] = []
        self.failed = False

    # --- helpers -------------------------------------------------------------------
    @property
    def program(self) -> str:
        """``cc`` or ``gcc``, whichever the user typed."""
        return self.request.program

    def report(self, diags: list[Diagnostic]) -> None:
        """Print and remember diagnostics; any error marks the run as failed."""
        _record(self.ctx, diags)
        self.diagnostics.extend(diags)
        if any(diag.is_error for diag in diags):
            self.failed = True

    def say(self, line: str) -> None:
        """A plain trailer line on stderr (``compilation terminated.``)."""
        self.ctx.stderr.write(line + "\n")

    def fatal(self, code: str, message: str, friendly: str) -> int:
        """Report a fatal driver error with gcc's trailer and give up (status 1)."""
        self.report([driver_error(code, message, friendly, Severity.FATAL)])
        self.say("compilation terminated.")
        return 1

    def resolve(self, name: str) -> str:
        """Absolute path of a name on the command line."""
        return self.ctx.resolve(name)

    # --- the steps ---------------------------------------------------------------------
    def run(self) -> int:
        """Do everything the command line asks for; returns the exit status."""
        status = self._check_command_line()
        if status is not None:
            return status
        if self.request.preprocess_only:
            return self._preprocess_all()
        results = [self._compile(name) for name in self.request.sources]
        if self.request.compile_only:
            return self._finish_objects(results)
        return self._link(results)

    def _check_command_line(self) -> int | None:
        """The checks gcc makes before compiling anything (None when all is fine)."""
        request = self.request
        if not request.inputs:
            return self.fatal("cc_no_input", f"{self.program}: fatal error: no input files",
                              "Tell cc which .c files to compile, for example: "
                              f"{self.program} main.c -o main")
        status = self._check_output()
        if status is not None:
            return status
        self._check_inputs()
        if self.failed:
            if not any(self.ctx.vfs.is_file(self.resolve(n)) for n in request.inputs):
                return self.fatal("cc_no_input",
                                  f"{self.program}: fatal error: no input files",
                                  "None of the files on the command line exists. Run `ls` "
                                  "to see the names in this folder.")
            return 1
        return None

    def _check_output(self) -> int | None:
        """``-o`` must not overwrite an input, nor name one file for several objects."""
        request = self.request
        if request.output is None:
            return None
        target = self.resolve(request.output)
        for name in request.inputs:
            if self.resolve(name) == target:
                return self.fatal(
                    "same_input_output",
                    f"{self.program}: fatal error: input file '{name}' is the same as "
                    "output file",
                    f"-o names the file cc writes. '{name}' is also an input, so it would "
                    "be overwritten by the program. Pick a new name, like "
                    f"`{self.program} {name} -o {_program_name(name)}`.")
        one_file_each = request.compile_only or request.preprocess_only
        if one_file_each and len(request.sources) > 1:
            return self.fatal(
                "cc_bad_option",
                f"{self.program}: fatal error: cannot specify '-o' with '-c', '-S' or '-E' "
                "with multiple files",
                "With -c every .c file becomes its own .o file, so one -o name cannot fit "
                "them all. Leave out -o, or compile the files one at a time.")
        return None

    def _check_inputs(self) -> None:
        """Every input must exist and be a .c or .o file."""
        for name in self.request.inputs:
            path = self.resolve(name)
            if self.ctx.vfs.is_dir(path):
                self.report([driver_error(
                    "is_a_directory", f"{self.program}: error: {name}: Is a directory",
                    f"'{name}' is a folder. Give cc the .c files inside it, for example "
                    f"`{self.program} {name}/main.c`.")])
            elif not self.ctx.vfs.is_file(path):
                self.report([driver_error(
                    "no_such_file", f"{self.program}: error: {name}: No such file or directory",
                    f"There is no file called {name} here. Check the name with `ls`.")])
            elif not name.endswith((SOURCE_SUFFIX, OBJECT_SUFFIX)):
                self.report([driver_error(
                    "cc_unknown_file_type",
                    f"{self.program}: error: {name}: file not recognized: file format not "
                    "recognized",
                    _unknown_type_friendly(name))])

    def _compile(self, name: str) -> CompileResult:
        """Compile one .c file and print what the compiler said."""
        result = compile_source(self.ctx.vfs, name, self.toolchain, self.request.options,
                                self.ctx.cwd)
        self.report(result.diagnostics)
        if any(diag.severity is Severity.FATAL for diag in result.diagnostics):
            self.say("compilation terminated.")
        if any(diag.message.endswith(WERROR_SUFFIX) for diag in result.diagnostics):
            self.say("cc1: all warnings being treated as errors")
        if result.ok:
            self.ctx.shell.count_construct(constructs.CC_COMPILE)
        return result

    def _finish_objects(self, results: list[CompileResult]) -> int:
        """``-c``: write ``x.o`` (in the current folder, or the ``-o`` name)."""
        objects = [result.object for result in results if result.object is not None]
        for obj in objects:
            target = self.request.output or object_name_for(obj.unit_name)
            self._write_output(target, obj.to_bytes(), executable=False)
        for name in self.request.inputs:
            if name.endswith(OBJECT_SUFFIX):
                self.ctx.stderr.write(f"{self.program}: warning: {name}: linker input file "
                                      "unused because linking not done\n")
        self._record_build(objects, None, "")
        return 1 if self.failed else 0

    def _link(self, results: list[CompileResult]) -> int:
        """Link the compiled units and the given .o files into the program."""
        compiled = {result.unit_name: result.object for result in results}
        objects: list[ObjectFile] = []
        for name in self.request.inputs:
            obj = compiled.get(name) if name.endswith(SOURCE_SUFFIX) else self._load_object(name)
            if obj is not None:
                objects.append(obj)
        output = self.request.output or DEFAULT_OUTPUT
        compiled_ok = all(result.ok for result in results)
        if self.failed:
            if compiled_ok:
                # Only an unreadable .o stopped the build: that is the linker's failure.
                self.say("collect2: error: ld returned 1 exit status")
            self._record_build(objects, None, output)
            return 1
        linked = link(objects, self.toolchain.builtins, known_sources=self._known_sources())
        self.report(linked.diagnostics)
        executable = Executable(objects) if linked.ok else None
        if executable is not None:
            self._write_output(output, executable.to_bytes(), executable=True)
        if linked.ok and not self.failed:
            self.ctx.shell.count_construct(constructs.LINK)
        else:
            self.say("collect2: error: ld returned 1 exit status")
        self._record_build(objects, linked, output, executable if not self.failed else None)
        return 1 if self.failed else 0

    def _load_object(self, name: str) -> ObjectFile | None:
        """Read a ``.o`` file given on the command line (``bad_object_file`` if not one)."""
        try:
            return ObjectFile.from_bytes(self.ctx.vfs.read_file(self.resolve(name)),
                                         self.toolchain)
        except BadObjectFile as error:
            self.report([error.diagnostic(name)])
        except VfsError as error:
            self.report([driver_error("no_such_file", f"{self.program}: error: {error}",
                                      "cc could not read this object file.")])
        return None

    def _known_sources(self) -> list[str]:
        """The other .c/.o files of the project folder, for better "undefined" hints."""
        given = set(self.request.inputs)
        try:
            names = self.ctx.vfs.listdir(self.ctx.cwd)
        except VfsError:
            return []
        return [name for name in names
                if name.endswith((SOURCE_SUFFIX, OBJECT_SUFFIX)) and name not in given]

    def _write_output(self, name: str, data: bytes, executable: bool) -> None:
        """Store the object file or program; a failure is reported like ld does."""
        path = self.resolve(name)
        try:
            self.ctx.vfs.write_file(path, data)
            if executable:
                self.ctx.vfs.set_executable(path)
        except VfsError as error:
            tool = "ld" if executable else self.program
            self.report([Diagnostic(
                Severity.FATAL, "cc_bad_option",
                f"{tool}: cannot open output file {name}: {error.reason}",
                f"cc could not write '{name}' ({error.reason.lower()}). Choose another "
                "name with -o.",
                None, Concept.BUILD_CC, Phase.LINK if executable else Phase.PREPROCESS)])

    def _record_build(self, objects: list[ObjectFile], linked: LinkResult | None,
                      output: str, executable: Executable | None = None) -> None:
        """Hand the ``BuildResult`` to the shell (``shell.last_build``)."""
        ok = not self.failed and (linked is None or linked.ok)
        build = BuildResult(ok, executable, objects, list(self.diagnostics),
                            [self.command_line], self.ctx.cwd, output, linked)
        self.ctx.shell.record_build(build)

    # --- -E ------------------------------------------------------------------------------
    def _preprocess_all(self) -> int:
        """``-E``: print the preprocessed text of every source (or write it to ``-o``)."""
        texts = [self._preprocess(name) for name in self.request.sources]
        text = "".join(texts)
        if self.request.output is not None:
            try:
                self.ctx.vfs.write_file(self.resolve(self.request.output), text)
            except VfsError as error:
                return self.fatal("cc_bad_option",
                                  f"{self.program}: fatal error: opening output file "
                                  f"{self.request.output}: {error.reason}",
                                  "cc could not write the output file.")
        else:
            self.ctx.out(text)
        return 1 if self.failed else 0

    def _preprocess(self, name: str) -> str:
        """The preprocessed text of one file, with gcc's ``# line "file"`` markers."""
        options = self.request.options
        resolver = IncludeResolverImpl(self.ctx.vfs, self.ctx.cwd, options.include_dirs,
                                       self.toolchain.headers)
        result = preprocess(self.ctx.vfs.read_text(self.resolve(name)), name, resolver,
                            dict(options.defines))
        self.report(result.diagnostics)
        if any(diag.severity is Severity.FATAL for diag in result.diagnostics):
            # gcc -E stops that file at a missing header and says so, like a compile.
            self.say("compilation terminated.")
        return render_preprocessed(result.lines)


def render_preprocessed(lines: list[PPLine]) -> str:
    """Preprocessed lines as ``gcc -E`` prints them.

    A ``# 12 "util.h"`` marker is written whenever the text jumps to another file or
    skips lines, so a reader can always tell where each line came from.
    """
    out: list[str] = []
    previous: tuple[str, int] | None = None
    for line in lines:
        if previous is None or previous != (line.file, line.line - 1):
            out.append(f'# {line.line} "{line.file}"')
        out.append(line.text)
        previous = (line.file, line.line)
    return "\n".join(out) + ("\n" if out else "")


def _program_name(source: str) -> str:
    """``src/main.c`` -> ``main``: a sensible -o name to suggest."""
    return posixpath.splitext(posixpath.basename(source))[0] or "program"


def _unknown_type_friendly(name: str) -> str:
    """Why cc refuses a file that is neither C source nor an object file."""
    if name.endswith(".h"):
        return (f"{name} is a header: it is not compiled on its own but copied into the .c "
                "files that #include it. Leave it off the cc command line.")
    return (f"cc compiles .c files and links .o files. {name} is neither, so cc does not "
            "know what to do with it.")


def cmd_cc(ctx: CommandContext, args: list[str]) -> int:
    """The ``cc``/``gcc`` command (registered by ``shell.toolchain_commands``)."""
    toolchain = getattr(ctx.shell, "toolchain", None)
    if toolchain is None:
        return ctx.fail("command_not_found",
                        f"{ctx.name}: the compiler is not installed in this workshop yet",
                        "The workshop's compiler arrives later in the story.", 127,
                        Concept.BUILD_CC)
    try:
        request = parse_cc_args(ctx.name, args, toolchain.default_options)
    except CcUsageError as error:
        diag = driver_error(error.code, error.message, error.friendly)
        _record(ctx, [diag])
        return 1
    command_line = shlex.join([ctx.name] + args)
    return CcRun(ctx, request, toolchain, command_line).run()


__all__ = ["CcRequest", "CcRun", "CcUsageError", "cmd_cc", "driver_error", "gcc_text",
           "indented_friendly", "parse_cc_args", "print_diagnostic", "render_preprocessed"]
