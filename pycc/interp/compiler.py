"""The compile driver: what ``cc`` does with one file, and with a whole project.

One translation unit goes through this pipeline:

    text --preprocess--> PPLine[] --lex--> Token[] --parse--> AST --check--> ObjectFile

Every stage reports gcc-style diagnostics; any error (or fatal error) means no object
file, exactly like gcc. ``build_project`` compiles several sources and links them into
an ``Executable``, which is what ``cc a.c b.c -o prog`` produces.

The checker lives in ``interp.checker``; ``load_checker`` is the single place that
finds it, so tools and tests can swap in another checker with the same signature.
"""
from __future__ import annotations

import shlex
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation, concept_for_code
from pycc.interp import astnodes as ast
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.header_hints import point_at_headers
from pycc.interp.hostio import DictSourceFiles, SourceFiles, join_path
from pycc.interp.includes import IncludeResolverImpl
from pycc.interp.lexer import tokenize
from pycc.interp.linker import (
    BadExecutable, Executable, LinkResult, ObjectFile, link, object_name_for,
)
from pycc.interp.parser import parse
from pycc.interp.preprocessor import PreprocessResult, preprocess
from pycc.interp.source import PPLine
from pycc.interp.symbols import UnitSymbols
from pycc.interp.toolchain import CompileOptions, Toolchain

# check(unit, registry, options) -> CheckResult(diagnostics, symbols)  
CheckFunction = Callable[[ast.TranslationUnit, BuiltinRegistry, CompileOptions], Any]

WERROR_SUFFIX = " [-Werror]"

# The note D3 attaches to every link failure of an existing executable: it linked when
# it was built, so a failure now means this machine offers other functions.
ANOTHER_MACHINE_NOTE = "this program was built by a toolchain with other device functions"


def load_checker() -> CheckFunction:
    """The checker the driver runs after parsing (``interp.checker.check``)."""
    from pycc.interp.checker import check  # imported late: the checker is the heaviest stage

    return check


# =========================================================================== results

@dataclass
class FrontEndResult:
    """Lex + parse + check of already preprocessed lines."""

    ast: ast.TranslationUnit | None
    symbols: UnitSymbols | None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing stops compilation (warnings are fine)."""
        return self.symbols is not None and not any(d.is_error for d in self.diagnostics)


@dataclass
class CompileResult:
    """The outcome of compiling one source file.

    Diagnostic paths are as given to cc, relative to ``cwd``.
    """

    ok: bool
    object: ObjectFile | None
    diagnostics: list[Diagnostic]
    pp_result: PreprocessResult | None
    cwd: str
    unit_name: str


@dataclass
class BuildResult:
    """The outcome of compiling and linking a project into one executable."""

    ok: bool
    executable: Executable | None
    objects: list[ObjectFile]
    diagnostics: list[Diagnostic]
    commands: list[str]
    cwd: str
    output_path: str = ""
    link_result: LinkResult | None = None


# =========================================================================== one unit

def run_front_end(lines: Sequence[PPLine], unit_name: str, builtins: BuiltinRegistry,
                  options: CompileOptions) -> FrontEndResult:
    """Lex, parse and (when those succeeded) check preprocessed lines.

    The checker only runs on a cleanly parsed unit: a partial AST after a syntax error
    makes the checker invent follow-up errors that would bury the real mistake.
    """
    diagnostics: list[Diagnostic] = []
    lexed = tokenize(list(lines), diagnostics)
    tokens = getattr(lexed, "tokens", lexed)          # LexResult or a plain list
    diagnostics.extend(getattr(lexed, "diagnostics", []))
    parsed = parse(tokens, unit_name)
    unit = getattr(parsed, "unit", None) or parsed.ast
    diagnostics = _in_source_order(diagnostics + list(parsed.diagnostics), lines)
    if any(diag.is_error for diag in diagnostics):
        return FrontEndResult(unit, None, diagnostics)
    checked = load_checker()(unit, builtins, options)
    return FrontEndResult(unit, checked.symbols, diagnostics + list(checked.diagnostics))


def compile_source(files: SourceFiles, path: str, toolchain: Toolchain,
                   options: CompileOptions | None = None, cwd: str = "/") -> CompileResult:
    """Compile one source file (``cc -c path``) into an ObjectFile."""
    opts = options.copy() if options is not None else toolchain.default_options.copy()
    full_path = join_path(cwd, path)
    text = _read_source(files, full_path)
    if text is None:
        diag = _driver_diagnostic(
            "no_such_file", f"cc: error: {path}: No such file or directory",
            f"There is no file called {path} in this folder. Check the name with 'ls'.")
        return CompileResult(False, None, [diag], None, cwd, path)
    resolver = IncludeResolverImpl(files, cwd, opts.include_dirs, toolchain.headers)
    pp_result = preprocess(text, path, resolver, dict(opts.defines))
    diagnostics = list(pp_result.diagnostics)
    if any(diag.severity is Severity.FATAL for diag in diagnostics):
        # A missing header or #error stops this unit, like gcc's "compilation terminated."
        return _compile_result(_apply_werror(diagnostics, opts), None, pp_result, cwd, path)
    front = run_front_end(pp_result.lines, path, toolchain.builtins, opts)
    # 'EOF' undeclared is a forgotten #include <stdio.h>, not a missing variable.
    point_at_headers(front.diagnostics, toolchain.headers, pp_result.includes,
                     toolchain.builtins)
    diagnostics = _apply_werror(diagnostics + front.diagnostics, opts)
    obj = None
    if front.ok and not any(diag.is_error for diag in diagnostics):
        obj = ObjectFile(path, full_path, front.ast, front.symbols, list(pp_result.lines),
                         opts, cwd)
    return _compile_result(diagnostics, obj, pp_result, cwd, path)


def compile_text(text: str, filename: str, toolchain: Toolchain,
                 options: CompileOptions | None = None) -> CompileResult:
    """Compile a string as if it were the file ``filename`` in ``/`` (tests and tools)."""
    files = DictSourceFiles({join_path("/", filename): text})
    return compile_source(files, filename, toolchain, options, cwd="/")


# =========================================================================== projects

def build_project(files: SourceFiles, sources: Iterable[str], output_path: str,
                  toolchain: Toolchain, options: CompileOptions | None = None,
                  cwd: str = "/", known_sources: Iterable[str] = ()) -> BuildResult:
    """Compile every source and link them: ``cc -Wall a.c b.c -o output_path``.

    Nothing is written: the caller stores ``executable.to_bytes()`` at ``output_path``.
    ``known_sources`` (other .c/.o files in the project) improve the "undefined
    reference" hints.
    """
    opts = options.copy() if options is not None else toolchain.default_options.copy()
    paths = list(sources)
    commands = [format_cc_command(paths, output_path, opts)]
    results = [compile_source(files, path, toolchain, opts, cwd) for path in paths]
    diagnostics = [diag for result in results for diag in result.diagnostics]
    objects = [result.object for result in results if result.object is not None]
    if not paths:
        diagnostics.append(_driver_diagnostic(
            "cc_no_input", "cc: fatal error: no input files",
            "Tell cc which .c files to compile, for example: cc main.c -o main"))
    if not paths or len(objects) != len(results):
        return BuildResult(False, None, objects, diagnostics, commands, cwd, output_path)
    linked = link(objects, toolchain.builtins, known_sources=known_sources)
    diagnostics.extend(linked.diagnostics)
    executable = Executable(objects) if linked.ok else None
    return BuildResult(linked.ok, executable, objects, diagnostics, commands, cwd,
                       output_path, linked)


def load_executable(data: bytes) -> Executable:
    """Decode executable bytes (objects not yet loaded); raises BadExecutable."""
    return Executable.from_bytes(data)


def link_executable(executable: Executable, builtins: BuiltinRegistry,
                    program_name: str = "a.out") -> LinkResult:
    """Load every object of ``executable`` against ``builtins`` and link them.

    This runs before every program run; it fails only when the program was built for
    a toolchain with other device functions or the file is damaged.
    """
    try:
        objects = [obj.load(builtins) for obj in executable.objects]
    except Exception as error:  # noqa: BLE001 - any failure here means "cannot run this file"
        reason = getattr(error, "reason", str(error))
        return LinkResult(False, None, [BadExecutable(reason).diagnostic(program_name)])
    linked = link(objects, builtins)
    for diag in linked.diagnostics:
        _explain_other_machine(diag)
    return linked


def _explain_other_machine(diag: Diagnostic) -> None:
    """Reword a link failure of a built program: the code was fine when it was built.

    "Write the function" would be wrong advice here; the function exists in the
    toolchain the program was built with.
    """
    if ANOTHER_MACHINE_NOTE not in diag.notes:
        diag.notes.append(ANOTHER_MACHINE_NOTE)
    if diag.code == "undefined_reference":
        name = _quoted_name(diag.message)
        diag.friendly = (f"This program was built by a toolchain that has '{name}'. The "
                         "toolchain here does not, so the program cannot start. Rebuild it "
                         "here with cc, using only the functions this toolchain provides.")


def _quoted_name(message: str) -> str:
    """The last 'quoted' word of an ld message (the missing name)."""
    pieces = message.split("'")
    return pieces[-2] if len(pieces) >= 3 else message


def format_cc_command(sources: Sequence[str], output_path: str,
                      options: CompileOptions) -> str:
    """The cc command line a build corresponds to, as the terminal would show it."""
    words = ["cc"]
    words += [flag for flag, on in (("-Wall", options.wall), ("-Wextra", options.wextra),
                                    ("-Werror", options.werror)) if on]
    words += [f"-I{directory}" for directory in options.include_dirs]
    words += [f"-D{name}" if value in (None, "") else f"-D{name}={value}"
              for name, value in options.defines.items()]
    words += list(sources)
    if output_path:
        words += ["-o", output_path]
    return " ".join(shlex.quote(word) for word in words)


# =========================================================================== helpers

def _read_source(files: SourceFiles, full_path: str) -> str | None:
    """The text of a source file, or None if it cannot be read."""
    if not files.is_file(full_path):
        return None
    try:
        return files.read_text(full_path)
    except Exception:  # noqa: BLE001 - each SourceFiles implementation raises its own errors
        return None


def _compile_result(diagnostics: list[Diagnostic], obj: ObjectFile | None,
                    pp_result: PreprocessResult, cwd: str, path: str) -> CompileResult:
    ok = obj is not None and not any(diag.is_error for diag in diagnostics)
    return CompileResult(ok, obj if ok else None, diagnostics, pp_result, cwd, path)


def _apply_werror(diagnostics: list[Diagnostic], options: CompileOptions) -> list[Diagnostic]:
    """``-Werror``: every warning becomes an error and says why, like gcc.

    The checker may already have rewritten its own warnings, so this is idempotent.
    """
    if not options.werror:
        return diagnostics
    for diag in diagnostics:
        if diag.severity is Severity.WARNING:
            diag.severity = Severity.ERROR
            if not diag.message.endswith(WERROR_SUFFIX):
                diag.message += WERROR_SUFFIX
    return diagnostics


def _in_source_order(diagnostics: list[Diagnostic], lines: Sequence[PPLine]) -> list[Diagnostic]:
    """Sort lexer and parser diagnostics into the order of the preprocessed lines.

    Sorting by line number alone would mix up a header's line 3 with main.c's line 3.
    """
    position: dict[tuple[str, int], int] = {}
    for index, line in enumerate(lines):
        position.setdefault((line.file, line.line), index)

    def key(diag: Diagnostic) -> tuple[int, int, int]:
        loc: SourceLocation | None = diag.location
        if loc is None:
            return (1, 0, 0)
        return (0, position.get((loc.file, loc.line), len(lines)), loc.column)

    return sorted(diagnostics, key=key)


def _driver_diagnostic(code: str, message: str, friendly: str) -> Diagnostic:
    """A diagnostic from cc itself (no file position)."""
    return Diagnostic(Severity.FATAL, code, message, friendly, None,
                      concept_for_code(code), Phase.PREPROCESS)


__all__ = [
    "ANOTHER_MACHINE_NOTE", "BuildResult", "CompileResult", "FrontEndResult", "WERROR_SUFFIX",
    "build_project", "compile_source", "compile_text", "format_cc_command",
    "link_executable", "load_checker", "load_executable", "object_name_for",
    "run_front_end",
]
