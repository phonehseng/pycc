"""The semantic checker: what the program *means*, before it runs.

The parser knows the shape of the program and the type of every declaration. The
checker walks the tree once more and answers everything else a C compiler decides
statically:

* what each name refers to (a local, a parameter, a global, a library function, an
  enum constant...): written onto ``Identifier.ref`` as a ``SymbolRef``;
* the type of every expression: written onto ``expr.ctype`` (after lvalue conversion,
  before array/function decay; see interp/astnodes.py);
* every conversion C performs silently: made explicit as ``ImplicitCast`` nodes, so
  the interpreter never has to re-derive a static type;
* gcc's errors and warnings, each with a stable code, gcc's wording and a friendly
  explanation for a beginner;
* the unit's file-scope symbols (``UnitSymbols``) for the linker.

Warnings belong to groups (``WARNING_GROUPS``): the default group is always on, the
others need ``-Wall`` / ``-Wextra``, and ``-Werror`` turns enabled warnings into errors.
``apply_warning_options`` applies those rules and is public so the compile driver can
run the parser's warnings through the same filter.
"""
from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pycc.core.concepts import Concept
from pycc.core.diagnostics import Diagnostic, Phase, Severity, SourceLocation
from pycc.interp import astnodes as ast
from pycc.interp.builtins import BuiltinRegistry
from pycc.interp.c_types import (
    BOOL, CHAR, DOUBLE, INT, PTRDIFF_T, SIZE_T, VOID_PTR, ArrayType, CType, EnumType,
    FunctionType, IntType, PointerType, StructType, decay, fits, format_type,
    integer_promote, is_compatible, usual_arithmetic_conversions, wrap_int,
)
from pycc.interp.constexpr import eval_const_expr
from pycc.interp.format_check import check_format, type_phrase
from pycc.interp.friendly import did_you_mean, make_diagnostic, redefinition_friendly
from pycc.interp.symbols import (
    SymbolDef, SymbolRef, UnitSymbols, external_key, is_static_key, static_local_key,
    unit_static_key,
)
from pycc.interp.toolchain import CompileOptions

# --- warning groups -------------------------------------------------

WARNING_GROUPS: dict[str, frozenset[str]] = {
    "default": frozenset({
        "implicit_declaration", "incompatible_pointer", "int_from_pointer",
        "pointer_from_int", "integer_overflow_constant", "string_literal_to_char_array",
        # gcc files -Waddress under -Wall; pycc keeps it always on, because
        # 'name == "yes"' is the classic beginner bug and silently never matches.
        "address_compare",
    }),
    "wall": frozenset({
        "unused_variable", "control_reaches_end", "assignment_as_condition",
        "uninitialized_use", "array_bounds", "format_mismatch", "unused_value",
        "pointer_int_comparison", "implicit_int", "missing_return_type",
    }),
    # missing_prototype is gcc's -Wmissing-prototypes; pycc files it under -Wextra.
    "wextra": frozenset({"sign_compare", "missing_prototype"}),
}

# Library functions that never return: code after a call to them is unreachable, so a
# function ending in 'exit(1);' does not "reach the end".
NORETURN_FUNCTIONS = frozenset({"exit", "abort", "_Exit"})

# A type for expressions that already produced an error. Diagnostics about operands of
# this type are suppressed, so one mistake does not cascade into five messages.
ERROR_TYPE = IntType("<error>", 4, True)

# Python frames one level of a deep expression costs in this checker, generously.
_RECURSION_LIMIT = 30_000


def warning_group(code: str) -> str:
    """The group a warning code belongs to: 'default', 'wall' or 'wextra'."""
    for group in ("wall", "wextra"):
        if code in WARNING_GROUPS[group]:
            return group
    return "default"


def warning_enabled(code: str, options: CompileOptions) -> bool:
    """Is a warning with this code shown under these options?"""
    group = warning_group(code)
    if group == "wall":
        return options.wall
    if group == "wextra":
        return options.wextra
    return True


def apply_warning_options(diagnostics: Iterable[Diagnostic],
                          options: CompileOptions | None) -> list[Diagnostic]:
    """Drop warnings whose group is off; with -Werror turn the rest into errors."""
    options = options or CompileOptions()
    result: list[Diagnostic] = []
    for diag in diagnostics:
        if diag.severity is not Severity.WARNING:
            result.append(diag)
        elif warning_enabled(diag.code, options):
            if options.werror:
                diag = dataclasses.replace(diag, severity=Severity.ERROR,
                                           message=diag.message + " [-Werror]",
                                           notes=list(diag.notes))
            result.append(diag)
    return result


# --- results ----------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Everything the checker found: diagnostics plus the unit's symbol table."""

    diagnostics: list[Diagnostic]
    symbols: UnitSymbols

    @property
    def errors(self) -> list[Diagnostic]:
        """The diagnostics that stop compilation."""
        return [diag for diag in self.diagnostics if diag.is_error]

    @property
    def ok(self) -> bool:
        """True when no errors were reported (warnings are fine)."""
        return not self.errors


def check(unit: ast.TranslationUnit, registry: BuiltinRegistry | None = None,
          options: CompileOptions | None = None) -> CheckResult:
    """Check one parsed translation unit and annotate its AST in place."""
    if sys.getrecursionlimit() < _RECURSION_LIMIT:
        sys.setrecursionlimit(_RECURSION_LIMIT)
    checker = Checker(unit, registry or BuiltinRegistry())
    try:
        checker.check_unit()
    except RecursionError:
        checker.error("nesting_too_deep", "expression is nested too deeply to compile",
                      None, "This code nests expressions thousands of levels deep. Split it "
                            "into smaller pieces.", concept=Concept.SYNTAX)
    return CheckResult(apply_warning_options(checker.diagnostics, options), checker.symbols)


# --- symbol table entries -----------------------------------------------------------------

@dataclass
class _Entry:
    """What one name means in one scope, plus what the warnings need to know about it."""

    name: str
    kind: str                       # a SymbolRef kind, or 'typedef'
    ctype: CType
    loc: SourceLocation
    key: str = ""
    value: int | None = None        # enum constants
    is_const: bool = False
    points_to_const: bool = False
    used: bool = False
    report_unused: bool = False     # locals: warn if never referenced
    track_init: bool = False        # scalar local without initializer
    assigned: bool = False          # a textual assignment has been seen
    warned_uninit: bool = False
    loop_depth: int = 0             # how many loops enclose the declaration

    def ref(self) -> SymbolRef:
        """The SymbolRef an Identifier naming this entry gets."""
        return SymbolRef(self.kind, self.key, self.ctype, self.value)  # type: ignore[arg-type]


@dataclass
class _Global:
    """File-level knowledge about one name, for redeclaration checks across scopes."""

    name: str
    kind: str                       # 'function' | 'variable'
    entry: _Entry
    loc: SourceLocation             # first declaration
    linkage: str                    # 'external' | 'internal'
    defined_at: SourceLocation | None = None
    has_prototype_decl: bool = False


@dataclass
class _Switch:
    """The case values and default label seen so far in one switch."""

    cases: dict[int, SourceLocation] = field(default_factory=dict)
    default: SourceLocation | None = None


@dataclass
class _Function:
    """The function whose body is being checked."""

    name: str
    ret: CType
    loops: list[ast.Stmt] = field(default_factory=list)   # innermost last
    breakables: int = 0             # loops and switches: where 'break' may appear
    switches: list[_Switch] = field(default_factory=list)
    undeclared: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Target:
    """Where a value is being converted to, for gcc's wording of conversion messages."""

    kind: str                       # 'assign' | 'init' | 'arg' | 'return'
    function: str = ""
    number: int = 0


class _InitCursor:
    """Walks the items of one brace list while brace elision consumes them."""

    def __init__(self, items: list[ast.Expr]) -> None:
        self.items = items
        self.pos = 0
        self.prechecked: int | None = None   # item already type-checked by a peek

    @property
    def done(self) -> bool:
        """True once every item of the list has been consumed."""
        return self.pos >= len(self.items)


# --- the checker ------------------------------------------------------------------------------

class Checker:
    """Checks one translation unit. Use ``check()``; this class holds the walk's state."""

    def __init__(self, unit: ast.TranslationUnit, registry: BuiltinRegistry) -> None:
        self.unit = unit
        self.registry = registry
        self.diagnostics: list[Diagnostic] = []
        self.symbols = UnitSymbols(unit.name)
        self._scopes: list[dict[str, _Entry]] = [{}]
        self._globals: dict[str, _Global] = {}
        self._function: _Function | None = None
        self._unevaluated = 0                    # inside sizeof: nothing is read
        self._address_operand: ast.Expr | None = None   # the operand of the current '&'
        # Names the unit defines, collected first: a reference to 'abs' is the
        # library's unless this unit defines its own (then it is a plain function).
        self._defined_functions = {d.name for d in unit.decls if isinstance(d, ast.FunctionDef)}
        self._defined_variables = {d.name for d in unit.decls
                                   if isinstance(d, ast.VarDecl) and not d.is_extern}

    # ================================================================ reporting
    def error(self, code: str, message: str, loc: SourceLocation | None, friendly: str, *,
              notes: list[str] | None = None, concept: Concept | None = None) -> None:
        """Record an error."""
        self.diagnostics.append(make_diagnostic(code, message, loc, friendly=friendly,
                                                phase=Phase.CHECK, notes=notes,
                                                concept=concept))

    def warn(self, code: str, message: str, loc: SourceLocation | None, friendly: str, *,
             notes: list[str] | None = None) -> None:
        """Record a warning (groups and -Werror are applied at the end)."""
        self.diagnostics.append(make_diagnostic(code, message, loc, friendly=friendly,
                                                severity=Severity.WARNING, phase=Phase.CHECK,
                                                notes=notes))

    # ================================================================ scopes
    def _push(self) -> None:
        self._scopes.append({})

    def _pop(self) -> None:
        for entry in self._scopes.pop().values():
            if entry.report_unused and not entry.used:
                self.warn("unused_variable", f"unused variable '{entry.name}'", entry.loc,
                          f"'{entry.name}' is declared but never used. Remove it, or use it "
                          "if you forgot to.")

    def _lookup(self, name: str) -> _Entry | None:
        for scope in reversed(self._scopes):
            entry = scope.get(name)
            if entry is not None:
                return entry
        return None

    def _visible_names(self) -> list[str]:
        return [name for scope in self._scopes for name, entry in scope.items()
                if entry.kind != "typedef"]

    def _declare(self, entry: _Entry, previous_note: str = "declaration") -> None:
        """Add a block-scope name; a second one in the same block is an error."""
        scope = self._scopes[-1]
        earlier = scope.get(entry.name)
        if earlier is not None and len(self._scopes) > 1:
            self.error("redefinition", f"redeclaration of '{entry.name}' with no linkage",
                       entry.loc, f"'{entry.name}' is declared twice in the same block (first "
                                  f"at {earlier.loc}). Give the second one a different name, "
                                  "or drop its type to assign the existing variable.",
                       notes=[f"previous {previous_note} of '{entry.name}' was here"])
        scope[entry.name] = entry

    # ConstScope protocol (interp/constexpr.py), so constant folding sees our scopes.
    def enum_constant(self, name: str) -> int | None:
        """The value of a visible enum constant."""
        entry = self._lookup(name)
        return entry.value if entry is not None and entry.kind == "enum_const" else None

    def object_type(self, name: str) -> CType | None:
        """The declared type of a visible variable (for 'sizeof arr / sizeof arr[0]')."""
        entry = self._lookup(name)
        if entry is None or entry.kind in ("typedef", "enum_const", "function", "builtin"):
            return None
        return entry.ctype

    def _const(self, expr: ast.Expr) -> int | None:
        return eval_const_expr(expr, self)

    # ================================================================ the unit
    def check_unit(self) -> None:
        """Check every top-level declaration in order."""
        handlers: dict[type, Callable[[Checker, ast.Decl], None]] = {
            ast.VarDecl: Checker._file_variable,
            ast.FunctionDecl: Checker._prototype,
            ast.FunctionDef: Checker._function_definition,
            ast.EnumDecl: Checker._enum,
            ast.TypedefDecl: Checker._typedef,
        }
        for decl in self.unit.decls:
            handler = handlers.get(type(decl))
            if handler is not None:
                handler(self, decl)

    def _enum(self, decl: ast.EnumDecl) -> None:
        for name, value in decl.constants:
            self._scopes[-1][name] = _Entry(name, "enum_const", INT, decl.loc, str(value), value)

    def _typedef(self, decl: ast.TypedefDecl) -> None:
        # Kept so an inner typedef hides an outer variable of the same name, as in C.
        self._scopes[-1][decl.name] = _Entry(decl.name, "typedef", decl.ctype, decl.loc)

    # ---------------------------------------------------------- functions
    def _prototype(self, decl: ast.FunctionDecl) -> None:
        entry = self._declare_function(decl, is_definition=False)
        self._scopes[-1].setdefault(decl.name, entry)

    def _declare_function(self, decl: ast.FunctionDecl | ast.FunctionDef,
                          is_definition: bool) -> _Entry:
        """Merge a prototype or definition with what the unit already knows."""
        name, ftype = decl.name, decl.ftype
        static = decl.storage == "static"
        known = self._globals.get(name)
        if known is None:
            known = self._new_function(decl, static)
        elif known.kind != "function":
            self._different_kind(name, decl.loc, known)
            return known.entry
        else:
            self._merge_function(known, decl, static, is_definition)
        if isinstance(decl, ast.FunctionDecl) and ftype.has_prototype:
            known.has_prototype_decl = True
        if is_definition and known.defined_at is None:
            known.defined_at = decl.loc
        decl.storage_key = known.entry.key  # type: ignore[attr-defined]
        self._record_function_symbol(known, decl, is_definition)
        return known.entry

    def _new_function(self, decl: ast.FunctionDecl | ast.FunctionDef, static: bool) -> _Global:
        name = decl.name
        key = unit_static_key(self.unit.name, name) if static else external_key(name)
        library = name not in self._defined_functions and name in self.registry.functions
        entry = _Entry(name, "builtin" if library else "function", decl.ftype, decl.loc, key)
        known = _Global(name, "function", entry, decl.loc, "internal" if static else "external")
        self._globals[name] = known
        return known

    def _merge_function(self, known: _Global, decl: ast.FunctionDecl | ast.FunctionDef,
                        static: bool, is_definition: bool) -> None:
        name, ftype = decl.name, decl.ftype
        old = known.entry.ctype
        if not is_compatible(old, ftype):
            word = "definition" if known.defined_at else "declaration"
            self.error("conflicting_types", f"conflicting types for '{name}'", decl.loc,
                       f"'{name}' was declared as '{_signature(name, old)}' (at {known.loc}) "
                       f"but here it is '{_signature(name, ftype)}'. Make the prototype and "
                       "the definition match exactly.",
                       notes=[f"previous {word} of '{name}' was here"])
            return
        if is_definition and known.defined_at is not None:
            self.error("redefinition", f"redefinition of '{name}'", decl.loc,
                       redefinition_friendly(name, known.defined_at, decl.loc),
                       notes=[f"previous definition of '{name}' was here"])
            return
        if static and known.linkage == "external":
            self.error("conflicting_types",
                       f"static declaration of '{name}' follows non-static declaration",
                       decl.loc, f"'{name}' was first declared without 'static'. Use 'static' "
                                 "on every declaration of it, or on none.",
                       notes=[f"previous declaration of '{name}' was here"])
            return
        if isinstance(old, FunctionType) and not old.has_prototype and ftype.has_prototype:
            known.entry.ctype = ftype  # 'int f();' then 'int f(int x)': now we know more

    def _different_kind(self, name: str, loc: SourceLocation, known: _Global) -> None:
        self.error("conflicting_types", f"'{name}' redeclared as different kind of symbol", loc,
                   f"'{name}' is already a {known.kind} (declared at {known.loc}). A name can "
                   "be a variable or a function, not both. Rename one of them.",
                   notes=[f"previous declaration of '{name}' was here"])

    def _record_function_symbol(self, known: _Global, decl: ast.FunctionDecl | ast.FunctionDef,
                                is_definition: bool) -> None:
        symbol = SymbolDef(decl.name, "function", known.linkage,  # type: ignore[arg-type]
                           known.entry.ctype, decl.loc, False, is_definition, known.entry.key)
        if is_definition:
            self.symbols.defined[decl.name] = symbol
        else:
            self.symbols.declared.setdefault(decl.name, symbol)

    def _function_definition(self, fdef: ast.FunctionDef) -> None:
        name, ftype = fdef.name, fdef.ftype
        if ftype.variadic:
            self.error("unsupported_feature", "variadic function definitions are not supported",
                       fdef.loc, f"'{name}' takes '...'. Functions with a variable number of "
                                 "arguments (like printf) cannot be written in pycc's C. "
                                 "Pass an array and a count instead.")
        self._check_main_signature(fdef)
        self._check_missing_prototype(fdef)
        entry = self._declare_function(fdef, is_definition=True)
        self._scopes[0].setdefault(name, entry)
        self._function = _Function(name, ftype.ret)
        self._push()   # parameters and the body's outermost block share one scope in C
        try:
            self._declare_parameters(fdef)
            self._block_items(fdef.body.items)
            self._check_falls_off(fdef)
        finally:
            self._pop()
            self._function = None

    def _check_main_signature(self, fdef: ast.FunctionDef) -> None:
        if fdef.name == "main" and fdef.ftype.ret != INT and fdef.ftype.ret is not ERROR_TYPE:
            self.warn("missing_return_type", "return type of 'main' is not 'int'", fdef.loc,
                      "main tells the shell whether the program worked, so it must return "
                      "int. Write 'int main(void)' and end with 'return 0;'.")

    def _check_missing_prototype(self, fdef: ast.FunctionDef) -> None:
        if fdef.storage == "static" or fdef.name == "main":
            return
        known = self._globals.get(fdef.name)
        if known is None or not known.has_prototype_decl:
            self.warn("missing_prototype", f"no previous prototype for '{fdef.name}'", fdef.loc,
                      f"'{fdef.name}' is defined without a prototype before it. Put "
                      f"'{_signature(fdef.name, fdef.ftype)};' in a header so other files call "
                      "it correctly, or make it static if only this file uses it.")

    def _declare_parameters(self, fdef: ast.FunctionDef) -> None:
        for index, (name, ctype) in enumerate(zip(fdef.param_names, fdef.ftype.params)):
            if not name:
                continue
            loc = fdef.param_locs[index] if index < len(fdef.param_locs) else fdef.loc
            is_const = index < len(fdef.param_is_const) and fdef.param_is_const[index]
            if name in self._scopes[-1]:
                self.error("redefinition", f"redefinition of parameter '{name}'", loc,
                           f"Two parameters of '{fdef.name}' are called '{name}'. Give each "
                           "parameter its own name.")
                continue
            self._scopes[-1][name] = _Entry(name, "param", ctype, loc, name, is_const=is_const)

    def _check_falls_off(self, fdef: ast.FunctionDef) -> None:
        ret = fdef.ftype.ret
        # C99: falling off the end of main returns 0, so main never gets this warning.
        if ret.is_void or fdef.name == "main" or not can_fall_through(fdef.body):
            return
        loc = fdef.body.end_loc or fdef.loc
        self.warn("control_reaches_end", "control reaches end of non-void function", loc,
                  f"'{fdef.name}' should return {_a(ret)}, but when the code reaches "
                  "the closing '}' there is no return, so the caller gets garbage. Add a "
                  "'return' for every path, for example at the end of the function.")

    # ---------------------------------------------------------- file-scope variables
    def _file_variable(self, decl: ast.VarDecl) -> None:
        name = decl.name
        static = decl.storage == "static"
        extern = decl.is_extern or decl.storage == "extern"
        known = self._globals.get(name)
        if known is None:
            known = self._new_global_variable(decl, static, extern)
            self._scopes[0][name] = known.entry
        elif known.kind != "variable":
            self._different_kind(name, decl.loc, known)
            return
        elif not self._merge_variable(known, decl):
            if decl.init is not None:
                decl.init = self._initializer(decl.ctype, decl.init)  # still annotate it
            return
        if not extern:
            self._require_complete_object(decl)
        decl.storage_key = known.entry.key  # type: ignore[attr-defined]
        if decl.init is not None:
            decl.init = self._initializer(decl.ctype, decl.init)
            self._require_constant_initializer(decl)
            known.defined_at = known.defined_at or decl.loc
        self._record_variable_symbol(known, decl, extern)

    def _new_global_variable(self, decl: ast.VarDecl, static: bool, extern: bool) -> _Global:
        name = decl.name
        if static:
            kind, key = "unit_static", unit_static_key(self.unit.name, name)
        elif extern and name not in self._defined_variables and name in self.registry.variables:
            kind, key = "builtin", external_key(name)
        else:
            kind, key = "global", external_key(name)
        entry = _Entry(name, kind, decl.ctype, decl.loc, key, is_const=decl.is_const,
                       points_to_const=decl.points_to_const)
        known = _Global(name, "variable", entry, decl.loc, "internal" if static else "external")
        self._globals[name] = known
        return known

    def _merge_variable(self, known: _Global, decl: ast.VarDecl) -> bool:
        """Combine a redeclaration with the earlier one; False after an error."""
        name = decl.name
        old = known.entry.ctype
        if not is_compatible(old, decl.ctype):
            self.error("conflicting_types", f"conflicting types for '{name}'", decl.loc,
                       f"'{name}' was declared as {_a(old)} (at {known.loc}) and "
                       f"now as {_a(decl.ctype)}. Use one type everywhere.",
                       notes=[f"previous declaration of '{name}' was here"])
            return False
        if decl.init is not None and known.defined_at is not None:
            self.error("redefinition", f"redefinition of '{name}'", decl.loc,
                       redefinition_friendly(name, known.defined_at, decl.loc),
                       notes=[f"previous definition of '{name}' was here"])
            return False
        if isinstance(old, ArrayType) and old.length is None:
            known.entry.ctype = decl.ctype   # 'extern int a[];' then 'int a[5];'
        return True

    def _record_variable_symbol(self, known: _Global, decl: ast.VarDecl, extern: bool) -> None:
        name = decl.name
        if extern and decl.init is None:
            self.symbols.declared.setdefault(name, SymbolDef(
                name, "variable", known.linkage, known.entry.ctype,  # type: ignore[arg-type]
                decl.loc, False, False, known.entry.key))
            return
        tentative = decl.init is None
        existing = self.symbols.defined.get(name)
        if existing is not None and not existing.tentative and tentative:
            return   # 'int x = 1; int x;': the real definition stays
        self.symbols.defined[name] = SymbolDef(
            name, "variable", known.linkage, known.entry.ctype,  # type: ignore[arg-type]
            decl.loc, tentative, True, known.entry.key)

    def _require_complete_object(self, decl: ast.VarDecl) -> bool:
        ctype = decl.ctype
        while isinstance(ctype, ArrayType):
            ctype = ctype.element
        if isinstance(ctype, StructType) and not ctype.is_complete:
            self.error("incomplete_type", f"storage size of '{decl.name}' isn't known", decl.loc,
                       f"'{format_type(ctype)}' is only declared, not defined, so C does not "
                       f"know how big '{decl.name}' is. Define the struct (with its members) "
                       "before this line, or use a pointer to it.")
            return False
        return True

    def _require_constant_initializer(self, decl: ast.VarDecl) -> None:
        """Globals and statics get their value before main runs: only constants allowed."""
        assert decl.init is not None
        for item in _initializer_items(decl.init):
            if any(_type(node) is ERROR_TYPE for node in ast.walk(item)
                   if isinstance(node, ast.Expr)):
                continue   # already reported; a second message would only confuse
            if not self._is_constant_value(item):
                self.error("bad_initializer", "initializer element is not constant",
                           _start(item), f"'{decl.name}' is set before the program starts, "
                           "when no code can run yet, so its initializer may only use "
                           "constants: numbers, characters, strings, sizeof, enum names and "
                           "addresses of globals. Give it its value inside a function "
                           "instead, for example at the start of main.", concept=Concept.SCOPE)
                return   # one message per declaration is enough

    def _is_constant_value(self, expr: ast.Expr) -> bool:
        """Is this checked expression something C can compute before the program runs?"""
        if isinstance(expr, ast.ImplicitCast):
            inner = _type(expr.expr)
            if isinstance(inner, (ArrayType, FunctionType)):
                return self._is_address_constant(expr.expr)   # 'int *p = global_array;'
            return self._is_constant_value(expr.expr)
        if isinstance(expr, (ast.IntLiteral, ast.FloatLiteral, ast.CharLiteral,
                             ast.StringLiteral, ast.SizeofType, ast.SizeofExpr)):
            return True
        if isinstance(expr, ast.Identifier):
            return self._is_constant_name(expr)
        if isinstance(expr, ast.Cast):
            return self._is_constant_value(expr.expr)
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "&":
                return self._is_address_constant(expr.operand)
            return expr.op in ("+", "-", "~", "!") and self._is_constant_value(expr.operand)
        if isinstance(expr, ast.BinaryOp):
            return self._is_constant_value(expr.left) and self._is_constant_value(expr.right)
        if isinstance(expr, ast.Conditional):
            return all(self._is_constant_value(part)
                       for part in (expr.cond, expr.then, expr.otherwise))
        return False   # calls, assignments, reads of variables, the comma operator

    def _is_constant_name(self, expr: ast.Identifier) -> bool:
        ref = getattr(expr, "ref", None)
        if ref is None:
            return False
        if ref.kind == "enum_const":
            return True
        # gcc (since version 8) also folds a const global, as in 'const int N = 5;'. A
        # const local is different: it gets its value only when the function runs.
        entry = self._lookup(expr.name)
        return (entry is not None and entry.is_const and entry.ctype.is_scalar
                and entry.kind in ("global", "unit_static", "static_local"))

    def _is_address_constant(self, expr: ast.Expr) -> bool:
        """Does ``expr`` name an object (or function) whose address is fixed for good?"""
        if isinstance(expr, ast.Identifier):
            ref = getattr(expr, "ref", None)
            return ref is not None and ref.kind in _STATIC_STORAGE_KINDS
        if isinstance(expr, ast.StringLiteral):
            return True
        if isinstance(expr, ast.Index):
            return self._is_constant_value(expr.base) and self._is_constant_value(expr.index)
        if isinstance(expr, ast.Member):
            if expr.arrow:
                return self._is_constant_value(expr.base)
            return self._is_address_constant(expr.base)
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            return self._is_constant_value(expr.operand)
        return False

    # ================================================================ statements
    def _block_items(self, items: list[ast.Node]) -> None:
        for item in items:
            self._stmt(item)

    def _stmt(self, stmt: ast.Node) -> None:
        handler = _STMT_HANDLERS.get(type(stmt))
        if handler is not None:
            handler(self, stmt)

    def _compound(self, stmt: ast.CompoundStmt) -> None:
        self._push()
        try:
            self._block_items(stmt.items)
        finally:
            self._pop()

    def _decl_stmt(self, stmt: ast.DeclStmt) -> None:
        for decl in stmt.decls:
            if isinstance(decl, ast.VarDecl):
                self._local_variable(decl)
            elif isinstance(decl, ast.FunctionDecl):
                entry = self._declare_function(decl, is_definition=False)
                self._scopes[-1][decl.name] = entry
            elif isinstance(decl, ast.EnumDecl):
                self._enum(decl)
            elif isinstance(decl, ast.TypedefDecl):
                self._typedef(decl)

    def _local_variable(self, decl: ast.VarDecl) -> None:
        if decl.is_extern or decl.storage == "extern":
            self._block_extern(decl)
            return
        name = decl.name
        function = self._function.name if self._function else ""
        if decl.storage == "static":
            key = static_local_key(self.unit.name, function, name, decl.loc)
            entry = _Entry(name, "static_local", decl.ctype, decl.loc, key,
                           is_const=decl.is_const, points_to_const=decl.points_to_const,
                           report_unused=True)
            self.symbols.static_locals[key] = SymbolDef(name, "variable", "internal",
                                                        decl.ctype, decl.loc, False, True, key)
        else:
            scalar = decl.ctype.is_scalar
            entry = _Entry(name, "local", decl.ctype, decl.loc, name, is_const=decl.is_const,
                           points_to_const=decl.points_to_const, report_unused=True,
                           track_init=scalar and decl.init is None,
                           loop_depth=len(self._function.loops) if self._function else 0)
        decl.storage_key = entry.key  # type: ignore[attr-defined]
        self._require_complete_object(decl)
        self._declare(entry)
        if decl.init is not None:
            decl.init = self._initializer(decl.ctype, decl.init)
            entry.assigned = True
            if decl.storage == "static":
                self._require_constant_initializer(decl)

    def _block_extern(self, decl: ast.VarDecl) -> None:
        """'extern int g;' inside a block names the global; it is not a new local."""
        known = self._globals.get(decl.name)
        if known is None:
            known = self._new_global_variable(decl, static=False, extern=True)
            self._record_variable_symbol(known, decl, extern=True)
        elif known.kind != "variable":
            self._different_kind(decl.name, decl.loc, known)
            return
        else:
            self._merge_variable(known, decl)
        decl.storage_key = known.entry.key  # type: ignore[attr-defined]
        self._declare(known.entry)

    def _expr_stmt(self, stmt: ast.ExprStmt) -> None:
        self._expr(stmt.expr)
        if _has_no_effect(stmt.expr):
            self._warn_no_effect(stmt.expr)

    def _warn_no_effect(self, expr: ast.Expr) -> None:
        if isinstance(expr, ast.BinaryOp) and expr.op == "==":
            friendly = ("This line compares two values and throws the answer away. Did you "
                        "mean '=' to store a value?")
        else:
            friendly = ("This line computes a value and then throws it away, so it does "
                        "nothing. Did you forget an assignment or a function call?")
        self.warn("unused_value", "statement with no effect", _start(expr), friendly)

    def _if(self, stmt: ast.IfStmt) -> None:
        stmt.cond = self._condition(stmt.cond)
        self._stmt(stmt.then)
        if stmt.otherwise is not None:
            self._stmt(stmt.otherwise)

    def _while(self, stmt: ast.WhileStmt) -> None:
        stmt.cond = self._condition(stmt.cond)
        self._loop_body(stmt, [stmt.body])

    def _do_while(self, stmt: ast.DoWhileStmt) -> None:
        self._loop_body(stmt, [stmt.body])
        stmt.cond = self._condition(stmt.cond)

    def _for(self, stmt: ast.ForStmt) -> None:
        self._push()   # 'for (int i = 0; ...)': i lives only inside the loop
        try:
            if isinstance(stmt.init, ast.DeclStmt):
                self._decl_stmt(stmt.init)
            elif isinstance(stmt.init, ast.Expr):
                self._expr(stmt.init)
            if stmt.cond is not None:
                stmt.cond = self._condition(stmt.cond)
            # The step runs after the body, so it belongs to the loop like the body does.
            self._loop_body(stmt, [stmt.step, stmt.body])
        finally:
            self._pop()

    def _loop_body(self, loop: ast.Stmt, parts: list[ast.Node | None]) -> None:
        """Check the parts of ``loop`` that repeat (its body, a for's step)."""
        function = self._function
        assert function is not None
        function.loops.append(loop)
        function.breakables += 1
        try:
            for part in parts:
                if isinstance(part, ast.Expr):
                    self._expr(part)
                elif part is not None:
                    self._stmt(part)
        finally:
            function.loops.pop()
            function.breakables -= 1

    def _return(self, stmt: ast.ReturnStmt) -> None:
        function = self._function
        assert function is not None
        ret = function.ret
        if stmt.value is None:
            if not ret.is_void and ret is not ERROR_TYPE:
                self.error("return_missing_value",
                           "'return' with no value, in function returning non-void", stmt.loc,
                           f"'{function.name}' promises to return {_a(ret)}, so "
                           "'return' needs a value here, like 'return 0;'.",
                           notes=["declared here"])
            return
        if ret.is_void:
            value = self._value(stmt.value)
            stmt.value = value
            if not _type(value).is_void:
                self.error("return_value_in_void",
                           "'return' with a value, in function returning void", stmt.loc,
                           f"'{function.name}' is declared void, so it cannot give back a "
                           "value. Remove the value, or change 'void' to the type you want "
                           "to return.", notes=["declared here"])
            return
        value = self._operand(stmt.value)
        stmt.value = self._convert(value, ret, _Target("return", function.name))

    def _break(self, stmt: ast.BreakStmt) -> None:
        if self._function is None or self._function.breakables == 0:
            self.error("break_outside_loop", "break statement not within loop or switch",
                       stmt.loc, "'break' jumps out of a loop or a switch, but this one is not "
                                 "inside either. Remove it, or move it into the loop.")

    def _continue(self, stmt: ast.ContinueStmt) -> None:
        if self._function is None or not self._function.loops:
            self.error("break_outside_loop", "continue statement not within a loop", stmt.loc,
                       "'continue' skips to the next round of a loop, but this one is not "
                       "inside a loop. Remove it, or move it into the loop.")

    def _switch(self, stmt: ast.SwitchStmt) -> None:
        function = self._function
        assert function is not None
        cond = self._operand(stmt.cond)
        ctype = _type(cond)
        if not _broken(ctype) and not ctype.is_integer:
            self.error("incompatible_types", "switch quantity not an integer", _start(cond),
                       f"switch only works on whole numbers (int, char, enums), not on "
                       f"{_a(ctype)}. Use if/else for other values.")
        elif ctype.is_integer:
            cond = self._cast_if_needed(cond, integer_promote(ctype))
        stmt.cond = cond
        function.switches.append(_Switch())
        function.breakables += 1
        try:
            self._stmt(stmt.body)
        finally:
            function.switches.pop()
            function.breakables -= 1

    def _case(self, stmt: ast.CaseLabel) -> None:
        self._expr(stmt.value)
        switch = self._current_switch()
        if switch is None:
            self.error("case_outside_switch", "case label not within a switch statement",
                       stmt.loc, "'case' labels only make sense inside a switch (...) { } "
                                 "block. Check the braces of the switch above.")
        elif stmt.const_value is not None:
            first = switch.cases.get(stmt.const_value)
            if first is not None:
                self.error("duplicate_case", "duplicate case value", stmt.loc,
                           f"Two case labels have the value {stmt.const_value} (the first at "
                           f"{first}). Each case in a switch must be different.",
                           notes=["previously used here"])
            else:
                switch.cases[stmt.const_value] = stmt.loc
        self._stmt(stmt.body)

    def _default(self, stmt: ast.DefaultLabel) -> None:
        switch = self._current_switch()
        if switch is None:
            self.error("case_outside_switch", "'default' label not within a switch statement",
                       stmt.loc, "'default' only makes sense inside a switch (...) { } block. "
                                 "Check the braces of the switch above.")
        elif switch.default is not None:
            self.error("duplicate_case", "multiple default labels in one switch", stmt.loc,
                       f"This switch already has a 'default' at {switch.default}. Keep one.",
                       notes=["this is the first default label"])
        else:
            switch.default = stmt.loc
        self._stmt(stmt.body)

    def _current_switch(self) -> _Switch | None:
        if self._function is None or not self._function.switches:
            return None
        return self._function.switches[-1]

    def _condition(self, cond: ast.Expr) -> ast.Expr:
        """An if/while/for/?: condition: any scalar, and gcc's '=' versus '==' warning."""
        if isinstance(cond, ast.Assignment) and cond.op == "=" and not cond.parenthesized:
            self.warn("assignment_as_condition",
                      "suggest parentheses around assignment used as truth value", _start(cond),
                      "'=' stores a value; '==' compares two values. If you meant to compare, "
                      f"write '{_text(cond.target) or 'x'} == ...'. If the assignment is on "
                      "purpose, wrap it in an extra pair of parentheses.")
        value = self._operand(cond)
        if isinstance(_type(value), StructType):
            self.error("incompatible_types", "used struct type value where scalar is required",
                       _start(value), "A whole struct cannot be true or false. Test one of its "
                                      "members instead, like 'if (s.count > 0)'.")
        return value

    # ================================================================ expressions
    def _expr(self, expr: ast.Expr) -> CType:
        """Type-check ``expr`` and annotate it (no decay); returns its type."""
        handler = _EXPR_HANDLERS.get(type(expr))
        result = handler(self, expr) if handler is not None else ERROR_TYPE
        if not isinstance(expr, ast.SizeofType):
            expr.ctype = result  # type: ignore[attr-defined]
        return result

    def _value(self, expr: ast.Expr) -> ast.Expr:
        """Check ``expr`` used as a value: arrays and functions decay to pointers."""
        self._expr(expr)
        return self._decay(expr)

    def _operand(self, expr: ast.Expr) -> ast.Expr:
        """Like ``_value``, but a void expression (a call to a void function) is an error."""
        value = self._value(expr)
        if _type(value).is_void:
            self.error("void_value", "void value not ignored as it ought to be", _start(value),
                       "This expression has type void: the function returns nothing, so there "
                       "is no value to use here. Call it on its own line, or make the function "
                       "return a value.", concept=Concept.FUNCTION_DESIGN)
            value.ctype = ERROR_TYPE  # type: ignore[attr-defined]
        return value

    def _decay(self, expr: ast.Expr) -> ast.Expr:
        ctype = _type(expr)
        if isinstance(ctype, (ArrayType, FunctionType)):
            return _wrap(expr, decay(ctype))
        return expr

    def _cast_if_needed(self, expr: ast.Expr, to: CType) -> ast.Expr:
        if _same_type(_type(expr), to):
            return expr
        return _wrap(expr, to)

    # ---------------------------------------------------------- leaves
    def _int_literal(self, expr: ast.IntLiteral) -> CType:
        return expr.ctype

    def _float_literal(self, expr: ast.FloatLiteral) -> CType:
        return expr.ctype

    def _char_literal(self, expr: ast.CharLiteral) -> CType:
        return INT   # 'a' is an int in C

    def _string_literal(self, expr: ast.StringLiteral) -> CType:
        return ArrayType(CHAR, len(expr.value) + 1)   # the hidden NUL counts

    def _identifier(self, expr: ast.Identifier, store: bool = False) -> CType:
        entry = self._lookup(expr.name)
        if entry is None or entry.kind == "typedef":
            self._undeclared(expr)
            expr.ref = SymbolRef("global", expr.name, ERROR_TYPE)  # type: ignore[attr-defined]
            return ERROR_TYPE
        entry.used = True
        expr.ref = entry.ref()  # type: ignore[attr-defined]
        if (entry.track_init and not store and not entry.assigned and not entry.warned_uninit
                and not self._unevaluated and not self._assigned_by_enclosing_loop(entry)):
            entry.warned_uninit = True
            self.warn("uninitialized_use", f"'{expr.name}' is used uninitialized", expr.loc,
                      f"'{expr.name}' is read here before it was given a value, so it holds "
                      f"leftover garbage. Initialize it where you declare it, for example "
                      f"'{format_type(entry.ctype)} {expr.name} = 0;'.")
        self._note_reference(entry, expr.loc)
        return entry.ctype

    def _assigned_by_enclosing_loop(self, entry: _Entry) -> bool:
        """Is this read inside a loop that gives the variable a value further down?

        'int prev; for (...) { if (i > 0) use(prev); prev = i; }' reads 'prev' before
        the assignment in the text, but on every round after the first the previous
        round has set it. gcc does not warn about this at -O0, so neither do we.
        """
        function = self._function
        if function is None or len(function.loops) <= entry.loop_depth:
            return False
        outermost = function.loops[entry.loop_depth]   # the first loop inside its scope
        return _assigns_name(outermost, entry.name)

    def _note_reference(self, entry: _Entry, loc: SourceLocation) -> None:
        """Remember uses of external names: the linker must find each one somewhere."""
        if entry.kind in ("global", "function", "builtin") and not is_static_key(entry.key):
            self.symbols.referenced.setdefault(entry.name, []).append(loc)

    def _undeclared(self, expr: ast.Identifier) -> None:
        name = expr.name
        function = self._function
        if function is not None:
            if name in function.undeclared:
                return   # gcc reports each name once per function
            function.undeclared.add(name)
        suggestion = did_you_mean(name, self._visible_names())
        if function is not None:
            message = f"'{name}' undeclared (first use in this function)"
            notes = ["each undeclared identifier is reported only once for each function it "
                     "appears in"] if len(function.undeclared) == 1 else []
        else:
            message, notes = f"'{name}' undeclared here (not in a function)", []
        if suggestion is not None:
            message += f"; did you mean '{suggestion}'?"
        self.error("undeclared_identifier", message, expr.loc,
                   self._undeclared_friendly(name, suggestion), notes=notes)

    def _undeclared_friendly(self, name: str, suggestion: str | None) -> str:
        header = self.registry.header_for(name)
        if header is not None:
            return (f"'{name}' is declared in <{header}>. Did you forget #include <{header}>? "
                    "Add it at the top of the file.")
        if suggestion is not None and suggestion.lower() == name.lower():
            return (f"C is case-sensitive: '{name}' and '{suggestion}' are different names. "
                    f"Did you mean '{suggestion}'?")
        text = f"'{name}' has not been declared at this point. "
        if suggestion is not None:
            return text + f"Did you mean '{suggestion}'?"
        return text + (f"A variable must be declared before it is used, for example "
                       f"'int {name} = 0;'.")

    # ---------------------------------------------------------- operators
    def _binary(self, expr: ast.BinaryOp) -> CType:
        op = expr.op
        if op in ("&&", "||"):
            expr.left = self._scalar_operand(expr.left, op)
            expr.right = self._scalar_operand(expr.right, op)
            return INT
        expr.left = self._operand(expr.left)
        expr.right = self._operand(expr.right)
        left, right = _type(expr.left), _type(expr.right)
        if _broken(left) or _broken(right):
            return ERROR_TYPE
        if op in ("+", "-"):
            return self._additive(expr, left, right)
        if op in _COMPARISON_OPS:
            return self._comparison(expr, left, right)
        if op in ("<<", ">>"):
            if not (left.is_integer and right.is_integer):
                return self._invalid_operands(expr, left, right)
            expr.left = self._cast_if_needed(expr.left, integer_promote(left))
            expr.right = self._cast_if_needed(expr.right, integer_promote(right))
            return integer_promote(left)
        needs_integers = op in ("%", "&", "|", "^")
        ok = (left.is_integer and right.is_integer) if needs_integers else (
            left.is_arithmetic and right.is_arithmetic)
        if not ok:
            return self._invalid_operands(expr, left, right)
        return self._arithmetic_conversions(expr, left, right)

    def _arithmetic_conversions(self, expr: ast.BinaryOp, left: CType, right: CType) -> CType:
        common = usual_arithmetic_conversions(left, right)
        expr.left = self._cast_if_needed(expr.left, common)
        expr.right = self._cast_if_needed(expr.right, common)
        return common

    def _additive(self, expr: ast.BinaryOp, left: CType, right: CType) -> CType:
        if left.is_arithmetic and right.is_arithmetic:
            return self._arithmetic_conversions(expr, left, right)
        if left.is_pointer and right.is_integer:
            return left
        if expr.op == "+" and left.is_integer and right.is_pointer:
            return right
        if expr.op == "-" and left.is_pointer and right.is_pointer:
            if _pointers_compatible(left, right):  # type: ignore[arg-type]
                return PTRDIFF_T
        return self._invalid_operands(expr, left, right)

    def _comparison(self, expr: ast.BinaryOp, left: CType, right: CType) -> CType:
        self._check_string_literal_compare(expr)
        if left.is_arithmetic and right.is_arithmetic:
            self._check_sign_compare(expr, left, right)
            self._arithmetic_conversions(expr, left, right)
            return INT
        if left.is_pointer and right.is_pointer:
            if not _pointers_compatible(left, right):  # type: ignore[arg-type]
                self.warn("incompatible_pointer", "comparison of distinct pointer types lacks a "
                          "cast", expr.loc, f"{_a(left, True)} is compared with "
                          f"{_a(right)}. Pointers to different types rarely point "
                          "at the same thing; check what you are comparing.")
            return INT
        if left.is_pointer and right.is_integer:
            expr.right = self._pointer_versus_integer(expr, expr.right, left)
            return INT
        if left.is_integer and right.is_pointer:
            expr.left = self._pointer_versus_integer(expr, expr.left, right)
            return INT
        self._invalid_operands(expr, left, right)
        return INT

    def _check_string_literal_compare(self, expr: ast.BinaryOp) -> None:
        """gcc's -Waddress: ``name == "yes"`` compares addresses, never letters."""
        if expr.op not in ("==", "!="):
            return
        if not (_is_string_literal(expr.left) or _is_string_literal(expr.right)):
            return
        self.warn("address_compare", "comparison with string literal results in unspecified "
                  "behavior", expr.loc, "Use strcmp to compare text. '==' only checks whether "
                  "both sides are stored at the same address, and a string literal never "
                  "shares an address with your array: write 'strcmp(a, \"text\") == 0'.")

    def _pointer_versus_integer(self, expr: ast.BinaryOp, number: ast.Expr,
                                pointer: CType) -> ast.Expr:
        if not self._is_null_constant(number):
            self.warn("pointer_int_comparison", "comparison between pointer and integer",
                      expr.loc, f"One side is a pointer ('{format_type(pointer)}'), the other a "
                      "number. Compare a pointer with another pointer or NULL; to compare the "
                      "value it points to, write '*p'.")
        return _wrap(number, pointer)

    def _check_sign_compare(self, expr: ast.BinaryOp, left: CType, right: CType) -> None:
        if not (left.is_integer and right.is_integer):
            return
        common = usual_arithmetic_conversions(left, right)
        if getattr(common, "signed", True):
            return
        left_signed = getattr(integer_promote(left), "signed", True)
        right_signed = getattr(integer_promote(right), "signed", True)
        if left_signed == right_signed:
            return
        signed_side = expr.left if left_signed else expr.right
        value = self._const(signed_side)
        if value is not None and value >= 0:
            return   # 'u < 10' is fine: the constant cannot be negative
        self.warn("sign_compare", "comparison of integer expressions of different signedness: "
                  f"'{format_type(left)}' and '{format_type(right)}'", expr.loc,
                  "One side is signed and the other unsigned, so C converts the signed one to "
                  "unsigned first: -1 becomes a huge number and the comparison goes wrong. "
                  "Make both sides the same signedness.")

    def _scalar_operand(self, operand: ast.Expr, op: str) -> ast.Expr:
        value = self._operand(operand)
        ctype = _type(value)
        if not _broken(ctype) and not ctype.is_scalar:
            self.error("invalid_operands", f"invalid operands to binary {op} (have "
                       f"'{format_type(ctype)}')", _start(value),
                       f"'{op}' needs values that can be true or false, like numbers or "
                       f"pointers, not {_a(ctype)}.")
        return value

    def _invalid_operands(self, expr: ast.BinaryOp, left: CType, right: CType) -> CType:
        op = expr.op
        if left.is_pointer and right.is_pointer:
            friendly = (f"'{op}' does not work on two pointers. Dereference them first (*p) to "
                        "work with the values they point to.")
        elif left.is_struct or right.is_struct:
            friendly = (f"A whole struct cannot be used with '{op}'. Use one of its members, "
                        "like s.x.")
        else:
            friendly = (f"'{op}' does not work on {_a(left)} and "
                        f"{_a(right)}.")
        self.error("invalid_operands", f"invalid operands to binary {op} (have "
                   f"'{format_type(left)}' and '{format_type(right)}')", expr.loc, friendly)
        return ERROR_TYPE

    def _unary(self, expr: ast.UnaryOp) -> CType:
        op = expr.op
        if op in ("++", "--"):
            return self._increment(expr, expr.operand, op)
        if op == "&":
            return self._address_of(expr)
        expr.operand = self._operand(expr.operand)
        ctype = _type(expr.operand)
        if _broken(ctype):
            return ERROR_TYPE
        if op == "*":
            if not isinstance(ctype, PointerType):
                return self._wrong_unary(expr, ctype, f"invalid type argument of unary '*' "
                                                      f"(have '{format_type(ctype)}')",
                                         "'*' reads what a pointer points to, but this is not "
                                         "a pointer. Did you mean to multiply, or is the "
                                         "variable missing a '*' in its declaration?")
            return ctype.target
        if op == "!":
            if not ctype.is_scalar:
                return self._wrong_unary(expr, ctype, "wrong type argument to unary exclamation "
                                         "mark", "'!' needs a number or a pointer.")
            return INT
        needs = ctype.is_integer if op == "~" else ctype.is_arithmetic
        if not needs:
            names = {"-": "unary minus", "+": "unary plus", "~": "bit-complement"}
            return self._wrong_unary(expr, ctype, f"wrong type argument to {names[op]}",
                                     f"'{op}' needs a number, not {_a(ctype)}.")
        promoted = integer_promote(ctype)
        expr.operand = self._cast_if_needed(expr.operand, promoted)
        return promoted

    def _wrong_unary(self, expr: ast.UnaryOp, ctype: CType, message: str, friendly: str) -> CType:
        self.error("invalid_operands", message, expr.loc, friendly)
        return ERROR_TYPE

    def _address_of(self, expr: ast.UnaryOp) -> CType:
        operand = expr.operand
        self._address_operand = operand
        if isinstance(operand, ast.Identifier):
            ctype = self._identifier(operand, store=True)
            operand.ctype = ctype  # type: ignore[attr-defined]
        else:
            ctype = self._expr(operand)
        self._address_operand = None
        if ctype is ERROR_TYPE:
            return ERROR_TYPE
        if not (_is_lvalue(operand) or isinstance(ctype, FunctionType)):
            self.error("lvalue_required", "lvalue required as unary '&' operand", expr.loc,
                       "'&' gives the address of a variable or array element. This is a "
                       "computed value with no address of its own.")
            return ERROR_TYPE
        self._mark_assigned(operand)   # scanf("%d", &x) gives x a value
        return PointerType(ctype)

    def _increment(self, expr: ast.UnaryOp | ast.PostfixOp, operand: ast.Expr, op: str) -> CType:
        ctype = self._expr(operand)
        action = "increment" if op == "++" else "decrement"
        if _broken(ctype):
            return ERROR_TYPE
        if not _is_lvalue(operand) or isinstance(ctype, ArrayType):
            self.error("lvalue_required", f"lvalue required as {action} operand", expr.loc,
                       f"'{op}' changes a variable, so it needs a variable or array element, "
                       "not a computed value.")
            return ERROR_TYPE
        if not ctype.is_scalar:
            self.error("invalid_operands", f"wrong type argument to {action}", expr.loc,
                       f"'{op}' works on numbers and pointers, not on {_a(ctype)}.")
            return ERROR_TYPE
        self._check_not_const(operand, action)
        self._mark_assigned(operand)
        return ctype

    def _postfix(self, expr: ast.PostfixOp) -> CType:
        return self._increment(expr, expr.operand, expr.op)

    def _mark_assigned(self, target: ast.Expr) -> None:
        if isinstance(target, ast.Identifier):
            entry = self._lookup(target.name)
            if entry is not None:
                entry.assigned = True

    # ---------------------------------------------------------- assignment
    def _assignment(self, expr: ast.Assignment) -> CType:
        target = expr.target
        if expr.op == "=" and isinstance(target, ast.Identifier):
            target_type = self._identifier(target, store=True)
            target.ctype = target_type  # type: ignore[attr-defined]
        else:
            target_type = self._expr(target)
        expr.value = self._operand(expr.value)
        if _broken(target_type) or not self._check_modifiable(target, target_type):
            return ERROR_TYPE if _broken(target_type) else target_type
        if expr.op == "=":
            expr.value = self._convert(expr.value, target_type, _Target("assign"))
        else:
            self._compound_assignment(expr, target_type)
        self._mark_assigned(target)
        return target_type

    def _check_modifiable(self, target: ast.Expr, ctype: CType) -> bool:
        if isinstance(ctype, ArrayType):
            self.error("assign_to_array", "assignment to expression with array type",
                       _start(target), "A whole array cannot be assigned in C. Copy it element "
                                       "by element in a loop, or use strcpy for strings: "
                                       "strcpy(name, \"Ada\").")
            return False
        if not _is_lvalue(target):
            self.error("lvalue_required", "lvalue required as left operand of assignment",
                       _start(target), "The left side of '=' must be a place that can hold a "
                                       "value, like a variable or an array element. This is a "
                                       "computed value. Did you mean '==' to compare?")
            return False
        return self._check_not_const(target, "assignment")

    def _check_not_const(self, target: ast.Expr, action: str) -> bool:
        problem = self._const_problem(target)
        if problem is None:
            return True
        message, name = problem
        self.error("assign_to_const", f"{action} of {message}", _start(target),
                   f"'{name}' is declared const, so it cannot be changed after it is "
                   "initialized. Remove 'const' from its declaration if it needs to change.")
        return False

    def _const_problem(self, target: ast.Expr) -> tuple[str, str] | None:
        """(gcc's wording, the const thing's name) if ``target`` is read-only."""
        if isinstance(target, ast.Identifier):
            entry = self._lookup(target.name)
            if entry is not None and entry.is_const:
                return f"read-only variable '{target.name}'", target.name
            return None
        if isinstance(target, (ast.UnaryOp, ast.Index)):
            base = target.operand if isinstance(target, ast.UnaryOp) else target.base
            holder = _strip(base)
            if isinstance(holder, ast.Identifier):
                entry = self._lookup(holder.name)
                if entry is not None and (entry.points_to_const or
                                          (entry.is_const and entry.ctype.is_array)):
                    return f"read-only location '{_text(target) or holder.name}'", holder.name
            return None
        if isinstance(target, ast.Member):
            holder = _strip(target.base)
            if isinstance(holder, ast.Identifier):
                entry = self._lookup(holder.name)
                const = entry is not None and (entry.points_to_const if target.arrow
                                               else entry.is_const)
                if const:
                    return f"member '{target.name}' in read-only object", holder.name
        return None

    def _compound_assignment(self, expr: ast.Assignment, target_type: CType) -> None:
        op = expr.op[:-1]
        value_type = _type(expr.value)
        if _broken(value_type):
            return
        if op in ("+", "-") and target_type.is_pointer and value_type.is_integer:
            return
        if op in ("<<", ">>"):
            if target_type.is_integer and value_type.is_integer:
                expr.value = self._cast_if_needed(expr.value, integer_promote(value_type))
                return
        elif op in ("%", "&", "|", "^"):
            if target_type.is_integer and value_type.is_integer:
                common = usual_arithmetic_conversions(target_type, value_type)
                expr.value = self._cast_if_needed(expr.value, common)
                return
        elif target_type.is_arithmetic and value_type.is_arithmetic:
            # The value joins the target's arithmetic type; the interpreter converts the
            # result back to the target's own type when it stores it.
            common = usual_arithmetic_conversions(target_type, value_type)
            expr.value = self._cast_if_needed(expr.value, common)
            return
        self.error("invalid_operands", f"invalid operands to binary {op} (have "
                   f"'{format_type(target_type)}' and '{format_type(value_type)}')", expr.loc,
                   f"'{expr.op}' does not work on {_a(target_type)} and "
                   f"{_a(value_type)}.")

    # ---------------------------------------------------------- conversions
    def _convert(self, expr: ast.Expr, to: CType, target: _Target) -> ast.Expr:
        """Convert a checked value to ``to`` as assignment does; warn or err like gcc."""
        src = _type(expr)
        if _broken(src) or to is ERROR_TYPE:
            return expr
        if isinstance(to, StructType) or isinstance(src, StructType):
            if isinstance(to, StructType) and isinstance(src, StructType) and \
                    is_compatible(to, src):
                return expr
            self._incompatible_types(expr, src, to, target)
            return expr
        if to.is_arithmetic and src.is_arithmetic:
            self._check_constant_overflow(expr, src, to)
            return self._cast_if_needed(expr, to)
        if isinstance(to, PointerType):
            return self._convert_to_pointer(expr, src, to, target)
        if to.is_integer and src.is_pointer:
            if to != BOOL:
                self._pointer_integer_warning("int_from_pointer", expr, src, to, target)
            return _wrap(expr, to)
        self._incompatible_types(expr, src, to, target)
        return expr

    def _convert_to_pointer(self, expr: ast.Expr, src: CType, to: PointerType,
                            target: _Target) -> ast.Expr:
        if isinstance(src, PointerType):
            if not _pointers_compatible(to, src):
                self.warn("incompatible_pointer", _incompatible_pointer_message(src, to, target),
                          _start(expr), f"{_a(src, True)} is used where "
                          f"{_a(to)} is expected. The two point to different "
                          "types, so the bytes would be read the wrong way. Check the types, "
                          "or cast explicitly if you really mean it.")
            return self._cast_if_needed(expr, to)
        if src.is_integer:
            if not self._is_null_constant(expr):
                self._pointer_integer_warning("pointer_from_int", expr, src, to, target)
            return _wrap(expr, to)
        self._incompatible_types(expr, src, to, target)
        return expr

    def _pointer_integer_warning(self, code: str, expr: ast.Expr, src: CType, to: CType,
                                 target: _Target) -> None:
        what = "pointer from integer" if code == "pointer_from_int" else "integer from pointer"
        message = _conversion_message(src, to, target) + f" makes {what} without a cast"
        if code == "pointer_from_int":
            friendly = (f"A plain number ('{format_type(src)}') is stored into a pointer "
                        f"('{format_type(to)}'). Pointers hold addresses: did you forget '&' "
                        "to take a variable's address?")
        elif isinstance(_strip(expr), ast.StringLiteral):
            friendly = ("\"...\" is a string, which is an address. For one character use "
                        "single quotes, like 'x'.")
        else:
            friendly = (f"An address ('{format_type(src)}') is stored into "
                        f"{_a(to)}. Did you forget '*' to read the value the "
                        "pointer points to?")
        self.warn(code, message, _start(expr), friendly)

    def _incompatible_types(self, expr: ast.Expr, src: CType, to: CType,
                            target: _Target) -> None:
        source, dest = format_type(src), format_type(to)
        messages = {
            "assign": f"incompatible types when assigning to type '{dest}' from type '{source}'",
            "init": f"incompatible types when initializing type '{dest}' using type '{source}'",
            "arg": f"incompatible type for argument {target.number} of '{target.function}'",
            "return": f"incompatible types when returning type '{source}' but '{dest}' was "
                      "expected",
        }
        friendly = f"{type_phrase(source, True)} cannot be turned into {type_phrase(dest)}."
        if isinstance(src, StructType):
            friendly += " Use one of the struct's members instead, like s.x."
        self.error("incompatible_types", messages[target.kind], _start(expr), friendly)

    def _check_constant_overflow(self, expr: ast.Expr, src: CType, to: CType) -> None:
        if not (to.is_integer and src.is_integer) or to == BOOL:
            return
        value = self._const(expr)
        if value is None or fits(value, to):
            return
        unsigned_target = not getattr(to, "signed", True)
        if unsigned_target and value < 0 and to.size >= integer_promote(src).size:
            return   # 'unsigned u = -1;' is a well-known idiom gcc accepts silently
        changed = wrap_int(value, to)
        low, high = to.min, to.max  # type: ignore[attr-defined]
        prefix = "unsigned conversion" if unsigned_target else "overflow in conversion"
        self.warn("integer_overflow_constant",
                  f"{prefix} from '{format_type(src)}' to '{format_type(to)}' changes value "
                  f"from '{value}' to '{changed}'", _start(expr),
                  f"{_a(to, True)} holds values from {low} to {high}, so {value} does not "
                  f"fit and becomes {changed}. Use a wider type such as int, or a smaller "
                  "value.")

    def _is_null_constant(self, expr: ast.Expr) -> bool:
        """0, or (void *)0: the values C lets you store into any pointer."""
        if isinstance(expr, ast.Cast) and expr.ctype == VOID_PTR:
            expr = expr.expr
        ctype = ast.expression_type(expr)
        return ctype is not None and ctype.is_integer and self._const(expr) == 0

    # ---------------------------------------------------------- calls
    def _call(self, expr: ast.Call) -> CType:
        func = expr.func
        name = func.name if isinstance(func, ast.Identifier) else None
        if name is not None and self._lookup(name) is None:
            self._implicit_declaration(func)  # type: ignore[arg-type]
        callee_type = self._expr(func)
        ftype = callee_type
        if isinstance(ftype, PointerType) and isinstance(ftype.target, FunctionType):
            ftype = ftype.target
        if not isinstance(ftype, FunctionType):
            if not _broken(callee_type):
                label = f"called object '{name}'" if name else "called object"
                self.error("call_non_function",
                           f"{label} is not a function or function pointer", expr.loc,
                           f"'{name or _text(func) or 'this'}' is "
                           f"{_a(callee_type)}, not a function, so it cannot be "
                           "called with (). Is a variable using the same name as a function?")
            expr.args = [self._value(arg) for arg in expr.args]
            return ERROR_TYPE
        self._arguments(expr, ftype, name or _text(func) or "function")
        return ftype.ret

    def _implicit_declaration(self, func: ast.Identifier) -> None:
        """Report a call to an undeclared function, then declare it as old C did."""
        name = func.name
        header = self.registry.header_for(name)
        candidates = [n for n in self._visible_names()
                      if (e := self._lookup(n)) is not None and e.ctype.is_function]
        suggestion = None if header else did_you_mean(name, candidates)
        message = f"implicit declaration of function '{name}'"
        if suggestion is not None:
            message += f"; did you mean '{suggestion}'?"
        if header is not None:
            friendly = (f"'{name}' is declared in <{header}>. Did you forget #include "
                        f"<{header}>? Add it at the top of the file.")
        elif suggestion is not None and suggestion.lower() == name.lower():
            friendly = (f"C is case-sensitive: '{name}' and '{suggestion}' are different "
                        f"names. Did you mean '{suggestion}'?")
        elif suggestion is not None:
            friendly = (f"C does not know a function called '{name}'. Did you mean "
                        f"'{suggestion}'? Otherwise declare it with a prototype before the call.")
        else:
            friendly = (f"C does not know a function called '{name}' at this point. Define it "
                        f"above the call, or add a prototype such as 'int {name}(void);' "
                        "before it.")
        self.error("implicit_declaration", message, func.loc, friendly)
        implicit = ast.FunctionDecl(func.loc, name, FunctionType(INT, (), False, False))
        entry = self._declare_function(implicit, is_definition=False)
        self._scopes[0].setdefault(name, entry)

    def _arguments(self, call: ast.Call, ftype: FunctionType, label: str) -> None:
        values = [self._operand(arg) for arg in call.args]
        fixed = len(ftype.params) if ftype.has_prototype else 0
        if ftype.has_prototype:
            self._check_argument_count(call, ftype, label, len(values))
            for index, param in enumerate(ftype.params[:len(values)]):
                values[index] = self._convert(values[index], param,
                                              _Target("arg", label, index + 1))
        for index in range(fixed, len(values)):
            values[index] = self._default_promotion(values[index])
        call.args = values
        info = self.registry.format_info(label)
        if info is not None:
            self._check_format_call(call, label, info)

    def _check_argument_count(self, call: ast.Call, ftype: FunctionType, label: str,
                              given: int) -> None:
        wanted = len(ftype.params)
        if given >= wanted and (given == wanted or ftype.variadic):
            return
        few = given < wanted
        code = "too_few_arguments" if few else "too_many_arguments"
        self.error(code, f"too {'few' if few else 'many'} arguments to function '{label}'",
                   call.loc, f"'{label}' takes {wanted} argument{'s' if wanted != 1 else ''} "
                             f"but this call passes {given}. Its prototype is "
                             f"'{_signature(label, ftype)}'.", notes=["declared here"])

    def _default_promotion(self, value: ast.Expr) -> ast.Expr:
        """What '...' arguments go through: float -> double, char/short -> int."""
        ctype = _type(value)
        if _broken(ctype):
            return value
        if ctype.is_float and ctype.size == 4:
            return _wrap(value, DOUBLE)
        if ctype.is_integer:
            return self._cast_if_needed(value, integer_promote(ctype))
        return value

    def _check_format_call(self, call: ast.Call, label: str, info: tuple[int, str]) -> None:
        index, kind = info
        if index >= len(call.args):
            return
        literal = _strip(call.args[index])
        if not isinstance(literal, ast.StringLiteral):
            return   # a format in a variable cannot be checked statically
        rest = call.args[index + 1:]
        types = [_type(arg) for arg in rest]
        if any(_broken(t) for t in types):
            return
        names = [_text(_strip(arg)) for arg in rest]
        text = literal.value.decode("latin-1")
        for issue in check_format(kind, text, types, index + 2, names):
            loc = _column_in_literal(literal, issue.offset)
            self.warn("format_mismatch", issue.message, loc, issue.friendly)

    # ---------------------------------------------------------- postfix
    def _index(self, expr: ast.Index) -> CType:
        in_address = self._address_operand is expr
        raw = self._expr(expr.base)
        expr.base = self._decay(expr.base)
        expr.index = self._operand(expr.index)
        base, index = _type(expr.base), _type(expr.index)
        if _broken(base) or _broken(index):
            return ERROR_TYPE
        if index.is_pointer and base.is_integer:
            # '2[a]' is legal C: normalise to 'a[2]' so the interpreter sees one shape.
            expr.base, expr.index = expr.index, expr.base
            base, index = index, base
        if not isinstance(base, PointerType):
            self.error("invalid_subscript",
                       "subscripted value is neither array nor pointer nor vector", expr.loc,
                       f"'[...]' picks an element of an array (or what a pointer points to), "
                       f"but this is {_a(base)}.")
            return ERROR_TYPE
        if not index.is_integer:
            self.error("invalid_subscript", "array subscript is not an integer", expr.loc,
                       f"An array index must be a whole number, not {_a(index)}.")
            return ERROR_TYPE
        if isinstance(raw, ArrayType) and not in_address and not self._unevaluated:
            self._check_bounds(expr, raw)   # 'sizeof a[10]' reads nothing, so it is fine
        return base.target

    def _check_bounds(self, expr: ast.Index, array: ArrayType) -> None:
        if array.length is None:
            return
        value = self._const(expr.index)
        if value is None or 0 <= value < array.length:
            return
        where = "above" if value >= array.length else "below"
        name = _text(_strip(expr.base)) or "the array"
        self.warn("array_bounds", f"array subscript {value} is {where} array bounds of "
                  f"'{format_type(array)}'", expr.loc,
                  f"'{name}' has {array.length} elements, numbered 0 to {array.length - 1}, so "
                  f"index {value} is outside it. Reading or writing there touches memory that "
                  "belongs to something else.")

    def _member(self, expr: ast.Member) -> CType:
        if expr.arrow:
            expr.base = self._operand(expr.base)
            base = _type(expr.base)
            holder = base.target if isinstance(base, PointerType) else None
            if not _broken(base) and not isinstance(holder, StructType):
                self._bad_arrow(expr, base)
                return ERROR_TYPE
        else:
            base = self._expr(expr.base)
            holder = base
            if not _broken(base) and not isinstance(base, StructType):
                self._bad_dot(expr, base)
                return ERROR_TYPE
        if _broken(base):
            return ERROR_TYPE
        assert isinstance(holder, StructType)
        return self._member_type(expr, holder)

    def _bad_arrow(self, expr: ast.Member, base: CType) -> None:
        text = _text(_strip(expr.base)) or "this"
        if isinstance(base, StructType):
            friendly = (f"'->' is for pointers to structs, but '{text}' is a struct itself. Use "
                        f"'.' instead: {text}.{expr.name}.")
        else:
            friendly = f"'->' needs a pointer to a struct, but '{text}' is {_a(base)}."
        self.error("not_a_struct", f"invalid type argument of '->' (have '{format_type(base)}')",
                   expr.loc, friendly)

    def _bad_dot(self, expr: ast.Member, base: CType) -> None:
        text = _text(_strip(expr.base)) or "this"
        if isinstance(base, PointerType) and isinstance(base.target, StructType):
            self.error("not_a_struct", f"'{text}' is a pointer; did you mean to use '->'?",
                       expr.loc, f"'{text}' holds the address of a struct, so reach its "
                                 f"members with '->': {text}->{expr.name}.")
            return
        self.error("not_a_struct",
                   f"request for member '{expr.name}' in something not a structure or union",
                   expr.loc, f"'.{expr.name}' only works on a struct, but '{text}' is "
                             f"{_a(base)}.")

    def _member_type(self, expr: ast.Member, holder: StructType) -> CType:
        if not holder.is_complete:
            self.error("incomplete_type", f"invalid use of undefined type "
                       f"'{format_type(holder)}'", expr.loc,
                       f"'{format_type(holder)}' is declared but its members are not defined "
                       "here, so '.' and '->' cannot reach them. Define the struct before "
                       "this line (or include the header that does).")
            return ERROR_TYPE
        member = holder.member(expr.name)
        if member is None:
            names = [m.name for m in holder.members or []]
            suggestion = did_you_mean(expr.name, names)
            message = f"'{format_type(holder)}' has no member named '{expr.name}'"
            friendly = f"'{format_type(holder)}' has the members {', '.join(names) or 'none'}."
            if suggestion is not None:
                message += f"; did you mean '{suggestion}'?"
                friendly = f"Did you mean '{suggestion}'? " + friendly
            self.error("no_member", message, expr.loc, friendly)
            return ERROR_TYPE
        return member.ctype

    # ---------------------------------------------------------- other expressions
    def _cast(self, expr: ast.Cast) -> CType:
        target = expr.ctype
        expr.expr = self._value(expr.expr)
        src = _type(expr.expr)
        if target.is_void or src is ERROR_TYPE:
            return target
        problem = None
        if src.is_void:
            problem = "void value not ignored as it ought to be"
        elif isinstance(target, StructType):
            problem = "conversion to non-scalar type requested"
        elif isinstance(src, StructType):
            wanted = "a pointer" if target.is_pointer else (
                "a floating-point" if target.is_float else "an integer")
            problem = f"aggregate value used where {wanted} was expected"
        elif target.is_float and src.is_pointer:
            problem = "pointer value used where a floating-point was expected"
        elif target.is_pointer and src.is_float:
            problem = "cannot convert to a pointer type"
        if problem is not None:
            code = "void_value" if src.is_void else "incompatible_types"
            self.error(code, problem, expr.loc, f"{_a(src, True)} cannot be cast to "
                                                f"{_a(target)}.")
        return target

    def _sizeof_type(self, expr: ast.SizeofType) -> CType:
        self._require_complete_for_sizeof(expr.ctype, expr.loc)
        return SIZE_T

    def _sizeof_expr(self, expr: ast.SizeofExpr) -> CType:
        self._unevaluated += 1   # sizeof never evaluates its operand: nothing is read
        try:
            ctype = self._expr(expr.expr)
        finally:
            self._unevaluated -= 1
        if not _broken(ctype):
            self._require_complete_for_sizeof(ctype, expr.loc)
        return SIZE_T

    def _require_complete_for_sizeof(self, ctype: CType, loc: SourceLocation) -> None:
        inner = ctype
        while isinstance(inner, ArrayType) and inner.length is not None:
            inner = inner.element
        incomplete = (isinstance(inner, StructType) and not inner.is_complete) or (
            isinstance(inner, ArrayType) and inner.length is None)
        if incomplete:
            self.error("incomplete_type", f"invalid application of 'sizeof' to incomplete type "
                       f"'{format_type(ctype)}'", loc,
                       f"The size of '{format_type(ctype)}' is not known here because its "
                       "definition has not been seen. Define it before this line.")

    def _conditional(self, expr: ast.Conditional) -> CType:
        expr.cond = self._condition(expr.cond)
        expr.then = self._value(expr.then)
        expr.otherwise = self._value(expr.otherwise)
        a, b = _type(expr.then), _type(expr.otherwise)
        if a is ERROR_TYPE or b is ERROR_TYPE:
            return ERROR_TYPE
        if a.is_arithmetic and b.is_arithmetic:
            common = usual_arithmetic_conversions(a, b)
            expr.then = self._cast_if_needed(expr.then, common)
            expr.otherwise = self._cast_if_needed(expr.otherwise, common)
            return common
        if a.is_void and b.is_void:
            return a
        if isinstance(a, PointerType) and isinstance(b, PointerType):
            return self._conditional_pointers(expr, a, b)
        if isinstance(a, PointerType) and b.is_integer:
            expr.otherwise = self._conditional_pointer_int(expr, expr.otherwise, a)
            return a
        if isinstance(b, PointerType) and a.is_integer:
            expr.then = self._conditional_pointer_int(expr, expr.then, b)
            return b
        if isinstance(a, StructType) and isinstance(b, StructType) and is_compatible(a, b):
            return a
        self.error("incompatible_types", "type mismatch in conditional expression", expr.loc,
                   f"Both sides of ':' must have matching types, but one is "
                   f"{_a(a)} and the other {_a(b)}.")
        return ERROR_TYPE

    def _conditional_pointers(self, expr: ast.Conditional, a: PointerType,
                              b: PointerType) -> CType:
        if _same_type(a, b) or b.target.is_void:
            return a if not b.target.is_void else b
        if a.target.is_void:
            return a
        if not _pointers_compatible(a, b):
            self.warn("incompatible_pointer", "pointer type mismatch in conditional expression",
                      expr.loc, f"One side of ':' is {_a(a)} and the other "
                      f"{_a(b)}. Make both point to the same type.")
        return a

    def _conditional_pointer_int(self, expr: ast.Conditional, number: ast.Expr,
                                 pointer: PointerType) -> ast.Expr:
        if not self._is_null_constant(number):
            self.warn("pointer_from_int", "pointer/integer type mismatch in conditional "
                      "expression", expr.loc, "One side of ':' is a pointer and the other a "
                      "plain number. Use NULL for 'no pointer'.")
        return _wrap(number, pointer)

    def _comma(self, expr: ast.Comma) -> CType:
        for item in expr.exprs[:-1]:
            self._expr(item)
        if not expr.exprs:
            return ERROR_TYPE
        expr.exprs[-1] = self._value(expr.exprs[-1])
        return _type(expr.exprs[-1])

    def _init_list_expr(self, expr: ast.InitList) -> CType:
        # A brace list outside a declaration cannot be parsed; annotate defensively.
        for item in expr.items:
            self._expr(item)
        return ERROR_TYPE

    def _implicit_cast_expr(self, expr: ast.ImplicitCast) -> CType:
        return expr.to

    # ================================================================ initializers
    def _initializer(self, target: CType, init: ast.Expr) -> ast.Expr:
        """Check an initializer for an object of type ``target``; returns the new node."""
        if isinstance(init, ast.InitList):
            self._braced(target, init, top=True)
            return init
        if isinstance(target, ArrayType):
            # 'char s[] = "abc"' copies the characters: no decay, no conversion. Any
            # other form was already reported by the parser.
            self._expr(init)
            return init
        return self._convert(self._operand(init), target, _Target("init"))

    def _braced(self, target: CType, init: ast.InitList, top: bool = False) -> None:
        init.ctype = target  # type: ignore[attr-defined]
        if (isinstance(target, ArrayType) and _is_char(target.element) and len(init.items) == 1
                and isinstance(init.items[0], ast.StringLiteral)):
            self._expr(init.items[0])   # 'char s[8] = {"abc"}' is the string form in braces
            return
        cursor = _InitCursor(init.items)
        if isinstance(target, (ArrayType, StructType)):
            self._fill(target, cursor)
        elif not cursor.done:
            self._fill_scalar(target, cursor)   # 'int x = {5};' is valid C
        if cursor.done:
            return
        # The parser already warns about too many items in a top-level array list.
        if not (top and isinstance(target, ArrayType)):
            kind = ("array" if isinstance(target, ArrayType) else
                    target.keyword if isinstance(target, StructType) else "scalar")
            extra = cursor.items[cursor.pos]
            self.error("too_many_initializers", f"excess elements in {kind} initializer",
                       _start(extra), f"This {kind} has room for fewer values than the list "
                                      "gives. Remove the extra values or make the object "
                                      "bigger.")
        for index in range(cursor.pos, len(cursor.items)):
            if index != cursor.prechecked:
                self._expr(cursor.items[index])

    def _fill(self, target: ArrayType | StructType, cursor: _InitCursor) -> None:
        """Consume items for an aggregate, entering sub-aggregates without braces."""
        if isinstance(target, ArrayType):
            count = 0
            while not cursor.done and (target.length is None or count < target.length):
                self._fill_one(target.element, cursor)
                count += 1
            return
        if not target.is_complete:
            self.error("incomplete_type", f"variable has incomplete type "
                       f"'{format_type(target)}'", _start(cursor.items[0]) if cursor.items
                       else None, f"'{format_type(target)}' is not defined yet, so it cannot "
                                  "be initialized. Define the struct first.")
            return
        members = (target.members or [])[:1] if target.is_union else (target.members or [])
        for member in members:
            if cursor.done:
                return
            self._fill_one(member.ctype, cursor)

    def _fill_one(self, sub: CType, cursor: _InitCursor) -> None:
        item = cursor.items[cursor.pos]
        if isinstance(item, ast.InitList):
            self._braced(sub, item)
            cursor.pos += 1
        elif isinstance(sub, ArrayType) and _is_char(sub.element) and \
                isinstance(_strip(item), ast.StringLiteral):
            literal = _strip(item)
            assert isinstance(literal, ast.StringLiteral)
            if cursor.pos != cursor.prechecked:
                self._expr(literal)
            cursor.items[cursor.pos] = literal   # a peek may have decayed it; undo that
            self._check_string_fits(sub, literal)
            cursor.pos += 1
        elif isinstance(sub, ArrayType):
            self._fill(sub, cursor)   # '{1, 2, 3, 4}' for int[2][2]: braces elided
        elif isinstance(sub, StructType):
            self._fill_struct_item(sub, cursor)
        else:
            self._fill_scalar(sub, cursor)

    def _fill_struct_item(self, sub: StructType, cursor: _InitCursor) -> None:
        """A struct element given without braces: a struct value, or elided braces."""
        pos = cursor.pos
        value = self._operand(cursor.items[pos])
        cursor.items[pos] = value
        cursor.prechecked = pos
        if isinstance(_type(value), StructType):
            cursor.items[pos] = self._convert(value, sub, _Target("init"))
            cursor.pos += 1
            return
        self._fill(sub, cursor)

    def _fill_scalar(self, sub: CType, cursor: _InitCursor) -> None:
        pos = cursor.pos
        item = cursor.items[pos]
        if pos == cursor.prechecked:
            value = item
        elif isinstance(item, ast.InitList):
            self._braced(sub, item)
            cursor.pos += 1
            return
        else:
            value = self._operand(item)
        cursor.items[pos] = self._convert(value, sub, _Target("init"))
        cursor.pos += 1

    def _check_string_fits(self, array: ArrayType, string: ast.StringLiteral) -> None:
        if array.length is not None and len(string.value) > array.length:
            self.warn("string_literal_to_char_array",
                      "initializer-string for array of 'char' is too long", string.loc,
                      f"The array holds {array.length} chars but the string needs "
                      f"{len(string.value) + 1} (one extra for the '\\0' at the end).")


# --- dispatch tables -------------------------------------------------------------------------

_STMT_HANDLERS: dict[type, Callable[[Checker, ast.Node], None]] = {
    ast.CompoundStmt: Checker._compound,
    ast.DeclStmt: Checker._decl_stmt,
    ast.ExprStmt: Checker._expr_stmt,
    ast.IfStmt: Checker._if,
    ast.WhileStmt: Checker._while,
    ast.DoWhileStmt: Checker._do_while,
    ast.ForStmt: Checker._for,
    ast.ReturnStmt: Checker._return,
    ast.BreakStmt: Checker._break,
    ast.ContinueStmt: Checker._continue,
    ast.SwitchStmt: Checker._switch,
    ast.CaseLabel: Checker._case,
    ast.DefaultLabel: Checker._default,
}  # type: ignore[dict-item]

_EXPR_HANDLERS: dict[type, Callable[[Checker, ast.Expr], CType]] = {
    ast.IntLiteral: Checker._int_literal,
    ast.FloatLiteral: Checker._float_literal,
    ast.CharLiteral: Checker._char_literal,
    ast.StringLiteral: Checker._string_literal,
    ast.Identifier: Checker._identifier,
    ast.BinaryOp: Checker._binary,
    ast.UnaryOp: Checker._unary,
    ast.PostfixOp: Checker._postfix,
    ast.Assignment: Checker._assignment,
    ast.Call: Checker._call,
    ast.Index: Checker._index,
    ast.Member: Checker._member,
    ast.Cast: Checker._cast,
    ast.SizeofType: Checker._sizeof_type,
    ast.SizeofExpr: Checker._sizeof_expr,
    ast.Conditional: Checker._conditional,
    ast.Comma: Checker._comma,
    ast.InitList: Checker._init_list_expr,
    ast.ImplicitCast: Checker._implicit_cast_expr,
}  # type: ignore[dict-item]

_COMPARISON_OPS = frozenset({"==", "!=", "<", ">", "<=", ">="})

# Names whose objects live for the whole run, so their addresses are constants.
_STATIC_STORAGE_KINDS = frozenset({"global", "unit_static", "static_local", "function",
                                   "builtin"})


# --- small helpers ---------------------------------------------------------------------------

def _is_string_literal(expr: ast.Expr) -> bool:
    """True for ``"text"`` itself or after the checker made its decay visible."""
    while isinstance(expr, ast.ImplicitCast):
        expr = expr.expr
    return isinstance(expr, ast.StringLiteral)


def _wrap(expr: ast.Expr, to: CType) -> ast.ImplicitCast:
    return ast.ImplicitCast(expr.loc, expr, to)


def _a(ctype: CType, capital: bool = False) -> str:
    """"an 'int'", "a 'struct P'": a type for friendly sentences."""
    return type_phrase(format_type(ctype), capital)


def _type(expr: ast.Expr) -> CType:
    """The checked type of a node's value (``sizeof(T)`` is a size_t, not a T)."""
    ctype = ast.expression_type(expr)
    return ERROR_TYPE if ctype is None else ctype


def _strip(expr: ast.Expr) -> ast.Expr:
    """The expression under any ImplicitCast wrappers."""
    while isinstance(expr, ast.ImplicitCast):
        expr = expr.expr
    return expr


def _broken(ctype: CType | None) -> bool:
    """True for types that already caused an error (or void, reported where it is used)."""
    return ctype is None or ctype is ERROR_TYPE or ctype.is_void


def _is_char(ctype: CType) -> bool:
    return isinstance(ctype, IntType) and ctype.size == 1


def _same_type(a: CType | None, b: CType) -> bool:
    """Same type for conversion purposes: an enum and int share one representation."""
    if a is b or a == b:
        return True
    if isinstance(a, EnumType):
        return b == INT or isinstance(b, EnumType)
    return isinstance(b, EnumType) and a == INT


def _pointers_compatible(a: PointerType, b: PointerType) -> bool:
    """May one pointer be stored into the other without a warning?"""
    if a.target.is_void or b.target.is_void or is_compatible(a, b):
        return True
    # char * versus unsigned char *: gcc only complains under -Wpointer-sign.
    return (isinstance(a.target, IntType) and isinstance(b.target, IntType)
            and a.target.size == b.target.size)


def _is_lvalue(expr: ast.Expr) -> bool:
    """Does ``expr`` designate an object (something with an address)?"""
    if isinstance(expr, ast.Identifier):
        ref = getattr(expr, "ref", None)
        if ref is None or ref.kind in ("function", "enum_const"):
            return False
        return not ref.ctype.is_function   # library variables (errno) are objects too
    if isinstance(expr, (ast.Index, ast.StringLiteral)):
        return True
    if isinstance(expr, ast.UnaryOp):
        return expr.op == "*"
    if isinstance(expr, ast.Member):
        return expr.arrow or _is_lvalue(expr.base)
    return False


def _start(expr: ast.Expr) -> SourceLocation:
    """Where an expression begins in the source (binary nodes sit at their operator)."""
    while True:
        if isinstance(expr, ast.ImplicitCast):
            expr = expr.expr
        elif isinstance(expr, ast.BinaryOp):
            expr = expr.left
        elif isinstance(expr, ast.Assignment):
            expr = expr.target
        elif isinstance(expr, (ast.Index, ast.Member)):
            expr = expr.base
        elif isinstance(expr, ast.PostfixOp):
            expr = expr.operand
        elif isinstance(expr, ast.Conditional):
            expr = expr.cond
        elif isinstance(expr, ast.Comma) and expr.exprs:
            expr = expr.exprs[0]
        else:
            return expr.loc


def _text(expr: ast.Expr) -> str | None:
    """Short source-like text for simple expressions ('p', '*p', 'a[2]', 's.x')."""
    expr = _strip(expr)
    if isinstance(expr, ast.Identifier):
        return expr.name
    if isinstance(expr, ast.IntLiteral):
        return expr.text or str(expr.value)
    if isinstance(expr, ast.UnaryOp) and expr.op == "*":
        inner = _text(expr.operand)
        return f"*{inner}" if inner else None
    if isinstance(expr, ast.Index):
        base, index = _text(expr.base), _text(expr.index)
        return f"{base}[{index}]" if base and index else None
    if isinstance(expr, ast.Member):
        base = _text(expr.base)
        return f"{base}{'->' if expr.arrow else '.'}{expr.name}" if base else None
    return None


def _signature(name: str, ftype: CType) -> str:
    """'int add(int, int)' for messages."""
    if not isinstance(ftype, FunctionType):
        return f"{format_type(ftype)} {name}"
    params = ", ".join(format_type(p) for p in ftype.params)
    if ftype.has_prototype and not ftype.params:
        params = "void"
    if ftype.variadic:
        params += ", ..."
    return f"{format_type(ftype.ret)} {name}({params})"


def _conversion_message(src: CType, to: CType, target: _Target) -> str:
    """gcc's lead-in for a conversion warning, by context."""
    source, dest = format_type(src), format_type(to)
    if target.kind == "assign":
        return f"assignment to '{dest}' from '{source}'"
    if target.kind == "init":
        return f"initialization of '{dest}' from '{source}'"
    if target.kind == "arg":
        return f"passing argument {target.number} of '{target.function}'"
    return f"returning '{source}' from a function with return type '{dest}'"


def _incompatible_pointer_message(src: CType, to: CType, target: _Target) -> str:
    source, dest = format_type(src), format_type(to)
    if target.kind == "assign":
        return f"assignment to '{dest}' from incompatible pointer type '{source}'"
    if target.kind == "init":
        return f"initialization of '{dest}' from incompatible pointer type '{source}'"
    if target.kind == "arg":
        return (f"passing argument {target.number} of '{target.function}' from incompatible "
                "pointer type")
    return f"returning '{source}' from a function with incompatible return type '{dest}'"


def _column_in_literal(literal: ast.StringLiteral, offset: int | None) -> SourceLocation:
    """Point at the '%' of a conversion inside a format string when we can be exact."""
    loc = literal.loc
    raw = literal.text
    if offset is None or not raw.startswith('"'):
        return loc
    prefix = raw[1:1 + offset]
    if "\\" in prefix or '"' in prefix or len(prefix) < offset:
        return loc   # escapes or joined literals: the start of the string is honest
    return SourceLocation(loc.file, loc.line, loc.column + 1 + offset)


def _initializer_items(init: ast.Expr) -> Iterable[ast.Expr]:
    """The scalar values inside a (possibly nested) brace initializer."""
    if isinstance(init, ast.InitList):
        for item in init.items:
            yield from _initializer_items(item)
    else:
        yield init


def _assigns_name(root: ast.Node, name: str) -> bool:
    """Does anything under ``root`` store into, increment or take the address of ``name``?"""
    for node in ast.walk(root):
        if isinstance(node, ast.Assignment):
            target = node.target
        elif isinstance(node, (ast.UnaryOp, ast.PostfixOp)) and node.op in ("&", "++", "--"):
            target = node.operand
        else:
            continue
        if isinstance(target, ast.Identifier) and target.name == name:
            return True
    return False


def _has_no_effect(expr: ast.Expr) -> bool:
    """True for an expression statement that changes nothing ('x == 5;')."""
    expr = _strip(expr)
    if isinstance(expr, (ast.Assignment, ast.Call, ast.PostfixOp, ast.Comma, ast.Cast,
                         ast.InitList)):
        return False
    if isinstance(expr, ast.UnaryOp):
        return expr.op not in ("++", "--") and _has_no_effect(expr.operand)
    if isinstance(expr, ast.BinaryOp):
        return _has_no_effect(expr.left) and _has_no_effect(expr.right)
    if isinstance(expr, ast.Conditional):
        return all(_has_no_effect(e) for e in (expr.cond, expr.then, expr.otherwise))
    if isinstance(expr, (ast.Index, ast.Member)):
        return _has_no_effect(expr.base)
    return isinstance(expr, (ast.Identifier, ast.IntLiteral, ast.FloatLiteral,
                             ast.CharLiteral, ast.StringLiteral, ast.SizeofExpr,
                             ast.SizeofType))


# --- flow analysis for control_reaches_end ------------------------------------------------------

@dataclass(frozen=True)
class _Flow:
    """How control can leave one statement.

    ``falls``: it can reach its own end and carry on with the next statement.
    ``breaks`` / ``continues``: a *reachable* break or continue inside it leaves the
    enclosing loop (or, for break, switch). Only reachable ones count, so the habit of
    writing 'return x; break;' in a switch does not make the switch look escapable.
    """

    falls: bool
    breaks: bool = False
    continues: bool = False


_STOPS = _Flow(False)
_FALLS = _Flow(True)


def can_fall_through(stmt: ast.Node) -> bool:
    """Can execution reach the end of ``stmt`` normally (and so continue after it)?

    Simple and conservative, like gcc's -Wreturn-type: return, exit(), infinite loops
    without a reachable break, if/else and switch-with-default where every branch stops.
    """
    return _flow(stmt).falls


def _flow(stmt: ast.Node) -> _Flow:
    handler = _FLOW.get(type(stmt))
    return handler(stmt) if handler is not None else _FALLS


def _flow_block(stmt: ast.CompoundStmt) -> _Flow:
    reachable, breaks, continues = True, False, False
    for item in stmt.items:
        if isinstance(item, (ast.CaseLabel, ast.DefaultLabel)):
            reachable = True   # a label is a way in, even right after a return
        if not reachable:
            continue           # dead code: its breaks and continues never run
        inner = _flow(item)
        breaks, continues = breaks or inner.breaks, continues or inner.continues
        reachable = inner.falls
    return _Flow(reachable, breaks, continues)


def _flow_if(stmt: ast.IfStmt) -> _Flow:
    then = _flow(stmt.then)
    otherwise = _flow(stmt.otherwise) if stmt.otherwise is not None else _FALLS
    return _Flow(then.falls or otherwise.falls, then.breaks or otherwise.breaks,
                 then.continues or otherwise.continues)


def _flow_expr(stmt: ast.ExprStmt) -> _Flow:
    expr = _strip(stmt.expr)
    if isinstance(expr, ast.Call):
        callee = _strip(expr.func)
        if isinstance(callee, ast.Identifier) and callee.name in NORETURN_FUNCTIONS:
            return _STOPS
    return _FALLS


def _always_true(cond: ast.Expr | None) -> bool:
    return cond is None or eval_const_expr(cond) not in (None, 0)


def _flow_loop(stmt: ast.WhileStmt | ast.ForStmt) -> _Flow:
    # The loop owns its breaks and continues: none of them escape further out.
    body = _flow(stmt.body)
    return _Flow(not _always_true(stmt.cond) or body.breaks)


def _flow_do(stmt: ast.DoWhileStmt) -> _Flow:
    body = _flow(stmt.body)
    if body.breaks:
        return _FALLS
    if _always_true(stmt.cond):
        return _STOPS
    # The condition is only tested when the body ends normally or continues.
    return _Flow(body.falls or body.continues)


def _flow_switch(stmt: ast.SwitchStmt) -> _Flow:
    body = _flow(stmt.body)
    falls = not _has_default(stmt.body) or body.breaks or body.falls
    # A switch owns its breaks, but a continue inside it belongs to the enclosing loop.
    return _Flow(falls, False, body.continues)


def _flow_label(stmt: ast.CaseLabel | ast.DefaultLabel) -> _Flow:
    return _flow(stmt.body)


_FLOW: dict[type, Callable[[ast.Node], _Flow]] = {
    ast.ReturnStmt: lambda stmt: _STOPS,
    ast.BreakStmt: lambda stmt: _Flow(False, breaks=True),
    ast.ContinueStmt: lambda stmt: _Flow(False, continues=True),
    ast.CompoundStmt: _flow_block,
    ast.IfStmt: _flow_if,
    ast.ExprStmt: _flow_expr,
    ast.WhileStmt: _flow_loop,
    ast.ForStmt: _flow_loop,
    ast.DoWhileStmt: _flow_do,
    ast.SwitchStmt: _flow_switch,
    ast.CaseLabel: _flow_label,
    ast.DefaultLabel: _flow_label,
}  # type: ignore[dict-item]


def _has_default(stmt: ast.Node) -> bool:
    """Does this switch body have a 'default:' of its own (not a nested switch's)?"""
    if isinstance(stmt, ast.DefaultLabel):
        return True
    if isinstance(stmt, (ast.SwitchStmt, ast.Expr)):
        return False
    return any(_has_default(child) for child in stmt.children())
