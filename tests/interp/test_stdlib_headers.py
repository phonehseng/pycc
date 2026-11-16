"""The header texts: they compile, they are guarded, and they match the registry."""
from __future__ import annotations

import pytest

from pycc.interp import astnodes as ast
from pycc.interp.c_types import ArrayType, IntType, StructType, format_type
from pycc.interp.lexer import tokenize
from pycc.interp.parser import parse, parse_text
from pycc.interp.preprocessor import DictResolver, preprocess
from pycc.interp.stdlib import load_headers, make_registry, system_headers
from pycc.interp.stdlib.errno_ import ERRNO_MACROS
from pycc.interp.stdlib.headers import STANDARD_HEADERS, header_text
from pycc.interp.toolchain import SystemHeaders

HEADERS = load_headers()

# What a translation unit needs to parse a lone prototype: the typedefs headers provide.
TYPEDEF_PRELUDE = ("typedef unsigned long size_t;\n"
                   "typedef struct _CVFILE { int _id; } FILE;\n"
                   "typedef long time_t;\ntypedef long clock_t;\n")


def compile_including(*names: str, body: str = "int main(void) { return 0; }\n"):
    """Preprocess, lex and parse a main.c that includes the given headers."""
    text = "".join(f"#include <{name}>\n" for name in names) + body
    pre = preprocess(text, "main.c", DictResolver(system_headers=HEADERS))
    lex_diagnostics: list = []
    tokens = tokenize(pre.lines, lex_diagnostics)
    parsed = parse(tokens, "main.c")
    return pre, lex_diagnostics + parsed.diagnostics, parsed.ast


def declarations(name: str) -> dict[str, ast.Decl]:
    _, _, unit = compile_including(name)
    return {decl.name: decl for decl in unit.decls if hasattr(decl, "name") and decl.name}


def normalized(text: str) -> str:
    return " ".join(text.replace(";", " ").split())


def test_all_twelve_headers_ship() -> None:
    assert set(HEADERS) == {"stdio.h", "stdlib.h", "string.h", "ctype.h", "math.h", "errno.h",
                            "stdbool.h", "stdint.h", "limits.h", "stddef.h", "assert.h", "time.h"}


def test_system_headers_returns_toolchain_object() -> None:
    headers = system_headers()
    assert isinstance(headers, SystemHeaders)
    assert headers.names() == sorted(STANDARD_HEADERS)
    assert headers.get("stdio.h") == header_text("stdio.h")
    assert headers.get("farm.h") is None


def test_header_text_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        header_text("conio.h")


@pytest.mark.parametrize("name", STANDARD_HEADERS)
def test_header_compiles_in_empty_main(name: str) -> None:
    pre, diagnostics, _ = compile_including(name)
    assert pre.diagnostics == []
    assert diagnostics == []


def test_all_headers_together_twice_compile() -> None:
    pre, diagnostics, _ = compile_including(*STANDARD_HEADERS, *STANDARD_HEADERS)
    assert pre.diagnostics == []
    assert diagnostics == []


@pytest.mark.parametrize("name", STANDARD_HEADERS)
def test_header_has_include_guard(name: str) -> None:
    pre, _, _ = compile_including(name)
    assert pre.guards[f"/usr/include/{name}"] is True


@pytest.mark.parametrize("name", STANDARD_HEADERS)
def test_header_never_mentions_unsupported_preprocessor_operators(name: str) -> None:
    assert "##" not in HEADERS[name]
    assert "__VA_ARGS__" not in HEADERS[name]


def test_every_registered_prototype_appears_verbatim_in_its_header() -> None:
    registry = make_registry()
    for spec in registry.functions.values():
        header = normalized(HEADERS[spec.header])
        assert normalized(spec.prototype) in header, spec.name


def test_every_header_function_is_registered_for_that_header() -> None:
    registry = make_registry()
    for name in STANDARD_HEADERS:
        for decl in declarations(name).values():
            if isinstance(decl, ast.FunctionDecl):
                assert registry.header_for(decl.name) == name, decl.name


def test_registered_prototypes_parse_to_the_header_types() -> None:
    registry = make_registry()
    for spec in registry.functions.values():
        header_decl = declarations(spec.header)[spec.name]
        result = parse_text(TYPEDEF_PRELUDE + spec.prototype + "\n", "proto.c")
        assert result.diagnostics == [], spec.name
        proto_decl = result.ast.decls[-1]
        assert isinstance(proto_decl, ast.FunctionDecl)
        assert format_type(proto_decl.ftype) == format_type(header_decl.ftype), spec.name


def test_header_declares_each_library_variable() -> None:
    registry = make_registry()
    for name, variable in registry.variables.items():
        decl = declarations(variable.header)[name]
        assert isinstance(decl, ast.VarDecl)
        assert decl.is_extern or decl.storage == "extern", name


def test_file_typedef_layout() -> None:
    decls = declarations("stdio.h")
    file_type = decls["FILE"].ctype
    assert isinstance(file_type, StructType) and file_type.tag == "_CVFILE"
    assert [m.name for m in file_type.members] == ["_id"] and file_type.size == 4


@pytest.mark.parametrize("name,size,signed", [
    ("int8_t", 1, True), ("uint8_t", 1, False), ("int16_t", 2, True), ("uint16_t", 2, False),
    ("int32_t", 4, True), ("uint32_t", 4, False), ("int64_t", 8, True), ("uint64_t", 8, False),
])
def test_exact_width_integer_typedefs(name: str, size: int, signed: bool) -> None:
    ctype = declarations("stdint.h")[name].ctype
    assert isinstance(ctype, IntType) and (ctype.size, ctype.signed) == (size, signed)


def test_size_t_is_unsigned_long_everywhere() -> None:
    for name in ("stdio.h", "stdlib.h", "string.h", "stddef.h"):
        assert format_type(declarations(name)["size_t"].ctype) == "unsigned long"


def macro_equals(macro: str, header: str, value: int) -> bool:
    """Ask the preprocessor's own #if arithmetic whether ``macro == value``."""
    body = f"#if ({macro}) == ({value})\nint yes;\n#endif\n"
    _, _, unit = compile_including(header, body=body)
    return any(getattr(decl, "name", None) == "yes" for decl in unit.decls)


@pytest.mark.parametrize("macro,header,value", [
    ("EOF", "stdio.h", -1), ("RAND_MAX", "stdlib.h", 32767), ("EXIT_FAILURE", "stdlib.h", 1),
    ("INT_MAX", "limits.h", 2**31 - 1), ("INT_MIN", "limits.h", -2**31),
    ("CHAR_BIT", "limits.h", 8),
    ("SHRT_MIN", "limits.h", -32768), ("UCHAR_MAX", "limits.h", 255),
    ("LONG_MIN", "limits.h", -2**63), ("INT32_MIN", "stdint.h", -2**31),
    ("true", "stdbool.h", 1), ("false", "stdbool.h", 0),
])
def test_macro_values(macro: str, header: str, value: int) -> None:
    assert macro_equals(macro, header, value)
    assert not macro_equals(macro, header, value + 1)


def test_errno_macros_match_the_python_table() -> None:
    pre, _, _ = compile_including("errno.h")
    defined = {name: int(pre.macros[name].body) for name in ERRNO_MACROS}
    assert defined == ERRNO_MACROS


def test_null_and_bool_macros() -> None:
    pre, _, _ = compile_including("stdio.h", "stdbool.h")
    assert pre.macros["NULL"].body == "((void *)0)"
    assert pre.macros["bool"].body == "_Bool"


def test_assert_header_enables_the_preprocessor_rewrite() -> None:
    pre, _, _ = compile_including("assert.h")
    assert "__CV_ASSERT__" in pre.macros
    assert "assert" not in pre.macros


def test_ndebug_turns_assert_into_nothing() -> None:
    text = "#define NDEBUG\n#include <assert.h>\nint main(void) { assert(0); return 0; }\n"
    pre = preprocess(text, "main.c", DictResolver(system_headers=HEADERS))
    assert "__CV_ASSERT__" not in pre.macros
    assert "((void)0)" in pre.text


def test_program_using_many_headers_parses() -> None:
    body = """
int compare(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
int main(void) {
    int values[4] = {3, 1, 2, 0};
    char line[64];
    FILE *f = fopen("data.txt", "r");
    if (f == NULL) { perror("fopen"); return EXIT_FAILURE; }
    while (fgets(line, sizeof line, f) != NULL) { printf("%zu\\n", strlen(line)); }
    fclose(f);
    qsort(values, 4, sizeof values[0], compare);
    bool ok = isdigit('7') && sqrt(16.0) == 4.0 && errno == 0 && INT_MAX > 0;
    int32_t small = INT32_MAX;
    time_t now = time(NULL);
    return ok && small && now >= 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
"""
    pre, diagnostics, unit = compile_including(*STANDARD_HEADERS, body=body)
    assert pre.diagnostics == [] and diagnostics == []
    main = [d for d in unit.decls if isinstance(d, ast.FunctionDef) and d.name == "main"]
    assert main and isinstance(declarations("stdio.h")["stdin"].ctype.target, StructType)
    assert not isinstance(declarations("stdio.h")["stdin"].ctype, ArrayType)
