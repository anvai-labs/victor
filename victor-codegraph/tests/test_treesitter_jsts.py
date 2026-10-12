"""JS/TS extraction tests — the donor stub fix.

Guarded with importorskip so the suite stays green where the grammar pack isn't
installed; where it is, these assert the stub is genuinely replaced (non-empty symbols).
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from victor_codegraph import CodeSymbolType, parse  # noqa: E402
from victor_codegraph.treesitter_parser import (
    GrammarUnavailable,
    parse_treesitter,
)  # noqa: E402

JS = """\
import { x } from "./x";

export function add(a, b) {
  return a + b;
}

const mul = (a, b) => a * b;

class Calc {
  constructor(seed) {
    this.seed = seed;
  }
  run(n) {
    return add(n, this.seed);
  }
}
"""

TS = """\
export function greet(name: string): string {
  return `hi ${name}`;
}

class Service {
  handle(req: Request): Response {
    return new Response();
  }
}
"""


def _names(language: str, src: str):
    try:
        parsed = parse_treesitter(src, f"f.{language}", language)
    except GrammarUnavailable:
        pytest.skip(f"{language} grammar not installed")
    return {s.simple_name for s in parsed.symbols}, parsed


def test_javascript_functions_classes_methods_arrow():
    names, parsed = _names("javascript", JS)
    # The donor stub returned []; here we must see real symbols.
    assert {"add", "Calc", "run"} <= names
    assert "mul" in names  # const arrow function
    assert any(s.symbol_type == CodeSymbolType.CLASS for s in parsed.symbols)
    assert any("import" in i for i in parsed.imports)


def test_typescript_functions_and_methods():
    names, _ = _names("typescript", TS)
    assert {"greet", "Service", "handle"} <= names


def test_chunk_jsts_is_size_capped():
    parsed = parse(JS, language="javascript", file_path="f.js")
    if not parsed.symbols:
        pytest.skip("javascript grammar not installed")
    assert parsed.symbols  # routed through tree-sitter, not the empty stub


@pytest.mark.parametrize(
    "annotation",
    [
        "export type X<T> = T extends object ? T : never;",
        "type Options = Record<string, object>;",
        "function accept(value: object): object { return value; }",
    ],
)
def test_typescript_object_keyword_is_not_an_object_literal_scope(annotation):
    source = annotation + "\nconst handlers = { run() { return 1; } };"
    parsed = parse(source, file_path="case.ts")
    shifted = parse("// heading\n\n" + source, file_path="case.ts")
    assert parsed.status.value == "success"
    assert not parsed.diagnostics
    methods = [symbol for symbol in parsed.symbols if symbol.simple_name == "run"]
    assert len(methods) == 1
    assert methods[0].scope_chain
    assert {symbol.id for symbol in parsed.symbols} == {symbol.id for symbol in shifted.symbols}


def test_inline_import_generic_grammar_limitation_stays_explicit():
    # The pinned grammar rejects this valid TypeScript import type in a generic
    # call. Do not hide the partial parse; an equivalent named alias is supported.
    inline = parse("const value = load<typeof import('child_process')>();", file_path="case.ts")
    assert inline.status.value == "partial"
    assert "syntax_error" in {diagnostic.code for diagnostic in inline.diagnostics}
    aliased = parse(
        "type ChildProcessModule = typeof import('child_process');"
        "const value = load<ChildProcessModule>();",
        file_path="case.ts",
    )
    assert aliased.status.value == "success"
    assert not aliased.diagnostics
