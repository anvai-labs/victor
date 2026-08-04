"""Relation extraction on the tree-sitter path (JS/TS focus).

Pins the multi-language parity contract: CALLS with in-file resolution
(confidence 1.0) and retained external targets (0.5), 1-based call sites,
EXTENDS/IMPLEMENTS from heritage clauses, full-scope-path CONTAINS, and
nested-symbol collection inside function bodies.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from victor_codegraph import CodeRelationType, CodeSymbolType  # noqa: E402
from victor_codegraph.treesitter_parser import (  # noqa: E402
    GrammarUnavailable,
    parse_treesitter,
)


def _parse(language: str, src: str, ext: str):
    try:
        return parse_treesitter(src, f"f.{ext}", language)
    except GrammarUnavailable:
        pytest.skip(f"{language} grammar not installed")


def _by_name(parsed):
    return {s.simple_name: s for s in parsed.symbols}


def _rels(parsed, rel_type):
    return [r for r in parsed.relations if r.relation_type == rel_type]


JS = """\
import { helper } from "./helper";

class Animal {}

class Dog extends Animal {
  constructor() {
    super();
  }
  bark() {
    growl();
    unknownExternal();
  }
}

function growl() {
  console.log("grr");
}
"""


def test_js_calls_resolved_and_retained():
    parsed = _parse("javascript", JS, "js")
    syms = _by_name(parsed)
    calls = _rels(parsed, CodeRelationType.CALLS)

    resolved = [r for r in calls if r.context == "growl"]
    assert len(resolved) == 1
    assert resolved[0].from_symbol_id == syms["bark"].id
    assert resolved[0].to_symbol_id == syms["growl"].id
    assert resolved[0].confidence == 1.0

    external = [r for r in calls if r.context == "unknownExternal"]
    assert len(external) == 1
    assert external[0].to_symbol_id == "unknownExternal"  # retained bare name
    assert external[0].confidence == 0.5


def test_js_call_sites_are_one_based():
    parsed = _parse("javascript", JS, "js")
    calls = _rels(parsed, CodeRelationType.CALLS)
    growl_call = next(r for r in calls if r.context == "growl")
    # `growl();` is on line 10 of the source (1-based).
    assert growl_call.call_site is not None
    assert growl_call.call_site.start_line == 10


def test_js_extends_resolved_in_file():
    parsed = _parse("javascript", JS, "js")
    syms = _by_name(parsed)
    extends = _rels(parsed, CodeRelationType.EXTENDS)
    assert len(extends) == 1
    assert extends[0].from_symbol_id == syms["Dog"].id
    assert extends[0].to_symbol_id == syms["Animal"].id
    assert extends[0].confidence == 1.0


def test_js_contains_class_members():
    parsed = _parse("javascript", JS, "js")
    syms = _by_name(parsed)
    contains = _rels(parsed, CodeRelationType.CONTAINS)
    pairs = {(r.from_symbol_id, r.to_symbol_id) for r in contains}
    assert (syms["Dog"].id, syms["bark"].id) in pairs
    assert (syms["Dog"].id, syms["constructor"].id) in pairs


def test_js_imports_collected():
    parsed = _parse("javascript", JS, "js")
    assert any("helper" in i for i in parsed.imports)


TS = """\
interface Speaker {
  speak(): void;
}

class Base<T> {}

class Dog extends Base<string> implements Speaker {
  speak(): void {
    helper();
  }
}

function helper(): void {}
"""


def test_ts_extends_and_implements_with_generics():
    parsed = _parse("typescript", TS, "ts")
    syms = _by_name(parsed)
    extends = _rels(parsed, CodeRelationType.EXTENDS)
    implements = _rels(parsed, CodeRelationType.IMPLEMENTS)

    assert [r.context for r in extends] == ["Base"]  # generic arg `string` skipped
    assert extends[0].to_symbol_id == syms["Base"].id
    assert [r.context for r in implements] == ["Speaker"]
    assert implements[0].to_symbol_id == syms["Speaker"].id
    assert implements[0].from_symbol_id == syms["Dog"].id


NESTED_JS = """\
function outer() {
  function inner() {
    deep();
  }
  const xs = [1].map(x => inner(x));
  return inner;
}
"""


def test_js_nested_functions_collected_with_scope():
    parsed = _parse("javascript", NESTED_JS, "js")
    syms = _by_name(parsed)
    # Nested named function is collected (previously dropped entirely).
    assert "inner" in syms
    assert syms["inner"].scope_chain == ["outer"]
    assert syms["inner"].symbol_type == CodeSymbolType.FUNCTION  # not METHOD
    # The unparenthesized arrow param must NOT become a symbol name.
    assert "x" not in syms

    contains = _rels(parsed, CodeRelationType.CONTAINS)
    pairs = {(r.from_symbol_id, r.to_symbol_id) for r in contains}
    assert (syms["outer"].id, syms["inner"].id) in pairs


def test_js_anonymous_callback_calls_belong_to_enclosing_function():
    parsed = _parse("javascript", NESTED_JS, "js")
    syms = _by_name(parsed)
    calls = _rels(parsed, CodeRelationType.CALLS)
    # `inner(x)` sits in an anonymous arrow inside outer -> attributed to outer.
    inner_call = next(r for r in calls if r.context == "inner")
    assert inner_call.from_symbol_id == syms["outer"].id
    assert inner_call.to_symbol_id == syms["inner"].id
    # `deep()` sits in the named nested function -> attributed to inner.
    deep_call = next(r for r in calls if r.context == "deep")
    assert deep_call.from_symbol_id == syms["inner"].id
    assert deep_call.confidence == 0.5


def test_recursion_is_retained_as_graph_evidence():
    src = "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n"
    parsed = _parse("javascript", src, "js")
    calls = _rels(parsed, CodeRelationType.CALLS)
    assert len(calls) == 2
    assert all(r.from_symbol_id == r.to_symbol_id for r in calls)
