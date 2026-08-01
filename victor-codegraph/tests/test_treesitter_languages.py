"""Per-language extraction parity fixtures (Go, Rust, Java, C, C++).

Each language asserts the same contract shape: named symbols with correct
types and scope, CALLS resolved in-file (1.0) with externals retained (0.5),
inheritance edges, CONTAINS for members, and collected import strings.
Grammar-gated: skips where the language pack (or a grammar) is missing.
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


GO = """\
package main

import "fmt"

type Base struct{}

type Svc struct {
    Base
    name string
}

func (s *Svc) Run() {
    fmt.Println(s.name)
    helper()
}

func helper() {}
"""


def test_go_structs_methods_and_embedding():
    parsed = _parse("go", GO, "go")
    syms = _by_name(parsed)

    assert syms["Base"].symbol_type == CodeSymbolType.STRUCT
    assert syms["Svc"].symbol_type == CodeSymbolType.STRUCT
    assert syms["Run"].symbol_type == CodeSymbolType.METHOD
    assert syms["Run"].metadata.get("receiver") == "Svc"
    assert syms["helper"].symbol_type == CodeSymbolType.FUNCTION

    # struct embedding => EXTENDS
    extends = _rels(parsed, CodeRelationType.EXTENDS)
    assert [(r.from_symbol_id, r.to_symbol_id) for r in extends] == [
        (syms["Svc"].id, syms["Base"].id)
    ]

    # receiver => CONTAINS struct -> method
    contains = _rels(parsed, CodeRelationType.CONTAINS)
    assert (syms["Svc"].id, syms["Run"].id) in {
        (r.from_symbol_id, r.to_symbol_id) for r in contains
    }

    calls = {r.context: r for r in _rels(parsed, CodeRelationType.CALLS)}
    assert calls["helper"].to_symbol_id == syms["helper"].id
    assert calls["helper"].confidence == 1.0
    assert calls["Println"].confidence == 0.5  # external, retained

    assert any("fmt" in i for i in parsed.imports)


RUST = """\
use std::fmt;

trait Speak {
    fn speak(&self);
}

struct Dog {
    name: String,
}

impl Speak for Dog {
    fn speak(&self) {
        helper();
    }
}

impl Dog {
    fn new(name: String) -> Self {
        Dog { name }
    }
}

fn helper() {}
"""


def test_rust_traits_impls_and_methods():
    parsed = _parse("rust", RUST, "rs")
    syms = _by_name(parsed)

    assert syms["Speak"].symbol_type == CodeSymbolType.TRAIT
    assert syms["Dog"].symbol_type == CodeSymbolType.STRUCT
    # impl blocks emit NO symbol of their own (they'd collide with the struct's
    # FQN/OID); their members scope to the type.
    dogs = [s for s in parsed.symbols if s.simple_name == "Dog"]
    assert len(dogs) == 1
    assert syms["speak"].symbol_type == CodeSymbolType.METHOD
    assert syms["speak"].scope_chain == ["Dog"]
    assert syms["new"].symbol_type == CodeSymbolType.CONSTRUCTOR
    assert syms["new"].scope_chain == ["Dog"]

    # `impl Speak for Dog` => Dog IMPLEMENTS Speak (attached to the TYPE).
    implements = _rels(parsed, CodeRelationType.IMPLEMENTS)
    assert [(r.from_symbol_id, r.to_symbol_id) for r in implements] == [
        (syms["Dog"].id, syms["Speak"].id)
    ]

    contains = {
        (r.from_symbol_id, r.to_symbol_id) for r in _rels(parsed, CodeRelationType.CONTAINS)
    }
    assert (syms["Dog"].id, syms["speak"].id) in contains
    assert (syms["Dog"].id, syms["new"].id) in contains

    calls = {r.context: r for r in _rels(parsed, CodeRelationType.CALLS)}
    assert calls["helper"].confidence == 1.0

    assert any("use std::fmt" in i for i in parsed.imports)


JAVA = """\
import java.util.List;

interface Speaker {
    void speak();
}

class Base {}

public class Dog extends Base implements Speaker {
    public void speak() {
        helper();
        System.out.println("woof");
    }

    static void helper() {}
}
"""


def test_java_inheritance_and_calls():
    parsed = _parse("java", JAVA, "java")
    syms = _by_name(parsed)

    assert syms["Speaker"].symbol_type == CodeSymbolType.INTERFACE
    assert syms["Dog"].symbol_type == CodeSymbolType.CLASS
    dog_members = [s for s in parsed.symbols if s.scope_chain == ["Dog"]]
    assert {s.simple_name for s in dog_members} == {"speak", "helper"}

    # exactly ONE edge each (java clauses are reachable as child AND field —
    # regression guard against double emission)
    extends = _rels(parsed, CodeRelationType.EXTENDS)
    implements = _rels(parsed, CodeRelationType.IMPLEMENTS)
    assert [(r.from_symbol_id, r.to_symbol_id) for r in extends] == [
        (syms["Dog"].id, syms["Base"].id)
    ]
    assert [(r.from_symbol_id, r.to_symbol_id) for r in implements] == [
        (syms["Dog"].id, syms["Speaker"].id)
    ]

    calls = {r.context: r for r in _rels(parsed, CodeRelationType.CALLS)}
    assert calls["helper"].confidence == 1.0
    assert calls["println"].confidence == 0.5

    assert any("java.util.List" in i for i in parsed.imports)


C = """\
#include <stdio.h>

struct point {
    int x;
    int y;
};

int add(int a, int b) {
    return a + b;
}

int main(void) {
    struct point p;
    int r = add(1, 2);
    printf("%d", r);
    return 0;
}
"""


def test_c_functions_structs_and_calls():
    parsed = _parse("c", C, "c")
    syms = _by_name(parsed)

    assert syms["point"].symbol_type == CodeSymbolType.STRUCT
    assert syms["add"].symbol_type == CodeSymbolType.FUNCTION
    assert syms["main"].symbol_type == CodeSymbolType.FUNCTION
    # `struct point p;` is a reference, not a second definition
    assert len([s for s in parsed.symbols if s.simple_name == "point"]) == 1

    calls = {r.context: r for r in _rels(parsed, CodeRelationType.CALLS)}
    assert calls["add"].to_symbol_id == syms["add"].id
    assert calls["add"].confidence == 1.0
    assert calls["printf"].confidence == 0.5

    assert any("stdio.h" in i for i in parsed.imports)


CPP = """\
#include <string>

class Base {
 public:
    void run();
};

class Derived : public Base {
 public:
    void go() {
        run();
    }
};
"""


def test_cpp_class_inheritance_and_methods():
    parsed = _parse("cpp", CPP, "cpp")
    syms = _by_name(parsed)

    assert syms["Base"].symbol_type == CodeSymbolType.CLASS
    assert syms["Derived"].symbol_type == CodeSymbolType.CLASS
    assert syms["go"].symbol_type == CodeSymbolType.METHOD
    assert syms["go"].scope_chain == ["Derived"]

    extends = _rels(parsed, CodeRelationType.EXTENDS)
    assert [(r.from_symbol_id, r.to_symbol_id) for r in extends] == [
        (syms["Derived"].id, syms["Base"].id)
    ]

    calls = {r.context: r for r in _rels(parsed, CodeRelationType.CALLS)}
    assert calls["run"].confidence == 0.5  # declaration-only member, unresolved


# ── OID stability twin of test_symbol_oid_parity.py for the tree-sitter path ──

STABLE_JS = """\
import { a } from "./a";

export function add(x, y) {
  return x + y;
}

class Calc {
  constructor(seed) {
    this.seed = seed;
  }
  run(n) {
    return add(n, this.seed);
  }
}

const mul = (x, y) => x * y;
"""


def _oids(parsed, repo="r1"):
    from victor_codegraph import stable_symbol_oid

    return {
        s.simple_name: stable_symbol_oid(repo, s.language, s.fully_qualified_name, s.signature)
        for s in parsed.symbols
    }


def test_treesitter_oids_survive_line_shift():
    parsed_a = _parse("javascript", STABLE_JS, "js")
    shifted = "// header comment\n\n\n" + STABLE_JS
    parsed_b = _parse("javascript", shifted, "js")
    assert _oids(parsed_a) == _oids(parsed_b)


def test_treesitter_oids_no_collisions():
    parsed = _parse("javascript", STABLE_JS, "js")
    from victor_codegraph import stable_symbol_oid

    oids = [
        stable_symbol_oid("r1", s.language, s.fully_qualified_name, s.signature)
        for s in parsed.symbols
    ]
    assert len(oids) == len(set(oids))


def test_treesitter_fqns_pinned_for_previously_emitted_symbols():
    """Hierarchy work must be ADDITIVE: v0.8.0-era symbols keep their FQNs."""
    parsed = _parse("javascript", STABLE_JS, "js")
    fqns = {s.simple_name: s.fully_qualified_name for s in parsed.symbols}
    assert fqns["add"] == "f.js::add"
    assert fqns["Calc"] == "f.js::Calc"
    assert fqns["run"] == "f.js::Calc::run"
    assert fqns["constructor"] == "f.js::Calc::constructor"
    assert fqns["mul"] == "f.js::mul"
