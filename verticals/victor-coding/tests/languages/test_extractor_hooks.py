# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Characterization tests for the callee/member-name extractor hooks.

For each language one snippet covers plain, member, chained, qualified,
generic, parenthesized and (where the language has them) macro / object
creation calls plus non-call member access. Two goldens per language were
captured from the pre-refactor per-plugin extractors and are pinned verbatim
(quirks included):

* ``callee``: ``_extract_callee_name(node)`` for EVERY node of the tree, as
  ``(line, column, node_type, name)`` for each non-``None`` result.
* ``child``: ``_extract_callee_name`` on a synthetic parent whose only child
  is each node in turn. This pins the per-child dispatch, i.e. the member /
  field / navigation / selector / scoped name derived from each node type.

Kotlin needs the production grammar wheel (the language pack ships a
different Kotlin grammar); every other language parses identically with
``tree_sitter_language_pack``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional, Tuple

import pytest

from victor_coding.languages.base import BaseLanguagePlugin, declarator_name, last_named_child
from victor_coding.languages.plugins import additional, cpp, go, java
from victor_coding.languages.plugins import python as python_plugin
from victor_coding.languages.plugins import rust

Row = Tuple[int, int, str, str]


def _parse_tree(language: str, source: str) -> Any:
    data = source.encode("utf-8")
    try:
        from victor_coding.codebase.tree_sitter_manager import get_parser

        return get_parser(language).parse(data)
    except Exception:
        if language == "kotlin":
            pytest.skip("production tree_sitter kotlin grammar not installed")
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language

        pack_name = {"csharp": "c_sharp"}.get(language, language)
        return Parser(get_language(pack_name)).parse(data)
    except Exception as exc:  # pragma: no cover - depends on installed grammars
        pytest.skip(f"no tree-sitter grammar available for {language}: {exc}")


def _walk(node: Any) -> Iterator[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _rows(plugin: BaseLanguagePlugin, root: Any, wrap: bool) -> List[Row]:
    rows: List[Row] = []
    for node in _walk(root):
        target = SimpleNamespace(type="__wrapper__", children=[node]) if wrap else node
        name: Optional[str] = plugin._extract_callee_name(target)
        if name is not None:
            rows.append((node.start_point[0] + 1, node.start_point[1], node.type, name))
    return rows


PYTHON_SOURCE = """\
import os.path

foo()
obj.method()
a.b.c.deep()
os.path.join("a", "b")
make()().after()
(get_fn)(1)
items[0].pop()
x = obj.attr.chain
def outer():
    super().__init__()
    return Klass[int].create()
async def runner():
    await client.session.fetch(url)
"""
PYTHON_CALLEE = [
    (1, 7, "dotted_name", "os"),
    (3, 0, "call", "foo"),
    (4, 0, "call", "method"),
    (4, 0, "attribute", "obj"),
    (5, 0, "call", "deep"),
    (5, 0, "attribute", "c"),
    (5, 0, "attribute", "b"),
    (5, 0, "attribute", "a"),
    (6, 0, "call", "join"),
    (6, 0, "attribute", "path"),
    (6, 0, "attribute", "os"),
    (7, 0, "call", "after"),
    (7, 0, "attribute", "after"),
    (7, 0, "call", "make"),
    (8, 0, "parenthesized_expression", "get_fn"),
    (9, 0, "call", "pop"),
    (9, 0, "attribute", "pop"),
    (9, 0, "subscript", "items"),
    (10, 0, "assignment", "x"),
    (10, 4, "attribute", "attr"),
    (10, 4, "attribute", "obj"),
    (11, 0, "function_definition", "outer"),
    (12, 4, "call", "__init__"),
    (12, 4, "attribute", "__init__"),
    (12, 4, "call", "super"),
    (13, 11, "call", "create"),
    (13, 11, "attribute", "create"),
    (13, 11, "subscript", "Klass"),
    (14, 0, "function_definition", "runner"),
    (15, 10, "call", "fetch"),
    (15, 10, "attribute", "session"),
    (15, 10, "attribute", "client"),
    (15, 30, "argument_list", "url"),
]
PYTHON_CHILD = [
    (1, 7, "identifier", "os"),
    (1, 10, "identifier", "path"),
    (3, 0, "identifier", "foo"),
    (4, 0, "attribute", "method"),
    (4, 0, "identifier", "obj"),
    (4, 4, "identifier", "method"),
    (5, 0, "attribute", "deep"),
    (5, 0, "attribute", "c"),
    (5, 0, "attribute", "b"),
    (5, 0, "identifier", "a"),
    (5, 2, "identifier", "b"),
    (5, 4, "identifier", "c"),
    (5, 6, "identifier", "deep"),
    (6, 0, "attribute", "join"),
    (6, 0, "attribute", "path"),
    (6, 0, "identifier", "os"),
    (6, 3, "identifier", "path"),
    (6, 8, "identifier", "join"),
    (7, 0, "attribute", "after"),
    (7, 0, "identifier", "make"),
    (7, 9, "identifier", "after"),
    (8, 1, "identifier", "get_fn"),
    (9, 0, "attribute", "pop"),
    (9, 0, "identifier", "items"),
    (9, 9, "identifier", "pop"),
    (10, 0, "identifier", "x"),
    (10, 4, "attribute", "chain"),
    (10, 4, "attribute", "attr"),
    (10, 4, "identifier", "obj"),
    (10, 8, "identifier", "attr"),
    (10, 13, "identifier", "chain"),
    (11, 4, "identifier", "outer"),
    (12, 4, "attribute", "__init__"),
    (12, 4, "identifier", "super"),
    (12, 12, "identifier", "__init__"),
    (13, 11, "attribute", "create"),
    (13, 11, "identifier", "Klass"),
    (13, 17, "identifier", "int"),
    (13, 22, "identifier", "create"),
    (14, 10, "identifier", "runner"),
    (15, 10, "attribute", "fetch"),
    (15, 10, "attribute", "session"),
    (15, 10, "identifier", "client"),
    (15, 17, "identifier", "session"),
    (15, 25, "identifier", "fetch"),
    (15, 31, "identifier", "url"),
]

GO_SOURCE = """\
package main

func main() {
\tfoo()
\tobj.Method()
\tpkg.Type.Method()
\tfmt.Println("x")
\t(fn)(1)
\t((pkg.Fn))(2)
\tgetObj().Next().Last()
\tGeneric[int](3)
\tx := s.field.inner
\t_ = x
\tfunc() { inner() }()
}
"""
GO_CALLEE = [
    (3, 0, "function_declaration", "main"),
    (4, 1, "call_expression", "foo"),
    (5, 1, "call_expression", "Method"),
    (5, 1, "selector_expression", "obj"),
    (6, 1, "call_expression", "Method"),
    (6, 1, "selector_expression", "Type"),
    (6, 1, "selector_expression", "pkg"),
    (7, 1, "call_expression", "Println"),
    (7, 1, "selector_expression", "fmt"),
    (8, 1, "call_expression", "fn"),
    (8, 1, "parenthesized_expression", "fn"),
    (9, 1, "call_expression", "Fn"),
    (9, 1, "parenthesized_expression", "Fn"),
    (9, 2, "parenthesized_expression", "Fn"),
    (9, 3, "selector_expression", "pkg"),
    (10, 1, "call_expression", "Last"),
    (10, 1, "call_expression", "Next"),
    (10, 1, "call_expression", "getObj"),
    (12, 1, "expression_list", "x"),
    (12, 6, "expression_list", "inner"),
    (12, 6, "selector_expression", "field"),
    (12, 6, "selector_expression", "s"),
    (13, 1, "expression_list", "_"),
    (13, 5, "expression_list", "x"),
    (14, 10, "call_expression", "inner"),
]
GO_CHILD = [
    (3, 5, "identifier", "main"),
    (4, 1, "identifier", "foo"),
    (5, 1, "selector_expression", "Method"),
    (5, 1, "identifier", "obj"),
    (6, 1, "selector_expression", "Method"),
    (6, 1, "selector_expression", "Type"),
    (6, 1, "identifier", "pkg"),
    (7, 1, "selector_expression", "Println"),
    (7, 1, "identifier", "fmt"),
    (8, 1, "parenthesized_expression", "fn"),
    (8, 2, "identifier", "fn"),
    (9, 1, "parenthesized_expression", "Fn"),
    (9, 2, "parenthesized_expression", "Fn"),
    (9, 3, "selector_expression", "Fn"),
    (9, 3, "identifier", "pkg"),
    (10, 1, "selector_expression", "Last"),
    (10, 1, "selector_expression", "Next"),
    (10, 1, "identifier", "getObj"),
    (12, 1, "identifier", "x"),
    (12, 6, "selector_expression", "inner"),
    (12, 6, "selector_expression", "field"),
    (12, 6, "identifier", "s"),
    (13, 1, "identifier", "_"),
    (13, 5, "identifier", "x"),
    (14, 10, "identifier", "inner"),
]

JAVA_SOURCE = """\
class Sample {
    void run() {
        foo();
        obj.method();
        a.b.c.deep();
        System.out.println("x");
        this.<String>generic();
        builder.one().two().three();
        new Foo();
        new com.example.Bar();
        new java.util.ArrayList<String>();
        super.close();
        int y = obj.field.inner;
    }
}
"""
JAVA_CALLEE = [
    (1, 0, "class_declaration", "Sample"),
    (2, 4, "method_declaration", "run"),
    (3, 8, "method_invocation", "foo"),
    (4, 8, "method_invocation", "method"),
    (5, 8, "method_invocation", "deep"),
    (5, 8, "field_access", "c"),
    (5, 8, "field_access", "a"),
    (6, 8, "method_invocation", "println"),
    (6, 8, "field_access", "System"),
    (7, 8, "method_invocation", "generic"),
    (8, 8, "method_invocation", "three"),
    (8, 8, "method_invocation", "two"),
    (8, 8, "method_invocation", "one"),
    (9, 8, "object_creation_expression", "Foo"),
    (12, 8, "method_invocation", "close"),
    (13, 12, "variable_declarator", "y"),
    (13, 16, "field_access", "inner"),
    (13, 16, "field_access", "obj"),
]
JAVA_CHILD = [
    (1, 6, "identifier", "Sample"),
    (2, 9, "identifier", "run"),
    (3, 8, "identifier", "foo"),
    (4, 8, "identifier", "obj"),
    (4, 12, "identifier", "method"),
    (5, 8, "identifier", "a"),
    (5, 10, "identifier", "b"),
    (5, 12, "identifier", "c"),
    (5, 14, "identifier", "deep"),
    (6, 8, "identifier", "System"),
    (6, 15, "identifier", "out"),
    (6, 19, "identifier", "println"),
    (7, 21, "identifier", "generic"),
    (8, 8, "identifier", "builder"),
    (8, 16, "identifier", "one"),
    (8, 22, "identifier", "two"),
    (8, 28, "identifier", "three"),
    (12, 14, "identifier", "close"),
    (13, 12, "identifier", "y"),
    (13, 16, "identifier", "obj"),
    (13, 20, "identifier", "field"),
    (13, 26, "identifier", "inner"),
]

CPP_SOURCE = """\
#include <memory>
void run() {
    foo();
    obj.method();
    ptr->method();
    a.b.c.deep();
    std::sort(v.begin(), v.end());
    ns::Klass::create();
    make_unique<Foo>();
    std::make_shared<Bar>(1);
    obj.template get<int>();
    builder.one().two();
    (fn)(1);
    int y = obj.field.inner;
}
"""
CPP_CALLEE = [
    (2, 5, "function_declarator", "run"),
    (3, 4, "call_expression", "foo"),
    (4, 4, "call_expression", "method"),
    (4, 4, "field_expression", "obj"),
    (5, 4, "call_expression", "method"),
    (5, 4, "field_expression", "ptr"),
    (6, 4, "call_expression", "deep"),
    (6, 4, "field_expression", "c"),
    (6, 4, "field_expression", "b"),
    (6, 4, "field_expression", "a"),
    (7, 4, "call_expression", "sort"),
    (7, 4, "qualified_identifier", "sort"),
    (7, 14, "call_expression", "begin"),
    (7, 14, "field_expression", "v"),
    (7, 25, "call_expression", "end"),
    (7, 25, "field_expression", "v"),
    (8, 4, "call_expression", "create"),
    (8, 4, "qualified_identifier", "create"),
    (8, 8, "qualified_identifier", "create"),
    (9, 4, "call_expression", "make_unique"),
    (9, 4, "template_function", "make_unique"),
    (10, 4, "qualified_identifier", "make_shared"),
    (10, 9, "template_function", "make_shared"),
    (11, 4, "field_expression", "obj"),
    (12, 4, "call_expression", "two"),
    (12, 4, "call_expression", "one"),
    (12, 4, "field_expression", "builder"),
    (13, 4, "parenthesized_expression", "fn"),
    (14, 8, "init_declarator", "y"),
    (14, 12, "field_expression", "field"),
    (14, 12, "field_expression", "obj"),
]
CPP_CHILD = [
    (2, 5, "identifier", "run"),
    (3, 4, "identifier", "foo"),
    (4, 4, "field_expression", "method"),
    (4, 4, "identifier", "obj"),
    (5, 4, "field_expression", "method"),
    (5, 4, "identifier", "ptr"),
    (6, 4, "field_expression", "deep"),
    (6, 4, "field_expression", "c"),
    (6, 4, "field_expression", "b"),
    (6, 4, "identifier", "a"),
    (7, 4, "qualified_identifier", "sort"),
    (7, 9, "identifier", "sort"),
    (7, 14, "field_expression", "begin"),
    (7, 14, "identifier", "v"),
    (7, 25, "field_expression", "end"),
    (7, 25, "identifier", "v"),
    (8, 4, "qualified_identifier", "create"),
    (8, 8, "qualified_identifier", "create"),
    (8, 15, "identifier", "create"),
    (9, 4, "template_function", "make_unique"),
    (9, 4, "identifier", "make_unique"),
    (10, 9, "template_function", "make_shared"),
    (10, 9, "identifier", "make_shared"),
    (11, 4, "identifier", "obj"),
    (12, 4, "field_expression", "two"),
    (12, 4, "field_expression", "one"),
    (12, 4, "identifier", "builder"),
    (13, 5, "identifier", "fn"),
    (14, 8, "identifier", "y"),
    (14, 12, "field_expression", "inner"),
    (14, 12, "field_expression", "field"),
    (14, 12, "identifier", "obj"),
]

C_SOURCE = """\
void run(struct s *p) {
    foo();
    p->method();
    s.field.fn();
    p->inner->deep(1);
    (fn)(2);
    (*fp)(3);
    int y = s.a.b;
}
"""
C_CALLEE = [
    (1, 5, "function_declarator", "run"),
    (1, 18, "pointer_declarator", "p"),
    (2, 4, "call_expression", "foo"),
    (3, 4, "call_expression", "method"),
    (3, 4, "field_expression", "p"),
    (4, 4, "call_expression", "fn"),
    (4, 4, "field_expression", "field"),
    (4, 4, "field_expression", "s"),
    (5, 4, "call_expression", "deep"),
    (5, 4, "field_expression", "inner"),
    (5, 4, "field_expression", "p"),
    (6, 4, "parenthesized_expression", "fn"),
    (7, 5, "pointer_expression", "fp"),
    (8, 8, "init_declarator", "y"),
    (8, 12, "field_expression", "a"),
    (8, 12, "field_expression", "s"),
]
C_CHILD = [
    (1, 5, "identifier", "run"),
    (1, 19, "identifier", "p"),
    (2, 4, "identifier", "foo"),
    (3, 4, "field_expression", "method"),
    (3, 4, "identifier", "p"),
    (4, 4, "field_expression", "fn"),
    (4, 4, "field_expression", "field"),
    (4, 4, "identifier", "s"),
    (5, 4, "field_expression", "deep"),
    (5, 4, "field_expression", "inner"),
    (5, 4, "identifier", "p"),
    (6, 5, "identifier", "fn"),
    (7, 6, "identifier", "fp"),
    (8, 8, "identifier", "y"),
    (8, 12, "field_expression", "b"),
    (8, 12, "field_expression", "a"),
    (8, 12, "identifier", "s"),
]

RUST_SOURCE = """\
fn run() {
    foo();
    obj.method();
    a.b.c.deep();
    std::mem::drop(x);
    Vec::<i32>::new();
    Foo::new().build();
    println!("x");
    std::println!("y");
    builder.one().two();
    (closure)(1);
    let y = s.field.inner;
    obj.generic::<u8>();
}
"""
RUST_CALLEE = [
    (1, 0, "function_item", "run"),
    (2, 4, "call_expression", "foo"),
    (3, 4, "call_expression", "method"),
    (3, 4, "field_expression", "obj"),
    (4, 4, "call_expression", "deep"),
    (4, 4, "field_expression", "c"),
    (4, 4, "field_expression", "b"),
    (4, 4, "field_expression", "a"),
    (5, 4, "call_expression", "drop"),
    (5, 4, "scoped_identifier", "mem"),
    (5, 4, "scoped_identifier", "std"),
    (5, 18, "arguments", "x"),
    (6, 4, "call_expression", "new"),
    (6, 4, "scoped_identifier", "new"),
    (7, 4, "call_expression", "build"),
    (7, 4, "call_expression", "new"),
    (7, 4, "scoped_identifier", "Foo"),
    (8, 4, "macro_invocation", "println"),
    (9, 4, "macro_invocation", "println"),
    (9, 4, "scoped_identifier", "std"),
    (10, 4, "call_expression", "two"),
    (10, 4, "call_expression", "one"),
    (10, 4, "field_expression", "builder"),
    (11, 4, "parenthesized_expression", "closure"),
    (12, 4, "let_declaration", "y"),
    (12, 12, "field_expression", "field"),
    (12, 12, "field_expression", "s"),
    (13, 4, "generic_function", "generic"),
    (13, 4, "field_expression", "obj"),
]
RUST_CHILD = [
    (1, 3, "identifier", "run"),
    (2, 4, "identifier", "foo"),
    (3, 4, "field_expression", "method"),
    (3, 4, "identifier", "obj"),
    (4, 4, "field_expression", "deep"),
    (4, 4, "field_expression", "c"),
    (4, 4, "field_expression", "b"),
    (4, 4, "identifier", "a"),
    (5, 4, "scoped_identifier", "drop"),
    (5, 4, "scoped_identifier", "mem"),
    (5, 4, "identifier", "std"),
    (5, 9, "identifier", "mem"),
    (5, 14, "identifier", "drop"),
    (5, 19, "identifier", "x"),
    (6, 4, "scoped_identifier", "new"),
    (6, 16, "identifier", "new"),
    (7, 4, "field_expression", "build"),
    (7, 4, "scoped_identifier", "new"),
    (7, 4, "identifier", "Foo"),
    (7, 9, "identifier", "new"),
    (8, 4, "identifier", "println"),
    (9, 4, "scoped_identifier", "println"),
    (9, 4, "identifier", "std"),
    (9, 9, "identifier", "println"),
    (10, 4, "field_expression", "two"),
    (10, 4, "field_expression", "one"),
    (10, 4, "identifier", "builder"),
    (11, 5, "identifier", "closure"),
    (12, 8, "identifier", "y"),
    (12, 12, "field_expression", "inner"),
    (12, 12, "field_expression", "field"),
    (12, 12, "identifier", "s"),
    (13, 4, "field_expression", "generic"),
    (13, 4, "identifier", "obj"),
]

KOTLIN_SOURCE = """\
fun run() {
    foo()
    obj.method()
    a.b.c.deep()
    obj?.safe()
    listOf<Int>(1)
    builder.one().two()
    Foo()
    val y = s.field.inner
}
"""
KOTLIN_CALLEE = [
    (1, 0, "function_declaration", "run"),
    (2, 4, "call_expression", "foo"),
    (3, 4, "call_expression", "method"),
    (3, 4, "navigation_expression", "obj"),
    (4, 4, "call_expression", "deep"),
    (4, 4, "navigation_expression", "c"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "navigation_expression", "a"),
    (5, 4, "call_expression", "safe"),
    (5, 4, "navigation_expression", "obj"),
    (6, 4, "binary_expression", "listOf"),
    (7, 4, "call_expression", "two"),
    (7, 4, "navigation_expression", "two"),
    (7, 4, "call_expression", "one"),
    (7, 4, "navigation_expression", "builder"),
    (8, 4, "call_expression", "Foo"),
    (9, 4, "property_declaration", "inner"),
    (9, 8, "variable_declaration", "y"),
    (9, 12, "navigation_expression", "field"),
    (9, 12, "navigation_expression", "s"),
]
KOTLIN_CHILD = [
    (1, 4, "identifier", "run"),
    (2, 4, "identifier", "foo"),
    (3, 4, "navigation_expression", "method"),
    (3, 4, "identifier", "obj"),
    (3, 8, "identifier", "method"),
    (4, 4, "navigation_expression", "deep"),
    (4, 4, "navigation_expression", "c"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "identifier", "a"),
    (4, 6, "identifier", "b"),
    (4, 8, "identifier", "c"),
    (4, 10, "identifier", "deep"),
    (5, 4, "navigation_expression", "safe"),
    (5, 4, "identifier", "obj"),
    (5, 9, "identifier", "safe"),
    (6, 4, "identifier", "listOf"),
    (6, 11, "identifier", "Int"),
    (7, 4, "navigation_expression", "two"),
    (7, 4, "navigation_expression", "one"),
    (7, 4, "identifier", "builder"),
    (7, 12, "identifier", "one"),
    (7, 18, "identifier", "two"),
    (8, 4, "identifier", "Foo"),
    (9, 8, "identifier", "y"),
    (9, 12, "navigation_expression", "inner"),
    (9, 12, "navigation_expression", "field"),
    (9, 12, "identifier", "s"),
    (9, 14, "identifier", "field"),
    (9, 20, "identifier", "inner"),
]

CSHARP_SOURCE = """\
class Sample {
    void Run() {
        Foo();
        obj.Method();
        a.b.c.Deep();
        Console.WriteLine("x");
        obj.Generic<int>();
        builder.One().Two();
        new Foo();
        new List<int>();
        var y = s.Field.Inner;
    }
}
"""
CSHARP_CALLEE = [
    (1, 0, "class_declaration", "Sample"),
    (2, 4, "method_declaration", "Run"),
    (3, 8, "invocation_expression", "Foo"),
    (4, 8, "invocation_expression", "Method"),
    (4, 8, "member_access_expression", "obj"),
    (5, 8, "invocation_expression", "Deep"),
    (5, 8, "member_access_expression", "c"),
    (5, 8, "member_access_expression", "b"),
    (5, 8, "member_access_expression", "a"),
    (6, 8, "invocation_expression", "WriteLine"),
    (6, 8, "member_access_expression", "Console"),
    (7, 8, "invocation_expression", "Generic"),
    (7, 8, "member_access_expression", "obj"),
    (7, 12, "generic_name", "Generic"),
    (8, 8, "invocation_expression", "Two"),
    (8, 8, "member_access_expression", "Two"),
    (8, 8, "invocation_expression", "One"),
    (8, 8, "member_access_expression", "builder"),
    (9, 8, "object_creation_expression", "Foo"),
    (10, 8, "object_creation_expression", "List"),
    (10, 12, "generic_name", "List"),
    (11, 12, "variable_declarator", "y"),
    (11, 16, "member_access_expression", "Field"),
    (11, 16, "member_access_expression", "s"),
]
CSHARP_CHILD = [
    (1, 6, "identifier", "Sample"),
    (2, 9, "identifier", "Run"),
    (3, 8, "identifier", "Foo"),
    (4, 8, "member_access_expression", "Method"),
    (4, 8, "identifier", "obj"),
    (4, 12, "identifier", "Method"),
    (5, 8, "member_access_expression", "Deep"),
    (5, 8, "member_access_expression", "c"),
    (5, 8, "member_access_expression", "b"),
    (5, 8, "identifier", "a"),
    (5, 10, "identifier", "b"),
    (5, 12, "identifier", "c"),
    (5, 14, "identifier", "Deep"),
    (6, 8, "member_access_expression", "WriteLine"),
    (6, 8, "identifier", "Console"),
    (6, 16, "identifier", "WriteLine"),
    (7, 8, "member_access_expression", "Generic"),
    (7, 8, "identifier", "obj"),
    (7, 12, "identifier", "Generic"),
    (8, 8, "member_access_expression", "Two"),
    (8, 8, "member_access_expression", "One"),
    (8, 8, "identifier", "builder"),
    (8, 16, "identifier", "One"),
    (8, 22, "identifier", "Two"),
    (9, 12, "identifier", "Foo"),
    (10, 12, "identifier", "List"),
    (11, 12, "identifier", "y"),
    (11, 16, "member_access_expression", "Inner"),
    (11, 16, "member_access_expression", "Field"),
    (11, 16, "identifier", "s"),
    (11, 18, "identifier", "Field"),
    (11, 24, "identifier", "Inner"),
]

RUBY_SOURCE = """\
def run
  foo
  foo()
  obj.method
  a.b.c.deep(1)
  obj&.safe
  Klass.new
  Mod::Klass.create
  builder.one.two
end
"""
RUBY_CALLEE = [
    (1, 0, "method", "run"),
    (1, 4, "identifier", "run"),
    (2, 2, "body_statement", "foo"),
    (2, 2, "identifier", "foo"),
    (3, 2, "call", "foo"),
    (3, 2, "identifier", "foo"),
    (4, 2, "call", "method"),
    (4, 2, "identifier", "obj"),
    (4, 6, "identifier", "method"),
    (5, 2, "call", "deep"),
    (5, 2, "call", "c"),
    (5, 2, "call", "b"),
    (5, 2, "identifier", "a"),
    (5, 4, "identifier", "b"),
    (5, 6, "identifier", "c"),
    (5, 8, "identifier", "deep"),
    (6, 2, "call", "safe"),
    (6, 2, "identifier", "obj"),
    (6, 7, "identifier", "safe"),
    (7, 2, "call", "new"),
    (7, 8, "identifier", "new"),
    (8, 2, "call", "create"),
    (8, 13, "identifier", "create"),
    (9, 2, "call", "two"),
    (9, 2, "call", "one"),
    (9, 2, "identifier", "builder"),
    (9, 10, "identifier", "one"),
    (9, 14, "identifier", "two"),
]
RUBY_CHILD = [
    (1, 4, "identifier", "run"),
    (2, 2, "identifier", "foo"),
    (3, 2, "identifier", "foo"),
    (4, 2, "identifier", "obj"),
    (4, 6, "identifier", "method"),
    (5, 2, "identifier", "a"),
    (5, 4, "identifier", "b"),
    (5, 6, "identifier", "c"),
    (5, 8, "identifier", "deep"),
    (6, 2, "identifier", "obj"),
    (6, 7, "identifier", "safe"),
    (7, 8, "identifier", "new"),
    (8, 13, "identifier", "create"),
    (9, 2, "identifier", "builder"),
    (9, 10, "identifier", "one"),
    (9, 14, "identifier", "two"),
]

PHP_SOURCE = """\
<?php
function run() {
    foo();
    $obj->method();
    $a->b->deep();
    Klass::create();
    \\Ns\\Klass::make();
    new Foo();
    new \\Ns\\Bar();
    $obj?->safe();
    $builder->one()->two();
}
"""
PHP_CALLEE = [
    (3, 4, "function_call_expression", "foo"),
    (4, 4, "member_call_expression", "method"),
    (5, 4, "member_call_expression", "deep"),
    (6, 4, "scoped_call_expression", "create"),
    (7, 4, "scoped_call_expression", "make"),
    (8, 4, "object_creation_expression", "Foo"),
    (11, 4, "member_call_expression", "two"),
    (11, 4, "member_call_expression", "one"),
]
PHP_CHILD = []

SWIFT_SOURCE = """\
func run() {
    foo()
    obj.method()
    a.b.c.deep()
    obj?.safe()
    Foo()
    builder.one().two()
    let y = s.field.inner
}
"""
SWIFT_CALLEE = [
    (1, 0, "function_declaration", "run"),
    (2, 4, "call_expression", "foo"),
    (3, 4, "call_expression", "method"),
    (3, 4, "navigation_expression", "obj"),
    (3, 7, "navigation_suffix", "method"),
    (4, 4, "call_expression", "b"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "navigation_expression", "a"),
    (4, 5, "navigation_suffix", "b"),
    (4, 7, "navigation_suffix", "c"),
    (4, 9, "navigation_suffix", "deep"),
    (5, 4, "call_expression", "safe"),
    (5, 4, "navigation_expression", "obj"),
    (5, 8, "navigation_suffix", "safe"),
    (6, 4, "call_expression", "Foo"),
    (7, 4, "call_expression", "two"),
    (7, 4, "call_expression", "one"),
    (7, 4, "navigation_expression", "builder"),
    (7, 11, "navigation_suffix", "one"),
    (7, 17, "navigation_suffix", "two"),
    (8, 4, "property_declaration", "field"),
    (8, 8, "pattern", "y"),
    (8, 12, "navigation_expression", "field"),
    (8, 12, "navigation_expression", "s"),
    (8, 13, "navigation_suffix", "field"),
    (8, 19, "navigation_suffix", "inner"),
]
SWIFT_CHILD = [
    (1, 5, "simple_identifier", "run"),
    (2, 4, "simple_identifier", "foo"),
    (3, 4, "navigation_expression", "method"),
    (3, 4, "simple_identifier", "obj"),
    (3, 8, "simple_identifier", "method"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "navigation_expression", "b"),
    (4, 4, "simple_identifier", "a"),
    (4, 6, "simple_identifier", "b"),
    (4, 8, "simple_identifier", "c"),
    (4, 10, "simple_identifier", "deep"),
    (5, 4, "navigation_expression", "safe"),
    (5, 4, "simple_identifier", "obj"),
    (5, 9, "simple_identifier", "safe"),
    (6, 4, "simple_identifier", "Foo"),
    (7, 4, "navigation_expression", "two"),
    (7, 4, "navigation_expression", "one"),
    (7, 4, "simple_identifier", "builder"),
    (7, 12, "simple_identifier", "one"),
    (7, 18, "simple_identifier", "two"),
    (8, 8, "simple_identifier", "y"),
    (8, 12, "navigation_expression", "field"),
    (8, 12, "navigation_expression", "field"),
    (8, 12, "simple_identifier", "s"),
    (8, 14, "simple_identifier", "field"),
    (8, 20, "simple_identifier", "inner"),
]

CASES = [
    pytest.param(
        python_plugin.PythonPlugin,
        "python",
        PYTHON_SOURCE,
        PYTHON_CALLEE,
        PYTHON_CHILD,
        id="python",
    ),
    pytest.param(go.GoPlugin, "go", GO_SOURCE, GO_CALLEE, GO_CHILD, id="go"),
    pytest.param(java.JavaPlugin, "java", JAVA_SOURCE, JAVA_CALLEE, JAVA_CHILD, id="java"),
    pytest.param(cpp.CppPlugin, "cpp", CPP_SOURCE, CPP_CALLEE, CPP_CHILD, id="cpp"),
    pytest.param(additional.CPlugin, "c", C_SOURCE, C_CALLEE, C_CHILD, id="c"),
    pytest.param(rust.RustPlugin, "rust", RUST_SOURCE, RUST_CALLEE, RUST_CHILD, id="rust"),
    pytest.param(
        additional.KotlinPlugin, "kotlin", KOTLIN_SOURCE, KOTLIN_CALLEE, KOTLIN_CHILD, id="kotlin"
    ),
    pytest.param(
        additional.CSharpPlugin, "csharp", CSHARP_SOURCE, CSHARP_CALLEE, CSHARP_CHILD, id="csharp"
    ),
    pytest.param(additional.RubyPlugin, "ruby", RUBY_SOURCE, RUBY_CALLEE, RUBY_CHILD, id="ruby"),
    pytest.param(additional.PhpPlugin, "php", PHP_SOURCE, PHP_CALLEE, PHP_CHILD, id="php"),
    pytest.param(
        additional.SwiftPlugin, "swift", SWIFT_SOURCE, SWIFT_CALLEE, SWIFT_CHILD, id="swift"
    ),
]


@pytest.mark.parametrize("plugin_cls,language,source,callee,child", CASES)
def test_callee_name_for_every_node(plugin_cls, language, source, callee, child) -> None:
    tree = _parse_tree(language, source)
    assert _rows(plugin_cls(), tree.root_node, wrap=False) == callee


@pytest.mark.parametrize("plugin_cls,language,source,callee,child", CASES)
def test_callee_name_per_child_dispatch(plugin_cls, language, source, callee, child) -> None:
    tree = _parse_tree(language, source)
    assert _rows(plugin_cls(), tree.root_node, wrap=True) == child


C_FUNCTIONS_SOURCE = """\
int plain(void) { return 0; }
static char *ptr_ret(int n) { return 0; }
void (*fp_ret(int x))(int) { return 0; }
struct s make(struct s v) { return v; }
"""
C_FUNCTION_NAMES = [(1, "plain"), (2, None), (3, None), (4, "make")]

CPP_FUNCTIONS_SOURCE = """\
int plain() { return 0; }
void Klass::method() {}
Klass::Klass() {}
Klass::~Klass() {}
bool Klass::operator==(const Klass &o) const { return true; }
int &ref_ret() { static int x; return x; }
template <typename T> T tmpl(T v) { return v; }
namespace ns { void inner() {} }
struct S { void inline_method() {} };
auto trailing() -> int { return 1; }
"""
CPP_FUNCTION_NAMES = [
    (1, "plain"),
    (2, None),
    (3, None),
    (4, None),
    (5, None),
    (6, None),
    (7, "tmpl"),
    (8, "inner"),
    (9, "inline_method"),
    (10, "trailing"),
]


@pytest.mark.parametrize(
    "language,source,extract,expected",
    [
        pytest.param(
            "c",
            C_FUNCTIONS_SOURCE,
            lambda node: additional.CPlugin()._extract_function_name_c(node),
            C_FUNCTION_NAMES,
            id="c",
        ),
        pytest.param(
            "cpp",
            CPP_FUNCTIONS_SOURCE,
            lambda node: cpp.CppPlugin()._extract_function_name(node),
            CPP_FUNCTION_NAMES,
            id="cpp",
        ),
    ],
)
def test_function_definition_names(language, source, extract, expected) -> None:
    tree = _parse_tree(language, source)
    names = [
        (node.start_point[0] + 1, extract(node))
        for node in _walk(tree.root_node)
        if node.type == "function_definition"
    ]
    assert sorted(names, key=lambda row: row[0]) == expected


# ---------------------------------------------------------------------------
# Shared helpers / base hook on synthetic nodes (grammar independent)
# ---------------------------------------------------------------------------


def _n(node_type: str, text: str = "", *children: Any) -> SimpleNamespace:
    return SimpleNamespace(type=node_type, text=text.encode(), children=list(children))


_TEXT = BaseLanguagePlugin._get_node_text


def test_last_named_child_takes_rightmost_leaf() -> None:
    path = _n("attr", "", _n("id", "obj"), _n(".", "."), _n("id", "method"))
    assert last_named_child(path, {"id"}, {"attr"}, _TEXT) == "method"


def test_last_named_child_recurses_into_nested_paths_only_when_listed() -> None:
    inner = _n("attr", "", _n("id", "a"), _n("id", "b"))
    outer = _n("attr", "", inner, _n("call_args", "()"))
    assert last_named_child(outer, {"id"}, {"attr"}, _TEXT) == "b"
    assert last_named_child(outer, {"id"}, frozenset(), _TEXT) is None


def test_last_named_child_skips_empty_nested_result() -> None:
    empty_nested = _n("attr", "", _n("other", "x"))
    node = _n("attr", "", _n("id", "left"), empty_nested)
    assert last_named_child(node, {"id"}, {"attr"}, _TEXT) == "left"


def test_last_named_child_leaf_wins_over_later_nested() -> None:
    node = _n("attr", "", _n("attr", "", _n("id", "deep")), _n("id", "tail"))
    assert last_named_child(node, {"id"}, {"attr"}, _TEXT) == "tail"


def test_declarator_name_prefers_function_declarator() -> None:
    node = _n(
        "function_definition",
        "",
        _n("identifier", "type_name"),
        _n("function_declarator", "", _n("identifier", "real"), _n("parameter_list", "()")),
    )
    assert declarator_name(node, {"identifier"}, _TEXT) == "real"


def test_declarator_name_looks_inside_declarator_child_and_falls_back() -> None:
    nested = _n(
        "function_definition",
        "",
        _n("function_declarator", "", _n("declarator", "", _n("field_identifier", "m"))),
    )
    assert declarator_name(nested, {"identifier", "field_identifier"}, _TEXT) == "m"
    assert declarator_name(nested, {"identifier"}, _TEXT) is None
    fallback = _n("function_definition", "", _n("identifier", "direct"))
    assert declarator_name(fallback, {"identifier"}, _TEXT) == "direct"


class _TablePlugin(BaseLanguagePlugin):
    _callee_name_types = frozenset({"id"})
    _callee_path_types = {"attr": (frozenset({"id"}), frozenset({"attr"}))}
    _callee_wrapper_types = frozenset({"paren"})

    def _create_config(self):  # pragma: no cover - not used
        raise NotImplementedError

    def _create_capabilities(self):  # pragma: no cover - not used
        raise NotImplementedError


class _NoTablePlugin(_TablePlugin):
    _callee_name_types = BaseLanguagePlugin._callee_name_types
    _callee_path_types = BaseLanguagePlugin._callee_path_types
    _callee_wrapper_types = BaseLanguagePlugin._callee_wrapper_types


def test_base_callee_hook_dispatches_name_path_and_wrapper() -> None:
    plugin = _TablePlugin()
    assert plugin._extract_callee_name(_n("call", "", _n("id", "foo"))) == "foo"
    member = _n("attr", "", _n("id", "obj"), _n("id", "m"))
    assert plugin._extract_callee_name(_n("call", "", member)) == "m"
    wrapped = _n("paren", "", _n("(", "("), member, _n(")", ")"))
    assert plugin._extract_callee_name(_n("call", "", wrapped)) == "m"


def test_base_callee_hook_first_matching_child_decides() -> None:
    plugin = _TablePlugin()
    nameless_path = _n("attr", "", _n("other", "x"))
    call = _n("call", "", _n("args", "()"), nameless_path, _n("id", "later"))
    assert plugin._extract_callee_name(call) is None
    assert plugin._extract_callee_name(_n("call", "", _n("args", "()"))) is None


def test_base_callee_hook_default_tables_find_nothing() -> None:
    plugin = _NoTablePlugin()
    member = _n("attr", "", _n("id", "obj"), _n("id", "m"))
    assert plugin._extract_callee_name(_n("call", "", _n("id", "foo"), member)) is None


@pytest.mark.parametrize(
    "plugin_cls,path_type,leaf_type,expected",
    [
        pytest.param(python_plugin.PythonPlugin, "attribute", "identifier", "inner", id="python"),
        pytest.param(go.GoPlugin, "selector_expression", "field_identifier", "inner", id="go"),
        pytest.param(additional.CPlugin, "field_expression", "field_identifier", None, id="c"),
        pytest.param(
            cpp.CppPlugin, "field_expression", "field_identifier", "inner", id="cpp-field"
        ),
        pytest.param(cpp.CppPlugin, "qualified_identifier", "identifier", "inner", id="cpp-qual"),
        pytest.param(
            additional.KotlinPlugin, "navigation_expression", "identifier", "inner", id="kotlin"
        ),
        pytest.param(
            rust.RustPlugin, "field_expression", "field_identifier", "inner", id="rust-field"
        ),
        pytest.param(rust.RustPlugin, "scoped_identifier", "identifier", "inner", id="rust-scoped"),
    ],
)
def test_callee_path_recursion_on_malformed_path(
    plugin_cls, path_type, leaf_type, expected
) -> None:
    """A path whose last child is an ERROR node: the name comes from the nested path.

    Well-formed trees always end a path in its name leaf, so this pins the
    per-language recursion choice (C does not recurse) on error-recovered trees.
    """
    nested = _n(path_type, "", _n(leaf_type, "recv"), _n(leaf_type, "inner"))
    call = _n("call_expression", "", _n(path_type, "", nested, _n("ERROR", "")))
    assert plugin_cls()._extract_callee_name(call) == expected
