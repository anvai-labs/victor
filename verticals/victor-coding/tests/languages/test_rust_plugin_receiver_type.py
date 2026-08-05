# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Receiver-type extraction tests for the Rust language plugin.

These tests pin down receiver-type inference for `obj.method()` calls so
the cross-file CALLS resolver in victor-ai can bind method calls to the
correct `impl T` block instead of fanning out to every `method` named
`callee_name` across the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_rust")

from victor_coding.codebase.tree_sitter_manager import get_parser
from victor_coding.languages.base import CallEdge
from victor_coding.languages.plugins.rust import RustPlugin


def _detect(source: str) -> list[CallEdge]:
    plugin = RustPlugin()
    parser = get_parser("rust")
    tree = parser.parse(bytes(source, "utf-8"))
    result = plugin.detect_calls_edges(tree, source, Path("scratch.rs"))
    return result.calls


def _by_callee(calls: list[CallEdge], name: str) -> CallEdge:
    matches = [c for c in calls if c.callee_name == name]
    assert matches, f"no CallEdge with callee_name={name!r}; have {[c.callee_name for c in calls]}"
    assert len(matches) == 1, f"expected exactly one {name!r} call, got {len(matches)}"
    return matches[0]


def test_call_edge_default_receiver_type_is_none():
    """Backwards compatibility: receiver_type defaults to None when not set."""
    edge = CallEdge(caller_name="a", callee_name="b")
    assert edge.receiver_type is None


def test_typed_let_binds_receiver_type():
    """`let x: Foo = ...; x.method()` → receiver_type='Foo'."""
    source = """
fn caller() {
    let x: Foo = make();
    x.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.caller_name == "caller"
    assert edge.receiver_type == "Foo"


def test_constructor_let_binds_receiver_type():
    """`let x = Foo::new(); x.method()` → receiver_type='Foo'."""
    source = """
fn caller() {
    let x = Foo::new();
    x.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Foo"


def test_self_call_inside_impl_block_uses_impl_type():
    """`impl Foo { fn x(&self) { self.method() } }` → receiver_type='Foo'."""
    source = """
impl Foo {
    fn outer(&self) {
        self.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Foo"


def test_typed_parameter_binds_receiver_type():
    """`fn x(arg: Bar) { arg.method() }` → receiver_type='Bar'."""
    source = """
fn caller(arg: Bar) {
    arg.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_typed_reference_parameter_strips_reference():
    """`fn x(arg: &Bar)` and `fn x(arg: &mut Bar)` both yield receiver_type='Bar'."""
    source = """
fn caller(arg: &Bar) {
    arg.method();
}
fn caller_mut(arg: &mut Baz) {
    arg.other();
}
"""
    calls = _detect(source)
    assert _by_callee(calls, "method").receiver_type == "Bar"
    assert _by_callee(calls, "other").receiver_type == "Baz"


def test_plain_function_call_has_no_receiver_type():
    """`func()` → receiver_type=None (not a method call)."""
    source = """
fn caller() {
    func();
}
"""
    edge = _by_callee(_detect(source), "func")
    assert edge.receiver_type is None


def test_call_on_untracked_variable_has_no_receiver_type():
    """`unknown.method()` (no binding visible in scope) → receiver_type=None."""
    source = """
fn caller() {
    unknown.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type is None


def test_self_call_outside_impl_block_has_no_receiver_type():
    """A bare `fn` not inside `impl T` cannot bind `self` to a type."""
    source = """
fn outer(this: &Self) {
    this.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    # `this` is just a parameter named "this" with type Self; we don't model Self outside impl.
    # Receiver type should fall back to None (or the literal "Self") — assert it's not a real type.
    assert edge.receiver_type in (None, "Self")


# -----------------------------------------------------------------------------
# Struct-field receivers: `self.field.method()` resolves through the struct's
# field declarations to bind against the field's type, not Self. Common Rust
# pattern; intra-file only (cross-file struct field types are out of scope).
# -----------------------------------------------------------------------------


def test_self_field_method_resolves_through_struct_field_type():
    """`self.field.method()` inside `impl Foo` with `struct Foo { field: Bar }` -> Bar."""
    source = """
struct Foo {
    field: Bar,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_self_field_method_strips_reference_in_field_type():
    """`field: &Bar` and `field: &mut Bar` both yield 'Bar'."""
    source = """
struct Foo {
    field: &'static Bar,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


def test_self_field_method_unwraps_generic_to_base_type():
    """`field: Vec<Bar>` resolves to 'Vec' (the base type; generic args dropped).

    Bar would be the right answer for `Vec`, `Option`, `Box`-style single-arg
    wrappers, but that requires whitelisting and we explicitly chose not to
    model standard-library smart-pointer transparency in this round.
    """
    source = """
struct Foo {
    field: Vec<Bar>,
}
impl Foo {
    fn outer(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Vec"


def test_chained_field_access_resolves_through_intermediate_struct():
    """`self.a.b.method()` walks through both struct field maps."""
    source = """
struct Outer {
    a: Inner,
}
struct Inner {
    b: Target,
}
impl Outer {
    fn run(&self) {
        self.a.b.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Target"


def test_unknown_struct_field_falls_back_to_none():
    """`self.nonexistent.method()` — struct doesn't declare the field, falls back to None."""
    source = """
struct Foo { real: Bar }
impl Foo {
    fn run(&self) {
        self.nonexistent.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type is None


def test_struct_field_inference_works_when_struct_appears_after_impl():
    """Rust allows forward references; the plugin must do a struct-collection
    pre-pass so a struct defined below its impl block still resolves."""
    source = """
impl Foo {
    fn run(&self) {
        self.field.method();
    }
}
struct Foo {
    field: Bar,
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Bar"


# -----------------------------------------------------------------------------
# Method-chain inference: `a.foo().bar()` resolves through the return type
# of `foo`. Requires extracting return types from impl-method signatures and
# threading them through the value-type walk.
# -----------------------------------------------------------------------------


def test_method_chain_resolves_through_self_method_return_type():
    """`self.foo().bar()` where `fn foo(&self) -> Bar` -> bar's receiver = 'Bar'."""
    source = """
impl Foo {
    fn foo(&self) -> Bar { todo!() }
    fn run(&self) {
        self.foo().bar();
    }
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type == "Bar"


def test_method_chain_resolves_through_typed_var_method_return():
    """`x.foo().bar()` where `x: Foo` and `fn foo(&self) -> Bar` -> 'Bar'."""
    source = """
impl Foo {
    fn foo(&self) -> Bar { todo!() }
}
fn caller(x: Foo) {
    x.foo().bar();
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type == "Bar"


def test_method_chain_strips_reference_in_return_type():
    """`fn foo(&self) -> &Bar` -> bar's receiver = 'Bar'."""
    source = """
impl Foo {
    fn foo(&self) -> &Bar { todo!() }
    fn run(&self) {
        self.foo().bar();
    }
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type == "Bar"


def test_method_chain_unwraps_generic_return_type_to_base():
    """`fn foo(&self) -> Vec<Bar>` -> 'Vec' (consistent with field-type handling)."""
    source = """
impl Foo {
    fn foo(&self) -> Vec<Bar> { todo!() }
    fn run(&self) {
        self.foo().bar();
    }
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type == "Vec"


def test_constructor_chain_resolves_to_constructor_type():
    """`Foo::new().method()` -> 'Foo' (Self constructor returns Self by convention)."""
    source = """
fn caller() {
    Foo::new().method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type == "Foo"


def test_method_chain_with_no_return_type_falls_back_to_none():
    """`fn foo(&self) { }` (returns unit) chained `.bar()` -> None."""
    source = """
impl Foo {
    fn foo(&self) { }
    fn run(&self) {
        self.foo().bar();
    }
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type is None


def test_method_chain_with_unknown_intermediate_method_falls_back_to_none():
    """`x.unknown_method().bar()` -> intermediate method not in impl table -> None."""
    source = """
fn caller(x: Foo) {
    x.unknown_method().bar();
}
"""
    edge = _by_callee(_detect(source), "bar")
    assert edge.receiver_type is None


# -----------------------------------------------------------------------------
# Self::method() — :: syntax instead of dot dispatch. Common inside impl
# blocks for associated functions and explicit self-calls; deserves binding
# even though it's syntactically not a `field_expression`.
# -----------------------------------------------------------------------------


def test_self_path_call_resolves_to_enclosing_impl_type():
    """`Self::method()` inside `impl Foo` -> receiver_type = 'Foo'."""
    source = """
impl Foo {
    fn alpha() {
        Self::beta();
    }
    fn beta() {}
}
"""
    edge = _by_callee(_detect(source), "beta")
    assert edge.receiver_type == "Foo"


def test_self_path_call_outside_impl_block_has_no_receiver_type():
    """`Self::method()` at module level is invalid Rust but should not crash; -> None."""
    source = """
fn standalone() {
    Self::method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.receiver_type is None


# -----------------------------------------------------------------------------
# Macro invocations (println!, format!, vec!, write!, ...) are not function
# calls -- they expand at compile time. Emitting them as CallEdges causes the
# resolver to fan out to user-defined functions with the same leaf name
# (observed inflation on `format` and `vec` against user impls). Macros must
# not produce edges.
# -----------------------------------------------------------------------------


def test_println_macro_does_not_produce_call_edge():
    """`println!("...")` is a macro, not a function call -- no CallEdge."""
    source = """
fn caller() {
    println!("hi");
}
"""
    calls = _detect(source)
    assert [c for c in calls if c.callee_name == "println"] == []


def test_format_macro_does_not_produce_call_edge():
    """`format!("...")` is a macro -- must not bind to user-defined `format` fns."""
    source = """
fn caller() {
    let s = format!("{}", 1);
}
"""
    calls = _detect(source)
    assert [c for c in calls if c.callee_name == "format"] == []


def test_vec_macro_does_not_produce_call_edge():
    """`vec![1,2,3]` is a macro."""
    source = """
fn caller() {
    let v = vec![1, 2, 3];
}
"""
    calls = _detect(source)
    assert [c for c in calls if c.callee_name == "vec"] == []


def test_real_function_with_same_name_as_macro_still_emits_edge():
    """A user function explicitly called `vec(args)` is still a function call.

    We're filtering by AST node type (macro_invocation), not by leaf name, so
    a real `fn vec()` invocation remains a CallEdge.
    """
    source = """
fn caller() {
    vec(1, 2, 3);
}
"""
    edge = _by_callee(_detect(source), "vec")
    assert edge.caller_name == "caller"


# -----------------------------------------------------------------------------
# is_method_call: flag for resolver-side fallback policy. Method calls
# (`obj.method()`) without an inferable receiver type should *not* fall back
# to name-only matching — the leaf name is ambiguous across types and binding
# to user-defined same-named methods of unrelated types is almost always
# wrong. Plain function calls (`func()`) keep the name-only fallback.
# -----------------------------------------------------------------------------


def test_call_edge_default_is_method_call_is_false():
    edge = CallEdge(caller_name="a", callee_name="b")
    assert edge.is_method_call is False


def test_method_syntax_call_marked_as_method_call():
    """`obj.method()` is a method call regardless of whether type can be inferred."""
    source = """
fn caller(x: SomeUnknownType) {
    x.method();
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.is_method_call is True


def test_plain_function_call_not_marked_as_method_call():
    """`func()` is a plain function call."""
    source = """
fn caller() {
    free_function();
}
"""
    edge = _by_callee(_detect(source), "free_function")
    assert edge.is_method_call is False


def test_scoped_path_call_not_marked_as_method_call():
    """`Foo::bar()` is a path call, not a method call (different dispatch semantics)."""
    source = """
fn caller() {
    SomeModule::function();
}
"""
    edge = _by_callee(_detect(source), "function")
    assert edge.is_method_call is False


def test_self_field_method_marked_as_method_call_when_inferred():
    """`self.field.method()` is a method call even when receiver_type is known."""
    source = """
struct Foo { field: Bar }
impl Foo {
    fn run(&self) {
        self.field.method();
    }
}
"""
    edge = _by_callee(_detect(source), "method")
    assert edge.is_method_call is True
    assert edge.receiver_type == "Bar"
