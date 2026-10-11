from __future__ import annotations

import pytest

from victor_codegraph import CodeRelationType, parse, parse_repo, to_proxima_records


def test_rust_qualified_calls_do_not_bind_unrelated_local_functions():
    parsed = parse(
        "mod a { pub fn target() {} } mod b { pub fn target() {} } "
        "fn caller() { a::target(); external::target(); value.target(); }",
        language="rust",
        file_path="src/lib.rs",
    )
    calls = [r for r in parsed.relations if r.relation_type == CodeRelationType.CALLS]
    targets = {s.id: s for s in parsed.symbols}
    resolved = [r for r in calls if r.target_ref is None]
    assert len(resolved) == 1
    assert targets[resolved[0].to_symbol_id].scope_chain == ["a"]
    assert {r.target_ref.qualifier for r in calls if r.target_ref} == {"external", "value"}


def test_rust_repository_resolves_grouped_alias_and_path_dependency(tmp_path):
    for crate in ("app", "engine"):
        (tmp_path / crate / "src").mkdir(parents=True)
    (tmp_path / "app/Cargo.toml").write_text(
        '[package]\nname="app"\n[dependencies]\nengine-alias={path="../engine"}\n'
    )
    (tmp_path / "engine/Cargo.toml").write_text('[package]\nname="engine"\n')
    (tmp_path / "app/src/lib.rs").write_text(
        "mod codec; use crate::codec::{decode as parse}; "
        "use engine_alias::run; fn caller() { parse(); run(); external::run(); }"
    )
    (tmp_path / "app/src/codec.rs").write_text("pub fn decode() {}")
    (tmp_path / "engine/src/lib.rs").write_text("pub fn run() {}")
    repository = parse_repo(tmp_path)
    calls = [r for r in repository.relations if r.relation_type.name == "CALLS"]
    symbols = {s.id: s for s in repository.symbols}
    assert {symbols[r.to_symbol_id].location.file_path for r in calls if not r.target_ref} == {
        "app/src/codec.rs",
        "engine/src/lib.rs",
    }
    assert len([r for r in calls if r.target_ref]) == 1
    imports = [r for r in repository.relations if r.relation_type.name == "IMPORTS"]
    assert len(imports) == 2
    assert all(r.target_ref is None for r in imports)


@pytest.mark.parametrize(
    "source,filename",
    [
        ("fn run() {} mod child { fn caller() { run(); } }", "src/lib.rs"),
        ("use crate::dep::run; fn caller() { run(); }", "src/lib.rs"),
        ("mod dep; use crate::dep::run; mod child { fn caller() { run(); } }", "src/lib.rs"),
        ("use crate::run; fn caller() { run(); }", "src/main.rs"),
        ("#[cfg(any())] mod dep; use crate::dep::run; fn caller() { run(); }", "src/lib.rs"),
        ("#[cfg(any())] fn run() {} fn caller() { run(); }", "src/lib.rs"),
        ("mod dep; #[cfg(any())] use crate::dep::run; fn caller() { run(); }", "src/lib.rs"),
        ("#![cfg(any())] mod dep; use crate::dep::run; fn caller() { run(); }", "src/lib.rs"),
        ('#[path="other.rs"] mod dep; use crate::dep::run; fn caller() { run(); }', "src/lib.rs"),
    ],
)
def test_rust_repository_does_not_invent_scope_or_module_membership(tmp_path, source, filename):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname="app"\n')
    (tmp_path / "src/lib.rs").write_text("pub fn run() {}")
    (tmp_path / "src/dep.rs").write_text("pub fn run() {}")
    (tmp_path / filename).write_text(source)
    parsed = parse_repo(tmp_path)
    calls = [r for r in parsed.relations if r.relation_type.name == "CALLS"]
    assert calls and all(r.target_ref is not None for r in calls)


@pytest.mark.parametrize(
    "target", ['[lib]\npath="/tmp/outside.rs"', '[[bin]]\npath="../../outside.rs"']
)
def test_rust_external_cargo_target_preserves_unresolved_import(tmp_path, target):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname="app"\n' + target)
    (tmp_path / "src/lib.rs").write_text("use crate::dep::run; fn caller() { run(); }")
    parsed = parse_repo(tmp_path)
    assert all(
        r.target_ref for r in parsed.relations if r.relation_type.name in {"CALLS", "IMPORTS"}
    )


@pytest.mark.parametrize("shared", [False, True])
def test_rust_binary_child_module_never_binds_library_crate(tmp_path, shared):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname="app"\n')
    (tmp_path / "src/lib.rs").write_text(("mod helper; " if shared else "") + "pub fn run() {}")
    (tmp_path / "src/main.rs").write_text("mod helper; fn main() {}")
    (tmp_path / "src/helper.rs").write_text("use crate::run; fn caller() { run(); }")
    parsed = parse_repo(tmp_path)
    calls = [r for r in parsed.relations if r.relation_type.name == "CALLS"]
    assert calls and all(r.target_ref is not None for r in calls)


def test_rust_grouped_glob_retains_import_without_guessing_calls(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname="app"\n')
    (tmp_path / "src/lib.rs").write_text("mod dep; use crate::dep::{*}; fn caller() { run(); }")
    (tmp_path / "src/dep.rs").write_text("pub fn run() {}")
    parsed = parse_repo(tmp_path)
    imports = [r for r in parsed.relations if r.relation_type.name == "IMPORTS"]
    assert len(imports) == 1
    calls = [r for r in parsed.relations if r.relation_type.name == "CALLS"]
    assert len(calls) == 1 and calls[0].target_ref is not None


def test_rust_grouped_import_preserves_unresolved_sibling_and_module_alias(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "Cargo.toml").write_text('[package]\nname="app"\n')
    (tmp_path / "src/lib.rs").write_text(
        "mod dep; use crate::{dep::{self as d, run}, absent::missing}; " "fn caller() { d::run(); }"
    )
    (tmp_path / "src/dep.rs").write_text("pub fn run() {}")
    parsed = parse_repo(tmp_path)
    imports = [r for r in parsed.relations if r.relation_type.name == "IMPORTS"]
    unresolved = [r for r in imports if r.target_ref is not None]
    assert len(unresolved) == 1
    assert unresolved[0].target_ref.name == "crate::absent::missing"
    calls = [r for r in parsed.relations if r.relation_type.name == "CALLS"]
    assert len(calls) == 1 and calls[0].target_ref is None


def _symbols(parsed):
    return {s.fully_qualified_name: s for s in parsed.symbols}


def test_in_file_resolution_is_scope_aware():
    parsed = parse(
        """
class A:
    def target(self): return 1
    def run(self): return self.target()

class B:
    def target(self): return 2
""",
        file_path="m.py",
    )
    symbols = _symbols(parsed)
    call = next(r for r in parsed.relations if r.relation_type == CodeRelationType.CALLS)
    assert call.from_symbol_id == symbols["m.py::A::run"].id
    assert call.to_symbol_id == symbols["m.py::A::target"].id


def test_recursive_calls_are_retained():
    parsed = parse("def f():\n    return f()\n", file_path="m.py")
    symbol = parsed.symbols[0]
    calls = [r for r in parsed.relations if r.relation_type == CodeRelationType.CALLS]
    assert len(calls) == 1
    assert calls[0].from_symbol_id == calls[0].to_symbol_id == symbol.id


def test_repository_pass_resolves_unique_cross_file_call(tmp_path):
    (tmp_path / "a.py").write_text("def target():\n    return 1\n")
    (tmp_path / "b.py").write_text("from a import target\n\ndef caller():\n    return target()\n")

    indexed = parse_repo(tmp_path, repo_id="repo")
    target = next(s for s in indexed.symbols if s.simple_name == "target")
    call = next(r for r in indexed.relations if r.relation_type == CodeRelationType.CALLS)
    assert call.to_symbol_id == target.id
    assert call.target_ref is None
    assert call.confidence == 1.0

    records = to_proxima_records(indexed, "repo")
    target_oid = next(
        r["oid"] for r in records if "code_symbol" in r["labels"] and r["props"]["name"] == "target"
    )
    call_record = next(
        r for r in records if "graph_edge" in r["labels"] and r["edge"]["edge_type"] == "CALLS"
    )
    assert call_record["edge"]["to_oid"] == target_oid


def test_repository_does_not_guess_cross_file_target_without_import(tmp_path):
    (tmp_path / "a.py").write_text("def target(): return 1\n")
    (tmp_path / "b.py").write_text("def caller(): return target()\n")
    indexed = parse_repo(tmp_path, repo_id="repo")
    call = next(r for r in indexed.relations if r.relation_type.name == "CALLS")
    assert call.target_ref is not None
    assert call.confidence < 1.0


def test_repository_index_emits_file_containment_and_import_edges(tmp_path):
    (tmp_path / "a.py").write_text("from b import target\n\ndef caller(): return target()\n")
    (tmp_path / "b.py").write_text("def target(): return 1\n")
    indexed = parse_repo(tmp_path, repo_id="repo")

    files = {s.location.file_path: s for s in indexed.symbols if s.symbol_type.name == "FILE"}
    assert set(files) == {"a.py", "b.py"}
    imports = [r for r in indexed.relations if r.relation_type.name == "IMPORTS"]
    assert len(imports) == 1
    assert imports[0].from_symbol_id == files["a.py"].id
    assert imports[0].to_symbol_id == files["b.py"].id
    assert imports[0].target_ref is None

    contains = [r for r in indexed.relations if r.relation_type.name == "CONTAINS"]
    caller = next(s for s in indexed.symbols if s.simple_name == "caller")
    assert any(
        r.from_symbol_id == files["a.py"].id and r.to_symbol_id == caller.id for r in contains
    )


def test_repository_index_resolves_package_relative_import(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.py").write_text("from .b import target\n")
    (package / "b.py").write_text("def target(): return 1\n")
    indexed = parse_repo(tmp_path, repo_id="repo")
    files = {s.location.file_path: s for s in indexed.symbols if s.symbol_type.name == "FILE"}
    relation = next(r for r in indexed.relations if r.relation_type.name == "IMPORTS")
    assert relation.from_symbol_id == files["pkg/a.py"].id
    assert relation.to_symbol_id == files["pkg/b.py"].id


def test_unresolved_relation_gets_explicit_external_node():
    parsed = parse("def caller():\n    return external_api()\n", file_path="m.py")
    call = parsed.relations[0]
    assert call.target_ref is not None
    assert call.target_ref.name == "external_api"

    records = to_proxima_records(parsed, "repo")
    node_oids = {r["oid"] for r in records if "oid" in r and "graph_node" in r["labels"]}
    edge = next(r for r in records if "graph_edge" in r["labels"])
    assert edge["edge"]["to_oid"] in node_oids
    assert any("external_symbol" in r["labels"] for r in records)
