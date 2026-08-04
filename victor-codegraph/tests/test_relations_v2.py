from __future__ import annotations

from victor_codegraph import CodeRelationType, parse, parse_repo, to_proxima_records


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
