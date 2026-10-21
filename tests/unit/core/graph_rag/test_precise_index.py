"""ADR-032 tier 3 — precise-index harvesting: unit + pipeline contracts.

All SCIP-dependent paths stay importorskip/tool-skip gated (no CI install
changes); the join core is tested against SimpleNamespace documents shaped
like the REAL scip-python output (``python <file>`<dotted path>`` symbols,
definition-role bit, per-document occurrences), and the vendored protobuf
bindings are exercised directly when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.core.graph_rag.precise_index import (
    PreciseHarvest,
    detect_precise_tool,
    join_occurrences,
    python_symbol_to_parts,
)


def _occ(symbol: str, line: int, roles: int):
    # Real scip-python shape: Occurrence.range is the deprecated packed-int32
    # encoding [startLine, startChar]; the typed single/multi_line_range
    # alternatives normalize in _occurrence_offset.
    return SimpleNamespace(symbol=symbol, symbol_roles=roles, range=[line, 0])


def _doc(path: str, occurrences, symbols=()):
    return SimpleNamespace(relative_path=path, occurrences=occurrences, symbols=list(symbols))


def _index(*docs):
    return SimpleNamespace(documents=list(docs))


@dataclass
class N:
    """GraphNode-shaped fixture: lowercase type, parent_id, no scope_chain."""

    node_id: str
    name: str
    file: str
    type: str = "function"
    parent_id: str | None = None
    scope_chain: list | None = None


def _nodes():
    return [
        N("fn-id", "helper", "src/m.py"),
        N("cls-id", "Worker", "src/m.py", type="class"),
        N("meth-id", "run", "src/m.py", type="method", parent_id="cls-id"),
        N("other-id", "helper", "src/other.py"),
    ]


def _sym(symbol: str, kind: int):
    return SimpleNamespace(symbol=symbol, kind=kind)


DEF = 1  # scip.proto SymbolRole definition bit


def test_symbol_parts_backtick_grammar():
    assert python_symbol_to_parts("python src/m.py`pkg.mod.func().") == {
        "file": "src/m.py",
        "path": "pkg.mod.func",
    }
    # Parameter lists and the trailing accessor both strip.
    assert python_symbol_to_parts("python src/m.py`mod.f(a, b)").get("path") == "mod.f"
    assert python_symbol_to_parts("rust crate::x") is None
    assert python_symbol_to_parts("python src/m.py helper") is None


def test_unique_file_name_join_resolves_functions_classes_methods():
    nodes = _nodes()
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.helper().", 1, DEF),
                _occ("python src/m.py`pkg.Worker().", 10, DEF),
                _occ("python src/m.py`pkg.Worker.run().", 20, DEF),
                # references: helper used inside run
                _occ("python src/m.py`pkg.helper().", 22, 0),
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes[:3]})
    assert harvest.resolved == 1
    assert harvest.edges == [
        (
            "meth-id",
            "fn-id",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]
    assert harvest.unjoined == 0


def test_cross_file_marker_rejected_and_own_file_used():
    nodes = _nodes()
    # The symbol's marker says another file -> must NOT join to src/m.py's
    # same-named node; the join key is the DOCUMENT's file anyway.
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/other.py`pkg.helper().", 1, DEF),
                _occ("python src/other.py`pkg.helper().", 3, 0),
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes[:3]})
    assert harvest.resolved == 0 and harvest.edges == []
    assert harvest.unjoined >= 1


def test_same_name_multiple_scopes_scope_chain_disambiguates():
    nodes = _nodes()
    # Two run methods in different classes: meth-id chains via parent_id
    # (the real GraphNode path), meth2-id carries an explicit chain.
    nodes.append(N("other-cls", "Other", "src/m.py", type="class"))
    nodes.append(N("meth2-id", "run", "src/m.py", type="method", parent_id="other-cls"))
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.Worker.run().", 5, DEF),
                _occ("python src/m.py`pkg.Other.run().", 10, DEF),
                _occ("python src/m.py`pkg.Worker.run().", 12, 0),
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes})
    # The reference at line 12 lies inside Other.run (def line 10): the
    # positional owner rule attributes it to meth2-id — correct scoping.
    assert harvest.edges and harvest.edges[0][:2] == ("meth2-id", "meth-id")


def test_ambiguity_stays_unjoined_never_guessed():
    nodes = _nodes()
    # Two same-named helpers in the SAME file at top level: unjoinable.
    nodes.append(N("dup-id", "helper", "src/m.py"))
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.helper().", 1, DEF),
                _occ("python src/m.py`pkg.helper().", 3, 0),
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes})
    assert harvest.resolved == 0 and harvest.edges == []
    assert harvest.unjoined == 2  # definition + reference both counted


def test_kind_disagreement_stays_unjoined():
    nodes = _nodes()
    # scip says helper is a class; the only node named helper is a function.
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.helper().", 1, DEF),
                _occ("python src/m.py`pkg.helper().", 3, 0),
            ],
            symbols=[_sym("python src/m.py`pkg.helper().", 5)],  # SymbolKind.CLASS
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes[:1]})
    assert harvest.resolved == 0 and harvest.edges == []
    assert harvest.unjoined == 2


def test_kind_agreement_selects_among_same_name_nodes():
    nodes = _nodes()
    # A class Worker and a function named Worker: scip's class kind must
    # bind the definition (and its references) to the class node only.
    nodes.append(N("fn-worker", "Worker", "src/m.py", type="function"))
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.Worker().", 4, DEF),
                _occ("python src/m.py`pkg.helper().", 10, DEF),
                _occ("python src/m.py`pkg.Worker().", 12, 0),  # ref inside helper
            ],
            symbols=[
                _sym("python src/m.py`pkg.Worker().", 5),
                _sym("python src/m.py`pkg.helper().", 3),
            ],
        )
    )
    harvest = join_occurrences(
        index, {"src/m.py": [n for n in nodes if n.file == "src/m.py"], "src/other.py": [nodes[3]]}
    )
    # Edge flows helper -> the CLASS node, never the same-named function.
    assert harvest.edges == [
        (
            "fn-id",
            "cls-id",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]


def test_reference_before_any_definition_counts_unjoined():
    nodes = _nodes()
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.helper().", 5, 0),  # ref BEFORE def line 9
                _occ("python src/m.py`pkg.helper().", 9, DEF),
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes[:1]})
    assert harvest.resolved == 0 and harvest.unjoined == 1  # ownerless ref counted


def test_self_reference_and_duplicate_edges_dropped():
    nodes = _nodes()
    index = _index(
        _doc(
            "src/m.py",
            [
                _occ("python src/m.py`pkg.helper().", 1, DEF),
                _occ("python src/m.py`pkg.helper().", 3, 0),  # self
                _occ("python src/m.py`pkg.helper().", 5, 0),  # dup pair
            ],
        )
    )
    harvest = join_occurrences(index, {"src/m.py": nodes[:1]})
    assert harvest.resolved == 0 and harvest.edges == []


def test_dump_command_timeout_label_and_tool_detection(monkeypatch):
    from victor.core.graph_rag.precise_index import harvest_scip_index

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/scip-python")
    import subprocess as subprocess_mod

    def timeout(*a, **k):
        raise subprocess_mod.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess_mod, "run", timeout)
    assert harvest_scip_index(Path("/tmp"), Path("/tmp/o.scip")) is None
    # No pip-module branch: absent binary -> None.
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert detect_precise_tool("auto") is None


def test_vendored_bindings_parse_real_schema():
    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2

    index = scip_pb2.Index()
    index.metadata.project_root = "/repo"
    document = index.documents.add()
    document.relative_path = "src/m.py"
    occurrence = document.occurrences.add()
    occurrence.symbol = "python src/m.py`pkg.helper()."
    occurrence.symbol_roles = 1
    occurrence.range.extend([1, 0])

    blob = index.SerializeToString()
    parsed = scip_pb2.Index()
    parsed.ParseFromString(blob)
    assert parsed.documents[0].occurrences[0].symbol == "python src/m.py`pkg.helper()."


@pytest.mark.asyncio
async def test_harvest_precise_edges_end_to_end_through_vendored_reader(tmp_path):
    """Real protobuf round-trip: vendored bindings write, read, and join."""
    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag import precise_index
    from victor.core.graph_rag._scip import scip_pb2

    index = scip_pb2.Index()
    doc = index.documents.add()
    doc.relative_path = "src/m.py"
    table = doc.symbols.add()
    table.symbol = "python src/m.py`pkg.helper()."
    table.kind = 3  # SymbolKind.FUNCTION
    d = doc.occurrences.add()
    d.symbol = "python src/m.py`pkg.helper()."
    d.symbol_roles = 1
    d.range.extend([1, 0])
    r = doc.occurrences.add()
    r.symbol = "python src/m.py`pkg.Worker()."
    r.symbol_roles = 0
    r.single_line_range.line = 5
    r.single_line_range.start_character = 4

    nodes = [N("fn-id", "helper", "src/m.py"), N("cls-id", "Worker", "src/m.py", type="class")]
    # Worker has no joinable definition occurrence: reference is ownerless.
    path = tmp_path / "dump.scip"
    path.write_bytes(index.SerializeToString())
    harvest = precise_index.harvest_precise_edges(path, {"src/m.py": nodes})
    assert harvest.skipped_reason is None
    assert harvest.resolved == 0 and harvest.edges == []
    assert harvest.unjoined == 1  # definition joined, ownerless ref counted


def test_harvest_precise_edges_unreadable_dump_reports_no_reader(tmp_path):
    from victor.core.graph_rag import precise_index

    bogus = tmp_path / "not-a-scip.proto"
    bogus.write_bytes(b"\x00garbage\xff")
    harvest = precise_index.harvest_precise_edges(bogus, {})
    assert harvest.skipped_reason == "no-reader"
    assert harvest.edges == []


def test_occurrence_offset_handles_all_range_encodings():
    from victor.core.graph_rag.precise_index import _occurrence_offset

    packed = SimpleNamespace(range=[3, 7])
    assert _occurrence_offset(packed) == 3 * 10_000 + 7
    empty = SimpleNamespace(range=[])
    assert _occurrence_offset(empty) == 0
    # Typed encodings against the REAL vendored schema: flat messages, and
    # unset proto3 message fields must not shadow the set one.
    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2

    single = scip_pb2.Occurrence()
    single.single_line_range.line = 2
    single.single_line_range.start_character = 9
    assert _occurrence_offset(single) == 2 * 10_000 + 9
    multi = scip_pb2.Occurrence()
    multi.multi_line_range.start_line = 4
    multi.multi_line_range.start_character = 1
    # single_line_range UNSET (default instance): must not win over the set
    # multi_line_range — the old nested-shape reader crashed or misread here.
    assert _occurrence_offset(multi) == 4 * 10_000 + 1
    assert _occurrence_offset(scip_pb2.Occurrence()) == 0


def test_scope_chain_walks_container_parents_only():
    from victor.core.graph_rag.precise_index import _scope_chain

    file_node = N("file-id", "m.py", "src/m.py", type="file")
    cls = N("cls-id", "Worker", "src/m.py", type="class", parent_id="file-id")
    fn = N("fn-id", "helper", "src/m.py", parent_id="cls-id")
    by_id = {n.node_id: n for n in (file_node, cls, fn)}
    # file node is NOT a container: walk stops at the class boundary.
    assert _scope_chain(fn, by_id) == ["Worker"]
    # explicit field wins over any parent structure.
    explicit = N("x", "run", "src/m.py", scope_chain=["Other"])
    assert _scope_chain(explicit, by_id) == ["Other"]
    # unknown parent id ends the walk without crashing.
    orphan = N("o", "solo", "src/m.py", parent_id="missing-id")
    assert _scope_chain(orphan, by_id) == []


def test_unique_join_unmapped_kind_keeps_population_typed_empty_unjoins():
    from victor.core.graph_rag.precise_index import _unique_join

    nodes = [N("fn-id", "helper", "src/m.py")]
    parts = {"file": "src/m.py", "path": "pkg.helper"}
    no_chain = lambda node: []  # noqa: E731 - trivial fixture lambda
    # Unmapped scip kind (e.g. a parameter): no filtering, join proceeds.
    assert _unique_join(parts, nodes, 999, no_chain) == "fn-id"
    # Mapped kind with zero matching-type nodes: unjoined, not guessed.
    assert _unique_join(parts, nodes, 5, no_chain) is None  # CLASS vs function node


def test_harvest_scip_index_oserror_degrades_to_none(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    from victor.core.graph_rag import precise_index

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/scip-python")

    def boom(*a, **k):
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess_mod, "run", boom)
    assert precise_index.harvest_scip_index(tmp_path, tmp_path / "o.scip") is None


@pytest.mark.asyncio
async def test_real_harvest_method_downgrades_without_toolchain(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag import precise_index
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    monkeypatch.setattr(precise_index, "harvest_scip_index", lambda *a, **k: None)

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_precise_index=True,
            incremental=False,
        )
        pipeline = GraphIndexingPipeline(store, config)
        harvest = await pipeline._harvest_precise_index(tmp_path)
        assert harvest.skipped_reason == "toolchain-or-dump-failed"
    finally:
        await store.close()
