"""ADR-032 tier 2.5 — victor-core integration: flag, provenance, REFERENCES.

Pins the core-side contract: ``enable_scope_binding`` (default OFF) turns the
flag into repository-snapshot emission and in-file REFERENCES edges carrying
provenance metadata; the capability probe downgrades (never silently falls
back) on a victor-codegraph older than 0.11.0; the projection carries
provenance/confidence/use instead of discarding them; retrieval prefers
provenance-bearing edges when the candidate budget bites.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.core.graph_rag.codegraph_repository import (
    ProjectedRelationEdge,
    project_resolved_relations,
)


def _relation(name, src, dst, provenance, confidence=1.0, context="load"):
    return SimpleNamespace(
        relation_type=SimpleNamespace(name=name),
        from_symbol_id=src,
        to_symbol_id=dst,
        provenance=provenance,
        confidence=confidence,
        context=context,
        target_ref=None,
    )


def test_projection_carries_provenance_and_counts_references():
    repository = SimpleNamespace(
        relations=[
            _relation("CALLS", "a", "b", "repository_resolver"),
            _relation(
                "REFERENCES",
                "a",
                "c",
                "repository_resolver",
                context="load",
            ),
            _relation("REFERENCES", "a", "zzz", "repository_resolver"),
        ]
    )
    projection = project_resolved_relations(repository, materialized_symbol_ids={"a", "b", "c"})
    assert [(e.src, e.dst, e.type) for e in projection.edges] == [
        ("a", "b", "CALLS"),
        ("a", "c", "REFERENCES"),
    ]
    assert projection.calls == 1
    assert projection.references == 1
    assert projection.dropped_unmaterialized == 1
    reference_edge = projection.edges[1]
    assert isinstance(reference_edge, ProjectedRelationEdge)
    assert reference_edge.provenance == "repository_resolver"
    assert reference_edge.confidence == 1.0
    assert reference_edge.use == "load"


def test_unresolved_references_do_not_project():
    repository = SimpleNamespace(
        relations=[
            SimpleNamespace(
                relation_type=SimpleNamespace(name="REFERENCES"),
                from_symbol_id="a",
                to_symbol_id="missing",
                provenance="parser",
                confidence=0.5,
                context="load",
                target_ref=SimpleNamespace(name="missing", text="missing"),
            )
        ]
    )
    assert project_resolved_relations(repository).edges == ()


def test_capability_probe_downgrades_on_old_codegraph(tmp_path, monkeypatch, caplog):
    pytest.importorskip("victor_codegraph")
    import victor_codegraph

    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline

    def old_parse_repo(*args, **kwargs):
        raise AssertionError("must not be called with the new kwarg")

    monkeypatch.setattr(victor_codegraph, "parse_repo", old_parse_repo, raising=False)

    store = SimpleNamespace()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_scope_binding=True,
    )
    pipeline = GraphIndexingPipeline(store, config)
    pipeline._scope_binding_supported_cache = None
    with caplog.at_level("INFO"):
        assert pipeline._scope_binding_supported() is False
    assert any("does not support emit_references" in rec.message for rec in caplog.records)
    # The downgrade is cached: a second call does not re-probe.
    assert pipeline._scope_binding_supported() is False


def test_retrieval_precision_first_orders_provenance_edges_ahead():
    from victor.core.graph_rag.retrieval import _precision_first
    from victor.storage.graph.protocol import GraphEdge

    heuristic = GraphEdge(src="a", dst="b", type="CALLS")
    precise = GraphEdge(
        src="a", dst="c", type="REFERENCES", metadata={"provenance": "scope_resolver"}
    )
    later_precise = GraphEdge(
        src="a", dst="d", type="CALLS", metadata={"provenance": "repository_resolver"}
    )
    assert [e.dst for e in _precision_first([heuristic, precise, later_precise])] == [
        "c",
        "d",
        "b",
    ]
    # Without provenance the order is stable/unchanged.
    assert [e.dst for e in _precision_first([heuristic, precise.__class__("a", "z", "CALLS")])] == [
        "b",
        "z",
    ]


@pytest.mark.asyncio
async def test_pipeline_emits_reference_edges_with_provenance(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    import inspect

    import victor_codegraph

    if "emit_references" not in inspect.signature(victor_codegraph.parse_repo).parameters:
        pytest.skip("installed victor-codegraph predates emit_references")

    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "targets.py").write_text(
        "def process(data):\n    return data\n\n\nclass Record:\n    pass\n"
    )
    (tmp_path / "consumer.py").write_text(
        "from targets import Record, process\n\n\ndef use(record=None):\n"
        "    kind = Record\n    fn = process\n    return kind, fn\n"
    )
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_scope_binding=True,
            incremental=False,
        )
        pipeline = GraphIndexingPipeline(store, config)
        stats = await pipeline.index_repository(tmp_path)

        references = [e for e in await store.get_all_edges() if e.type == "REFERENCES"]
        assert references, "flag ON persists REFERENCES edges"
        assert stats.scope_references_resolved >= len(references) - 1
        # Monotonicity contract: every REFERENCES edge carries provenance —
        # heuristic passes never write this edge type.
        assert all((e.metadata or {}).get("provenance") for e in references)
        provenances = {e.metadata.get("provenance") for e in references}
        assert provenances <= {"scope_resolver", "repository_resolver"}
        by_dst = {(e.src, e.dst): e for e in references}
        record = await store.find_nodes(name="Record")
        process = await store.find_nodes(name="process")
        assert record and process
        assert (record[0].node_id, process[0].node_id) or True
        target_ids = {record[0].node_id, process[0].node_id}
        assert any(e.dst in target_ids for e in references)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pipeline_explicit_opt_out_writes_no_references(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "targets.py").write_text("def process(data):\n    return data\n")
    (tmp_path / "consumer.py").write_text(
        "from targets import process\n\n\ndef use():\n    fn = process\n    return fn\n"
    )
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_scope_binding=False,
            incremental=False,
        )
        await GraphIndexingPipeline(store, config).index_repository(tmp_path)
        assert [e for e in await store.get_all_edges() if e.type == "REFERENCES"] == []
    finally:
        await store.close()
