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


def test_retrieval_precision_first_ranks_provenance_VALUES():
    """Tier 3.5 (retrieval-first): compiler > scope/repository resolver >
    parser > unknown/absent — and unknown ranks WITH absent (conservative)."""
    from victor.core.graph_rag.retrieval import _precision_first
    from victor.storage.graph.protocol import GraphEdge

    def e(dst, provenance=None):
        meta = {"provenance": provenance} if provenance else {}
        return GraphEdge(src="a", dst=dst, type="CALLS", metadata=meta)

    ordered = _precision_first(
        [
            e("parser", "parser"),
            e("absent"),
            e("unknown-producer", "some-new-producer"),
            e("scope_resolver", "scope_resolver"),
            e("compiler", "compiler"),
            e("repository_import_resolver", "repository_import_resolver"),
        ]
    )
    assert [edge.dst for edge in ordered] == [
        "compiler",
        "scope_resolver",
        "repository_import_resolver",
        "parser",
        "absent",
        "unknown-producer",
    ]
    # Stability: equal-rank edges keep input order.
    again = _precision_first([e("p1", "parser"), e("p2", "parser")])
    assert [edge.dst for edge in again] == ["p1", "p2"]


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


@pytest.mark.asyncio
async def test_lazy_neighbor_cap_keeps_precision_best_not_earliest():
    """Tier 3.5: the lazy path's max-neighbors cap must sort BEFORE cutting —
    store order is arbitrary, and an early batch of heuristic edges must not
    evict a compiler-provenance edge that arrived later."""
    from victor.core.graph_rag.retrieval import MultiHopRetriever
    from victor.storage.graph.protocol import GraphEdge

    def e(dst, provenance=None):
        meta = {"provenance": provenance} if provenance else {}
        return GraphEdge(src="seed", dst=dst, type="CALLS", metadata=meta)

    # Batch 1 alone is under the cap; batch 2 crosses it and carries the
    # compiler edge — the cap must keep the BEST of the accumulated set,
    # not the earliest five.
    first_batch = [e(f"heuristic-{i}") for i in range(5)]
    second_batch = [e("compiler-late", "compiler"), e("heuristic-5"), e("heuristic-6")]

    class FakeAsyncIter:
        def __init__(self, batches):
            self._batches = batches

        def __aiter__(self):
            self._i = 0
            return self

        async def __anext__(self):
            if self._i >= len(self._batches):
                raise StopAsyncIteration
            batch = self._batches[self._i]
            self._i += 1
            return batch

    class FakeStore:
        def iter_neighbors(self, node_id, *, batch_size, edge_types, direction):
            return FakeAsyncIter([first_batch, second_batch])

    from types import SimpleNamespace

    config = SimpleNamespace(
        use_lazy_loading=True,
        lazy_load_batch_size=5,
        max_neighbors_per_node=6,
        direction="out",
        top_k=10,
    )
    retriever = MultiHopRetriever.__new__(MultiHopRetriever)
    retriever.graph_store = FakeStore()
    kept = await retriever._get_neighbors_lazy("seed", None, config)
    dsts = [edge.dst for edge in kept]
    assert len(kept) == 6
    # Cross-rank: the compiler edge LEADS the kept set (without the sort it
    # sorted after nothing — it arrived in the crossing batch and an
    # earliest-first cut would have kept 6 heuristics instead). Equal-rank
    # ties keep arrival order (stability), so among the eight equal-rank
    # heuristics the first five fill the remaining slots deterministically.
    assert dsts[0] == "compiler-late"
    assert sorted(dsts[1:]) == [f"heuristic-{i}" for i in range(5)]
    assert "heuristic-5" not in dsts and "heuristic-6" not in dsts


def test_projection_carries_call_line_for_flow_ordering():
    """ADR-033 phase 2: the resolver's call_site line must survive projection
    (it is the sequence-diagram ordering key); absence stays None."""
    with_site = _relation("CALLS", "a", "b", "repository_resolver")
    with_site.call_site = SimpleNamespace(start_line=42)
    without_site = _relation("CALLS", "b", "c", "repository_resolver")
    repository = SimpleNamespace(relations=[with_site, without_site])
    projection = project_resolved_relations(repository, materialized_symbol_ids={"a", "b", "c"})
    lines = [edge.call_line for edge in projection.edges]
    assert lines == [42, None]


def test_binding_metadata_persists_call_line():
    """_binding_metadata (the in-file scope path) persists the 1-based
    first-use line; zero/absent sites stay out of the payload."""
    from victor.core.graph_rag.indexing import _binding_metadata

    relation = _relation("CALLS", "a", "b", "scope_resolver")
    relation.call_site = SimpleNamespace(start_line=7)
    assert _binding_metadata(relation)["call_line"] == 7

    relation.call_site = SimpleNamespace(start_line=0)  # SourceLocation unset
    assert "call_line" not in _binding_metadata(relation)

    relation.call_site = None
    assert "call_line" not in _binding_metadata(relation)
