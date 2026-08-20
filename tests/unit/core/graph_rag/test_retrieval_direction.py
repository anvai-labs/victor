# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Retrieval must be able to walk edges backwards.

"Who calls `X`?" and "what breaks if I change `X`?" are the questions a code
graph exists to answer, and both are *inward* walks: the answer is the set of
nodes with an edge **into** the target. Until this change every neighbour fetch
in `MultiHopRetriever` hardcoded ``direction="out"`` and read ``edge.dst``, so
those questions were structurally unanswerable — the walk left the target and
went looking at what the target itself calls.

Nothing caught it. The store has supported ``direction="in"`` all along and is
tested for it; `impact_analysis` passes direction correctly. The retriever —
which powers context assembly, `code_search(mode="graph")` and
`graph_semantic_search` — sat between them with the value pinned, and no test
asserted that an expansion ever *recovers a known neighbour set*. Tests asserted
shapes (seeds present, ``<= top_k`` returned, round-trip counts); all of them
pass against a walk that returns the wrong nodes, or none.

So these tests assert recall against ground truth the fixture knows by
construction, which is the only property that fails when direction is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from victor.core.graph_rag import retrieval as retrieval_module
from victor.core.graph_rag.config import RetrievalConfig
from victor.core.graph_rag.retrieval import MultiHopRetriever
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore

CALLERS = 6
CALLEES = 3


@pytest.fixture(autouse=True)
def _isolate_query_cache() -> Any:
    """Give every test a private, non-persisted cache.

    Retrieval memoizes on (query, config, repo). These stores report no
    ``repo_root``, so without isolation one test is served another's result —
    and a direction test served a cached outward answer would pass while
    measuring nothing. Persistence is off so the L2 layer does not write into
    the shared ``project.db`` that other suites count rows in.
    """
    from victor.core.graph_rag import query_cache as query_cache_module

    query_cache_module.reset_graph_query_cache()
    original = query_cache_module.GraphQueryCacheConfig

    class _EphemeralConfig(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("persist", False)
            super().__init__(*args, **kwargs)

    query_cache_module.GraphQueryCacheConfig = _EphemeralConfig
    try:
        yield
    finally:
        query_cache_module.GraphQueryCacheConfig = original
        query_cache_module.reset_graph_query_cache()


@pytest.fixture
async def store(tmp_path: Path) -> SqliteGraphStore:
    """A hub with a known caller set and a known callee set.

    ``target`` is called by ``CALLERS`` distinct functions and itself calls
    ``CALLEES`` others. The two sets are disjoint, so an outward walk cannot
    accidentally satisfy an inward assertion or vice versa — which is exactly
    the confusion that let the bug live.
    """
    graph = SqliteGraphStore(project_path=tmp_path / "direction.db")
    await graph.initialize()

    nodes: List[GraphNode] = [
        GraphNode(
            node_id="target",
            type="function",
            name="target_symbol",
            file="target.py",
            line=1,
            lang="python",
        )
    ]
    edges: List[GraphEdge] = []

    for i in range(CALLERS):
        nodes.append(
            GraphNode(
                node_id=f"caller_{i}",
                type="function",
                name=f"caller_{i}",
                file=f"callers/mod_{i}.py",
                line=10 + i,
                lang="python",
            )
        )
        edges.append(GraphEdge(src=f"caller_{i}", dst="target", type="CALLS"))

    for i in range(CALLEES):
        nodes.append(
            GraphNode(
                node_id=f"callee_{i}",
                type="function",
                name=f"callee_{i}",
                file=f"callees/mod_{i}.py",
                line=20 + i,
                lang="python",
            )
        )
        edges.append(GraphEdge(src="target", dst=f"callee_{i}", type="CALLS"))

    await graph.upsert_nodes(nodes)
    await graph.upsert_edges(edges)
    return graph


def _config(mode: str, **overrides: Any) -> RetrievalConfig:
    """One config per test; `mode` is part of the cache key, so vary it."""
    params: dict[str, Any] = {
        "seed_count": 3,
        "max_hops": 1,
        "top_k": 25,
        "mode": mode,
        "edge_types": {"CALLS"},
    }
    params.update(overrides)
    return RetrievalConfig(**params)


async def _retrieved_ids(store: SqliteGraphStore, config: RetrievalConfig) -> set[str]:
    result = await MultiHopRetriever(store, config).retrieve("target_symbol", config)
    return {node.node_id for node in result.nodes}


@pytest.mark.asyncio
async def test_inward_traversal_recovers_the_full_caller_set(store: SqliteGraphStore) -> None:
    """The headline gap: "who calls target_symbol" must answer with the callers."""
    found = await _retrieved_ids(store, _config("dir-in", direction="in"))

    expected = {f"caller_{i}" for i in range(CALLERS)}
    assert expected <= found, f"missing callers: {sorted(expected - found)}"


@pytest.mark.asyncio
async def test_inward_traversal_does_not_return_callees(store: SqliteGraphStore) -> None:
    """Teeth for the assertion above: an outward walk must not satisfy it.

    Without this, a `direction="both"` implementation of `"in"` would pass the
    recall test while still answering the wrong question.
    """
    found = await _retrieved_ids(store, _config("dir-in-exclusive", direction="in"))

    assert not {f"callee_{i}" for i in range(CALLEES)} & found


@pytest.mark.asyncio
async def test_outward_traversal_is_unchanged_and_is_the_default(
    store: SqliteGraphStore,
) -> None:
    """The knob must not move existing behaviour: unset still means outward."""
    explicit = await _retrieved_ids(store, _config("dir-out", direction="out"))
    default = await _retrieved_ids(store, _config("dir-default"))

    callees = {f"callee_{i}" for i in range(CALLEES)}
    assert callees <= explicit
    assert callees <= default
    assert not {f"caller_{i}" for i in range(CALLERS)} & default


@pytest.mark.asyncio
async def test_both_directions_returns_callers_and_callees(store: SqliteGraphStore) -> None:
    found = await _retrieved_ids(store, _config("dir-both", direction="both"))

    assert {f"caller_{i}" for i in range(CALLERS)} <= found
    assert {f"callee_{i}" for i in range(CALLEES)} <= found


@pytest.mark.asyncio
async def test_unknown_direction_falls_back_to_outward(store: SqliteGraphStore) -> None:
    """A typo degrades to the documented default, it does not abort a traversal.

    The store raises ValueError on an unrecognised direction, and that would
    surface mid-walk as a warning-and-empty-neighbours — a silent wrong answer.
    Normalizing at the boundary keeps the failure mode legible.
    """
    found = await _retrieved_ids(store, _config("dir-bogus", direction="sideways"))

    assert {f"callee_{i}" for i in range(CALLEES)} <= found


@pytest.mark.asyncio
async def test_direction_participates_in_the_cache_key(store: SqliteGraphStore) -> None:
    """Same query, opposite directions, different answers — so different entries.

    Direction was added to the retrieval config after the cache key was written;
    had it not been added to the key too, whichever direction ran first would be
    served to the other.
    """
    inward = await _retrieved_ids(store, _config("dir-cache", direction="in"))
    outward = await _retrieved_ids(store, _config("dir-cache", direction="out"))

    assert {f"caller_{i}" for i in range(CALLERS)} <= inward
    assert {f"callee_{i}" for i in range(CALLEES)} <= outward
    assert inward != outward


@pytest.mark.asyncio
async def test_non_outward_request_stays_off_the_batched_traversal(
    store: SqliteGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`multi_hop_traverse_parallel` takes no direction and walks outward only.

    It is opt-in today, so this guards against a future flip silently answering
    inward questions with outward results.
    """
    monkeypatch.setattr(MultiHopRetriever, "_should_use_parallel", lambda self, config: True)
    calls: list[str] = []
    original = store.multi_hop_traverse_parallel

    async def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("parallel")
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "multi_hop_traverse_parallel", spy)

    found = await _retrieved_ids(store, _config("dir-parallel-guard", direction="in"))

    assert calls == []
    assert {f"caller_{i}" for i in range(CALLERS)} <= found


@pytest.mark.asyncio
async def test_context_assembly_surfaces_callers(store: SqliteGraphStore) -> None:
    """The knob has to be *set* somewhere, or it ships dead.

    A config option no production path selects is indistinguishable from the
    bug it fixes — this repo has a long history of exactly that. This asserts
    at the product surface: context assembled for a task mentioning a symbol
    includes the code that calls it, which is what "what breaks if I change
    this" needs and what the outward-only walk could never provide.
    """
    from victor.context.graph_context_builder import GraphEnhancedContextBuilder

    builder = GraphEnhancedContextBuilder(store)
    nodes = await builder._identify_relevant_symbols("target_symbol", max_symbols=25)

    found = {node.node_id for node in nodes}
    assert {f"caller_{i}" for i in range(CALLERS)} <= found


def test_neighbor_endpoint_skips_self_loops() -> None:
    """A self-loop has no neighbour; returning the node re-enqueues itself."""
    loop = GraphEdge(src="n", dst="n", type="CALLS")

    assert retrieval_module._neighbor_endpoint(loop, "n", "both") is None
