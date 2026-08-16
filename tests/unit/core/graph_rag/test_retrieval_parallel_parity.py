# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""`retrieve_parallel` must return what `retrieve` returns, only faster.

`MultiHopRetriever` has carried two expansion strategies for some time. The
serial one (`retrieve`) walks a BFS queue one node at a time, issuing a
`get_node_by_id` **and** a `get_neighbors` per node. The parallel one
(`retrieve_parallel`) hands the whole seed set to
`graph_store.multi_hop_traverse_parallel` in one call.

Only the serial one is reachable from production code. The sole reference to the
parallel one anywhere in the tree was::

    assert hasattr(retriever, "retrieve_parallel")

which cannot detect that the method is unused, that it returns different results,
or that it raises. All three were true.

The two strategies are deliberately **not** byte-identical, and the reason is
worth recording. Serial expansion runs ``while queue and len(results) <
config.top_k``: it stops the moment it has ``top_k`` nodes, then ranks and slices
to ``top_k``. The candidate set is therefore chosen entirely by BFS discovery
order and ranking is decorative — it reorders a set traversal already truncated,
and can never promote a distant-but-relevant node over a near-but-irrelevant one.

Parallel expansion walks the k-hop neighbourhood and ranks **all** of it before
taking ``top_k``, so ranking finally decides the answer. That is a deliberate
improvement, not a regression, so these tests assert the properties that must
hold rather than equality with the older behaviour:

1. **A superset of candidates, properly ranked** — seeds present, never more than
   ``top_k`` returned, scores consistent with hop distance.
2. **Fewer round trips.** That is the entire point; an efficiency claim that is
   not asserted is a comment, and this one was wrong for long enough to matter.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

from victor.core.graph_rag.config import RetrievalConfig
from victor.core.graph_rag import retrieval as retrieval_module
from victor.core.graph_rag.retrieval import MultiHopRetriever
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture(autouse=True)
def _isolate_query_cache() -> Any:
    """Retrieval memoizes results, and the cache outlives an in-process reset.

    `reset_graph_query_cache()` clears the L1 singleton, but there is an L2 layer
    persisted in the repo's `project.db`, and these stores report no `repo_root`
    so every test shares one cache key. The reset alone is therefore not enough —
    each test also varies `mode`, which `RetrievalConfig` documents as part of the
    cache key.

    This is not incidental tidying: without it the round-trip assertion below
    "passed" with serial performing **zero** traversal calls, because it was
    served an earlier test's cached result. A performance test that silently
    measures a cache hit is worse than no test.

    Persistence is also disabled: the L2 layer writes into the shared
    ``project.db``, which other suites assert on (``delete_by_repo`` counts rows
    there), so leaving it on makes these tests corrupt unrelated ones.
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
    """A small deterministic call graph: 4 seeds, each with a fan-out chain.

    Shaped so multi-hop actually has somewhere to go — a flat graph would let a
    broken expansion pass by returning the seeds and nothing else.
    """
    graph = SqliteGraphStore(project_path=tmp_path / "parity.db")
    await graph.initialize()

    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    for seed in range(4):
        nodes.append(
            GraphNode(
                node_id=f"handler_{seed}",
                type="function",
                name=f"handler_{seed}",
                file=f"mod_{seed}.py",
                line=1,
                lang="python",
            )
        )
        for hop1 in range(3):
            mid = f"handler_{seed}_call_{hop1}"
            nodes.append(
                GraphNode(
                    node_id=mid,
                    type="function",
                    name=f"callee_{seed}_{hop1}",
                    file=f"mod_{seed}.py",
                    line=10 + hop1,
                    lang="python",
                )
            )
            edges.append(GraphEdge(src=f"handler_{seed}", dst=mid, type="CALLS"))
            leaf = f"{mid}_leaf"
            nodes.append(
                GraphNode(
                    node_id=leaf,
                    type="function",
                    name=f"leaf_{seed}_{hop1}",
                    file=f"mod_{seed}.py",
                    line=100 + hop1,
                    lang="python",
                )
            )
            edges.append(GraphEdge(src=mid, dst=leaf, type="CALLS"))

    await graph.upsert_nodes(nodes)
    await graph.upsert_edges(edges)
    return graph


class CountingStore:
    """Delegating wrapper that counts the calls each strategy makes.

    Deliberately a wrapper rather than a mock: the point is to measure the real
    store's call pattern, not to assert against a hand-written script of calls.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: Dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        async def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] = self.calls.get(name, 0) + 1
            return await attr(*args, **kwargs)

        return counted

    @property
    def traversal_calls(self) -> int:
        """Calls that cost a round trip to expand the frontier."""
        return self.calls.get("get_neighbors", 0) + self.calls.get("multi_hop_traverse_parallel", 0)


def _config(mode: str, **overrides: Any) -> RetrievalConfig:
    """Config for one test. `mode` participates in the cache key — give each test
    its own so a cached result from a sibling test can never be served here."""
    cfg = RetrievalConfig(seed_count=4, max_hops=2, top_k=10, mode=mode)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


async def test_parallel_retrieval_ranks_a_superset_of_the_serial_candidates(
    store: SqliteGraphStore,
) -> None:
    """Parallel must consider at least what serial considered, and rank it.

    Serial stops traversing at ``top_k``; parallel walks the neighbourhood and
    ranks it. So parallel's *candidate pool* is a superset even though the final
    ``top_k`` may differ — that difference is the improvement, and it is only
    legitimate if the seeds still survive and ranking actually ran.
    """
    retriever = MultiHopRetriever(store, _config("superset", enable_parallel=True))

    serial = await retriever.retrieve("handler")
    parallel = await retriever.retrieve_parallel("handler")

    assert parallel.nodes, "parallel retrieval returned nothing"
    assert parallel.seed_nodes == serial.seed_nodes, "the two paths disagree on seeds"

    # Ranking must have produced a score and a hop distance for everything
    # returned — the old parallel branch returned raw traversal output with
    # neither, which is what made it unusable as a drop-in.
    returned = {n.node_id for n in parallel.nodes}
    assert returned <= set(parallel.scores), "some returned nodes carry no score"
    assert returned <= set(parallel.hop_distances), "some returned nodes carry no hop distance"

    # Seeds are distance 0 and must outrank anything further away.
    for seed in parallel.seed_nodes:
        if seed in parallel.scores:
            assert parallel.hop_distances[seed] == 0
            for node_id, distance in parallel.hop_distances.items():
                if distance > 0:
                    assert parallel.scores[seed] >= parallel.scores[node_id], (
                        f"seed {seed} scored below {node_id} at distance {distance}; "
                        "score decay by distance is not being applied"
                    )


async def test_parallel_retrieval_applies_the_top_k_limit(
    store: SqliteGraphStore,
) -> None:
    """top_k is a budget, not a suggestion — the parallel branch skipped it."""
    retriever = MultiHopRetriever(store, _config("topk", top_k=3, enable_parallel=True))

    result = await retriever.retrieve_parallel("handler")

    assert (
        len(result.nodes) <= 3
    ), f"top_k=3 but parallel retrieval returned {len(result.nodes)} nodes"


async def test_parallel_retrieval_preserves_nonzero_hop_distances(
    store: SqliteGraphStore,
) -> None:
    """The final result must not flatten every traversed node back to distance zero."""
    retriever = MultiHopRetriever(store, _config("hop-distance", enable_parallel=True))

    result = await retriever.retrieve_parallel("handler")

    non_seeds = {node.node_id for node in result.nodes} - set(result.seed_nodes)
    assert non_seeds, "fixture did not return any expanded nodes"
    assert any(
        result.hop_distances[node_id] > 0 for node_id in non_seeds
    ), "parallel retrieval flattened every expanded node to hop distance zero"


async def test_parallel_retrieval_assembles_internal_edges_once(
    store: SqliteGraphStore,
) -> None:
    """Ranking the traversal must not repeat the final neighbor batch read."""
    counting = CountingStore(store)
    retriever = MultiHopRetriever(counting, _config("edge-assembly", enable_parallel=True))

    await retriever.retrieve_parallel("handler")

    assert counting.calls.get("get_neighbors_batch", 0) == 1


async def test_parallel_retrieval_expands_in_fewer_round_trips(
    store: SqliteGraphStore,
) -> None:
    """The efficiency claim, asserted rather than assumed.

    Serial expansion costs one `get_neighbors` per visited node; parallel
    expansion delegates the whole walk to the store in a single call. Without
    this assertion the production path could quietly keep using the serial
    strategy — which is exactly what happened.
    """
    serial_store = CountingStore(store)
    parallel_store = CountingStore(store)

    # Call each strategy explicitly rather than through `retrieve`, which now
    # dispatches to serial by default — routing through it would compare serial
    # with itself and pass vacuously.
    await MultiHopRetriever(serial_store, _config("roundtrip-serial"))._retrieve_serial("handler")
    # `enable_parallel` because batching is opt-in (it measured slower on sparse
    # code graphs); this test is about its call pattern, so switch it on.
    await MultiHopRetriever(
        parallel_store, _config("roundtrip-parallel", enable_parallel=True)
    ).retrieve_parallel("handler")

    assert parallel_store.traversal_calls < serial_store.traversal_calls, (
        f"parallel retrieval made {parallel_store.traversal_calls} traversal calls "
        f"vs serial's {serial_store.traversal_calls}; it is not actually batching"
    )


async def test_falls_back_to_serial_when_parallel_is_disabled(
    store: SqliteGraphStore,
) -> None:
    """An explicit opt-out must still produce correct results, not an error."""
    retriever = MultiHopRetriever(store, _config("fallback", enable_parallel=False))

    result = await retriever.retrieve_parallel("handler")

    assert result.nodes, "disabling parallel must fall back to serial, not return nothing"


async def test_parallel_fallback_reuses_the_seed_search(
    store: SqliteGraphStore,
) -> None:
    """Finding too few actual seeds must not repeat the remote seed query.

    Dispatch uses the configured ``seed_count`` while this fallback necessarily
    uses the number actually returned.  A sparse match therefore reaches the
    parallel method, discovers one seed, and falls back.  Re-running discovery
    in the serial helper doubles the dominant remote query without changing the
    result.
    """
    counting = CountingStore(store)
    retriever = MultiHopRetriever(
        counting,
        _config(
            "fallback-seeds-once",
            enable_parallel=True,
            parallel_min_batch_size=3,
        ),
    )

    result = await retriever.retrieve_parallel("handler_0_call_0_leaf")

    assert result.nodes
    assert (
        counting.calls.get("search_symbols", 0) == 1
    ), "parallel fallback repeated seed discovery before serial traversal"


async def test_edges_between_does_not_read_the_whole_graph(
    store: SqliteGraphStore,
) -> None:
    """Edge assembly must be scoped to the returned nodes.

    `_get_edges_between` used to call `get_all_edges()` and filter in Python —
    a whole-graph read per query for the few edges among `top_k` nodes. SQLite
    hid it at ~2.9 ms; the ProximaDB backend measured 732 ms for the same call,
    which made it the dominant cost of a retrieval returning a handful of nodes.
    """
    counting = CountingStore(store)
    retriever = MultiHopRetriever(counting, _config("scoped-edges"))

    result = await retriever.retrieve("handler")

    assert (
        counting.calls.get("get_all_edges", 0) == 0
    ), "retrieval read the entire edge set; it should ask only about the nodes it is returning"
    # Still correct: every edge returned must connect two returned nodes.
    returned = {n.node_id for n in result.nodes}
    for edge in result.edges:
        assert edge.src in returned and edge.dst in returned


async def test_outgoing_batches_reconstruct_every_internal_edge(
    store: SqliteGraphStore,
) -> None:
    """Outgoing reads from every returned node equal the induced edge set.

    This exhausts every subset of a seven-node component. It specifically
    guards the non-obvious completeness argument behind ``direction="out"``:
    an edge internal to the result has its source in the queried set, so it must
    appear in that source's outgoing batch even when its destination was reached
    by some unrelated path.
    """
    retriever = MultiHopRetriever(store, _config("induced-edge-property"))
    all_edges = await store.get_all_edges()
    component = sorted(
        {
            node_id
            for edge in all_edges
            for node_id in (edge.src, edge.dst)
            if node_id.startswith("handler_0")
        }
    )
    assert len(component) == 7

    for size in range(len(component) + 1):
        for chosen in combinations(component, size):
            node_ids = set(chosen)
            actual = await retriever._get_edges_between(node_ids, retriever.config)
            actual_keys = {(edge.src, edge.dst, str(edge.type)) for edge in actual}
            expected_keys = {
                (edge.src, edge.dst, str(edge.type))
                for edge in all_edges
                if edge.src in node_ids and edge.dst in node_ids
            }
            assert actual_keys == expected_keys, f"wrong induced edges for {sorted(node_ids)}"


async def test_serial_retrieval_considers_the_whole_neighbourhood(
    store: SqliteGraphStore,
) -> None:
    """Ranking must decide the result on the serial path too.

    `_retrieve_serial` ran `while queue and len(results) < top_k`, stopping
    traversal as soon as it had `top_k` nodes and only then ranking them. The
    candidate set was therefore chosen by BFS discovery order, and ranking could
    only reorder what traversal had already settled.

    The observable is the number of candidates considered, not the nodes
    returned: with score decaying by distance the nearest `top_k` win either way,
    so the returned set looks identical while the *choice* behind it is
    different. The serial BFS fetches each visited node, so counting
    `get_node_by_id` counts candidates. The fixture holds 28 nodes (4 seeds,
    12 one-hop, 12 two-hop) against a `top_k` of 10 — an implementation that
    still truncates during traversal stops at ~10.
    """
    counting = CountingStore(store)
    retriever = MultiHopRetriever(counting, _config("serial-full-rank", top_k=10))

    result = await retriever._retrieve_serial("handler")

    considered = counting.calls.get("get_node_by_id", 0)
    assert considered > 10, (
        f"serial traversal considered only {considered} candidates for top_k=10; "
        "it is still truncating during traversal, so ranking cannot change the "
        "result set"
    )
    assert len(result.nodes) <= 10, "top_k must still bound the returned set"


async def test_serial_retrieval_fails_closed_when_candidate_budget_binds(
    store: SqliteGraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A safety budget must not silently turn BFS order into ranking."""
    monkeypatch.setattr(retrieval_module, "_CANDIDATE_CEILING", 3)
    retriever = MultiHopRetriever(store, _config("serial-budget", top_k=2))

    with pytest.raises(RuntimeError, match="candidate.*budget"):
        await retriever._retrieve_serial("handler")
