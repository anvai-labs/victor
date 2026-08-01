# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""graph_semantic_search mode dispatch + hybrid RRF seed fusion.

Pins the fix for the decorative `mode` parameter: semantic/structural/hybrid
now run genuinely different seed strategies, hybrid fuses FTS5 + vector legs
via RRF, and fused seed scores carry through the hop-distance decay.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.tools.graph_query_tool import (
    _hybrid_seed_scores,
    _structural_seed_scores,
    graph_semantic_search,
)


def _node(node_id: str, name: str) -> GraphNode:
    return GraphNode(node_id=node_id, type="function", name=name, file="f.py", line=1)


def _hit(node_id: str, score: float):
    return SimpleNamespace(metadata={"node_id": node_id}, content=f"content {node_id}", score=score)


# ── seed-score helpers ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structural_seed_scores_rank_decayed():
    store = MagicMock(spec=[])
    store.search_symbols = AsyncMock(return_value=[_node("a", "alpha"), _node("b", "beta")])

    scores = await _structural_seed_scores(store, "alpha", 5)
    assert scores == {"a": 1.0, "b": 0.5}


@pytest.mark.asyncio
async def test_hybrid_fuses_keyword_and_vector_legs():
    # `spec=[]` so hasattr(store, "semantic_search") is False -> provider path.
    store = MagicMock(spec=[])
    store.search_symbols = AsyncMock(return_value=[_node("kw1", "alpha"), _node("both", "beta")])

    provider = MagicMock()
    provider.search_similar = AsyncMock(return_value=[_hit("both", 0.9), _hit("vec1", 0.7)])

    with patch("victor.tools.graph_query_tool._vector_provider_for", return_value=provider):
        scores, note = await _hybrid_seed_scores(store, "beta", 4)

    assert note is None
    # The node present in BOTH legs wins RRF fusion (normalized to 1.0).
    assert set(scores) == {"kw1", "both", "vec1"}
    assert scores["both"] == 1.0
    assert scores["kw1"] < 1.0
    assert scores["vec1"] < 1.0


@pytest.mark.asyncio
async def test_hybrid_degrades_to_structural_without_vectors():
    store = MagicMock(spec=[])
    store.search_symbols = AsyncMock(return_value=[_node("a", "alpha")])

    provider = MagicMock()
    provider.search_similar = AsyncMock(return_value=[])  # nothing embedded

    with patch("victor.tools.graph_query_tool._vector_provider_for", return_value=provider):
        scores, note = await _hybrid_seed_scores(store, "alpha", 4)

    assert scores == {"a": 1.0}
    assert note is not None
    assert "victor graph index --embeddings" in note


@pytest.mark.asyncio
async def test_hybrid_uses_store_semantic_search_when_available():
    store = MagicMock(spec=[])
    store.search_symbols = AsyncMock(return_value=[_node("kw1", "alpha")])
    store.semantic_search = AsyncMock(return_value=[_node("vec1", "gamma")])

    service = MagicMock()
    service.embed_text = AsyncMock(return_value=[0.1, 0.2])

    with patch("victor.storage.embeddings.service.get_embedding_service", return_value=service):
        scores, note = await _hybrid_seed_scores(store, "alpha", 4)

    store.semantic_search.assert_awaited_once()
    assert set(scores) == {"kw1", "vec1"}
    assert note is None


# ── expand_from_seeds: fused scores feed hop-distance decay ──────────────────


@pytest.mark.asyncio
async def test_expand_from_seeds_carries_seed_scores_through_decay():
    store = MagicMock(spec=[])
    nodes = {"seed": _node("seed", "alpha"), "nbr": _node("nbr", "beta")}
    store.get_node_by_id = AsyncMock(side_effect=lambda nid: nodes.get(nid))
    store.get_neighbors = AsyncMock(
        side_effect=lambda nid, **kw: (
            [GraphEdge(src="seed", dst="nbr", type="CALLS")] if nid == "seed" else []
        )
    )

    config = RetrievalConfig(
        seed_count=5,
        max_hops=2,
        top_k=10,
        centrality_weight=0.0,
        size_penalty_weight=0.0,
    )
    retriever = MultiHopRetriever(store, config)

    result = await retriever.expand_from_seeds({"seed": 0.5}, "alpha", config)

    # Seed: 0.5 * 1.5 (seed boost). Neighbor: 0.5 * 0.7 (1-hop decay).
    assert result.scores["seed"] == pytest.approx(0.75)
    assert result.scores["nbr"] == pytest.approx(0.35)
    assert result.hop_distances == {"seed": 0, "nbr": 1}


# ── tool-level mode dispatch ─────────────────────────────────────────────────


def _enabled_flags():
    manager = MagicMock()
    manager.is_enabled.return_value = True
    return manager


@pytest.mark.asyncio
async def test_tool_semantic_mode_uses_retrieve():
    store = MagicMock()
    store.initialize = AsyncMock()

    retriever = MagicMock()
    from victor.core.graph_rag.retrieval import RetrievalResult

    retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(nodes=[], edges=[], subgraphs=[], query="q")
    )
    retriever.expand_from_seeds = AsyncMock()

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch("victor.storage.graph.create_graph_store", return_value=store),
        patch("victor.core.graph_rag.MultiHopRetriever", return_value=retriever),
    ):
        out = await graph_semantic_search("find auth", mode="semantic")

    retriever.retrieve.assert_awaited_once()
    retriever.expand_from_seeds.assert_not_awaited()
    assert out["metadata"]["mode"] == "semantic"


@pytest.mark.asyncio
async def test_tool_structural_mode_expands_from_fts_seeds():
    store = MagicMock()
    store.initialize = AsyncMock()
    store.search_symbols = AsyncMock(return_value=[_node("a", "alpha")])

    retriever = MagicMock()
    from victor.core.graph_rag.retrieval import RetrievalResult

    retriever.retrieve = AsyncMock()
    retriever.expand_from_seeds = AsyncMock(
        return_value=RetrievalResult(nodes=[], edges=[], subgraphs=[], query="q", seed_nodes=["a"])
    )

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch("victor.storage.graph.create_graph_store", return_value=store),
        patch("victor.core.graph_rag.MultiHopRetriever", return_value=retriever),
    ):
        out = await graph_semantic_search("find auth", mode="structural")

    retriever.retrieve.assert_not_awaited()
    retriever.expand_from_seeds.assert_awaited_once()
    seed_scores, _query, _config = retriever.expand_from_seeds.await_args.args
    assert seed_scores == {"a": 1.0}
    assert out["metadata"]["mode"] == "structural"


@pytest.mark.asyncio
async def test_tool_hybrid_mode_reports_degradation_note():
    store = MagicMock(spec=[])
    store.initialize = AsyncMock()
    store.search_symbols = AsyncMock(return_value=[_node("a", "alpha")])

    retriever = MagicMock()
    from victor.core.graph_rag.retrieval import RetrievalResult

    retriever.expand_from_seeds = AsyncMock(
        return_value=RetrievalResult(nodes=[], edges=[], subgraphs=[], query="q", seed_nodes=["a"])
    )

    provider = MagicMock()
    provider.search_similar = AsyncMock(return_value=[])

    with (
        patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=_enabled_flags(),
        ),
        patch("victor.storage.graph.create_graph_store", return_value=store),
        patch("victor.core.graph_rag.MultiHopRetriever", return_value=retriever),
        patch("victor.tools.graph_query_tool._vector_provider_for", return_value=provider),
    ):
        out = await graph_semantic_search("find auth", mode="hybrid")

    assert out["metadata"]["mode"] == "hybrid"
    assert "note" in out["metadata"]


# ── cache keys must not collide across modes ─────────────────────────────────


def test_cache_key_includes_mode():
    from victor.core.graph_rag.query_cache import _create_query_cache_key

    semantic = RetrievalConfig(mode="semantic")
    hybrid = RetrievalConfig(mode="hybrid")
    key_a = _create_query_cache_key("find auth", semantic, "/repo")
    key_b = _create_query_cache_key("find auth", hybrid, "/repo")
    assert key_a != key_b
