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

"""Real semantic similarity in the requirement graph (closes the PH5-004 TODO).

Pins: embedding cosine replaces the textual fallback in _semantic_similarity,
FTS symbol candidates are re-scored (not flat 0.8), and every degraded-service
shape (absent, raising, zero-vector) falls back instead of zeroing mappings.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.core.graph_rag import requirement_graph as rg
from victor.core.graph_rag.requirement_graph import (
    RequirementGraphBuilder,
    RequirementSimilarityCalculator,
)
from victor.storage.graph.protocol import GraphNode
from victor.storage.embeddings.service import EmbeddingService


def _req(node_id: str, name: str, description: str = "") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="requirement",
        name=name,
        file="requirements.md",
        metadata={"description": description} if description else {},
    )


def _sym(node_id: str, name: str, docstring: str = "") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="function",
        name=name,
        file="f.py",
        line=1,
        docstring=docstring,
    )


def _service(vectors: dict[str, list[float]]):
    """Mock service returning fixed vectors per text (prefix-matched)."""

    def _lookup(text: str) -> np.ndarray:
        for key, vec in vectors.items():
            if key in text:
                return np.array(vec, dtype=np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    service = MagicMock()
    service.embed_text = AsyncMock(side_effect=lambda text, use_cache=True: _lookup(text))
    service.embed_batch = AsyncMock(side_effect=lambda texts: np.stack([_lookup(t) for t in texts]))
    service.cosine_similarity = EmbeddingService.cosine_similarity
    return service


# ── _semantic_similarity ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_semantic_similarity_uses_embedding_cosine(monkeypatch):
    service = _service({"login": [1.0, 0.0, 0.0], "authentication": [0.8, 0.6, 0.0]})
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    calc = RequirementSimilarityCalculator(graph_store=MagicMock())
    score = await calc._semantic_similarity(
        _req("r1", "user login"), _req("r2", "authentication flow")
    )
    assert score == pytest.approx(0.8, abs=1e-5)  # cos([1,0,0],[0.8,0.6,0]) = 0.8


@pytest.mark.asyncio
async def test_semantic_similarity_clamps_negative_cosine(monkeypatch):
    service = _service({"aaa": [1.0, 0.0, 0.0], "bbb": [-1.0, 0.0, 0.0]})
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    calc = RequirementSimilarityCalculator(graph_store=MagicMock())
    score = await calc._semantic_similarity(_req("r1", "aaa"), _req("r2", "bbb"))
    assert score == 0.0


@pytest.mark.asyncio
async def test_semantic_similarity_falls_back_without_service(monkeypatch):
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: None)

    calc = RequirementSimilarityCalculator(graph_store=MagicMock())
    r1 = _req("r1", "validate user input")
    r2 = _req("r2", "validate user input")
    score = await calc._semantic_similarity(r1, r2)
    assert score == calc._textual_similarity(r1, r2)  # identical text -> 1.0
    assert score == 1.0


@pytest.mark.asyncio
async def test_semantic_similarity_zero_vectors_fall_back_to_textual(monkeypatch):
    """A degraded service (model missing emits zero vectors) must not zero scores."""
    service = _service({"": [0.0, 0.0, 0.0]})
    service.embed_text = AsyncMock(return_value=np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    calc = RequirementSimilarityCalculator(graph_store=MagicMock())
    r1 = _req("r1", "validate user input")
    r2 = _req("r2", "validate user input")
    assert await calc._semantic_similarity(r1, r2) == 1.0  # textual fallback


@pytest.mark.asyncio
async def test_find_similar_requirements_semantic_gated_by_default_on(monkeypatch):
    service = _service({"login": [1.0, 0.0, 0.0], "authentication": [0.9, 0.436, 0.0]})
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    source = _req("r1", "login")
    other = _req("r2", "authentication")
    store = MagicMock()
    store.get_node_by_id = AsyncMock(return_value=source)

    calc = RequirementSimilarityCalculator(graph_store=store)  # use_embeddings default
    monkeypatch.setattr(calc, "_get_all_requirements", AsyncMock(return_value=[source, other]))

    results = await calc.find_similar_requirements("r1", threshold=0.5, similarity_type="semantic")
    assert len(results) == 1
    assert results[0].similar_requirement_id == "r2"
    assert results[0].similarity_type == "semantic"
    assert results[0].similarity_score == pytest.approx(0.9, abs=1e-3)


# ── _find_similar_symbols re-scoring ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_symbols_rescored_and_reranked_by_cosine(monkeypatch):
    service = _service(
        {
            "authenticate requirement": [1.0, 0.0, 0.0],
            "verify_password": [0.9, 0.436, 0.0],  # cos ~0.9
            "render_chart": [0.0, 1.0, 0.0],  # cos 0.0
        }
    )
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    store = MagicMock()
    # FTS returns the WORSE candidate first — semantic re-ranking must flip it.
    store.search_symbols = AsyncMock(
        return_value=[_sym("s2", "render_chart"), _sym("s1", "verify_password")]
    )

    builder = RequirementGraphBuilder(graph_store=store)
    symbols = await builder._find_similar_symbols("r1", "authenticate requirement", max_symbols=2)

    assert [s.symbol_id for s in symbols] == ["s1", "s2"]
    assert symbols[0].confidence == pytest.approx(0.9, abs=1e-3)
    assert symbols[1].confidence == 0.0


@pytest.mark.asyncio
async def test_symbols_cut_to_max_after_rerank(monkeypatch):
    service = _service({"authenticate": [1.0, 0.0, 0.0]})
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    store = MagicMock()
    store.search_symbols = AsyncMock(return_value=[_sym(f"s{i}", f"fn_{i}") for i in range(6)])

    builder = RequirementGraphBuilder(graph_store=store)
    symbols = await builder._find_similar_symbols("authenticate", "authenticate", max_symbols=3)

    assert len(symbols) == 3
    # Over-fetch happened so re-ranking had extra candidates.
    assert store.search_symbols.await_args.kwargs.get("limit") == 6


@pytest.mark.asyncio
async def test_symbols_flat_default_without_service(monkeypatch):
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: None)

    store = MagicMock()
    store.search_symbols = AsyncMock(return_value=[_sym("s1", "fn")])

    builder = RequirementGraphBuilder(graph_store=store)
    symbols = await builder._find_similar_symbols("r1", "anything", max_symbols=5)
    assert [s.confidence for s in symbols] == [0.8]


@pytest.mark.asyncio
async def test_symbols_zero_vector_service_keeps_default_confidence(monkeypatch):
    service = MagicMock()
    service.embed_text = AsyncMock(return_value=np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(rg, "_get_embedding_service", lambda: service)

    store = MagicMock()
    store.search_symbols = AsyncMock(return_value=[_sym("s1", "fn")])

    builder = RequirementGraphBuilder(graph_store=store)
    symbols = await builder._find_similar_symbols("r1", "anything", max_symbols=5)
    # Zero vectors would zero every confidence and drop all SATISFIES edges.
    assert [s.confidence for s in symbols] == [0.8]
