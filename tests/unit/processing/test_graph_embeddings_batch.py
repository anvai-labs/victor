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

"""GraphAwareEmbedder batching behavior.

Pins the fix for the per-node embed_text loop: a batch of nodes must produce
exactly ONE EmbeddingService.embed_batch call.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.processing.graph_embeddings import GraphAwareEmbedder, GraphEmbeddingConfig
from victor.storage.graph.protocol import GraphNode


def _node(i: int) -> GraphNode:
    return GraphNode(
        node_id=f"n{i}",
        type="function",
        name=f"func_{i}",
        file="f.py",
        line=i * 10,
        signature=f"func_{i}(x)",
        docstring=f"does thing {i}",
    )


def _service(dim: int = 4):
    import numpy as np

    service = MagicMock()
    service.embed_batch = AsyncMock(
        side_effect=lambda texts: np.ones((len(texts), dim), dtype=np.float32)
    )
    service.embed_text = AsyncMock(return_value=np.ones(dim, dtype=np.float32))
    return service


@pytest.mark.asyncio
async def test_embed_batch_makes_single_service_call():
    service = _service()
    embedder = GraphAwareEmbedder(
        config=GraphEmbeddingConfig(structural_weight=0.0),
        embedding_service=service,
    )
    nodes = [_node(i) for i in range(7)]

    embeddings = await embedder.embed_batch(nodes, graph_store=MagicMock())

    service.embed_batch.assert_awaited_once()
    (texts,) = service.embed_batch.await_args.args
    assert len(texts) == 7
    assert set(embeddings) == {f"n{i}" for i in range(7)}
    # Per-node embed_text must not be used on the batch path.
    service.embed_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_batch_zero_structural_weight_skips_graph_context():
    service = _service()
    embedder = GraphAwareEmbedder(
        config=GraphEmbeddingConfig(structural_weight=0.0),
        embedding_service=service,
    )
    graph_store = MagicMock()
    graph_store.get_neighbors = AsyncMock(return_value=[])

    await embedder.embed_batch([_node(1)], graph_store)

    graph_store.get_neighbors.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_batch_structural_weight_uses_graph_context():
    service = _service()
    embedder = GraphAwareEmbedder(
        config=GraphEmbeddingConfig(structural_weight=0.3, semantic_weight=0.7),
        embedding_service=service,
    )
    graph_store = MagicMock()
    graph_store.get_neighbors = AsyncMock(return_value=[])

    result = await embedder.embed_batch([_node(1)], graph_store)

    graph_store.get_neighbors.assert_awaited()
    assert "n1" in result


@pytest.mark.asyncio
async def test_embed_batch_empty_nodes():
    embedder = GraphAwareEmbedder(embedding_service=_service())
    assert await embedder.embed_batch([], MagicMock()) == {}


def test_node_text_is_name_signature_docstring():
    node = _node(3)
    text = GraphAwareEmbedder.node_text(node)
    assert "func_3" in text
    assert "func_3(x)" in text
    assert "does thing 3" in text
