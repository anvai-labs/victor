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

"""Graph-aware embeddings for code intelligence.

This module implements structure-aware embeddings that capture both:
- Semantic content (text embedding)
- Structural context (graph neighborhood)

Based on research from GraphCodeBERT and code graph embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphNode, GraphEdge

logger = logging.getLogger(__name__)

# Import EmbeddingService lazily to avoid circular imports
_embedding_service: Optional[Any] = None


def _get_embedding_service() -> Any:
    """Get the EmbeddingService singleton (lazy import)."""
    global _embedding_service
    if _embedding_service is None:
        try:
            from victor.storage.embeddings.service import get_embedding_service

            _embedding_service = get_embedding_service()
            logger.debug("GraphAwareEmbedder using EmbeddingService for embeddings")
        except ImportError:
            logger.warning(
                "EmbeddingService not available, using hash-based fallback. "
                "Install sentence-transformers for better embeddings."
            )
            _embedding_service = False  # Mark as unavailable
    return _embedding_service if _embedding_service is not False else None


@dataclass
class GraphEmbeddingConfig:
    """Configuration for graph-aware embeddings.

    Attributes:
        neighborhood_radius: Radius for neighborhood encoding
        include_edge_types: Whether to include edge type information
        structural_weight: Weight for structural component (0-1)
        semantic_weight: Weight for semantic component (0-1)
        max_neighbors: Maximum neighbors to include in encoding
    """

    neighborhood_radius: int = 2
    include_edge_types: bool = True
    structural_weight: float = 0.3
    semantic_weight: float = 0.7
    max_neighbors: int = 50


@dataclass
class GraphContext:
    """Structural context for a node.

    Attributes:
        node: The focal node
        neighbors: Neighbor nodes within radius
        edge_types: Edge types to neighbors
        paths: Shortest paths to neighbors
    """

    node: GraphNode
    neighbors: List[GraphNode] = field(default_factory=list)
    edge_types: Dict[str, List[str]] = field(default_factory=dict)
    paths: Dict[str, List[str]] = field(default_factory=dict)


class GraphAwareEmbedder:
    """Generate embeddings that capture graph structure.

    This class creates embeddings that combine:
    1. Text embedding of node content (via EmbeddingService)
    2. Structural encoding of neighborhood

    Attributes:
        config: Embedding configuration
        embedding_service: Optional EmbeddingService (auto-resolved if None)
    """

    def __init__(
        self,
        config: GraphEmbeddingConfig | None = None,
        embedding_service: Any = None,
    ) -> None:
        """Initialize the graph-aware embedder.

        Args:
            config: GraphEmbeddingConfig instance
            embedding_service: Optional EmbeddingService (auto-resolved from
                singleton/DI container if None). Pass explicitly for testing.
        """
        self.config = config or GraphEmbeddingConfig()
        self._explicit_service = embedding_service

    async def embed_with_context(
        self,
        node: GraphNode,
        graph_store: Any,
    ) -> List[float]:
        """Generate embedding with graph context.

        Args:
            node: Node to embed
            graph_store: Graph store for neighborhood queries

        Returns:
            Combined embedding vector
        """
        # 1. Get text embedding
        text_embedding = await self._get_text_embedding(node)

        # 2. Get structural context
        context = await self._get_graph_context(node, graph_store)

        # 3. Encode structure
        structural_embedding = self._encode_structure(context)

        # 4. Combine embeddings
        combined = self._combine_embeddings(
            text_embedding,
            structural_embedding,
        )

        return combined

    @staticmethod
    def node_text(node: GraphNode) -> str:
        """The text actually embedded for a node (name + signature + docstring).

        Also the input to the staleness fingerprint (``content_version``) used by
        the indexing pipeline — keep the two in lockstep.
        """
        return " ".join(filter(None, [node.name, node.signature or "", node.docstring or ""]))

    async def embed_batch(
        self,
        nodes: List[GraphNode],
        graph_store: Any,
    ) -> Dict[str, List[float]]:
        """Generate embeddings for multiple nodes.

        Text embeddings are generated in ONE ``EmbeddingService.embed_batch``
        call (the old per-node ``embed_text`` loop defeated the service's
        adaptive batching). Structural mixing is applied per node only when
        ``structural_weight > 0`` — leave it at 0 for vectors persisted for
        search, so stored vectors stay comparable to plain-text query vectors.

        Args:
            nodes: Nodes to embed
            graph_store: Graph store for neighborhood queries

        Returns:
            Dict mapping node IDs to embeddings
        """
        if not nodes:
            return {}

        texts = [self.node_text(n) for n in nodes]
        service = self._explicit_service or _get_embedding_service()
        text_vectors: List[List[float]]
        if service is not None:
            try:
                import numpy as np

                batch = await service.embed_batch(texts)
                text_vectors = [v.tolist() if isinstance(v, np.ndarray) else list(v) for v in batch]
            except Exception as e:
                logger.warning(f"Batch embedding failed, using hash fallback: {e}")
                text_vectors = [self._hash_encode(t) for t in texts]
        else:
            text_vectors = [self._hash_encode(t) for t in texts]

        embeddings: Dict[str, List[float]] = {}
        for node, vec in zip(nodes, text_vectors):
            try:
                if self.config.structural_weight > 0:
                    context = await self._get_graph_context(node, graph_store)
                    structural = self._encode_structure(context)
                    vec = self._combine_embeddings(vec, structural)
                embeddings[node.node_id] = vec
            except Exception as e:
                logger.warning(f"Failed to embed {node.node_id}: {e}")

        return embeddings

    async def _get_text_embedding(self, node: GraphNode) -> List[float]:
        """Get text embedding for a node.

        Args:
            node: Node to embed

        Returns:
            Text embedding vector
        """
        text = self.node_text(node)

        # Use embedding service if available
        service = self._explicit_service or _get_embedding_service()
        if service is not None:
            try:
                import numpy as np

                embedding = await service.embed_text(text, use_cache=True)
                return embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            except Exception as e:
                logger.warning(f"EmbeddingService encoding failed: {e}")

        # Fallback: simple hash-based encoding
        return self._hash_encode(text)

    def _hash_encode(self, text: str) -> List[float]:
        """Generate a simple hash-based encoding.

        Args:
            text: Text to encode

        Returns:
            Hash-based embedding vector (384 dimensions)
        """
        import hashlib

        # Generate hash values for different dimensions
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Expand to 384 dimensions
        embedding: List[float] = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            # Convert byte to float in [-1, 1]
            val = (hash_bytes[byte_idx] / 127.5) - 1.0
            embedding.append(val)

        return embedding

    async def _get_graph_context(
        self,
        node: GraphNode,
        graph_store: Any,
    ) -> GraphContext:
        """Get graph context for a node.

        Args:
            node: Focal node
            graph_store: Graph store

        Returns:
            GraphContext with neighborhood info
        """
        context = GraphContext(node=node)

        try:
            # Get neighbors within radius
            edges = await graph_store.get_neighbors(
                node.node_id,
                direction="both",
                max_depth=self.config.neighborhood_radius,
            )

            # Collect unique neighbors
            seen: set[str] = set()
            for edge in edges:
                neighbor_id = edge.dst if edge.src == node.node_id else edge.src
                if neighbor_id not in seen and neighbor_id != node.node_id:
                    seen.add(neighbor_id)

                    # Get neighbor node
                    neighbor = await graph_store.get_node_by_id(neighbor_id)
                    if neighbor:
                        context.neighbors.append(neighbor)

                        # Track edge types
                        if edge.type not in context.edge_types:
                            context.edge_types[edge.type] = []
                        context.edge_types[edge.type].append(neighbor_id)

        except Exception as e:
            logger.warning(f"Error getting graph context: {e}")

        return context

    def _encode_structure(self, context: GraphContext) -> List[float]:
        """Encode graph structure into embedding.

        Args:
            context: Graph context

        Returns:
            Structural encoding vector
        """
        # Simple encoding based on neighborhood statistics
        features = [
            len(context.neighbors),  # Neighbor count
            len(context.edge_types),  # Edge type diversity
            # Node type distribution
            sum(1 for n in context.neighbors if n.type == "function"),
            sum(1 for n in context.neighbors if n.type == "class"),
            sum(1 for n in context.neighbors if n.type == "method"),
        ]

        # Expand to 384 dimensions
        import hashlib

        feature_str = ",".join(map(str, features))
        hash_obj = hashlib.sha256(feature_str.encode())
        hash_bytes = hash_obj.digest()

        encoding: List[float] = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] / 127.5) - 1.0
            encoding.append(val)

        return encoding

    def _combine_embeddings(
        self,
        text_embedding: List[float],
        structural_embedding: List[float],
    ) -> List[float]:
        """Combine text and structural embeddings.

        Args:
            text_embedding: Text embedding vector
            structural_embedding: Structural encoding vector

        Returns:
            Combined embedding vector
        """
        if len(text_embedding) != len(structural_embedding):
            # Truncate to min length
            min_len = min(len(text_embedding), len(structural_embedding))
            text_embedding = text_embedding[:min_len]
            structural_embedding = structural_embedding[:min_len]

        # Weighted combination
        combined = []
        for t, s in zip(text_embedding, structural_embedding):
            val = t * self.config.semantic_weight + s * self.config.structural_weight
            combined.append(val)

        return combined


__all__ = [
    "GraphAwareEmbedder",
    "GraphEmbeddingConfig",
    "GraphContext",
]
