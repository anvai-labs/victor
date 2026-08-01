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

"""Graph retrieval - G-Retrieval stage of Graph RAG.

This module implements the second stage of the Graph RAG pipeline:
multi-hop graph traversal for context retrieval.

Based on GraphCodeAgent methodology:
1. Dense retrieval for seed nodes (vector search)
2. Multi-hop expansion (BFS on graph)
3. Pruning by relevance
4. Ranking with decay-with-distance
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import (
        GraphEdge,
        GraphNode,
        GraphStoreProtocol,
        Subgraph,
    )

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a graph retrieval operation.

    Attributes:
        nodes: Retrieved graph nodes
        edges: Edges between retrieved nodes
        subgraphs: Cached subgraphs used
        query: Original query
        seed_nodes: Initial seed nodes from dense retrieval
        hop_distances: Distance from query for each node
        scores: Relevance scores for each node
        execution_time_ms: Time taken for retrieval
        metadata: Additional metadata
    """

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    subgraphs: List[Subgraph]
    query: str
    seed_nodes: List[str] = field(default_factory=list)
    hop_distances: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "subgraph_count": len(self.subgraphs),
            "query": self.query,
            "seed_count": len(self.seed_nodes),
            "top_nodes": [
                {"node_id": n.node_id, "name": n.name, "file": n.file} for n in self.nodes[:5]
            ],
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class NodeWithScore:
    """A node with its associated relevance score.

    Attributes:
        node: The graph node
        score: Relevance score (0-1)
        distance: Hop distance from query
        path: Path of node IDs from seed to this node
    """

    node: GraphNode
    score: float
    distance: int
    path: List[str] = field(default_factory=list)


class MultiHopRetriever:
    """Multi-hop graph traversal for context retrieval.

    This class implements the GraphCodeAgent retrieval strategy:
    1. Find seed nodes via semantic/vector search
    2. Expand via BFS traversal on graph structure
    3. Rank by combined semantic and structural relevance

    Attributes:
        graph_store: Graph store for traversal
        config: Retrieval configuration
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        config: Any,  # RetrievalConfig
    ) -> None:
        """Initialize the multi-hop retriever.

        Args:
            graph_store: Graph store for traversal
            config: RetrievalConfig instance
        """
        self.graph_store = graph_store
        self.config = config

    async def retrieve(
        self,
        query: str,
        config: Any | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant context via multi-hop graph traversal.

        Args:
            query: Natural language query
            config: Optional config override

        Returns:
            RetrievalResult with retrieved nodes and edges
        """
        start_time = time.time()
        cfg = config or self.config

        logger.debug(f"Multi-hop retrieval for query: {query}")

        # Try cache first (PH4-005: Graph query cache)
        cache_result = await self._try_cache(query, cfg)
        if cache_result is not None:
            return cache_result

        # Stage 1: Find seed nodes via dense retrieval
        seed_nodes = await self._find_seed_nodes(query, cfg)
        seed_ids = [n.node_id for n in seed_nodes]

        logger.debug(f"Found {len(seed_ids)} seed nodes")

        # Stage 2: Multi-hop expansion
        expanded_nodes = await self._multi_hop_expand(seed_ids, cfg)

        logger.debug(f"Expanded to {len(expanded_nodes)} nodes")

        # Stage 3: Rank and prune
        ranked_nodes = self._rank_nodes(
            expanded_nodes,
            seed_ids,
            query,
            cfg,
        )

        # Apply top-k limit
        final_nodes = ranked_nodes[: cfg.top_k]

        # Get edges between retrieved nodes
        node_ids = {n.node.node_id for n in final_nodes}
        edges = await self._get_edges_between(node_ids, cfg)

        # Build result
        result = RetrievalResult(
            nodes=[n.node for n in final_nodes],
            edges=edges,
            subgraphs=[],
            query=query,
            seed_nodes=seed_ids,
            hop_distances={n.node.node_id: n.distance for n in final_nodes},
            scores={n.node.node_id: n.score for n in final_nodes},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

        logger.debug(
            f"Retrieval complete: {len(result.nodes)} nodes, "
            f"{len(result.edges)} edges in {result.execution_time_ms:.1f}ms"
        )

        # Cache the result (PH4-005: Graph query cache)
        await self._save_to_cache(query, cfg, result)

        return result

    async def expand_from_seeds(
        self,
        seed_scores: Dict[str, float],
        query: str,
        config: Any | None = None,
    ) -> RetrievalResult:
        """Multi-hop expansion from externally ranked seeds.

        Used by hybrid/structural search modes that fuse their own seed set
        (e.g. FTS5 + vector RRF fusion in graph_semantic_search): each seed's
        score feeds the hop-distance decay so the fused ranking carries into
        the expansion instead of being flattened to a uniform 1.0.

        Args:
            seed_scores: node_id -> initial score (ideally normalized to [0,1]
                so centrality/size adjustments stay proportionate)
            query: Original query (for ranking context / result metadata)
            config: Optional config override

        Returns:
            RetrievalResult with expanded, ranked nodes and their edges
        """
        start_time = time.time()
        cfg = config or self.config
        seed_ids = list(seed_scores)

        expanded_nodes = await self._multi_hop_expand(seed_ids, cfg, seed_scores=seed_scores)
        ranked_nodes = self._rank_nodes(expanded_nodes, seed_ids, query, cfg)
        final_nodes = ranked_nodes[: cfg.top_k]

        node_ids = {n.node.node_id for n in final_nodes}
        edges = await self._get_edges_between(node_ids, cfg)

        return RetrievalResult(
            nodes=[n.node for n in final_nodes],
            edges=edges,
            subgraphs=[],
            query=query,
            seed_nodes=seed_ids,
            hop_distances={n.node.node_id: n.distance for n in final_nodes},
            scores={n.node.node_id: n.score for n in final_nodes},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _find_seed_nodes(
        self,
        query: str,
        config: Any,
    ) -> List[GraphNode]:
        """Find seed nodes via dense/semantic retrieval.

        Args:
            query: Natural language query
            config: Retrieval configuration

        Returns:
            List of seed nodes
        """
        # Try semantic search via graph store
        try:
            nodes = await self.graph_store.search_symbols(
                query,
                limit=config.seed_count,
            )
            if nodes:
                return nodes[: config.seed_count]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # Fallback: use keyword matching in find_nodes
        keywords = self._extract_keywords(query)
        all_nodes: List[GraphNode] = []

        for keyword in keywords[:5]:
            try:
                found = await self.graph_store.find_nodes(name=keyword)
                all_nodes.extend(found)
            except Exception:
                pass

        # Deduplicate and limit
        seen: Set[str] = set()
        unique_nodes = []
        for node in all_nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_nodes.append(node)

        return unique_nodes[: config.seed_count]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from a query.

        Args:
            query: Query string

        Returns:
            List of keywords
        """
        import re

        # Extract alphanumeric sequences
        words = re.findall(r"\b\w+\b", query.lower())

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "how",
            "what",
            "when",
            "where",
            "why",
            "which",
            "who",
            "does",
            "do",
            "can",
            "could",
            "should",
            "would",
            "get",
            "find",
            "show",
        }

        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords

    async def _multi_hop_expand(
        self,
        seed_ids: List[str],
        config: Any,
        seed_scores: Optional[Dict[str, float]] = None,
    ) -> List[NodeWithScore]:
        """Expand from seed nodes via multi-hop BFS traversal.

        Args:
            seed_ids: Seed node IDs
            config: Retrieval configuration
            seed_scores: Optional per-seed initial scores (e.g. RRF-fused hybrid
                scores); a node's score decays from ITS seed's score, not a
                uniform 1.0, so fused ranking carries through the expansion.

        Returns:
            List of nodes with scores and distances
        """
        results: Dict[str, NodeWithScore] = {}
        visited: Set[str] = set()
        queue: deque[Tuple[str, int, List[str]]] = deque()

        # Initialize queue with seed nodes
        for seed_id in seed_ids:
            queue.append((seed_id, 0, [seed_id]))
            visited.add(seed_id)

        # Check if lazy loading is enabled (PH4-006)
        use_lazy_loading = self._use_lazy_loading(config)

        # BFS traversal
        while queue and len(results) < config.top_k:
            node_id, distance, path = queue.popleft()

            # Get node
            try:
                node = await self.graph_store.get_node_by_id(node_id)
                if node is None:
                    continue
            except Exception:
                continue

            # Calculate initial score (decay with distance from the seed's score)
            base = seed_scores.get(path[0], 1.0) if seed_scores else 1.0
            score = self._decay_score(base, distance, config.max_hops)

            # Store result
            results[node_id] = NodeWithScore(
                node=node,
                score=score,
                distance=distance,
                path=path,
            )

            # Expand neighbors if within max hops
            if distance < config.max_hops:
                try:
                    edge_types = config.edge_types

                    # Use lazy loading if enabled and available (PH4-006)
                    if use_lazy_loading and hasattr(self.graph_store, "iter_neighbors"):
                        neighbors = await self._get_neighbors_lazy(node_id, edge_types, config)
                    else:
                        neighbors = await self.graph_store.get_neighbors(
                            node_id,
                            edge_types=edge_types,
                            direction="out",
                            max_depth=1,
                        )

                    for edge in neighbors:
                        neighbor_id = edge.dst
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            new_path = path + [neighbor_id]
                            queue.append((neighbor_id, distance + 1, new_path))

                except Exception as e:
                    logger.warning(f"Error getting neighbors for {node_id}: {e}")

        return list(results.values())

    def _use_lazy_loading(self, config: Any) -> bool:
        """Check if lazy loading should be used for retrieval.

        Args:
            config: Retrieval configuration

        Returns:
            True if lazy loading is enabled and available
        """
        # Check if lazy loading is enabled in config
        if hasattr(config, "use_lazy_loading"):
            return config.use_lazy_loading

        # Check if we should use lazy loading based on result size
        # For large top_k, use lazy loading to reduce memory
        if config.top_k > 100:
            return True

        return False

    async def _get_neighbors_lazy(
        self,
        node_id: str,
        edge_types: Any,
        config: Any,
    ) -> List[GraphEdge]:
        """Get neighbors using lazy iterator (PH4-006).

        Args:
            node_id: Node ID to get neighbors for
            edge_types: Edge type filter
            config: Retrieval configuration

        Returns:
            List of neighbor edges (materialized from iterator)
        """
        edges: List[GraphEdge] = []
        batch_size = getattr(config, "lazy_load_batch_size", 50)

        try:
            # Iterate over neighbors in batches
            iter_neighbors = self.graph_store.iter_neighbors(
                node_id,
                batch_size=batch_size,
                edge_types=edge_types if edge_types else None,
                direction="out",
            )

            async for batch in iter_neighbors:
                edges.extend(batch)

                # Early termination if we have enough neighbors
                max_neighbors = getattr(config, "max_neighbors_per_node", 100)
                if len(edges) >= max_neighbors:
                    edges = edges[:max_neighbors]
                    break

        except Exception as e:
            logger.warning(f"Lazy neighbor iteration failed for {node_id}: {e}")
            # Fall back to regular get_neighbors
            edges = await self.graph_store.get_neighbors(
                node_id,
                edge_types=edge_types,
                direction="out",
                max_depth=1,
            )

        return edges

    def _decay_score(
        self,
        initial_score: float,
        distance: int,
        max_hops: int,
    ) -> float:
        """Apply distance decay to a score.

        Args:
            initial_score: Initial score
            distance: Hop distance
            max_hops: Maximum hops

        Returns:
            Decayed score
        """
        if distance == 0:
            return initial_score

        # Exponential decay
        decay_factor = 0.7**distance
        return initial_score * decay_factor

    def _rank_nodes(
        self,
        nodes: List[NodeWithScore],
        seed_ids: List[str],
        query: str,
        config: Any,
    ) -> List[NodeWithScore]:
        """Rank nodes by combined relevance.

        Args:
            nodes: Nodes with initial scores
            seed_ids: Seed node IDs
            query: Original query
            config: Retrieval configuration

        Returns:
            Sorted list of nodes
        """
        # Calculate combined scores
        for nws in nodes:
            # Base score from distance decay
            combined = nws.score

            # Boost seed nodes
            if nws.node.node_id in seed_ids:
                combined *= 1.5

            # Apply centrality boost if configured
            if config.centrality_weight > 0:
                centrality = self._estimate_centrality(nws.node)
                combined = (
                    combined * (1 - config.centrality_weight)
                    + centrality * config.centrality_weight
                )

            # Apply size penalty for large contexts
            if config.size_penalty_weight > 0:
                size_penalty = self._calculate_size_penalty(nws.node)
                combined *= 1 - size_penalty * config.size_penalty_weight

            nws.score = combined

        # Sort by score descending
        return sorted(nodes, key=lambda n: n.score, reverse=True)

    def _estimate_centrality(self, node: GraphNode) -> float:
        """Estimate node centrality (simplified).

        Args:
            node: Graph node

        Returns:
            Estimated centrality (0-1)
        """
        # Simple heuristic: nodes with more connections are more central
        # This is a simplified version; real implementation would use
        # PageRank or betweenness centrality
        base_score = 0.5

        # Boost public symbols
        if node.visibility == "public":
            base_score += 0.2

        # Boost definitions over statements
        if node.type in {"function", "class", "method"}:
            base_score += 0.2

        return min(base_score, 1.0)

    def _calculate_size_penalty(self, node: GraphNode) -> float:
        """Calculate penalty for large nodes (long functions, etc.).

        Args:
            node: Graph node

        Returns:
            Size penalty (0-1)
        """
        if node.line is None or node.end_line is None:
            return 0.0

        size = node.end_line - node.line + 1

        # Penalty increases with size, capped at 0.5
        return min(size / 200, 0.5)

    async def _get_edges_between(
        self,
        node_ids: Set[str],
        config: Any,
    ) -> List[GraphEdge]:
        """Get edges between the retrieved nodes.

        Args:
            node_ids: Set of node IDs
            config: Retrieval configuration

        Returns:
            List of edges
        """
        edges: List[GraphEdge] = []

        # Get all edges and filter
        try:
            all_edges = await self.graph_store.get_all_edges()

            for edge in all_edges:
                if edge.src in node_ids and edge.dst in node_ids:
                    # Filter by edge type if specified
                    if config.edge_types is None or edge.type in config.edge_types:
                        edges.append(edge)
        except Exception as e:
            logger.warning(f"Error getting edges: {e}")

        return edges

    def _cache_repo_path(self) -> Optional[str]:
        """Resolve the repo path used to scope cache entries.

        Prefers the public ``repo_root`` property exposed by graph stores;
        falls back to the legacy ``_root_path`` attribute for third-party stores.
        """
        root = getattr(self.graph_store, "repo_root", None)
        if root is None:
            root = getattr(self.graph_store, "_root_path", None)
        return str(root) if root is not None else None

    async def _try_cache(
        self,
        query: str,
        config: Any,
    ) -> Optional[RetrievalResult]:
        """Try to get result from cache.

        Args:
            query: Query string
            config: Retrieval configuration

        Returns:
            Cached RetrievalResult or None
        """
        try:
            from victor.core.graph_rag.query_cache import get_graph_query_cache

            cache = get_graph_query_cache()
            return cache.get(query, config, self._cache_repo_path())
        except ImportError:
            # Cache module not available
            return None
        except Exception as e:
            logger.warning(f"Error accessing graph query cache: {e}")
            return None

    async def _save_to_cache(
        self,
        query: str,
        config: Any,
        result: RetrievalResult,
    ) -> None:
        """Save result to cache.

        Args:
            query: Query string
            config: Retrieval configuration
            result: RetrievalResult to cache
        """
        try:
            from victor.core.graph_rag.query_cache import get_graph_query_cache

            cache = get_graph_query_cache()
            cache.put(query, config, result, self._cache_repo_path())
        except ImportError:
            # Cache module not available
            pass
        except Exception as e:
            logger.warning(f"Error saving to graph query cache: {e}")

    async def retrieve_parallel(
        self,
        query: str,
        config: Any | None = None,
    ) -> RetrievalResult:
        """Retrieve using parallel multi-hop traversal (PH4-007).

        Uses parallel graph traversal for faster multi-hop expansion
        when there are multiple seed nodes.

        Args:
            query: Natural language query
            config: Optional config override

        Returns:
            RetrievalResult with retrieved nodes and edges
        """
        start_time = time.time()
        cfg = config or self.config

        logger.debug(f"Parallel multi-hop retrieval for query: {query}")

        # Check if parallel traversal is available and beneficial
        use_parallel = self._should_use_parallel(cfg)

        if not use_parallel:
            # Fall back to regular retrieval
            return await self.retrieve(query, cfg)

        # Stage 1: Find seed nodes
        seed_nodes = await self._find_seed_nodes(query, cfg)
        seed_ids = [n.node_id for n in seed_nodes]

        if len(seed_ids) < getattr(cfg, "parallel_min_batch_size", 3):
            # Not enough seeds to benefit from parallel
            return await self.retrieve(query, cfg)

        logger.debug(f"Using parallel traversal with {len(seed_ids)} seeds")

        # Stage 2: Parallel multi-hop expansion
        max_workers = getattr(cfg, "max_workers", 4)
        edge_types = getattr(cfg, "edge_types", None)

        # Use graph store's parallel traversal if available
        if hasattr(self.graph_store, "multi_hop_traverse_parallel"):
            from victor.storage.graph.protocol import GraphQueryResult

            graph_result = await self.graph_store.multi_hop_traverse_parallel(
                start_node_ids=seed_ids,
                max_hops=cfg.max_hops,
                edge_types=edge_types,
                max_nodes=cfg.max_nodes,
                max_workers=max_workers,
            )

            # Convert to RetrievalResult
            result = RetrievalResult(
                nodes=graph_result.nodes,
                edges=graph_result.edges,
                subgraphs=[],
                query=query,
                seed_nodes=seed_ids,
                execution_time_ms=graph_result.execution_time_ms,
                metadata=graph_result.metadata,
            )
        else:
            # Fall back to regular expansion
            expanded_nodes = await self._multi_hop_expand(seed_ids, cfg)

            # Build result
            node_ids = {n.node.node_id for n in expanded_nodes}
            edges = await self._get_edges_between(node_ids, cfg)

            result = RetrievalResult(
                nodes=[n.node for n in expanded_nodes],
                edges=edges,
                subgraphs=[],
                query=query,
                seed_nodes=seed_ids,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Stage 3: Rank and prune
        ranked_nodes = self._rank_nodes(
            [NodeWithScore(n, result.scores.get(n.node_id, 0.5), 0, []) for n in result.nodes],
            seed_ids,
            query,
            cfg,
        )

        # Apply top-k limit
        final_nodes = ranked_nodes[: cfg.top_k]

        # Get edges between retrieved nodes
        node_ids = {n.node.node_id for n in final_nodes}
        edges = await self._get_edges_between(node_ids, cfg)

        # Build final result
        result = RetrievalResult(
            nodes=[n.node for n in final_nodes],
            edges=edges,
            subgraphs=[],
            query=query,
            seed_nodes=seed_ids,
            hop_distances={n.node.node_id: n.distance for n in final_nodes},
            scores={n.node.node_id: n.score for n in final_nodes},
            execution_time_ms=(time.time() - start_time) * 1000,
        )

        # Cache the result
        await self._save_to_cache(query, cfg, result)

        return result

    def _should_use_parallel(self, config: Any) -> bool:
        """Check if parallel traversal should be used.

        Args:
            config: Retrieval configuration

        Returns:
            True if parallel traversal is beneficial
        """
        # Check if explicitly disabled
        if hasattr(config, "enable_parallel"):
            if not config.enable_parallel:
                return False

        # Check if we have enough seeds to benefit
        seed_count = getattr(config, "seed_count", 5)
        min_batch = getattr(config, "parallel_min_batch_size", 3)

        return seed_count >= min_batch


__all__ = ["MultiHopRetriever", "RetrievalResult", "NodeWithScore"]
