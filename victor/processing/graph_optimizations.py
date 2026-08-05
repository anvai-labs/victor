# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Optimization utilities for graph operations (PH4-008).

This module provides optimization strategies and utilities for graph operations:
- Batch operation optimization
- Query plan optimization
- Index usage optimization
- Caching strategies
- Connection pooling

Usage:
    from victor.processing.graph_optimizations import (
        optimize_batch_size,
        suggest_index_strategy,
        GraphOptimizationHints,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeRecommendation:
    """Recommendation for batch size of a specific operation.

    Attributes:
        operation: Name of the operation (e.g., "indexing", "retrieval")
        recommended_size: Suggested batch size
        reason: Explanation for the recommendation
    """

    operation: str
    recommended_size: int
    reason: str = ""


@dataclass
class GraphOptimizationHints:
    """Hints for optimizing graph operations.

    Attributes:
        batch_size_hint: Suggested batch size for operations
        use_parallel: Whether parallelization is beneficial
        cache_key_hint: Hint for cache key generation
        preferred_traversal: Preferred traversal strategy
        skip_optimization: Whether to skip optimization for this operation
        batch_size_recommendations: Per-operation batch size recommendations
    """

    batch_size_hint: Optional[int] = None
    use_parallel: bool = False
    cache_key_hint: Optional[str] = None
    preferred_traversal: str = "sequential"  # sequential, parallel, hybrid
    skip_optimization: bool = False
    batch_size_recommendations: List[BatchSizeRecommendation] = field(default_factory=list)


@dataclass
class OperationProfile:
    """Profile data for an operation type.

    Attributes:
        avg_time_ms: Average execution time
        call_count: Number of times called
        node_count: Average number of nodes processed
        edge_count: Average number of edges processed
        last_optimized: Timestamp of last optimization
    """

    avg_time_ms: float = 0.0
    call_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    last_optimized: Optional[float] = None


class GraphOptimizer:
    """Optimizer for graph operations based on profiling data (PH4-008).

        Analyzes operation profiles and suggests optimizations:
    - Optimal batch sizes for bulk operations
    - When to use parallel vs sequential processing
    - Cache hit/miss patterns
    - Index usage recommendations
    """

    def __init__(self) -> None:
        """Initialize the optimizer."""
        self._profiles: Dict[str, OperationProfile] = {}
        self._optimization_history: List[Dict[str, Any]] = []

    def analyze_operation(
        self,
        operation_name: str,
        profile_data: Dict[str, Any],
    ) -> GraphOptimizationHints:
        """Analyze operation profile and generate optimization hints.

        Args:
            operation_name: Name of the operation
            profile_data: Profile data from profiler

        Returns:
            Optimization hints for the operation
        """
        hints = GraphOptimizationHints()

        avg_time = profile_data.get("avg_time_ms", 0)
        call_count = profile_data.get("call_count", 0)
        node_count = profile_data.get("node_count", 0)

        # Skip optimization for rarely called operations
        if call_count < 5:
            hints.skip_optimization = True
            return hints

        # Parallelization decision based on operation characteristics
        if avg_time > 50 and node_count > 10:
            hints.use_parallel = True
            hints.batch_size_hint = min(50, max(4, node_count // 4))

        # Cache decision
        if call_count > 10 and avg_time < 10:
            hints.cache_key_hint = operation_name

        # Batch size optimization
        if node_count > 100:
            hints.batch_size_hint = self._calculate_optimal_batch_size(avg_time, node_count)

        return hints

    def _calculate_optimal_batch_size(
        self,
        avg_time_ms: float,
        node_count: int,
    ) -> int:
        """Calculate optimal batch size based on operation characteristics.

        Args:
            avg_time_ms: Average execution time
            node_count: Number of nodes to process

        Returns:
            Optimal batch size
        """
        # For fast operations (<1ms), use larger batches
        if avg_time_ms < 1:
            return min(500, max(50, node_count))

        # For slow operations (>100ms), use smaller batches
        if avg_time_ms > 100:
            return max(10, 50)

        # Medium operations - moderate batch size
        return 100

    def record_optimization(
        self,
        operation_name: str,
        optimization_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Record an optimization that was applied.

        Args:
            operation_name: Name of the operation
            optimization_type: Type of optimization applied
            details: Optimization details
        """
        import time

        self._optimization_history.append(
            {
                "operation": operation_name,
                "optimization": optimization_type,
                "details": details,
                "timestamp": time.time(),
            }
        )

        # Update profile last optimized time
        if operation_name in self._profiles:
            self._profiles[operation_name].last_optimized = time.time()

    async def analyze_graph(
        self,
        graph_store: Any = None,
        profile_data: Optional[Dict[str, Any]] = None,
    ) -> GraphOptimizationHints:
        """Analyze graph store and profile data to produce optimization hints.

        Args:
            graph_store: Optional graph store instance for live statistics
            profile_data: Optional dict with keys such as nodes, edges, etc.

        Returns:
            GraphOptimizationHints with recommendations
        """
        if profile_data is None:
            profile_data = {}

        hints = GraphOptimizationHints()
        node_count = profile_data.get("nodes", 0)
        edge_count = profile_data.get("edges", 0)

        # Build per-operation batch size recommendations
        recommendations: List[BatchSizeRecommendation] = []

        # Indexing recommendation
        indexing_batch = self._calculate_optimal_batch_size(10.0, max(1, node_count))
        recommendations.append(
            BatchSizeRecommendation(
                operation="indexing",
                recommended_size=indexing_batch,
                reason=f"Based on {node_count} nodes",
            )
        )

        # Retrieval recommendation
        retrieval_batch = self._calculate_optimal_batch_size(5.0, max(1, edge_count))
        recommendations.append(
            BatchSizeRecommendation(
                operation="retrieval",
                recommended_size=retrieval_batch,
                reason=f"Based on {edge_count} edges",
            )
        )

        hints.batch_size_recommendations = recommendations
        hints.batch_size_hint = indexing_batch

        # Parallelization heuristic
        if node_count > 50 or edge_count > 200:
            hints.use_parallel = True
            hints.preferred_traversal = "parallel"

        return hints

    def get_optimization_history(
        self,
        operation_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get optimization history.

        Args:
            operation_name: Optional operation name filter
            limit: Maximum number of records to return

        Returns:
            List of optimization records
        """
        history = self._optimization_history

        if operation_name:
            history = [h for h in history if h.get("operation") == operation_name]

        return history[-limit:]

    def suggest_index_strategy(
        self,
        graph_stats: Dict[str, Any],
    ) -> List[str]:
        """Suggest indexing strategy based on graph statistics.

        Args:
            graph_stats: Graph statistics from stats() call

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        node_count = graph_stats.get("nodes", 0)
        edge_count = graph_stats.get("edges", 0)

        if node_count > 10000:
            recommendations.append(
                "Large graph detected (>10K nodes) - "
                "consider partitioning or subgraph extraction"
            )

        if edge_count > node_count * 10:
            recommendations.append(
                f"High edge-to-node ratio ({edge_count/node_count:.1f}:1) - "
                "consider edge type filtering or pruning"
            )

        if edge_count > 50000:
            recommendations.append(
                "Very large edge count - consider compressed storage " "or edge type aggregation"
            )

        return recommendations


def optimize_batch_size(
    operation_type: str,
    current_batch_size: int,
    avg_time_ms: float,
    node_count: int,
) -> int:
    """Dynamically optimize batch size based on performance.

    Args:
        operation_type: Type of operation (e.g., "get_neighbors", "multi_hop")
        current_batch_size: Current batch size being used
        avg_time_ms: Average execution time
        node_count: Number of nodes being processed

    Returns:
        Optimized batch size
    """
    # For very fast operations, increase batch size
    if avg_time_ms < 1:
        return min(current_batch_size * 2, 500)

    # For slow operations, decrease batch size
    if avg_time_ms > 100:
        return max(current_batch_size // 2, 10)

    # For I/O-bound operations, moderate batch size
    if 1 <= avg_time_ms <= 50:
        return current_batch_size

    return current_batch_size


def suggest_query_plan(
    query_type: str,
    graph_stats: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggest optimal query plan based on graph characteristics.

    Args:
        query_type: Type of query (traversal, search, aggregation)
        graph_stats: Graph statistics
        config: Query configuration

    Returns:
        Query plan with optimization hints
    """
    plan = {
        "strategy": "sequential",
        "use_cache": True,
        "use_lazy_loading": False,
        "use_parallel": False,
        "estimated_nodes": min(config.get("max_nodes", 100), graph_stats.get("nodes", 0)),
    }

    node_count = graph_stats.get("nodes", 0)
    max_nodes = config.get("max_nodes", 100)

    # For large graphs, use lazy loading
    if node_count > 10000:
        plan["use_lazy_loading"] = True
        plan["batch_size"] = 100

    # For multi-hop with many seed nodes, use parallel
    seed_count = config.get("seed_count", 5)
    if seed_count >= 5 and max_nodes > 50:
        plan["use_parallel"] = True
        plan["strategy"] = "parallel"

    # For targeted queries on large graphs, use index
    if query_type == "targeted" and node_count > 1000:
        plan["use_index"] = True

    return plan


class GraphOperationCache:
    """Operation-level cache for graph query results (PH4-008).

    Provides fine-grained caching for individual graph operations
    like get_neighbors, find_nodes, etc. This is different from the
    query-level cache in that it caches at the operation level.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
    ) -> None:
        """Initialize the operation cache.

        Args:
            max_size: Maximum number of cached operations
            ttl_seconds: Time-to-live for cache entries
        """
        self._max_size = max_size
        _ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached operation result.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        return self._cache.get(key)

    def put(self, key: str, result: Any) -> None:
        """Cache operation result.

        Args:
            key: Cache key
            result: Result to cache
        """
        # Simple LRU-style eviction when cache is full
        if len(self._cache) >= self._max_size:
            # Remove oldest entry (first in dict iteration order in Python 3.7+)
            if self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        self._cache[key] = result

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached operation.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was found and removed
        """
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all cached operations."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": 0.0,  # Would need to track hits/misses
        }


def create_cache_key(
    operation: str,
    params: Dict[str, Any],
) -> str:
    """Create a cache key for a graph operation.

    Args:
        operation: Operation name (e.g., "get_neighbors", "find_nodes")
        params: Operation parameters

    Returns:
        Stable cache key string
    """
    import hashlib
    import json

    # Normalize params
    normalized = {
        k: sorted(v) if isinstance(v, list) else v
        for k, v in sorted(params.items())
        if v is not None
    }

    key_string = f"{operation}:{json.dumps(normalized, sort_keys=True)}"
    return hashlib.sha256(key_string.encode()).hexdigest()


__all__ = [
    "BatchSizeRecommendation",
    "GraphOptimizer",
    "GraphOptimizationHints",
    "OperationProfile",
    "optimize_batch_size",
    "suggest_query_plan",
    "GraphOperationCache",
    "create_cache_key",
]
