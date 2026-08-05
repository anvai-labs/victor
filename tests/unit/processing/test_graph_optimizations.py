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

"""Tests for graph optimization utilities (PH4-008)."""

from __future__ import annotations

import time

import pytest

from victor.processing.graph_optimizations import (
    GraphOperationCache,
    GraphOptimizer,
    GraphOptimizationHints,
    create_cache_key,
    optimize_batch_size,
    suggest_query_plan,
)


class TestGraphOptimizationHints:
    """Tests for GraphOptimizationHints dataclass."""

    def test_default_hints(self):
        """Test default hint values."""
        hints = GraphOptimizationHints()

        assert hints.batch_size_hint is None
        assert hints.use_parallel is False
        assert hints.cache_key_hint is None
        assert hints.preferred_traversal == "sequential"
        assert hints.skip_optimization is False

    def test_custom_hints(self):
        """Test custom hint values."""
        hints = GraphOptimizationHints(
            batch_size_hint=50,
            use_parallel=True,
            cache_key_hint="test_key",
            preferred_traversal="parallel",
            skip_optimization=False,
        )

        assert hints.batch_size_hint == 50
        assert hints.use_parallel is True
        assert hints.cache_key_hint == "test_key"
        assert hints.preferred_traversal == "parallel"
        assert hints.skip_optimization is False


class TestGraphOptimizer:
    """Tests for GraphOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = GraphOptimizer()

        assert optimizer._profiles == {}
        assert optimizer._optimization_history == []

    def test_analyze_operation_skip_rare(self):
        """Test that rarely called operations are skipped."""
        optimizer = GraphOptimizer()

        profile_data = {
            "avg_time_ms": 100.0,
            "call_count": 2,  # Below threshold of 5
            "node_count": 10,
        }

        hints = optimizer.analyze_operation("rare_operation", profile_data)

        assert hints.skip_optimization is True
        assert hints.use_parallel is False

    def test_analyze_operation_parallel_recommendation(self):
        """Test parallel recommendation for slow operations."""
        optimizer = GraphOptimizer()

        profile_data = {
            "avg_time_ms": 100.0,  # > 50ms
            "call_count": 10,
            "node_count": 20,  # > 10 nodes
        }

        hints = optimizer.analyze_operation("slow_operation", profile_data)

        assert hints.use_parallel is True
        assert hints.batch_size_hint is not None
        assert 4 <= hints.batch_size_hint <= 50

    def test_analyze_operation_cache_recommendation(self):
        """Test cache recommendation for fast, frequent operations."""
        optimizer = GraphOptimizer()

        profile_data = {
            "avg_time_ms": 5.0,  # < 10ms
            "call_count": 15,  # > 10
            "node_count": 5,
        }

        hints = optimizer.analyze_operation("cacheable_operation", profile_data)

        assert hints.cache_key_hint == "cacheable_operation"

    def test_analyze_operation_batch_size_large_graph(self):
        """Test batch size calculation for large node counts."""
        optimizer = GraphOptimizer()

        profile_data = {
            "avg_time_ms": 50.0,
            "call_count": 10,
            "node_count": 150,  # > 100 nodes
        }

        hints = optimizer.analyze_operation("large_operation", profile_data)

        assert hints.batch_size_hint is not None

    def test_calculate_optimal_batch_size_fast(self):
        """Test batch size for fast operations."""
        optimizer = GraphOptimizer()

        batch_size = optimizer._calculate_optimal_batch_size(0.5, 100)

        # For very fast operations (<1ms), use larger batches
        assert 50 <= batch_size <= 500

    def test_calculate_optimal_batch_size_slow(self):
        """Test batch size for slow operations."""
        optimizer = GraphOptimizer()

        batch_size = optimizer._calculate_optimal_batch_size(150.0, 100)

        # For slow operations (>100ms), use smaller batches
        assert batch_size == 50

    def test_calculate_optimal_batch_size_medium(self):
        """Test batch size for medium-speed operations."""
        optimizer = GraphOptimizer()

        batch_size = optimizer._calculate_optimal_batch_size(50.0, 100)

        assert batch_size == 100  # Medium operations default

    def test_record_optimization(self):
        """Test recording an optimization."""
        optimizer = GraphOptimizer()

        optimizer.record_optimization(
            "test_operation",
            "batch_size",
            {"old": 10, "new": 50},
        )

        assert len(optimizer._optimization_history) == 1

        history = optimizer.get_optimization_history()
        assert history[0]["operation"] == "test_operation"
        assert history[0]["optimization"] == "batch_size"

    def test_get_optimization_history(self):
        """Test retrieving optimization history."""
        optimizer = GraphOptimizer()

        optimizer.record_optimization("op1", "opt1", {})
        optimizer.record_optimization("op2", "opt2", {})
        optimizer.record_optimization("op3", "opt3", {})

        history = optimizer.get_optimization_history(limit=2)

        assert len(history) == 2
        # Should get the last 2
        assert history[0]["operation"] == "op2"
        assert history[1]["operation"] == "op3"

    def test_get_optimization_history_filtered(self):
        """Test filtering history by operation name."""
        optimizer = GraphOptimizer()

        optimizer.record_optimization("op1", "opt1", {})
        optimizer.record_optimization("op1", "opt2", {})
        optimizer.record_optimization("op2", "opt3", {})

        history = optimizer.get_optimization_history(operation_name="op1")

        assert len(history) == 2
        assert all(h["operation"] == "op1" for h in history)

    def test_suggest_index_strategy_large_graph(self):
        """Test index suggestions for large graphs."""
        optimizer = GraphOptimizer()

        stats = {
            "nodes": 15000,  # > 10000
            "edges": 20000,
        }

        recommendations = optimizer.suggest_index_strategy(stats)

        assert len(recommendations) > 0
        assert any("Large graph" in r for r in recommendations)

    def test_suggest_index_strategy_high_edge_ratio(self):
        """Test index suggestions for high edge-to-node ratio."""
        optimizer = GraphOptimizer()

        stats = {
            "nodes": 1000,
            "edges": 15000,  # > 10 * nodes
        }

        recommendations = optimizer.suggest_index_strategy(stats)

        assert len(recommendations) > 0
        assert any("High edge-to-node" in r for r in recommendations)

    def test_suggest_index_strategy_very_large_edges(self):
        """Test index suggestions for very large edge count."""
        optimizer = GraphOptimizer()

        stats = {
            "nodes": 5000,
            "edges": 60000,  # > 50000
        }

        recommendations = optimizer.suggest_index_strategy(stats)

        assert len(recommendations) > 0
        assert any("Very large edge" in r for r in recommendations)


class TestOptimizeBatchSize:
    """Tests for optimize_batch_size function."""

    def test_increase_batch_for_fast_operations(self):
        """Test batch size increase for very fast operations."""
        new_size = optimize_batch_size(
            operation_type="get_neighbors",
            current_batch_size=50,
            avg_time_ms=0.5,  # < 1ms
            node_count=100,
        )

        assert new_size == 100  # Doubled, capped at 500

    def test_decrease_batch_for_slow_operations(self):
        """Test batch size decrease for slow operations."""
        new_size = optimize_batch_size(
            operation_type="multi_hop",
            current_batch_size=100,
            avg_time_ms=150.0,  # > 100ms
            node_count=100,
        )

        assert new_size == 50  # Halved, min 10

    def test_no_change_for_medium_operations(self):
        """Test no batch size change for medium-speed operations."""
        current_size = 100
        new_size = optimize_batch_size(
            operation_type="traversal",
            current_batch_size=current_size,
            avg_time_ms=25.0,  # 1-50ms range
            node_count=100,
        )

        assert new_size == current_size

    def test_batch_size_upper_limit(self):
        """Test batch size upper limit."""
        new_size = optimize_batch_size(
            operation_type="fast_op",
            current_batch_size=400,
            avg_time_ms=0.5,
            node_count=100,
        )

        assert new_size == 500  # Capped at 500

    def test_batch_size_lower_limit(self):
        """Test batch size lower limit."""
        new_size = optimize_batch_size(
            operation_type="slow_op",
            current_batch_size=15,
            avg_time_ms=200.0,
            node_count=100,
        )

        assert new_size == 10  # Minimum


class TestSuggestQueryPlan:
    """Tests for suggest_query_plan function."""

    def test_default_plan(self):
        """Test default query plan."""
        plan = suggest_query_plan(
            query_type="traversal",
            graph_stats={"nodes": 1000, "edges": 2000},
            config={"max_nodes": 100, "seed_count": 3},  # Small seed count
        )

        assert plan["strategy"] == "sequential"  # Sequential with small seed_count
        assert plan["use_cache"] is True
        assert plan["use_lazy_loading"] is False
        assert plan["use_parallel"] is False
        assert plan["estimated_nodes"] == 100

    def test_lazy_loading_for_large_graph(self):
        """Test lazy loading suggestion for large graphs."""
        plan = suggest_query_plan(
            query_type="traversal",
            graph_stats={"nodes": 15000, "edges": 30000},  # > 10000
            config={"max_nodes": 100},
        )

        assert plan["use_lazy_loading"] is True
        assert plan["batch_size"] == 100

    def test_parallel_for_multiple_seeds(self):
        """Test parallel suggestion for multiple seed nodes."""
        plan = suggest_query_plan(
            query_type="multi_hop",
            graph_stats={"nodes": 1000, "edges": 2000},
            config={
                "max_nodes": 100,
                "seed_count": 5,  # >= 5
            },
        )

        assert plan["use_parallel"] is True
        assert plan["strategy"] == "parallel"

    def test_index_for_targeted_query(self):
        """Test index usage for targeted queries on large graphs."""
        plan = suggest_query_plan(
            query_type="targeted",
            graph_stats={"nodes": 5000, "edges": 10000},  # > 1000
            config={"max_nodes": 100},
        )

        assert plan["use_index"] is True


class TestGraphOperationCache:
    """Tests for GraphOperationCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with defaults."""
        cache = GraphOperationCache()

        assert cache._max_size == 1000
        assert cache._cache == {}

    def test_cache_custom_initialization(self):
        """Test cache initialization with custom values."""
        cache = GraphOperationCache(
            max_size=500,
            ttl_seconds=600,
        )

        assert cache._max_size == 500
        # Note: ttl_seconds is stored but not used in basic implementation

    def test_cache_put_and_get(self):
        """Test putting and getting cached values."""
        cache = GraphOperationCache()

        cache.put("key1", {"result": "value1"})
        result = cache.get("key1")

        assert result == {"result": "value1"}

    def test_cache_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        cache = GraphOperationCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_invalidate(self):
        """Test invalidating a cache entry."""
        cache = GraphOperationCache()

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        removed = cache.invalidate("key1")
        assert removed is True
        assert cache.get("key1") is None

    def test_cache_invalidate_nonexistent(self):
        """Test invalidating non-existent key returns False."""
        cache = GraphOperationCache()

        removed = cache.invalidate("nonexistent")

        assert removed is False

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = GraphOperationCache()

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache._cache) == 0

    def test_cache_eviction(self):
        """Test LRU-style eviction when cache is full."""
        cache = GraphOperationCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1

        # In Python 3.7+, dict maintains insertion order
        # So the first item (key1) should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_stats(self):
        """Test getting cache statistics."""
        cache = GraphOperationCache(max_size=100)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert "hit_rate" in stats


class TestCreateCacheKey:
    """Tests for create_cache_key function."""

    def test_cache_key_simple(self):
        """Test creating a simple cache key."""
        key = create_cache_key(
            operation="get_neighbors",
            params={"node_id": "test_node"},
        )

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex length

    def test_cache_key_different_operations(self):
        """Test that different operations produce different keys."""
        key1 = create_cache_key("operation1", {"param": "value"})
        key2 = create_cache_key("operation2", {"param": "value"})

        assert key1 != key2

    def test_cache_key_different_params(self):
        """Test that different parameters produce different keys."""
        key1 = create_cache_key("operation", {"param": "value1"})
        key2 = create_cache_key("operation", {"param": "value2"})

        assert key1 != key2

    def test_cache_key_param_order_independence(self):
        """Test that parameter order doesn't affect key."""
        key1 = create_cache_key(
            "operation",
            {"a": 1, "b": 2, "c": 3},
        )
        key2 = create_cache_key(
            "operation",
            {"c": 3, "a": 1, "b": 2},
        )

        assert key1 == key2

    def test_cache_key_list_normalization(self):
        """Test that lists are normalized for cache key."""
        key1 = create_cache_key(
            "operation",
            {"items": [3, 1, 2]},
        )
        key2 = create_cache_key(
            "operation",
            {"items": [1, 2, 3]},
        )

        # Lists should be sorted, so keys should match
        assert key1 == key2

    def test_cache_key_none_values_filtered(self):
        """Test that None values are filtered from cache key."""
        key1 = create_cache_key(
            "operation",
            {"a": 1, "b": None, "c": 3},
        )
        key2 = create_cache_key(
            "operation",
            {"a": 1, "c": 3},
        )

        # None values should be filtered out
        assert key1 == key2


class TestOptimizationIntegration:
    """Integration tests for optimization utilities."""

    def test_end_to_end_optimization_flow(self):
        """Test complete optimization workflow."""
        optimizer = GraphOptimizer()

        # Simulate profiling data collection
        profile_data = {
            "avg_time_ms": 80.0,
            "call_count": 20,
            "node_count": 150,
        }

        # Get optimization hints
        hints = optimizer.analyze_operation("test_operation", profile_data)

        # Apply optimizations
        if hints.batch_size_hint:
            # Record the optimization
            optimizer.record_optimization(
                "test_operation",
                "batch_size",
                {"new_size": hints.batch_size_hint},
            )

        # Verify hints
        assert hints.use_parallel is True
        assert hints.batch_size_hint is not None

        # Verify history
        history = optimizer.get_optimization_history()
        assert len(history) == 1

    def test_cache_integration_with_optimizer(self):
        """Test using cache with optimizer recommendations."""
        cache = GraphOperationCache(max_size=100)
        optimizer = GraphOptimizer()

        # Get cache recommendation
        profile_data = {
            "avg_time_ms": 5.0,
            "call_count": 15,
            "node_count": 5,
        }

        hints = optimizer.analyze_operation("cacheable_op", profile_data)

        # Use the cache key hint
        if hints.cache_key_hint:
            cache.put(hints.cache_key_hint, {"result": "cached_data"})

            # Retrieve from cache
            result = cache.get(hints.cache_key_hint)
            assert result == {"result": "cached_data"}

    def test_query_plan_with_graph_stats(self):
        """Test query planning based on graph statistics."""
        optimizer = GraphOptimizer()

        # Get index recommendations
        stats = {
            "nodes": 20000,
            "edges": 300000,
        }

        recommendations = optimizer.suggest_index_strategy(stats)

        # Get query plan
        plan = suggest_query_plan(
            query_type="traversal",
            graph_stats=stats,
            config={"max_nodes": 100},
        )

        # Both should suggest optimizations for large graphs
        assert len(recommendations) > 0
        assert plan["use_lazy_loading"] is True
