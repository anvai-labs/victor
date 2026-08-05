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

"""Tests for victor.framework.graph_cache module (CompiledGraphCache).

These tests verify the compiled graph caching infrastructure including:
- Graph hash computation
- Cache operations (get, put, get_or_compile)
- Cache invalidation
- Cache statistics
- Global singleton management
"""

import pytest
from typing import TypedDict, List
from unittest.mock import MagicMock, patch

from victor.framework.graph_cache import (
    CompiledGraphCache,
    CompiledGraphCacheConfig,
    get_compiled_graph_cache,
    configure_compiled_graph_cache,
    reset_compiled_graph_cache,
)
from victor.framework.graph import (
    StateGraph,
    CompiledGraph,
    END,
)

# =============================================================================
# Test State Types
# =============================================================================


class SimpleState(TypedDict):
    """Simple state for testing."""

    value: int
    history: List[str]


# =============================================================================
# Node Functions for Testing
# =============================================================================


async def increment_node(state: SimpleState) -> SimpleState:
    """Node that increments value."""
    state["value"] += 1
    state["history"].append("increment")
    return state


async def double_node(state: SimpleState) -> SimpleState:
    """Node that doubles value."""
    state["value"] *= 2
    state["history"].append("double")
    return state


def sync_node(state: SimpleState) -> SimpleState:
    """Synchronous node for testing."""
    state["value"] += 10
    state["history"].append("sync")
    return state


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache_config():
    """Create a cache configuration for tests."""
    return CompiledGraphCacheConfig(
        enabled=True,
        max_entries=10,
        ttl_seconds=60,
    )


@pytest.fixture
def cache(cache_config):
    """Create a cache instance for tests."""
    return CompiledGraphCache(cache_config)


@pytest.fixture
def simple_graph():
    """Create a simple linear graph."""
    graph = StateGraph(SimpleState)
    graph.add_node("inc", increment_node)
    graph.add_node("double", double_node)
    graph.add_edge("inc", "double")
    graph.add_edge("double", END)
    graph.set_entry_point("inc")
    return graph


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset global cache before and after each test."""
    reset_compiled_graph_cache()
    yield
    reset_compiled_graph_cache()


# =============================================================================
# CompiledGraphCacheConfig Tests
# =============================================================================


class TestCompiledGraphCacheConfig:
    """Tests for CompiledGraphCacheConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CompiledGraphCacheConfig()

        assert config.enabled is True
        assert config.max_entries == 50
        assert config.ttl_seconds == 3600

    def test_custom_values(self):
        """Config should accept custom values."""
        config = CompiledGraphCacheConfig(
            enabled=False,
            max_entries=100,
            ttl_seconds=7200,
        )

        assert config.enabled is False
        assert config.max_entries == 100
        assert config.ttl_seconds == 7200


# =============================================================================
# CompiledGraphCache Tests
# =============================================================================


class TestCompiledGraphCache:
    """Tests for CompiledGraphCache."""

    def test_init_enabled(self, cache_config):
        """Cache should initialize when enabled."""
        cache = CompiledGraphCache(cache_config)

        assert cache._config.enabled is True
        assert cache._cache is not None

    def test_init_disabled(self):
        """Cache should not create internal cache when disabled."""
        config = CompiledGraphCacheConfig(enabled=False)
        cache = CompiledGraphCache(config)

        assert cache._config.enabled is False
        assert cache._cache is None

    def test_init_default_config(self):
        """Cache should use default config when none provided."""
        cache = CompiledGraphCache()

        assert cache._config.enabled is True
        assert cache._config.max_entries == 50


class TestGraphHashComputation:
    """Tests for graph hash computation."""

    def test_identical_graphs_same_hash(self, cache, simple_graph):
        """Identical graphs should produce the same hash."""
        # Create a second identical graph
        graph2 = StateGraph(SimpleState)
        graph2.add_node("inc", increment_node)
        graph2.add_node("double", double_node)
        graph2.add_edge("inc", "double")
        graph2.add_edge("double", END)
        graph2.set_entry_point("inc")

        hash1 = cache._compute_graph_hash(simple_graph)
        hash2 = cache._compute_graph_hash(graph2)

        assert hash1 == hash2

    def test_different_nodes_different_hash(self, cache, simple_graph):
        """Graphs with different nodes should have different hashes."""
        graph2 = StateGraph(SimpleState)
        graph2.add_node("sync", sync_node)  # Different node
        graph2.add_node("double", double_node)
        graph2.add_edge("sync", "double")
        graph2.add_edge("double", END)
        graph2.set_entry_point("sync")

        hash1 = cache._compute_graph_hash(simple_graph)
        hash2 = cache._compute_graph_hash(graph2)

        assert hash1 != hash2

    def test_same_name_different_function_body_different_hash(self, cache):
        """Function implementation changes should invalidate the graph hash."""

        async def version_one(state: SimpleState) -> SimpleState:
            state["value"] += 1
            state["history"].append("v1")
            return state

        async def version_two(state: SimpleState) -> SimpleState:
            state["value"] += 2
            state["history"].append("v2")
            return state

        version_one.__name__ = "dynamic_node"
        version_two.__name__ = "dynamic_node"

        graph1 = StateGraph(SimpleState)
        graph1.add_node("dynamic", version_one)
        graph1.add_edge("dynamic", END)
        graph1.set_entry_point("dynamic")

        graph2 = StateGraph(SimpleState)
        graph2.add_node("dynamic", version_two)
        graph2.add_edge("dynamic", END)
        graph2.set_entry_point("dynamic")

        assert cache._compute_graph_hash(graph1) != cache._compute_graph_hash(graph2)

    def test_different_edges_different_hash(self, cache, simple_graph):
        """Graphs with different edges should have different hashes."""
        graph2 = StateGraph(SimpleState)
        graph2.add_node("inc", increment_node)
        graph2.add_node("double", double_node)
        # Different edge structure - direct to END
        graph2.add_edge("inc", END)
        graph2.set_entry_point("inc")

        hash1 = cache._compute_graph_hash(simple_graph)
        hash2 = cache._compute_graph_hash(graph2)

        assert hash1 != hash2

    def test_different_entry_point_different_hash(self, cache):
        """Graphs with different entry points should have different hashes."""
        graph1 = StateGraph(SimpleState)
        graph1.add_node("inc", increment_node)
        graph1.add_node("double", double_node)
        graph1.add_edge("inc", "double")
        graph1.add_edge("double", END)
        graph1.set_entry_point("inc")

        graph2 = StateGraph(SimpleState)
        graph2.add_node("inc", increment_node)
        graph2.add_node("double", double_node)
        graph2.add_edge("inc", "double")
        graph2.add_edge("double", END)
        graph2.set_entry_point("double")  # Different entry point

        hash1 = cache._compute_graph_hash(graph1)
        hash2 = cache._compute_graph_hash(graph2)

        assert hash1 != hash2

    def test_hash_is_string(self, cache, simple_graph):
        """Hash should be a hex string."""
        hash_value = cache._compute_graph_hash(simple_graph)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length


class TestCacheOperations:
    """Tests for cache get/put operations."""

    def test_get_miss(self, cache, simple_graph):
        """get should return None for uncached graph."""
        result = cache.get(simple_graph)

        assert result is None

    def test_put_and_get(self, cache, simple_graph):
        """put should cache compiled graph, get should retrieve it."""
        compiled = simple_graph.compile()

        cache.put(simple_graph, compiled)
        result = cache.get(simple_graph)

        assert result is compiled

    def test_get_disabled_cache(self, simple_graph):
        """get should return None when cache is disabled."""
        config = CompiledGraphCacheConfig(enabled=False)
        cache = CompiledGraphCache(config)

        compiled = simple_graph.compile()
        cache.put(simple_graph, compiled)
        result = cache.get(simple_graph)

        assert result is None

    def test_put_disabled_cache(self, simple_graph):
        """put should return False when cache is disabled."""
        config = CompiledGraphCacheConfig(enabled=False)
        cache = CompiledGraphCache(config)

        compiled = simple_graph.compile()
        result = cache.put(simple_graph, compiled)

        assert result is False


class TestGetOrCompile:
    """Tests for get_or_compile convenience method."""

    def test_get_or_compile_cache_miss(self, cache, simple_graph):
        """get_or_compile should compile and cache on miss."""
        result = cache.get_or_compile(simple_graph)

        assert isinstance(result, CompiledGraph)

        # Should be in cache now
        cached = cache.get(simple_graph)
        assert cached is result

    def test_get_or_compile_cache_hit(self, cache, simple_graph):
        """get_or_compile should return cached on hit."""
        # First call - cache miss
        result1 = cache.get_or_compile(simple_graph)

        # Second call - cache hit
        result2 = cache.get_or_compile(simple_graph)

        # Should be same instance
        assert result1 is result2

        # Stats should show hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_existing(self, cache, simple_graph):
        """invalidate should remove specific graph from cache."""
        compiled = simple_graph.compile()
        cache.put(simple_graph, compiled)

        result = cache.invalidate(simple_graph)

        assert result is True
        assert cache.get(simple_graph) is None

    def test_invalidate_nonexistent(self, cache, simple_graph):
        """invalidate should return False for uncached graph."""
        result = cache.invalidate(simple_graph)

        assert result is False

    def test_invalidate_all(self, cache):
        """invalidate_all should clear all entries."""
        # Add multiple graphs
        graph1 = StateGraph(SimpleState)
        graph1.add_node("inc", increment_node)
        graph1.set_entry_point("inc")
        graph1.add_edge("inc", END)

        graph2 = StateGraph(SimpleState)
        graph2.add_node("double", double_node)
        graph2.set_entry_point("double")
        graph2.add_edge("double", END)

        cache.put(graph1, graph1.compile())
        cache.put(graph2, graph2.compile())

        count = cache.invalidate_all()

        assert count == 2
        assert cache.get(graph1) is None
        assert cache.get(graph2) is None


class TestCacheStats:
    """Tests for cache statistics."""

    def test_initial_stats(self, cache):
        """Initial stats should be zero."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["compilations"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["current_size"] == 0
        assert stats["enabled"] is True

    def test_stats_after_operations(self, cache, simple_graph):
        """Stats should update after operations."""
        # Miss
        cache.get(simple_graph)

        # Compile and put
        compiled = simple_graph.compile()
        cache.put(simple_graph, compiled)

        # Hit
        cache.get(simple_graph)

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["current_size"] == 1

    def test_stats_compilations(self, cache, simple_graph):
        """Stats should track compilations via get_or_compile."""
        cache.get_or_compile(simple_graph)

        stats = cache.get_stats()

        assert stats["compilations"] == 1


# =============================================================================
# Global Singleton Tests
# =============================================================================


class TestGlobalSingleton:
    """Tests for global cache singleton."""

    def test_get_returns_singleton(self):
        """get_compiled_graph_cache should return same instance."""
        cache1 = get_compiled_graph_cache()
        cache2 = get_compiled_graph_cache()

        assert cache1 is cache2

    def test_configure_replaces_singleton(self):
        """configure_compiled_graph_cache should replace singleton."""
        old_cache = get_compiled_graph_cache()

        config = CompiledGraphCacheConfig(max_entries=100)
        configure_compiled_graph_cache(config)

        new_cache = get_compiled_graph_cache()

        assert new_cache is not old_cache
        assert new_cache._config.max_entries == 100

    def test_reset_clears_singleton(self):
        """reset_compiled_graph_cache should clear singleton."""
        cache1 = get_compiled_graph_cache()
        reset_compiled_graph_cache()
        cache2 = get_compiled_graph_cache()

        assert cache1 is not cache2


# =============================================================================
# Integration with CacheCoordinator Tests
# =============================================================================


class TestCacheCoordinatorIntegration:
    """Tests for CacheCoordinator integration with graph cache."""

    def test_coordinator_with_graph_cache(self):
        """CacheCoordinator should accept graph_cache parameter."""
        from victor.framework.coordinators.cache_coordinator import CacheCoordinator

        graph_cache = CompiledGraphCache()
        coordinator = CacheCoordinator(graph_cache=graph_cache)

        assert coordinator._graph_cache is graph_cache

    def test_coordinator_clear_includes_graph_cache(self):
        """clear_cache should clear graph cache."""
        from victor.framework.coordinators.cache_coordinator import CacheCoordinator

        graph_cache = CompiledGraphCache()

        # Add an entry to graph cache
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.set_entry_point("inc")
        graph.add_edge("inc", END)
        graph_cache.put(graph, graph.compile())

        coordinator = CacheCoordinator(graph_cache=graph_cache)
        cleared = coordinator.clear_cache()

        assert cleared >= 1
        assert graph_cache.get(graph) is None

    def test_coordinator_get_stats_includes_graph_cache(self):
        """get_stats should include graph_cache stats."""
        from victor.framework.coordinators.cache_coordinator import CacheCoordinator

        graph_cache = CompiledGraphCache()
        coordinator = CacheCoordinator(graph_cache=graph_cache)

        stats = coordinator.get_stats()

        assert "graph_cache" in stats

    def test_coordinator_get_graph_cache_stats(self):
        """get_graph_cache_stats should return graph cache stats."""
        from victor.framework.coordinators.cache_coordinator import CacheCoordinator

        graph_cache = CompiledGraphCache()
        coordinator = CacheCoordinator(graph_cache=graph_cache)

        stats = coordinator.get_graph_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "enabled" in stats

    def test_coordinator_set_graph_cache(self):
        """set_graph_cache should update graph cache."""
        from victor.framework.coordinators.cache_coordinator import CacheCoordinator

        coordinator = CacheCoordinator()
        new_cache = CompiledGraphCache()

        coordinator.set_graph_cache(new_cache)

        assert coordinator._graph_cache is new_cache


# =============================================================================
# Tool Schema Caching Tests (D.2)
# =============================================================================


class TestToolSchemaCaching:
    """Tests for tool schema caching in ToolRegistry."""

    def test_schema_cache_initialization(self):
        """ToolRegistry should initialize schema cache."""
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()

        assert hasattr(registry, "_schema_cache")
        assert hasattr(registry, "_schema_cache_lock")
        assert registry._schema_cache[True] is None
        assert registry._schema_cache[False] is None

    def test_schema_cache_invalidation_on_register(self):
        """Schema cache should be invalidated on register."""
        from victor.tools.registry import ToolRegistry
        from victor.tools.base import BaseTool, ToolResult

        class DummyTool(BaseTool):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "A dummy tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ok")

        registry = ToolRegistry()

        # Populate cache
        registry.get_tool_schemas(only_enabled=True)
        assert registry._schema_cache[True] is not None
        cached_version = registry._schema_cache[True][0]

        # Register should bump version, making cache stale
        registry.register(DummyTool())
        assert registry._schema_cache_version > cached_version

    def test_schema_cache_invalidation_on_enable_disable(self):
        """Schema cache should be invalidated on enable/disable."""
        from victor.tools.registry import ToolRegistry
        from victor.tools.base import BaseTool, ToolResult

        class DummyTool(BaseTool):
            @property
            def name(self) -> str:
                return "dummy2"

            @property
            def description(self) -> str:
                return "A dummy tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ok")

        registry = ToolRegistry()
        registry.register(DummyTool())

        # Populate cache
        registry.get_tool_schemas(only_enabled=True)
        assert registry._schema_cache[True] is not None
        version_after_populate = registry._schema_cache_version

        # Disable should bump version, making cache stale
        registry.disable_tool("dummy2")
        version_after_disable = registry._schema_cache_version
        assert version_after_disable > version_after_populate

        # Populate cache again
        registry.get_tool_schemas(only_enabled=True)
        assert registry._schema_cache[True] is not None

        # Enable should bump version again
        registry.enable_tool("dummy2")
        assert registry._schema_cache_version > version_after_disable

    def test_schema_cache_hit(self):
        """get_tool_schemas should return cached schemas on hit."""
        from victor.tools.registry import ToolRegistry
        from victor.tools.base import BaseTool, ToolResult

        class DummyTool(BaseTool):
            @property
            def name(self) -> str:
                return "dummy3"

            @property
            def description(self) -> str:
                return "A dummy tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ok")

        registry = ToolRegistry()
        registry.register(DummyTool())

        # First call - cache miss
        schemas1 = registry.get_tool_schemas(only_enabled=True)

        # Second call - cache hit (same list)
        schemas2 = registry.get_tool_schemas(only_enabled=True)

        assert schemas1 is schemas2
