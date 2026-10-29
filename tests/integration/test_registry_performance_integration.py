"""Integration tests for registry performance optimizations.

These tests validate that performance optimizations work correctly
under realistic scenarios and integrate properly with existing code.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.tools.registry import ToolRegistry
from victor.tools.batch_registration import BatchRegistrar, register_tools_batch
from victor.tools.base import BaseTool
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)
from victor.tools.metadata import ToolMetadata
from victor.core.feature_flag_cache import FeatureFlagCache
from victor.tools.query_cache import QueryCache
from victor.core.feature_flags import FeatureFlag
from typing import Dict, Any, List


class TestTool(BaseTool):
    """Test tool for integration testing."""

    def __init__(self, name: str, description: str, tags: List[str] = None):
        self._name = name
        self._description = description
        self._tags = tags or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=self._description,
            category=ExecutionCategory.READ_ONLY,
            access_mode=AccessMode.READONLY,
            cost_tier=CostTier.FREE,
            danger_level=DangerLevel.SAFE,
            priority=Priority.MEDIUM,
            tags=self._tags,
        )

    async def execute(self, _exec_ctx, **kwargs):
        from victor.tools.base import ToolResult

        return ToolResult(success=True, output="test output")


class TestBatchRegistrationIntegration:
    """Integration tests for batch registration."""

    def test_batch_registration_with_validation(self):
        """Test batch registration with validation errors."""
        registry = ToolRegistry()

        # Create tools with some invalid ones
        tools = []
        for i in range(10):
            tool = TestTool(name=f"tool_{i}", description=f"Test tool {i}", tags=[f"tag_{i % 3}"])
            tools.append(tool)

        # Add duplicate tool name (will be filtered by validation)
        duplicate_tool = TestTool(name="tool_5", description="Duplicate")
        tools.append(duplicate_tool)

        # Register batch
        registrar = BatchRegistrar(registry)
        result = registrar.register_batch(tools, fail_fast=False)

        # Should have registered 10 tools (duplicate handled gracefully)
        assert result.success_count >= 10
        assert result.duration_ms > 0
        assert result.cache_invalidations == 1  # Single invalidation

    def test_batch_registration_chunked(self):
        """Test batch registration with chunking."""
        registry = ToolRegistry()

        # Create 250 tools (should trigger 3 chunks of 100)
        tools = []
        for i in range(250):
            tool = TestTool(name=f"tool_{i}", description=f"Test tool {i}", tags=[f"tag_{i % 10}"])
            tools.append(tool)

        # Register with chunking
        registrar = BatchRegistrar(registry)
        result = registrar.register_batch(tools, chunk_size=100)

        # All tools should be registered
        assert result.success_count == 250
        assert result.total_count == 250
        assert result.cache_invalidations == 1  # Still single invalidation

    def test_batch_registration_convenience_function(self):
        """Test convenience function for batch registration."""
        registry = ToolRegistry()

        tools = [
            TestTool(name=f"tool_{i}", description=f"Test tool {i}", tags=[f"tag_{i % 3}"])
            for i in range(50)
        ]

        # Use convenience function
        result = register_tools_batch(registry, tools)

        assert result.success_count == 50
        assert result.failure_count == 0

    def test_batch_registration_preserves_registry_state(self):
        """Test that batch registration preserves registry consistency."""
        registry = ToolRegistry()

        # Register initial tools
        initial_tools = [
            TestTool(name=f"initial_{i}", description=f"Initial {i}") for i in range(10)
        ]
        for tool in initial_tools:
            registry.register(tool)

        # Register batch
        batch_tools = [TestTool(name=f"batch_{i}", description=f"Batch {i}") for i in range(20)]
        registrar = BatchRegistrar(registry)
        result = registrar.register_batch(batch_tools)

        # Both sets should be present
        assert result.success_count == 20
        for i in range(10):
            assert registry.get(f"initial_{i}") is not None
        for i in range(20):
            assert registry.get(f"batch_{i}") is not None


class TestFeatureFlagCacheIntegration:
    """Integration tests for feature flag caching."""

    def test_feature_flag_cache_with_real_flags(self):
        """Test feature flag cache with actual feature flags."""
        cache = FeatureFlagCache(scoped=True)

        # Check multiple flags
        flags_to_check = [
            FeatureFlag.USE_EDGE_MODEL,
            FeatureFlag.USE_LEARNING_FROM_EXECUTION,
            FeatureFlag.USE_EDGE_MODEL,
        ]

        results = {}
        for flag in flags_to_check:
            results[flag] = cache.is_enabled(flag)

        # All should return boolean
        for flag, value in results.items():
            assert isinstance(value, bool)

        # Check cache statistics
        stats = cache.get_stats()
        assert stats["total_checks"] == 3
        assert stats["cache_size"] >= 1  # At least one cached

    def test_feature_flag_cache_invalidation(self):
        """Test feature flag cache invalidation."""
        cache = FeatureFlagCache(scoped=True)

        # Cache some flags
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        cache.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)

        stats_before = cache.get_stats()
        assert stats_before["cache_size"] >= 1

        # Invalidate all
        cache.invalidate()

        stats_after = cache.get_stats()
        assert stats_after["cache_size"] == 0

    def test_feature_flag_cache_with_context_manager(self):
        """Test feature flag cache with context manager."""
        with FeatureFlagCache.scope(ttl_seconds=60) as cache:
            # Perform some checks
            for _ in range(10):
                cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)

            stats = cache.get_stats()
            assert stats["total_checks"] == 10
            assert stats["hits"] == 9  # First miss, then 9 hits

        # Cache should be cleared after exit
        stats_final = cache.get_stats()
        assert stats_final["cache_size"] == 0


class TestQueryCacheIntegration:
    """Integration tests for query result caching."""

    def test_query_cache_with_registry(self):
        """Test query cache with actual registry operations."""
        registry = ToolRegistry()
        cache = QueryCache()

        # Populate registry
        tools = [
            TestTool(name=f"tool_{i}", description=f"Test tool {i}", tags=[f"tag_{i % 3}"])
            for i in range(20)
        ]
        with registry.batch_update():
            for tool in tools:
                registry.register(tool)

        # Cache some queries
        def get_tool_5():
            return registry.get("tool_5")

        result1 = cache.get("tool_5", get_tool_5)
        result2 = cache.get("tool_5", get_tool_5)

        assert result1 is not None
        assert result2 is not None
        assert result1 == result2

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_query_cache_with_tags(self):
        """Test query cache with tag-based invalidation."""
        cache = QueryCache()

        # Cache entries with tags
        cache.get("key1", lambda: "value1", tags={"tool:foo"})
        cache.get("key2", lambda: "value2", tags={"tool:bar"})
        cache.get("key3", lambda: "value3", tags={"tag:testing"})

        assert cache.get_stats()["cache_size"] == 3

        # Invalidate entries with specific tag
        count = cache.invalidate_tags({"tool:foo"})
        assert count == 1
        assert cache.get_stats()["cache_size"] == 2

    def test_query_cache_expiration(self):
        """Test query cache with TTL expiration."""
        cache = QueryCache(default_ttl_seconds=0.001)  # 1ms TTL
        import time

        cache.get("key1", lambda: "value1")
        assert cache.get_stats()["cache_size"] == 1

        # Wait for expiration
        time.sleep(0.01)

        # Try to get again - should recompute
        cache.get("key1", lambda: "value1")
        stats = cache.get_stats()
        assert stats["misses"] == 2  # Two misses due to expiration


class TestCachingIntegration:
    """Integration tests for combined caching strategies."""

    def test_combined_caching_workflow(self):
        """Test realistic workflow with all caching layers."""
        registry = ToolRegistry()

        # Feature flag caching
        with FeatureFlagCache.scope() as flag_cache:
            # Query caching
            query_cache = QueryCache()

            # Batch registration
            tools = [
                TestTool(
                    name=f"tool_{i}",
                    description=f"Test tool {i}",
                    tags=[f"tag_{i % 3}"],
                )
                for i in range(50)
            ]

            # Register with batch API
            registrar = BatchRegistrar(registry)
            result = registrar.register_batch(tools)

            assert result.success_count == 50

            # Perform cached queries
            def get_tool(name):
                return registry.get(name)

            for i in range(10):
                tool = query_cache.get(f"tool_{i}", lambda idx=i: get_tool(f"tool_{idx}"))
                assert tool is not None

            # Check statistics
            flag_stats = flag_cache.get_stats()
            query_stats = query_cache.get_stats()

            assert flag_stats["total_checks"] >= 0
            assert query_stats["cache_size"] >= 1

    def test_caching_with_concurrent_access(self):
        """Test caching behavior under concurrent-like access."""
        registry = ToolRegistry()
        cache = QueryCache()

        # Populate registry
        tools = [TestTool(name=f"tool_{i}", description=f"Test tool {i}") for i in range(100)]
        with registry.batch_update():
            for tool in tools:
                registry.register(tool)

        # Simulate repeated access patterns (cache hits)
        results = []
        for i in range(100):
            # Access same tools multiple times to test caching
            tool_name = f"tool_{i % 10}"  # Only 10 unique tools
            result = cache.get(tool_name, lambda name=tool_name: registry.get(name))
            results.append(result)

        # All lookups should succeed
        assert all(r is not None for r in results)

        # Check cache effectiveness - should have hits since we repeated tools
        stats = cache.get_stats()
        assert stats["hits"] >= 90  # 100 accesses, only 10 unique keys


class TestOptimizationErrorHandling:
    """Integration tests for error handling in optimizations."""

    def test_batch_registration_with_failures(self):
        """Test batch registration handles individual failures gracefully."""
        registry = ToolRegistry()

        # Create mix of valid and invalid tools
        tools = []
        for i in range(10):
            tool = TestTool(name=f"tool_{i}", description=f"Test tool {i}")
            tools.append(tool)

        # Add a tool that will fail (same name as existing)
        duplicate = TestTool(name="tool_5", description="Duplicate")
        tools.append(duplicate)

        # Register with fail_fast=False
        registrar = BatchRegistrar(registry)
        result = registrar.register_batch(tools, fail_fast=False)

        # Should have some successes
        assert result.success_count >= 10

        # Should handle duplicate gracefully
        assert result.total_count >= 10

    def test_cache_handles_errors_gracefully(self):
        """Test that caching handles computation errors."""
        cache = QueryCache()

        def failing_computation():
            raise ValueError("Computation failed")

        # First call should raise error
        with pytest.raises(ValueError):
            cache.get("failing_key", failing_computation)

        # Cache should not store failed result
        assert cache.get_stats()["cache_size"] == 0

    def test_feature_flag_cache_handles_missing_flags(self):
        """Test feature flag cache with default values."""
        cache = FeatureFlagCache(scoped=True)

        # Use a flag that might not exist
        result = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL, default=True)
        assert isinstance(result, bool)


class TestBackwardCompatibility:
    """Integration tests for backward compatibility."""

    def test_existing_code_works_unchanged(self):
        """Test that existing registration code still works."""
        registry = ToolRegistry()

        # Old-style individual registration
        for i in range(10):
            tool = TestTool(name=f"tool_{i}", description=f"Test tool {i}")
            registry.register(tool)

        # All tools should be registered
        for i in range(10):
            assert registry.get(f"tool_{i}") is not None

    def test_batch_update_context_still_works(self):
        """Test that existing batch_update context still works."""
        registry = ToolRegistry()

        # Old-style batch update
        with registry.batch_update():
            for i in range(10):
                tool = TestTool(name=f"tool_{i}", description=f"Test tool {i}")
                registry.register(tool)

        # All tools should be registered
        for i in range(10):
            assert registry.get(f"tool_{i}") is not None

    def test_new_apis_are_additive(self):
        """Test that new APIs don't break existing ones."""
        registry = ToolRegistry()

        # Mix old and new styles
        # Old style
        for i in range(5):
            tool = TestTool(name=f"old_{i}", description=f"Old style {i}")
            registry.register(tool)

        # New style
        new_tools = [TestTool(name=f"new_{i}", description=f"New style {i}") for i in range(5)]
        registrar = BatchRegistrar(registry)
        result = registrar.register_batch(new_tools)

        # Both should work
        assert result.success_count == 5
        for i in range(5):
            assert registry.get(f"old_{i}") is not None
            assert registry.get(f"new_{i}") is not None
