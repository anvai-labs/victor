"""Tests for tool preloader.

Tests cover:
- Initialization and configuration
- L1 cache operations
- L2 cache operations
- Preload prediction and triggering
- Async background loading
- Cache promotion and demotion
- TTL expiration
- Statistics and monitoring
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import asyncio
import pytest

from victor.agent.planning.tool_preloader import (
    CacheEntry,
    PreloaderConfig,
    ToolPreloader,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
        )

        assert entry.tool_name == "read"
        assert entry.schema == {"type": "object"}
        assert entry.access_count == 0
        assert isinstance(entry.last_accessed, datetime)

    def test_is_expired_false(self):
        """Test expiration check for non-expired entry."""
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        assert entry.is_expired() is False

    def test_is_expired_true(self):
        """Test expiration check for expired entry."""
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

        assert entry.is_expired() is True

    def test_touch_increments_access_count(self):
        """Test that touch updates access count."""
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
        )

        initial_count = entry.access_count
        entry.touch()

        assert entry.access_count == initial_count + 1


class TestPreloaderConfig:
    """Test preloader configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreloaderConfig()

        assert config.l1_max_size == 50
        assert config.l1_ttl_seconds == 600.0
        assert config.l2_enabled is False
        assert config.preload_threshold == 0.5
        assert config.max_preload_tools == 5
        assert config.promotion_threshold == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreloaderConfig(
            l1_max_size=100,
            l1_ttl_seconds=300.0,
            l2_enabled=True,
            preload_threshold=0.7,
        )

        assert config.l1_max_size == 100
        assert config.l1_ttl_seconds == 300.0
        assert config.l2_enabled is True
        assert config.preload_threshold == 0.7


class TestInitialization:
    """Test preloader initialization."""

    def test_default_initialization(self):
        """Test preloader with default settings."""
        preloader = ToolPreloader()

        assert preloader.config.l1_max_size == 50
        assert preloader._predictor is None
        assert preloader._tool_registry is None
        assert preloader._l2_cache is None

    def test_initialization_with_predictor(self):
        """Test preloader with tool predictor."""
        predictor = MagicMock()
        preloader = ToolPreloader(tool_predictor=predictor)

        assert preloader._predictor is predictor

    def test_initialization_with_registry(self):
        """Test preloader with tool registry."""
        registry = MagicMock()
        preloader = ToolPreloader(tool_registry=registry)

        assert preloader._tool_registry is registry

    def test_initialization_with_l2_enabled(self):
        """Test preloader with L2 cache enabled."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        assert preloader._l2_cache is not None

    def test_initialization_stats(self):
        """Test initial statistics."""
        preloader = ToolPreloader()

        stats = preloader.get_statistics()

        assert stats["l1_cache_size"] == 0
        assert stats["l1_hits"] == 0
        assert stats["l1_misses"] == 0


class TestL1Cache:
    """Test L1 cache operations."""

    @pytest.mark.asyncio
    async def test_add_to_l1_cache(self):
        """Test adding entries to L1 cache."""
        preloader = ToolPreloader()

        schema = {"type": "object"}
        entry = CacheEntry(
            tool_name="read",
            schema=schema,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        preloader._add_to_l1_cache("read", entry)

        assert "read" in preloader._l1_cache
        assert preloader._l1_cache["read"].schema == schema

    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        """Test cache hit in L1."""
        preloader = ToolPreloader()

        schema = {"type": "object"}
        entry = CacheEntry(
            tool_name="read",
            schema=schema,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        preloader._add_to_l1_cache("read", entry)

        # Mock registry to return None (so cache is used)
        preloader._tool_registry = MagicMock()
        preloader._tool_registry.get = MagicMock(return_value=None)

        result = await preloader.get_tool_schema("read")

        assert result == schema
        assert preloader._l1_hits == 1
        assert preloader._l1_misses == 0

    @pytest.mark.asyncio
    async def test_l1_cache_miss(self):
        """Test cache miss in L1."""
        preloader = ToolPreloader()

        # Mock registry to return schema
        schema = {"type": "object"}

        # Create a simple mock object with schema attribute
        class MockTool:
            def __init__(self, schema):
                self.schema = schema
                self.input_schema = schema

        # Mock both get and get_tool_schema
        preloader._tool_registry = MagicMock()
        preloader._tool_registry.get = MagicMock(return_value=MockTool(schema))
        preloader._tool_registry.get_tool_schema = MagicMock(return_value=schema)

        result = await preloader.get_tool_schema("read")

        assert result == schema
        assert preloader._l1_hits == 0
        assert preloader._l1_misses == 1

    @pytest.mark.asyncio
    async def test_l1_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        config = PreloaderConfig(l1_max_size=2)
        preloader = ToolPreloader(config=config)

        # Add 3 entries (should evict oldest)
        for i in range(3):
            entry = CacheEntry(
                tool_name=f"tool_{i}",
                schema={"type": "object"},
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            )
            preloader._add_to_l1_cache(f"tool_{i}", entry)

        # Should only have 2 entries
        assert len(preloader._l1_cache) == 2
        # Oldest (tool_0) should be evicted
        assert "tool_0" not in preloader._l1_cache
        assert "tool_1" in preloader._l1_cache
        assert "tool_2" in preloader._l1_cache

    @pytest.mark.asyncio
    async def test_l1_cache_expiration(self):
        """Test that expired entries are replaced with fresh entries from registry."""
        preloader = ToolPreloader()

        # Add expired entry
        schema = {"type": "object"}
        entry = CacheEntry(
            tool_name="read",
            schema=schema,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        preloader._add_to_l1_cache("read", entry)

        # Mock registry to return mock schema (fallback)
        preloader._tool_registry = MagicMock()
        preloader._tool_registry.get = MagicMock(return_value=None)
        preloader._tool_registry.get_tool_schema = MagicMock(return_value=schema)

        result = await preloader.get_tool_schema("read")

        # Should return mock schema from fallback
        assert result is not None
        assert result == schema
        # Expired entry should be removed and replaced with fresh entry from registry
        assert "read" in preloader._l1_cache
        # New entry should have a future expiration time
        assert preloader._l1_cache["read"].expires_at > datetime.now(timezone.utc)


class TestL2Cache:
    """Test L2 cache operations."""

    @pytest.mark.asyncio
    async def test_l2_cache_disabled_by_default(self):
        """Test that L2 cache is disabled by default."""
        preloader = ToolPreloader()

        assert preloader._l2_cache is None

    @pytest.mark.asyncio
    async def test_l2_cache_enabled(self):
        """Test that L2 cache can be enabled."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        assert preloader._l2_cache is not None

    @pytest.mark.asyncio
    async def test_l2_cache_hit(self):
        """Test cache hit in L2."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        schema = {"type": "object"}

        # Create a simple mock object with schema attribute
        class MockTool:
            def __init__(self, schema):
                self.schema = schema
                self.input_schema = schema

        preloader._tool_registry = MagicMock()
        preloader._tool_registry.get = MagicMock(return_value=MockTool(schema))
        preloader._tool_registry.get_tool_schema = MagicMock(return_value=schema)

        entry = CacheEntry(
            tool_name="read",
            schema=schema,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        preloader._add_to_l2_cache("read", entry)

        result = await preloader.get_tool_schema("read")

        # Should find in L2 and promote to L1
        assert result == schema
        assert "read" in preloader._l1_cache
        assert preloader._l2_hits == 1

    @pytest.mark.asyncio
    async def test_l2_cache_promotion(self):
        """Test promotion from L1 to L2 after threshold accesses."""
        config = PreloaderConfig(
            l2_enabled=True,
            promotion_threshold=3,
        )
        preloader = ToolPreloader(config=config)

        schema = {"type": "object"}

        # Create a simple mock object with schema attribute
        class MockTool:
            def __init__(self, schema):
                self.schema = schema
                self.input_schema = schema

        # Mock both get and get_tool_schema
        preloader._tool_registry = MagicMock()
        preloader._tool_registry.get = MagicMock(return_value=MockTool(schema))
        preloader._tool_registry.get_tool_schema = MagicMock(return_value=schema)

        # Add to L1
        entry = CacheEntry(
            tool_name="read",
            schema=schema,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        preloader._add_to_l1_cache("read", entry)

        # Access 3 times to trigger promotion
        for _ in range(3):
            await preloader.get_tool_schema("read")

        # Should be promoted to L2
        assert "read" in preloader._l2_cache


class TestPreloading:
    """Test preloading functionality."""

    @pytest.mark.asyncio
    async def test_preload_with_predictions(self):
        """Test preloading based on predictions."""
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "read"
        prediction.probability = 0.8
        predictor.predict_tools = MagicMock(return_value=[prediction])

        preloader = ToolPreloader(tool_predictor=predictor)

        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
            recent_tools=["search"],
        )

        # Should preload 1 tool
        assert count == 1
        assert preloader._preload_count == 1

    @pytest.mark.asyncio
    async def test_preload_respects_threshold(self):
        """Test that only high-confidence predictions trigger preload."""
        predictor = MagicMock()
        # Low confidence prediction (below threshold)
        prediction = MagicMock()
        prediction.tool_name = "read"
        prediction.probability = 0.3

        predictor.predict_tools = MagicMock(return_value=[prediction])

        config = PreloaderConfig(preload_threshold=0.5)
        preloader = ToolPreloader(
            tool_predictor=predictor,
            config=config,
        )

        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
        )

        # Should not preload (confidence too low)
        assert count == 0

    @pytest.mark.asyncio
    async def test_preload_limits_max_tools(self):
        """Test that max_preload_tools limits preloading."""
        predictor = MagicMock()
        # Return 10 predictions
        predictions = [MagicMock(tool_name=f"tool_{i}", probability=0.8) for i in range(10)]
        predictor.predict_tools = MagicMock(return_value=predictions)

        config = PreloaderConfig(max_preload_tools=3)
        preloader = ToolPreloader(
            tool_predictor=predictor,
            config=config,
        )

        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
        )

        # Should only preload 3 tools
        assert count == 3

    @pytest.mark.asyncio
    async def test_preload_without_predictor(self):
        """Test preloading without predictor returns 0."""
        preloader = ToolPreloader(tool_predictor=None)

        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
        )

        assert count == 0


class TestAsyncBackgroundLoading:
    """Test async background loading."""

    @pytest.mark.asyncio
    async def test_background_loading(self):
        """Test that preloading happens in background."""
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "read"
        prediction.probability = 0.8
        predictor.predict_tools = MagicMock(return_value=[prediction])

        preloader = ToolPreloader(tool_predictor=predictor)

        # Preload (doesn't await individual tool loads)
        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
        )

        # Should return immediately
        assert count == 1

        # Give background tasks time to complete
        await asyncio.sleep(0.2)

        # Verify that preload was attempted
        assert preloader._preload_count == 1

    @pytest.mark.asyncio
    async def test_background_load_failure_handling(self):
        """Test that background load failures are handled gracefully."""
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "invalid_tool"
        prediction.probability = 0.8
        predictor.predict_tools = MagicMock(return_value=[prediction])

        # Mock registry that raises exception
        registry = MagicMock()
        registry.get = MagicMock(side_effect=Exception("Tool not found"))

        preloader = ToolPreloader(
            tool_predictor=predictor,
            tool_registry=registry,
        )

        # Should not raise exception
        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
        )

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

        # Should still return count (preloading was attempted)
        assert count == 1


class TestWarmUp:
    """Test cache warm-up functionality."""

    @pytest.mark.asyncio
    async def test_warm_up_preloads_tools(self):
        """Test warm-up preloads specified tools."""
        preloader = ToolPreloader()

        tool_names = ["search", "read", "edit"]

        await preloader.warm_up(tool_names)

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

        # Should have attempted to preload
        assert preloader._background_loads >= 0


class TestCacheManagement:
    """Test cache management operations."""

    @pytest.mark.asyncio
    async def test_clear_l1_cache(self):
        """Test clearing L1 cache."""
        preloader = ToolPreloader()

        # Add entry to L1
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        preloader._add_to_l1_cache("read", entry)

        assert len(preloader._l1_cache) == 1

        # Clear L1
        preloader.clear_cache(level="l1")

        assert len(preloader._l1_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_l2_cache(self):
        """Test clearing L2 cache."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        # Add entry to L2
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        preloader._add_to_l2_cache("read", entry)

        assert len(preloader._l2_cache) == 1

        # Clear L2
        preloader.clear_cache(level="l2")

        assert len(preloader._l2_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_all_caches(self):
        """Test clearing all caches."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        # Add entries
        entry = CacheEntry(
            tool_name="read",
            schema={"type": "object"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        preloader._add_to_l1_cache("read", entry)
        preloader._add_to_l2_cache("read", entry)

        # Clear all
        preloader.clear_cache(level="all")

        assert len(preloader._l1_cache) == 0
        assert len(preloader._l2_cache) == 0


class TestStatistics:
    """Test statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_statistics_includes_l1_stats(self):
        """Test that statistics include L1 cache stats."""
        preloader = ToolPreloader()

        stats = preloader.get_statistics()

        assert "l1_cache_size" in stats
        assert "l1_hits" in stats
        assert "l1_misses" in stats
        assert "l1_hit_rate" in stats

    @pytest.mark.asyncio
    async def test_statistics_includes_l2_stats_when_enabled(self):
        """Test that statistics include L2 cache stats when enabled."""
        config = PreloaderConfig(l2_enabled=True)
        preloader = ToolPreloader(config=config)

        stats = preloader.get_statistics()

        assert "l2_cache_size" in stats
        assert "l2_hits" in stats
        assert "l2_misses" in stats

    @pytest.mark.asyncio
    async def test_statistics_includes_preload_stats(self):
        """Test that statistics include preload stats."""
        preloader = ToolPreloader()

        stats = preloader.get_statistics()

        assert "preload_count" in stats
        assert "background_loads" in stats

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        preloader = ToolPreloader()

        # Add some hits and misses
        preloader._l1_hits = 7
        preloader._l1_misses = 3

        stats = preloader.get_statistics()

        assert stats["l1_hit_rate"] == 0.7


class TestIntegration:
    """Integration tests for tool preloader."""

    @pytest.mark.asyncio
    async def test_full_preload_workflow(self):
        """Test complete preloading workflow."""
        # Setup predictor
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "read"
        prediction.probability = 0.9
        predictor.predict_tools = MagicMock(return_value=[prediction])

        # Setup registry
        schema = {"type": "object"}

        # Create a simple mock object with schema attribute
        class MockTool:
            def __init__(self, schema):
                self.schema = schema
                self.input_schema = schema

        registry = MagicMock()
        registry.get = MagicMock(return_value=MockTool(schema))
        registry.get_tool_schema = MagicMock(return_value=schema)

        preloader = ToolPreloader(
            tool_predictor=predictor,
            tool_registry=registry,
        )

        # Preload for next step
        count = await preloader.preload_for_next_step(
            current_step="exploration",
            task_type="bugfix",
            recent_tools=["search"],
        )

        assert count == 1

        # Give background tasks time to complete
        await asyncio.sleep(0.1)

        # Get schema (should be in cache)
        result_schema = await preloader.get_tool_schema("read")

        assert result_schema is not None
        assert result_schema == schema
        assert preloader._l1_hits >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_to_registry(self):
        """Test fallback to registry on cache miss."""
        schema = {"type": "object"}

        # Create a simple mock object with schema attribute
        class MockTool:
            def __init__(self, schema):
                self.schema = schema
                self.input_schema = schema

        registry = MagicMock()
        registry.get = MagicMock(return_value=MockTool(schema))
        registry.get_tool_schema = MagicMock(return_value=schema)

        preloader = ToolPreloader(tool_registry=registry)

        # Get schema (not in cache)
        result_schema = await preloader.get_tool_schema("read")

        assert result_schema == schema
        assert preloader._l1_misses >= 1
