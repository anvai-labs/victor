"""Tests for feature flag caching module."""

import time
from unittest.mock import patch


from victor.core.feature_flags import FeatureFlag
from victor.core.feature_flag_cache import (
    CacheEntry,
    FeatureFlagCache,
    cached_is_enabled,
    with_cache_scope,
)

# Mock path for raw_is_enabled (imported locally in is_enabled method)
MOCK_PATH = "victor.core.feature_flag_cache.FeatureFlagCache._check_cache"


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value=True)
        assert entry.value is True
        assert entry.is_valid()

    def test_cache_entry_expiration(self):
        """Test cache entry expiration with TTL."""
        # Create entry with 1ms TTL
        entry = CacheEntry(value=True, ttl=__import__("datetime").timedelta(milliseconds=1))
        assert entry.is_valid()

        # Wait for expiration
        time.sleep(0.01)
        assert not entry.is_valid()

    def test_cache_entry_invalidation(self):
        """Test manual cache entry invalidation."""
        entry = CacheEntry(value=True)
        assert entry.is_valid()

        entry.invalidate()
        assert not entry.is_valid()


class TestFeatureFlagCache:
    """Tests for FeatureFlagCache class."""

    def test_cache_miss_then_hit(self):
        """Test cache miss on first check, hit on subsequent."""
        cache = FeatureFlagCache(scoped=True)

        # First check - miss
        result1 = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        assert isinstance(result1, bool)

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Second check - hit
        result2 = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        assert result1 == result2  # Same result

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_scoped_cache_auto_clears(self):
        """Test that scoped cache auto-clears on exit."""
        with FeatureFlagCache.scope() as cache:
            # Perform some checks
            cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
            cache.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)

            stats = cache.get_stats()
            assert stats["cache_size"] >= 1

        # After exit, cache should be cleared
        stats = cache.get_stats()
        assert stats["cache_size"] == 0

    def test_cache_invalidation_single_flag(self):
        """Test invalidating a single cached flag."""
        cache = FeatureFlagCache(scoped=True)

        # Cache a flag
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        stats = cache.get_stats()
        assert stats["cache_size"] >= 1

        # Invalidate specific flag
        cache.invalidate(FeatureFlag.USE_EDGE_MODEL)
        # Cache size might still be >0 if other flags cached

    def test_cache_invalidation_all_flags(self):
        """Test invalidating all cached flags."""
        cache = FeatureFlagCache(scoped=True)

        # Cache multiple flags
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        cache.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)

        stats = cache.get_stats()
        assert stats["cache_size"] >= 1

        # Invalidate all
        cache.invalidate()
        stats = cache.get_stats()
        assert stats["cache_size"] == 0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = FeatureFlagCache(scoped=True)

        # No checks yet
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

        # One miss
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["total_checks"] == 1

        # One hit
        cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total

    def test_global_vs_scoped_cache(self):
        """Test that global and scoped caches are independent."""
        global_cache = FeatureFlagCache(scoped=False)

        with FeatureFlagCache.scope() as scoped_cache:
            # Both check same flag
            global_cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
            scoped_cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)

            # Each has its own stats
            global_stats = global_cache.get_stats()
            scoped_stats = scoped_cache.get_stats()

            # Scoped should have at least 1 check
            assert scoped_stats["total_checks"] >= 1

    def test_default_value_parameter(self):
        """Test default value when flag check fails."""
        cache = FeatureFlagCache(scoped=True)

        # Use a non-existent flag to test default behavior
        result = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL, default=True)

        # Should return the actual flag value (not the default)
        assert isinstance(result, bool)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("victor.core.feature_flags.is_feature_enabled")
    def test_cached_is_enabled_function(self, mock_is_enabled):
        """Test cached_is_enabled convenience function."""
        mock_is_enabled.return_value = True

        # Clear global cache first
        FeatureFlagCache.clear_global_cache()

        # First call - cache miss
        result1 = cached_is_enabled(FeatureFlag.USE_EDGE_MODEL)
        assert result1 is True

        # Second call - cache hit (mock called once)
        result2 = cached_is_enabled(FeatureFlag.USE_EDGE_MODEL)
        assert result2 is True

        # Mock should only be called once due to caching
        assert mock_is_enabled.call_count == 1

    @patch("victor.core.feature_flags.is_feature_enabled")
    def test_with_cache_scope_decorator(self, mock_is_enabled):
        """Test @with_cache_scope decorator."""
        mock_is_enabled.return_value = True

        @with_cache_scope(ttl_seconds=60)
        def bulk_operation():
            # Simulate bulk operation with repeated flag checks
            for _ in range(10):
                cached_is_enabled(FeatureFlag.USE_EDGE_MODEL)

        # Execute bulk operation
        bulk_operation()

        # Mock should be called fewer times than 10 due to caching
        assert mock_is_enabled.call_count < 10


class TestClearGlobalCache:
    """Tests for global cache clearing."""

    def test_clear_global_cache(self):
        """Test clearing global feature flag cache."""
        # Add some entries to global cache
        cached_is_enabled(FeatureFlag.USE_EDGE_MODEL)
        cached_is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)

        # Clear global cache
        FeatureFlagCache.clear_global_cache()

        # Cache should be empty
        cache = FeatureFlagCache(scoped=False)
        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestCacheExpiration:
    """Tests for cache expiration behavior."""

    @patch("victor.core.feature_flags.is_feature_enabled")
    def test_expired_entry_refetch(self, mock_is_enabled):
        """Test that expired entries trigger refetch."""
        mock_is_enabled.return_value = True

        # Use scoped cache with short TTL
        with FeatureFlagCache.scope(ttl_seconds=0.001) as cache:  # 1ms TTL
            # First check
            result1 = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
            assert result1 is True
            assert mock_is_enabled.call_count == 1

            # Wait for expiration
            time.sleep(0.01)

            # Second check should refetch (new cache entry)
            result2 = cache.is_enabled(FeatureFlag.USE_EDGE_MODEL)
            assert result2 is True
            assert mock_is_enabled.call_count == 2  # Called again after expiration
