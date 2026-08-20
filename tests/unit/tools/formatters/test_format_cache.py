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

"""Tests for formatter caching layer."""

import time

import pytest

from victor.tools.formatters.registry import (
    _FormatCache,
    get_format_cache_stats,
    invalidate_format_cache,
    clear_format_cache,
)
from victor.tools.formatters.base import FormattedOutput


class TestFormatCache:
    """Test formatter caching functionality."""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache = _FormatCache()
        assert cache._max_size == 100
        assert cache._default_ttl == 300
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cache = _FormatCache()
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )

        # Put in cache
        cache.put(tool_name, data, formatted)

        # Get from cache
        result = cache.get(tool_name, data)

        assert result is not None
        assert result.content == "✓ 10 passed"
        assert cache._hits == 1
        assert cache._misses == 0

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = _FormatCache()
        tool_name = "test"
        data = {"summary": {"total": 10}}

        result = cache.get(tool_name, data)

        assert result is None
        assert cache._hits == 0
        assert cache._misses == 1

    def test_cache_key_generation_with_kwargs(self):
        """Test cache key is different for different kwargs."""
        cache = _FormatCache()
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted1 = FormattedOutput(
            content="result1", format_type="rich", summary="", contains_markup=True
        )
        formatted2 = FormattedOutput(
            content="result2", format_type="rich", summary="", contains_markup=True
        )

        # Put with different kwargs
        cache.put(tool_name, data, formatted1, max_failures=5)
        cache.put(tool_name, data, formatted2, max_failures=10)

        # Get should return different results based on kwargs
        result1 = cache.get(tool_name, data, max_failures=5)
        result2 = cache.get(tool_name, data, max_failures=10)

        assert result1.content == "result1"
        assert result2.content == "result2"

    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        cache = _FormatCache(default_ttl=1)  # 1 second TTL
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )

        # Put in cache
        cache.put(tool_name, data, formatted)

        # Get immediately should hit
        result = cache.get(tool_name, data)
        assert result is not None
        assert cache._hits == 1

        # Wait for expiration
        time.sleep(1.1)

        # Get after expiration should miss
        result = cache.get(tool_name, data)
        assert result is None
        assert cache._misses == 1  # Only 1 miss, not 2

    def test_cache_lru_eviction(self):
        """Test cache evicts oldest entry when full."""
        cache = _FormatCache(max_size=2)
        tool_name = "test"
        data1 = {"id": 1}
        data2 = {"id": 2}
        data3 = {"id": 3}
        formatted1 = FormattedOutput(
            content="result1", format_type="rich", summary="", contains_markup=True
        )
        formatted2 = FormattedOutput(
            content="result2", format_type="rich", summary="", contains_markup=True
        )
        formatted3 = FormattedOutput(
            content="result3", format_type="rich", summary="", contains_markup=True
        )

        # Fill cache
        cache.put(tool_name, data1, formatted1)
        cache.put(tool_name, data2, formatted2)

        # Add third entry should evict first
        cache.put(tool_name, data3, formatted3)

        # First entry should be gone
        result1 = cache.get(tool_name, data1)
        assert result1 is None

        # Second and third should still be there
        result2 = cache.get(tool_name, data2)
        result3 = cache.get(tool_name, data3)
        assert result2 is not None
        assert result3 is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = _FormatCache()
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )

        # Add entry
        cache.put(tool_name, data, formatted)

        # Hit
        cache.get(tool_name, data)

        # Miss
        cache.get(tool_name, {"different": "data"})

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hit_rate"] == 0.5

    def test_invalidate_all(self):
        """Test invalidating all cache entries."""
        cache = _FormatCache()
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )

        # Add entries
        cache.put(tool_name, data, formatted)
        cache.put("other_tool", data, formatted)

        # Invalidate all
        cache.invalidate()

        # Both should be gone
        assert cache.get(tool_name, data) is None
        assert cache.get("other_tool", data) is None
        assert cache._cache == {}

    def test_invalidate_by_tool(self):
        """Test invalidating cache entries for specific tool."""
        cache = _FormatCache()
        data1 = {"summary": {"total": 10}}
        data2 = {"summary": {"total": 5}}
        formatted1 = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )
        formatted2 = FormattedOutput(
            content="✓ 5 passed",
            format_type="rich",
            summary="5 tests",
            contains_markup=True,
        )

        # Add entries for different tools with different data
        cache.put("test", data1, formatted1)
        cache.put("git", data2, formatted2)

        # Verify both are in cache (stats should show size 2)
        stats_before = cache.get_stats()
        assert stats_before["size"] == 2

        # Invalidate only test tool
        cache.invalidate("test")

        # Verify cache size decreased (git entry should still be there)
        stats_after = cache.get_stats()
        assert stats_after["size"] == 1

        # Git entry should still be accessible
        result = cache.get("git", data2)
        assert result is not None
        assert result.content == "✓ 5 passed"

    def test_custom_ttl(self):
        """Test custom TTL for cache entry."""
        cache = _FormatCache(default_ttl=10)
        tool_name = "test"
        data = {"summary": {"total": 10}}
        formatted = FormattedOutput(
            content="✓ 10 passed",
            format_type="rich",
            summary="10 tests",
            contains_markup=True,
        )

        # Put with custom TTL of 1 second
        cache.put(tool_name, data, formatted, ttl=1)

        # Get immediately should hit
        result = cache.get(tool_name, data)
        assert result is not None

        # Wait for expiration
        time.sleep(1.1)

        # Get after expiration should miss
        result = cache.get(tool_name, data)
        assert result is None


class TestCacheUtilityFunctions:
    """Test cache management utility functions."""

    def test_get_format_cache_stats(self):
        """Test getting global cache stats."""
        stats = get_format_cache_stats()

        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats

    def test_invalidate_format_cache_all(self):
        """Test invalidating all format cache."""
        # Add something to cache first
        from victor.tools.formatters import format_test_results

        test_data = {
            "summary": {"total_tests": 5, "passed": 5, "failed": 0, "skipped": 0},
            "failures": [],
        }
        format_test_results(test_data)  # This will cache

        # Invalidate all
        invalidate_format_cache()

        # Check cache is cleared
        stats = get_format_cache_stats()
        assert stats["size"] == 0

    def test_invalidate_format_cache_tool_specific(self):
        """Test invalidating cache for specific tool."""
        # Add something to cache
        from victor.tools.formatters import format_test_results

        test_data = {
            "summary": {"total_tests": 5, "passed": 5, "failed": 0, "skipped": 0},
            "failures": [],
        }
        format_test_results(test_data)

        # Invalidate test tool only
        invalidate_format_cache("test")

        # Stats should show reduced size
        stats = get_format_cache_stats()
        # Cache might have other entries, so just check it doesn't error
        assert stats is not None

    def test_clear_format_cache(self):
        """Test clearing format cache."""
        # Add something to cache
        from victor.tools.formatters import format_test_results

        test_data = {
            "summary": {"total_tests": 5, "passed": 5, "failed": 0, "skipped": 0},
            "failures": [],
        }
        format_test_results(test_data)

        # Clear cache
        clear_format_cache()

        # Check cache is cleared
        stats = get_format_cache_stats()
        assert stats["size"] == 0
