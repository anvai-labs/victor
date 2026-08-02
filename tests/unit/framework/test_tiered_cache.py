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

"""Tests for the TieredCache module.

Tests L1 (memory) and L2 (disk) caching, statistics,
and specialized caches (ResponseCache, EmbeddingCache).
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import (
    TieredCache,
    ResponseCache,
    EmbeddingCache,
)
from victor.storage.cache.manager import CacheManager

# ============================================================================
# TieredCache Tests
# ============================================================================


class TestTieredCacheBasic:
    """Basic TieredCache tests."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a tiered cache with temp directory."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=True,
            memory_max_size=100,
            memory_ttl=60,
            disk_path=temp_cache_dir,
            disk_ttl=300,
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self, cache):
        """Test get for nonexistent key."""
        assert cache.get("nonexistent") is None

    def test_namespaced_keys(self, cache):
        """Test namespace isolation."""
        cache.set("key", "value1", namespace="ns1")
        cache.set("key", "value2", namespace="ns2")

        assert cache.get("key", namespace="ns1") == "value1"
        assert cache.get("key", namespace="ns2") == "value2"

    def test_delete(self, cache):
        """Test delete operation."""
        cache.set("key", "value")
        assert cache.get("key") == "value"

        cache.delete("key")
        assert cache.get("key") is None

    def test_delete_nonexistent(self, cache):
        """Test delete of nonexistent key."""
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear_all(self, cache):
        """Test clearing all entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        count = cache.clear()
        assert count >= 3

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_clear_namespace(self, cache):
        """Test clearing specific namespace."""
        cache.set("key1", "value1", namespace="ns1")
        cache.set("key2", "value2", namespace="ns1")
        cache.set("key3", "value3", namespace="ns2")

        cache.clear(namespace="ns1")

        assert cache.get("key1", namespace="ns1") is None
        assert cache.get("key2", namespace="ns1") is None
        assert cache.get("key3", namespace="ns2") == "value3"


class TestTieredCacheStatistics:
    """Tests for cache statistics."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a tiered cache."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=True,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_initial_stats(self, cache):
        """Test initial statistics are zero."""
        stats = cache.get_stats()
        assert stats["memory_hits"] == 0
        assert stats["memory_misses"] == 0
        assert stats["disk_hits"] == 0
        assert stats["disk_misses"] == 0
        assert stats["sets"] == 0

    def test_stats_after_operations(self, cache):
        """Test statistics after cache operations."""
        # Set
        cache.set("key1", "value1")
        stats = cache.get_stats()
        assert stats["sets"] == 1

        # Memory hit
        cache.get("key1")
        stats = cache.get_stats()
        assert stats["memory_hits"] == 1

        # Miss
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["memory_misses"] >= 1

    def test_hit_rate_calculation(self, cache):
        """Test hit rate calculation."""
        # Create some hits and misses
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        # 2 hits out of 3 requests (2 hits + 1 miss)
        assert stats["memory_hit_rate"] > 0


class TestTieredCacheTiering:
    """Tests for L1/L2 tiering behavior."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a tiered cache."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=True,
            memory_max_size=10,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_promotes_from_disk_to_memory(self, cache):
        """Test that disk cache hits are promoted to memory."""
        # Set in cache (goes to both L1 and L2)
        cache.set("key1", "value1")

        # Clear memory cache only (simulate memory eviction)
        cache._memory_cache.clear()

        # Get should find in disk and promote to memory
        result = cache.get("key1")
        assert result == "value1"

        # Now should be in memory
        assert "default:key1" in cache._memory_cache

    def test_stores_in_both_tiers(self, cache):
        """Test that set stores in both L1 and L2."""
        cache.set("key1", "value1")

        # Check both caches
        assert cache._memory_cache.get("default:key1") == "value1"
        assert cache._disk_cache.get("default:key1") == "value1"


class TestTieredCacheMemoryOnly:
    """Tests for memory-only cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a memory-only cache."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=False,
        )
        return TieredCache(config)

    def test_memory_only(self, cache):
        """Test memory-only cache."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache._disk_cache is None


class TestTieredCacheDiskOnly:
    """Tests for disk-only cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a disk-only cache."""
        config = CacheConfig(
            enable_memory=False,
            enable_disk=True,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_disk_only(self, cache):
        """Test disk-only cache."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache._memory_cache is None


class TestTieredCacheWarmup:
    """Tests for cache warmup."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a tiered cache."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=True,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_warmup(self, cache):
        """Test cache warmup with data."""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        count = cache.warmup(data)
        assert count == 3

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"


class TestTieredCacheContextManager:
    """Tests for context manager behavior."""

    def test_context_manager(self, tmp_path):
        """Test cache as context manager."""
        config = CacheConfig(
            enable_memory=True,
            enable_disk=True,
            disk_path=tmp_path / "cache",
        )

        with TieredCache(config) as cache:
            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"


class TestTieredCacheLongKeys:
    """Tests for long key handling."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a tiered cache."""
        config = CacheConfig(
            enable_memory=True,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_long_key_hashing(self, cache):
        """Test that long keys are hashed."""
        long_key = "a" * 300  # > 200 chars
        cache.set(long_key, "value1")

        # Should still be retrievable
        assert cache.get(long_key) == "value1"


class TestTieredCacheThreadSafety:
    """Tests for thread safety."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a tiered cache."""
        config = CacheConfig(
            enable_memory=True,
            disk_path=tmp_path / "cache",
        )
        cache = TieredCache(config)
        yield cache
        cache.close()

    def test_concurrent_operations(self, cache):
        """Test concurrent cache operations."""
        import threading

        def write_values(start, count):
            for i in range(start, start + count):
                cache.set(f"key{i}", f"value{i}")

        def read_values(start, count):
            for i in range(start, start + count):
                cache.get(f"key{i}")

        # Write from multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_values, args=(i * 10, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Read from multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=read_values, args=(i * 10, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all values
        for i in range(50):
            assert cache.get(f"key{i}") == f"value{i}"


# ============================================================================
# ResponseCache Tests
# ============================================================================


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a response cache."""
        tiered = TieredCache(CacheConfig(disk_path=tmp_path / "cache"))
        yield ResponseCache(tiered)
        tiered.close()

    def test_cache_response(self, cache):
        """Test caching a response."""
        prompt = "What is 2+2?"
        model = "gpt-4"
        temperature = 0.7
        response = "4"

        cache.cache_response(prompt, model, temperature, response)
        result = cache.get_response(prompt, model, temperature)
        assert result == response

    def test_different_prompts(self, cache):
        """Test different prompts are cached separately."""
        cache.cache_response("Prompt 1", "gpt-4", 0.7, "Response 1")
        cache.cache_response("Prompt 2", "gpt-4", 0.7, "Response 2")

        assert cache.get_response("Prompt 1", "gpt-4", 0.7) == "Response 1"
        assert cache.get_response("Prompt 2", "gpt-4", 0.7) == "Response 2"

    def test_different_models(self, cache):
        """Test different models are cached separately."""
        prompt = "What is 2+2?"
        cache.cache_response(prompt, "gpt-4", 0.7, "GPT-4 Response")
        cache.cache_response(prompt, "claude-3", 0.7, "Claude Response")

        assert cache.get_response(prompt, "gpt-4", 0.7) == "GPT-4 Response"
        assert cache.get_response(prompt, "claude-3", 0.7) == "Claude Response"

    def test_different_temperatures(self, cache):
        """Test different temperatures are cached separately."""
        prompt = "What is 2+2?"
        cache.cache_response(prompt, "gpt-4", 0.0, "Deterministic")
        cache.cache_response(prompt, "gpt-4", 1.0, "Creative")

        assert cache.get_response(prompt, "gpt-4", 0.0) == "Deterministic"
        assert cache.get_response(prompt, "gpt-4", 1.0) == "Creative"


# ============================================================================
# EmbeddingCache Tests
# ============================================================================


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create an embedding cache."""
        tiered = TieredCache(CacheConfig(disk_path=tmp_path / "cache"))
        yield EmbeddingCache(tiered)
        tiered.close()

    def test_cache_embedding(self, cache):
        """Test caching an embedding."""
        text = "Hello world"
        model = "text-embedding-ada-002"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        cache.cache_embedding(text, model, embedding)
        result = cache.get_embedding(text, model)
        assert result == embedding

    def test_different_texts(self, cache):
        """Test different texts are cached separately."""
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        cache.cache_embedding("Text 1", "model", embedding1)
        cache.cache_embedding("Text 2", "model", embedding2)

        assert cache.get_embedding("Text 1", "model") == embedding1
        assert cache.get_embedding("Text 2", "model") == embedding2

    def test_different_models(self, cache):
        """Test different models are cached separately."""
        text = "Hello world"
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        cache.cache_embedding(text, "model1", embedding1)
        cache.cache_embedding(text, "model2", embedding2)

        assert cache.get_embedding(text, "model1") == embedding1
        assert cache.get_embedding(text, "model2") == embedding2


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_cache_manager_usage(self, tmp_path):
        """Test CacheManager provides namespace-based caching."""
        config = CacheConfig(disk_path=tmp_path / "cache")
        cache = CacheManager(config)
        ns = cache.namespace("test")
        ns.set("key", "value")
        assert ns.get("key") == "value"
