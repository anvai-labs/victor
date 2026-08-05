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

"""Tests for the unified CacheManager."""

import pytest
import tempfile
from pathlib import Path

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.manager import (
    CacheManager,
    CacheNamespace,
    CacheStats,
    get_cache_manager,
    set_cache_manager,
    reset_cache_manager,
    get_tools_cache,
    get_embeddings_cache,
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_config(temp_cache_dir):
    """Create cache config with temp directory."""
    return CacheConfig(
        disk_path=temp_cache_dir / "cache",
        enable_disk=True,
        enable_memory=True,
    )


@pytest.fixture
def cache_manager(cache_config):
    """Create a cache manager for testing."""
    manager = CacheManager(cache_config)
    yield manager
    manager.close()


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(memory_hits=10, disk_hits=5)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits/misses."""
        stats = CacheStats(
            memory_hits=5,
            memory_misses=5,
            disk_hits=5,
            disk_misses=5,
        )
        assert stats.hit_rate == 0.5

    def test_memory_hit_rate(self):
        """Test memory-specific hit rate."""
        stats = CacheStats(memory_hits=8, memory_misses=2)
        assert stats.memory_hit_rate == 0.8


class TestCacheNamespace:
    """Tests for CacheNamespace class."""

    def test_namespace_set_get(self, cache_manager):
        """Test setting and getting within namespace."""
        ns = cache_manager.namespace("test")

        ns.set("key1", "value1")
        result = ns.get("key1")

        assert result == "value1"

    def test_namespace_isolation(self, cache_manager):
        """Test that namespaces are isolated."""
        ns1 = cache_manager.namespace("ns1")
        ns2 = cache_manager.namespace("ns2")

        ns1.set("key", "value1")
        ns2.set("key", "value2")

        assert ns1.get("key") == "value1"
        assert ns2.get("key") == "value2"

    def test_namespace_delete(self, cache_manager):
        """Test deleting from namespace."""
        ns = cache_manager.namespace("test")

        ns.set("key", "value")
        assert ns.get("key") == "value"

        ns.delete("key")
        assert ns.get("key") is None

    def test_namespace_clear(self, cache_manager):
        """Test clearing a namespace."""
        ns = cache_manager.namespace("test")

        ns.set("key1", "value1")
        ns.set("key2", "value2")

        ns.clear()

        assert ns.get("key1") is None
        assert ns.get("key2") is None

    def test_namespace_get_or_set(self, cache_manager):
        """Test get_or_set helper."""
        ns = cache_manager.namespace("test")
        compute_count = [0]

        def factory():
            compute_count[0] += 1
            return "computed_value"

        # First call computes
        result1 = ns.get_or_set("key", factory)
        assert result1 == "computed_value"
        assert compute_count[0] == 1

        # Second call uses cache
        result2 = ns.get_or_set("key", factory)
        assert result2 == "computed_value"
        assert compute_count[0] == 1  # Not called again

    def test_namespace_property(self, cache_manager):
        """Test namespace property."""
        ns = cache_manager.namespace("my_namespace")
        assert ns.namespace == "my_namespace"


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_basic_set_get(self, cache_manager):
        """Test basic set/get operations."""
        cache_manager.set("key", "value")
        result = cache_manager.get("key")

        assert result == "value"

    def test_set_get_with_namespace(self, cache_manager):
        """Test set/get with explicit namespace."""
        cache_manager.set("key", "value", namespace="custom")
        result = cache_manager.get("key", namespace="custom")

        assert result == "value"

    def test_delete(self, cache_manager):
        """Test delete operation."""
        cache_manager.set("key", "value")
        cache_manager.delete("key")

        assert cache_manager.get("key") is None

    def test_clear_namespace(self, cache_manager):
        """Test clearing specific namespace."""
        cache_manager.set("key1", "value1", namespace="ns1")
        cache_manager.set("key2", "value2", namespace="ns1")
        cache_manager.set("key3", "value3", namespace="ns2")

        cache_manager.clear_namespace("ns1")

        assert cache_manager.get("key1", namespace="ns1") is None
        assert cache_manager.get("key2", namespace="ns1") is None
        assert cache_manager.get("key3", namespace="ns2") == "value3"

    def test_clear_all(self, cache_manager):
        """Test clearing all cache."""
        cache_manager.set("key1", "value1", namespace="ns1")
        cache_manager.set("key2", "value2", namespace="ns2")

        cache_manager.clear_all()

        assert cache_manager.get("key1", namespace="ns1") is None
        assert cache_manager.get("key2", namespace="ns2") is None

    def test_get_stats(self, cache_manager):
        """Test getting cache statistics."""
        cache_manager.set("key", "value")
        cache_manager.get("key")  # Hit
        cache_manager.get("missing")  # Miss

        stats = cache_manager.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.total_sets >= 1

    def test_namespace_returns_same_instance(self, cache_manager):
        """Test that namespace() returns same instance for same name."""
        ns1 = cache_manager.namespace("test")
        ns2 = cache_manager.namespace("test")

        assert ns1 is ns2

    def test_config_property(self, cache_config, cache_manager):
        """Test config property."""
        assert cache_manager.config is cache_config

    def test_well_known_namespaces(self, cache_manager):
        """Test well-known namespace constants."""
        assert CacheManager.NAMESPACE_TOOLS == "tools"
        assert CacheManager.NAMESPACE_EMBEDDINGS == "embeddings"
        assert CacheManager.NAMESPACE_RESPONSES == "responses"
        assert CacheManager.NAMESPACE_CODE_SEARCH == "code_search"
        assert CacheManager.NAMESPACE_METADATA == "metadata"


class TestGlobalCacheManager:
    """Tests for global cache manager functions."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_cache_manager()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_cache_manager()

    def test_get_cache_manager(self, temp_cache_dir):
        """Test getting global cache manager."""
        config = CacheConfig(disk_path=temp_cache_dir / "cache")
        manager = get_cache_manager(config)

        assert isinstance(manager, CacheManager)

    def test_get_cache_manager_same_instance(self, temp_cache_dir):
        """Test that get_cache_manager returns same instance."""
        config = CacheConfig(disk_path=temp_cache_dir / "cache")
        manager1 = get_cache_manager(config)
        manager2 = get_cache_manager()

        assert manager1 is manager2

    def test_set_cache_manager(self, cache_config):
        """Test setting global cache manager."""
        manager = CacheManager(cache_config)
        set_cache_manager(manager)

        assert get_cache_manager() is manager

    def test_reset_cache_manager(self, temp_cache_dir):
        """Test resetting global cache manager."""
        config = CacheConfig(disk_path=temp_cache_dir / "cache")
        manager1 = get_cache_manager(config)
        reset_cache_manager()
        manager2 = get_cache_manager(config)

        assert manager1 is not manager2


class TestConvenienceFunctions:
    """Tests for convenience cache accessor functions."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_cache_manager()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_cache_manager()

    def test_get_tools_cache(self, temp_cache_dir):
        """Test get_tools_cache convenience function."""
        config = CacheConfig(disk_path=temp_cache_dir / "cache")
        get_cache_manager(config)

        cache = get_tools_cache()

        assert isinstance(cache, CacheNamespace)
        assert cache.namespace == "tools"

    def test_get_embeddings_cache(self, temp_cache_dir):
        """Test get_embeddings_cache convenience function."""
        config = CacheConfig(disk_path=temp_cache_dir / "cache")
        get_cache_manager(config)

        cache = get_embeddings_cache()

        assert isinstance(cache, CacheNamespace)
        assert cache.namespace == "embeddings"
