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

"""Tests for graph query cache (PH4-005)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag.query_cache import (
    GraphQueryCache,
    GraphQueryCacheConfig,
    get_graph_query_cache,
    configure_graph_query_cache,
    reset_graph_query_cache,
    _normalize_query,
    _create_query_cache_key,
)


@dataclass
class MockConfig:
    """Mock retrieval configuration."""

    seed_count: int = 5
    max_hops: int = 2
    top_k: int = 10
    edge_types: list | None = None


@dataclass
class MockNode:
    """Mock graph node."""

    node_id: str
    type: str
    name: str
    file: str
    line: int | None = None
    end_line: int | None = None
    lang: str | None = None


@dataclass
class MockEdge:
    """Mock graph edge."""

    src: str
    dst: str
    type: str


@dataclass
class MockRetrievalResult:
    """Mock retrieval result."""

    nodes: list
    edges: list
    subgraphs: list
    query: str
    seed_nodes: list = None
    hop_distances: dict = None
    scores: dict = None
    execution_time_ms: float = 100.0
    metadata: dict = None

    def __post_init__(self):
        if self.seed_nodes is None:
            self.seed_nodes = []
        if self.hop_distances is None:
            self.hop_distances = {}
        if self.scores is None:
            self.scores = {}
        if self.metadata is None:
            self.metadata = {}


class TestQueryNormalization:
    """Tests for query normalization."""

    def test_normalize_lowercase(self):
        """Test that normalization converts to lowercase."""
        assert _normalize_query("FIND Authentication") == "authentication"

    def test_normalize_whitespace(self):
        """Test that normalization removes extra whitespace."""
        # Note: "find" is also removed as a prefix
        assert _normalize_query("find   authentication") == "authentication"

    def test_normalize_remove_prefixes(self):
        """Test that common question prefixes are removed."""
        # "how do I find" -> "find authentication" (find is separate)
        assert _normalize_query("how do I find authentication") == "find authentication"
        # "what is the" is removed
        assert _normalize_query("what is the authentication") == "authentication"
        # "show me" is removed
        assert _normalize_query("show me authentication") == "authentication"

    def test_normalize_code_patterns(self):
        """Test normalization of code-related queries."""
        assert _normalize_query("search for user login") == "user login"
        assert _normalize_query("look for auth function") == "auth function"


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_key_with_query(self):
        """Test key generation with just a query."""
        config = MockConfig()
        key = _create_query_cache_key("find auth", config)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex length

    def test_key_different_queries(self):
        """Test that different queries produce different keys."""
        config = MockConfig()
        key1 = _create_query_cache_key("find auth", config)
        key2 = _create_query_cache_key("find user", config)
        assert key1 != key2

    def test_key_different_configs(self):
        """Test that different configs produce different keys."""
        config1 = MockConfig(max_hops=2)
        config2 = MockConfig(max_hops=3)
        key1 = _create_query_cache_key("find auth", config1)
        key2 = _create_query_cache_key("find auth", config2)
        assert key1 != key2

    def test_key_with_repo(self):
        """Test that repo path affects scoping."""
        config = MockConfig()
        key1 = _create_query_cache_key("find auth", config, "/path/to/repo1")
        key2 = _create_query_cache_key("find auth", config, "/path/to/repo2")
        assert key1 != key2

    def test_key_normalized_vs_not(self):
        """Test that normalization affects key generation."""
        config = MockConfig()
        key1 = _create_query_cache_key("How do I find auth", config, normalize=True)
        key2 = _create_query_cache_key("How do I find auth", config, normalize=False)
        # Normalized version should have "find auth" prefix stripped
        assert key1 != key2

    def test_key_same_normalized_queries(self):
        """Test that equivalent queries produce the same key when normalized."""
        config = MockConfig()
        # "how do i find auth" -> "find auth" (how do i prefix removed)
        # "find auth" -> "auth" (find prefix removed)
        # These should NOT be equal since they normalize differently
        key1 = _create_query_cache_key("How do I find auth", config, normalize=True)
        key2 = _create_query_cache_key("find auth", config, normalize=True)
        assert key1 != key2

        # But these should be equal (both normalize to "auth")
        key3 = _create_query_cache_key("find auth", config, normalize=True)
        key4 = _create_query_cache_key("show me auth", config, normalize=True)
        assert key3 == key4


class TestGraphQueryCache:
    """Tests for GraphQueryCache."""

    def test_cache_init_default_config(self):
        """Test cache initialization with default config."""
        cache = GraphQueryCache()
        assert cache._config.enabled is True
        assert cache._config.max_entries == 100
        assert cache._config.ttl_seconds == 3600
        assert cache._cache is not None

    def test_cache_init_custom_config(self):
        """Test cache initialization with custom config."""
        config = GraphQueryCacheConfig(
            enabled=True,
            max_entries=50,
            ttl_seconds=1800,
        )
        cache = GraphQueryCache(config)
        assert cache._config.max_entries == 50
        assert cache._config.ttl_seconds == 1800

    def test_cache_init_disabled(self):
        """Test cache initialization when disabled."""
        config = GraphQueryCacheConfig(enabled=False)
        cache = GraphQueryCache(config)
        assert cache._cache is None

    def test_cache_put_and_get(self):
        """Test putting and getting cached results."""
        cache = GraphQueryCache()
        config = MockConfig()

        # Create a mock result
        result = MockRetrievalResult(
            nodes=[MockNode("n1", "function", "auth", "auth.py", 10)],
            edges=[MockEdge("n1", "n2", "CALLS")],
            subgraphs=[],
            query="find auth",
        )

        # Put and get
        assert cache.put("find auth", config, result, "/path/to/repo") is True
        cached = cache.get("find auth", config, "/path/to/repo")

        assert cached is not None
        assert cached.query == "find auth"
        assert len(cached.nodes) == 1
        assert cached.nodes[0].name == "auth"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = cache.get("nonexistent query", config)
        assert result is None

    def test_cache_hit_increments_stats(self):
        """Test that cache hits increment stats."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test", config, result)
        cache.get("test", config)

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss_increments_stats(self):
        """Test that cache misses increment stats."""
        cache = GraphQueryCache()
        config = MockConfig()

        cache.get("nonexistent", config)

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_cache_put_increments_stats(self):
        """Test that puts increment stats."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test", config, result)

        stats = cache.get_stats()
        assert stats["puts"] == 1

    def test_cache_invalidate(self):
        """Test invalidating a specific entry."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test", config, result)
        assert cache.get("test", config) is not None

        assert cache.invalidate("test", config) is True
        assert cache.get("test", config) is None

    def test_cache_invalidate_nonexistent(self):
        """Test invalidating a non-existent entry returns False."""
        cache = GraphQueryCache()
        config = MockConfig()

        assert cache.invalidate("nonexistent", config) is False

    def test_cache_invalidate_repo(self):
        """Test invalidating all entries for a repository."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        # Add entries for different repos
        cache.put("test1", config, result, "/repo1")
        cache.put("test2", config, result, "/repo1")
        cache.put("test3", config, result, "/repo2")

        # Invalidate repo1
        count = cache.invalidate_repo("/repo1")
        assert count == 2

        # Verify repo1 entries are gone, repo2 remains
        assert cache.get("test1", config, "/repo1") is None
        assert cache.get("test2", config, "/repo1") is None
        assert cache.get("test3", config, "/repo2") is not None

    def test_cache_invalidate_all(self):
        """Test invalidating all entries."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test1", config, result)
        cache.put("test2", config, result)

        count = cache.invalidate_all()
        assert count == 2

        assert cache.get("test1", config) is None
        assert cache.get("test2", config) is None

    def test_cache_stats_hit_rate(self):
        """Test that hit rate is calculated correctly."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test", config, result)

        # 2 hits, 1 miss
        cache.get("test", config)
        cache.get("test", config)
        cache.get("nonexistent", config)

        stats = cache.get_stats()
        assert stats["hit_rate"] == 2 / 3

    def test_cache_stats_current_size(self):
        """Test that current size is tracked correctly."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        assert cache.get_stats()["current_size"] == 0

        cache.put("test1", config, result)
        assert cache.get_stats()["current_size"] == 1

        cache.put("test2", config, result)
        assert cache.get_stats()["current_size"] == 2

    def test_cache_stats_repo_count(self):
        """Test that repository count is tracked correctly."""
        cache = GraphQueryCache()
        config = MockConfig()

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        cache.put("test1", config, result, "/repo1")
        assert cache.get_stats()["repo_count"] == 1

        cache.put("test2", config, result, "/repo2")
        assert cache.get_stats()["repo_count"] == 2

    def test_cache_disabled_returns_none(self):
        """Test that disabled cache always returns None."""
        config = GraphQueryCacheConfig(enabled=False)
        cache = GraphQueryCache(config)

        result = MockRetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
        )

        assert cache.put("test", MockConfig(), result) is False
        assert cache.get("test", MockConfig()) is None
        assert cache.invalidate("test", MockConfig()) is False
        assert cache.invalidate_all() == 0


class TestGlobalCacheSingleton:
    """Tests for global cache singleton."""

    def test_get_singleton(self):
        """Test that get_graph_query_cache returns a singleton."""
        reset_graph_query_cache()

        cache1 = get_graph_query_cache()
        cache2 = get_graph_query_cache()

        assert cache1 is cache2

    def test_configure_singleton(self):
        """Test that configure_graph_query_cache replaces the singleton."""
        reset_graph_query_cache()

        cache1 = get_graph_query_cache()
        assert cache1._config.max_entries == 100

        config = GraphQueryCacheConfig(max_entries=50)
        configure_graph_query_cache(config)

        cache2 = get_graph_query_cache()
        assert cache2._config.max_entries == 50
        assert cache1 is not cache2

    def test_reset_singleton(self):
        """Test that reset_graph_query_cache clears the singleton."""
        reset_graph_query_cache()

        cache1 = get_graph_query_cache()
        reset_graph_query_cache()

        cache2 = get_graph_query_cache()
        assert cache1 is not cache2


class TestRepoScoping:
    """Tests for full-path repo scoping of cache keys and invalidation."""

    def test_key_same_basename_different_repos_differ(self):
        """Repos sharing a basename must produce distinct cache keys."""
        config = MockConfig()
        key_a = _create_query_cache_key("find auth", config, "/home/a/project")
        key_b = _create_query_cache_key("find auth", config, "/home/b/project")
        assert key_a != key_b

    def test_invalidate_repo_scoped_to_full_path(self):
        """Invalidating one repo must not evict a same-named sibling repo."""
        cache = GraphQueryCache()
        config = MockConfig()
        result = MockRetrievalResult(nodes=[], edges=[], subgraphs=[], query="test")

        cache.put("q", config, result, "/home/a/project")
        cache.put("q", config, result, "/home/b/project")

        count = cache.invalidate_repo("/home/a/project")
        assert count == 1
        assert cache.get("q", config, "/home/a/project") is None
        assert cache.get("q", config, "/home/b/project") is not None

    def test_retriever_repo_path_prefers_repo_root(self):
        """MultiHopRetriever should scope the cache via the public repo_root."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        graph_store = MagicMock()
        graph_store.repo_root = Path("/test/repo")
        retriever = MultiHopRetriever(graph_store, MockConfig())
        assert retriever._cache_repo_path() == str(Path("/test/repo"))

    def test_retriever_repo_path_falls_back_to_root_path(self):
        """Stores without repo_root fall back to the legacy _root_path attribute."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        graph_store = MagicMock(spec=[])
        graph_store._root_path = Path("/legacy/repo")
        retriever = MultiHopRetriever(graph_store, MockConfig())
        assert retriever._cache_repo_path() == str(Path("/legacy/repo"))

    def test_retriever_repo_path_none_when_unavailable(self):
        """No scoping attribute at all yields None (unscoped cache entries)."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        graph_store = MagicMock(spec=[])
        retriever = MultiHopRetriever(graph_store, MockConfig())
        assert retriever._cache_repo_path() is None


class TestCacheIntegration:
    """Integration tests for cache with MultiHopRetriever."""

    @pytest.mark.asyncio
    async def test_cache_with_retriever_integration(self):
        """Test that cache works with MultiHopRetriever."""
        from victor.core.graph_rag.query_cache import get_graph_query_cache
        from victor.core.graph_rag.retrieval import RetrievalResult

        reset_graph_query_cache()
        cache = get_graph_query_cache()

        # Mock graph store
        graph_store = MagicMock()
        graph_store._root_path = Path("/test/repo")

        # Mock config
        config = MockConfig(max_hops=2, seed_count=5, top_k=10)

        # Create a mock result
        result = RetrievalResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="find auth",
        )

        # Cache and retrieve
        cache.put("find auth", config, result, "/test/repo")
        cached = cache.get("find auth", config, "/test/repo")

        assert cached is not None
        assert cached.query == "find auth"

    def test_cache_serialization_preserves_data(self):
        """Test that serialization/deserialization preserves result data."""
        cache = GraphQueryCache()
        config = MockConfig()

        # Create a result with various data
        result = MockRetrievalResult(
            nodes=[
                MockNode("n1", "function", "auth", "auth.py", 10, 20, "python"),
                MockNode("n2", "class", "User", "user.py", 5, 100, "python"),
            ],
            edges=[
                MockEdge("n1", "n2", "CALLS"),
                MockEdge("n2", "n1", "REFERENCES"),
            ],
            subgraphs=[],
            query="find auth",
            seed_nodes=["n1"],
            hop_distances={"n1": 0, "n2": 1},
            scores={"n1": 1.0, "n2": 0.7},
            execution_time_ms=150.5,
            metadata={"cache_hit": False},
        )

        # Put and get
        cache.put("test", config, result)
        cached = cache.get("test", config)

        assert cached is not None
        assert cached.query == "find auth"
        assert len(cached.nodes) == 2
        assert cached.nodes[0].name == "auth"
        assert cached.nodes[1].name == "User"
        assert len(cached.edges) == 2
        assert cached.seed_nodes == ["n1"]
        assert cached.hop_distances == {"n1": 0, "n2": 1}
        assert cached.scores == {"n1": 1.0, "n2": 0.7}
        assert cached.execution_time_ms == 150.5
        assert cached.metadata == {"cache_hit": False}
