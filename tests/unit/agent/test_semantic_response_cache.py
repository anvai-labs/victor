"""Tests for semantic response cache."""

import sys

import pytest
import time
import numpy as np
from victor.agent.semantic_response_cache import (
    SemanticResponseCache,
    CachedResponse,
    get_semantic_cache,
)


class TestSemanticResponseCache:
    """Test semantic response cache functionality."""

    @pytest.fixture(autouse=True)
    def _deterministic_embeddings(self, monkeypatch):
        """Make the cache hermetic by replacing the real embedding model.

        ``SemanticResponseCache._embed`` lazily loads a real
        ``SentenceTransformer("BAAI/bge-small-en-v1.5")`` — a network/HuggingFace
        model download. On CI that download is intermittently slow/unavailable;
        when it fails, ``set()`` silently skips storing the entry, which made the
        storage/stats/eviction assertions fail non-deterministically. These tests
        only need identical text -> identical vector (a hit) and different text ->
        a near-orthogonal vector (a miss), not real semantics, so a deterministic
        offline embedding removes the flakiness without changing behavior.
        """
        import hashlib

        def _fake_embed(_self, text: str) -> np.ndarray:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
            vec = np.random.RandomState(seed).randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm else vec

        monkeypatch.setattr(SemanticResponseCache, "_embed", _fake_embed)

    def test_initialization(self):
        """Test cache initialization with defaults."""
        cache = SemanticResponseCache()
        assert cache.similarity_threshold == 0.92
        assert cache.ttl_seconds == 24 * 3600
        assert cache.max_entries == 1000

    def test_initialization_custom(self):
        """Test cache initialization with custom parameters."""
        cache = SemanticResponseCache(
            similarity_threshold=0.95,
            ttl_hours=12.0,
            max_entries=500,
        )
        assert cache.similarity_threshold == 0.95
        assert cache.ttl_seconds == 12 * 3600
        assert cache.max_entries == 500

    def test_cache_miss_initially(self):
        """Test that cache returns None for non-existent queries."""
        cache = SemanticResponseCache()
        result = cache.get("How do I search code?")
        assert result is None

    def test_cache_set_and_get(self):
        """Test caching and retrieving responses."""
        cache = SemanticResponseCache()

        # Cache a response
        cache.set(
            query="How do I search code?",
            response="Use the code_search tool with semantic mode.",
        )

        # Retrieve it
        result = cache.get("How do I search code?")
        assert result is not None
        assert result["response"] == "Use the code_search tool with semantic mode."
        assert "similarity" in result
        assert result["similarity"] >= 0.92

    def test_cache_similarity_threshold(self):
        """Test that similar queries hit the cache."""
        cache = SemanticResponseCache(similarity_threshold=0.90)

        cache.set(
            query="How to search Python files?",
            response="Use code_search with ext='.py'",
        )

        # Similar query should hit cache
        # Note: This depends on sentence-transformers being installed
        # and embeddings being sufficiently similar
        result = cache.get("How to search Python files?")  # Exact match
        assert result is not None
        assert result["similarity"] >= 0.90

    def test_cache_miss_dissimilar_query(self):
        """Test that dissimilar queries miss the cache."""
        cache = SemanticResponseCache()

        cache.set(
            query="How do I search code?",
            response="Use code_search tool",
        )

        # Completely different query should miss
        result = cache.get("What is the weather today?")
        assert result is None

    def test_cache_with_metadata(self):
        """Test caching with metadata."""
        cache = SemanticResponseCache()

        metadata = {"model": "gpt-4", "tokens": 150}
        cache.set(
            query="Test query",
            response="Test response",
            metadata=metadata,
        )

        result = cache.get("Test query")
        assert result is not None
        assert result["metadata"] == metadata

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = SemanticResponseCache()

        # No queries yet
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Add entry
        cache.set(query="Test", response="Response")

        # Cache hit
        cache.get("Test")

        # Cache miss
        cache.get("Different query")

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_max_entries_eviction(self):
        """Test LRU eviction when max entries exceeded."""
        cache = SemanticResponseCache(max_entries=2)

        cache.set(query="A", response="Response A")
        cache.set(query="B", response="Response B")
        cache.set(query="C", response="Response C")  # Should evict A or B

        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["evictions"] == 1

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = SemanticResponseCache()

        cache.set(query="Test", response="Response")
        assert cache.get_stats()["entries"] == 1

        cache.clear()
        assert cache.get_stats()["entries"] == 0

    def test_global_singleton(self):
        """Test global cache singleton."""
        cache1 = get_semantic_cache()
        cache2 = get_semantic_cache()

        assert cache1 is cache2

    def test_cached_response_is_expired(self):
        """Test cache entry expiration."""
        cached = CachedResponse(
            response="Test",
            embedding=np.array([1.0, 0.0]),
            timestamp=time.time() - 100,  # 100 seconds ago
            ttl=50,  # 50 second TTL
        )
        assert cached.is_expired()

        cached.ttl = 200  # 200 second TTL
        assert not cached.is_expired()

    def test_cached_response_similarity_score(self):
        """Test similarity score calculation."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])  # Identical

        cached = CachedResponse(
            response="Test",
            embedding=embedding1,
            timestamp=time.time(),
            ttl=3600,
        )

        # Identical embeddings should have similarity 1.0
        score = cached.similarity_score(embedding2)
        assert score == pytest.approx(1.0, rel=1e-5)

        # Orthogonal embeddings should have similarity 0.0
        embedding3 = np.array([0.0, 1.0, 0.0])
        score = cached.similarity_score(embedding3)
        assert score == pytest.approx(0.0, abs=1e-5)


def test_module_import_survives_without_numpy(monkeypatch):
    """Bare-install simulation: reloading the module with numpy poisoned must
    take the ImportError guard branch (NUMPY_AVAILABLE False, np None)."""
    import importlib

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "victor.agent.semantic_response_cache", raising=False)
    reloaded = importlib.import_module("victor.agent.semantic_response_cache")
    try:
        assert reloaded.NUMPY_AVAILABLE is False
        assert reloaded.np is None
    finally:
        monkeypatch.undo()
        importlib.reload(importlib.import_module("victor.agent.semantic_response_cache"))
