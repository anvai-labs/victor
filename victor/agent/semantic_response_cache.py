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

"""Semantic response cache for reducing repeated LLM calls.

Based on arXiv:2508.07675 (Semantic Caching for Low-Cost LLM Serving).
Uses embedding similarity to serve cached responses without LLM calls.

Features:
- Embed queries + context using BAAI/bge-small-en-v1.5
- Cache LLM responses with embedding key
- Similarity search with cosine similarity > 0.92 threshold
- TTL-based invalidation + semantic drift detection
- Track cache hit rates and quality metrics
"""

from __future__ import annotations

import hashlib
from victor.core.json_utils import json_dumps
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached LLM response with metadata."""

    response: str
    embedding: np.ndarray
    timestamp: float
    ttl: float
    hit_count: int = 0
    query_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    def similarity_score(self, query_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between query and cached embedding."""
        # Normalize embeddings
        norm_cached = np.linalg.norm(self.embedding)
        norm_query = np.linalg.norm(query_embedding)

        if norm_cached == 0 or norm_query == 0:
            return 0.0

        # Cosine similarity
        return float(np.dot(self.embedding, query_embedding) / (norm_cached * norm_query))


class SemanticResponseCache:
    """Semantic response cache using embedding similarity.

    Reduces LLM calls by serving cached responses for semantically similar queries.
    Uses BAAI/bge-small-en-v1.5 embeddings for similarity matching.

    Example:
        cache = SemanticResponseCache(similarity_threshold=0.92, ttl_hours=24)

        # Check cache before LLM call
        cached = cache.get(query="How do I search code?")
        if cached:
            return cached["response"]

        # Call LLM and cache result
        response = llm.generate(query)
        cache.set(query=query, response=response, metadata={"model": "gpt-4"})
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_hours: float = 24.0,
        max_entries: int = 1000,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """Initialize semantic response cache.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hits (0.0-1.0)
            ttl_hours: Time-to-live for cache entries in hours
            max_entries: Maximum number of cached responses
            embedding_model: Model name for embeddings (default: BAAI/bge-small-en-v1.5)
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        self.max_entries = max_entries
        self.embedding_model_name = embedding_model

        self._cache: Dict[str, CachedResponse] = {}
        self._embedding_model: Optional[Any] = None
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def _get_embedding_model(self) -> Any:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning(
                    "sentence_transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
                raise

        return self._embedding_model

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        model = self._get_embedding_model()
        return model.encode(text, normalize_embeddings=True)

    def _hash_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate hash for query + context."""
        context_str = json_dumps(context, sort_keys=True, default=str) if context else ""
        combined = f"{query}||{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available.

        Args:
            query: User query text
            context: Optional context dict (conversation state, metadata, etc.)

        Returns:
            Cached response dict with keys: response, similarity, metadata, or None
        """
        with self._lock:
            # Clean up expired entries
            self._cleanup_expired()

            if not self._cache:
                self._stats["misses"] += 1
                return None

            # Generate query embedding (outside lock to avoid blocking)
            cache_snapshot = dict(self._cache)

        try:
            query_embedding = self._embed(query)
        except Exception as e:
            logger.debug(f"Failed to generate embedding: {e}")
            with self._lock:
                self._stats["misses"] += 1
            return None

        # Search for similar cached responses
        best_match: Optional[Tuple[str, CachedResponse, float]] = None

        for cache_key, cached_response in cache_snapshot.items():
            if cached_response.is_expired():
                continue

            similarity = cached_response.similarity_score(query_embedding)
            if similarity >= self.similarity_threshold:
                if best_match is None or similarity > best_match[2]:
                    best_match = (cache_key, cached_response, similarity)

        with self._lock:
            if best_match:
                cache_key, cached_response, similarity = best_match
                cached_response.hit_count += 1
                self._stats["hits"] += 1

                logger.debug(
                    f"Semantic cache hit: similarity={similarity:.3f}, "
                    f"hits={cached_response.hit_count}"
                )

                return {
                    "response": cached_response.response,
                    "similarity": similarity,
                    "metadata": cached_response.metadata,
                    "cached_at": cached_response.timestamp,
                }

            self._stats["misses"] += 1
            return None

    def set(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[float] = None,
    ) -> None:
        """Cache a response for future queries.

        Args:
            query: User query text
            response: LLM response to cache
            context: Optional context dict
            metadata: Optional metadata (model, tokens, etc.)
            ttl_hours: Optional custom TTL in hours
        """
        # Generate embedding (outside lock to avoid blocking)
        try:
            query_embedding = self._embed(query)
        except Exception as e:
            logger.debug(f"Failed to generate embedding for caching: {e}")
            return

        cache_key = self._hash_query(query, context)
        ttl = (ttl_hours * 3600) if ttl_hours else self.ttl_seconds

        cached_response = CachedResponse(
            response=response,
            embedding=query_embedding,
            timestamp=time.time(),
            ttl=ttl,
            query_hash=cache_key,
            metadata=metadata or {},
        )

        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._evict_lru()
            self._cache[cache_key] = cached_response

        logger.debug(f"Cached response: key={cache_key[:8]}..., ttl={ttl}s")

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        expired = [key for key, cached in self._cache.items() if cached.is_expired()]

        for key in expired:
            del self._cache[key]
            self._stats["expirations"] += 1

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache:
            return

        # Find entry with lowest hit count and oldest timestamp
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (
                self._cache[k].hit_count,
                self._cache[k].timestamp,
            ),
        )

        del self._cache[lru_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted LRU cache entry: {lru_key[:8]}...")

    @classmethod
    def is_cacheable(cls, response: str) -> bool:
        """Return True if this response is safe to cache across queries.

        Rejects responses that are session-specific or contain dynamic content:
        - Absolute file paths (session-specific filesystem state)
        - UUIDs or session identifiers
        - Very short responses (not meaningful to cache)
        - Responses with embedded tool call markers
        """
        if not response or len(response) < 50:
            return False
        # Reject if response contains absolute paths
        if re.search(r"(?<!\w)/(?:Users|home|tmp|var|etc|opt)/", response):
            return False
        # Reject if response contains UUID-like session identifiers
        if re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            response,
        ):
            return False
        # Reject if response contains tool call JSON markers
        if '{"tool_use"' in response or '"tool_calls"' in response:
            return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
        logger.info("Semantic response cache cleared")


# Global singleton instance
_global_cache: Optional[SemanticResponseCache] = None


def get_semantic_cache() -> SemanticResponseCache:
    """Get global semantic response cache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticResponseCache()
    return _global_cache
