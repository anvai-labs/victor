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

"""In-memory Tool Result Cache with FAISS for semantic similarity.

This module provides ephemeral caching of tool results using FAISS for
fast in-memory vector similarity search. No persistence - cache rebuilds
each session for optimal performance.

Key Features:
- FAISS IndexFlatIP for cosine similarity (fast, in-memory)
- LRU + TTL eviction (memcache-style)
- Per-tool configurable TTL and similarity thresholds
- File modification tracking for invalidation
- Zero disk I/O overhead

Usage:
    cache = ToolResultCache(embedding_service)

    # Check for cached result before executing tool
    cached = await cache.get("read", {"path": "src/main.py"})
    if cached:
        return cached  # Skip tool execution, save tokens!

    # After tool execution, cache the result
    result = await tool.execute(args)
    await cache.put("read", {"path": "src/main.py"}, result)

Design:
- Ephemeral: No persistence, rebuilds each session
- FAISS IndexFlatIP: Exact search, fast for <10K entries
- LRU eviction: Removes least recently used when full
- TTL expiry: Automatic cleanup of stale entries
"""

import hashlib
import heapq
from victor.core.json_utils import json_dumps
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ToolResultCacheEntry:
    """A cached tool result with metadata."""

    tool_name: str
    args_hash: str
    arguments: Dict[str, Any]
    result: Any
    embedding: np.ndarray
    created_at: float
    expires_at: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    file_path: Optional[str] = None
    file_mtime: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    def is_file_stale(self) -> bool:
        """Check if underlying file has changed."""
        if not self.file_path or not self.file_mtime:
            return False
        try:
            current_mtime = Path(self.file_path).stat().st_mtime
            return current_mtime > self.file_mtime
        except OSError:
            return True  # File gone = stale

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "evictions": self.evictions,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "tokens_saved": self.tokens_saved,
        }


# TTL configuration per tool (seconds)
# Since we use mtime-based invalidation for file operations,
# TTLs can be extended - they're just fallback expiry.
TOOL_TTL = {
    # File content - mtime-based invalidation, long TTL as fallback
    "read": 3600,  # 1 hour (mtime handles freshness)
    "grep": 1800,  # 30 min (mtime for files, TTL for pattern changes)
    "code_search": 1800,  # 30 min
    "semantic_code_search": 1800,  # 30 min
    # Directory structure - mtime-based, extended TTL
    "ls": 3600,  # 1 hour
    "overview": 3600,  # 1 hour
    "find": 1800,  # 30 min
    "glob": 1800,  # 30 min
    # Symbol/reference lookups - index-backed, long TTL
    "symbol": 3600,  # 1 hour
    "refs": 3600,  # 1 hour
    "graph": 3600,  # 1 hour
    # Never cache (side effects or volatile)
    "shell": 0,
    "write": 0,
    "edit": 0,
    "git": 0,
    "test": 0,
}

# Similarity thresholds per tool (higher = stricter matching)
SIMILARITY_THRESHOLD = {
    "read": 0.98,  # Path must be nearly identical
    "ls": 0.95,  # Directory path must match closely
    "grep": 0.90,  # Pattern + path matching
    "code_search": 0.85,  # Query similarity more flexible
    "semantic_code_search": 0.80,  # Semantic queries most flexible
    "overview": 0.90,
    "symbol": 0.95,
    "refs": 0.95,
    "find": 0.90,
    "graph": 0.90,
}


class ToolResultCache:
    """In-memory tool result cache with FAISS similarity search.

    Uses FAISS IndexFlatIP for fast cosine similarity search and
    LRU+TTL eviction for memory management. Fully ephemeral - no
    disk persistence.

    Thread-safe with lock protection for concurrent access.

    Attributes:
        embedding_service: Service for generating embeddings
        max_entries: Maximum cache entries before LRU eviction
        stats: Cache performance statistics
    """

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        max_entries: int = 500,
        cleanup_interval: float = 60.0,
    ):
        """Initialize in-memory cache.

        Args:
            embedding_service: Service for generating embeddings
            max_entries: Maximum entries before LRU eviction (default 500)
            cleanup_interval: Seconds between TTL cleanup runs (default 60)
        """
        self.embedding_service = embedding_service
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        self.stats = CacheStats()

        # Core data structures
        self._entries: OrderedDict[str, ToolResultCacheEntry] = OrderedDict()
        self._embeddings: List[np.ndarray] = []
        self._hash_to_idx: Dict[str, int] = {}
        self._faiss_index = None
        self._embedding_dim: Optional[int] = None

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup
        self._last_cleanup = time.time()

    def _ensure_faiss_index(self, dim: int) -> None:
        """Lazily initialize FAISS index."""
        if self._faiss_index is None:
            try:
                import faiss

                # IndexFlatIP for cosine similarity (with normalized vectors)
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._embedding_dim = dim
                logger.debug(f"FAISS index initialized with dim={dim}")
            except ImportError:
                logger.warning("FAISS not available, falling back to numpy search")
                self._embedding_dim = dim

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _hash_args(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate hash for tool+arguments."""
        normalized = json_dumps(
            {"tool": tool_name, "args": arguments},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _create_query_text(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create text representation for embedding."""
        parts = [f"tool:{tool_name}"]
        for key, value in sorted(arguments.items()):
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            parts.append(f"{key}:{value}")
        return " ".join(parts)

    def _maybe_cleanup(self) -> None:
        """Run TTL cleanup if interval elapsed."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.is_expired() or entry.is_file_stale()
        ]

        for key in expired_keys:
            self._remove_entry(key)
            self.stats.expirations += 1

        if expired_keys:
            logger.debug(f"TTL cleanup: removed {len(expired_keys)} expired entries")

    def _remove_entry(self, args_hash: str) -> None:
        """Remove entry by hash (internal, assumes lock held)."""
        if args_hash in self._entries:
            del self._entries[args_hash]
            # Note: We don't remove from FAISS index (would require rebuild)
            # Instead, we check entry existence on retrieval
            if args_hash in self._hash_to_idx:
                del self._hash_to_idx[args_hash]

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while len(self._entries) >= self.max_entries:
            # OrderedDict: first item is oldest
            oldest_key = next(iter(self._entries))
            self._remove_entry(oldest_key)
            self.stats.evictions += 1

    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from current entries."""
        if not self._entries or self._embedding_dim is None:
            return

        try:
            import faiss

            self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
            self._embeddings = []
            self._hash_to_idx = {}

            for idx, (args_hash, entry) in enumerate(self._entries.items()):
                normalized = self._normalize(entry.embedding)
                self._embeddings.append(normalized)
                self._hash_to_idx[args_hash] = idx

            if self._embeddings:
                vectors = np.vstack(self._embeddings).astype(np.float32)
                self._faiss_index.add(vectors)

        except ImportError:
            pass  # Fall back to numpy search

    async def get(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Optional[Any]:
        """Get cached result for similar tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            threshold: Override similarity threshold

        Returns:
            Cached result if found and valid, None otherwise
        """
        # Check if tool is cacheable
        ttl = TOOL_TTL.get(tool_name, 0)
        if ttl == 0:
            self.stats.misses += 1
            return None

        with self._lock:
            self._maybe_cleanup()

            # Fast path: exact hash match
            args_hash = self._hash_args(tool_name, arguments)
            if args_hash in self._entries:
                entry = self._entries[args_hash]
                if not entry.is_expired() and not entry.is_file_stale():
                    entry.touch()
                    # Move to end (most recently used)
                    self._entries.move_to_end(args_hash)
                    self.stats.hits += 1
                    self.stats.tokens_saved += self._estimate_tokens(entry.result)
                    return entry.result
                else:
                    self._remove_entry(args_hash)
                    self.stats.expirations += 1

            # Semantic similarity search
            if not self._entries:
                self.stats.misses += 1
                return None

            try:
                # Generate query embedding
                query_text = self._create_query_text(tool_name, arguments)
                embedding = await self.embedding_service.embed_async(query_text)

                if embedding is None:
                    self.stats.misses += 1
                    return None

                query_vec = self._normalize(np.array(embedding, dtype=np.float32))
                sim_threshold = threshold or SIMILARITY_THRESHOLD.get(tool_name, 0.90)

                # Search using FAISS or numpy fallback
                best_match = self._search_similar(query_vec, tool_name, sim_threshold)

                if best_match:
                    best_match.touch()
                    self._entries.move_to_end(best_match.args_hash)
                    self.stats.hits += 1
                    self.stats.tokens_saved += self._estimate_tokens(best_match.result)
                    return best_match.result

            except Exception as e:
                logger.debug(f"Cache lookup error: {e}")

            self.stats.misses += 1
            return None

    def _search_similar(
        self,
        query_vec: np.ndarray,
        tool_name: str,
        threshold: float,
    ) -> Optional[ToolResultCacheEntry]:
        """Search for similar entry using FAISS or numpy."""
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            try:
                # FAISS search
                query = query_vec.reshape(1, -1).astype(np.float32)
                distances, indices = self._faiss_index.search(query, min(10, len(self._entries)))

                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:
                        continue
                    similarity = float(dist)  # Already cosine similarity for IndexFlatIP
                    if similarity < threshold:
                        continue

                    # Find entry by index
                    for args_hash, stored_idx in self._hash_to_idx.items():
                        if stored_idx == idx and args_hash in self._entries:
                            entry = self._entries[args_hash]
                            if entry.tool_name == tool_name and not entry.is_expired():
                                return entry
                return None

            except Exception as e:
                logger.debug(f"FAISS search error: {e}")

        # Numpy fallback
        best_entry = None
        best_similarity = threshold

        for entry in self._entries.values():
            if entry.tool_name != tool_name or entry.is_expired():
                continue

            similarity = float(np.dot(query_vec, self._normalize(entry.embedding)))
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        return best_entry

    async def put(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        file_path: Optional[str] = None,
    ) -> bool:
        """Store tool result in cache.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool execution result
            file_path: Optional file path for invalidation tracking

        Returns:
            True if stored successfully
        """
        ttl = TOOL_TTL.get(tool_name, 0)
        if ttl == 0:
            return False

        try:
            # Generate embedding
            query_text = self._create_query_text(tool_name, arguments)
            embedding = await self.embedding_service.embed_async(query_text)

            if embedding is None:
                return False

            embedding_array = np.array(embedding, dtype=np.float32)
            self._ensure_faiss_index(len(embedding_array))

            now = time.time()
            args_hash = self._hash_args(tool_name, arguments)

            # Get file mtime if applicable
            file_mtime = None
            if file_path:
                try:
                    file_mtime = Path(file_path).stat().st_mtime
                except OSError:
                    pass

            entry = ToolResultCacheEntry(
                tool_name=tool_name,
                args_hash=args_hash,
                arguments=arguments,
                result=result,
                embedding=embedding_array,
                created_at=now,
                expires_at=now + ttl,
                file_path=file_path,
                file_mtime=file_mtime,
            )

            with self._lock:
                # Evict if needed
                self._evict_lru()

                # Store entry
                self._entries[args_hash] = entry

                # Add to FAISS index
                if self._faiss_index is not None:
                    normalized = self._normalize(embedding_array).reshape(1, -1).astype(np.float32)
                    self._faiss_index.add(normalized)
                    self._hash_to_idx[args_hash] = self._faiss_index.ntotal - 1

                self._embeddings.append(embedding_array)

            logger.debug(f"Cached {tool_name} result (ttl={ttl}s)")
            return True

        except Exception as e:
            logger.debug(f"Cache store error: {e}")
            return False

    def _estimate_tokens(self, result: Any) -> int:
        """Estimate tokens saved by cache hit."""
        try:
            result_str = json_dumps(result, default=str)
            return len(result_str) // 4  # Rough estimate
        except Exception:
            return 100  # Default estimate

    def invalidate(self, file_path: str) -> int:
        """Invalidate all cache entries for a file.

        Args:
            file_path: Path to invalidated file

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            to_remove = [
                key for key, entry in self._entries.items() if entry.file_path == file_path
            ]
            for key in to_remove:
                self._remove_entry(key)
                self.stats.invalidations += 1

            if to_remove:
                logger.debug(f"Invalidated {len(to_remove)} entries for {file_path}")

            return len(to_remove)

    def invalidate_tool(self, tool_name: str) -> int:
        """Invalidate all cache entries for a tool.

        Args:
            tool_name: Tool name to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            to_remove = [
                key for key, entry in self._entries.items() if entry.tool_name == tool_name
            ]
            for key in to_remove:
                self._remove_entry(key)
                self.stats.invalidations += 1

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self.stats.to_dict()
            stats["entries"] = len(self._entries)
            stats["max_entries"] = self.max_entries
            return stats

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._embeddings.clear()
            self._hash_to_idx.clear()
            if self._faiss_index is not None:
                self._faiss_index.reset()
            logger.info("Tool result cache cleared")


# Singleton instance for global access
_cache_instance: Optional[ToolResultCache] = None


def get_tool_cache(embedding_service: "EmbeddingService") -> ToolResultCache:
    """Get or create global tool cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ToolResultCache(embedding_service)
    return _cache_instance


def reset_tool_cache() -> None:
    """Reset global tool cache (for testing)."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.clear()
    _cache_instance = None
