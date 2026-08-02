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

"""Tiered cache manager with memory and disk caching."""

import hashlib
import json
import logging
from typing import Any, Optional, Dict, List
import threading

from cachetools import TTLCache  # type: ignore[import-untyped]
import diskcache

from victor.storage.cache.config import CacheConfig

logger = logging.getLogger(__name__)


class TieredCache:
    """Tiered cache with L1 (memory) and L2 (disk) caching.

    Architecture:
    - L1: Fast in-memory cache using cachetools (TTL-based)
    - L2: Persistent disk cache using diskcache (survives restarts)

    Features:
    - Automatic tiering (checks memory first, then disk)
    - Thread-safe operations
    - TTL support at both levels
    - Size limits
    - Statistics tracking
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        cache_eviction_learner: Optional[Any] = None,
    ):
        """Initialize tiered cache.

        Args:
            config: Cache configuration (uses defaults if None)
            cache_eviction_learner: Optional CacheEvictionLearner for RL-based eviction
        """
        self.config = config or CacheConfig()
        self._lock = threading.RLock()
        self._cache_eviction_learner = cache_eviction_learner

        # Statistics
        self._stats: Dict[str, float | int] = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "sets": 0,
            "evictions": 0,
            "rl_evictions": 0,
        }

        # Metadata for RL learning (key -> tool_name, set_time, hit_count)
        self._entry_metadata: Dict[str, Dict[str, Any]] = {}

        # Initialize L1 memory cache
        if self.config.enable_memory:
            self._memory_cache: Optional[TTLCache] = TTLCache(
                maxsize=self.config.memory_max_size,
                ttl=self.config.memory_ttl,
            )
        else:
            self._memory_cache = None

        # Initialize L2 disk cache
        if self.config.enable_disk:
            self._disk_cache: Optional[diskcache.Cache] = diskcache.Cache(
                directory=str(self.config.disk_path),
                size_limit=self.config.disk_max_size,
            )
        else:
            self._disk_cache = None

        if cache_eviction_learner:
            logger.info("RL: TieredCache using unified CacheEvictionLearner")

        logger.info(
            "Cache initialized: memory=%s, disk=%s",
            self.config.enable_memory,
            self.config.enable_disk,
        )

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache.

        Checks L1 (memory) first, then L2 (disk).
        If found in disk, promotes to memory.

        Args:
            key: Cache key
            namespace: Cache namespace for organization

        Returns:
            Cached value or None if not found
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            # Try L1 memory cache
            if self._memory_cache is not None:
                try:
                    value = self._memory_cache.get(cache_key)
                    if value is not None:
                        self._stats["memory_hits"] += 1
                        logger.debug("Memory cache hit: %s", cache_key)
                        return value
                    self._stats["memory_misses"] += 1
                except KeyError:
                    self._stats["memory_misses"] += 1

            # Try L2 disk cache
            if self._disk_cache is not None:
                try:
                    value = self._disk_cache.get(cache_key)
                    if value is not None:
                        self._stats["disk_hits"] += 1
                        logger.debug("Disk cache hit: %s", cache_key)

                        # Promote to memory cache
                        if self._memory_cache is not None:
                            self._memory_cache[cache_key] = value

                        return value
                    self._stats["disk_misses"] += 1
                except KeyError:
                    self._stats["disk_misses"] += 1

            logger.debug("Cache miss: %s", cache_key)
            return None

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.

        Stores in both L1 (memory) and L2 (disk) if enabled.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Custom TTL in seconds (uses config default if None)

        Returns:
            True if successfully cached
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            success = False

            # Store in L1 memory cache
            if self._memory_cache is not None:
                try:
                    self._memory_cache[cache_key] = value
                    success = True
                except Exception as e:
                    logger.warning("Failed to store in memory cache: %s", e)

            # Store in L2 disk cache
            if self._disk_cache is not None:
                try:
                    expire_time = ttl or self.config.disk_ttl
                    self._disk_cache.set(cache_key, value, expire=expire_time)
                    success = True
                except Exception as e:
                    logger.warning("Failed to store in disk cache: %s", e)

            if success:
                self._stats["sets"] += 1
                logger.debug("Cached: %s", cache_key)

            return success

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache.

        Removes from both L1 and L2 if present.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            True if deleted from at least one cache
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            deleted = False

            # Remove from L1
            if self._memory_cache is not None:
                try:
                    del self._memory_cache[cache_key]
                    deleted = True
                except KeyError:
                    pass

            # Remove from L2
            if self._disk_cache is not None:
                try:
                    deleted = self._disk_cache.delete(cache_key) or deleted
                except Exception as e:
                    logger.warning("Failed to delete from disk cache: %s", e)

            if deleted:
                logger.debug("Deleted from cache: %s", cache_key)

            return deleted

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            namespace: If provided, clear only this namespace. Otherwise clear all.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = 0

            if namespace:
                # Clear specific namespace
                if self._memory_cache is not None:
                    keys_to_delete = [
                        k for k in self._memory_cache.keys() if k.startswith(f"{namespace}:")
                    ]
                    for key in keys_to_delete:
                        try:
                            del self._memory_cache[key]
                            count += 1
                        except KeyError:
                            pass

                if self._disk_cache is not None:
                    # diskcache namespace clear by iteration
                    to_delete = [
                        k for k in self._disk_cache.iterkeys() if k.startswith(f"{namespace}:")
                    ]
                    for key in to_delete:
                        try:
                            del self._disk_cache[key]
                            count += 1
                        except Exception:
                            pass
            else:
                # Clear all
                if self._memory_cache is not None:
                    count += len(self._memory_cache)
                    self._memory_cache.clear()

                if self._disk_cache is not None:
                    count += len(self._disk_cache)
                    self._disk_cache.clear()

            logger.info("Cleared %d cache entries", count)
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            stats = self._stats.copy()

            # Calculate hit rates
            total_memory = stats["memory_hits"] + stats["memory_misses"]
            total_disk = stats["disk_hits"] + stats["disk_misses"]

            stats["memory_hit_rate"] = (
                stats["memory_hits"] / total_memory if total_memory > 0 else 0
            )
            stats["disk_hit_rate"] = stats["disk_hits"] / total_disk if total_disk > 0 else 0

            # Add size info
            if self._memory_cache is not None:
                stats["memory_size"] = len(self._memory_cache)
                stats["memory_max_size"] = self.config.memory_max_size

            if self._disk_cache is not None:
                stats["disk_size"] = len(self._disk_cache)
                stats["disk_volume"] = self._disk_cache.volume()

            return stats

    def set_with_tool(
        self,
        key: str,
        value: Any,
        tool_name: str,
        namespace: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with tool metadata for RL learning.

        Args:
            key: Cache key
            value: Value to cache
            tool_name: Name of the tool that produced this result
            namespace: Cache namespace
            ttl: Custom TTL in seconds

        Returns:
            True if successfully cached
        """
        import time

        cache_key = self._make_key(key, namespace)

        # Store metadata for RL learning
        self._entry_metadata[cache_key] = {
            "tool_name": tool_name,
            "set_time": time.time(),
            "hit_count": 0,
        }

        return self.set(key, value, namespace, ttl)

    def get_with_rl(
        self,
        key: str,
        namespace: str = "default",
        tool_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Get value from cache and record hit/miss for RL learning.

        Args:
            key: Cache key
            namespace: Cache namespace
            tool_name: Tool name for RL tracking (optional)

        Returns:
            Cached value or None
        """
        import time

        cache_key = self._make_key(key, namespace)
        value = self.get(key, namespace)

        # Update metadata and record to RL learner
        if cache_key in self._entry_metadata:
            meta = self._entry_metadata[cache_key]
            meta["hit_count"] += 1

            # Record hit to RL learner
            self._record_cache_hit(
                cache_key=cache_key,
                tool_name=meta.get("tool_name", tool_name or "unknown"),
                age_seconds=time.time() - meta.get("set_time", time.time()),
            )
        elif value is None and tool_name:
            # Record miss
            self._record_cache_miss(tool_name=tool_name)

        return value

    def _record_cache_hit(
        self,
        cache_key: str,
        tool_name: str,
        age_seconds: float,
    ) -> None:
        """Record cache hit to RL learner."""
        if not self._cache_eviction_learner:
            return

        try:
            from victor.framework.rl.base import RLOutcome

            # Get current utilization
            utilization = self._get_utilization()

            meta = self._entry_metadata.get(cache_key, {})
            hit_count = meta.get("hit_count", 1)

            outcome = RLOutcome(
                success=True,
                quality_score=1.0,
                task_type="cache",
                metadata={
                    "state_key": self._build_state_key(
                        utilization, age_seconds, hit_count, tool_name
                    ),
                    "action": "keep",
                    "tool_name": tool_name,
                    "hit_after": 1,
                },
            )
            self._cache_eviction_learner.record_outcome(outcome)
            logger.debug(f"RL: Recorded cache hit for {tool_name}")

            # Emit RL hook event for cache hit
            self._emit_cache_event(tool_name=tool_name, cache_hit=True)

        except Exception as e:
            logger.debug(f"RL: Failed to record cache hit: {e}")

    def _record_cache_miss(self, tool_name: str) -> None:
        """Record cache miss to RL learner."""
        if not self._cache_eviction_learner:
            return

        try:
            from victor.framework.rl.base import RLOutcome

            outcome = RLOutcome(
                success=False,
                quality_score=0.0,
                task_type="cache",
                metadata={
                    "state_key": f"miss:{tool_name}",
                    "action": "evict",  # Entry was evicted (or never cached)
                    "tool_name": tool_name,
                    "hit_after": 0,
                },
            )
            self._cache_eviction_learner.record_outcome(outcome)
            logger.debug(f"RL: Recorded cache miss for {tool_name}")

            # Emit RL hook event for cache miss
            self._emit_cache_event(tool_name=tool_name, cache_hit=False)

        except Exception as e:
            logger.debug(f"RL: Failed to record cache miss: {e}")

    def _emit_cache_event(self, tool_name: str, cache_hit: bool) -> None:
        """Emit RL event for cache access.

        Args:
            tool_name: Name of the tool
            cache_hit: Whether this was a cache hit
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            event = RLEvent(
                type=RLEventType.CACHE_ACCESS,
                tool_name=tool_name,
                cache_hit=cache_hit,
                success=cache_hit,
                quality_score=1.0 if cache_hit else 0.0,
                metadata={
                    "utilization": self._get_utilization(),
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Cache event emission failed: {e}")

    def _get_utilization(self) -> float:
        """Get current cache utilization (0-1)."""
        if self._memory_cache is not None:
            return len(self._memory_cache) / max(self.config.memory_max_size, 1)
        return 0.0

    def _build_state_key(
        self,
        utilization: float,
        age_seconds: float,
        hit_count: int,
        tool_name: str,
    ) -> str:
        """Build state key for RL learner."""
        # Discretize utilization
        if utilization < 0.25:
            util_bucket = "low"
        elif utilization < 0.5:
            util_bucket = "mid_low"
        elif utilization < 0.75:
            util_bucket = "mid_high"
        else:
            util_bucket = "high"

        # Discretize age
        if age_seconds < 60:
            age_bucket = "fresh"
        elif age_seconds < 300:
            age_bucket = "recent"
        elif age_seconds < 900:
            age_bucket = "aging"
        else:
            age_bucket = "old"

        # Discretize hits
        if hit_count == 0:
            hit_bucket = "cold"
        elif hit_count <= 2:
            hit_bucket = "warm"
        else:
            hit_bucket = "hot"

        # Get tool type
        tool_type = tool_name.split("_")[0] if "_" in tool_name else tool_name

        return f"{util_bucket}:{age_bucket}:{hit_bucket}:{tool_type}"

    def smart_evict(self, count: int = 1) -> int:
        """Evict entries using RL-learned policy.

        Uses the CacheEvictionLearner to decide which entries to evict
        based on learned value estimates.

        Args:
            count: Number of entries to evict

        Returns:
            Number of entries actually evicted
        """
        import time

        if not self._cache_eviction_learner:
            return 0

        evicted = 0

        with self._lock:
            if self._memory_cache is None:
                return 0

            # Score all entries using RL
            candidates = []
            for cache_key in list(self._memory_cache.keys()):
                meta = self._entry_metadata.get(cache_key, {})
                tool_name = meta.get("tool_name", "unknown")
                age_seconds = time.time() - meta.get("set_time", time.time())
                hit_count = meta.get("hit_count", 0)

                # Get eviction decision from RL
                action, confidence = self._cache_eviction_learner.get_eviction_decision(
                    utilization=self._get_utilization(),
                    age_seconds=age_seconds,
                    hit_count=hit_count,
                    tool_name=tool_name,
                )

                if action == "evict":
                    candidates.append((cache_key, confidence))

            # Sort by confidence (higher confidence = more sure about eviction)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Evict top candidates
            for cache_key, _ in candidates[:count]:
                try:
                    del self._memory_cache[cache_key]
                    if cache_key in self._entry_metadata:
                        del self._entry_metadata[cache_key]
                    evicted += 1
                    self._stats["evictions"] += 1
                    self._stats["rl_evictions"] += 1
                except KeyError:
                    pass

            if evicted > 0:
                logger.info(f"RL: Smart eviction removed {evicted} entries")

        return evicted

    def _make_key(self, key: str, namespace: str) -> str:
        """Create cache key with namespace.

        Args:
            key: Original key
            namespace: Namespace for organization

        Returns:
            Namespaced cache key
        """
        # Hash long keys to keep them manageable
        if len(key) > 200:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            return f"{namespace}:{key_hash}"

        return f"{namespace}:{key}"

    def warmup(self, data: Dict[str, Any], namespace: str = "default") -> int:
        """Warm up cache with data.

        Args:
            data: Dictionary of key-value pairs to cache
            namespace: Cache namespace

        Returns:
            Number of entries cached
        """
        count = 0
        for key, value in data.items():
            if self.set(key, value, namespace):
                count += 1

        logger.info("Warmed up cache with %d entries", count)
        return count

    def close(self) -> None:
        """Close cache connections and cleanup."""
        if self._disk_cache is not None:
            try:
                self._disk_cache.close()
                logger.info("Disk cache closed")
            except Exception as e:
                logger.warning("Error closing disk cache: %s", e)

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration.

        Returns a dictionary formatted for the ObservabilityManager.
        Includes cache statistics like hit rates, sizes, and eviction counts.

        Returns:
            Dictionary with observability data including:
            - source_id: Unique identifier for this cache
            - source_type: Type of metrics source ("cache")
            - stats: Cache statistics dictionary

        Example:
            data = cache.get_observability_data()
            print(f"Hit rate: {data['stats']['memory_hit_rate']:.2%}")
        """
        stats = self.get_stats()
        return {
            "source_id": f"cache:tiered:{id(self)}",
            "source_type": "cache",
            "stats": stats,
        }

    def __enter__(self) -> "TieredCache":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()


class ResponseCache:
    """Specialized cache for LLM responses.

    Caches expensive LLM API calls to reduce costs and latency.
    """

    def __init__(self, cache: Optional["TieredCache"] = None):
        """Initialize response cache.

        Args:
            cache: TieredCache instance (creates default if None)
        """
        self.cache = cache or TieredCache()
        self.namespace = "responses"

    def get_response(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get cached response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting

        Returns:
            Cached response or None
        """
        key = self._make_response_key(prompt, model, temperature)
        return self.cache.get(key, self.namespace)

    def cache_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache LLM response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting
            response: Response to cache
            ttl: Custom TTL (uses default if None)

        Returns:
            True if successfully cached
        """
        key = self._make_response_key(prompt, model, temperature)
        return self.cache.set(key, response, self.namespace, ttl)

    def _make_response_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create cache key for response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting

        Returns:
            Cache key
        """
        # Include model and temperature in key for uniqueness
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class EmbeddingCache:
    """Specialized cache for embeddings.

    Caches expensive embedding computations for semantic search.
    """

    def __init__(self, cache: Optional["TieredCache"] = None):
        """Initialize embedding cache.

        Args:
            cache: TieredCache instance
        """
        self.cache = cache or TieredCache()
        self.namespace = "embeddings"

    def get_embedding(self, text: str, model: str) -> Optional[list]:
        """Get cached embedding.

        Args:
            text: Text to get embedding for
            model: Embedding model name

        Returns:
            Cached embedding vector or None
        """
        key = self._make_embedding_key(text, model)
        return self.cache.get(key, self.namespace)

    def cache_embedding(
        self, text: str, model: str, embedding: list, ttl: Optional[int] = None
    ) -> bool:
        """Cache embedding.

        Args:
            text: Text
            model: Embedding model
            embedding: Embedding vector
            ttl: Custom TTL

        Returns:
            True if successfully cached
        """
        key = self._make_embedding_key(text, model)
        return self.cache.set(key, embedding, self.namespace, ttl)

    def get_batch_embeddings(
        self,
        texts: List[str],
        model: str,
        embedding_fn: Optional[callable] = None,
    ) -> List[Optional[list]]:
        """Get embeddings for multiple texts, computing missing ones.

        Args:
            texts: List of texts to get embeddings for
            model: Embedding model name
            embedding_fn: Optional function to compute embeddings for missing texts.
                           Should take (text, model) and return embedding list.

        Returns:
            List of embedding vectors (None for texts that couldn't be computed)

        Example:
            def compute_embedding(text, model):
                return provider.get_embedding(text, model)

            embeddings = cache.get_batch_embeddings(
                texts=["hello", "world"],
                model="text-embedding-3-small",
                embedding_fn=compute_embedding
            )
        """
        results = []
        missing_indices = []

        # First pass: check cache for all texts
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text, model)
            if embedding is not None:
                results.append(embedding)
            else:
                results.append(None)
                missing_indices.append((i, text))

        # Second pass: compute missing embeddings if function provided
        if embedding_fn and missing_indices:
            for i, text in missing_indices:
                try:
                    embedding = embedding_fn(text, model)
                    if embedding is not None:
                        self.cache_embedding(text, model, embedding)
                        results[i] = embedding
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for text {i}: {e}")
                    results[i] = None

        return results

    def preload_embeddings(
        self,
        texts: List[str],
        model: str,
        embedding_fn: callable,
        batch_size: int = 10,
    ) -> Dict[str, int]:
        """Preload embeddings for a batch of texts.

        Computes and caches embeddings for texts that aren't already cached.
        Useful for warming up the cache during initialization.

        Args:
            texts: List of texts to preload embeddings for
            model: Embedding model name
            embedding_fn: Function to compute embeddings
            batch_size: Number of texts to process in each batch

        Returns:
            Dictionary with statistics:
            - "total": Total number of texts
            - "cached": Number already cached
            - "computed": Number of embeddings computed
            - "failed": Number that failed to compute

        Example:
            def compute_embeddings_batch(texts, model):
                return [provider.get_embedding(t, model) for t in texts]

            stats = cache.preload_embeddings(
                texts=["hello", "world", "test"],
                model="text-embedding-3-small",
                embedding_fn=lambda t, m: provider.get_embedding(t, m)
            )
            print(f"Computed {stats['computed']} embeddings")
        """
        stats = {
            "total": len(texts),
            "cached": 0,
            "computed": 0,
            "failed": 0,
        }

        # Check which texts are already cached
        to_compute = []
        for text in texts:
            if self.get_embedding(text, model) is not None:
                stats["cached"] += 1
            else:
                to_compute.append(text)

        # Compute embeddings in batches
        if to_compute and embedding_fn:
            logger.info(f"Preloading {len(to_compute)} embeddings...")

            for i in range(0, len(to_compute), batch_size):
                batch = to_compute[i : i + batch_size]
                for text in batch:
                    try:
                        embedding = embedding_fn(text, model)
                        if embedding is not None:
                            self.cache_embedding(text, model, embedding)
                            stats["computed"] += 1
                        else:
                            stats["failed"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to preload embedding: {e}")
                        stats["failed"] += 1

        logger.info(
            f"Preload complete: {stats['cached']} cached, "
            f"{stats['computed']} computed, {stats['failed']} failed"
        )

        return stats

    def _make_embedding_key(self, text: str, model: str) -> str:
        """Create cache key for embedding.

        Args:
            text: Text
            model: Model name

        Returns:
            Cache key
        """
        key_str = f"{model}:{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()
