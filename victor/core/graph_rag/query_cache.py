# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph query result cache for multi-hop retrieval (PH4-005).

Provides caching for GraphQueryTool and MultiHopRetriever results to avoid
redundant graph traversals for identical or similar queries.

Cache Key Strategy:
- Query text (normalized)
- Retrieval parameters: max_hops, seed_count, top_k, edge_types
- Graph store fingerprint (repo path + last update time)

Invalidation Strategy:
- TTL-based expiration (configurable, default 3600s)
- Manual invalidation on graph updates
- Path-based invalidation when files are modified

Integration:
- Registered with CacheRegistry as "graph_query"
- Used by MultiHopRetriever in graph_rag/retrieval.py
- Used by GraphQueryTool in tools/graph_query_tool.py

Example:
    from victor.core.graph_rag.query_cache import get_graph_query_cache

    cache = get_graph_query_cache()

    # Try to get cached result
    result = cache.get(query, config)
    if result is None:
        result = await retriever.retrieve(query, config)
        cache.put(query, config, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from cachetools import TTLCache

if TYPE_CHECKING:
    from victor.core.graph_rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GraphQueryCacheConfig:
    """Configuration for graph query caching.

    Attributes:
        enabled: Whether caching is enabled (default: True)
        max_entries: Maximum number of query results to cache (default: 100)
        ttl_seconds: Time-to-live for cache entries in seconds (default: 3600)
        normalize_queries: Whether to normalize query text for better hits (default: True)
        cache_by_repo: Whether to scope cache by repository path (default: True)
    """

    enabled: bool = True
    max_entries: int = 100
    ttl_seconds: int = 3600
    normalize_queries: bool = True
    cache_by_repo: bool = True
    # Persist entries to the repo's project.db (derived, rebuildable state) so
    # cache hits survive process restarts — without this, one-shot CLI runs
    # (`victor graph query ...`) never hit the cache at all.
    persist: bool = True


@dataclass
class CachedQueryResult:
    """A cached graph query result.

    Attributes:
        query_hash: Hash of the query and config
        result: The RetrievalResult (serialized)
        created_at: Timestamp when entry was created
        hit_count: Number of times this entry was accessed
        repo_path: Repository path for scoping
    """

    query_hash: str
    result: Dict[str, Any]  # Serialized RetrievalResult
    created_at: float
    hit_count: int = 0
    repo_path: Optional[str] = None


class _SqlitePersistence:
    """Per-repo L2 cache layer in ``project.db`` (derived, rebuildable state).

    Every operation is best-effort: any sqlite failure degrades the cache to
    memory-only rather than surfacing on the query path.
    """

    _TABLE = "graph_query_cache"

    def __init__(self, repo_key: str) -> None:
        from victor.core.database import get_project_database

        self._repo_key = repo_key
        self._db = get_project_database(Path(repo_key))
        conn = self._db.get_connection()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                query_hash TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at REAL NOT NULL,
                repo_key TEXT NOT NULL
            )
            """)
        conn.commit()

    def get(self, cache_key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
        conn = self._db.get_connection()
        row = conn.execute(
            f"SELECT result, created_at FROM {self._TABLE} WHERE query_hash = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        if time.time() - float(row[1]) > ttl_seconds:
            conn.execute(f"DELETE FROM {self._TABLE} WHERE query_hash = ?", (cache_key,))
            conn.commit()
            return None
        return json.loads(row[0])

    def put(self, cache_key: str, payload: Dict[str, Any], max_entries: int) -> None:
        conn = self._db.get_connection()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._TABLE}(query_hash, result, created_at, repo_key)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, json.dumps(payload), time.time(), self._repo_key),
        )
        # Size cap: keep the newest max_entries rows.
        conn.execute(
            f"""
            DELETE FROM {self._TABLE} WHERE query_hash NOT IN (
                SELECT query_hash FROM {self._TABLE}
                ORDER BY created_at DESC LIMIT ?
            )
            """,
            (max_entries,),
        )
        conn.commit()

    def invalidate_repo(self) -> int:
        conn = self._db.get_connection()
        cur = conn.execute(f"DELETE FROM {self._TABLE}")
        conn.commit()
        return int(cur.rowcount or 0)


def _normalize_query(query: str) -> str:
    """Normalize query text for cache key generation.

    Normalization steps:
    1. Lowercase
    2. Remove extra whitespace
    3. Remove common stop words
    4. Normalize code patterns (e.g., "find X" == "search for X")

    Args:
        query: Raw query string

    Returns:
        Normalized query string
    """
    # Lowercase and strip
    normalized = query.lower().strip()

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove common question words at the start
    prefixes_to_remove = [
        "how do i ",
        "how can i ",
        "how to ",
        "how do we ",
        "what is the ",
        "what are the ",
        "what ",
        "where is the ",
        "where are the ",
        "where ",
        "which ",
        "who ",
        "when ",
        "why ",
        "can you ",
        "could you ",
        "would you ",
        "find ",
        "search for ",
        "look for ",
        "get ",
        "show me ",
        "list ",
        "display ",
    ]

    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    return normalized.strip()


def _repo_key(repo_path: str) -> str:
    """Normalize a repo path to a stable scoping key.

    Uses the full resolved path (not just the directory name) so two repos
    that happen to share a basename never collide in the cache.
    """
    return str(Path(repo_path).expanduser().resolve(strict=False))


def _create_query_cache_key(
    query: str,
    config: Any,
    repo_path: Optional[str] = None,
    normalize: bool = True,
) -> str:
    """Create a stable cache key for a graph query.

    Args:
        query: Query string
        config: RetrievalConfig with parameters
        repo_path: Optional repository path for scoping
        normalize: Whether to normalize the query

    Returns:
        SHA-256 hash key for caching
    """
    # Normalize query if enabled
    query_text = _normalize_query(query) if normalize else query

    # Extract relevant config parameters
    config_params = {
        "seed_count": getattr(config, "seed_count", 5),
        "max_hops": getattr(config, "max_hops", 2),
        "top_k": getattr(config, "top_k", 10),
        "edge_types": sorted(getattr(config, "edge_types", None) or []),
        # Seed strategy: semantic/structural/hybrid results must not collide.
        "mode": getattr(config, "mode", "semantic"),
    }

    # Build key parts
    key_parts = [query_text]

    if repo_path:
        # Full resolved path: this is an in-process cache, so cross-machine
        # portability doesn't apply, and basename-only keys collide for
        # same-named repos.
        key_parts.append(_repo_key(repo_path))

    key_parts.append(json.dumps(config_params, sort_keys=True))

    # Generate hash
    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


class GraphQueryCache:
    """Cache for graph query results from MultiHopRetriever.

    Caches RetrievalResult objects to avoid redundant graph traversals.
    Uses TTL-based expiration with configurable size limits.

    Thread-safe implementation using locks.

    Attributes:
        config: Cache configuration

    Example:
        cache = GraphQueryCache()

        # Try to get cached result
        result = cache.get(query, config)
        if result is None:
            result = await retriever.retrieve(query, config)
            cache.put(query, config, result)

        # Get cache stats
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, config: Optional[GraphQueryCacheConfig] = None) -> None:
        """Initialize graph query cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or GraphQueryCacheConfig()
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats: Dict[str, int | float] = {
            "hits": 0,
            "misses": 0,
            "puts": 0,
            "evictions": 0,
            "invalidations": 0,
            "persistent_hits": 0,
        }

        # Track queries by repository for selective invalidation
        self._repo_index: Dict[str, Set[str]] = {}

        # Lazily opened per-repo sqlite L2 layers (False = open failed, don't retry)
        self._persistence: Dict[str, Any] = {}

        # Initialize cache if enabled
        if self._config.enabled:
            self._cache: Optional[TTLCache[str, CachedQueryResult]] = TTLCache(
                maxsize=self._config.max_entries,
                ttl=self._config.ttl_seconds,
            )
            logger.info(
                f"Graph query cache initialized: max_entries={self._config.max_entries}, "
                f"ttl={self._config.ttl_seconds}s"
            )
        else:
            self._cache = None
            logger.debug("Graph query cache disabled")

    def _persistence_for(self, repo_path: Optional[str]) -> Optional[_SqlitePersistence]:
        """Lazy per-repo L2 layer; a failed open is remembered and not retried."""
        if not self._config.persist or not repo_path:
            return None
        repo_key = _repo_key(repo_path)
        backend = self._persistence.get(repo_key)
        if backend is False:
            return None
        if backend is None:
            try:
                backend = _SqlitePersistence(repo_key)
                self._persistence[repo_key] = backend
            except Exception as e:
                logger.debug(f"Query cache persistence unavailable for {repo_key}: {e}")
                self._persistence[repo_key] = False
                return None
        return backend

    def _serialize_result(self, result: "RetrievalResult") -> Dict[str, Any]:
        """Serialize RetrievalResult for caching.

        Args:
            result: RetrievalResult to serialize

        Returns:
            Dictionary representation suitable for caching
        """
        # Convert nodes to minimal representation
        nodes_data = [
            {
                "node_id": n.node_id,
                "type": n.type,
                "name": n.name,
                "file": n.file,
                "line": n.line,
                "end_line": n.end_line,
                "lang": n.lang,
            }
            for n in result.nodes
        ]

        # Convert edges to minimal representation
        edges_data = [{"src": e.src, "dst": e.dst, "type": e.type} for e in result.edges]

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "query": result.query,
            "seed_nodes": result.seed_nodes,
            "hop_distances": result.hop_distances,
            "scores": result.scores,
            "execution_time_ms": result.execution_time_ms,
            "metadata": result.metadata,
            "cached_at": time.time(),
        }

    def _deserialize_result(self, data: Dict[str, Any]) -> "RetrievalResult":
        """Deserialize cached data back to RetrievalResult.

        Args:
            data: Cached dictionary data

        Returns:
            RetrievalResult instance
        """
        from victor.core.graph_rag.retrieval import RetrievalResult
        from victor.storage.graph.protocol import GraphEdge, GraphNode

        # Reconstruct nodes
        nodes = [
            GraphNode(
                node_id=n["node_id"],
                type=n["type"],
                name=n["name"],
                file=n["file"],
                line=n.get("line"),
                end_line=n.get("end_line"),
                lang=n.get("lang"),
            )
            for n in data.get("nodes", [])
        ]

        # Reconstruct edges
        edges = [
            GraphEdge(
                src=e["src"],
                dst=e["dst"],
                type=e["type"],
                weight=1.0,
            )
            for e in data.get("edges", [])
        ]

        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            subgraphs=[],  # Subgraphs not cached
            query=data["query"],
            seed_nodes=data.get("seed_nodes", []),
            hop_distances=data.get("hop_distances", {}),
            scores=data.get("scores", {}),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            metadata=data.get("metadata", {}),
        )

    def get(
        self,
        query: str,
        config: Any,
        repo_path: Optional[str] = None,
    ) -> Optional["RetrievalResult"]:
        """Get cached query result.

        Args:
            query: Query string
            config: Retrieval configuration
            repo_path: Optional repository path for scoping

        Returns:
            Cached RetrievalResult or None if not found
        """
        if not self._config.enabled or self._cache is None:
            return None

        cache_key = _create_query_cache_key(
            query,
            config,
            repo_path,
            normalize=self._config.normalize_queries,
        )

        with self._lock:
            try:
                entry: Optional[CachedQueryResult] = self._cache.get(cache_key)
                if entry is not None:
                    entry.hit_count += 1
                    self._stats["hits"] += 1
                    logger.debug(f"Graph query cache hit: {query[:50]}...")
                    return self._deserialize_result(entry.result)
            except KeyError:
                pass

            # L2: persisted entries survive process restarts (one-shot CLI runs).
            backend = self._persistence_for(repo_path)
            if backend is not None:
                try:
                    payload = backend.get(cache_key, self._config.ttl_seconds)
                except Exception as e:
                    logger.debug(f"Query cache persistence read failed: {e}")
                    payload = None
                if payload is not None:
                    # Promote to L1 for the rest of this process.
                    self._cache[cache_key] = CachedQueryResult(
                        query_hash=cache_key,
                        result=payload,
                        created_at=float(payload.get("cached_at", time.time())),
                        repo_path=repo_path,
                    )
                    if repo_path:
                        self._repo_index.setdefault(_repo_key(repo_path), set()).add(cache_key)
                    self._stats["hits"] += 1
                    self._stats["persistent_hits"] += 1
                    logger.debug(f"Graph query cache hit (persisted): {query[:50]}...")
                    return self._deserialize_result(payload)

            self._stats["misses"] += 1
            logger.debug(f"Graph query cache miss: {query[:50]}...")
            return None

    def put(
        self,
        query: str,
        config: Any,
        result: "RetrievalResult",
        repo_path: Optional[str] = None,
    ) -> bool:
        """Cache a query result.

        Args:
            query: Query string
            config: Retrieval configuration
            result: RetrievalResult to cache
            repo_path: Optional repository path for scoping

        Returns:
            True if successfully cached
        """
        if not self._config.enabled or self._cache is None:
            return False

        cache_key = _create_query_cache_key(
            query,
            config,
            repo_path,
            normalize=self._config.normalize_queries,
        )

        with self._lock:
            try:
                payload = self._serialize_result(result)
                entry = CachedQueryResult(
                    query_hash=cache_key,
                    result=payload,
                    created_at=time.time(),
                    repo_path=repo_path,
                )
                self._cache[cache_key] = entry
                self._stats["puts"] += 1

                # Update repo index
                if repo_path:
                    self._repo_index.setdefault(_repo_key(repo_path), set()).add(cache_key)

                # L2 write-through so future processes hit too.
                backend = self._persistence_for(repo_path)
                if backend is not None:
                    try:
                        backend.put(cache_key, payload, self._config.max_entries)
                    except Exception as e:
                        logger.debug(f"Query cache persistence write failed: {e}")

                logger.debug(f"Graph query cached: {query[:50]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed to cache graph query result: {e}")
                return False

    def invalidate(
        self,
        query: str,
        config: Any,
        repo_path: Optional[str] = None,
    ) -> bool:
        """Invalidate cache entry for a specific query.

        Args:
            query: Query string
            config: Retrieval configuration
            repo_path: Optional repository path

        Returns:
            True if entry was found and removed
        """
        if not self._config.enabled or self._cache is None:
            return False

        cache_key = _create_query_cache_key(
            query,
            config,
            repo_path,
            normalize=self._config.normalize_queries,
        )

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats["invalidations"] += 1
                logger.debug(f"Graph query invalidated: {query[:50]}...")
                return True
            return False

    def invalidate_repo(self, repo_path: str) -> int:
        """Invalidate all cached queries for a repository.

        Useful when the graph for a repository is updated.

        Args:
            repo_path: Path to the repository

        Returns:
            Number of entries invalidated
        """
        if not self._config.enabled or self._cache is None:
            return 0

        repo_key = _repo_key(repo_path)
        count = 0

        with self._lock:
            keys_to_remove = self._repo_index.get(repo_key, set()).copy()

            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                    count += 1

            if repo_key in self._repo_index:
                del self._repo_index[repo_key]

            # L2: drop persisted rows too — an index refresh in THIS process
            # must invalidate entries a FUTURE process would otherwise load.
            backend = self._persistence_for(repo_path)
            if backend is not None:
                try:
                    count += backend.invalidate_repo()
                except Exception as e:
                    logger.debug(f"Query cache persistence invalidation failed: {e}")

            self._stats["invalidations"] += count

        if count > 0:
            logger.info(f"Invalidated {count} graph query cache entries for repo: {repo_key}")

        return count

    def invalidate_all(self) -> int:
        """Invalidate all cached query results.

        Returns:
            Number of entries cleared
        """
        if not self._config.enabled or self._cache is None:
            return 0

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._repo_index.clear()
            # Clear every L2 layer this process has opened.
            for backend in self._persistence.values():
                if backend:
                    try:
                        count += backend.invalidate_repo()
                    except Exception as e:
                        logger.debug(f"Query cache persistence clear failed: {e}")
            logger.info(f"Cleared {count} graph query cache entries")
            self._stats["invalidations"] += count
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - puts: Number of entries added
            - hit_rate: Hit rate (0.0 to 1.0)
            - current_size: Current number of cached queries
            - max_size: Maximum cache size
            - enabled: Whether cache is enabled
            - ttl_seconds: TTL for entries
            - repo_count: Number of repositories with cached queries
        """
        with self._lock:
            stats = self._stats.copy()

            total = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0

            if self._cache is not None:
                stats["current_size"] = len(self._cache)
                stats["max_size"] = self._config.max_entries
            else:
                stats["current_size"] = 0
                stats["max_size"] = 0

            stats["enabled"] = self._config.enabled
            stats["ttl_seconds"] = self._config.ttl_seconds
            stats["repo_count"] = len(self._repo_index)

            return stats


# Global graph query cache instance
_global_graph_query_cache: Optional[GraphQueryCache] = None
_global_cache_lock = threading.Lock()


def get_graph_query_cache() -> GraphQueryCache:
    """Get the global graph query cache singleton.

    Returns:
        Global GraphQueryCache instance
    """
    global _global_graph_query_cache
    if _global_graph_query_cache is None:
        with _global_cache_lock:
            # Double-check pattern for thread safety
            if _global_graph_query_cache is None:
                _global_graph_query_cache = GraphQueryCache()
    return _global_graph_query_cache


def configure_graph_query_cache(config: GraphQueryCacheConfig) -> None:
    """Configure the global graph query cache.

    Args:
        config: Configuration to apply
    """
    global _global_graph_query_cache
    with _global_cache_lock:
        _global_graph_query_cache = GraphQueryCache(config)
        logger.info(f"Graph query cache configured: enabled={config.enabled}")


def reset_graph_query_cache() -> None:
    """Reset the global graph query cache (primarily for testing).

    Clears the global cache instance, allowing a fresh instance to be created
    on the next get_graph_query_cache() call.
    """
    global _global_graph_query_cache
    with _global_cache_lock:
        _global_graph_query_cache = None


__all__ = [
    "GraphQueryCacheConfig",
    "GraphQueryCache",
    "get_graph_query_cache",
    "configure_graph_query_cache",
    "reset_graph_query_cache",
    "_normalize_query",
    "_create_query_cache_key",
]
