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

"""Compiled graph cache for StateGraph workflows.

Provides caching for compiled StateGraphs based on structural hash,
avoiding redundant compilation for identical graph structures.

The cache uses SHA-256 hashing of graph structure (nodes, edges, entry_point,
state_schema) to identify equivalent graphs. This enables efficient reuse
of compiled graphs across multiple executions.

Example:
    from victor.framework.graph_cache import get_compiled_graph_cache

    cache = get_compiled_graph_cache()

    # Get or compile with automatic caching
    compiled = cache.get_or_compile(graph, checkpointer=my_checkpointer)

    # Check cache stats
    stats = cache.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.2%}")
"""

from __future__ import annotations

import hashlib
import inspect
from victor.core.json_utils import json_dumps
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
    TypeVar,
)

from cachetools import TTLCache  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from victor.framework.graph import (
        CheckpointerProtocol,
        CompiledGraph,
        StateGraph,
    )

logger = logging.getLogger(__name__)

StateType = TypeVar("StateType", bound=Dict[str, Any])


@dataclass
class CompiledGraphCacheConfig:
    """Configuration for compiled graph caching.

    Attributes:
        enabled: Whether caching is enabled (default: True)
        max_entries: Maximum number of compiled graphs to cache (default: 50)
        ttl_seconds: Time-to-live for cache entries in seconds (default: 3600)
    """

    enabled: bool = True
    max_entries: int = 50
    ttl_seconds: int = 3600


@dataclass
class GraphCacheEntry:
    """A cached compiled graph entry.

    Attributes:
        graph_hash: SHA-256 hash of graph structure
        compiled: The compiled graph instance
        created_at: Timestamp when entry was created
        hit_count: Number of times this entry was accessed
    """

    graph_hash: str
    compiled: "CompiledGraph"
    created_at: float
    hit_count: int = 0


class CompiledGraphCache:
    """Cache for compiled StateGraphs based on structural hash.

    Caches compiled StateGraph instances to avoid redundant compilation
    for graphs with identical structure. Uses SHA-256 hashing of nodes,
    edges, entry_point, and state_schema to identify equivalent graphs.

    Thread-safe implementation using locks.

    Attributes:
        config: Cache configuration

    Example:
        cache = CompiledGraphCache()

        # Try to get cached compiled graph
        compiled = cache.get(graph)
        if compiled is None:
            compiled = graph.compile()
            cache.put(graph, compiled)

        # Or use convenience method
        compiled = cache.get_or_compile(graph)
    """

    def __init__(self, config: Optional[CompiledGraphCacheConfig] = None) -> None:
        """Initialize compiled graph cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or CompiledGraphCacheConfig()
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "compilations": 0,
            "evictions": 0,
        }

        # Initialize cache if enabled
        if self._config.enabled:
            self._cache: Optional[TTLCache[str, GraphCacheEntry]] = TTLCache(
                maxsize=self._config.max_entries,
                ttl=self._config.ttl_seconds,
            )
            logger.info(
                f"Compiled graph cache initialized: max_entries={self._config.max_entries}, "
                f"ttl={self._config.ttl_seconds}s"
            )
        else:
            self._cache = None
            logger.debug("Compiled graph cache disabled")

    @staticmethod
    def _compute_function_fingerprint(func: Any) -> Optional[str]:
        """Return a stable fingerprint for callable implementation changes."""
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = None

        if source:
            return hashlib.sha256(source.encode()).hexdigest()[:16]

        code = getattr(func, "__code__", None)
        if code is None:
            return None

        code_payload = {
            "co_code": code.co_code.hex(),
            "co_consts": repr(code.co_consts),
            "co_names": code.co_names,
            "defaults": repr(getattr(func, "__defaults__", None)),
            "kwdefaults": repr(getattr(func, "__kwdefaults__", None)),
        }
        payload = json_dumps(code_payload, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _compute_graph_hash(self, graph: "StateGraph") -> str:
        """Compute SHA-256 hash of graph structure.

        Hash is based on:
        - Node IDs and their function names
        - Edge structure (source, target, type)
        - Entry point
        - State schema name (if available)

        Args:
            graph: The StateGraph to hash

        Returns:
            SHA-256 hash string of graph structure
        """
        # Build hash data from graph structure
        hash_data: Dict[str, Any] = {
            "nodes": {},
            "edges": {},
            "entry_point": graph._entry_point,
            "state_schema": None,
        }

        # Add node information
        for node_id, node in graph._nodes.items():
            hash_data["nodes"][node_id] = {
                "id": node.id,
                "func_name": getattr(node.func, "__name__", str(node.func)),
                "func_module": getattr(node.func, "__module__", "unknown"),
                "func_fingerprint": self._compute_function_fingerprint(node.func),
            }

        # Add edge information
        for source, edges in graph._edges.items():
            hash_data["edges"][source] = []
            for edge in edges:
                edge_data = {
                    "source": edge.source,
                    "target": (
                        edge.target
                        if isinstance(edge.target, str)
                        else (
                            sorted(edge.target.items())
                            if isinstance(edge.target, dict)
                            else str(edge.target)
                        )
                    ),
                    "type": edge.edge_type.value,
                }
                if edge.condition is not None:
                    edge_data["condition_name"] = getattr(
                        edge.condition, "__name__", str(edge.condition)
                    )
                hash_data["edges"][source].append(edge_data)

        # Add state schema information
        if graph._state_schema is not None:
            hash_data["state_schema"] = {
                "name": graph._state_schema.__name__,
                "module": getattr(graph._state_schema, "__module__", "unknown"),
            }

        # Generate hash
        hash_str = json_dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def get(self, graph: "StateGraph") -> Optional["CompiledGraph"]:
        """Get cached compiled graph.

        Args:
            graph: The StateGraph to look up

        Returns:
            Cached CompiledGraph or None if not found
        """
        if not self._config.enabled or self._cache is None:
            return None

        graph_hash = self._compute_graph_hash(graph)

        with self._lock:
            try:
                entry: Optional[GraphCacheEntry] = self._cache.get(graph_hash)
                if entry is not None:
                    entry.hit_count += 1
                    self._stats["hits"] += 1
                    logger.debug(f"Graph cache hit: hash={graph_hash[:16]}...")
                    return entry.compiled
            except KeyError:
                pass

            self._stats["misses"] += 1
            logger.debug(f"Graph cache miss: hash={graph_hash[:16]}...")
            return None

    def put(self, graph: "StateGraph", compiled: "CompiledGraph") -> bool:
        """Cache a compiled graph.

        Args:
            graph: The source StateGraph (used for hash computation)
            compiled: The compiled graph to cache

        Returns:
            True if successfully cached
        """
        if not self._config.enabled or self._cache is None:
            return False

        graph_hash = self._compute_graph_hash(graph)

        with self._lock:
            try:
                entry = GraphCacheEntry(
                    graph_hash=graph_hash,
                    compiled=compiled,
                    created_at=time.time(),
                )
                self._cache[graph_hash] = entry
                logger.debug(f"Graph cached: hash={graph_hash[:16]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed to cache compiled graph: {e}")
                return False

    def get_or_compile(
        self,
        graph: "StateGraph",
        checkpointer: Optional["CheckpointerProtocol"] = None,
        **config_kwargs: Any,
    ) -> "CompiledGraph":
        """Get cached compiled graph or compile and cache.

        Convenience method that combines get(), compile(), and put() operations.
        If the graph is found in cache, returns the cached version.
        Otherwise, compiles the graph, caches it, and returns it.

        Args:
            graph: The StateGraph to compile
            checkpointer: Optional checkpointer for persistence
            **config_kwargs: Additional config options for compilation

        Returns:
            CompiledGraph (from cache or freshly compiled)
        """
        # Try cache first
        cached = self.get(graph)
        if cached is not None:
            return cached

        # Compile and cache
        with self._lock:
            self._stats["compilations"] += 1

        compiled = graph.compile(checkpointer=checkpointer, **config_kwargs)
        self.put(graph, compiled)

        return compiled

    def invalidate(self, graph: "StateGraph") -> bool:
        """Invalidate cache entry for a specific graph.

        Args:
            graph: The StateGraph to invalidate

        Returns:
            True if entry was found and removed
        """
        if not self._config.enabled or self._cache is None:
            return False

        graph_hash = self._compute_graph_hash(graph)

        with self._lock:
            if graph_hash in self._cache:
                del self._cache[graph_hash]
                logger.debug(f"Graph invalidated: hash={graph_hash[:16]}...")
                return True
            return False

    def invalidate_all(self) -> int:
        """Invalidate all cached compiled graphs.

        Returns:
            Number of entries cleared
        """
        if not self._config.enabled or self._cache is None:
            return 0

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} compiled graph cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - compilations: Number of graph compilations
            - hit_rate: Hit rate (0.0 to 1.0)
            - current_size: Current number of cached graphs
            - max_size: Maximum cache size
            - enabled: Whether cache is enabled
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

            return stats


# Global compiled graph cache instance
_global_compiled_graph_cache: Optional[CompiledGraphCache] = None
_global_cache_lock = threading.Lock()


def get_compiled_graph_cache() -> CompiledGraphCache:
    """Get the global compiled graph cache singleton.

    Returns:
        Global CompiledGraphCache instance
    """
    global _global_compiled_graph_cache
    if _global_compiled_graph_cache is None:
        with _global_cache_lock:
            # Double-check pattern for thread safety
            if _global_compiled_graph_cache is None:
                _global_compiled_graph_cache = CompiledGraphCache()
    return _global_compiled_graph_cache


def configure_compiled_graph_cache(config: CompiledGraphCacheConfig) -> None:
    """Configure the global compiled graph cache.

    Args:
        config: Configuration to apply
    """
    global _global_compiled_graph_cache
    with _global_cache_lock:
        _global_compiled_graph_cache = CompiledGraphCache(config)
        logger.info(f"Compiled graph cache configured: enabled={config.enabled}")


def reset_compiled_graph_cache() -> None:
    """Reset the global compiled graph cache (primarily for testing).

    Clears the global cache instance, allowing a fresh instance to be created
    on the next get_compiled_graph_cache() call.
    """
    global _global_compiled_graph_cache
    with _global_cache_lock:
        _global_compiled_graph_cache = None


__all__ = [
    "CompiledGraphCacheConfig",
    "CompiledGraphCache",
    "get_compiled_graph_cache",
    "configure_compiled_graph_cache",
    "reset_compiled_graph_cache",
]
