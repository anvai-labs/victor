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

"""Generic result cache for extended caching beyond tool results.

This module extends caching to non-tool results, providing a unified
caching interface for various result types:

- LLM response results
- Embedding computation results
- Parsing results
- Validation results
- Path-based invalidation for dependent results

Design Pattern: Strategy Pattern + Decorator Pattern
- Extends TieredCache for generic result caching
- Path-based invalidation for dependent results
- Result type tracking for selective invalidation

Phase 3: Improve Performance with Extended Caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Callable
from pathlib import Path

from victor.storage.cache.tiered_cache import TieredCache
from victor.storage.cache.config import CacheConfig

logger = logging.getLogger(__name__)


class ResultType(Enum):
    """Types of cached results for selective invalidation."""

    LLM_RESPONSE = "llm_response"
    EMBEDDING = "embedding"
    PARSING = "parsing"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    ANALYSIS = "analysis"
    SEARCH = "search"
    CODE_ACTION = "code_action"
    TOOL_SELECTION = "tool_selection"
    OTHER = "other"


@dataclass
class CacheDependency:
    """Represents a dependency relationship between cached results.

    Attributes:
        key: The cache key that depends on others
        depends_on: List of cache keys this result depends on
        result_type: Type of the cached result
        created_at: Timestamp when the dependency was created
    """

    key: str
    depends_on: List[str]
    result_type: ResultType
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "depends_on": self.depends_on,
            "result_type": self.result_type.value,
            "created_at": self.created_at,
        }


class InvalidationStrategy(Enum):
    """Strategies for cache invalidation."""

    EXACT = "exact"  # Invalidate only exact key matches
    PREFIX = "prefix"  # Invalidate all keys with prefix
    DEPENDENCY = "dependency"  # Invalidate based on dependency graph


def _create_cache_key(
    result_type: ResultType,
    identifier: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a stable cache key for a generic result.

    Args:
        result_type: Type of the result
        identifier: Unique identifier for the result
        params: Optional parameters that affect the result

    Returns:
        Stable hash key for caching
    """
    key_parts = [result_type.value, identifier]
    if params:
        try:
            params_str = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            params_str = str(params)
        key_parts.append(params_str)
    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


def _create_tool_selection_cache_key(
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    conversation_depth: int = 0,
    stage: Optional[str] = None,
    use_semantic: bool = True,
) -> str:
    """Create a stable cache key for tool selection results.

    Tool selection is expensive due to embedding computation. This creates
    a deterministic cache key based on all factors that affect selection:
    - User message content
    - Last 2 conversation messages (for context)
    - Conversation depth (affects adaptive thresholds)
    - Current stage (affects tool filtering)
    - Selection method (semantic vs keyword)

    Args:
        user_message: The user's input message
        conversation_history: Optional conversation history for context
        conversation_depth: Number of messages in conversation
        stage: Current conversation stage (e.g., "EXPLORATION", "EXECUTION")
        use_semantic: Whether semantic selection was used

    Returns:
        Stable hash key for caching tool selection results
    """
    # For conversation history, only use last 2 messages for context
    # Too much history makes cache keys too specific
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        # Take last 2 messages max
        recent_history = (
            conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
        )
        # Create a normalized string representation
        history_parts = []
        for msg in recent_history:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]  # Truncate long messages
                history_parts.append(f"{role}:{content}")
        history_context = "|".join(history_parts)

    # Normalize message (lowercase, strip extra whitespace)
    normalized_message = " ".join(user_message.lower().strip().split())

    # Bucket conversation depth into ranges to improve cache hit rate.
    # Exact depth changes every iteration, causing cache misses.
    if conversation_depth < 5:
        depth_bucket = "short"
    elif conversation_depth < 20:
        depth_bucket = "medium"
    else:
        depth_bucket = "long"

    # Build key parts
    key_parts = [
        ResultType.TOOL_SELECTION.value,
        normalized_message,
        history_context,
        depth_bucket,
        stage or "none",
        "semantic" if use_semantic else "keyword",
    ]

    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


class GenericResultCache:
    """Generic cache for non-tool results.

    Extends caching beyond tool results to include:
    - LLM response results
    - Embedding computation results
    - Parsing results
    - Validation results

    Features:
    - Result type tracking for selective invalidation
    - Path-based invalidation for dependent results
    - Dependency tracking for cascading invalidations
    - TTL support per result type
    - Thread-safe operations

    Integration Point:
        Use in ToolCoordinator for extended caching

    Example:
        cache = GenericResultCache()

        # Cache an LLM response
        cache.set(
            ResultType.LLM_RESPONSE,
            "chat_completion_123",
            {"response": "..."},
            ttl=300
        )

        # Cache with path dependency
        cache.set_with_dependencies(
            ResultType.CODE_ACTION,
            "edit_file_456",
            {"success": True},
            depends_on=["/path/to/file.py"],
            ttl=600
        )

        # Invalidate by path
        cache.invalidate_by_paths(["/path/to/file.py"])
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        default_ttl: int = 300,
        enable_dependency_tracking: bool = True,
    ):
        """Initialize the generic result cache.

        Args:
            config: Cache configuration (uses defaults if None)
            default_ttl: Default TTL in seconds for cached results
            enable_dependency_tracking: Whether to track dependencies
        """
        self._config = config or CacheConfig()
        self._default_ttl = default_ttl
        self._enable_dependency_tracking = enable_dependency_tracking

        # Initialize tiered cache
        self._cache = TieredCache(config=self._config)

        # Dependency tracking: key -> CacheDependency
        self._dependencies: Dict[str, CacheDependency] = {}
        self._dependency_lock = threading.RLock()

        # Path index: path -> set of cache keys (paths are normalized)
        self._path_index: Dict[str, Set[str]] = {}

        # Result type index: result_type -> set of keys
        self._type_index: Dict[ResultType, Set[str]] = {
            result_type: set() for result_type in ResultType
        }

        # Statistics
        self._stats: Dict[str, int | float] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
            "dependency_invalidations": 0,
        }

        logger.info(
            "GenericResultCache initialized: ttl=%ds, dependency_tracking=%s",
            default_ttl,
            enable_dependency_tracking,
        )

    @staticmethod
    def _normalize_path(path_str: str) -> str:
        """Normalize file paths for consistent cache key generation."""
        if "/" in path_str or "\\" in path_str:
            try:
                return str(Path(path_str).resolve())
            except (OSError, ValueError):
                return path_str
        return path_str

    def get(
        self,
        result_type: ResultType,
        identifier: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get a cached result.

        Args:
            result_type: Type of the result
            identifier: Unique identifier for the result
            params: Optional parameters that affect the result

        Returns:
            Cached result or None if not found
        """
        key = _create_cache_key(result_type, identifier, params)

        result = self._cache.get(key)
        if result is not None:
            with self._dependency_lock:
                self._stats["hits"] += 1
            logger.debug(f"Cache hit: {result_type.value}:{identifier}")
        else:
            with self._dependency_lock:
                self._stats["misses"] += 1
            logger.debug(f"Cache miss: {result_type.value}:{identifier}")

        return result

    def set(
        self,
        result_type: ResultType,
        identifier: str,
        value: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a cached result.

        Args:
            result_type: Type of the result
            identifier: Unique identifier for the result
            value: Result value to cache
            params: Optional parameters that affect the result
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = _create_cache_key(result_type, identifier, params)
        cache_ttl = ttl if ttl is not None else self._default_ttl

        self._cache.set(key, value, ttl=cache_ttl)

        with self._dependency_lock:
            self._stats["sets"] += 1
            self._type_index[result_type].add(key)

        logger.debug(f"Cached: {result_type.value}:{identifier} (ttl={cache_ttl}s)")

    def set_with_dependencies(
        self,
        result_type: ResultType,
        identifier: str,
        value: Any,
        depends_on: List[str],
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a cached result with dependency tracking.

        Args:
            result_type: Type of the result
            identifier: Unique identifier for the result
            value: Result value to cache
            depends_on: List of file paths or cache keys this depends on
            params: Optional parameters
            ttl: Time-to-live in seconds
        """
        key = _create_cache_key(result_type, identifier, params)
        cache_ttl = ttl if ttl is not None else self._default_ttl

        # Cache the result
        self._cache.set(key, value, ttl=cache_ttl)

        if self._enable_dependency_tracking:
            with self._dependency_lock:
                # Create dependency record
                dependency = CacheDependency(
                    key=key,
                    depends_on=depends_on,
                    result_type=result_type,
                )
                self._dependencies[key] = dependency

                # Update path index for path-based dependencies
                for path_str in depends_on:
                    # Check if it looks like a file path
                    if "/" in path_str or "\\" in path_str:
                        normalized = self._normalize_path(path_str)
                        self._path_index.setdefault(normalized, set()).add(key)

                # Update type index
                self._type_index[result_type].add(key)
                self._stats["sets"] += 1

        logger.debug(
            f"Cached with dependencies: {result_type.value}:{identifier} "
            f"(depends_on={len(depends_on)}, ttl={cache_ttl}s)"
        )

    def invalidate(
        self,
        result_type: ResultType,
        identifier: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Invalidate a specific cached result.

        Args:
            result_type: Type of the result
            identifier: Unique identifier for the result
            params: Optional parameters

        Returns:
            True if key was found and invalidated, False otherwise
        """
        key = _create_cache_key(result_type, identifier, params)

        # Remove from cache
        existed = self._cache.get(key) is not None
        self._cache.delete(key)

        # Remove dependencies
        with self._dependency_lock:
            if key in self._dependencies:
                del self._dependencies[key]
            self._stats["invalidations"] += 1

            # Remove from type index
            self._type_index[result_type].discard(key)

        logger.debug(f"Invalidated: {result_type.value}:{identifier}")
        return existed

    def invalidate_by_type(self, result_type: ResultType) -> int:
        """Invalidate all cached results of a specific type.

        Args:
            result_type: Type of results to invalidate

        Returns:
            Number of keys invalidated
        """
        with self._dependency_lock:
            keys_to_invalidate = self._type_index.get(result_type, set()).copy()

        for key in keys_to_invalidate:
            self._cache.delete(key)
            if key in self._dependencies:
                del self._dependencies[key]

        self._type_index[result_type].clear()

        count = len(keys_to_invalidate)
        self._stats["invalidations"] += count
        logger.info(f"Invalidated {count} results of type: {result_type.value}")
        return count

    def invalidate_by_paths(self, paths: List[str]) -> int:
        """Invalidate results that depend on given paths.

        Args:
            paths: List of file paths to invalidate

        Returns:
            Number of cache entries invalidated
        """
        keys_to_invalidate = set()

        with self._dependency_lock:
            for path in paths:
                path_str = self._normalize_path(str(path))
                # Get keys that depend on this path
                dependent_keys = self._path_index.get(path_str, set())
                keys_to_invalidate.update(dependent_keys)

                # Clear path index entry
                if path_str in self._path_index:
                    del self._path_index[path_str]

            # Also cascade invalidate dependent results
            visited = set()
            to_process = list(keys_to_invalidate)

            while to_process:
                current_key = to_process.pop()
                if current_key in visited:
                    continue

                visited.add(current_key)
                keys_to_invalidate.add(current_key)

                # Find what depends on this key
                for dep in self._dependencies.values():
                    if current_key in dep.depends_on:
                        to_process.append(dep.key)

        # Invalidate all affected keys
        for key in keys_to_invalidate:
            self._cache.delete(key)
            if key in self._dependencies:
                del self._dependencies[key]

        count = len(keys_to_invalidate)
        self._stats["dependency_invalidations"] += count
        logger.info(f"Invalidated {count} results by path dependency")

        return count

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a given prefix.

        Args:
            prefix: Key prefix to invalidate

        Returns:
            Number of keys invalidated
        """
        # This is expensive - requires scanning all keys
        # In practice, use type-based or path-based invalidation instead
        count = 0

        with self._dependency_lock:
            # Scan type index for matching keys
            for result_type, keys in self._type_index.items():
                for key in list(keys):
                    if key.startswith(prefix):
                        self._cache.delete(key)
                        keys.discard(key)
                        if key in self._dependencies:
                            del self._dependencies[key]
                        count += 1

        self._stats["invalidations"] += count
        logger.info(f"Invalidated {count} results by prefix: {prefix}")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._dependency_lock:
            stats_copy = self._stats.copy()
            stats_copy["dependency_count"] = len(self._dependencies)
            stats_copy["path_count"] = len(self._path_index)
            stats_copy["type_counts"] = {
                rt.value: len(keys) for rt, keys in self._type_index.items()
            }
            return stats_copy

    def clear(self) -> None:
        """Clear all cached results and dependencies."""
        self._cache.clear()

        with self._dependency_lock:
            self._dependencies.clear()
            self._path_index.clear()
            for result_type in self._type_index:
                self._type_index[result_type].clear()

        logger.info("Cleared all cached results")

    def get_dependencies(
        self, result_type: ResultType, identifier: str
    ) -> Optional[CacheDependency]:
        """Get dependency information for a cached result.

        Args:
            result_type: Type of the result
            identifier: Unique identifier

        Returns:
            CacheDependency or None if not found
        """
        key = _create_cache_key(result_type, identifier)
        return self._dependencies.get(key)

    def get_dependent_keys(self, cache_key: str) -> List[str]:
        """Get all keys that depend on a given cache key.

        Args:
            cache_key: Cache key to find dependents for

        Returns:
            List of dependent cache keys
        """
        dependents = []

        for dep in self._dependencies.values():
            if cache_key in dep.depends_on:
                dependents.append(dep.key)

        return dependents


# Cache key utilities for common result types


class CacheKeys:
    """Utility class for generating cache keys for common result types."""

    @staticmethod
    def llm_response(model: str, prompt_hash: str, **kwargs) -> str:
        """Generate cache key for LLM response."""
        return _create_cache_key(
            ResultType.LLM_RESPONSE,
            f"{model}:{prompt_hash}",
            kwargs if kwargs else None,
        )

    @staticmethod
    def embedding(text: str, model: str = "default") -> str:
        """Generate cache key for embedding result."""
        return _create_cache_key(
            ResultType.EMBEDDING,
            f"{model}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
            {"model": model},
        )

    @staticmethod
    def parsing(file_path: str, parse_type: str) -> str:
        """Generate cache key for parsing result."""
        return _create_cache_key(
            ResultType.PARSING,
            f"{parse_type}:{Path(file_path).name}",
            {"path": file_path, "type": parse_type},
        )

    @staticmethod
    def validation(target: str, validation_type: str) -> str:
        """Generate cache key for validation result."""
        return _create_cache_key(
            ResultType.VALIDATION,
            f"{validation_type}:{target}",
            {"target": target, "type": validation_type},
        )

    @staticmethod
    def code_action(file_path: str, action: str) -> str:
        """Generate cache key for code action result."""
        return _create_cache_key(
            ResultType.CODE_ACTION,
            f"{action}:{Path(file_path).name}",
            {"path": file_path, "action": action},
        )


__all__ = [
    "GenericResultCache",
    "ResultType",
    "CacheDependency",
    "InvalidationStrategy",
    "CacheKeys",
]
