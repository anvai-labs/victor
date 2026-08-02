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

"""Workflow node caching for deterministic workflow steps.

Provides caching for workflow nodes that have deterministic outputs
given their inputs, improving performance for repeated workflow executions.

Cacheable node types:
- TransformNode: Pure functions that transform context data
- ConditionNode: Deterministic branching based on context

Non-cacheable node types:
- AgentNode: Non-deterministic (LLM responses vary)
- ParallelNode: Orchestration node (no direct output)
- HITLNode: Human-in-the-loop (inherently non-deterministic)

Example:
    from victor.workflows.cache import WorkflowCache, WorkflowCacheConfig

    # Create cache with custom config
    config = WorkflowCacheConfig(enabled=True, ttl_seconds=3600)
    cache = WorkflowCache(config)

    # Check for cached result
    result = cache.get(node, context)
    if result is not None:
        return result  # Cache hit

    # Execute node and cache result
    result = await execute_node(node, context)
    cache.set(node, context, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from cachetools import TTLCache  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from victor_contracts.workflows import NodeResult
    from victor.workflows.definition import (
        ConditionNode,
        TransformNode,
        WorkflowDefinition,
        WorkflowNode,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Dependency Graph for Cascading Cache Invalidation
# =============================================================================


@dataclass
class DependencyGraph:
    """Tracks node dependencies for cascading cache invalidation.

    When a node's output changes, all downstream nodes that depend on it
    should also be invalidated. This graph tracks those relationships.

    All verticals benefit from automatic dependency-aware invalidation
    without needing to manually track which cache entries to clear.

    Attributes:
        dependents: Maps node_id -> set of nodes that depend on it (downstream)
        dependencies: Maps node_id -> set of nodes it depends on (upstream)

    Example:
        # Build graph from workflow
        graph = DependencyGraph.from_workflow(workflow)

        # When node 'A' changes, find all affected nodes
        cascade = graph.get_cascade_set('A')
        # Returns {'B', 'C', 'D'} if B->A, C->B, D->C
    """

    dependents: Dict[str, Set[str]] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)

    def add_dependency(self, node_id: str, depends_on: str) -> None:
        """Record that node_id depends on depends_on.

        Args:
            node_id: The dependent node (downstream)
            depends_on: The dependency node (upstream)
        """
        if node_id not in self.dependencies:
            self.dependencies[node_id] = set()
        self.dependencies[node_id].add(depends_on)

        if depends_on not in self.dependents:
            self.dependents[depends_on] = set()
        self.dependents[depends_on].add(node_id)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get nodes that this node depends on (upstream)."""
        return self.dependencies.get(node_id, set())

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get nodes that depend on this node (downstream)."""
        return self.dependents.get(node_id, set())

    def get_cascade_set(self, node_id: str) -> Set[str]:
        """Get all nodes that should be invalidated when node_id changes.

        Performs breadth-first traversal to find all downstream dependents.

        Args:
            node_id: The node that changed

        Returns:
            Set of all node IDs that should be invalidated
        """
        result: Set[str] = set()
        to_process = list(self.dependents.get(node_id, set()))

        while to_process:
            current = to_process.pop(0)
            if current not in result:
                result.add(current)
                # Add all dependents of this node too
                to_process.extend(self.dependents.get(current, set()))

        return result

    def get_all_upstream(self, node_id: str) -> Set[str]:
        """Get all upstream dependencies (transitive)."""
        result: Set[str] = set()
        to_process = list(self.dependencies.get(node_id, set()))

        while to_process:
            current = to_process.pop(0)
            if current not in result:
                result.add(current)
                to_process.extend(self.dependencies.get(current, set()))

        return result

    @classmethod
    def from_workflow(cls, workflow: "WorkflowDefinition") -> "DependencyGraph":
        """Build dependency graph from workflow definition.

        Analyzes next_nodes, branches, and parallel_nodes to determine
        which nodes depend on which.

        Args:
            workflow: The workflow definition to analyze

        Returns:
            DependencyGraph with all node relationships
        """
        graph = cls()

        for node_id, node in workflow.nodes.items():
            # next_nodes: subsequent nodes depend on current node
            next_nodes = getattr(node, "next_nodes", None) or []
            for next_id in next_nodes:
                graph.add_dependency(next_id, node_id)

            # branches: branch targets depend on condition node
            branches = getattr(node, "branches", None) or {}
            for branch_target in branches.values():
                if branch_target:
                    graph.add_dependency(branch_target, node_id)

            # parallel_nodes: don't depend on each other
            # They're executed concurrently, not sequentially

        return graph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for debugging/storage."""
        return {
            "dependents": {k: list(v) for k, v in self.dependents.items()},
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
        }


class CascadingInvalidator:
    """Handles cascading cache invalidation based on dependencies.

    When a node's output changes, this class automatically invalidates
    all cache entries for dependent nodes, ensuring consistency.

    All verticals benefit from automatic cascading invalidation.

    Example:
        graph = DependencyGraph.from_workflow(workflow)
        invalidator = CascadingInvalidator(cache, graph)

        # When node 'A' output changes
        count = invalidator.invalidate_with_cascade('A')
        # Invalidates A and all downstream dependents
    """

    def __init__(self, cache: "WorkflowCache", graph: DependencyGraph):
        """Initialize cascading invalidator.

        Args:
            cache: The workflow cache to invalidate
            graph: Dependency graph for cascade calculation
        """
        self.cache = cache
        self.graph = graph

    def invalidate_with_cascade(self, node_id: str) -> int:
        """Invalidate node and all dependent nodes.

        Args:
            node_id: The node that changed

        Returns:
            Total number of cache entries invalidated
        """
        # Get all nodes to invalidate
        to_invalidate = {node_id} | self.graph.get_cascade_set(node_id)
        total = 0

        for nid in to_invalidate:
            count = self.cache.invalidate(nid)
            total += count
            if count > 0:
                logger.debug(f"Invalidated {count} cache entries for node {nid}")

        if total > 0:
            logger.debug(
                f"Cascade invalidation from '{node_id}': "
                f"{len(to_invalidate)} nodes, {total} entries"
            )

        return total

    def invalidate_upstream(self, node_id: str) -> int:
        """Invalidate all upstream dependencies (and their dependents).

        Use when a node's inputs have become invalid.

        Args:
            node_id: The node whose inputs are invalid

        Returns:
            Total entries invalidated
        """
        upstream = self.graph.get_all_upstream(node_id)
        total = 0

        for nid in upstream:
            # For each upstream, invalidate it and its cascade
            total += self.invalidate_with_cascade(nid)

        return total


@dataclass
class WorkflowCacheConfig:
    """Configuration for workflow node caching.

    Attributes:
        enabled: Whether caching is enabled (default: False)
        ttl_seconds: Time-to-live for cache entries in seconds
        max_entries: Maximum number of entries in memory cache
        cacheable_node_types: Node types that can be cached
        excluded_context_keys: Context keys to exclude from cache key generation
    """

    enabled: bool = False
    ttl_seconds: int = 3600  # 1 hour default
    max_entries: int = 500
    cacheable_node_types: Set[str] = field(default_factory=lambda: {"transform", "condition"})
    excluded_context_keys: Set[str] = field(
        default_factory=lambda: {"_internal", "_debug", "_timestamps"}
    )


@dataclass
class WorkflowNodeCacheEntry:
    """A cached node result.

    Attributes:
        node_id: ID of the cached node
        result: The cached NodeResult
        created_at: Timestamp when entry was created
        hit_count: Number of times this entry was accessed
    """

    node_id: str
    result: "NodeResult"
    created_at: float
    hit_count: int = 0


class WorkflowCache:
    """Cache for workflow node execution results.

    Provides in-memory caching for deterministic workflow nodes
    (TransformNode, ConditionNode) to avoid redundant computation
    during workflow re-execution.

    Thread-safe implementation using locks.

    Attributes:
        config: Cache configuration
        stats: Cache statistics (hits, misses, etc.)

    Example:
        cache = WorkflowCache(WorkflowCacheConfig(enabled=True))

        # Try to get cached result
        result = cache.get(node, context)
        if result is None:
            result = await execute_node(node, context)
            cache.set(node, context, result)
    """

    def __init__(self, config: Optional[WorkflowCacheConfig] = None):
        """Initialize workflow cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or WorkflowCacheConfig()
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "skipped_non_cacheable": 0,
        }

        # Initialize cache if enabled
        if self.config.enabled:
            self._cache: Optional[TTLCache] = TTLCache(
                maxsize=self.config.max_entries,
                ttl=self.config.ttl_seconds,
            )
            logger.debug(
                f"Workflow cache initialized: max_entries={self.config.max_entries}, "
                f"ttl={self.config.ttl_seconds}s"
            )
        else:
            self._cache = None
            logger.debug("Workflow cache disabled")

    def is_cacheable(self, node: "WorkflowNode") -> bool:
        """Check if a node type is cacheable.

        Args:
            node: The workflow node to check

        Returns:
            True if the node type can be cached
        """
        if not self.config.enabled:
            return False

        node_type = node.node_type.value if hasattr(node, "node_type") else "unknown"
        return node_type in self.config.cacheable_node_types

    def get(
        self,
        node: "WorkflowNode",
        context: Dict[str, Any],
    ) -> Optional["NodeResult"]:
        """Get cached result for a node.

        Args:
            node: The workflow node
            context: Current workflow context

        Returns:
            Cached NodeResult or None if not found
        """
        if not self.config.enabled or self._cache is None:
            return None

        if not self.is_cacheable(node):
            self._stats["skipped_non_cacheable"] += 1
            return None

        cache_key = self._make_cache_key(node, context)

        with self._lock:
            try:
                entry: Optional[WorkflowNodeCacheEntry] = self._cache.get(cache_key)
                if entry is not None:
                    entry.hit_count += 1
                    self._stats["hits"] += 1
                    logger.debug(f"Workflow cache hit: node={node.id}, key={cache_key[:16]}...")
                    return entry.result
            except KeyError:
                pass

            self._stats["misses"] += 1
            logger.debug(f"Workflow cache miss: node={node.id}")
            return None

    def set(
        self,
        node: "WorkflowNode",
        context: Dict[str, Any],
        result: "NodeResult",
    ) -> bool:
        """Cache a node result.

        Args:
            node: The workflow node
            context: Workflow context at execution time
            result: The execution result to cache

        Returns:
            True if successfully cached
        """
        if not self.config.enabled or self._cache is None:
            return False

        if not self.is_cacheable(node):
            return False

        # Don't cache failed results
        if not result.success:
            logger.debug(f"Skipping cache for failed node: {node.id}")
            return False

        cache_key = self._make_cache_key(node, context)

        with self._lock:
            entry = WorkflowNodeCacheEntry(
                node_id=node.id,
                result=result,
                created_at=time.time(),
            )

            try:
                self._cache[cache_key] = entry
                self._stats["sets"] += 1
                logger.debug(f"Workflow cache set: node={node.id}, key={cache_key[:16]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed to cache workflow result: {e}")
                return False

    def invalidate(self, node_id: str) -> int:
        """Invalidate all cache entries for a specific node.

        Args:
            node_id: ID of the node to invalidate

        Returns:
            Number of entries invalidated
        """
        if not self.config.enabled or self._cache is None:
            return 0

        with self._lock:
            keys_to_delete = [
                k
                for k, v in self._cache.items()
                if isinstance(v, WorkflowNodeCacheEntry) and v.node_id == node_id
            ]

            count = 0
            for key in keys_to_delete:
                try:
                    del self._cache[key]
                    count += 1
                except KeyError:
                    pass

            if count > 0:
                logger.debug(f"Invalidated {count} cache entries for node: {node_id}")

            return count

    def set_dependency_graph(self, graph: DependencyGraph) -> None:
        """Set dependency graph for cascading invalidation.

        When a dependency graph is set, the cache gains the ability to
        automatically invalidate all dependent cache entries when a
        node's output changes.

        Args:
            graph: Dependency graph built from workflow definition
        """
        self._dependency_graph = graph
        self._invalidator = CascadingInvalidator(self, graph)
        logger.debug("Dependency graph set for cascading invalidation")

    def invalidate_cascade(self, node_id: str) -> int:
        """Invalidate node and all dependent nodes.

        If a dependency graph has been set via `set_dependency_graph()`,
        this method will invalidate the specified node and all nodes
        that depend on it (cascading invalidation).

        If no dependency graph is set, falls back to simple invalidation.

        Args:
            node_id: ID of the node whose output changed

        Returns:
            Total number of cache entries invalidated
        """
        if hasattr(self, "_invalidator") and self._invalidator is not None:
            return self._invalidator.invalidate_with_cascade(node_id)
        return self.invalidate(node_id)

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        if not self.config.enabled or self._cache is None:
            return 0

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"Cleared {count} workflow cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            stats = self._stats.copy()

            total = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0

            if self._cache is not None:
                stats["current_size"] = len(self._cache)
                stats["max_size"] = self.config.max_entries
            else:
                stats["current_size"] = 0
                stats["max_size"] = 0

            stats["enabled"] = self.config.enabled

            return stats

    def _make_cache_key(
        self,
        node: "WorkflowNode",
        context: Dict[str, Any],
    ) -> str:
        """Generate cache key for a node execution.

        Cache key is based on:
        - Node ID (unique identifier)
        - Node type
        - Relevant context values (filtered)

        Args:
            node: The workflow node
            context: Current workflow context

        Returns:
            Hash-based cache key
        """
        # Get relevant context for this node type
        relevant_context = self._get_relevant_context(node, context)

        # Build key data
        key_data = {
            "node_id": node.id,
            "node_type": (node.node_type.value if hasattr(node, "node_type") else "unknown"),
            "context": relevant_context,
        }

        # Add node-specific data for more accurate cache matching
        if hasattr(node, "transform"):
            # For transform nodes, include function identity if possible
            key_data["transform_name"] = getattr(node.transform, "__name__", "anonymous")
        elif hasattr(node, "condition"):
            # For condition nodes, include condition function identity
            key_data["condition_name"] = getattr(node.condition, "__name__", "anonymous")

        # Generate hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_relevant_context(
        self,
        node: "WorkflowNode",
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant context values for cache key.

        Filters out excluded keys and limits depth to avoid
        huge cache keys.

        Args:
            node: The workflow node
            context: Full workflow context

        Returns:
            Filtered context dictionary
        """
        relevant = {}

        for key, value in context.items():
            # Skip excluded keys
            if key in self.config.excluded_context_keys:
                continue

            # Skip private keys
            if key.startswith("_"):
                continue

            # Limit value size for large strings/lists
            if isinstance(value, str) and len(value) > 1000:
                # Use hash for large strings
                relevant[key] = f"hash:{hashlib.sha256(value.encode()).hexdigest()[:16]}"
            elif isinstance(value, (list, dict)) and len(str(value)) > 1000:
                # Use hash for large collections
                relevant[key] = f"hash:{hashlib.sha256(str(value).encode()).hexdigest()[:16]}"
            else:
                relevant[key] = value

        return relevant


class WorkflowCacheManager:
    """Manager for workflow caching with namespace support.

    Provides a higher-level interface for managing workflow caches
    across multiple workflows with namespace isolation.

    Example:
        manager = WorkflowCacheManager()
        cache = manager.get_cache("my_workflow")
        result = cache.get(node, context)
    """

    def __init__(self, default_config: Optional[WorkflowCacheConfig] = None):
        """Initialize cache manager.

        Args:
            default_config: Default configuration for new caches
        """
        self._default_config = default_config or WorkflowCacheConfig()
        self._caches: Dict[str, WorkflowCache] = {}
        self._lock = threading.Lock()

    def get_cache(
        self,
        workflow_name: str,
        config: Optional[WorkflowCacheConfig] = None,
    ) -> WorkflowCache:
        """Get or create cache for a workflow.

        Args:
            workflow_name: Name of the workflow
            config: Optional override configuration

        Returns:
            WorkflowCache instance
        """
        with self._lock:
            if workflow_name not in self._caches:
                cache_config = config or self._default_config
                self._caches[workflow_name] = WorkflowCache(cache_config)
                logger.debug(f"Created workflow cache for: {workflow_name}")

            return self._caches[workflow_name]

    def clear_workflow(self, workflow_name: str) -> int:
        """Clear cache for a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Number of entries cleared
        """
        with self._lock:
            if workflow_name in self._caches:
                return self._caches[workflow_name].clear()
            return 0

    def clear_all(self) -> int:
        """Clear all workflow caches.

        Returns:
            Total entries cleared across all workflows
        """
        with self._lock:
            total = 0
            for cache in self._caches.values():
                total += cache.clear()
            return total

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workflow caches.

        Returns:
            Dictionary mapping workflow names to their stats
        """
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}


# =============================================================================
# Workflow Definition Cache (P1 - Scalability)
# =============================================================================


@dataclass
class DefinitionCacheConfig:
    """Configuration for workflow definition caching.

    Attributes:
        enabled: Whether caching is enabled
        ttl_seconds: Time-to-live for cache entries
        max_entries: Maximum cached definitions
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 100

    @classmethod
    def from_settings(cls, settings: Any) -> "DefinitionCacheConfig":
        """Create config from Settings instance."""
        return cls(
            enabled=getattr(settings, "workflow_definition_cache_enabled", True),
            ttl_seconds=getattr(settings, "workflow_definition_cache_ttl", 3600),
            max_entries=getattr(settings, "workflow_definition_cache_max_entries", 100),
        )


class WorkflowDefinitionCache:
    """TTL + mtime-based cache for parsed WorkflowDefinitions.

    Caches parsed YAML workflow definitions to avoid redundant parsing.
    Invalidation is based on both TTL and file modification time.

    Example:
        cache = WorkflowDefinitionCache(config)

        # Check cache
        definition = cache.get(path, workflow_name, config_hash)
        if definition is None:
            # Parse YAML and cache
            definition = parse_workflow(path, workflow_name)
            cache.put(path, workflow_name, config_hash, definition)
    """

    def __init__(self, config: Optional[DefinitionCacheConfig] = None):
        """Initialize the definition cache.

        Args:
            config: Cache configuration
        """
        self._config = config or DefinitionCacheConfig()
        self._lock = threading.RLock()

        if self._config.enabled:
            self._cache: Optional[TTLCache] = TTLCache(
                maxsize=self._config.max_entries,
                ttl=self._config.ttl_seconds,
            )
            logger.debug(
                f"Workflow definition cache initialized: max_entries={self._config.max_entries}, "
                f"ttl={self._config.ttl_seconds}s"
            )
        else:
            self._cache = None
            logger.debug("Workflow definition cache disabled")

        # Statistics
        self._stats = {"hits": 0, "misses": 0, "invalidations": 0}

    def _make_key(
        self,
        path: Path,
        workflow_name: str,
        config_hash: int,
        mtime: float,
    ) -> str:
        """Generate cache key.

        Key includes file path, workflow name, config hash, and mtime
        to ensure cache invalidation on any change.
        """
        key_data = f"{path.resolve()}:{workflow_name}:{config_hash}:{mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        path: Path,
        workflow_name: str,
        config_hash: int,
    ) -> Optional["WorkflowDefinition"]:
        """Get cached workflow definition.

        Args:
            path: Path to YAML file
            workflow_name: Name of workflow in YAML
            config_hash: Hash of workflow config for cache key

        Returns:
            Cached WorkflowDefinition or None if not found/stale
        """
        if not self._config.enabled or self._cache is None:
            return None

        try:
            mtime = path.stat().st_mtime
        except (OSError, FileNotFoundError):
            return None

        key = self._make_key(path, workflow_name, config_hash, mtime)

        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._stats["hits"] += 1
                logger.debug(f"Definition cache hit: {workflow_name}")
                return result

            self._stats["misses"] += 1
            return None

    def put(
        self,
        path: Path,
        workflow_name: str,
        config_hash: int,
        definition: "WorkflowDefinition",
    ) -> bool:
        """Cache a workflow definition.

        Args:
            path: Path to YAML file
            workflow_name: Name of workflow
            config_hash: Hash of workflow config
            definition: Parsed WorkflowDefinition to cache

        Returns:
            True if cached successfully
        """
        if not self._config.enabled or self._cache is None:
            return False

        try:
            mtime = path.stat().st_mtime
        except (OSError, FileNotFoundError):
            return False

        key = self._make_key(path, workflow_name, config_hash, mtime)

        with self._lock:
            self._cache[key] = definition
            logger.debug(f"Definition cached: {workflow_name}")
            return True

    def invalidate(self, path: Path) -> int:
        """Invalidate all cached definitions for a file path.

        Args:
            path: Path to invalidate

        Returns:
            Number of entries invalidated
        """
        if not self._config.enabled or self._cache is None:
            return 0

        path_str = str(path.resolve())

        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if path_str in k]  # Key contains path

            for key in keys_to_delete:
                del self._cache[key]

            count = len(keys_to_delete)
            if count > 0:
                self._stats["invalidations"] += count
                logger.debug(f"Invalidated {count} definition cache entries for: {path}")

            return count

    def clear(self) -> int:
        """Clear all cached definitions.

        Returns:
            Number of entries cleared
        """
        if not self._config.enabled or self._cache is None:
            return 0

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"Cleared {count} definition cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            total = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
            stats["size"] = len(self._cache) if self._cache else 0
            stats["enabled"] = self._config.enabled
            return stats


# Global definition cache instance
_global_definition_cache: Optional[WorkflowDefinitionCache] = None


def get_workflow_definition_cache() -> WorkflowDefinitionCache:
    """Get the global workflow definition cache.

    Returns:
        Global WorkflowDefinitionCache instance
    """
    global _global_definition_cache
    if _global_definition_cache is None:
        _global_definition_cache = WorkflowDefinitionCache()
    return _global_definition_cache


def configure_workflow_definition_cache(config: DefinitionCacheConfig) -> None:
    """Configure the global workflow definition cache.

    Args:
        config: Configuration to apply
    """
    global _global_definition_cache
    _global_definition_cache = WorkflowDefinitionCache(config)
    logger.debug(f"Workflow definition cache configured: enabled={config.enabled}")


# Global cache manager instance
_global_cache_manager: Optional[WorkflowCacheManager] = None


def get_workflow_cache_manager() -> WorkflowCacheManager:
    """Get the global workflow cache manager.

    Returns:
        Global WorkflowCacheManager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = WorkflowCacheManager()
    return _global_cache_manager


def configure_workflow_cache(config: WorkflowCacheConfig) -> None:
    """Configure the global workflow cache manager.

    Args:
        config: Configuration to apply
    """
    global _global_cache_manager
    _global_cache_manager = WorkflowCacheManager(config)
    logger.debug(f"Workflow cache configured: enabled={config.enabled}")


__all__ = [
    # Dependency tracking
    "DependencyGraph",
    "CascadingInvalidator",
    # Cache config and entries
    "WorkflowCacheConfig",
    "WorkflowNodeCacheEntry",
    "WorkflowCache",
    "WorkflowCacheManager",
    # Global management
    "get_workflow_cache_manager",
    "configure_workflow_cache",
    # Definition cache
    "DefinitionCacheConfig",
    "WorkflowDefinitionCache",
    "get_workflow_definition_cache",
    "configure_workflow_definition_cache",
]
