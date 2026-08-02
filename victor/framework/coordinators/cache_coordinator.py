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

"""Cache Coordinator.

Handles workflow caching configuration and management, including:
- Result caching (for executed workflow outputs)
- Definition caching (for parsed YAML workflow definitions)
- Graph caching (for compiled StateGraph instances)

Features:
- Enable/disable result caching with configurable TTL
- Clear workflow caches
- Get cache statistics
- Integration with WorkflowCacheManager, WorkflowDefinitionCache, and CompiledGraphCache
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

if TYPE_CHECKING:
    from victor.framework.graph_cache import CompiledGraphCache
    from victor.workflows.cache import WorkflowCacheManager, WorkflowDefinitionCache

logger = logging.getLogger(__name__)


class CacheCoordinator:
    """Coordinator for workflow cache management.

    Provides a unified interface for managing workflow caches, including:
    - Result caching (executed workflow outputs)
    - Definition caching (parsed YAML workflow definitions)
    - Graph caching (compiled StateGraph instances)
    - Cache configuration (enable/disable, TTL)
    - Cache statistics and monitoring

    Example:
        coordinator = CacheCoordinator()

        # Enable caching with 1-hour TTL
        coordinator.enable_caching(ttl_seconds=3600)

        # Get cache stats
        stats = coordinator.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # Clear all caches
        coordinator.clear_cache()
    """

    def __init__(
        self,
        cache_manager: Optional["WorkflowCacheManager"] = None,
        definition_cache: Optional["WorkflowDefinitionCache"] = None,
        graph_cache: Optional["CompiledGraphCache"] = None,
    ) -> None:
        """Initialize the cache coordinator.

        Args:
            cache_manager: Optional WorkflowCacheManager for result caching
            definition_cache: Optional WorkflowDefinitionCache for definition caching
            graph_cache: Optional CompiledGraphCache for compiled graph caching
        """
        self._cache_manager = cache_manager
        self._definition_cache = definition_cache
        self._graph_cache = graph_cache
        self._caching_enabled = True
        self._cache_ttl_seconds = 3600

    @property
    def caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._caching_enabled

    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds."""
        return self._cache_ttl_seconds

    def _get_cache_manager(self) -> "WorkflowCacheManager":
        """Get or create cache manager."""
        if self._cache_manager is None:
            from victor.workflows.cache import get_workflow_cache_manager

            self._cache_manager = get_workflow_cache_manager()
        return self._cache_manager

    def _get_definition_cache(self) -> "WorkflowDefinitionCache":
        """Get or create definition cache."""
        if self._definition_cache is None:
            from victor.workflows.cache import get_workflow_definition_cache

            self._definition_cache = get_workflow_definition_cache()
        return self._definition_cache

    def _get_graph_cache(self) -> "CompiledGraphCache":
        """Get or create compiled graph cache."""
        if self._graph_cache is None:
            from victor.framework.graph_cache import get_compiled_graph_cache

            self._graph_cache = get_compiled_graph_cache()
        return self._graph_cache

    def enable_caching(self, ttl_seconds: int = 3600) -> None:
        """Enable result caching.

        Enables caching of workflow execution results with the specified
        TTL. Already-cached results will still be used until they expire.

        Args:
            ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
        """
        self._caching_enabled = True
        self._cache_ttl_seconds = ttl_seconds
        logger.info(f"Workflow caching enabled with TTL={ttl_seconds}s")

    def disable_caching(self) -> None:
        """Disable result caching.

        Disables caching of new workflow execution results. Already-cached
        results will still be used until they expire or are cleared.
        """
        self._caching_enabled = False
        logger.info("Workflow caching disabled")

    def clear_cache(self) -> int:
        """Clear all cached results.

        Clears result caches, definition caches, and graph caches.

        Returns:
            Total number of entries cleared
        """
        total_cleared = 0

        # Clear result caches
        if self._cache_manager:
            result_cleared = self._cache_manager.clear_all()
            total_cleared += result_cleared
            logger.debug(f"Cleared {result_cleared} result cache entries")

        # Clear definition cache
        if self._definition_cache:
            def_cleared = self._definition_cache.clear()
            total_cleared += def_cleared
            logger.debug(f"Cleared {def_cleared} definition cache entries")

        # Clear graph cache
        if self._graph_cache:
            graph_cleared = self._graph_cache.invalidate_all()
            total_cleared += graph_cleared
            logger.debug(f"Cleared {graph_cleared} graph cache entries")

        logger.info(f"Cleared {total_cleared} total workflow cache entries")
        return total_cleared

    def clear_workflow_cache(self, workflow_name: str) -> int:
        """Clear cache for a specific workflow.

        Args:
            workflow_name: Name of the workflow to clear

        Returns:
            Number of entries cleared
        """
        if self._cache_manager:
            count = self._cache_manager.clear_workflow(workflow_name)
            logger.debug(f"Cleared {count} cache entries for workflow: {workflow_name}")
            return count
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns comprehensive statistics for result caching,
        definition caching, and graph caching.

        Returns:
            Dictionary with cache statistics:
            - enabled: Whether caching is enabled
            - ttl_seconds: Current TTL setting
            - result_caches: Stats per workflow
            - definition_cache: Definition cache stats
            - graph_cache: Compiled graph cache stats
        """
        stats: Dict[str, Any] = {
            "enabled": self._caching_enabled,
            "ttl_seconds": self._cache_ttl_seconds,
            "result_caches": {},
            "definition_cache": {},
            "graph_cache": {},
        }

        # Get result cache stats
        if self._cache_manager:
            stats["result_caches"] = self._cache_manager.get_all_stats()

        # Get definition cache stats
        if self._definition_cache:
            stats["definition_cache"] = self._definition_cache.get_stats()

        # Get graph cache stats
        if self._graph_cache:
            stats["graph_cache"] = self._graph_cache.get_stats()

        return stats

    def get_result_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get result cache statistics for all workflows.

        Returns:
            Dictionary mapping workflow names to their stats
        """
        if self._cache_manager:
            return self._cache_manager.get_all_stats()
        return {}

    def get_definition_cache_stats(self) -> Dict[str, Any]:
        """Get definition cache statistics.

        Returns:
            Dictionary with definition cache stats
        """
        if self._definition_cache:
            return self._definition_cache.get_stats()
        return {}

    def get_graph_cache_stats(self) -> Dict[str, Any]:
        """Get compiled graph cache statistics.

        Returns:
            Dictionary with graph cache stats
        """
        if self._graph_cache:
            return self._graph_cache.get_stats()
        return {}

    def set_cache_manager(self, cache_manager: "WorkflowCacheManager") -> None:
        """Set custom cache manager.

        Args:
            cache_manager: WorkflowCacheManager instance
        """
        self._cache_manager = cache_manager

    def set_definition_cache(self, definition_cache: "WorkflowDefinitionCache") -> None:
        """Set custom definition cache.

        Args:
            definition_cache: WorkflowDefinitionCache instance
        """
        self._definition_cache = definition_cache

    def set_graph_cache(self, graph_cache: "CompiledGraphCache") -> None:
        """Set custom compiled graph cache.

        Args:
            graph_cache: CompiledGraphCache instance
        """
        self._graph_cache = graph_cache


__all__ = ["CacheCoordinator"]
