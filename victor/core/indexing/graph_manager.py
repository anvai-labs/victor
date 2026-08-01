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

"""Graph management with automatic updates.

This module provides automatic graph invalidation and updates based on
file system changes detected by the file watcher service.

Pattern: Singleton + Observer + Cache + Repository
"""

from __future__ import annotations

import asyncio
import errno
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from victor.core.indexing.file_watcher import (
    FileChangeEvent,
    FileChangeType,
    FileWatcherRegistry,
)
from victor.core.indexing.index_lock import IndexLockRegistry

logger = logging.getLogger(__name__)

__all__ = ["GraphManager", "classify_refresh_error"]


def _extract_os_error_code(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an OS error number from an exception chain."""
    current: Optional[BaseException] = exc
    while current is not None:
        if isinstance(current, OSError) and current.errno is not None:
            return current.errno
        current = current.__cause__ or current.__context__
    return None


def classify_refresh_error(exc: Exception, *, failure_count: int = 1) -> Dict[str, Any]:
    """Classify a graph refresh failure for health tracking and retry policy."""
    error_code = _extract_os_error_code(exc)
    retry_delay_seconds = 10.0
    recoverable = True
    severity = "warning"
    category = "unexpected_refresh_failure"
    operator_guidance = "Inspect the graph refresh stack trace if the failure persists."

    if isinstance(exc, FileNotFoundError) or error_code == errno.ENOENT:
        category = "transient_missing_file"
        retry_delay_seconds = 0.5
        operator_guidance = "Retry after the filesystem settles; this commonly happens during deletes or temp-file churn."
    elif error_code in {errno.EMFILE, errno.ENFILE} or "Too many open files" in str(exc):
        category = "resource_exhaustion"
        retry_delay_seconds = min(60.0, 5.0 * max(1, failure_count))
        operator_guidance = (
            "Reduce concurrent watcher/index activity or raise the open-file limit before retrying."
        )
    elif isinstance(exc, TimeoutError):
        category = "lock_timeout"
        retry_delay_seconds = min(30.0, 2.0 * max(1, failure_count))
        operator_guidance = (
            "Another process may be indexing this project; retry after the lock clears."
        )
    else:
        recoverable = False
        severity = "error"
        operator_guidance = (
            "Inspect the exception and graph indexing pipeline for a deterministic bug."
        )

    return {
        "error_code": error_code,
        "category": category,
        "recoverable": recoverable,
        "severity": severity,
        "retry_delay_seconds": retry_delay_seconds,
        "operator_guidance": operator_guidance,
    }


class GraphManager:
    """Manages code graph with automatic updates.

    Provides automatic cache invalidation and incremental updates
    based on file system changes. Subscribes to file watcher events
    and maintains per-mode graph caching.

    Pattern: Singleton + Observer + Cache

    Example:
        >>> manager = GraphManager.get_instance()
        >>> graph, built = await manager.get_or_build_graph(
        ...     root=Path("/my/project"),
        ...     mode="pagerank",
        ...     exec_ctx=None
        ... )
        >>> # Graph is automatically updated when files change
    """

    _instance: Optional["GraphManager"] = None

    def __init__(self):
        """Initialize GraphManager."""
        self._graph_cache: Dict[str, Dict] = {}
        self._cache_lock = asyncio.Lock()
        self._watcher_subscribed: set[str] = set()
        self._watcher_callbacks: Dict[str, Any] = {}
        self._background_refresh: Dict[str, Dict[str, Any]] = {}
        self._refresh_tasks: Dict[str, asyncio.Task[None]] = {}
        self._refresh_failures: Dict[str, Dict[str, Any]] = {}
        self._last_refresh_completed_at: Dict[str, float] = {}
        self._last_refresh_source_mtime: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "GraphManager":
        """Get singleton instance.

        Returns:
            GraphManager singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_or_build_graph(
        self,
        root: Path,
        mode: str,
        exec_ctx: Optional[Dict] = None,
    ) -> Tuple[Dict[str, any], bool]:
        """Get cached graph or build/update it.

        Args:
            root: Root path of codebase
            mode: Graph mode (pagerank, centrality, trace, etc.)
            exec_ctx: Execution context

        Returns:
            Tuple of (graph, was_built)
        """
        root_str = str(root.resolve())
        cache_key = f"{root_str}:{mode}"

        # Check cache without lock (fast path)
        cache_entry = self._graph_cache.get(cache_key)
        if cache_entry and not cache_entry.get("stale", False):
            logger.info(f"[GraphManager] Cache hit for {cache_key}")
            return cache_entry["graph"], False

        # Acquire lock for this path and mode
        lock_registry = IndexLockRegistry.get_instance()
        # Use mode-specific lock path to allow concurrent different modes
        path_lock = await lock_registry.acquire_lock(root / mode)

        async with path_lock:
            # Double-check cache inside lock
            cache_entry = self._graph_cache.get(cache_key)
            if cache_entry and not cache_entry.get("stale", False):
                return cache_entry["graph"], False

            # Build graph
            logger.info(f"[GraphManager] Building graph for {cache_key}")

            # Subscribe to file changes and keep the persisted graph fresh.
            await self.ensure_background_refresh(root, exec_ctx=exec_ctx)

            try:
                from victor.core.utils.capability_loader import load_graph_tool_module

                module = load_graph_tool_module()
                graph = module.graph
            except ImportError:
                logger.warning("victor-coding not installed, returning empty graph")
                return {}, False

            try:
                result = await graph(
                    path=str(root),
                    mode=mode,
                    reindex=True,  # Force fresh graph
                    _exec_ctx=exec_ctx,
                )

                # Cache result
                self._graph_cache[cache_key] = {
                    "graph": result,
                    "built_at": datetime.now().timestamp(),
                    "stale": False,
                }

                logger.info(f"[GraphManager] Graph built for {cache_key}")
                return result, True

            except Exception as e:
                logger.error(f"[GraphManager] Failed to build graph for {cache_key}: {e}")
                raise

    async def ensure_background_refresh(
        self,
        root: Path,
        *,
        enable_ccg: bool = True,
        exec_ctx: Optional[Dict] = None,
        poll_interval_seconds: float = 1.0,
        debounce_seconds: float = 0.3,
        min_refresh_interval_seconds: float = 30.0,
        build_now: bool = False,
        on_refresh_complete: Optional[Any] = None,
        on_refresh_error: Optional[Any] = None,
    ) -> Optional[Any]:
        """Ensure file watching plus incremental persisted-graph refresh for a root.

        Args:
            root: Root path to watch and incrementally index
            enable_ccg: Whether refreshes should rebuild Code Context Graph data
            exec_ctx: Optional execution context
            poll_interval_seconds: Watcher poll interval
            debounce_seconds: Watcher debounce window
            min_refresh_interval_seconds: Minimum delay between successful refresh passes
            build_now: If True, run one incremental refresh immediately

        Returns:
            Initial refresh stats when ``build_now`` is True, otherwise None.
        """
        root = root.resolve()
        root_str = str(root)
        self._background_refresh[root_str] = {
            "enable_ccg": enable_ccg,
            "exec_ctx": exec_ctx,
            "pending": False,
            "on_refresh_complete": on_refresh_complete,
            "on_refresh_error": on_refresh_error,
            "min_refresh_interval_seconds": max(0.0, float(min_refresh_interval_seconds)),
        }

        initial_stats = None
        if build_now:
            initial_stats = await self._refresh_graph_index(root)

        await self._ensure_file_watcher(
            root,
            exec_ctx,
            poll_interval_seconds=poll_interval_seconds,
            debounce_seconds=debounce_seconds,
        )
        return initial_stats

    async def _ensure_file_watcher(
        self,
        root: Path,
        exec_ctx: Optional[Dict] = None,
        *,
        poll_interval_seconds: float = 1.0,
        debounce_seconds: float = 0.3,
    ) -> None:
        """Ensure file watcher is subscribed for this root.

        Args:
            root: Root path to watch
            exec_ctx: Execution context
            poll_interval_seconds: Poll interval for file detection
            debounce_seconds: Debounce interval before publishing events
        """
        root_str = str(root.resolve())

        # Check if already subscribed
        if root_str in self._watcher_subscribed:
            return

        # Get file watcher
        watcher_registry = FileWatcherRegistry.get_instance()
        file_watcher = await watcher_registry.get_watcher(
            root,
            poll_interval_seconds=poll_interval_seconds,
            debounce_seconds=debounce_seconds,
        )

        # Subscribe to file changes
        def callback(event: FileChangeEvent) -> asyncio.Task[Any]:
            return asyncio.create_task(self._on_file_change(event, root, exec_ctx))

        file_watcher.subscribe(callback)

        # Mark as subscribed
        self._watcher_subscribed.add(root_str)
        self._watcher_callbacks[root_str] = callback

        logger.info(f"[GraphManager] Subscribed to file watcher for {root_str}")

    async def _unsubscribe_file_watcher(self, root_str: str) -> None:
        """Detach GraphManager from a root watcher and stop it if no subscribers remain."""
        callback = self._watcher_callbacks.pop(root_str, None)
        self._watcher_subscribed.discard(root_str)
        if callback is None:
            return

        watcher_registry = FileWatcherRegistry.get_instance()
        watcher = getattr(watcher_registry, "_watchers", {}).get(root_str)
        if watcher is None:
            return

        watcher.unsubscribe(callback)
        logger.info(f"[GraphManager] Unsubscribed from file watcher for {root_str}")

        if watcher.get_stats().get("subscribers", 0) == 0:
            await watcher_registry.stop_watcher(Path(root_str))

    async def _on_file_change(
        self,
        event: FileChangeEvent,
        root: Path,
        exec_ctx: Optional[Dict] = None,
    ) -> None:
        """Handle file change event.

        Invalidates all cached graphs for this root.

        Args:
            event: File change event
            root: Root path of codebase
            exec_ctx: Execution context
        """
        root_str = str(root.resolve())

        # Mark all graphs for this root as stale
        stale_count = 0
        for key, cache_entry in self._graph_cache.items():
            if key.startswith(root_str):
                cache_entry["stale"] = True
                stale_count += 1
                logger.debug(f"[GraphManager] Graph marked stale: {key}")

        if root_str in self._background_refresh:
            self._schedule_background_refresh(root)

        if stale_count > 0:
            logger.info(
                f"[GraphManager] Marked {stale_count} graph(s) stale for {root_str} "
                f"due to {event.change_type.value}: {event.path}"
            )

    def _schedule_background_refresh(self, root: Path) -> None:
        """Schedule incremental refresh of the persisted graph for a root."""
        root_str = str(root.resolve())
        refresh_config = self._background_refresh.get(root_str)
        if refresh_config is None:
            return

        active_task = self._refresh_tasks.get(root_str)
        if active_task is not None and not active_task.done():
            refresh_config["pending"] = True
            return

        refresh_config["pending"] = False
        retry_delay_seconds = 0.0
        failure = self._refresh_failures.get(root_str)
        if isinstance(failure, dict):
            next_retry_at = failure.get("next_retry_at")
            if isinstance(next_retry_at, (int, float)):
                retry_delay_seconds = max(0.0, float(next_retry_at) - datetime.now().timestamp())
                if retry_delay_seconds > 0:
                    failure["suppressed_events"] = int(failure.get("suppressed_events", 0)) + 1

        cooldown_delay_seconds = self._refresh_cooldown_delay(root_str, refresh_config)
        startup_delay_seconds = max(retry_delay_seconds, cooldown_delay_seconds)

        if startup_delay_seconds > 0:
            delay_reason = (
                "retry_backoff" if retry_delay_seconds >= cooldown_delay_seconds else "cooldown"
            )
            logger.debug(
                "[GraphManager] Delaying background refresh for %s by %.2fs (%s)",
                root_str,
                startup_delay_seconds,
                delay_reason,
            )
            self._refresh_tasks[root_str] = asyncio.create_task(
                self._run_refresh_loop(root.resolve(), startup_delay_seconds=startup_delay_seconds)
            )
            return

        self._refresh_tasks[root_str] = asyncio.create_task(self._run_refresh_loop(root.resolve()))

    def _refresh_cooldown_delay(
        self,
        root_str: str,
        refresh_config: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Return remaining cooldown before another successful refresh pass may start."""
        config = refresh_config or self._background_refresh.get(root_str) or {}
        min_interval = float(config.get("min_refresh_interval_seconds", 0.0) or 0.0)
        if min_interval <= 0:
            return 0.0

        last_completed = self._last_refresh_completed_at.get(root_str)
        if last_completed is None:
            return 0.0

        elapsed = datetime.now().timestamp() - float(last_completed)
        return max(0.0, min_interval - elapsed)

    def _source_mtime_already_refreshed(self, root_str: str, source_mtime: float) -> bool:
        """Return True when the watched source tree has not advanced since last refresh."""
        last_source_mtime = self._last_refresh_source_mtime.get(root_str)
        if last_source_mtime is None:
            return False
        return float(last_source_mtime) >= float(source_mtime)

    async def _run_refresh_loop(self, root: Path, startup_delay_seconds: float = 0.0) -> None:
        """Run one or more coalesced incremental refresh passes for a root."""
        root_str = str(root.resolve())
        current_task = asyncio.current_task()
        try:
            if startup_delay_seconds > 0:
                await asyncio.sleep(startup_delay_seconds)

            while True:
                try:
                    await self._refresh_graph_index(root)
                except FileNotFoundError as exc:
                    logger.info(
                        "[GraphManager] Background refresh saw transient missing file for %s; retrying once: %s",
                        root_str,
                        exc,
                    )
                    await asyncio.sleep(0.1)
                    await self._refresh_graph_index(root)
                self._refresh_failures.pop(root_str, None)
                refresh_config = self._background_refresh.get(root_str)
                if not refresh_config or not refresh_config.get("pending", False):
                    break
                refresh_config["pending"] = False
                cooldown_delay_seconds = self._refresh_cooldown_delay(root_str, refresh_config)
                if cooldown_delay_seconds > 0:
                    await asyncio.sleep(cooldown_delay_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = self._record_refresh_failure(root, exc)
            await self._notify_refresh_error(root, exc)
            logger.info(
                "[GraphManager] Background refresh deferred for %s: category=%s recoverable=%s retry_in=%.2fs error=%s",
                root_str,
                failure.get("category", "unknown"),
                failure.get("recoverable", False),
                float(failure.get("retry_delay_seconds", 0.0) or 0.0),
                exc,
            )
        finally:
            task = self._refresh_tasks.get(root_str)
            if task is current_task:
                self._refresh_tasks.pop(root_str, None)

    def _record_refresh_failure(self, root: Path, exc: Exception) -> Dict[str, Any]:
        """Track a recoverable background refresh failure for a root."""
        root_str = str(root.resolve())
        existing = self._refresh_failures.get(root_str, {})
        count = int(existing.get("count", 0)) + 1
        classification = classify_refresh_error(exc, failure_count=count)
        failure = {
            "count": count,
            "last_error": str(exc),
            "error_type": type(exc).__name__,
            "last_failed_at": datetime.now().timestamp(),
            "suppressed_events": int(existing.get("suppressed_events", 0)),
            **classification,
        }
        failure["next_retry_at"] = failure["last_failed_at"] + float(
            failure.get("retry_delay_seconds", 0.0) or 0.0
        )
        self._refresh_failures[root_str] = failure
        return failure

    async def _notify_refresh_error(self, root: Path, exc: Exception) -> None:
        """Notify optional background refresh error callbacks."""
        refresh_config = self._background_refresh.get(str(root.resolve()))
        if not refresh_config:
            return

        callback = refresh_config.get("on_refresh_error")
        if not callable(callback):
            return

        callback_result = callback(root, exc)
        if inspect.isawaitable(callback_result):
            await callback_result

    @staticmethod
    def _background_embeddings_enabled() -> bool:
        """Whether background refreshes should also generate node embeddings.

        Gated by VICTOR_USE_GRAPH_EMBEDDINGS (opt-in: embedding costs model
        inference on every refresh). ``victor graph index --embeddings``
        bypasses this gate for explicit one-shot runs.
        """
        try:
            from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

            return get_feature_flag_manager().is_enabled(FeatureFlag.USE_GRAPH_EMBEDDINGS)
        except Exception:
            return False

    async def _refresh_graph_index(self, root: Path) -> Any:
        """Incrementally refresh the persisted graph and synthetic edges for a root."""
        from victor.core.graph_rag import (
            GraphIndexConfig,
            GraphIndexingPipeline,
            GraphIndexStats,
        )
        from victor.core.indexing.index_lock import IndexLockRegistry
        from victor.core.indexing.graph_enrichment import ensure_project_graph_enriched
        from victor.storage.graph import create_graph_store
        from victor.tools.common import latest_mtime

        root = root.resolve()
        root_str = str(root)
        refresh_config = self._background_refresh.get(root_str, {})
        enable_ccg = bool(refresh_config.get("enable_ccg", True))
        refresh_started_at = datetime.now().timestamp()
        repo_mtime = latest_mtime(root)
        stats = GraphIndexStats()

        if self._source_mtime_already_refreshed(root_str, repo_mtime):
            stats.processing_time_seconds = datetime.now().timestamp() - refresh_started_at
            logger.debug("[GraphManager] Source tree already current for %s", root_str)
        else:
            logger.info("[GraphManager] Starting incremental graph refresh for %s", root_str)
            phase1_start = datetime.now().timestamp()

            lock_registry = IndexLockRegistry.get_instance()
            path_lock = await lock_registry.acquire_lock(root)

            phase1_time = datetime.now().timestamp() - phase1_start
            logger.debug(
                "[GraphManager] Lock acquisition took %.2fs for %s",
                phase1_time,
                root_str,
            )

            async with path_lock:
                phase2_start = datetime.now().timestamp()

                # "auto" honors the per-repo .victor/graph_backend marker so a
                # repo flipped to ProximaDB is populated through the same
                # incremental refresh path; defaults to sqlite.
                graph_store = create_graph_store("auto", root)
                config = GraphIndexConfig(
                    root_path=root,
                    enable_ccg=enable_ccg,
                    enable_embeddings=self._background_embeddings_enabled(),
                    enable_subgraph_cache=False,
                    incremental=True,  # Use incremental updates (default, but explicit for clarity)
                )

                phase2_time = datetime.now().timestamp() - phase2_start
                logger.debug(
                    "[GraphManager] Graph store initialization took %.2fs for %s",
                    phase2_time,
                    root_str,
                )

                phase3_start = datetime.now().timestamp()
                pipeline = GraphIndexingPipeline(graph_store, config)
                stats = await pipeline.index_repository()

                phase3_time = datetime.now().timestamp() - phase3_start
                logger.debug(
                    "[GraphManager] Repository indexing took %.2fs for %s (parsed %d files)",
                    phase3_time,
                    root_str,
                    stats.files_processed + stats.files_unchanged,
                )

                phase4_start = datetime.now().timestamp()
                ensure_project_graph_enriched(root, latest_mtime=repo_mtime)

                phase4_time = datetime.now().timestamp() - phase4_start
                logger.debug(
                    "[GraphManager] Graph enrichment took %.2fs for %s",
                    phase4_time,
                    root_str,
                )

        total_time = datetime.now().timestamp() - refresh_started_at

        if stats.files_processed or stats.files_deleted:
            logger.info(
                "[GraphManager] Incremental graph refresh complete for %s "
                "(changed=%d deleted=%d unchanged=%d duration=%.2fs)",
                root_str,
                stats.files_processed,
                stats.files_deleted,
                stats.files_unchanged,
                total_time,
            )
            self._invalidate_query_cache(root_str)
        else:
            logger.debug(
                "[GraphManager] Graph already current for %s (check took %.2fs)",
                root_str,
                total_time,
            )

        callback = refresh_config.get("on_refresh_complete")
        if callable(callback):
            callback_result = callback(root, stats)
            if inspect.isawaitable(callback_result):
                await callback_result

        self._last_refresh_completed_at[root_str] = datetime.now().timestamp()
        self._last_refresh_source_mtime[root_str] = repo_mtime
        return stats

    @staticmethod
    def _invalidate_query_cache(root_str: str) -> None:
        """Drop cached graph query results for a root after its graph changed."""
        try:
            from victor.core.graph_rag.query_cache import get_graph_query_cache

            get_graph_query_cache().invalidate_repo(root_str)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("[GraphManager] Query cache invalidation failed for %s: %s", root_str, e)

    async def wait_for_refresh(self, root: Path, timeout: float = 5.0) -> bool:
        """Wait for an active background refresh to complete for a root."""
        root_str = str(root.resolve())
        task = self._refresh_tasks.get(root_str)
        if task is None:
            return True

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop_background_refresh(self, root: Optional[Path] = None) -> int:
        """Cancel active background refresh tasks and clear refresh configuration."""
        if root is None:
            roots = list(self._background_refresh.keys())
        else:
            roots = [str(root.resolve())]

        stopped = 0
        for root_str in roots:
            task = self._refresh_tasks.pop(root_str, None)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                stopped += 1
            self._background_refresh.pop(root_str, None)
            self._refresh_failures.pop(root_str, None)
            self._last_refresh_completed_at.pop(root_str, None)
            self._last_refresh_source_mtime.pop(root_str, None)
            await self._unsubscribe_file_watcher(root_str)

        return stopped

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about graph cache.

        Returns:
            Dict with current statistics
        """
        total_graphs = len(self._graph_cache)
        stale_graphs = sum(1 for e in self._graph_cache.values() if e.get("stale", False))
        watched_roots = len(self._watcher_subscribed)

        return {
            "total_graphs": total_graphs,
            "fresh_graphs": total_graphs - stale_graphs,
            "stale_graphs": stale_graphs,
            "watched_roots": watched_roots,
            "background_refresh_roots": len(self._background_refresh),
            "active_refresh_tasks": sum(
                1 for task in self._refresh_tasks.values() if not task.done()
            ),
            "refresh_failures": dict(self._refresh_failures),
            "graph_details": {
                key: {
                    "built_at": entry.get("built_at"),
                    "stale": entry.get("stale", False),
                }
                for key, entry in self._graph_cache.items()
            },
        }

    async def invalidate_root(self, root: Path) -> int:
        """Invalidate all cached graphs for a specific root.

        Args:
            root: Root path to invalidate

        Returns:
            Number of graphs invalidated
        """
        root_str = str(root.resolve())
        invalidated = 0

        async with self._cache_lock:
            for key, cache_entry in self._graph_cache.items():
                if key.startswith(root_str):
                    cache_entry["stale"] = True
                    invalidated += 1

        if invalidated > 0:
            logger.info(f"[GraphManager] Invalidated {invalidated} graph(s) for {root_str}")

        return invalidated

    async def clear_cache(self, root: Optional[Path] = None) -> int:
        """Clear cached graphs.

        Args:
            root: Optional root path to clear. If None, clears all graphs.

        Returns:
            Number of graphs cleared
        """
        cleared = 0

        async with self._cache_lock:
            if root is None:
                cleared = len(self._graph_cache)
                self._graph_cache.clear()
                logger.info(f"[GraphManager] Cleared all {cleared} graph(s)")
            else:
                root_str = str(root.resolve())
                keys_to_remove = [key for key in self._graph_cache if key.startswith(root_str)]

                for key in keys_to_remove:
                    del self._graph_cache[key]
                    cleared += 1

                logger.info(f"[GraphManager] Cleared {cleared} graph(s) for {root_str}")

        return cleared
