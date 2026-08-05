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

"""File watcher initialization utilities.

This module provides helper functions for initializing file watchers
across different contexts (CLI, ToolExecutor, API server).

Pattern: Service Locator + Initialization Helper
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from victor.core.indexing.file_watcher import FileWatcherRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "initialize_file_watchers",
    "stop_file_watchers",
    "get_project_paths_from_context",
]


async def initialize_file_watchers(
    project_paths: List[Path],
    exec_ctx: Optional[dict] = None,
) -> None:
    """Initialize file watchers for all project paths.

    Creates file watchers for each project path to enable automatic
    cache invalidation and incremental updates.

    Args:
        project_paths: List of project root paths to watch
        exec_ctx: Optional execution context for passing to watchers
    """
    if not project_paths:
        logger.debug("[FileWatcherInitializer] No project paths to watch")
        return

    registry = FileWatcherRegistry.get_instance()
    initialized = []

    for project_path in project_paths:
        try:
            # Resolve path to canonical form
            resolved_path = project_path.resolve()

            # Get or create watcher (cached per path)
            await registry.get_watcher(
                resolved_path,
                poll_interval_seconds=1.0,
                debounce_seconds=0.3,
            )

            initialized.append(str(resolved_path))
            logger.info(f"[FileWatcherInitializer] Watching {resolved_path}")

        except Exception as e:
            logger.error(
                f"[FileWatcherInitializer] Failed to start watcher for {project_path}: {e}"
            )

    if initialized:
        logger.info(f"[FileWatcherInitializer] Initialized {len(initialized)} file watcher(s)")


async def stop_file_watchers(
    project_paths: Optional[List[Path]] = None,
) -> None:
    """Stop file watchers for project paths.

    Args:
        project_paths: List of project paths to stop watching.
                      If None, stops all watchers.
    """
    registry = FileWatcherRegistry.get_instance()

    if project_paths is None:
        # Stop all watchers
        await registry.stop_all()
        logger.info("[FileWatcherInitializer] Stopped all file watchers")
        return

    # Stop specific watchers
    for project_path in project_paths:
        try:
            stopped = await registry.stop_watcher(project_path)
            if stopped:
                logger.info(f"[FileWatcherInitializer] Stopped watcher for {project_path}")
        except Exception as e:
            logger.error(f"[FileWatcherInitializer] Failed to stop watcher for {project_path}: {e}")


def get_project_paths_from_context(exec_ctx: Optional[dict]) -> List[Path]:
    """Extract project paths from execution context.

    Args:
        exec_ctx: Execution context that may contain project paths

    Returns:
        List of project root paths
    """
    if not exec_ctx:
        return []

    paths = []

    # Check for current working directory
    cwd = exec_ctx.get("cwd")
    if cwd:
        paths.append(Path(cwd))

    # Check for explicit project paths
    project_paths = exec_ctx.get("project_paths")
    if project_paths:
        if isinstance(project_paths, list):
            paths.extend([Path(p) for p in project_paths])
        else:
            paths.append(Path(project_paths))

    # Check for settings with project path
    settings = exec_ctx.get("settings")
    if settings:
        project_path = getattr(settings, "project_path", None)
        if project_path:
            paths.append(Path(project_path))

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in paths:
        path_str = str(path.resolve())
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path)

    return unique_paths


async def initialize_from_context(exec_ctx: Optional[dict]) -> None:
    """Initialize file watchers based on execution context.

    Automatically detects project paths from context and initializes
    file watchers for them.

    Args:
        exec_ctx: Execution context that may contain project paths
    """
    project_paths = get_project_paths_from_context(exec_ctx)

    if not project_paths:
        # Default to current working directory
        import os

        project_paths = [Path(os.getcwd())]

    await initialize_file_watchers(project_paths, exec_ctx)


async def cleanup_session() -> None:
    """Cleanup file watchers for session shutdown.

    Stops all file watchers and releases resources.
    Should be called on session/agent shutdown.
    """
    registry = FileWatcherRegistry.get_instance()

    stats = registry.get_stats()
    total_watchers = stats.get("total_watchers", 0)

    if total_watchers > 0:
        logger.info(f"[FileWatcherInitializer] Cleaning up {total_watchers} watcher(s)")
        await registry.stop_all()
    else:
        logger.debug("[FileWatcherInitializer] No watchers to cleanup")
