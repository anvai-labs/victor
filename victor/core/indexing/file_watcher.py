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

"""File watching service for codebase change detection.

This module provides a polling-based file watching service that detects
file system changes and publishes events to subscribers. Uses debouncing
to prevent event storms from rapid file saves.

Pattern: Observer + Event Emitter + Singleton Registry
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from victor.core.graph_rag.exclude_patterns import (
    get_exclusion_patterns,
    is_path_excluded,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FileChangeType",
    "FileChangeEvent",
    "FileWatcherService",
    "FileWatcherRegistry",
]


class FileChangeType(Enum):
    """Types of file system changes."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChangeEvent:
    """File system change event.

    Attributes:
        path: Path that changed
        change_type: Type of change
        timestamp: When the change was detected
        old_path: Original path (for rename events)
    """

    path: Path
    change_type: FileChangeType
    timestamp: datetime
    old_path: Optional[Path] = None

    def __str__(self) -> str:
        return f"{self.change_type.value}: {self.path}"


class FileWatcherService:
    """Watch codebase for file changes and publish events.

    Uses polling-based approach (cross-platform, no external deps).
    Debounces rapid changes to prevent event storms.

    Pattern: Observer + Event Emitter

    Example:
        >>> watcher = FileWatcherService(Path("/my/project"))
        >>> watcher.subscribe(lambda e: print(f"Changed: {e.path}"))
        >>> await watcher.start()
        >>> # ... make changes to files ...
        >>> await watcher.stop()

    Thread Safety:
        Fully thread-safe with asyncio.Lock for subscriber management
        and async task for polling loop.
    """

    def __init__(
        self,
        root: Path,
        poll_interval_seconds: float = 1.0,
        debounce_seconds: float = 0.3,
        exclude_patterns: Optional[Set[str]] = None,
    ):
        """Initialize file watcher.

        Args:
            root: Root directory to watch
            poll_interval_seconds: How often to poll for changes (default: 1s)
            debounce_seconds: How long to wait before publishing events (default: 300ms)
            exclude_patterns: File patterns to exclude from watching
        """
        self.root = root.resolve()
        self.poll_interval = poll_interval_seconds
        self.debounce_seconds = debounce_seconds
        self.exclude_patterns = (
            set(exclude_patterns)
            if exclude_patterns is not None
            else self._default_exclude_patterns()
        )
        # Cache once: `is_path_excluded` expects a list, and we hit
        # `_should_exclude` thousands of times per poll.
        self._exclude_patterns_list = list(self.exclude_patterns)
        # Literal name components extracted from globs (e.g. "**/.git/**" → ".git").
        # Lets us short-circuit the expensive fnmatch loop for the common case.
        self._excluded_components = self._extract_literal_components(self.exclude_patterns)

        # State tracking
        self._file_mtimes: Dict[str, float] = {}
        self._watching = False
        self._task: Optional[asyncio.Task] = None

        # Debounce tracking
        self._pending_changes: Dict[str, FileChangeEvent] = {}
        self._debounce_task: Optional[asyncio.Task] = None
        self._debounce_lock = asyncio.Lock()

        # Event subscribers (thread-safe set operations in CPython)
        self._subscribers: Set[Callable[[FileChangeEvent], None]] = set()

        # Statistics
        self._stats = {
            "files_watched": 0,
            "changes_detected": 0,
            "events_published": 0,
            "events_debounced": 0,
        }

    def _default_exclude_patterns(self) -> Set[str]:
        """Default patterns to exclude from watching.

        Returns:
            Set of glob patterns to exclude
        """
        return set(get_exclusion_patterns(self.root))

    @staticmethod
    def _extract_literal_components(patterns: Set[str]) -> Set[str]:
        """Extract literal name components from glob patterns for O(1) checks.

        Patterns like "**/.git/**" reduce to ".git"; patterns that still
        contain wildcards or path separators are left to the fnmatch path.
        """
        components: Set[str] = set()
        for pattern in patterns:
            cleaned = pattern
            # Strip recursive prefixes/suffixes used by the universal patterns.
            for marker in ("**/", "/**"):
                while marker in cleaned:
                    cleaned = cleaned.replace(marker, "")
            cleaned = cleaned.strip("/")
            if cleaned and "*" not in cleaned and "?" not in cleaned and "/" not in cleaned:
                components.add(cleaned)
        return components

    def subscribe(
        self,
        callback: Callable[[FileChangeEvent], None],
    ) -> None:
        """Subscribe to file change events.

        Args:
            callback: Function to call on file changes (can be sync or async)
        """
        self._subscribers.add(callback)
        logger.info(f"[FileWatcher] Subscriber added (total: {len(self._subscribers)})")

    def unsubscribe(
        self,
        callback: Callable[[FileChangeEvent], None],
    ) -> None:
        """Unsubscribe from file change events.

        Args:
            callback: Previously subscribed callback
        """
        self._subscribers.discard(callback)
        logger.info(f"[FileWatcher] Subscriber removed (total: {len(self._subscribers)})")

    async def start(self) -> None:
        """Start watching file system for changes."""
        if self._watching:
            logger.warning(f"[FileWatcher] Already watching {self.root}")
            return

        self._watching = True

        # Initial scan
        self._file_mtimes = await self._scan_directory()
        self._stats["files_watched"] = len(self._file_mtimes)
        logger.info(f"[FileWatcher] Initial scan: {len(self._file_mtimes)} files in {self.root}")

        # Start polling loop
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"[FileWatcher] Started watching {self.root}")

    async def stop(self) -> None:
        """Stop watching file system."""
        if not self._watching:
            return

        self._watching = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass

        logger.info(f"[FileWatcher] Stopped watching {self.root}")

    async def _poll_loop(self) -> None:
        """Main polling loop to detect file changes."""
        while self._watching:
            try:
                changes = await self._detect_changes()

                if changes:
                    # Debounce changes
                    async with self._debounce_lock:
                        for change in changes:
                            self._pending_changes[str(change.path)] = change

                        self._stats["changes_detected"] += len(changes)

                        # Reset debounce timer
                        if self._debounce_task:
                            self._debounce_task.cancel()

                        self._debounce_task = asyncio.create_task(self._debounce_and_publish())

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("[FileWatcher] Poll loop cancelled")
                break
            except Exception as e:
                logger.error(f"[FileWatcher] Error in poll loop: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _detect_changes(self) -> List[FileChangeEvent]:
        """Detect file system changes since last poll.

        Returns:
            List of file change events
        """
        current_mtimes = await self._scan_directory()
        changes = []

        # Check for new or modified files
        for path_str, mtime in current_mtimes.items():
            if path_str not in self._file_mtimes:
                # New file
                changes.append(
                    FileChangeEvent(
                        path=Path(path_str),
                        change_type=FileChangeType.CREATED,
                        timestamp=datetime.now(),
                    )
                )
            elif mtime > self._file_mtimes[path_str]:
                # Modified file
                changes.append(
                    FileChangeEvent(
                        path=Path(path_str),
                        change_type=FileChangeType.MODIFIED,
                        timestamp=datetime.now(),
                    )
                )

        # Check for deleted files
        for path_str in self._file_mtimes:
            if path_str not in current_mtimes:
                changes.append(
                    FileChangeEvent(
                        path=Path(path_str),
                        change_type=FileChangeType.DELETED,
                        timestamp=datetime.now(),
                    )
                )

        # Update state
        self._file_mtimes = current_mtimes
        self._stats["files_watched"] = len(self._file_mtimes)

        return changes

    async def _debounce_and_publish(self) -> None:
        """Wait for debounce period, then publish all pending changes."""
        await asyncio.sleep(self.debounce_seconds)

        # Get changes to publish
        async with self._debounce_lock:
            changes_to_publish = list(self._pending_changes.values())
            self._pending_changes.clear()

        # Publish changes
        for change in changes_to_publish:
            await self._publish_event(change)

        # Track debounced events
        debounced_count = len(changes_to_publish)
        if debounced_count > 0:
            self._stats["events_debounced"] += debounced_count
            logger.debug(f"[FileWatcher] Published {debounced_count} debounced events")

    async def _publish_event(self, event: FileChangeEvent) -> None:
        """Publish event to all subscribers.

        Args:
            event: File change event to publish
        """
        logger.debug(f"[FileWatcher] Publishing: {event}")

        # Get current subscribers (copy to avoid modification during iteration)
        subscribers = list(self._subscribers)

        # Publish to all subscribers
        for callback in subscribers:
            try:
                # Call subscriber (may be sync or async)
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[FileWatcher] Subscriber error: {e}")

        self._stats["events_published"] += 1

    async def _scan_directory(self) -> Dict[str, float]:
        """Scan directory and get mtimes for all files.

        Returns:
            Dict mapping file paths to modification times
        """
        return await asyncio.to_thread(self._scan_directory_sync)

    def _scan_directory_sync(self) -> Dict[str, float]:
        """Synchronous scanner used from a worker thread."""
        mtimes = {}

        try:
            for dirpath, dirnames, filenames in os.walk(self.root):
                current_dir = Path(dirpath)
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if not self._should_exclude(current_dir / dirname)
                ]

                for filename in filenames:
                    file_path = current_dir / filename

                    # Check exclude patterns
                    if self._should_exclude(file_path):
                        continue

                    try:
                        mtime = file_path.stat().st_mtime
                        mtimes[str(file_path)] = mtime
                    except Exception as e:
                        logger.warning(f"[FileWatcher] Error getting mtime for {file_path}: {e}")
        except Exception as e:
            logger.error(f"[FileWatcher] Error scanning directory {self.root}: {e}")

        return mtimes

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from watching.

        Args:
            path: Path to check

        Returns:
            True if path should be excluded, False otherwise
        """
        # Fast path: any path component is a literal excluded name (.git,
        # __pycache__, node_modules, .victor, ...). Avoids the fnmatch loop
        # for the overwhelmingly common case during polling.
        excluded_components = self._excluded_components
        if excluded_components:
            try:
                parts = path.relative_to(self.root).parts
            except ValueError:
                parts = path.parts
            for part in parts:
                if part in excluded_components:
                    return True

        # Slow path: full glob matching for wildcards (*.pyc, cmake-build-*, ...).
        return is_path_excluded(path, self.root, self._exclude_patterns_list)

    def get_stats(self) -> Dict[str, int]:
        """Get file watcher statistics.

        Returns:
            Dict with current statistics
        """
        return {
            **self._stats,
            "subscribers": len(self._subscribers),
            "watching": 1 if self._watching else 0,
        }


class FileWatcherRegistry:
    """Global registry of file watchers (one per root path).

    Manages file watcher lifecycle and ensures only one watcher per path.

    Pattern: Singleton Registry

    Example:
        >>> registry = FileWatcherRegistry.get_instance()
        >>> watcher = await registry.get_watcher(Path("/my/project"))
        >>> # ... use watcher ...
        >>> await registry.stop_all()
    """

    _instance: Optional["FileWatcherRegistry"] = None
    _watchers: Dict[str, FileWatcherService] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        """Initialize FileWatcherRegistry."""
        pass

    @classmethod
    def get_instance(cls) -> "FileWatcherRegistry":
        """Get singleton instance.

        Returns:
            FileWatcherRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _find_overlapping_watcher(
        root: Path, watchers: Dict[str, FileWatcherService]
    ) -> Optional[str]:
        """Find an existing watcher that covers the requested root path.

        Checks if:
        1. An existing watcher is a parent of the requested root (most common case)
        2. The requested root is a parent of an existing watcher (should reuse existing watchers)

        Args:
            root: Path to watch
            watchers: Dictionary of existing watchers

        Returns:
            Path string of overlapping watcher, or None
        """
        root_str = str(root.resolve())
        root_normalized = os.path.normpath(root_str) + os.sep

        for watcher_path_str in watchers.keys():
            watcher_normalized = os.path.normpath(watcher_path_str) + os.sep

            # Check if existing watcher is a parent of requested root
            if root_normalized.startswith(watcher_normalized):
                logger.debug(
                    f"[FileWatcherRegistry] Reusing parent watcher {watcher_path_str} for {root_str}"
                )
                return watcher_path_str

            # Check if requested root is a parent of existing watcher
            if watcher_normalized.startswith(root_normalized):
                logger.debug(
                    f"[FileWatcherRegistry] Requested root {root_str} is parent of existing watcher {watcher_path_str}, will consolidate"
                )
                return watcher_path_str

        return None

    async def get_watcher(
        self,
        root: Path,
        **kwargs,
    ) -> FileWatcherService:
        """Get or create file watcher for root path.

        Automatically deduplicates overlapping watchers by reusing parent
        watchers when possible. Skips watching empty directories.

        Args:
            root: Root directory to watch
            **kwargs: Additional arguments for FileWatcherService

        Returns:
            FileWatcherService instance (cached or new)
        """
        import os

        root = root.resolve()
        root_str = str(root)

        # Check for overlapping watchers without lock
        overlapping = self._find_overlapping_watcher(root, self._watchers)
        if overlapping:
            logger.info(
                f"[FileWatcherRegistry] Using existing watcher {overlapping} for {root_str} instead of creating duplicate"
            )
            return self._watchers[overlapping]

        # Fast path - check for exact match without lock
        if root_str in self._watchers:
            return self._watchers[root_str]

        # Slow path - acquire lock
        async with self._lock:
            # Double-check for exact match inside lock
            if root_str in self._watchers:
                return self._watchers[root_str]

            # Check again for overlaps inside lock
            overlapping = self._find_overlapping_watcher(root, self._watchers)
            if overlapping:
                logger.info(
                    f"[FileWatcherRegistry] Using existing watcher {overlapping} for {root_str} instead of creating duplicate"
                )
                return self._watchers[overlapping]

            # Check if directory is empty (skip watching)
            try:
                if not root.exists() or not root.is_dir():
                    logger.warning(
                        f"[FileWatcherRegistry] Path {root_str} does not exist or is not a directory"
                    )
                    # Return a dummy watcher that does nothing
                    return FileWatcherService(root, **kwargs)

                file_count = sum(1 for _ in root.iterdir() if _.is_file())
                if file_count == 0:
                    logger.info(
                        f"[FileWatcherRegistry] Skipping empty directory {root_str} (0 files), not watching"
                    )
                    # Return a watcher that won't poll anything
                    return FileWatcherService(root, **kwargs)

            except (OSError, PermissionError) as exc:
                logger.warning(
                    f"[FileWatcherRegistry] Cannot access {root_str}: {exc}, creating watcher anyway"
                )

            # Create new watcher
            watcher = FileWatcherService(root, **kwargs)
            await watcher.start()
            self._watchers[root_str] = watcher
            logger.info(
                f"[FileWatcherRegistry] Created watcher for {root_str} "
                f"(total watchers: {len(self._watchers)})"
            )

            return self._watchers[root_str]

    async def stop_watcher(self, root: Path) -> bool:
        """Stop watcher for specific root path.

        Args:
            root: Root path whose watcher should be stopped

        Returns:
            True if watcher was stopped, False if not found
        """
        root_str = str(root.resolve())

        async with self._lock:
            if root_str in self._watchers:
                await self._watchers[root_str].stop()
                del self._watchers[root_str]
                logger.info(
                    f"[FileWatcherRegistry] Stopped watcher for {root_str} "
                    f"(remaining: {len(self._watchers)})"
                )
                return True

            return False

    async def stop_all(self) -> None:
        """Stop all file watchers.

        Useful for cleanup or shutdown.
        """
        async with self._lock:
            for watcher in self._watchers.values():
                await watcher.stop()

            count = len(self._watchers)
            self._watchers.clear()

            logger.info(f"[FileWatcherRegistry] Stopped {count} watchers")

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all watchers.

        Returns:
            Dict mapping root paths to watcher statistics
        """
        return {
            "total_watchers": len(self._watchers),
            "watcher_details": {
                root: watcher.get_stats() for root, watcher in self._watchers.items()
            },
        }
