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

"""Index-level locking to prevent concurrent indexing of same path.

This module provides per-path async locking to prevent multiple concurrent
indexing operations on the same codebase path. Uses singleton pattern with
double-checked locking for thread-safe access.

Provides two levels of locking:
1. In-process locking (asyncio.Lock) - prevents concurrent indexing in same process
2. Cross-process locking (file-based) - prevents concurrent indexing across processes

Pattern: Singleton + Per-Resource Locking + Double-Checked Locking + File Locking
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["IndexLockRegistry", "FileLock"]


class FileLock:
    """Cross-process file-based lock using fcntl.

    Prevents multiple processes from indexing the same path concurrently.
    Uses exclusive file locking with automatic cleanup on process exit.

    Pattern: RAII (Resource Acquisition Is Initialization)

    Example:
        >>> lock = FileLock(Path("/my/project/.victor/index.lock"))
        >>> lock.acquire()
        >>> try:
        ...     # Exclusive access to index this path
        ...     await build_index()
        ... finally:
        ...     lock.release()
    """

    def __init__(self, lock_file: Path):
        """Initialize file lock.

        Args:
            lock_file: Path to lock file (will be created if needed)
        """
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd: Optional[int] = None

    def acquire(self, timeout: float = 300.0, shared: bool = False) -> bool:
        """Acquire lock on file.

        Args:
            timeout: Maximum time to wait for lock (default: 5 minutes)
            shared: If True, acquire a shared lock (multiple readers allowed).
                    If False, acquire an exclusive lock (one writer).

        Returns:
            True if lock acquired, False if timeout

        Raises:
            IOError: If lock cannot be acquired due to system errors
        """
        start_time = time.time()
        current_pid = os.getpid()

        lock_op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        retries = 0
        while True:
            try:
                # Open file (create if needed).
                # Note: shared locks on newly created files are always successful,
                # but they won't block exclusive locks unless the file exists.
                self._lock_fd = os.open(
                    self.lock_file,
                    os.O_CREAT | os.O_RDWR,
                    0o644,
                )

                # Try to acquire lock (non-blocking)
                fcntl.flock(self._lock_fd, lock_op | fcntl.LOCK_NB)

                # Write PID to lock file if exclusive
                if not shared:
                    os.ftruncate(self._lock_fd, 0)
                    os.write(self._lock_fd, str(current_pid).encode())

                if retries > 0:
                    mode = "shared" if shared else "exclusive"
                    logger.debug(
                        f"[FileLock] Acquired {mode} lock {self.lock_file} after {retries} retries (PID: {current_pid})"
                    )
                return True

            except IOError as e:
                # Lock is held by another process
                if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    if self._lock_fd is not None:
                        os.close(self._lock_fd)
                        self._lock_fd = None

                    # If we've waited longer than timeout, give up
                    if time.time() - start_time >= timeout:
                        mode = "shared" if shared else "exclusive"
                        logger.warning(
                            f"[FileLock] Timeout waiting for {mode} lock {self.lock_file} after {time.time() - start_time:.1f}s"
                        )
                        return False

                    # Back off exponentially, max 1s
                    sleep_time = min(0.1 * (1.5**retries), 1.0)
                    time.sleep(sleep_time)
                    retries += 1
                    continue

                # Other error
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                raise IOError(f"Failed to acquire lock: {e}")

    def release(self) -> None:
        """Release the lock and close the file descriptor.

        The lock file itself is intentionally left on disk. Unlinking a contended
        lock file races with other holders: a second process can create and lock
        a fresh inode for the same path while a first holder still locks the
        now-orphaned inode, so both believe they hold the lock. Leaving the (tiny)
        file in place keeps the inode stable — essential now that shared/exclusive
        locks coexist.
        """
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                logger.debug(f"[FileLock] Released lock {self.lock_file}")
            except Exception as e:
                logger.warning(f"[FileLock] Error releasing lock: {e}")
            finally:
                self._lock_fd = None

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Destructor - ensure lock is released."""
        self.release()


class _AsyncRWLock:
    """In-process readers-writer lock with writer preference.

    Multiple readers may hold the lock concurrently; a writer holds it
    exclusively. Writer-preference (new readers wait while a writer is queued)
    prevents writer starvation under a steady stream of readers. This is the
    in-process counterpart to the cross-process shared/exclusive ``FileLock`` so
    a "shared" acquisition is genuinely concurrent at both levels.
    """

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    async def acquire_read(self) -> None:
        async with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                await self._cond.wait()
            self._readers += 1

    async def release_read(self) -> None:
        async with self._cond:
            if self._readers > 0:
                self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_write(self) -> None:
        async with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True

    async def release_write(self) -> None:
        async with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    def is_held(self) -> bool:
        """Whether the lock is currently held by any reader or a writer."""
        return self._writer_active or self._readers > 0


class _PathLockHandle:
    """Async context manager that combines in-process and file locking per use."""

    def __init__(
        self,
        registry: "IndexLockRegistry",
        path_str: str,
        rw_lock: "_AsyncRWLock",
        lock_file: Optional[Path],
        timeout_seconds: float,
        shared: bool = False,
    ) -> None:
        self._registry = registry
        self._path_str = path_str
        self._rw_lock = rw_lock
        self._lock_file = lock_file
        self._timeout_seconds = timeout_seconds
        self._shared = shared
        self._file_lock: Optional[FileLock] = None

    async def _acquire_in_process(self) -> None:
        if self._shared:
            await self._rw_lock.acquire_read()
        else:
            await self._rw_lock.acquire_write()

    async def _release_in_process(self) -> None:
        if self._shared:
            await self._rw_lock.release_read()
        else:
            await self._rw_lock.release_write()

    async def __aenter__(self) -> None:
        await self._acquire_in_process()
        try:
            if self._lock_file is not None:
                file_lock = FileLock(self._lock_file)
                loop = asyncio.get_running_loop()
                acquired = await loop.run_in_executor(
                    None,
                    lambda: file_lock.acquire(timeout=self._timeout_seconds, shared=self._shared),
                )
                if not acquired:
                    raise TimeoutError(
                        f"Failed to acquire index lock for {self._path_str} "
                        f"after {self._timeout_seconds:g} seconds "
                        "(another process may be indexing)"
                    )
                self._file_lock = file_lock
                logger.info(
                    "[IndexLockRegistry] Acquired cross-process %s lock for %s",
                    "shared" if self._shared else "exclusive",
                    self._path_str,
                )
            return None
        except Exception:
            await self._release_in_process()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self._file_lock is not None:
                self._file_lock.release()
                self._file_lock = None
        finally:
            self._registry._mark_lock_used_path(self._path_str)
            await self._release_in_process()
        return False


class IndexLockRegistry:
    """Global registry of path-specific async locks.

    Prevents multiple concurrent indexing operations on the same path
    by providing a unique asyncio.Lock for each unique codebase path.

    Uses singleton pattern with double-checked locking:
    1. Check cache without lock (fast path)
    2. Acquire registry lock
    3. Double-check and create if needed
    4. Return path-specific lock

    Example:
        >>> registry = IndexLockRegistry.get_instance()
        >>> path_lock = await registry.acquire_lock(Path("/my/project"))
        >>> async with path_lock:
        ...     # Exclusive access to index this path
        ...     await build_index()

    Thread Safety:
        Fully thread-safe with asyncio.Lock for registry access
        and per-path asyncio.Lock instances for resource protection.
    """

    _instance: Optional["IndexLockRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        """Initialize IndexLockRegistry."""
        self._path_locks: Dict[str, _AsyncRWLock] = {}
        self._lock_files: Dict[str, Path] = {}
        self._registry_lock = asyncio.Lock()
        self._lock_stats: Dict[str, Dict] = {}  # Track lock usage stats

    @classmethod
    def get_instance(cls) -> "IndexLockRegistry":
        """Get singleton instance.

        Returns:
            IndexLockRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def acquire_lock(
        self,
        path: Path,
        use_file_lock: bool = True,
        timeout_seconds: float = 300.0,
        shared: bool = False,
    ) -> _PathLockHandle:
        """Acquire or create lock for specific path.

        Uses double-checked locking pattern for performance:
        - Fast path: Check cache without registry lock
        - Slow path: Acquire registry lock and create if needed
        - Cross-process: Also acquires file lock for multi-process safety

        Args:
            path: File system path to lock
            use_file_lock: Whether to use file-based locking for cross-process safety
            timeout_seconds: Maximum time to wait for the file lock

        Returns:
            Async context manager for this specific path

        Raises:
            TimeoutError: If file lock cannot be acquired within timeout
        """
        path_str = str(path.resolve())

        # Fast path - check cache without lock
        if path_str in self._path_locks:
            # Update stats
            if path_str in self._lock_stats:
                self._lock_stats[path_str]["cache_hits"] += 1
            return _PathLockHandle(
                self,
                path_str,
                self._path_locks[path_str],
                self._lock_files.get(path_str) if use_file_lock else None,
                timeout_seconds,
                shared=shared,
            )

        # Slow path - acquire registry lock
        start_time = time.time()
        async with self._registry_lock:
            # Double-check inside lock
            if path_str in self._path_locks:
                # Another task created it while we waited
                return _PathLockHandle(
                    self,
                    path_str,
                    self._path_locks[path_str],
                    self._lock_files.get(path_str) if use_file_lock else None,
                    timeout_seconds,
                )

            lock_file: Optional[Path] = None
            if use_file_lock:
                from victor.config.settings import get_project_paths

                project_paths = get_project_paths(path)
                lock_file = project_paths.project_victor_dir / "index.lock"
                self._lock_files[path_str] = lock_file

            # Create new in-process lock
            self._path_locks[path_str] = _AsyncRWLock()

            # Initialize stats
            self._lock_stats[path_str] = {
                "created_at": time.time(),
                "cache_hits": 0,
                "waits": 0,
                "total_wait_time_ms": 0,
                "has_file_lock": use_file_lock,
            }

            logger.info(
                f"[IndexLockRegistry] Created lock for {path_str} "
                f"(total locks: {len(self._path_locks)})"
            )

            # Track wait time
            wait_time_ms = (time.time() - start_time) * 1000
            self._lock_stats[path_str]["waits"] += 1
            self._lock_stats[path_str]["total_wait_time_ms"] += wait_time_ms

            return _PathLockHandle(
                self,
                path_str,
                self._path_locks[path_str],
                lock_file if use_file_lock else None,
                timeout_seconds,
            )

    async def cleanup_idle_locks(
        self,
        max_idle_seconds: int = 3600,
    ) -> int:
        """Remove locks that haven't been used recently.

        Args:
            max_idle_seconds: Remove locks idle longer than this (default: 1 hour)

        Returns:
            Number of locks removed
        """
        if max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be positive")

        removed = 0
        current_time = time.time()

        async with self._registry_lock:
            # Find idle locks
            idle_paths = [
                path_str
                for path_str, stats in self._lock_stats.items()
                if current_time - stats.get("last_used", stats["created_at"]) > max_idle_seconds
            ]

            # Remove idle locks
            for path_str in idle_paths:
                lock = self._path_locks.get(path_str)
                if lock is not None and lock.is_held():
                    continue

                del self._path_locks[path_str]
                self._lock_files.pop(path_str, None)
                del self._lock_stats[path_str]
                removed += 1

            if removed > 0:
                logger.info(
                    f"[IndexLockRegistry] Cleaned up {removed} idle locks "
                    f"(remaining: {len(self._path_locks)})"
                )

        return removed

    def mark_lock_used(self, path: Path) -> None:
        """Mark lock as recently used.

        Call this after successfully using a lock to track activity.

        Args:
            path: Path whose lock was used
        """
        path_str = str(path.resolve())
        self._mark_lock_used_path(path_str)

    def _mark_lock_used_path(self, path_str: str) -> None:
        """Mark a lock entry as recently used by its normalized path string."""
        if path_str in self._lock_stats:
            self._lock_stats[path_str]["last_used"] = time.time()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics about lock usage.

        Returns:
            Dict mapping path strings to lock statistics
        """
        return {
            "total_locks": len(self._path_locks),
            "lock_details": dict(self._lock_stats),
        }

    async def reset(self) -> None:
        """Reset all locks (mainly for testing).

        Warning: This will release all locks and clear the cache.
        Use with caution in production.
        """
        async with self._registry_lock:
            self._lock_files.clear()
            self._path_locks.clear()
            self._lock_stats.clear()
            logger.warning("[IndexLockRegistry] All locks cleared")
