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

"""Shared infrastructure for dynamic module loading with hot-reload support.

This module provides the DynamicModuleLoader base class that encapsulates
common module loading infrastructure used by:
- CapabilityLoader (victor/framework/capability_loader.py)
- ToolPluginRegistry (victor/tools/plugin_registry.py)

Key features:
- Module cache invalidation for hot-reload
- File watching with watchdog
- Debounced reload timers
- Submodule discovery and invalidation

Design Pattern: Template Method
==============================
DynamicModuleLoader provides the infrastructure for module loading,
while subclasses implement the specific loading/registration logic.

Usage:
    from victor.framework.module_loader import DynamicModuleLoader

    class MyLoader(DynamicModuleLoader):
        def __init__(self):
            super().__init__(watch_dirs=[Path("~/.myapp/plugins")])

        def _on_module_reloaded(self, module_name: str) -> None:
            # Handle module reload
            self.reload_my_resources(module_name)
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from watchdog.observers import Observer

logger = logging.getLogger(__name__)


# =============================================================================
# Debounced Reload Timer
# =============================================================================


class DebouncedReloadTimer:
    """Timer that debounces rapid file changes.

    This prevents multiple reloads when multiple files change in quick
    succession (e.g., during a batch save operation).

    Attributes:
        delay: Debounce delay in seconds (default 0.5s)
    """

    def __init__(self, delay: float = 0.5):
        """Initialize debounced timer.

        Args:
            delay: Debounce delay in seconds
        """
        self.delay = delay
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(
        self,
        key: str,
        callback: Callable[[], None],
    ) -> None:
        """Schedule a debounced callback.

        If a callback is already scheduled for this key, it will be
        cancelled and rescheduled.

        Args:
            key: Unique key for this callback (usually module name)
            callback: Callable to invoke after debounce delay
        """
        with self._lock:
            # Cancel existing timer
            if key in self._timers:
                self._timers[key].cancel()

            # Create new timer
            timer = threading.Timer(self.delay, self._execute, args=(key, callback))
            timer.daemon = True
            timer.start()
            self._timers[key] = timer

    def _execute(self, key: str, callback: Callable[[], None]) -> None:
        """Execute callback and cleanup timer.

        Args:
            key: Timer key
            callback: Callback to execute
        """
        with self._lock:
            self._timers.pop(key, None)

        try:
            callback()
        except Exception as e:
            logger.error(f"Debounced callback failed for '{key}': {e}")

    def cancel(self, key: str) -> bool:
        """Cancel a scheduled callback.

        Args:
            key: Timer key to cancel

        Returns:
            True if timer was cancelled
        """
        with self._lock:
            timer = self._timers.pop(key, None)
            if timer:
                timer.cancel()
                return True
            return False

    def cancel_all(self) -> int:
        """Cancel all pending timers.

        Returns:
            Number of timers cancelled
        """
        with self._lock:
            count = len(self._timers)
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            return count


# =============================================================================
# Dynamic Module Loader Base Class
# =============================================================================


class DynamicModuleLoader:
    """Base class for dynamic module loading with hot-reload support.

    This class provides shared infrastructure for:
    - Module cache invalidation
    - File watching with debounced reloads
    - Submodule tracking and invalidation

    Subclasses should implement:
    - _on_module_reloaded(module_name): Handle module reload
    - _get_module_path(module_name): Get file path for a module

    Thread Safety:
        The module invalidation and file watching are thread-safe.
        Subclasses should use appropriate synchronization for their
        own state.

    Attributes:
        watch_dirs: Directories being watched for file changes
        debounce_delay: Delay before triggering reload (default 0.5s)
    """

    def __init__(
        self,
        watch_dirs: Optional[List[Path]] = None,
        debounce_delay: float = 0.5,
    ) -> None:
        """Initialize the module loader.

        Args:
            watch_dirs: Directories to watch for file changes
            debounce_delay: Delay before triggering reload (default 0.5s)
        """
        self._watch_dirs: List[Path] = []
        if watch_dirs:
            for dir_path in watch_dirs:
                expanded = Path(dir_path).expanduser()
                if expanded.exists():
                    self._watch_dirs.append(expanded)

        self._debounce_delay = debounce_delay
        self._debounce_timer = DebouncedReloadTimer(delay=debounce_delay)

        # File watcher state
        self._observer: Optional["Observer"] = None
        self._file_handler: Optional[Any] = None

        # Module tracking
        self._loaded_modules: Dict[str, Any] = {}
        self._modules_lock = threading.Lock()
        self._module_paths: Dict[str, Path] = {}  # module -> file path

    @property
    def watch_dirs(self) -> List[Path]:
        """Get list of watched directories."""
        return list(self._watch_dirs)

    @property
    def debounce_delay(self) -> float:
        """Get debounce delay in seconds."""
        return self._debounce_delay

    # =========================================================================
    # Module Cache Invalidation
    # =========================================================================

    def invalidate_module(self, module_name: str) -> int:
        """Invalidate Python module cache for hot-reload.

        This removes the module and all its submodules from sys.modules,
        ensuring subsequent imports will load fresh code from disk.

        Args:
            module_name: Module name to invalidate

        Returns:
            Number of modules invalidated (including submodules)
        """
        invalidated = 0

        # Remove from internal tracking
        self._loaded_modules.pop(module_name, None)

        # Lock protects sys.modules from concurrent invalidation
        with self._modules_lock:
            # Remove main module
            if module_name in sys.modules:
                del sys.modules[module_name]
                invalidated += 1
                logger.debug(f"Invalidated module: {module_name}")

            # Remove submodules
            prefix = f"{module_name}."
            to_remove = [name for name in sys.modules if name.startswith(prefix)]
            for name in to_remove:
                del sys.modules[name]
                invalidated += 1
                logger.debug(f"Invalidated submodule: {name}")

        # Clear import caches (thread-safe per CPython)
        importlib.invalidate_caches()

        return invalidated

    def invalidate_modules_in_path(
        self,
        base_module_name: str,
        directory_path: Path,
    ) -> int:
        """Invalidate all modules loaded from a directory.

        This is useful for plugin directories where multiple modules
        may be loaded from the same directory tree.

        Args:
            base_module_name: Base module name for the plugin
            directory_path: Path to plugin directory

        Returns:
            Number of modules invalidated
        """
        invalidated = 0
        path_str = str(directory_path.resolve())

        # Lock protects sys.modules from concurrent invalidation
        with self._modules_lock:
            # Remove the main module
            if base_module_name in sys.modules:
                del sys.modules[base_module_name]
                invalidated += 1
                logger.debug(f"Invalidated module: {base_module_name}")

            # Remove any modules that came from this directory
            modules_to_remove = []
            for mod_name, mod in list(sys.modules.items()):
                if mod is None:
                    continue

                try:
                    mod_file = getattr(mod, "__file__", None)
                    if mod_file and path_str in str(Path(mod_file).resolve()):
                        modules_to_remove.append(mod_name)
                except Exception:
                    continue

            for mod_name in modules_to_remove:
                del sys.modules[mod_name]
                invalidated += 1
                logger.debug(f"Invalidated module from path: {mod_name}")

        # Clear import caches (thread-safe per CPython)
        importlib.invalidate_caches()

        return invalidated

    # =========================================================================
    # File Watching
    # =========================================================================

    def setup_file_watcher(
        self,
        dirs: Optional[List[Path]] = None,
        on_change: Optional[Callable[[str, str], None]] = None,
        recursive: bool = True,
    ) -> bool:
        """Set up file watching with debounced reloads.

        Args:
            dirs: Directories to watch (defaults to self._watch_dirs)
            on_change: Callback(file_path, event_type) for file changes.
                      event_type is one of: "modified", "created", "deleted"
            recursive: Watch directories recursively (default True)

        Returns:
            True if watching started successfully
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "watchdog not installed - file watching unavailable. "
                "Install with: pip install watchdog"
            )
            return False

        if self._observer is not None:
            logger.warning("File watcher already running")
            return False

        watch_dirs = dirs if dirs is not None else self._watch_dirs
        if not watch_dirs:
            logger.warning("No directories configured for watching")
            return False

        # Create file event handler
        loader = self

        class ModuleFileHandler(FileSystemEventHandler):
            """Handler for module file changes."""

            def __init__(handler_self):
                super().__init__()

            def _handle_change(
                handler_self,
                file_path: str,
                event_type: str,
            ) -> None:
                """Handle file change with debouncing."""
                if not file_path.endswith(".py"):
                    return

                path_obj = Path(file_path)

                # Find which module this file belongs to
                module_name = loader._find_module_for_file(path_obj)

                if module_name:
                    # Schedule debounced reload
                    def do_reload():
                        try:
                            loader._on_module_changed(
                                module_name,
                                path_obj,
                                event_type,
                            )
                            if on_change:
                                on_change(file_path, event_type)
                        except Exception as e:
                            logger.error(f"Error handling {event_type} for '{module_name}': {e}")

                    loader._debounce_timer.schedule(module_name, do_reload)

            def on_modified(handler_self, event) -> None:
                if not event.is_directory:
                    handler_self._handle_change(event.src_path, "modified")

            def on_created(handler_self, event) -> None:
                if not event.is_directory:
                    handler_self._handle_change(event.src_path, "created")

            def on_deleted(handler_self, event) -> None:
                if not event.is_directory:
                    handler_self._handle_change(event.src_path, "deleted")

        self._observer = Observer()
        self._file_handler = ModuleFileHandler()

        # Watch all directories
        watched_count = 0
        for watch_dir in watch_dirs:
            if watch_dir.exists():
                self._observer.schedule(
                    self._file_handler,
                    str(watch_dir),
                    recursive=recursive,
                )
                watched_count += 1
                logger.debug(f"Watching directory: {watch_dir}")

        if watched_count == 0:
            logger.warning("No valid directories to watch")
            self._observer = None
            self._file_handler = None
            return False

        self._observer.start()
        logger.info(f"Started file watcher for {watched_count} directories")
        return True

    def stop_file_watcher(self) -> None:
        """Stop the file watcher."""
        # Cancel pending debounced reloads
        cancelled = self._debounce_timer.cancel_all()
        if cancelled:
            logger.debug(f"Cancelled {cancelled} pending reload(s)")

        # Stop observer
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
            self._file_handler = None
            logger.info("Stopped file watcher")

    @property
    def is_watching(self) -> bool:
        """Check if file watcher is running."""
        return self._observer is not None

    # =========================================================================
    # Template Methods (Override in Subclasses)
    # =========================================================================

    def _find_module_for_file(self, file_path: Path) -> Optional[str]:
        """Find which loaded module a file belongs to.

        Override this method to implement module discovery logic.

        Args:
            file_path: Path to the changed file

        Returns:
            Module name or None if not found
        """
        file_path = file_path.resolve()

        for mod_name, mod in self._loaded_modules.items():
            mod_file = getattr(mod, "__file__", None)
            if mod_file and Path(mod_file).resolve() == file_path:
                return mod_name

        return None

    def _on_module_changed(
        self,
        module_name: str,
        file_path: Path,
        event_type: str,
    ) -> None:
        """Handle a module file change.

        Override this method to implement reload logic.

        Args:
            module_name: Name of the changed module
            file_path: Path to the changed file
            event_type: Type of change ("modified", "created", "deleted")
        """
        logger.debug(f"Module '{module_name}' {event_type}: {file_path}")

    # =========================================================================
    # Module Tracking
    # =========================================================================

    def track_module(
        self,
        module_name: str,
        module: Any,
        file_path: Optional[Path] = None,
    ) -> None:
        """Track a loaded module for hot-reload support.

        Args:
            module_name: Name of the module
            module: The module object
            file_path: Optional file path (defaults to module.__file__)
        """
        self._loaded_modules[module_name] = module

        if file_path:
            self._module_paths[module_name] = file_path
        elif hasattr(module, "__file__") and module.__file__:
            self._module_paths[module_name] = Path(module.__file__)

    def untrack_module(self, module_name: str) -> bool:
        """Untrack a loaded module.

        Args:
            module_name: Name of the module to untrack

        Returns:
            True if module was tracked
        """
        was_tracked = module_name in self._loaded_modules
        self._loaded_modules.pop(module_name, None)
        self._module_paths.pop(module_name, None)
        return was_tracked

    def get_tracked_modules(self) -> List[str]:
        """Get list of tracked module names.

        Returns:
            List of module names
        """
        return list(self._loaded_modules.keys())

    def get_module_path(self, module_name: str) -> Optional[Path]:
        """Get file path for a tracked module.

        Args:
            module_name: Name of the module

        Returns:
            Path or None if not tracked
        """
        return self._module_paths.get(module_name)

    # =========================================================================
    # Watch Directory Management
    # =========================================================================

    def add_watch_dir(self, path: Path) -> bool:
        """Add a directory to watch.

        If file watching is active, the new directory will be added
        to the watcher.

        Args:
            path: Directory path to add

        Returns:
            True if directory was added
        """
        path = Path(path).expanduser()
        if path in self._watch_dirs:
            return False

        self._watch_dirs.append(path)

        # Add to active watcher if running
        if self._observer is not None and path.exists():
            self._observer.schedule(
                self._file_handler,
                str(path),
                recursive=True,
            )
            logger.debug(f"Added watch directory: {path}")

        return True

    def remove_watch_dir(self, path: Path) -> bool:
        """Remove a directory from watching.

        Note: This doesn't immediately stop watching the directory
        if the watcher is running. Call stop_file_watcher() and
        setup_file_watcher() to apply the change.

        Args:
            path: Directory path to remove

        Returns:
            True if directory was removed
        """
        path = Path(path).expanduser()
        if path in self._watch_dirs:
            self._watch_dirs.remove(path)
            return True
        return False

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "DynamicModuleLoader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup."""
        self.stop_file_watcher()


# =============================================================================
# Entry Point Cache
# =============================================================================


@dataclass
class CachedEntryPoints:
    """Cached entry points data with metadata.

    Attributes:
        group: Entry point group name
        entries: Dictionary of name -> module path
        env_hash: Hash of package environment when cached
        timestamp: When the cache was created
        ttl: Time-to-live in seconds
    """

    group: str
    entries: Dict[str, str]  # name -> module:attr
    env_hash: str
    timestamp: float
    ttl: float = 3600.0  # 1 hour default

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if cache entry has expired.

        Args:
            current_time: Current time (defaults to time.time())

        Returns:
            True if expired
        """
        import time

        now = current_time or time.time()
        return (now - self.timestamp) > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "group": self.group,
            "entries": self.entries,
            "env_hash": self.env_hash,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedEntryPoints":
        """Deserialize from dictionary."""
        return cls(
            group=data["group"],
            entries=data["entries"],
            env_hash=data["env_hash"],
            timestamp=data["timestamp"],
            ttl=data.get("ttl", 3600.0),
        )


class EntryPointCache:
    """Cache for Python entry point scanning results.

    Entry point scanning via importlib.metadata.entry_points() can be slow
    as it scans all installed packages. This cache:

    1. Stores scan results in memory for fast repeated access
    2. Persists to disk for cross-session reuse
    3. Uses environment hash to detect package changes
    4. Provides TTL-based expiration as fallback

    Thread Safety:
        The cache is thread-safe for reads. Writes use a lock for
        atomic updates.

    Usage:
        cache = EntryPointCache()

        # Get entry points for a group (cached)
        eps = cache.get_entry_points("victor.plugins")

        # Force refresh if needed
        eps = cache.get_entry_points("victor.plugins", force_refresh=True)

        # Async version
        eps = await cache.get_entry_points_async("victor.tools")

    Attributes:
        cache_dir: Directory for persistent cache storage
        default_ttl: Default TTL in seconds for cache entries
    """

    # Singleton instance
    _instance: Optional["EntryPointCache"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: float = 3600.0,
    ) -> None:
        """Initialize entry point cache.

        Args:
            cache_dir: Directory for cache file (defaults to ~/.victor/cache)
            default_ttl: Default TTL in seconds (default 1 hour)
        """
        self._cache_dir = cache_dir or Path.home() / ".victor" / "cache"
        self._cache_file = self._cache_dir / "entry_points.json"
        self._default_ttl = default_ttl

        # In-memory cache
        self._memory_cache: Dict[str, CachedEntryPoints] = {}
        self._env_hash: Optional[str] = None
        self._cache_lock = threading.RLock()

        # Load from disk on init
        self._load_from_disk()

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        default_ttl: float = 3600.0,
    ) -> "EntryPointCache":
        """Get or create singleton instance.

        Args:
            cache_dir: Directory for cache file
            default_ttl: Default TTL in seconds

        Returns:
            Singleton EntryPointCache instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir, default_ttl)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _compute_env_hash(self) -> str:
        """Compute hash of installed package environment.

        Uses hash of sorted package names and versions to detect
        when the environment has changed (packages installed/removed).

        Returns:
            Hash string
        """
        import hashlib

        try:
            # Get installed packages (Python 3.10+)
            from importlib.metadata import distributions

            # Build sorted list of package:version pairs
            packages = []
            for dist in distributions():
                name = dist.metadata.get("Name", "unknown")
                version = dist.metadata.get("Version", "0.0.0")
                packages.append(f"{name}=={version}")

            packages.sort()
            package_str = "\n".join(packages)

            # Hash the package list
            return hashlib.sha256(package_str.encode()).hexdigest()[:16]

        except Exception as e:
            logger.warning(f"Failed to compute env hash: {e}")
            # Return timestamp-based hash as fallback
            import time

            return f"time_{int(time.time())}"

    def _compute_installation_fingerprint(self) -> str:
        """Compute lightweight installation fingerprint for warm-start checks.

        This avoids full package enumeration by hashing a small set of
        interpreter and site-package directory mtimes.
        """
        import hashlib
        import site
        import sys

        components = [
            f"exe:{sys.executable}",
            f"py:{sys.version_info.major}.{sys.version_info.minor}",
        ]
        candidate_paths = []
        try:
            candidate_paths.extend(site.getsitepackages())
        except Exception:
            pass

        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidate_paths.append(user_site)

        for raw_path in sorted({p for p in candidate_paths if p}):
            try:
                resolved = Path(raw_path).resolve()
                stat = resolved.stat()
                components.append(f"path:{resolved}:{stat.st_mtime_ns}:{stat.st_size}")
            except Exception:
                components.append(f"path:{raw_path}:missing")

        payload = "|".join(components)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_env_hash(self) -> str:
        """Get cached or compute environment hash.

        Returns:
            Environment hash string
        """
        if self._env_hash is None:
            self._env_hash = self._compute_env_hash()
        return self._env_hash

    def _load_from_disk(self) -> None:
        """Load cache from disk if available."""
        if not self._cache_file.exists():
            return

        try:
            import json

            with open(self._cache_file, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("Entry point cache file has invalid top-level format")
                return

            # Warm-start fast path: if installation fingerprint matches, reuse persisted env hash
            # and avoid the expensive full package-distribution scan.
            current_hash: Optional[str] = None
            meta = data.get("_meta")
            if isinstance(meta, dict):
                cached_env_hash = meta.get("env_hash")
                cached_fingerprint = meta.get("install_fingerprint")
                if isinstance(cached_env_hash, str) and isinstance(cached_fingerprint, str):
                    current_fingerprint = self._compute_installation_fingerprint()
                    if current_fingerprint == cached_fingerprint:
                        self._env_hash = cached_env_hash
                        current_hash = cached_env_hash
                        logger.debug(
                            "Reused cached env hash via installation fingerprint fast path"
                        )

            # Fallback for legacy cache files or fingerprint mismatch.
            if current_hash is None:
                current_hash = self._get_env_hash()

            with self._cache_lock:
                for group, entry_data in data.items():
                    if group == "_meta":
                        continue
                    try:
                        cached = CachedEntryPoints.from_dict(entry_data)
                        # Only load if env hash matches and not expired
                        if cached.env_hash == current_hash and not cached.is_expired():
                            self._memory_cache[group] = cached
                            logger.debug(f"Loaded cached entry points for '{group}'")
                    except Exception as e:
                        logger.warning(f"Failed to load cache entry for '{group}': {e}")

            if self._memory_cache:
                logger.info(f"Loaded {len(self._memory_cache)} entry point groups from cache")

        except Exception as e:
            logger.warning(f"Failed to load entry point cache from disk: {e}")

    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        try:
            import json
            import time

            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            with self._cache_lock:
                data = {group: cached.to_dict() for group, cached in self._memory_cache.items()}
                if self._env_hash is not None:
                    data["_meta"] = {
                        "schema_version": 1,
                        "env_hash": self._env_hash,
                        "install_fingerprint": self._compute_installation_fingerprint(),
                        "saved_at": time.time(),
                    }

            with open(self._cache_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved entry point cache to {self._cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save entry point cache to disk: {e}")

    def get_entry_points(
        self,
        group: str,
        force_refresh: bool = False,
    ) -> Dict[str, str]:
        """Get entry points for a group (cached).

        Args:
            group: Entry point group name (e.g., "victor.plugins")
            force_refresh: Force refresh from metadata (bypass cache)

        Returns:
            Dictionary mapping entry point names to module:attr strings
        """
        import time

        current_hash = self._get_env_hash()
        should_refresh_registry = force_refresh

        with self._cache_lock:
            cached = self._memory_cache.get(group)

            # Check cache validity
            if not force_refresh and cached is not None:
                if cached.env_hash == current_hash and not cached.is_expired():
                    logger.debug(f"Cache hit for entry point group '{group}'")
                    return cached.entries.copy()
                should_refresh_registry = True

        # Cache miss or invalid - scan entry points
        logger.debug(f"Scanning entry points for group '{group}'")
        entries = self._scan_entry_points(group, force_refresh=should_refresh_registry)

        # Store in cache
        cached_entry = CachedEntryPoints(
            group=group,
            entries=entries,
            env_hash=current_hash,
            timestamp=time.time(),
            ttl=self._default_ttl,
        )

        with self._cache_lock:
            self._memory_cache[group] = cached_entry

        # Persist to disk (async-safe)
        self._save_to_disk()

        return entries.copy()

    def _scan_entry_points(self, group: str, *, force_refresh: bool = False) -> Dict[str, str]:
        """Scan entry points for a group.

        Args:
            group: Entry point group name
            force_refresh: Whether to invalidate the shared registry first

        Returns:
            Dictionary mapping names to module:attr strings
        """
        entries = {}

        try:
            from victor.framework.entry_point_registry import get_entry_point_values

            entries = get_entry_point_values(group, force=force_refresh)

        except Exception as e:
            logger.warning(f"Failed to scan entry points for '{group}': {e}")

        return entries

    async def get_entry_points_async(
        self,
        group: str,
        force_refresh: bool = False,
    ) -> Dict[str, str]:
        """Get entry points for a group asynchronously.

        Offloads the potentially slow scan to a thread pool.

        Args:
            group: Entry point group name
            force_refresh: Force refresh from metadata

        Returns:
            Dictionary mapping entry point names to module:attr strings
        """
        import asyncio

        return await asyncio.to_thread(self.get_entry_points, group, force_refresh)

    def invalidate(self, group: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            group: Specific group to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        with self._cache_lock:
            if group:
                if group in self._memory_cache:
                    del self._memory_cache[group]
                    self._save_to_disk()
                    count = 1
                else:
                    count = 0
            else:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                self._env_hash = None  # Force re-compute
                self._save_to_disk()

        try:
            from victor.framework.entry_point_registry import get_entry_point_registry

            get_entry_point_registry().invalidate()
        except Exception as e:
            logger.debug(f"Failed to invalidate shared entry point registry: {e}")

        return count

    def invalidate_on_env_change(self) -> bool:
        """Check if environment changed and invalidate if so.

        Returns:
            True if cache was invalidated due to env change
        """
        old_hash = self._env_hash
        self._env_hash = None  # Force re-compute
        new_hash = self._get_env_hash()

        if old_hash != new_hash:
            logger.info("Package environment changed, invalidating entry point cache")
            self.invalidate()
            return True
        return False

    def get_cached_groups(self) -> List[str]:
        """Get list of cached entry point groups.

        Returns:
            List of group names
        """
        with self._cache_lock:
            return list(self._memory_cache.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        import time

        with self._cache_lock:
            stats = {
                "groups_cached": len(self._memory_cache),
                "env_hash": self._env_hash,
                "cache_file": str(self._cache_file),
                "groups": {},
            }

            now = time.time()
            for group, cached in self._memory_cache.items():
                stats["groups"][group] = {
                    "entries": len(cached.entries),
                    "age_seconds": now - cached.timestamp,
                    "ttl_remaining": max(0, cached.ttl - (now - cached.timestamp)),
                    "expired": cached.is_expired(now),
                }

            return stats


def get_entry_point_cache() -> EntryPointCache:
    """Get the singleton entry point cache instance.

    Returns:
        Global EntryPointCache instance
    """
    return EntryPointCache.get_instance()


__all__ = [
    "DynamicModuleLoader",
    "DebouncedReloadTimer",
    "EntryPointCache",
    "CachedEntryPoints",
    "get_entry_point_cache",
]
