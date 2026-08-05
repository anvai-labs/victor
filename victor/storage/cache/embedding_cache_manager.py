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

"""Unified embedding cache management for Victor.

This module provides centralized cache management for all embedding caches:
- Tool embeddings (~65KB)
- Task/intent classifiers (~800KB)
- Conversation embeddings (variable)
- Tiered result cache (variable)

Benefits:
- Single source of truth for cache operations
- Consistent versioning and validation
- Reusable across CLI, orchestrator, and background tasks
- Enables periodic cache refresh and incremental updates

Usage:
    from victor.storage.cache.embedding_cache_manager import EmbeddingCacheManager

    manager = EmbeddingCacheManager.get_instance()

    # Get cache status
    status = manager.get_status()
    for cache in status.caches:
        print(f"{cache.name}: {cache.file_count} files ({cache.size_str})")

    # Clear specific caches
    manager.clear(["tool", "intent"])

    # Rebuild task classifiers
    await manager.rebuild_task_classifiers()
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from victor.core.async_utils import run_sync

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of embedding caches."""

    TOOL = "tool"
    INTENT = "intent"
    TIERED = "tiered"
    CONVERSATION = "conversation"


@dataclass
class CacheFileInfo:
    """Information about a single cache file."""

    name: str
    size: int
    mtime: datetime

    @property
    def size_str(self) -> str:
        """Human-readable size string."""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        return f"{self.size / (1024 * 1024):.1f} MB"

    @property
    def age_str(self) -> str:
        """Human-readable age string."""
        delta = datetime.now() - self.mtime
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        return "just now"


@dataclass
class CacheInfo:
    """Information about a cache category."""

    cache_type: CacheType
    name: str
    description: str
    path: Path
    pattern: str
    files: List[CacheFileInfo] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def size_str(self) -> str:
        size = self.total_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    @property
    def newest(self) -> Optional[datetime]:
        if not self.files:
            return None
        return max(f.mtime for f in self.files)

    @property
    def age_str(self) -> str:
        if not self.newest:
            return "never"
        delta = datetime.now() - self.newest
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        return "just now"

    @property
    def is_empty(self) -> bool:
        return self.file_count == 0


@dataclass
class CacheStatus:
    """Overall cache status."""

    caches: List[CacheInfo]

    @property
    def total_files(self) -> int:
        return sum(c.file_count for c in self.caches)

    @property
    def total_size(self) -> int:
        return sum(c.total_size for c in self.caches)

    @property
    def total_size_str(self) -> str:
        size = self.total_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    def get_cache(self, cache_type: CacheType) -> Optional[CacheInfo]:
        """Get cache info by type."""
        for cache in self.caches:
            if cache.cache_type == cache_type:
                return cache
        return None


@dataclass
class ClearResult:
    """Result of cache clear operation."""

    cleared_files: int = 0
    cleared_size: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class EmbeddingCacheManager:
    """Centralized manager for all embedding caches.

    Provides consistent operations for:
    - Getting cache status
    - Clearing caches
    - Rebuilding caches
    - Periodic maintenance

    Thread-safe singleton implementation.
    """

    _instance: Optional["EmbeddingCacheManager"] = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingCacheManager":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        """Initialize cache manager with default paths."""
        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        self._global_embeddings = paths.global_embeddings_dir
        self._global_cache = paths.global_victor_dir / "cache"
        self._project_embeddings = paths.project_victor_dir / "embeddings"

        # Cache category definitions
        self._categories: Dict[CacheType, Dict] = {
            CacheType.TOOL: {
                "name": "Tool Embeddings",
                "desc": "Semantic tool selection (project-isolated)",
                "path": self._global_embeddings,
                "pattern": "tool_embeddings_*_*.pkl",  # model_hash pattern
            },
            CacheType.INTENT: {
                "name": "Task Classifier",
                "desc": "Task type detection (unified)",
                "path": self._global_embeddings,
                "pattern": "task_classifier_collection.pkl",
            },
            CacheType.TIERED: {
                "name": "Tiered Cache",
                "desc": "Tool result caching",
                "path": self._global_cache,
                "pattern": "**/*",
            },
            CacheType.CONVERSATION: {
                "name": "Conversation Embeddings",
                "desc": "Semantic conversation search",
                "path": self._project_embeddings,
                "pattern": "**/*",
            },
        }

    def _scan_cache_files(self, path: Path, pattern: str) -> List[CacheFileInfo]:
        """Scan directory for cache files matching pattern."""
        files = []
        if not path.exists():
            return files

        for f in path.glob(pattern):
            if f.is_file():
                try:
                    stat = f.stat()
                    files.append(
                        CacheFileInfo(
                            name=f.name,
                            size=stat.st_size,
                            mtime=datetime.fromtimestamp(stat.st_mtime),
                        )
                    )
                except OSError:
                    pass

        return sorted(files, key=lambda x: x.mtime, reverse=True)

    def get_status(self, include_tiered: bool = False) -> CacheStatus:
        """Get current status of embedding caches.

        Args:
            include_tiered: If True, also include tiered cache (tool results).
                           Default False since tiered cache is not embeddings.

        Returns:
            CacheStatus with cache info for each category.
        """
        caches = []
        for cache_type, cat in self._categories.items():
            # Skip tiered cache by default (it's tool results, not embeddings)
            if cache_type == CacheType.TIERED and not include_tiered:
                continue
            files = self._scan_cache_files(cat["path"], cat["pattern"])
            caches.append(
                CacheInfo(
                    cache_type=cache_type,
                    name=cat["name"],
                    description=cat["desc"],
                    path=cat["path"],
                    pattern=cat["pattern"],
                    files=files,
                )
            )
        return CacheStatus(caches=caches)

    def get_cache_info(self, cache_type: CacheType) -> CacheInfo:
        """Get info for a specific cache type."""
        cat = self._categories[cache_type]
        files = self._scan_cache_files(cat["path"], cat["pattern"])
        return CacheInfo(
            cache_type=cache_type,
            name=cat["name"],
            description=cat["desc"],
            path=cat["path"],
            pattern=cat["pattern"],
            files=files,
        )

    def clear(
        self,
        cache_types: Optional[List[CacheType]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ClearResult:
        """Clear specified caches.

        Args:
            cache_types: List of cache types to clear (None = all)
            progress_callback: Optional callback for progress updates

        Returns:
            ClearResult with counts and any errors
        """
        if cache_types is None:
            cache_types = list(CacheType)

        result = ClearResult()

        for cache_type in cache_types:
            cat = self._categories[cache_type]
            path = cat["path"]
            pattern = cat["pattern"]

            if not path.exists():
                if progress_callback:
                    progress_callback(f"  {cat['name']}: empty")
                continue

            try:
                if pattern == "**/*":
                    # Clear entire directory
                    file_count = sum(1 for _ in path.glob("**/*") if _.is_file())
                    total_size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

                    if file_count > 0:
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)
                        result.cleared_files += file_count
                        result.cleared_size += total_size
                        if progress_callback:
                            size_str = (
                                f"{total_size / 1024:.1f} KB"
                                if total_size >= 1024
                                else f"{total_size} B"
                            )
                            progress_callback(f"  {cat['name']}: {file_count} files ({size_str})")
                    else:
                        if progress_callback:
                            progress_callback(f"  {cat['name']}: empty")
                else:
                    # Clear matching files only
                    files = list(path.glob(pattern))
                    file_count = len(files)
                    total_size = sum(f.stat().st_size for f in files if f.is_file())

                    if file_count > 0:
                        for f in files:
                            if f.is_file():
                                f.unlink()
                        result.cleared_files += file_count
                        result.cleared_size += total_size
                        if progress_callback:
                            size_str = (
                                f"{total_size / 1024:.1f} KB"
                                if total_size >= 1024
                                else f"{total_size} B"
                            )
                            progress_callback(f"  {cat['name']}: {file_count} files ({size_str})")
                    else:
                        if progress_callback:
                            progress_callback(f"  {cat['name']}: empty")

            except Exception as e:
                error_msg = f"{cat['name']}: {e}"
                result.errors.append(error_msg)
                logger.warning(f"Failed to clear cache {cat['name']}: {e}")
                if progress_callback:
                    progress_callback(f"  {cat['name']}: ERROR - {e}")

        return result

    async def rebuild_task_classifiers(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Rebuild task/intent classifier embeddings.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Number of phrases indexed
        """
        from victor.storage.embeddings.service import get_embedding_service
        from victor.storage.embeddings.task_classifier import TaskTypeClassifier

        if progress_callback:
            progress_callback("Loading embedding model...")

        service = get_embedding_service()

        if progress_callback:
            progress_callback("Rebuilding task classifiers...")

        # Reset singleton to force reinitialization
        TaskTypeClassifier.reset_instance()
        classifier = TaskTypeClassifier.get_instance(embedding_service=service)
        classifier.initialize_sync()

        # Count total phrases
        phrase_count = sum(len(phrases) for phrases in classifier._phrase_lists.values())

        if progress_callback:
            progress_callback(f"Task classifiers rebuilt ({phrase_count} phrases)")

        logger.info(f"EmbeddingCacheManager: rebuilt task classifiers with {phrase_count} phrases")
        return phrase_count

    def rebuild_task_classifiers_sync(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Sync version of rebuild_task_classifiers.

        This method can only be called from synchronous contexts.
        If called from an async context, it will raise a RuntimeError.

        Raises:
            RuntimeError: If called from an async context (use await rebuild_task_classifiers() instead)

        Returns:
            Number of phrases indexed
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run_sync(self.rebuild_task_classifiers(progress_callback))

        raise RuntimeError(
            "Cannot call rebuild_task_classifiers_sync from async context. "
            "Use 'await rebuild_task_classifiers()' instead."
        )

    def needs_rebuild(self, cache_type: CacheType, max_age_hours: int = 24) -> bool:
        """Check if cache needs rebuilding based on age.

        Args:
            cache_type: Type of cache to check
            max_age_hours: Maximum age in hours before rebuild is needed

        Returns:
            True if cache is empty or older than max_age_hours
        """
        info = self.get_cache_info(cache_type)
        if info.is_empty:
            return True

        if info.newest:
            age_hours = (datetime.now() - info.newest).total_seconds() / 3600
            return age_hours > max_age_hours

        return True

    def invalidate_tool_cache(self) -> None:
        """Invalidate tool embeddings cache.

        Call this when tools are added/removed/modified.
        The cache will be rebuilt on next use.
        """
        self.clear([CacheType.TOOL])
        logger.info("EmbeddingCacheManager: tool cache invalidated")

    def invalidate_intent_cache(self) -> None:
        """Invalidate intent/task classifier cache.

        Call this when classifier phrases are modified.
        """
        self.clear([CacheType.INTENT])
        TaskTypeClassifier = __import__(
            "victor.storage.embeddings.task_classifier", fromlist=["TaskTypeClassifier"]
        ).TaskTypeClassifier
        TaskTypeClassifier.reset_instance()
        logger.info("EmbeddingCacheManager: intent cache invalidated")
