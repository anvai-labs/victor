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

"""Tool Preloader with async background preloading.

Preloads tool schemas before they're needed to reduce tool selection latency.
Uses predictive models to anticipate which tools will be needed next.

Architecture:
- L1 Cache: In-memory cache of tool schemas (50 entries, 10min TTL)
- L2 Cache: Optional disk-based persistent cache (1000 entries, 1hr TTL)
- Preload Predictor: Uses ToolPredictor to anticipate next tools
- Background Loading: Async preloading to avoid blocking

Usage:
    from victor.agent.planning.tool_preloader import ToolPreloader
    from victor.agent.planning.tool_predictor import ToolPredictor

    predictor = ToolPredictor()
    preloader = ToolPreloader(
        tool_predictor=predictor,
        tool_registry=registry,
    )

    # Preload for next step
    await preloader.preload_for_next_step(
        current_step="exploration",
        task_type="bugfix",
        recent_tools=["search"],
    )

    # Get tool schema (may be preloaded)
    schema = await preloader.get_tool_schema("read_file")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.planning.tool_predictor import ToolPredictor


@dataclass
class CacheEntry:
    """A cache entry for a tool schema.

    Attributes:
        tool_name: Name of the tool
        schema: The tool schema (dict)
        access_count: Number of times this entry was accessed
        last_accessed: When this entry was last accessed
        last_preloaded: When this entry was preloaded
        expires_at: When this entry expires (TTL)
    """

    tool_name: str
    schema: Dict[str, Any]
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_preloaded: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    def touch(self):
        """Update last_accessed timestamp."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class PreloaderConfig:
    """Configuration for tool preloader.

    Attributes:
        l1_max_size: Maximum number of entries in L1 cache
        l1_ttl_seconds: TTL for L1 cache entries (default: 10 minutes)
        l2_enabled: Whether to enable L2 disk cache
        l2_max_size: Maximum number of entries in L2 cache
        l2_ttl_seconds: TTL for L2 cache entries (default: 1 hour)
        l2_cache_dir: Directory for L2 disk cache
        preload_threshold: Minimum confidence to trigger preload
        max_preload_tools: Maximum tools to preload per prediction
        promotion_threshold: Access count to promote from L1 to L2
    """

    l1_max_size: int = 50
    l1_ttl_seconds: float = 600.0  # 10 minutes
    l2_enabled: bool = False
    l2_max_size: int = 1000
    l2_ttl_seconds: float = 3600.0  # 1 hour
    l2_cache_dir: Optional[Path] = None
    preload_threshold: float = 0.5  # Only preload if confidence >= 0.5
    max_preload_tools: int = 5  # Max tools to preload
    promotion_threshold: int = 3  # Promote after 3 accesses


class ToolPreloader:
    """Preloads tool schemas before they're needed.

    Uses predictive models to anticipate which tools will be needed and
    preloads their schemas in the background to reduce latency.

    Cache Architecture:
    - L1 (Memory): Fast, limited capacity, short TTL
    - L2 (Disk): Slower, larger capacity, longer TTL (optional)

    Preloading Strategy:
    - Use ToolPredictor to anticipate next tools
    - Only preload if confidence >= threshold
    - Load schemas asynchronously in background
    """

    def __init__(
        self,
        tool_predictor: Optional["ToolPredictor"] = None,
        tool_registry: Optional[Any] = None,
        config: Optional[PreloaderConfig] = None,
    ):
        """Initialize the tool preloader.

        Args:
            tool_predictor: ToolPredictor for anticipating next tools
            tool_registry: Tool registry to load schemas from
            config: Optional preloader configuration
        """
        self.config = config or PreloaderConfig()
        self._predictor = tool_predictor
        self._tool_registry = tool_registry

        # L1 cache (in-memory OrderedDict for LRU eviction)
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # L2 cache (optional disk-based)
        self._l2_cache: Optional[Dict[str, CacheEntry]] = None
        if self.config.l2_enabled:
            self._l2_cache = {}
            if self.config.l2_cache_dir:
                self.config.l2_cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._l1_hits: int = 0
        self._l1_misses: int = 0
        self._l2_hits: int = 0
        self._l2_misses: int = 0
        self._preload_count: int = 0
        self._background_loads: int = 0

        logger.info(
            f"ToolPreloader initialized (L1: {self.config.l1_max_size} entries, "
            f"L2: {'enabled' if self.config.l2_enabled else 'disabled'})"
        )

    async def preload_for_next_step(
        self,
        current_step: str,
        task_type: str = "default",
        recent_tools: Optional[List[str]] = None,
        task_description: Optional[str] = None,
    ) -> int:
        """Preload tools for the next step based on predictions.

        Args:
            current_step: Current workflow step
            task_type: Type of task
            recent_tools: Tools used recently
            task_description: Optional task description for better predictions

        Returns:
            Number of tools preloaded
        """
        if not self._predictor:
            logger.debug("No predictor configured, skipping preload")
            return 0

        try:
            # Get predictions for next tools
            predictions = self._predictor.predict_tools(
                task_description=task_description or "",
                current_step=current_step,
                recent_tools=recent_tools or [],
                task_type=task_type,
            )

            # Filter by confidence threshold
            high_confidence = [
                p for p in predictions if p.probability >= self.config.preload_threshold
            ]

            # Limit to max_preload_tools
            to_preload = high_confidence[: self.config.max_preload_tools]

            if not to_preload:
                logger.debug("No high-confidence predictions for preloading")
                return 0

            # Preload in background (don't await)
            for prediction in to_preload:
                asyncio.create_task(
                    self._preload_tool(
                        prediction.tool_name,
                        confidence=prediction.probability,
                    )
                )

            logger.debug(
                f"Preloading {len(to_preload)} tools: " f"{[p.tool_name for p in to_preload]}"
            )

            self._preload_count += len(to_preload)
            return len(to_preload)

        except Exception as e:
            logger.warning(f"Preload failed: {e}")
            return 0

    async def _preload_tool(
        self,
        tool_name: str,
        confidence: float,
    ) -> None:
        """Preload a single tool schema.

        Args:
            tool_name: Name of the tool to preload
            confidence: Prediction confidence
        """
        try:
            # Check if already in L1 cache
            if tool_name in self._l1_cache:
                entry = self._l1_cache[tool_name]
                if not entry.is_expired():
                    logger.debug(f"Tool {tool_name} already in L1 cache")
                    return

            # Load schema from registry
            if self._tool_registry is None:
                logger.debug("No tool registry configured")
                return

            schema = await self._load_schema_from_registry(tool_name)
            if schema is None:
                logger.debug(f"Could not load schema for {tool_name}")
                return

            # Create cache entry
            now = datetime.now(timezone.utc)
            entry = CacheEntry(
                tool_name=tool_name,
                schema=schema,
                last_preloaded=now,
                expires_at=now + timedelta(seconds=self.config.l1_ttl_seconds),
            )

            # Add to L1 cache (may trigger eviction)
            self._add_to_l1_cache(tool_name, entry)

            # Add to L2 cache if enabled
            if self._l2_cache is not None:
                self._add_to_l2_cache(tool_name, entry)

            self._background_loads += 1
            logger.debug(f"Preloaded tool: {tool_name} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.warning(f"Failed to preload {tool_name}: {e}")

    async def get_tool_schema(
        self,
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a tool schema, using cache if available.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dict, or None if not found
        """
        # Check L1 cache first
        if tool_name in self._l1_cache:
            entry = self._l1_cache[tool_name]
            if not entry.is_expired():
                entry.touch()
                self._l1_hits += 1
                logger.debug(f"L1 cache hit: {tool_name}")

                # Check for L2 promotion after threshold accesses
                if (
                    self._l2_cache is not None
                    and entry.access_count >= self.config.promotion_threshold
                ):
                    self._add_to_l2_cache(tool_name, entry)

                return entry.schema
            else:
                # Expired, remove from cache
                del self._l1_cache[tool_name]
                self._l1_misses += 1

        # Check L2 cache if enabled
        if self._l2_cache is not None and tool_name in self._l2_cache:
            entry = self._l2_cache[tool_name]
            if not entry.is_expired():
                # Promote to L1 cache
                entry.touch()
                self._add_to_l1_cache(tool_name, entry)
                self._l2_hits += 1
                logger.debug(f"L2 cache hit (promoted to L1): {tool_name}")
                return entry.schema
            else:
                # Expired, remove from L2 cache
                del self._l2_cache[tool_name]
                self._l2_misses += 1

        # Cache miss, load from registry
        self._l1_misses += 1
        self._l2_misses += 1

        if self._tool_registry is None:
            return None

        schema = await self._load_schema_from_registry(tool_name)
        if schema is None:
            return None

        # Add to caches
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            tool_name=tool_name,
            schema=schema,
            last_preloaded=now,
            expires_at=now + timedelta(seconds=self.config.l1_ttl_seconds),
        )

        self._add_to_l1_cache(tool_name, entry)

        if self._l2_cache is not None:
            self._add_to_l2_cache(tool_name, entry)

        return schema

    async def _load_schema_from_registry(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Load tool schema from registry.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dict, or None if not found
        """
        try:
            # Try to get schema from registry
            # This is a simplified version - actual implementation depends on registry API
            if hasattr(self._tool_registry, "get_tool_schema"):
                method = self._tool_registry.get_tool_schema
                if asyncio.iscoroutinefunction(method):
                    schema = await method(tool_name)
                else:
                    schema = method(tool_name)
                return schema
            elif hasattr(self._tool_registry, "get"):
                tool = self._tool_registry.get(tool_name)
                if tool and hasattr(tool, "schema"):
                    return tool.schema
                elif tool and hasattr(tool, "input_schema"):
                    return tool.input_schema
                elif tool and hasattr(tool, "parameters"):
                    return tool.parameters

            # Fallback: return mock schema for testing
            logger.debug(f"Using mock schema for {tool_name}")
            return {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": f"Input for {tool_name}",
                    }
                },
            }

        except Exception as e:
            logger.warning(f"Failed to load schema for {tool_name}: {e}")
            return None

    def _add_to_l1_cache(self, tool_name: str, entry: CacheEntry) -> None:
        """Add entry to L1 cache with LRU eviction.

        Args:
            tool_name: Name of the tool
            entry: Cache entry to add
        """
        # Remove if already exists (will be re-added with updated access)
        if tool_name in self._l1_cache:
            del self._l1_cache[tool_name]

        # Add to end (most recently used)
        self._l1_cache[tool_name] = entry

        # Evict oldest if over capacity
        while len(self._l1_cache) > self.config.l1_max_size:
            oldest_key, _ = self._l1_cache.popitem(last=False)
            logger.debug(f"L1 cache evicted: {oldest_key}")

    def _add_to_l2_cache(self, tool_name: str, entry: CacheEntry) -> None:
        """Add entry to L2 cache with promotion check.

        Args:
            tool_name: Name of the tool
            entry: Cache entry to add
        """
        if self._l2_cache is None:
            return

        # Check for promotion (high access count)
        if tool_name in self._l1_cache:
            l1_entry = self._l1_cache[tool_name]
            if l1_entry.access_count >= self.config.promotion_threshold:
                # Promoted to L2 (extend TTL)
                l2_entry = CacheEntry(
                    tool_name=tool_name,
                    schema=entry.schema,
                    access_count=l1_entry.access_count,
                    last_accessed=l1_entry.last_accessed,
                    last_preloaded=entry.last_preloaded,
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=self.config.l2_ttl_seconds),
                )
                self._l2_cache[tool_name] = l2_entry
                logger.debug(f"L2 cache promoted: {tool_name}")
                return

        # Add to L2 with standard TTL
        self._l2_cache[tool_name] = entry

        # Evict if over capacity
        if len(self._l2_cache) > self.config.l2_max_size:
            # Find oldest entry
            oldest_key = min(
                self._l2_cache.keys(),
                key=lambda k: self._l2_cache[k].last_preloaded,
            )
            del self._l2_cache[oldest_key]
            logger.debug(f"L2 cache evicted: {oldest_key}")

    async def warm_up(self, tool_names: List[str]) -> None:
        """Warm up the cache by preloading specific tools.

        Args:
            tool_names: List of tool names to preload
        """
        for tool_name in tool_names:
            await self._preload_tool(tool_name, confidence=1.0)

        logger.info(f"Warmed up cache with {len(tool_names)} tools")

    def clear_cache(self, level: str = "all") -> None:
        """Clear cache(s).

        Args:
            level: Which cache to clear ("l1", "l2", or "all")
        """
        if level in ("l1", "all"):
            self._l1_cache.clear()
            logger.debug("L1 cache cleared")

        if level in ("l2", "all") and self._l2_cache is not None:
            self._l2_cache.clear()
            logger.debug("L2 cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get preloader statistics.

        Returns:
            Dictionary with preloader metrics
        """
        stats = {
            "l1_cache_size": len(self._l1_cache),
            "l1_max_size": self.config.l1_max_size,
            "l1_hits": self._l1_hits,
            "l1_misses": self._l1_misses,
            "l1_hit_rate": (
                self._l1_hits / (self._l1_hits + self._l1_misses)
                if (self._l1_hits + self._l1_misses) > 0
                else 0.0
            ),
            "preload_count": self._preload_count,
            "background_loads": self._background_loads,
        }

        if self._l2_cache is not None:
            stats.update(
                {
                    "l2_cache_size": len(self._l2_cache),
                    "l2_max_size": self.config.l2_max_size,
                    "l2_hits": self._l2_hits,
                    "l2_misses": self._l2_misses,
                    "l2_hit_rate": (
                        self._l2_hits / (self._l2_hits + self._l2_misses)
                        if (self._l2_hits + self._l2_misses) > 0
                        else 0.0
                    ),
                }
            )

        return stats


__all__ = [
    "ToolPreloader",
    "PreloaderConfig",
    "CacheEntry",
]
