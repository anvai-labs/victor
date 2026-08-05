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

"""Unified cache management tool for Victor's caching system.

Consolidates all cache operations into a single tool for better token efficiency.
Supports: stats, clear, info.
"""

from typing import Dict, Any, Optional
import logging

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.storage.cache.tiered_cache import TieredCache

logger = logging.getLogger(__name__)


def _get_cache_manager(
    context: Optional[Dict[str, Any]] = None,
) -> Optional[TieredCache]:
    """Get cache manager from execution context.

    Args:
        context: Tool execution context

    Returns:
        TieredCache instance if available in context, None otherwise
    """
    if context:
        return context.get("cache_manager")
    return None


async def _do_stats(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get cache statistics."""
    cache_manager = _get_cache_manager(context)
    if cache_manager is None:
        return {"success": False, "error": "Cache manager not initialized"}

    stats = cache_manager.get_stats()

    # Build formatted report
    report = []
    report.append("Cache Statistics")
    report.append("=" * 70)
    report.append("")

    report.append("Performance:")
    report.append(f"  Memory Hit Rate: {stats.get('memory_hit_rate', 0):.2%}")
    report.append(f"  Disk Hit Rate: {stats.get('disk_hit_rate', 0):.2%}")
    report.append("")

    report.append("Counts:")
    report.append(f"  Memory Hits: {stats.get('memory_hits', 0)}")
    report.append(f"  Memory Misses: {stats.get('memory_misses', 0)}")
    report.append(f"  Disk Hits: {stats.get('disk_hits', 0)}")
    report.append(f"  Disk Misses: {stats.get('disk_misses', 0)}")
    report.append(f"  Total Sets: {stats.get('sets', 0)}")
    report.append("")

    if "memory_size" in stats:
        report.append("Memory Cache:")
        report.append(f"  Current Size: {stats['memory_size']}/{stats.get('memory_max_size', 0)}")
        report.append("")

    if "disk_size" in stats:
        report.append("Disk Cache:")
        report.append(f"  Entries: {stats['disk_size']}")
        if "disk_volume" in stats:
            volume_mb = stats["disk_volume"] / (1024 * 1024)
            report.append(f"  Volume: {volume_mb:.2f} MB")

    return {"success": True, "stats": stats, "formatted_report": "\n".join(report)}


async def _do_clear(
    namespace: Optional[str] = None, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Clear cache entries."""
    cache_manager = _get_cache_manager(context)
    if cache_manager is None:
        return {"success": False, "error": "Cache manager not initialized"}

    try:
        count = cache_manager.clear(namespace)

        if namespace:
            message = f"Cleared {count} entries from namespace '{namespace}'"
        else:
            message = f"Cleared all cache ({count} entries)"

        return {"success": True, "cleared_count": count, "message": message}

    except Exception as e:
        logger.exception("Failed to clear cache")
        return {"success": False, "error": f"Failed to clear cache: {str(e)}"}


async def _do_info(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get cache configuration info."""
    cache_manager = _get_cache_manager(context)
    if cache_manager is None:
        return {"success": False, "error": "Cache manager not initialized"}

    config = cache_manager.config

    report = []
    report.append("Cache Configuration")
    report.append("=" * 70)
    report.append("")

    report.append("Architecture: Tiered (L1 Memory + L2 Disk)")
    report.append("")

    report.append("L1 Memory Cache:")
    report.append(f"  Enabled: {config.enable_memory}")
    if config.enable_memory:
        report.append(f"  Max Size: {config.memory_max_size} entries")
        report.append(f"  TTL: {config.memory_ttl} seconds ({config.memory_ttl//60} min)")
    report.append("")

    report.append("L2 Disk Cache:")
    report.append(f"  Enabled: {config.enable_disk}")
    if config.enable_disk:
        size_mb = config.disk_max_size / (1024 * 1024)
        report.append(f"  Max Size: {size_mb:.0f} MB")
        report.append(f"  TTL: {config.disk_ttl} seconds ({config.disk_ttl//86400} days)")
        report.append(f"  Path: {config.disk_path}")
    report.append("")

    report.append("Features:")
    report.append("  - Automatic tiering (memory -> disk -> source)")
    report.append("  - TTL-based expiration")
    report.append("  - Thread-safe operations")
    report.append("  - Persistent across restarts (disk cache)")
    report.append("  - Zero external dependencies")

    return {
        "success": True,
        "config": {
            "enable_memory": config.enable_memory,
            "memory_max_size": config.memory_max_size,
            "memory_ttl": config.memory_ttl,
            "enable_disk": config.enable_disk,
            "disk_max_size": config.disk_max_size,
            "disk_ttl": config.disk_ttl,
            "disk_path": str(config.disk_path),
        },
        "formatted_report": "\n".join(report),
    }


@tool(
    category="cache",
    priority=Priority.LOW,  # Administrative tool
    access_mode=AccessMode.MIXED,  # Reads stats and can clear cache
    danger_level=DangerLevel.LOW,  # Cache clearing is safe
    keywords=["cache", "clear", "stats", "memory", "disk"],
)
async def cache(
    action: str,
    namespace: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Manage Victor caches: inspect, clear, and optimize cached data.

    Actions:
    - stats: Get cache statistics and performance metrics
    - clear: Clear cache entries (optionally by namespace)
    - info: Get cache configuration information

    Args:
        action: Operation to perform - 'stats', 'clear', or 'info'.
        namespace: Optional namespace for clear action (e.g., "responses", "embeddings").
                  If not provided with clear, clears all cache.
        context: Tool execution context containing cache_manager.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - For stats: stats dict, formatted_report
        - For clear: cleared_count, message
        - For info: config dict, formatted_report
        - error: Error message if failed

    Examples:
        # Get cache statistics
        cache(action="stats")

        # Clear all cache
        cache(action="clear")

        # Clear specific namespace
        cache(action="clear", namespace="responses")

        # Get cache configuration
        cache(action="info")
    """
    action_lower = action.lower().strip()

    if action_lower == "stats":
        return await _do_stats(context)

    elif action_lower == "clear":
        return await _do_clear(namespace, context)

    elif action_lower == "info":
        return await _do_info(context)

    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}. Valid actions: stats, clear, info",
        }
