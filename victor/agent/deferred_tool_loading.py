# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Budget-aware compaction with deferred tool loading (P2-3).

Replaces large tool results with lightweight placeholders that can be
loaded on-demand. This reduces context usage while preserving the ability
to access previous tool results when needed.

Key benefits:
- Reduces context usage by replacing large tool results with placeholders
- Preserves ability to restore results when needed
- LRU eviction prevents unbounded memory growth
- Tracks bytes saved for metrics

Usage:
    manager = DeferredLoadingManager()
    result_id = manager.store_result("read", {"path": "file.py"}, content)
    placeholder = create_placeholder_for_tool_result("read", {"path": "file.py"}, content, result_id)
    # Later...
    restored = restore_tool_result(placeholder, manager)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeferredToolResult:
    """A stored tool result that can be restored later.

    Attributes:
        tool_name: Name of the tool that produced the result
        tool_args: Arguments passed to the tool
        content: The original tool result content
        timestamp: Unix timestamp when stored
    """

    tool_name: str
    tool_args: Dict[str, Any]
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultPlaceholder:
    """A placeholder for a deferred tool result.

    This lightweight object replaces large tool results in the context,
    with enough information to restore the original if needed.

    Attributes:
        tool_name: Name of the tool that produced the result
        tool_args: Arguments passed to the tool
        original_length: Length of the original content
        result_id: ID to use for restoration
        content: Placeholder text content
    """

    tool_name: str
    tool_args: Dict[str, Any]
    original_length: int
    result_id: str
    content: str = field(init=False)

    def __post_init__(self):
        """Generate placeholder content."""
        self.content = (
            f"[Deferred tool result: {self.tool_name} "
            f"({self.original_length} chars, use result_id '{self.result_id}' to restore)]"
        )


@dataclass
class DeferredLoadingConfig:
    """Configuration for deferred tool loading.

    Attributes:
        min_size_to_defer: Minimum content size to trigger deferral
        defer_tool_results: Whether to defer tool results
        defer_assistant: Whether to defer large assistant messages
        max_stored_results: Maximum number of results to store (LRU eviction)
        defer_roles: Set of roles that can be deferred
    """

    min_size_to_defer: int = 1000  # chars
    defer_tool_results: bool = True
    defer_assistant: bool = False
    max_stored_results: int = 100
    defer_roles: set = field(default_factory=lambda: {"tool"})


class DeferredLoadingManager:
    """Manages deferred tool results with LRU eviction.

    Stores large tool results and provides restoration capability.
    Uses LRU eviction to prevent unbounded memory growth.

    Example:
        manager = DeferredLoadingManager()
        result_id = manager.store_result("read", {"path": "file.py"}, content)
        restored = manager.get_result(result_id)
    """

    def __init__(self, config: Optional[DeferredLoadingConfig] = None):
        """Initialize the deferred loading manager.

        Args:
            config: Optional custom configuration
        """
        self.config = config or DeferredLoadingConfig()
        self._store: Dict[str, DeferredToolResult] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._total_bytes_saved = 0

    def store_result(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        content: str,
    ) -> str:
        """Store a tool result for later retrieval.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            content: The result content

        Returns:
            Result ID for restoration
        """
        # Check if we need to evict
        if len(self._store) >= self.config.max_stored_results:
            self._evict_lru()

        import hashlib

        result_id = hashlib.md5(
            f"{tool_name}:{str(tool_args)}:{content[:100]}".encode(), usedforsecurity=False
        ).hexdigest()[:12]

        result = DeferredToolResult(
            tool_name=tool_name,
            tool_args=tool_args,
            content=content,
        )

        self._store[result_id] = result
        self._access_order.append(result_id)
        self._total_bytes_saved += len(content)

        logger.debug(
            f"Stored deferred tool result: {result_id} " f"({len(content)} chars from {tool_name})"
        )

        return result_id

    def get_result(self, result_id: str) -> Optional[DeferredToolResult]:
        """Get a stored tool result.

        Args:
            result_id: The result ID from store_result

        Returns:
            The stored result, or None if not found
        """
        if result_id not in self._store:
            return None

        # Update access order for LRU
        if result_id in self._access_order:
            self._access_order.remove(result_id)
        self._access_order.append(result_id)

        return self._store[result_id]

    def _evict_lru(self) -> None:
        """Evict the least recently used result."""
        if not self._access_order:
            return

        lru_id = self._access_order.pop(0)
        if lru_id in self._store:
            evicted = self._store.pop(lru_id)
            logger.debug(
                f"Evicted LRU tool result: {lru_id} "
                f"({len(evicted.content)} chars from {evicted.tool_name})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the manager.

        Returns:
            Dictionary with stats
        """
        return {
            "total_stored": len(self._store),
            "max_capacity": self.config.max_stored_results,
            "total_bytes_saved": self._total_bytes_saved,
            "utilization": len(self._store) / self.config.max_stored_results,
        }

    def clear(self) -> None:
        """Clear all stored results."""
        self._store.clear()
        self._access_order.clear()
        self._total_bytes_saved = 0


def should_defer_tool_result(
    role: str,
    content: str,
    config: Optional[DeferredLoadingConfig] = None,
) -> bool:
    """Check if a tool result should be deferred.

    Args:
        role: Message role (tool, assistant, etc.)
        content: Message content
        config: Optional custom configuration

    Returns:
        True if the result should be deferred
    """
    config = config or DeferredLoadingConfig()

    # Check if deferral is enabled
    if not config.defer_tool_results:
        return False

    # Check role
    if role not in config.defer_roles:
        return False

    # Check size
    return len(content) >= config.min_size_to_defer


def create_placeholder_for_tool_result(
    tool_name: str,
    tool_args: Dict[str, Any],
    content: str,
    result_id: str,
) -> ToolResultPlaceholder:
    """Create a placeholder for a deferred tool result.

    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments
        content: Original content (for length calculation)
        result_id: Result ID for restoration

    Returns:
        ToolResultPlaceholder object
    """
    return ToolResultPlaceholder(
        tool_name=tool_name,
        tool_args=tool_args,
        original_length=len(content),
        result_id=result_id,
    )


def restore_tool_result(
    placeholder: ToolResultPlaceholder,
    manager: DeferredLoadingManager,
) -> Optional[str]:
    """Restore a tool result from a placeholder.

    Args:
        placeholder: The placeholder object
        manager: The manager that stored the result

    Returns:
        The original content, or None if not found
    """
    result = manager.get_result(placeholder.result_id)
    if result is None:
        return None

    return result.content


# Singleton instance for convenience
_default_manager: Optional[DeferredLoadingManager] = None


def get_deferred_result_store() -> DeferredLoadingManager:
    """Get the singleton deferred result store.

    Returns:
        The default DeferredLoadingManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = DeferredLoadingManager()
    return _default_manager
