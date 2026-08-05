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

"""Tool event emitter for tracking tool execution.

SOLID Principles:
- SRP: Focused solely on tool execution events
- OCP: Extensible via inheritance (can add custom tool tracking)
- LSP: Substitutable with IToolEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("tool.start") instead of category-based
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

import contextlib
import logging
import time
import traceback
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from victor.observability.emitters.base import IToolEventEmitter
from victor.core.events import ObservabilityBus, SyncEventWrapper

logger = logging.getLogger(__name__)


class ToolEventEmitter(IToolEventEmitter):
    """Emits tool execution events to ObservabilityBus.

    Thread-safe, performant tool execution tracking.
    Handles errors gracefully to avoid impacting tool execution.

    Example:
        >>> emitter = ToolEventEmitter()
        >>> emitter.tool_start("read_file", {"path": "file.txt"}, agent_id="agent-1")
        >>> # ... execute tool ...
        >>> emitter.tool_end("read_file", 150.0, result="file contents")

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("tool.start", "tool.result") instead of category-based routing.
    """

    def __init__(self, bus: Optional[ObservabilityBus] = None):
        """Initialize the tool event emitter.

        Args:
            bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._bus = bus
        self._sync_wrapper: Optional[SyncEventWrapper] = None
        self._enabled = True

    def _get_bus(self) -> Optional[ObservabilityBus]:
        """Get ObservabilityBus instance.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        if self._bus:
            return self._bus

        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _get_sync_wrapper(self) -> Optional[SyncEventWrapper]:
        """Get sync wrapper for gradual migration.

        Returns:
            SyncEventWrapper instance or None if unavailable
        """
        if self._sync_wrapper:
            return self._sync_wrapper

        bus = self._get_bus()
        if bus:
            self._sync_wrapper = SyncEventWrapper(bus.backend)
            return self._sync_wrapper

        return None

    async def emit_async(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> bool:
        """Emit a tool event asynchronously.

        Args:
            topic: Event topic (e.g., "tool.start", "tool.result")
            data: Event payload

        Returns:
            True if emission succeeded, False otherwise
        """
        if not self._enabled:
            return False

        bus = self._get_bus()
        if bus:
            try:
                # Add category metadata for observability features
                data_with_category = {
                    **data,
                    "category": "tool",
                }
                return await bus.emit(topic, data_with_category)
            except Exception as e:
                logger.debug(f"Failed to emit tool event: {e}")
                return False

        return False

    def emit(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a tool event synchronously (for gradual migration).

        This method wraps the async emit_async() method using emit_event_sync()
        so sync callers can reuse the shared async/sync bridge behavior.

        Args:
            topic: Event topic (e.g., "tool.start", "tool.result")
            data: Event payload
        """
        try:
            from victor.core.events.emit_helper import emit_event_sync

            bus = self._get_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic=topic,
                    data=data,
                    source="ToolEventEmitter",
                )
        except Exception as e:
            logger.debug(f"Failed to emit tool event: {e}")

    async def tool_start_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ) -> bool:
        """Emit tool execution start event asynchronously.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            **metadata: Additional metadata (agent_id, session_id, etc.)

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="tool.start",
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                **metadata,
            },
        )

    def tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ) -> None:
        """Emit tool execution start event (sync wrapper).

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            **metadata: Additional metadata (agent_id, session_id, etc.)
        """
        self.emit(
            topic="tool.start",
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                **metadata,
            },
        )

    async def tool_end_async(
        self,
        tool_name: str,
        duration_ms: float,
        result: Optional[Any] = None,
        **metadata: Any,
    ) -> bool:
        """Emit tool execution end event asynchronously (success).

        Args:
            tool_name: Name of the tool that completed
            duration_ms: Execution duration in milliseconds
            result: Tool result (will be truncated if large)
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        # Truncate large results for event data
        result_str = None
        if result is not None:
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"

        return await self.emit_async(
            topic="tool.result",
            data={
                "tool_name": tool_name,
                "success": True,
                "duration_ms": duration_ms,
                "result": result_str,
                **metadata,
            },
        )

    def tool_end(
        self,
        tool_name: str,
        duration_ms: float,
        result: Optional[Any] = None,
        **metadata: Any,
    ) -> None:
        """Emit tool execution end event (sync wrapper).

        Args:
            tool_name: Name of the tool that completed
            duration_ms: Execution duration in milliseconds
            result: Tool result (will be truncated if large)
            **metadata: Additional metadata
        """
        # Truncate large results for event data
        result_str = None
        if result is not None:
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"

        self.emit(
            topic="tool.result",
            data={
                "tool_name": tool_name,
                "success": True,
                "duration_ms": duration_ms,
                "result": result_str,
                **metadata,
            },
        )

    async def tool_failure_async(
        self,
        tool_name: str,
        duration_ms: float,
        error: Exception,
        **metadata: Any,
    ) -> bool:
        """Emit tool execution failure event asynchronously.

        Args:
            tool_name: Name of the tool that failed
            duration_ms: Execution duration before failure
            error: The exception that occurred
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="tool.error",
            data={
                "tool_name": tool_name,
                "success": False,
                "duration_ms": duration_ms,
                "error": str(error),
                "error_type": type(error).__name__,
                **metadata,
            },
        )

    def tool_failure(
        self,
        tool_name: str,
        duration_ms: float,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit tool execution failure event (sync wrapper).

        Args:
            tool_name: Name of the tool that failed
            duration_ms: Execution duration before failure
            error: The exception that occurred
            **metadata: Additional metadata
        """
        self.emit(
            topic="tool.error",
            data={
                "tool_name": tool_name,
                "success": False,
                "duration_ms": duration_ms,
                "error": str(error),
                "error_type": type(error).__name__,
                **metadata,
            },
        )

    @contextlib.contextmanager
    def track_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ):
        """Context manager for tracking tool execution.

        Automatically emits start/end events and handles exceptions.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            **metadata: Additional metadata

        Yields:
            None

        Example:
            >>> with emitter.track_tool("read_file", {"path": "file.txt"}):
            ...     result = await tool(**arguments)
        """
        start_time = time.time()
        self.tool_start(tool_name, arguments, **metadata)

        try:
            yield
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.tool_failure(tool_name, duration_ms, e, **metadata)
            raise

        duration_ms = (time.time() - start_time) * 1000
        # Note: Result will be emitted by caller if needed
        # This context manager tracks timing and errors only
        self.tool_end(tool_name, duration_ms, **metadata)

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if event emission is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled
