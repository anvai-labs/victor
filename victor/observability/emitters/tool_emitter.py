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
import time
from typing import Any, Dict, Optional

from victor.observability.emitters.base import BaseEventEmitter, IToolEventEmitter


class ToolEventEmitter(BaseEventEmitter, IToolEventEmitter):
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

    _CATEGORY = "tool"

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
