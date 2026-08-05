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

"""Debug protocol message types for workflow debugging.

This module provides protocol message types for communication between
debug clients and the debugging system. Messages are JSON-serializable
and can be sent over WebSocket, REST, or other transport.

Key Classes:
    DebugMessageType: Types of debug protocol messages
    DebugMessage: Debug protocol message

Example:
    from victor.framework.debugging.protocol import (
        DebugMessage,
        DebugMessageType,
    )

    # Client command
    msg = DebugMessage(
        type=DebugMessageType.SET_BREAKPOINTS,
        session_id="debug-123",
        data={"node_id": "analyze", "position": "before"}
    )

    # Server event
    msg = DebugMessage(
        type=DebugMessageType.BREAKPOINT_HIT,
        session_id="debug-123",
        data={"node_id": "analyze", "state": {...}}
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from victor.framework.debugging.breakpoints import BreakpointPosition


class DebugMessageType(Enum):
    """Types of debug protocol messages.

    Client -> Server (Commands):
        SET_BREAKPOINTS: Set a new breakpoint
        CLEAR_BREAKPOINTS: Clear a breakpoint
        CONTINUE: Continue execution
        STEP_OVER: Step over next node
        STEP_INTO: Step into next node
        STEP_OUT: Step out of current sub-workflow
        PAUSE: Request immediate pause
        INSPECT_STATE: Get current workflow state
        GET_DIFF: Get state diff

    Server -> Client (Events):
        BREAKPOINT_HIT: Breakpoint was hit
        STATE_UPDATE: State update notification
        EXCEPTION: Exception occurred
        PAUSED: Execution paused
        RESUMED: Execution resumed
        COMPLETED: Execution completed
    """

    # Client -> Server (Commands)
    SET_BREAKPOINTS = "set_breakpoints"
    CLEAR_BREAKPOINTS = "clear_breakpoints"
    CONTINUE = "continue"
    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"
    PAUSE = "pause"
    INSPECT_STATE = "inspect_state"
    GET_DIFF = "get_diff"

    # Server -> Client (Events)
    BREAKPOINT_HIT = "breakpoint_hit"
    STATE_UPDATE = "state_update"
    EXCEPTION = "exception"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"


@dataclass
class DebugMessage:
    """Debug protocol message.

    Messages are JSON-serializable and can be sent over
    WebSocket, REST, or other transport.

    Attributes:
        type: Message type
        session_id: Debug session ID
        data: Message payload (varies by type)
        timestamp: Message timestamp
        request_id: Optional correlation ID for request-response

    Example:
        # Client command
        msg = DebugMessage(
            type=DebugMessageType.SET_BREAKPOINTS,
            session_id="debug-123",
            data={"node_id": "analyze", "position": "before"}
        )

        # Server event
        msg = DebugMessage(
            type=DebugMessageType.BREAKPOINT_HIT,
            session_id="debug-123",
            data={"node_id": "analyze", "state": {...}}
        )
    """

    type: DebugMessageType
    session_id: str
    data: Dict[str, Any]
    timestamp: float
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of message
        """
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugMessage":
        """Create from dictionary.

        Args:
            data: Dictionary representation of message

        Returns:
            DebugMessage instance
        """
        return cls(
            type=DebugMessageType(data["type"]),
            session_id=data["session_id"],
            data=data["data"],
            timestamp=data["timestamp"],
            request_id=data.get("request_id"),
        )


class DebugProtocolHandler:
    """Handles debug protocol messages (facade pattern).

    Routes messages to appropriate components and handles
    request/response correlation.

    Attributes:
        session: DebugSession instance
        _event_bus: EventBus for emitting events

    Example:
        handler = DebugProtocolHandler(session, event_bus)

        # Handle client message
        await handler.handle_message(message)

        # Emit event to client
        handler.emit_breakpoint_hit(node_id, state)
    """

    def __init__(self, session: Any, event_bus: Any) -> None:
        """Initialize protocol handler.

        Args:
            session: DebugSession instance
            event_bus: EventBus for events
        """
        self.session = session
        self._event_bus = event_bus

    async def handle_message(self, message: DebugMessage) -> Optional[DebugMessage]:
        """Handle incoming debug protocol message.

        Args:
            message: DebugMessage from client

        Returns:
            Optional response message

        Raises:
            ValueError: If message type is unknown
        """
        handlers = {
            DebugMessageType.SET_BREAKPOINTS: self._handle_set_breakpoints,
            DebugMessageType.CLEAR_BREAKPOINTS: self._handle_clear_breakpoints,
            DebugMessageType.CONTINUE: self._handle_continue,
            DebugMessageType.STEP_OVER: self._handle_step_over,
            DebugMessageType.STEP_INTO: self._handle_step_into,
            DebugMessageType.STEP_OUT: self._handle_step_out,
            DebugMessageType.PAUSE: self._handle_pause,
            DebugMessageType.INSPECT_STATE: self._handle_inspect_state,
            DebugMessageType.GET_DIFF: self._handle_get_diff,
        }

        handler = handlers.get(message.type)
        if not handler:
            raise ValueError(f"Unknown message type: {message.type}")

        return await handler(message)

    async def _handle_set_breakpoints(self, message: DebugMessage) -> DebugMessage:
        """Handle SET_BREAKPOINTS message.

        Args:
            message: Set breakpoints message

        Returns:
            Response message
        """
        data = message.data
        bp = self.session.set_breakpoint(
            node_id=data.get("node_id"),
            position=BreakpointPosition(data.get("position", "before")),
        )

        return DebugMessage(
            type=DebugMessageType.BREAKPOINT_HIT,
            session_id=message.session_id,
            data={"breakpoint_id": bp.id},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_clear_breakpoints(self, message: DebugMessage) -> DebugMessage:
        """Handle CLEAR_BREAKPOINTS message.

        Args:
            message: Clear breakpoints message

        Returns:
            Response message
        """
        breakpoint_id = message.data.get("breakpoint_id")
        cleared = self.session.clear_breakpoint(breakpoint_id)

        return DebugMessage(
            type=DebugMessageType.BREAKPOINT_HIT,
            session_id=message.session_id,
            data={"cleared": cleared},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_continue(self, message: DebugMessage) -> DebugMessage:
        """Handle CONTINUE message.

        Args:
            message: Continue message

        Returns:
            Response message
        """
        await self.session.continue_execution()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_step_over(self, message: DebugMessage) -> DebugMessage:
        """Handle STEP_OVER message.

        Args:
            message: Step over message

        Returns:
            Response message
        """
        await self.session.step_over()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={"step_mode": "step_over"},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_step_into(self, message: DebugMessage) -> DebugMessage:
        """Handle STEP_INTO message.

        Args:
            message: Step into message

        Returns:
            Response message
        """
        await self.session.step_into()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={"step_mode": "step_into"},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_step_out(self, message: DebugMessage) -> DebugMessage:
        """Handle STEP_OUT message.

        Args:
            message: Step out message

        Returns:
            Response message
        """
        await self.session.step_out()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={"step_mode": "step_out"},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_pause(self, message: DebugMessage) -> DebugMessage:
        """Handle PAUSE message.

        Args:
            message: Pause message

        Returns:
            Response message
        """
        await self.session.pause()

        return DebugMessage(
            type=DebugMessageType.PAUSED,
            session_id=message.session_id,
            data={},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_inspect_state(self, message: DebugMessage) -> DebugMessage:
        """Handle INSPECT_STATE message.

        Args:
            message: Inspect state message

        Returns:
            Response message with state
        """
        state = self.session.get_current_state()

        return DebugMessage(
            type=DebugMessageType.STATE_UPDATE,
            session_id=message.session_id,
            data={"state": state},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_get_diff(self, message: DebugMessage) -> DebugMessage:
        """Handle GET_DIFF message.

        Args:
            message: Get diff message

        Returns:
            Response message with diff
        """
        # This is a placeholder - actual implementation would need
        # to track before/after states
        return DebugMessage(
            type=DebugMessageType.STATE_UPDATE,
            session_id=message.session_id,
            data={"diff": {}},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    def emit_breakpoint_hit(self, node_id: str, state: Dict[str, Any]) -> None:
        """Emit BREAKPOINT_HIT event to client.

        Args:
            node_id: Node ID where breakpoint was hit
            state: Current workflow state
        """
        try:
            import asyncio

            if asyncio.iscoroutinefunction(self._event_bus.emit):
                asyncio.create_task(
                    self._event_bus.emit(
                        topic="debug.breakpoint.hit",
                        data={
                            "session_id": self.session.config.session_id,
                            "node_id": node_id,
                        },
                    )
                )
            else:
                self._event_bus.emit(
                    topic="debug.breakpoint.hit",
                    data={
                        "session_id": self.session.config.session_id,
                        "node_id": node_id,
                    },
                )
        except Exception:
            pass  # Event emission failures shouldn't break debugging
