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

"""Inter-agent communication infrastructure for teams.

This module provides the communication backbone for agent teams:
- AgentMessage: Structured messages between agents
- TeamMessageBus: Central message routing and delivery
- TeamSharedMemory: Thread-safe shared context store

Design Principles:
- Async-first message passing
- Type-safe message structure
- Thread-safe shared memory
- Full message history for debugging and learning

Example:
    # Create message bus for a team
    bus = TeamMessageBus(team_id="auth_team")
    bus.register_agent("researcher")
    bus.register_agent("executor")

    # Send a discovery message
    await bus.send(AgentMessage(
        type=MessageType.DISCOVERY,
        from_agent="researcher",
        content="Found 5 authentication endpoints",
        data={"endpoints": ["/login", "/logout", "/refresh", "/register", "/verify"]}
    ))

    # Executor receives the message
    msg = await bus.receive("executor")
    print(msg.content)  # "Found 5 authentication endpoints"
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

# Import canonical types from victor.teams.types
from victor.teams.types import MessageType, AgentMessage as CanonicalAgentMessage

logger = logging.getLogger(__name__)


# MessageType is imported from victor.teams.types (canonical location)
# See victor.teams.types.MessageType for documentation


@dataclass
class AgentMessage:
    """A message between agents in a team.

    NOTE: This is a legacy wrapper for backward compatibility.
    For new code, prefer using victor.teams.types.AgentMessage directly.

    Messages are the primary communication mechanism between team members.
    They support both directed (to specific agent) and broadcast (to all)
    communication patterns.

    Attributes:
        id: Unique message identifier
        type: Category of message (discovery, request, etc.)
        from_agent: ID of the sending agent
        to_agent: ID of recipient (None = broadcast to all)
        content: Human-readable message content
        data: Structured data payload
        timestamp: When the message was created
        reply_to: ID of message being replied to (for threading)
        priority: Message priority (higher = more important)

    Example:
        msg = AgentMessage(
            type=MessageType.DISCOVERY,
            from_agent="researcher",
            content="Found database connection pool implementation",
            data={"file": "src/db/pool.py", "lines": [45, 120]}
        )
    """

    type: MessageType
    from_agent: str
    content: str = ""
    to_agent: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    priority: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "priority": self.priority,
        }

    def to_context_string(self, presentation: Optional["PresentationProtocol"] = None) -> str:
        """Format message for inclusion in agent context.

        Args:
            presentation: Optional presentation adapter for icons.
                If None, creates default adapter.

        Returns:
            Formatted string for agent context.
        """
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            presentation = create_presentation_adapter()

        arrow = presentation.icon("arrow_right", with_color=False)
        header = f"[{self.type.value.upper()}] {self.from_agent}"
        if self.to_agent:
            header += f" {arrow} {self.to_agent}"
        return f"{header}: {self.content}"


class TeamMessageBus:
    """Central message bus for agent communication within a team.

    Routes messages between agents, maintains message history, and supports
    both synchronous and asynchronous message delivery.

    Attributes:
        team_id: Identifier for the team using this bus
        message_log: Complete history of all messages

    Thread Safety:
        The message bus is thread-safe for concurrent access.

    Example:
        bus = TeamMessageBus("refactoring_team")
        bus.register_agent("researcher")
        bus.register_agent("executor")

        # Send message
        await bus.send(AgentMessage(
            type=MessageType.DISCOVERY,
            from_agent="researcher",
            content="Found legacy code in auth module",
        ))

        # Receive messages
        msg = await bus.receive("executor", timeout=5.0)
    """

    def __init__(self, team_id: str):
        """Initialize message bus for a team.

        Args:
            team_id: Unique identifier for the team
        """
        self.team_id = team_id
        self._queues: Dict[str, asyncio.Queue[AgentMessage]] = {}
        self._message_log: List[AgentMessage] = []
        self._subscribers: Dict[MessageType, List[Callable[[AgentMessage], None]]] = {}
        self._lock = RLock()
        self._registered_agents: Set[str] = set()
        self._lsp: Optional[Any] = None  # Language Server Protocol capability

        logger.debug(f"Created message bus for team: {team_id}")

    def register_agent(self, agent_id: str) -> None:
        """Register an agent to receive messages.

        Must be called before an agent can send or receive messages.

        Args:
            agent_id: Unique identifier for the agent
        """
        with self._lock:
            if agent_id not in self._registered_agents:
                self._queues[agent_id] = asyncio.Queue()
                self._registered_agents.add(agent_id)
                logger.debug(f"Registered agent '{agent_id}' on bus '{self.team_id}'")

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the message bus.

        Args:
            agent_id: Agent to unregister
        """
        with self._lock:
            self._registered_agents.discard(agent_id)
            self._queues.pop(agent_id, None)
            logger.debug(f"Unregistered agent '{agent_id}' from bus '{self.team_id}'")

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence in team communication.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for team communication.

        Enables language intelligence features for code-related messages
        exchanged between team members.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability
        logger.debug(f"LSP capability registered for team '{self.team_id}'")

    async def send(self, message: AgentMessage) -> None:
        """Send a message to specific agent or broadcast to all.

        If to_agent is set, delivers only to that agent.
        If to_agent is None, broadcasts to all agents except sender.

        Args:
            message: Message to send

        Raises:
            ValueError: If sender is not registered
        """
        with self._lock:
            if message.from_agent not in self._registered_agents:
                raise ValueError(
                    f"Sender '{message.from_agent}' not registered on bus '{self.team_id}'"
                )

            # Log message
            self._message_log.append(message)

        # Route message
        if message.to_agent:
            # Directed message
            if message.to_agent in self._queues:
                await self._queues[message.to_agent].put(message)
                logger.debug(
                    f"Delivered {message.type.value} from '{message.from_agent}' "
                    f"to '{message.to_agent}'"
                )
            else:
                logger.warning(
                    f"Target agent '{message.to_agent}' not found on bus '{self.team_id}'"
                )
        else:
            # Broadcast to all except sender
            for agent_id, queue in self._queues.items():
                if agent_id != message.from_agent:
                    await queue.put(message)

            logger.debug(
                f"Broadcast {message.type.value} from '{message.from_agent}' "
                f"to {len(self._queues) - 1} agents"
            )

        # Notify subscribers
        if message.type in self._subscribers:
            for callback in self._subscribers[message.type]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {e}")

    async def receive(
        self,
        agent_id: str,
        timeout: float = 0,
        message_types: Optional[List[MessageType]] = None,
    ) -> Optional[AgentMessage]:
        """Receive next message for an agent.

        Args:
            agent_id: Agent receiving the message
            timeout: Max wait time in seconds (0 = no wait, None = wait forever)
            message_types: Filter for specific message types

        Returns:
            Next message for agent, or None if timeout/no message
        """
        if agent_id not in self._queues:
            logger.warning(f"Agent '{agent_id}' not registered on bus '{self.team_id}'")
            return None

        try:
            if timeout == 0:
                # Non-blocking check
                try:
                    msg = self._queues[agent_id].get_nowait()
                except asyncio.QueueEmpty:
                    return None
            elif timeout is None:
                # Wait forever
                msg = await self._queues[agent_id].get()
            else:
                # Wait with timeout
                msg = await asyncio.wait_for(
                    self._queues[agent_id].get(),
                    timeout=timeout,
                )

            # Filter by type if specified
            if message_types and msg.type not in message_types:
                # Put back and return None (or could keep searching)
                await self._queues[agent_id].put(msg)
                return None

            return msg

        except asyncio.TimeoutError:
            return None

    def get_pending_count(self, agent_id: str) -> int:
        """Get number of pending messages for an agent.

        Args:
            agent_id: Agent to check

        Returns:
            Number of pending messages
        """
        if agent_id not in self._queues:
            return 0
        return self._queues[agent_id].qsize()

    def get_message_log(
        self,
        message_type: Optional[MessageType] = None,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """Get filtered message history.

        Args:
            message_type: Filter by message type
            from_agent: Filter by sender
            to_agent: Filter by recipient
            limit: Maximum messages to return

        Returns:
            List of matching messages (newest first)
        """
        with self._lock:
            messages = self._message_log.copy()

        # Apply filters
        if message_type:
            messages = [m for m in messages if m.type == message_type]
        if from_agent:
            messages = [m for m in messages if m.from_agent == from_agent]
        if to_agent:
            messages = [m for m in messages if m.to_agent == to_agent]

        # Return most recent, limited
        return list(reversed(messages[-limit:]))

    def get_discoveries(self, agent_id: Optional[str] = None) -> List[AgentMessage]:
        """Get all discovery messages, optionally filtered by agent.

        Args:
            agent_id: Filter by sender agent

        Returns:
            List of discovery messages
        """
        return self.get_message_log(
            message_type=MessageType.DISCOVERY,
            from_agent=agent_id,
        )

    def subscribe(
        self,
        message_type: MessageType,
        callback: Callable[[AgentMessage], None],
    ) -> None:
        """Subscribe to messages of a specific type.

        Callback is invoked synchronously when matching messages are sent.

        Args:
            message_type: Type of messages to subscribe to
            callback: Function to call with each matching message
        """
        with self._lock:
            if message_type not in self._subscribers:
                self._subscribers[message_type] = []
            self._subscribers[message_type].append(callback)

    def get_context_summary(self, max_messages: int = 20) -> str:
        """Get summary of recent messages for agent context injection.

        Args:
            max_messages: Maximum messages to include

        Returns:
            Formatted string for inclusion in agent context
        """
        messages = self.get_message_log(limit=max_messages)
        if not messages:
            return "## Team Communication\n\nNo messages exchanged yet."

        lines = ["## Team Communication\n"]
        for msg in messages:
            lines.append(f"- {msg.to_context_string()}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all message history and queues."""
        with self._lock:
            self._message_log.clear()
            for queue in self._queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break


class TeamSharedMemory:
    """Thread-safe shared memory for team context.

    Provides a key-value store that all team members can read and write.
    Tracks which agents contributed each value for attribution.

    Thread Safety:
        All operations are thread-safe using RLock.

    Example:
        memory = TeamSharedMemory()

        # Researcher stores findings
        memory.set("api_endpoints", ["/login", "/logout"], "researcher")
        memory.append("files_analyzed", "auth.py", "researcher")

        # Executor reads findings
        endpoints = memory.get("api_endpoints")
        files = memory.get("files_analyzed")
    """

    def __init__(self):
        """Initialize empty shared memory."""
        self._data: Dict[str, Any] = {}
        self._contributors: Dict[str, Set[str]] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = RLock()

    def set(self, key: str, value: Any, agent_id: str) -> None:
        """Set a shared value.

        Overwrites any existing value for the key.

        Args:
            key: Key to store value under
            value: Value to store
            agent_id: ID of agent storing the value
        """
        with self._lock:
            self._data[key] = value
            if key not in self._contributors:
                self._contributors[key] = set()
            self._contributors[key].add(agent_id)
            self._timestamps[key] = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a shared value.

        Args:
            key: Key to retrieve
            default: Value to return if key not found

        Returns:
            Stored value or default
        """
        return self._data.get(key, default)

    def append(self, key: str, value: Any, agent_id: str) -> None:
        """Append to a list value.

        Creates list if key doesn't exist.

        Args:
            key: Key for the list
            value: Value to append
            agent_id: ID of agent appending
        """
        with self._lock:
            if key not in self._data:
                self._data[key] = []
            if not isinstance(self._data[key], list):
                raise TypeError(f"Cannot append to non-list value at key '{key}'")
            self._data[key].append(value)
            if key not in self._contributors:
                self._contributors[key] = set()
            self._contributors[key].add(agent_id)
            self._timestamps[key] = time.time()

    def update(self, key: str, updates: Dict[str, Any], agent_id: str) -> None:
        """Update a dictionary value.

        Creates dict if key doesn't exist.

        Args:
            key: Key for the dictionary
            updates: Key-value pairs to update
            agent_id: ID of agent updating
        """
        with self._lock:
            if key not in self._data:
                self._data[key] = {}
            if not isinstance(self._data[key], dict):
                raise TypeError(f"Cannot update non-dict value at key '{key}'")
            self._data[key].update(updates)
            if key not in self._contributors:
                self._contributors[key] = set()
            self._contributors[key].add(agent_id)
            self._timestamps[key] = time.time()

    def has(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        return key in self._data

    def keys(self) -> List[str]:
        """Get all keys in shared memory.

        Returns:
            List of all keys
        """
        return list(self._data.keys())

    def get_contributors(self, key: str) -> Set[str]:
        """Get agents that contributed to a key.

        Args:
            key: Key to check

        Returns:
            Set of agent IDs
        """
        return self._contributors.get(key, set()).copy()

    def get_all(self) -> Dict[str, Any]:
        """Get snapshot of all shared data.

        Returns:
            Copy of all data
        """
        with self._lock:
            return dict(self._data)

    def get_summary(self, max_value_length: int = 500) -> str:
        """Get human-readable summary for agent context.

        Args:
            max_value_length: Maximum characters per value

        Returns:
            Formatted summary string
        """
        with self._lock:
            if not self._data:
                return "## Shared Team Knowledge\n\nNo shared data yet."

            lines = ["## Shared Team Knowledge\n"]
            for key, value in self._data.items():
                contributors = ", ".join(self._contributors.get(key, set()))
                value_str = str(value)
                if len(value_str) > max_value_length:
                    value_str = value_str[:max_value_length] + "..."
                lines.append(f"### {key}")
                lines.append(f"*Contributors: {contributors}*")
                lines.append(f"```\n{value_str}\n```")
                lines.append("")
            return "\n".join(lines)

    def clear(self) -> None:
        """Clear all shared memory."""
        with self._lock:
            self._data.clear()
            self._contributors.clear()
            self._timestamps.clear()


__all__ = [
    "MessageType",
    "AgentMessage",
    "TeamMessageBus",
    "TeamSharedMemory",
]
