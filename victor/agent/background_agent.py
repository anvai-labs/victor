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

"""Background Agent Manager for async task execution.

This module provides infrastructure for running AI agents in the background,
similar to Cursor's parallel agent execution. Agents run asynchronously and
report progress via WebSocket events.

Features:
- Async agent execution (non-blocking)
- Progress tracking and status updates
- WebSocket event broadcasting
- Cancellation support
- Concurrent agent limit

Usage:
    from victor.agent.services.protocols import ChatServiceProtocol

    manager = BackgroundAgentManager(orchestrator, chat_service, max_concurrent=4)

    # Start a background agent
    agent_id = await manager.start_agent(
        task="Refactor the auth module",
        mode="build",
    )

    # Check status
    status = manager.get_agent_status(agent_id)

    # Cancel if needed
    await manager.cancel_agent(agent_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.async_utils import run_sync
from victor.agent.services.protocols import ChatServiceProtocol

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ToolCallRecord:
    """Record of a tool call made by an agent."""

    id: str
    name: str
    status: str  # pending, running, success, error
    start_time: float
    end_time: Optional[float] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BackgroundAgent:
    """Represents a background agent task.

    Phase 4 Refactoring:
    Now includes optional orchestrator reference for unified agent creation.
    """

    id: str
    name: str
    description: str
    task: str
    mode: str  # build, plan, explore
    orchestrator: Optional[Any] = None  # Phase 4: Optional orchestrator reference
    status: AgentStatus = AgentStatus.PENDING
    progress: int = 0  # 0-100
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    output: Optional[str] = None
    error: Optional[str] = None

    # Internal state
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task": self.task,
            "mode": self.mode,
            "status": self.status.value,
            "progress": self.progress,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "status": tc.status,
                    "start_time": tc.start_time,
                    "end_time": tc.end_time,
                    "result": tc.result[:200] if tc.result else None,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "output": self.output[:500] if self.output else None,
            "error": self.error,
        }


# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]


class BackgroundAgentManager:
    """Manages background agent execution.

    Phase 4 Refactoring:
    Now supports OrchestratorFactory for unified agent creation, ensuring
    consistent code maintenance and eliminating code proliferation (SOLID DIP).

    Provides async agent execution with progress tracking and WebSocket
    event broadcasting for real-time UI updates.

    Note: This manager shares a single orchestrator across all background
    agents for efficiency. Agent instances are state trackers, not separate
    agent implementations.
    """

    def __init__(
        self,
        orchestrator: Any,
        chat_service: ChatServiceProtocol,
        max_concurrent: int = 4,
        event_callback: Optional[EventCallback] = None,
        factory: Optional[Any] = None,
    ):
        """Initialize the background agent manager.

        Args:
            orchestrator: The AgentOrchestrator instance (shared by all agents)
            chat_service: The ChatService for streaming responses
            max_concurrent: Maximum concurrent agents (default: 4)
            event_callback: Callback for broadcasting events
            factory: Optional OrchestratorFactory for unified agent creation
                    (recommended for consistency with Phase 4 architecture)
        """
        self._orchestrator = orchestrator
        self._chat_service = chat_service
        self._factory = factory  # Store factory for unified creation
        self._max_concurrent = max_concurrent
        self._event_callback = event_callback

        self._agents: Dict[str, BackgroundAgent] = {}
        self._running_tasks: Set[str] = set()
        self._lock = asyncio.Lock()

        logger.info(f"BackgroundAgentManager initialized (max_concurrent={max_concurrent})")

    @property
    def active_count(self) -> int:
        """Get count of active (running/pending) agents."""
        return len(self._running_tasks)

    @classmethod
    def from_factory(
        cls,
        factory: Any,
        chat_service: ChatServiceProtocol,
        max_concurrent: int = 4,
        event_callback: Optional[EventCallback] = None,
    ) -> "BackgroundAgentManager":
        """Create BackgroundAgentManager from OrchestratorFactory.

        Phase 4: Recommended factory method for unified agent creation.

        Args:
            factory: OrchestratorFactory instance
            chat_service: The ChatService for streaming responses
            max_concurrent: Maximum concurrent agents
            event_callback: Callback for broadcasting events

        Returns:
            BackgroundAgentManager instance with factory-created orchestrator

        Example:
            from victor.agent.orchestrator_factory import OrchestratorFactory
            from victor.config.settings import load_settings

            settings = load_settings()
            factory = OrchestratorFactory(settings, provider, model)
            manager = BackgroundAgentManager.from_factory(factory, chat_service)
        """
        # Create orchestrator using factory (this is the unified creation path)
        orchestrator = run_sync(factory.create_agent(mode="foreground"))

        # Return manager with factory stored for consistency
        return cls(
            orchestrator=orchestrator,
            chat_service=chat_service,
            max_concurrent=max_concurrent,
            event_callback=event_callback,
            factory=factory,  # Store factory for future use
        )

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for broadcasting updates."""
        self._event_callback = callback

    async def start_agent(
        self,
        task: str,
        mode: str = "build",
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> str:
        """Start a new background agent.

        Phase 4 Refactoring:
        Accepts optional UnifiedAgentConfig and uses OrchestratorFactory if available
        for unified agent creation. Falls back to direct instantiation for
        backward compatibility.

        Args:
            task: The task/prompt for the agent
            mode: Agent mode (build, plan, explore)
            name: Optional display name
            description: Optional description
            config: Optional UnifiedAgentConfig for advanced configuration

        Returns:
            Agent ID

        Raises:
            RuntimeError: If max concurrent agents reached
        """
        async with self._lock:
            if len(self._running_tasks) >= self._max_concurrent:
                raise RuntimeError(
                    f"Maximum concurrent agents ({self._max_concurrent}) reached. "
                    f"Wait for an agent to complete or cancel one."
                )

            agent_id = f"agent-{uuid.uuid4().hex[:8]}"

            # Generate name from task if not provided
            if not name:
                name = task[:40] + ("..." if len(task) > 40 else "")

            # Phase 4: Use factory if available for unified agent creation
            if self._factory and config:
                # Import here to avoid circular dependency
                from victor.agent.config import UnifiedAgentConfig

                # Convert to UnifiedAgentConfig if needed
                if not isinstance(config, UnifiedAgentConfig):
                    config = UnifiedAgentConfig(
                        mode="background",
                        task=task,
                        mode_type=mode,
                    )

                # Use factory to create agent wrapper (gets orchestrator)
                agent_wrapper = await self._factory.create_agent(
                    mode="background",
                    task=task,
                    mode_type=mode,
                    config=config,
                )

                # Create BackgroundAgent state tracker
                agent = BackgroundAgent(
                    id=agent_id,
                    name=name,
                    description=description or task,
                    task=task,
                    mode=mode,
                    orchestrator=(
                        agent_wrapper._orchestrator
                        if hasattr(agent_wrapper, "_orchestrator")
                        else self._orchestrator
                    ),
                )
            else:
                # Legacy path: direct BackgroundAgent instantiation
                agent = BackgroundAgent(
                    id=agent_id,
                    name=name,
                    description=description or task,
                    task=task,
                    mode=mode,
                    orchestrator=self._orchestrator,
                )

            self._agents[agent_id] = agent
            self._running_tasks.add(agent_id)

            # Start the agent task
            agent._task = asyncio.create_task(self._run_agent(agent))

            logger.info(f"Started background agent: {agent_id} - {name}")
            self._emit_event("agent_started", agent.to_dict())

            return agent_id

    async def cancel_agent(self, agent_id: str) -> bool:
        """Cancel a running agent.

        Args:
            agent_id: ID of the agent to cancel

        Returns:
            True if cancelled, False if not found or not running
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        if agent.status not in (AgentStatus.RUNNING, AgentStatus.PENDING):
            return False

        agent._cancelled = True

        if agent._task and not agent._task.done():
            agent._task.cancel()

        agent.status = AgentStatus.CANCELLED
        agent.end_time = time.time()

        async with self._lock:
            self._running_tasks.discard(agent_id)

        logger.info(f"Cancelled agent: {agent_id}")
        self._emit_event("agent_cancelled", agent.to_dict())

        return True

    def get_agent(self, agent_id: str) -> Optional[BackgroundAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status as dictionary."""
        agent = self._agents.get(agent_id)
        return agent.to_dict() if agent else None

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List agents, optionally filtered by status.

        Args:
            status: Filter by status (optional)
            limit: Maximum agents to return

        Returns:
            List of agent dictionaries
        """
        agents = list(self._agents.values())

        if status:
            agents = [a for a in agents if a.status == status]

        # Sort by start time (newest first)
        agents.sort(key=lambda a: a.start_time, reverse=True)

        return [a.to_dict() for a in agents[:limit]]

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled agents.

        Returns:
            Number of agents cleared
        """
        to_remove = [
            agent_id
            for agent_id, agent in self._agents.items()
            if agent.status
            in (
                AgentStatus.COMPLETED,
                AgentStatus.ERROR,
                AgentStatus.CANCELLED,
            )
        ]

        for agent_id in to_remove:
            del self._agents[agent_id]

        logger.info(f"Cleared {len(to_remove)} completed agents")
        return len(to_remove)

    async def _run_agent(self, agent: BackgroundAgent) -> None:
        """Execute an agent task.

        This runs in the background and updates agent status/progress.
        """
        try:
            agent.status = AgentStatus.RUNNING
            self._emit_event("agent_running", agent.to_dict())

            # Set mode on orchestrator
            if hasattr(self._orchestrator, "set_mode"):
                self._orchestrator.set_mode(agent.mode)

            # Execute the task with tool call tracking
            response_chunks = []
            tool_call_count = 0

            async for chunk in self._chat_service.stream_chat(agent.task):
                if agent._cancelled:
                    break

                chunk_type = chunk.get("type", "")

                if chunk_type == "content":
                    response_chunks.append(chunk.get("content", ""))

                elif chunk_type == "tool_call":
                    tool_call_count += 1
                    tc_data = chunk.get("tool_call", {})

                    tool_call = ToolCallRecord(
                        id=tc_data.get("id", f"tc-{tool_call_count}"),
                        name=tc_data.get("name", "unknown"),
                        status="running",
                        start_time=time.time(),
                        arguments=tc_data.get("arguments"),
                    )
                    agent.tool_calls.append(tool_call)

                    # Update progress (rough estimate)
                    agent.progress = min(90, agent.progress + 10)

                    self._emit_event(
                        "agent_tool_call",
                        {
                            "agent_id": agent.id,
                            "tool_call": {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "status": tool_call.status,
                            },
                        },
                    )

                elif chunk_type == "tool_result":
                    # Update the last tool call with result
                    if agent.tool_calls:
                        tc = agent.tool_calls[-1]
                        tc.status = "success"
                        tc.end_time = time.time()
                        tc.result = str(chunk.get("result", ""))[:500]

                        self._emit_event(
                            "agent_tool_result",
                            {
                                "agent_id": agent.id,
                                "tool_call_id": tc.id,
                                "status": tc.status,
                            },
                        )

            # Agent completed successfully
            agent.output = "".join(response_chunks)
            agent.status = AgentStatus.COMPLETED
            agent.progress = 100
            agent.end_time = time.time()

            logger.info(f"Agent completed: {agent.id}")
            self._emit_event("agent_completed", agent.to_dict())

        except asyncio.CancelledError:
            agent.status = AgentStatus.CANCELLED
            agent.end_time = time.time()
            logger.info(f"Agent cancelled: {agent.id}")

        except Exception as e:
            agent.status = AgentStatus.ERROR
            agent.error = str(e)
            agent.end_time = time.time()
            logger.exception(f"Agent error: {agent.id}")
            self._emit_event(
                "agent_error",
                {
                    "agent_id": agent.id,
                    "error": str(e),
                },
            )

        finally:
            async with self._lock:
                self._running_tasks.discard(agent.id)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event via the callback."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")


# Global instance (lazy initialization)
_agent_manager: Optional[BackgroundAgentManager] = None


def get_agent_manager() -> Optional[BackgroundAgentManager]:
    """Get the global agent manager instance."""
    return _agent_manager


def init_agent_manager(
    orchestrator: Any,
    chat_service: ChatServiceProtocol,
    max_concurrent: int = 4,
    event_callback: Optional[EventCallback] = None,
) -> BackgroundAgentManager:
    """Initialize the global agent manager.

    Args:
        orchestrator: The AgentOrchestrator instance
        chat_service: The ChatService for streaming responses
        max_concurrent: Maximum concurrent agents
        event_callback: Callback for broadcasting events

    Returns:
        The initialized BackgroundAgentManager
    """
    global _agent_manager
    _agent_manager = BackgroundAgentManager(
        orchestrator=orchestrator,
        chat_service=chat_service,
        max_concurrent=max_concurrent,
        event_callback=event_callback,
    )
    return _agent_manager


__all__ = [
    "AgentStatus",
    "ToolCallRecord",
    "BackgroundAgent",
    "BackgroundAgentManager",
    "get_agent_manager",
    "init_agent_manager",
]
