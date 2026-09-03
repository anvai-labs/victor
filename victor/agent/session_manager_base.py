"""Unified session management for CLI and one-shot modes.

This module provides a unified interface for session management,
eliminating divergence between interactive and one-shot session handling.

SOLID Principles Applied:
- Single Responsibility: Each class handles one aspect of session management
- Interface Segregation: Separate interfaces for different session operations
- Dependency Inversion: High-level code depends on abstractions, not concrete implementations
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class SessionMode(Enum):
    """Mode of the session."""

    INTERACTIVE = "interactive"
    ONESHOT = "oneshot"


@dataclass
class SessionConfig:
    """Configuration for a session."""

    mode: SessionMode
    provider: str
    model: str
    profile: str
    thinking: bool = False
    vertical: Optional[str] = None
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None
    mode_name: Optional[str] = None  # build, plan, explore

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            "thinking": self.thinking,
            "vertical": self.vertical,
            "tool_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "mode_name": self.mode_name,
        }


@dataclass
class SessionMetrics:
    """Metrics for a session."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tool_calls: int = 0
    tokens_used: int = 0
    iterations: int = 0
    success: bool = True

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "iterations": self.iterations,
            "success": self.success,
        }


class ISessionHandler(ABC):
    """Interface for session handling.

    Defines the contract for session lifecycle management.
    Concrete implementations can provide different behaviors
    for CLI and one-shot modes.
    """

    @abstractmethod
    async def initialize(self, config: SessionConfig) -> AgentOrchestrator:
        """Initialize a new session.

        Args:
            config: Session configuration

        Returns:
            Initialized agent orchestrator
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        agent: AgentOrchestrator,
        message: str,
        stream: bool = True,
    ) -> str:
        """Process a user message.

        Args:
            agent: Agent orchestrator
            message: User message
            stream: Whether to stream response

        Returns:
            Assistant response content
        """
        pass

    @abstractmethod
    async def cleanup(self, agent: AgentOrchestrator, metrics: SessionMetrics) -> None:
        """Clean up resources after session.

        Args:
            agent: Agent orchestrator
            metrics: Session metrics to finalize
        """
        pass


class BaseSessionHandler(ISessionHandler):
    """Base implementation of session handler.

    Provides common functionality for all session types.
    """

    def __init__(self, strategy_factory: Optional[Any] = None):
        """Initialize handler.

        Args:
            strategy_factory: Optional factory for agent creation strategy
        """
        self._strategy_factory = strategy_factory

    async def initialize(self, config: SessionConfig) -> AgentOrchestrator:
        """Initialize session with agent creation."""
        from victor.agent.creation_strategies import (
            AgentCreationContext,
            AgentCreationFactory,
        )
        from victor.config.settings import load_settings
        from victor.core.verticals import get_vertical

        # fresh=True: AgentFactory monkey-patches load_profiles on the instance
        # it receives (agent_factory._apply_profile_overrides) — must never be
        # the shared process snapshot (adversarial-review finding).
        settings = load_settings(fresh=True)
        vertical_class = get_vertical(config.vertical) if config.vertical else None

        context = AgentCreationContext(
            settings=settings,
            profile=config.profile,
            provider=config.provider,
            model=config.model,
            thinking=config.thinking,
            vertical=vertical_class,
            tool_budget=config.tool_budget,
            max_iterations=config.max_iterations,
            mode=config.mode_name,
            metadata={"session_mode": config.mode.value},
        )

        factory = AgentCreationFactory()
        return await factory.create_agent(context)

    async def process_message(
        self,
        agent: AgentOrchestrator,
        message: str,
        stream: bool = True,
    ) -> str:
        """Process message with optional streaming."""
        if stream and agent.provider.supports_streaming():
            content_buffer = ""
            async for chunk in agent.stream_chat(message):
                if hasattr(chunk, "content") and chunk.content:
                    content_buffer += chunk.content
            return content_buffer
        else:
            response = await agent.chat(message)
            return response.content

    async def cleanup(self, agent: AgentOrchestrator, metrics: SessionMetrics) -> None:
        """Clean up session resources."""
        metrics.end_time = time.time()
        if hasattr(agent, "get_session_metrics"):
            agent_metrics = agent.get_session_metrics()
            if agent_metrics:
                metrics.tool_calls = agent_metrics.get("tool_calls", metrics.tool_calls)
                metrics.tokens_used = agent_metrics.get("tokens", metrics.tokens_used)
                metrics.iterations = agent_metrics.get("iterations", metrics.iterations)

        # Graceful shutdown
        from victor.ui.commands.utils import graceful_shutdown

        await graceful_shutdown(agent)


class OneshotSessionHandler(BaseSessionHandler):
    """Handler for one-shot sessions.

    One-shot sessions process a single message and exit.
    No conversation state is maintained.
    """

    async def execute(self, config: SessionConfig, message: str) -> SessionMetrics:
        """Execute a one-shot session.

        Args:
            config: Session configuration
            message: Message to process

        Returns:
            Session metrics
        """
        metrics = SessionMetrics()

        try:
            agent = await self.initialize(config)
            response = await self.process_message(agent, message, stream=True)
            metrics.success = True
            return metrics
        except Exception as e:
            logger.error(f"One-shot session failed: {e}")
            metrics.success = False
            raise
        finally:
            if "agent" in locals():
                await self.cleanup(agent, metrics)


class InteractiveSessionHandler(BaseSessionHandler):
    """Handler for interactive CLI sessions.

    Interactive sessions maintain conversation state across
    multiple messages in a REPL-style interface.
    """

    def __init__(self, on_message_callback: Optional[Callable] = None):
        """Initialize interactive handler.

        Args:
            on_message_callback: Optional callback for each message
        """
        super().__init__()
        self._on_message = on_message_callback
        self._agent: Optional[AgentOrchestrator] = None

    async def start_repl(self, config: SessionConfig) -> None:
        """Start the REPL loop.

        Args:
            config: Session configuration
        """
        self._agent = await self.initialize(config)
        metrics = SessionMetrics()

        try:
            while True:
                try:
                    # This is a template - actual REPL implementation
                    # would use rich.prompt.Prompt or similar
                    user_input = await self._get_user_input()
                    if not user_input:
                        continue

                    if user_input.lower() in ("exit", "quit"):
                        break

                    response = await self.process_message(self._agent, user_input)

                    if self._on_message:
                        await self._on_message(response)

                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break

        finally:
            await self.cleanup(self._agent, metrics)

    async def _get_user_input(self) -> str:
        """Get user input (to be implemented by concrete class)."""
        raise NotImplementedError("Subclasses must implement _get_user_input")


__all__ = [
    "SessionMode",
    "SessionConfig",
    "SessionMetrics",
    "ISessionHandler",
    "BaseSessionHandler",
    "OneshotSessionHandler",
    "InteractiveSessionHandler",
]
