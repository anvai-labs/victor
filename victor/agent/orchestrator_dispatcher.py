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

"""Orchestrator dispatcher with lazy loading.

This module provides a protocol-based dispatch facade that lazy loads
the AgentOrchestrator implementation only when first accessed.

Design Patterns:
- Facade Pattern: Simplified interface to complex subsystem
- Lazy Loading: Defer expensive imports until first use
- Virtual Proxy: Placeholder that creates real object on first access

Example:
    >>> from victor.agent.orchestrator_dispatcher import get_orchestrator_dispatcher
    >>>
    >>> # Zero imports until first use
    >>> dispatcher = get_orchestrator_dispatcher()
    >>> result = await dispatcher.run("Do something")
    >>> # AgentOrchestrator imported only now
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.agent.orchestrator_protocols import IAgentOrchestrator

logger = logging.getLogger(__name__)

__all__ = [
    "OrchestratorDispatcher",
    "get_orchestrator_dispatcher",
]


class OrchestratorDispatcher:
    """
    Protocol-based dispatch facade with lazy implementation loading.

    This facade provides zero-import access to orchestrator functionality.
    The actual AgentOrchestrator is only loaded on first method call, significantly
    reducing startup import time and dependency complexity.

    Benefits:
        - Zero runtime imports at instantiation
        - Protocol-based type safety
        - Transparent to calling code
        - Compatible with existing AgentOrchestrator usage

    Example:
        >>> dispatcher = OrchestratorDispatcher()
        >>> # No imports yet
        >>>
        >>> result = await dispatcher.run("Do something")
        >>> # AgentOrchestrator imported only now
        >>>
        >>> # Can also access properties
        >>> session_id = dispatcher.session_id
    """

    def __init__(self) -> None:
        """Initialize the dispatcher with lazy loading."""
        self._impl: Optional[IAgentOrchestrator] = None
        self._factory_called: bool = False

    def _get_implementation(self) -> IAgentOrchestrator:
        """
        Lazy load implementation on first access.

        Returns:
            AgentOrchestrator instance

        Raises:
            RuntimeError: If orchestrator creation fails
        """
        if self._impl is not None:
            return self._impl

        try:
            # Import only when needed
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.core.bootstrap import ensure_bootstrapped
            from victor.core.async_utils import run_sync
            from victor.config.settings import load_settings

            # Ensure DI container is bootstrapped
            settings = load_settings()
            ensure_bootstrapped(settings)

            # Create orchestrator
            self._impl = run_sync(
                AgentOrchestrator.from_settings(
                    settings=settings,
                    profile_name="default",
                )
            )
            self._factory_called = True

            logger.debug("OrchestratorDispatcher: Lazy loaded AgentOrchestrator")
            return self._impl

        except Exception as e:
            logger.error(f"OrchestratorDispatcher: Failed to load AgentOrchestrator: {e}")
            raise RuntimeError(f"Failed to initialize orchestrator: {e}") from e

    # Delegate all protocol methods
    async def run(self, task: str, **kwargs: Any) -> Any:
        """Execute a single-turn task."""
        return await self._get_implementation().run(task, **kwargs)

    async def stream(self, task: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream task execution with real-time events."""
        async for event in self._get_implementation().stream(task, **kwargs):
            yield event

    def chat(self, **kwargs: Any) -> Any:
        """Create a multi-turn chat session."""
        return self._get_implementation().chat(**kwargs)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a single tool."""
        return await self._get_implementation().execute_tool(
            tool_name,
            arguments,
            **kwargs,
        )

    async def aclose(self) -> None:
        """Close orchestrator and release resources."""
        if self._impl is not None:
            await self._impl.aclose()

    @property
    def messages(self) -> List[Any]:
        """Get conversation messages."""
        return self._get_implementation().messages

    @property
    def settings(self) -> Any:
        """Get orchestrator settings."""
        return self._get_implementation().settings

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        impl = self._get_implementation()

        session_id = getattr(impl, "session_id", None)
        if session_id:
            return session_id

        active_session_id = getattr(impl, "active_session_id", None)
        if active_session_id:
            return active_session_id

        getter = getattr(impl, "get_memory_session_id", None)
        if callable(getter):
            memory_session_id = getter()
            if memory_session_id:
                return memory_session_id

        return ""

    def reset(self) -> None:
        """
        Reset the dispatcher (useful for testing).

        Clears the cached implementation, forcing a reload on next access.
        """
        self._impl = None
        self._factory_called = False
        logger.debug("OrchestratorDispatcher: Reset")


# Singleton instance
_orchestrator_dispatcher: Optional[OrchestratorDispatcher] = None


def get_orchestrator_dispatcher() -> OrchestratorDispatcher:
    """
    Get the global orchestrator dispatcher singleton.

    Returns:
        OrchestratorDispatcher instance

    Example:
        >>> from victor.agent.orchestrator_dispatcher import get_orchestrator_dispatcher
        >>>
        >>> dispatcher = get_orchestrator_dispatcher()
        >>> result = await dispatcher.run("Do something")
    """
    global _orchestrator_dispatcher
    if _orchestrator_dispatcher is None:
        _orchestrator_dispatcher = OrchestratorDispatcher()
    return _orchestrator_dispatcher
