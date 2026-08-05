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

"""Agent orchestrator protocol for breaking circular dependencies.

This module defines the IAgentOrchestrator protocol that captures the core
interface of AgentOrchestrator without requiring its import. This breaks
circular dependencies where modules need orchestrator functionality.

Problem:
    - victor/agent/orchestrator.py imports from providers, tools, workflows
    - Those modules often need orchestrator features for context
    - Direct imports create circular dependency chains

Solution:
    - Define IAgentOrchestrator protocol in neutral location
    - Modules depend on protocol, not concrete class
    - Orchestrator implements protocol implicitly (duck typing)

Usage:
    # Instead of:
    from victor.agent.orchestrator import AgentOrchestrator
    def process(orchestrator: AgentOrchestrator): ...

    # Use:
    from victor.protocols.agent import IAgentOrchestrator
    def process(orchestrator: IAgentOrchestrator): ...

Design Principles:
    - DIP: Depend on abstractions (protocol), not concretions (class)
    - ISP: Protocol contains only commonly-needed methods
    - OCP: New features added via protocol extension, not modification
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.providers.base import StreamChunk


@runtime_checkable
class IAgentOrchestrator(Protocol):
    """Protocol for agent orchestrator functionality.

    This protocol defines the minimal interface for agent orchestration
    that other modules commonly need. It enables dependency inversion
    and breaks circular import chains.

    Implementations:
        - AgentOrchestrator (full implementation)
        - Mock implementations for testing

    Categories of functionality:
        1. Chat/completion interface
        2. Provider/model access
        3. Tool registry access
        4. Session state access
        5. Configuration access
    """

    # =========================================================================
    # CHAT/COMPLETION INTERFACE
    # =========================================================================

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a message and get a response.

        Args:
            message: User message
            stream: Whether to stream the response
            **kwargs: Additional provider-specific options

        Returns:
            Response from the LLM (string or structured response)
        """
        ...

    async def stream_chat(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncIterator["StreamChunk"]:
        """Stream a chat response.

        Args:
            message: User message
            **kwargs: Additional provider-specific options

        Yields:
            StreamChunk objects with response content
        """
        ...

    # =========================================================================
    # PROVIDER/MODEL ACCESS
    # =========================================================================

    @property
    def provider(self) -> Any:
        """Get the current LLM provider instance.

        Returns:
            BaseProvider instance
        """
        ...

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider.

        Returns:
            Provider name (e.g., 'anthropic', 'openai', 'ollama')
        """
        ...

    @property
    def model(self) -> str:
        """Get the current model identifier.

        Returns:
            Model name (e.g., 'claude-sonnet-4-20250514', 'gpt-4o')
        """
        ...

    @property
    def temperature(self) -> float:
        """Get the temperature setting for sampling.

        Returns:
            Temperature value (0.0 to 1.0+)
        """
        ...

    # =========================================================================
    # TOOL REGISTRY ACCESS
    # =========================================================================

    @property
    def tool_registry(self) -> Any:
        """Get the tool registry.

        Returns:
            ToolRegistry instance with available tools
        """
        ...

    @property
    def allowed_tools(self) -> Optional[List[str]]:
        """Get list of allowed tool names, if restricted.

        Returns:
            List of allowed tool names, or None if all allowed
        """
        ...

    # =========================================================================
    # SESSION STATE ACCESS
    # =========================================================================

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used in this session.

        Returns:
            Count of tool calls made
        """
        ...

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tool names in order.

        Returns:
            List of tool names that have been executed
        """
        ...

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Get set of failed tool call signatures.

        Returns:
            Set of (tool_name, args_hash) for failed calls
        """
        ...

    @property
    def observed_files(self) -> Set[str]:
        """Get set of files observed during session.

        Returns:
            Set of file paths that have been read
        """
        ...

    # =========================================================================
    # CONFIGURATION ACCESS
    # =========================================================================

    @property
    def settings(self) -> Any:
        """Get configuration settings.

        Returns:
            Settings object with agent configuration
        """
        ...

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this session.

        Returns:
            Maximum number of tool calls allowed
        """
        ...

    @property
    def mode(self) -> Any:
        """Get the current agent mode.

        Returns:
            AgentMode enum value (BUILD, PLAN, EXPLORE)
        """
        ...


@runtime_checkable
class IAgentOrchestratorFactory(Protocol):
    """Protocol for creating agent orchestrators.

    This enables dependency injection of orchestrator creation
    without importing the concrete factory.
    """

    def create(
        self,
        provider_name: str = "anthropic",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> IAgentOrchestrator:
        """Create a new agent orchestrator.

        Args:
            provider_name: Name of LLM provider to use
            model: Model identifier (provider-specific)
            **kwargs: Additional configuration

        Returns:
            Configured IAgentOrchestrator instance
        """
        ...


__all__ = [
    "IAgentOrchestrator",
    "IAgentOrchestratorFactory",
]
