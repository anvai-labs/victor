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

"""Protocols for SubAgent dependencies - ISP compliant.

This module defines the minimal interface required by SubAgent instances,
following the Interface Segregation Principle (ISP). Instead of depending
on the full AgentOrchestrator interface, sub-agents depend only on the
specific capabilities they need.

Design Principles:
- ISP Compliance: SubAgentContext contains only methods needed by SubAgent
- DIP Compliance: SubAgent depends on abstractions, not concrete implementations
- Adapter Pattern: SubAgentContextAdapter bridges AgentOrchestrator to protocol
- OCP Compliance: RoleToolProvider enables extension without modifying code

Usage:
    from victor.agent.subagents.protocols import (
        SubAgentContext,
        SubAgentContextAdapter,
        RoleToolProvider,
    )

    # Type hint with protocol
    def create_subagent(context: SubAgentContext) -> SubAgent:
        return SubAgent(config, context)

    # Adapt from orchestrator
    context = SubAgentContextAdapter(orchestrator)
    subagent = create_subagent(context)

    # Custom role provider for vertical-specific tools
    class InvestmentRoleProvider:
        def get_tools(self, role, vertical=None):
            return ["read", "sec_filing", "valuation", ...]

Example:
    # Direct usage with adapter
    adapter = SubAgentContextAdapter(parent_orchestrator)
    print(f"Using provider: {adapter.provider_name}")
    print(f"Model: {adapter.model}")

    # Protocol enables testing with mocks
    mock_context = MagicMock(spec=SubAgentContext)
    mock_context.provider_name = "test_provider"
    mock_context.model = "test_model"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator


# Re-export from canonical location for backward compatibility
from victor.core.shared_types import SubAgentContext  # noqa: F401


class SubAgentContextAdapter:
    """Adapter to create SubAgentContext from AgentOrchestrator.

    This adapter implements the Adapter pattern to bridge the gap between
    the full AgentOrchestrator interface and the minimal SubAgentContext
    protocol that SubAgent needs.

    The adapter:
    - Wraps an existing AgentOrchestrator instance
    - Exposes only the properties defined in SubAgentContext
    - Delegates to the wrapped orchestrator for implementation

    Usage:
        orchestrator = AgentOrchestrator(settings, provider="anthropic")
        context = SubAgentContextAdapter(orchestrator)

        # Now use context instead of orchestrator for SubAgent creation
        subagent = SubAgent(config, context)

    Attributes:
        _orchestrator: The wrapped AgentOrchestrator instance
    """

    def __init__(self, orchestrator: "AgentOrchestrator"):
        """Initialize adapter with orchestrator.

        Args:
            orchestrator: AgentOrchestrator instance to adapt
        """
        self._orchestrator = orchestrator

    @property
    def settings(self) -> Any:
        """Get configuration settings from orchestrator.

        Returns:
            Settings object from the wrapped orchestrator
        """
        return self._orchestrator.settings

    @property
    def provider(self) -> Any:
        """Get provider instance from orchestrator.

        Returns:
            BaseProvider instance from the wrapped orchestrator
        """
        return self._orchestrator.provider

    @property
    def provider_name(self) -> str:
        """Get provider name from orchestrator.

        Returns:
            Provider name from the wrapped orchestrator
        """
        return self._orchestrator.provider_name

    @property
    def model(self) -> str:
        """Get model identifier from orchestrator.

        Returns:
            Model identifier from the wrapped orchestrator
        """
        return self._orchestrator.model

    @property
    def tool_registry(self) -> Any:
        """Get tool registry from orchestrator.

        Returns:
            Tool registry from the wrapped orchestrator
        """
        # AgentOrchestrator uses 'tools' as the ToolRegistry attribute
        return self._orchestrator.tools

    @property
    def temperature(self) -> float:
        """Get temperature setting from orchestrator.

        Returns:
            Temperature value from the wrapped orchestrator
        """
        return self._orchestrator.temperature

    @property
    def reasoning_effort(self) -> Optional[str]:
        """Get the parent orchestrator's reasoning_effort (None = unset).

        Lets a sub-agent inherit the parent's reasoning effort when the member
        spec doesn't override it.
        """
        return getattr(self._orchestrator, "reasoning_effort", None)

    @property
    def vertical_context(self) -> Any:
        """Get vertical context from orchestrator.

        Returns:
            VerticalContext from the wrapped orchestrator, or None
        """
        return getattr(self._orchestrator, "_vertical_context", None)


@runtime_checkable
class RoleToolProvider(Protocol):
    """Protocol for providing role-specific tool configurations.

    This enables vertical-aware tool configuration following OCP.
    Third-party verticals can implement this protocol to provide
    custom tools for each subagent role.

    Example:
        class InvestmentRoleProvider:
            def get_tools_for_role(self, role, vertical=None):
                if role == "researcher":
                    return ["read", "sec_filing", "valuation", "market_data"]
                return ["read"]

            def get_budget_for_role(self, role):
                return 20

            def get_context_limit_for_role(self, role):
                return 50000
    """

    def get_tools_for_role(
        self,
        role: str,
        vertical: Optional[str] = None,
    ) -> List[str]:
        """Get tools available for a role, optionally within a vertical.

        Args:
            role: Role name (researcher, planner, executor, etc.)
            vertical: Optional vertical name (coding, investment, etc.)

        Returns:
            List of tool names available for this role
        """
        ...

    def get_budget_for_role(self, role: str) -> int:
        """Get tool budget for a role.

        Args:
            role: Role name

        Returns:
            Maximum number of tool calls allowed
        """
        ...

    def get_context_limit_for_role(self, role: str) -> int:
        """Get context character limit for a role.

        Args:
            role: Role name

        Returns:
            Maximum context characters
        """
        ...


class DefaultRoleToolProvider:
    """Default role tool provider with hardcoded tools.

    Maintains backward compatibility with existing ROLE_DEFAULT_TOOLS.
    Can be extended or replaced for vertical-specific behavior.
    """

    # Core tools available to all roles
    CORE_TOOLS = ["read", "ls", "grep"]

    # Role-specific default tools (coding-focused for backward compat)
    ROLE_TOOLS: Dict[str, List[str]] = {
        "researcher": [
            "read",
            "ls",
            "grep",
            "search",
            "code_search",
            "semantic_code_search",
            "web_search",
            "web_fetch",
        ],
        "planner": ["read", "ls", "grep", "search", "plan_files"],
        "executor": [
            "read",
            "write",
            "edit",
            "ls",
            "grep",
            "search",
            "shell",
            "test",
            "git",
        ],
        "reviewer": ["read", "ls", "grep", "search", "git", "test", "shell"],
        "tester": ["read", "write", "ls", "grep", "search", "test", "shell"],
    }

    ROLE_BUDGETS: Dict[str, int] = {
        "researcher": 15,
        "planner": 10,
        "executor": 30,
        "reviewer": 15,
        "tester": 20,
    }

    ROLE_CONTEXT_LIMITS: Dict[str, int] = {
        "researcher": 50000,
        "planner": 30000,
        "executor": 80000,
        "reviewer": 40000,
        "tester": 50000,
    }

    def get_tools_for_role(
        self,
        role: str,
        vertical: Optional[str] = None,
    ) -> List[str]:
        """Get tools for role. Vertical parameter reserved for extension."""
        role_lower = role.lower()
        return self.ROLE_TOOLS.get(role_lower, self.CORE_TOOLS)

    def get_budget_for_role(self, role: str) -> int:
        """Get budget for role."""
        return self.ROLE_BUDGETS.get(role.lower(), 15)

    def get_context_limit_for_role(self, role: str) -> int:
        """Get context limit for role."""
        return self.ROLE_CONTEXT_LIMITS.get(role.lower(), 50000)


# Global default provider instance
_default_role_provider: Optional[RoleToolProvider] = None


def get_role_tool_provider() -> RoleToolProvider:
    """Get the global role tool provider.

    Returns:
        RoleToolProvider instance (default or custom-registered)
    """
    global _default_role_provider
    if _default_role_provider is None:
        _default_role_provider = DefaultRoleToolProvider()
    return _default_role_provider


def set_role_tool_provider(provider: RoleToolProvider) -> None:
    """Set a custom role tool provider.

    This allows verticals to provide custom role configurations.

    Args:
        provider: Custom RoleToolProvider implementation

    Example:
        from victor_invest.role_provider import InvestmentRoleProvider
        set_role_tool_provider(InvestmentRoleProvider())
    """
    global _default_role_provider
    _default_role_provider = provider


__all__ = [
    "SubAgentContext",
    "SubAgentContextAdapter",
    "RoleToolProvider",
    "DefaultRoleToolProvider",
    "get_role_tool_provider",
    "set_role_tool_provider",
]
