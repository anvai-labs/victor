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

"""ISP-Compliant Vertical Provider Protocols.

This module holds the vertical provider protocols that are still defined
locally because their hook shape is not (yet) part of the SDK surface.

The provider protocols that the SDK also defines — MiddlewareProvider,
SafetyProvider, WorkflowProvider, TeamProvider, RLProvider,
EnrichmentProvider, ToolProvider, HandlerProvider, CapabilityProvider and
ModeConfigProvider — are canonically defined in ``victor_contracts`` and
re-exported by the ``victor.core.verticals.protocols`` package. Import them
from the package (or from ``victor_contracts`` directly), not from here:

    from victor.core.verticals.protocols import MiddlewareProvider

The protocols below follow the Interface Segregation Principle (ISP): the
framework uses isinstance() checks against them to determine which
capabilities a vertical supports.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

# =============================================================================
# Prompt Contributor Provider Protocol
# =============================================================================


@runtime_checkable
class PromptContributorProvider(Protocol):
    """Protocol for verticals that provide prompt contributors.

    Prompt contributors add vertical-specific hints, sections, and
    task type mappings to prompts.

    Example:
        class ExampleVertical(VerticalBase, PromptContributorProvider):
            @classmethod
            def get_prompt_contributor(cls) -> Optional[Any]:
                return ExamplePromptContributor()

            @classmethod
            def get_task_type_hints(cls) -> Dict[str, Any]:
                return {
                    "search": {"hint": "Use search tools", "priority_tools": ["search"]},
                    "synthesize": {"hint": "Combine sources", "priority_tools": ["write"]},
                }
    """

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Get prompt contributor for this vertical.

        Returns:
            Prompt contributor (PromptContributorProtocol) or None
        """
        ...

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get task-type-specific prompt hints.

        Returns:
            Dictionary mapping task types to hint configurations
        """
        ...


# =============================================================================
# Tool Dependency Provider Protocol
# =============================================================================


@runtime_checkable
class ToolDependencyProvider(Protocol):
    """Protocol for verticals that provide tool dependencies.

    Tool dependency providers define relationships between tools,
    enabling intelligent tool sequencing and pre-requisite handling.

    Example:
        class ExampleVertical(VerticalBase, ToolDependencyProvider):
            @classmethod
            def get_tool_dependency_provider(cls) -> Optional[Any]:
                return ExampleToolDependencyProvider()
    """

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Get tool dependency provider for this vertical.

        Returns:
            Tool dependency provider (ToolDependencyProviderProtocol) or None
        """
        ...


# =============================================================================
# Tiered Tool Config Provider Protocol
# =============================================================================


@runtime_checkable
class TieredToolConfigProvider(Protocol):
    """Protocol for verticals that provide tiered tool configuration.

    Tiered tool configuration enables context-efficient tool management with
    three tiers:
    1. Mandatory: Always included tools (e.g., read, ls)
    2. Vertical Core: Always included for this vertical
    3. Semantic Pool: Selected based on query similarity

    Example:
        class ExampleVertical(VerticalBase, TieredToolConfigProvider):
            @classmethod
            def get_tiered_tool_config(cls) -> Optional[Any]:
                return {
                    "basic_tools": {"read", "ls"},
                    "standard_tools": {"web_search", "web_fetch"},
                    "advanced_tools": {"write", "edit"},
                }
    """

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get tiered tool configuration for this vertical.

        Returns:
            TieredToolConfig instance or None
        """
        ...


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for verticals that provide DI services.

    Service providers register vertical-specific services with the
    dependency injection container.

    Example:
        class ExampleVertical(VerticalBase, ServiceProvider):
            @classmethod
            def get_service_provider(cls) -> Optional[Any]:
                return ExampleServiceProvider()
    """

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Get service provider for this vertical.

        Returns:
            Service provider (ServiceProviderProtocol) or None
        """
        ...


__all__ = [
    # Additional provider protocols (locally defined; the rest are SDK-backed
    # re-exports from the victor.core.verticals.protocols package)
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
]
