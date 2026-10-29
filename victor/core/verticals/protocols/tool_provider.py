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

"""Tool Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for tool selection and
tool dependency management. Following ISP, these protocols are focused
on a single responsibility: tool configuration and selection.

Usage:
    from victor.core.verticals.protocols.tool_provider import (
        ToolSelectionStrategyProtocol,
        ToolSelectionContext,
        ToolSelectionResult,
        VerticalToolSelectionProviderProtocol,
    )

    class CodingToolStrategy(ToolSelectionStrategyProtocol):
        def select_tools(self, context: ToolSelectionContext) -> ToolSelectionResult:
            if context.task_type == "edit":
                return ToolSelectionResult(
                    priority_tools=["read", "edit"],
                    budget_override=10,
                )
            return ToolSelectionResult()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor_contracts.verticals.protocols.promoted_types import (
    ToolSelectionContext as ToolSelectionContext,
    ToolSelectionResult as ToolSelectionResult,
)

# =============================================================================
# Tool Selection Data Types
# =============================================================================


# The selection data types are defined once, in the SDK (promoted from here).
# The vertical-specific name is kept as an alias; ToolSelectionContext is the
# SDK's own name for the same type.
VerticalToolSelectionContext = ToolSelectionContext


# =============================================================================
# Tool Selection Strategy Protocol
# =============================================================================


@runtime_checkable
class ToolSelectionStrategyProtocol(Protocol):
    """Protocol for vertical-specific tool selection strategies.

    Enables verticals to customize tool selection based on domain knowledge.
    This follows the Strategy Pattern (OCP) and Dependency Inversion (DIP).

    The strategy is consulted during tool selection to:
    1. Prioritize domain-relevant tools
    2. Exclude inappropriate tools for the task
    3. Adjust tool weights for semantic scoring
    4. Override tool budgets based on task complexity

    Example:
        class CodingToolSelectionStrategy(ToolSelectionStrategyProtocol):
            def select_tools(
                self,
                context: ToolSelectionContext,
            ) -> ToolSelectionResult:
                if context.task_type == "refactor":
                    return ToolSelectionResult(
                        priority_tools=["read", "edit", "search"],
                        tool_weights={"edit": 0.9, "write": 0.7},
                        budget_override=15,
                        reasoning="Refactoring requires read-edit cycles",
                    )
                return ToolSelectionResult()

            def get_task_tool_mapping(self) -> Dict[str, List[str]]:
                return {
                    "edit": ["read", "edit", "search"],
                    "debug": ["read", "shell", "search"],
                    "test": ["shell", "read", "write"],
                }
    """

    def select_tools(
        self,
        context: ToolSelectionContext,
    ) -> ToolSelectionResult:
        """Select tools based on vertical-specific strategy.

        Args:
            context: Selection context with task and conversation info

        Returns:
            ToolSelectionResult with prioritized/excluded tools and weights
        """
        ...

    def get_task_tool_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of task types to priority tools.

        Returns:
            Dict mapping task type names to ordered list of priority tools
        """
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy.

        Lower values are processed first when multiple strategies exist.

        Returns:
            Priority value (default 50)
        """
        return 50


# =============================================================================
# Vertical Tool Selection Provider Protocol
# =============================================================================


@runtime_checkable
class VerticalToolSelectionProviderProtocol(Protocol):
    """Protocol for verticals providing tool selection strategies.

    This protocol enables type-safe isinstance() checks when integrating
    vertical tool selection with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalToolSelectionProviderProtocol):
            @classmethod
            def get_tool_selection_strategy(cls) -> Optional[ToolSelectionStrategyProtocol]:
                return CodingToolSelectionStrategy()
    """

    @classmethod
    def get_tool_selection_strategy(cls) -> Optional[ToolSelectionStrategyProtocol]:
        """Get the tool selection strategy for this vertical.

        Returns:
            ToolSelectionStrategyProtocol implementation or None
        """
        ...


# =============================================================================
# Tiered Tool Config Provider Protocol
# =============================================================================


@runtime_checkable
class TieredToolConfigProviderProtocol(Protocol):
    """Protocol for verticals providing tiered tool configuration.

    Tiered tool configuration enables context-efficient tool management with
    three tiers:
    1. Mandatory: Always included tools (e.g., read, ls, grep)
    2. Vertical Core: Always included for this vertical (e.g., web_search for research)
    3. Semantic Pool: Dynamically selected based on task context

    This protocol follows ISP (Interface Segregation Principle) by focusing
    solely on tiered tool configuration, separate from tool selection strategy.

    Example:
        class ResearchVertical(VerticalBase, TieredToolConfigProviderProtocol):
            @classmethod
            def get_tiered_tool_config(cls) -> TieredToolConfig:
                return TieredToolConfig(
                    mandatory={"read", "ls", "grep"},
                    vertical_core={"web_search", "web_fetch", "overview"},
                    readonly_only_for_analysis=True,
                )

            @classmethod
            def get_tool_tier_name(cls) -> str:
                return "research"  # For registry lookup

    Integration:
        The ToolTierRegistry singleton can be used to register and retrieve
        tiered configurations across verticals, enabling cross-vertical
        tier management and inheritance.
    """

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get the tiered tool configuration for this vertical.

        Returns:
            TieredToolConfig instance or None if not using tiered configuration
        """
        ...

    @classmethod
    def get_tool_tier_name(cls) -> str:
        """Get the tier name for registry lookup.

        Returns:
            Tier name string (typically the vertical name)
        """
        ...


@runtime_checkable
class VerticalTieredToolProviderProtocol(Protocol):
    """Protocol for verticals that provide tiered tool management.

    This protocol enables type-safe isinstance() checks when integrating
    tiered tool configuration with the framework's VerticalIntegrationPipeline.

    The protocol is @runtime_checkable, allowing:
        if isinstance(vertical, VerticalTieredToolProviderProtocol):
            config = vertical.get_tiered_tool_config()
    """

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get the tiered tool configuration for this vertical.

        Returns:
            TieredToolConfig instance or None
        """
        ...

    # get_tiered_tools() removed (E5 M3) — use get_tiered_tool_config() instead


__all__ = [
    "ToolSelectionContext",
    "ToolSelectionResult",
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
]
