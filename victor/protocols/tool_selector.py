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

"""Tool selector protocol for unified tool selection interface.

This module defines the IToolSelector protocol that all tool selector
implementations should follow. This eliminates the need for parallel
implementations with incompatible interfaces.

Implementations:
    - SemanticToolSelector: Embedding-based semantic similarity matching
    - KeywordToolSelector: Keyword and pattern-based matching
    - HybridToolSelector: Combination of semantic and keyword approaches

Design Principles:
    - ISP: Protocol defines minimal interface needed by consumers
    - DIP: Consumers depend on protocol, not concrete implementations
    - OCP: New selectors can be added without modifying existing code

Usage:
    from victor.protocols.tool_selector import IToolSelector

    def select_tools(selector: IToolSelector, task: str) -> List[str]:
        return selector.select_tools(task, limit=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry


class ToolSelectionStrategy(Enum):
    """Enumeration of tool selection strategies."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ToolSelectionResult:
    """Result from tool selection operation.

    Attributes:
        tool_names: Ordered list of selected tool names (most relevant first)
        scores: Dictionary mapping tool names to relevance scores (0.0-1.0)
        strategy_used: Which selection strategy produced this result
        metadata: Additional metadata about the selection process
    """

    tool_names: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    strategy_used: ToolSelectionStrategy = ToolSelectionStrategy.HYBRID
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_tool(self) -> Optional[str]:
        """Get the top-ranked tool name."""
        return self.tool_names[0] if self.tool_names else None

    def filter_by_score(self, min_score: float) -> "ToolSelectionResult":
        """Return new result with only tools above minimum score.

        Args:
            min_score: Minimum score threshold (0.0-1.0)

        Returns:
            New ToolSelectionResult with filtered tools
        """
        filtered_names = [
            name for name in self.tool_names if self.scores.get(name, 0.0) >= min_score
        ]
        filtered_scores = {
            name: self.scores[name] for name in filtered_names if name in self.scores
        }
        return ToolSelectionResult(
            tool_names=filtered_names,
            scores=filtered_scores,
            strategy_used=self.strategy_used,
            metadata=self.metadata,
        )


@dataclass
class ToolSelectionContext:
    """Context information for tool selection.

    Provides additional context to help selectors make better decisions.

    Attributes:
        task_description: The task or query to select tools for
        conversation_stage: Current stage of the conversation (e.g., "planning", "executing")
        previous_tools: Tools that have already been used in this session
        failed_tools: Tools that failed in previous attempts
        model_name: Current LLM model being used
        provider_name: Current LLM provider
        enabled_tools: Set of explicitly enabled tool names (None = all enabled)
        disabled_tools: Set of explicitly disabled tool names
        cost_budget: Optional cost budget constraint
    """

    task_description: str
    conversation_stage: Optional[str] = None
    previous_tools: List[str] = field(default_factory=list)
    failed_tools: Set[str] = field(default_factory=set)
    model_name: str = ""
    provider_name: str = ""
    enabled_tools: Optional[Set[str]] = None
    disabled_tools: Set[str] = field(default_factory=set)
    cost_budget: Optional[float] = None


@runtime_checkable
class IToolSelector(Protocol):
    """Protocol for tool selection implementations.

    This protocol defines the minimal interface that all tool selectors
    must implement. It enables dependency injection and testing while
    allowing different selection strategies.

    Implementations should be stateless or thread-safe, as selectors
    may be shared across multiple concurrent sessions.
    """

    def select_tools(
        self,
        task: str,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[ToolSelectionContext] = None,
    ) -> ToolSelectionResult:
        """Select relevant tools for a task.

        Args:
            task: Task description or query to match tools against
            limit: Maximum number of tools to return
            min_score: Minimum relevance score threshold (0.0-1.0)
            context: Optional additional context for selection

        Returns:
            ToolSelectionResult with ranked tool names and scores
        """
        ...

    def get_tool_score(
        self,
        tool_name: str,
        task: str,
        *,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        """Get relevance score for a specific tool.

        Args:
            tool_name: Name of the tool to score
            task: Task description to score against
            context: Optional additional context

        Returns:
            Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)
        """
        ...

    @property
    def strategy(self) -> ToolSelectionStrategy:
        """Get the selection strategy used by this selector."""
        ...


@runtime_checkable
class IConfigurableToolSelector(IToolSelector, Protocol):
    """Extended protocol for configurable tool selectors.

    Adds configuration and state management capabilities to the base
    tool selector interface.
    """

    def configure(
        self,
        *,
        enabled_tools: Optional[Set[str]] = None,
        disabled_tools: Optional[Set[str]] = None,
        cost_tier_filter: Optional[str] = None,
    ) -> None:
        """Configure selector options.

        Args:
            enabled_tools: Set of tool names to enable (None = all)
            disabled_tools: Set of tool names to disable
            cost_tier_filter: Filter by cost tier (e.g., "FREE", "LOW")
        """
        ...

    def reset_configuration(self) -> None:
        """Reset selector to default configuration."""
        ...

    def get_available_tools(self) -> List[str]:
        """Get list of all available tool names.

        Returns:
            List of tool names that can be selected
        """
        ...


@runtime_checkable
class IToolSelectorFactory(Protocol):
    """Protocol for creating tool selectors.

    Enables dependency injection of selector creation without
    importing concrete factory implementations.
    """

    def create_selector(
        self,
        strategy: ToolSelectionStrategy = ToolSelectionStrategy.HYBRID,
        tool_registry: Optional["ToolRegistry"] = None,
        **kwargs: Any,
    ) -> IToolSelector:
        """Create a new tool selector.

        Args:
            strategy: Selection strategy to use
            tool_registry: Optional tool registry (uses default if None)
            **kwargs: Additional configuration options

        Returns:
            Configured IToolSelector instance
        """
        ...


__all__ = [
    "IToolSelector",
    "IConfigurableToolSelector",
    "IToolSelectorFactory",
    "ToolSelectionResult",
    "ToolSelectionContext",
    "ToolSelectionStrategy",
]
