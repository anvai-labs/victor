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

"""Base tool selector implementing IToolSelector protocol.

This module provides a base class for tool selectors that implements
the common functionality shared across all selector strategies.

Design Pattern: Template Method
==============================
BaseToolSelector provides the infrastructure and template methods.
Subclasses implement the specific scoring algorithm via _score_tool().

Usage:
    from victor.tools.selection.base_selector import BaseToolSelector

    class MySelector(BaseToolSelector):
        def _score_tool(self, tool_name: str, task: str, context: ...) -> float:
            # Implement scoring logic
            return score
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.config.defaults import ToolDefaults
from victor.protocols.tool_selector import (
    IConfigurableToolSelector,
    ToolSelectionContext,
    ToolSelectionResult,
    ToolSelectionStrategy,
)

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseToolSelector(ABC):
    """Abstract base class for tool selectors.

    Provides common infrastructure for tool selection while allowing
    different scoring strategies via the _score_tool() method.

    Implements IConfigurableToolSelector protocol.

    Attributes:
        tool_registry: Registry of available tools
        enabled_tools: Set of enabled tool names (None = all)
        disabled_tools: Set of disabled tool names
        model_name: Current model name for context
        provider_name: Current provider name for context
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        *,
        enabled_tools: Optional[Set[str]] = None,
        disabled_tools: Optional[Set[str]] = None,
        model_name: str = "",
        provider_name: str = "",
    ) -> None:
        """Initialize the base tool selector.

        Args:
            tool_registry: Registry of available tools
            enabled_tools: Set of enabled tool names (None = all)
            disabled_tools: Set of disabled tool names
            model_name: Current model name for context
            provider_name: Current provider name for context
        """
        self._tool_registry = tool_registry
        self._enabled_tools = enabled_tools
        self._disabled_tools = disabled_tools or set()
        self._model_name = model_name
        self._provider_name = provider_name

    @property
    @abstractmethod
    def strategy(self) -> ToolSelectionStrategy:
        """Get the selection strategy used by this selector."""
        ...

    @abstractmethod
    def _score_tool(
        self,
        tool_name: str,
        task: str,
        context: Optional[ToolSelectionContext] = None,
    ) -> float:
        """Score a tool's relevance to a task.

        Subclasses must implement this method with their specific
        scoring algorithm.

        Args:
            tool_name: Name of the tool to score
            task: Task description to score against
            context: Optional additional context

        Returns:
            Relevance score from 0.0 to 1.0
        """
        ...

    def select_tools(
        self,
        task: str,
        *,
        limit: int = ToolDefaults.TOOL_SELECTION_LIMIT,
        min_score: float = ToolDefaults.TOOL_SELECTION_MIN_SCORE,
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
        # Get available tools (filtered by enabled/disabled)
        available_tools = self._get_available_tools(context)

        if not available_tools:
            return ToolSelectionResult(
                tool_names=[],
                scores={},
                strategy_used=self.strategy,
                metadata={"reason": "no_available_tools"},
            )

        # Score all available tools
        scores: Dict[str, float] = {}
        for tool_name in available_tools:
            try:
                score = self._score_tool(tool_name, task, context)
                if score >= min_score:
                    scores[tool_name] = score
            except Exception as e:
                logger.debug(f"Error scoring tool '{tool_name}': {e}")
                continue

        # Sort by score descending
        sorted_tools = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

        # Apply limit
        selected_tools = sorted_tools[:limit]

        return ToolSelectionResult(
            tool_names=selected_tools,
            scores={t: scores[t] for t in selected_tools},
            strategy_used=self.strategy,
            metadata={
                "total_available": len(available_tools),
                "scored": len(scores),
                "min_score_used": min_score,
            },
        )

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
            Relevance score from 0.0 to 1.0
        """
        # Check if tool is available
        available_tools = self._get_available_tools(context)
        if tool_name not in available_tools:
            return 0.0

        try:
            return self._score_tool(tool_name, task, context)
        except Exception as e:
            logger.debug(f"Error scoring tool '{tool_name}': {e}")
            return 0.0

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
        if enabled_tools is not None:
            self._enabled_tools = enabled_tools

        if disabled_tools is not None:
            self._disabled_tools = disabled_tools

        # Cost tier filtering is handled by subclasses if needed
        if cost_tier_filter is not None:
            self._apply_cost_tier_filter(cost_tier_filter)

    def reset_configuration(self) -> None:
        """Reset selector to default configuration."""
        self._enabled_tools = None
        self._disabled_tools = set()

    def get_available_tools(self) -> List[str]:
        """Get list of all available tool names.

        Returns:
            List of tool names that can be selected
        """
        return list(self._get_available_tools())

    def _get_available_tools(
        self,
        context: Optional[ToolSelectionContext] = None,
    ) -> Set[str]:
        """Get set of available tools after filtering.

        Args:
            context: Optional context with additional filters

        Returns:
            Set of available tool names
        """
        # Start with all tools from registry
        all_tools = set(self._tool_registry.list_tools())

        # Apply enabled filter
        if self._enabled_tools is not None:
            all_tools &= self._enabled_tools

        # Apply disabled filter
        all_tools -= self._disabled_tools

        # Apply context filters if provided
        if context:
            if context.enabled_tools is not None:
                all_tools &= context.enabled_tools
            all_tools -= context.disabled_tools
            all_tools -= context.failed_tools

        return all_tools

    def _apply_cost_tier_filter(self, cost_tier: str) -> None:
        """Apply cost tier filtering.

        Override in subclasses that support cost tier filtering.

        Args:
            cost_tier: Cost tier to filter by
        """
        # Default implementation logs that cost tier filtering is not supported
        # Subclasses can override to implement cost tier filtering
        logger.debug(
            f"{type(self).__name__} does not support cost tier filtering, "
            f"ignoring cost_tier={cost_tier}"
        )

    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description or empty string
        """
        tool = self._tool_registry.get_tool(tool_name)
        if tool:
            return getattr(tool, "description", "")
        return ""


__all__ = ["BaseToolSelector"]
