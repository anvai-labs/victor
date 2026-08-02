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

"""Tool graph builder for dependency graph setup.

This module handles tool dependency graph construction and planning,
extracted from ToolRegistrar as part of SRP compliance refactoring.

Single Responsibility: Build and query tool dependency graphs for planning.

Design Pattern: Builder Pattern
- Constructs tool dependency graphs
- Registers tool input/output specifications
- Provides tool planning based on goals

Usage:
    from victor.agent.tool_graph_builder import ToolGraphBuilder

    builder = ToolGraphBuilder(
        registry=tool_registry,
        tool_graph=tool_dependency_graph,
    )
    builder.build()

    # Plan tools for goals
    plan = builder.plan_for_goals(["summary", "security_report"])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from victor.providers.base import ToolDefinition
from victor.tools.registry import ToolRegistry, CostTier

logger = logging.getLogger(__name__)


@dataclass
class ToolGraphConfig:
    """Configuration for tool graph builder.

    Attributes:
        enabled: Whether tool graph is enabled
        include_cost_tiers: Include cost tier information in graph
    """

    enabled: bool = True
    include_cost_tiers: bool = True


@dataclass
class GraphBuildResult:
    """Result of tool graph building.

    Attributes:
        tools_registered: Number of tools registered in graph
        errors: List of errors encountered during building
    """

    tools_registered: int = 0
    errors: List[str] = field(default_factory=list)


class ToolGraphBuilder:
    """Builds and manages tool dependency graphs.

    Single Responsibility: Tool dependency graph construction and planning.

    This class handles:
    - Registering tool input/output specifications
    - Building tool dependency relationships
    - Planning tool sequences for goals
    - Inferring goals from user messages

    Extracted from ToolRegistrar for SRP compliance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        tool_graph: Optional[Any] = None,
        config: Optional[ToolGraphConfig] = None,
    ):
        """Initialize the tool graph builder.

        Args:
            registry: Tool registry for tool lookups
            tool_graph: Tool dependency graph instance
            config: Optional graph builder configuration
        """
        self._registry = registry
        self._tool_graph = tool_graph
        self._config = config or ToolGraphConfig()
        self._built = False

    @property
    def is_built(self) -> bool:
        """Check if graph has been built."""
        return self._built

    @property
    def tool_graph(self) -> Optional[Any]:
        """Get the underlying tool graph."""
        return self._tool_graph

    def build(self) -> GraphBuildResult:
        """Build the tool dependency graph.

        This method registers tool input/output specifications
        with cost tiers for intelligent planning.

        Returns:
            GraphBuildResult with building statistics
        """
        result = GraphBuildResult()

        if not self._config.enabled or not self._tool_graph:
            logger.debug("Tool graph disabled or not available")
            self._built = True
            return result

        try:
            result.tools_registered = self._register_tool_dependencies()
        except Exception as e:
            result.errors.append(str(e))
            logger.warning(f"Failed to build tool graph: {e}")

        self._built = True
        logger.debug(f"ToolGraphBuilder: registered {result.tools_registered} tools")
        return result

    def _register_tool_dependencies(self) -> int:
        """Register tool input/output specs for planning with cost tiers.

        Returns:
            Number of tools registered in dependency graph
        """
        if not self._tool_graph:
            return 0

        registered = 0

        # Search tools - FREE tier (local operations)
        registered += self._add_tool(
            "code_search",
            inputs=["query"],
            outputs=["file_candidates"],
            cost_tier=CostTier.FREE,
        )

        registered += self._add_tool(
            "semantic_code_search",
            inputs=["query"],
            outputs=["file_candidates"],
            cost_tier=CostTier.FREE,
        )

        # File operations - FREE tier
        registered += self._add_tool(
            "read_file",
            inputs=["file_candidates"],
            outputs=["file_contents"],
            cost_tier=CostTier.FREE,
        )

        # Analysis tools - LOW tier
        registered += self._add_tool(
            "analyze_docs",
            inputs=["file_contents"],
            outputs=["summary"],
            cost_tier=CostTier.LOW,
        )

        registered += self._add_tool(
            "code_review",
            inputs=["file_contents"],
            outputs=["summary"],
            cost_tier=CostTier.LOW,
        )

        registered += self._add_tool(
            "generate_docs",
            inputs=["file_contents"],
            outputs=["documentation"],
            cost_tier=CostTier.LOW,
        )

        registered += self._add_tool(
            "security_scan",
            inputs=["file_contents"],
            outputs=["security_report"],
            cost_tier=CostTier.LOW,
        )

        registered += self._add_tool(
            "analyze_metrics",
            inputs=["file_contents"],
            outputs=["metrics_report"],
            cost_tier=CostTier.LOW,
        )

        return registered

    def _add_tool(
        self,
        name: str,
        inputs: List[str],
        outputs: List[str],
        cost_tier: CostTier,
    ) -> int:
        """Add a tool to the dependency graph.

        Args:
            name: Tool name
            inputs: Required input types
            outputs: Produced output types
            cost_tier: Cost tier for prioritization

        Returns:
            1 if successful, 0 if failed
        """
        try:
            if self._config.include_cost_tiers:
                self._tool_graph.add_tool(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                    cost_tier=cost_tier,
                )
            else:
                self._tool_graph.add_tool(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                )
            return 1
        except Exception as e:
            logger.debug(f"Failed to add tool {name} to graph: {e}")
            return 0

    def plan_for_goals(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[ToolDefinition]:
        """Plan a sequence of tools to satisfy goals using the dependency graph.

        Args:
            goals: List of goals to achieve (e.g., ["summary", "security_report"])
            available_inputs: Already available inputs

        Returns:
            List of ToolDefinition objects for the planned tools
        """
        if not goals or not self._tool_graph:
            return []

        available = available_inputs or []
        plan_names = self._tool_graph.plan(goals, available)
        tool_defs: List[ToolDefinition] = []

        for name in plan_names:
            tool = self._registry.get(name)
            if tool and self._registry.is_tool_enabled(name):
                tool_defs.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )

        return tool_defs

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request.

        Uses keyword matching to identify what the user wants to achieve.

        Args:
            user_message: User's message

        Returns:
            List of inferred goal names
        """
        text = user_message.lower()
        goals: List[str] = []

        if any(kw in text for kw in ["summarize", "summary", "analyze", "overview"]):
            goals.append("summary")

        if any(kw in text for kw in ["review", "code review", "audit"]):
            goals.append("summary")

        if any(kw in text for kw in ["doc", "documentation", "readme"]):
            goals.append("documentation")

        if any(kw in text for kw in ["security", "vulnerability", "secret", "scan"]):
            goals.append("security_report")

        if any(kw in text for kw in ["complexity", "metrics", "maintainability", "technical debt"]):
            goals.append("metrics_report")

        return goals


__all__ = ["ToolGraphBuilder", "ToolGraphConfig", "GraphBuildResult"]
