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

"""Optimization strategies for workflow improvement.

This module provides concrete implementations of optimization strategies
that can be applied to workflows to improve performance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from victor.optimization.models import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
    NodeStatistics,
    OptimizationOpportunity,
    OptimizationStrategyType,
    WorkflowProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowChange:
    """Represents a change to be applied to a workflow.

    Attributes:
        change_type: Type of change (remove_node, create_parallel, substitute_tool, etc.)
        target: Target node/tool ID
        description: Human-readable description
        metadata: Additional metadata about the change
    """

    change_type: str
    target: str
    description: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseOptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    @abstractmethod
    def can_apply(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> bool:
        """Check if strategy can be applied to given bottleneck."""
        pass

    @abstractmethod
    def generate_opportunity(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> Optional[OptimizationOpportunity]:
        """Generate optimization opportunity."""
        pass

    @property
    @abstractmethod
    def risk_level(self) -> BottleneckSeverity:
        """Risk level of applying this strategy."""
        pass


class PruningStrategy(BaseOptimizationStrategy):
    """Removes unnecessary or failing nodes from workflow.

    This strategy identifies nodes that can be safely removed:
    - Nodes with low success rates
    - Nodes whose outputs are never used
    - Redundant nodes performing duplicate operations
    """

    def can_apply(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> bool:
        """Pruning applies to unreliable or redundant nodes."""
        return bottleneck.type in [
            BottleneckType.UNRELIABLE_NODE,
            BottleneckType.REDUNDANT_OPERATION,
            BottleneckType.UNUSED_OUTPUT,
        ]

    def generate_opportunity(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> Optional[OptimizationOpportunity]:
        """Generate pruning opportunity."""
        if not bottleneck.node_id:
            return None

        if bottleneck.type == BottleneckType.UNRELIABLE_NODE:
            stats = profile.node_stats.get(bottleneck.node_id)
            if not stats:
                return None

            return OptimizationOpportunity(
                strategy_type=OptimizationStrategyType.PRUNING,
                target=bottleneck.node_id,
                description=(
                    f"Remove consistently failing node '{bottleneck.node_id}' "
                    f"(success rate: {stats.success_rate:.1%})"
                ),
                expected_improvement=1.0 - stats.success_rate,
                risk_level=BottleneckSeverity.HIGH,
                estimated_cost_reduction=stats.total_cost,
                estimated_duration_reduction=stats.avg_duration,
                confidence=0.6,
                metadata={
                    "reason": "low_success_rate",
                    "current_success_rate": stats.success_rate,
                },
            )

        elif bottleneck.type == BottleneckType.UNUSED_OUTPUT:
            return OptimizationOpportunity(
                strategy_type=OptimizationStrategyType.PRUNING,
                target=bottleneck.node_id,
                description=(f"Remove node '{bottleneck.node_id}' with unused outputs"),
                expected_improvement=0.1,  # Small improvement
                risk_level=BottleneckSeverity.MEDIUM,
                confidence=0.8,
                metadata={
                    "reason": "unused_output",
                },
            )

        return None

    @property
    def risk_level(self) -> BottleneckSeverity:
        return BottleneckSeverity.HIGH


@dataclass
class ParallelGroup:
    """Represents a group of nodes that can execute in parallel.

    Attributes:
        node_ids: IDs of nodes in the group
        estimated_speedup: Expected speedup from parallelization
        sequential_duration: Total duration if executed sequentially
        parallel_duration: Expected duration if executed in parallel
    """

    node_ids: List[str]
    estimated_speedup: float
    sequential_duration: float
    parallel_duration: float


class ParallelizationStrategy(BaseOptimizationStrategy):
    """Converts sequential node execution to parallel where possible.

    This strategy identifies independent nodes that can execute concurrently:
    - Nodes with no data dependencies
    - Nodes that don't modify the same state keys
    - Nodes that can be grouped for parallel execution
    """

    def __init__(self):
        """Initialize parallelization strategy."""
        self._dependency_cache: Dict[str, Set[str]] = {}

    def can_apply(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> bool:
        """Parallelization applies to slow or dominant nodes."""
        return bottleneck.type in [
            BottleneckType.SLOW_NODE,
            BottleneckType.DOMINANT_NODE,
        ]

    def generate_opportunity(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> Optional[OptimizationOpportunity]:
        """Generate parallelization opportunity."""
        if not bottleneck.node_id:
            return None

        # Find parallelizable nodes
        parallel_groups = self._find_parallelizable_nodes(profile)

        # Find group containing the bottleneck node
        for group in parallel_groups:
            if bottleneck.node_id in group.node_ids:
                return OptimizationOpportunity(
                    strategy_type=OptimizationStrategyType.PARALLELIZATION,
                    target=",".join(group.node_ids),
                    description=(
                        f"Execute {len(group.node_ids)} nodes in parallel "
                        f"(potential speedup: {group.estimated_speedup:.2f}x)"
                    ),
                    expected_improvement=(group.estimated_speedup - 1) / group.estimated_speedup,
                    risk_level=BottleneckSeverity.MEDIUM,
                    estimated_duration_reduction=(
                        group.sequential_duration - group.parallel_duration
                    ),
                    confidence=0.7,
                    metadata={
                        "node_ids": group.node_ids,
                        "estimated_speedup": group.estimated_speedup,
                        "sequential_duration": group.sequential_duration,
                        "parallel_duration": group.parallel_duration,
                    },
                )

        return None

    def _find_parallelizable_nodes(
        self,
        profile: WorkflowProfile,
    ) -> List[ParallelGroup]:
        """Identify groups of nodes that can execute in parallel.

        Args:
            profile: Workflow profile

        Returns:
            List of parallelizable groups
        """
        # Build dependency graph
        deps = self._build_dependency_graph(profile)

        parallel_groups = []
        visited = set()

        # For each node, find independent nodes
        for node_id in profile.node_stats.keys():
            if node_id in visited:
                continue

            # Find all nodes that can execute with this node
            independent_nodes = self._find_independent_nodes(
                node_id,
                deps,
                visited,
                profile.node_stats.keys(),
            )

            if len(independent_nodes) > 1:
                # Calculate estimated speedup
                durations = [profile.node_stats[n].p95_duration for n in independent_nodes]
                sequential_time = sum(durations)
                parallel_time = max(durations)
                speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

                parallel_groups.append(
                    ParallelGroup(
                        node_ids=list(independent_nodes),
                        estimated_speedup=speedup,
                        sequential_duration=sequential_time,
                        parallel_duration=parallel_time,
                    )
                )

        return parallel_groups

    def _build_dependency_graph(
        self,
        profile: WorkflowProfile,
    ) -> Dict[str, Set[str]]:
        """Build dependency graph showing which nodes depend on which.

        This is a simplified implementation. A full implementation would:
        1. Analyze StateGraph structure
        2. Check state read/write patterns
        3. Identify explicit edge dependencies

        Args:
            profile: Workflow profile

        Returns:
            Dictionary mapping node IDs to their dependencies
        """
        # Simplified: assume no dependencies for MVP
        # In production, this would analyze the actual workflow structure
        return {node_id: set() for node_id in profile.node_stats.keys()}

    def _find_independent_nodes(
        self,
        node_id: str,
        deps: Dict[str, Set[str]],
        visited: Set[str],
        all_nodes: List[str],
    ) -> Set[str]:
        """Find nodes that are independent of the given node.

        Args:
            node_id: Starting node
            deps: Dependency graph
            visited: Set of already visited nodes
            all_nodes: All node IDs

        Returns:
            Set of independent node IDs
        """
        independent = {node_id}
        visited.add(node_id)

        node_deps = deps.get(node_id, set())

        for other_node in all_nodes:
            if other_node in visited:
                continue

            other_deps = deps.get(other_node, set())

            # Check if nodes are independent (no dependencies in either direction)
            if other_node not in node_deps and node_id not in other_deps:
                independent.add(other_node)
                visited.add(other_node)

        return independent

    @property
    def risk_level(self) -> BottleneckSeverity:
        return BottleneckSeverity.MEDIUM


class ToolSelectionStrategy(BaseOptimizationStrategy):
    """Selects optimal tools based on performance metrics.

    This strategy identifies tools that can be swapped for better alternatives:
    - Expensive tools with cheaper alternatives
    - Slow tools with faster alternatives
    - Unreliable tools with more stable alternatives
    """

    # Tool alternatives database (simplified for MVP)
    TOOL_ALTERNATIVES = {
        "claude_opus": [
            "claude_sonnet",  # Cheaper, still high quality
            "gpt_4o",  # Comparable quality, different pricing
        ],
        "gpt_4_turbo": [
            "gpt_4o_mini",  # Much cheaper
            "claude_haiku",  # Fast and cheap
        ],
        "web_search": [
            "cached_search",  # Use cached results when possible
            "local_search",  # Search local index first
        ],
    }

    def can_apply(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> bool:
        """Tool selection applies to expensive tools."""
        return bottleneck.type == BottleneckType.EXPENSIVE_TOOL

    def generate_opportunity(
        self,
        bottleneck: Bottleneck,
        profile: WorkflowProfile,
    ) -> Optional[OptimizationOpportunity]:
        """Generate tool selection opportunity."""
        if not bottleneck.tool_id:
            return None

        # Find alternatives
        alternatives = self.TOOL_ALTERNATIVES.get(bottleneck.tool_id, [])

        if not alternatives:
            return None

        # Use first alternative (simplified - should rank by cost/performance)
        best_alternative = alternatives[0]

        # Estimate improvement
        cost_reduction = 0.5  # Assume 50% cost reduction

        return OptimizationOpportunity(
            strategy_type=OptimizationStrategyType.TOOL_SELECTION,
            target=f"{bottleneck.tool_id} -> {best_alternative}",
            description=(
                f"Replace '{bottleneck.tool_id}' with '{best_alternative}' "
                f"(expected {cost_reduction:.0%} cost reduction)"
            ),
            expected_improvement=cost_reduction,
            risk_level=BottleneckSeverity.LOW,
            estimated_cost_reduction=cost_reduction,
            confidence=0.7,
            metadata={
                "current_tool": bottleneck.tool_id,
                "suggested_tool": best_alternative,
                "alternatives": alternatives,
            },
        )

    @property
    def risk_level(self) -> BottleneckSeverity:
        return BottleneckSeverity.LOW


# Strategy factory
def create_strategy(
    strategy_type: OptimizationStrategyType,
) -> BaseOptimizationStrategy:
    """Create strategy instance by type.

    Args:
        strategy_type: Type of strategy to create

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy type is not supported
    """
    strategies = {
        OptimizationStrategyType.PRUNING: PruningStrategy,
        OptimizationStrategyType.PARALLELIZATION: ParallelizationStrategy,
        OptimizationStrategyType.TOOL_SELECTION: ToolSelectionStrategy,
    }

    strategy_class = strategies.get(strategy_type)

    if not strategy_class:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

    return strategy_class()
