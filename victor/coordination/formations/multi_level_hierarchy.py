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

"""Multi-level hierarchy formation for divide-and-conquer coordination.

This module provides MultiLevelHierarchyFormation, which implements
hierarchical agent coordination with multiple levels.

Formation Pattern:
    Level 0: Coordinator
    Level 1: Team Leads (3-5)
    Level 2: Team Members (each under a lead)

Implements divide-and-conquer pattern for large tasks.

SOLID Principles:
- SRP: Hierarchy logic only
- OCP: Extensible via depth and branching factor
- LSP: Substitutable with other formations
- DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.coordination.formations.multi_level_hierarchy import (
        MultiLevelHierarchyFormation,
        HierarchyNode,
    )
    from victor.coordination.formations.base import TeamContext

    # Build hierarchy tree
    leaf1 = HierarchyNode(agent=worker_agent_1)
    leaf2 = HierarchyNode(agent=worker_agent_2)
    lead = HierarchyNode(agent=lead_agent, children=[leaf1, leaf2])
    coordinator = HierarchyNode(agent=coordinator_agent, children=[lead])

    # Create formation with hierarchy
    formation = MultiLevelHierarchyFormation(hierarchy=coordinator)

    # Execute
    results = await formation.execute(agents, context, task)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """Node in agent hierarchy tree.

    Represents an agent with optional children (subordinates).
    Forms a tree structure for hierarchical coordination.

    Attributes:
        agent: The agent for this node
        children: List of child nodes (subordinates)
        parent: Parent node (None for root)
        level: Depth level in hierarchy (0 for root)

    Example:
        >>> leaf1 = HierarchyNode(agent=worker1)
        >>> leaf2 = HierarchyNode(agent=worker2)
        >>> lead = HierarchyNode(agent=team_lead, children=[leaf1, leaf2])
        >>> coordinator = HierarchyNode(agent=coordinator, children=[lead])
    """

    agent: Any
    children: List["HierarchyNode"] = field(default_factory=list)
    parent: Optional["HierarchyNode"] = None
    level: int = 0

    def __post_init__(self):
        """Set parent references and level for children."""
        for child in self.children:
            child.parent = self
            child.level = self.level + 1

    def get_depth(self) -> int:
        """Get the depth of this hierarchy tree.

        Returns:
            Maximum depth from this node (1 if leaf)
        """
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)


class MultiLevelHierarchyFormation(BaseFormationStrategy):
    """Multi-level hierarchical coordination for divide-and-conquer.

    Formation Pattern:
        Level 0: Coordinator
            ├── Level 1: Team Lead 1
            │   ├── Level 2: Member 1
            │   └── Level 2: Member 2
            └── Level 1: Team Lead 2
                ├── Level 2: Member 3
                └── Level 2: Member 4

    Implements divide-and-conquer pattern:
    1. Coordinator receives task
    2. Task split among team leads
    3. Leads delegate to members
    4. Results aggregated up the hierarchy

    SOLID: SRP (hierarchy logic only), OCP (extensible depth)

    Attributes:
        hierarchy: Root hierarchy node
        max_depth: Maximum hierarchy depth to execute
        split_strategy: Strategy for splitting tasks (line/count/auto)

    Example:
        >>> # Build 3-level hierarchy
        >>> leaf1 = HierarchyNode(agent=worker1)
        >>> leaf2 = HierarchyNode(agent=worker2)
        >>> lead = HierarchyNode(agent=team_lead, children=[leaf1, leaf2])
        >>> root = HierarchyNode(agent=coordinator, children=[lead])
        >>>
        >>> formation = MultiLevelHierarchyFormation(hierarchy=root)
        >>> results = await formation.execute(agents, context, task)
        >>>
        >>> print(f"Depth: {results[0].metadata['hierarchy_levels']}")
    """

    def __init__(
        self,
        hierarchy: HierarchyNode,
        max_depth: int = 3,
        split_strategy: str = "auto",
    ):
        """Initialize the multi-level hierarchy formation.

        Args:
            hierarchy: Root hierarchy node containing entire tree
            max_depth: Maximum depth to execute (default: 3)
            split_strategy: Strategy for splitting tasks
                - "line": Split by lines
                - "count": Split by count
                - "auto": Automatic based on task size
        """
        self.hierarchy = hierarchy
        self.max_depth = max_depth
        self.split_strategy = split_strategy

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task using hierarchical divide-and-conquer.

        Args:
            agents: List of agents (not used, hierarchy defines agents)
            context: Team context
            task: Task message to process

        Returns:
            List of MemberResult with metadata:
                - result: Aggregated result from hierarchy
                - hierarchy_levels: Number of levels in hierarchy
                - nodes_executed: Total nodes executed
        """
        try:
            # Execute from root of hierarchy
            result = await self._execute_node(self.hierarchy, task, context)

            return [
                MemberResult(
                    member_id="multi_level_hierarchy",
                    success=True,
                    output=str(result) if result else "",
                    metadata={
                        "hierarchy_levels": self.hierarchy.get_depth(),
                        "nodes_executed": self._count_nodes(self.hierarchy),
                        "formation": "multi_level_hierarchy",
                    },
                )
            ]
        except Exception as e:
            logger.error(f"MultiLevelHierarchyFormation failed: {e}")
            return [
                MemberResult(
                    member_id="multi_level_hierarchy",
                    success=False,
                    output="",
                    error=str(e),
                )
            ]

    async def _execute_node(
        self, node: HierarchyNode, task: AgentMessage, context: TeamContext
    ) -> Any:
        """Execute a hierarchy node (recursive).

        Args:
            node: Hierarchy node to execute
            task: Task message for this node
            context: Team context

        Returns:
            Result from node execution
        """
        # If leaf node (no children), execute directly
        if not node.children:
            logger.debug(f"Executing leaf node: {node.agent.id}")
            return await node.agent.execute(task.content, context=context.shared_state)

        # If internal node, delegate to children
        logger.debug(f"Executing internal node: {node.agent.id} with {len(node.children)} children")

        # Split task for children
        subtasks = self._split_task(task.content, len(node.children))

        # Execute children in parallel (or sequentially)
        child_results = []
        for child, subtask in zip(node.children, subtasks):
            from victor.teams.types import MessageType

            child_task = AgentMessage(
                sender_id="hierarchy",
                content=subtask,
                message_type=MessageType.TASK,
            )
            result = await self._execute_node(child, child_task, context)
            child_results.append(result)

        # Aggregate results from children
        return await self._aggregate_results(child_results)

    def _split_task(self, task: str, num_parts: int) -> List[str]:
        """Split task into subtasks for children.

        Args:
            task: Task description to split
            num_parts: Number of parts to split into

        Returns:
            List of subtask strings
        """
        if num_parts <= 1:
            return [task]

        # Strategy: line-based splitting
        if self.split_strategy == "line" or (self.split_strategy == "auto" and len(task) > 500):
            lines = task.split("\n")
            chunk_size = max(1, len(lines) // num_parts)
            subtasks = []

            for i in range(0, len(lines), chunk_size):
                chunk = "\n".join(lines[i : i + chunk_size])
                subtasks.append(chunk)

            return subtasks[:num_parts]

        # Strategy: count-based splitting
        elif self.split_strategy == "count":
            char_size = max(1, len(task) // num_parts)
            subtasks = []

            for i in range(0, len(task), char_size):
                chunk = task[i : i + char_size]
                subtasks.append(chunk)

            return subtasks[:num_parts]

        # Auto: simple equal splitting
        else:
            part_size = len(task) // num_parts
            return [task[i : i + part_size] for i in range(0, len(task), part_size)][:num_parts]

    async def _aggregate_results(self, results: List[Any]) -> Any:
        """Aggregate results from child nodes.

        Args:
            results: List of results from children

        Returns:
            Aggregated result
        """
        # Simple aggregation: concatenate with separators
        aggregated = []
        for result in results:
            if isinstance(result, str):
                aggregated.append(result)
            else:
                aggregated.append(str(result))

        return "\n\n".join(aggregated)

    def _count_nodes(self, node: HierarchyNode) -> int:
        """Count total nodes in hierarchy tree.

        Args:
            node: Root node to count from

        Returns:
            Total number of nodes
        """
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _validate_node(self, node: HierarchyNode) -> bool:
        """Validate hierarchy node and its children.

        Args:
            node: Node to validate

        Returns:
            True if node and all children are valid
        """
        if not node.agent:
            logger.error("Hierarchy node missing agent")
            return False

        for child in node.children:
            if not self._validate_node(child):
                return False

        return True

    def validate_context(self, context: TeamContext) -> bool:
        """Validate hierarchy structure.

        Args:
            context: Team context (not used for hierarchy validation)

        Returns:
            True if hierarchy structure is valid
        """
        return self._validate_node(self.hierarchy)

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Multi-level hierarchy doesn't support early termination
        as it needs to complete the full tree.

        Returns:
            False (no early termination)
        """
        return False


__all__ = ["MultiLevelHierarchyFormation", "HierarchyNode"]
