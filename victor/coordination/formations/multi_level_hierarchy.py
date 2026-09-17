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
    Level 0: Supervisor
    Level 1: Supervisors
    Level 2: Members

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
import json
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
        Level 0: Supervisor
            ├── Level 1: Supervisor 1
            │   ├── Level 2: Member 1
            │   └── Level 2: Member 2
            └── Level 1: Supervisor 2
                ├── Level 2: Member 3
                └── Level 2: Member 4

    Implements divide-and-conquer pattern:
    1. Root supervisor receives task
    2. Task split among child supervisors
    3. Supervisors delegate to members
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
        hierarchy: Optional[HierarchyNode] = None,
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
        if max_depth < 1 or split_strategy not in {"line", "count", "auto"}:
            raise ValueError("Invalid hierarchy depth or split strategy")
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
        if agents:
            return await self._execute_participants(agents, context, task)
        if self.hierarchy is None:
            return [
                MemberResult(
                    member_id="hierarchy", success=False, output="", error="No members available"
                )
            ]
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

    async def _execute_participants(self, agents, context, task):
        """Execute a validated member-ID tree, retaining each member's outcome."""
        from victor.teams.types import MessageType

        by_id = {agent.id: agent for agent in agents}
        tree = context.get("hierarchy")
        max_depth = context.get("hierarchy_max_depth", self.max_depth)
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("hierarchy_max_depth must be a positive integer")
        if tree is None:
            # Stable binary tree in member order. Explicit trees are recommended
            # when supervisor placement matters.
            nodes = [{"member_id": agent.id, "children": []} for agent in agents]
            for index in range(1, len(nodes)):
                nodes[(index - 1) // 2]["children"].append(nodes[index])
            tree = nodes[0]
        seen = set()

        def validate(node, depth):
            if not isinstance(node, dict) or set(node) - {"member_id", "children"}:
                raise ValueError("Hierarchy nodes require member_id and optional children")
            member_id = node.get("member_id")
            if not isinstance(member_id, str) or member_id not in by_id or member_id in seen:
                raise ValueError("Hierarchy contains an unknown or duplicate member ID")
            if depth > max_depth:
                raise ValueError("Hierarchy exceeds hierarchy_max_depth")
            seen.add(member_id)
            children = node.get("children", [])
            if not isinstance(children, list):
                raise ValueError("Hierarchy children must be a list")
            for child in children:
                validate(child, depth + 1)

        validate(tree, 1)
        if seen != set(by_id):
            raise ValueError("Hierarchy must include every member exactly once")
        splitter = MultiLevelHierarchyFormation(
            split_strategy=context.get("hierarchy_split_strategy", self.split_strategy)
        )
        results = []

        async def visit(node, objective, depth):
            children = node.get("children", [])
            findings = []
            for child, subtask in zip(children, splitter._split_task(objective, len(children))):
                findings.append(await visit(child, subtask, depth + 1))
            prompt = (
                f"Objective: {objective}\n"
                "Output: condensed findings or file references. Use only assigned tools and sources.\n"
                "Boundary: handle only this assigned portion; do not duplicate other members' work."
            )
            if children:
                prompt += (
                    "\nSynthesize these structured member outcomes; preserve failures and do not redo their tasks:\n"
                    + json.dumps(findings)
                )
            result = await by_id[node["member_id"]].execute(
                AgentMessage(
                    sender_id=task.sender_id,
                    content=prompt,
                    message_type=MessageType.TASK,
                    data=task.data,
                ),
                context,
            )
            result.metadata.update(
                formation="multi_level_hierarchy",
                hierarchy_level=depth,
                hierarchy_root=node is tree,
            )
            results.append(result)
            return {
                "member_id": result.member_id,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }

        await visit(tree, task.content, 1)
        return results

    def supports_durable_pause(self) -> bool:
        """False: the recursive tree cursor is not persisted; approval stays inline."""
        return False

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
        # Balanced partitions preserve the remainder and handle tasks shorter than
        # the member count. Line endings stay attached to their original lines.
        units = (
            task.splitlines(keepends=True)
            if self.split_strategy == "line" or (self.split_strategy == "auto" and "\n" in task)
            else list(task)
        )
        size, remainder = divmod(len(units), num_parts)
        chunks = []
        offset = 0
        for index in range(num_parts):
            end = offset + size + (index < remainder)
            chunks.append("".join(units[offset:end]))
            offset = end
        return chunks

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
        return self.hierarchy is not None and self._validate_node(self.hierarchy)

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Multi-level hierarchy doesn't support early termination
        as it needs to complete the full tree.

        Returns:
            False (no early termination)
        """
        return False


__all__ = ["MultiLevelHierarchyFormation", "HierarchyNode"]
