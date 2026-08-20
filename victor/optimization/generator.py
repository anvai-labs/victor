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

"""Workflow variant generation for optimization.

This module provides the WorkflowVariantGenerator class for creating
optimized workflow variants based on optimization opportunities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy

from victor.optimization.models import (
    OptimizationOpportunity,
    OptimizationStrategyType,
    WorkflowProfile,
)
from victor.optimization.strategies import (
    WorkflowChange,
    create_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowVariant:
    """Represents an optimized workflow variant.

    Attributes:
        variant_id: Unique identifier for this variant
        base_workflow_id: Original workflow ID
        changes: List of changes applied to create this variant
        expected_improvement: Expected improvement from applying changes
        risk_level: Overall risk level
        estimated_cost_reduction: Expected cost reduction in USD
        estimated_duration_reduction: Expected duration reduction in seconds
        metadata: Additional metadata
        config: Modified workflow configuration
    """

    variant_id: str
    base_workflow_id: str
    changes: List[WorkflowChange]
    expected_improvement: float
    risk_level: str
    estimated_cost_reduction: float = 0.0
    estimated_duration_reduction: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "variant_id": self.variant_id,
            "base_workflow_id": self.base_workflow_id,
            "changes": [
                {
                    "type": c.change_type,
                    "target": c.target,
                    "description": c.description,
                    "metadata": c.metadata,
                }
                for c in self.changes
            ],
            "expected_improvement": self.expected_improvement,
            "risk_level": self.risk_level,
            "estimated_cost_reduction": self.estimated_cost_reduction,
            "estimated_duration_reduction": self.estimated_duration_reduction,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"WorkflowVariant(id={self.variant_id}, "
            f"changes={len(self.changes)}, "
            f"improvement={self.expected_improvement:.1%}, "
            f"risk={self.risk_level})"
        )


class WorkflowVariantGenerator:
    """Generates optimized workflow variants.

    The generator applies optimization strategies to create workflow variants
    with improved performance characteristics.

    Example:
        generator = WorkflowVariantGenerator()

        variant = await generator.generate_variant(
            workflow_config=config,
            opportunity=opportunity,
            profile=profile,
        )

        # Validate variant
        is_valid = await generator.validate_variant(variant)
    """

    def __init__(
        self,
        validate_structure: bool = True,
        max_variants_per_opportunity: int = 3,
    ):
        """Initialize the variant generator.

        Args:
            validate_structure: Whether to validate variant structure
            max_variants_per_opportunity: Maximum variants to generate per opportunity
        """
        self.validate_structure = validate_structure
        self.max_variants_per_opportunity = max_variants_per_opportunity
        self._variant_counter = 0

    async def generate_variant(
        self,
        workflow_config: Dict[str, Any],
        opportunity: OptimizationOpportunity,
        profile: WorkflowProfile,
    ) -> Optional[WorkflowVariant]:
        """Generate an optimized workflow variant.

        Args:
            workflow_config: Original workflow configuration
            opportunity: Optimization opportunity to apply
            profile: Workflow performance profile

        Returns:
            WorkflowVariant if successful, None otherwise
        """
        logger.info(
            f"Generating variant for opportunity: {opportunity.strategy_type.value} "
            f"on {opportunity.target}"
        )

        # Deep copy config to avoid modifying original
        variant_config = deepcopy(workflow_config)

        # Apply optimization based on strategy type
        changes = []
        success = False

        if opportunity.strategy_type == OptimizationStrategyType.PRUNING:
            success, changes = await self._apply_pruning(
                variant_config,
                opportunity,
                profile,
            )

        elif opportunity.strategy_type == OptimizationStrategyType.PARALLELIZATION:
            success, changes = await self._apply_parallelization(
                variant_config,
                opportunity,
                profile,
            )

        elif opportunity.strategy_type == OptimizationStrategyType.TOOL_SELECTION:
            success, changes = await self._apply_tool_selection(
                variant_config,
                opportunity,
                profile,
            )

        if not success:
            logger.warning(f"Failed to apply optimization: {opportunity.strategy_type.value}")
            return None

        # Create variant
        self._variant_counter += 1
        variant_id = f"{profile.workflow_id}_variant_{self._variant_counter}"

        variant = WorkflowVariant(
            variant_id=variant_id,
            base_workflow_id=profile.workflow_id,
            changes=changes,
            expected_improvement=opportunity.expected_improvement,
            risk_level=opportunity.risk_level.value,
            estimated_cost_reduction=opportunity.estimated_cost_reduction,
            estimated_duration_reduction=opportunity.estimated_duration_reduction,
            metadata={
                "strategy_type": opportunity.strategy_type.value,
                "target": opportunity.target,
                "confidence": opportunity.confidence,
            },
            config=variant_config,
        )

        # Validate variant structure
        if self.validate_structure:
            is_valid = await self.validate_variant(variant, profile)

            if not is_valid:
                logger.warning(f"Variant {variant_id} failed validation")
                return None

        logger.info(f"Generated variant: {variant_id}")
        return variant

    async def _apply_pruning(
        self,
        config: Dict[str, Any],
        opportunity: OptimizationOpportunity,
        profile: WorkflowProfile,
    ) -> tuple[bool, List[WorkflowChange]]:
        """Apply pruning strategy to remove nodes.

        Args:
            config: Workflow configuration
            opportunity: Optimization opportunity
            profile: Workflow profile

        Returns:
            Tuple of (success, changes)
        """
        changes = []

        # Parse target node IDs
        target_nodes = opportunity.target.split(",")

        for node_id in target_nodes:
            node_id = node_id.strip()

            # Check if node exists in config
            if "nodes" in config and node_id in config["nodes"]:
                # Remove node
                removed_node = config["nodes"].pop(node_id)

                # Update edges to skip removed node
                if "edges" in config:
                    # Remove edges involving this node
                    config["edges"] = [
                        edge
                        for edge in config["edges"]
                        if edge.get("source") != node_id and edge.get("target") != node_id
                    ]

                changes.append(
                    WorkflowChange(
                        change_type="remove_node",
                        target=node_id,
                        description=f"Removed node '{node_id}' (success rate: {profile.node_stats[node_id].success_rate:.1%})",
                        metadata={
                            "removed_node": removed_node,
                            "reason": "pruning",
                        },
                    )
                )

                logger.info(f"Pruned node: {node_id}")

        return len(changes) > 0, changes

    async def _apply_parallelization(
        self,
        config: Dict[str, Any],
        opportunity: OptimizationOpportunity,
        profile: WorkflowProfile,
    ) -> tuple[bool, List[WorkflowChange]]:
        """Apply parallelization strategy to group nodes.

        Args:
            config: Workflow configuration
            opportunity: Optimization opportunity
            profile: Workflow profile

        Returns:
            Tuple of (success, changes)
        """
        changes = []

        # Parse target node IDs
        target_nodes = [n.strip() for n in opportunity.target.split(",")]

        if len(target_nodes) < 2:
            return False, changes

        # Create parallel node
        parallel_node_id = f"parallel_{'_'.join(target_nodes[:2])}"

        parallel_config = {
            "type": "parallel",
            "node_ids": target_nodes,
            "join_strategy": "all_complete",
            "error_strategy": "fail_fast",
        }

        # Add parallel node to config
        if "nodes" not in config:
            config["nodes"] = {}

        config["nodes"][parallel_node_id] = parallel_config

        # Update edges to route through parallel node
        # (Simplified - full implementation would handle complex edge routing)
        if "edges" in config:
            # Find edges to/from target nodes
            incoming_edges = [
                edge for edge in config["edges"] if edge.get("target") in target_nodes
            ]
            outgoing_edges = [
                edge for edge in config["edges"] if edge.get("source") in target_nodes
            ]

            # Remove old edges
            config["edges"] = [
                edge
                for edge in config["edges"]
                if edge not in incoming_edges and edge not in outgoing_edges
            ]

            # Add new edges through parallel node
            for edge in incoming_edges:
                config["edges"].append(
                    {
                        "source": edge["source"],
                        "target": parallel_node_id,
                    }
                )

            for edge in outgoing_edges:
                config["edges"].append(
                    {
                        "source": parallel_node_id,
                        "target": edge["target"],
                    }
                )

        changes.append(
            WorkflowChange(
                change_type="create_parallel_node",
                target=",".join(target_nodes),
                description=f"Created parallel node '{parallel_node_id}' with {len(target_nodes)} nodes",
                metadata={
                    "parallel_node_id": parallel_node_id,
                    "node_ids": target_nodes,
                    "estimated_speedup": opportunity.metadata.get("estimated_speedup", 2.0),
                },
            )
        )

        logger.info(f"Created parallel node: {parallel_node_id}")

        return True, changes

    async def _apply_tool_selection(
        self,
        config: Dict[str, Any],
        opportunity: OptimizationOpportunity,
        profile: WorkflowProfile,
    ) -> tuple[bool, List[WorkflowChange]]:
        """Apply tool selection strategy to swap tools.

        Args:
            config: Workflow configuration
            opportunity: Optimization opportunity
            profile: Workflow profile

        Returns:
            Tuple of (success, changes)
        """
        changes = []

        # Parse tool mapping (e.g., "tool_a -> tool_b")
        if " -> " not in opportunity.target:
            return False, changes

        current_tool, new_tool = opportunity.target.split(" -> ", 1)
        current_tool = current_tool.strip()
        new_tool = new_tool.strip()

        # Find nodes using the current tool and update them
        # This is a simplified implementation
        if "nodes" in config:
            for node_id, node_config in config["nodes"].items():
                if isinstance(node_config, dict):
                    # Check if node uses the tool
                    if node_config.get("tool") == current_tool:
                        # Update tool
                        old_tool = node_config.get("tool")
                        node_config["tool"] = new_tool

                        changes.append(
                            WorkflowChange(
                                change_type="substitute_tool",
                                target=node_id,
                                description=f"Substituted tool '{old_tool}' with '{new_tool}' in node '{node_id}'",
                                metadata={
                                    "old_tool": old_tool,
                                    "new_tool": new_tool,
                                    "node_id": node_id,
                                },
                            )
                        )

                        logger.info(f"Substituted tool in node {node_id}: {old_tool} -> {new_tool}")

        return len(changes) > 0, changes

    async def validate_variant(
        self,
        variant: WorkflowVariant,
        profile: WorkflowProfile,
    ) -> bool:
        """Validate that variant is well-formed.

        Args:
            variant: Workflow variant to validate
            profile: Original workflow profile

        Returns:
            True if valid, False otherwise
        """
        logger.info(f"Validating variant: {variant.variant_id}")

        config = variant.config

        # Check 1: All nodes have required fields
        if "nodes" in config:
            for node_id, node_config in config["nodes"].items():
                if not isinstance(node_config, dict):
                    logger.warning(f"Node {node_id} config is not a dict")
                    return False

                # Check for required fields based on node type
                node_type = node_config.get("type", "agent")
                if node_type == "agent":
                    if "goal" not in node_config and "prompt" not in node_config:
                        logger.warning(f"Agent node {node_id} missing goal/prompt")
                        return False

        # Check 2: No orphan nodes (all nodes reachable from start)
        if "edges" in config and "nodes" in config:
            reachable_nodes = self._find_reachable_nodes(config)
            all_nodes = set(config["nodes"].keys())

            orphan_nodes = all_nodes - reachable_nodes
            if orphan_nodes:
                logger.warning(f"Found orphan nodes: {orphan_nodes}")
                # This is a warning, not necessarily invalid
                # Some workflows may have intentional disconnected components

        # Check 3: No cycles (unless max_iterations is set)
        has_cycle = self._detect_cycles(config)
        if has_cycle:
            # Check if max_iterations is set to handle cycles
            has_max_iter = any(
                node.get("max_iterations", 0) > 0 for node in config.get("nodes", {}).values()
            )

            if not has_max_iter:
                logger.warning("Detected cycle without max_iterations")
                return False

        # Check 4: At least one node exists
        if not config.get("nodes"):
            logger.warning("Variant has no nodes")
            return False

        logger.info(f"Variant {variant.variant_id} is valid")
        return True

    def _find_reachable_nodes(
        self,
        config: Dict[str, Any],
    ) -> Set[str]:
        """Find all nodes reachable from start nodes.

        Args:
            config: Workflow configuration

        Returns:
            Set of reachable node IDs
        """
        reachable = set()
        to_visit = set()

        # Start from nodes with no incoming edges
        if "edges" in config and "nodes" in config:
            all_nodes = set(config["nodes"].keys())
            targets = {edge.get("target") for edge in config["edges"]}

            start_nodes = all_nodes - targets
            to_visit.update(start_nodes)

            # BFS traversal
            while to_visit:
                node = to_visit.pop()
                reachable.add(node)

                # Find outgoing edges
                for edge in config.get("edges", []):
                    if edge.get("source") == node:
                        target = edge.get("target")
                        if target and target not in reachable:
                            to_visit.add(target)

        return reachable

    def _detect_cycles(
        self,
        config: Dict[str, Any],
    ) -> bool:
        """Detect if workflow contains cycles.

        Args:
            config: Workflow configuration

        Returns:
            True if cycle detected, False otherwise
        """
        if "edges" not in config or "nodes" not in config:
            return False

        # Build adjacency list
        adj = {node_id: [] for node_id in config["nodes"].keys()}

        for edge in config["edges"]:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adj[source].append(target)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in adj:
            if node not in visited:
                if dfs(node):
                    return True

        return False
