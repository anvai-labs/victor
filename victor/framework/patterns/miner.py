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

"""Pattern mining implementation.

This module provides the PatternMiner class that discovers collaboration
patterns from workflow execution traces.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from victor.framework.patterns.types import (
    CollaborationPattern,
    PatternCategory,
    PatternMetrics,
    PatternStatus,
    WorkflowExecutionTrace,
)

logger = logging.getLogger(__name__)


class PatternMiner:
    """Discovers collaboration patterns from execution traces.

    Implements PatternMinerProtocol for pattern discovery.

    Example:
        miner = PatternMiner(min_occurrences=3)
        traces = [...]  # List of execution traces
        patterns = await miner.mine_from_traces(traces)
        for pattern in patterns:
            print(f"Found: {pattern.name} (success rate: {pattern.success_rate:.1%})")
    """

    def __init__(
        self,
        min_occurrences: int = 2,
        min_success_rate: float = 0.5,
    ):
        """Initialize pattern miner.

        Args:
            min_occurrences: Minimum occurrences for pattern discovery
            min_success_rate: Minimum success rate for pattern
        """
        self.min_occurrences = min_occurrences
        self.min_success_rate = min_success_rate

    async def mine_from_traces(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Extract patterns from execution traces.

        Args:
            traces: List of workflow execution traces

        Returns:
            List of discovered collaboration patterns
        """
        if not traces:
            return []

        logger.info(f"Mining patterns from {len(traces)} traces")

        # Group traces by similar structure
        structure_groups = self._group_by_structure(traces)

        patterns = []
        for structure_key, group_traces in structure_groups.items():
            if len(group_traces) < self.min_occurrences:
                continue

            # Analyze the group
            pattern = self._create_pattern_from_group(structure_key, group_traces)

            if pattern and pattern.success_rate >= self.min_success_rate:
                patterns.append(pattern)

        logger.info(f"Discovered {len(patterns)} patterns")
        return patterns

    async def analyze_execution_order(
        self,
        trace: WorkflowExecutionTrace,
    ) -> Dict[str, Any]:
        """Analyze execution order patterns.

        Args:
            trace: Single execution trace

        Returns:
            Dictionary with execution pattern analysis
        """
        analysis = {
            "workflow_id": trace.workflow_id,
            "node_count": len(trace.nodes_executed),
            "execution_sequence": trace.execution_order,
            "category": self._detect_category(trace),
        }

        # Detect parallelism
        parallel_groups = self._detect_parallel_groups(trace)
        analysis["parallel_groups"] = parallel_groups
        analysis["has_parallelism"] = len(parallel_groups) > 0

        # Detect hierarchical patterns
        analysis["has_hierarchy"] = self._detect_hierarchy(trace)

        return analysis

    async def detect_formations(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> Dict[PatternCategory, int]:
        """Detect pattern categories from traces.

        Args:
            traces: List of execution traces

        Returns:
            Dictionary mapping category to frequency
        """
        category_counts: Dict[PatternCategory, int] = defaultdict(int)

        for trace in traces:
            category = self._detect_category(trace)
            category_counts[category] += 1

        return dict(category_counts)

    def _group_by_structure(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> Dict[str, List[WorkflowExecutionTrace]]:
        """Group traces by similar execution structure.

        Args:
            traces: List of traces to group

        Returns:
            Dictionary mapping structure key to traces
        """
        groups: Dict[str, List[WorkflowExecutionTrace]] = defaultdict(list)

        for trace in traces:
            # Create structure key from execution order
            key = self._create_structure_key(trace)
            groups[key].append(trace)

        return dict(groups)

    def _create_structure_key(self, trace: WorkflowExecutionTrace) -> str:
        """Create a key representing the execution structure.

        Args:
            trace: Execution trace

        Returns:
            Structure key string
        """
        # Use sorted nodes for consistent grouping
        nodes = sorted(trace.nodes_executed)

        # Include execution order pattern
        order_pattern = self._get_order_pattern(trace.execution_order)

        return f"{'-'.join(nodes)}|{order_pattern}"

    def _get_order_pattern(self, execution_order: List[str]) -> str:
        """Extract execution order pattern.

        Args:
            execution_order: List of node IDs in execution order

        Returns:
            Pattern string (e.g., "sequential", "parallel:2", etc.)
        """
        if not execution_order:
            return "empty"

        # Check for sequential pattern
        if len(execution_order) == len(set(execution_order)):
            # Each node appears once - simple sequential
            return "sequential"

        # Detect parallel groups
        seen = set()
        parallel_count = 0
        for node in execution_order:
            if node in seen:
                parallel_count += 1
            seen.add(node)

        if parallel_count > 0:
            return f"parallel:{parallel_count}"

        return "mixed"

    def _create_pattern_from_group(
        self,
        structure_key: str,
        traces: List[WorkflowExecutionTrace],
    ) -> Optional[CollaborationPattern]:
        """Create a pattern from a group of similar traces.

        Args:
            structure_key: Structure key for the group
            traces: List of traces in the group

        Returns:
            CollaborationPattern or None if not valid
        """
        if not traces:
            return None

        # Calculate success count (rate computed by PatternMetrics)
        success_count = sum(1 for t in traces if t.success)

        # Get representative trace
        representative = traces[0]

        # Determine category
        category = self._detect_category(representative)

        # Build participant specs from traces
        participants = self._extract_participants(traces)

        # Calculate metrics
        avg_duration = sum(t.duration_ms for t in traces) / len(traces)
        avg_cost = sum(t.cost for t in traces) / len(traces)

        metrics = PatternMetrics(
            usage_count=len(traces),
            success_count=success_count,
            avg_duration_ms=avg_duration,
            avg_cost=avg_cost,
        )

        # Create pattern name
        name = self._generate_pattern_name(category, participants)

        return CollaborationPattern(
            name=name,
            description=f"Pattern discovered from {len(traces)} executions",
            category=category,
            participants=participants,
            workflow={
                "structure_key": structure_key,
                "execution_order": representative.execution_order,
                "nodes": representative.nodes_executed,
            },
            status=PatternStatus.DISCOVERED,
            metrics=metrics,
        )

    def _detect_category(self, trace: WorkflowExecutionTrace) -> PatternCategory:
        """Detect the pattern category from a trace.

        Args:
            trace: Execution trace

        Returns:
            PatternCategory
        """
        # Check for parallel execution
        parallel_groups = self._detect_parallel_groups(trace)
        if parallel_groups:
            if self._detect_hierarchy(trace):
                return PatternCategory.MIXED
            return PatternCategory.PARALLEL

        # Check for hierarchy
        if self._detect_hierarchy(trace):
            return PatternCategory.HIERARCHICAL

        # Default to sequential
        return PatternCategory.SEQUENTIAL

    def _detect_parallel_groups(self, trace: WorkflowExecutionTrace) -> List[List[str]]:
        """Detect groups of parallel nodes.

        Args:
            trace: Execution trace

        Returns:
            List of parallel node groups
        """
        # Simple heuristic: nodes that appear close together might be parallel
        # This would be more sophisticated with actual timing data
        parallel_groups = []

        metadata = trace.metadata
        if "parallel_nodes" in metadata:
            return metadata["parallel_nodes"]

        # Heuristic: look for common prefixes suggesting parallel execution
        current_group = []
        for node in trace.execution_order:
            if "_parallel_" in node or node.startswith("parallel_"):
                current_group.append(node)
            else:
                if len(current_group) >= 2:
                    parallel_groups.append(current_group)
                current_group = []

        if len(current_group) >= 2:
            parallel_groups.append(current_group)

        return parallel_groups

    def _detect_hierarchy(self, trace: WorkflowExecutionTrace) -> bool:
        """Detect if trace shows hierarchical pattern.

        Args:
            trace: Execution trace

        Returns:
            True if hierarchical pattern detected
        """
        # Check for supervisor/specialist naming, with legacy manager/worker
        # trace labels accepted for historical executions.
        has_supervisor = any(
            "supervisor" in n.lower() or "manager" in n.lower() or "coordinator" in n.lower()
            for n in trace.nodes_executed
        )
        has_specialist = any(
            "specialist" in n.lower() or "worker" in n.lower() or "agent" in n.lower()
            for n in trace.nodes_executed
        )

        return has_supervisor and has_specialist

    def _extract_participants(self, traces: List[WorkflowExecutionTrace]) -> List[Dict[str, Any]]:
        """Extract participant specs from traces.

        Args:
            traces: List of traces

        Returns:
            List of participant specifications
        """
        # Collect all unique nodes across traces
        all_nodes: Dict[str, int] = defaultdict(int)
        for trace in traces:
            for node in trace.nodes_executed:
                all_nodes[node] += 1

        # Create participant specs
        participants = []
        for node, count in all_nodes.items():
            role = "executor"
            if (
                "supervisor" in node.lower()
                or "manager" in node.lower()
                or "coordinator" in node.lower()
            ):
                role = "supervisor"
            elif "reviewer" in node.lower():
                role = "reviewer"
            elif "planner" in node.lower():
                role = "planner"

            participants.append(
                {
                    "id": node,
                    "role": role,
                    "occurrence_rate": count / len(traces),
                }
            )

        return participants

    def _generate_pattern_name(
        self,
        category: PatternCategory,
        participants: List[Dict[str, Any]],
    ) -> str:
        """Generate a descriptive pattern name.

        Args:
            category: Pattern category
            participants: Participant list

        Returns:
            Pattern name
        """
        # Count roles
        role_counts = defaultdict(int)
        for p in participants:
            role_counts[p["role"]] += 1

        # Build name parts
        parts = [category.value.title()]

        if role_counts["supervisor"] > 0:
            parts.append(f"{role_counts['supervisor']}S")
        if role_counts["executor"] > 0:
            parts.append(f"{role_counts['executor']}E")
        if role_counts["reviewer"] > 0:
            parts.append(f"{role_counts['reviewer']}R")

        return " ".join(parts)


__all__ = [
    "PatternMiner",
]
