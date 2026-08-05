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

"""Unified tool execution graph for intelligent tool selection.

This module provides a unified graph abstraction that supports multiple paradigms:
- Dependency-based (coding format): ToolDependency with depends_on/enables
- Transition-based (devops format): Dict[str, List[Tuple[str, float]]]
- Sequence-based: Common tool sequences for workflow patterns
- IO-based (planning format): Input/output tool relationships

The graph enables intelligent tool selection by:
- Suggesting next tools based on transition probabilities
- Validating prerequisites before tool execution
- Planning tool sequences to achieve goals
- Detecting workflow patterns

Example usage:
    from victor.tools.tool_graph import (
        ToolExecutionGraph,
        ToolNode,
    )

    # Create graph
    graph = ToolExecutionGraph()

    # Add dependencies (coding format)
    graph.add_dependency(
        tool_name="edit_files",
        depends_on={"read_file"},
        enables={"run_tests"},
        weight=0.9,
    )

    # Add transitions (devops format)
    graph.add_transitions({
        "read_file": [("edit_files", 0.8), ("code_search", 0.6)],
    })

    # Add sequences (common patterns)
    graph.add_sequence(["read_file", "edit_files", "run_tests"])

    # Suggest next tools
    suggestions = graph.suggest_next_tools("read_file", history=["list_directory"])
    # Returns: [("edit_files", 0.9), ("code_search", 0.7), ...]

    # Get prerequisites
    prereqs = graph.get_prerequisites("edit_files")
    # Returns: {"read_file"}

    # Plan to achieve goal
    plan = graph.plan_for_goal({"run_tests"}, available={"read_file", "edit_files"})
    # Returns: ["read_file", "edit_files", "run_tests"]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class ToolDependency:
    """Simplified tool dependency specification.

    A compact representation of tool dependencies used for defining
    relationships between tools in the execution graph.

    Attributes:
        tool_name: Name of the tool
        depends_on: List of tools that must be called before this one
        enables: List of tools that are enabled after this one succeeds
        transition_weight: Weight for transitions from this tool (0.0 to 1.0)
    """

    tool_name: str
    depends_on: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)
    transition_weight: float = 1.0


@dataclass
class ToolNode:
    """Node in the tool execution graph.

    Attributes:
        name: Tool name
        depends_on: Tools that should be called before this one
        enables: Tools that are enabled after this one succeeds
        inputs: Data types this tool requires
        outputs: Data types this tool produces
        weight: Default transition weight for this tool
        metadata: Additional tool-specific data
    """

    name: str
    depends_on: Set[str] = field(default_factory=set)
    enables: Set[str] = field(default_factory=set)
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    weight: float = 1.0
    metadata: Dict[str, any] = field(default_factory=dict)

    @classmethod
    def from_dependency(cls, dep: ToolDependency) -> "ToolNode":
        """Create a ToolNode from a ToolDependency."""
        return cls(
            name=dep.tool_name,
            depends_on=set(dep.depends_on),
            enables=set(dep.enables),
            weight=dep.transition_weight,
        )


@dataclass
class ToolTransition:
    """Transition between tools in the graph.

    Attributes:
        from_tool: Source tool
        to_tool: Target tool
        weight: Transition probability weight (0.0 to 1.0)
        relationship: Type of relationship (depends_on, enables, sequence, etc.)
    """

    from_tool: str
    to_tool: str
    weight: float = 0.5
    relationship: str = "transition"


class ToolExecutionGraph:
    """Unified graph for tool execution patterns.

    Supports multiple paradigms for defining tool relationships:
    - Dependencies (depends_on/enables)
    - Transitions (probability weights)
    - Sequences (common patterns)
    - IO relationships (inputs/outputs)

    The graph provides intelligent suggestions based on:
    - Historical patterns
    - Dependency satisfaction
    - Workflow progression
    """

    def __init__(self, name: str = "default"):
        """Initialize the graph.

        Args:
            name: Graph identifier (e.g., vertical name)
        """
        self.name = name
        self._nodes: Dict[str, ToolNode] = {}
        self._transitions: Dict[str, List[ToolTransition]] = defaultdict(list)
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        self._sequences: List[List[str]] = []
        self._clusters: Dict[str, Set[str]] = {}

    # =========================================================================
    # Node Management
    # =========================================================================

    def add_node(
        self,
        name: str,
        depends_on: Optional[Set[str]] = None,
        enables: Optional[Set[str]] = None,
        inputs: Optional[Set[str]] = None,
        outputs: Optional[Set[str]] = None,
        weight: float = 1.0,
    ) -> ToolNode:
        """Add or update a tool node.

        Args:
            name: Tool name
            depends_on: Prerequisites
            enables: Tools enabled after success
            inputs: Required input data types
            outputs: Produced output data types
            weight: Default weight

        Returns:
            The created or updated ToolNode
        """
        if name in self._nodes:
            node = self._nodes[name]
            if depends_on:
                node.depends_on.update(depends_on)
            if enables:
                node.enables.update(enables)
            if inputs:
                node.inputs.update(inputs)
            if outputs:
                node.outputs.update(outputs)
            node.weight = weight
        else:
            node = ToolNode(
                name=name,
                depends_on=depends_on or set(),
                enables=enables or set(),
                inputs=inputs or set(),
                outputs=outputs or set(),
                weight=weight,
            )
            self._nodes[name] = node

        # Update reverse dependencies
        for dep in node.depends_on:
            self._reverse_deps[dep].add(name)

        # Create transitions from dependencies
        for dep in node.depends_on:
            self._add_transition(dep, name, weight * 0.9, "depends_on")

        for enabled in node.enables:
            self._add_transition(name, enabled, weight * 0.8, "enables")

        return node

    def get_node(self, name: str) -> Optional[ToolNode]:
        """Get a tool node by name.

        Args:
            name: Tool name

        Returns:
            ToolNode or None
        """
        return self._nodes.get(name)

    # =========================================================================
    # Dependency Format (Coding Style)
    # =========================================================================

    def add_dependency(
        self,
        tool_name_or_dep: str | ToolDependency,
        depends_on: Optional[Set[str]] = None,
        enables: Optional[Set[str]] = None,
        weight: float = 1.0,
    ) -> None:
        """Add tool dependency (coding format).

        Can accept either explicit parameters or a ToolDependency dataclass.

        Args:
            tool_name_or_dep: Tool name string or ToolDependency object
            depends_on: Tools that should be called first (ignored if ToolDependency)
            enables: Tools enabled after success (ignored if ToolDependency)
            weight: Transition weight (ignored if ToolDependency)
        """
        if isinstance(tool_name_or_dep, ToolDependency):
            dep = tool_name_or_dep
            self.add_node(
                name=dep.tool_name,
                depends_on=set(dep.depends_on),
                enables=set(dep.enables),
                weight=dep.transition_weight,
            )
        else:
            self.add_node(
                name=tool_name_or_dep,
                depends_on=depends_on,
                enables=enables,
                weight=weight,
            )

    def add_dependencies(
        self,
        dependencies: List[Tuple[str, Set[str], Set[str], float]],
    ) -> None:
        """Add multiple dependencies.

        Args:
            dependencies: List of (tool_name, depends_on, enables, weight) tuples
        """
        for tool_name, depends_on, enables, weight in dependencies:
            self.add_dependency(tool_name, depends_on, enables, weight)

    # =========================================================================
    # Transition Format (DevOps Style)
    # =========================================================================

    def _add_transition(
        self,
        from_tool: str,
        to_tool: str,
        weight: float,
        relationship: str = "transition",
    ) -> None:
        """Add a single transition.

        Args:
            from_tool: Source tool
            to_tool: Target tool
            weight: Transition weight
            relationship: Type of relationship
        """
        # Check if transition already exists
        existing = [t for t in self._transitions[from_tool] if t.to_tool == to_tool]
        if existing:
            # Update weight if new one is higher
            if weight > existing[0].weight:
                existing[0].weight = weight
        else:
            self._transitions[from_tool].append(
                ToolTransition(from_tool, to_tool, weight, relationship)
            )

    def add_transitions(
        self,
        transitions: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Add transitions (devops format).

        Args:
            transitions: Dict mapping source tool to list of (target, weight) tuples
        """
        for from_tool, targets in transitions.items():
            # Ensure source node exists
            if from_tool not in self._nodes:
                self.add_node(from_tool)

            for to_tool, weight in targets:
                # Ensure target node exists
                if to_tool not in self._nodes:
                    self.add_node(to_tool)
                self._add_transition(from_tool, to_tool, weight, "transition")

    # =========================================================================
    # Sequence Format
    # =========================================================================

    def add_sequence(self, sequence: List[str], weight: float = 0.7) -> None:
        """Add a tool sequence pattern.

        Args:
            sequence: Ordered list of tool names
            weight: Weight for sequence-based transitions
        """
        self._sequences.append(sequence)

        # Create transitions for adjacent tools
        for i, tool in enumerate(sequence[:-1]):
            next_tool = sequence[i + 1]
            self._add_transition(tool, next_tool, weight, "sequence")

    def add_sequences(self, sequences: List[List[str]], weight: float = 0.7) -> None:
        """Add multiple sequences.

        Args:
            sequences: List of tool sequences
            weight: Weight for all sequences
        """
        for seq in sequences:
            self.add_sequence(seq, weight)

    # =========================================================================
    # IO Format (Planning Style)
    # =========================================================================

    def add_io_tool(
        self,
        name: str,
        inputs: Set[str],
        outputs: Set[str],
        weight: float = 1.0,
    ) -> None:
        """Add tool with IO relationships (planning format).

        Tools that produce outputs needed by other tools' inputs
        automatically create transitions.

        Args:
            name: Tool name
            inputs: Required input data types
            outputs: Produced output data types
            weight: Default weight
        """
        self.add_node(name, inputs=inputs, outputs=outputs, weight=weight)

        # Create transitions based on IO relationships
        for existing_name, existing_node in self._nodes.items():
            if existing_name == name:
                continue

            # If existing tool produces inputs we need, add transition
            if existing_node.outputs & inputs:
                self._add_transition(existing_name, name, weight * 0.8, "io_dependency")

            # If we produce outputs needed by existing tool, add transition
            if outputs & existing_node.inputs:
                self._add_transition(name, existing_name, weight * 0.8, "io_enables")

    # =========================================================================
    # Cluster Management
    # =========================================================================

    def add_cluster(self, name: str, tools: Set[str]) -> None:
        """Add a tool cluster (tools that work well together).

        Args:
            name: Cluster name
            tools: Set of tool names in cluster
        """
        self._clusters[name] = tools

    def get_cluster_tools(self, tool_name: str) -> Set[str]:
        """Get tools in the same cluster as a given tool.

        Args:
            tool_name: Tool to find cluster for

        Returns:
            Set of tools in same cluster(s)
        """
        related = set()
        for cluster_tools in self._clusters.values():
            if tool_name in cluster_tools:
                related.update(cluster_tools)
        related.discard(tool_name)
        return related

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_prerequisites(self, tool_name: str) -> Set[str]:
        """Get prerequisite tools.

        Args:
            tool_name: Tool to get prerequisites for

        Returns:
            Set of prerequisite tool names
        """
        node = self._nodes.get(tool_name)
        if node:
            return node.depends_on.copy()
        return set()

    def get_enabled_tools(self, tool_name: str) -> Set[str]:
        """Get tools enabled by a tool.

        Args:
            tool_name: Tool that enables others

        Returns:
            Set of enabled tool names
        """
        node = self._nodes.get(tool_name)
        if node:
            return node.enables.copy()
        return set()

    def get_dependents(self, tool_name: str) -> Set[str]:
        """Get tools that depend on a tool.

        Args:
            tool_name: Tool that others depend on

        Returns:
            Set of dependent tool names
        """
        return self._reverse_deps.get(tool_name, set()).copy()

    def suggest_next_tools(
        self,
        current_tool: str,
        history: Optional[List[str]] = None,
        available_tools: Optional[Set[str]] = None,
        max_suggestions: int = 5,
    ) -> List[Tuple[str, float]]:
        """Suggest next tools based on current state.

        Args:
            current_tool: Currently executing tool
            history: Previously used tools
            available_tools: Tools that can be used (None = all)
            max_suggestions: Maximum suggestions to return

        Returns:
            List of (tool_name, score) tuples, sorted by score
        """
        history = history or []
        recent_set = set(history[-3:]) if len(history) >= 3 else set(history)

        candidates: Dict[str, float] = {}

        # 1. Add transition-based candidates
        for transition in self._transitions.get(current_tool, []):
            if available_tools is None or transition.to_tool in available_tools:
                score = transition.weight
                # Reduce score for recently used tools
                if transition.to_tool in recent_set:
                    score *= 0.5
                candidates[transition.to_tool] = max(candidates.get(transition.to_tool, 0), score)

        # 2. Add cluster-related tools with lower score
        cluster_tools = self.get_cluster_tools(current_tool)
        for tool in cluster_tools:
            if available_tools is None or tool in available_tools:
                if tool not in candidates:
                    score = 0.4
                    if tool in recent_set:
                        score *= 0.5
                    candidates[tool] = score

        # 3. Add enabled tools from node
        node = self._nodes.get(current_tool)
        if node:
            for enabled in node.enables:
                if available_tools is None or enabled in available_tools:
                    score = node.weight * 0.85
                    if enabled in recent_set:
                        score *= 0.5
                    candidates[enabled] = max(candidates.get(enabled, 0), score)

        # Sort by score and return top suggestions
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:max_suggestions]

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        """Get transition weight between two tools.

        Args:
            from_tool: Source tool
            to_tool: Target tool

        Returns:
            Transition weight (0.0 to 1.0), or 0.3 default
        """
        # Check direct transitions
        for transition in self._transitions.get(from_tool, []):
            if transition.to_tool == to_tool:
                return transition.weight

        # Check if to_tool depends on from_tool
        to_node = self._nodes.get(to_tool)
        if to_node and from_tool in to_node.depends_on:
            return to_node.weight * 0.9

        # Check if from_tool enables to_tool
        from_node = self._nodes.get(from_tool)
        if from_node and to_tool in from_node.enables:
            return from_node.weight * 0.8

        # Check sequence adjacency
        for seq in self._sequences:
            for i, tool in enumerate(seq[:-1]):
                if tool == from_tool and seq[i + 1] == to_tool:
                    return 0.6

        return 0.3  # Default weight

    def plan_for_goal(
        self,
        goal_tools: Set[str],
        available_tools: Optional[Set[str]] = None,
        current_state: Optional[Set[str]] = None,
    ) -> List[str]:
        """Plan tool sequence to achieve goal.

        Uses simple dependency-based planning with transitive dependency resolution.

        Args:
            goal_tools: Tools that need to be executed
            available_tools: Tools that can be used
            current_state: Tools already executed

        Returns:
            Ordered list of tools to execute
        """
        current_state = current_state or set()
        plan: List[str] = []

        # Expand goals to include all transitive dependencies
        all_needed: Set[str] = set()
        to_process = list(goal_tools)
        while to_process:
            tool = to_process.pop()
            if tool in all_needed or tool in current_state:
                continue
            all_needed.add(tool)
            prereqs = self.get_prerequisites(tool)
            for prereq in prereqs:
                if prereq not in all_needed and prereq not in current_state:
                    to_process.append(prereq)

        remaining = all_needed - current_state

        # Simple topological sort based on dependencies
        while remaining:
            # Find tools with satisfied dependencies
            ready = set()
            for tool in remaining:
                prereqs = self.get_prerequisites(tool)
                if prereqs <= (current_state | set(plan)):
                    if available_tools is None or tool in available_tools:
                        ready.add(tool)

            if not ready:
                # No tools can be executed - add remaining anyway
                for tool in remaining:
                    if available_tools is None or tool in available_tools:
                        plan.append(tool)
                break

            # Add ready tools (sorted for determinism)
            for tool in sorted(ready):
                plan.append(tool)

            remaining -= ready

        return plan

    def validate_execution(
        self, tool_name: str, executed_tools: Set[str]
    ) -> Tuple[bool, List[str]]:
        """Validate if a tool can be executed.

        Args:
            tool_name: Tool to validate
            executed_tools: Previously executed tools

        Returns:
            Tuple of (is_valid, list of missing prerequisites)
        """
        prereqs = self.get_prerequisites(tool_name)
        missing = prereqs - executed_tools
        return len(missing) == 0, list(missing)

    def get_next_tools(self, current_tool: str) -> List[Tuple[str, float]]:
        """Get enabled tools with their transition weights.

        Returns tools that can be called after the current tool based on
        dependencies and transitions in the graph.

        Args:
            current_tool: Currently executing tool

        Returns:
            List of (tool_name, weight) tuples sorted by weight descending
        """
        candidates: Dict[str, float] = {}

        # 1. Add tools enabled by this tool
        node = self._nodes.get(current_tool)
        if node:
            for enabled in node.enables:
                candidates[enabled] = node.weight * 0.85

        # 2. Add transition-based candidates
        for transition in self._transitions.get(current_tool, []):
            candidates[transition.to_tool] = max(
                candidates.get(transition.to_tool, 0), transition.weight
            )

        # 3. Add tools that depend on this tool (reverse lookup)
        for tool_name, deps in self._reverse_deps.items():
            if current_tool in deps:
                # This tool depends on current_tool, so it's a candidate
                dep_node = self._nodes.get(tool_name)
                if dep_node:
                    candidates[tool_name] = max(candidates.get(tool_name, 0), dep_node.weight * 0.9)

        # Sort by weight descending
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    def validate_sequence(self, sequence: List[str]) -> bool:
        """Validate if a tool sequence respects dependencies.

        Checks that each tool in the sequence has its dependencies
        satisfied by tools that appear earlier in the sequence.

        Args:
            sequence: List of tool names in execution order

        Returns:
            True if the sequence is valid (all dependencies satisfied),
            False if any tool's dependencies are not met
        """
        executed: Set[str] = set()

        for tool_name in sequence:
            prereqs = self.get_prerequisites(tool_name)
            if prereqs and not prereqs.issubset(executed):
                # Tool has unmet dependencies
                return False
            executed.add(tool_name)

        return True

    # =========================================================================
    # Merge and Export
    # =========================================================================

    def merge(self, other: "ToolExecutionGraph") -> None:
        """Merge another graph into this one.

        Args:
            other: Graph to merge
        """
        # Merge nodes
        for name, node in other._nodes.items():
            self.add_node(
                name,
                depends_on=node.depends_on,
                enables=node.enables,
                inputs=node.inputs,
                outputs=node.outputs,
                weight=node.weight,
            )

        # Merge transitions
        for from_tool, transitions in other._transitions.items():
            for trans in transitions:
                self._add_transition(
                    trans.from_tool, trans.to_tool, trans.weight, trans.relationship
                )

        # Merge sequences
        self._sequences.extend(other._sequences)

        # Merge clusters
        self._clusters.update(other._clusters)

    def to_dict(self) -> Dict:
        """Export graph as dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "nodes": {
                name: {
                    "depends_on": list(node.depends_on),
                    "enables": list(node.enables),
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "weight": node.weight,
                }
                for name, node in self._nodes.items()
            },
            "transitions": {
                from_tool: [(t.to_tool, t.weight) for t in transitions]
                for from_tool, transitions in self._transitions.items()
            },
            "sequences": self._sequences,
            "clusters": {k: list(v) for k, v in self._clusters.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolExecutionGraph":
        """Create graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New ToolExecutionGraph
        """
        graph = cls(data.get("name", "default"))

        for name, node_data in data.get("nodes", {}).items():
            graph.add_node(
                name,
                depends_on=set(node_data.get("depends_on", [])),
                enables=set(node_data.get("enables", [])),
                inputs=set(node_data.get("inputs", [])),
                outputs=set(node_data.get("outputs", [])),
                weight=node_data.get("weight", 1.0),
            )

        for from_tool, transitions in data.get("transitions", {}).items():
            for to_tool, weight in transitions:
                graph._add_transition(from_tool, to_tool, weight, "imported")

        for seq in data.get("sequences", []):
            graph._sequences.append(seq)

        for name, tools in data.get("clusters", {}).items():
            graph._clusters[name] = set(tools)

        return graph

    @property
    def all_tools(self) -> Set[str]:
        """Get all tool names in the graph."""
        return set(self._nodes.keys())

    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    @property
    def transition_count(self) -> int:
        """Get number of transitions."""
        return sum(len(t) for t in self._transitions.values())


# =============================================================================
# Graph Registry
# =============================================================================


class ToolGraphRegistry:
    """Registry for tool execution graphs.

    Provides centralized access to graphs for different verticals.
    """

    _instance: Optional["ToolGraphRegistry"] = None

    def __init__(self):
        """Initialize the registry."""
        self._graphs: Dict[str, ToolExecutionGraph] = {}
        self._default_graph = ToolExecutionGraph("default")

    @classmethod
    def get_instance(cls) -> "ToolGraphRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register_graph(self, name: str, graph: ToolExecutionGraph) -> None:
        """Register a graph for a vertical.

        Args:
            name: Vertical name
            graph: Tool execution graph
        """
        self._graphs[name] = graph

    def get_graph(self, name: str) -> ToolExecutionGraph:
        """Get graph for a vertical.

        Args:
            name: Vertical name

        Returns:
            ToolExecutionGraph (falls back to default)
        """
        return self._graphs.get(name, self._default_graph)

    def get_merged_graph(self, names: List[str]) -> ToolExecutionGraph:
        """Get merged graph from multiple verticals.

        Args:
            names: List of vertical names

        Returns:
            Merged ToolExecutionGraph
        """
        merged = ToolExecutionGraph("merged")
        for name in names:
            if name in self._graphs:
                merged.merge(self._graphs[name])
        return merged

    def list_graphs(self) -> List[str]:
        """List registered graph names."""
        return list(self._graphs.keys())


__all__ = [
    # Dataclasses
    "ToolDependency",
    "ToolNode",
    "ToolTransition",
    # Main class
    "ToolExecutionGraph",
    # Registry
    "ToolGraphRegistry",
]
