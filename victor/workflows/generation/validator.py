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

"""Multi-layer workflow validation system.

This module provides comprehensive validation for dynamically generated workflows
across four layers: schema, structure, semantic, and security.

Design Principles (SOLID):
    - SRP: Each validator handles one validation layer
    - OCP: Extensible via new validation rules
    - LSP: All validators implement the same interface
    - ISP: Focused validation methods per layer
    - DIP: High-level modules depend on validation abstractions

Key Features:
    - Layer 1: Schema validation (Pydantic-based)
    - Layer 2: Graph structure validation (reachability, cycles)
    - Layer 3: Semantic validation (node types, tools)
    - Layer 4: Security validation (resource limits, safety)

Example:
    from victor.workflows.generation import WorkflowValidator

    validator = WorkflowValidator(strict_mode=True)
    result = validator.validate(workflow_dict)

    if not result.is_valid:
        print(f"Found {len(result.all_errors)} validation errors")
        for error in result.critical_errors:
            print(f"  {error.location}: {error.message}")
"""

import copy
import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from victor.workflows.generation.requirements import RequirementValidationResult

from victor.workflows.generation.types import (
    ErrorCategory,
    ErrorSeverity,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
)

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Layer 1: Schema validation (Pydantic-based).

    Validates workflow structure against expected schema:
    - Required fields present (nodes, edges, entry_point)
    - Field types correct (nodes is list, edges is list, etc.)
    - Field values within valid ranges (tool_budget > 0, timeout >= 0)
    - String constraints (non-empty node IDs, valid edge types)
    - List constraints (non-empty node list, unique node IDs)
    """

    # Valid node types
    VALID_NODE_TYPES = {
        "agent",
        "compute",
        "condition",
        "parallel",
        "transform",
        "team",
        "hitl",
    }

    # Valid agent roles
    VALID_AGENT_ROLES = {
        "researcher",
        "planner",
        "executor",
        "reviewer",
        "writer",
        "analyst",
        "coordinator",
    }

    # Valid team formations
    VALID_TEAM_FORMATIONS = {
        "sequential",
        "parallel",
        "hierarchical",
        "pipeline",
        "consensus",
    }

    # Valid join strategies
    VALID_JOIN_STRATEGIES = {"all", "any", "merge"}

    def __init__(self, strict_mode: bool = True):
        """Initialize schema validator.

        Args:
            strict_mode: If True, all errors fail validation. If False, warnings for non-critical.
        """
        self.strict_mode = strict_mode

    def validate(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate workflow schema.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check required top-level fields
        errors.extend(self._validate_required_fields(workflow))

        # Check nodes if present
        if "nodes" in workflow:
            errors.extend(self._validate_nodes(workflow.get("nodes", [])))

        # Check edges if present
        if "edges" in workflow:
            errors.extend(
                self._validate_edges(workflow.get("edges", []), workflow.get("nodes", []))
            )

        # Check entry point if present
        if "entry_point" in workflow:
            errors.extend(self._validate_entry_point(workflow))

        # Check workflow-level config
        errors.extend(self._validate_workflow_config(workflow))

        return errors

    def _validate_required_fields(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate required top-level fields."""
        errors = []
        required_fields = ["nodes", "entry_point"]

        for field in required_fields:
            if field not in workflow:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.CRITICAL,
                        message=f"Missing required field: '{field}'",
                        location=f"workflow.{field}",
                        suggestion=f"Add '{field}' field to workflow definition",
                    )
                )

        return errors

    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[WorkflowValidationError]:
        """Validate nodes list."""
        errors = []

        # Check if nodes list is non-empty
        if not nodes:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.CRITICAL,
                    message="Nodes list is empty",
                    location="workflow.nodes",
                    suggestion="Add at least one node to the workflow",
                )
            )
            return errors

        # Check for unique node IDs
        node_ids = [node.get("id") for node in nodes if "id" in node]
        duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]

        if duplicates:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message=f"Duplicate node IDs found: {set(duplicates)}",
                    location="workflow.nodes",
                    suggestion="Ensure all node IDs are unique",
                )
            )

        # Validate each node
        for i, node in enumerate(nodes):
            errors.extend(self._validate_node(node, i))

        return errors

    def _validate_node(self, node: Dict[str, Any], index: int) -> List[WorkflowValidationError]:
        """Validate individual node."""
        errors = []
        node_id = node.get("id", f"nodes[{index}]")
        location = f"nodes[{node_id}]"

        # Check required fields
        if "id" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Node missing 'id' field",
                    location=f"nodes[{index}]",
                    suggestion="Add unique 'id' field to node",
                )
            )

        if "type" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.CRITICAL,
                    message="Node missing 'type' field",
                    location=location,
                    suggestion=f"Add 'type' field (valid: {self.VALID_NODE_TYPES})",
                )
            )
            return errors  # Can't validate further without type

        # Validate node type
        node_type = node.get("type")
        if node_type not in self.VALID_NODE_TYPES:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message=f"Invalid node type: '{node_type}'",
                    location=location,
                    suggestion=f"Use one of: {self.VALID_NODE_TYPES}",
                )
            )

        # Type-specific validation
        if node_type == "agent":
            errors.extend(self._validate_agent_node(node, location))
        elif node_type == "compute":
            errors.extend(self._validate_compute_node(node, location))
        elif node_type == "condition":
            errors.extend(self._validate_condition_node(node, location))
        elif node_type == "parallel":
            errors.extend(self._validate_parallel_node(node, location))
        elif node_type == "transform":
            errors.extend(self._validate_transform_node(node, location))
        elif node_type == "team":
            errors.extend(self._validate_team_node(node, location))

        return errors

    def _validate_agent_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate agent node fields."""
        errors = []

        # Check required fields
        if "role" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Agent node missing 'role' field",
                    location=location,
                    suggestion=f"Add 'role' field (valid: {self.VALID_AGENT_ROLES})",
                )
            )
        elif node.get("role") not in self.VALID_AGENT_ROLES:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message=f"Invalid agent role: '{node.get('role')}'",
                    location=f"{location}.role",
                    suggestion=f"Use one of: {self.VALID_AGENT_ROLES}",
                )
            )

        if "goal" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Agent node missing 'goal' field",
                    location=location,
                    suggestion="Add 'goal' field with task description",
                )
            )

        # Validate tool_budget if present
        if "tool_budget" in node:
            tool_budget = node.get("tool_budget")
            if not isinstance(tool_budget, int):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message=f"tool_budget must be integer, got {type(tool_budget).__name__}",
                        location=f"{location}.tool_budget",
                        suggestion="Convert tool_budget to integer",
                    )
                )
            elif not (1 <= tool_budget <= 500):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message=f"tool_budget {tool_budget} out of range [1, 500]",
                        location=f"{location}.tool_budget",
                        suggestion="Set tool_budget between 1 and 500",
                    )
                )

        return errors

    def _validate_compute_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate compute node fields."""
        errors = []

        # Compute nodes must have tools OR handler
        has_tools = "tools" in node and node["tools"]
        has_handler = "handler" in node and node["handler"]

        if not has_tools and not has_handler:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Compute node must specify 'tools' or 'handler'",
                    location=location,
                    suggestion="Add 'tools' list or 'handler' function name",
                )
            )

        return errors

    def _validate_condition_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate condition node fields."""
        errors = []

        if "branches" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Condition node missing 'branches' field",
                    location=location,
                    suggestion="Add 'branches' mapping (e.g., {'continue': 'node1', 'done': '__end__'})",
                )
            )

        return errors

    def _validate_parallel_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate parallel node fields."""
        errors = []

        if "parallel_nodes" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Parallel node missing 'parallel_nodes' field",
                    location=location,
                    suggestion="Add 'parallel_nodes' list with node IDs to execute in parallel",
                )
            )

        if "join_strategy" in node:
            join_strategy = node.get("join_strategy")
            if join_strategy not in self.VALID_JOIN_STRATEGIES:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message=f"Invalid join_strategy: '{join_strategy}'",
                        location=f"{location}.join_strategy",
                        suggestion=f"Use one of: {self.VALID_JOIN_STRATEGIES}",
                    )
                )

        return errors

    def _validate_transform_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate transform node fields."""
        errors = []

        if "transform" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Transform node missing 'transform' field",
                    location=location,
                    suggestion="Add 'transform' field with transformation function name",
                )
            )

        return errors

    def _validate_team_node(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate team node fields."""
        errors = []

        if "team_formation" in node:
            formation = node.get("team_formation")
            if formation not in self.VALID_TEAM_FORMATIONS:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message=f"Invalid team_formation: '{formation}'",
                        location=f"{location}.team_formation",
                        suggestion=f"Use one of: {self.VALID_TEAM_FORMATIONS}",
                    )
                )

        if "members" not in node:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.ERROR,
                    message="Team node missing 'members' field",
                    location=location,
                    suggestion="Add 'members' list with team member definitions",
                )
            )

        return errors

    def _validate_edges(
        self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]
    ) -> List[WorkflowValidationError]:
        """Validate edges list."""
        errors = []

        # Get valid node IDs
        node_ids = {node.get("id") for node in nodes if "id" in node}
        node_ids.add("__end__")  # END is always valid

        # Validate each edge
        for i, edge in enumerate(edges):
            location = f"edges[{i}]"

            # Check required fields
            if "source" not in edge:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message="Edge missing 'source' field",
                        location=location,
                        suggestion="Add 'source' field with source node ID",
                    )
                )
                continue  # Can't validate further without source

            # Check target
            if "target" not in edge:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.ERROR,
                        message="Edge missing 'target' field",
                        location=location,
                        suggestion="Add 'target' field with target node ID or '__end__'",
                    )
                )

        return errors

    def _validate_entry_point(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate entry point."""
        errors = []
        entry_point = workflow.get("entry_point")
        nodes = workflow.get("nodes", [])
        node_ids = {node.get("id") for node in nodes if "id" in node}

        if entry_point not in node_ids:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SCHEMA,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Entry point '{entry_point}' not found in nodes",
                    location="workflow.entry_point",
                    suggestion=f"Set entry_point to one of: {list(node_ids)}",
                )
            )

        return errors

    def _validate_workflow_config(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate workflow-level configuration."""
        errors = []

        # Validate max_iterations
        if "max_iterations" in workflow:
            max_iter = workflow.get("max_iterations")
            if not isinstance(max_iter, int):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.WARNING,
                        message=f"max_iterations must be integer, got {type(max_iter).__name__}",
                        location="workflow.max_iterations",
                        suggestion="Convert max_iterations to integer",
                    )
                )
            elif not (1 <= max_iter <= 500):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.WARNING,
                        message=f"max_iterations {max_iter} out of range [1, 500]",
                        location="workflow.max_iterations",
                        suggestion="Set max_iterations between 1 and 500",
                    )
                )

        # Validate max_timeout_seconds
        if "max_timeout_seconds" in workflow:
            timeout = workflow.get("max_timeout_seconds")
            if not isinstance(timeout, (int, float)):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.WARNING,
                        message=f"max_timeout_seconds must be numeric, got {type(timeout).__name__}",
                        location="workflow.max_timeout_seconds",
                        suggestion="Convert max_timeout_seconds to float",
                    )
                )
            elif timeout < 0:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.WARNING,
                        message=f"max_timeout_seconds must be non-negative, got {timeout}",
                        location="workflow.max_timeout_seconds",
                        suggestion="Set max_timeout_seconds to >= 0",
                    )
                )

        return errors


class GraphStructureValidator:
    """Layer 2: Graph structure validation.

    Validates workflow graph topology:
    - All nodes reachable from entry point
    - No orphan nodes (except entry point)
    - Edge references valid nodes
    - No unconditional cycles
    - At least one path to END
    """

    def __init__(self, max_cycle_depth: int = 10):
        """Initialize graph structure validator.

        Args:
            max_cycle_depth: Maximum allowed cycle depth
        """
        self.max_cycle_depth = max_cycle_depth

    def validate(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate graph structure.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            List of validation errors
        """
        errors = []

        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])
        entry_point = workflow.get("entry_point")

        if not nodes or not entry_point:
            return errors  # Schema validation will catch these

        # Build adjacency list
        graph = self._build_graph(nodes, edges)

        # Check node reachability
        errors.extend(self._check_reachability(graph, entry_point, nodes))

        # Check edge references
        errors.extend(self._check_edge_references(graph, nodes, edges))

        # Check for invalid cycles
        errors.extend(self._check_cycles(graph, nodes))

        # Check for dead ends
        errors.extend(self._check_dead_ends(graph, nodes))

        return errors

    def _build_graph(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Build adjacency list representation."""
        graph = {node.get("id"): [] for node in nodes if "id" in node}

        for edge in edges:
            source = edge.get("source")
            if source and source in graph:
                target = edge.get("target")
                if target:
                    if isinstance(target, str):
                        graph[source].append(target)
                    elif isinstance(target, dict):
                        # Conditional edge - add all branch targets
                        graph[source].extend(target.values())

        return graph

    def _check_reachability(
        self, graph: Dict[str, List[str]], entry_point: str, nodes: List[Dict[str, Any]]
    ) -> List[WorkflowValidationError]:
        """Check all nodes are reachable from entry point."""
        errors = []

        if entry_point not in graph:
            return errors  # Schema validation will catch this

        # BFS from entry point
        visited = set()
        queue = [entry_point]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            queue.extend(graph.get(current, []))

        # Find unreachable nodes
        node_ids = {node.get("id") for node in nodes if "id" in node}
        unreachable = node_ids - visited - {"__end__"}

        for node_id in unreachable:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.STRUCTURE,
                    severity=ErrorSeverity.ERROR,
                    message=f"Node '{node_id}' is not reachable from entry point '{entry_point}'",
                    location=f"nodes[{node_id}]",
                    suggestion=f"Add edge from existing node to '{node_id}' or set as entry_point",
                )
            )

        return errors

    def _check_edge_references(
        self,
        graph: Dict[str, List[str]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[WorkflowValidationError]:
        """Check edge references exist."""
        errors = []

        node_ids = {node.get("id") for node in nodes if "id" in node}
        node_ids.add("__end__")  # END is always valid

        for i, edge in enumerate(edges):
            source = edge.get("source")
            target = edge.get("target")

            # Check source exists
            if source and source not in node_ids:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.STRUCTURE,
                        severity=ErrorSeverity.ERROR,
                        message=f"Edge source '{source}' not found in nodes",
                        location=f"edges[{i}].source",
                        suggestion=f"Fix source to reference existing node: {list(node_ids)}",
                    )
                )

            # Check target exists
            if target:
                if isinstance(target, str) and target not in node_ids:
                    errors.append(
                        WorkflowValidationError(
                            category=ErrorCategory.STRUCTURE,
                            severity=ErrorSeverity.ERROR,
                            message=f"Edge target '{target}' not found in nodes",
                            location=f"edges[{i}].target",
                            suggestion=f"Fix target to reference existing node: {list(node_ids)}",
                        )
                    )
                elif isinstance(target, dict):
                    # Check all branch targets
                    for branch_name, branch_target in target.items():
                        if branch_target not in node_ids:
                            errors.append(
                                WorkflowValidationError(
                                    category=ErrorCategory.STRUCTURE,
                                    severity=ErrorSeverity.ERROR,
                                    message=f"Conditional branch '{branch_name}' target '{branch_target}' not found",
                                    location=f"edges[{i}].target.{branch_name}",
                                    suggestion=f"Fix branch target to: {list(node_ids)}",
                                )
                            )

        return errors

    def _check_cycles(
        self, graph: Dict[str, List[str]], nodes: List[Dict[str, Any]]
    ) -> List[WorkflowValidationError]:
        """Detect invalid cycles (cycles without condition nodes)."""
        errors = []

        # Build node type lookup
        node_types = {node.get("id"): node.get("type") for node in nodes if "id" in node}
        condition_nodes = {n.get("id") for n in nodes if n.get("type") == "condition"}

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            """DFS to detect cycles, returns cycle path if found."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in node_types:
                    continue  # Skip END marker
                if neighbor not in visited:
                    result = dfs(neighbor, path.copy())
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.remove(node)
            return None

        for node_id in graph:
            if node_id not in visited:
                cycle_path = dfs(node_id, [])
                if cycle_path:
                    # Check if cycle contains a condition node
                    has_condition = any(n in condition_nodes for n in cycle_path)

                    if not has_condition:
                        errors.append(
                            WorkflowValidationError(
                                category=ErrorCategory.STRUCTURE,
                                severity=ErrorSeverity.CRITICAL,
                                message=f"Unconditional cycle detected: {' -> '.join(cycle_path)}",
                                location="edges",
                                suggestion="Add a condition node in the cycle or restructure to avoid cycle",
                            )
                        )

        return errors

    def _check_dead_ends(
        self, graph: Dict[str, List[str]], nodes: List[Dict[str, Any]]
    ) -> List[WorkflowValidationError]:
        """Check for dead-end nodes."""
        errors = []

        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            # Skip if node has outgoing edges
            if graph.get(node_id):
                continue

            # Skip if node is a terminal node (no outgoing edges expected)
            node_type = node.get("type")
            if node_type in ["hitl"]:
                continue

            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.STRUCTURE,
                    severity=ErrorSeverity.WARNING,
                    message=f"Node '{node_id}' has no outgoing edges (dead end)",
                    location=f"nodes[{node_id}]",
                    suggestion="Add edge to another node or to '__end__'",
                )
            )

        return errors


class SemanticValidator:
    """Layer 3: Semantic validation.

    Validates node semantics:
    - Node types have correct configurations
    - Tools are available in registry
    - Handlers are registered
    - Conditions are valid
    """

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        handler_registry: Optional[Dict[str, Any]] = None,
        strict_mode: bool = True,
    ):
        """Initialize semantic validator.

        Args:
            tool_registry: Tool registry for tool availability checks
            handler_registry: Handler registry for handler availability checks
            strict_mode: If True, unavailable tools/handlers are errors
        """
        self.tool_registry = tool_registry
        self.handler_registry = handler_registry or {}
        self.strict_mode = strict_mode

    def validate(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate workflow semantics.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            List of validation errors
        """
        errors = []

        nodes = workflow.get("nodes", [])

        # Validate each node's semantics
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            errors.extend(self._validate_node_semantics(node))

        return errors

    def _validate_node_semantics(self, node: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate node semantics."""
        errors = []
        node_id = node.get("id", "unknown")
        node_type = node.get("type")
        location = f"nodes[{node_id}]"

        if node_type == "agent":
            errors.extend(self._validate_agent_semantics(node, location))
        elif node_type == "compute":
            errors.extend(self._validate_compute_semantics(node, location))
        elif node_type == "condition":
            errors.extend(self._validate_condition_semantics(node, location))

        return errors

    def _validate_agent_semantics(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate agent node semantics."""
        errors = []

        # Validate tools if specified
        if "tools" in node and node["tools"]:
            tools = node["tools"]
            if not isinstance(tools, list):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SEMANTIC,
                        severity=ErrorSeverity.ERROR,
                        message=f"Agent 'tools' must be a list, got {type(tools).__name__}",
                        location=f"{location}.tools",
                        suggestion="Convert tools to a list",
                    )
                )
            else:
                # Check tool availability (if registry available)
                if self.tool_registry:
                    for tool_name in tools:
                        if not self._tool_exists(tool_name):
                            severity = (
                                ErrorSeverity.ERROR if self.strict_mode else ErrorSeverity.WARNING
                            )
                            errors.append(
                                WorkflowValidationError(
                                    category=ErrorCategory.SEMANTIC,
                                    severity=severity,
                                    message=f"Tool '{tool_name}' not found in registry",
                                    location=f"{location}.tools",
                                    suggestion="Verify tool is registered or remove from tools list",
                                )
                            )

        return errors

    def _validate_compute_semantics(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate compute node semantics."""
        errors = []

        # Validate handler if specified
        if "handler" in node and node["handler"]:
            handler_name = node["handler"]
            if handler_name not in self.handler_registry:
                severity = ErrorSeverity.ERROR if self.strict_mode else ErrorSeverity.WARNING
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SEMANTIC,
                        severity=severity,
                        message=f"Handler '{handler_name}' not registered",
                        location=f"{location}.handler",
                        suggestion=f"Register handler or use one of: {list(self.handler_registry.keys())}",
                    )
                )

        # Validate tools if specified
        if "tools" in node and node["tools"]:
            tools = node["tools"]
            if self.tool_registry:
                for tool_name in tools:
                    if not self._tool_exists(tool_name):
                        severity = (
                            ErrorSeverity.ERROR if self.strict_mode else ErrorSeverity.WARNING
                        )
                        errors.append(
                            WorkflowValidationError(
                                category=ErrorCategory.SEMANTIC,
                                severity=severity,
                                message=f"Tool '{tool_name}' not found in registry",
                                location=f"{location}.tools",
                                suggestion="Verify tool is registered or remove from tools list",
                            )
                        )

        return errors

    def _validate_condition_semantics(
        self, node: Dict[str, Any], location: str
    ) -> List[WorkflowValidationError]:
        """Validate condition node semantics."""
        errors = []

        # Validate branches
        if "branches" in node and node["branches"]:
            branches = node["branches"]
            if not isinstance(branches, dict):
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SEMANTIC,
                        severity=ErrorSeverity.ERROR,
                        message=f"Condition 'branches' must be a dict, got {type(branches).__name__}",
                        location=f"{location}.branches",
                        suggestion="Convert branches to a dict mapping branch names to node IDs",
                    )
                )

        return errors

    def _tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists in registry."""
        if not self.tool_registry:
            return True  # Skip validation if no registry

        try:
            # Try to get tool from registry
            return self.tool_registry.get_tool(tool_name) is not None
        except Exception:
            return False


class SecurityValidator:
    """Layer 4: Security validation.

    Validates security and safety constraints:
    - Resource limits (tool budget, timeout, iterations)
    - Dangerous tool combinations
    - Airgapped mode constraints
    - Infinite loop prevention
    """

    def __init__(
        self,
        max_tool_budget: int = 500,
        max_timeout_seconds: float = 3600.0,
        max_iterations: int = 50,
        max_parallel_branches: int = 10,
        airgapped_mode: bool = False,
    ):
        """Initialize security validator.

        Args:
            max_tool_budget: Maximum total tool budget
            max_timeout_seconds: Maximum execution timeout
            max_iterations: Maximum workflow iterations
            max_parallel_branches: Maximum parallel branches
            airgapped_mode: Whether airgapped mode is enabled
        """
        self.max_tool_budget = max_tool_budget
        self.max_timeout_seconds = max_timeout_seconds
        self.max_iterations = max_iterations
        self.max_parallel_branches = max_parallel_branches
        self.airgapped_mode = airgapped_mode

        # Dangerous tool combinations
        self.dangerous_combinations = [
            ({"file_delete", "file_write"}, "bash_execute"),
        ]

        # Network tools (blocked in airgapped mode)
        self.network_tools = {"web_search", "http_request", "api_call", "fetch_url"}

    def validate(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Validate security and safety constraints.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check resource limits
        errors.extend(self._check_resource_limits(workflow))

        # Check tool combinations
        errors.extend(self._check_tool_combinations(workflow))

        # Check airgapped mode constraints
        if self.airgapped_mode:
            errors.extend(self._check_airgapped_constraints(workflow))

        # Check for infinite loops
        errors.extend(self._check_infinite_loops(workflow))

        return errors

    def _check_resource_limits(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Check workflow doesn't exceed resource limits."""
        errors = []

        nodes = workflow.get("nodes", [])

        # Sum tool budgets
        total_tool_budget = 0
        for node in nodes:
            if node.get("type") == "agent":
                budget = node.get("tool_budget", 15)
                # Convert to int if it's a string
                if isinstance(budget, str):
                    try:
                        budget = int(budget)
                    except (ValueError, TypeError):
                        budget = 15  # Use default if conversion fails
                total_tool_budget += budget

        if total_tool_budget > self.max_tool_budget:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SECURITY,
                    severity=ErrorSeverity.ERROR,
                    message=f"Total tool budget {total_tool_budget} exceeds limit {self.max_tool_budget}",
                    location="workflow",
                    suggestion="Reduce tool budgets or increase max_tool_budget",
                )
            )

        # Check parallel branches
        for node in nodes:
            if node.get("type") == "parallel" and "parallel_nodes" in node:
                parallel_count = len(node.get("parallel_nodes", []))
                if parallel_count > self.max_parallel_branches:
                    errors.append(
                        WorkflowValidationError(
                            category=ErrorCategory.SECURITY,
                            severity=ErrorSeverity.ERROR,
                            message=f"Parallel node '{node.get('id')}' has {parallel_count} branches, exceeds limit {self.max_parallel_branches}",
                            location=f"nodes[{node.get('id')}]",
                            suggestion="Reduce parallel branches or increase max_parallel_branches",
                        )
                    )

        return errors

    def _check_tool_combinations(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Check for dangerous tool combinations."""
        errors = []

        nodes = workflow.get("nodes", [])

        # Collect all tools used in workflow
        all_tools = set()
        for node in nodes:
            if "tools" in node and node["tools"]:
                all_tools.update(node["tools"])

        # Check dangerous combinations
        for tool_set, dangerous_tool in self.dangerous_combinations:
            if tool_set.issubset(all_tools) and dangerous_tool in all_tools:
                errors.append(
                    WorkflowValidationError(
                        category=ErrorCategory.SECURITY,
                        severity=ErrorSeverity.CRITICAL,
                        message=f"Dangerous tool combination: {tool_set} + {dangerous_tool}",
                        location="workflow",
                        suggestion=f"Remove {dangerous_tool} or restructure workflow to avoid combination",
                    )
                )

        return errors

    def _check_airgapped_constraints(
        self, workflow: Dict[str, Any]
    ) -> List[WorkflowValidationError]:
        """Check airgapped mode constraints."""
        errors = []

        nodes = workflow.get("nodes", [])

        # Collect all tools
        all_tools = set()
        for node in nodes:
            if "tools" in node and node["tools"]:
                all_tools.update(node["tools"])

        # Check for network tools
        network_tools_found = all_tools & self.network_tools
        if network_tools_found:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SECURITY,
                    severity=ErrorSeverity.ERROR,
                    message=f"Network tools not allowed in airgapped mode: {network_tools_found}",
                    location="workflow",
                    suggestion="Remove network tools or disable airgapped mode",
                )
            )

        return errors

    def _check_infinite_loops(self, workflow: Dict[str, Any]) -> List[WorkflowValidationError]:
        """Check for potential infinite loops."""
        errors = []

        # Check if workflow has max_iterations set
        max_iterations = workflow.get("max_iterations")
        if max_iterations and max_iterations > self.max_iterations:
            errors.append(
                WorkflowValidationError(
                    category=ErrorCategory.SECURITY,
                    severity=ErrorSeverity.WARNING,
                    message=f"max_iterations {max_iterations} exceeds recommended limit {self.max_iterations}",
                    location="workflow.max_iterations",
                    suggestion=f"Consider reducing max_iterations to {self.max_iterations} or less",
                )
            )

        return errors


class WorkflowValidator:
    """Main validator orchestrating all 4 validation layers.

    This is the primary interface for workflow validation.

    Example:
        validator = WorkflowValidator(strict_mode=True)

        result = validator.validate(workflow_dict)

        if not result.is_valid:
            print(f"Found {len(result.all_errors)} errors")
            for error in result.critical_errors:
                print(f"  {error}")
    """

    def __init__(
        self,
        strict_mode: bool = True,
        tool_registry: Optional[Any] = None,
        handler_registry: Optional[Dict[str, Any]] = None,
        airgapped_mode: bool = False,
    ):
        """Initialize workflow validator.

        Args:
            strict_mode: If True, all errors fail validation
            tool_registry: Tool registry for tool availability checks
            handler_registry: Handler registry for handler availability checks
            airgapped_mode: Whether airgapped mode is enabled
        """
        self.strict_mode = strict_mode
        self.airgapped_mode = airgapped_mode

        # Initialize validators
        self.schema_validator = SchemaValidator(strict_mode=strict_mode)
        self.structure_validator = GraphStructureValidator()
        self.semantic_validator = SemanticValidator(
            tool_registry=tool_registry,
            handler_registry=handler_registry,
            strict_mode=strict_mode,
        )
        self.security_validator = SecurityValidator(airgapped_mode=airgapped_mode)

    def validate(
        self, workflow: Dict[str, Any], workflow_name: Optional[str] = None
    ) -> WorkflowGenerationValidationResult:
        """Validate workflow across all 4 layers.

        Args:
            workflow: Workflow definition dictionary or object with to_dict()
            workflow_name: Optional workflow name for reporting

        Returns:
            WorkflowGenerationValidationResult with all errors from all layers
        """
        # Convert workflow to dict if needed
        if hasattr(workflow, "to_dict"):
            workflow_dict = workflow.to_dict()
        elif isinstance(workflow, dict):
            workflow_dict = workflow
        else:
            return WorkflowGenerationValidationResult(
                is_valid=False,
                schema_errors=[
                    WorkflowValidationError(
                        category=ErrorCategory.SCHEMA,
                        severity=ErrorSeverity.CRITICAL,
                        message="Workflow must be a dict or have to_dict() method",
                        location="workflow",
                        suggestion="Provide workflow as dictionary",
                    )
                ],
                workflow_name=workflow_name,
            )

        # Run all 4 validation layers
        schema_errors = self.schema_validator.validate(workflow_dict)
        structure_errors = self.structure_validator.validate(workflow_dict)
        semantic_errors = self.semantic_validator.validate(workflow_dict)
        security_errors = self.security_validator.validate(workflow_dict)

        # Determine if valid (no critical or error-level issues)
        all_errors = schema_errors + structure_errors + semantic_errors + security_errors
        has_blocking_errors = any(
            e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR] for e in all_errors
        )

        return WorkflowGenerationValidationResult(
            is_valid=not has_blocking_errors,
            schema_errors=schema_errors,
            structure_errors=structure_errors,
            semantic_errors=semantic_errors,
            security_errors=security_errors,
            workflow_name=workflow_name,
        )

    def validate_layer(self, workflow: Dict[str, Any], layer: str) -> List[WorkflowValidationError]:
        """Validate a single layer.

        Useful for targeted validation.

        Args:
            workflow: Workflow definition dictionary
            layer: Layer name (schema, structure, semantic, security)

        Returns:
            List of validation errors from that layer

        Raises:
            ValueError: If layer name is invalid
        """
        # Convert workflow to dict if needed
        if hasattr(workflow, "to_dict"):
            workflow_dict = workflow.to_dict()
        else:
            workflow_dict = workflow

        if layer == "schema":
            return self.schema_validator.validate(workflow_dict)
        elif layer == "structure":
            return self.structure_validator.validate(workflow_dict)
        elif layer == "semantic":
            return self.semantic_validator.validate(workflow_dict)
        elif layer == "security":
            return self.security_validator.validate(workflow_dict)
        else:
            raise ValueError(
                f"Invalid layer: {layer}. " f"Valid layers: schema, structure, semantic, security"
            )


__all__ = [
    "WorkflowValidator",
    "SchemaValidator",
    "GraphStructureValidator",
    "SemanticValidator",
    "SecurityValidator",
    "RequirementValidator",  # Added for requirements extraction validation
]


# =============================================================================
# Requirement Validator (for WorkflowRequirements, not workflow definitions)
# =============================================================================


class RequirementValidator:
    """Validate workflow requirements for completeness and consistency.

    This is a separate validator for WorkflowRequirements objects (from the
    requirements extraction system), not workflow definitions.

    Checks:
    - Completeness: All required info present?
    - Consistency: No contradictions?
    - Feasibility: Can this be implemented?
    - Specificity: Enough detail for generation?

    Attributes:
        _min_tasks: Minimum number of tasks required
        _max_tasks: Maximum number of tasks (warning threshold)
        _min_description_length: Minimum characters for task descriptions

    Example:
        from victor.workflows.generation import RequirementValidator

        validator = RequirementValidator()
        result = validator.validate(requirements)

        if not result.is_valid:
            for error in result.errors:
                print(f"{error.severity}: {error.message}")
    """

    def __init__(
        self,
        min_tasks: int = 1,
        max_tasks: int = 20,
        min_description_length: int = 10,
    ):
        """Initialize requirement validator.

        Args:
            min_tasks: Minimum number of tasks required
            max_tasks: Maximum number of tasks before warning
            min_description_length: Minimum characters for task descriptions
        """
        self._min_tasks = min_tasks
        self._max_tasks = max_tasks
        self._min_description_length = min_description_length

        # Valid values for enum-like fields
        self._valid_task_types = {
            "agent",
            "compute",
            "condition",
            "transform",
            "parallel",
        }
        self._valid_agent_roles = {
            "researcher",
            "executor",
            "planner",
            "reviewer",
            "writer",
            "analyst",
            "coordinator",
        }
        self._valid_verticals = {
            "coding",
            "devops",
            "research",
            "rag",
            "dataanalysis",
            "benchmark",
        }
        self._valid_execution_orders = {
            "sequential",
            "parallel",
            "mixed",
            "conditional",
        }
        self._valid_cost_tiers = {"FREE", "LOW", "MEDIUM", "HIGH"}
        self._valid_retry_policies = {
            "retry",
            "fail_fast",
            "continue",
            "fallback",
        }
        self._valid_environments = {"local", "cloud", "sandbox"}

    def validate(self, requirements) -> "RequirementValidationResult":
        """Perform comprehensive validation.

        Args:
            requirements: WorkflowRequirements to validate

        Returns:
            RequirementWorkflowGenerationValidationResult with errors, warnings, and recommendations
        """
        # Import here to avoid circular dependency
        from victor.workflows.generation.requirements import (
            RequirementValidationResult,
            RequirementValidationError,
        )

        errors = []
        warnings = []
        recommendations = []

        # Completeness checks
        completeness_errors = self._check_completeness(requirements)
        errors.extend(completeness_errors)

        # Consistency checks
        consistency_errors = self._check_consistency(requirements)
        errors.extend(consistency_errors[0])  # Critical errors
        warnings.extend(consistency_errors[1])  # Warnings

        # Feasibility checks
        feasibility_errors = self._check_feasibility(requirements)
        errors.extend(feasibility_errors[0])  # Critical errors
        warnings.extend(feasibility_errors[1])  # Warnings

        # Specificity checks
        specificity_warnings = self._check_specificity(requirements)
        warnings.extend(specificity_warnings)

        # Generate recommendations
        recommendations = self._generate_recommendations(requirements, errors, warnings)

        # Compute score
        score = self._compute_score(requirements, errors, warnings)

        # Determine validity
        is_valid = len([e for e in errors if e.severity in ("critical", "error")]) == 0

        return RequirementValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            score=score,
        )

    def _check_completeness(self, requirements) -> List:
        """Check for missing required information."""
        from victor.workflows.generation.requirements import RequirementValidationError

        errors = []

        # Must have at least one task
        if len(requirements.functional.tasks) < self._min_tasks:
            errors.append(
                RequirementValidationError(
                    field="functional.tasks",
                    message=f"No tasks defined (minimum: {self._min_tasks})",
                    severity="critical",
                    suggestion="Add at least one task to the workflow",
                )
            )

        # Each task must have description
        for task in requirements.functional.tasks:
            if not task.description:
                errors.append(
                    RequirementValidationError(
                        field=f"functional.tasks.{task.id}",
                        message=f"Task '{task.id}' missing description",
                        severity="critical",
                        suggestion="Provide a description for this task",
                    )
                )
            elif len(task.description) < self._min_description_length:
                errors.append(
                    RequirementValidationError(
                        field=f"functional.tasks.{task.id}.description",
                        message=f"Task '{task.id}' description too short",
                        severity="error",
                        suggestion=f"Provide a more detailed description (min {self._min_description_length} chars)",
                    )
                )

            # Each task must have valid type
            if task.task_type not in self._valid_task_types:
                errors.append(
                    RequirementValidationError(
                        field=f"functional.tasks.{task.id}.task_type",
                        message=f"Invalid task type: '{task.task_type}'",
                        severity="critical",
                        suggestion=f"Use one of: {', '.join(self._valid_task_types)}",
                    )
                )

            # Agent tasks must have role
            if task.task_type == "agent" and not task.role:
                errors.append(
                    RequirementValidationError(
                        field=f"functional.tasks.{task.id}.role",
                        message=f"Agent task '{task.id}' missing role",
                        severity="critical",
                        suggestion=f"Specify agent role: {', '.join(self._valid_agent_roles)}",
                    )
                )
            elif task.task_type == "agent" and task.role not in self._valid_agent_roles:
                errors.append(
                    RequirementValidationError(
                        field=f"functional.tasks.{task.id}.role",
                        message=f"Invalid agent role: '{task.role}'",
                        severity="error",
                        suggestion=f"Use one of: {', '.join(self._valid_agent_roles)}",
                    )
                )

        # Branches must have conditions
        for branch in requirements.structural.branches:
            if not branch.condition:
                errors.append(
                    RequirementValidationError(
                        field=f"structural.branches.{branch.condition_id}",
                        message=f"Branch '{branch.condition_id}' missing condition",
                        severity="critical",
                        suggestion="Specify the condition for this branch",
                    )
                )

        # Must have context (vertical)
        if not requirements.context.vertical:
            errors.append(
                RequirementValidationError(
                    field="context.vertical",
                    message="Domain vertical not specified",
                    severity="error",
                    suggestion=f"Specify one of: {', '.join(self._valid_verticals)}",
                )
            )
        elif requirements.context.vertical not in self._valid_verticals:
            errors.append(
                RequirementValidationError(
                    field="context.vertical",
                    message=f"Invalid vertical: '{requirements.context.vertical}'",
                    severity="error",
                    suggestion=f"Use one of: {', '.join(self._valid_verticals)}",
                )
            )

        return errors

    def _check_consistency(self, requirements) -> tuple:
        """Check for contradictions.

        Returns:
            Tuple of (errors, warnings)
        """
        from victor.workflows.generation.requirements import RequirementValidationError

        errors = []
        warnings = []

        # Check for circular dependencies
        dependencies = requirements.structural.dependencies
        if dependencies:
            cycles = self._detect_cycles(dependencies)
            if cycles:
                cycle_path = " → ".join(cycles[0])
                errors.append(
                    RequirementValidationError(
                        field="structural.dependencies",
                        message=f"Circular dependency: {cycle_path}",
                        severity="critical",
                        suggestion="Remove one dependency to break the cycle",
                    )
                )

        # Check: Parallel execution with dependencies
        if requirements.structural.execution_order == "parallel" and dependencies:
            warnings.append(
                RequirementValidationError(
                    field="structural.execution_order",
                    message="Parallel execution with dependencies may be inefficient",
                    severity="warning",
                    suggestion="Consider sequential execution or remove dependencies",
                )
            )

        # Check: Conflicting quality constraints
        if requirements.quality.max_duration_seconds and len(requirements.functional.tasks) > 0:
            min_time = len(requirements.functional.tasks) * 30  # 30s per task min
            if requirements.quality.max_duration_seconds < min_time:
                errors.append(
                    RequirementValidationError(
                        field="quality.max_duration_seconds",
                        message=f"Timeout ({requirements.quality.max_duration_seconds}s) too short for {len(requirements.functional.tasks)} tasks (minimum: {min_time}s)",
                        severity="error",
                        suggestion=f"Increase timeout to at least {min_time}s",
                    )
                )

        # Check: Invalid enum values
        if requirements.structural.execution_order not in self._valid_execution_orders:
            errors.append(
                RequirementValidationError(
                    field="structural.execution_order",
                    message=f"Invalid execution order: '{requirements.structural.execution_order}'",
                    severity="error",
                    suggestion=f"Use one of: {', '.join(self._valid_execution_orders)}",
                )
            )

        if requirements.quality.max_cost_tier not in self._valid_cost_tiers:
            errors.append(
                RequirementValidationError(
                    field="quality.max_cost_tier",
                    message=f"Invalid cost tier: '{requirements.quality.max_cost_tier}'",
                    severity="error",
                    suggestion=f"Use one of: {', '.join(self._valid_cost_tiers)}",
                )
            )

        if requirements.quality.retry_policy not in self._valid_retry_policies:
            errors.append(
                RequirementValidationError(
                    field="quality.retry_policy",
                    message=f"Invalid retry policy: '{requirements.quality.retry_policy}'",
                    severity="error",
                    suggestion=f"Use one of: {', '.join(self._valid_retry_policies)}",
                )
            )

        if requirements.context.environment not in self._valid_environments:
            errors.append(
                RequirementValidationError(
                    field="context.environment",
                    message=f"Invalid environment: '{requirements.context.environment}'",
                    severity="error",
                    suggestion=f"Use one of: {', '.join(self._valid_environments)}",
                )
            )

        return errors, warnings

    def _check_feasibility(self, requirements) -> tuple:
        """Check if requirements can be implemented.

        Returns:
            Tuple of (errors, warnings)
        """
        from victor.workflows.generation.requirements import RequirementValidationError

        errors = []
        warnings = []

        # Check: Too many tasks for single workflow
        if len(requirements.functional.tasks) > self._max_tasks:
            warnings.append(
                RequirementValidationError(
                    field="functional.tasks",
                    message=f"Too many tasks ({len(requirements.functional.tasks)}), consider splitting into multiple workflows",
                    severity="warning",
                    suggestion=f"Keep workflows under {self._max_tasks} tasks for maintainability",
                )
            )

        # Check: Complex structures
        if len(requirements.structural.branches) > 5:
            warnings.append(
                RequirementValidationError(
                    field="structural.branches",
                    message=f"Too many branches ({len(requirements.structural.branches)}), workflow may be hard to debug",
                    severity="warning",
                    suggestion="Simplify logic or split into sub-workflows",
                )
            )

        # Check: Task dependencies reference valid tasks
        task_ids = {task.id for task in requirements.functional.tasks}
        for task_id, deps in requirements.structural.dependencies.items():
            if task_id not in task_ids:
                errors.append(
                    RequirementValidationError(
                        field=f"structural.dependencies.{task_id}",
                        message=f"Task '{task_id}' not found in dependencies",
                        severity="error",
                        suggestion="Remove or fix dependency reference",
                    )
                )
            for dep_id in deps:
                if dep_id not in task_ids:
                    errors.append(
                        RequirementValidationError(
                            field=f"structural.dependencies.{task_id}",
                            message=f"Dependency '{dep_id}' not found",
                            severity="error",
                            suggestion="Remove invalid dependency",
                        )
                    )

        # Check: Branch references valid tasks
        for branch in requirements.structural.branches:
            if branch.true_branch != "end" and branch.true_branch not in task_ids:
                errors.append(
                    RequirementValidationError(
                        field=f"structural.branches.{branch.condition_id}.true_branch",
                        message=f"Branch references unknown task: '{branch.true_branch}'",
                        severity="error",
                        suggestion="Fix branch target or set to 'end'",
                    )
                )
            if branch.false_branch != "end" and branch.false_branch not in task_ids:
                errors.append(
                    RequirementValidationError(
                        field=f"structural.branches.{branch.condition_id}.false_branch",
                        message=f"Branch references unknown task: '{branch.false_branch}'",
                        severity="error",
                        suggestion="Fix branch target or set to 'end'",
                    )
                )

        return errors, warnings

    def _check_specificity(self, requirements) -> List:
        """Check if requirements are specific enough.

        Args:
            requirements: Requirements to check

        Returns:
            List of warnings
        """
        from victor.workflows.generation.requirements import RequirementValidationError
        import re

        warnings = []

        # Check for vague task descriptions
        vague_patterns = [
            r"\bsomething\b",
            r"\bstuff\b",
            r"\bthings\b",
            r"\bhandle it\b",
            r"\bprocess\b",
        ]

        for task in requirements.functional.tasks:
            for pattern in vague_patterns:
                if re.search(pattern, task.description, re.IGNORECASE):
                    warnings.append(
                        RequirementValidationError(
                            field=f"functional.tasks.{task.id}.description",
                            message=f"Task '{task.id}' has vague description",
                            severity="warning",
                            suggestion="Be more specific about what this task does",
                        )
                    )
                    break

        # Check: Success criteria measurable
        for criteria in requirements.functional.success_criteria:
            # Check for measurable indicators
            if not any(
                indicator in criteria.lower()
                for indicator in ["%", "pass", "fail", "score", "error", "success"]
            ):
                warnings.append(
                    RequirementValidationError(
                        field="functional.success_criteria",
                        message=f"Success criteria may not be measurable: '{criteria}'",
                        severity="info",
                        suggestion="Add specific metrics or outcomes",
                    )
                )

        return warnings

    def _generate_recommendations(self, requirements, errors, warnings) -> List[str]:
        """Generate recommendations for improvement.

        Args:
            requirements: Requirements being validated
            errors: Errors found
            warnings: Warnings found

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Based on error count
        if len(errors) > 5:
            recommendations.append("Consider breaking this into multiple workflows")

        # Based on task count
        if len(requirements.functional.tasks) > 15:
            recommendations.append("Workflow has many tasks - consider sub-workflows")

        # Based on structure complexity
        if len(requirements.structural.branches) > 3:
            recommendations.append("Complex branching logic - add comments/documentation")

        # Based on quality constraints
        if not requirements.quality.max_duration_seconds:
            recommendations.append("Consider adding a timeout for safety")

        if not requirements.functional.success_criteria:
            recommendations.append("Add success criteria to verify workflow completion")

        # Based on vertical
        if requirements.context.vertical == "coding" and not requirements.context.project_context:
            recommendations.append("Add project context (language, framework) for better results")

        return recommendations

    def _compute_score(self, requirements, errors, warnings) -> float:
        """Compute overall quality score (0.0-1.0).

        Args:
            requirements: Requirements being scored
            errors: Errors found
            warnings: Warnings found

        Returns:
            Quality score
        """
        # Base score
        score = 1.0

        # Deduct for errors
        for error in errors:
            if error.severity == "critical":
                score -= 0.2
            elif error.severity == "error":
                score -= 0.1

        # Deduct for warnings
        for warning in warnings:
            if warning.severity == "warning":
                score -= 0.05
            elif warning.severity == "info":
                score -= 0.01

        # Bonus for good practices
        if requirements.functional.success_criteria:
            score += 0.05

        if requirements.quality.max_duration_seconds:
            score += 0.05

        if requirements.context.project_context:
            score += 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _detect_cycles(self, dependencies) -> List:
        """Detect cycles in dependency graph using DFS.

        Args:
            dependencies: Adjacency list (task_id -> [dep_ids])

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in dependencies:
            if node not in visited:
                dfs(node)

        return cycles


# Update __all__ to include RequirementValidator
__all__.append("RequirementValidator")
