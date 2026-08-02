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

"""Focused protocols for dynamic workflow adaptation.

This module defines ISP-compliant protocols for runtime workflow graph
modification. Each protocol has a single, well-defined responsibility.

Protocol Separation:
- GraphValidator: Validation only - no modification logic
- GraphApplier: Application only - assumes already validated
- GraphRollback: Rollback only - separate concern
- ImpactAnalyzer: Impact analysis - separate concern

Following the refinement plan, these protocols extend existing Victor
infrastructure and follow SOLID principles.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


# =============================================================================
# Data Classes
# =============================================================================


class ModificationType:
    """Types of graph modifications."""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MODIFY_NODE = "modify_node"
    MODIFY_EDGE = "modify_edge"
    REPLACE_SUBGRAPH = "replace_subgraph"


class GraphModification:
    """A proposed modification to a workflow graph."""

    def __init__(
        self,
        modification_type: str,
        description: str,
        node_data: Optional[Dict[str, Any]] = None,
        edge_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize graph modification.

        Args:
            modification_type: Type of modification (from ModificationType)
            description: Human-readable description
            node_data: Node-related data (for node modifications)
            edge_data: Edge-related data (for edge modifications)
            metadata: Optional additional metadata
        """
        self.modification_type = modification_type
        self.description = description
        self.node_data = node_data or {}
        self.edge_data = edge_data or {}
        self.metadata = metadata or {}
        self.id = f"{modification_type}_{id(self)}"


class AdaptationValidationResult:
    """Result of adaptation validation operation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
            metadata: Optional additional metadata
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}


class CompiledGraph:
    """Compiled workflow graph (alias for type clarity)."""

    def __init__(self, graph: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize compiled graph wrapper.

        Args:
            graph: Compiled LangGraph Runnable
            metadata: Optional graph metadata
        """
        self.graph = graph
        self.metadata = metadata or {}


class ModifiedGraph:
    """Result of graph modification."""

    def __init__(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
        changes: Dict[str, Any],
    ) -> None:
        """Initialize modified graph.

        Args:
            graph: Modified compiled graph
            modification: Modification that was applied
            changes: Description of changes made
        """
        self.graph = graph
        self.modification = modification
        self.changes = changes


class AdaptationImpact:
    """Impact analysis for graph modification."""

    def __init__(
        self,
        affects_nodes: List[str],
        affects_edges: List[tuple[str, str]],
        execution_path_change: bool,
        performance_impact: str,  # "positive", "neutral", "negative"
        risk_level: str,  # "low", "medium", "high"
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize adaptation impact.

        Args:
            affects_nodes: List of affected node IDs
            affects_edges: List of affected edges (source, target)
            execution_path_change: Whether execution path changes
            performance_impact: Expected performance impact
            risk_level: Risk level of modification
            details: Optional additional details
        """
        self.affects_nodes = affects_nodes
        self.affects_edges = affects_edges
        self.execution_path_change = execution_path_change
        self.performance_impact = performance_impact
        self.risk_level = risk_level
        self.details = details or {}


# =============================================================================
# Graph Validation Protocol
# =============================================================================


@runtime_checkable
class GraphValidator(Protocol):
    """Graph validation - single responsibility.

    Validates proposed modifications to workflow graphs.

    Single Responsibility: Validation ONLY (no modification logic).

    Key Methods:
    - validate: Validate modification against graph
    - validate_syntax: Validate syntax of modification data
    - validate_semantics: Validate semantic correctness

    This protocol is SEPARATE from GraphApplier because:
    - Validation can be swapped independently
    - Multiple validators can be chained
    - Validation failures should not modify graph
    - Follows Single Responsibility Principle

    Reuse Strategy:
    - Extend victor.core.workflow_validation.WorkflowValidator
    - Reuse existing validation infrastructure
    - Add adaptation-specific validation rules
    """

    async def validate(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> AdaptationValidationResult:
        """Validate modification against graph.

        Args:
            graph: Compiled graph to validate against
            modification: Proposed modification

        Returns:
            AdaptationValidationResult with is_valid flag and any errors
        """
        ...

    async def validate_syntax(
        self,
        modification: GraphModification,
    ) -> AdaptationValidationResult:
        """Validate syntax of modification data.

        Args:
            modification: Modification to validate

        Returns:
            AdaptationValidationResult for syntax only
        """
        ...

    async def validate_semantics(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> AdaptationValidationResult:
        """Validate semantic correctness.

        Args:
            graph: Compiled graph
            modification: Modification to validate

        Returns:
            AdaptationValidationResult for semantic validation
        """
        ...


# =============================================================================
# Graph Application Protocol
# =============================================================================


@runtime_checkable
class GraphApplier(Protocol):
    """Graph modification application - single responsibility.

    Applies validated modifications to workflow graphs.

    Single Responsibility: Application ONLY (assumes already validated).

    Key Methods:
    - apply: Apply modification to graph
    - apply_batch: Apply multiple modifications atomically
    - preview: Preview changes without applying

    This protocol is SEPARATE from GraphValidator because:
    - Application logic independent of validation
    - Can be confident validation already happened
    - Different application strategies (immediate, batch, etc.)
    - Follows Single Responsibility Principle
    """

    async def apply(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> ModifiedGraph:
        """Apply modification to graph.

        Args:
            graph: Compiled graph to modify
            modification: Validated modification to apply

        Returns:
            ModifiedGraph with changes applied

        Raises:
            ValueError: If modification invalid (should be validated first)
        """
        ...

    async def apply_batch(
        self,
        graph: CompiledGraph,
        modifications: List[GraphModification],
    ) -> ModifiedGraph:
        """Apply multiple modifications atomically.

        Args:
            graph: Compiled graph to modify
            modifications: List of validated modifications

        Returns:
            ModifiedGraph with all changes applied

        Raises:
            ValueError: If any modification invalid or batch fails
        """
        ...

    async def preview(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> Dict[str, Any]:
        """Preview changes without applying.

        Args:
            graph: Compiled graph
            modification: Modification to preview

        Returns:
            Dictionary describing changes that would be made
        """
        ...


# =============================================================================
# Graph Rollback Protocol
# =============================================================================


@runtime_checkable
class GraphRollback(Protocol):
    """Graph modification rollback - single responsibility.

    Rolls back applied modifications to workflow graphs.

    Single Responsibility: Rollback ONLY (separate concern).

    Key Methods:
    - rollback: Rollback a specific modification
    - rollback_to_checkpoint: Rollback to saved checkpoint
    - create_checkpoint: Create rollback checkpoint

    This protocol is SEPARATE from GraphApplier because:
    - Rollback logic independent of application
    - Different rollback strategies (immediate, deferred, etc.)
    - Can have multiple rollback implementations
    - Follows Single Responsibility Principle
    """

    async def rollback(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> CompiledGraph:
        """Rollback a specific modification.

        Args:
            graph: Current graph state
            modification: Modification to rollback

        Returns:
            CompiledGraph with modification rolled back

        Raises:
            ValueError: If modification cannot be rolled back
        """
        ...

    async def rollback_to_checkpoint(
        self,
        graph: CompiledGraph,
        checkpoint_id: str,
    ) -> CompiledGraph:
        """Rollback to saved checkpoint.

        Args:
            graph: Current graph state
            checkpoint_id: Checkpoint ID to rollback to

        Returns:
            CompiledGraph rolled back to checkpoint
        """
        ...

    async def create_checkpoint(
        self,
        graph: CompiledGraph,
    ) -> str:
        """Create rollback checkpoint.

        Args:
            graph: Current graph state

        Returns:
            Checkpoint ID for later rollback
        """
        ...


# =============================================================================
# Impact Analysis Protocol
# =============================================================================


@runtime_checkable
class ImpactAnalyzer(Protocol):
    """Impact analysis for graph modifications.

    Single Responsibility: Analyze impact of proposed modifications.

    Key Methods:
    - calculate_impact: Calculate impact of modification
    - compare_states: Compare before/after states
    - estimate_performance: Estimate performance impact

    This protocol is SEPARATE from GraphValidator because:
    - Analysis is different from validation (can be valid but high impact)
    - Impact estimation independent of correctness
    - Can be used for planning before validation
    - Follows Single Responsibility Principle
    """

    async def calculate_impact(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> AdaptationImpact:
        """Calculate impact of modification.

        Args:
            graph: Current graph state
            modification: Proposed modification

        Returns:
            AdaptationImpact with analysis details
        """
        ...

    async def compare_states(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> AdaptationImpact:
        """Compare before/after states.

        Args:
            before: State before modification
            after: State after modification

        Returns:
            AdaptationImpact with comparison details
        """
        ...

    async def estimate_performance(
        self,
        graph: CompiledGraph,
        modification: GraphModification,
    ) -> str:
        """Estimate performance impact.

        Args:
            graph: Current graph state
            modification: Proposed modification

        Returns:
            Performance impact: "positive", "neutral", or "negative"
        """
        ...
