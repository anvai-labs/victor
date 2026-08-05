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

"""AdaptableGraph wrapper for runtime workflow modification.

This module provides the AdaptableGraph class that wraps CompiledGraph
and enables safe runtime modifications.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from victor.framework.adaptation.types import (
    AdaptationCheckpoint,
    AdaptationConfig,
    AdaptationImpact,
    AdaptationResult,
    AdaptationValidationResult,
    GraphModification,
    ModificationType,
    RiskLevel,
)

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class AdaptableGraph:
    """Wrapper around CompiledGraph supporting safe runtime modifications.

    AdaptableGraph provides:
    - Safe modification operations with validation
    - Automatic checkpointing for rollback
    - Impact analysis before modifications
    - Rate limiting and circuit breaker
    - Audit trail of all modifications

    Example:
        # Wrap existing compiled graph
        adaptable = AdaptableGraph(compiled_graph)

        # Configure adaptation behavior
        adaptable.configure(AdaptationConfig(
            enable_auto_checkpoint=True,
            max_risk_level=RiskLevel.MEDIUM,
        ))

        # Apply modification
        modification = GraphModification(
            modification_type=ModificationType.ADD_RETRY,
            description="Add retry to flaky node",
            target_node="api_call",
            data={"max_retries": 3, "backoff": "exponential"},
        )

        result = await adaptable.adapt(modification)
        if not result.success:
            print(f"Adaptation failed: {result.error}")

    Attributes:
        graph: The underlying compiled graph
        config: Adaptation configuration
        checkpoints: List of checkpoints for rollback
        history: List of applied modifications
    """

    def __init__(
        self,
        graph: "Runnable",
        config: Optional[AdaptationConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize AdaptableGraph.

        Args:
            graph: Compiled LangGraph Runnable to wrap
            config: Adaptation configuration
            metadata: Optional graph metadata
        """
        self._graph = graph
        self._config = config or AdaptationConfig()
        self._metadata = metadata or {}

        # Tracking state
        self._checkpoints: List[AdaptationCheckpoint] = []
        self._history: List[AdaptationResult] = []
        self._modification_count = 0

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None

    @property
    def graph(self) -> "Runnable":
        """Get the underlying compiled graph."""
        return self._graph

    @property
    def config(self) -> AdaptationConfig:
        """Get adaptation configuration."""
        return self._config

    @property
    def checkpoints(self) -> List[AdaptationCheckpoint]:
        """Get list of checkpoints."""
        return list(self._checkpoints)

    @property
    def history(self) -> List[AdaptationResult]:
        """Get modification history."""
        return list(self._history)

    def configure(self, config: AdaptationConfig) -> None:
        """Update adaptation configuration.

        Args:
            config: New configuration
        """
        self._config = config
        logger.debug(f"Updated adaptation config: {config}")

    async def adapt(
        self,
        modification: GraphModification,
        skip_validation: bool = False,
        skip_impact_analysis: bool = False,
    ) -> AdaptationResult:
        """Apply an adaptation to the graph.

        Args:
            modification: Modification to apply
            skip_validation: Skip validation step
            skip_impact_analysis: Skip impact analysis

        Returns:
            AdaptationResult with success status and details
        """
        start_time = time.time()

        # Check circuit breaker
        if self._is_circuit_open():
            return AdaptationResult(
                success=False,
                modification=modification,
                error="Circuit breaker open - too many recent failures",
            )

        # Check rate limit
        if self._modification_count >= self._config.max_modifications_per_execution:
            return AdaptationResult(
                success=False,
                modification=modification,
                error=f"Rate limit exceeded ({self._config.max_modifications_per_execution})",
            )

        logger.info(f"Applying adaptation: {modification.description}")

        try:
            # Step 1: Validate
            if self._config.require_validation and not skip_validation:
                validation = await self.validate(modification)
                if not validation.is_valid:
                    return AdaptationResult(
                        success=False,
                        modification=modification,
                        error=f"Validation failed: {validation.errors}",
                    )

            # Step 2: Impact analysis
            impact = None
            if self._config.require_impact_analysis and not skip_impact_analysis:
                impact = await self.analyze_impact(modification)
                if impact.risk_level.value > self._config.max_risk_level.value:
                    return AdaptationResult(
                        success=False,
                        modification=modification,
                        impact=impact,
                        error=f"Risk level {impact.risk_level.value} exceeds max {self._config.max_risk_level.value}",
                    )

            # Step 3: Create checkpoint
            checkpoint_id = None
            if self._config.enable_auto_checkpoint:
                checkpoint_id = await self.create_checkpoint(f"Before: {modification.description}")

            # Step 4: Apply modification
            try:
                await self._apply_modification(modification)
                self._modification_count += 1
                self._consecutive_failures = 0

                execution_time = (time.time() - start_time) * 1000

                result = AdaptationResult(
                    success=True,
                    modification=modification,
                    checkpoint_id=checkpoint_id,
                    impact=impact,
                    execution_time_ms=execution_time,
                )

                self._history.append(result)
                logger.info(
                    f"Adaptation successful: {modification.description} "
                    f"({execution_time:.1f}ms)"
                )

                return result

            except Exception as e:
                # Rollback on error if configured
                if self._config.rollback_on_error and checkpoint_id:
                    logger.warning(f"Rolling back due to error: {e}")
                    await self.rollback_to(checkpoint_id)

                raise

        except Exception as e:
            self._consecutive_failures += 1
            self._check_circuit_breaker()

            execution_time = (time.time() - start_time) * 1000

            result = AdaptationResult(
                success=False,
                modification=modification,
                execution_time_ms=execution_time,
                error=str(e),
            )

            self._history.append(result)
            logger.error(f"Adaptation failed: {e}")

            return result

    async def validate(self, modification: GraphModification) -> AdaptationValidationResult:
        """Validate a proposed modification.

        Args:
            modification: Modification to validate

        Returns:
            AdaptationValidationResult with validation details
        """
        errors = []
        warnings = []

        # Basic validation
        if not modification.description:
            warnings.append("Modification has no description")

        # Type-specific validation
        if modification.modification_type in [
            ModificationType.ADD_NODE,
            ModificationType.REMOVE_NODE,
            ModificationType.MODIFY_NODE,
        ]:
            if not modification.target_node:
                errors.append("Node modification requires target_node")

        if modification.modification_type in [
            ModificationType.ADD_EDGE,
            ModificationType.REMOVE_EDGE,
            ModificationType.MODIFY_EDGE,
        ]:
            if not modification.target_edge:
                errors.append("Edge modification requires target_edge")

        # Validate node exists for modifications (not additions)
        if modification.modification_type in [
            ModificationType.REMOVE_NODE,
            ModificationType.MODIFY_NODE,
        ]:
            if modification.target_node and not self._node_exists(modification.target_node):
                errors.append(f"Target node does not exist: {modification.target_node}")

        return AdaptationValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def analyze_impact(self, modification: GraphModification) -> AdaptationImpact:
        """Analyze impact of a proposed modification.

        Args:
            modification: Modification to analyze

        Returns:
            AdaptationImpact with analysis details
        """
        affects_nodes = []
        affects_edges = []
        execution_path_change = False
        performance_impact = "neutral"
        risk_level = RiskLevel.LOW

        # Analyze based on modification type
        if modification.modification_type == ModificationType.ADD_NODE:
            affects_nodes = [modification.target_node] if modification.target_node else []
            risk_level = RiskLevel.MEDIUM
            execution_path_change = True

        elif modification.modification_type == ModificationType.REMOVE_NODE:
            affects_nodes = [modification.target_node] if modification.target_node else []
            risk_level = RiskLevel.HIGH
            execution_path_change = True
            performance_impact = "positive" if modification.data.get("is_redundant") else "negative"

        elif modification.modification_type == ModificationType.ADD_PARALLELIZATION:
            affects_nodes = modification.data.get("nodes_to_parallelize", [])
            risk_level = RiskLevel.MEDIUM
            execution_path_change = True
            performance_impact = "positive"

        elif modification.modification_type == ModificationType.ADD_RETRY:
            affects_nodes = [modification.target_node] if modification.target_node else []
            risk_level = RiskLevel.LOW
            performance_impact = "neutral"

        elif modification.modification_type == ModificationType.ADD_CIRCUIT_BREAKER:
            affects_nodes = [modification.target_node] if modification.target_node else []
            risk_level = RiskLevel.LOW
            performance_impact = "positive"

        elif modification.modification_type == ModificationType.ADD_CACHING:
            affects_nodes = [modification.target_node] if modification.target_node else []
            risk_level = RiskLevel.LOW
            performance_impact = "positive"

        elif modification.modification_type == ModificationType.REPLACE_SUBGRAPH:
            affects_nodes = modification.data.get("nodes_to_replace", [])
            risk_level = RiskLevel.CRITICAL
            execution_path_change = True

        # Handle edges
        if modification.target_edge:
            affects_edges = [modification.target_edge]

        return AdaptationImpact(
            affects_nodes=affects_nodes,
            affects_edges=affects_edges,
            execution_path_change=execution_path_change,
            performance_impact=performance_impact,
            risk_level=risk_level,
        )

    async def create_checkpoint(self, description: str = "") -> str:
        """Create a checkpoint for rollback.

        Args:
            description: Checkpoint description

        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid4())[:8]

        # Serialize graph state (implementation depends on graph type)
        graph_state = self._serialize_graph_state()

        checkpoint = AdaptationCheckpoint(
            id=checkpoint_id,
            graph_state=graph_state,
            description=description,
        )

        self._checkpoints.append(checkpoint)

        # Limit checkpoint history
        max_checkpoints = 20
        if len(self._checkpoints) > max_checkpoints:
            self._checkpoints = self._checkpoints[-max_checkpoints:]

        logger.debug(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id

    async def rollback_to(self, checkpoint_id: str) -> bool:
        """Rollback to a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to rollback to

        Returns:
            True if rollback successful
        """
        checkpoint = next(
            (cp for cp in self._checkpoints if cp.id == checkpoint_id),
            None,
        )

        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        try:
            self._restore_graph_state(checkpoint.graph_state)
            logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def rollback_last(self) -> bool:
        """Rollback the last modification.

        Returns:
            True if rollback successful
        """
        if not self._checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False

        last_checkpoint = self._checkpoints[-1]
        return await self.rollback_to(last_checkpoint.id)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _node_exists(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        # This depends on the actual graph structure
        # For now, return True (validation depends on graph implementation)
        if hasattr(self._graph, "nodes"):
            return node_id in self._graph.nodes
        return True

    def _serialize_graph_state(self) -> Dict[str, Any]:
        """Serialize graph state for checkpointing."""
        # This is a simplified implementation
        # Full implementation would depend on graph structure
        return {
            "metadata": copy.deepcopy(self._metadata),
            "modification_count": self._modification_count,
            "timestamp": datetime.now().isoformat(),
        }

    def _restore_graph_state(self, state: Dict[str, Any]) -> None:
        """Restore graph state from checkpoint."""
        self._metadata = state.get("metadata", {})
        self._modification_count = state.get("modification_count", 0)

    async def _apply_modification(self, modification: GraphModification) -> None:
        """Apply a modification to the graph.

        This is a hook for actual graph modification logic.
        Implementation depends on the graph type.
        """
        logger.debug(f"Applying modification: {modification.modification_type.value}")

        # Record modification in metadata
        if "modifications" not in self._metadata:
            self._metadata["modifications"] = []

        self._metadata["modifications"].append(
            {
                "id": modification.id,
                "type": modification.modification_type.value,
                "description": modification.description,
                "target_node": modification.target_node,
                "target_edge": modification.target_edge,
                "applied_at": datetime.now().isoformat(),
            }
        )

        # Actual graph modification would happen here
        # This depends on the StateGraph/CompiledGraph implementation

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_open:
            return False

        # Check if circuit should close
        if self._circuit_open_until and datetime.now() > self._circuit_open_until:
            self._circuit_open = False
            self._circuit_open_until = None
            self._consecutive_failures = 0
            logger.info("Circuit breaker closed")
            return False

        return True

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should open."""
        failure_threshold = 3
        if self._consecutive_failures >= failure_threshold:
            self._circuit_open = True
            # Open for 60 seconds
            from datetime import timedelta

            self._circuit_open_until = datetime.now() + timedelta(seconds=60)
            logger.warning("Circuit breaker opened due to consecutive failures")


__all__ = [
    "AdaptableGraph",
]
