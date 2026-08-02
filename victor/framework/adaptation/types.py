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

"""Type definitions for dynamic workflow adaptation.

This module provides the core data types used throughout the adaptation
framework, including modification types, results, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


class ModificationType(Enum):
    """Types of graph modifications."""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MODIFY_NODE = "modify_node"
    MODIFY_EDGE = "modify_edge"
    REPLACE_SUBGRAPH = "replace_subgraph"
    ADD_PARALLELIZATION = "add_parallelization"
    ADD_RETRY = "add_retry"
    ADD_CIRCUIT_BREAKER = "add_circuit_breaker"
    ADD_CACHING = "add_caching"


class AdaptationTrigger(Enum):
    """What triggered the adaptation."""

    PERFORMANCE = "performance"  # Performance metrics
    ERROR = "error"  # Error rates
    TIMEOUT = "timeout"  # Timeouts
    RESOURCE = "resource"  # Resource constraints
    MANUAL = "manual"  # Manual trigger
    SCHEDULED = "scheduled"  # Scheduled adaptation
    LEARNING = "learning"  # ML-based learning


class RiskLevel(Enum):
    """Risk level of modification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GraphModification:
    """A proposed modification to a workflow graph.

    Attributes:
        modification_type: Type of modification
        description: Human-readable description
        target_node: Target node ID (for node modifications)
        target_edge: Target edge (source, target) (for edge modifications)
        data: Modification-specific data
        trigger: What triggered this adaptation
        priority: Priority (higher = more urgent)
        metadata: Additional metadata
    """

    modification_type: ModificationType
    description: str
    target_node: Optional[str] = None
    target_edge: Optional[tuple[str, str]] = None
    data: Dict[str, Any] = field(default_factory=dict)
    trigger: AdaptationTrigger = AdaptationTrigger.MANUAL
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationValidationResult:
    """Result of adaptation validation operation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        suggestions: Suggested fixes
        metadata: Additional metadata
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "AdaptationValidationResult") -> "AdaptationValidationResult":
        """Merge two validation results."""
        return AdaptationValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            suggestions=self.suggestions + other.suggestions,
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class AdaptationImpact:
    """Impact analysis for graph modification.

    Attributes:
        affects_nodes: List of affected node IDs
        affects_edges: List of affected edges
        execution_path_change: Whether execution path changes
        performance_impact: Expected performance impact
        risk_level: Risk level of modification
        estimated_overhead_ms: Estimated overhead in milliseconds
        details: Additional details
    """

    affects_nodes: List[str] = field(default_factory=list)
    affects_edges: List[tuple[str, str]] = field(default_factory=list)
    execution_path_change: bool = False
    performance_impact: str = "neutral"  # "positive", "neutral", "negative"
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_overhead_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationCheckpoint:
    """Checkpoint for rollback support.

    Attributes:
        id: Checkpoint ID
        graph_state: Serialized graph state
        created_at: When checkpoint was created
        description: Description of graph state
        metadata: Additional metadata
    """

    id: str
    graph_state: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of applying an adaptation.

    Attributes:
        success: Whether adaptation succeeded
        modification: The modification that was applied
        checkpoint_id: Checkpoint created before modification
        impact: Impact analysis
        execution_time_ms: Time to apply modification
        error: Error message if failed
    """

    success: bool
    modification: GraphModification
    checkpoint_id: Optional[str] = None
    impact: Optional[AdaptationImpact] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AdaptationConfig:
    """Configuration for adaptation behavior.

    Attributes:
        enable_auto_checkpoint: Create checkpoints automatically
        max_modifications_per_execution: Rate limit
        require_validation: Always validate before applying
        require_impact_analysis: Always analyze impact first
        max_risk_level: Maximum allowed risk level
        rollback_on_error: Automatically rollback on errors
    """

    enable_auto_checkpoint: bool = True
    max_modifications_per_execution: int = 10
    require_validation: bool = True
    require_impact_analysis: bool = True
    max_risk_level: RiskLevel = RiskLevel.MEDIUM
    rollback_on_error: bool = True


@dataclass
class AdaptationStrategy:
    """Strategy for when/how to adapt.

    Attributes:
        name: Strategy name
        description: Human-readable description
        trigger_condition: Condition for triggering adaptation
        modification_generator: Function to generate modifications
        priority: Strategy priority (higher = earlier)
        enabled: Whether strategy is enabled
    """

    name: str
    description: str
    trigger_condition: Callable[[Dict[str, Any]], bool]
    modification_generator: Callable[[Dict[str, Any]], List[GraphModification]]
    priority: int = 0
    enabled: bool = True


__all__ = [
    "ModificationType",
    "AdaptationTrigger",
    "RiskLevel",
    "GraphModification",
    "AdaptationValidationResult",
    "AdaptationImpact",
    "AdaptationCheckpoint",
    "AdaptationResult",
    "AdaptationConfig",
    "AdaptationStrategy",
]
