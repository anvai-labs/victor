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

"""Type definitions for emergent collaboration patterns.

This module provides the core data types used throughout the patterns
framework, including patterns, contexts, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from victor.core.constants import DEFAULT_VERTICAL


class PatternStatus(Enum):
    """Status of a collaboration pattern."""

    DISCOVERED = "discovered"  # Newly mined
    VALIDATED = "validated"  # Passed validation
    APPROVED = "approved"  # Human approved
    ACTIVE = "active"  # In active use
    DEPRECATED = "deprecated"  # No longer recommended
    FAILED = "failed"  # Failed validation


class PatternCategory(Enum):
    """Category of collaboration pattern."""

    SEQUENTIAL = "sequential"  # Tasks in sequence
    PARALLEL = "parallel"  # Tasks in parallel
    HIERARCHICAL = "hierarchical"  # Supervisor-specialist
    COLLABORATIVE = "collaborative"  # Peer discussion
    COMPETITIVE = "competitive"  # Multiple attempts
    MIXED = "mixed"  # Combination


@dataclass
class PatternValidationResult:
    """Result of pattern validation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        quality_score: Quality score (0-1)
        safety_score: Safety score (0-1)
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    safety_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMetrics:
    """Metrics for a collaboration pattern.

    Attributes:
        usage_count: Number of times pattern used
        success_count: Number of successful uses
        avg_duration_ms: Average execution duration
        avg_cost: Average execution cost
        last_used: When pattern was last used
    """

    usage_count: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0.0
    avg_cost: float = 0.0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


@dataclass
class TaskContext:
    """Context for pattern recommendation.

    Attributes:
        task_description: What needs to be done
        required_capabilities: Required skills/capabilities
        vertical: Domain vertical (coding, devops, etc.)
        complexity: Task complexity level
        constraints: Additional constraints
    """

    task_description: str
    required_capabilities: List[str] = field(default_factory=list)
    vertical: str = DEFAULT_VERTICAL
    complexity: str = "medium"  # low, medium, high
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationPattern:
    """A discovered or defined collaboration pattern.

    Represents a team formation and collaboration style that can be
    applied to tasks.

    Attributes:
        id: Unique pattern identifier
        name: Human-readable name
        description: Pattern description
        category: Pattern category
        participants: Participant specifications
        workflow: Workflow structure
        status: Current status
        metrics: Usage metrics
    """

    name: str
    description: str
    category: PatternCategory
    participants: List[Dict[str, Any]]
    workflow: Dict[str, Any] = field(default_factory=dict)
    status: PatternStatus = PatternStatus.DISCOVERED
    metrics: PatternMetrics = field(default_factory=PatternMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Get success rate from metrics."""
        return self.metrics.success_rate

    def record_usage(self, success: bool, duration_ms: float, cost: float = 0.0) -> None:
        """Record a usage of this pattern.

        Args:
            success: Whether usage was successful
            duration_ms: Execution duration
            cost: Execution cost
        """
        m = self.metrics
        m.usage_count += 1
        if success:
            m.success_count += 1

        # Update running averages
        n = m.usage_count
        m.avg_duration_ms = (m.avg_duration_ms * (n - 1) + duration_ms) / n
        m.avg_cost = (m.avg_cost * (n - 1) + cost) / n
        m.last_used = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "participants": self.participants,
            "workflow": self.workflow,
            "status": self.status.value,
            "metrics": {
                "usage_count": self.metrics.usage_count,
                "success_count": self.metrics.success_count,
                "success_rate": self.metrics.success_rate,
                "avg_duration_ms": self.metrics.avg_duration_ms,
                "avg_cost": self.metrics.avg_cost,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborationPattern":
        """Create from dictionary."""
        metrics_data = data.get("metrics", {})
        metrics = PatternMetrics(
            usage_count=metrics_data.get("usage_count", 0),
            success_count=metrics_data.get("success_count", 0),
            avg_duration_ms=metrics_data.get("avg_duration_ms", 0.0),
            avg_cost=metrics_data.get("avg_cost", 0.0),
        )

        return cls(
            id=data.get("id", str(uuid4())[:8]),
            name=data["name"],
            description=data.get("description", ""),
            category=PatternCategory(data.get("category", "sequential")),
            participants=data.get("participants", []),
            workflow=data.get("workflow", {}),
            status=PatternStatus(data.get("status", "discovered")),
            metrics=metrics,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatternRecommendation:
    """A recommendation for using a pattern.

    Attributes:
        pattern: The recommended pattern
        score: Recommendation score (0-1)
        rationale: Why this pattern is recommended
        expected_benefits: Expected benefits
        potential_risks: Potential risks
    """

    pattern: CollaborationPattern
    score: float
    rationale: str
    expected_benefits: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionTrace:
    """Execution trace for pattern mining.

    Records what happened during a workflow execution for analysis.

    Attributes:
        workflow_id: Workflow identifier
        execution_id: Unique execution identifier
        nodes_executed: Nodes that were executed
        execution_order: Order of execution
        success: Whether execution succeeded
    """

    workflow_id: str
    execution_id: str
    nodes_executed: List[str]
    execution_order: List[str]
    success: bool
    duration_ms: float = 0.0
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


__all__ = [
    "PatternStatus",
    "PatternCategory",
    "PatternValidationResult",
    "PatternMetrics",
    "TaskContext",
    "CollaborationPattern",
    "PatternRecommendation",
    "WorkflowExecutionTrace",
]
