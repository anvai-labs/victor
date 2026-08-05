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

"""Data models for workflow optimization.

This module defines the core data structures used throughout the optimization
system, including profiles, statistics, bottlenecks, and opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    SLOW_NODE = "slow_node"
    DOMINANT_NODE = "dominant_node"
    UNRELIABLE_NODE = "unreliable_node"
    EXPENSIVE_TOOL = "expensive_tool"
    MISSING_CACHING = "missing_caching"
    REDUNDANT_OPERATION = "redundant_operation"
    UNUSED_OUTPUT = "unused_output"


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationStrategyType(Enum):
    """Types of optimization strategies."""

    PRUNING = "pruning"
    PARALLELIZATION = "parallelization"
    TOOL_SELECTION = "tool_selection"
    MODEL_SELECTION = "model_selection"
    CACHING = "caching"
    BATCHING = "batching"


@dataclass
class NodeStatistics:
    """Detailed statistics for a single workflow node.

    Attributes:
        node_id: Unique identifier for the node
        avg_duration: Average execution duration in seconds
        p50_duration: 50th percentile duration
        p95_duration: 95th percentile duration
        p99_duration: 99th percentile duration
        success_rate: Success rate (0.0 to 1.0)
        error_types: Dictionary mapping error types to counts
        avg_input_tokens: Average input tokens consumed
        avg_output_tokens: Average output tokens generated
        token_efficiency: Ratio of output to input tokens
        tool_calls: Dictionary mapping tool names to call statistics
        total_cost: Total cost in USD
        avg_memory_mb: Average memory usage in MB
        avg_cpu_percent: Average CPU usage percentage
    """

    node_id: str
    avg_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    success_rate: float
    error_types: Dict[str, int] = field(default_factory=dict)
    avg_input_tokens: int = 0
    avg_output_tokens: int = 0
    token_efficiency: float = 0.0
    tool_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_cost: float = 0.0
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "avg_duration": self.avg_duration,
            "p50_duration": self.p50_duration,
            "p95_duration": self.p95_duration,
            "p99_duration": self.p99_duration,
            "success_rate": self.success_rate,
            "error_types": self.error_types,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "token_efficiency": self.token_efficiency,
            "tool_calls": self.tool_calls,
            "total_cost": self.total_cost,
            "avg_memory_mb": self.avg_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
        }


@dataclass
class Bottleneck:
    """Represents a performance bottleneck in a workflow.

    Attributes:
        type: Type of bottleneck
        severity: Severity level
        node_id: Affected node ID (if applicable)
        tool_id: Affected tool ID (if applicable)
        metric: Metric that triggered the bottleneck
        value: Actual value of the metric
        threshold: Threshold value that was exceeded
        suggestion: Human-readable suggestion for fixing the bottleneck
        confidence: Confidence in this bottleneck detection (0.0 to 1.0)
    """

    type: BottleneckType
    severity: BottleneckSeverity
    metric: str
    value: float
    threshold: float
    suggestion: str
    node_id: Optional[str] = None
    tool_id: Optional[str] = None
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "node_id": self.node_id,
            "tool_id": self.tool_id,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        """String representation."""
        location = self.node_id or self.tool_id or "unknown"
        return (
            f"{self.type.value}: {location} - "
            f"{self.metric}={self.value:.2f} (threshold={self.threshold:.2f})"
        )


@dataclass
class OptimizationOpportunity:
    """Represents an opportunity to optimize a workflow.

    Attributes:
        strategy_type: Type of optimization strategy to apply
        target: Target node/tool ID
        description: Human-readable description
        expected_improvement: Expected improvement as percentage (0.0 to 1.0)
        risk_level: Risk level (LOW/MEDIUM/HIGH)
        estimated_cost_reduction: Expected cost reduction in USD
        estimated_duration_reduction: Expected duration reduction in seconds
        confidence: Confidence in the opportunity estimate (0.0 to 1.0)
        metadata: Additional metadata about the opportunity
    """

    strategy_type: OptimizationStrategyType
    target: str
    description: str
    expected_improvement: float
    risk_level: BottleneckSeverity
    estimated_cost_reduction: float = 0.0
    estimated_duration_reduction: float = 0.0
    confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy_type": self.strategy_type.value,
            "target": self.target,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "risk_level": self.risk_level.value,
            "estimated_cost_reduction": self.estimated_cost_reduction,
            "estimated_duration_reduction": self.estimated_duration_reduction,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.strategy_type.value}: {self.target} - "
            f"{self.expected_improvement:.1%} improvement "
            f"(risk={self.risk_level.value}, confidence={self.confidence:.1%})"
        )


@dataclass
class OptimizationStrategy:
    """Configuration for an optimization strategy.

    Attributes:
        name: Unique name for the strategy
        strategy_type: Type of optimization
        enabled: Whether this strategy is enabled
        priority: Priority for execution (higher = earlier)
        max_iterations: Maximum number of iterations for this strategy
        parameters: Strategy-specific parameters
    """

    name: str
    strategy_type: OptimizationStrategyType
    enabled: bool = True
    priority: int = 0
    max_iterations: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "max_iterations": self.max_iterations,
            "parameters": self.parameters,
        }


@dataclass
class WorkflowProfile:
    """Comprehensive performance profile for a workflow.

    Attributes:
        workflow_id: Unique identifier for the workflow
        node_stats: Per-node statistics
        bottlenecks: Detected bottlenecks
        opportunities: Optimization opportunities
        total_duration: Total workflow duration
        total_cost: Total workflow cost
        total_tokens: Total tokens consumed
        success_rate: Overall success rate
        created_at: Timestamp when profile was created
        num_executions: Number of executions analyzed
    """

    workflow_id: str
    node_stats: Dict[str, NodeStatistics]
    bottlenecks: List[Bottleneck]
    opportunities: List[OptimizationOpportunity]
    total_duration: float
    total_cost: float
    total_tokens: int
    success_rate: float
    created_at: datetime = field(default_factory=datetime.now)
    num_executions: int = 1

    def get_slowest_nodes(self, n: int = 5) -> List[NodeStatistics]:
        """Get the N slowest nodes by average duration."""
        return sorted(
            self.node_stats.values(),
            key=lambda s: s.avg_duration,
            reverse=True,
        )[:n]

    def get_most_expensive_nodes(self, n: int = 5) -> List[NodeStatistics]:
        """Get the N most expensive nodes by total cost."""
        return sorted(
            self.node_stats.values(),
            key=lambda s: s.total_cost,
            reverse=True,
        )[:n]

    def get_least_reliable_nodes(self, n: int = 5) -> List[NodeStatistics]:
        """Get the N least reliable nodes by success rate."""
        return sorted(
            self.node_stats.values(),
            key=lambda s: s.success_rate,
        )[:n]

    def get_high_impact_opportunities(
        self,
        min_confidence: float = 0.7,
        min_improvement: float = 0.1,
    ) -> List[OptimizationOpportunity]:
        """Get high-impact optimization opportunities.

        Args:
            min_confidence: Minimum confidence threshold
            min_improvement: Minimum expected improvement threshold

        Returns:
            List of high-impact opportunities sorted by expected improvement
        """
        return [
            opp
            for opp in self.opportunities
            if opp.confidence >= min_confidence and opp.expected_improvement >= min_improvement
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "node_stats": {node_id: stats.to_dict() for node_id, stats in self.node_stats.items()},
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "opportunities": [o.to_dict() for o in self.opportunities],
            "total_duration": self.total_duration,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "num_executions": self.num_executions,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"WorkflowProfile(id={self.workflow_id}, "
            f"duration={self.total_duration:.2f}s, "
            f"cost=${self.total_cost:.4f}, "
            f"success_rate={self.success_rate:.1%}, "
            f"bottlenecks={len(self.bottlenecks)}, "
            f"opportunities={len(self.opportunities)})"
        )
