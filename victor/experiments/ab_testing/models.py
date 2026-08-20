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

"""Data models for A/B testing.

This module defines the core data structures for experiments, variants,
and metrics using Pydantic for validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
import time
import uuid

# =============================================================================
# Enums for A/B Testing
# =============================================================================


class AllocationStrategy(str, Enum):
    """Traffic allocation strategies."""

    RANDOM = "random"
    STICKY = "sticky"
    ROUND_ROBIN = "round_robin"


class ExperimentRecommendation(str, Enum):
    """Recommendation for A/B test experiment.

    Determines what action to take based on test results:
    - DEPLOY_WINNER: Clear winner found, deploy to production
    - CONTINUE: Need more data, continue collecting
    - INCONCLUSIVE: No clear winner, inconclusive results
    """

    DEPLOY_WINNER = "deploy_winner"  # Clear winner, deploy
    CONTINUE = "continue"  # Need more data
    INCONCLUSIVE = "inconclusive"  # No clear winner


class AggregationMethod(str, Enum):
    """Method for aggregating metric values.

    Determines how to aggregate multiple measurements:
    - MEAN: Arithmetic mean (average)
    - MEDIAN: Median value (middle)
    - P95: 95th percentile
    - SUM: Sum of all values
    """

    MEAN = "mean"  # Arithmetic mean
    MEDIAN = "median"  # Median value
    P95 = "p95"  # 95th percentile
    SUM = "sum"  # Sum of values


class WorkflowType(str, Enum):
    """Type of workflow definition.

    Determines how workflows are defined:
    - YAML: YAML workflow definition
    - STATEGRAPH: StateGraph programmatic definition
    - DEFINITION: Workflow definition format
    """

    YAML = "yaml"  # YAML workflow definition
    STATEGRAPH = "stategraph"  # StateGraph programmatic definition
    DEFINITION = "definition"  # Workflow definition format


class VariantStatus(str, Enum):
    """Status of experiment variant.

    Runtime status of variants:
    - RUNNING: Currently executing
    - PAUSED: Temporarily paused
    - FAILED: Failed with error
    """

    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Temporarily paused
    FAILED = "failed"  # Failed with error


@dataclass
class ExperimentVariant:
    """A single variant in an A/B test.

    Attributes:
        variant_id: Unique identifier (e.g., "control", "treatment_a")
        name: Human-readable name
        description: Optional description
        workflow_type: Type of workflow (yaml, stategraph, definition)
        workflow_config: Variant-specific workflow configuration
        parameter_overrides: Optional parameter overrides for this variant
        traffic_weight: Percentage of traffic (0-1)
        is_control: Whether this is the control variant
        tags: Optional metadata tags
    """

    variant_id: str
    name: str
    description: str = ""

    # Workflow configuration
    workflow_type: WorkflowType = WorkflowType.YAML
    workflow_config: Dict[str, Any] = field(default_factory=dict)

    # Parameter overrides (optional)
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    # Example: {"model": "claude-opus-4-5", "tool_budget": 20, "temperature": 0.7}

    # Traffic allocation
    traffic_weight: float = 0.5

    # Variant metadata
    is_control: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentMetric:
    """A metric to track in an experiment.

    Attributes:
        metric_id: Unique identifier
        name: Human-readable name
        description: Optional description
        metric_type: Type of metric
        optimization_goal: Goal (minimize, maximize, target)
        target_value: For "target" goals
        success_threshold: Minimum acceptable value
        relative_improvement: Percentage improvement needed
        aggregation_method: How to aggregate (mean, median, p95, sum)
    """

    metric_id: str
    name: str
    description: str = ""

    # Metric definition
    metric_type: Literal[
        "execution_time",
        "token_usage",
        "tool_calls_count",
        "success_rate",
        "cost",
        "custom",
    ] = "execution_time"

    # Optimization goal
    optimization_goal: Literal["minimize", "maximize", "target"] = "maximize"
    target_value: Optional[float] = None

    # Success criteria
    success_threshold: Optional[float] = None
    relative_improvement: Optional[float] = None

    # Statistical settings
    aggregation_method: AggregationMethod = AggregationMethod.MEAN


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        experiment_id: Unique identifier (auto-generated)
        name: Human-readable name
        description: Optional description
        hypothesis: What we're testing
        variants: List of variants to test
        primary_metric: Main decision metric
        secondary_metrics: Optional secondary metrics
        min_sample_size: Minimum samples per variant
        max_duration_seconds: Optional time limit
        max_iterations: Optional execution count limit
        significance_level: Alpha (p-value threshold)
        statistical_power: Statistical power (1 - beta)
        confidence_interval: For reporting
        enable_early_stopping: Whether to enable early stopping
        early_stopping_threshold: Confidence threshold for early stopping
        targeting_rules: Optional targeting rules
        tags: Optional metadata tags
        created_at: Creation timestamp
        created_by: Creator identifier
    """

    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    hypothesis: str = ""

    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)

    # Metrics to track
    primary_metric: Optional[ExperimentMetric] = None
    secondary_metrics: List[ExperimentMetric] = field(default_factory=list)

    # Experiment constraints
    min_sample_size: int = 100
    max_duration_seconds: Optional[int] = None
    max_iterations: Optional[int] = None

    # Statistical significance
    significance_level: float = 0.05
    statistical_power: float = 0.8
    confidence_interval: float = 0.95

    # Early stopping
    enable_early_stopping: bool = False
    early_stopping_threshold: float = 0.99

    # Target audience
    targeting_rules: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"


@dataclass
class ExperimentStatus:
    """Current status of an experiment.

    Attributes:
        status: Current status
        started_at: Optional start timestamp
        completed_at: Optional completion timestamp
        paused_at: Optional pause timestamp
        current_iteration: Current iteration count
        total_samples: Total samples across all variants
        variant_samples: Per-variant sample counts
        variant_status: Per-variant status
    """

    status: Literal[
        "draft",
        "running",
        "paused",
        "completed",
        "analyzed",
        "rolled_out",
        "archived",
    ] = "draft"

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    paused_at: Optional[float] = None

    current_iteration: int = 0
    total_samples: int = 0

    # Variant status
    variant_samples: Dict[str, int] = field(default_factory=dict)
    variant_status: Dict[str, VariantStatus] = field(default_factory=dict)


@dataclass
class VariantResult:
    """Results for a single variant.

    Attributes:
        variant_id: Variant identifier
        primary_metric_value: Value of primary metric
        primary_metric_std: Standard deviation of primary metric
        primary_metric_samples: Number of samples
        secondary_metrics: Optional secondary metrics
        relative_improvement: Relative improvement vs control
        absolute_improvement: Absolute improvement vs control
        confidence_interval: Optional confidence interval
        effect_size: Optional Cohen's d
    """

    variant_id: str

    # Primary metric
    primary_metric_value: float = 0.0
    primary_metric_std: float = 0.0
    primary_metric_samples: int = 0

    # Secondary metrics
    secondary_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Performance vs control
    relative_improvement: Optional[float] = None
    absolute_improvement: Optional[float] = None

    # Statistical measures
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None


@dataclass
class ExperimentResult:
    """Results from a completed experiment.

    Attributes:
        experiment_id: Experiment identifier
        winning_variant_id: Optional winning variant
        confidence: Optional confidence in winner
        statistical_significance: Whether difference is significant
        p_value: Optional p-value
        confidence_interval: Optional confidence interval
        variant_results: Per-variant results
        recommendation: Recommendation (deploy_winner, continue, inconclusive)
        reasoning: Explanation of recommendation
        analyzed_at: Analysis timestamp
        total_samples: Total samples
        total_duration_seconds: Total duration
    """

    experiment_id: str

    # Winner determination
    winning_variant_id: Optional[str] = None
    confidence: Optional[float] = None

    # Statistical tests
    statistical_significance: bool = False
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    # Per-variant results
    variant_results: Dict[str, VariantResult] = field(default_factory=dict)

    # Recommendations
    recommendation: ExperimentRecommendation = ExperimentRecommendation.INCONCLUSIVE
    reasoning: str = ""

    # Metadata
    analyzed_at: float = field(default_factory=time.time)
    total_samples: int = 0
    total_duration_seconds: float = 0.0


@dataclass
class ExecutionMetrics:
    """Metrics collected from each workflow execution.

    Attributes:
        execution_id: Unique execution identifier
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        user_id: User identifier
        execution_time: Total duration in seconds
        node_times: Per-node durations
        prompt_tokens: Prompt token count
        completion_tokens: Completion token count
        total_tokens: Total token count
        tool_calls_count: Number of tool calls
        tool_calls_by_name: Per-name tool call counts
        tool_errors: Number of tool errors
        success: Whether execution succeeded
        error_message: Optional error message
        estimated_cost: Estimated cost in USD
        custom_metrics: Optional custom metrics
        timestamp: Execution timestamp
        workflow_name: Workflow name
        workflow_type: Workflow type
    """

    execution_id: str
    experiment_id: str
    variant_id: str
    user_id: str

    # Timing
    execution_time: float = 0.0
    node_times: Dict[str, float] = field(default_factory=dict)

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Tool usage
    tool_calls_count: int = 0
    tool_calls_by_name: Dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0

    # Success
    success: bool = True
    error_message: Optional[str] = None

    # Cost
    estimated_cost: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    workflow_name: str = ""
    workflow_type: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated metrics per variant.

    Attributes:
        variant_id: Variant identifier
        sample_count: Number of samples
        execution_time_mean: Mean execution time
        execution_time_median: Median execution time
        execution_time_std: Std dev of execution time
        execution_time_p95: 95th percentile execution time
        execution_time_ci: Confidence interval for execution time
        total_tokens_mean: Mean token usage
        total_tokens_median: Median token usage
        total_tokens_sum: Total token usage
        tool_calls_mean: Mean tool calls
        tool_errors_total: Total tool errors
        tool_error_rate: Tool error rate
        success_count: Number of successful executions
        success_rate: Success rate
        success_rate_ci: Confidence interval for success rate
        total_cost: Total cost
        cost_per_execution_mean: Mean cost per execution
        custom_metrics_aggregated: Aggregated custom metrics
    """

    variant_id: str

    # Sample size
    sample_count: int = 0

    # Execution time
    execution_time_mean: float = 0.0
    execution_time_median: float = 0.0
    execution_time_std: float = 0.0
    execution_time_p95: float = 0.0
    execution_time_ci: Tuple[float, float] = (0.0, 0.0)

    # Token usage
    total_tokens_mean: float = 0.0
    total_tokens_median: float = 0.0
    total_tokens_sum: int = 0

    # Tool usage
    tool_calls_mean: float = 0.0
    tool_errors_total: int = 0
    tool_error_rate: float = 0.0

    # Success rate
    success_count: int = 0
    success_rate: float = 0.0
    success_rate_ci: Tuple[float, float] = (0.0, 0.0)

    # Cost
    total_cost: float = 0.0
    cost_per_execution_mean: float = 0.0

    # Custom metrics
    custom_metrics_aggregated: Dict[str, Dict[str, float]] = field(default_factory=dict)
