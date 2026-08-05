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

"""Protocol definitions for the recovery system.

This module defines the contracts (protocols) that all recovery components must follow.
Using Protocol (structural subtyping) over ABC (nominal subtyping) because:

1. Duck typing with contracts: Components can implement protocols implicitly
2. Testability: Easy to create test doubles without inheritance
3. Flexibility: External code can implement protocols without importing
4. Composition: Multiple protocols can be combined easily

Why Protocol over Duck Typing alone:
- Static type checking catches contract violations at development time
- Self-documenting interfaces
- IDE support for autocomplete and refactoring
- Runtime isinstance() checks available with @runtime_checkable

Contract-based design ensures:
- Failure modes are explicitly typed (FailureType enum)
- Recovery actions are well-defined (RecoveryAction dataclass)
- State transitions are traceable (RecoveryResult)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)


class FailureType(Enum):
    """Enumeration of detectable failure modes.

    Each failure type maps to specific recovery strategies.
    Using explicit enum ensures all failure modes are handled.
    """

    EMPTY_RESPONSE = auto()  # Model returned no content or tool calls
    STUCK_LOOP = auto()  # Model keeps planning but not executing
    HALLUCINATED_TOOL = auto()  # Model mentions tool but doesn't call it
    TIMEOUT_APPROACHING = auto()  # Session time limit approaching
    CONTEXT_OVERFLOW = auto()  # Context window near capacity
    REPEATED_RESPONSE = auto()  # Model repeating same response
    LOW_QUALITY = auto()  # Response quality below threshold
    TOOL_BUDGET_EXHAUSTED = auto()  # Tool call budget depleted
    ITERATION_LIMIT = auto()  # Max iterations reached
    PROVIDER_ERROR = auto()  # Provider returned an error
    RATE_LIMITED = auto()  # Provider rate limit hit


class StrategyRecoveryAction(Enum):
    """Actions that recovery strategies can recommend.

    Defines the vocabulary of recovery interventions.

    Renamed from RecoveryAction to be semantically distinct:
    - ErrorRecoveryAction (victor.agent.error_recovery): Tool error recovery (string values)
    - StrategyRecoveryAction (here): Recovery strategy actions (auto enum values)
    - OrchestratorRecoveryAction (victor.agent.orchestrator_recovery): Orchestrator recovery dataclass
    """

    CONTINUE = auto()  # No recovery needed, continue normally
    PROMPT_TOOL_CALL = auto()  # Inject prompt to encourage tool call
    FORCE_SUMMARY = auto()  # Force model to summarize and finish
    ADJUST_TEMPERATURE = auto()  # Change temperature for retry
    SWITCH_MODEL = auto()  # Fall back to different model
    COMPACT_CONTEXT = auto()  # Reduce context size
    RETRY_WITH_TEMPLATE = auto()  # Retry with specific prompt template
    WAIT_AND_RETRY = auto()  # Wait for rate limit, then retry
    ABORT = auto()  # Give up and return error


# Backward compatibility alias
RecoveryAction = StrategyRecoveryAction


class RatioLevel(str, Enum):
    """Discretized ratio bucket for recovery state."""

    LOW = "low"  # < 0.25
    MID_LOW = "mid_low"  # 0.25-0.5
    MID_HIGH = "mid_high"  # 0.5-0.75
    HIGH = "high"  # > 0.75


class QualityLevel(str, Enum):
    """Discretized quality score bucket."""

    POOR = "poor"  # < 0.4
    FAIR = "fair"  # 0.4-0.6
    GOOD = "good"  # 0.6-0.8
    EXCELLENT = "excellent"  # > 0.8


class TemperatureLevel(str, Enum):
    """Discretized temperature bucket for recovery."""

    LOW = "low"  # < 0.3
    MEDIUM = "medium"  # 0.3-0.6
    HIGH = "high"  # 0.6-0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass(frozen=True)
class RecoveryContext:
    """Immutable context for recovery decisions.

    Captures the complete state needed to make recovery decisions.
    Frozen to prevent accidental mutation during processing.
    """

    # Current state
    failure_type: FailureType
    content: str = ""
    tool_calls_made: int = 0
    tool_budget: int = 10
    iteration_count: int = 0
    max_iterations: int = 50
    elapsed_time_seconds: float = 0.0
    session_idle_timeout: float = 180.0

    # Model info
    provider_name: str = ""
    model_name: str = ""
    current_temperature: float = 0.7

    # History for pattern detection
    consecutive_failures: int = 0
    mentioned_tools: Tuple[str, ...] = ()
    recent_responses: Tuple[str, ...] = ()  # Last N responses for loop detection

    # Quality metrics
    last_quality_score: float = 0.5
    grounding_score: float = 1.0

    # Task context
    task_type: str = "general"
    is_analysis_task: bool = False
    is_action_task: bool = False

    def to_state_key(self) -> str:
        """Generate a hashable state key for Q-learning.

        Discretizes continuous values for state space manageability.
        """
        # Discretize continuous values
        budget_ratio = self._discretize_ratio(self.tool_calls_made / max(self.tool_budget, 1))
        iter_ratio = self._discretize_ratio(self.iteration_count / max(self.max_iterations, 1))
        time_ratio = self._discretize_ratio(
            self.elapsed_time_seconds / max(self.session_idle_timeout, 1)
        )
        quality_bucket = self._discretize_quality(self.last_quality_score)
        temp_bucket = self._discretize_temperature(self.current_temperature)

        key = (
            f"{self.failure_type.name}:"
            f"{self.provider_name}:{self.model_name}:"
            f"{budget_ratio.value}:{iter_ratio.value}:{time_ratio.value}:"
            f"{quality_bucket.value}:{temp_bucket.value}:"
            f"{self.task_type}:{self.consecutive_failures}"
        )
        return key

    def _discretize_ratio(self, ratio: float) -> RatioLevel:
        if ratio < 0.25:
            return RatioLevel.LOW
        elif ratio < 0.5:
            return RatioLevel.MID_LOW
        elif ratio < 0.75:
            return RatioLevel.MID_HIGH
        else:
            return RatioLevel.HIGH

    def _discretize_quality(self, score: float) -> QualityLevel:
        if score < 0.4:
            return QualityLevel.POOR
        elif score < 0.6:
            return QualityLevel.FAIR
        elif score < 0.8:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT

    def _discretize_temperature(self, temp: float) -> TemperatureLevel:
        if temp < 0.3:
            return TemperatureLevel.LOW
        elif temp < 0.6:
            return TemperatureLevel.MEDIUM
        elif temp < 0.9:
            return TemperatureLevel.HIGH
        else:
            return TemperatureLevel.VERY_HIGH


@dataclass
class RecoveryResult:
    """Result of a recovery attempt.

    Contains the recommended action and all metadata needed for execution.
    """

    action: RecoveryAction
    success: bool = False
    message: Optional[str] = None  # Message to inject into conversation
    new_temperature: Optional[float] = None
    fallback_model: Optional[str] = None
    wait_seconds: Optional[float] = None
    prompt_template_id: Optional[str] = None

    # Metadata for learning
    strategy_name: str = ""
    confidence: float = 0.5
    reason: str = ""

    # For telemetry
    timestamp: datetime = field(default_factory=datetime.now)
    context_hash: str = ""  # Hash of RecoveryContext for correlation


@dataclass
class PromptTemplate:
    """A prompt template with metadata for learning.

    Templates are versioned and tracked for effectiveness.
    """

    id: str
    name: str
    template: str
    failure_types: List[FailureType]
    provider_patterns: List[str] = field(default_factory=list)  # Glob patterns
    model_patterns: List[str] = field(default_factory=list)

    # Learning metrics
    usage_count: int = 0
    success_count: int = 0
    avg_quality_improvement: float = 0.0

    # Versioning
    version: int = 1
    parent_id: Optional[str] = None  # For template evolution tracking

    @property
    def success_rate(self) -> float:
        """Calculate success rate of this template."""
        if self.usage_count == 0:
            return 0.5  # Prior probability
        return self.success_count / self.usage_count

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)


@dataclass
class TemperaturePolicy:
    """Policy for temperature adjustment.

    Defines how temperature should change based on failure patterns.
    """

    failure_type: FailureType
    base_adjustment: float = 0.1  # How much to adjust per failure
    max_temperature: float = 1.0
    min_temperature: float = 0.1
    decay_factor: float = 0.9  # Decay adjustment over consecutive attempts

    def calculate_temperature(
        self,
        current_temp: float,
        consecutive_failures: int,
    ) -> float:
        """Calculate new temperature based on failures."""
        # Adjust with decay for consecutive failures
        adjustment = self.base_adjustment * (self.decay_factor**consecutive_failures)
        new_temp = current_temp + adjustment
        return max(self.min_temperature, min(self.max_temperature, new_temp))


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol defining the contract for recovery strategies.

    All recovery strategies must implement this interface.
    Using @runtime_checkable allows isinstance() checks.

    Single Responsibility: Each implementation handles one failure type.
    Open/Closed: New strategies can be added without modifying this protocol.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this strategy."""
        ...

    @property
    def handles_failure_types(self) -> List[FailureType]:
        """List of failure types this strategy can handle."""
        ...

    def can_handle(self, context: RecoveryContext) -> bool:
        """Check if this strategy can handle the given context.

        Args:
            context: Current recovery context

        Returns:
            True if this strategy should be considered
        """
        ...

    def calculate_priority(self, context: RecoveryContext) -> float:
        """Calculate priority score for this strategy in current context.

        Higher scores indicate this strategy should be tried first.
        Allows learned weighting of strategies.

        Args:
            context: Current recovery context

        Returns:
            Priority score (0.0 to 1.0)
        """
        ...

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery for the given context.

        Args:
            context: Current recovery context

        Returns:
            RecoveryResult with recommended action
        """
        ...

    def record_outcome(
        self, context: RecoveryContext, result: RecoveryResult, success: bool
    ) -> None:
        """Record the outcome of a recovery attempt for learning.

        Args:
            context: The context that was recovered
            result: The result that was applied
            success: Whether the recovery was successful
        """
        ...


@runtime_checkable
class TelemetryCollector(Protocol):
    """Protocol for collecting recovery telemetry.

    Separates telemetry concerns from recovery logic.
    Interface Segregation: Only defines telemetry-related methods.
    """

    def record_failure(self, context: RecoveryContext) -> None:
        """Record a failure occurrence."""
        ...

    def record_recovery_attempt(self, context: RecoveryContext, result: RecoveryResult) -> None:
        """Record a recovery attempt."""
        ...

    def record_recovery_outcome(
        self,
        context: RecoveryContext,
        result: RecoveryResult,
        success: bool,
        quality_improvement: float,
    ) -> None:
        """Record the outcome of a recovery attempt."""
        ...

    def get_failure_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for a time window."""
        ...

    def get_strategy_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics per strategy."""
        ...


@runtime_checkable
class QLearningStoreProtocol(Protocol):
    """Protocol for Q-learning state storage.

    NOTE: The framework provides a concrete implementation in:
    victor.agent.adaptive_mode_controller.QLearningStore

    This protocol allows the recovery system to work with that existing
    implementation via duck typing, supporting Dependency Inversion.

    Usage:
        from victor.agent.adaptive_mode_controller import QLearningStore
        q_store = QLearningStore()  # Uses existing framework Q-learning

    The protocol defines a minimal interface for recovery-specific Q-learning.
    """

    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for state-action pair.

        Note: action_key is a string (e.g., RecoveryAction.name) to match
        the existing framework's QLearningStore interface.
        """
        ...

    def update_q_value(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: Optional[str] = None,
    ) -> None:
        """Update Q-value based on reward."""
        ...

    @property
    def exploration_rate(self) -> float:
        """Current exploration rate for epsilon-greedy."""
        ...


# Type alias for compatibility - can use either protocol or concrete type
QLearningStore = QLearningStoreProtocol


@runtime_checkable
class ModelFallbackPolicy(Protocol):
    """Protocol for model fallback decisions.

    Defines how to select fallback models.
    """

    def get_fallback_model(
        self,
        current_provider: str,
        current_model: str,
        failure_type: FailureType,
    ) -> Optional[Tuple[str, str]]:
        """Get fallback provider and model.

        Args:
            current_provider: Current provider name
            current_model: Current model name
            failure_type: Type of failure that triggered fallback

        Returns:
            Tuple of (provider, model) or None if no fallback available
        """
        ...

    def record_model_failure(
        self,
        provider: str,
        model: str,
        failure_type: FailureType,
    ) -> None:
        """Record a model failure for circuit breaker logic."""
        ...

    def is_model_available(self, provider: str, model: str) -> bool:
        """Check if a model is currently available (not circuit-broken)."""
        ...


# Type variable for generic strategy composition
T = TypeVar("T", bound=RecoveryStrategy)
