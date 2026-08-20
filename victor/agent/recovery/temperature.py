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

"""Progressive temperature adjustment with Q-learning integration.

This module provides intelligent temperature adjustment based on:
- Failure type (different failures need different temperature responses)
- Model characteristics (some models respond better to temperature changes)
- Historical success patterns (Q-learning-based optimization)

Temperature adjustment philosophy:
- Stuck loops: INCREASE temperature to encourage diversity
- Hallucinations: DECREASE temperature for more deterministic output
- Empty responses: Moderate increase to encourage output
- Low quality: Adjust based on learned optimal value
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from victor.agent.recovery.protocols import (
    FailureType,
    QLearningStore,
    RecoveryContext,
    TemperaturePolicy,
)
from victor.framework.temperature.defaults import (
    MODEL_TEMPERATURE_RANGES as _FRAMEWORK_MODEL_TEMPERATURE_RANGES,
)

logger = logging.getLogger(__name__)


@dataclass
class TemperatureState:
    """Tracks temperature adjustment state for a session."""

    initial_temperature: float
    current_temperature: float
    adjustments_made: int = 0
    failure_history: List[FailureType] = field(default_factory=list)

    def record_adjustment(
        self,
        new_temp: float,
        failure_type: FailureType,
    ) -> None:
        """Record a temperature adjustment."""
        self.current_temperature = new_temp
        self.adjustments_made += 1
        self.failure_history.append(failure_type)


class ProgressiveTemperatureAdjuster:
    """Intelligent temperature adjustment with Q-learning.

    Features:
    - Per-failure-type policies
    - Model-specific learned adjustments
    - Progressive adjustment with decay
    - Q-learning for optimal temperature discovery

    Follows Open/Closed Principle: New policies can be added without modification.
    """

    # Default policies per failure type
    DEFAULT_POLICIES: Dict[FailureType, TemperaturePolicy] = {
        FailureType.EMPTY_RESPONSE: TemperaturePolicy(
            failure_type=FailureType.EMPTY_RESPONSE,
            base_adjustment=0.15,  # Increase to encourage output
            max_temperature=1.0,
            min_temperature=0.3,
            decay_factor=0.8,
        ),
        FailureType.STUCK_LOOP: TemperaturePolicy(
            failure_type=FailureType.STUCK_LOOP,
            base_adjustment=0.2,  # Larger increase for diversity
            max_temperature=1.0,
            min_temperature=0.4,
            decay_factor=0.7,
        ),
        FailureType.HALLUCINATED_TOOL: TemperaturePolicy(
            failure_type=FailureType.HALLUCINATED_TOOL,
            base_adjustment=-0.1,  # Decrease for more deterministic output
            max_temperature=0.8,
            min_temperature=0.1,
            decay_factor=0.9,
        ),
        FailureType.REPEATED_RESPONSE: TemperaturePolicy(
            failure_type=FailureType.REPEATED_RESPONSE,
            base_adjustment=0.25,  # Large increase to break repetition
            max_temperature=1.0,
            min_temperature=0.5,
            decay_factor=0.6,
        ),
        FailureType.LOW_QUALITY: TemperaturePolicy(
            failure_type=FailureType.LOW_QUALITY,
            base_adjustment=0.1,  # Moderate adjustment
            max_temperature=0.9,
            min_temperature=0.2,
            decay_factor=0.85,
        ),
    }

    # Model-specific effective temperature ranges. Single source of truth lives in
    # victor.framework.temperature.defaults (ADR-013); referenced here to avoid duplication.
    MODEL_TEMPERATURE_RANGES: Dict[str, Tuple[float, float]] = _FRAMEWORK_MODEL_TEMPERATURE_RANGES

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        custom_policies: Optional[Dict[FailureType, TemperaturePolicy]] = None,
    ):
        self._q_store = q_store
        self._policies = {**self.DEFAULT_POLICIES}
        if custom_policies:
            self._policies.update(custom_policies)

        # Track per-model learned optimal temperatures
        self._learned_optima: Dict[str, Dict[str, float]] = {}
        # Format: {model_pattern: {failure_type_name: optimal_temp}}

        # Session state
        self._sessions: Dict[str, TemperatureState] = {}

    def get_adjusted_temperature(
        self,
        context: RecoveryContext,
        session_id: Optional[str] = None,
    ) -> Tuple[float, str]:
        """Calculate adjusted temperature for recovery.

        Args:
            context: Current recovery context
            session_id: Optional session ID for state tracking

        Returns:
            Tuple of (new_temperature, reason)
        """
        failure_type = context.failure_type
        current_temp = context.current_temperature
        model_name = context.model_name.lower()

        # Get or create session state
        if session_id:
            if session_id not in self._sessions:
                self._sessions[session_id] = TemperatureState(
                    initial_temperature=current_temp,
                    current_temperature=current_temp,
                )
            session = self._sessions[session_id]
        else:
            session = None

        # Get base policy for failure type
        policy = self._policies.get(
            failure_type,
            TemperaturePolicy(failure_type=failure_type),
        )

        # Calculate base adjustment
        consecutive = context.consecutive_failures
        adjustment = policy.base_adjustment * (policy.decay_factor**consecutive)

        # Apply model-specific bounds
        min_temp, max_temp = self._get_model_bounds(model_name)
        policy_min = max(policy.min_temperature, min_temp)
        policy_max = min(policy.max_temperature, max_temp)

        # Check Q-learning for learned optimal
        if self._q_store:
            learned_temp = self._get_learned_temperature(context)
            if learned_temp is not None:
                # Blend learned with calculated
                new_temp = 0.7 * learned_temp + 0.3 * (current_temp + adjustment)
                reason = f"Q-learned optimal ({learned_temp:.2f}) blended with policy"
            else:
                new_temp = current_temp + adjustment
                reason = f"Policy adjustment ({adjustment:+.2f}) for {failure_type.name}"
        else:
            new_temp = current_temp + adjustment
            reason = f"Policy adjustment ({adjustment:+.2f}) for {failure_type.name}"

        # Clamp to bounds
        new_temp = max(policy_min, min(policy_max, new_temp))

        # Update session state
        if session:
            session.record_adjustment(new_temp, failure_type)
            if session.adjustments_made > 3:
                reason += f" (adjustment #{session.adjustments_made})"

        logger.debug(f"Temperature adjustment: {current_temp:.2f} -> {new_temp:.2f} " f"({reason})")

        return new_temp, reason

    def _get_model_bounds(self, model_name: str) -> Tuple[float, float]:
        """Get effective temperature bounds for a model."""
        for pattern, bounds in self.MODEL_TEMPERATURE_RANGES.items():
            if pattern in model_name:
                return bounds
        return (0.0, 1.0)  # Default full range

    def _get_learned_temperature(
        self,
        context: RecoveryContext,
    ) -> Optional[float]:
        """Get Q-learned optimal temperature if available."""
        if not self._q_store:
            return None

        # Check learned optima cache
        model_key = self._get_model_key(context.model_name)
        failure_key = context.failure_type.name

        if model_key in self._learned_optima:
            if failure_key in self._learned_optima[model_key]:
                return self._learned_optima[model_key][failure_key]

        return None

    def _get_model_key(self, model_name: str) -> str:
        """Get a normalized model key for caching."""
        model_lower = model_name.lower()
        for pattern in self.MODEL_TEMPERATURE_RANGES:
            if pattern in model_lower:
                return pattern
        return "default"

    def record_outcome(
        self,
        context: RecoveryContext,
        new_temperature: float,
        success: bool,
        quality_improvement: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Record outcome for Q-learning.

        Args:
            context: Recovery context
            new_temperature: Temperature that was used
            success: Whether recovery was successful
            quality_improvement: Change in quality score
            session_id: Optional session ID
        """
        if not self._q_store:
            return

        # Calculate reward based on outcome
        reward = 0.0
        if success:
            reward = 0.5 + quality_improvement
        else:
            reward = -0.3

        # Create state key that includes temperature
        state_key = f"{context.to_state_key()}:temp={new_temperature:.1f}"

        # Update Q-value (using ADJUST_TEMPERATURE action)
        from victor.agent.recovery.protocols import RecoveryAction

        self._q_store.update_q_value(
            state_key,
            RecoveryAction.ADJUST_TEMPERATURE,
            reward,
        )

        # Update learned optima if this was a good outcome
        if success and quality_improvement > 0.1:
            model_key = self._get_model_key(context.model_name)
            failure_key = context.failure_type.name

            if model_key not in self._learned_optima:
                self._learned_optima[model_key] = {}

            # Exponential moving average of optimal temperature
            current_optimal = self._learned_optima[model_key].get(failure_key)
            if current_optimal is None:
                self._learned_optima[model_key][failure_key] = new_temperature
            else:
                # Blend: 80% old, 20% new
                self._learned_optima[model_key][failure_key] = (
                    0.8 * current_optimal + 0.2 * new_temperature
                )

            logger.debug(
                f"Updated learned optimal temperature for {model_key}/{failure_key}: "
                f"{self._learned_optima[model_key][failure_key]:.2f}"
            )

    def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a session."""
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        return {
            "initial_temperature": session.initial_temperature,
            "current_temperature": session.current_temperature,
            "adjustments_made": session.adjustments_made,
            "failure_types": [ft.name for ft in session.failure_history],
        }

    def get_learned_optima(self) -> Dict[str, Dict[str, float]]:
        """Get all learned optimal temperatures."""
        return dict(self._learned_optima)
