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

"""Policy manager for safe RL deployment.

This module provides high-level policy management including:
- Automatic checkpointing based on outcomes
- Rollback on performance degradation
- Shadow mode for risk-free evaluation
- Gradual rollout control

Policy Lifecycle:
1. Development: Policy trained with exploration
2. Staging: Policy tested in shadow mode
3. Canary: Small percentage of traffic
4. Production: Full deployment
5. Deprecated: Marked for removal

Shadow Mode:
- New policy runs in parallel
- Outcomes recorded but not used for decisions
- Compare performance before promoting

Sprint 6: Observability & Polish
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.rl.checkpoint_store import (
    CheckpointStore,
    get_checkpoint_store,
)

if TYPE_CHECKING:
    from victor.framework.rl.base import BaseLearner
    from victor.framework.rl.coordinator import RLCoordinator

logger = logging.getLogger(__name__)


class PolicyStage(str, Enum):
    """Policy lifecycle stages."""

    DEVELOPMENT = "development"  # Active training
    STAGING = "staging"  # Shadow mode testing
    CANARY = "canary"  # Limited production traffic
    PRODUCTION = "production"  # Full deployment
    DEPRECATED = "deprecated"  # Marked for removal


@dataclass
class PolicyState:
    """State of a managed policy.

    Attributes:
        learner_name: Name of the learner
        current_version: Current active version
        stage: Lifecycle stage
        shadow_version: Version running in shadow mode
        canary_traffic: Percentage of traffic for canary (0-100)
        performance_baseline: Baseline performance metrics
        last_checkpoint_outcomes: Outcomes since last checkpoint
        auto_checkpoint_threshold: Outcomes before auto-checkpoint
    """

    learner_name: str
    current_version: str = "v0.0.0"
    stage: PolicyStage = PolicyStage.DEVELOPMENT
    shadow_version: Optional[str] = None
    canary_traffic: int = 0
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    last_checkpoint_outcomes: int = 0
    auto_checkpoint_threshold: int = 100

    # Performance tracking
    recent_success_rate: float = 0.5
    recent_quality_score: float = 0.5


@dataclass
class RollbackEvent:
    """Record of a rollback event.

    Attributes:
        learner_name: Name of the learner
        from_version: Version rolled back from
        to_version: Version rolled back to
        reason: Reason for rollback
        metrics_before: Metrics before rollback
        timestamp: When rollback occurred
    """

    learner_name: str
    from_version: str
    to_version: str
    reason: str
    metrics_before: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PolicyManager:
    """Manager for safe policy deployment.

    Coordinates policy lifecycle, checkpointing, and rollback
    for all RL learners.

    Features:
    1. Auto-Checkpointing: Create checkpoints after N outcomes
    2. Performance Monitoring: Track success rate and quality
    3. Auto-Rollback: Revert to previous version on degradation
    4. Shadow Mode: Test new policies without affecting outcomes
    5. Gradual Rollout: Canary deployments with traffic control

    Usage:
        manager = PolicyManager(coordinator, checkpoint_store)

        # Enable auto-checkpointing
        manager.enable_auto_checkpoint("tool_selector", threshold=100)

        # Record outcomes (triggers auto-checkpoint if threshold met)
        manager.record_outcome("tool_selector", success=True, quality=0.85)

        # Check for degradation
        if manager.should_rollback("tool_selector"):
            manager.rollback("tool_selector")

        # Shadow mode testing
        manager.start_shadow_mode("tool_selector", "v2.0.0")
        shadow_result = manager.get_shadow_recommendation("tool_selector", ...)
    """

    # Degradation detection parameters
    DEGRADATION_WINDOW = 50
    DEGRADATION_THRESHOLD = 0.15  # 15% drop triggers rollback
    MIN_SAMPLES_FOR_ROLLBACK = 30

    # Auto-checkpoint defaults
    DEFAULT_CHECKPOINT_THRESHOLD = 100

    def __init__(
        self,
        coordinator: Optional["RLCoordinator"] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
    ):
        """Initialize policy manager.

        Args:
            coordinator: RL coordinator for learner access
            checkpoint_store: Store for policy checkpoints
        """
        self._coordinator = coordinator
        self._checkpoint_store = checkpoint_store or get_checkpoint_store()

        # Policy states per learner
        self._policy_states: Dict[str, PolicyState] = {}

        # Recent outcomes for degradation detection
        self._recent_outcomes: Dict[str, List[Dict[str, Any]]] = {}

        # Rollback history
        self._rollback_history: List[RollbackEvent] = []

        # Shadow mode states
        self._shadow_states: Dict[str, Dict[str, Any]] = {}

    def set_coordinator(self, coordinator: "RLCoordinator") -> None:
        """Set the RL coordinator.

        Args:
            coordinator: RLCoordinator instance
        """
        self._coordinator = coordinator

    def get_policy_state(self, learner_name: str) -> PolicyState:
        """Get or create policy state for a learner.

        Args:
            learner_name: Name of the learner

        Returns:
            PolicyState for the learner
        """
        if learner_name not in self._policy_states:
            self._policy_states[learner_name] = PolicyState(learner_name=learner_name)
        return self._policy_states[learner_name]

    def enable_auto_checkpoint(
        self,
        learner_name: str,
        threshold: int = DEFAULT_CHECKPOINT_THRESHOLD,
    ) -> None:
        """Enable automatic checkpointing for a learner.

        Args:
            learner_name: Name of the learner
            threshold: Outcomes between checkpoints
        """
        state = self.get_policy_state(learner_name)
        state.auto_checkpoint_threshold = threshold

        logger.info(
            f"PolicyManager: Enabled auto-checkpoint for {learner_name} "
            f"every {threshold} outcomes"
        )

    def record_outcome(
        self,
        learner_name: str,
        success: bool,
        quality_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an outcome and check for auto-checkpoint/rollback.

        Args:
            learner_name: Name of the learner
            success: Whether the outcome was successful
            quality_score: Quality score (0-1)
            metadata: Optional additional metadata
        """
        state = self.get_policy_state(learner_name)

        # Track outcome
        if learner_name not in self._recent_outcomes:
            self._recent_outcomes[learner_name] = []

        self._recent_outcomes[learner_name].append(
            {
                "success": success,
                "quality": quality_score,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )

        # Keep only recent outcomes
        self._recent_outcomes[learner_name] = self._recent_outcomes[learner_name][
            -self.DEGRADATION_WINDOW * 2 :
        ]

        # Update running metrics
        recent = self._recent_outcomes[learner_name][-self.DEGRADATION_WINDOW :]
        if recent:
            state.recent_success_rate = sum(o["success"] for o in recent) / len(recent)
            state.recent_quality_score = sum(o["quality"] for o in recent) / len(recent)

        # Increment outcome counter
        state.last_checkpoint_outcomes += 1

        # Check for auto-checkpoint
        if state.last_checkpoint_outcomes >= state.auto_checkpoint_threshold:
            self._auto_checkpoint(learner_name)

        # Check for degradation and auto-rollback
        if self.should_rollback(learner_name):
            logger.warning(
                f"PolicyManager: Degradation detected for {learner_name}, "
                f"triggering auto-rollback"
            )
            self.rollback(learner_name, reason="Auto-rollback due to degradation")

    def _auto_checkpoint(self, learner_name: str) -> None:
        """Create automatic checkpoint for a learner.

        Args:
            learner_name: Name of the learner
        """
        if not self._coordinator:
            return

        learner = self._coordinator.get_learner(learner_name)
        if not learner:
            return

        state = self.get_policy_state(learner_name)

        # Generate version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get learner state
        try:
            learner_state = self._export_learner_state(learner)

            # Get current checkpoint for lineage
            latest = self._checkpoint_store.get_latest_checkpoint(learner_name)
            parent_id = latest.checkpoint_id if latest else None

            # Create checkpoint
            self._checkpoint_store.create_checkpoint(
                learner_name=learner_name,
                version=version,
                state=learner_state,
                metadata={
                    "success_rate": state.recent_success_rate,
                    "quality_score": state.recent_quality_score,
                    "outcomes_since_last": state.last_checkpoint_outcomes,
                },
                parent_id=parent_id,
                tags=["auto"],
            )

            # Update state
            state.current_version = version
            state.last_checkpoint_outcomes = 0

            logger.info(
                f"PolicyManager: Auto-checkpoint {version} for {learner_name} "
                f"(success_rate={state.recent_success_rate:.2f})"
            )

        except Exception as e:
            logger.error(f"PolicyManager: Failed to auto-checkpoint {learner_name}: {e}")

    def _export_learner_state(self, learner: "BaseLearner") -> Dict[str, Any]:
        """Export learner state for checkpointing.

        Args:
            learner: Learner instance

        Returns:
            Serializable state dictionary
        """
        state = {"name": learner.name}

        # Export Q-values if available
        if hasattr(learner, "_q_values"):
            state["q_values"] = dict(learner._q_values)

        # Export weights if available
        if hasattr(learner, "_weights"):
            state["weights"] = {
                k: dict(v) if isinstance(v, dict) else v for k, v in learner._weights.items()
            }

        # Export sample counts
        if hasattr(learner, "_sample_counts"):
            state["sample_counts"] = dict(learner._sample_counts)

        # Export metrics
        try:
            state["metrics"] = learner.export_metrics()
        except Exception:
            pass

        return state

    def _import_learner_state(self, learner: "BaseLearner", state: Dict[str, Any]) -> None:
        """Import state into a learner.

        Args:
            learner: Learner instance
            state: State dictionary to import
        """
        # Import Q-values
        if "q_values" in state and hasattr(learner, "_q_values"):
            learner._q_values.clear()
            learner._q_values.update(state["q_values"])

        # Import weights
        if "weights" in state and hasattr(learner, "_weights"):
            learner._weights.clear()
            learner._weights.update(state["weights"])

        # Import sample counts
        if "sample_counts" in state and hasattr(learner, "_sample_counts"):
            learner._sample_counts.clear()
            learner._sample_counts.update(state["sample_counts"])

    def should_rollback(self, learner_name: str) -> bool:
        """Check if a learner should be rolled back.

        Args:
            learner_name: Name of the learner

        Returns:
            True if degradation detected
        """
        outcomes = self._recent_outcomes.get(learner_name, [])

        if len(outcomes) < self.MIN_SAMPLES_FOR_ROLLBACK:
            return False

        state = self.get_policy_state(learner_name)

        # Compare to baseline if available
        if state.performance_baseline:
            baseline_success = state.performance_baseline.get("success_rate", 0.5)
            if state.recent_success_rate < baseline_success - self.DEGRADATION_THRESHOLD:
                return True

        # Compare recent to earlier outcomes
        mid = len(outcomes) // 2
        if mid < 10:
            return False

        early_rate = sum(o["success"] for o in outcomes[:mid]) / mid
        recent_rate = sum(o["success"] for o in outcomes[mid:]) / (len(outcomes) - mid)

        return early_rate - recent_rate > self.DEGRADATION_THRESHOLD

    def rollback(
        self,
        learner_name: str,
        to_version: Optional[str] = None,
        reason: str = "Manual rollback",
    ) -> bool:
        """Rollback a learner to a previous version.

        Args:
            learner_name: Name of the learner
            to_version: Specific version to rollback to (or previous if None)
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        if not self._coordinator:
            return False

        state = self.get_policy_state(learner_name)

        # Find target version
        if to_version:
            target_checkpoint = self._checkpoint_store.get_checkpoint(learner_name, to_version)
        else:
            # Get previous version
            checkpoints = self._checkpoint_store.list_checkpoints(learner_name)
            if len(checkpoints) < 2:
                logger.warning(f"PolicyManager: No previous checkpoint for {learner_name}")
                return False

            # Find current checkpoint index
            current_idx = next(
                (i for i, cp in enumerate(checkpoints) if cp.version == state.current_version),
                -1,
            )

            if current_idx < 0 or current_idx >= len(checkpoints) - 1:
                target_checkpoint = checkpoints[1] if len(checkpoints) > 1 else None
            else:
                target_checkpoint = checkpoints[current_idx + 1]

        if not target_checkpoint:
            logger.warning("PolicyManager: Target checkpoint not found for rollback")
            return False

        # Get learner
        learner = self._coordinator.get_learner(learner_name)
        if not learner:
            return False

        # Record rollback event
        event = RollbackEvent(
            learner_name=learner_name,
            from_version=state.current_version,
            to_version=target_checkpoint.version,
            reason=reason,
            metrics_before={
                "success_rate": state.recent_success_rate,
                "quality_score": state.recent_quality_score,
            },
        )
        self._rollback_history.append(event)

        # Apply rollback
        try:
            self._import_learner_state(learner, target_checkpoint.state)
            state.current_version = target_checkpoint.version

            # Reset performance tracking
            self._recent_outcomes[learner_name] = []
            state.last_checkpoint_outcomes = 0

            logger.info(
                f"PolicyManager: Rolled back {learner_name} from "
                f"{event.from_version} to {event.to_version} ({reason})"
            )

            return True

        except Exception as e:
            logger.error(f"PolicyManager: Rollback failed for {learner_name}: {e}")
            return False

    def start_shadow_mode(
        self,
        learner_name: str,
        shadow_version: str,
    ) -> bool:
        """Start shadow mode for a learner.

        Args:
            learner_name: Name of the learner
            shadow_version: Version to run in shadow mode

        Returns:
            True if started successfully
        """
        checkpoint = self._checkpoint_store.get_checkpoint(learner_name, shadow_version)
        if not checkpoint:
            logger.warning(f"PolicyManager: Shadow version {shadow_version} not found")
            return False

        state = self.get_policy_state(learner_name)
        state.shadow_version = shadow_version
        state.stage = PolicyStage.STAGING

        # Initialize shadow tracking
        self._shadow_states[learner_name] = {
            "version": shadow_version,
            "state": checkpoint.state,
            "outcomes": [],
        }

        logger.info(
            f"PolicyManager: Started shadow mode for {learner_name} "
            f"with version {shadow_version}"
        )

        return True

    def stop_shadow_mode(self, learner_name: str) -> Dict[str, Any]:
        """Stop shadow mode and return comparison results.

        Args:
            learner_name: Name of the learner

        Returns:
            Dictionary with comparison metrics
        """
        state = self.get_policy_state(learner_name)
        state.shadow_version = None
        state.stage = PolicyStage.DEVELOPMENT

        shadow_data = self._shadow_states.pop(learner_name, {})
        shadow_outcomes = shadow_data.get("outcomes", [])

        # Compare outcomes
        main_outcomes = self._recent_outcomes.get(learner_name, [])

        comparison = {
            "shadow_version": shadow_data.get("version"),
            "shadow_outcomes": len(shadow_outcomes),
            "main_outcomes": len(main_outcomes),
            "shadow_success_rate": (
                sum(o.get("success", False) for o in shadow_outcomes) / len(shadow_outcomes)
                if shadow_outcomes
                else 0.0
            ),
            "main_success_rate": (
                sum(o.get("success", False) for o in main_outcomes) / len(main_outcomes)
                if main_outcomes
                else 0.0
            ),
        }

        logger.info(
            f"PolicyManager: Stopped shadow mode for {learner_name}, "
            f"shadow_success={comparison['shadow_success_rate']:.2f}, "
            f"main_success={comparison['main_success_rate']:.2f}"
        )

        return comparison

    def promote_shadow(self, learner_name: str) -> bool:
        """Promote shadow version to production.

        Args:
            learner_name: Name of the learner

        Returns:
            True if promoted successfully
        """
        if learner_name not in self._shadow_states:
            return False

        shadow_data = self._shadow_states[learner_name]
        shadow_version = shadow_data.get("version")

        if not shadow_version:
            return False

        # Stop shadow mode and promote
        self.stop_shadow_mode(learner_name)

        # Apply shadow state to main learner
        if self._coordinator:
            learner = self._coordinator.get_learner(learner_name)
            if learner:
                self._import_learner_state(learner, shadow_data["state"])

        state = self.get_policy_state(learner_name)
        state.current_version = shadow_version
        state.stage = PolicyStage.PRODUCTION

        logger.info(
            f"PolicyManager: Promoted {learner_name} shadow version "
            f"{shadow_version} to production"
        )

        return True

    def set_canary_traffic(self, learner_name: str, percentage: int) -> None:
        """Set canary traffic percentage.

        Args:
            learner_name: Name of the learner
            percentage: Traffic percentage (0-100)
        """
        state = self.get_policy_state(learner_name)
        state.canary_traffic = max(0, min(100, percentage))
        state.stage = PolicyStage.CANARY if percentage > 0 else PolicyStage.PRODUCTION

        logger.info(f"PolicyManager: Set canary traffic for {learner_name} to {percentage}%")

    def set_performance_baseline(
        self,
        learner_name: str,
        success_rate: float,
        quality_score: float,
    ) -> None:
        """Set performance baseline for degradation detection.

        Args:
            learner_name: Name of the learner
            success_rate: Baseline success rate
            quality_score: Baseline quality score
        """
        state = self.get_policy_state(learner_name)
        state.performance_baseline = {
            "success_rate": success_rate,
            "quality_score": quality_score,
        }

    def get_rollback_history(
        self,
        learner_name: Optional[str] = None,
    ) -> List[RollbackEvent]:
        """Get rollback history.

        Args:
            learner_name: Optional filter by learner

        Returns:
            List of RollbackEvent
        """
        if learner_name:
            return [e for e in self._rollback_history if e.learner_name == learner_name]
        return self._rollback_history

    def export_metrics(self) -> Dict[str, Any]:
        """Export policy manager metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "managed_policies": len(self._policy_states),
            "policies_by_stage": {
                stage.value: sum(1 for s in self._policy_states.values() if s.stage == stage)
                for stage in PolicyStage
            },
            "total_rollbacks": len(self._rollback_history),
            "shadow_mode_active": len(self._shadow_states),
            "checkpoint_store": self._checkpoint_store.export_metrics(),
        }


# Global singleton
_policy_manager: Optional[PolicyManager] = None


def get_policy_manager(
    coordinator: Optional["RLCoordinator"] = None,
    checkpoint_store: Optional[CheckpointStore] = None,
) -> PolicyManager:
    """Get global policy manager (lazy init).

    Args:
        coordinator: Optional RL coordinator
        checkpoint_store: Optional checkpoint store

    Returns:
        PolicyManager singleton
    """
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = PolicyManager(coordinator, checkpoint_store)
    return _policy_manager
