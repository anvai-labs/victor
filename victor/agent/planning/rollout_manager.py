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

"""Rollout manager for gradual feature rollout.

Manages the gradual rollout of predictive features with:
- Consistent hashing for percentage-based rollout
- Monitoring and metrics collection
- Automatic rollback on error thresholds
- Progress tracking and reporting
"""

from __future__ import annotations

import hashlib
from victor.core.json_utils import json_dumps, json_loads
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RolloutStage(str, Enum):
    """Rollout stages for gradual deployment."""

    CANARY = "canary"  # 1% - Initial testing
    EARLY_ADOPTERS = "early_adopters"  # 10% - Broader testing
    BETA = "beta"  # 50% - Majority of users
    GENERAL = "general"  # 100% - All users


@dataclass
class RolloutMetrics:
    """Metrics for rollout monitoring.

    Attributes:
        total_requests: Total number of requests
        predictive_requests: Requests using predictive features
        errors: Number of errors with predictive features
        latency_ms: Average latency in milliseconds
        last_updated: When metrics were last updated
    """

    total_requests: int = 0
    predictive_requests: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.predictive_requests == 0:
            return 0.0
        return self.errors / self.predictive_requests

    def rollout_percentage(self) -> float:
        """Calculate actual rollout percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.predictive_requests / self.total_requests) * 100


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout.

    Attributes:
        stages: Rollout stages with percentages
        error_threshold: Error rate threshold for automatic rollback
        min_requests_before_rollout: Minimum requests before moving to next stage
        cooldown_seconds: Cooldown between stage transitions
    """

    stages: Dict[RolloutStage, int] = field(
        default_factory=lambda: {
            RolloutStage.CANARY: 1,
            RolloutStage.EARLY_ADOPTERS: 10,
            RolloutStage.BETA: 50,
            RolloutStage.GENERAL: 100,
        }
    )
    error_threshold: float = 0.05  # 5% error rate triggers rollback
    min_requests_before_rollout: int = 100
    cooldown_seconds: int = 3600  # 1 hour between stages


class RolloutManager:
    """Manages gradual rollout of predictive features.

    The rollout manager provides:
    - Consistent hashing for percentage-based rollout
    - Metrics collection and monitoring
    - Automatic rollback on error thresholds
    - Stage progression tracking

    Example:
        manager = RolloutManager(config=RolloutConfig())

        # Check if predictive features should be used
        if manager.should_use_predictive(session_id="abc123"):
            # Use predictive features
            pass

        # Record metrics
        manager.record_request(
            session_id="abc123",
            used_predictive=True,
            success=True,
            latency_ms=150,
        )
    """

    def __init__(
        self,
        config: Optional[RolloutConfig] = None,
        metrics_path: Optional[Path] = None,
    ):
        """Initialize the rollout manager.

        Args:
            config: Rollout configuration
            metrics_path: Path to store metrics for persistence
        """
        self.config = config or RolloutConfig()
        self.metrics_path = metrics_path or Path.home() / ".victor" / "metrics" / "rollout.jsonl"

        # Current stage
        self._current_stage = RolloutStage.CANARY
        self._stage_start_time = time.time()

        # Metrics tracking
        self._metrics: Dict[str, RolloutMetrics] = {}

        # Load existing metrics if available
        self._load_metrics()

        logger.info(
            f"RolloutManager initialized (stage={self._current_stage}, "
            f"percentage={self.config.stages[self._current_stage]}%)"
        )

    def should_use_predictive(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Determine if predictive features should be used for this request.

        Uses consistent hashing to ensure the same session/user always
        gets the same treatment during a rollout stage.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for user-based rollout

        Returns:
            True if predictive features should be used
        """
        # Get current rollout percentage
        rollout_percentage = self.config.stages[self._current_stage]

        if rollout_percentage >= 100:
            return True

        if rollout_percentage <= 0:
            return False

        # Use consistent hashing based on session_id or user_id
        identifier = user_id or session_id
        hash_value = self._hash_identifier(identifier)

        # Check if hash falls within rollout percentage
        return (hash_value % 100) < rollout_percentage

    def record_request(
        self,
        session_id: str,
        used_predictive: bool,
        success: bool,
        latency_ms: float = 0.0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record metrics for a request.

        Args:
            session_id: Session identifier
            used_predictive: Whether predictive features were used
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            error_message: Optional error message if failed
        """
        # Get or create metrics for this session
        if session_id not in self._metrics:
            self._metrics[session_id] = RolloutMetrics()

        metrics = self._metrics[session_id]
        metrics.total_requests += 1

        if used_predictive:
            metrics.predictive_requests += 1
            if not success:
                metrics.errors += 1
                logger.warning(f"Predictive request failed (session={session_id}): {error_message}")

        # Update latency (exponential moving average)
        if latency_ms > 0:
            alpha = 0.1  # Smoothing factor
            if metrics.latency_ms == 0:
                metrics.latency_ms = latency_ms
            else:
                metrics.latency_ms = alpha * latency_ms + (1 - alpha) * metrics.latency_ms

        metrics.last_updated = datetime.now(timezone.utc)

        # Check if we should rollback
        if metrics.error_rate() > self.config.error_threshold:
            logger.error(
                f"Error rate {metrics.error_rate():.2%} exceeds threshold "
                f"{self.config.error_threshold:.2%}, considering rollback"
            )

        # Save metrics periodically
        if metrics.total_requests % 10 == 0:
            self._save_metrics()

    def can_advance_to_next_stage(self) -> bool:
        """Check if rollout can advance to the next stage.

        Returns:
            True if criteria met for advancing to next stage
        """
        # Check cooldown
        elapsed = time.time() - self._stage_start_time
        if elapsed < self.config.cooldown_seconds:
            logger.debug(f"Cooldown not met ({elapsed:.0f}s < {self.config.cooldown_seconds}s)")
            return False

        # Check minimum requests
        total_requests = sum(m.total_requests for m in self._metrics.values())
        if total_requests < self.config.min_requests_before_rollout:
            logger.debug(
                f"Minimum requests not met ({total_requests} < "
                f"{self.config.min_requests_before_rollout})"
            )
            return False

        # Check error rate
        overall_error_rate = self._get_overall_error_rate()
        if overall_error_rate > self.config.error_threshold:
            logger.warning(f"Cannot advance: error rate {overall_error_rate:.2%} exceeds threshold")
            return False

        return True

    def advance_to_next_stage(self) -> Optional[RolloutStage]:
        """Advance to the next rollout stage.

        Returns:
            New stage if advanced, None if already at final stage
        """
        stages = list(RolloutStage)
        current_index = stages.index(self._current_stage)

        if current_index >= len(stages) - 1:
            logger.info("Already at final rollout stage")
            return None

        if not self.can_advance_to_next_stage():
            logger.warning("Cannot advance to next stage yet")
            return None

        new_stage = stages[current_index + 1]
        old_stage = self._current_stage
        self._current_stage = new_stage
        self._stage_start_time = time.time()

        logger.info(
            f"Advanced rollout from {old_stage} ({self.config.stages[old_stage]}%) "
            f"to {new_stage} ({self.config.stages[new_stage]}%)"
        )

        return new_stage

    def rollback(self, reason: str = "manual") -> RolloutStage:
        """Rollback to canary stage.

        Args:
            reason: Reason for rollback

        Returns:
            The stage we rolled back to
        """
        old_stage = self._current_stage
        self._current_stage = RolloutStage.CANARY
        self._stage_start_time = time.time()

        logger.warning(f"Rolled back from {old_stage} to {RolloutStage.CANARY}: {reason}")

        return RolloutStage.CANARY

    def get_current_stage(self) -> RolloutStage:
        """Get the current rollout stage.

        Returns:
            Current rollout stage
        """
        return self._current_stage

    def get_rollout_percentage(self) -> int:
        """Get the current rollout percentage.

        Returns:
            Current rollout percentage (0-100)
        """
        return self.config.stages[self._current_stage]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of rollout metrics.

        Returns:
            Dictionary with metrics summary
        """
        total_requests = sum(m.total_requests for m in self._metrics.values())
        total_predictive = sum(m.predictive_requests for m in self._metrics.values())
        total_errors = sum(m.errors for m in self._metrics.values())

        # Calculate average latency
        latencies = [m.latency_ms for m in self._metrics.values() if m.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "current_stage": self._current_stage.value,
            "rollout_percentage": self.get_rollout_percentage(),
            "total_sessions": len(self._metrics),
            "total_requests": total_requests,
            "predictive_requests": total_predictive,
            "errors": total_errors,
            "error_rate": (total_errors / total_predictive if total_predictive > 0 else 0.0),
            "actual_rollout_percentage": (
                (total_predictive / total_requests * 100) if total_requests > 0 else 0.0
            ),
            "avg_latency_ms": avg_latency,
            "time_in_stage": time.time() - self._stage_start_time,
        }

    def _hash_identifier(self, identifier: str) -> int:
        """Hash an identifier to a consistent integer.

        Args:
            identifier: String to hash

        Returns:
            Integer hash value
        """
        return int(hashlib.md5(identifier.encode()).hexdigest(), 16)

    def _get_overall_error_rate(self) -> float:
        """Get overall error rate across all sessions.

        Returns:
            Error rate (0.0-1.0)
        """
        total_predictive = sum(m.predictive_requests for m in self._metrics.values())
        total_errors = sum(m.errors for m in self._metrics.values())

        if total_predictive == 0:
            return 0.0

        return total_errors / total_predictive

    def _load_metrics(self) -> None:
        """Load metrics from disk if available."""
        if not self.metrics_path.exists():
            return

        try:
            with open(self.metrics_path, "r") as f:
                for line in f:
                    data = json_loads(line.strip())
                    session_id = data.get("session_id")
                    if session_id:
                        self._metrics[session_id] = RolloutMetrics(**data)

            logger.debug(f"Loaded metrics for {len(self._metrics)} sessions")

        except Exception as e:
            logger.warning(f"Failed to load metrics: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metrics_path, "w") as f:
                for session_id, metrics in self._metrics.items():
                    data = {
                        "session_id": session_id,
                        "total_requests": metrics.total_requests,
                        "predictive_requests": metrics.predictive_requests,
                        "errors": metrics.errors,
                        "latency_ms": metrics.latency_ms,
                        "last_updated": metrics.last_updated.isoformat(),
                    }
                    f.write(json_dumps(data) + "\n")

        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


__all__ = [
    "RolloutStage",
    "RolloutMetrics",
    "RolloutConfig",
    "RolloutManager",
]
