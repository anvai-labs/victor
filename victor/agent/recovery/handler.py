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

"""RecoveryHandler - Delegation wrapper for DI integration.

This module provides a thin delegation layer between the orchestrator
and RecoveryCoordinator, enabling:
- DI container registration via RecoveryHandlerProtocol
- Clean separation of concerns
- Easy testing through mock substitution
- Consistent interface with other orchestrator services

DELEGATION PATTERN:
==================
RecoveryHandler delegates to RecoveryCoordinator while implementing
RecoveryHandlerProtocol. This allows:
- Protocol-based DI registration
- Orchestrator to depend on abstraction (protocol) not concrete type
- Future implementation swaps without changing orchestrator

Usage:
    # Via DI container (preferred)
    handler = container.get(RecoveryHandlerProtocol)

    # Direct instantiation (for testing)
    handler = RecoveryHandler.create(settings=settings)

    # Detect and recover
    failure = handler.detect_failure(content, tool_calls, ...)
    if failure:
        outcome = await handler.recover(failure, provider, model, ...)
        handler.record_outcome(success=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.recovery.coordinator import RecoveryCoordinator, RecoveryOutcome
from victor.agent.recovery.protocols import FailureType, RecoveryAction

if TYPE_CHECKING:
    from victor.agent.context_compactor import ContextCompactor
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class RecoveryHandler:
    """Delegation wrapper for RecoveryCoordinator with DI support.

    This class:
    1. Implements RecoveryHandlerProtocol for DI registration
    2. Delegates to RecoveryCoordinator for actual recovery logic
    3. Provides factory methods for easy instantiation
    4. Maintains session state for multi-turn recovery

    Architecture:
        RecoveryHandler (protocol impl) -> RecoveryCoordinator (logic)
                                               |
                                               v
                                    Framework Components:
                                    - QLearningStore
                                    - UsageAnalytics
                                    - ContextCompactor
    """

    def __init__(
        self,
        coordinator: Optional[RecoveryCoordinator] = None,
        enabled: bool = True,
    ):
        """Initialize handler with optional coordinator.

        Args:
            coordinator: RecoveryCoordinator instance (or None for disabled mode)
            enabled: Whether recovery is enabled
        """
        self._coordinator = coordinator
        self._enabled = enabled
        self._session_id: Optional[str] = None

        # Track recent responses for loop detection
        self._recent_responses: List[str] = []
        self._max_recent_responses = 5

        # Track consecutive failures for escalation
        self._consecutive_failures = 0

    @classmethod
    def create(
        cls,
        settings: Optional["Settings"] = None,
        data_dir: Optional[Path] = None,
    ) -> "RecoveryHandler":
        """Create handler with framework integration.

        Args:
            settings: VictorSettings for configuration
            data_dir: Directory for recovery data persistence

        Returns:
            Configured RecoveryHandler
        """
        enabled = True
        if settings:
            enabled = getattr(settings, "enable_recovery_system", True)

        if not enabled:
            logger.debug("RecoveryHandler created in disabled mode")
            return cls(coordinator=None, enabled=False)

        try:
            # Get data directory
            if data_dir is None and settings:
                cache_dir = getattr(settings, "cache_dir", None)
                if cache_dir:
                    data_dir = Path(cache_dir) / "recovery"
                    data_dir.mkdir(parents=True, exist_ok=True)

            coordinator = RecoveryCoordinator.create_with_framework(
                settings=settings,
                data_dir=data_dir,
            )
            logger.debug("RecoveryHandler created with framework integration")
            return cls(coordinator=coordinator, enabled=True)

        except Exception as e:
            logger.warning(f"RecoveryCoordinator creation failed, using disabled mode: {e}")
            return cls(coordinator=None, enabled=False)

    @classmethod
    def create_null(cls) -> "RecoveryHandler":
        """Create a null/no-op handler for testing or disabled mode."""
        return cls(coordinator=None, enabled=False)

    def set_context_compactor(self, compactor: "ContextCompactor") -> None:
        """Wire in context compactor for proactive compaction.

        Called by orchestrator after creating the compactor.
        """
        if self._coordinator:
            self._coordinator.set_context_compactor(compactor)

    def set_session_id(self, session_id: str) -> None:
        """Set current session ID for tracking."""
        self._session_id = session_id
        if self._coordinator:
            self._coordinator.reset_session(session_id)

    def detect_failure(
        self,
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        mentioned_tools: Optional[List[str]] = None,
        elapsed_time: float = 0.0,
        session_idle_timeout: float = 180.0,
        quality_score: float = 0.5,
        consecutive_failures: int = 0,
        recent_responses: Optional[List[str]] = None,
        context_utilization: Optional[float] = None,
    ) -> Optional[FailureType]:
        """Detect failure type from response characteristics.

        Delegates to RecoveryCoordinator.detect_failure()

        Args:
            content: Response content
            tool_calls: Tool calls made
            mentioned_tools: Tools mentioned but not called
            elapsed_time: Session elapsed time
            session_idle_timeout: Session time limit
            quality_score: Response quality score
            consecutive_failures: Count of consecutive failures
            recent_responses: Recent responses for loop detection
            context_utilization: Context usage ratio

        Returns:
            FailureType if failure detected, None otherwise
        """
        if not self._enabled or not self._coordinator:
            return None

        # Use internal tracking if recent_responses not provided
        if recent_responses is None:
            recent_responses = self._recent_responses

        # Use internal consecutive failure count if not provided
        if consecutive_failures == 0:
            consecutive_failures = self._consecutive_failures

        return self._coordinator.detect_failure(
            content=content,
            tool_calls=tool_calls,
            mentioned_tools=mentioned_tools,
            elapsed_time=elapsed_time,
            session_idle_timeout=session_idle_timeout,
            quality_score=quality_score,
            consecutive_failures=consecutive_failures,
            recent_responses=recent_responses,
            context_utilization=context_utilization,
        )

    async def recover(
        self,
        failure_type: FailureType,
        provider: str,
        model: str,
        content: str = "",
        tool_calls_made: int = 0,
        tool_budget: int = 10,
        iteration_count: int = 0,
        max_iterations: int = 50,
        elapsed_time: float = 0.0,
        session_idle_timeout: float = 180.0,
        current_temperature: float = 0.7,
        consecutive_failures: int = 0,
        mentioned_tools: Optional[List[str]] = None,
        recent_responses: Optional[List[str]] = None,
        quality_score: float = 0.5,
        task_type: str = "general",
        is_analysis_task: bool = False,
        is_action_task: bool = False,
        session_id: Optional[str] = None,
    ) -> RecoveryOutcome:
        """Attempt recovery using appropriate strategy.

        Delegates to RecoveryCoordinator.recover()

        Returns:
            RecoveryOutcome with action to take
        """
        if not self._enabled or not self._coordinator:
            # Return no-op outcome
            from victor.agent.recovery.protocols import RecoveryResult

            return RecoveryOutcome(
                result=RecoveryResult(
                    action=RecoveryAction.CONTINUE,
                    success=True,
                    strategy_name="disabled",
                    reason="Recovery system disabled",
                )
            )

        # Track this failure
        self._consecutive_failures += 1

        # Use internal tracking if not provided
        if recent_responses is None:
            recent_responses = self._recent_responses
        if consecutive_failures == 0:
            consecutive_failures = self._consecutive_failures
        if session_id is None:
            session_id = self._session_id

        outcome = await self._coordinator.recover(
            failure_type=failure_type,
            provider=provider,
            model=model,
            content=content,
            tool_calls_made=tool_calls_made,
            tool_budget=tool_budget,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            elapsed_time=elapsed_time,
            session_idle_timeout=session_idle_timeout,
            current_temperature=current_temperature,
            consecutive_failures=consecutive_failures,
            mentioned_tools=mentioned_tools,
            recent_responses=recent_responses,
            quality_score=quality_score,
            task_type=task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            session_id=session_id,
        )

        return outcome

    def record_outcome(
        self,
        success: bool,
        quality_improvement: float = 0.0,
    ) -> None:
        """Record recovery outcome for Q-learning.

        Delegates to RecoveryCoordinator.record_outcome()
        """
        if not self._enabled or not self._coordinator:
            return

        self._coordinator.record_outcome(
            success=success,
            quality_improvement=quality_improvement,
        )

        # Reset consecutive failures on success
        if success:
            self._consecutive_failures = 0

    def track_response(self, content: str) -> None:
        """Track a response for loop detection.

        Call this for each model response to enable stuck loop detection.
        """
        if content and len(content) > 10:
            self._recent_responses.append(content)
            if len(self._recent_responses) > self._max_recent_responses:
                self._recent_responses.pop(0)

    def reset_session(self, session_id: str) -> None:
        """Reset recovery state for a new session."""
        self._session_id = session_id
        self._recent_responses.clear()
        self._consecutive_failures = 0

        if self._coordinator:
            self._coordinator.reset_session(session_id)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about recovery system."""
        diagnostics = {
            "enabled": self._enabled,
            "session_id": self._session_id,
            "consecutive_failures": self._consecutive_failures,
            "recent_responses_count": len(self._recent_responses),
        }

        if self._coordinator:
            diagnostics.update(self._coordinator.get_diagnostics())

        return diagnostics

    @property
    def enabled(self) -> bool:
        """Whether recovery is enabled."""
        return self._enabled

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count."""
        return self._consecutive_failures


def create_recovery_handler(
    settings: Optional["Settings"] = None,
    data_dir: Optional[Path] = None,
) -> RecoveryHandler:
    """Factory function for creating RecoveryHandler.

    Convenience function for DI container registration.

    Args:
        settings: VictorSettings for configuration
        data_dir: Directory for recovery data persistence

    Returns:
        Configured RecoveryHandler
    """
    return RecoveryHandler.create(settings=settings, data_dir=data_dir)
