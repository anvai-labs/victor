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

"""Orchestrator recovery integration - submodule for recovery handling.

This module provides a clean delegation layer for recovery logic, extracting
complex recovery handling from the main orchestrator stream_chat loop.

ARCHITECTURE:
============
This is an internal orchestrator submodule (not registered in DI).
The orchestrator creates and owns this component, delegating recovery
decisions to it.

Responsibilities:
- Detect failures from model responses
- Apply recovery strategies via RecoveryHandler
- Track recovery state across iterations
- Provide recovery prompts and temperature adjustments

Integration:
    # In orchestrator __init__
    self._recovery_integration = OrchestratorRecoveryIntegration(
        recovery_handler=self._recovery_handler,
        settings=self.settings,
    )

    # In stream_chat loop
    recovery_action = await self._recovery_integration.handle_response(
        content=full_content,
        tool_calls=tool_calls,
        ...
    )
    if recovery_action.requires_retry:
        # Apply recovery_action.message, temperature, etc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.recovery import RecoveryHandler, RecoveryOutcome
    from victor.agent.recovery.protocols import FailureType
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RecoveryState:
    """Mutable state for recovery tracking across iterations."""

    consecutive_empty_responses: int = 0
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    last_failure_type: Optional["FailureType"] = None
    recent_responses: List[str] = field(default_factory=list)
    session_start_time: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Reset state for new session."""
        self.consecutive_empty_responses = 0
        self.consecutive_failures = 0
        self.recovery_attempts = 0
        self.last_failure_type = None
        self.recent_responses.clear()
        self.session_start_time = time.time()

    def track_response(self, content: str, max_recent: int = 5) -> None:
        """Track a response for loop detection."""
        if content and len(content) > 10:
            self.recent_responses.append(content)
            if len(self.recent_responses) > max_recent:
                self.recent_responses.pop(0)

    def on_success(self) -> None:
        """Reset failure counters on success."""
        self.consecutive_empty_responses = 0
        self.consecutive_failures = 0
        self.last_failure_type = None

    def on_failure(self, failure_type: "FailureType") -> None:
        """Track a failure."""
        self.consecutive_failures += 1
        self.last_failure_type = failure_type


@dataclass
class OrchestratorRecoveryAction:
    """Action to take after recovery analysis.

    Renamed from RecoveryAction to be semantically distinct:
    - ErrorRecoveryAction (victor.agent.error_recovery): Tool error recovery enum
    - StrategyRecoveryAction (victor.agent.recovery.protocols): Recovery strategy enum
    - OrchestratorRecoveryAction (here): Orchestrator recovery action dataclass
    """

    # Action type
    action: str  # "continue", "retry", "abort", "force_summary"

    # For retry actions
    message: Optional[str] = None
    new_temperature: Optional[float] = None

    # For model switch
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None

    # Metadata
    reason: str = ""
    failure_type: Optional["FailureType"] = None
    strategy_name: str = ""
    confidence: float = 0.0

    @property
    def requires_retry(self) -> bool:
        """Whether this action requires retrying the model call."""
        return self.action in ("retry", "force_summary")

    @property
    def requires_model_switch(self) -> bool:
        """Whether this action requires switching models."""
        return self.fallback_provider is not None


# Backward compatibility alias
RecoveryAction = OrchestratorRecoveryAction


class OrchestratorRecoveryIntegration:
    """Submodule for recovery handling in orchestrator.

    This class encapsulates recovery logic, making the main stream_chat
    loop cleaner and more maintainable.

    Design:
    - Owned by orchestrator (not registered in DI)
    - Delegates to RecoveryHandler (which IS in DI)
    - Tracks recovery state across iterations
    - Provides clean API for recovery decisions
    """

    def __init__(
        self,
        recovery_handler: Optional["RecoveryHandler"],
        settings: "Settings",
        session_id: Optional[str] = None,
    ):
        """Initialize recovery integration.

        Args:
            recovery_handler: DI-provided RecoveryHandler (or None if disabled)
            settings: Application settings
            session_id: Optional session ID for state tracking
        """
        self._handler = recovery_handler
        self._settings = settings
        self._session_id = session_id or f"session-{int(time.time())}"

        # Recovery state tracking
        self._state = RecoveryState()

        # Configuration
        self._empty_response_threshold = getattr(settings, "recovery_empty_response_threshold", 3)
        self._max_recovery_attempts = getattr(settings, "recovery_max_attempts", 5)

        # Initialize handler with session
        if self._handler:
            self._handler.set_session_id(self._session_id)

    @property
    def enabled(self) -> bool:
        """Whether recovery is enabled."""
        return self._handler is not None and getattr(self._handler, "enabled", True)

    @property
    def state(self) -> RecoveryState:
        """Current recovery state."""
        return self._state

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset for a new session."""
        self._session_id = session_id or f"session-{int(time.time())}"
        self._state.reset()
        if self._handler:
            self._handler.reset_session(self._session_id)

    async def handle_response(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        provider_name: str,
        model_name: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int,
        max_iterations: int,
        current_temperature: float,
        quality_score: float = 0.5,
        task_type: str = "general",
        is_analysis_task: bool = False,
        is_action_task: bool = False,
        context_utilization: Optional[float] = None,
    ) -> RecoveryAction:
        """Handle a model response and determine recovery action.

        This is the main entry point called from stream_chat.

        Args:
            content: Model response content
            tool_calls: Tool calls made (if any)
            mentioned_tools: Tools mentioned but not called
            provider_name: Current provider name
            model_name: Current model name
            tool_calls_made: Number of tool calls made in session
            tool_budget: Total tool budget
            iteration_count: Current iteration
            max_iterations: Maximum iterations
            current_temperature: Current temperature
            quality_score: Response quality score
            task_type: Classified task type
            is_analysis_task: Whether task is analysis-oriented
            is_action_task: Whether task is action-oriented
            context_utilization: Context window utilization ratio

        Returns:
            RecoveryAction with action to take
        """
        # Track this response
        self._state.track_response(content)

        # Calculate elapsed time
        elapsed_time = time.time() - self._state.session_start_time
        session_idle_timeout = getattr(self._settings, "session_idle_timeout", 180)

        # Check for normal completion (has content and/or tool calls)
        if (content and len(content) > 50) or tool_calls:
            self._state.on_success()
            return RecoveryAction(
                action="continue",
                reason="Response has content or tool calls",
            )

        # If recovery is disabled, return continue
        if not self.enabled:
            return RecoveryAction(
                action="continue",
                reason="Recovery disabled",
            )

        # Detect failure type
        failure_type = self._handler.detect_failure(
            content=content,
            tool_calls=tool_calls,
            mentioned_tools=mentioned_tools,
            elapsed_time=elapsed_time,
            session_idle_timeout=session_idle_timeout,
            quality_score=quality_score,
            consecutive_failures=self._state.consecutive_failures,
            recent_responses=self._state.recent_responses,
            context_utilization=context_utilization,
        )

        if failure_type is None:
            self._state.on_success()
            return RecoveryAction(
                action="continue",
                reason="No failure detected",
            )

        # Track the failure
        self._state.on_failure(failure_type)
        self._state.recovery_attempts += 1

        # Check if we've exceeded recovery attempts
        if self._state.recovery_attempts >= self._max_recovery_attempts:
            return RecoveryAction(
                action="abort",
                failure_type=failure_type,
                reason=f"Max recovery attempts ({self._max_recovery_attempts}) exceeded",
            )

        # Get recovery outcome from handler
        outcome = await self._handler.recover(
            failure_type=failure_type,
            provider=provider_name,
            model=model_name,
            content=content,
            tool_calls_made=tool_calls_made,
            tool_budget=tool_budget,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            elapsed_time=elapsed_time,
            session_idle_timeout=session_idle_timeout,
            current_temperature=current_temperature,
            consecutive_failures=self._state.consecutive_failures,
            mentioned_tools=mentioned_tools,
            recent_responses=self._state.recent_responses,
            quality_score=quality_score,
            task_type=task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            session_id=self._session_id,
        )

        # Convert RecoveryOutcome to RecoveryAction
        return self._convert_outcome(outcome, failure_type)

    def _convert_outcome(
        self,
        outcome: "RecoveryOutcome",
        failure_type: "FailureType",
    ) -> RecoveryAction:
        """Convert RecoveryOutcome to RecoveryAction."""
        from victor.agent.recovery.protocols import RecoveryAction as RA

        result = outcome.result

        # Map recovery action to our action type
        action_map = {
            RA.CONTINUE: "continue",
            RA.PROMPT_TOOL_CALL: "retry",
            RA.FORCE_SUMMARY: "force_summary",
            RA.ADJUST_TEMPERATURE: "retry",
            RA.SWITCH_MODEL: "retry",
            RA.COMPACT_CONTEXT: "retry",
            RA.RETRY_WITH_TEMPLATE: "retry",
            RA.WAIT_AND_RETRY: "retry",
            RA.ABORT: "abort",
        }

        action_type = action_map.get(result.action, "continue")

        return RecoveryAction(
            action=action_type,
            message=result.message,
            new_temperature=outcome.new_temperature or result.new_temperature,
            fallback_provider=outcome.fallback_provider,
            fallback_model=outcome.fallback_model,
            reason=result.reason,
            failure_type=failure_type,
            strategy_name=result.strategy_name,
            confidence=result.confidence,
        )

    def record_outcome(self, success: bool, quality_improvement: float = 0.0) -> None:
        """Record the outcome of a recovery attempt.

        Call this after applying recovery action to provide feedback
        for Q-learning.
        """
        if self._handler:
            self._handler.record_outcome(success, quality_improvement)

        if success:
            self._state.on_success()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diagnostics = {
            "enabled": self.enabled,
            "session_id": self._session_id,
            "state": {
                "consecutive_empty_responses": self._state.consecutive_empty_responses,
                "consecutive_failures": self._state.consecutive_failures,
                "recovery_attempts": self._state.recovery_attempts,
                "last_failure_type": (
                    self._state.last_failure_type.name if self._state.last_failure_type else None
                ),
                "recent_responses_count": len(self._state.recent_responses),
            },
        }

        if self._handler:
            diagnostics["handler"] = self._handler.get_diagnostics()

        return diagnostics


def create_recovery_integration(
    recovery_handler: Optional["RecoveryHandler"],
    settings: "Settings",
    session_id: Optional[str] = None,
) -> OrchestratorRecoveryIntegration:
    """Factory function for creating recovery integration.

    Args:
        recovery_handler: RecoveryHandler from DI container
        settings: Application settings
        session_id: Optional session ID

    Returns:
        Configured OrchestratorRecoveryIntegration
    """
    return OrchestratorRecoveryIntegration(
        recovery_handler=recovery_handler,
        settings=settings,
        session_id=session_id,
    )
