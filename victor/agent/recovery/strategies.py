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

"""Concrete recovery strategy implementations.

Each strategy follows Single Responsibility Principle - handling one failure type.
Strategies can be composed using CompositeRecoveryStrategy.

The strategies integrate with Q-learning for adaptive behavior:
- Each strategy maintains success/failure counts
- Q-learning weights influence priority calculation
- Template selection is learned over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.agent.recovery.protocols import (
    FailureType,
    QLearningStore,
    RecoveryAction,
    RecoveryContext,
    RecoveryResult,
    RecoveryStrategy,
)
from victor.framework.temperature import escalate_temperature

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for tracking strategy effectiveness."""

    attempts: int = 0
    successes: int = 0
    total_quality_improvement: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.5  # Prior
        return self.successes / self.attempts

    @property
    def avg_quality_improvement(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_quality_improvement / self.attempts


class BaseRecoveryStrategy:
    """Base class providing common functionality for recovery strategies.

    Not using ABC because we want Protocol-based structural typing.
    This provides default implementations that can be overridden.
    """

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        base_priority: float = 0.5,
    ):
        self._q_store = q_store
        self._base_priority = base_priority
        self._metrics = StrategyMetrics()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def handles_failure_types(self) -> List[FailureType]:
        return []  # Override in subclasses

    def can_handle(self, context: RecoveryContext) -> bool:
        return context.failure_type in self.handles_failure_types

    def calculate_priority(self, context: RecoveryContext) -> float:
        """Calculate priority using base + Q-learning adjustment."""
        priority = self._base_priority

        # Adjust based on Q-learning if available
        if self._q_store:
            state_key = context.to_state_key()
            # Get all actions and find the best one
            all_actions = self._q_store.get_all_actions(state_key)
            if all_actions:
                best_action_key = max(all_actions, key=all_actions.get)
                q_value = all_actions[best_action_key]
                # Boost priority if Q-learning suggests our action type
                if self._matches_action_key(best_action_key):
                    priority += q_value * 0.3  # 30% boost from Q-learning

        # Adjust based on historical success rate
        priority += (self._metrics.success_rate - 0.5) * 0.2

        return min(1.0, max(0.0, priority))

    def _matches_action(self, action: RecoveryAction) -> bool:
        """Check if an action matches this strategy's typical actions.

        Override in subclasses.
        """
        return False

    def _matches_action_key(self, action_key: str) -> bool:
        """Check if an action key string matches this strategy's typical actions.

        Override in subclasses for Q-learning priority adjustment.
        """
        return False

    def record_outcome(
        self,
        context: RecoveryContext,
        result: RecoveryResult,
        success: bool,
    ) -> None:
        """Record outcome for learning."""
        self._metrics.attempts += 1
        if success:
            self._metrics.successes += 1

        # Update Q-learning if available
        if self._q_store:
            state_key = context.to_state_key()
            reward = 1.0 if success else -0.5
            # Convert enum to string name for Q-learning store
            action_key = (
                result.action.name if hasattr(result.action, "name") else str(result.action)
            )
            self._q_store.update_q_value(state_key, action_key, reward)


class EmptyResponseRecovery(BaseRecoveryStrategy):
    """Recovery strategy for empty model responses.

    Handles cases where model returns no content and no tool calls.
    Uses progressive prompting with escalating assertiveness.
    """

    # Escalating prompt templates
    PROMPT_TEMPLATES = [
        # Level 1: Gentle reminder
        (
            "Your response was empty. Please provide a response. "
            "You can either make a tool call or provide text content."
        ),
        # Level 2: More specific
        (
            "No response received. Please either:\n"
            "1. Call a tool (read, search, ls) to gather information, OR\n"
            "2. Provide your analysis based on what you know.\n\n"
            "Respond NOW."
        ),
        # Level 3: Assertive with structure
        (
            "CRITICAL: Empty response detected. You MUST respond with one of:\n\n"
            "OPTION A - Make a tool call:\n"
            '```json\n{{"name": "read", "arguments": {{"path": "..."}}}}\n```\n\n'
            "OPTION B - Provide text response:\n"
            "Start your response with 'Based on my analysis...' and provide content.\n\n"
            "Choose one option NOW."
        ),
    ]

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        max_retries: int = 3,
    ):
        super().__init__(q_store, base_priority=0.8)
        self._max_retries = max_retries

    @property
    def handles_failure_types(self) -> List[FailureType]:
        return [FailureType.EMPTY_RESPONSE]

    def _matches_action(self, action: RecoveryAction) -> bool:
        return action in (
            RecoveryAction.PROMPT_TOOL_CALL,
            RecoveryAction.ADJUST_TEMPERATURE,
        )

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery from empty response."""
        # Select prompt based on consecutive failures
        prompt_idx = min(context.consecutive_failures, len(self.PROMPT_TEMPLATES) - 1)
        prompt = self.PROMPT_TEMPLATES[prompt_idx]

        # Determine action based on failure count
        if context.consecutive_failures >= self._max_retries:
            # Give up on empty response recovery, force summary
            return RecoveryResult(
                action=RecoveryAction.FORCE_SUMMARY,
                success=False,
                message=(
                    "Multiple empty responses received. Please provide a summary "
                    "of what you know about this task, or explain what is preventing "
                    "you from responding."
                ),
                strategy_name=self.name,
                confidence=0.9,
                reason=f"Max retries ({self._max_retries}) exceeded for empty response",
            )

        # Check if temperature adjustment might help
        if context.consecutive_failures >= 2 and context.current_temperature < 0.9:
            return RecoveryResult(
                action=RecoveryAction.ADJUST_TEMPERATURE,
                success=True,
                message=prompt,
                new_temperature=escalate_temperature(context.current_temperature, 0.2, cap=1.0),
                strategy_name=self.name,
                confidence=0.7,
                reason=f"Empty response retry {context.consecutive_failures + 1} with temperature boost",
            )

        return RecoveryResult(
            action=RecoveryAction.PROMPT_TOOL_CALL,
            success=True,
            message=prompt,
            strategy_name=self.name,
            confidence=0.75,
            reason=f"Empty response retry {context.consecutive_failures + 1}",
        )


class StuckLoopRecovery(BaseRecoveryStrategy):
    """Recovery strategy for stuck planning loops.

    Handles cases where model keeps saying what it will do but never does it.
    Uses pattern detection and assertive prompting.
    """

    PROMPT_TEMPLATES = [
        # Level 1: Direct instruction
        (
            "You appear to be stuck in a planning loop. You keep describing what "
            "you will do but are not making actual tool calls.\n\n"
            "Please either:\n"
            "1. Make an ACTUAL tool call NOW (not just describe it), OR\n"
            "2. Provide your response based on what you already know."
        ),
        # Level 2: More assertive
        (
            "STUCK LOOP DETECTED. Your responses are not making progress.\n\n"
            "DO NOT describe what you will do. DO one of these:\n"
            "- CALL a tool: read(), search(), ls()\n"
            "- PROVIDE your answer NOW\n\n"
            "Action, not words."
        ),
        # Level 3: Force completion
        (
            "FINAL WARNING: You have been planning without executing.\n\n"
            "Provide your FINAL RESPONSE now. No more tool calls will be allowed. "
            "Summarize what you know and answer the user's question."
        ),
    ]

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        loop_threshold: int = 2,
    ):
        super().__init__(q_store, base_priority=0.85)
        self._loop_threshold = loop_threshold

    @property
    def handles_failure_types(self) -> List[FailureType]:
        return [FailureType.STUCK_LOOP, FailureType.REPEATED_RESPONSE]

    def _matches_action(self, action: RecoveryAction) -> bool:
        return action in (RecoveryAction.FORCE_SUMMARY, RecoveryAction.PROMPT_TOOL_CALL)

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery from stuck loop."""
        prompt_idx = min(context.consecutive_failures, len(self.PROMPT_TEMPLATES) - 1)
        prompt = self.PROMPT_TEMPLATES[prompt_idx]

        # If many failures, force summary
        if context.consecutive_failures >= len(self.PROMPT_TEMPLATES):
            return RecoveryResult(
                action=RecoveryAction.FORCE_SUMMARY,
                success=True,
                message=prompt,
                strategy_name=self.name,
                confidence=0.9,
                reason="Stuck loop - forcing completion",
            )

        # If model has made some tool calls, encourage completion
        if context.tool_calls_made > 0:
            return RecoveryResult(
                action=RecoveryAction.FORCE_SUMMARY,
                success=True,
                message=(
                    f"You have made {context.tool_calls_made} tool calls. "
                    "Please provide your analysis NOW based on what you've learned. "
                    "No more exploration needed."
                ),
                strategy_name=self.name,
                confidence=0.85,
                reason="Stuck loop with prior tool calls - requesting summary",
            )

        return RecoveryResult(
            action=RecoveryAction.PROMPT_TOOL_CALL,
            success=True,
            message=prompt,
            strategy_name=self.name,
            confidence=0.75,
            reason=f"Stuck loop recovery attempt {context.consecutive_failures + 1}",
        )


class HallucinatedToolRecovery(BaseRecoveryStrategy):
    """Recovery strategy for hallucinated tool calls.

    Handles cases where model mentions tools but doesn't actually call them.
    """

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        max_hallucinations: int = 3,
    ):
        super().__init__(q_store, base_priority=0.9)
        self._max_hallucinations = max_hallucinations

    @property
    def handles_failure_types(self) -> List[FailureType]:
        return [FailureType.HALLUCINATED_TOOL]

    def _matches_action(self, action: RecoveryAction) -> bool:
        return action == RecoveryAction.RETRY_WITH_TEMPLATE

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt recovery from hallucinated tool call."""
        mentioned = ", ".join(context.mentioned_tools) if context.mentioned_tools else "tools"

        if context.consecutive_failures >= self._max_hallucinations:
            # Give up on tool calls, force text response
            return RecoveryResult(
                action=RecoveryAction.FORCE_SUMMARY,
                success=True,
                message=(
                    f"You have mentioned {mentioned} multiple times but never called them. "
                    "You appear unable to make tool calls. Please provide your response "
                    "NOW based only on what you already know. Do NOT mention any tools."
                ),
                strategy_name=self.name,
                confidence=0.9,
                reason=f"Max hallucinations ({self._max_hallucinations}) exceeded",
            )

        # First few attempts: teach correct format
        if context.consecutive_failures == 0:
            message = (
                f"You mentioned {mentioned} but did not make an actual tool call. "
                "To call a tool, you must use the proper format.\n\n"
                "Please ACTUALLY call the tool now, or provide your analysis."
            )
        else:
            message = (
                f"CRITICAL: You mentioned {mentioned} but did NOT execute a tool call.\n\n"
                "Your response contained TEXT describing what you would do, "
                "but no actual tool invocation.\n\n"
                "You MUST respond with an ACTUAL tool call. "
                "If you cannot call tools, provide your answer NOW."
            )

        return RecoveryResult(
            action=RecoveryAction.RETRY_WITH_TEMPLATE,
            success=True,
            message=message,
            strategy_name=self.name,
            confidence=0.8,
            reason=f"Hallucinated tool recovery attempt {context.consecutive_failures + 1}",
        )


class TimeoutRecovery(BaseRecoveryStrategy):
    """Recovery strategy for approaching timeouts.

    Handles cases where session time is running out.
    """

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        warning_threshold: float = 0.8,  # 80% of time limit
    ):
        super().__init__(q_store, base_priority=0.95)
        self._warning_threshold = warning_threshold

    @property
    def handles_failure_types(self) -> List[FailureType]:
        return [FailureType.TIMEOUT_APPROACHING]

    def _matches_action(self, action: RecoveryAction) -> bool:
        return action == RecoveryAction.FORCE_SUMMARY

    def can_handle(self, context: RecoveryContext) -> bool:
        """Check if approaching timeout."""
        if context.failure_type == FailureType.TIMEOUT_APPROACHING:
            return True
        # Also handle preemptively
        time_ratio = context.elapsed_time_seconds / max(context.session_idle_timeout, 1)
        return time_ratio >= self._warning_threshold

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Force completion due to timeout."""
        remaining = context.session_idle_timeout - context.elapsed_time_seconds
        remaining_str = f"{remaining:.0f}s" if remaining > 0 else "no time"

        return RecoveryResult(
            action=RecoveryAction.FORCE_SUMMARY,
            success=True,
            message=(
                f"TIME LIMIT APPROACHING ({remaining_str} remaining).\n\n"
                "Provide your response NOW. Summarize your findings and "
                "answer the user's question. Do NOT call any more tools."
            ),
            strategy_name=self.name,
            confidence=0.95,
            reason=f"Timeout approaching - {remaining_str} remaining",
        )


class CompositeRecoveryStrategy(BaseRecoveryStrategy):
    """Composite strategy that delegates to specialized strategies.

    Uses Strategy pattern to select the best recovery approach.
    Implements priority-based strategy selection with Q-learning weighting.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseRecoveryStrategy]] = None,
        q_store: Optional[QLearningStore] = None,
    ):
        super().__init__(q_store, base_priority=1.0)
        self._strategies = strategies or self._create_default_strategies(q_store)

    def _create_default_strategies(
        self,
        q_store: Optional[QLearningStore],
    ) -> List[BaseRecoveryStrategy]:
        """Create default set of recovery strategies."""
        return [
            EmptyResponseRecovery(q_store),
            StuckLoopRecovery(q_store),
            HallucinatedToolRecovery(q_store),
            TimeoutRecovery(q_store),
        ]

    @property
    def handles_failure_types(self) -> List[FailureType]:
        """Returns all failure types handled by child strategies."""
        types = set()
        for strategy in self._strategies:
            types.update(strategy.handles_failure_types)
        return list(types)

    def can_handle(self, context: RecoveryContext) -> bool:
        """Check if any child strategy can handle."""
        return any(s.can_handle(context) for s in self._strategies)

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Select best strategy and delegate recovery."""
        # Find applicable strategies
        applicable = [s for s in self._strategies if s.can_handle(context)]

        if not applicable:
            logger.warning(f"No strategy found for failure type: {context.failure_type}")
            return RecoveryResult(
                action=RecoveryAction.CONTINUE,
                success=False,
                strategy_name=self.name,
                confidence=0.0,
                reason=f"No strategy for {context.failure_type}",
            )

        # Sort by priority
        applicable.sort(key=lambda s: s.calculate_priority(context), reverse=True)

        # Use highest priority strategy
        best_strategy = applicable[0]
        logger.debug(
            f"Selected recovery strategy: {best_strategy.name} "
            f"(priority={best_strategy.calculate_priority(context):.2f})"
        )

        result = await best_strategy.recover(context)
        result.strategy_name = best_strategy.name
        return result

    def record_outcome(
        self,
        context: RecoveryContext,
        result: RecoveryResult,
        success: bool,
    ) -> None:
        """Record outcome to the strategy that was used."""
        super().record_outcome(context, result, success)

        # Find and update the strategy that was used
        for strategy in self._strategies:
            if strategy.name == result.strategy_name:
                strategy.record_outcome(context, result, success)
                break

    def add_strategy(self, strategy: BaseRecoveryStrategy) -> None:
        """Add a new recovery strategy."""
        self._strategies.append(strategy)

    def get_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all strategies."""
        return {
            s.name: {
                "attempts": s._metrics.attempts,
                "success_rate": s._metrics.success_rate,
                "avg_quality_improvement": s._metrics.avg_quality_improvement,
                "handles": [ft.name for ft in s.handles_failure_types],
            }
            for s in self._strategies
        }
