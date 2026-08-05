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

"""Iteration coordinator for streaming chat loop.

This module provides the IterationCoordinator class which handles loop control
decisions for the streaming chat implementation. It encapsulates:
- Budget limit checking
- Loop detection coordination
- Completion criteria evaluation
- Pre-iteration and post-iteration checks

Design Pattern: Strategy + Coordinator
======================================
The coordinator delegates to specialized handlers (StreamingChatHandler) for
specific checks while providing the orchestration logic for loop control.
This enables clean separation between decision-making (coordinator) and
action execution (handler).

Usage:
    coordinator = IterationCoordinator(
        handler=streaming_handler,
        loop_detector=unified_tracker,
        settings=settings,
    )

    while coordinator.should_continue(ctx, last_result):
        # Pre-iteration checks
        pre_check = coordinator.pre_iteration_check(ctx)
        if pre_check is not None:
            yield from pre_check.chunks
            if pre_check.should_break:
                break

        # ... perform iteration ...

        # Post-iteration checks
        post_check = coordinator.post_iteration_check(ctx, result)
        if post_check is not None:
            yield from post_check.chunks
            if post_check.should_break:
                break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    create_break_result,
    create_force_completion_result,
)
from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class LoopDetectorProtocol(Protocol):
    """Protocol for loop detection."""

    def check_stop(self, tool_calls: List[Dict[str, Any]], content: str) -> tuple[bool, str, str]:
        """Check if loop should stop.

        Returns:
            Tuple of (should_stop, stop_reason_value, stop_hint)
        """
        ...

    def get_warning(self) -> Optional[str]:
        """Get warning message if approaching loop limit."""
        ...


@dataclass
class CoordinatorConfig:
    """Configuration for IterationCoordinator.

    Attributes:
        session_idle_timeout: Maximum idle time in seconds.
        budget_warning_threshold: Tool calls before budget warning.
        budget_warning_pct: Fraction of tool budget used before warning.
        budget_warning_remaining: Warn when this many or fewer tool calls remain.
        consecutive_blocked_limit: Blocked attempts before force.
        total_blocked_limit: Total blocked attempts before force.
        max_empty_responses: Empty responses before recovery.
    """

    session_idle_timeout: float = 180.0
    budget_warning_threshold: int = 250
    budget_warning_pct: float = 0.8
    budget_warning_remaining: int = 5
    consecutive_blocked_limit: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
    total_blocked_limit: int = DEFAULT_BLOCKED_TOTAL_THRESHOLD
    max_empty_responses: int = 3


class IterationCoordinator:
    """Coordinator for streaming chat loop control.

    This class encapsulates the decision-making logic for when to continue
    or stop the streaming loop. It coordinates between:
    - Budget tracking and limits
    - Loop detection
    - Time limits
    - Completion criteria

    The coordinator uses the Strategy pattern, delegating specific checks
    to the StreamingChatHandler while providing orchestration logic.
    """

    def __init__(
        self,
        handler: "StreamingChatHandler",
        loop_detector: Optional[LoopDetectorProtocol] = None,
        settings: Optional["Settings"] = None,
        config: Optional[CoordinatorConfig] = None,
    ):
        """Initialize the coordinator.

        Args:
            handler: The streaming chat handler for checks and actions.
            loop_detector: Optional loop detector (unified tracker).
            settings: Optional application settings.
            config: Optional coordinator configuration.
        """
        self._handler = handler
        self._loop_detector = loop_detector
        self._settings = settings
        if config is not None:
            self._config = config
        elif settings is not None:
            self._config = CoordinatorConfig(
                budget_warning_threshold=getattr(
                    settings, "tool_call_budget_warning_threshold", 250
                ),
                budget_warning_pct=getattr(settings, "tool_call_budget_warning_pct", 0.8),
                budget_warning_remaining=getattr(settings, "tool_call_budget_warning_remaining", 5),
            )
        else:
            self._config = CoordinatorConfig()

    @property
    def config(self) -> CoordinatorConfig:
        """Get the coordinator configuration."""
        return self._config

    def _sync_blocked_thresholds(self, ctx: StreamingChatContext) -> None:
        """Ensure context-based blocked checks use coordinator-configured limits."""
        ctx.max_blocked_before_force = self._config.consecutive_blocked_limit

    def should_continue(
        self,
        ctx: StreamingChatContext,
        last_result: Optional[IterationResult] = None,
    ) -> bool:
        """Determine if the streaming loop should continue.

        This is the primary entry point for loop control decisions.
        It checks all termination conditions in priority order.

        Args:
            ctx: The streaming context.
            last_result: The result from the previous iteration.

        Returns:
            True if loop should continue, False otherwise.
        """
        self._sync_blocked_thresholds(ctx)

        # Check if last result says to break
        if last_result is not None and last_result.should_break:
            return False

        # Check iteration limit
        if ctx.is_over_iteration_limit():
            logger.info(f"Stopping: iteration limit ({ctx.max_total_iterations}) reached")
            return False

        # Check force completion without pending tool calls
        if ctx.should_force_completion():
            has_pending_tools = last_result and last_result.has_tool_calls
            if not has_pending_tools:
                logger.info("Stopping: force completion triggered (no pending tools)")
                return False

        # Check budget exhaustion
        if ctx.is_budget_exhausted():
            logger.info(f"Stopping: tool budget ({ctx.tool_budget}) exhausted")
            return False

        return True

    def pre_iteration_check(
        self,
        ctx: StreamingChatContext,
    ) -> Optional[IterationResult]:
        """Perform pre-iteration checks.

        These checks run before each iteration to catch conditions
        that should stop the loop early.

        Args:
            ctx: The streaming context.

        Returns:
            IterationResult if loop should stop/yield, None to continue.
        """
        self._sync_blocked_thresholds(ctx)

        # Increment iteration counter
        ctx.increment_iteration()

        # Check time limit
        time_result = self._handler.check_time_limit(ctx)
        if time_result is not None:
            return time_result

        # Check iteration limit
        iter_result = self._handler.check_iteration_limit(ctx)
        if iter_result is not None:
            return iter_result

        # Check force completion
        force_result = self._handler.check_force_completion(ctx)
        if force_result is not None:
            return force_result

        return None

    def post_iteration_check(
        self,
        ctx: StreamingChatContext,
        result: IterationResult,
    ) -> Optional[IterationResult]:
        """Perform post-iteration checks.

        These checks run after each iteration to evaluate the outcome
        and decide on next steps.

        Args:
            ctx: The streaming context.
            result: The result from the current iteration.

        Returns:
            IterationResult if additional action needed, None otherwise.
        """
        # Check budget warning
        budget_result = self._handler.check_tool_budget(
            ctx,
            self._config.budget_warning_threshold,
            self._config.budget_warning_pct,
            self._config.budget_warning_remaining,
        )
        if budget_result is not None:
            return budget_result

        # Check progress and force completion if stuck
        if self._handler.check_progress_and_force(ctx):
            return create_force_completion_result(
                f"Progress check failed: {ctx.tool_calls_used} tool calls "
                f"but only {len(ctx.unique_resources)} unique resources"
            )

        return None

    def check_loop_detection(
        self,
        ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        content: str,
    ) -> tuple[bool, Optional[IterationResult]]:
        """Check loop detection and return stop decision.

        Args:
            ctx: The streaming context.
            tool_calls: Current tool calls.
            content: Current response content.

        Returns:
            Tuple of (should_stop, optional_result).
        """
        if self._loop_detector is None:
            return False, None

        should_stop, stop_reason, stop_hint = self._loop_detector.check_stop(tool_calls, content)

        if should_stop:
            logger.warning(f"Loop detected: {stop_reason} - {stop_hint}")
            ctx.force_completion = True
            result = self._handler.handle_force_completion(ctx, stop_reason, stop_hint)
            if result is not None:
                return True, result
            return True, create_force_completion_result(stop_hint)

        # Check for warning
        warning = self._loop_detector.get_warning()
        if warning:
            chunk = self._handler.handle_loop_warning(ctx, warning)
            if chunk:
                result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
                result.add_chunk(chunk)
                return False, result

        return False, None

    def check_blocked_threshold(
        self,
        ctx: StreamingChatContext,
        all_blocked: bool,
    ) -> Optional[IterationResult]:
        """Check if blocked tool attempts exceed thresholds.

        Args:
            ctx: The streaming context.
            all_blocked: Whether all tool calls were blocked.

        Returns:
            IterationResult if force completion triggered.
        """
        return self._handler.check_blocked_threshold(
            ctx,
            all_blocked,
            consecutive_limit=self._config.consecutive_blocked_limit,
            total_limit=self._config.total_blocked_limit,
        )

    def handle_empty_response(
        self,
        ctx: StreamingChatContext,
        has_tool_calls: bool,
    ) -> Optional[IterationResult]:
        """Handle an empty response from the model.

        Args:
            ctx: The streaming context.
            has_tool_calls: Whether there are pending tool calls.

        Returns:
            IterationResult if action needed.
        """
        # Check for natural completion
        natural = self._handler.check_natural_completion(
            ctx, has_tool_calls, ctx.total_accumulated_chars
        )
        if natural is not None:
            return natural

        # Track empty response
        return self._handler.handle_empty_response(ctx)

    def check_budget_for_tools(
        self,
        ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[StreamChunk]]:
        """Check budget and potentially truncate tool calls.

        Args:
            ctx: The streaming context.
            tool_calls: Requested tool calls.

        Returns:
            Tuple of (allowed_tool_calls, warning_chunks).
        """
        chunks: List[StreamChunk] = []

        if ctx.is_budget_exhausted():
            chunks.extend(self._handler.get_budget_exhausted_chunks(ctx))
            return [], chunks

        # Truncate to remaining budget
        truncated = self._handler.truncate_tool_calls(tool_calls, ctx)
        if len(truncated) < len(tool_calls):
            logger.info(
                f"Truncated tool calls from {len(tool_calls)} to {len(truncated)} "
                f"(budget: {ctx.get_remaining_budget()})"
            )

        return truncated, chunks

    def filter_blocked_tools(
        self,
        ctx: StreamingChatContext,
        tool_calls: List[Dict[str, Any]],
        block_checker: Callable[[str, Dict[str, Any]], Optional[str]],
    ) -> tuple[List[Dict[str, Any]], List[StreamChunk], bool]:
        """Filter out blocked tool calls.

        Args:
            ctx: The streaming context.
            tool_calls: Tool calls to filter.
            block_checker: Function to check if tool is blocked.

        Returns:
            Tuple of (allowed_calls, blocked_chunks, all_blocked).
        """
        filtered, blocked_chunks, blocked_count = self._handler.filter_blocked_tool_calls(
            ctx, tool_calls, block_checker
        )
        all_blocked = blocked_count > 0 and len(filtered) == 0
        return filtered, blocked_chunks, all_blocked

    def handle_continuation_decision(
        self,
        ctx: StreamingChatContext,
        has_tool_calls: bool,
        has_content: bool,
        content_length: int,
    ) -> Optional[IterationResult]:
        """Make continuation decision based on response characteristics.

        This method handles the complex logic of deciding whether to continue
        the loop based on the model's response.

        Args:
            ctx: The streaming context.
            has_tool_calls: Whether response has tool calls.
            has_content: Whether response has content.
            content_length: Length of response content.

        Returns:
            IterationResult if loop should stop, None to continue.
        """
        # With tool calls, always continue to execute them
        if has_tool_calls:
            return None

        # Force completion with substantial content
        if ctx.force_completion and has_content:
            logger.info("Force completion: yielding final content")
            return create_break_result()

        # Natural completion: substantial content without tools
        if has_content and ctx.has_substantial_content():
            # Let the handler decide
            result = self._handler.check_natural_completion(ctx, has_tool_calls, content_length)
            if result is not None:
                return result

        # Empty response handling
        if not has_content and not has_tool_calls:
            return self.handle_empty_response(ctx, has_tool_calls)

        return None


def create_coordinator(
    handler: "StreamingChatHandler",
    loop_detector: Optional[LoopDetectorProtocol] = None,
    settings: Optional["Settings"] = None,
    config: Optional[CoordinatorConfig] = None,
) -> IterationCoordinator:
    """Factory function to create an IterationCoordinator.

    Args:
        handler: The streaming chat handler.
        loop_detector: Optional loop detector.
        settings: Optional application settings.
        config: Optional coordinator configuration.

    Returns:
        Configured IterationCoordinator instance.
    """
    return IterationCoordinator(
        handler=handler,
        loop_detector=loop_detector,
        settings=settings,
        config=config,
    )
