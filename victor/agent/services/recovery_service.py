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

"""Recovery service implementation.

Extracts error recovery from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Error classification and analysis
- Recovery action selection
- Automatic retry with exponential backoff
- Recovery metrics and tracking
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.streaming.context import StreamingChatContext
    from victor.evaluation.htir import ETCLOVGLayer

logger = logging.getLogger(__name__)

from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)
from victor.providers.base import StreamChunk
from dataclasses import dataclass


@dataclass
class StreamingRecoveryContext:
    """Context for streaming recovery decisions.

    This dataclass encapsulates all state needed for streaming recovery coordination.
    Moved from recovery_compat.py to recovery_service.py for canonical service ownership.

    NOTE: This is distinct from victor.agent.recovery.protocols.StreamingRecoveryContext
    which is used by the SOLID recovery system for failure detection/strategy.

    Attributes:
        iteration: Current iteration number
        elapsed_time: Elapsed time since session start (seconds)
        tool_calls_used: Number of tool calls made so far
        tool_budget: Maximum allowed tool calls
        max_iterations: Maximum allowed iterations
        session_start_time: Unix timestamp when session started
        last_quality_score: Quality score from last iteration (0-1)
        streaming_context: StreamingChatContext for current session
        provider_name: Name of current LLM provider
        model: Name of current model
        temperature: Current temperature setting
        unified_task_type: Task type from UnifiedTaskTracker
        is_analysis_task: Whether this is an analysis task
        is_action_task: Whether this is an action task
    """

    iteration: int
    elapsed_time: float
    tool_calls_used: int
    tool_budget: int
    max_iterations: int
    session_start_time: float
    last_quality_score: float
    streaming_context: "StreamingChatContext"
    provider_name: str
    model: str
    temperature: float
    unified_task_type: Any  # TaskType from UnifiedTaskTracker
    is_analysis_task: bool
    is_action_task: bool


class RecoveryStrategy(str, Enum):
    """Error recovery strategy selection."""

    RETRY = "retry"  # Retry the operation immediately
    BACKOFF = "backoff"  # Retry with exponential backoff
    FALLBACK = "fallback"  # Use fallback method
    GIVE_UP = "give_up"  # Abort the operation


def resolve_recovery_layer_attribution_enabled(settings: Any) -> bool:
    """Resolve the ADR-012 prong-2 recovery-attribution flag: env → AgentSettings → default off.

    Mirrors :func:`victor.framework.effect_gate.resolve_effect_gate_enabled` so both opt-in gates
    read consistently. Env override ``VICTOR_RECOVERY_LAYER_ATTRIBUTION`` wins; otherwise
    ``settings.agent.recovery_layer_attribution`` (default ``False``).
    """
    env = os.environ.get("VICTOR_RECOVERY_LAYER_ATTRIBUTION")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(getattr(settings, "agent", None), "recovery_layer_attribution", False))


class RecoveryContextImpl:
    """Implementation of recovery context."""

    def __init__(
        self,
        error: Exception,
        error_type: str,
        attempt_count: int,
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self.error = error
        self.error_type = error_type
        self.attempt_count = attempt_count
        self.state = state
        self.metadata = metadata


class RecoveryService:
    """[CANONICAL] Service for error recovery and resilience.

    The target implementation for recovery operations following the
    state-passed architectural pattern. Supersedes RecoveryController.

    This service follows SOLID principles:
    - SRP: Only handles recovery operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements RecoveryServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        service = RecoveryService()
        error_type = await service.classify_error(error)
        context = RecoveryContextImpl(error, error_type, 1, {}, {})
        success = await service.execute_recovery(context)
    """

    def bind_runtime_components(
        self,
        *,
        recovery_coordinator: Optional[Any] = None,
        recovery_handler: Optional[Any] = None,
        recovery_integration: Optional[Any] = None,
        streaming_handler: Optional[Any] = None,
        context_compactor: Optional[Any] = None,
        unified_tracker: Optional[Any] = None,
        settings: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        presentation: Optional[Any] = None,
    ) -> None:
        """Bind live runtime collaborators after bootstrap."""
        if recovery_coordinator is not None:
            self._recovery_coordinator = recovery_coordinator
        if recovery_handler is not None:
            self._recovery_handler = recovery_handler
        if recovery_integration is not None:
            self._recovery_integration = recovery_integration
        if streaming_handler is not None:
            self._streaming_handler = streaming_handler
        if context_compactor is not None:
            self._context_compactor = context_compactor
        if unified_tracker is not None:
            self._unified_tracker = unified_tracker
        if settings is not None:
            self._settings = settings
            self._layer_attribution_enabled = resolve_recovery_layer_attribution_enabled(settings)
        if event_bus is not None:
            self._event_bus = event_bus
        if presentation is not None:
            self._presentation = presentation

    def has_native_streaming_runtime(self) -> bool:
        """Whether the service has enough runtime state to own streaming recovery."""
        return (
            self._streaming_handler is not None
            and self._settings is not None
            and self._unified_tracker is not None
            and self._recovery_integration is not None
        )

    def _icon(self, name: str) -> str:
        if self._presentation is None:
            return ""
        try:
            return self._presentation.icon(name, with_color=False)
        except Exception:
            return ""

    def _emit_async_event(
        self,
        *,
        topic: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
    ) -> None:
        """Emit an observability event without blocking the recovery path."""
        if self._event_bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            kwargs = {"topic": topic, "data": data}
            if source is not None:
                kwargs["source"] = source
            loop.create_task(self._event_bus.emit(**kwargs))
        except Exception as exc:
            self._logger.debug("Failed to emit recovery event %s: %s", topic, exc)

    @staticmethod
    def _task_type_value(task_type: Any) -> Any:
        return getattr(task_type, "value", task_type)

    def classify_failure_layer(self, context: RecoveryContextImpl) -> Optional["ETCLOVGLayer"]:
        """Attribute this failure to an ETCLOVG harness layer (ADR-012 / EVR-5, prong 2).

        Flag-guarded (default off — returns ``None`` immediately). When enabled, reads the failing
        tool name from the recovery context (``metadata``/``state``) and reuses HTIR's layer
        inference, so recovery failures become attributable to a harness *layer* (Execution /
        Tooling / Context-Memory / …) instead of only an exception type. Records a per-layer metric
        and stamps ``context.metadata['failure_layer']`` for downstream handlers; never changes the
        selected action (that remains a separate, future opt-in).

        Returns the attributed :class:`ETCLOVGLayer`, or ``None`` when disabled or no tool is in
        flight (e.g. a provider-level error).
        """
        if not self._layer_attribution_enabled:
            return None
        from victor.evaluation.htir import attribute_failure_layer

        raw = (
            context.metadata.get("failing_tool")
            or context.metadata.get("tool_name")
            or context.state.get("failing_tool")
            or context.state.get("tool_name")
        )
        layer = attribute_failure_layer(raw if isinstance(raw, str) else None)
        if layer is None:
            return None
        by_layer = self._metrics.setdefault("by_layer", {})
        by_layer[layer.value] = by_layer.get(layer.value, 0) + 1
        context.metadata["failure_layer"] = layer.value
        self._logger.debug(
            "Recovery failure attributed to ETCLOVG layer '%s' (tool=%s)", layer.value, raw
        )
        return layer

    async def classify_error(self, error: Exception) -> str:
        """Classify an error for recovery strategy selection.

        Args:
            error: The error to classify

        Returns:
            Error type string
        """
        error_type = type(error).__name__

        # Map common errors to types
        error_mapping = {
            "TimeoutError": "timeout",
            "RateLimitError": "rate_limit",
            "AuthError": "auth",
            "ConnectionError": "connection",
        }

        return error_mapping.get(error_type, "unknown")

    async def select_recovery_action(
        self,
        context: RecoveryContextImpl,
    ) -> str:
        """Select appropriate recovery action for the context.

        Args:
            context: Recovery context with error and state info

        Returns:
            Recovery action name
        """
        error_type = context.error_type

        # Select action based on error type
        action_mapping = {
            "timeout": "retry",
            "rate_limit": "backoff",
            "auth": "fail",
            "connection": "retry",
            "unknown": "retry",
        }

        return action_mapping.get(error_type, "fail")

    async def execute_recovery(
        self,
        context: RecoveryContextImpl,
    ) -> bool:
        """Execute appropriate recovery action for the context.

        Args:
            context: Recovery context with error and state info

        Returns:
            True if recovery succeeded, False otherwise
        """
        self._metrics["total_attempts"] += 1

        # ADR-012 prong 2: attribute the failure to a harness layer (flag-guarded, no-op by default).
        self.classify_failure_layer(context)

        action = await self.select_recovery_action(context)

        self._logger.info(f"Executing recovery action: {action} for error: {context.error_type}")

        # Track by error type
        error_type = context.error_type
        if error_type not in self._metrics["by_error_type"]:
            self._metrics["by_error_type"][error_type] = {
                "attempts": 0,
                "successes": 0,
            }
        self._metrics["by_error_type"][error_type]["attempts"] += 1

        # Execute action
        success = False

        if action == "retry":
            success = await self._retry_action(context)
        elif action == "backoff":
            success = await self._backoff_action(context)
        elif action == "fail":
            success = False
        else:
            success = False

        # Track results
        if success:
            self._metrics["successful_recoveries"] += 1
            self._metrics["by_error_type"][error_type]["successes"] += 1
        else:
            self._metrics["failed_recoveries"] += 1

        return success

    def can_retry(
        self,
        error: Exception,
        attempt_count: int,
    ) -> bool:
        """Check if an operation can be retried.

        Args:
            error: The error that occurred
            attempt_count: Number of attempts made so far

        Returns:
            True if operation can be retried, False otherwise
        """
        # Don't retry auth errors
        if isinstance(error, (PermissionError, AuthError)):
            return False

        # Check attempt limit
        return attempt_count < self._max_retry_attempts

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery statistics and metrics.

        Returns:
            Dictionary with recovery metrics
        """
        total = self._metrics["total_attempts"]
        successful = self._metrics["successful_recoveries"]

        return {
            **self._metrics,
            "success_rate": successful / total if total > 0 else 0.0,
        }

    def reset_metrics(self) -> None:
        """Reset recovery metrics."""
        self._metrics = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "by_error_type": {},
            "by_layer": {},
        }

    def is_healthy(self) -> bool:
        """Check if the recovery service is healthy.

        Returns:
            True if the service is healthy
        """
        return self._max_retry_attempts > 0

    # ==========================================================================
    # Stuck Loop Detection
    # ==========================================================================

    def detect_stuck_loop(
        self,
        content: str,
        recent_responses: Optional[List[str]] = None,
    ) -> bool:
        """Detect stuck planning loop pattern in responses.

        Identifies when the agent is stuck in a planning phase by looking
        for repeated planning phrases like "I'm going to read", "let me check",
        etc. across multiple responses.

        Args:
            content: Current response content
            recent_responses: Optional list of recent responses for context

        Returns:
            True if stuck loop detected, False otherwise

        Example:
            if service.detect_stuck_loop(response, recent_responses):
                # Agent is stuck, trigger recovery
        """
        if not content:
            return False

        import re

        # Planning patterns that indicate stuck behavior
        planning_patterns = [
            r"\bi[''`]?m\s+going\s+to\s+(read|examine|check|call|use)\b",
            r"\bi\s+will\s+now\s+(read|examine|check|call|use)\b",
            r"\blet\s+me\s+(read|check|examine|look\s+at)\b",
        ]

        # Count planning phrases in current content
        planning_count = 0
        for pattern in planning_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                planning_count += 1

        # Check recent responses if available
        if recent_responses:
            for response in recent_responses[-3:]:  # Check last 3 responses
                for pattern in planning_patterns:
                    if re.search(pattern, response, re.IGNORECASE):
                        planning_count += 1

        # Stuck if 2+ patterns in current or 3+ across recent responses
        is_stuck = planning_count >= 3

        if is_stuck:
            self._logger.warning(f"Stuck loop detected: {planning_count} planning patterns found")

        return is_stuck

    def detect_repeated_response(
        self,
        content: str,
        recent_responses: List[str],
    ) -> bool:
        """Detect repeated response pattern.

        Checks if the current content is very similar to recent responses,
        indicating the agent is repeating itself without making progress.

        Args:
            content: Current response content
            recent_responses: List of recent responses to compare against

        Returns:
            True if repeated response detected, False otherwise

        Example:
            if service.detect_repeated_response(response, recent):
                # Agent is repeating itself
        """
        if not content or not recent_responses:
            return False

        content_lower = content.lower().strip()

        # Check against last 3 responses
        for response in recent_responses[-3:]:
            response_lower = response.lower().strip()

            # Only compare if both are substantial (>50 chars)
            if len(content_lower) > 50 and len(response_lower) > 50:
                min_len = min(len(content_lower), len(response_lower))
                prefix_len = min(200, min_len)

                # Check if first 200 chars (or full length) are identical
                if content_lower[:prefix_len] == response_lower[:prefix_len]:
                    self._logger.warning("Repeated response detected: identical prefix found")
                    return True

        return False

    # ==========================================================================
    # Retry Logic and Backoff Strategies
    # ==========================================================================

    async def retry_with_exponential_backoff(
        self,
        func: Callable[..., Any],
        *args: Any,
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        jitter: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Retry a function with exponential backoff.

        Executes the given function with retry logic using exponential
        backoff. Optionally adds jitter to prevent thundering herd.

        Args:
            func: Function to retry
            *args: Positional arguments for the function
            max_attempts: Maximum retry attempts (uses service default if None)
            base_delay: Base delay in seconds (uses service default if None)
            max_delay: Maximum delay in seconds (uses service default if None)
            jitter: Whether to add random jitter to delays
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            Exception: If all retry attempts fail

        Example:
            result = await service.retry_with_exponential_backoff(
                api_call,
                "param1",
                "param2",
                max_attempts=3,
                jitter=True
            )
        """
        max_attempts = max_attempts or self._max_retry_attempts
        base_delay = base_delay or self._base_retry_delay
        max_delay = max_delay or self._max_retry_delay

        last_exception = None

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff
                    delay = base_delay * (2**attempt)
                    delay = min(delay, max_delay)

                    # Add jitter if enabled
                    if jitter:
                        import random

                        delay = delay * (0.5 + random.random() * 0.5)

                    self._logger.info(
                        f"Retry attempt {attempt + 1}/{max_attempts} "
                        f"after {delay:.1f}s delay: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    self._logger.error(f"All {max_attempts} retry attempts failed: {e}")

        raise last_exception

    async def retry_with_jitter(
        self,
        func: Callable[..., Any],
        *args: Any,
        jitter_range: tuple[float, float] = (0.5, 1.5),
        **kwargs: Any,
    ) -> Any:
        """Retry a function with randomized jitter delay.

        Adds random jitter to retry delays to prevent synchronized
        retry storms (thundering herd problem).

        Args:
            func: Function to retry
            *args: Positional arguments for the function
            jitter_range: (min_multiplier, max_multiplier) for jitter
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Example:
            result = await service.retry_with_jitter(
                api_call,
                jitter_range=(0.5, 1.5)
            )
        """
        import random

        min_mult, max_mult = jitter_range
        last_exception = None

        for attempt in range(self._max_retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self._max_retry_attempts - 1:
                    # Calculate base delay
                    base_delay = self._base_retry_delay * (2**attempt)
                    base_delay = min(base_delay, self._max_retry_delay)

                    # Apply jitter
                    delay = base_delay * random.uniform(min_mult, max_mult)

                    self._logger.info(
                        f"Retry with jitter: attempt {attempt + 1}, " f"delay {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        raise last_exception

    def calculate_backoff_delay(
        self,
        attempt: int,
        strategy: str = "exponential",
    ) -> float:
        """Calculate retry delay based on strategy.

        Supports multiple backoff strategies:
        - exponential: delay = base * 2^attempt
        - linear: delay = base * (attempt + 1)
        - fixed: delay = base
        - fibonacci: delay = base * fib(attempt)

        Args:
            attempt: Current attempt number (0-indexed)
            strategy: Backoff strategy to use

        Returns:
            Delay in seconds

        Example:
            delay = service.calculate_backoff_delay(2, strategy="exponential")
        """
        if strategy == "exponential":
            delay = self._base_retry_delay * (2**attempt)
        elif strategy == "linear":
            delay = self._base_retry_delay * (attempt + 1)
        elif strategy == "fixed":
            delay = self._base_retry_delay
        elif strategy == "fibonacci":
            # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, ...
            def fib(n):
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(n):
                    a, b = b, a + b
                return a

            delay = self._base_retry_delay * fib(attempt + 1)
        else:
            # Default to exponential
            delay = self._base_retry_delay * (2**attempt)

        return min(delay, self._max_retry_delay)

    # ==========================================================================
    # Circuit Breaker Pattern
    # ==========================================================================

    def should_attempt_recovery(
        self,
        error_type: str,
        consecutive_failures: int = 0,
    ) -> bool:
        """Determine if recovery should be attempted based on error type.

        Uses circuit breaker pattern logic to prevent repeated recovery
        attempts for persistent errors.

        Args:
            error_type: Type of error (from classify_error)
            consecutive_failures: Number of consecutive failures

        Returns:
            True if recovery should be attempted

        Example:
            if service.should_attempt_recovery(error_type, failures):
                await service.execute_recovery(context)
        """
        # Don't attempt recovery for certain error types
        no_recovery_types = {
            "auth",  # Authentication errors
            "validation",  # Validation errors
        }

        if error_type in no_recovery_types:
            return False

        # Don't attempt if too many consecutive failures
        max_consecutive_failures = 5
        if consecutive_failures >= max_consecutive_failures:
            self._logger.warning(
                f"Circuit breaker open: {consecutive_failures} " f"consecutive failures"
            )
            return False

        return True

    async def handle_recovery_with_integration(
        self,
        ctx: Any,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]] = None,
        message_adder: Any = None,
    ) -> Any:
        """Handle streaming recovery through the canonical runtime service."""
        if self._recovery_integration is not None:
            from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

            if not self._recovery_integration.enabled:
                return OrchestratorRecoveryAction(action="continue", reason="Recovery disabled")

            context_utilization = None
            if self._context_compactor is not None:
                try:
                    stats = self._context_compactor.get_statistics()
                    context_utilization = stats.get("current_utilization")
                except Exception as exc:
                    self._logger.debug("Failed to get context utilization for recovery: %s", exc)

            recovery_action = await self._recovery_integration.handle_response(
                content=full_content,
                tool_calls=tool_calls,
                mentioned_tools=mentioned_tools,
                provider_name=ctx.provider_name,
                model_name=ctx.model,
                tool_calls_made=ctx.tool_calls_used,
                tool_budget=ctx.tool_budget,
                iteration_count=ctx.streaming_context.total_iterations,
                max_iterations=ctx.streaming_context.max_total_iterations,
                current_temperature=ctx.temperature,
                quality_score=ctx.last_quality_score,
                task_type=self._task_type_value(ctx.unified_task_type),
                is_analysis_task=ctx.is_analysis_task,
                is_action_task=ctx.is_action_task,
                context_utilization=context_utilization,
            )

            if recovery_action.action != "continue":
                self._logger.info(
                    "Recovery integration: action=%s, reason=%s, failure_type=%s, strategy=%s",
                    recovery_action.action,
                    recovery_action.reason,
                    recovery_action.failure_type,
                    recovery_action.strategy_name,
                )
            return recovery_action

        if self._recovery_coordinator is not None:
            return await self._recovery_coordinator.handle_recovery_with_integration(
                ctx,
                full_content,
                tool_calls,
                mentioned_tools,
                message_adder=message_adder,
            )

        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        return OrchestratorRecoveryAction(
            action="continue",
            reason="Recovery coordinator unavailable",
        )

    def apply_recovery_action(
        self,
        recovery_action: Any,
        ctx: Any,
        message_adder: Any = None,
    ) -> Any:
        """Apply a recovery action through the canonical runtime service."""
        if recovery_action.action == "continue":
            return None

        self._emit_async_event(
            topic=f"state.recovery.action_{recovery_action.action}",
            data={
                "action": recovery_action.action,
                "reason": recovery_action.reason,
                "category": "state",
            },
            source="RecoveryService",
        )

        if recovery_action.action == "retry":
            if recovery_action.message and message_adder:
                message_adder("user", recovery_action.message)
            if recovery_action.new_temperature is not None:
                self._logger.debug(
                    "Recovery: adjusting temperature from %s to %s",
                    ctx.temperature,
                    recovery_action.new_temperature,
                )
            return None

        if recovery_action.action == "force_summary":
            ctx.streaming_context.force_completion = True
            if recovery_action.message and message_adder:
                message_adder("user", recovery_action.message)
            elif message_adder:
                message_adder(
                    "user",
                    "Please provide a brief summary of what you've accomplished and any findings.",
                )
            return None

        if recovery_action.action == "abort":
            self._emit_async_event(
                topic="error.raised",
                data={
                    "error_type": "RuntimeError",
                    "error_message": f"Session aborted: {recovery_action.reason}",
                    "category": "error",
                    "recoverable": False,
                    "context": {
                        "failure_type": recovery_action.failure_type,
                        "iteration": ctx.iteration,
                    },
                },
            )
            return StreamChunk(
                content=f"\n[recovery] Session aborted: {recovery_action.reason}\n",
                is_final=True,
            )

        return None

    def check_natural_completion(
        self,
        ctx: Any,
        has_tool_calls: bool,
        content_length: int,
    ) -> Any:
        """Check whether the current streaming turn should terminate naturally."""
        if self._streaming_handler is not None:
            result = self._streaming_handler.check_natural_completion(
                ctx.streaming_context,
                has_tool_calls,
                content_length,
            )
            if result:
                return StreamChunk(content="", is_final=True)
            return None

        if self._recovery_coordinator is not None:
            return self._recovery_coordinator.check_natural_completion(
                ctx,
                has_tool_calls,
                content_length,
            )
        return None

    def handle_empty_response(
        self,
        ctx: Any,
    ) -> Any:
        """Handle an empty model response during streaming."""
        if self._streaming_handler is not None:
            result = self._streaming_handler.handle_empty_response(ctx.streaming_context)
            if result and result.chunks:
                self._emit_async_event(
                    topic="error.raised",
                    data={
                        "error_type": "RuntimeError",
                        "error_message": "Empty model response",
                        "category": "error",
                        "recoverable": True,
                        "context": {
                            "iteration": ctx.iteration,
                            "provider": ctx.provider_name,
                            "model": ctx.model,
                            "force_completion": ctx.streaming_context.force_completion,
                        },
                    },
                )
                return result.chunks[0], ctx.streaming_context.force_completion
            return None, False

        if self._recovery_coordinator is not None:
            return self._recovery_coordinator.handle_empty_response(ctx)
        return None, False

    def get_recovery_fallback_message(
        self,
        ctx: Any,
    ) -> str:
        """Get the fallback recovery message for streaming."""
        if self._recovery_coordinator is not None:
            return self._recovery_coordinator.get_recovery_fallback_message(ctx)
        return (
            "I apologize, but I'm having difficulty completing this task. "
            "Here's a summary of what I've accomplished so far..."
        )

    def check_tool_budget(
        self,
        ctx: Any,
        warning_threshold: int = 250,
        warning_pct: float = 0.8,
        warning_remaining: int = 5,
    ) -> Any:
        """Check whether the streaming tool budget is approaching exhaustion."""
        if self._streaming_handler is not None:
            result = self._streaming_handler.check_tool_budget(
                ctx.streaming_context,
                warning_threshold,
                warning_pct,
                warning_remaining,
            )
            if result and result.chunks:
                return result.chunks[0]
            return None

        if self._recovery_coordinator is not None:
            try:
                return self._recovery_coordinator.check_tool_budget(
                    ctx,
                    warning_threshold,
                    warning_pct,
                    warning_remaining,
                )
            except TypeError:
                return self._recovery_coordinator.check_tool_budget(ctx, warning_threshold)
        return None

    def truncate_tool_calls(
        self,
        ctx: Any,
        tool_calls: List[Dict[str, Any]],
        max_calls: int,
    ) -> Any:
        """Truncate tool calls to the allowed budget."""
        if len(tool_calls) <= max_calls:
            return tool_calls, False
        self._logger.info(
            "Truncating %d tool calls to budget limit of %d",
            len(tool_calls),
            max_calls,
        )
        return tool_calls[:max_calls], True

    def filter_blocked_tool_calls(
        self,
        ctx: Any,
        tool_calls: List[Dict[str, Any]],
    ) -> Any:
        """Filter tool calls that are blocked by runtime safeguards."""
        if self._streaming_handler is not None and self._unified_tracker is not None:
            return self._streaming_handler.filter_blocked_tool_calls(
                ctx.streaming_context,
                tool_calls,
                self._unified_tracker.is_blocked_after_warning,
            )

        if self._recovery_coordinator is not None:
            return self._recovery_coordinator.filter_blocked_tool_calls(ctx, tool_calls)
        return tool_calls, [], 0

    def check_blocked_threshold(
        self,
        ctx: Any,
        all_blocked: bool,
    ) -> Any:
        """Check whether blocked-tool thresholds require recovery action."""
        if self._streaming_handler is not None and self._settings is not None:
            consecutive_limit = getattr(
                self._settings,
                "recovery_blocked_consecutive_threshold",
                DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
            )
            total_limit = getattr(
                self._settings,
                "recovery_blocked_total_threshold",
                DEFAULT_BLOCKED_TOTAL_THRESHOLD,
            )
            result = self._streaming_handler.check_blocked_threshold(
                ctx.streaming_context,
                all_blocked,
                consecutive_limit,
                total_limit,
            )
            if result:
                warning_icon = self._icon("warning")
                chunk = (
                    result.chunks[0]
                    if result.chunks
                    else StreamChunk(
                        content=f"\n[loop] {warning_icon} Multiple blocked attempts - forcing completion\n"
                    )
                )
                return chunk, result.clear_tool_calls
            return None

        if self._recovery_coordinator is not None:
            return self._recovery_coordinator.check_blocked_threshold(ctx, all_blocked)
        return None

    def check_force_action(
        self,
        ctx: Any,
    ) -> Any:
        """Check whether recovery should force a follow-up action."""
        return False, None

    def get_circuit_state(
        self,
        failure_count: int,
        last_failure_time: Optional[float] = None,
        timeout_seconds: float = 60.0,
    ) -> Dict[str, Any]:
        """Get circuit breaker state information.

        Returns the current state of the circuit breaker based on
        failure count and time since last failure.

        Args:
            failure_count: Number of failures
            last_failure_time: Unix timestamp of last failure
            timeout_seconds: Timeout for circuit to reset

        Returns:
            Dictionary with circuit state information

        Example:
            state = service.get_circuit_state(failures, last_failure)
            if state["is_open"]:
                # Circuit is open, don't attempt recovery
        """
        import time

        is_open = failure_count >= 5
        should_reset = False

        if is_open and last_failure_time:
            time_since_failure = time.time() - last_failure_time
            should_reset = time_since_failure > timeout_seconds

            if should_reset:
                # Circuit would reset after timeout
                is_open = False

        return {
            "is_open": is_open,
            "is_half_open": not is_open and failure_count > 0,
            "failure_count": failure_count,
            "should_reset": should_reset,
            "can_attempt": not is_open,
        }

    # ==========================================================================
    # Recovery Strategy Selection
    # ==========================================================================

    def select_recovery_strategy(
        self,
        error_type: str,
        attempt_count: int,
        context: Optional[RecoveryContextImpl] = None,
    ) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on error and context.

        Analyzes the error type and attempt count to recommend the
        best recovery strategy.

        Args:
            error_type: Type of error
            attempt_count: Current retry attempt number
            context: Optional recovery context

        Returns:
            Recovery strategy enum value

        Example:
            strategy = service.select_recovery_strategy(error_type, attempts)
            # Returns: RecoveryStrategy.RETRY, FALLBACK, BACKOFF, or GIVE_UP
        """
        # Don't retry certain errors
        if error_type in {"auth", "validation"}:
            return RecoveryStrategy.GIVE_UP

        # Rate limiting: backoff
        if error_type == "rate_limit":
            if attempt_count < 3:
                return RecoveryStrategy.BACKOFF
            else:
                return RecoveryStrategy.FALLBACK

        # Connection errors: retry with backoff
        if error_type == "connection":
            if attempt_count < self._max_retry_attempts:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.FALLBACK

        # Timeout errors: retry with backoff
        if error_type == "timeout":
            if attempt_count < 2:
                return RecoveryStrategy.BACKOFF
            else:
                return RecoveryStrategy.FALLBACK

        # Unknown: retry conservatively
        if attempt_count < 2:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.BACKOFF

    def get_recovery_config(
        self,
        error_type: str,
    ) -> Dict[str, Any]:
        """Get recovery configuration for an error type.

        Returns the recommended recovery configuration including
        max retries, delays, and fallback options.

        Args:
            error_type: Type of error

        Returns:
            Dictionary with recovery configuration

        Example:
            config = service.get_recovery_config("connection")
            # {"max_retries": 3, "strategy": "exponential", ...}
        """
        configs = {
            "connection": {
                "max_retries": 3,
                "strategy": "exponential",
                "base_delay": 1.0,
                "use_jitter": True,
                "fallback_enabled": True,
            },
            "timeout": {
                "max_retries": 2,
                "strategy": "linear",
                "base_delay": 2.0,
                "use_jitter": False,
                "fallback_enabled": True,
            },
            "rate_limit": {
                "max_retries": 5,
                "strategy": "exponential",
                "base_delay": 5.0,
                "use_jitter": True,
                "fallback_enabled": False,
            },
            "auth": {
                "max_retries": 0,
                "strategy": "none",
                "base_delay": 0,
                "use_jitter": False,
                "fallback_enabled": False,
            },
        }

        return configs.get(
            error_type,
            {
                "max_retries": self._max_retry_attempts,
                "strategy": "exponential",
                "base_delay": self._base_retry_delay,
                "use_jitter": True,
                "fallback_enabled": True,
            },
        )

    # ==========================================================================
    # Fallback Provider Switching
    # ==========================================================================

    def __init__(
        self,
        max_retry_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
    ):
        """Initialize the recovery service.

        Args:
            max_retry_attempts: Maximum retry attempts
            base_retry_delay: Base delay for exponential backoff
            max_retry_delay: Maximum delay for retries
        """
        self._max_retry_attempts = max_retry_attempts
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._metrics: Dict[str, Any] = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "by_error_type": {},
        }
        # ADR-012 / EVR-5 prong 2: attribute failures to a harness layer (flag-guarded, default
        # off; set from bound settings via resolve_recovery_layer_attribution_enabled).
        self._layer_attribution_enabled = False
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        self._recovery_coordinator: Optional[Any] = None
        self._recovery_handler: Optional[Any] = None
        self._recovery_integration: Optional[Any] = None
        self._streaming_handler: Optional[Any] = None
        self._context_compactor: Optional[Any] = None
        self._unified_tracker: Optional[Any] = None
        self._settings: Optional[Any] = None
        self._event_bus: Optional[Any] = None
        self._presentation: Optional[Any] = None

        # Provider health tracking
        self._provider_health: Dict[str, Dict[str, Any]] = {}
        self._primary_provider: Optional[str] = None
        self._fallback_providers: List[str] = []

        # Model fallback configuration
        # Updated 2025-04-17 with within-provider fallback chains
        # IMPORTANT: Fallbacks are WITHIN provider family only to avoid API incompatibility
        # - Anthropic models → Anthropic models only
        # - OpenAI models → OpenAI models only
        # - Google models → Google models only
        # This ensures API compatibility, consistent response formats, and no cross-provider issues
        self._model_fallbacks: Dict[str, List[str]] = {
            # ================================================================
            # ANTHROPIC MODELS - Within-Provider Fallback Chain
            # ================================================================
            # Claude Opus 4.6 (latest, most intelligent) → Sonnet 4.6 → Haiku 4.5
            "claude-opus-4-6": [
                "claude-sonnet-4-6",  # Same provider, next tier
                "claude-haiku-4-5",  # Same provider, fastest
            ],
            # Claude Sonnet 4.6 → Haiku 4.5
            "claude-sonnet-4-6": [
                "claude-haiku-4-5",  # Same provider only
            ],
            # Claude Haiku 4.5 (fastest, no further fallback within Anthropic)
            "claude-haiku-4-5": [
                # End of Anthropic chain - no fallback
            ],
            # ================================================================
            # OPENAI MODELS - Within-Provider Fallback Chain
            # ================================================================
            # GPT 5.4 (latest, most intelligent) → GPT 5.4-mini
            "gpt-5-4": [
                "gpt-5-4-mini",  # Same provider only
            ],
            # GPT 5.4-mini (fastest, no further fallback within OpenAI)
            "gpt-5-4-mini": [
                # End of OpenAI chain - no fallback
            ],
            # ================================================================
            # GOOGLE MODELS - Within-Provider Fallback Chain
            # ================================================================
            # Gemini 3.1 (latest) → Gemini 2.5 Flash
            "gemini-3-1": [
                "gemini-2-5-flash",  # Same provider only
            ],
            # Gemini 2.5 Flash → Gemini 1.5 Flash
            "gemini-2-5-flash": [
                "gemini-1-5-flash",  # Same provider only
            ],
            # Gemini 1.5 Flash (fastest, no further fallback within Google)
            "gemini-1-5-flash": [
                # End of Google chain - no fallback
            ],
            # ================================================================
            # LEGACY MODEL NAMES - Mapped to Latest Within-Provider Chains
            # ================================================================
            # Legacy Anthropic models
            "claude-3-7-sonnet-20250219": [
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5",
            ],
            "claude-3-5-sonnet-20241022": ["claude-sonnet-4-6", "claude-haiku-4-5"],
            "claude-3-5-haiku-20241022": ["claude-haiku-4-5"],
            "claude-3-opus-20240229": [
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5",
            ],
            "claude-3-sonnet-20240229": ["claude-sonnet-4-6", "claude-haiku-4-5"],
            "claude-3-haiku-20240307": ["claude-haiku-4-5"],
            # Legacy OpenAI models
            "gpt-4o": ["gpt-5-4", "gpt-5-4-mini"],
            "gpt-4o-mini": ["gpt-5-4-mini"],
            "gpt-4-turbo": ["gpt-5-4-mini"],
            # Legacy Google models
            "gemini-2-0-flash-thinking": [
                "gemini-3-1",
                "gemini-2-5-flash",
                "gemini-1-5-flash",
            ],
            "gemini-1-5-pro": ["gemini-3-1", "gemini-2-5-flash"],
            # ================================================================
            # OTHER PROVIDERS - Within-Provider Chains (can be extended)
            # ================================================================
            # Mistral models (example)
            "mistral-large": [
                "mistral-medium",
                "mistral-small",
            ],
            # DeepSeek models (example)
            "deepseek-reasoner": [
                "deepseek-chat",
            ],
            # Add more providers as needed
        }

        # Metrics
        self._metrics = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "provider_switches": 0,
            "model_fallbacks": 0,
            "by_error_type": {},
            # ADR-012 prong 2: recovery attempts attributed per ETCLOVG harness layer.
            "by_layer": {},
        }

        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    def configure_provider_chain(
        self,
        primary_provider: str,
        fallback_providers: List[str],
    ) -> None:
        """Configure the provider fallback chain.

        Sets the primary provider and ordered list of fallback providers
        to use when the primary fails.

        Args:
            primary_provider: Primary provider to use
            fallback_providers: Ordered list of fallback providers

        Example:
            service.configure_provider_chain(
                primary_provider="anthropic",
                fallback_providers=["openai", "google"]
            )
        """
        self._primary_provider = primary_provider
        self._fallback_providers = fallback_providers

        # Initialize health tracking for all providers
        for provider in [primary_provider] + fallback_providers:
            if provider not in self._provider_health:
                self._provider_health[provider] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "last_failure_time": None,
                    "consecutive_failures": 0,
                    "is_available": True,
                }

        self._logger.info(
            f"Configured provider chain: primary={primary_provider}, "
            f"fallbacks={fallback_providers}"
        )

    def get_primary_provider(self) -> Optional[str]:
        """Get the current primary provider.

        Returns:
            Primary provider name or None if not configured

        Example:
            primary = service.get_primary_provider()
        """
        return self._primary_provider

    def get_fallback_providers(self) -> List[str]:
        """Get the configured fallback providers.

        Returns:
            Ordered list of fallback provider names

        Example:
            fallbacks = service.get_fallback_providers()
        """
        return self._fallback_providers.copy()

    def get_provider_health(self, provider: str) -> Dict[str, Any]:
        """Get health statistics for a provider.

        Returns success rate, failure count, availability status.

        Args:
            provider: Provider name to check

        Returns:
            Dictionary with provider health information

        Example:
            health = service.get_provider_health("anthropic")
            # {
            #   "total_requests": 100,
            #   "successful_requests": 95,
            #   "failed_requests": 5,
            #   "success_rate": 0.95,
            #   "consecutive_failures": 0,
            #   "is_available": True
            # }
        """
        if provider not in self._provider_health:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "consecutive_failures": 0,
                "is_available": False,
            }

        health = self._provider_health[provider]
        total = health["total_requests"]

        return {
            "total_requests": total,
            "successful_requests": health["successful_requests"],
            "failed_requests": health["failed_requests"],
            "success_rate": health["successful_requests"] / total if total > 0 else 0.0,
            "consecutive_failures": health["consecutive_failures"],
            "is_available": health["is_available"],
        }

    def should_switch_provider(
        self,
        current_provider: str,
        error_type: str,
        consecutive_failures: int = 0,
    ) -> bool:
        """Determine if should switch to fallback provider.

        Analyzes provider health and error patterns to recommend
        provider switching.

        Args:
            current_provider: Current provider name
            error_type: Type of error
            consecutive_failures: Consecutive failure count

        Returns:
            True if should switch to fallback provider

        Example:
            if service.should_switch_provider("anthropic", "rate_limit", 3):
                # Switch to fallback provider
        """
        # Check if we have fallback providers available
        if not self._fallback_providers:
            return False

        # Get current provider health
        health = self.get_provider_health(current_provider)

        # Switch if:
        # 1. Provider is unavailable
        # 2. High consecutive failures (> 5)
        # 3. Low success rate (< 50% with > 10 requests)
        # 4. Rate limit errors with many failures

        if not health["is_available"]:
            self._logger.warning(f"Provider {current_provider} is unavailable, should switch")
            return True

        if health["consecutive_failures"] >= 5:
            self._logger.warning(
                f"Provider {current_provider} has {health['consecutive_failures']} "
                f"consecutive failures, should switch"
            )
            return True

        if health["total_requests"] >= 10 and health["success_rate"] < 0.5:
            self._logger.warning(
                f"Provider {current_provider} has low success rate "
                f"({health['success_rate']:.1%}), should switch"
            )
            return True

        if error_type == "rate_limit" and consecutive_failures >= 3:
            self._logger.warning(
                f"Provider {current_provider} has {consecutive_failures} "
                f"rate limit errors, should switch"
            )
            return True

        return False

    def get_next_provider(self, current_provider: Optional[str] = None) -> Optional[str]:
        """Get the next provider in the fallback chain.

        Returns the next available provider after the current one.
        Wraps around to primary if no fallbacks available.

        Args:
            current_provider: Current provider (or None to get primary)

        Returns:
            Next provider name or None

        Example:
            next_provider = service.get_next_provider("anthropic")
        """
        if not self._primary_provider:
            return None

        # If no current provider, return primary
        if not current_provider:
            return self._primary_provider

        # If current is primary, return first fallback
        if current_provider == self._primary_provider:
            if self._fallback_providers:
                return self._fallback_providers[0]

            # No fallbacks, return primary
            return self._primary_provider

        # Find current provider in chain
        all_providers = [self._primary_provider] + self._fallback_providers

        try:
            current_index = all_providers.index(current_provider)
            next_index = (current_index + 1) % len(all_providers)
            return all_providers[next_index]
        except ValueError:
            # Current provider not in chain, return primary
            return self._primary_provider

    def update_provider_health(
        self,
        provider: str,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """Update provider health tracking after a request.

        Tracks success/failure rates and consecutive failures to determine
        provider availability.

        Args:
            provider: Provider name
            success: Whether the request succeeded
            error_type: Optional error type for classification

        Example:
            service.update_provider_health("anthropic", success=True)
        """
        if provider not in self._provider_health:
            self._provider_health[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "last_failure_time": None,
                "consecutive_failures": 0,
                "is_available": True,
            }

        health = self._provider_health[provider]
        health["total_requests"] += 1

        if success:
            health["successful_requests"] += 1
            health["consecutive_failures"] = 0
            health["is_available"] = True

            # Log recovery after success
            if health["consecutive_failures"] > 0:
                self._logger.info(
                    f"Provider {provider} recovered after "
                    f"{health['consecutive_failures']} failures"
                )
        else:
            health["failed_requests"] += 1
            health["consecutive_failures"] += 1
            health["last_failure_time"] = datetime.now().isoformat()

            # Check if should mark unavailable
            if health["consecutive_failures"] >= 5 or (
                health["total_requests"] >= 10
                and health["successful_requests"] / health["total_requests"] < 0.5
            ):
                health["is_available"] = False
                self._logger.warning(
                    f"Provider {provider} marked as unavailable: "
                    f"{health['consecutive_failures']} consecutive failures, "
                    f"success_rate={health['successful_requests']/health['total_requests']:.1%}"
                )

    # ==========================================================================
    # Model Fallback Strategies
    # ==========================================================================

    def get_model_fallbacks(self, model: str) -> List[str]:
        """Get fallback models for a given model.

        Returns ordered list of fallback models to try if the primary
        model fails.

        Args:
            model: Model name (e.g., "claude-opus-4-5")

        Returns:
            List of fallback model names

        Example:
            fallbacks = service.get_model_fallbacks("claude-opus-4-5")
            # ["claude-sonnet-4", "claude-3-5-haiku"]
        """
        return self._model_fallbacks.get(model, [])

    def can_use_model_fallback(
        self,
        model: str,
        error_type: str,
    ) -> bool:
        """Check if model fallback is available for the error type.

        Determines if there are fallback models available and if the error
        type supports model fallback.

        Args:
            model: Current model name
            error_type: Type of error

        Returns:
            True if model fallback is available

        Example:
            if service.can_use_model_fallback("claude-opus-4-5", "timeout"):
                # Try fallback model
        """
        # Check if fallbacks exist
        fallbacks = self.get_model_fallbacks(model)
        if not fallbacks:
            return False

        # Certain error types don't support model fallback
        no_fallback_errors = {"auth", "validation", "permission"}
        if error_type in no_fallback_errors:
            return False

        return True

    def get_fallback_model(
        self,
        model: str,
        error_type: str,
        attempt_count: int = 0,
    ) -> Optional[str]:
        """Get the appropriate fallback model.

        Returns the best fallback model based on error type and
        attempt count.

        Args:
            model: Current model name
            error_type: Type of error
            attempt_count: Current retry attempt number

        Returns:
            Fallback model name or None

        Example:
            fallback = service.get_fallback_model("claude-opus-4-5", "timeout")
            # "claude-sonnet-4"
        """
        fallbacks = self.get_model_fallbacks(model)

        if not fallbacks:
            return None

        # For rate limiting, skip to later fallback
        if error_type == "rate_limit":
            # Use index based on attempt count
            index = min(attempt_count, len(fallbacks) - 1)
            return fallbacks[index]

        # For other errors, use first fallback
        return fallbacks[0] if fallbacks else None

    def get_recovery_chain(self) -> Dict[str, Any]:
        """Get the complete recovery chain configuration.

        Returns information about primary provider, fallback providers,
        model fallbacks, and health status.

        Returns:
            Dictionary with recovery chain information

        Example:
            chain = service.get_recovery_chain()
            # {
            #   "primary_provider": "anthropic",
            #   "fallback_providers": ["openai", "google"],
            #   "provider_health": {...},
            #   "model_fallbacks": {...}
            # }
        """
        return {
            "primary_provider": self._primary_provider,
            "fallback_providers": self._fallback_providers.copy(),
            "provider_health": {
                provider: self.get_provider_health(provider)
                for provider in [self._primary_provider] + self._fallback_providers
                if provider in self._provider_health
            },
            "model_fallbacks": self._model_fallbacks.copy(),
            "has_fallback_providers": len(self._fallback_providers) > 0,
            "has_model_fallbacks": len(self._model_fallbacks) > 0,
        }

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    async def _retry_action(self, context: RecoveryContextImpl) -> bool:
        """Execute retry action.

        Args:
            context: Recovery context

        Returns:
            True if retry should be attempted
        """
        # Just indicate retry is possible
        # The actual retry is handled by the caller
        return context.attempt_count < self._max_retry_attempts

    async def _backoff_action(self, context: RecoveryContextImpl) -> bool:
        """Execute backoff action.

        Args:
            context: Recovery context

        Returns:
            True if backoff completed successfully
        """
        delay = self._calculate_retry_delay(context.attempt_count)
        self._logger.info(f"Backing off for {delay:.1f} seconds")
        await asyncio.sleep(delay)
        return True

    def _calculate_retry_delay(
        self,
        attempt: int,
    ) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number

        Returns:
            Delay in seconds
        """
        delay = self._base_retry_delay * (2**attempt)
        return min(delay, self._max_retry_delay)

    # ==========================================================================
    # Reset and Cleanup
    # ==========================================================================

    def reset(self) -> None:
        """Reset retry state and provider health tracking.

        Clears all retry counters, resets provider health status,
        and restores providers to healthy state. Useful for
        starting fresh or after configuration changes.

        Example:
            service.reset()
            # All retry state cleared, providers marked healthy
        """
        # Reset provider health
        for provider in self._provider_health:
            self._provider_health[provider] = {
                "consecutive_failures": 0,
                "last_failure_time": None,
                "last_success_time": None,
                "is_available": True,
            }

        self._logger.info("Reset recovery service state")


class AuthError(Exception):
    """Authentication error."""

    pass
