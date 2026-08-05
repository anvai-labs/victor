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

"""Recovery service protocol.

Defines the interface for error recovery and resilience operations.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from enum import Enum


@runtime_checkable
class RecoveryAction(Protocol):
    """Represents a recovery action that can be taken."""

    @property
    def name(self) -> str:
        """Action name."""
        ...

    @property
    def description(self) -> str:
        """Action description."""
        ...

    async def execute(self, context: "RecoveryContext") -> bool:
        """Execute the recovery action.

        Args:
            context: Recovery context with error and state info

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...


@runtime_checkable
class RecoveryContext(Protocol):
    """Context information for recovery decisions.

    Provides all necessary information for making recovery
    decisions and executing recovery actions.
    """

    @property
    def error(self) -> Exception:
        """The error that occurred."""
        ...

    @property
    def error_type(self) -> str:
        """Type of error (e.g., 'timeout', 'rate_limit', 'auth')."""
        ...

    @property
    def attempt_count(self) -> int:
        """Number of attempts made so far."""
        ...

    @property
    def state(self) -> Dict[str, Any]:
        """Current state snapshot."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata for recovery."""
        ...


@runtime_checkable
class RecoveryServiceProtocol(Protocol):
    """[CANONICAL] Protocol for error recovery and resilience service.

    This protocol represents the target architecture for recovery operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on recovery-related operations.

    Methods:
        classify_error: Classify an error for recovery strategy
        select_recovery_action: Select appropriate recovery action
        execute_recovery: Execute a recovery action
        can_retry: Check if operation can be retried
        get_recovery_metrics: Get recovery statistics

    Example:
        class MyRecoveryService(RecoveryServiceProtocol):
            def __init__(self):
                self._metrics = {}

            async def classify_error(self, error):
                if isinstance(error, TimeoutError):
                    return "timeout"
                elif isinstance(error, RateLimitError):
                    return "rate_limit"
                return "unknown"

            async def select_recovery_action(self, context):
                error_type = context.error_type
                if error_type == "timeout":
                    return RetryAction(max_attempts=3)
                elif error_type == "rate_limit":
                    return BackoffAction(delay=60)
                return FailAction()
    """

    async def classify_error(self, error: Exception) -> str:
        """Classify an error for recovery strategy selection.

        Analyzes the error to determine its type and appropriate
        recovery strategy.

        Args:
            error: The error to classify

        Returns:
            Error type string (e.g., 'timeout', 'rate_limit', 'auth')

        Example:
            try:
                await provider.chat(messages)
            except Exception as e:
                error_type = await recovery_service.classify_error(e)
                if error_type == "timeout":
                    # Handle timeout
                    pass
        """
        ...

    async def select_recovery_action(
        self,
        context: "RecoveryContext",
    ) -> "RecoveryAction":
        """Select appropriate recovery action for the context.

        Analyzes the error context and selects the best recovery
        action based on error type, attempt count, and state.

        Args:
            context: Recovery context with error and state info

        Returns:
            Selected recovery action

        Example:
            context = RecoveryContext(
                error=error,
                attempt_count=2,
                state=current_state
            )
            action = await recovery_service.select_recovery_action(context)
            success = await action.execute(context)
        """
        ...

    async def execute_recovery(
        self,
        context: "RecoveryContext",
    ) -> bool:
        """Execute appropriate recovery action for the context.

        Selects and executes the best recovery action automatically.

        Args:
            context: Recovery context with error and state info

        Returns:
            True if recovery succeeded, False otherwise

        Example:
            try:
                result = await operation()
            except Exception as e:
                context = RecoveryContext(error=e, attempt_count=1)
                if await recovery_service.execute_recovery(context):
                    # Recovery succeeded, retry operation
                    result = await operation()
        """
        ...

    def can_retry(
        self,
        error: Exception,
        attempt_count: int,
    ) -> bool:
        """Check if an operation can be retried.

        Determines whether the error is retryable and whether
        the attempt limit has been reached.

        Args:
            error: The error that occurred
            attempt_count: Number of attempts made so far

        Returns:
            True if operation can be retried, False otherwise

        Example:
            try:
                result = await operation()
            except Exception as e:
                if recovery_service.can_retry(e, attempt_count):
                    # Retry operation
                    pass
                else:
                    # Give up
                    raise
        """
        ...

    def should_attempt_recovery(
        self,
        error_type: str,
        consecutive_failures: int = 0,
    ) -> bool:
        """Check whether recovery should be attempted for the given error type."""
        ...

    async def handle_recovery_with_integration(
        self,
        ctx: Any,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]] = None,
        message_adder: Any = None,
    ) -> Any:
        """Handle streaming recovery through the canonical runtime path."""
        ...

    def apply_recovery_action(
        self,
        recovery_action: Any,
        ctx: Any,
        message_adder: Any = None,
    ) -> Any:
        """Apply a recovery action through the canonical runtime path."""
        ...

    def check_natural_completion(
        self,
        ctx: Any,
        has_tool_calls: bool,
        content_length: int,
    ) -> Any:
        """Check whether the current streaming turn should terminate naturally."""
        ...

    def handle_empty_response(
        self,
        ctx: Any,
    ) -> Any:
        """Handle an empty model response during streaming."""
        ...

    def get_recovery_fallback_message(
        self,
        ctx: Any,
    ) -> str:
        """Get the fallback recovery message for streaming."""
        ...

    def check_tool_budget(
        self,
        ctx: Any,
        warning_threshold: int = 250,
        warning_pct: float = 0.8,
        warning_remaining: int = 5,
    ) -> Any:
        """Check whether the streaming tool budget is approaching exhaustion."""
        ...

    def truncate_tool_calls(
        self,
        ctx: Any,
        tool_calls: List[Dict[str, Any]],
        max_calls: int,
    ) -> Any:
        """Truncate tool calls to the allowed budget."""
        ...

    def filter_blocked_tool_calls(
        self,
        ctx: Any,
        tool_calls: List[Dict[str, Any]],
    ) -> Any:
        """Filter tool calls that are blocked by runtime safeguards."""
        ...

    def check_blocked_threshold(
        self,
        ctx: Any,
        all_blocked: bool,
    ) -> Any:
        """Check whether blocked-tool thresholds require recovery action."""
        ...

    def check_force_action(
        self,
        ctx: Any,
    ) -> Any:
        """Check whether recovery should force a follow-up action."""
        ...

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery statistics and metrics.

        Returns metrics about recovery operations including:
        - Total recovery attempts
        - Success rate by error type
        - Average recovery time
        - Circuit breaker state

        Returns:
            Dictionary with recovery metrics

        Example:
            metrics = recovery_service.get_recovery_metrics()
            print(f"Recovery success rate: {metrics['success_rate']:.1%}")
        """
        ...

    def reset_metrics(self) -> None:
        """Reset recovery metrics.

        Useful for testing or starting fresh metrics collection.
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the recovery service is healthy.

        A healthy recovery service should:
        - Have recovery actions configured
        - Have metrics tracking enabled
        - Not be in a degraded state

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker functionality.

    Circuit breakers prevent cascading failures by stopping
    requests to failing services.
    """

    async def record_success(self, service: str) -> None:
        """Record a successful operation for a service.

        Args:
            service: Service name
        """
        ...

    async def record_failure(self, service: str, error: Exception) -> None:
        """Record a failed operation for a service.

        Args:
            service: Service name
            error: Error that occurred
        """
        ...

    async def is_circuit_open(self, service: str) -> bool:
        """Check if circuit is open for a service.

        An open circuit blocks requests to the service.

        Args:
            service: Service name

        Returns:
            True if circuit is open, False otherwise
        """
        ...

    async def reset_circuit(self, service: str) -> None:
        """Reset circuit breaker for a service.

        Args:
            service: Service name
        """
        ...

    def get_circuit_state(self, service: str) -> Dict[str, Any]:
        """Get circuit breaker state for a service.

        Args:
            service: Service name

        Returns:
            Circuit state information
        """
        ...


@runtime_checkable
class RetryProtocol(Protocol):
    """Protocol for retry logic.

    Handles retry operations with exponential backoff and
    jitter for resilience.
    """

    async def should_retry(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int,
    ) -> bool:
        """Determine if operation should be retried.

        Args:
            error: The error that occurred
            attempt: Current attempt number
            max_attempts: Maximum retry attempts

        Returns:
            True if should retry, False otherwise
        """
        ...

    async def get_retry_delay(
        self,
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Delay in seconds before next retry
        """
        ...

    async def execute_with_retry(
        self,
        operation: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """Execute operation with automatic retry.

        Args:
            operation: Async operation to execute
            max_attempts: Maximum retry attempts
            base_delay: Base delay for exponential backoff

        Returns:
            Operation result

        Raises:
            Last error if all retries fail

        Example:
            result = await retry_service.execute_with_retry(
                lambda: provider.chat(messages),
                max_attempts=3
            )
        """
        ...


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol for recovery strategy implementations.

    Defines the interface for custom recovery strategies.
    """

    async def can_recover(self, context: "RecoveryContext") -> bool:
        """Check if this strategy can recover from the error.

        Args:
            context: Recovery context

        Returns:
            True if strategy can handle this error
        """
        ...

    async def recover(self, context: "RecoveryContext") -> bool:
        """Execute recovery strategy.

        Args:
            context: Recovery context

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...

    @property
    def priority(self) -> int:
        """Strategy priority (higher = tried first)."""
        ...

    @property
    def name(self) -> str:
        """Strategy name."""
        ...
