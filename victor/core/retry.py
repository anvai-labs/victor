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

"""Unified retry strategies for Victor.

This module provides a standardized interface for retry logic used across:
- Tool execution (ToolExecutor)
- Provider API calls
- MCP client connections
- Any other retriable operations

The goal is to replace scattered retry implementations with a single,
configurable, well-tested strategy system.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryOutcome(Enum):
    """Possible outcomes of a retry decision."""

    RETRY = "retry"
    ABORT = "abort"
    SUCCESS = "success"


@dataclass
class RetryContext:
    """Context passed to retry strategies for decision-making.

    Tracks the current state of a retriable operation including
    attempt count, elapsed time, and exception history.
    """

    attempt: int = 0
    max_attempts: int = 3
    start_time: float = field(default_factory=time.time)
    last_exception: Optional[Exception] = None
    exceptions: list[Exception] = field(default_factory=list)
    total_delay: float = 0.0

    @property
    def elapsed(self) -> float:
        """Time elapsed since first attempt."""
        return time.time() - self.start_time

    @property
    def attempts_remaining(self) -> int:
        """Number of attempts remaining."""
        return max(0, self.max_attempts - self.attempt)

    def record_exception(self, exc: Exception) -> None:
        """Record an exception from a failed attempt."""
        self.last_exception = exc
        self.exceptions.append(exc)

    def record_delay(self, delay: float) -> None:
        """Record delay time."""
        self.total_delay += delay


class BaseRetryStrategy(ABC):
    """Abstract base class for retry strategies.

    Renamed from RetryStrategy to be semantically distinct:
    - BaseRetryStrategy (here): Abstract base with should_retry(), get_delay() methods
    - ProviderRetryStrategy (victor.providers.resilience): Concrete provider retry with execute()
    - BatchRetryStrategy (victor.workflows.batch_executor): Enum for batch retry modes

    Implementations define when and how long to wait between retries.
    This provides a consistent interface for all retry logic in Victor.
    """

    @abstractmethod
    def should_retry(self, context: RetryContext) -> bool:
        """Determine if another retry attempt should be made.

        Args:
            context: Current retry context with attempt info

        Returns:
            True if should retry, False to abort
        """
        pass

    @abstractmethod
    def get_delay(self, context: RetryContext) -> float:
        """Calculate delay before next retry attempt.

        Args:
            context: Current retry context

        Returns:
            Delay in seconds before next attempt
        """
        pass

    def on_retry(self, context: RetryContext) -> None:  # noqa: B027
        """Hook called before each retry attempt.

        Override to add logging, metrics, or other side effects.

        Args:
            context: Current retry context
        """

    def on_success(self, context: RetryContext) -> None:  # noqa: B027
        """Hook called on successful completion.

        Args:
            context: Final retry context
        """

    def on_failure(self, context: RetryContext) -> None:  # noqa: B027
        """Hook called when all retries exhausted.

        Args:
            context: Final retry context
        """


def compute_backoff_delay(
    attempt: int,
    base_delay: float,
    *,
    multiplier: float = 2.0,
    max_delay: Optional[float] = None,
    jitter: float = 0.0,
) -> float:
    """Compute a single exponential-backoff delay.

    Pure helper that centralizes the exponential-backoff delay formula so
    call sites do not reimplement ``base * multiplier ** attempt`` inline.
    The delay is ``base_delay * multiplier ** attempt``, optionally capped at
    ``max_delay`` and perturbed by symmetric ``+/- jitter`` fractional jitter.

    Args:
        attempt: Zero-based exponent for the growth term. A value of ``0``
            yields ``base_delay``; ``1`` yields ``base_delay * multiplier``.
            Callers whose loop counter starts at 1 should pass ``counter - 1``.
        base_delay: Delay in seconds for ``attempt == 0``.
        multiplier: Exponential growth factor (default ``2.0`` = doubling).
        max_delay: Optional cap in seconds. ``None`` leaves the delay uncapped.
        jitter: Symmetric jitter fraction applied after the cap. ``0.0``
            disables jitter (deterministic); ``0.1`` gives +/-10% randomness.

    Returns:
        Delay in seconds, never negative.
    """
    delay = base_delay * (multiplier**attempt)
    if max_delay is not None:
        delay = min(delay, max_delay)
    if jitter > 0:
        jitter_amount = delay * jitter
        delay += random.uniform(-jitter_amount, jitter_amount)
    return max(0.0, delay)


class ExponentialBackoffStrategy(BaseRetryStrategy):
    """Exponential backoff with optional jitter.

    Default strategy for most operations. Delay doubles after each
    attempt up to a maximum, with optional random jitter to prevent
    thundering herd problems.

    Delay formula: min(max_delay, base_delay * (multiplier ^ attempt)) * (1 + jitter)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    ):
        """Initialize exponential backoff strategy.

        Args:
            max_attempts: Maximum number of attempts (including initial)
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            multiplier: Exponential multiplier (default 2.0 = doubling)
            jitter: Random jitter factor (0.1 = ±10% randomness)
            retryable_exceptions: Only retry these exception types (None = all)
            non_retryable_exceptions: Never retry these exception types
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions or set()

    def should_retry(self, context: RetryContext) -> bool:
        """Check if retry should occur based on attempts and exception type."""
        # Check attempt limit
        if context.attempt >= self.max_attempts:
            return False

        # Check exception type if we have one
        if context.last_exception:
            exc_type = type(context.last_exception)

            # Never retry non-retryable exceptions
            if exc_type in self.non_retryable_exceptions:
                return False

            # If retryable_exceptions specified, only retry those
            if self.retryable_exceptions is not None:
                return exc_type in self.retryable_exceptions

        return True

    def get_delay(self, context: RetryContext) -> float:
        """Calculate exponential delay with jitter."""
        # Calculate base exponential delay
        delay = self.base_delay * (self.multiplier**context.attempt)

        # Apply max cap
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def on_retry(self, context: RetryContext) -> None:
        """Log retry attempt."""
        logger.debug(
            f"Retry attempt {context.attempt + 1}/{self.max_attempts} "
            f"after {context.total_delay:.2f}s total delay"
        )


class LinearBackoffStrategy(BaseRetryStrategy):
    """Linear backoff - delay increases linearly with each attempt.

    Useful for operations where rapid initial retries are acceptable
    but slower backoff is needed for persistent failures.

    Delay formula: base_delay + (increment * attempt)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 30.0,
    ):
        """Initialize linear backoff strategy.

        Args:
            max_attempts: Maximum number of attempts
            base_delay: Initial delay in seconds
            increment: Delay increase per attempt
            max_delay: Maximum delay cap
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay

    def should_retry(self, context: RetryContext) -> bool:
        """Check if more attempts remain."""
        return context.attempt < self.max_attempts

    def get_delay(self, context: RetryContext) -> float:
        """Calculate linear delay."""
        delay = self.base_delay + (self.increment * context.attempt)
        return min(delay, self.max_delay)


class FixedDelayStrategy(BaseRetryStrategy):
    """Fixed delay between retries - no backoff.

    Suitable for operations where the failure is likely transient
    and consistent retry timing is preferred.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
    ):
        """Initialize fixed delay strategy.

        Args:
            max_attempts: Maximum number of attempts
            delay: Fixed delay between attempts
        """
        self.max_attempts = max_attempts
        self.delay = delay

    def should_retry(self, context: RetryContext) -> bool:
        """Check if more attempts remain."""
        return context.attempt < self.max_attempts

    def get_delay(self, context: RetryContext) -> float:
        """Return fixed delay."""
        return self.delay


class NoRetryStrategy(BaseRetryStrategy):
    """No retry - fail immediately on first error.

    Useful for operations that should not be retried,
    or when wrapping already-retriable operations.
    """

    def should_retry(self, context: RetryContext) -> bool:
        """Never retry."""
        return False

    def get_delay(self, context: RetryContext) -> float:
        """No delay needed."""
        return 0.0


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    context: RetryContext = field(default_factory=RetryContext)

    @property
    def attempts(self) -> int:
        """Number of attempts made."""
        return self.context.attempt

    @property
    def total_time(self) -> float:
        """Total time including delays."""
        return self.context.elapsed


class RetryExecutor:
    """Executes operations with retry logic.

    Provides both synchronous and asynchronous execution with
    configurable retry strategies.
    """

    def __init__(self, strategy: Optional[BaseRetryStrategy] = None):
        """Initialize executor with a retry strategy.

        Args:
            strategy: Retry strategy to use (default: ExponentialBackoffStrategy)
        """
        self.strategy = strategy or ExponentialBackoffStrategy()

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> "RetryResult":
        """Execute an async function with retries.

        This is an alias for execute_async for backward compatibility.
        Use this when executing async functions in an async context.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with success status, result, and context
        """
        return await self.execute_async(func, *args, **kwargs)

    async def execute_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute an async function with retries.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with success status, result, and context
        """
        context = RetryContext(max_attempts=getattr(self.strategy, "max_attempts", 3))

        while True:
            context.attempt += 1

            try:
                result = await func(*args, **kwargs)
                self.strategy.on_success(context)
                return RetryResult(
                    success=True,
                    result=result,
                    context=context,
                )
            except Exception as e:
                context.record_exception(e)

                if self.strategy.should_retry(context):
                    self.strategy.on_retry(context)
                    delay = self.strategy.get_delay(context)
                    context.record_delay(delay)

                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.strategy.on_failure(context)
                    return RetryResult(
                        success=False,
                        exception=e,
                        context=context,
                    )

    def execute_sync(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute a sync function with retries.

        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with success status, result, and context
        """
        context = RetryContext(max_attempts=getattr(self.strategy, "max_attempts", 3))

        while True:
            context.attempt += 1

            try:
                result = func(*args, **kwargs)
                self.strategy.on_success(context)
                return RetryResult(
                    success=True,
                    result=result,
                    context=context,
                )
            except Exception as e:
                context.record_exception(e)

                if self.strategy.should_retry(context):
                    self.strategy.on_retry(context)
                    delay = self.strategy.get_delay(context)
                    context.record_delay(delay)

                    if delay > 0:
                        time.sleep(delay)
                else:
                    self.strategy.on_failure(context)
                    return RetryResult(
                        success=False,
                        exception=e,
                        context=context,
                    )


def _handle_retry_result(
    result: "RetryResult",
    raise_on_failure: bool,
) -> Optional[T]:
    """Handle retry result - shared logic for sync and async wrappers.

    Args:
        result: Result from executor.execute_sync() or executor.execute_async()
        raise_on_failure: Whether to raise exception on final failure

    Returns:
        Result value if successful, None if failed and not raising

    Raises:
        Exception: If result failed and raise_on_failure is True
    """
    if result.success:
        return cast(T, result.result)
    elif raise_on_failure and result.exception:
        raise result.exception
    else:
        return None  # type: ignore


def with_retry(
    strategy: Optional[BaseRetryStrategy] = None,
    raise_on_failure: bool = True,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for async functions with retry logic.

    Args:
        strategy: Retry strategy (default: ExponentialBackoffStrategy)
        raise_on_failure: Whether to raise exception on final failure

    Usage:
        @with_retry(ExponentialBackoffStrategy(max_attempts=5))
        async def flaky_api_call():
            ...
    """
    executor = RetryExecutor(strategy)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            result = await executor.execute_async(func, *args, **kwargs)
            return _handle_retry_result(result, raise_on_failure)

        return wrapper

    return decorator


def with_retry_sync(
    strategy: Optional[BaseRetryStrategy] = None,
    raise_on_failure: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync functions with retry logic.

    Args:
        strategy: Retry strategy (default: ExponentialBackoffStrategy)
        raise_on_failure: Whether to raise exception on final failure

    Usage:
        @with_retry_sync(max_attempts=3)
        def flaky_file_op():
            ...
    """
    executor = RetryExecutor(strategy)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = executor.execute_sync(func, *args, **kwargs)
            return _handle_retry_result(result, raise_on_failure)

        return wrapper

    return decorator


# =============================================================================
# Pre-configured strategies for common use cases
# =============================================================================


def tool_retry_strategy(
    max_retries: int = 3, base_delay: float = 1.0
) -> ExponentialBackoffStrategy:
    """Create retry strategy optimized for tool execution.

    Uses moderate delays with jitter to handle transient failures
    from external tools (file I/O, git commands, etc.).

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries

    Returns:
        Configured ExponentialBackoffStrategy
    """
    return ExponentialBackoffStrategy(
        max_attempts=max_retries,
        base_delay=base_delay,
        max_delay=30.0,
        multiplier=2.0,
        jitter=0.1,
    )


def provider_retry_strategy(max_retries: int = 3) -> ExponentialBackoffStrategy:
    """Create retry strategy optimized for LLM provider API calls.

    Uses longer delays to respect rate limits, with significant jitter
    to prevent thundering herd on provider recovery.

    Args:
        max_retries: Maximum retry attempts

    Returns:
        Configured ExponentialBackoffStrategy
    """
    return ExponentialBackoffStrategy(
        max_attempts=max_retries,
        base_delay=2.0,
        max_delay=60.0,
        multiplier=2.0,
        jitter=0.25,  # More jitter for distributed systems
    )


def connection_retry_strategy(max_attempts: int = 5) -> ExponentialBackoffStrategy:
    """Create retry strategy for connection establishment.

    Uses aggressive retries initially with longer backoff for
    persistent connection failures (MCP, WebSocket, etc.).

    Args:
        max_attempts: Maximum connection attempts

    Returns:
        Configured ExponentialBackoffStrategy
    """
    return ExponentialBackoffStrategy(
        max_attempts=max_attempts,
        base_delay=0.5,  # Quick first retry
        max_delay=30.0,
        multiplier=2.0,
        jitter=0.15,
    )
