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

"""Resilience patterns for robust distributed systems.

This module implements enterprise-grade resilience patterns:
- Circuit Breaker: Prevents cascading failures (subclasses canonical from providers)
- Retry with exponential backoff: Handles transient failures
- Bulkhead: Isolates failures to prevent system-wide impact
- Timeout: Prevents indefinite blocking

Design Patterns:
- State Pattern: Circuit breaker states (Closed, Open, HalfOpen)
- Decorator Pattern: Wrap functions with resilience behavior
- Strategy Pattern: Configurable backoff strategies

Example:
    from victor.observability.resilience import (
        ObservableCircuitBreaker,
        retry_with_backoff,
        Bulkhead,
    )

    # Circuit breaker for external API (with metrics support)
    breaker = ObservableCircuitBreaker(
        failure_threshold=5,
        timeout_seconds=30.0,
    )

    @breaker
    async def call_external_api():
        ...

    # Retry with exponential backoff
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_data():
        ...

    # Bulkhead for resource isolation
    bulkhead = Bulkhead(max_concurrent=10)
    async with bulkhead:
        await process_request()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, TYPE_CHECKING

# Import shared circuit-breaker types from core to avoid cross-layer coupling
from victor.core.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig as CanonicalCircuitBreakerConfig,
    CircuitBreakerError,
)

# Import the canonical CircuitBreaker to subclass
from victor.providers.circuit_breaker import CircuitBreaker as CanonicalCircuitBreaker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Backoff Strategies (Strategy Pattern)
# =============================================================================


class BackoffStrategy(ABC):
    """Abstract base for backoff calculation strategies."""

    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for given attempt.

        Args:
            attempt: Current attempt number (0-indexed).
            base_delay: Base delay in seconds.

        Returns:
            Delay in seconds before next retry.
        """
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff: delay = base * (multiplier ^ attempt).

    With jitter to prevent thundering herd.
    """

    def __init__(
        self,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
    ) -> None:
        """Initialize exponential backoff.

        Args:
            multiplier: Exponential multiplier.
            max_delay: Maximum delay cap.
            jitter: Random jitter factor (0-1).
        """
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter

    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        delay = min(base_delay * (self.multiplier**attempt), self.max_delay)
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


class LinearBackoff(BackoffStrategy):
    """Linear backoff: delay = base * attempt."""

    def __init__(self, max_delay: float = 60.0) -> None:
        self.max_delay = max_delay

    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return min(base_delay * (attempt + 1), self.max_delay)


class ConstantBackoff(BackoffStrategy):
    """Constant delay between retries."""

    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        return base_delay


class DecorrelatedJitterBackoff(BackoffStrategy):
    """AWS-style decorrelated jitter for better distribution.

    sleep = min(cap, random_between(base, sleep * 3))
    """

    def __init__(self, max_delay: float = 60.0) -> None:
        self.max_delay = max_delay
        self._last_delay: Optional[float] = None

    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        if self._last_delay is None:
            self._last_delay = base_delay

        delay = random.uniform(base_delay, self._last_delay * 3)
        delay = min(delay, self.max_delay)
        self._last_delay = delay
        return delay


# =============================================================================
# Retry Decorator
# =============================================================================


@dataclass
class ObservabilityRetryConfig:
    """Configuration for observability retry behavior.

    Renamed from RetryConfig to be semantically distinct:
    - ObservabilityRetryConfig (here): With BackoffStrategy and on_retry callback
    - ProviderRetryConfig (victor.providers.resilience): Provider-specific with retryable_patterns
    - AgentRetryConfig (victor.agent.resilience): Agent-specific with jitter flag
    """

    max_retries: int = 3
    base_delay: float = 1.0
    backoff_strategy: BackoffStrategy = field(default_factory=ExponentialBackoff)
    retryable_exceptions: tuple = (Exception,)
    on_retry: Optional[Callable[[int, Exception, float], None]] = None


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: Optional[BackoffStrategy] = None,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """Decorator for retry with configurable backoff (COMPATIBILITY WRAPPER).

    .. deprecated::
        This decorator is a compatibility wrapper around the canonical retry
        implementation in victor.core.retry. For new code, use:

        - `from victor.core.retry import with_retry` for async functions
        - `from victor.core.retry import with_retry_sync` for sync functions
        - `RetryExecutor` class for more control

        This wrapper will be removed in version 0.10.0.

    The canonical implementation provides:
    - More robust error handling with RetryContext
    - Better observability with hooks (on_retry, on_success, on_failure)
    - Support for multiple backoff strategies (exponential, linear, fixed, none)
    - Jitter to prevent thundering herd problems
    - Exception type filtering (retryable/non-retryable)

    Args:
        max_retries: Maximum retry attempts.
        base_delay: Base delay between retries.
        backoff_strategy: Strategy for calculating delays (ignored, uses canonical).
        retryable_exceptions: Exceptions that trigger retry.
        on_retry: Callback called on each retry (preserved for compatibility).

    Returns:
        Decorated function with retry behavior.

    Example:
        # Old (deprecated):
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def fetch_data():
            response = await client.get(url)
            return response.json()

        # New (recommended):
        from victor.core.retry import with_retry, ExponentialBackoffStrategy

        strategy = ExponentialBackoffStrategy(max_attempts=4, base_delay=1.0)
        executor = RetryExecutor(strategy)

        @executor.execute_async
        async def fetch_data():
            response = await client.get(url)
            return response.json()
    """
    # Import canonical retry implementation
    from victor.core.retry import (
        with_retry,
        with_retry_sync,
        ExponentialBackoffStrategy as CanonicalExponentialBackoffStrategy,
        RetryExecutor,
    )

    # Map legacy parameters to canonical strategy
    # Note: backoff_strategy parameter is ignored - canonical handles this internally
    # Convert default (Exception,) to None to mean "retry all exceptions"
    # The canonical implementation checks for exact type match, not subclass relationships
    canonical_retryable = (
        None
        if retryable_exceptions == (Exception,)
        else (set(retryable_exceptions) if retryable_exceptions else None)
    )

    strategy = CanonicalExponentialBackoffStrategy(
        max_attempts=max_retries + 1,  # +1 because max_attempts includes initial
        base_delay=base_delay,
        max_delay=base_delay * 32,  # Cap at ~32x base
        multiplier=2.0,
        jitter=0.1,
        retryable_exceptions=canonical_retryable,
    )

    # Create executor with strategy
    executor = RetryExecutor(strategy)

    def decorator(func: F) -> F:
        # Preserve on_retry callback if provided
        if on_retry:
            strategy.on_retry = lambda ctx: (
                on_retry(ctx.attempt, ctx.last_exception, strategy.get_delay(ctx))
                if ctx.last_exception
                else None
            )

        # Use canonical executor
        if asyncio.iscoroutinefunction(func):
            # For async functions, wrap with executor
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await executor.execute_async(func, *args, **kwargs)
                # Extract result from RetryResult for backward compatibility
                if result.success:
                    return result.result  # type: ignore
                else:
                    if result.exception:
                        raise result.exception
                    raise RuntimeError("Retry execution failed with no exception")

            # Preserve original function metadata
            async_wrapper.__name__ = func.__name__
            async_wrapper.__doc__ = func.__doc__
            async_wrapper.__module__ = func.__module__
            async_wrapper.__wrapped__ = func  # type: ignore
            return async_wrapper  # type: ignore
        else:
            # For sync functions, wrap with executor
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = executor.execute(func, *args, **kwargs)
                # Extract result from RetryResult for backward compatibility
                if result.success:
                    return result.result  # type: ignore
                else:
                    if result.exception:
                        raise result.exception
                    raise RuntimeError("Retry execution failed with no exception")

            # Preserve original function metadata
            sync_wrapper.__name__ = func.__name__
            sync_wrapper.__doc__ = func.__doc__
            sync_wrapper.__module__ = func.__module__
            sync_wrapper.__wrapped__ = func  # type: ignore
            return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Circuit Breaker (Observable Subclass)
# =============================================================================

# ObservableCircuitBreaker extends the canonical CircuitBreaker from
# victor.providers.circuit_breaker with metrics collection support.


class ObservableCircuitBreaker(CanonicalCircuitBreaker):
    """Circuit breaker with observability features for metrics and callbacks.

    Extends the canonical CircuitBreaker from victor.providers.circuit_breaker
    with a get_metrics() method for observability integration.

    This is the recommended circuit breaker for observability use cases.
    For simple decorator/context manager usage, use CanonicalCircuitBreaker directly.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Calls fail immediately, waiting for recovery timeout
    - HALF_OPEN: Limited calls allowed to test recovery

    Example:
        breaker = ObservableCircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        @breaker
        async def call_service():
            return await http_client.get(url)

        # Get metrics for observability
        metrics = breaker.get_metrics()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: tuple = (),
        name: Optional[str] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
        *,
        timeout_seconds: Optional[float] = None,  # Canonical parameter name
    ) -> None:
        """Initialize observable circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
            success_threshold: Successes in half-open to close.
            recovery_timeout: Seconds before trying half-open (canonical name).
            half_open_max_calls: Max concurrent calls in half-open.
            excluded_exceptions: Exceptions that don't count as failures.
            name: Optional name for logging/metrics.
            on_state_change: Callback for state transitions (old_state, new_state).
            timeout_seconds: Alias for recovery_timeout (for compatibility).
        """
        # Support both parameter names for compatibility
        actual_timeout = timeout_seconds if timeout_seconds is not None else recovery_timeout

        # Wrap callback to match parent's 3-param signature (old, new, name)
        # Parent expects (old_state, new_state, name), we provide (old_state, new_state)
        wrapped_callback: Optional[Callable] = None
        if on_state_change is not None:

            def wrapped_callback(old: CircuitState, new: CircuitState, cb_name: str) -> None:
                on_state_change(old, new)  # type: ignore

        # Initialize parent canonical circuit breaker
        super().__init__(
            failure_threshold=failure_threshold,
            recovery_timeout=actual_timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold,
            excluded_exceptions=excluded_exceptions,
            name=name or "observable_circuit_breaker",
            on_state_change=wrapped_callback,
        )

    # Backward-compatible properties for accessing private attributes
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count."""
        return self._success_count

    @property
    def timeout_seconds(self) -> float:
        """Get recovery timeout in seconds (backward compatible alias)."""
        return self.recovery_timeout

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for observability.

        Returns:
            Dict with current circuit breaker state and statistics.
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "failure_threshold": self.failure_threshold,
            "timeout_seconds": self.recovery_timeout,  # Backward compatible key
            "recovery_timeout": self.recovery_timeout,
            "half_open_calls": self._half_open_calls,
        }


# Backward compatibility alias - exports ObservableCircuitBreaker as CircuitBreaker
CircuitBreaker = ObservableCircuitBreaker


# =============================================================================
# Bulkhead Pattern
# =============================================================================


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    pass


class Bulkhead:
    """Bulkhead pattern for resource isolation.

    Limits concurrent access to a resource to prevent
    one component from exhausting system resources.

    Uses asyncio.Semaphore for limiting concurrency.

    Example:
        # Limit concurrent DB connections
        db_bulkhead = Bulkhead(max_concurrent=10, name="database")

        async with db_bulkhead:
            await db.query(...)

        # With timeout
        async with db_bulkhead.acquire(timeout=5.0):
            await db.query(...)
    """

    def __init__(
        self,
        max_concurrent: int,
        max_waiting: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent executions.
            max_waiting: Maximum waiting queue size.
            name: Optional name for logging.
        """
        self._max_concurrent = max_concurrent
        self._max_waiting = max_waiting
        self._name = name or "bulkhead"
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting_count = 0
        self._active_count = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Get number of active executions."""
        return self._active_count

    @property
    def waiting_count(self) -> int:
        """Get number of waiting requests."""
        return self._waiting_count

    @property
    def available(self) -> int:
        """Get available slots."""
        return self._max_concurrent - self._active_count

    async def acquire(self, timeout: Optional[float] = None) -> "BulkheadContext":
        """Acquire bulkhead slot with optional timeout.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            BulkheadContext for use with 'async with'.

        Raises:
            BulkheadFullError: If queue is full or timeout exceeded.
        """
        async with self._lock:
            if self._max_waiting is not None and self._waiting_count >= self._max_waiting:
                raise BulkheadFullError(
                    f"Bulkhead '{self._name}' queue full: {self._waiting_count}/{self._max_waiting}"
                )
            self._waiting_count += 1

        try:
            if timeout is not None:
                acquired = await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                if not acquired:
                    raise BulkheadFullError(f"Bulkhead '{self._name}' timeout")
            else:
                await self._semaphore.acquire()

            async with self._lock:
                self._waiting_count -= 1
                self._active_count += 1

            return BulkheadContext(self)

        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting_count -= 1
            raise BulkheadFullError(f"Bulkhead '{self._name}' timeout after {timeout}s")

    async def release(self) -> None:
        """Release bulkhead slot."""
        async with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    async def __aenter__(self) -> "Bulkhead":
        """Enter bulkhead context."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit bulkhead context."""
        await self.release()

    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "name": self._name,
            "max_concurrent": self._max_concurrent,
            "active_count": self._active_count,
            "waiting_count": self._waiting_count,
            "available": self.available,
        }


class BulkheadContext:
    """Context manager for bulkhead."""

    def __init__(self, bulkhead: Bulkhead) -> None:
        self._bulkhead = bulkhead

    async def __aenter__(self) -> "BulkheadContext":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._bulkhead.release()


# =============================================================================
# Timeout Pattern
# =============================================================================


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


def with_timeout(
    timeout: float,
    error_message: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to add timeout to async functions.

    Args:
        timeout: Maximum seconds to wait.
        error_message: Custom error message.

    Returns:
        Decorated function with timeout.

    Example:
        @with_timeout(5.0)
        async def fetch_data():
            return await slow_operation()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                msg = error_message or f"{func.__name__} timed out after {timeout}s"
                raise TimeoutError(msg)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter.

    Controls the rate of operations using the token bucket algorithm.

    Example:
        limiter = RateLimiter(rate=10, capacity=20)  # 10 req/s, burst of 20

        if await limiter.acquire():
            await make_request()
        else:
            # Rate limited, handle accordingly
            pass
    """

    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            rate: Tokens per second.
            capacity: Maximum tokens (burst capacity).
            name: Optional name for logging.
        """
        self._rate = rate
        self._capacity = capacity or rate
        self._name = name or "rate_limiter"
        self._tokens = self._capacity
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens acquired, False if rate limited.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update

            # Refill tokens based on elapsed time
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_token(self, tokens: float = 1.0) -> None:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire.
        """
        while not await self.acquire(tokens):
            # Calculate wait time
            async with self._lock:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate
            await asyncio.sleep(wait_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            "name": self._name,
            "rate": self._rate,
            "capacity": self._capacity,
            "tokens_available": self._tokens,
        }


# =============================================================================
# Composite Resilience (Facade Pattern)
# =============================================================================


class ResiliencePolicy:
    """Combines multiple resilience patterns.

    Applies patterns in order: Rate Limit -> Bulkhead -> Circuit Breaker -> Retry -> Timeout

    Example:
        policy = ResiliencePolicy(
            circuit_breaker=ObservableCircuitBreaker(failure_threshold=5),
            retry_config=ObservabilityRetryConfig(max_retries=3),
            bulkhead=Bulkhead(max_concurrent=10),
            timeout=5.0,
        )

        @policy
        async def protected_call():
            return await external_service()
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[ObservabilityRetryConfig] = None,
        bulkhead: Optional[Bulkhead] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize resilience policy.

        Args:
            circuit_breaker: Circuit breaker instance.
            retry_config: Retry configuration.
            bulkhead: Bulkhead instance.
            rate_limiter: Rate limiter instance.
            timeout: Timeout in seconds.
            name: Policy name.
        """
        self._circuit_breaker = circuit_breaker
        self._retry_config = retry_config
        self._bulkhead = bulkhead
        self._rate_limiter = rate_limiter
        self._timeout = timeout
        self._name = name or "resilience_policy"

    def __call__(self, func: F) -> F:
        """Apply resilience policy to function."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(func, *args, **kwargs)

        return wrapper  # type: ignore

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with resilience policy.

        Args:
            func: Function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.
        """
        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.wait_for_token()

        # Bulkhead
        async def execute_with_bulkhead() -> Any:
            if self._bulkhead:
                async with self._bulkhead:
                    return await self._execute_inner(func, *args, **kwargs)
            return await self._execute_inner(func, *args, **kwargs)

        return await execute_with_bulkhead()

    async def _execute_inner(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute with circuit breaker, retry, and timeout."""

        async def execute_once() -> Any:
            # Circuit breaker check - raises CircuitBreakerError if blocked
            if self._circuit_breaker:
                await self._circuit_breaker._check_state()

            try:
                # Apply timeout
                if self._timeout:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=self._timeout)
                else:
                    result = await func(*args, **kwargs)

                # Record success
                if self._circuit_breaker:
                    self._circuit_breaker._record_success()

                return result

            except Exception as e:
                # Record failure
                if self._circuit_breaker:
                    self._circuit_breaker._record_failure(e)
                raise

        # Apply retry if configured
        if self._retry_config:
            last_exception: Optional[Exception] = None
            strategy = self._retry_config.backoff_strategy

            for attempt in range(self._retry_config.max_retries + 1):
                try:
                    return await execute_once()
                except self._retry_config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == self._retry_config.max_retries:
                        raise

                    delay = strategy.calculate_delay(attempt, self._retry_config.base_delay)

                    if self._retry_config.on_retry:
                        self._retry_config.on_retry(attempt + 1, e, delay)

                    await asyncio.sleep(delay)

            raise last_exception  # type: ignore

        return await execute_once()

    def get_metrics(self) -> Dict[str, Any]:
        """Get all resilience metrics."""
        metrics: Dict[str, Any] = {"name": self._name}

        if self._circuit_breaker:
            metrics["circuit_breaker"] = self._circuit_breaker.get_metrics()
        if self._bulkhead:
            metrics["bulkhead"] = self._bulkhead.get_metrics()
        if self._rate_limiter:
            metrics["rate_limiter"] = self._rate_limiter.get_metrics()

        return metrics
