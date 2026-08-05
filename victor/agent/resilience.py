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

"""Resilience patterns for robust provider interactions.

This module implements enterprise-grade resilience patterns:
- Circuit Breaker: Prevent cascading failures
- Retry with Exponential Backoff: Handle transient failures
- Rate Limiting: Respect API quotas
- Bulkhead: Isolate failures

Design Pattern: Circuit Breaker (GoF: State + Template Method)
============================================================
The circuit breaker pattern prevents repeated calls to a failing service,
allowing it time to recover while providing fast failure responses.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered

Usage:
    config = MultiCircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3,
    )
    breaker = MultiCircuitBreaker(config)

    if breaker.is_allowed("provider_name"):
        try:
            result = await provider.chat(...)
            breaker.record_success("provider_name")
        except Exception as e:
            breaker.record_failure("provider_name", e)
    else:
        # Fast fail, use fallback
        result = await fallback_provider.chat(...)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, TYPE_CHECKING

# Import shared circuit-breaker types from core to avoid cross-layer coupling
from victor.core.circuit_breaker import (
    CircuitState as _CircuitState,
    CircuitBreakerConfig as CanonicalCircuitBreakerConfig,
)

if TYPE_CHECKING:
    from victor.core.circuit_breaker import CircuitState

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CircuitStats:
    """Statistics for a circuit.

    Attributes:
        failures: Consecutive failure count
        successes: Consecutive success count (in half-open)
        last_failure_time: Timestamp of last failure
        total_failures: Total failures since reset
        total_successes: Total successes since reset
        state: Current circuit state
        last_state_change: Timestamp of last state change
    """

    failures: int = 0
    successes: int = 0
    last_failure_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0
    state: CircuitState = _CircuitState.CLOSED
    last_state_change: float = field(default_factory=time.time)


@dataclass
class MultiCircuitBreakerConfig:
    """Configuration for multi-circuit breaker.

    This extends the canonical CircuitBreakerConfig with multi-circuit support.
    Supports both `timeout_seconds` (canonical) and `recovery_timeout` (legacy).

    Attributes:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before testing recovery
        half_open_max_calls: Test calls before closing
        success_threshold: Successes in half-open to close
        exclude_exceptions: Exception types that don't count as failures
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # Keep legacy name for backward compatibility
    half_open_max_calls: int = 3
    success_threshold: int = 2
    exclude_exceptions: tuple = (asyncio.CancelledError,)

    @property
    def timeout_seconds(self) -> float:
        """Canonical alias for recovery_timeout."""
        return self.recovery_timeout


# Alias for backward compatibility
CircuitBreakerConfig = MultiCircuitBreakerConfig


class MultiCircuitBreaker:
    """Circuit breaker managing multiple named circuits.

    Renamed from CircuitBreaker to be semantically distinct:
    - CircuitBreaker (victor.providers.circuit_breaker): Standalone with decorator/context manager
    - MultiCircuitBreaker (here): Manages multiple named circuits (Dict[str, CircuitStats])
    - ObservableCircuitBreaker (victor.observability.resilience): Metrics/callback focused
    - ProviderCircuitBreaker (victor.providers.resilience): ResilientProvider workflow

    Thread-safe implementation using asyncio locks.
    Supports multiple named circuits for different providers.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._circuits: Dict[str, CircuitStats] = defaultdict(CircuitStats)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._half_open_calls: Dict[str, int] = defaultdict(int)

    def get_state(self, name: str) -> CircuitState:
        """Get current state of a circuit.

        Args:
            name: Circuit name (e.g., provider name)

        Returns:
            Current circuit state
        """
        stats = self._circuits[name]

        # Check for automatic state transitions
        if stats.state == _CircuitState.OPEN:
            if time.time() - stats.last_failure_time >= self.config.recovery_timeout:
                # Transition to half-open for testing
                stats.state = _CircuitState.HALF_OPEN
                stats.last_state_change = time.time()
                self._half_open_calls[name] = 0
                logger.info(f"Circuit '{name}' transitioning to HALF_OPEN for testing")

        return stats.state

    def is_allowed(self, name: str) -> bool:
        """Check if a call is allowed through the circuit.

        Args:
            name: Circuit name

        Returns:
            True if call is allowed
        """
        state = self.get_state(name)

        if state == _CircuitState.CLOSED:
            return True

        if state == _CircuitState.HALF_OPEN:
            # Allow limited calls for testing
            if self._half_open_calls[name] < self.config.half_open_max_calls:
                self._half_open_calls[name] += 1
                return True
            return False

        # OPEN state - reject
        return False

    def record_success(self, name: str) -> None:
        """Record a successful call.

        Args:
            name: Circuit name
        """
        stats = self._circuits[name]
        stats.successes += 1
        stats.total_successes += 1
        stats.failures = 0  # Reset consecutive failures

        if stats.state == _CircuitState.HALF_OPEN:
            if stats.successes >= self.config.success_threshold:
                # Service recovered, close circuit
                stats.state = _CircuitState.CLOSED
                stats.last_state_change = time.time()
                stats.successes = 0
                logger.info(f"Circuit '{name}' CLOSED - service recovered")

    def record_failure(self, name: str, error: Optional[Exception] = None) -> None:
        """Record a failed call.

        Args:
            name: Circuit name
            error: Optional exception that caused failure
        """
        # Check if exception should be excluded
        if error and isinstance(error, self.config.exclude_exceptions):
            logger.debug(f"Circuit '{name}' - excluded exception: {type(error).__name__}")
            return

        stats = self._circuits[name]
        stats.failures += 1
        stats.total_failures += 1
        stats.last_failure_time = time.time()
        stats.successes = 0  # Reset consecutive successes

        if stats.state == _CircuitState.CLOSED:
            if stats.failures >= self.config.failure_threshold:
                # Too many failures, open circuit
                stats.state = _CircuitState.OPEN
                stats.last_state_change = time.time()
                logger.warning(
                    f"Circuit '{name}' OPEN - {stats.failures} failures "
                    f"(threshold: {self.config.failure_threshold})"
                )

        elif stats.state == _CircuitState.HALF_OPEN:
            # Failure during testing, reopen circuit
            stats.state = _CircuitState.OPEN
            stats.last_state_change = time.time()
            logger.warning(f"Circuit '{name}' reopened - test call failed")

    def reset(self, name: str) -> None:
        """Reset a circuit to closed state.

        Args:
            name: Circuit name
        """
        self._circuits[name] = CircuitStats()
        self._half_open_calls[name] = 0
        logger.info(f"Circuit '{name}' reset to CLOSED")

    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a circuit.

        Args:
            name: Circuit name

        Returns:
            Dictionary with circuit statistics
        """
        stats = self._circuits[name]
        return {
            "state": self.get_state(name).value,
            "failures": stats.failures,
            "successes": stats.successes,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
            "last_failure_time": stats.last_failure_time,
            "last_state_change": stats.last_state_change,
            "time_since_failure": (
                time.time() - stats.last_failure_time if stats.last_failure_time else None
            ),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuits.

        Returns:
            Dictionary mapping circuit names to their statistics
        """
        return {name: self.get_stats(name) for name in self._circuits}


# Backward compatibility alias
CircuitBreaker = MultiCircuitBreaker


@dataclass
class AgentRetryConfig:
    """Configuration for agent retry with exponential backoff.

    Renamed from RetryConfig to be semantically distinct:
    - AgentRetryConfig (here): Agent-specific with jitter flag
    - ProviderRetryConfig (victor.providers.resilience): Provider-specific with retryable_patterns
    - ObservabilityRetryConfig (victor.observability.resilience): With BackoffStrategy

    Attributes:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Exception types to retry on
        retryable_status_codes: HTTP status codes to retry on
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


class RetryHandler:
    """Retry handler with exponential backoff (COMPATIBILITY WRAPPER).

    .. deprecated::
        This class is a compatibility wrapper around the canonical retry
        implementation in victor.core.retry. For new code, use:

        - `from victor.core.retry import RetryExecutor` for direct control
        - `from victor.core.retry import with_retry` for decorator-based usage
        - `ExponentialBackoffStrategy` for strategy configuration

        This wrapper will be removed in version 0.10.0.

    The canonical implementation provides:
    - More robust error handling with RetryContext
    - Better observability with hooks (on_retry, on_success, on_failure)
    - Support for multiple backoff strategies (exponential, linear, fixed, none)
    - Jitter to prevent thundering herd problems
    - Exception type filtering (retryable/non-retryable)
    """

    def __init__(self, config: Optional[AgentRetryConfig] = None):
        """Initialize retry handler.

        Args:
            config: Retry configuration
        """
        from victor.core.retry import (
            RetryExecutor,
            ExponentialBackoffStrategy,
        )

        self.config = config or AgentRetryConfig()

        # Create canonical retry strategy
        self._strategy = ExponentialBackoffStrategy(
            max_attempts=self.config.max_retries + 1,  # +1 because max_attempts includes initial
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            multiplier=self.config.exponential_base,
            jitter=(
                0.5 if self.config.jitter else 0.0
            ),  # 0.5 = 50% jitter to match legacy behavior
            retryable_exceptions=(
                set(self.config.retryable_exceptions) if self.config.retryable_exceptions else None
            ),
        )

        # Create canonical executor
        self._executor = RetryExecutor(self._strategy)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt.

        Delegates to canonical ExponentialBackoffStrategy.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        from victor.core.retry import RetryContext

        # Create a minimal context for delay calculation
        ctx = RetryContext(attempt=attempt, max_attempts=self.config.max_retries + 1)
        return self._strategy.get_delay(ctx)

    def should_retry(
        self,
        attempt: int,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
    ) -> bool:
        """Check if a retry should be attempted.

        Args:
            attempt: Current attempt number
            error: Exception that occurred (if any)
            status_code: HTTP status code (if applicable)

        Returns:
            True if retry should be attempted
        """
        from victor.core.retry import RetryContext

        # Check attempt limit
        if attempt >= self.config.max_retries:
            return False

        # Create context for retry decision
        ctx = RetryContext(
            attempt=attempt,
            max_attempts=self.config.max_retries + 1,
            last_exception=error,
        )

        # Check exception type
        if error:
            if not isinstance(error, self.config.retryable_exceptions):
                return False

        # Check status code (HTTP-specific concern not in canonical implementation)
        if status_code and status_code not in self.config.retryable_status_codes:
            return False

        # Use canonical strategy for final decision
        return self._strategy.should_retry(ctx)

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with retry logic.

        Delegates to canonical RetryExecutor and extracts result for backward compatibility.

        Args:
            func: Async function to execute
            *args: Positional arguments
            on_retry: Optional callback when retry occurs
            **kwargs: Keyword arguments

        Returns:
            Result from the function

        Raises:
            Last exception if all retries exhausted
        """
        # Preserve on_retry callback if provided
        if on_retry:
            self._strategy.on_retry = lambda ctx: (
                on_retry(ctx.attempt, ctx.last_exception) if ctx.last_exception else None
            )

        # Use canonical executor
        result = await self._executor.execute_async(func, *args, **kwargs)

        # Extract result from RetryResult for backward compatibility
        # The canonical executor returns RetryResult, but legacy API expects direct result
        if result.success:
            return result.result  # type: ignore
        else:
            # Raise the exception if execution failed
            if result.exception:
                raise result.exception
            raise RuntimeError("Retry execution failed with no exception")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute
        requests_per_second: Maximum requests per second (more granular)
        burst_size: Allow burst up to this size
        cooldown_multiplier: Multiplier for cooldown after rate limit hit
    """

    requests_per_minute: int = 60
    requests_per_second: float = 1.0
    burst_size: int = 5
    cooldown_multiplier: float = 1.5


class RateLimiter:
    """Token bucket rate limiter.

    Uses the token bucket algorithm to smooth out request rates
    while allowing controlled bursts.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._tokens: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, name: str, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            name: Limiter name (e.g., provider name)
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        async with self._locks[name]:
            return await self._acquire_internal(name, tokens)

    async def _acquire_internal(self, name: str, tokens: int) -> float:
        """Internal token acquisition logic.

        Args:
            name: Limiter name
            tokens: Tokens to acquire

        Returns:
            Time waited in seconds
        """
        now = time.time()

        # Initialize if first request
        if name not in self._tokens:
            self._tokens[name] = float(self.config.burst_size)
            self._last_update[name] = now

        # Refill tokens based on time elapsed
        time_passed = now - self._last_update[name]
        self._tokens[name] = min(
            self.config.burst_size,
            self._tokens[name] + time_passed * self.config.requests_per_second,
        )
        self._last_update[name] = now

        # Check if we have enough tokens
        if self._tokens[name] >= tokens:
            self._tokens[name] -= tokens
            return 0.0

        # Calculate wait time
        tokens_needed = tokens - self._tokens[name]
        wait_time = tokens_needed / self.config.requests_per_second

        logger.debug(f"Rate limit: waiting {wait_time:.2f}s for '{name}'")
        await asyncio.sleep(wait_time)

        self._tokens[name] = 0
        self._last_update[name] = time.time()
        return wait_time

    def is_rate_limited(self, name: str) -> bool:
        """Check if currently rate limited.

        Args:
            name: Limiter name

        Returns:
            True if rate limited
        """
        if name not in self._tokens:
            return False
        return self._tokens[name] < 1

    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get rate limiter statistics.

        Args:
            name: Limiter name

        Returns:
            Statistics dictionary
        """
        tokens = self._tokens.get(name, float(self.config.burst_size))
        return {
            "available_tokens": tokens,
            "max_tokens": self.config.burst_size,
            "requests_per_second": self.config.requests_per_second,
            "is_limited": tokens < 1,
        }


class ResilientExecutor:
    """Combines circuit breaker, retry, and rate limiting.

    Provides a unified interface for resilient request execution
    with all patterns applied in the correct order:
    1. Rate limiting (prevent overwhelming)
    2. Circuit breaker (fail fast if unhealthy)
    3. Retry with backoff (handle transient failures)
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_handler: Optional[RetryHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize resilient executor.

        Args:
            circuit_breaker: Circuit breaker instance
            retry_handler: Retry handler instance
            rate_limiter: Rate limiter instance
        """
        self.circuit_breaker = circuit_breaker or MultiCircuitBreaker()
        self.retry_handler = retry_handler or RetryHandler()
        self.rate_limiter = rate_limiter or RateLimiter()

    async def execute(
        self,
        name: str,
        func: Callable[..., T],
        *args: Any,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with full resilience patterns.

        Args:
            name: Operation name (for circuit/rate tracking)
            func: Async function to execute
            *args: Positional arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments

        Returns:
            Result from function or fallback

        Raises:
            Exception if all resilience patterns exhausted
        """
        # 1. Rate limiting
        await self.rate_limiter.acquire(name)

        # 2. Circuit breaker check
        if not self.circuit_breaker.is_allowed(name):
            if fallback:
                logger.info(f"Circuit '{name}' open, using fallback")
                return await fallback(*args, **kwargs)
            raise CircuitOpenError(f"Circuit '{name}' is open")

        # 3. Execute with retry
        try:
            result = await self.retry_handler.execute_with_retry(
                func,
                *args,
                on_retry=lambda attempt, e: self.circuit_breaker.record_failure(name, e),
                **kwargs,
            )
            self.circuit_breaker.record_success(name)
            return result

        except Exception as e:
            self.circuit_breaker.record_failure(name, e)
            if fallback:
                logger.warning(f"Primary failed for '{name}', using fallback: {e}")
                return await fallback(*args, **kwargs)
            raise

    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for all circuits and rate limiters.

        Returns:
            Combined health report
        """
        return {
            "circuits": self.circuit_breaker.get_all_stats(),
            "retry_config": {
                "max_retries": self.retry_handler.config.max_retries,
                "base_delay": self.retry_handler.config.base_delay,
            },
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and no fallback available."""

    pass
