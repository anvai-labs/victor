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

"""Standalone circuit breaker pattern for provider resilience.

Implements the circuit breaker pattern to prevent cascading failures
when external LLM providers are unavailable or experiencing issues.

This module provides a standalone CircuitBreaker with decorator and context
manager support, plus a CircuitBreakerRegistry for managing multiple breakers.
Shared state/config/error types are defined in victor.core.circuit_breaker and
re-exported here for backward compatibility.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed

When to use this module:
- For decorator-based circuit breaking on individual functions/methods
- For managing circuit breakers across multiple services via CircuitBreakerRegistry
- For standalone circuit breaker usage outside of ResilientProvider

Related module:
- victor.providers.resilience: Contains an embedded CircuitBreaker optimized for
  the ResilientProvider workflow with retry strategies and fallback chains.
  Use ResilientProvider for a complete resilience solution that includes
  circuit breaking, retries, and fallback providers.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Optional, TypeVar, Awaitable
from functools import wraps

from victor.core.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    This is the canonical CircuitBreaker implementation for Victor.
    All modules should import from here instead of defining their own.

    Usage with parameters:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            name="my_service",
        )

    Usage with config:
        config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=30.0)
        breaker = CircuitBreaker.from_config("my_service", config)

    As decorator:
        @breaker
        async def call_provider():
            return await provider.chat(...)

    As context manager:
        async with breaker:
            result = await provider.chat(...)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        excluded_exceptions: Optional[tuple] = None,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable] = None,
        on_call_rejected: Optional[Callable] = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying recovery
            half_open_max_calls: Max concurrent calls in half-open state
            success_threshold: Successes needed in half-open to close circuit
            excluded_exceptions: Exceptions that don't count as failures
            name: Name for logging/identification
            config: Optional config object (overrides individual parameters)
            on_state_change: Callback(old_state, new_state, name) on state transitions
            on_call_rejected: Callback(name, retry_after) when calls are rejected
        """
        # If config provided, use its values
        if config is not None:
            failure_threshold = config.failure_threshold
            recovery_timeout = config.recovery_timeout
            half_open_max_calls = config.half_open_max_calls
            success_threshold = config.success_threshold

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.excluded_exceptions = excluded_exceptions or ()
        self.name = name

        # Observability callbacks
        self._on_state_change = on_state_change
        self._on_call_rejected = on_call_rejected

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_exception: Optional[Exception] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejected = 0
        self._state_changes: deque[tuple[float, CircuitState, CircuitState]] = deque(maxlen=100)

    @staticmethod
    def _coerce_state(state: Any) -> Any:
        """Normalize legacy string state values to CircuitState enums."""
        if isinstance(state, CircuitState):
            return state
        if isinstance(state, str):
            normalized = state.strip().lower()
            for candidate in CircuitState:
                if normalized in {candidate.value, candidate.name.lower()}:
                    return candidate
        return state

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic recovery."""
        coerced = self._coerce_state(self._state)
        if isinstance(coerced, CircuitState) and coerced is not self._state:
            self._state = coerced
        if self._state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if execution is allowed based on current state.

        This method is used by workflow executors to determine if a node
        should attempt execution through this circuit breaker.

        Returns:
            True if execution is allowed (CLOSED, or HALF_OPEN with slots)
            False if execution is blocked (OPEN, or HALF_OPEN at max calls)
        """
        state = self.state  # This may trigger OPEN -> HALF_OPEN

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # HALF_OPEN: check if we have available slots
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution.

        Call this after a successful operation to potentially
        transition from HALF_OPEN back to CLOSED state.
        """
        self._record_success()

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed execution.

        Call this after a failed operation to potentially
        transition to OPEN state.
        """
        self._record_failure(exception)

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._coerce_state(self._state)
        new_state = self._coerce_state(new_state)
        if not isinstance(old_state, CircuitState) or not isinstance(new_state, CircuitState):
            return
        self._state = old_state
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._state_changes.append((time.time(), old_state, new_state))
            logger.info(
                f"Circuit breaker '{self.name}' transitioned: "
                f"{old_state.value} -> {new_state.value}"
            )

            # Invoke observability callback
            if self._on_state_change is not None:
                try:
                    self._on_state_change(old_state, new_state, self.name)
                except Exception:
                    pass  # Callback errors must not break the breaker

            # Reset counters on state change
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        current_state = self.state
        if current_state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif current_state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed call."""
        self._total_failures += 1
        self._last_failure_time = time.time()
        self._last_exception = exception

        current_state = self.state
        if current_state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens circuit
            self._transition_to(CircuitState.OPEN)
        elif current_state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    async def _check_state(self) -> None:
        """Check if request should be allowed based on current state."""
        state = self.state  # This may trigger OPEN -> HALF_OPEN

        if state == CircuitState.OPEN:
            retry_after = self.recovery_timeout
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                retry_after = max(0, self.recovery_timeout - elapsed)

            self._total_rejected += 1

            # Invoke observability callback
            if self._on_call_rejected is not None:
                try:
                    self._on_call_rejected(self.name, retry_after)
                except Exception:
                    pass  # Callback errors must not break the breaker

            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. " f"Retry after {retry_after:.1f}s",
                state=state,
                retry_after=retry_after,
                last_error=self._last_exception,
            )

        if state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached",
                        state=state,
                        retry_after=1.0,
                        last_error=self._last_exception,
                    )
                self._half_open_calls += 1

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises and circuit stays closed
        """
        self._total_calls += 1
        await self._check_state()

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.excluded_exceptions:
            # These exceptions don't count as failures
            raise
        except Exception as exc:
            self._record_failure(exc)
            raise

    def __call__(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Decorator for wrapping async functions with circuit breaker."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def __aenter__(self) -> "CircuitBreaker":
        """Context manager entry - check state."""
        await self._check_state()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - record success/failure."""
        if exc_type is None:
            self._record_success()
        elif not isinstance(exc_val, self.excluded_exceptions):
            self._record_failure(exc_val)

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_exception = None
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejected": self._total_rejected,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "state_changes": len(self._state_changes),
        }

    @classmethod
    def from_config(
        cls,
        name: str,
        config: CircuitBreakerConfig,
        excluded_exceptions: Optional[tuple] = None,
    ) -> "CircuitBreaker":
        """Create CircuitBreaker from config object.

        Args:
            name: Name for logging/identification
            config: CircuitBreakerConfig with settings
            excluded_exceptions: Optional exceptions that don't count as failures

        Returns:
            Configured CircuitBreaker instance
        """
        return cls(
            name=name,
            config=config,
            excluded_exceptions=excluded_exceptions,
        )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _breakers: dict[str, CircuitBreaker] = {}
    _observability_bus: Optional[Any] = None

    @classmethod
    def get_or_create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            name: Unique name for the circuit breaker
            **kwargs: Arguments for CircuitBreaker if creating new

        Returns:
            CircuitBreaker instance
        """
        if name not in cls._breakers:
            breaker = CircuitBreaker(name=name, **kwargs)
            cls._breakers[name] = breaker
            # Auto-wire observability if bus is set
            if cls._observability_bus is not None:
                cls._wire_breaker(breaker)
        return cls._breakers[name]

    @classmethod
    def get(cls, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return cls._breakers.get(name)

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        for breaker in cls._breakers.values():
            breaker.reset()

    @classmethod
    def get_all_stats(cls) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in cls._breakers.items()}

    @classmethod
    def wire_observability(cls, bus: Any) -> None:
        """Wire an ObservabilityBus to all circuit breakers.

        Sets callbacks on all existing breakers and ensures new breakers
        created via get_or_create() are also wired automatically.

        Args:
            bus: ObservabilityBus instance with emit_metric(name, value, tags) method
        """
        cls._observability_bus = bus

        # Wire all existing breakers
        for breaker in cls._breakers.values():
            cls._wire_breaker(breaker)

    @classmethod
    def _wire_breaker(cls, breaker: CircuitBreaker) -> None:
        """Wire observability callbacks to a single breaker."""
        bus = cls._observability_bus
        if bus is None:
            return

        def on_state_change(old_state: CircuitState, new_state: CircuitState, name: str) -> None:
            bus.emit_metric(
                "circuit_breaker.state_change",
                1.0,
                {"breaker": name, "from": old_state.value, "to": new_state.value},
            )

        def on_call_rejected(name: str, retry_after: float) -> None:
            bus.emit_metric(
                "circuit_breaker.rejected",
                1.0,
                {"breaker": name, "retry_after": retry_after},
            )

        breaker._on_state_change = on_state_change
        breaker._on_call_rejected = on_call_rejected
