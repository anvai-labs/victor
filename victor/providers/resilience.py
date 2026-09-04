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

"""
Error Recovery and Resilience System for LLM Providers.

This module provides robust error handling for production systems:
- Circuit breaker pattern for failure isolation
- Retry strategy with exponential backoff and jitter
- Fallback provider chain support
- Rate limit handling with Retry-After respect

Note: This module contains an embedded CircuitBreaker implementation optimized
for the ResilientProvider workflow. For standalone circuit breaker usage with
decorator/context manager support, see victor.providers.circuit_breaker.

Usage:
    from victor.providers.resilience import ResilientProvider, CircuitBreakerConfig

    # Wrap provider with resilience features
    resilient = ResilientProvider(
        provider=anthropic_provider,
        fallback_providers=[openai_provider],
    )

    # Use like normal provider
    response = await resilient.chat(messages, model=model)
"""

import asyncio
import contextlib
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    from victor.core.retry import RetryContext

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Import shared circuit-breaker types from core to avoid cross-layer coupling
from victor.core.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerError as CanonicalCircuitBreakerError,
)
from victor.core.context import bind_call_id_once, call_id

# Import the canonical CircuitBreaker for composition
from victor.providers import failure_taxonomy
from victor.providers.circuit_breaker import CircuitBreaker as CanonicalCircuitBreaker


# CircuitOpenError is a compatibility adapter for the canonical CircuitBreakerError
# It provides the CircuitOpenError API that existing code expects
class CircuitOpenError(CanonicalCircuitBreakerError):
    """Raised when circuit breaker is open.

    Compatibility adapter for CircuitBreakerError that provides the
    CircuitOpenError API (circuit_name, retry_after attributes).
    """

    def __init__(self, circuit_name: str, retry_after: Optional[float] = None):
        # Convert CircuitOpenError signature to CircuitBreakerError signature
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        message = f"Circuit breaker '{circuit_name}' is open"
        if retry_after is not None:
            message += f". Retry after {retry_after:.1f}s"
        # Call parent with state=OPEN and retry_after
        from victor.core.circuit_breaker import CircuitState

        super().__init__(message, state=CircuitState.OPEN, retry_after=retry_after or 0)


@dataclass
class CircuitBreakerState:
    """
    Runtime state of circuit breaker.

    DEPRECATED: This is kept for backward compatibility only.
    New code should use the canonical CircuitBreaker from victor.providers.circuit_breaker.
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    half_open_calls: int = 0
    consecutive_successes: int = 0


class ProviderCircuitBreaker:
    """
    Circuit breaker adapter for provider resilience workflow.

    This is now a thin wrapper around the canonical CircuitBreaker from
    victor.providers.circuit_breaker. It provides the ProviderCircuitBreaker
    API (is_available, time_until_retry, execute) for backward compatibility
    while delegating all state management to the canonical implementation.

    Design: Composition over reimplementation
    - Wraps CanonicalCircuitBreaker instead of duplicating logic
    - Provides is_available/time_until_retry properties that ResilientProvider expects
    - Maintains backward compatibility with existing code

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting requests immediately
    - HALF_OPEN: Testing if service recovered

    Usage:
        cb = ProviderCircuitBreaker("anthropic")

        async def make_request():
            return await provider.chat(messages, model=model)

        try:
            result = await cb.execute(make_request)
        except CircuitOpenError:
            # Circuit is open, use fallback
            pass
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # Create the canonical circuit breaker with our config
        self._breaker = CanonicalCircuitBreaker(
            name=name,
            config=self.config,
        )

        logger.debug(
            f"CircuitBreaker '{name}' initialized. "
            f"Threshold: {self.config.failure_threshold}, "
            f"Timeout: {self.config.timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._breaker.state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        # Access the internal state from the canonical breaker
        return self._breaker._failure_count

    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        # Delegate to canonical breaker's can_execute method
        return self._breaker.can_execute()

    @property
    def time_until_retry(self) -> Optional[float]:
        """Seconds until circuit might allow requests."""
        if self.state != CircuitState.OPEN:
            return None

        if self._breaker._last_failure_time:
            import time

            elapsed = time.time() - self._breaker._last_failure_time
            remaining = self._breaker.recovery_timeout - elapsed
            return max(0, remaining)

        return None

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.is_available:
            raise CircuitOpenError(self.name, self.time_until_retry)

        try:
            result = await func(*args, **kwargs)
            self._breaker.record_success()
            return result
        except Exception as e:
            self._breaker.record_failure(e)
            raise

    def reset(self):
        """Reset circuit breaker to initial state."""
        # Reset the canonical breaker's state
        self._breaker._state = CircuitState.CLOSED
        self._breaker._failure_count = 0
        self._breaker._success_count = 0
        self._breaker._last_failure_time = None
        self._breaker._last_exception = None
        self._breaker._half_open_calls = 0
        logger.info(f"CircuitBreaker '{self.name}' reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self._breaker._success_count,
            "is_available": self.is_available,
            "time_until_retry": self.time_until_retry,
            "last_failure_time": (
                datetime.fromtimestamp(self._breaker._last_failure_time).isoformat()
                if self._breaker._last_failure_time
                else None
            ),
        }


# Backward compatibility alias
CircuitBreaker = ProviderCircuitBreaker


@dataclass
class ProviderRetryConfig:
    """Configuration for provider retry strategy.

    Renamed from RetryConfig to be semantically distinct:
    - ProviderRetryConfig (here): Provider-specific with retryable_patterns
    - AgentRetryConfig (victor.agent.resilience): Agent-specific with jitter flag
    - ObservabilityRetryConfig (victor.observability.resilience): With BackoffStrategy

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay_seconds: Initial delay between retries
        max_delay_seconds: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter_factor: Random jitter factor (0.0 to 1.0)
        retryable_exceptions: Exception types that should be retried
        retryable_status_codes: HTTP status codes that should be retried
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1

    # Retryable error conditions
    # CircuitBreakerError/CircuitOpenError are deliberately NOT retryable:
    # an open circuit must fail fast into the fallback chain instead of being
    # slept through for retries × timeout (matches BaseProvider's own retry
    # set, which excludes them with the same fail-fast intent).
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        # httpx transport errors: RemoteProtocolError, ReadError, etc.
        # "Server disconnected without sending a response"
    )

    # Extended patterns that catch httpx transport errors by class name
    retryable_exception_names: tuple = failure_taxonomy.RETRYABLE_EXCEPTION_NAMES

    retryable_status_codes: tuple = (429, 500, 502, 503, 504)

    # Retryable error message patterns
    retryable_patterns: tuple = (
        r"connection.?error",
        r"connection.?reset",
        r"bad.?record.?mac",
        r"rate.?limit",
        r"overloaded",
        r"capacity",
        r"temporarily.?unavailable",
        r"server.?error",
        r"timeout",
    )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, max_retries: int, last_error: Exception):
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(f"Max retries ({max_retries}) exhausted. Last error: {last_error}")


class ProviderRetryStrategy:
    """
    Intelligent provider retry strategy with exponential backoff and jitter.

    This class now extends BaseRetryStrategy from victor.core.retry to
    eliminate duplicate retry logic. Provider-specific features (retryable
    patterns, status codes, Retry-After headers) are handled via adapter methods.

    Renamed from RetryStrategy to be semantically distinct:
    - ProviderRetryStrategy (here): Concrete provider retry with provider-specific logic
    - BaseRetryStrategy (victor.core.retry): Abstract base with should_retry(), get_delay()
    - BatchRetryStrategy (victor.workflows.batch_executor): Enum for batch retry modes

    Features:
    - Exponential backoff with configurable base
    - Random jitter to prevent thundering herd
    - Respects Retry-After headers from rate limit responses
    - Configurable retryable conditions (provider-specific patterns, status codes)
    - Integrates with core BaseRetryStrategy for consistent retry behavior

    Architecture:
        Uses BaseRetryStrategy internally for retry decision logic, while
        providing provider-specific enhancements through adapter methods.

    Usage:
        retry = ProviderRetryStrategy()

        async def make_request():
            return await provider.chat(messages, model=model)

        result = await retry.execute(make_request)
    """

    def __init__(self, config: Optional[ProviderRetryConfig] = None):
        """Initialize retry strategy.

        Args:
            config: Retry configuration
        """
        from victor.core.retry import (
            BaseRetryStrategy,
            ExponentialBackoffStrategy,
            RetryContext,
        )

        self.config = config or ProviderRetryConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.retryable_patterns
        ]

        # Use core BaseRetryStrategy for standard retry logic
        # ProviderRetryConfig maps to ExponentialBackoffStrategy parameters
        self._base_strategy = ExponentialBackoffStrategy(
            max_attempts=self.config.max_retries
            + 1,  # BaseRetryStrategy counts attempts, ProviderRetryStrategy counts retries
            base_delay=self.config.base_delay_seconds,
            max_delay=self.config.max_delay_seconds,
            multiplier=self.config.exponential_base,
            jitter=self.config.jitter_factor,
        )

        logger.debug(
            f"ProviderRetryStrategy initialized. "
            f"Max retries: {self.config.max_retries}, "
            f"Base delay: {self.config.base_delay_seconds}s"
        )

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        retry_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> T:
        """Execute function with retry logic.

        Uses core BaseRetryStrategy for retry decisions, with provider-specific
        enhancements for error detection and Retry-After header handling.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            retry_event_callback: Optional observer called before each retry
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            RetryExhaustedError: If all retries are exhausted
        """
        from victor.core.retry import RetryContext

        last_exception: Optional[Exception] = None
        context = RetryContext(max_attempts=self.config.max_retries + 1)

        while context.attempt < context.max_attempts:
            context.attempt += 1

            try:
                result = await func(*args, **kwargs)
                # Notify base strategy of success
                self._base_strategy.on_success(context)
                return result

            except Exception as e:
                last_exception = e
                context.record_exception(e)

                # Check provider-specific retryability (patterns, status codes, etc.)
                if not self._is_retryable(e):
                    logger.debug(f"Non-retryable error: {e}")
                    raise

                # Use base strategy to determine if we should retry
                if not self._base_strategy.should_retry(context):
                    self._base_strategy.on_failure(context)
                    raise RetryExhaustedError(context.attempt - 1, e)

                # Calculate delay using base strategy, with provider-specific override
                delay = self._get_delay(context, e)

                # Notify base strategy of retry
                self._base_strategy.on_retry(context)
                context.record_delay(delay)

                error_detail = str(e) or repr(e) or type(e).__name__
                retry_event = {
                    "attempt": context.attempt,
                    "max_retries": self.config.max_retries,
                    "delay_seconds": delay,
                    "error": error_detail,
                    "error_type": type(e).__name__,
                }
                if retry_event_callback is not None:
                    retry_event_callback(retry_event)
                logger.warning(
                    "Retry %d/%d after %.2fs. Error: %s",
                    context.attempt,
                    self.config.max_retries,
                    delay,
                    error_detail,
                )

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise RetryExhaustedError(context.attempt, last_exception)
        raise RuntimeError("Unexpected state in retry loop")

    def _get_delay(self, context: "RetryContext", error: Exception) -> float:
        """Calculate delay before next retry attempt.

        Uses provider-specific Retry-After header if available, otherwise
        delegates to base strategy's delay calculation.

        Args:
            context: Current retry context
            error: Exception that triggered retry

        Returns:
            Delay in seconds before next attempt
        """
        # Check for Retry-After header in error (provider-specific)
        retry_after = self._extract_retry_after(error)
        if retry_after is not None:
            return min(retry_after, self.config.max_delay_seconds)

        # Use base strategy's delay calculation
        return self._base_strategy.get_delay(context)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if error should be retried
        """
        if self._is_hard_rate_limit(error):
            return False

        seen: set[int] = set()
        current: Optional[BaseException] = error

        while isinstance(current, Exception) and id(current) not in seen:
            seen.add(id(current))

            # Circuit-breaker errors are never retryable via the FUZZY
            # classifiers below: an open circuit must fail fast into the
            # fallback chain. The message-substring and status-code checks
            # would otherwise re-admit them whenever the breaker NAME (which
            # embeds base_url) or the retry_after value contains a
            # retryable-looking number (e.g. localhost:5000), and the
            # cause-chain walk would re-admit them when the breaker error
            # was raised from a retryable transport error
            # (adversarial-review finding). An EXPLICIT opt-in — the config
            # listing the breaker error in retryable_exceptions — still
            # wins, preserving the retry-after-cooldown capability.
            if isinstance(current, (CanonicalCircuitBreakerError, CircuitOpenError)):
                return isinstance(current, self.config.retryable_exceptions)

            # Check exception type
            if isinstance(current, self.config.retryable_exceptions):
                return True

            # Check exception class name (catches httpx/openai transport errors
            # without requiring those libraries as direct dependencies here)
            error_class = type(current).__name__
            if hasattr(self.config, "retryable_exception_names"):
                for name in self.config.retryable_exception_names:
                    if error_class == name:
                        return True
                # Also check parent classes
                for parent in type(current).__mro__:
                    if parent.__name__ in self.config.retryable_exception_names:
                        return True

            # Check error message patterns
            error_str = str(current).lower()
            for pattern in self._compiled_patterns:
                if pattern.search(error_str):
                    return True

            # Check for status code in error
            for status_code in self.config.retryable_status_codes:
                if str(status_code) in error_str:
                    return True

            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

        return False

    def _is_hard_rate_limit(self, error: Exception) -> bool:
        """Check whether a 429 is a quota/billing failure rather than a transient limit."""
        seen: set[int] = set()
        current: Optional[BaseException] = error

        while isinstance(current, Exception) and id(current) not in seen:
            seen.add(id(current))

            status = (
                getattr(current, "status_code", None)
                or getattr(current, "code", None)
                or getattr(getattr(current, "response", None), "status_code", None)
            )
            if status == 429:
                error_text_parts = [str(current)]
                response = getattr(current, "response", None)
                response_text = getattr(response, "text", None)
                if isinstance(response_text, str) and response_text:
                    error_text_parts.append(response_text)
                error_text = " ".join(error_text_parts).lower()
                if any(token in error_text for token in failure_taxonomy.HARD_RATE_LIMIT_TOKENS):
                    return True

            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

        return False

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After value from error if present.

        Args:
            error: Exception to check

        Returns:
            Retry-After value in seconds, or None
        """
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after_header = headers.get("retry-after")
            if retry_after_header:
                try:
                    return float(retry_after_header)
                except ValueError:
                    pass

        error_str = str(error)

        # Look for "retry after X seconds" pattern
        patterns = [
            r"retry[- ]?after[:\s]+(\d+(?:\.\d+)?)",
            r"wait[:\s]+(\d+(?:\.\d+)?)\s*s",
            r"(\d+(?:\.\d+)?)\s*seconds?\s*(?:before|until)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None


class ProviderUnavailableError(Exception):
    """Raised when no providers are available."""

    def __init__(
        self,
        primary_error: Exception,
        fallback_errors: Optional[List[Exception]] = None,
    ):
        self.primary_error = primary_error
        self.fallback_errors = fallback_errors or []

        errors_summary = [f"Primary: {primary_error}"]
        for i, err in enumerate(self.fallback_errors):
            errors_summary.append(f"Fallback {i + 1}: {err}")

        super().__init__(f"All providers failed. {'; '.join(errors_summary)}")


class ResilientProvider:
    """
    Wrapper that adds resilience features to any provider.

    Features:
    - Circuit breaker per provider
    - Retry with exponential backoff
    - Fallback to alternative providers
    - Request timeout handling
    - Comprehensive error logging
    - ObservabilityBus integration for fallback notifications

    Usage:
        resilient = ResilientProvider(
            provider=anthropic_provider,
            fallback_providers=[openai_provider, ollama_provider],
        )

        response = await resilient.chat(messages, model=model)
    """

    _observability_bus: ClassVar[Optional[Any]] = None

    @classmethod
    def wire_observability(cls, bus: Any) -> None:
        """Wire an ObservabilityBus for provider fallback notifications.

        Follows the same pattern as ``CircuitBreakerRegistry.wire_observability``.
        When wired, lifecycle events are emitted on fallback activation and
        exhaustion so the dashboard and other subscribers gain visibility.

        Args:
            bus: ObservabilityBus instance with ``emit_lifecycle_event()`` method.
        """
        cls._observability_bus = bus

    def __init__(
        self,
        provider: Any,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[ProviderRetryConfig] = None,
        fallback_providers: Optional[List[Any]] = None,
        request_timeout: float = 120.0,
    ):
        """Initialize resilient provider.

        Args:
            provider: Primary provider instance
            circuit_config: Circuit breaker configuration
            retry_config: Retry strategy configuration
            fallback_providers: List of fallback providers
            request_timeout: Request timeout in seconds. This is the single
                source of truth for timeouts. Set this lower than SDK-level
                timeouts (typically 60s for cloud, 300s for local) so that
                asyncio.wait_for fires first with a clean TimeoutError.
        """
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.request_timeout = request_timeout

        # Get provider name
        self._provider_name = getattr(provider, "name", "unknown")

        # Create circuit breaker
        self.circuit_breaker = ProviderCircuitBreaker(
            name=f"cb_{self._provider_name}",
            config=circuit_config,
        )

        # Create retry strategy
        self.retry_strategy = ProviderRetryStrategy(config=retry_config)

        # Create circuit breakers for fallbacks
        self._fallback_circuits: Dict[str, ProviderCircuitBreaker] = {}
        for fb in self.fallback_providers:
            fb_name = getattr(fb, "name", f"fallback_{len(self._fallback_circuits)}")
            self._fallback_circuits[fb_name] = ProviderCircuitBreaker(
                name=f"cb_{fb_name}",
                config=circuit_config,
            )

        # Statistics
        self._stats = {
            "total_requests": 0,
            "primary_successes": 0,
            "fallback_successes": 0,
            "total_failures": 0,
            "retry_attempts": 0,
        }

        logger.info(
            f"ResilientProvider initialized for '{self._provider_name}'. "
            f"Fallbacks: {len(self.fallback_providers)}, "
            f"Timeout: {request_timeout}s"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return self._provider_name

    def supports_tools(self) -> bool:
        """Check if provider supports tools."""
        return getattr(self.provider, "supports_tools", lambda: False)()

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return getattr(self.provider, "supports_streaming", lambda: False)()

    async def chat(
        self,
        messages: List[Any],
        *,
        model: str,
        **kwargs,
    ) -> Any:
        """Execute chat with resilience features.

        One invocation of this method is ONE logical call for gateway metering
        (sandhi TD-0021 P4): the logical-call id is bound here — outermost
        binding wins — so every retry and every fallback of this call carries
        the same ``Idempotency-Key`` and the gateway's meter counts the call
        once (enforcement still counts every physical settlement). Only a
        plain coroutine gets this treatment — see ``stream`` for why an async
        generator cannot.

        Args:
            messages: List of messages
            model: Model identifier
            **kwargs: Additional arguments for provider

        Returns:
            Chat response

        Raises:
            ProviderUnavailableError: If all providers fail
        """
        try:
            token = bind_call_id_once(uuid.uuid4().hex)
        except Exception:  # pragma: no cover - os.urandom failure must not block transport
            token = None  # unbound: the transport mints per invocation instead
        try:
            return await self._chat_with_retry(messages, model=model, **kwargs)
        finally:
            if token is not None:
                call_id.reset(token)

    async def _chat_with_retry(
        self,
        messages: List[Any],
        *,
        model: str,
        **kwargs,
    ) -> Any:
        self._stats["total_requests"] += 1
        primary_retry_events: List[Dict[str, Any]] = []

        async def _execute_primary():
            return await asyncio.wait_for(
                self.provider.chat(messages, model=model, **kwargs),
                timeout=self.request_timeout,
            )

        # Try primary provider with circuit breaker and retry
        try:
            result = await self.circuit_breaker.execute(
                self.retry_strategy.execute,
                _execute_primary,
                retry_event_callback=primary_retry_events.append,
            )
            self._stats["retry_attempts"] += len(primary_retry_events)
            self._stats["primary_successes"] += 1
            self._attach_provider_retry_diagnostics(
                result,
                provider=self._provider_name,
                model=model,
                retry_events=primary_retry_events,
            )
            return result

        except CircuitOpenError as e:
            self._stats["retry_attempts"] += len(primary_retry_events)
            logger.warning(f"Primary provider circuit open: {e}")
            primary_error = e
        except Exception as e:
            self._stats["retry_attempts"] += len(primary_retry_events)
            logger.warning(f"Primary provider failed: {e}")
            primary_error = e

        # Try fallback providers
        fallback_errors: List[Exception] = []

        for fallback in self.fallback_providers:
            fb_name = getattr(fallback, "name", "unknown")
            fb_circuit = self._fallback_circuits.get(fb_name)

            try:
                logger.info(f"Trying fallback provider: {fb_name}")

                async def _execute_fallback(fb=fallback):
                    return await asyncio.wait_for(
                        fb.chat(messages, model=model, **kwargs),
                        timeout=self.request_timeout,
                    )

                if fb_circuit:
                    result = await fb_circuit.execute(_execute_fallback)
                else:
                    result = await _execute_fallback()

                self._stats["fallback_successes"] += 1
                logger.info(f"Fallback provider '{fb_name}' succeeded")
                self._attach_provider_retry_diagnostics(
                    result,
                    provider=self._provider_name,
                    model=model,
                    retry_events=primary_retry_events,
                )
                if self.__class__._observability_bus is not None:
                    self.__class__._observability_bus.emit_lifecycle_event(
                        "provider.fallback.activated",
                        {
                            "primary": self._provider_name,
                            "fallback": fb_name,
                            "error": str(primary_error),
                            "model": model,
                        },
                    )
                return result

            except CircuitOpenError as e:
                logger.warning(f"Fallback '{fb_name}' circuit open: {e}")
                fallback_errors.append(e)
            except Exception as e:
                logger.warning(f"Fallback provider '{fb_name}' failed: {e}")
                fallback_errors.append(e)
                continue

        # All providers failed
        self._stats["total_failures"] += 1
        if self.__class__._observability_bus is not None:
            self.__class__._observability_bus.emit_lifecycle_event(
                "provider.fallback.exhausted",
                {
                    "primary": self._provider_name,
                    "fallback_count": len(self.fallback_providers),
                    "error": str(primary_error),
                    "model": model,
                },
            )
        raise ProviderUnavailableError(primary_error, fallback_errors)

    @staticmethod
    def _attach_provider_retry_diagnostics(
        response: Any,
        *,
        provider: str,
        model: str,
        retry_events: List[Dict[str, Any]],
    ) -> None:
        """Attach retry diagnostics to response metadata when the response supports it."""
        if not retry_events or not hasattr(response, "metadata"):
            return

        metadata = getattr(response, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}

        diagnostics = metadata.get("provider_retry_diagnostics")
        if not isinstance(diagnostics, list):
            diagnostics = [diagnostics] if diagnostics else []

        diagnostics.append(
            {
                "provider": provider,
                "model": model,
                "retry_count": len(retry_events),
                "last_error": retry_events[-1].get("error", ""),
                "events": retry_events,
            }
        )
        metadata["provider_retry_diagnostics"] = diagnostics
        response.metadata = metadata

    async def stream(
        self,
        messages: List[Any],
        *,
        model: str,
        **kwargs,
    ):
        """Stream chat with resilience features.

        Deliberately does NOT bind the logical-call id, unlike ``chat``: this is
        an async GENERATOR, and a ``ContextVar.set()`` in a generator frame
        executes in the resumer's context — an early consumer break without
        ``aclosing`` would leak the binding into the caller's context (every
        later logical call in that task reusing a stale ``Idempotency-Key`` and
        being dropped by the gateway's dedup = silent under-metering), and the
        asyncgen finalizer resets the token from a different context
        (``ValueError``). The transport therefore mints a per-invocation key on
        the stream path; stream-setup retries are counted per attempt — the
        fail-toward-counting direction sandhi specifies (ADR-0005 D3).

        Note: Streaming has limited retry capability. If the stream fails
        midway, it cannot be resumed.

        Args:
            messages: List of messages
            model: Model identifier
            **kwargs: Additional arguments for provider

        Yields:
            Stream chunks
        """
        self._stats["total_requests"] += 1
        partial_content: list = []

        # For streaming, we only retry the initial connection
        async def _start_stream():
            return self.provider.stream(messages, model=model, **kwargs)

        try:
            stream = await self.circuit_breaker.execute(
                self.retry_strategy.execute,
                _start_stream,
            )

            # aclosing the inner stream is load-bearing: aclose() on THIS generator does not
            # cascade GeneratorExit into a sub-generator iterated via `async for`, so an early
            # consumer break would otherwise leave the underlying provider/httpx byte stream
            # suspended and finalized off-task by GC — the source of "async generator ignored
            # GeneratorExit" / "exit cancel scope in a different task" spam every turn.
            async with contextlib.aclosing(stream) as inner_stream:
                async for chunk in inner_stream:
                    partial_content.append(chunk)
                    yield chunk

            self._stats["primary_successes"] += 1

        except (CircuitOpenError, Exception) as primary_err:
            # Build augmented messages for fallback with partial content
            fallback_messages = list(messages)
            if partial_content:
                collected = "".join(getattr(c, "content", "") or "" for c in partial_content)
                if collected.strip():
                    fallback_messages.append(
                        {
                            "role": "assistant",
                            "content": collected,
                        }
                    )
                    logger.info(
                        "Passing %d chars of partial content to fallback",
                        len(collected),
                    )

            # Try fallbacks for streaming
            for fallback in self.fallback_providers:
                fb_name = getattr(fallback, "name", "unknown")
                try:
                    logger.info(f"Trying fallback stream: {fb_name}")
                    stream = fallback.stream(fallback_messages, model=model, **kwargs)
                    async with contextlib.aclosing(stream) as inner_stream:
                        async for chunk in inner_stream:
                            yield chunk
                    self._stats["fallback_successes"] += 1
                    if self.__class__._observability_bus is not None:
                        self.__class__._observability_bus.emit_lifecycle_event(
                            "provider.fallback.activated",
                            {
                                "primary": self._provider_name,
                                "fallback": fb_name,
                                "error": str(primary_err),
                                "model": model,
                                "streaming": True,
                            },
                        )
                    return
                except Exception as e:
                    logger.warning(f"Fallback stream '{fb_name}' failed: {e}")
                    continue

            self._stats["total_failures"] += 1
            if self.__class__._observability_bus is not None:
                self.__class__._observability_bus.emit_lifecycle_event(
                    "provider.fallback.exhausted",
                    {
                        "primary": self._provider_name,
                        "fallback_count": len(self.fallback_providers),
                        "error": str(primary_err),
                        "model": model,
                        "streaming": True,
                    },
                )
            raise primary_err

    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        return {
            **self._stats,
            "primary_circuit": self.circuit_breaker.get_stats(),
            "fallback_circuits": {
                name: cb.get_stats() for name, cb in self._fallback_circuits.items()
            },
            "success_rate": (
                (self._stats["primary_successes"] + self._stats["fallback_successes"])
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0
                else 0
            ),
        }

    def reset_circuits(self):
        """Reset all circuit breakers."""
        self.circuit_breaker.reset()
        for cb in self._fallback_circuits.values():
            cb.reset()
        logger.info("All circuit breakers reset")
