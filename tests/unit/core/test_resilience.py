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

"""Tests for resilience patterns (circuit breaker, retry, rate limiter)."""

import asyncio
import time

import pytest

from victor.agent.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    RateLimiter,
    RateLimitConfig,
    ResilientExecutor,
    AgentRetryConfig,
    RetryHandler,
)
from victor.core.circuit_breaker import CircuitState


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Circuit should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.get_state("test") == CircuitState.CLOSED
        assert breaker.is_allowed("test") is True

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit should open after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        # Record failures
        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.CLOSED

        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.CLOSED

        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.OPEN
        assert breaker.is_allowed("test") is False

    def test_success_resets_failure_count(self):
        """Success should reset consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        breaker.record_failure("test")
        breaker.record_failure("test")
        breaker.record_success("test")

        # Failure count should be reset
        breaker.record_failure("test")
        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.CLOSED

    def test_circuit_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure("test")
        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to HALF_OPEN
        assert breaker.get_state("test") == CircuitState.HALF_OPEN
        assert breaker.is_allowed("test") is True

    def test_half_open_closes_on_success(self):
        """Circuit should close after successful test calls in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, success_threshold=2
        )
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure("test")
        breaker.record_failure("test")

        # Wait for recovery
        time.sleep(0.15)
        breaker.get_state("test")  # Trigger transition

        # Successful test calls
        breaker.record_success("test")
        assert breaker.get_state("test") == CircuitState.HALF_OPEN

        breaker.record_success("test")
        assert breaker.get_state("test") == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Circuit should reopen on failure during HALF_OPEN."""
        # Use longer timeout for reliable testing under parallel load
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05)
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure("test")
        breaker.record_failure("test")

        # Wait for recovery (3x timeout for reliability)
        time.sleep(0.20)
        breaker.get_state("test")  # Trigger transition to HALF_OPEN

        # Failure during test
        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.OPEN

    def test_multiple_circuits_are_independent(self):
        """Different circuits should be independent."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        # Open circuit A
        breaker.record_failure("A")
        breaker.record_failure("A")
        assert breaker.get_state("A") == CircuitState.OPEN

        # Circuit B should still be closed
        assert breaker.get_state("B") == CircuitState.CLOSED
        assert breaker.is_allowed("B") is True

    def test_reset_clears_circuit_state(self):
        """Reset should return circuit to initial state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure("test")
        breaker.record_failure("test")
        assert breaker.get_state("test") == CircuitState.OPEN

        # Reset
        breaker.reset("test")
        assert breaker.get_state("test") == CircuitState.CLOSED
        assert breaker.is_allowed("test") is True

    def test_get_stats_returns_circuit_info(self):
        """get_stats should return circuit statistics."""
        breaker = CircuitBreaker()
        breaker.record_success("test")
        breaker.record_failure("test")

        stats = breaker.get_stats("test")

        assert "state" in stats
        assert "failures" in stats
        assert "successes" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats
        assert stats["total_failures"] == 1
        assert stats["total_successes"] == 1

    def test_excluded_exceptions_not_counted(self):
        """Excluded exceptions should not count as failures."""
        config = CircuitBreakerConfig(failure_threshold=2, exclude_exceptions=(ValueError,))
        breaker = CircuitBreaker(config)

        # Record excluded exception
        breaker.record_failure("test", ValueError("test"))
        breaker.record_failure("test", ValueError("test"))

        # Circuit should still be closed
        assert breaker.get_state("test") == CircuitState.CLOSED


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_calculate_delay_exponential_backoff(self):
        """Delay should increase exponentially."""
        config = AgentRetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        handler = RetryHandler(config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 2.0
        assert handler.calculate_delay(2) == 4.0
        assert handler.calculate_delay(3) == 8.0

    def test_calculate_delay_respects_max(self):
        """Delay should not exceed max_delay."""
        config = AgentRetryConfig(base_delay=10.0, max_delay=15.0, jitter=False)
        handler = RetryHandler(config)

        assert handler.calculate_delay(5) == 15.0

    def test_should_retry_respects_max_retries(self):
        """Should not retry beyond max_retries."""
        config = AgentRetryConfig(max_retries=3)
        handler = RetryHandler(config)

        assert handler.should_retry(0, error=ConnectionError()) is True
        assert handler.should_retry(2, error=ConnectionError()) is True
        assert handler.should_retry(3, error=ConnectionError()) is False

    def test_should_retry_only_retryable_exceptions(self):
        """Should only retry on retryable exceptions."""
        config = AgentRetryConfig(retryable_exceptions=(ConnectionError,))
        handler = RetryHandler(config)

        assert handler.should_retry(0, error=ConnectionError()) is True
        assert handler.should_retry(0, error=ValueError()) is False

    def test_should_retry_on_status_codes(self):
        """Should retry on retryable status codes."""
        config = AgentRetryConfig(retryable_status_codes=(429, 500))
        handler = RetryHandler(config)

        assert handler.should_retry(0, status_code=429) is True
        assert handler.should_retry(0, status_code=500) is True
        assert handler.should_retry(0, status_code=400) is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Should succeed on first try if no error."""
        handler = RetryHandler()

        async def success_func():
            return "success"

        result = await handler.execute_with_retry(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventually_succeeds(self):
        """Should retry and eventually succeed."""
        config = AgentRetryConfig(
            max_retries=3,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions=(ConnectionError,),
        )
        handler = RetryHandler(config)

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await handler.execute_with_retry(flaky_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Should raise after exhausting retries."""
        config = AgentRetryConfig(
            max_retries=2,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions=(ConnectionError,),
        )
        handler = RetryHandler(config)

        async def always_fail():
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            await handler.execute_with_retry(always_fail)


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_returns_immediately_when_tokens_available(self):
        """Should return immediately when tokens are available."""
        config = RateLimitConfig(burst_size=5, requests_per_second=10.0)
        limiter = RateLimiter(config)

        wait_time = await limiter.acquire("test")
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_waits_when_tokens_exhausted(self):
        """Should wait when tokens are exhausted."""
        config = RateLimitConfig(burst_size=2, requests_per_second=100.0)
        limiter = RateLimiter(config)

        # Exhaust tokens
        await limiter.acquire("test", tokens=2)

        # Next request should wait
        start = time.time()
        await limiter.acquire("test")
        elapsed = time.time() - start

        # Should have waited for at least one token refill
        assert elapsed >= 0.005  # At least 5ms

    def test_is_rate_limited(self):
        """Should detect when rate limited."""
        config = RateLimitConfig(burst_size=1, requests_per_second=1.0)
        limiter = RateLimiter(config)

        # Initially not limited
        assert limiter.is_rate_limited("test") is False

    def test_get_stats(self):
        """Should return limiter statistics."""
        config = RateLimitConfig(burst_size=5, requests_per_second=10.0)
        limiter = RateLimiter(config)

        stats = limiter.get_stats("test")

        assert "available_tokens" in stats
        assert "max_tokens" in stats
        assert "requests_per_second" in stats
        assert stats["max_tokens"] == 5
        assert stats["requests_per_second"] == 10.0


class TestResilientExecutor:
    """Tests for ResilientExecutor."""

    @pytest.mark.asyncio
    async def test_execute_success_path(self):
        """Should execute successfully on happy path."""
        executor = ResilientExecutor()

        async def success_func():
            return "result"

        result = await executor.execute("test", success_func)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_records_success_in_circuit(self):
        """Successful execution should record success in circuit."""
        executor = ResilientExecutor()

        async def success_func():
            return "result"

        await executor.execute("test", success_func)

        stats = executor.circuit_breaker.get_stats("test")
        assert stats["total_successes"] == 1

    @pytest.mark.asyncio
    async def test_execute_uses_fallback_when_circuit_open(self):
        """Should use fallback when circuit is open."""
        circuit = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        circuit.record_failure("test")  # Open circuit

        executor = ResilientExecutor(circuit_breaker=circuit)

        async def primary():
            return "primary"

        async def fallback():
            return "fallback"

        result = await executor.execute("test", primary, fallback=fallback)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_execute_raises_circuit_open_error_without_fallback(self):
        """Should raise CircuitOpenError when circuit open and no fallback."""
        circuit = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        circuit.record_failure("test")  # Open circuit

        executor = ResilientExecutor(circuit_breaker=circuit)

        async def primary():
            return "primary"

        with pytest.raises(CircuitOpenError):
            await executor.execute("test", primary)

    @pytest.mark.asyncio
    async def test_execute_uses_fallback_on_failure(self):
        """Should use fallback when primary fails and retries exhausted."""
        retry = RetryHandler(
            AgentRetryConfig(max_retries=1, base_delay=0.01, retryable_exceptions=(ValueError,))
        )
        executor = ResilientExecutor(retry_handler=retry)

        async def primary():
            raise ValueError("fail")

        async def fallback():
            return "fallback"

        result = await executor.execute("test", primary, fallback=fallback)
        assert result == "fallback"

    def test_get_health_report(self):
        """Should return combined health report."""
        executor = ResilientExecutor()

        report = executor.get_health_report()

        assert "circuits" in report
        assert "retry_config" in report
