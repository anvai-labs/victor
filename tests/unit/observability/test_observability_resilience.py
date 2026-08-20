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

"""Tests for observability resilience patterns module."""

import asyncio
import pytest

from victor.observability.resilience import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerError,
    ConstantBackoff,
    ExponentialBackoff,
    LinearBackoff,
    RateLimiter,
    ResiliencePolicy,
    ObservabilityRetryConfig,
    retry_with_backoff,
    with_timeout,
    TimeoutError,
)
from victor.core.circuit_breaker import CircuitState

# =============================================================================
# Backoff Strategy Tests
# =============================================================================


class TestExponentialBackoff:
    """Tests for ExponentialBackoff strategy."""

    def test_basic_exponential(self):
        """Test basic exponential backoff calculation."""
        strategy = ExponentialBackoff(multiplier=2.0, max_delay=60.0, jitter=0.0)

        assert strategy.calculate_delay(0, 1.0) == 1.0
        assert strategy.calculate_delay(1, 1.0) == 2.0
        assert strategy.calculate_delay(2, 1.0) == 4.0
        assert strategy.calculate_delay(3, 1.0) == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        strategy = ExponentialBackoff(multiplier=2.0, max_delay=10.0, jitter=0.0)

        assert strategy.calculate_delay(10, 1.0) == 10.0  # Capped

    def test_jitter_adds_variation(self):
        """Test that jitter adds variation to delays."""
        strategy = ExponentialBackoff(multiplier=2.0, max_delay=60.0, jitter=0.1)

        delays = [strategy.calculate_delay(2, 1.0) for _ in range(10)]

        # Should have some variation
        assert len(set(delays)) > 1


class TestLinearBackoff:
    """Tests for LinearBackoff strategy."""

    def test_linear_increase(self):
        """Test linear backoff increases linearly."""
        strategy = LinearBackoff(max_delay=60.0)

        assert strategy.calculate_delay(0, 1.0) == 1.0
        assert strategy.calculate_delay(1, 1.0) == 2.0
        assert strategy.calculate_delay(2, 1.0) == 3.0


class TestConstantBackoff:
    """Tests for ConstantBackoff strategy."""

    def test_constant_delay(self):
        """Test constant backoff returns same delay."""
        strategy = ConstantBackoff()

        assert strategy.calculate_delay(0, 5.0) == 5.0
        assert strategy.calculate_delay(5, 5.0) == 5.0
        assert strategy.calculate_delay(10, 5.0) == 5.0


# =============================================================================
# Retry Decorator Tests
# =============================================================================


class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_successful_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test exception raised after retries exhausted."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await always_fail()

        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retryable_exceptions_filter(self):
        """Test only specified exceptions trigger retry."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        async def specific_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            await specific_error()

        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is invoked."""
        retries = []

        def on_retry_cb(attempt, error, delay):
            retries.append((attempt, str(error), delay))

        @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=on_retry_cb)
        async def flaky():
            if len(retries) < 2:
                raise ValueError("Error")
            return "ok"

        await flaky()

        assert len(retries) == 2


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker pattern."""

    @pytest.mark.asyncio
    async def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        for i in range(3):
            try:
                async with breaker:
                    raise ValueError(f"Error {i}")
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count >= 3

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        # Trip the circuit
        try:
            async with breaker:
                raise ValueError("Trip")
        except ValueError:
            pass

        # Verify rejection
        with pytest.raises(CircuitBreakerError):
            async with breaker:
                pass

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test circuit enters half-open after recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            success_threshold=1,  # Close after just one success
            recovery_timeout=0.1,
        )

        # Trip the circuit
        try:
            async with breaker:
                raise ValueError("Trip")
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should allow test call (half-open) and close after success
        async with breaker:
            pass  # Success

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_closes_on_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            success_threshold=2,
            recovery_timeout=0.1,
        )

        # Trip
        try:
            async with breaker:
                raise ValueError("Trip")
        except ValueError:
            pass

        await asyncio.sleep(0.15)

        # Two successes needed
        async with breaker:
            pass
        async with breaker:
            pass

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker(failure_threshold=3)

        call_count = 0

        @breaker
        async def protected_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await protected_func()

        assert result == "success"
        assert call_count == 1

    def test_reset(self):
        """Test circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker._failure_count = 10
        breaker._state = CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_get_metrics(self):
        """Test metrics collection."""
        breaker = CircuitBreaker(failure_threshold=5, name="test_breaker")

        metrics = breaker.get_metrics()

        assert metrics["name"] == "test_breaker"
        assert metrics["state"] == "closed"
        assert metrics["failure_threshold"] == 5

    @pytest.mark.asyncio
    async def test_state_change_callback(self):
        """Test state change callback is invoked."""
        changes = []

        def on_change(old, new):
            changes.append((old.value, new.value))

        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            on_state_change=on_change,
        )

        # Trip
        try:
            async with breaker:
                raise ValueError("Trip")
        except ValueError:
            pass

        assert ("closed", "open") in changes


# =============================================================================
# Bulkhead Tests
# =============================================================================


class TestBulkhead:
    """Tests for Bulkhead pattern."""

    @pytest.mark.asyncio
    async def test_limits_concurrent_access(self):
        """Test bulkhead limits concurrent executions."""
        bulkhead = Bulkhead(max_concurrent=2)
        concurrent = 0
        max_concurrent = 0

        async def worker():
            nonlocal concurrent, max_concurrent
            async with bulkhead:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
                await asyncio.sleep(0.05)
                concurrent -= 1

        await asyncio.gather(*[worker() for _ in range(5)])

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_acquire_with_timeout(self):
        """Test acquire with timeout."""
        bulkhead = Bulkhead(max_concurrent=1)

        # Acquire first slot
        async with bulkhead:
            # Second acquire should timeout
            with pytest.raises(BulkheadFullError):
                ctx = await bulkhead.acquire(timeout=0.1)
                async with ctx:
                    pass

    @pytest.mark.asyncio
    async def test_available_slots(self):
        """Test available slot tracking."""
        bulkhead = Bulkhead(max_concurrent=3)

        assert bulkhead.available == 3

        async with bulkhead:
            assert bulkhead.available == 2
            assert bulkhead.active_count == 1

        assert bulkhead.available == 3

    def test_get_metrics(self):
        """Test metrics collection."""
        bulkhead = Bulkhead(max_concurrent=5, name="test_bulkhead")

        metrics = bulkhead.get_metrics()

        assert metrics["name"] == "test_bulkhead"
        assert metrics["max_concurrent"] == 5


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter pattern."""

    @pytest.mark.asyncio
    async def test_allows_within_rate(self):
        """Test requests within rate are allowed."""
        limiter = RateLimiter(rate=10, capacity=10)

        results = [await limiter.acquire() for _ in range(5)]

        assert all(results)

    @pytest.mark.asyncio
    async def test_blocks_over_capacity(self):
        """Test requests over capacity are blocked."""
        limiter = RateLimiter(rate=10, capacity=2)

        # Exhaust capacity
        await limiter.acquire()
        await limiter.acquire()

        # Should fail
        result = await limiter.acquire()

        assert not result

    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        """Test tokens refill over time."""
        limiter = RateLimiter(rate=100, capacity=1)

        # Use token
        await limiter.acquire()

        # Should fail immediately
        assert not await limiter.acquire()

        # Wait for refill
        await asyncio.sleep(0.02)

        # Should work now
        assert await limiter.acquire()


# =============================================================================
# Timeout Decorator Tests
# =============================================================================


class TestTimeoutDecorator:
    """Tests for with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Test function completing within timeout."""

        @with_timeout(1.0)
        async def fast_func():
            return "fast"

        result = await fast_func()

        assert result == "fast"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Test timeout exception is raised."""

        @with_timeout(0.1)
        async def slow_func():
            await asyncio.sleep(1.0)
            return "slow"

        with pytest.raises(TimeoutError):
            await slow_func()


# =============================================================================
# Resilience Policy Tests
# =============================================================================


class TestResiliencePolicy:
    """Tests for composite ResiliencePolicy."""

    @pytest.mark.asyncio
    async def test_combined_policies(self):
        """Test combined resilience policies."""
        policy = ResiliencePolicy(
            circuit_breaker=CircuitBreaker(failure_threshold=5),
            retry_config=ObservabilityRetryConfig(max_retries=2, base_delay=0.01),
            timeout=1.0,
        )

        @policy
        async def protected():
            return "success"

        result = await protected()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_within_circuit_breaker(self):
        """Test retry works within circuit breaker."""
        call_count = 0

        policy = ResiliencePolicy(
            circuit_breaker=CircuitBreaker(failure_threshold=10),
            retry_config=ObservabilityRetryConfig(max_retries=2, base_delay=0.01),
        )

        @policy
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Flaky")
            return "ok"

        result = await flaky()

        assert result == "ok"
        assert call_count == 3

    def test_get_metrics(self):
        """Test combined metrics."""
        policy = ResiliencePolicy(
            circuit_breaker=CircuitBreaker(failure_threshold=5, name="cb"),
            bulkhead=Bulkhead(max_concurrent=10, name="bh"),
            rate_limiter=RateLimiter(rate=100, name="rl"),
            name="test_policy",
        )

        metrics = policy.get_metrics()

        assert "circuit_breaker" in metrics
        assert "bulkhead" in metrics
        assert "rate_limiter" in metrics
