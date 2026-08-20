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

"""Unit tests for CircuitBreaker."""

import asyncio
import pytest

from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestSharedCircuitBreakerTypes:
    """Tests for shared core circuit-breaker type exports."""

    def test_provider_module_reexports_core_types(self) -> None:
        """providers.circuit_breaker should expose shared core types."""
        from victor.core.circuit_breaker import CircuitBreakerConfig as CoreConfig
        from victor.core.circuit_breaker import CircuitBreakerError as CoreError
        from victor.core.circuit_breaker import CircuitState as CoreState

        assert CircuitBreakerConfig is CoreConfig
        assert CircuitBreakerError is CoreError
        assert CircuitState is CoreState


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create a fresh circuit breaker for each test."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
            success_threshold=2,
            name="test_breaker",
        )

    def test_init_default_state(self, breaker: CircuitBreaker) -> None:
        """Test breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker: CircuitBreaker) -> None:
        """Test successful call passes through."""

        async def success_func() -> str:
            return "success"

        result = await breaker.execute(success_func)
        assert result == "success"
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_opens_after_failures(self, breaker: CircuitBreaker) -> None:
        """Test circuit opens after reaching failure threshold."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Fail up to threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, breaker: CircuitBreaker) -> None:
        """Test open circuit rejects calls immediately."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Now calls should be rejected
        async def should_not_run() -> str:
            return "should not reach here"

        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.execute(should_not_run)

        assert exc_info.value.state == CircuitState.OPEN
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker: CircuitBreaker) -> None:
        """Test circuit transitions to half-open after recovery timeout."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should now be half-open
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self, breaker: CircuitBreaker) -> None:
        """Test half-open circuit closes after successful calls."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        async def success_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful calls should close it (need success_threshold successes)
        await breaker.execute(success_func)
        await breaker.execute(success_func)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_opens_on_failure(self, breaker: CircuitBreaker) -> None:
        """Test half-open circuit reopens on failure."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self) -> None:
        """Test that excluded exceptions don't count as failures."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            excluded_exceptions=(ValueError,),
            name="test_excluded",
        )

        async def raise_value_error() -> None:
            raise ValueError("This is excluded")

        # These should not open the circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.execute(raise_value_error)

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_decorator_usage(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker as decorator."""
        call_count = 0

        @breaker
        async def decorated_func() -> str:
            nonlocal call_count
            call_count += 1
            return "decorated result"

        result = await decorated_func()
        assert result == "decorated result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_success(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker as async context manager."""
        async with breaker:
            pass  # Success
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, breaker: CircuitBreaker) -> None:
        """Test context manager records failure on exception."""
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("Simulated failure")

        assert breaker.is_open

    def test_reset(self, breaker: CircuitBreaker) -> None:
        """Test manual reset returns to closed state."""
        # Force open state by modifying internal counter
        breaker._failure_count = breaker.failure_threshold
        breaker._record_failure()
        assert breaker.is_open

        breaker.reset()
        assert breaker.is_closed

    def test_get_stats(self, breaker: CircuitBreaker) -> None:
        """Test statistics collection."""
        stats = breaker.get_stats()
        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        CircuitBreakerRegistry._breakers.clear()

    def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance."""
        breaker1 = CircuitBreakerRegistry.get_or_create("test", failure_threshold=5)
        breaker2 = CircuitBreakerRegistry.get_or_create("test", failure_threshold=10)

        assert breaker1 is breaker2
        assert breaker1.failure_threshold == 5  # First creation wins

    def test_get(self) -> None:
        """Test get returns existing breaker."""
        CircuitBreakerRegistry.get_or_create("existing")
        assert CircuitBreakerRegistry.get("existing") is not None
        assert CircuitBreakerRegistry.get("nonexistent") is None

    def test_reset_all(self) -> None:
        """Test reset_all resets all breakers."""
        breaker1 = CircuitBreakerRegistry.get_or_create("b1", failure_threshold=1)
        breaker2 = CircuitBreakerRegistry.get_or_create("b2", failure_threshold=1)

        # Open both
        breaker1._failure_count = 1
        breaker1._record_failure()
        breaker2._failure_count = 1
        breaker2._record_failure()

        CircuitBreakerRegistry.reset_all()

        assert breaker1.is_closed
        assert breaker2.is_closed

    def test_reset_all_normalizes_legacy_string_states(self) -> None:
        """reset_all should tolerate legacy string-valued breaker states."""
        breaker = CircuitBreakerRegistry.get_or_create("legacy")
        breaker._state = "OPEN"
        breaker._last_failure_time = None

        CircuitBreakerRegistry.reset_all()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    def test_get_all_stats(self) -> None:
        """Test getting stats from all breakers."""
        CircuitBreakerRegistry.get_or_create("stats1")
        CircuitBreakerRegistry.get_or_create("stats2")

        stats = CircuitBreakerRegistry.get_all_stats()
        assert "stats1" in stats
        assert "stats2" in stats


class TestCircuitBreakerEdgeCases:
    """Edge case tests for CircuitBreaker."""

    def test_should_attempt_recovery_no_last_failure(self) -> None:
        """Test recovery check when no failure recorded (covers line 135)."""
        breaker = CircuitBreaker(name="test_no_failure")
        # Set state to OPEN without recording last_failure_time
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = None
        # _should_attempt_recovery should return True
        assert breaker._should_attempt_recovery() is True

    @pytest.mark.asyncio
    async def test_half_open_max_calls_exceeded(self) -> None:
        """Test rejection when half_open_max_calls is exceeded (covers line 201)."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_calls=1,
            name="test_max_calls",
        )

        async def failing_func() -> None:
            raise ValueError("fail")

        async def success_func() -> str:
            await asyncio.sleep(0.1)  # Slow to keep half-open occupied
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)

        # Wait for recovery
        await asyncio.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # Start first call (will occupy the half-open slot)
        # We need to simulate the scenario where max calls is reached
        breaker._half_open_calls = breaker.half_open_max_calls

        # Now try another call - should be rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.execute(success_func)

        assert exc_info.value.state == CircuitState.HALF_OPEN
        assert exc_info.value.retry_after == 1.0

    def test_transition_to_same_state_no_change(self) -> None:
        """Test transition to same state doesn't add to state_changes."""
        breaker = CircuitBreaker(name="test_same_state")
        initial_changes = len(breaker._state_changes)
        breaker._transition_to(CircuitState.CLOSED)
        assert len(breaker._state_changes) == initial_changes

    @pytest.mark.asyncio
    async def test_execute_with_args_and_kwargs(self) -> None:
        """Test execute passes args and kwargs correctly."""
        breaker = CircuitBreaker(name="test_args")

        async def func_with_args(x: int, y: str = "default") -> str:
            return f"{x}-{y}"

        result = await breaker.execute(func_with_args, 42, y="custom")
        assert result == "42-custom"

    def test_success_in_closed_resets_failure_count(self) -> None:
        """Test success in closed state resets failure count."""
        breaker = CircuitBreaker(failure_threshold=5, name="test_reset")
        breaker._failure_count = 3
        breaker._record_success()
        assert breaker._failure_count == 0

    def test_state_property_normalizes_legacy_string_values(self) -> None:
        """Legacy string state values should be coerced to CircuitState enums."""
        breaker = CircuitBreaker(name="legacy_state")
        breaker._state = "HALF_OPEN"

        assert breaker.state == CircuitState.HALF_OPEN


class TestCircuitBreakerObservability:
    """Tests for circuit breaker observability callbacks."""

    def test_on_state_change_callback(self) -> None:
        """Test callback fires on state transition."""
        changes = []

        def on_change(old, new, name):
            changes.append((old, new, name))

        breaker = CircuitBreaker(
            failure_threshold=2,
            name="test_cb",
            on_state_change=on_change,
        )

        # Trigger CLOSED -> OPEN
        breaker._failure_count = 1
        breaker._record_failure()

        assert len(changes) == 1
        assert changes[0] == (CircuitState.CLOSED, CircuitState.OPEN, "test_cb")

    @pytest.mark.asyncio
    async def test_on_call_rejected_callback(self) -> None:
        """Test callback fires on call rejection."""
        rejections = []

        def on_rejected(name, retry_after):
            rejections.append((name, retry_after))

        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60.0,
            name="test_rej",
            on_call_rejected=on_rejected,
        )

        # Open the circuit
        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.execute(fail)

        # Now trigger rejection
        with pytest.raises(CircuitBreakerError):
            await breaker.execute(fail)

        assert len(rejections) == 1
        assert rejections[0][0] == "test_rej"
        assert rejections[0][1] > 0

    def test_callback_exception_does_not_propagate(self) -> None:
        """Test bad callback doesn't break breaker."""

        def bad_callback(*args):
            raise RuntimeError("callback error")

        breaker = CircuitBreaker(
            failure_threshold=2,
            name="test_bad_cb",
            on_state_change=bad_callback,
        )

        # Should not raise despite bad callback
        breaker._failure_count = 1
        breaker._record_failure()
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerRegistryObservability:
    """Tests for CircuitBreakerRegistry observability wiring."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        CircuitBreakerRegistry._breakers.clear()
        CircuitBreakerRegistry._observability_bus = None

    def test_wire_observability(self) -> None:
        """Test wire_observability sets callbacks on existing breakers."""
        breaker = CircuitBreakerRegistry.get_or_create("obs_test", failure_threshold=2)

        metrics = []

        class MockBus:
            def emit_metric(self, name, value, tags):
                metrics.append((name, value, tags))

        CircuitBreakerRegistry.wire_observability(MockBus())

        # Trigger state change
        breaker._failure_count = 1
        breaker._record_failure()

        assert len(metrics) == 1
        assert metrics[0][0] == "circuit_breaker.state_change"
        assert metrics[0][2]["breaker"] == "obs_test"

    def test_new_breakers_auto_wired(self) -> None:
        """Test breakers created after wiring also get callbacks."""
        metrics = []

        class MockBus:
            def emit_metric(self, name, value, tags):
                metrics.append((name, value, tags))

        CircuitBreakerRegistry.wire_observability(MockBus())

        # Create breaker AFTER wiring
        breaker = CircuitBreakerRegistry.get_or_create("auto_wired", failure_threshold=1)

        # Trigger state change
        breaker._record_failure()

        assert len(metrics) == 1
        assert metrics[0][0] == "circuit_breaker.state_change"
        assert metrics[0][2]["breaker"] == "auto_wired"
