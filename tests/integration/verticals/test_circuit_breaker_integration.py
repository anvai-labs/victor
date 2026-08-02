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

"""Integration tests for CircuitBreaker with providers."""

import pytest
from unittest.mock import MagicMock

from victor.providers.base import BaseProvider, Message, CompletionResponse
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)


class MockProvider(BaseProvider):
    """Mock provider for testing circuit breaker integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_response = None
        self._should_fail = False
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def set_response(self, response):
        """Set the mock response to return."""
        self._mock_response = response

    def set_should_fail(self, fail: bool):
        """Set whether calls should fail."""
        self._should_fail = fail

    async def chat(self, messages, *, model, **kwargs):
        """Mock chat that uses circuit breaker."""

        async def _actual_chat():
            self._call_count += 1
            if self._should_fail:
                raise ValueError("Simulated provider failure")
            return self._mock_response or CompletionResponse(content="Mock response")

        return await self._execute_with_circuit_breaker(_actual_chat)

    async def stream(self, messages, *, model, **kwargs):
        """Mock stream."""
        yield MagicMock()

    async def close(self):
        """Close mock provider."""
        pass


@pytest.fixture(autouse=True)
def clear_circuit_breaker_registry():
    """Clear circuit breaker registry before each test."""
    CircuitBreakerRegistry._breakers.clear()
    yield
    CircuitBreakerRegistry._breakers.clear()


class TestProviderCircuitBreakerIntegration:
    """Tests for circuit breaker integration with providers."""

    def test_provider_has_circuit_breaker(self):
        """Test that provider initializes with circuit breaker."""
        provider = MockProvider(use_circuit_breaker=True)
        assert provider.circuit_breaker is not None
        assert isinstance(provider.circuit_breaker, CircuitBreaker)

    def test_provider_without_circuit_breaker(self):
        """Test provider can be created without circuit breaker."""
        provider = MockProvider(use_circuit_breaker=False)
        assert provider.circuit_breaker is None

    def test_circuit_breaker_registered(self):
        """Test that circuit breaker is registered in global registry."""
        provider = MockProvider(use_circuit_breaker=True)
        breaker = CircuitBreakerRegistry.get(provider.circuit_breaker.name)
        assert breaker is provider.circuit_breaker

    @pytest.mark.asyncio
    async def test_successful_call_through_circuit_breaker(self):
        """Test successful call passes through circuit breaker."""
        provider = MockProvider(use_circuit_breaker=True)
        provider.set_response(CompletionResponse(content="Success"))

        messages = [Message(role="user", content="Hello")]
        response = await provider.chat(messages, model="test")

        assert response.content == "Success"
        assert provider.circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        provider = MockProvider(
            use_circuit_breaker=True,
            circuit_breaker_failure_threshold=3,
        )
        provider.set_should_fail(True)

        messages = [Message(role="user", content="Hello")]

        # Fail up to threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                await provider.chat(messages, model="test")

        # Circuit should now be open
        assert provider.circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self):
        """Test that open circuit rejects calls immediately."""
        provider = MockProvider(
            use_circuit_breaker=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=60.0,  # Long timeout
        )
        provider.set_should_fail(True)

        messages = [Message(role="user", content="Hello")]

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await provider.chat(messages, model="test")

        assert provider.circuit_breaker.is_open
        call_count_before = provider._call_count

        # This call should be rejected without calling the actual method
        with pytest.raises(CircuitBreakerError) as exc_info:
            await provider.chat(messages, model="test")

        assert exc_info.value.state == CircuitState.OPEN
        assert provider._call_count == call_count_before  # No new call made

    def test_is_circuit_open_method(self):
        """Test is_circuit_open helper method."""
        provider = MockProvider(use_circuit_breaker=True)
        assert not provider.is_circuit_open()

        # Force open
        provider.circuit_breaker._failure_count = provider.circuit_breaker.failure_threshold
        provider.circuit_breaker._record_failure()

        assert provider.is_circuit_open()

    def test_reset_circuit_breaker(self):
        """Test manual circuit breaker reset."""
        provider = MockProvider(use_circuit_breaker=True)

        # Force open
        provider.circuit_breaker._failure_count = provider.circuit_breaker.failure_threshold
        provider.circuit_breaker._record_failure()
        assert provider.is_circuit_open()

        # Reset
        provider.reset_circuit_breaker()
        assert not provider.is_circuit_open()

    def test_get_circuit_breaker_stats(self):
        """Test getting circuit breaker statistics."""
        provider = MockProvider(use_circuit_breaker=True)
        stats = provider.get_circuit_breaker_stats()

        assert stats is not None
        assert "name" in stats
        assert "state" in stats
        assert stats["state"] == "closed"

    def test_provider_without_breaker_returns_no_stats(self):
        """Test that provider without breaker returns None for stats."""
        provider = MockProvider(use_circuit_breaker=False)
        assert provider.get_circuit_breaker_stats() is None


class TestCircuitBreakerScopedPerInstance:
    """Tests for circuit breaker isolation across provider instances."""

    def test_same_provider_type_gets_distinct_breakers(self):
        """Fresh provider instances should not inherit breaker state."""
        provider1 = MockProvider(use_circuit_breaker=True)
        provider2 = MockProvider(use_circuit_breaker=True)

        assert provider1.circuit_breaker is not provider2.circuit_breaker
        assert provider1.circuit_breaker.name != provider2.circuit_breaker.name

    @pytest.mark.asyncio
    async def test_failure_in_one_does_not_affect_other(self):
        """Failures in one provider instance should not poison another."""
        provider1 = MockProvider(
            use_circuit_breaker=True,
            circuit_breaker_failure_threshold=2,
        )
        provider2 = MockProvider(use_circuit_breaker=True)
        provider1.set_should_fail(True)

        messages = [Message(role="user", content="Hello")]

        # Fail through provider1
        for _ in range(2):
            with pytest.raises(ValueError):
                await provider1.chat(messages, model="test")

        assert provider1.is_circuit_open()
        assert not provider2.is_circuit_open()
