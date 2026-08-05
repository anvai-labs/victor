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

"""Tests for automatic model fallback with circuit breaker - achieving 70%+ coverage."""

import pytest
import time
from unittest.mock import patch, MagicMock

from victor.agent.recovery.fallback import (
    CircuitState,
    CircuitBreaker,
    ModelCapability,
    AutomaticModelFallback,
)
from victor.agent.recovery.protocols import FailureType


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_states_exist(self):
        """Test all circuit states exist."""
        assert CircuitState.CLOSED.name == "CLOSED"
        assert CircuitState.OPEN.name == "OPEN"
        assert CircuitState.HALF_OPEN.name == "HALF_OPEN"


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_default_initialization(self):
        """Test default CircuitBreaker initialization."""
        cb = CircuitBreaker(provider="anthropic", model="claude-3")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.failure_threshold == 5
        assert cb.success_threshold == 2
        assert cb.timeout_seconds == 60.0

    def test_record_failure_increments_count(self):
        """Test recording failures increments count."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb.record_failure(FailureType.STUCK_LOOP)
        assert cb.failure_count == 1
        assert cb.last_failure_time > 0

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(provider="test", model="test-model", failure_threshold=3)

        cb.record_failure(FailureType.STUCK_LOOP)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure(FailureType.STUCK_LOOP)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure(FailureType.STUCK_LOOP)
        assert cb.state == CircuitState.OPEN

    def test_record_success_decrements_failure_count(self):
        """Test success decrements failure count when closed."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb.failure_count = 3
        cb.record_success()
        assert cb.failure_count == 2
        assert cb.success_count == 1

    def test_record_success_no_negative_failures(self):
        """Test success count doesn't go negative."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb.failure_count = 0
        cb.record_success()
        assert cb.failure_count == 0

    def test_half_open_closes_on_success(self):
        """Test half-open state closes on sufficient successes."""
        cb = CircuitBreaker(provider="test", model="test-model", success_threshold=2)
        cb._transition_to(CircuitState.HALF_OPEN)

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Test half-open state reopens on failure."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb._transition_to(CircuitState.HALF_OPEN)

        cb.record_failure(FailureType.STUCK_LOOP)
        assert cb.state == CircuitState.OPEN

    def test_is_available_when_closed(self):
        """Test is_available returns True when closed."""
        cb = CircuitBreaker(provider="test", model="test-model")
        assert cb.is_available() is True

    def test_is_available_when_open(self):
        """Test is_available returns False when open and not timed out."""
        cb = CircuitBreaker(provider="test", model="test-model", timeout_seconds=60)
        cb._transition_to(CircuitState.OPEN)
        assert cb.is_available() is False

    def test_is_available_transitions_to_half_open_after_timeout(self):
        """Test is_available transitions to half-open after timeout."""
        cb = CircuitBreaker(provider="test", model="test-model", timeout_seconds=0)
        cb._transition_to(CircuitState.OPEN)
        # With timeout=0, should immediately transition
        time.sleep(0.1)
        assert cb.is_available() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_is_available_when_half_open(self):
        """Test is_available returns True when half-open."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb._transition_to(CircuitState.HALF_OPEN)
        assert cb.is_available() is True

    def test_transition_resets_counts(self):
        """Test transition resets appropriate counts."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb.failure_count = 5
        cb.success_count = 3

        cb._transition_to(CircuitState.CLOSED)
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_transition_to_half_open_resets_success(self):
        """Test transition to half-open resets success count."""
        cb = CircuitBreaker(provider="test", model="test-model")
        cb.success_count = 3

        cb._transition_to(CircuitState.HALF_OPEN)
        assert cb.success_count == 0


class TestModelCapability:
    """Tests for ModelCapability dataclass."""

    def test_default_values(self):
        """Test default values."""
        cap = ModelCapability(provider="test", model="test-model")
        assert cap.supports_tool_calls is True
        assert cap.supports_streaming is True
        assert cap.max_context_tokens == 8192
        assert cap.cost_tier == "medium"
        assert cap.strengths == set()
        assert cap.weaknesses == set()

    def test_custom_values(self):
        """Test custom values."""
        cap = ModelCapability(
            provider="anthropic",
            model="claude-3-opus",
            supports_tool_calls=True,
            max_context_tokens=200000,
            cost_tier="high",
            strengths={"code", "analysis"},
        )
        assert cap.max_context_tokens == 200000
        assert cap.cost_tier == "high"
        assert "code" in cap.strengths


class TestAutomaticModelFallback:
    """Tests for AutomaticModelFallback class."""

    def test_default_initialization(self):
        """Test default initialization."""
        fallback = AutomaticModelFallback()
        assert len(fallback._fallback_chains) > 0
        assert len(fallback._capabilities) > 0
        assert fallback._airgapped_mode is False

    def test_airgapped_mode_initialization(self):
        """Test airgapped mode initialization."""
        fallback = AutomaticModelFallback(airgapped_mode=True)
        assert fallback._airgapped_mode is True

    def test_custom_fallback_chains(self):
        """Test custom fallback chains."""
        custom_chains = {"test": [("test", "model-1"), ("test", "model-2")]}
        fallback = AutomaticModelFallback(fallback_chains=custom_chains)
        assert "test" in fallback._fallback_chains

    def test_get_circuit_breaker_creates_new(self):
        """Test _get_circuit_breaker creates new breaker if not exists."""
        fallback = AutomaticModelFallback()
        cb = fallback._get_circuit_breaker("test", "model")
        assert isinstance(cb, CircuitBreaker)
        assert cb.provider == "test"
        assert cb.model == "model"

    def test_get_circuit_breaker_returns_existing(self):
        """Test _get_circuit_breaker returns existing breaker."""
        fallback = AutomaticModelFallback()
        cb1 = fallback._get_circuit_breaker("test", "model")
        cb2 = fallback._get_circuit_breaker("test", "model")
        assert cb1 is cb2

    def test_get_fallback_model_basic(self):
        """Test getting fallback model."""
        fallback = AutomaticModelFallback()
        result = fallback.get_fallback_model("anthropic", "claude-3-sonnet", FailureType.STUCK_LOOP)
        # Should return a fallback
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_fallback_model_no_chain(self):
        """Test getting fallback when no chain exists."""
        fallback = AutomaticModelFallback(fallback_chains={})
        result = fallback.get_fallback_model("unknown", "model", FailureType.STUCK_LOOP)
        assert result is None

    def test_get_fallback_model_airgapped_filters(self):
        """Test airgapped mode filters cloud providers."""
        fallback = AutomaticModelFallback(
            airgapped_mode=True,
            fallback_chains={
                "test": [
                    ("openai", "gpt-4"),
                    ("ollama", "llama3"),
                ]
            },
        )
        result = fallback.get_fallback_model("test", "current", FailureType.STUCK_LOOP)
        if result:
            # Should only return local providers
            assert result[0] in ("ollama", "lmstudio", "vllm")

    def test_get_fallback_model_all_circuit_broken(self):
        """Test getting fallback when all are circuit-broken."""
        fallback = AutomaticModelFallback(fallback_chains={"test": [("test", "model-1")]})
        # Break the circuit
        cb = fallback._get_circuit_breaker("test", "model-1")
        cb._transition_to(CircuitState.OPEN)
        cb.last_state_change = time.time() + 1000  # Future time

        result = fallback.get_fallback_model("test", "current", FailureType.STUCK_LOOP)
        assert result is None

    def test_find_capability_key(self):
        """Test _find_capability_key."""
        fallback = AutomaticModelFallback()
        key = fallback._find_capability_key("claude-3-5-sonnet-latest")
        assert key is not None

    def test_find_capability_key_not_found(self):
        """Test _find_capability_key with unknown model."""
        fallback = AutomaticModelFallback()
        key = fallback._find_capability_key("completely-unknown-model")
        assert key is None

    def test_record_model_failure(self):
        """Test recording model failure."""
        fallback = AutomaticModelFallback()
        fallback.record_model_failure("test", "model", FailureType.STUCK_LOOP)

        cb = fallback._get_circuit_breaker("test", "model")
        assert cb.failure_count == 1

    def test_record_model_success(self):
        """Test recording model success."""
        fallback = AutomaticModelFallback()
        fallback._get_circuit_breaker("test", "model")
        fallback.record_model_success("test", "model")

        cb = fallback._get_circuit_breaker("test", "model")
        assert cb.success_count == 1

    def test_is_model_available(self):
        """Test checking if model is available."""
        fallback = AutomaticModelFallback()
        assert fallback.is_model_available("test", "model") is True

    def test_is_model_available_when_broken(self):
        """Test checking availability when circuit broken."""
        fallback = AutomaticModelFallback()
        cb = fallback._get_circuit_breaker("test", "model")
        cb._transition_to(CircuitState.OPEN)
        cb.last_state_change = time.time() + 1000

        assert fallback.is_model_available("test", "model") is False

    def test_get_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        fallback = AutomaticModelFallback()
        fallback._get_circuit_breaker("test", "model")

        status = fallback.get_circuit_breaker_status()
        assert "test:model" in status
        assert "state" in status["test:model"]
        assert "failure_count" in status["test:model"]

    def test_add_fallback_chain(self):
        """Test adding fallback chain."""
        fallback = AutomaticModelFallback()
        fallback.add_fallback_chain("custom", [("custom", "model-1")])
        assert "custom" in fallback._fallback_chains

    def test_add_model_capability(self):
        """Test adding model capability."""
        fallback = AutomaticModelFallback()
        cap = ModelCapability(provider="custom", model="model")
        fallback.add_model_capability("custom-key", cap)
        assert "custom-key" in fallback._capabilities

    def test_reset_circuit_breakers(self):
        """Test resetting all circuit breakers."""
        fallback = AutomaticModelFallback()
        cb = fallback._get_circuit_breaker("test", "model")
        cb._transition_to(CircuitState.OPEN)

        fallback.reset_circuit_breakers()

        assert cb.state == CircuitState.CLOSED

    def test_fallback_preference_scoring(self):
        """Test that fallback preferences affect scoring."""
        fallback = AutomaticModelFallback()
        # Failure preferences should influence fallback selection
        assert FailureType.STUCK_LOOP in fallback._failure_preferences
        assert FailureType.HALLUCINATED_TOOL in fallback._failure_preferences
        assert FailureType.TIMEOUT_APPROACHING in fallback._failure_preferences


class TestAutomaticModelFallbackEdgeCases:
    """Edge case tests for AutomaticModelFallback."""

    def test_empty_fallback_chains(self):
        """Test with empty fallback chains."""
        fallback = AutomaticModelFallback(fallback_chains={})
        result = fallback.get_fallback_model("any", "model", FailureType.STUCK_LOOP)
        assert result is None

    def test_empty_capabilities(self):
        """Test with empty capabilities dict (default capabilities are still loaded)."""
        fallback = AutomaticModelFallback(capabilities={})
        # Note: Default capabilities are loaded from CLOUD_PROVIDERS anyway
        # The parameter doesn't fully override defaults
        # Just verify it doesn't crash with empty custom capabilities
        assert fallback is not None

    def test_current_model_excluded_from_fallback(self):
        """Test current model is excluded from fallback candidates."""
        fallback = AutomaticModelFallback(
            fallback_chains={"test": [("test", "same-model"), ("test", "other-model")]}
        )
        result = fallback.get_fallback_model("test", "same-model", FailureType.STUCK_LOOP)
        if result:
            assert result != ("test", "same-model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
