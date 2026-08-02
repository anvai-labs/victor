# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for workflow resilience facade."""

import pytest

from victor.workflows.resilience import (
    # Re-exported from core
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    RetryExecutor,
    RetryResult,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    # Workflow-specific adapters
    retry_policy_to_strategy,
    node_retry_strategy,
    get_node_circuit_breaker,
)
from victor.workflows.protocols import RetryPolicy


class TestResilienceFacadeExports:
    """Test that facade exports all required components."""

    def test_circuit_breaker_exported(self):
        """CircuitBreaker should be accessible from facade."""
        assert CircuitBreaker is not None
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_registry_exported(self):
        """CircuitBreakerRegistry should be accessible from facade."""
        assert CircuitBreakerRegistry is not None
        # Reset to ensure clean state
        CircuitBreakerRegistry.reset_all()
        breaker = CircuitBreakerRegistry.get_or_create("test_node")
        assert breaker is not None

    def test_circuit_state_exported(self):
        """CircuitState should be accessible from facade."""
        assert CircuitState.CLOSED is not None
        assert CircuitState.OPEN is not None
        assert CircuitState.HALF_OPEN is not None

    def test_retry_executor_exported(self):
        """RetryExecutor should be accessible from facade."""
        assert RetryExecutor is not None

    def test_retry_strategies_exported(self):
        """Retry strategies should be accessible from facade."""
        assert ExponentialBackoffStrategy is not None
        assert FixedDelayStrategy is not None


class TestRetryPolicyAdapter:
    """Test retry_policy_to_strategy adapter function."""

    def test_exponential_backoff_policy(self):
        """Policy with exponential_backoff=True should create ExponentialBackoffStrategy."""
        policy = RetryPolicy(
            max_retries=3,
            delay_seconds=1.0,
            exponential_backoff=True,
        )
        strategy = retry_policy_to_strategy(policy)
        assert isinstance(strategy, ExponentialBackoffStrategy)

    def test_fixed_delay_policy(self):
        """Policy with exponential_backoff=False should create FixedDelayStrategy."""
        policy = RetryPolicy(
            max_retries=3,
            delay_seconds=2.0,
            exponential_backoff=False,
        )
        strategy = retry_policy_to_strategy(policy)
        assert isinstance(strategy, FixedDelayStrategy)

    def test_max_attempts_conversion(self):
        """max_retries should be converted to max_attempts correctly."""
        policy = RetryPolicy(max_retries=5, delay_seconds=1.0)
        strategy = retry_policy_to_strategy(policy)
        # max_attempts = max_retries + 1 (includes initial attempt)
        assert strategy.max_attempts == 6

    def test_delay_seconds_preserved(self):
        """delay_seconds should be preserved in strategy."""
        policy = RetryPolicy(max_retries=3, delay_seconds=2.5, exponential_backoff=False)
        strategy = retry_policy_to_strategy(policy)
        assert strategy.delay == 2.5


class TestNodeRetryStrategy:
    """Test node_retry_strategy helper function."""

    def test_default_creates_exponential(self):
        """Default should create exponential backoff strategy."""
        strategy = node_retry_strategy()
        assert isinstance(strategy, ExponentialBackoffStrategy)

    def test_exponential_true(self):
        """exponential=True should create ExponentialBackoffStrategy."""
        strategy = node_retry_strategy(max_retries=3, delay_seconds=1.0, exponential=True)
        assert isinstance(strategy, ExponentialBackoffStrategy)

    def test_exponential_false(self):
        """exponential=False should create FixedDelayStrategy."""
        strategy = node_retry_strategy(max_retries=3, delay_seconds=1.0, exponential=False)
        assert isinstance(strategy, FixedDelayStrategy)

    def test_custom_max_retries(self):
        """Custom max_retries should be applied."""
        strategy = node_retry_strategy(max_retries=5)
        assert strategy.max_attempts == 6

    def test_custom_delay(self):
        """Custom delay_seconds should be applied."""
        strategy = node_retry_strategy(delay_seconds=3.0, exponential=False)
        assert strategy.delay == 3.0


class TestGetNodeCircuitBreaker:
    """Test get_node_circuit_breaker helper function."""

    def test_creates_circuit_breaker(self):
        """Should create a circuit breaker for the node."""
        breaker = get_node_circuit_breaker("test_node_123_unique")
        assert breaker is not None
        assert "test_node_123_unique" in breaker.name

    def test_returns_same_breaker(self):
        """Should return same breaker for same node_id."""
        breaker1 = get_node_circuit_breaker("same_node_unique")
        breaker2 = get_node_circuit_breaker("same_node_unique")
        assert breaker1 is breaker2

    def test_different_nodes_different_breakers(self):
        """Different node_ids should get different breakers."""
        breaker1 = get_node_circuit_breaker("node_a_unique")
        breaker2 = get_node_circuit_breaker("node_b_unique")
        assert breaker1 is not breaker2

    def test_custom_failure_threshold(self):
        """Custom failure_threshold should be respected."""
        breaker = get_node_circuit_breaker("custom_node_unique", failure_threshold=10)
        assert breaker.failure_threshold == 10

    def test_custom_recovery_timeout(self):
        """Custom recovery_timeout should be respected."""
        breaker = get_node_circuit_breaker("timeout_node_unique", recovery_timeout=60.0)
        assert breaker.recovery_timeout == 60.0
