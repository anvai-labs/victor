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

"""Workflow-layer facade for resilience patterns.

This module provides workflow-specific access to Victor's resilience infrastructure
WITHOUT duplicating any implementations. It re-exports existing components and adds
thin adapter functions to bridge workflow protocols with core retry strategies.

Architecture:
    workflow.RetryPolicy -> adapter -> core.BaseRetryStrategy -> core.RetryExecutor
    workflow nodes -> CircuitBreakerRegistry -> providers.CircuitBreaker
    circuit state/config/error types -> core.circuit_breaker

Design Pattern: Facade + Adapter
- Re-exports existing CircuitBreaker and RetryExecutor (no duplication)
- Adds `retry_policy_to_strategy` adapter for workflow RetryPolicy
- Provides workflow-specific pre-configured strategies

Example:
    from victor.workflows.resilience import (
        # Circuit Breaker (from providers)
        CircuitBreaker,
        CircuitBreakerRegistry,
        CircuitState,

        # Retry (from core)
        RetryExecutor,
        RetryResult,
        with_retry,

        # Workflow-specific adapter
        retry_policy_to_strategy,
        node_retry_strategy,
    )

    # Convert workflow RetryPolicy to core BaseRetryStrategy
    from victor.workflows.protocols import RetryPolicy
    policy = RetryPolicy(max_retries=5, delay_seconds=2.0)
    strategy = retry_policy_to_strategy(policy)

    # Use with executor
    executor = RetryExecutor(strategy)
    result = await executor.execute_async(my_func)

    # Or use circuit breaker
    breaker = CircuitBreakerRegistry.get_or_create("agent_node_xyz")
    result = await breaker.execute(my_func)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Re-export Shared Circuit Breaker Types (from core - no duplication)
# =============================================================================
from victor.core.circuit_breaker import (
    CircuitBreakerError,
    CircuitState,
)

# =============================================================================
# Re-export Circuit Breaker (from providers - no duplication)
# =============================================================================
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)

# =============================================================================
# Re-export Retry Components (from core - no duplication)
# =============================================================================
from victor.core.retry import (
    BaseRetryStrategy,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
    RetryContext,
    RetryExecutor,
    RetryOutcome,
    RetryResult,
    with_retry,
    with_retry_sync,
)

if TYPE_CHECKING:
    from victor.workflows.protocols import RetryPolicy


# =============================================================================
# Workflow-Specific Adapters
# =============================================================================


def retry_policy_to_strategy(policy: "RetryPolicy") -> BaseRetryStrategy:
    """Convert workflow RetryPolicy to core BaseRetryStrategy.

    This adapter bridges the workflow protocol (RetryPolicy dataclass)
    with the core retry infrastructure (BaseRetryStrategy ABC).

    Args:
        policy: Workflow RetryPolicy from protocols.py

    Returns:
        Configured BaseRetryStrategy matching the policy settings

    Example:
        from victor.workflows.protocols import RetryPolicy
        policy = RetryPolicy(max_retries=3, delay_seconds=1.0)
        strategy = retry_policy_to_strategy(policy)
    """
    if policy.exponential_backoff:
        return ExponentialBackoffStrategy(
            max_attempts=policy.max_retries + 1,  # max_attempts includes initial
            base_delay=policy.delay_seconds,
            max_delay=policy.delay_seconds * 32,  # Cap at ~32x base
            multiplier=2.0,
            jitter=0.1,
            retryable_exceptions=(
                set(policy.retry_on_exceptions) if policy.retry_on_exceptions else None
            ),
        )
    else:
        return FixedDelayStrategy(
            max_attempts=policy.max_retries + 1,
            delay=policy.delay_seconds,
        )


def node_retry_strategy(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    exponential: bool = True,
) -> BaseRetryStrategy:
    """Create retry strategy optimized for workflow node execution.

    Pre-configured strategy for workflow nodes with sensible defaults
    for handling LLM API failures, tool timeouts, etc.

    Args:
        max_retries: Maximum retry attempts (not including initial)
        delay_seconds: Base delay between retries
        exponential: Use exponential backoff (True) or fixed delay (False)

    Returns:
        Configured BaseRetryStrategy
    """
    if exponential:
        return ExponentialBackoffStrategy(
            max_attempts=max_retries + 1,
            base_delay=delay_seconds,
            max_delay=60.0,
            multiplier=2.0,
            jitter=0.15,
        )
    else:
        return FixedDelayStrategy(
            max_attempts=max_retries + 1,
            delay=delay_seconds,
        )


def get_node_circuit_breaker(
    node_id: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a workflow node.

    Uses CircuitBreakerRegistry to ensure consistent breaker state
    across workflow executions for the same node.

    Args:
        node_id: Unique identifier for the node
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery

    Returns:
        CircuitBreaker instance for the node
    """
    return CircuitBreakerRegistry.get_or_create(
        name=f"workflow_node_{node_id}",
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


__all__ = [
    # Circuit Breaker (re-exported from providers)
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Retry (re-exported from core)
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    "LinearBackoffStrategy",
    "NoRetryStrategy",
    "RetryContext",
    "RetryExecutor",
    "RetryOutcome",
    "RetryResult",
    "BaseRetryStrategy",
    "with_retry",
    "with_retry_sync",
    # Workflow-specific adapters
    "retry_policy_to_strategy",
    "node_retry_strategy",
    "get_node_circuit_breaker",
]
