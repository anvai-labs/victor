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

"""Adaptation strategies for dynamic workflow modification.

This module provides pre-built adaptation strategies that can be used
with AdaptableGraph for common optimization patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from victor.framework.adaptation.types import (
    AdaptationStrategy,
    AdaptationTrigger,
    GraphModification,
    ModificationType,
)

logger = logging.getLogger(__name__)


def create_retry_strategy(
    error_threshold: int = 3,
    max_retries: int = 3,
    backoff: str = "exponential",
) -> AdaptationStrategy:
    """Create a strategy that adds retry logic to failing nodes.

    Args:
        error_threshold: Number of errors before triggering
        max_retries: Max retries to configure
        backoff: Backoff strategy ("constant", "linear", "exponential")

    Returns:
        AdaptationStrategy for retry logic
    """

    def trigger_condition(context: Dict[str, Any]) -> bool:
        """Check if node has exceeded error threshold."""
        error_counts = context.get("error_counts", {})
        for node_id, count in error_counts.items():
            if count >= error_threshold:
                # Don't add retry if already has retry
                if node_id not in context.get("nodes_with_retry", set()):
                    return True
        return False

    def modification_generator(context: Dict[str, Any]) -> List[GraphModification]:
        """Generate retry modifications for failing nodes."""
        modifications = []
        error_counts = context.get("error_counts", {})
        nodes_with_retry = context.get("nodes_with_retry", set())

        for node_id, count in error_counts.items():
            if count >= error_threshold and node_id not in nodes_with_retry:
                modifications.append(
                    GraphModification(
                        modification_type=ModificationType.ADD_RETRY,
                        description=f"Add retry to {node_id} after {count} errors",
                        target_node=node_id,
                        data={
                            "max_retries": max_retries,
                            "backoff": backoff,
                            "initial_delay_ms": 1000,
                        },
                        trigger=AdaptationTrigger.ERROR,
                        priority=count,  # Higher error count = higher priority
                    )
                )

        return modifications

    return AdaptationStrategy(
        name="auto_retry",
        description=f"Add retry logic after {error_threshold} failures",
        trigger_condition=trigger_condition,
        modification_generator=modification_generator,
        priority=10,
    )


def create_circuit_breaker_strategy(
    failure_threshold: int = 5,
    reset_timeout_seconds: int = 60,
) -> AdaptationStrategy:
    """Create a strategy that adds circuit breakers to unreliable nodes.

    Args:
        failure_threshold: Failures before opening circuit
        reset_timeout_seconds: Time before attempting reset

    Returns:
        AdaptationStrategy for circuit breakers
    """

    def trigger_condition(context: Dict[str, Any]) -> bool:
        """Check if node should have circuit breaker."""
        recent_failures = context.get("recent_failures", {})
        for node_id, failures in recent_failures.items():
            if len(failures) >= failure_threshold:
                if node_id not in context.get("nodes_with_circuit_breaker", set()):
                    return True
        return False

    def modification_generator(context: Dict[str, Any]) -> List[GraphModification]:
        """Generate circuit breaker modifications."""
        modifications = []
        recent_failures = context.get("recent_failures", {})
        existing_breakers = context.get("nodes_with_circuit_breaker", set())

        for node_id, failures in recent_failures.items():
            if len(failures) >= failure_threshold and node_id not in existing_breakers:
                modifications.append(
                    GraphModification(
                        modification_type=ModificationType.ADD_CIRCUIT_BREAKER,
                        description=f"Add circuit breaker to {node_id}",
                        target_node=node_id,
                        data={
                            "failure_threshold": failure_threshold,
                            "reset_timeout_seconds": reset_timeout_seconds,
                        },
                        trigger=AdaptationTrigger.ERROR,
                        priority=len(failures),
                    )
                )

        return modifications

    return AdaptationStrategy(
        name="auto_circuit_breaker",
        description=f"Add circuit breaker after {failure_threshold} failures",
        trigger_condition=trigger_condition,
        modification_generator=modification_generator,
        priority=20,
    )


def create_parallelization_strategy(
    min_sequential_nodes: int = 3,
    latency_threshold_ms: float = 1000.0,
) -> AdaptationStrategy:
    """Create a strategy that parallelizes sequential independent nodes.

    Args:
        min_sequential_nodes: Minimum sequential nodes to consider
        latency_threshold_ms: Latency threshold to trigger optimization

    Returns:
        AdaptationStrategy for parallelization
    """

    def trigger_condition(context: Dict[str, Any]) -> bool:
        """Check if parallelization would help."""
        # Check latency
        total_latency = context.get("total_latency_ms", 0)
        if total_latency < latency_threshold_ms:
            return False

        # Check for parallelizable nodes
        node_dependencies = context.get("node_dependencies", {})
        if len(node_dependencies) < min_sequential_nodes:
            return False

        # Find independent nodes
        independent_groups = _find_independent_groups(node_dependencies)
        return any(len(group) >= 2 for group in independent_groups)

    def modification_generator(context: Dict[str, Any]) -> List[GraphModification]:
        """Generate parallelization modifications."""
        modifications = []
        node_dependencies = context.get("node_dependencies", {})

        # Find groups of independent nodes
        independent_groups = _find_independent_groups(node_dependencies)

        for i, group in enumerate(independent_groups):
            if len(group) >= 2:
                modifications.append(
                    GraphModification(
                        modification_type=ModificationType.ADD_PARALLELIZATION,
                        description=f"Parallelize {len(group)} independent nodes",
                        data={
                            "nodes_to_parallelize": list(group),
                            "group_id": f"parallel_group_{i}",
                        },
                        trigger=AdaptationTrigger.PERFORMANCE,
                        priority=len(group),
                    )
                )

        return modifications

    return AdaptationStrategy(
        name="auto_parallelize",
        description=f"Parallelize independent nodes when latency > {latency_threshold_ms}ms",
        trigger_condition=trigger_condition,
        modification_generator=modification_generator,
        priority=5,
    )


def create_caching_strategy(
    hit_rate_threshold: float = 0.3,
    idempotent_only: bool = True,
) -> AdaptationStrategy:
    """Create a strategy that adds caching to frequently called nodes.

    Args:
        hit_rate_threshold: Minimum estimated cache hit rate to add caching
        idempotent_only: Only cache idempotent operations

    Returns:
        AdaptationStrategy for caching
    """

    def trigger_condition(context: Dict[str, Any]) -> bool:
        """Check if caching would help."""
        node_call_patterns = context.get("node_call_patterns", {})

        for node_id, patterns in node_call_patterns.items():
            if node_id in context.get("nodes_with_caching", set()):
                continue

            # Check if idempotent
            if idempotent_only and not patterns.get("is_idempotent", False):
                continue

            # Estimate cache hit rate
            duplicate_ratio = patterns.get("duplicate_input_ratio", 0)
            if duplicate_ratio >= hit_rate_threshold:
                return True

        return False

    def modification_generator(context: Dict[str, Any]) -> List[GraphModification]:
        """Generate caching modifications."""
        modifications = []
        node_call_patterns = context.get("node_call_patterns", {})
        existing_caching = context.get("nodes_with_caching", set())

        for node_id, patterns in node_call_patterns.items():
            if node_id in existing_caching:
                continue

            if idempotent_only and not patterns.get("is_idempotent", False):
                continue

            duplicate_ratio = patterns.get("duplicate_input_ratio", 0)
            if duplicate_ratio >= hit_rate_threshold:
                modifications.append(
                    GraphModification(
                        modification_type=ModificationType.ADD_CACHING,
                        description=f"Add caching to {node_id} (est. hit rate: {duplicate_ratio:.1%})",
                        target_node=node_id,
                        data={
                            "cache_type": "lru",
                            "max_size": 1000,
                            "ttl_seconds": 3600,
                            "estimated_hit_rate": duplicate_ratio,
                        },
                        trigger=AdaptationTrigger.PERFORMANCE,
                        priority=int(duplicate_ratio * 100),
                    )
                )

        return modifications

    return AdaptationStrategy(
        name="auto_cache",
        description=f"Add caching when estimated hit rate > {hit_rate_threshold:.0%}",
        trigger_condition=trigger_condition,
        modification_generator=modification_generator,
        priority=15,
    )


def _find_independent_groups(
    node_dependencies: Dict[str, List[str]],
) -> List[set]:
    """Find groups of nodes that can run in parallel.

    Args:
        node_dependencies: Dict mapping node_id -> list of dependencies

    Returns:
        List of sets of independent nodes
    """
    groups = []
    remaining_nodes = set(node_dependencies.keys())

    while remaining_nodes:
        # Find nodes with no remaining dependencies
        independent = set()
        for node_id in remaining_nodes:
            deps = set(node_dependencies.get(node_id, []))
            if not deps.intersection(remaining_nodes - {node_id}):
                independent.add(node_id)

        if not independent:
            # Break cycle - just take remaining nodes
            groups.append(remaining_nodes.copy())
            break

        if len(independent) >= 2:
            groups.append(independent)

        remaining_nodes -= independent

    return groups


# Pre-built strategy collection
DEFAULT_STRATEGIES = [
    create_retry_strategy(),
    create_circuit_breaker_strategy(),
    create_parallelization_strategy(),
    create_caching_strategy(),
]


__all__ = [
    "AdaptationStrategy",
    "create_retry_strategy",
    "create_circuit_breaker_strategy",
    "create_parallelization_strategy",
    "create_caching_strategy",
    "DEFAULT_STRATEGIES",
]
