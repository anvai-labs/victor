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

"""Traffic allocation strategies for A/B testing.

This module implements various strategies for allocating users to experiment variants.
"""

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from victor.experiments.ab_testing.models import (
    AllocationStrategy,
    ExperimentConfig,
    ExperimentVariant,
)


class TrafficAllocator(ABC):
    """Base class for traffic allocation strategies.

    Subclasses must implement the allocate_variant method.
    """

    @abstractmethod
    async def allocate_variant(
        self,
        user_id: str,
        experiment: ExperimentConfig,
        context: Optional[Dict] = None,
    ) -> str:
        """Allocate a user to a variant.

        Args:
            user_id: Unique user identifier
            experiment: Experiment configuration
            context: Optional context for allocation decisions

        Returns:
            variant_id: The allocated variant ID

        Raises:
            ValueError: If allocation fails
        """
        pass

    async def get_allocation_stats(
        self,
        experiment_id: str,
    ) -> Dict[str, int]:
        """Get allocation statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping variant_id to allocation count
        """
        # Default implementation - subclasses can override for persistence
        return {}


class RandomAllocator(TrafficAllocator):
    """Random (weighted) allocation strategy.

    Pros:
        - Simple to implement
        - Unbiased allocation
        - Works well for large sample sizes

    Cons:
        - Same user may get different variants across runs
        - No consistency for user experience
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize random allocator.

        Args:
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    async def allocate_variant(
        self,
        user_id: str,
        experiment: ExperimentConfig,
        context: Optional[Dict] = None,
    ) -> str:
        """Allocate user to variant using weighted random selection.

        Args:
            user_id: User identifier (not used in random allocation)
            experiment: Experiment configuration with variants
            context: Optional context (not used in random allocation)

        Returns:
            variant_id: The allocated variant ID

        Raises:
            ValueError: If no variants are defined or weights are invalid
        """
        if not experiment.variants:
            raise ValueError("Experiment has no variants defined")

        # Extract weights and variant IDs
        variants = experiment.variants
        weights = [v.traffic_weight for v in variants]
        variant_ids = [v.variant_id for v in variants]

        # Validate weights
        if not all(w >= 0 for w in weights):
            raise ValueError("Traffic weights must be non-negative")

        if sum(weights) == 0:
            raise ValueError("Sum of traffic weights must be positive")

        # Weighted random choice
        chosen_index = random.choices(range(len(variants)), weights=weights, k=1)[0]

        return variant_ids[chosen_index]


class StickyAllocator(TrafficAllocator):
    """Sticky allocation using consistent hashing.

    Pros:
        - Same user always gets same variant
        - Consistent user experience
        - Easier to debug

    Cons:
        - Requires user identification
        - Can lead to imbalance if few users
    """

    def __init__(self):
        """Initialize sticky allocator."""
        # Cache for allocations to avoid recomputation
        self._allocation_cache: Dict[str, Dict[str, str]] = {}

    async def allocate_variant(
        self,
        user_id: str,
        experiment: ExperimentConfig,
        context: Optional[Dict] = None,
    ) -> str:
        """Allocate user to variant using consistent hashing.

        Args:
            user_id: Unique user identifier
            experiment: Experiment configuration with variants
            context: Optional context (not used in sticky allocation)

        Returns:
            variant_id: The allocated variant ID

        Raises:
            ValueError: If no variants are defined or weights are invalid
        """
        if not experiment.variants:
            raise ValueError("Experiment has no variants defined")

        # Check cache
        experiment_id = experiment.experiment_id
        if experiment_id not in self._allocation_cache:
            self._allocation_cache[experiment_id] = {}

        if user_id in self._allocation_cache[experiment_id]:
            # Return cached allocation
            return self._allocation_cache[experiment_id][user_id]

        # Create stable hash from user_id
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)

        # Normalize to [0, 1]
        normalized = hash_value / (2**256 - 1)

        # Find variant based on cumulative weights
        variants = experiment.variants
        cumulative = 0.0

        for variant in variants:
            cumulative += variant.traffic_weight
            if normalized <= cumulative:
                # Cache and return allocation
                self._allocation_cache[experiment_id][user_id] = variant.variant_id
                return variant.variant_id

        # Fallback to last variant (handles floating point precision issues)
        variant_id = variants[-1].variant_id
        self._allocation_cache[experiment_id][user_id] = variant_id
        return variant_id

    async def get_allocation_stats(
        self,
        experiment_id: str,
    ) -> Dict[str, int]:
        """Get allocation statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping variant_id to allocation count
        """
        if experiment_id not in self._allocation_cache:
            return {}

        # Count allocations per variant
        stats: Dict[str, int] = {}
        for variant_id in self._allocation_cache[experiment_id].values():
            stats[variant_id] = stats.get(variant_id, 0) + 1

        return stats


class RoundRobinAllocator(TrafficAllocator):
    """Round-robin allocation strategy.

    Pros:
        - Perfect balance across variants
        - Predictable distribution
        - Simple to verify

    Cons:
        - Not random (can introduce bias)
        - Requires state tracking
        - Less suitable for statistical testing
    """

    def __init__(self):
        """Initialize round-robin allocator."""
        # Counter for each experiment
        self._counters: Dict[str, int] = {}

    async def allocate_variant(
        self,
        user_id: str,
        experiment: ExperimentConfig,
        context: Optional[Dict] = None,
    ) -> str:
        """Allocate user to variant using round-robin.

        Args:
            user_id: User identifier (not used in round-robin)
            experiment: Experiment configuration with variants
            context: Optional context (not used in round-robin)

        Returns:
            variant_id: The allocated variant ID

        Raises:
            ValueError: If no variants are defined
        """
        if not experiment.variants:
            raise ValueError("Experiment has no variants defined")

        experiment_id = experiment.experiment_id
        variants = experiment.variants

        # Initialize counter if needed
        if experiment_id not in self._counters:
            self._counters[experiment_id] = 0

        # Get current counter value
        counter = self._counters[experiment_id]

        # Select variant
        variant_index = counter % len(variants)
        variant_id = variants[variant_index].variant_id

        # Increment counter
        self._counters[experiment_id] = counter + 1

        return variant_id

    async def get_allocation_stats(
        self,
        experiment_id: str,
    ) -> Dict[str, int]:
        """Get allocation statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping variant_id to allocation count
        """
        if experiment_id not in self._counters:
            return {}

        # Return total allocations
        total = self._counters[experiment_id]
        return {"total_allocations": total}


def create_allocator(strategy: AllocationStrategy) -> TrafficAllocator:
    """Factory function to create allocator instances.

    Args:
        strategy: The allocation strategy to use

    Returns:
        TrafficAllocator: An instance of the appropriate allocator

    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == AllocationStrategy.RANDOM:
        return RandomAllocator()
    elif strategy == AllocationStrategy.STICKY:
        return StickyAllocator()
    elif strategy == AllocationStrategy.ROUND_ROBIN:
        return RoundRobinAllocator()
    else:
        raise ValueError(f"Unknown allocation strategy: {strategy}")
