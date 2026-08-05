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

"""Unit tests for A/B testing allocator."""

import asyncio

import pytest

from victor.experiments.ab_testing.models import (
    AllocationStrategy,
    ExperimentConfig,
    ExperimentMetric,
    ExperimentVariant,
)
from victor.experiments.ab_testing.allocator import (
    RandomAllocator,
    RoundRobinAllocator,
    StickyAllocator,
    create_allocator,
)


@pytest.fixture
def sample_experiment():
    """Create a sample experiment for testing."""
    control = ExperimentVariant(
        variant_id="control",
        name="Control",
        workflow_type="yaml",
        workflow_config={"path": "test.yaml"},
        traffic_weight=0.5,
        is_control=True,
    )

    treatment = ExperimentVariant(
        variant_id="treatment",
        name="Treatment",
        workflow_type="yaml",
        workflow_config={"path": "test.yaml"},
        traffic_weight=0.5,
        is_control=False,
    )

    return ExperimentConfig(
        name="Test Experiment",
        hypothesis="Treatment is better",
        variants=[control, treatment],
        primary_metric=ExperimentMetric(
            metric_id="test_metric",
            name="Test Metric",
            metric_type="execution_time",
        ),
    )


class TestRandomAllocator:
    """Tests for RandomAllocator."""

    @pytest.mark.asyncio
    async def test_allocate_returns_valid_variant(self, sample_experiment):
        """Test that allocation returns a valid variant ID."""
        allocator = RandomAllocator()
        variant_id = await allocator.allocate_variant("user_123", sample_experiment)

        assert variant_id in {"control", "treatment"}

    @pytest.mark.asyncio
    async def test_allocate_distribution(self, sample_experiment):
        """Test that allocation follows distribution weights."""
        allocator = RandomAllocator(seed=42)

        # Allocate many times
        allocations = {"control": 0, "treatment": 0}
        for i in range(1000):
            user_id = f"user_{i}"
            variant_id = await allocator.allocate_variant(user_id, sample_experiment)
            allocations[variant_id] += 1

        # Check distribution (should be roughly 50/50)
        total = sum(allocations.values())
        control_pct = allocations["control"] / total
        treatment_pct = allocations["treatment"] / total

        # Allow 10% tolerance
        assert 0.40 <= control_pct <= 0.60
        assert 0.40 <= treatment_pct <= 0.60

    @pytest.mark.asyncio
    async def test_allocate_with_seed_is_reproducible(self, sample_experiment):
        """Test that seeding produces reproducible results."""
        import random

        # Reset random state before each test
        random.seed(100)

        allocator1 = RandomAllocator(seed=42)
        allocations1 = [
            await allocator1.allocate_variant(f"user_{i}", sample_experiment) for i in range(10)
        ]

        # Reset random state before second test
        random.seed(100)

        allocator2 = RandomAllocator(seed=42)
        allocations2 = [
            await allocator2.allocate_variant(f"user_{i}", sample_experiment) for i in range(10)
        ]

        assert allocations1 == allocations2

    @pytest.mark.asyncio
    async def test_allocate_invalid_experiment_raises(self):
        """Test that invalid experiment raises ValueError."""
        allocator = RandomAllocator()

        # No variants
        with pytest.raises(ValueError, match="no variants"):
            await allocator.allocate_variant("user_123", ExperimentConfig(name="Empty"))


class TestStickyAllocator:
    """Tests for StickyAllocator."""

    @pytest.mark.asyncio
    async def test_allocate_is_consistent(self, sample_experiment):
        """Test that same user always gets same variant."""
        allocator = StickyAllocator()

        variant1 = await allocator.allocate_variant("user_123", sample_experiment)
        variant2 = await allocator.allocate_variant("user_123", sample_experiment)
        variant3 = await allocator.allocate_variant("user_123", sample_experiment)

        assert variant1 == variant2 == variant3

    @pytest.mark.asyncio
    async def test_allocate_different_users(self, sample_experiment):
        """Test that different users can get different variants."""
        allocator = StickyAllocator()

        # Allocate many users
        allocations = set()
        for i in range(100):
            user_id = f"user_{i}"
            variant_id = await allocator.allocate_variant(user_id, sample_experiment)
            allocations.add(variant_id)

        # Should have both variants allocated
        assert len(allocations) == 2

    @pytest.mark.asyncio
    async def test_get_allocation_stats(self, sample_experiment):
        """Test that allocation stats are tracked."""
        allocator = StickyAllocator()

        # Allocate some users
        for i in range(10):
            user_id = f"user_{i}"
            await allocator.allocate_variant(user_id, sample_experiment)

        # Get stats
        stats = await allocator.get_allocation_stats(sample_experiment.experiment_id)

        # Should have 10 allocations
        assert sum(stats.values()) == 10


class TestRoundRobinAllocator:
    """Tests for RoundRobinAllocator."""

    @pytest.mark.asyncio
    async def test_allocate_rotates_variants(self, sample_experiment):
        """Test that allocation rotates through variants."""
        allocator = RoundRobinAllocator()

        variant1 = await allocator.allocate_variant("user_1", sample_experiment)
        variant2 = await allocator.allocate_variant("user_2", sample_experiment)
        variant3 = await allocator.allocate_variant("user_3", sample_experiment)
        variant4 = await allocator.allocate_variant("user_4", sample_experiment)

        # Should alternate: control, treatment, control, treatment
        assert variant1 == "control"
        assert variant2 == "treatment"
        assert variant3 == "control"
        assert variant4 == "treatment"

    @pytest.mark.asyncio
    async def test_allocate_balanced_distribution(self, sample_experiment):
        """Test that round-robin creates balanced distribution."""
        allocator = RoundRobinAllocator()

        allocations = {"control": 0, "treatment": 0}
        for i in range(100):
            user_id = f"user_{i}"
            variant_id = await allocator.allocate_variant(user_id, sample_experiment)
            allocations[variant_id] += 1

        # Should be perfectly balanced
        assert allocations["control"] == 50
        assert allocations["treatment"] == 50

    @pytest.mark.asyncio
    async def test_get_allocation_stats(self, sample_experiment):
        """Test that allocation stats are tracked."""
        allocator = RoundRobinAllocator()

        # Allocate some users
        for i in range(10):
            await allocator.allocate_variant(f"user_{i}", sample_experiment)

        # Get stats
        stats = await allocator.get_allocation_stats(sample_experiment.experiment_id)

        # Should have 10 total allocations
        assert stats["total_allocations"] == 10


class TestCreateAllocator:
    """Tests for create_allocator factory function."""

    def test_create_random_allocator(self):
        """Test creating RandomAllocator."""
        allocator = create_allocator(AllocationStrategy.RANDOM)
        assert isinstance(allocator, RandomAllocator)

    def test_create_sticky_allocator(self):
        """Test creating StickyAllocator."""
        allocator = create_allocator(AllocationStrategy.STICKY)
        assert isinstance(allocator, StickyAllocator)

    def test_create_round_robin_allocator(self):
        """Test creating RoundRobinAllocator."""
        allocator = create_allocator(AllocationStrategy.ROUND_ROBIN)
        assert isinstance(allocator, RoundRobinAllocator)

    def test_create_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown allocation strategy"):
            # Create an invalid strategy for testing
            from enum import Enum

            class InvalidStrategy(Enum):
                INVALID = "invalid"

            create_allocator(InvalidStrategy.INVALID)


class TestAllocatorIntegration:
    """Integration tests for allocators."""

    @pytest.mark.asyncio
    async def test_three_variant_distribution(self):
        """Test allocation with three variants."""
        # Create experiment with 3 variants
        variants = [
            ExperimentVariant(
                variant_id="v1",
                name="Variant 1",
                workflow_type="yaml",
                workflow_config={},
                traffic_weight=0.33,
            ),
            ExperimentVariant(
                variant_id="v2",
                name="Variant 2",
                workflow_type="yaml",
                workflow_config={},
                traffic_weight=0.33,
            ),
            ExperimentVariant(
                variant_id="v3",
                name="Variant 3",
                workflow_type="yaml",
                workflow_config={},
                traffic_weight=0.34,
            ),
        ]

        experiment = ExperimentConfig(
            name="Three Variant Test",
            variants=variants,
            primary_metric=ExperimentMetric(
                metric_id="test",
                name="Test",
                metric_type="execution_time",
            ),
        )

        # Test random allocator
        allocator = RandomAllocator(seed=42)
        allocations = {"v1": 0, "v2": 0, "v3": 0}

        for i in range(1000):
            variant_id = await allocator.allocate_variant(f"user_{i}", experiment)
            allocations[variant_id] += 1

        # Check all variants were allocated
        assert all(count > 0 for count in allocations.values())

        # Check distribution is roughly correct
        total = sum(allocations.values())
        for variant_id, count in allocations.items():
            pct = count / total
            # Allow 10% tolerance
            assert 0.23 <= pct <= 0.40
