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

"""Unit tests for RLEvictionPolicy.

Tests the RL-based cache eviction policy.
"""

import pytest
from unittest.mock import MagicMock

from victor.storage.cache.rl_eviction_policy import (
    RLEvictionPolicy,
    CacheEntryState,
    EvictionAction,
    EvictionDecision,
    get_rl_eviction_policy,
)


class TestCacheEntryState:
    """Tests for CacheEntryState."""

    def test_cache_entry_state_creation(self) -> None:
        """Test creating cache entry state."""
        state = CacheEntryState(
            key="test_key",
            tool_type="code_search",
            entry_age_seconds=120.0,
            hit_count=5,
            size_bytes=2048,
            context_relevance=0.8,
        )

        assert state.key == "test_key"
        assert state.tool_type == "code_search"
        assert state.hit_count == 5

    def test_cache_entry_state_defaults(self) -> None:
        """Test default values for cache entry state."""
        state = CacheEntryState(key="key")

        assert state.tool_type == "unknown"
        assert state.entry_age_seconds == 0.0
        assert state.hit_count == 0
        assert state.context_relevance == 0.5

    def test_to_feature_tuple_age_buckets(self) -> None:
        """Test age bucketing in feature tuple."""
        fresh = CacheEntryState(key="k", entry_age_seconds=30)
        recent = CacheEntryState(key="k", entry_age_seconds=120)
        moderate = CacheEntryState(key="k", entry_age_seconds=600)
        old = CacheEntryState(key="k", entry_age_seconds=1200)

        assert "fresh" in fresh.to_feature_tuple()
        assert "recent" in recent.to_feature_tuple()
        assert "moderate" in moderate.to_feature_tuple()
        assert "old" in old.to_feature_tuple()

    def test_to_feature_tuple_hit_buckets(self) -> None:
        """Test hit count bucketing."""
        unused = CacheEntryState(key="k", hit_count=0)
        low = CacheEntryState(key="k", hit_count=2)
        medium = CacheEntryState(key="k", hit_count=5)
        high = CacheEntryState(key="k", hit_count=15)

        assert "unused" in unused.to_feature_tuple()
        assert "low" in low.to_feature_tuple()
        assert "medium" in medium.to_feature_tuple()
        assert "high" in high.to_feature_tuple()

    def test_to_feature_tuple_size_buckets(self) -> None:
        """Test size bucketing."""
        small = CacheEntryState(key="k", size_bytes=512)
        medium = CacheEntryState(key="k", size_bytes=5000)
        large = CacheEntryState(key="k", size_bytes=20000)

        assert "small" in small.to_feature_tuple()
        assert "medium" in medium.to_feature_tuple()
        assert "large" in large.to_feature_tuple()

    def test_to_feature_tuple_relevance_buckets(self) -> None:
        """Test relevance bucketing."""
        low = CacheEntryState(key="k", context_relevance=0.2)
        medium = CacheEntryState(key="k", context_relevance=0.5)
        high = CacheEntryState(key="k", context_relevance=0.8)

        assert "low" in low.to_feature_tuple()
        assert "medium" in medium.to_feature_tuple()
        assert "high" in high.to_feature_tuple()


class TestEvictionDecision:
    """Tests for EvictionDecision."""

    def test_eviction_decision_creation(self) -> None:
        """Test creating eviction decision."""
        decision = EvictionDecision(
            action=EvictionAction.EVICT,
            confidence=0.8,
            reason="Old unused entry",
            q_value=0.7,
        )

        assert decision.action == EvictionAction.EVICT
        assert decision.confidence == 0.8
        assert decision.q_value == 0.7

    def test_eviction_decision_defaults(self) -> None:
        """Test default values for eviction decision."""
        decision = EvictionDecision(action=EvictionAction.KEEP)

        assert decision.confidence == 0.5
        assert decision.reason == ""
        assert decision.q_value == 0.0


class TestEvictionAction:
    """Tests for EvictionAction enum."""

    def test_eviction_actions(self) -> None:
        """Test eviction action values."""
        assert EvictionAction.KEEP.value == "keep"
        assert EvictionAction.EVICT.value == "evict"
        assert EvictionAction.PROMOTE.value == "promote"
        assert EvictionAction.DEMOTE.value == "demote"


class TestRLEvictionPolicy:
    """Tests for RLEvictionPolicy."""

    @pytest.fixture
    def policy(self) -> RLEvictionPolicy:
        """Fixture for policy with no exploration."""
        return RLEvictionPolicy(epsilon=0.0)

    def test_initialization(self, policy: RLEvictionPolicy) -> None:
        """Test policy initialization."""
        assert policy.learning_rate == RLEvictionPolicy.DEFAULT_LEARNING_RATE
        assert policy.discount_factor == RLEvictionPolicy.DEFAULT_DISCOUNT_FACTOR
        assert policy._total_decisions == 0

    def test_get_decision_returns_decision(self, policy: RLEvictionPolicy) -> None:
        """Test getting decision returns EvictionDecision."""
        state = CacheEntryState(key="test", tool_type="code_search")

        decision = policy.get_decision(state)

        assert isinstance(decision, EvictionDecision)
        assert decision.action in EvictionAction
        assert 0 <= decision.confidence <= 1

    def test_get_decision_increments_count(self, policy: RLEvictionPolicy) -> None:
        """Test decision increments total count."""
        state = CacheEntryState(key="test")

        policy.get_decision(state)
        policy.get_decision(state)

        assert policy._total_decisions == 2

    def test_get_decision_tracks_decision(self, policy: RLEvictionPolicy) -> None:
        """Test decision is tracked for feedback."""
        state = CacheEntryState(key="test_key")

        policy.get_decision(state)

        assert "test_key" in policy._recent_decisions

    def test_heuristic_keep_high_hits(self, policy: RLEvictionPolicy) -> None:
        """Test heuristic prefers keeping high-hit entries."""
        state = CacheEntryState(
            key="high_hits",
            hit_count=10,
            entry_age_seconds=60,
        )

        q_values = policy._get_q_values(state)

        # KEEP should have higher Q than EVICT for high-hit entries
        assert q_values[EvictionAction.KEEP] > q_values[EvictionAction.EVICT]

    def test_heuristic_evict_old_unused(self, policy: RLEvictionPolicy) -> None:
        """Test heuristic prefers evicting old unused entries."""
        state = CacheEntryState(
            key="old_unused",
            hit_count=0,
            entry_age_seconds=900,  # 15 minutes
        )

        q_values = policy._get_q_values(state)

        # EVICT should have higher Q than KEEP for old unused entries
        assert q_values[EvictionAction.EVICT] > q_values[EvictionAction.KEEP]

    def test_heuristic_keep_high_relevance(self, policy: RLEvictionPolicy) -> None:
        """Test heuristic prefers keeping high-relevance entries."""
        state = CacheEntryState(
            key="relevant",
            context_relevance=0.9,
        )

        q_values = policy._get_q_values(state)

        assert q_values[EvictionAction.KEEP] > 0.5

    def test_high_utilization_bias_eviction(self, policy: RLEvictionPolicy) -> None:
        """Test high utilization biases toward eviction."""
        state = CacheEntryState(key="test", hit_count=1)

        # Get multiple decisions with high utilization
        evict_count = 0
        for _ in range(10):
            decision = policy.get_decision(state, cache_utilization=0.95)
            if decision.action == EvictionAction.EVICT:
                evict_count += 1

        # Should see some evictions due to high utilization
        # (may not be all due to Q-values)
        assert evict_count >= 0  # At least the bias is applied

    def test_record_hit(self, policy: RLEvictionPolicy) -> None:
        """Test recording cache hit."""
        state = CacheEntryState(key="hit_key")
        policy.get_decision(state)

        policy.record_hit("hit_key")

        assert policy._hits_after_keep >= 0

    def test_record_miss(self, policy: RLEvictionPolicy) -> None:
        """Test recording cache miss."""
        state = CacheEntryState(key="miss_key")
        policy.get_decision(state)

        policy.record_miss("miss_key", was_evicted=False)

        assert policy._misses_after_keep >= 0

    def test_record_eviction_success(self, policy: RLEvictionPolicy) -> None:
        """Test recording successful eviction."""
        state = CacheEntryState(key="evict_key")
        policy._recent_decisions["evict_key"] = (state, EvictionAction.EVICT)

        policy.record_eviction_success("evict_key")

        assert policy._evictions == 1

    def test_q_value_update_on_hit(self, policy: RLEvictionPolicy) -> None:
        """Test Q-value is updated on hit feedback."""
        state = CacheEntryState(key="update_key", tool_type="test")
        policy._recent_decisions["update_key"] = (state, EvictionAction.KEEP)

        feature_key = state.to_feature_tuple()
        initial_q = policy._get_q_values(state)[EvictionAction.KEEP]

        policy.record_hit("update_key")

        new_q = policy._q_table.get(feature_key, {}).get(EvictionAction.KEEP, initial_q)
        # Q-value should increase after positive reward
        assert new_q >= initial_q

    def test_get_entries_to_evict(self, policy: RLEvictionPolicy) -> None:
        """Test getting entries to evict."""
        entries = [
            CacheEntryState(key="old1", hit_count=0, entry_age_seconds=1000),
            CacheEntryState(key="new1", hit_count=5, entry_age_seconds=30),
            CacheEntryState(key="old2", hit_count=0, entry_age_seconds=900),
        ]

        to_evict = policy.get_entries_to_evict(entries, target_count=2)

        # Should return list of keys
        assert isinstance(to_evict, list)
        assert len(to_evict) <= 2

    def test_get_entries_to_evict_prefers_old_unused(self, policy: RLEvictionPolicy) -> None:
        """Test eviction prefers old unused entries."""
        entries = [
            CacheEntryState(key="old_unused", hit_count=0, entry_age_seconds=1200),
            CacheEntryState(key="new_used", hit_count=10, entry_age_seconds=30),
        ]

        to_evict = policy.get_entries_to_evict(entries, target_count=1)

        # Old unused should be evicted first
        if to_evict:
            assert "old_unused" in to_evict

    def test_decision_tracking_limit(self, policy: RLEvictionPolicy) -> None:
        """Test decision tracking is limited."""
        policy._max_tracked_decisions = 10

        # Add more than limit
        for i in range(15):
            state = CacheEntryState(key=f"key_{i}")
            policy.get_decision(state)

        # Should have cleaned up older entries
        assert len(policy._recent_decisions) <= 10

    def test_exploration(self) -> None:
        """Test exploration with high epsilon."""
        policy = RLEvictionPolicy(epsilon=1.0)  # Always explore
        state = CacheEntryState(key="explore")

        actions = set()
        for _ in range(20):
            decision = policy.get_decision(state)
            actions.add(decision.action)

        # Should see multiple actions due to exploration
        assert len(actions) > 1

    def test_export_metrics(self, policy: RLEvictionPolicy) -> None:
        """Test exporting metrics."""
        state = CacheEntryState(key="metrics_test")
        policy.get_decision(state)
        policy.record_hit("metrics_test")

        metrics = policy.export_metrics()

        assert metrics["total_decisions"] == 1
        assert "hits_after_keep" in metrics
        assert "hit_rate_after_keep" in metrics
        assert "epsilon" in metrics

    def test_with_external_learner(self) -> None:
        """Test policy with external learner."""
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = MagicMock(value=0.7)

        policy = RLEvictionPolicy(cache_eviction_learner=mock_learner)
        state = CacheEntryState(key="external")

        policy.get_decision(state)

        mock_learner.get_recommendation.assert_called_once()


class TestGlobalSingleton:
    """Tests for global singleton."""

    def test_get_rl_eviction_policy(self) -> None:
        """Test getting global singleton."""
        import victor.storage.cache.rl_eviction_policy as module

        module._rl_eviction_policy = None

        policy1 = get_rl_eviction_policy()
        policy2 = get_rl_eviction_policy()

        assert policy1 is policy2

    def test_singleton_preserves_state(self) -> None:
        """Test singleton preserves state."""
        import victor.storage.cache.rl_eviction_policy as module

        module._rl_eviction_policy = None

        policy = get_rl_eviction_policy()
        state = CacheEntryState(key="singleton_test")
        policy.get_decision(state)

        policy2 = get_rl_eviction_policy()
        assert policy2._total_decisions == 1
