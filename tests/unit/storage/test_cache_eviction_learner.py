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

"""Unit tests for cache eviction RL learner.

Tests the CacheEvictionLearner which uses Q-learning to learn
optimal cache eviction decisions.
"""

import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.cache_eviction import (
    CacheEvictionLearner,
    CacheEvictionAction,
)
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    """Fixture for RLCoordinator, ensuring a clean database for each test."""
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> CacheEvictionLearner:
    """Fixture for CacheEvictionLearner."""
    return coordinator.get_learner("cache_eviction")  # type: ignore


def _record_eviction_outcome(
    learner: CacheEvictionLearner,
    state_key: str = "medium:recent:low:search",
    action: str = CacheEvictionAction.KEEP,
    tool_name: str = "code_search",
    *,
    hit_after: bool = True,
    memory_saved: int = 0,
    latency_delta: float = 0.0,
) -> None:
    """Helper to record a cache eviction outcome."""
    outcome = RLOutcome(
        provider="cache",
        model="tiered_cache",
        task_type="eviction",
        success=hit_after if action == CacheEvictionAction.KEEP else True,
        quality_score=1.0 if hit_after else 0.5,
        metadata={
            "state_key": state_key,
            "action": action,
            "tool_name": tool_name,
            "hit_after": 1 if hit_after else 0,
            "memory_saved": memory_saved,
            "latency_delta": latency_delta,
        },
    )
    learner.record_outcome(outcome)


def _get_q_value_from_db(
    coordinator: RLCoordinator,
    state_key: str,
    action: str,
) -> Tuple[float, int]:
    """Helper to retrieve Q-value and visit count from the database."""
    cursor = coordinator.db.cursor()
    cursor.execute(
        f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
        f"WHERE learner_id = 'cache_eviction' AND state_key = ? AND action_key = ?",
        (state_key, action),
    )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0.0, 0)


class TestCacheEvictionLearner:
    """Tests for CacheEvictionLearner."""

    def test_initialization(self, learner: CacheEvictionLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "cache_eviction"
        assert learner.learning_rate == 0.1
        assert learner.discount_factor == 0.95
        assert learner.epsilon == 0.1

        # Check tables using correct names from Tables enum
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_Q_VALUE}';"
        )
        assert cursor.fetchone() is not None, f"Table {Tables.RL_Q_VALUE} not found"
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TASK_STAT}';"
        )
        assert cursor.fetchone() is not None, f"Table {Tables.RL_TASK_STAT} not found"
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TRANSITION}';"
        )
        assert cursor.fetchone() is not None, f"Table {Tables.RL_TRANSITION} not found"

    def test_record_single_outcome(
        self, coordinator: RLCoordinator, learner: CacheEvictionLearner
    ) -> None:
        """Recording one outcome updates Q-values and counts."""
        state_key = "low:fresh:zero:search"
        action = CacheEvictionAction.KEEP

        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=action,
            hit_after=True,
        )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action)
        assert count == 1
        assert q_value != 0.0  # Should have been updated from default

    def test_q_learning_update(
        self, coordinator: RLCoordinator, learner: CacheEvictionLearner
    ) -> None:
        """Q-values update correctly with repeated outcomes."""
        state_key = "high:warm:medium:search"
        action = CacheEvictionAction.KEEP

        # Record multiple successful keeps (cache hits)
        for _ in range(5):
            _record_eviction_outcome(
                learner,
                state_key=state_key,
                action=action,
                hit_after=True,
            )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action)
        assert count == 5
        assert q_value > 0  # Positive reward for hits should increase Q-value

    def test_eviction_reward(
        self, coordinator: RLCoordinator, learner: CacheEvictionLearner
    ) -> None:
        """Eviction with memory savings has positive reward."""
        state_key = "critical:stale:zero:other"
        action = CacheEvictionAction.EVICT

        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=action,
            hit_after=False,
            memory_saved=500000,  # 500KB saved
        )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action)
        assert count == 1
        assert q_value >= 0  # Memory savings contribute positively

    def test_keep_without_hit_penalty(
        self, coordinator: RLCoordinator, learner: CacheEvictionLearner
    ) -> None:
        """Keeping an entry that doesn't get hit should decrease Q-value."""
        state_key = "high:warm:low:compute"
        action = CacheEvictionAction.KEEP

        # Record keeps without subsequent hits (wasted space)
        for _ in range(5):
            _record_eviction_outcome(
                learner,
                state_key=state_key,
                action=action,
                hit_after=False,  # No hit after keeping
            )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action)
        assert count == 5
        assert q_value < 0  # Should be penalized for keeping unused entries

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        state_key = "medium:recent:low:search"
        action = CacheEvictionAction.KEEP

        reset_database()
        db_path = tmp_path / "rl_test.db"
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("cache_eviction")  # type: ignore

        _record_eviction_outcome(learner1, state_key=state_key, action=action)
        reset_database()

        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("cache_eviction")  # type: ignore

        q_value, count = _get_q_value_from_db(coordinator2, state_key, action)
        assert count == 1
        assert q_value != 0.0

        # Check state was loaded correctly
        assert state_key in learner2._q_values
        assert action in learner2._q_values[state_key]

        reset_database()

    def test_get_recommendation_exploitation(self, learner: CacheEvictionLearner) -> None:
        """Test get_recommendation returns best action in exploitation mode."""
        state_key = "high:stale:zero:other"

        # Record outcomes for different actions
        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=CacheEvictionAction.EVICT,
            hit_after=False,
            memory_saved=100000,
        )
        # Record multiple evictions to build up confidence
        for _ in range(3):
            _record_eviction_outcome(
                learner,
                state_key=state_key,
                action=CacheEvictionAction.EVICT,
                hit_after=False,
                memory_saved=100000,
            )

        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=CacheEvictionAction.KEEP,
            hit_after=False,  # Keep but no hit = bad
        )

        # Force exploitation (epsilon=0)
        learner.epsilon = 0.0
        rec = learner.get_recommendation(state_key, "", "eviction")

        assert rec is not None
        assert rec.value == CacheEvictionAction.EVICT  # Higher Q-value action

    def test_get_recommendation_exploration(self, learner: CacheEvictionLearner) -> None:
        """Test get_recommendation can explore with high epsilon."""
        state_key = "medium:recent:low:search"

        _record_eviction_outcome(learner, state_key=state_key, action=CacheEvictionAction.KEEP)

        # Force exploration (epsilon=1.0 always explores; control the learner's
        # injectable RNG rather than the stdlib global).
        learner.epsilon = 1.0
        with patch.object(learner._rng, "random", return_value=0.5):
            rec = learner.get_recommendation(state_key, "", "eviction")

        assert rec is not None
        assert rec.is_baseline is True
        assert "Exploration" in rec.reason

    def test_get_eviction_decision(self, learner: CacheEvictionLearner) -> None:
        """Test convenience method for eviction decision."""
        action, confidence = learner.get_eviction_decision(
            utilization=0.8,
            age_seconds=120,
            hit_count=2,
            tool_name="code_search",
        )

        assert action in learner.ACTIONS
        assert 0.0 <= confidence <= 1.0

    def test_tool_value_tracking(self, learner: CacheEvictionLearner) -> None:
        """Test that tool values are tracked correctly."""
        tool_name = "semantic_code_search"

        # Record multiple outcomes with hits
        for _ in range(5):
            _record_eviction_outcome(
                learner,
                tool_name=tool_name,
                action=CacheEvictionAction.KEEP,
                hit_after=True,
            )

        value = learner.get_tool_value(tool_name)
        assert value == 1.0  # All hits, max value

        # Record some misses
        for _ in range(5):
            _record_eviction_outcome(
                learner,
                tool_name=tool_name,
                action=CacheEvictionAction.KEEP,
                hit_after=False,
            )

        value = learner.get_tool_value(tool_name)
        assert value == 0.5  # 5 hits, 5 misses = 50%

    def test_discretization(self, learner: CacheEvictionLearner) -> None:
        """Test state discretization functions."""
        # Utilization
        assert learner._discretize_utilization(0.3) == "low"
        assert learner._discretize_utilization(0.6) == "medium"
        assert learner._discretize_utilization(0.8) == "high"
        assert learner._discretize_utilization(0.95) == "critical"

        # Age
        assert learner._discretize_age(30) == "fresh"
        assert learner._discretize_age(120) == "recent"
        assert learner._discretize_age(1800) == "warm"
        assert learner._discretize_age(7200) == "stale"

        # Hits
        assert learner._discretize_hits(0) == "zero"
        assert learner._discretize_hits(1) == "low"
        assert learner._discretize_hits(5) == "medium"
        assert learner._discretize_hits(15) == "high"

    def test_tool_type_categorization(self, learner: CacheEvictionLearner) -> None:
        """Test tool type categorization."""
        assert learner._get_tool_type("code_search") == "search"
        assert learner._get_tool_type("semantic_code_search") == "search"
        assert learner._get_tool_type("read") == "read"
        assert learner._get_tool_type("list_directory") == "read"
        assert learner._get_tool_type("code_review") == "compute"
        assert learner._get_tool_type("unknown_tool") == "other"

    def test_compute_reward(self, learner: CacheEvictionLearner) -> None:
        """Test reward computation."""
        # Keep with hit = positive
        outcome_hit = RLOutcome(
            provider="cache",
            model="tiered_cache",
            task_type="eviction",
            success=True,
            quality_score=1.0,
            metadata={
                "action": CacheEvictionAction.KEEP,
                "hit_after": 1,
                "memory_saved": 0,
                "latency_delta": 0,
            },
        )
        reward = learner._compute_reward(outcome_hit)
        assert reward > 0  # Should be positive for hit

        # Keep without hit = negative
        outcome_miss = RLOutcome(
            provider="cache",
            model="tiered_cache",
            task_type="eviction",
            success=False,
            quality_score=0.5,
            metadata={
                "action": CacheEvictionAction.KEEP,
                "hit_after": 0,
                "memory_saved": 0,
                "latency_delta": 0,
            },
        )
        reward = learner._compute_reward(outcome_miss)
        assert reward < 0  # Should be negative for miss

    def test_export_metrics(self, learner: CacheEvictionLearner) -> None:
        """Test metrics export."""
        _record_eviction_outcome(learner, action=CacheEvictionAction.KEEP, tool_name="code_search")
        _record_eviction_outcome(learner, action=CacheEvictionAction.EVICT, tool_name="web_search")

        metrics = learner.export_metrics()

        assert metrics["learner"] == "cache_eviction"
        assert metrics["total_decisions"] == 2
        assert metrics["epsilon"] == 0.1
        assert metrics["learning_rate"] == 0.1
        assert metrics["discount_factor"] == 0.95

    def test_no_data_returns_baseline(self, learner: CacheEvictionLearner) -> None:
        """Test that unknown state returns baseline recommendation."""
        rec = learner.get_recommendation("unknown:state:key", "", "eviction")

        assert rec is not None
        assert rec.is_baseline is True
        assert rec.value == CacheEvictionAction.KEEP  # Default to keep
        assert rec.confidence == 0.3
        assert rec.sample_size == 0

    def test_promote_demote_actions(
        self, coordinator: RLCoordinator, learner: CacheEvictionLearner
    ) -> None:
        """Test promote and demote actions work correctly."""
        state_key = "low:fresh:high:search"

        # Record promote with hit = very positive
        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=CacheEvictionAction.PROMOTE_L1,
            hit_after=True,
        )

        q_value, count = _get_q_value_from_db(
            coordinator, state_key, CacheEvictionAction.PROMOTE_L1
        )
        assert count == 1
        assert q_value > 0  # Promotion with hit is good

        # Record demote with hit = still ok
        _record_eviction_outcome(
            learner,
            state_key=state_key,
            action=CacheEvictionAction.DEMOTE_L2,
            hit_after=True,
        )

        q_value, count = _get_q_value_from_db(coordinator, state_key, CacheEvictionAction.DEMOTE_L2)
        assert count == 1
        assert q_value >= 0  # Demotion with hit is acceptable
