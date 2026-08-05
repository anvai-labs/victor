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

"""Tests for correlation tracking for dependence-aware pooling."""

import sqlite3

import pytest

from victor.framework.rl.learners.correlation_tracker import CorrelationTracker


class TestCorrelationTrackerInit:
    """Test CorrelationTracker initialization."""

    def test_init_with_db(self, tmp_path):
        """Test initialization with database connection."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        assert tracker.name == "correlation_tracker"
        assert tracker.db == conn

    def test_init_creates_table(self, tmp_path):
        """Test that initialization creates correlation table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Check that table was created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_agent_correlations'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "rl_agent_correlations"


class TestPredictionPairRecording:
    """Test recording of prediction pairs."""

    def test_record_agreeing_predictions(self, tmp_path):
        """Test recording a pair of agreeing predictions."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record agreeing predictions
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        # Check database
        cursor = conn.execute(
            "SELECT agreement_count, disagreement_count, total_pairs, correlation_coefficient "
            "FROM rl_agent_correlations "
            "WHERE agent_id_1 = 'agent_a' AND agent_id_2 = 'agent_b'"
        )

        row = cursor.fetchone()
        assert row is not None
        agreement_count, disagreement_count, total_pairs, correlation = row

        assert agreement_count == 1
        assert disagreement_count == 0
        assert total_pairs == 1
        assert correlation == 1.0  # Perfect agreement

    def test_record_disagreeing_predictions(self, tmp_path):
        """Test recording a pair of disagreeing predictions."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record disagreeing predictions
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="failure",
            actual_outcome="success",
        )

        # Check database
        cursor = conn.execute(
            "SELECT agreement_count, disagreement_count, total_pairs, correlation_coefficient "
            "FROM rl_agent_correlations "
            "WHERE agent_id_1 = 'agent_a' AND agent_id_2 = 'agent_b'"
        )

        row = cursor.fetchone()
        assert row is not None
        agreement_count, disagreement_count, total_pairs, correlation = row

        assert agreement_count == 0
        assert disagreement_count == 1
        assert total_pairs == 1
        assert correlation == -1.0  # Perfect disagreement

    def test_record_multiple_pairs(self, tmp_path):
        """Test recording multiple prediction pairs."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record multiple pairs
        for _ in range(7):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        for _ in range(3):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="failure",
                actual_outcome="success",
            )

        # Check database
        cursor = conn.execute(
            "SELECT agreement_count, disagreement_count, total_pairs, correlation_coefficient "
            "FROM rl_agent_correlations "
            "WHERE agent_id_1 = 'agent_a' AND agent_id_2 = 'agent_b'"
        )

        row = cursor.fetchone()
        assert row is not None
        agreement_count, disagreement_count, total_pairs, correlation = row

        assert agreement_count == 7
        assert disagreement_count == 3
        assert total_pairs == 10
        assert correlation == 0.4  # (7 - 3) / 10

    def test_normalizes_agent_order(self, tmp_path):
        """Test that agent order is normalized to avoid duplicates."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record with different orders
        tracker.record_prediction_pair(
            agent_id_1="agent_b",
            agent_id_2="agent_a",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        # Check that only one record exists
        cursor = conn.execute(
            "SELECT COUNT(*) FROM rl_agent_correlations "
            "WHERE agent_id_1 = 'agent_a' AND agent_id_2 = 'agent_b'"
        )

        count = cursor.fetchone()[0]
        assert count == 1

        # Check that it has both recordings
        cursor = conn.execute(
            "SELECT total_pairs FROM rl_agent_correlations "
            "WHERE agent_id_1 = 'agent_a' AND agent_id_2 = 'agent_b'"
        )

        total_pairs = cursor.fetchone()[0]
        assert total_pairs == 2


class TestCorrelationRetrieval:
    """Test correlation coefficient retrieval."""

    def test_get_correlation_exists(self, tmp_path):
        """Test getting correlation for existing pair."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record some pairs
        for _ in range(8):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        for _ in range(2):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="failure",
                actual_outcome="success",
            )

        # Get correlation
        correlation = tracker.get_correlation("agent_a", "agent_b")

        assert correlation == 0.6  # (8 - 2) / 10

    def test_get_correlation_not_exists(self, tmp_path):
        """Test getting correlation for non-existent pair."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Get correlation for pair that doesn't exist
        correlation = tracker.get_correlation("agent_a", "agent_b")

        assert correlation == 0.0  # Default to no correlation

    def test_get_correlation_order_independent(self, tmp_path):
        """Test that get_correlation is order-independent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record pairs
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        # Get correlation in both orders
        correlation_1 = tracker.get_correlation("agent_a", "agent_b")
        correlation_2 = tracker.get_correlation("agent_b", "agent_a")

        assert correlation_1 == correlation_2


class TestCorrelationMatrix:
    """Test correlation matrix computation."""

    def test_get_correlation_matrix(self, tmp_path):
        """Test getting correlation matrix for multiple agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record pairs for three agents
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        tracker.record_prediction_pair(
            agent_id_1="agent_b",
            agent_id_2="agent_c",
            prediction_1="success",
            prediction_2="failure",
            actual_outcome="success",
        )

        # Get correlation matrix
        matrix = tracker.get_correlation_matrix(["agent_a", "agent_b", "agent_c"])

        # Check self-correlations
        assert matrix["agent_a"]["agent_a"] == 1.0
        assert matrix["agent_b"]["agent_b"] == 1.0
        assert matrix["agent_c"]["agent_c"] == 1.0

        # Check pairwise correlations
        assert matrix["agent_a"]["agent_b"] == 1.0  # Agreed
        assert matrix["agent_b"]["agent_a"] == 1.0

        assert matrix["agent_b"]["agent_c"] == -1.0  # Disagreed
        assert matrix["agent_c"]["agent_b"] == -1.0

        # agent_a and agent_c have no data
        assert matrix["agent_a"]["agent_c"] == 0.0
        assert matrix["agent_c"]["agent_a"] == 0.0

    def test_get_correlation_matrix_empty(self, tmp_path):
        """Test getting correlation matrix with no agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        matrix = tracker.get_correlation_matrix([])

        assert matrix == {}


class TestEffectiveSampleSize:
    """Test effective sample size computation."""

    def test_compute_effective_sample_size_independent(self, tmp_path):
        """Test ESS with independent agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Independent agents (no correlation data)
        agent_ids = ["agent_a", "agent_b", "agent_c"]
        weights = {"agent_a": 1.0, "agent_b": 1.0, "agent_c": 1.0}

        ess = tracker.compute_effective_sample_size(agent_ids, weights)

        # ESS should equal sum of weights (3.0)
        assert ess == 3.0

    def test_compute_effective_sample_size_correlated(self, tmp_path):
        """Test ESS with correlated agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Create highly correlated agents
        for _ in range(10):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        agent_ids = ["agent_a", "agent_b", "agent_c"]
        weights = {"agent_a": 1.0, "agent_b": 1.0, "agent_c": 1.0}

        ess = tracker.compute_effective_sample_size(agent_ids, weights)

        # ESS should be less than sum of weights due to correlation
        assert ess < 3.0
        assert ess > 1.0  # But still positive

    def test_compute_effective_sample_size_empty(self, tmp_path):
        """Test ESS with empty agent list."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        ess = tracker.compute_effective_sample_size([], {})

        assert ess == 0.0


class TestAdjustedReliabilityWeights:
    """Test reliability weight adjustment for correlations."""

    def test_get_adjusted_weights_independent(self, tmp_path):
        """Test weight adjustment with independent agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Independent agents (no correlation data)
        agent_ids = ["agent_a", "agent_b", "agent_c"]
        base_weights = {"agent_a": 1.5, "agent_b": 1.2, "agent_c": 0.8}

        adjusted = tracker.get_adjusted_reliability_weights(agent_ids, base_weights)

        # Weights should remain the same
        assert adjusted == base_weights

    def test_get_adjusted_weights_correlated(self, tmp_path):
        """Test weight adjustment with correlated agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Create highly correlated agents
        for _ in range(10):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        agent_ids = ["agent_a", "agent_b", "agent_c"]
        base_weights = {"agent_a": 1.5, "agent_b": 1.2, "agent_c": 0.8}

        adjusted = tracker.get_adjusted_reliability_weights(agent_ids, base_weights)

        # agent_a and agent_b should be downweighted
        assert adjusted["agent_a"] < base_weights["agent_a"]
        assert adjusted["agent_b"] < base_weights["agent_b"]

        # agent_c should remain the same
        assert adjusted["agent_c"] == base_weights["agent_c"]

    def test_get_adjusted_weights_minimum(self, tmp_path):
        """Test that adjusted weights have minimum value."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Create perfectly correlated agents
        for _ in range(10):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        agent_ids = ["agent_a", "agent_b"]
        base_weights = {"agent_a": 1.0, "agent_b": 1.0}

        adjusted = tracker.get_adjusted_reliability_weights(agent_ids, base_weights)

        # Both should be downweighted but not below minimum
        assert all(weight >= 0.1 for weight in adjusted.values())


class TestCorrelationStatistics:
    """Test correlation statistics retrieval."""

    def test_get_correlation_stats_all(self, tmp_path):
        """Test getting correlation statistics for all agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record some pairs
        for _ in range(7):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        for _ in range(3):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="failure",
                actual_outcome="success",
            )

        # Get stats
        stats = tracker.get_correlation_stats()

        assert "agent_a" in stats
        assert "agent_b" in stats

        assert "agent_b" in stats["agent_a"]["correlations"]
        assert stats["agent_a"]["correlations"]["agent_b"]["correlation"] == 0.4  # (7-3)/10
        assert stats["agent_a"]["correlations"]["agent_b"]["agreement_rate"] == 0.7
        assert stats["agent_a"]["correlations"]["agent_b"]["total_pairs"] == 10

    def test_get_correlation_stats_single_agent(self, tmp_path):
        """Test getting correlation statistics for a specific agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record pairs with agent_a
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_c",
            prediction_1="success",
            prediction_2="failure",
            actual_outcome="success",
        )

        # Get stats for agent_a
        stats = tracker.get_correlation_stats("agent_a")

        assert "agent_a" in stats
        assert "agent_b" in stats["agent_a"]["correlations"]
        assert "agent_c" in stats["agent_a"]["correlations"]

    def test_get_highly_correlated_pairs(self, tmp_path):
        """Test getting highly correlated agent pairs."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Create highly correlated pair
        for _ in range(10):
            tracker.record_prediction_pair(
                agent_id_1="agent_a",
                agent_id_2="agent_b",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )

        # Create moderately correlated pair
        for _ in range(7):
            tracker.record_prediction_pair(
                agent_id_1="agent_b",
                agent_id_2="agent_c",
                prediction_1="success",
                prediction_2="success",
                actual_outcome="success",
            )
        for _ in range(3):
            tracker.record_prediction_pair(
                agent_id_1="agent_b",
                agent_id_2="agent_c",
                prediction_1="success",
                prediction_2="failure",
                actual_outcome="success",
            )

        # Get highly correlated pairs (threshold 0.7)
        highly_correlated = tracker.get_highly_correlated_pairs(threshold=0.7)

        # Should find agent_a-agent_b (correlation 1.0)
        assert ("agent_a", "agent_b", 1.0) in highly_correlated

        # Should not find agent_b-agent_c (correlation 0.4)
        assert not any("agent_b" in pair and "agent_c" in pair for pair in highly_correlated)


class TestCleanup:
    """Test cleanup of old correlation data."""

    def test_cleanup_old_correlations(self, tmp_path):
        """Test cleaning up old correlation records."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        tracker = CorrelationTracker(
            name="correlation_tracker",
            db_connection=conn,
        )

        # Record some pairs
        tracker.record_prediction_pair(
            agent_id_1="agent_a",
            agent_id_2="agent_b",
            prediction_1="success",
            prediction_2="success",
            actual_outcome="success",
        )

        # Clean up very old records (100 days)
        # This should not delete the recent record
        deleted = tracker.cleanup_old_correlations(days_old=100)

        # Record should still exist
        cursor = conn.execute("SELECT COUNT(*) FROM rl_agent_correlations")
        count = cursor.fetchone()[0]

        assert count == 1
        assert deleted == 0  # Nothing old enough to delete
