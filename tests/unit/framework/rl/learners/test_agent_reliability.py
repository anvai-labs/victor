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

"""Tests for AgentReliabilityLearner - learns reliability weights α_i for agents."""

import sqlite3

import pytest

from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner


class TestAgentReliabilityLearnerInit:
    """Test AgentReliabilityLearner initialization."""

    def test_init_with_database(self, tmp_path):
        """Test initialization with database connection."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        assert learner.name == "agent_reliability"
        assert learner.db == conn

    def test_init_creates_table(self, tmp_path):
        """Test that initialization creates the database table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Check that table was created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_agent_reliability'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "rl_agent_reliability"

    def test_table_schema(self, tmp_path):
        """Test that table has correct schema."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Check table structure
        cursor = conn.execute("PRAGMA table_info(rl_agent_reliability)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "agent_id" in columns
        assert "alpha_reliability" in columns
        assert "beta_reliability" in columns
        assert "sample_count" in columns
        assert "last_updated" in columns


class TestReliabilityRecording:
    """Test recording agent prediction results."""

    def test_record_correct_prediction(self, tmp_path):
        """Test recording a correct prediction."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record a correct prediction
        learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=True,
            calibration_error=0.1,
        )

        # Check that it was recorded
        cursor = conn.execute(
            "SELECT alpha_reliability, beta_reliability, sample_count FROM rl_agent_reliability "
            "WHERE agent_id='agent_a'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        # Weight = 1.0 + (0.5 - 0.1) = 1.4
        # Alpha = 1.0 + 1.4 = 2.4
        assert abs(alpha - 2.4) < 0.01  # Prior 1.0 + weighted correct
        assert beta == 1.0  # Prior 1.0 (no incorrect)
        assert count == 1

    def test_record_incorrect_prediction(self, tmp_path):
        """Test recording an incorrect prediction."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record an incorrect prediction
        learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=False,
            calibration_error=0.5,
        )

        # Check that it was recorded
        cursor = conn.execute(
            "SELECT alpha_reliability, beta_reliability, sample_count FROM rl_agent_reliability "
            "WHERE agent_id='agent_a'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        assert alpha == 1.0  # Prior 1.0 (no correct)
        assert beta == 2.0  # Prior 1.0 + incorrect
        assert count == 1

    def test_record_mixed_predictions(self, tmp_path):
        """Test recording mixed correct/incorrect predictions."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record 8 correct, 2 incorrect (80% accuracy)
        for i in range(8):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )
        for i in range(2):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.3,
            )

        # Check parameters
        cursor = conn.execute(
            "SELECT alpha_reliability, beta_reliability, sample_count FROM rl_agent_reliability "
            "WHERE agent_id='agent_a'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        # Correct: weight = 1.0 + (0.5 - 0.1) = 1.4, alpha = 1.0 + 8*1.4 = 12.2
        # Incorrect: weight = 1.0 + (0.5 - 0.3) = 1.2, beta = 1.0 + 2*1.2 = 3.4
        assert abs(alpha - 12.2) < 0.1
        assert abs(beta - 3.4) < 0.1
        assert count == 10

    def test_calibration_error_influences_update(self, tmp_path):
        """Test that calibration error influences reliability updates."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record correct prediction with high error (less weight)
        learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=True,
            calibration_error=0.8,  # High error
        )

        # Record correct prediction with low error (more weight)
        learner.record_prediction_result(
            agent_id="agent_b",
            was_correct=True,
            calibration_error=0.1,  # Low error
        )

        # Agent B should have higher reliability
        cursor = conn.execute(
            "SELECT agent_id, alpha_reliability, beta_reliability FROM rl_agent_reliability "
            "WHERE agent_id IN ('agent_a', 'agent_b')"
        )
        results = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # Both should have same counts but different weights
        assert results["agent_a"][0] < results["agent_b"][0]  # Agent A has lower alpha


class TestReliabilityWeightComputation:
    """Test reliability weight computation."""

    def test_get_reliability_weight_new_agent(self, tmp_path):
        """Test getting reliability for new agent (neutral weight)."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # New agent has no data
        weight = learner.get_reliability_weight("new_agent")

        # Should return neutral weight (1.0)
        assert weight == 1.0

    def test_get_reliability_weight_reliable_agent(self, tmp_path):
        """Test getting reliability for reliable agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record 90% accuracy (9 correct, 1 incorrect)
        for i in range(9):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )
        learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=False,
            calibration_error=0.3,
        )

        # Get reliability weight
        weight = learner.get_reliability_weight("agent_a")

        # Should be > 1.0 (upweight reliable agent)
        assert weight > 1.0

    def test_get_reliability_weight_unreliable_agent(self, tmp_path):
        """Test getting reliability for unreliable agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record 30% accuracy (3 correct, 7 incorrect)
        for i in range(3):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )
        for i in range(7):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.5,
            )

        # Get reliability weight
        weight = learner.get_reliability_weight("agent_a")

        # Should be < 1.0 (downweight unreliable agent)
        assert weight < 1.0

    def test_reliability_weight_thompson_sampling(self, tmp_path):
        """Test that reliability weight uses Thompson sampling."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record some observations
        for i in range(5):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )

        # Sample reliability multiple times
        weights = []
        for _ in range(100):
            weight = learner.get_reliability_weight("agent_a")
            weights.append(weight)

        # Samples should vary (Thompson sampling)
        assert max(weights) > min(weights)
        # But most should be > 1.0 (agent is reliable)
        assert sum(weights) / len(weights) > 1.0

    def test_reliability_weight_sampling_does_not_cross_neutral_boundary(self, tmp_path):
        """Sampled weights should not invert the learned reliability direction."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        for _ in range(3):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )
        for _ in range(7):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.5,
            )

        weights = [learner.get_reliability_weight("agent_a") for _ in range(100)]

        assert all(weight <= 1.0 for weight in weights)
        assert min(weights) < max(weights)

    def test_get_expected_reliability_weight_is_deterministic(self, tmp_path):
        """Posterior-mean reliability weight should be stable for ranking decisions."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        for _ in range(5):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )

        weights = [learner.get_expected_reliability_weight("agent_a") for _ in range(10)]

        assert len(set(weights)) == 1
        assert weights[0] > 1.0


class TestReliabilityStatistics:
    """Test reliability statistics and metrics."""

    def test_get_reliability_stats(self, tmp_path):
        """Test getting reliability statistics for an agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record observations
        for i in range(7):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )
        for i in range(3):
            learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.4,
            )

        stats = learner.get_agent_reliability_stats("agent_a")

        assert stats is not None
        assert "alpha" in stats
        assert "beta" in stats
        assert "sample_count" in stats
        assert "expected_reliability" in stats
        assert stats["sample_count"] == 10
        assert stats["expected_reliability"] > 0.5  # Should be > 0.5 for 70% accuracy

    def test_get_all_reliability_stats(self, tmp_path):
        """Test getting statistics for all agents."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record observations for multiple agents
        learner.record_prediction_result("agent_a", True, 0.1)
        learner.record_prediction_result("agent_b", False, 0.5)
        learner.record_prediction_result("agent_c", True, 0.1)

        all_stats = learner.get_all_agent_stats()

        assert len(all_stats) == 3
        assert "agent_a" in all_stats
        assert "agent_b" in all_stats
        assert "agent_c" in all_stats


class TestPersistence:
    """Test database persistence and recovery."""

    def test_persistence_across_instances(self, tmp_path):
        """Test that reliability data persists across learner instances."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # First instance
        learner1 = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        learner1.record_prediction_result(
            agent_id="agent_a",
            was_correct=True,
            calibration_error=0.1,
        )

        # Second instance (same database)
        learner2 = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Should be able to retrieve previous data
        weight = learner2.get_reliability_weight("agent_a")

        # Should have learned from first instance
        assert weight != 1.0  # Not neutral prior

    def test_get_recommendation(self, tmp_path):
        """Test getting RL recommendation for reliability learner."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        # Record some observations
        learner.record_prediction_result(
            agent_id="agent_a",
            was_correct=True,
            calibration_error=0.1,
        )

        # Get recommendation
        recommendation = learner.get_recommendation(
            context={
                "agent_id": "agent_a",
            }
        )

        assert recommendation is not None
        assert recommendation.value == "agent_reliability"
        assert recommendation.confidence > 0
        assert "learner_name" in recommendation.metadata
