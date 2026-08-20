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

"""Tests for ObservationModelLearner - learns P(agent_message | task_outcome)."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.base import RLOutcome


class TestObservationModelLearnerInit:
    """Test ObservationModelLearner initialization."""

    def test_init_with_database(self, tmp_path):
        """Test initialization with database connection."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        assert learner.name == "observation_model"
        assert learner.db == conn

    def test_init_creates_table(self, tmp_path):
        """Test that initialization creates the database table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Check that table was created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_observation_model'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "rl_observation_model"

    def test_table_schema(self, tmp_path):
        """Test that table has correct schema."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Check table structure
        cursor = conn.execute("PRAGMA table_info(rl_observation_model)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "agent_id" in columns
        assert "outcome_type" in columns
        assert "message_category" in columns
        assert "alpha" in columns
        assert "beta" in columns
        assert "sample_count" in columns
        assert "last_updated" in columns


class TestObservationRecording:
    """Test recording agent observations and learning outcomes."""

    def test_record_observation_affirm(self, tmp_path):
        """Test recording an affirming observation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Record an observation where agent said "success" and outcome was success
        learner.record_observation(
            agent_id="agent_a",
            message="The code should work",
            actual_outcome="success",
            confidence=0.9,
        )

        # Check that it was recorded
        cursor = conn.execute(
            "SELECT alpha, beta, sample_count FROM rl_observation_model "
            "WHERE agent_id='agent_a' AND outcome_type='success' AND message_category='affirm'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        assert alpha == 2.0  # Prior 1.0 + success
        assert beta == 1.0  # Prior 1.0 (no failure)
        assert count == 1

    def test_record_observation_deny(self, tmp_path):
        """Test recording a denying observation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Agent said "This won't work" (deny) and outcome was "failure" (correct prediction)
        learner.record_observation(
            agent_id="agent_a",
            message="This won't work",
            actual_outcome="failure",
            confidence=0.8,
        )

        # Check that it was recorded as deny with correct prediction
        cursor = conn.execute(
            "SELECT alpha, beta, sample_count FROM rl_observation_model "
            "WHERE agent_id='agent_a' AND outcome_type='failure' AND message_category='deny'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        assert alpha == 2.0  # Prior 1.0 + correct prediction
        assert beta == 1.0  # Prior 1.0 (no incorrect prediction)
        assert count == 1

    def test_record_multiple_observations(self, tmp_path):
        """Test recording multiple observations for same agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Record 5 successful affirmations
        for i in range(5):
            learner.record_observation(
                agent_id="agent_a",
                message="Looks good",
                actual_outcome="success",
                confidence=0.8,
            )

        # Check parameters
        cursor = conn.execute(
            "SELECT alpha, beta, sample_count FROM rl_observation_model "
            "WHERE agent_id='agent_a' AND outcome_type='success' AND message_category='affirm'"
        )
        result = cursor.fetchone()

        assert result is not None
        alpha, beta, count = result
        assert alpha == 6.0  # 1.0 + 5 successes
        assert beta == 1.0  # 1.0 prior
        assert count == 5

    def test_message_categorization(self, tmp_path):
        """Test that messages are correctly categorized."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Test different message categories
        test_cases = [
            ("Yes, that works", "affirm", "success"),
            ("That won't work", "deny", "failure"),
            ("I'm not sure", "uncertain", "success"),
            ("Error occurred", "error", "failure"),
        ]

        for message, expected_category, outcome in test_cases:
            learner.record_observation(
                agent_id="agent_a",
                message=message,
                actual_outcome=outcome,
                confidence=0.7,
            )

            # Verify category
            cursor = conn.execute(
                "SELECT message_category FROM rl_observation_model "
                "WHERE agent_id='agent_a' AND message_category=?",
                (expected_category,),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == expected_category


class TestLikelihoodComputation:
    """Test likelihood computation P(message | outcome)."""

    def test_get_likelihood_for_observed_agent(self, tmp_path):
        """Test getting likelihood for agent with observations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        # Seed RNG so the single Thompson draw is deterministic (avoids shard flake)
        learner.seed_rng(0)

        # Record 8 affirmations for success, 2 failures
        # Use "Yes, correct" which both categorizers recognize as "affirm"
        for i in range(8):
            learner.record_observation(
                agent_id="agent_a",
                message="Yes, correct",
                actual_outcome="success",
                confidence=0.8,
            )
        for i in range(2):
            learner.record_observation(
                agent_id="agent_a",
                message="Yes, correct",
                actual_outcome="failure",
                confidence=0.8,
            )

        # Get likelihood for "affirm" message given "success" outcome
        likelihood = learner.get_likelihood(
            agent_id="agent_a",
            message="Yes, correct",
            outcome="success",
        )

        # Should be high (agent is good at predicting success)
        assert likelihood > 0.7

    def test_get_likelihood_for_new_agent(self, tmp_path):
        """Test getting likelihood for agent with no observations."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        # Seed RNG so Thompson sampling is deterministic (avoids shard flake)
        learner.seed_rng(0)

        # New agent has no observations
        # Thompson sampling from Beta(1,1) gives values in [0,1] with mean 0.5
        likelihoods = []
        for _ in range(10):
            likelihood = learner.get_likelihood(
                agent_id="new_agent",
                message="Any message",
                outcome="success",
            )
            likelihoods.append(likelihood)

        # Average should be close to 0.5 (mean of Beta(1,1))
        avg_likelihood = sum(likelihoods) / len(likelihoods)
        assert 0.2 < avg_likelihood < 0.8  # Allow for Thompson sampling variance

    def test_likelihood_message_category_fallback(self, tmp_path):
        """Test likelihood computation when exact message not found."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        # Seed RNG so the Thompson draw is deterministic (avoids shard flake)
        learner.seed_rng(0)

        # Record some observations with "Yes" keyword (affirm category)
        for _ in range(5):
            learner.record_observation(
                agent_id="agent_a",
                message="Yes, this works",
                actual_outcome="success",
                confidence=0.8,
            )

        # Get likelihood for different message with same category
        likelihood = learner.get_likelihood(
            agent_id="agent_a",
            message="Yes, correct",  # Also affirm category (has "Yes")
            outcome="success",
        )

        # Should use category-level likelihood
        assert likelihood > 0.5


class TestCalibrationTracking:
    """Test agent calibration metrics and reliability scoring."""

    def test_get_agent_calibration_good_agent(self, tmp_path):
        """Test calibration for well-calibrated agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Agent with 80% success rate (8/10 correct)
        for i in range(8):
            learner.record_observation(
                agent_id="agent_a",
                message="Good prediction",
                actual_outcome="success",
                confidence=0.8,
            )
        for i in range(2):
            learner.record_observation(
                agent_id="agent_a",
                message="Good prediction",
                actual_outcome="failure",
                confidence=0.8,
            )

        calibration = learner.get_agent_calibration("agent_a")

        # Should have low calibration error (good calibration)
        assert "affirm" in calibration
        assert calibration["affirm"]["expected_prob"] > 0.7
        assert calibration["affirm"]["calibration_error"] < 0.2

    def test_get_agent_calibration_overconfident(self, tmp_path):
        """Test calibration for overconfident agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Overconfident agent: predicts 0.9 but only 50% success
        for i in range(5):
            learner.record_observation(
                agent_id="agent_b",
                message="Very confident",
                actual_outcome="success",
                confidence=0.9,
            )
        for i in range(5):
            learner.record_observation(
                agent_id="agent_b",
                message="Very confident",
                actual_outcome="failure",
                confidence=0.9,
            )

        calibration = learner.get_agent_calibration("agent_b")

        # Should have some calibration error (mixed predictions)
        # The agent is equally good/bad for success and failure outcomes
        # Error is (|0.857-1.0| + |0.143-0.0|) / 2 = 0.143
        assert calibration["affirm"]["calibration_error"] > 0.1

    def test_get_agent_calibration_new_agent(self, tmp_path):
        """Test calibration for agent with no data."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        calibration = learner.get_agent_calibration("new_agent")

        # Should return default calibration
        assert "affirm" in calibration
        assert "expected_prob" in calibration["affirm"]
        assert calibration["affirm"]["expected_prob"] == 0.5  # Uniform prior


class TestThompsonSampling:
    """Test Thompson sampling for likelihood exploration."""

    def test_thompson_sampling_likelihood(self, tmp_path):
        """Test that get_likelihood uses Thompson sampling."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        # Seed RNG so the 100-sample Thompson sequence is deterministic
        learner.seed_rng(0)

        # Record observations to establish parameters
        # Use "Yes" which both categorizers will recognize as "affirm"
        for _ in range(10):
            learner.record_observation(
                agent_id="agent_a",
                message="Yes, this works",
                actual_outcome="success",
                confidence=0.8,
            )

        # Sample likelihood multiple times
        samples = []
        for _ in range(100):
            likelihood = learner.get_likelihood(
                agent_id="agent_a",
                message="Yes, this works",
                outcome="success",
            )
            samples.append(likelihood)

        # Samples should vary (Thompson sampling)
        # But most should be high (since agent is good)
        assert max(samples) > min(samples)
        assert sum(samples) / len(samples) > 0.7  # Average should be high


class TestDatabasePersistence:
    """Test database persistence and recovery."""

    def test_persistence_across_instances(self, tmp_path):
        """Test that observations persist across learner instances."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # First instance
        learner1 = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        learner1.record_observation(
            agent_id="agent_a",
            message="Test",
            actual_outcome="success",
            confidence=0.8,
        )

        # Second instance (same database)
        learner2 = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Should be able to retrieve previous observations
        likelihood = learner2.get_likelihood(
            agent_id="agent_a",
            message="Test",
            outcome="success",
        )

        # Should have learned from first instance
        assert likelihood != 0.5  # Not uniform prior

    def test_get_recommendation(self, tmp_path):
        """Test getting RL recommendation for observation model."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )

        # Record some observations
        learner.record_observation(
            agent_id="agent_a",
            message="Test",
            actual_outcome="success",
            confidence=0.8,
        )

        # Get recommendation (should return likelihood info)
        recommendation = learner.get_recommendation(
            context={
                "agent_id": "agent_a",
                "message": "Test",
                "outcome_type": "success",
            }
        )

        assert recommendation is not None
        assert recommendation.value == "observation_model"
        assert recommendation.confidence > 0
        assert "learner_name" in recommendation.metadata
