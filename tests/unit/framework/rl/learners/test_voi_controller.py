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

"""Tests for VoIController - Expected Value of Information computation."""

import sqlite3

import pytest

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.voi_controller import VoIController
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType


class TestVoIControllerInit:
    """Test VoIController initialization."""

    def test_init_with_learners(self, tmp_path):
        """Test initialization with observation and reliability learners."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        assert controller.name == "voi_controller"
        assert controller.observation_learner == observation_learner
        assert controller.reliability_learner == reliability_learner
        assert controller.db == conn

    def test_init_creates_table(self, tmp_path):
        """Test that initialization creates the database table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Check that table was created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_voi_history'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "rl_voi_history"


class TestVoIComputation:
    """Test Value of Information computation."""

    def test_compute_voi_high_uncertainty(self, tmp_path):
        """Test VoI computation when uncertainty is high."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Create task analysis with high uncertainty
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Compute VoI
        voi = controller.compute_voi(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=0.1,
        )

        # VoI should be positive (high uncertainty)
        assert voi > 0

    def test_compute_voi_low_uncertainty(self, tmp_path):
        """Test VoI computation when uncertainty is low."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Create task analysis with low uncertainty
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.99, "failure": 0.01},
        )

        # Compute VoI
        voi = controller.compute_voi(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=0.1,
        )

        # VoI should be low (low uncertainty)
        assert voi < 0.2

    def test_compute_voi_with_reliable_agent(self, tmp_path):
        """Test VoI computation with reliable agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Train agent to be reliable
        for i in range(9):
            reliability_learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.1,
            )

        # Create task analysis with uncertainty
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Compute VoI
        voi = controller.compute_voi(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=0.1,
        )

        # VoI should be higher for reliable agent
        assert voi > 0.2

    def test_compute_voi_with_unreliable_agent(self, tmp_path):
        """Test VoI computation with unreliable agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Train agent to be unreliable
        for i in range(7):
            reliability_learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.5,
            )
        for i in range(3):
            reliability_learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=True,
                calibration_error=0.3,
            )

        # Create task analysis with uncertainty
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Compute VoI
        voi = controller.compute_voi(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=0.1,
        )

        # VoI should be lower for unreliable agent
        assert voi < 0.3


class TestQueryDecisions:
    """Test query decision making."""

    def test_should_query_positive_voi(self, tmp_path):
        """Test decision when VoI is positive."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Should query when VoI > cost
        should_query = controller.should_query(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=0.05,  # Low cost
        )

        assert should_query is True

    def test_should_query_negative_voi(self, tmp_path):
        """Test decision when VoI is negative."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.99, "failure": 0.01},  # Low uncertainty
        )

        # Should not query when cost > VoI
        should_query = controller.should_query(
            task_analysis=analysis,
            agent_id="agent_a",
            query_cost=1.0,  # High cost
        )

        assert should_query is False

    def test_rank_agents_by_voi(self, tmp_path):
        """Test ranking multiple agents by VoI."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Train agents with different reliabilities
        for i in range(9):
            reliability_learner.record_prediction_result("agent_a", True, 0.1)
        for i in range(5):
            reliability_learner.record_prediction_result("agent_b", True, 0.2)
        for i in range(3):
            reliability_learner.record_prediction_result("agent_c", True, 0.3)

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Rank agents
        rankings = controller.rank_agents_by_voi(
            task_analysis=analysis,
            agent_ids=["agent_a", "agent_b", "agent_c"],
            query_cost=0.1,
        )

        # Agent A should be ranked highest (most reliable)
        assert rankings[0]["agent_id"] == "agent_a"
        assert rankings[0]["voi"] > rankings[1]["voi"]


class TestVoITracking:
    """Test VoI history tracking."""

    def test_record_query_outcome(self, tmp_path):
        """Test recording query outcome for learning."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Record query outcome
        controller.record_query_outcome(
            agent_id="agent_a",
            predicted_voi=0.5,
            actual_information_gain=0.6,
            query_cost=0.1,
            was_beneficial=True,
        )

        # Check that it was recorded
        cursor = conn.execute(
            "SELECT predicted_voi, actual_information_gain, was_beneficial FROM rl_voi_history "
            "WHERE agent_id='agent_a'"
        )
        result = cursor.fetchone()

        assert result is not None
        predicted_voi, actual_gain, beneficial = result
        assert abs(predicted_voi - 0.5) < 0.01
        assert abs(actual_gain - 0.6) < 0.01
        assert beneficial == 1  # SQLite stores boolean as integer

    def test_get_voi_statistics(self, tmp_path):
        """Test getting VoI statistics for an agent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        # Record some outcomes
        controller.record_query_outcome("agent_a", 0.5, 0.6, 0.1, True)
        controller.record_query_outcome("agent_a", 0.4, 0.3, 0.1, True)
        controller.record_query_outcome("agent_a", 0.6, 0.2, 0.1, False)

        # Get statistics
        stats = controller.get_agent_voi_stats("agent_a")

        assert stats is not None
        assert stats["query_count"] == 3
        assert stats["beneficial_count"] == 2
        assert "avg_predicted_voi" in stats
        assert "avg_actual_gain" in stats


class TestRecommendations:
    """Test RL recommendations."""

    def test_get_recommendation(self, tmp_path):
        """Test getting VoI recommendation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        observation_learner = ObservationModelLearner(
            name="observation_model",
            db_connection=conn,
        )
        reliability_learner = AgentReliabilityLearner(
            name="agent_reliability",
            db_connection=conn,
        )

        controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Get recommendation
        recommendation = controller.get_recommendation(
            context={
                "task_analysis": analysis,
                "agent_id": "agent_a",
                "query_cost": 0.1,
            }
        )

        assert recommendation is not None
        assert recommendation.value == "voi_controller"
        assert "voi" in recommendation.metadata
