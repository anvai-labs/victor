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

"""Tests for Bayesian orchestration integration."""

import sqlite3

import pytest

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType
from victor.framework.rl.orchestration.bayesian_orchestrator import (
    BayesianOrchestrationService,
)
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.voi_controller import VoIController


class TestBayesianOrchestrationServiceInit:
    """Test Bayesian orchestration service initialization."""

    def test_init_with_learners(self, tmp_path):
        """Test initialization with all required learners."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        assert service.observation_learner == observation_learner
        assert service.reliability_learner == reliability_learner
        assert service.voi_controller == voi_controller

    def test_init_creates_tables(self, tmp_path):
        """Test that initialization creates required tables."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Check that belief history table was created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rl_belief_history'"
        )
        tables = cursor.fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "rl_belief_history"


class TestBeliefStateManagement:
    """Test belief state creation and tracking."""

    def test_create_belief_state(self, tmp_path):
        """Test creating a belief state for a task."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        assert belief is not None
        assert belief.outcome_belief == {"success": 0.5, "failure": 0.5}
        assert belief.belief_entropy > 0
        assert belief.complexity == TaskComplexity.SIMPLE

    def test_get_belief_state(self, tmp_path):
        """Test retrieving a belief state."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.6, "failure": 0.4},
        )

        # Retrieve belief state
        retrieved = service.get_belief_state(belief_id=belief.belief_id)

        assert retrieved is not None
        assert retrieved.outcome_belief == {"success": 0.6, "failure": 0.4}


class TestBayesianUpdates:
    """Test Bayesian posterior updates."""

    def test_update_belief_with_agent_message(self, tmp_path):
        """Test updating belief with agent message."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Update belief with agent message
        updated_belief = service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            message="This will work",
            confidence=0.8,
        )

        # Belief should have shifted
        assert updated_belief is not None
        # Success probability should have increased (agent affirmed)
        assert updated_belief.outcome_belief["success"] > 0.5

    def test_update_belief_with_reliability_weighting(self, tmp_path):
        """Test that reliability weights affect belief updates."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Train agent to be unreliable
        for i in range(7):
            reliability_learner.record_prediction_result(
                agent_id="agent_a",
                was_correct=False,
                calibration_error=0.5,
            )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # get_reliability_weight does Thompson sampling from the learner's
        # injectable RNG (self._rng). Seed it so the sampled weight — and thus
        # the belief shift — is deterministic (no flake, no stdlib patching).
        # seed=1 yields success≈0.67, comfortably under the 0.75 threshold.
        reliability_learner.seed_rng(1)

        # Update with unreliable agent
        updated_belief = service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            message="This will work",
            confidence=0.9,
        )

        # Belief should shift less due to low reliability
        # (compared to a reliable agent)
        # With 7 incorrect predictions, reliability is low but not zero
        assert updated_belief.outcome_belief["success"] < 0.75


class TestVoIDecisions:
    """Test Value of Information decision making."""

    def test_should_query_agent(self, tmp_path):
        """Test decision about whether to query an agent."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state with high uncertainty
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Should query when uncertainty is high
        should_query = service.should_query_agent(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            query_cost=0.1,
        )

        assert should_query is True

    def test_select_best_agent_to_query(self, tmp_path):
        """Test selecting the best agent to query."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Train agents with different reliabilities
        for i in range(9):
            reliability_learner.record_prediction_result("agent_a", True, 0.1)
        for i in range(5):
            reliability_learner.record_prediction_result("agent_b", True, 0.2)

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Select best agent
        best_agent = service.select_best_agent_to_query(
            belief_id=belief.belief_id,
            agent_ids=["agent_a", "agent_b"],
            query_cost=0.1,
        )

        # Should select agent_a (more reliable)
        assert best_agent == "agent_a"


class TestOutcomeRecording:
    """Test recording task outcomes for learning."""

    def test_record_task_outcome(self, tmp_path):
        """Test recording task outcome."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Record task outcome
        service.record_task_outcome(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            actual_outcome="success",
            agent_message="This will work",
            agent_confidence=0.8,
        )

        # Check that observation was recorded
        stats = observation_learner.get_calibration_stats()
        assert "agent_a" in stats

        # Check that reliability was updated
        reliability_stats = reliability_learner.get_agent_reliability_stats("agent_a")
        assert reliability_stats is not None


class TestBeliefHistory:
    """Test belief state history tracking."""

    def test_belief_history_tracked(self, tmp_path):
        """Test that belief state changes are tracked."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # Create belief state
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # Update belief multiple times
        service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            message="Yes",
            confidence=0.8,
        )
        service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_b",
            message="No",
            confidence=0.7,
        )

        # Get belief history
        history = service.get_belief_history(belief_id=belief.belief_id)

        assert len(history) >= 2  # Initial + 2 updates
        assert history[0]["success_prob"] == 0.5  # Initial


class TestIntegration:
    """Test full integration scenarios."""

    def test_full_bayesian_workflow(self, tmp_path):
        """Test complete Bayesian workflow."""
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
        voi_controller = VoIController(
            name="voi_controller",
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
        )

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=observation_learner,
            reliability_learner=reliability_learner,
            voi_controller=voi_controller,
        )

        # 1. Create belief state for task
        belief = service.create_belief_state(
            task_type="code_edit",
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            initial_belief={"success": 0.5, "failure": 0.5},
        )

        # 2. Decide whether to query agents
        should_query = service.should_query_agent(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            query_cost=0.1,
        )
        assert should_query is True

        # 3. Update belief with agent message
        updated_belief = service.update_belief_with_message(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            message="Yes, this works",
            confidence=0.8,
        )
        assert updated_belief.outcome_belief["success"] > 0.5

        # 4. Record task outcome
        service.record_task_outcome(
            belief_id=belief.belief_id,
            agent_id="agent_a",
            actual_outcome="success",
            agent_message="Yes, this works",
            agent_confidence=0.8,
        )

        # 5. Verify learning occurred
        stats = observation_learner.get_calibration_stats()
        assert "agent_a" in stats
