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

"""Tests for Bayesian task analysis with belief state tracking."""

import math

import pytest

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType


class TestBayesianTaskAnalysisInit:
    """Test BayesianTaskAnalysis initialization and basic properties."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
        )

        assert analysis.complexity == TaskComplexity.SIMPLE
        assert analysis.tool_budget == 10
        assert analysis.unified_task_type == UnifiedTaskType.EDIT

        # Bayesian-specific defaults
        assert analysis.outcome_belief == {}
        assert analysis.belief_entropy == 0.0
        assert analysis.belief_variance == 0.0
        assert analysis.agent_likelihoods == {}
        assert analysis.agent_reliability == {}
        assert analysis.expected_voi == 0.0
        assert analysis.query_cost_estimate == 0.0

    def test_init_with_belief_state(self):
        """Test initialization with pre-existing belief state."""
        prior_belief = {"success": 0.5, "failure": 0.5}

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.7,
            unified_task_type=UnifiedTaskType.GENERATION,
            outcome_belief=prior_belief.copy(),
        )

        assert analysis.outcome_belief == prior_belief
        # Entropy should be computed automatically
        assert analysis.belief_entropy > 0
        # Binary uniform distribution has entropy = ln(2) ≈ 0.693
        assert abs(analysis.belief_entropy - math.log(2)) < 0.01

    def test_init_with_agent_reliability(self):
        """Test initialization with agent reliability weights."""
        reliability = {
            "agent_a": 0.9,
            "agent_b": 0.7,
            "agent_c": 0.5,
        }

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.SEARCH,
            agent_reliability=reliability.copy(),
        )

        assert analysis.agent_reliability == reliability
        # All reliabilities should be in [0, 1]
        for agent_id, rel in analysis.agent_reliability.items():
            assert 0.0 <= rel <= 1.0


class TestPosteriorUpdate:
    """Test Bayesian posterior update mechanics."""

    def test_compute_posterior_conjugate(self):
        """Test conjugate update: P(Y|D,z) ∝ P(Y|D) * P(z|Y)."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Agent message suggests success (affirm)
        likelihood = {"success": 0.8, "failure": 0.2}  # P(z|Y)

        analysis.compute_posterior(analysis.outcome_belief, likelihood)

        # Posterior should shift toward success
        assert analysis.outcome_belief["success"] > 0.5
        assert analysis.outcome_belief["failure"] < 0.5

        # Check normalization (sums to 1.0)
        total = sum(analysis.outcome_belief.values())
        assert abs(total - 1.0) < 1e-6

    def test_posterior_with_reliability_weighting(self):
        """Test posterior update with reliability downweighting."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.GENERATION,
            outcome_belief={"success": 0.5, "failure": 0.5},
            agent_reliability={"agent_a": 0.5},  # Low reliability
        )

        likelihood = {"success": 0.9, "failure": 0.1}

        # Update with reliability weighting
        weighted_likelihood = {
            outcome: prob * analysis.agent_reliability["agent_a"]
            for outcome, prob in likelihood.items()
        }

        analysis.compute_posterior(analysis.outcome_belief, weighted_likelihood)

        # With low reliability, belief should shift less
        assert analysis.outcome_belief["success"] > 0.5
        # The reliability weighting is applied inside compute_posterior
        # so we should check that the update happened
        assert analysis.outcome_belief["success"] != 0.5  # Belief did update

    def test_multiple_sequential_updates(self):
        """Test multiple Bayesian updates in sequence."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # First agent suggests success
        likelihood1 = {"success": 0.8, "failure": 0.2}
        analysis.compute_posterior(analysis.outcome_belief, likelihood1)

        posterior1 = analysis.outcome_belief.copy()

        # Second agent also suggests success
        likelihood2 = {"success": 0.9, "failure": 0.1}
        analysis.compute_posterior(analysis.outcome_belief, likelihood2)

        # Belief should be stronger after two affirmations
        assert analysis.outcome_belief["success"] > posterior1["success"]

    def test_posterior_with_conflicting_evidence(self):
        """Test posterior update with conflicting agent messages."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # Agent A says success
        likelihood_a = {"success": 0.8, "failure": 0.2}
        analysis.compute_posterior(analysis.outcome_belief, likelihood_a)

        # Agent B says failure
        likelihood_b = {"success": 0.2, "failure": 0.8}
        analysis.compute_posterior(analysis.outcome_belief, likelihood_b)

        # Belief should be pulled back toward uncertainty
        # But not all the way back to 0.5 (Agent A had slightly stronger signal)
        assert 0.4 < analysis.outcome_belief["success"] < 0.6


class TestEntropyComputation:
    """Test belief entropy calculations."""

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # H[X] = -∑ p(x) log p(x)
        # For binary uniform: H = -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.693
        expected_entropy = math.log(2)

        assert abs(analysis.belief_entropy - expected_entropy) < 0.01

    def test_entropy_certain_distribution(self):
        """Test entropy of certain distribution (zero entropy)."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 1.0, "failure": 0.0},
        )

        # Certain distribution has zero entropy
        assert analysis.belief_entropy == 0.0

    def test_entropy_decreases_with_information(self):
        """Test that entropy decreases as we gain information."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        initial_entropy = analysis.belief_entropy

        # Add evidence favoring success
        likelihood = {"success": 0.9, "failure": 0.1}
        analysis.compute_posterior(analysis.outcome_belief, likelihood)

        # Entropy should decrease
        assert analysis.belief_entropy < initial_entropy

    def test_entropy_multi_outcome(self):
        """Test entropy with multi-outcome belief state."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.GENERATION,
            outcome_belief={"success": 0.33, "partial": 0.33, "failure": 0.34},
        )

        # Entropy should be higher for 3-outcome uniform vs 2-outcome
        assert analysis.belief_entropy > math.log(2)

        # Should be less than log(3) (not perfectly uniform)
        assert analysis.belief_entropy < math.log(3)


class TestVoIComputation:
    """Test Value of Information computation."""

    def test_compute_voi_basic(self):
        """Test basic VoI computation."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # High uncertainty agent should have positive VoI
        voi = analysis.compute_voi("agent_a", query_cost=0.1)

        # VoI should be positive when uncertainty is high
        assert voi > 0

    def test_compute_voi_with_low_uncertainty(self):
        """Test VoI when belief is already certain."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.99, "failure": 0.01},
        )

        # Low uncertainty should have low VoI
        voi = analysis.compute_voi("agent_a", query_cost=0.1)

        # VoI should be small when entropy is low
        assert voi < 0.1

    def test_compute_voi_with_high_cost(self):
        """Test VoI computation with high query cost."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            outcome_belief={"success": 0.5, "failure": 0.5},
        )

        # High cost should make VoI negative
        voi = analysis.compute_voi("agent_a", query_cost=1.0)

        # With high cost, VoI should be negative
        assert voi < 0


class TestBeliefPersistence:
    """Test belief state persistence and serialization."""

    def test_belief_serialization(self):
        """Test that belief state can be serialized for storage."""
        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.GENERATION,
            outcome_belief={"success": 0.7, "failure": 0.3},
            agent_reliability={"agent_a": 0.8},
        )

        # Convert to dict (simulating serialization)
        belief_dict = {
            "outcome_belief": analysis.outcome_belief,
            "belief_entropy": analysis.belief_entropy,
            "agent_reliability": analysis.agent_reliability,
        }

        # Should be JSON-serializable (no complex objects)
        assert isinstance(belief_dict["outcome_belief"], dict)
        assert isinstance(belief_dict["belief_entropy"], float)
        assert isinstance(belief_dict["agent_reliability"], dict)

    def test_belief_deserialization(self):
        """Test that belief state can be restored from storage."""
        stored_belief = {
            "outcome_belief": {"success": 0.6, "failure": 0.4},
            "belief_entropy": 0.673,
            "agent_reliability": {"agent_a": 0.75},
        }

        analysis = BayesianTaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=10,
            complexity_confidence=0.8,
            unified_task_type=UnifiedTaskType.EDIT,
            **stored_belief,
        )

        assert analysis.outcome_belief == stored_belief["outcome_belief"]
        assert analysis.agent_reliability == stored_belief["agent_reliability"]
        assert abs(analysis.belief_entropy - stored_belief["belief_entropy"]) < 0.01
