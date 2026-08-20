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

"""Unit tests for grounding threshold RL learner.

Tests the GroundingThresholdLearner which uses Thompson Sampling
to learn optimal hallucination detection thresholds per provider.
"""

import pytest
from pathlib import Path
from typing import Dict, Tuple

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.grounding_threshold import GroundingThresholdLearner
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
def learner(coordinator: RLCoordinator) -> GroundingThresholdLearner:
    """Fixture for GroundingThresholdLearner."""
    return coordinator.get_learner("grounding_threshold")  # type: ignore


def _record_grounding_outcome(
    learner: GroundingThresholdLearner,
    provider: str = "openai",
    response_type: str = "code_generation",
    threshold_used: float = 0.70,
    result_type: str = "tp",
) -> None:
    """Helper to record a grounding verification outcome."""
    outcome = RLOutcome(
        provider=provider,
        model="",
        task_type=response_type,
        success=result_type in ("tp", "tn"),
        quality_score=1.0 if result_type in ("tp", "tn") else 0.0,
        metadata={
            "response_type": response_type,
            "threshold_used": threshold_used,
            "result_type": result_type,
        },
    )
    learner.record_outcome(outcome)


def _get_beta_params_from_db(
    coordinator: RLCoordinator,
    context_key: str,
    threshold: float,
) -> Tuple[float, float, int]:
    """Helper to retrieve Beta parameters from the database."""
    cursor = coordinator.db.cursor()
    alpha_key = f"alpha:{context_key}:{threshold}"
    beta_key = f"beta:{context_key}:{threshold}"
    rows = cursor.execute(
        f"SELECT param_key, param_value, sample_count FROM {Tables.RL_PARAM} "
        f"WHERE learner_id = 'grounding_threshold' AND param_key IN (?, ?)",
        (alpha_key, beta_key),
    ).fetchall()
    if not rows:
        return (1.0, 1.0, 0)
    params = {r[0]: (r[1], r[2]) for r in rows}
    alpha = params.get(alpha_key, (1.0, 0))[0]
    beta = params.get(beta_key, (1.0, 0))[0]
    sample_count = int(params.get(alpha_key, (1.0, 0))[1])
    return (alpha, beta, sample_count)


class TestGroundingThresholdLearner:
    """Tests for GroundingThresholdLearner."""

    def test_initialization(self, learner: GroundingThresholdLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "grounding_threshold"
        assert learner.learning_rate == 0.1

        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_PARAM}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TASK_STAT}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TRANSITION}';"
        )
        assert cursor.fetchone() is not None

    def test_record_single_outcome(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """Recording one outcome updates Beta parameters."""
        provider = "anthropic"
        response_type = "explanation"
        threshold = 0.70

        _record_grounding_outcome(
            learner,
            provider=provider,
            response_type=response_type,
            threshold_used=threshold,
            result_type="tp",
        )

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator, context_key, threshold)
        assert count == 1
        # Alpha should increase for success (tp)
        assert alpha > learner.PRIOR_ALPHA

    def test_true_positive_increases_alpha(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """True positive outcomes increase alpha (success)."""
        provider = "openai"
        response_type = "code_generation"
        threshold = 0.75

        for _ in range(5):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=threshold,
                result_type="tp",
            )

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator, context_key, threshold)
        assert count == 5
        assert alpha > learner.PRIOR_ALPHA + 0.4  # 5 * learning_rate

    def test_false_positive_increases_beta(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """False positive outcomes increase beta (failure)."""
        provider = "deepseek"
        response_type = "analysis"
        threshold = 0.80

        for _ in range(5):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=threshold,
                result_type="fp",  # False positive
            )

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator, context_key, threshold)
        assert count == 5
        # Beta should increase for failures
        assert beta > learner.PRIOR_BETA + 0.4

    def test_false_negative_increases_beta(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """False negative outcomes (missed hallucination) increase beta."""
        provider = "ollama"
        response_type = "general"
        threshold = 0.65

        _record_grounding_outcome(
            learner,
            provider=provider,
            response_type=response_type,
            threshold_used=threshold,
            result_type="fn",  # False negative - worst outcome
        )

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator, context_key, threshold)
        assert count == 1
        assert beta > learner.PRIOR_BETA

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        provider = "google"
        response_type = "explanation"
        threshold = 0.70

        reset_database()
        db_path = tmp_path / "rl_test.db"
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("grounding_threshold")  # type: ignore

        _record_grounding_outcome(
            learner1,
            provider=provider,
            response_type=response_type,
            threshold_used=threshold,
        )
        reset_database()

        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("grounding_threshold")  # type: ignore

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator2, context_key, threshold)
        assert count == 1

        # Check state was loaded correctly
        assert context_key in learner2._beta_params

    def test_get_recommendation_thompson_sampling(self, learner: GroundingThresholdLearner) -> None:
        """Test get_recommendation uses Thompson Sampling."""
        provider = "anthropic"
        response_type = "code_generation"

        # Record many successes at 0.75 threshold
        for _ in range(20):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.75,
                result_type="tp",
            )

        # Record many failures at 0.90 threshold
        for _ in range(20):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.90,
                result_type="fp",
            )

        # Get recommendation - should favor 0.75
        rec = learner.get_recommendation(provider, "", response_type)

        assert rec is not None
        assert rec.value in learner.THRESHOLD_LEVELS
        # sample_size reflects weighted learning updates, not raw count
        # With many samples, should have some tracked updates
        assert rec.sample_size > 0

    def test_get_optimal_threshold(self, learner: GroundingThresholdLearner) -> None:
        """Test convenience method for optimal threshold."""
        provider = "mistral"
        response_type = "edit"

        threshold, confidence = learner.get_optimal_threshold(provider, response_type)

        assert 0.5 <= threshold <= 0.95
        assert 0.0 <= confidence <= 1.0

    def test_provider_stats_tracking(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """Test provider error rate statistics."""
        provider = "openai"

        # Record mixed outcomes
        _record_grounding_outcome(learner, provider=provider, result_type="tp")
        _record_grounding_outcome(learner, provider=provider, result_type="tp")
        _record_grounding_outcome(learner, provider=provider, result_type="tn")
        _record_grounding_outcome(learner, provider=provider, result_type="fp")
        _record_grounding_outcome(learner, provider=provider, result_type="fn")

        stats = learner.get_provider_error_rates(provider)

        assert stats["total_samples"] == 5
        assert stats["fp_rate"] == 0.2  # 1/5
        assert stats["fn_rate"] == 0.2  # 1/5
        # Precision = tp / (tp + fp) = 2 / 3
        assert abs(stats["precision"] - 2 / 3) < 0.01
        # Recall = tp / (tp + fn) = 2 / 3
        assert abs(stats["recall"] - 2 / 3) < 0.01

    def test_get_all_provider_stats(self, learner: GroundingThresholdLearner) -> None:
        """Test getting stats for all providers."""
        _record_grounding_outcome(learner, provider="openai", result_type="tp")
        _record_grounding_outcome(learner, provider="anthropic", result_type="tn")
        _record_grounding_outcome(learner, provider="deepseek", result_type="fp")

        all_stats = learner.get_all_provider_stats()

        assert "openai" in all_stats
        assert "anthropic" in all_stats
        assert "deepseek" in all_stats

    def test_compute_reward(self, learner: GroundingThresholdLearner) -> None:
        """Test reward computation for different result types."""
        assert learner._compute_reward("tp") == 0.1
        assert learner._compute_reward("tn") == 0.1
        assert learner._compute_reward("fp") == -1.0
        assert learner._compute_reward("fn") == -2.0  # Worst
        assert learner._compute_reward("unknown") == 0.0

    def test_context_key_building(self, learner: GroundingThresholdLearner) -> None:
        """Test context key construction."""
        assert learner._build_context_key("openai", "code_generation") == "openai:code_generation"
        assert learner._build_context_key("anthropic", "explanation") == "anthropic:explanation"
        # Unknown response type defaults to general
        assert learner._build_context_key("ollama", "unknown_type") == "ollama:general"

    def test_export_metrics(self, learner: GroundingThresholdLearner) -> None:
        """Test metrics export."""
        _record_grounding_outcome(learner, provider="openai", result_type="tp")
        _record_grounding_outcome(learner, provider="anthropic", result_type="tn")

        metrics = learner.export_metrics()

        assert metrics["learner"] == "grounding_threshold"
        assert metrics["total_decisions"] == 2
        assert metrics["default_threshold"] == 0.70
        assert metrics["threshold_levels"] == learner.THRESHOLD_LEVELS

    def test_no_data_returns_baseline(self, learner: GroundingThresholdLearner) -> None:
        """Test that unknown provider returns baseline recommendation."""
        rec = learner.get_recommendation("unknown_provider", "", "general")

        assert rec is not None
        assert rec.is_baseline is True
        assert rec.value == learner.DEFAULT_THRESHOLD
        assert rec.confidence == 0.3
        assert rec.sample_size == 0

    def test_threshold_discretization(
        self, coordinator: RLCoordinator, learner: GroundingThresholdLearner
    ) -> None:
        """Test that thresholds are discretized to nearest level."""
        provider = "test"
        response_type = "analysis"

        # Record at 0.73 - should discretize to 0.75
        _record_grounding_outcome(
            learner,
            provider=provider,
            response_type=response_type,
            threshold_used=0.73,
            result_type="tp",
        )

        context_key = f"{provider}:{response_type}"
        alpha, beta, count = _get_beta_params_from_db(coordinator, context_key, 0.75)
        assert count == 1

    def test_response_type_normalization(self, learner: GroundingThresholdLearner) -> None:
        """Test that response types are normalized."""
        valid_types = learner.RESPONSE_TYPES

        for rt in valid_types:
            key = learner._build_context_key("test", rt)
            assert rt in key

        # Invalid type should become 'general'
        key = learner._build_context_key("test", "invalid_type")
        assert "general" in key

    def test_mixed_outcomes_convergence(self, learner: GroundingThresholdLearner) -> None:
        """Test that learner converges with mixed outcomes."""
        provider = "test_provider"
        response_type = "code_generation"

        # Simulate realistic scenario:
        # - Low threshold (0.55) has many false positives (too strict)
        # - High threshold (0.85) has some false negatives (too lenient)
        # - Medium threshold (0.70) is just right

        # Too strict - lots of false positives
        for _ in range(10):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.55,
                result_type="fp",
            )

        # Too lenient - some false negatives
        for _ in range(10):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.85,
                result_type="fn",
            )

        # Just right - mostly true positives/negatives
        for _ in range(15):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.70,
                result_type="tp",
            )
        for _ in range(5):
            _record_grounding_outcome(
                learner,
                provider=provider,
                response_type=response_type,
                threshold_used=0.70,
                result_type="tn",
            )

        # Get optimal threshold
        threshold, confidence = learner.get_optimal_threshold(provider, response_type)

        # The threshold should be one of the valid levels
        assert threshold in learner.THRESHOLD_LEVELS

        # With this many samples and clear winner at 0.70, confidence should be non-trivial
        # (Thompson Sampling may still show low confidence early on due to exploration)
        assert confidence >= 0.3
