"""Unit tests for model selector RL learner."""

import pytest
from unittest.mock import patch
from pathlib import Path
from typing import Optional, Tuple

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.model_selector import (
    ModelSelectorLearner,
    SelectionStrategy,
)
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    """Fixture for RLCoordinator, ensuring a clean database for each test."""
    # Reset the global singleton to ensure fresh database for each test
    reset_database()
    db_path = tmp_path / "rl_test.db"
    # Initialize the database singleton with temp path BEFORE creating coordinator
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    # Reset again after the test to clean up
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> ModelSelectorLearner:
    """Fixture for ModelSelectorLearner."""
    return coordinator.get_learner("model_selector")  # type: ignore


def _record_selection_outcome(
    learner: ModelSelectorLearner,
    provider: str = "ollama",
    model: str = "qwen3-coder-tools:30b",
    task_type: str = "analysis",
    *,
    success: bool = True,
    quality_score: float = 0.8,
    latency_seconds: float = 5.0,
    token_count: int = 1000,
    tool_calls_made: int = 5,
    user_satisfaction: Optional[float] = None,
) -> None:
    """Helper to record a single model selection outcome."""
    outcome = RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "latency_seconds": latency_seconds,
            "token_count": token_count,
            "tool_calls_made": tool_calls_made,
            "user_satisfaction": user_satisfaction,
        },
    )
    learner.record_outcome(outcome)


def _get_q_value_from_db(
    coordinator: RLCoordinator,
    provider: str,
    task_type: Optional[str] = None,
) -> Tuple[float, int]:
    """Helper to retrieve Q-value and selection count from the database."""
    cursor = coordinator.db.cursor()
    if task_type:
        cursor.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'model_selector' AND state_key = ? AND action_key = ?",
            (provider, task_type),
        )
    else:
        cursor.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'model_selector' AND state_key = ? AND action_key = 'select'",
            (provider,),
        )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0.5, 0)  # Default values


class TestModelSelectorLearner:
    def test_initialization(self, learner: ModelSelectorLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "model_selector"
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_Q_VALUE}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_PARAM}';"
        )
        assert cursor.fetchone() is not None

    def test_record_single_outcome(
        self, coordinator: RLCoordinator, learner: ModelSelectorLearner
    ) -> None:
        """Recording one outcome updates Q-values and counts."""
        _record_selection_outcome(
            learner,
            provider="anthropic",
            model="claude-3-opus",
            task_type="analysis",
            success=True,
            quality_score=0.9,
            latency_seconds=10.0,
        )

        q_value, count = _get_q_value_from_db(coordinator, "anthropic")
        assert count == 1
        assert q_value != 0.5  # Should have been updated

        task_q_value, task_count = _get_q_value_from_db(coordinator, "anthropic", "analysis")
        assert task_count == 1
        assert task_q_value != 0.5

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        # Reset to ensure clean state
        reset_database()
        db_path = tmp_path / "rl_test.db"

        # Initialize DB singleton with temp path
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("model_selector")  # type: ignore
        initial_epsilon_learner1 = learner1.epsilon

        _record_selection_outcome(learner1, provider="openai")

        # Reset the singleton to simulate new session (don't call close directly)
        reset_database()

        # Re-initialize with same temp path to test persistence
        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("model_selector")  # type: ignore

        q_value, count = _get_q_value_from_db(coordinator2, "openai")
        assert count == 1
        assert q_value != 0.5
        assert (
            learner2.epsilon < initial_epsilon_learner1
        )  # Epsilon should decay from initial value
        assert learner2.epsilon == learner1.epsilon  # The actual decayed value should be loaded

        # Clean up
        reset_database()

    @pytest.mark.parametrize(
        "strategy, expected_reason",
        [
            (SelectionStrategy.EPSILON_GREEDY, "Exploration"),
            (SelectionStrategy.EXPLOIT_ONLY, "Best Q-value"),
        ],
    )
    def test_get_recommendation_strategies(
        self,
        learner: ModelSelectorLearner,
        strategy: SelectionStrategy,
        expected_reason: str,
    ) -> None:
        """Test different selection strategies."""
        learner.strategy = strategy
        learner.epsilon = (
            1.0 if strategy == SelectionStrategy.EPSILON_GREEDY else 0.0
        )  # Force exploration/exploitation

        _record_selection_outcome(
            learner, provider="provider_A", quality_score=0.9, task_type="analysis"
        )
        _record_selection_outcome(
            learner, provider="provider_B", quality_score=0.1, task_type="analysis"
        )

        # Force the exploration draw. The learner draws from its per-instance
        # ``self._rng`` (base.py), NOT module-level random — patching the module
        # was a no-op, leaving this a ~1-in-100 flake (ε decays to 0.99 after the
        # two recorded outcomes, so the unpatched draw failed whenever rng >= ε).
        with patch.object(
            learner._rng, "random", return_value=0.1
        ):  # 0.1 < ε for EPSILON_GREEDY; EXPLOIT_ONLY ignores the draw
            # Get recommendation for providers A and B
            rec = learner.get_recommendation(
                provider='["provider_A", "provider_B"]',
                model="any-model",
                task_type="analysis",
            )
        assert rec is not None
        assert rec.value in ["provider_A", "provider_B"]
        assert expected_reason in rec.reason

    def test_ucb_strategy(self, learner: ModelSelectorLearner) -> None:
        """Test UCB selection strategy."""
        learner.strategy = SelectionStrategy.UCB

        # Populate some data
        _record_selection_outcome(
            learner,
            provider="provider_X",
            quality_score=0.8,
            task_type="coding",
            latency_seconds=1.0,
        )
        _record_selection_outcome(
            learner,
            provider="provider_X",
            quality_score=0.8,
            task_type="coding",
            latency_seconds=1.0,
        )
        _record_selection_outcome(
            learner,
            provider="provider_Y",
            quality_score=0.2,
            task_type="coding",
            latency_seconds=10.0,
        )

        rec = learner.get_recommendation(
            provider='["provider_X", "provider_Y"]',
            model="any-model",
            task_type="coding",
        )
        assert rec is not None
        # Provider_X should have higher Q-value and thus be selected more often
        # This is a probabilistic test, so we can't assert exact selection, but
        # for a small number of selections and high difference, X should win.
        # For simplicity, we'll just check it returns one of them.
        assert rec.value in ["provider_X", "provider_Y"]


class TestModelSelectorMultiProviderABTesting:
    """A/B coverage proving the router is provider-agnostic.

    Motivation: the chat banner's "Routing hint" was anthropic-only because the
    only training data came from runs that always used anthropic. These tests
    feed BALANCED, multi-provider outcomes and assert the learner picks the best
    provider on merit — never a hardcoded default, and symmetric under provider
    name/order — so a single-provider dev/test habit can't bias the routing.
    """

    # Deliberately mix non-anthropic winners so a default-to-anthropic bug fails.
    @pytest.mark.parametrize(
        "providers, best",
        [
            (["anthropic", "openai", "ollama", "zai"], "openai"),
            (["anthropic", "openai", "ollama", "zai"], "ollama"),
            (["anthropic", "openai", "ollama", "zai"], "zai"),
            (["openai", "anthropic"], "anthropic"),
            (["deepseek", "groq", "mistral"], "groq"),
        ],
    )
    def test_router_picks_best_provider_on_merit(
        self, learner: ModelSelectorLearner, providers, best
    ):
        import json

        # Deterministic merit-based selection (no exploration noise).
        learner.strategy = SelectionStrategy.EXPLOIT_ONLY

        # Give the `best` provider clearly higher quality; others clearly lower.
        for _ in range(8):
            for p in providers:
                _record_selection_outcome(
                    learner,
                    provider=p,
                    task_type="coding",
                    quality_score=0.95 if p == best else 0.2,
                    success=(p == best),
                )

        rec = learner.get_recommendation(json.dumps(providers), "", "coding")
        assert rec is not None
        assert rec.value == best, f"expected {best}, got {rec.value} for {providers}"

    def test_recommendation_is_symmetric_under_provider_order(self, learner: ModelSelectorLearner):
        import json

        learner.strategy = SelectionStrategy.EXPLOIT_ONLY
        # `success` is what drives the learned Q-value; give openai a clear edge.
        for _ in range(8):
            _record_selection_outcome(
                learner,
                provider="openai",
                success=True,
                quality_score=0.9,
                task_type="coding",
            )
            _record_selection_outcome(
                learner,
                provider="anthropic",
                success=False,
                quality_score=0.2,
                task_type="coding",
            )

        rec_a = learner.get_recommendation(json.dumps(["anthropic", "openai"]), "", "coding")
        rec_b = learner.get_recommendation(json.dumps(["openai", "anthropic"]), "", "coding")
        # Order of the available list must not change the merit-based winner.
        assert rec_a.value == rec_b.value == "openai"

    def test_cold_start_does_not_default_to_anthropic(self, learner: ModelSelectorLearner):
        import json

        # No outcomes recorded for anyone: the learner must not confidently
        # recommend a specific provider (which previously surfaced anthropic).
        rec = learner.get_recommendation(json.dumps(["anthropic", "openai", "ollama"]), "", "chat")
        if rec is not None:
            # Any cold-start pick must be low-confidence (untrained), never a
            # high-Q anthropic suggestion like the polluted banner showed.
            assert rec.confidence < 0.6
