# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Priority 4 Phase 2 — predictive tool selection and model threshold learning."""

import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.learners.tool_selector import ToolSelectorLearner
from victor.framework.rl.learners.model_selector import ModelSelectorLearner
from victor.core.schema import Tables

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_tool_learner() -> ToolSelectorLearner:
    return ToolSelectorLearner(name="tool_selector", db_connection=_make_db())


def _make_model_learner() -> ModelSelectorLearner:
    return ModelSelectorLearner(name="model_selector", db_connection=_make_db())


def _tool_outcome(tool_name: str, success: bool = True, tools_used: list = None) -> RLOutcome:
    return RLOutcome(
        provider="anthropic",
        model="claude-sonnet-4-6",
        task_type="analysis",
        success=success,
        quality_score=0.8 if success else 0.3,
        metadata={
            "tool_name": tool_name,
            "tool_success": success,
            "tools_used": tools_used or [tool_name],
        },
    )


# ---------------------------------------------------------------------------
# ToolSelectorLearner Phase 2 extensions
# ---------------------------------------------------------------------------


class TestToolSelectorPredictor:
    def test_predictor_is_none_initially(self):
        learner = _make_tool_learner()
        assert learner._predictor is None

    def test_analytics_is_none_initially(self):
        learner = _make_tool_learner()
        assert learner._analytics is None

    def test_get_next_tool_prediction_returns_string_or_none(self):
        learner = _make_tool_learner()
        result = learner.get_next_tool_prediction(
            task_description="read a Python file",
            current_step="exploration",
            recent_tools=[],
            task_type="analysis",
        )
        assert result is None or isinstance(result, str)

    def test_get_next_tool_prediction_delegates_to_predictor(self):
        learner = _make_tool_learner()
        mock_prediction = MagicMock()
        mock_prediction.tool_name = "read"
        mock_predictor = MagicMock()
        mock_predictor.predict_tools.return_value = [mock_prediction]
        learner._predictor = mock_predictor

        result = learner.get_next_tool_prediction(
            task_description="read a file",
            current_step="exploration",
            recent_tools=["search"],
            task_type="analysis",
        )
        assert result == "read"
        mock_predictor.predict_tools.assert_called_once_with(
            task_description="read a file",
            current_step="exploration",
            recent_tools=["search"],
            task_type="analysis",
        )

    def test_get_next_tool_prediction_returns_none_on_empty_list(self):
        learner = _make_tool_learner()
        mock_predictor = MagicMock()
        mock_predictor.predict_tools.return_value = []
        learner._predictor = mock_predictor
        result = learner.get_next_tool_prediction("no match task", recent_tools=[])
        assert result is None

    def test_get_next_tool_prediction_returns_none_on_predictor_error(self):
        learner = _make_tool_learner()
        mock_predictor = MagicMock()
        mock_predictor.predict_tools.side_effect = RuntimeError("embedding service down")
        learner._predictor = mock_predictor
        result = learner.get_next_tool_prediction("task", recent_tools=[])
        assert result is None

    def test_does_not_reimplement_prediction_logic(self):
        """get_next_tool_prediction must delegate to predictor.predict_tools, not reimplement."""
        import inspect

        src = inspect.getsource(ToolSelectorLearner.get_next_tool_prediction)
        assert "predict_tools" in src, "Must delegate to existing ToolPredictor.predict_tools()"
        assert "keyword" not in src.lower(), "Must not reimplement keyword matching"
        assert "semantic" not in src.lower(), "Must not reimplement semantic similarity"


class TestToolSelectorCooccurrenceFeedback:
    def test_feed_outcome_to_predictor_calls_tracker(self):
        learner = _make_tool_learner()
        mock_tracker = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor._cooccurrence_tracker = mock_tracker
        learner._predictor = mock_predictor

        outcome = _tool_outcome("read", success=True, tools_used=["search", "read"])
        learner.feed_outcome_to_predictor(outcome)

        mock_tracker.record_tool_sequence.assert_called_once_with(
            tools=["search", "read"],
            task_type="analysis",
            success=True,
        )

    def test_feed_outcome_falls_back_to_tool_name_when_no_tools_used(self):
        learner = _make_tool_learner()
        mock_tracker = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor._cooccurrence_tracker = mock_tracker
        learner._predictor = mock_predictor

        outcome = _tool_outcome("write", success=True)
        outcome.metadata.pop("tools_used", None)
        learner.feed_outcome_to_predictor(outcome)

        mock_tracker.record_tool_sequence.assert_called_once_with(
            tools=["write"],
            task_type="analysis",
            success=True,
        )

    def test_feed_outcome_noop_when_predictor_unavailable(self):
        """Must not raise when predictor hasn't been loaded yet."""
        learner = _make_tool_learner()
        assert learner._predictor is None
        outcome = _tool_outcome("read")
        learner.feed_outcome_to_predictor(outcome)  # Should not raise

    def test_record_outcome_calls_feed_outcome(self):
        """record_outcome() override must also feed the predictor tracker."""
        learner = _make_tool_learner()
        mock_tracker = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor._cooccurrence_tracker = mock_tracker
        learner._predictor = mock_predictor

        outcome = _tool_outcome("grep", success=True, tools_used=["grep"])
        learner.record_outcome(outcome)

        mock_tracker.record_tool_sequence.assert_called_once()


class TestToolSelectorAnalyticsEnhancedRankings:
    def test_falls_back_to_base_rankings_when_analytics_unavailable(self):
        learner = _make_tool_learner()
        assert learner._analytics is None
        result = learner.get_analytics_enhanced_rankings(["read", "write", "search"], "analysis")
        assert isinstance(result, list)

    def test_blends_analytics_success_rate(self):
        learner = _make_tool_learner()
        # Seed some Q-values
        learner._tool_q_values["read"] = 0.8
        learner._tool_q_values["write"] = 0.6
        learner._tool_selection_counts["read"] = 30
        learner._tool_selection_counts["write"] = 30

        mock_analytics = MagicMock()
        mock_analytics.get_tool_insights.side_effect = lambda t: {
            "success_rate": 0.9 if t == "read" else 0.4,
        }
        learner._analytics = mock_analytics

        rankings = learner.get_analytics_enhanced_rankings(["read", "write"], "analysis")
        tool_names = [r[0] for r in rankings]
        assert tool_names[0] == "read"

    def test_returns_list_of_tuples(self):
        learner = _make_tool_learner()
        result = learner.get_analytics_enhanced_rankings(["read"], "analysis")
        assert isinstance(result, list)
        if result:
            name, score, conf = result[0]
            assert isinstance(name, str)
            assert 0.0 <= score <= 1.0

    def test_export_metrics_includes_wiring_flags(self):
        learner = _make_tool_learner()
        metrics = learner.export_metrics()
        assert "predictor_wired" in metrics
        assert "analytics_wired" in metrics
        assert metrics["predictor_wired"] is False
        assert metrics["analytics_wired"] is False


# ---------------------------------------------------------------------------
# ModelSelectorLearner Phase 2 — learned confidence thresholds
# ---------------------------------------------------------------------------


class TestModelSelectorConfidenceThresholds:
    def test_learn_confidence_threshold_stores_observation(self):
        learner = _make_model_learner()
        learner.learn_confidence_threshold("task_type", 0.8, used_llm=False, success=True)
        assert len(learner._threshold_observations["task_type"]) == 1

    def test_get_optimal_threshold_none_when_insufficient_data(self):
        learner = _make_model_learner()
        learner.learn_confidence_threshold("task_type", 0.8, used_llm=False, success=True)
        assert learner.get_optimal_threshold("task_type") is None  # need 10+ obs

    def test_get_optimal_threshold_returns_float_with_enough_data(self):
        learner = _make_model_learner()
        for i in range(20):
            learner.learn_confidence_threshold(
                "task_type",
                heuristic_confidence=0.9 if i % 2 == 0 else 0.3,
                used_llm=(i % 2 == 1),
                success=(i % 3 != 0),
            )
        threshold = learner.get_optimal_threshold("task_type")
        assert threshold is not None
        assert 0.0 < threshold < 1.0

    def test_get_optimal_threshold_different_types_independent(self):
        learner = _make_model_learner()
        for i in range(15):
            learner.learn_confidence_threshold("tool_necessity", 0.9, False, True)
            learner.learn_confidence_threshold("task_type", 0.3, True, True)

        t1 = learner.get_optimal_threshold("tool_necessity")
        t2 = learner.get_optimal_threshold("task_type")
        # Both should exist but may differ
        assert t1 is not None
        assert t2 is not None

    def test_observations_capped_at_200(self):
        learner = _make_model_learner()
        for i in range(250):
            learner.learn_confidence_threshold("type_x", 0.5, False, True)
        assert len(learner._threshold_observations["type_x"]) == 200

    def test_threshold_persisted_to_db(self):
        learner = _make_model_learner()
        for i in range(10):
            learner.learn_confidence_threshold("tool_necessity", 0.7, False, True)

        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT param_key, value_text FROM {Tables.RL_PARAM} "
            f"WHERE learner_id = 'model_selector' AND param_key = 'threshold:tool_necessity'",
        )
        row = cursor.fetchone()
        assert row is not None

    def test_threshold_loaded_from_db_on_init(self):
        db = _make_db()
        learner1 = ModelSelectorLearner(name="model_selector", db_connection=db)
        for i in range(15):
            learner1.learn_confidence_threshold("task_type", 0.8, False, True)

        learner2 = ModelSelectorLearner(name="model_selector", db_connection=db)
        assert len(learner2._threshold_observations.get("task_type", [])) > 0

    def test_get_optimal_threshold_returns_none_for_unknown_type(self):
        learner = _make_model_learner()
        assert learner.get_optimal_threshold("unknown_decision_type") is None

    def test_existing_q_learning_still_works_after_extension(self):
        """Verify Phase 2 additions don't break the existing Q-learning."""
        learner = _make_model_learner()
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="analysis",
            success=True,
            quality_score=0.85,
        )
        learner.record_outcome(outcome)
        rec = learner.get_recommendation("anthropic", "claude-sonnet-4-6", "analysis")
        assert rec is not None


# ---------------------------------------------------------------------------
# Coordinator integration
# ---------------------------------------------------------------------------


class TestCoordinatorPhase2Integration:
    def test_coordinator_tool_selector_has_predictor_methods(self):
        from victor.framework.rl.coordinator import get_rl_coordinator

        coord = get_rl_coordinator()
        learner = coord.get_learner("tool_selector")
        assert hasattr(learner, "get_next_tool_prediction")
        assert hasattr(learner, "feed_outcome_to_predictor")
        assert hasattr(learner, "get_analytics_enhanced_rankings")

    def test_coordinator_model_selector_has_threshold_methods(self):
        from victor.framework.rl.coordinator import get_rl_coordinator

        coord = get_rl_coordinator()
        learner = coord.get_learner("model_selector")
        assert hasattr(learner, "learn_confidence_threshold")
        assert hasattr(learner, "get_optimal_threshold")
        assert hasattr(learner, "load_threshold_observations")
