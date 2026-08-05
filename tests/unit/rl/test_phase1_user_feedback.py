# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Priority 4 Phase 1 — user feedback learner and meta-learning."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.framework.rl.learners.user_feedback import (
    UserFeedbackLearner,
    create_outcome_with_user_feedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_learner() -> UserFeedbackLearner:
    return UserFeedbackLearner(name="user_feedback", db_connection=_make_db(), learning_rate=0.1)


# ---------------------------------------------------------------------------
# create_outcome_with_user_feedback helper
# ---------------------------------------------------------------------------


class TestCreateOutcomeWithUserFeedback:
    def test_returns_rl_outcome(self):
        outcome = create_outcome_with_user_feedback(session_id="s1", rating=0.8)
        assert isinstance(outcome, RLOutcome)

    def test_rating_maps_to_quality_score(self):
        outcome = create_outcome_with_user_feedback(session_id="s1", rating=0.75)
        assert outcome.quality_score == 0.75

    def test_feedback_source_is_user(self):
        outcome = create_outcome_with_user_feedback(session_id="s1", rating=0.9)
        assert outcome.metadata["feedback_source"] == "user"

    def test_optional_fields_stored_in_metadata(self):
        outcome = create_outcome_with_user_feedback(
            session_id="s1",
            rating=0.6,
            feedback="Good",
            helpful=True,
            correction="Fix X",
        )
        assert outcome.metadata["user_feedback"] == "Good"
        assert outcome.metadata["helpful"] is True
        assert outcome.metadata["correction"] == "Fix X"

    def test_session_id_in_metadata(self):
        outcome = create_outcome_with_user_feedback(session_id="abc-123", rating=0.5)
        assert outcome.metadata["session_id"] == "abc-123"

    def test_provider_and_task_type_are_canonical(self):
        outcome = create_outcome_with_user_feedback(session_id="s1", rating=0.8)
        assert outcome.provider == "user"
        assert outcome.task_type == "feedback"
        assert outcome.success is True

    def test_reuses_rl_outcome_quality_score_no_new_field(self):
        """Verify we're NOT adding a separate user_rating field."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RLOutcome)}
        assert "user_rating" not in fields
        assert "feedback_score" not in fields
        assert "quality_score" in fields  # reused


# ---------------------------------------------------------------------------
# UserFeedbackLearner
# ---------------------------------------------------------------------------


class TestUserFeedbackLearner:
    def test_inherits_from_base_learner(self):
        from victor.framework.rl.base import BaseLearner

        learner = _make_learner()
        assert isinstance(learner, BaseLearner)

    def test_record_outcome_accepts_user_feedback(self):
        learner = _make_learner()
        outcome = create_outcome_with_user_feedback(session_id="s1", rating=0.8)
        learner.record_outcome(outcome)
        stats = learner.get_feedback_stats()
        assert stats["total_feedback"] == 1
        assert abs(stats["avg_rating"] - 0.8) < 1e-6

    def test_record_outcome_ignores_non_user_feedback(self):
        learner = _make_learner()
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="tool_call",
            success=True,
            quality_score=0.9,
            metadata={"feedback_source": "auto"},
        )
        learner.record_outcome(outcome)
        stats = learner.get_feedback_stats()
        assert stats["total_feedback"] == 0

    def test_multiple_feedbacks_averaged(self):
        learner = _make_learner()
        for rating in [0.6, 0.8, 1.0]:
            learner.record_outcome(
                create_outcome_with_user_feedback(session_id=f"s{rating}", rating=rating)
            )
        stats = learner.get_feedback_stats()
        assert stats["total_feedback"] == 3
        assert abs(stats["avg_rating"] - 0.8) < 1e-6

    def test_get_feedback_stats_empty(self):
        learner = _make_learner()
        stats = learner.get_feedback_stats()
        assert stats["total_feedback"] == 0

    def test_get_recommendation_none_when_no_feedback(self):
        learner = _make_learner()
        rec = learner.get_recommendation("anthropic", "claude", "tool_call")
        assert rec is None

    def test_get_recommendation_returns_rl_recommendation(self):
        learner = _make_learner()
        for i in range(5):
            learner.record_outcome(
                create_outcome_with_user_feedback(session_id=f"s{i}", rating=0.85)
            )
        rec = learner.get_recommendation("anthropic", "claude", "tool_call")
        assert isinstance(rec, RLRecommendation)
        assert "avg_user_rating" in rec.value
        assert rec.value["sample_count"] == 5

    def test_confidence_increases_with_more_samples(self):
        learner = _make_learner()
        for i in range(3):
            learner.record_outcome(
                create_outcome_with_user_feedback(session_id=f"s{i}", rating=0.8)
            )
        low_conf = learner.get_recommendation("a", "m", "t").confidence

        for i in range(10):
            learner.record_outcome(
                create_outcome_with_user_feedback(session_id=f"x{i}", rating=0.8)
            )
        high_conf = learner.get_recommendation("a", "m", "t").confidence
        assert high_conf > low_conf

    def test_persists_summary_to_db(self):
        learner = _make_learner()
        learner.record_outcome(create_outcome_with_user_feedback(session_id="s1", rating=0.9))
        cursor = learner.db.cursor()
        cursor.execute("SELECT * FROM rl_user_feedback_summary WHERE context_key = 's1'")
        row = cursor.fetchone()
        assert row is not None
        assert abs(dict(row)["avg_rating"] - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Coordinator integration — user_feedback learner is retrievable
# ---------------------------------------------------------------------------


class TestCoordinatorUserFeedbackIntegration:
    def test_coordinator_can_retrieve_user_feedback_learner(self):
        from victor.framework.rl.coordinator import get_rl_coordinator

        coord = get_rl_coordinator()
        learner = coord.get_learner("user_feedback")
        assert learner is not None
        assert isinstance(learner, UserFeedbackLearner)

    def test_record_outcome_writes_feedback_source_column(self):
        from victor.framework.rl.coordinator import get_rl_coordinator
        from victor.core.schema import Tables

        coord = get_rl_coordinator()
        outcome = create_outcome_with_user_feedback(session_id="unit-test", rating=0.7)
        coord.record_outcome("user_feedback", outcome)

        cursor = coord.db.cursor()
        cursor.execute(
            f"SELECT feedback_source, session_id FROM {Tables.RL_OUTCOME} "
            f"WHERE feedback_source = 'user' LIMIT 1"
        )
        row = cursor.fetchone()
        assert row is not None
        assert dict(row)["feedback_source"] == "user"
        assert dict(row)["session_id"] == "unit-test"


# ---------------------------------------------------------------------------
# UsageAnalytics.persist_to_rl_database
# ---------------------------------------------------------------------------


class TestUsageAnalyticsPersistBridge:
    def setup_method(self):
        from victor.agent.usage_analytics import UsageAnalytics

        UsageAnalytics.reset_instance()
        self.ua = UsageAnalytics.get_instance()

    def teardown_method(self):
        from victor.agent.usage_analytics import UsageAnalytics

        UsageAnalytics.reset_instance()

    def test_returns_false_when_no_sessions(self):
        result = self.ua.persist_to_rl_database()
        assert result is False

    def test_returns_true_after_session(self):
        self.ua.start_session()
        self.ua.record_turn()
        self.ua.end_session()
        result = self.ua.persist_to_rl_database()
        assert result is True

    def test_does_not_duplicate_session_aggregation_logic(self):
        """persist_to_rl_database must call get_session_summary(), not reimplement it."""
        import inspect

        src = inspect.getsource(self.ua.persist_to_rl_database)
        assert (
            "get_session_summary" in src
        ), "persist_to_rl_database must delegate to get_session_summary()"


# ---------------------------------------------------------------------------
# MetaLearningCoordinator
# ---------------------------------------------------------------------------


class TestMetaLearningCoordinator:
    def test_importable(self):
        from victor.framework.rl.meta_learning import (
            MetaLearningCoordinator,
        )  # noqa: F401

    def test_get_meta_learning_coordinator_returns_instance(self):
        from victor.framework.rl.meta_learning import (
            get_meta_learning_coordinator,
            MetaLearningCoordinator,
        )

        coord = get_meta_learning_coordinator()
        assert isinstance(coord, MetaLearningCoordinator)

    def test_aggregate_session_metrics_returns_dict(self):
        from victor.framework.rl.meta_learning import MetaLearningCoordinator
        from victor.agent.usage_analytics import UsageAnalytics

        UsageAnalytics.reset_instance()
        coord = MetaLearningCoordinator()
        result = coord.aggregate_session_metrics()
        assert isinstance(result, dict)
        UsageAnalytics.reset_instance()

    def test_aggregate_session_metrics_with_data(self):
        from victor.framework.rl.meta_learning import MetaLearningCoordinator
        from victor.agent.usage_analytics import UsageAnalytics

        UsageAnalytics.reset_instance()
        ua = UsageAnalytics.get_instance()
        ua.start_session()
        ua.record_turn()
        ua.end_session()

        coord = MetaLearningCoordinator()
        result = coord.aggregate_session_metrics()
        assert "avg_turns_per_session" in result
        UsageAnalytics.reset_instance()

    def test_detect_long_term_trends_no_data(self):
        from victor.framework.rl.meta_learning import MetaLearningCoordinator

        coord = MetaLearningCoordinator()
        trends = coord.detect_long_term_trends(repo_id="no-data-repo")
        assert isinstance(trends, dict)
        assert trends.get("status") in ("no_historical_data", "insufficient_data", "ok")

    def test_get_consolidated_recommendations_returns_list(self):
        from victor.framework.rl.meta_learning import MetaLearningCoordinator

        coord = MetaLearningCoordinator()
        recs = coord.get_consolidated_recommendations()
        assert isinstance(recs, list)

    def test_extends_rl_coordinator_not_replaces(self):
        from victor.framework.rl.meta_learning import MetaLearningCoordinator
        from victor.framework.rl.coordinator import RLCoordinator

        assert issubclass(MetaLearningCoordinator, RLCoordinator)
