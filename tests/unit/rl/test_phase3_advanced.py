# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Priority 4 Phase 3 — transfer learning, preference learning, explainability."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.framework.rl.learners.cross_vertical import CrossVerticalLearner
from victor.framework.rl.learners.quality_weights import (
    QualityWeightLearner,
    QualityDimension,
)
from victor.framework.rl.explainability import RecommendationExplainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_cross_vertical() -> CrossVerticalLearner:
    db = _make_db()
    # Create rl_outcome table so cross_vertical queries work in isolation
    from victor.core.schema import Tables, Schema

    db.execute(Schema.RL_OUTCOME)
    db.commit()
    return CrossVerticalLearner(name="cross_vertical", db_connection=db)


def _make_quality_weights() -> QualityWeightLearner:
    return QualityWeightLearner(name="quality_weights", db_connection=_make_db())


# ---------------------------------------------------------------------------
# Transfer Learning — CrossVerticalLearner
# ---------------------------------------------------------------------------


class TestCrossVerticalExportPatterns:
    def test_export_returns_dict_with_schema_version(self):
        learner = _make_cross_vertical()
        exported = learner.export_patterns()
        assert isinstance(exported, dict)
        assert exported["schema_version"] == 1
        assert "patterns" in exported
        assert "exported_at" in exported

    def test_export_empty_when_no_patterns(self):
        learner = _make_cross_vertical()
        exported = learner.export_patterns()
        assert exported["pattern_count"] == 0
        assert exported["patterns"] == []

    def test_export_serializable_to_json(self):
        learner = _make_cross_vertical()
        exported = learner.export_patterns()
        # Must not raise
        json_str = json.dumps(exported)
        assert len(json_str) > 0

    def test_export_scopes_to_repo_id(self):
        learner = _make_cross_vertical()
        exported = learner.export_patterns(repo_id="project-a")
        assert exported["source_repo_id"] == "project-a"

    def test_export_respects_min_confidence(self):
        learner = _make_cross_vertical()
        exported_strict = learner.export_patterns(min_confidence=0.9)
        exported_loose = learner.export_patterns(min_confidence=0.1)
        # With no data both are empty; at least the filter is applied
        assert exported_strict["pattern_count"] <= exported_loose["pattern_count"]


class TestCrossVerticalImportPatterns:
    def test_import_unknown_schema_returns_0(self):
        learner = _make_cross_vertical()
        count = learner.import_patterns({"schema_version": 99, "patterns": []})
        assert count == 0

    def test_import_empty_patterns_returns_0(self):
        learner = _make_cross_vertical()
        count = learner.import_patterns({"schema_version": 1, "patterns": []})
        assert count == 0

    def test_import_then_export_roundtrip(self):
        source = _make_cross_vertical()
        target = _make_cross_vertical()

        # Seed a pattern into source via DB directly
        cursor = source.db.cursor()
        from victor.core.schema import Tables

        cursor.execute(
            f"""INSERT INTO {Tables.RL_PATTERN}
            (pattern_id, task_type, pattern_name, avg_quality, confidence,
             source_verticals, recommended_mode, recommendation,
             sample_count, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            (
                "p1",
                "analysis",
                "high_quality",
                0.85,
                0.75,
                json.dumps(["coding", "devops"]),
                "BUILD",
                "Use structured output",
                25,
            ),
        )
        source.db.commit()

        exported = source.export_patterns(min_confidence=0.5)
        assert exported["pattern_count"] == 1

        imported = target.import_patterns(exported, source_repo_id="src")
        assert imported == 1

    def test_import_applies_confidence_decay(self):
        source = _make_cross_vertical()
        target = _make_cross_vertical()

        from victor.core.schema import Tables

        cursor = source.db.cursor()
        cursor.execute(
            f"""INSERT INTO {Tables.RL_PATTERN}
            (pattern_id, task_type, pattern_name, avg_quality, confidence,
             source_verticals, recommended_mode, recommendation,
             sample_count, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            (
                "p2",
                "edit",
                "edit_pattern",
                0.80,
                0.90,
                json.dumps(["coding"]),
                None,
                "Verify before edit",
                30,
            ),
        )
        source.db.commit()

        exported = source.export_patterns(min_confidence=0.5)
        target.import_patterns(exported, source_repo_id="src", confidence_decay=0.8)

        # Verify decay was applied (0.9 * 0.8 = 0.72)
        from victor.core.schema import Tables

        cursor2 = target.db.cursor()
        cursor2.execute(f"SELECT confidence FROM {Tables.RL_PATTERN} WHERE task_type='edit'")
        row = cursor2.fetchone()
        assert row is not None
        assert abs(dict(row)["confidence"] - 0.72) < 0.01

    def test_import_is_idempotent(self):
        source = _make_cross_vertical()
        target = _make_cross_vertical()
        data = {
            "schema_version": 1,
            "patterns": [
                {
                    "task_type": "analysis",
                    "pattern_name": "p",
                    "avg_quality": 0.8,
                    "confidence": 0.7,
                    "source_verticals": ["coding"],
                    "recommended_mode": None,
                    "recommendation": "test",
                    "sample_count": 10,
                }
            ],
        }
        c1 = target.import_patterns(data, source_repo_id="src")
        c2 = target.import_patterns(data, source_repo_id="src")
        assert c1 == 1
        assert c2 == 0  # INSERT OR IGNORE — second import is a no-op


class TestCrossVerticalAdaptPatterns:
    def test_adapt_patterns_returns_list(self):
        learner = _make_cross_vertical()
        result = learner.adapt_patterns("coding", "research")
        assert isinstance(result, list)

    def test_adapt_patterns_returns_rl_recommendations(self):
        learner = _make_cross_vertical()
        # Seed some outcomes for source vertical
        from victor.core.schema import Tables

        cursor = learner.db.cursor()
        for i in range(12):
            cursor.execute(
                f"""INSERT INTO {Tables.RL_OUTCOME}
                (learner_id, provider, model, task_type, vertical, success, quality_score, metadata)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    "cross_vertical",
                    "anthropic",
                    "claude",
                    "analysis",
                    "coding",
                    1,
                    0.85,
                    "{}",
                ),
            )
        learner.db.commit()

        recs = learner.adapt_patterns("coding", "research", min_confidence=0.0)
        assert len(recs) > 0
        for rec in recs:
            assert isinstance(rec, RLRecommendation)
            assert rec.metadata["transfer_type"] == "domain_adaptation"
            assert rec.metadata["source_vertical"] == "coding"
            assert rec.metadata["target_vertical"] == "research"

    def test_adapt_confidence_has_domain_shift_penalty(self):
        learner = _make_cross_vertical()
        from victor.core.schema import Tables

        cursor = learner.db.cursor()
        for i in range(50):
            cursor.execute(
                f"""INSERT INTO {Tables.RL_OUTCOME}
                (learner_id, provider, model, task_type, vertical, success, quality_score, metadata)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    "cross_vertical",
                    "anthropic",
                    "claude",
                    "edit",
                    "coding",
                    1,
                    0.95,
                    "{}",
                ),
            )
        learner.db.commit()

        recs = learner.adapt_patterns("coding", "devops", min_confidence=0.0)
        for rec in recs:
            # Domain-shift penalty applied (0.85 multiplier)
            assert rec.confidence < 0.85
            assert rec.metadata["confidence_decay"] == 0.85


# ---------------------------------------------------------------------------
# Preference Learning — QualityWeightLearner
# ---------------------------------------------------------------------------


class TestQualityWeightsPreferenceLearning:
    def test_record_user_preference_stores_weight(self):
        learner = _make_quality_weights()
        learner.record_user_preference("user-1", QualityDimension.ACCURACY, 2.0, "analysis")
        prefs = learner._user_preferences.get("user-1", {}).get("analysis", {})
        assert QualityDimension.ACCURACY in prefs
        assert prefs[QualityDimension.ACCURACY] == 2.0

    def test_record_user_preference_clamps_to_valid_range(self):
        learner = _make_quality_weights()
        learner.record_user_preference("u1", QualityDimension.RELEVANCE, 99.0)
        prefs = learner._user_preferences["u1"]["default"]
        assert prefs[QualityDimension.RELEVANCE] <= learner.MAX_WEIGHT

        learner.record_user_preference("u1", QualityDimension.RELEVANCE, -5.0)
        prefs = learner._user_preferences["u1"]["default"]
        assert prefs[QualityDimension.RELEVANCE] >= learner.MIN_WEIGHT

    def test_get_personalized_weights_returns_dict(self):
        learner = _make_quality_weights()
        weights = learner.get_personalized_weights("new-user", "analysis")
        assert isinstance(weights, dict)
        assert QualityDimension.RELEVANCE in weights

    def test_get_personalized_weights_blends_preference(self):
        learner = _make_quality_weights()
        # Force a known global weight
        learner._weights["analysis"] = dict.fromkeys(QualityDimension.ALL, 1.0)
        learner.record_user_preference("u1", QualityDimension.CODE_QUALITY, 3.0, "analysis")

        blended = learner.get_personalized_weights("u1", "analysis")
        # Expect 70% * 1.0 + 30% * 3.0 = 1.6
        assert abs(blended[QualityDimension.CODE_QUALITY] - 1.6) < 0.01

    def test_get_personalized_weights_fallback_to_global(self):
        learner = _make_quality_weights()
        learner._weights["analysis"] = dict.fromkeys(QualityDimension.ALL, 1.5)
        weights = learner.get_personalized_weights("unknown-user", "analysis")
        for dim in QualityDimension.ALL:
            assert weights[dim] == 1.5

    def test_preference_persisted_to_db(self):
        from victor.core.schema import Tables

        learner = _make_quality_weights()
        learner.record_user_preference("u2", QualityDimension.CONCISENESS, 0.5, "default")
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT param_value FROM {Tables.RL_PARAM} "
            f"WHERE learner_id = 'quality_weights' AND param_key LIKE 'user_weight:u2:%'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert abs(row[0] - 0.5) < 0.01

    def test_preference_loaded_cross_session(self):
        db = _make_db()
        learner1 = QualityWeightLearner(name="quality_weights", db_connection=db)
        learner1.record_user_preference("u3", QualityDimension.ACCURACY, 2.5, "default")

        learner2 = QualityWeightLearner(name="quality_weights", db_connection=db)
        learner2._load_user_preferences()
        prefs = learner2._user_preferences.get("u3", {}).get("default", {})
        assert abs(prefs.get(QualityDimension.ACCURACY, 0) - 2.5) < 0.01

    def test_multiple_users_independent(self):
        learner = _make_quality_weights()
        learner.record_user_preference("alice", QualityDimension.ACCURACY, 2.0)
        learner.record_user_preference("bob", QualityDimension.ACCURACY, 0.5)
        alice_w = learner.get_personalized_weights("alice")
        bob_w = learner.get_personalized_weights("bob")
        # Alice prefers accuracy more than Bob
        assert alice_w[QualityDimension.ACCURACY] > bob_w[QualityDimension.ACCURACY]

    def test_export_metrics_reports_preference_count(self):
        learner = _make_quality_weights()
        learner.record_user_preference("u1", QualityDimension.RELEVANCE, 1.8)
        learner.record_user_preference("u2", QualityDimension.RELEVANCE, 1.2)
        metrics = learner.export_metrics()
        assert metrics["user_preference_count"] == 2


# ---------------------------------------------------------------------------
# Explainability — RecommendationExplainer
# ---------------------------------------------------------------------------


class TestRecommendationExplainer:
    def _make_explainer(self) -> RecommendationExplainer:
        mock_coord = MagicMock()
        mock_learner = MagicMock()
        mock_learner.get_tool_stats.return_value = {
            "q_value": 0.78,
            "selection_count": 42,
            "success_rate": 0.85,
            "task_q_values": {"analysis": 0.81},
        }
        mock_learner._analytics = None
        mock_learner._predictor = None
        mock_coord.get_learner.return_value = mock_learner
        return RecommendationExplainer(mock_coord)

    def test_explain_tool_recommendation_returns_dict(self):
        explainer = self._make_explainer()
        result = explainer.explain_tool_recommendation("read", "analysis")
        assert isinstance(result, dict)
        assert result["tool"] == "read"
        assert result["task_type"] == "analysis"

    def test_explain_tool_recommendation_has_signals(self):
        explainer = self._make_explainer()
        result = explainer.explain_tool_recommendation("read", "analysis")
        assert "signals" in result
        assert len(result["signals"]) >= 1
        for s in result["signals"]:
            assert "source" in s
            assert "description" in s

    def test_explain_tool_recommendation_has_confidence_label(self):
        explainer = self._make_explainer()
        result = explainer.explain_tool_recommendation("read", "analysis")
        assert result["confidence_label"] in ("low", "medium", "high")

    def test_explain_tool_recommendation_has_summary(self):
        explainer = self._make_explainer()
        result = explainer.explain_tool_recommendation("read", "analysis")
        assert isinstance(result["summary"], str)
        assert "read" in result["summary"]

    def test_explain_tool_rankings_preserves_order(self):
        explainer = self._make_explainer()
        rankings = [("read", 0.85, 0.9), ("write", 0.72, 0.8), ("search", 0.6, 0.7)]
        results = explainer.explain_tool_rankings(rankings, "analysis")
        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert results[0]["tool"] == "read"
        assert results[2]["rank"] == 3

    def test_explain_tool_rankings_includes_ranking_score(self):
        explainer = self._make_explainer()
        rankings = [("grep", 0.75, 0.85)]
        results = explainer.explain_tool_rankings(rankings, "search")
        assert results[0]["ranking_score"] == 0.75
        assert results[0]["ranking_confidence"] == 0.85

    def test_explain_model_recommendation_returns_dict(self):
        mock_coord = MagicMock()
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = RLRecommendation(
            value=0.8,
            confidence=0.75,
            reason="High Q-value from 50 sessions",
            sample_size=50,
            is_baseline=False,
        )
        mock_learner.get_optimal_threshold.return_value = 0.65
        mock_coord.get_learner.return_value = mock_learner
        explainer = RecommendationExplainer(mock_coord)

        result = explainer.explain_model_recommendation("anthropic", "analysis")
        assert result["provider"] == "anthropic"
        assert len(result["signals"]) >= 2
        assert any(s["source"] == "learned_threshold" for s in result["signals"])

    def test_annotate_recommendation_injects_explanation(self):
        explainer = self._make_explainer()
        rec = RLRecommendation(
            value=0.8,
            confidence=0.75,
            reason="test",
            sample_size=10,
        )
        annotated = explainer.annotate_recommendation(rec, "tool_selector", {"tool": "read"})
        assert "explanation" in annotated.metadata
        assert annotated.metadata["explanation"]["learner"] == "tool_selector"
        assert annotated.metadata["explanation"]["confidence_label"] in (
            "low",
            "medium",
            "high",
        )

    def test_annotate_recommendation_does_not_modify_value(self):
        explainer = self._make_explainer()
        rec = RLRecommendation(
            value=0.88,
            confidence=0.72,
            reason="orig",
            sample_size=15,
        )
        explainer.annotate_recommendation(rec, "tool_selector", {})
        assert rec.value == 0.88
        assert rec.confidence == 0.72
        assert rec.reason == "orig"

    def test_explain_no_learner_data_returns_default(self):
        mock_coord = MagicMock()
        mock_coord.get_learner.return_value = None
        explainer = RecommendationExplainer(mock_coord)
        result = explainer.explain_tool_recommendation("unknown", "default")
        assert result["signals"][0]["source"] == "default"
        assert result["blended_score"] == 0.5

    def test_confidence_label_boundaries(self):
        mock_coord = MagicMock()
        mock_coord.get_learner.return_value = None
        explainer = RecommendationExplainer(mock_coord)
        assert explainer._confidence_label(0.2) == "low"
        assert explainer._confidence_label(0.5) == "medium"
        assert explainer._confidence_label(0.8) == "high"


# ---------------------------------------------------------------------------
# End-to-end: coordinator integration
# ---------------------------------------------------------------------------


class TestPhase3CoordinatorIntegration:
    def test_cross_vertical_learner_has_transfer_methods(self):
        from victor.framework.rl.coordinator import get_rl_coordinator

        coord = get_rl_coordinator()
        learner = coord.get_learner("cross_vertical")
        assert hasattr(learner, "export_patterns")
        assert hasattr(learner, "import_patterns")
        assert hasattr(learner, "adapt_patterns")

    def test_quality_weights_learner_has_preference_methods(self):
        from victor.framework.rl.coordinator import get_rl_coordinator

        coord = get_rl_coordinator()
        learner = coord.get_learner("quality_weights")
        assert hasattr(learner, "record_user_preference")
        assert hasattr(learner, "get_personalized_weights")

    def test_explainability_importable(self):
        from victor.framework.rl.explainability import (
            RecommendationExplainer,
        )  # noqa: F401

    def test_explainer_works_with_real_coordinator(self):
        from victor.framework.rl.coordinator import get_rl_coordinator
        from victor.framework.rl.explainability import RecommendationExplainer

        coord = get_rl_coordinator()
        explainer = RecommendationExplainer(coord)
        result = explainer.explain_tool_recommendation("read", "analysis")
        assert isinstance(result, dict)
        assert "signals" in result
