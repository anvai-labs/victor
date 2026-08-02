# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Priority 4 Phase 4: Production Readiness test suite.

Validates feature flag gating, monitoring integration, and gradual rollout
controls introduced in Weeks 23-24 of the Priority 4 design.

Covers:
  - USE_LEARNING_FROM_EXECUTION flag existence and default state
  - get_meta_learning_coordinator() respects the flag
  - get_recommendation_explainer() respects the flag
  - Priority 4 Prometheus metrics emitted when flag enabled
  - Priority 4 Prometheus metrics absent when flag disabled
  - No new DB tables created by Phase 4 code
"""

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

# ---------------------------------------------------------------------------
# Part 1: Feature Flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """Feature flag existence and default behaviour."""

    def test_flag_exists_in_enum(self):
        """USE_LEARNING_FROM_EXECUTION must be a member of FeatureFlag."""
        assert hasattr(
            FeatureFlag, "USE_LEARNING_FROM_EXECUTION"
        ), "FeatureFlag.USE_LEARNING_FROM_EXECUTION missing — add it to feature_flags.py"

    def test_flag_value_string(self):
        """Enum value must be the snake_case string."""
        assert FeatureFlag.USE_LEARNING_FROM_EXECUTION.value == "use_learning_from_execution"

    def test_env_var_name(self):
        """get_env_var_name() must return VICTOR_USE_LEARNING_FROM_EXECUTION."""
        assert FeatureFlag.USE_LEARNING_FROM_EXECUTION.get_env_var_name() == (
            "VICTOR_USE_LEARNING_FROM_EXECUTION"
        )

    def test_flag_defaults_to_enabled(self):
        """Flag must default to True (gradual rollout: on by default)."""
        mgr = get_feature_flag_manager()
        # Without env override, default_enabled=True in FeatureFlagConfig
        with patch.dict("os.environ", {}, clear=False):
            # Remove env var if set
            import os

            env_key = "VICTOR_USE_LEARNING_FROM_EXECUTION"
            old = os.environ.pop(env_key, None)
            try:
                assert mgr.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION) is True
            finally:
                if old is not None:
                    os.environ[env_key] = old

    def test_flag_can_be_disabled_via_env(self):
        """Setting env var to 'false' must disable the flag."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            mgr = get_feature_flag_manager()
            assert mgr.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION) is False

    def test_flag_can_be_enabled_via_env(self):
        """Setting env var to 'true' must enable the flag."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            mgr = get_feature_flag_manager()
            assert mgr.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION) is True


# ---------------------------------------------------------------------------
# Part 2: MetaLearningCoordinator Flag Gating
# ---------------------------------------------------------------------------


class TestMetaLearningCoordinatorGating:
    """get_meta_learning_coordinator() must respect the feature flag."""

    def test_returns_meta_coordinator_when_enabled(self):
        """When flag is enabled, must return MetaLearningCoordinator."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            # Reset singleton to pick up flag change
            import victor.framework.rl.meta_learning as ml_mod

            ml_mod._meta_coordinator = None

            from victor.framework.rl.meta_learning import (
                get_meta_learning_coordinator,
                MetaLearningCoordinator,
            )

            coord = get_meta_learning_coordinator()
            assert isinstance(
                coord, MetaLearningCoordinator
            ), f"Expected MetaLearningCoordinator, got {type(coord)}"
            ml_mod._meta_coordinator = None  # cleanup

    def test_returns_base_coordinator_when_disabled(self):
        """When flag is disabled, must fall back to base RLCoordinator."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            import victor.framework.rl.meta_learning as ml_mod

            ml_mod._meta_coordinator = None

            from victor.framework.rl.meta_learning import get_meta_learning_coordinator
            from victor.framework.rl.coordinator import RLCoordinator
            from victor.framework.rl.meta_learning import MetaLearningCoordinator

            coord = get_meta_learning_coordinator()
            assert not isinstance(
                coord, MetaLearningCoordinator
            ), "Expected base RLCoordinator when flag disabled, got MetaLearningCoordinator"
            assert isinstance(coord, RLCoordinator)
            ml_mod._meta_coordinator = None  # cleanup


# ---------------------------------------------------------------------------
# Part 3: RecommendationExplainer Flag Gating
# ---------------------------------------------------------------------------


class TestExplainerGating:
    """get_recommendation_explainer() must respect the feature flag."""

    def test_returns_explainer_when_enabled(self):
        """When flag is enabled, must return RecommendationExplainer."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            from victor.framework.rl.explainability import (
                get_recommendation_explainer,
                RecommendationExplainer,
            )

            explainer = get_recommendation_explainer()
            assert isinstance(explainer, RecommendationExplainer)

    def test_returns_none_when_disabled(self):
        """When flag is disabled, must return None."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            from victor.framework.rl.explainability import get_recommendation_explainer

            result = get_recommendation_explainer()
            assert result is None, f"Expected None when flag disabled, got {type(result)}"

    def test_returns_none_allows_safe_skip(self):
        """None return must allow callers to skip explanation safely (no AttributeError)."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            from victor.framework.rl.explainability import get_recommendation_explainer

            explainer = get_recommendation_explainer()
            # Safe usage pattern: if explainer is not None: explainer.explain_...
            if explainer is not None:
                pytest.fail("Expected None, not a real explainer when disabled")
            # No crash — the None check is the correct guard


# ---------------------------------------------------------------------------
# Part 4: Prometheus Metrics Integration
# ---------------------------------------------------------------------------


class TestPrometheusMetrics:
    """Priority 4 metrics appear in Prometheus output when flag enabled."""

    def _make_exporter_with_mock_coord(self):
        """Create RLMetricsExporter with a minimal mock coordinator."""
        from victor.framework.rl.metrics import RLMetricsExporter

        exporter = RLMetricsExporter()

        mock_coord = MagicMock()
        # user_feedback learner
        fb_learner = MagicMock()
        fb_learner.get_feedback_stats.return_value = {
            "total_feedback": 5,
            "avg_rating": 0.82,
            "contexts_with_feedback": 3,
        }
        # model_selector learner
        model_learner = MagicMock()
        model_learner.get_optimal_threshold.return_value = 0.65

        # quality_weights learner
        qw_learner = MagicMock()
        qw_learner.export_metrics.return_value = {"user_preference_count": 7}

        def _get_learner(name):
            if name == "user_feedback":
                return fb_learner
            if name == "model_selector":
                return model_learner
            if name == "quality_weights":
                return qw_learner
            return MagicMock()

        mock_coord.get_learner.side_effect = _get_learner
        mock_coord.db.cursor.return_value.fetchall.return_value = []
        mock_coord.db.cursor.return_value.fetchone.return_value = (0, 0, 0)
        exporter.set_coordinator(mock_coord)
        return exporter

    def test_priority4_metrics_present_when_enabled(self):
        """Prometheus output must include user_feedback and threshold metrics when flag on."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            exporter = self._make_exporter_with_mock_coord()
            output = exporter.export_prometheus()

        assert (
            "victor_rl_user_feedback_total" in output
        ), "Expected victor_rl_user_feedback_total in Prometheus output when flag enabled"
        assert "victor_rl_user_feedback_avg_rating" in output

    def test_priority4_metrics_absent_when_disabled(self):
        """Prometheus output must NOT include Priority 4 metrics when flag off."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            exporter = self._make_exporter_with_mock_coord()
            output = exporter.export_prometheus()

        assert (
            "victor_rl_user_feedback_total" not in output
        ), "Priority 4 metrics must be absent when USE_LEARNING_FROM_EXECUTION=false"

    def test_base_metrics_always_present(self):
        """Core RL metrics must always appear regardless of Priority 4 flag."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "false"}):
            exporter = self._make_exporter_with_mock_coord()
            output = exporter.export_prometheus()

        assert "victor_rl_outcomes_total" in output
        assert "victor_rl_success_rate" in output
        assert "victor_rl_active_learners" in output

    def test_priority4_prometheus_export_does_not_raise(self):
        """_export_priority4_metrics() must not raise even if learner data is missing."""
        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            from victor.framework.rl.metrics import RLMetricsExporter

            exporter = RLMetricsExporter()

            # Coordinator with learners that raise
            mock_coord = MagicMock()
            mock_coord.get_learner.side_effect = RuntimeError("DB unavailable")
            mock_coord.db.cursor.return_value.fetchall.return_value = []
            mock_coord.db.cursor.return_value.fetchone.return_value = (0, 0, 0)
            exporter.set_coordinator(mock_coord)

            output = exporter.export_prometheus()
            # Must not raise and must still contain core metrics
            assert "victor_rl_outcomes_total" in output


# ---------------------------------------------------------------------------
# Part 5: No New Tables (schema hygiene)
# ---------------------------------------------------------------------------


class TestSchemaHygiene:
    """Phase 4 must not introduce new database tables."""

    ALLOWED_TABLES = {
        "rl_outcome",
        "rl_metric",
        "rl_q_value",
        "rl_transition",
        "rl_param",
        "rl_task_stat",
        "rl_pattern",
        "rl_user_feedback_summary",
        "sys_metadata",
    }

    def _get_tables(self, db_path: str) -> set:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        return tables

    def test_no_new_tables_after_flag_check(self, tmp_path):
        """Importing and calling Phase 4 factories must not create unknown tables."""
        import victor.framework.rl.meta_learning as ml_mod

        ml_mod._meta_coordinator = None

        db_path = str(tmp_path / "test.db")

        with patch.dict("os.environ", {"VICTOR_USE_LEARNING_FROM_EXECUTION": "true"}):
            from victor.framework.rl.meta_learning import MetaLearningCoordinator

            coord = MetaLearningCoordinator(db_path=db_path)

        tables = self._get_tables(db_path)
        unknown = tables - self.ALLOWED_TABLES
        assert not unknown, (
            f"Phase 4 created unexpected DB tables: {unknown}. "
            "New data should use existing tables or metadata fields."
        )
        ml_mod._meta_coordinator = None
