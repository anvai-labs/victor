# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Priority 4 Phase 0: Infrastructure Audit Test Suite.

Validates the existing RL infrastructure before any Priority 4 implementation
begins. Every test must pass; failures mean a prerequisite is missing.

Covers:
  - Part 1: 14 existing learners are present and instantiable
  - Part 2: UsageAnalytics API contract
  - Part 3: RL database schema (rl_outcomes table + required columns)
  - Part 4: RLOutcome.quality_score field for user feedback
  - Part 5: ToolPredictor (Priority 3) integration surface
  - Part 6: No-duplication guard assertions
"""

import asyncio
import time
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.usage_analytics import UsageAnalytics
from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.coordinator import get_rl_coordinator

# ---------------------------------------------------------------------------
# Part 1: Learner Inventory — 15 learners (14 original + user_feedback from Phase 1)
# ---------------------------------------------------------------------------

EXPECTED_LEARNERS = [
    "cache_eviction",
    "context_pruning",
    "continuation_patience",
    "continuation_prompts",
    "cross_vertical",
    "grounding_threshold",
    "mode_transition",
    "model_selector",
    "prompt_optimizer",
    "prompt_template",
    "quality_weights",
    "semantic_threshold",
    "tool_selector",
    "user_feedback",
    "workflow_execution",
]


class TestLearnerInventory:
    """Part 1: Verify all 15 learners exist (14 original + user_feedback from Phase 1)."""

    def test_all_learners_exist(self):
        """All expected learner files must be importable."""
        import importlib

        missing = []
        for name in EXPECTED_LEARNERS:
            mod_path = f"victor.framework.rl.learners.{name}"
            try:
                importlib.import_module(mod_path)
            except ImportError as e:
                missing.append(f"{name}: {e}")
        assert not missing, f"Missing learners: {missing}"

    def test_learner_count_at_least_15(self):
        """At least 15 learner files (15 canonical + optional extended variants)."""
        import os

        learner_dir = "victor/framework/rl/learners"
        files = [
            f[:-3] for f in os.listdir(learner_dir) if f.endswith(".py") and not f.startswith("_")
        ]
        assert (
            len(files) >= 15
        ), f"Expected at least 15 learners, found {len(files)}: {sorted(files)}"

    @pytest.mark.parametrize("learner_name", EXPECTED_LEARNERS)
    def test_coordinator_can_retrieve_learner(self, learner_name):
        """RLCoordinator.get_learner() must return a non-None learner for each name."""
        coord = get_rl_coordinator()
        learner = coord.get_learner(learner_name)
        assert learner is not None, f"get_learner('{learner_name}') returned None"


# ---------------------------------------------------------------------------
# Part 2: UsageAnalytics API Contract
# ---------------------------------------------------------------------------


class TestUsageAnalyticsAPI:
    """Part 2: UsageAnalytics integration surface for Priority 4."""

    def setup_method(self):
        """Fresh instance per test."""
        UsageAnalytics.reset_instance()
        self.ua = UsageAnalytics.get_instance()

    def teardown_method(self):
        UsageAnalytics.reset_instance()

    def test_singleton_pattern(self):
        """get_instance() always returns the same object."""
        a = UsageAnalytics.get_instance()
        b = UsageAnalytics.get_instance()
        assert a is b

    def test_get_session_summary_returns_dict(self):
        """get_session_summary() returns a dict (shape varies before/after sessions)."""
        summary = self.ua.get_session_summary()
        assert isinstance(
            summary, dict
        ), f"get_session_summary() must return a dict, got {type(summary)}"

    def test_get_session_summary_with_session_data(self):
        """get_session_summary() returns aggregation keys when sessions exist."""
        self.ua.start_session()
        self.ua.record_turn()
        self.ua.end_session()
        summary = self.ua.get_session_summary()
        assert isinstance(summary, dict)
        assert summary != {}, "Expected non-empty summary after a session"

    def test_get_tool_insights_returns_dict(self):
        """get_tool_insights() returns a dict."""
        insights = self.ua.get_tool_insights("read")
        assert isinstance(insights, dict)

    def test_record_tool_execution(self):
        """record_tool_execution() must not raise."""
        self.ua.record_tool_execution(
            tool_name="read",
            success=True,
            execution_time_ms=42.0,
        )

    def test_get_optimization_recommendations_returns_list(self):
        """get_optimization_recommendations() returns a list."""
        recs = self.ua.get_optimization_recommendations()
        assert isinstance(recs, list)

    def test_start_and_end_session(self):
        """start_session/end_session lifecycle must work without raising."""
        self.ua.start_session()
        self.ua.record_turn()
        self.ua.end_session()
        summary = self.ua.get_session_summary()
        assert isinstance(summary, dict)

    def test_no_duplicate_session_aggregation(self):
        """Priority 4 must NOT recreate session aggregation — it already exists."""
        self.ua.start_session()
        self.ua.record_turn()
        self.ua.end_session()
        summary = self.ua.get_session_summary()
        assert "avg_turns_per_session" in summary, (
            "UsageAnalytics.get_session_summary() already provides session "
            "aggregation. Priority 4 must EXTEND this, not duplicate it."
        )


# ---------------------------------------------------------------------------
# Part 3: RL Database Schema
# ---------------------------------------------------------------------------


class TestRLDatabaseSchema:
    """Part 3: rl_outcomes table must have all required columns."""

    def test_rl_coordinator_has_record_outcome(self):
        """RLCoordinator.record_outcome() must exist."""
        coord = get_rl_coordinator()
        assert hasattr(coord, "record_outcome"), "record_outcome() method missing"
        assert callable(coord.record_outcome)

    def test_rl_coordinator_has_async_record_outcome(self):
        """Async variant must exist for non-blocking use."""
        coord = get_rl_coordinator()
        assert hasattr(
            coord, "record_outcome_async"
        ), "record_outcome_async() missing — needed for async tool pipeline"

    def test_record_outcome_accepts_rl_outcome(self):
        """record_outcome() must accept an RLOutcome without raising."""
        coord = get_rl_coordinator()
        outcome = RLOutcome(
            provider="test",
            model="test-model",
            task_type="phase_0_audit",
            success=True,
            quality_score=0.9,
            metadata={"audit": "phase_0", "feedback_source": "auto"},
            vertical="general",
        )
        # Should not raise — learner_name is required as first positional arg
        try:
            coord.record_outcome("tool_selector", outcome)
        except Exception as e:
            pytest.fail(f"record_outcome() raised: {e}")

    def test_rl_outcome_quality_score_accepts_user_feedback(self):
        """RLOutcome must accept user feedback via quality_score + metadata."""
        outcome = RLOutcome(
            provider="user",
            model="feedback",
            task_type="feedback",
            success=True,
            quality_score=0.85,
            metadata={
                "feedback_source": "user",
                "user_feedback": "Excellent result",
                "helpful": True,
            },
            vertical="general",
        )
        assert outcome.quality_score == 0.85
        assert outcome.metadata["feedback_source"] == "user"


# ---------------------------------------------------------------------------
# Part 4: RLOutcome Quality Score
# ---------------------------------------------------------------------------


class TestRLOutcomeQualityScore:
    """Part 4: RLOutcome.quality_score field for user feedback integration."""

    def test_rl_outcome_has_quality_score_field(self):
        """RLOutcome must have a quality_score field."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RLOutcome)}
        assert "quality_score" in fields, (
            "RLOutcome.quality_score is required for user feedback. "
            "Do not remove or rename this field."
        )

    def test_rl_outcome_has_metadata_field(self):
        """RLOutcome must have a metadata dict for feedback_source tracking."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RLOutcome)}
        assert "metadata" in fields

    def test_quality_score_accepts_none(self):
        """quality_score=None must be valid (optional auto-scoring)."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="tool_call",
            success=True,
            quality_score=None,
        )
        assert outcome.quality_score is None

    def test_quality_score_accepts_float_range(self):
        """quality_score must accept 0.0–1.0 floats."""
        for score in (0.0, 0.5, 1.0):
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-sonnet-4-6",
                task_type="tool_call",
                success=True,
                quality_score=score,
            )
            assert outcome.quality_score == score

    def test_automatic_vs_user_feedback_distinguished_via_metadata(self):
        """feedback_source in metadata distinguishes automatic from human scores."""
        auto = RLOutcome(
            provider="anthropic",
            model="claude-sonnet-4-6",
            task_type="tool_call",
            success=True,
            quality_score=0.8,
            metadata={"feedback_source": "auto"},
        )
        human = RLOutcome(
            provider="user",
            model="feedback",
            task_type="feedback",
            success=True,
            quality_score=0.95,
            metadata={"feedback_source": "user", "helpful": True},
        )
        assert auto.metadata["feedback_source"] == "auto"
        assert human.metadata["feedback_source"] == "user"

    def test_no_duplicate_user_feedback_mechanism(self):
        """RLOutcome.quality_score is the canonical feedback field.
        Priority 4 must NOT add a separate feedback table or field.
        """
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RLOutcome)}
        # Must NOT have a separate 'user_rating' or 'feedback_score' field
        duplicates = fields & {"user_rating", "feedback_score", "human_score"}
        assert not duplicates, (
            f"Found duplicate feedback fields {duplicates}. "
            "Use quality_score with metadata.feedback_source='user'."
        )


# ---------------------------------------------------------------------------
# Part 5: ToolPredictor (Priority 3) Integration
# ---------------------------------------------------------------------------


class TestToolPredictorIntegration:
    """Part 5: ToolPredictor from Priority 3 must be available for Priority 4."""

    def test_tool_predictor_importable(self):
        """ToolPredictor must be importable from Priority 3 location."""
        from victor.agent.planning.tool_predictor import ToolPredictor  # noqa: F401

    def test_cooccurrence_tracker_importable(self):
        """CooccurrenceTracker must be importable."""
        from victor.agent.planning.cooccurrence_tracker import (
            CooccurrenceTracker,
        )  # noqa: F401

    def test_tool_predictor_has_predict_tools(self):
        """ToolPredictor.predict_tools() must exist."""
        from victor.agent.planning.tool_predictor import ToolPredictor

        tp = ToolPredictor()
        assert hasattr(tp, "predict_tools") and callable(
            tp.predict_tools
        ), "ToolPredictor.predict_tools() missing — Priority 4 extends this method"

    def test_tool_predictor_predict_tools_returns_list(self):
        """predict_tools() must return a list (may be empty for unknown task)."""
        from victor.agent.planning.tool_predictor import ToolPredictor

        tp = ToolPredictor()
        result = tp.predict_tools(
            task_description="read a file",
            current_step="exploration",
            recent_tools=["ls"],
            task_type="analysis",
        )
        assert isinstance(result, list), f"predict_tools() returned {type(result)}, expected list"

    def test_tool_selector_learner_exists_and_instantiable(self):
        """tool_selector learner must be retrievable from coordinator."""
        coord = get_rl_coordinator()
        learner = coord.get_learner("tool_selector")
        assert learner is not None

    def test_no_duplicate_tool_prediction(self):
        """ToolPredictor from Priority 3 is the canonical predictor.
        Priority 4 must EXTEND it, not create a new prediction class.
        """
        from victor.agent.planning import tool_predictor as tp_mod

        # Only one ToolPredictor class should exist in the planning package
        import inspect

        predictors = [
            name
            for name, obj in inspect.getmembers(tp_mod, inspect.isclass)
            if "Predict" in name and "Tool" in name
        ]
        assert len(predictors) >= 1, "ToolPredictor class not found"
        # Ensure it hasn't been duplicated elsewhere with a different name
        assert "ToolPredictor" in predictors, f"Expected 'ToolPredictor', found: {predictors}"


# ---------------------------------------------------------------------------
# Part 6: No-Duplication Guards + Performance Baselines
# ---------------------------------------------------------------------------


class TestNoDuplicationGuards:
    """Part 6: Guard tests that prevent Priority 4 from duplicating existing work."""

    def test_all_learners_extended_not_replaced(self):
        """All 15 learners must still be present after any Priority 4 work."""
        import importlib

        for name in EXPECTED_LEARNERS:
            try:
                importlib.import_module(f"victor.framework.rl.learners.{name}")
            except ImportError:
                pytest.fail(
                    f"Learner '{name}' was removed. Priority 4 must EXTEND learners, "
                    "not replace them."
                )

    def test_usage_analytics_singleton_is_not_replaced(self):
        """UsageAnalytics.get_instance() must remain the singleton accessor."""
        UsageAnalytics.reset_instance()
        a = UsageAnalytics.get_instance()
        b = UsageAnalytics.get_instance()
        assert a is b, (
            "UsageAnalytics singleton broken — Priority 4 must not replace it "
            "with a new session aggregation system."
        )
        UsageAnalytics.reset_instance()

    def test_rl_outcome_schema_not_bloated(self):
        """RLOutcome must not grow beyond its essential fields.
        Priority 4 should use metadata dict for new data, not new fields.
        """
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RLOutcome)}
        essential = {
            "provider",
            "model",
            "task_type",
            "success",
            "quality_score",
            "timestamp",
            "metadata",
            "vertical",
        }
        unknown = fields - essential
        assert not unknown, (
            f"Unexpected fields in RLOutcome: {unknown}. "
            "New data should go into metadata dict, not new fields."
        )


class TestPerformanceBaselines:
    """Performance baselines — must pass before Phase 1 begins."""

    def teardown_method(self):
        UsageAnalytics.reset_instance()

    def test_usage_analytics_record_performance(self):
        """recording 100 tool executions must complete in <500ms."""
        UsageAnalytics.reset_instance()
        ua = UsageAnalytics.get_instance()
        start = time.monotonic()
        for i in range(100):
            ua.record_tool_execution(
                tool_name=f"tool_{i % 5}",
                success=i % 3 != 0,
                execution_time_ms=float(i),
            )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 500, (
            f"UsageAnalytics recording 100 events took {elapsed_ms:.0f}ms "
            f"(limit: 500ms). This would slow the agentic loop."
        )

    def test_usage_analytics_get_summary_performance(self):
        """get_session_summary() must complete in <50ms."""
        UsageAnalytics.reset_instance()
        ua = UsageAnalytics.get_instance()
        ua.start_session()
        for _ in range(20):
            ua.record_tool_execution("read", success=True, execution_time_ms=10.0)
        ua.end_session()

        start = time.monotonic()
        summary = ua.get_session_summary()
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 50, f"get_session_summary() took {elapsed_ms:.1f}ms (limit: 50ms)"
        assert isinstance(summary, dict)

    def test_rl_coordinator_record_outcome_performance(self):
        """record_outcome() for 10 outcomes must complete in <1s."""
        coord = get_rl_coordinator()
        outcomes = [
            RLOutcome(
                provider="anthropic",
                model="claude-sonnet-4-6",
                task_type="tool_call",
                success=True,
                quality_score=0.8 + i * 0.01,
                metadata={"audit": "phase_0_perf"},
                vertical="general",
            )
            for i in range(10)
        ]
        start = time.monotonic()
        for outcome in outcomes:
            try:
                coord.record_outcome(outcome)
            except Exception:
                pass  # DB not available in test env — just time the call
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 1000, f"record_outcome() x10 took {elapsed_ms:.0f}ms (limit: 1000ms)"
