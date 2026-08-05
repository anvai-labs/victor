# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Behavioral tests for data analysis workflow escape hatches."""

from victor_dataanalysis.escape_hatches import (
    CONDITIONS,
    TRANSFORMS,
    aggregate_model_results,
    analysis_confidence,
    merge_parallel_stats,
    model_selection_criteria,
    quality_threshold,
    should_retry_cleaning,
    should_tune_more,
)


class TestShouldRetryCleaning:
    def test_validation_passed_is_done(self):
        assert should_retry_cleaning({"validation_passed": True}) == "done"

    def test_failed_validation_retries(self):
        ctx = {"validation_passed": False, "iteration": 1, "max_iterations": 3}
        assert should_retry_cleaning(ctx) == "retry"

    def test_max_iterations_stops_retrying(self):
        ctx = {"validation_passed": False, "iteration": 3, "max_iterations": 3}
        assert should_retry_cleaning(ctx) == "done"


class TestShouldTuneMore:
    def test_threshold_met_is_done(self):
        ctx = {"metrics": {"primary_metric": 0.95}, "iteration": 1}
        assert should_tune_more(ctx) == "done"

    def test_below_threshold_tunes(self):
        ctx = {"metrics": {"primary_metric": 0.5}, "iteration": 1, "max_iterations": 3}
        assert should_tune_more(ctx) == "tune"

    def test_custom_threshold_respected(self):
        ctx = {
            "metrics": {"primary_metric": 0.75},
            "performance_threshold": 0.7,
            "iteration": 1,
        }
        assert should_tune_more(ctx) == "done"

    def test_iteration_budget_exhausted_is_done(self):
        ctx = {"metrics": {"primary_metric": 0.1}, "iteration": 3, "max_iterations": 3}
        assert should_tune_more(ctx) == "done"


class TestQualityThreshold:
    def test_high_quality(self):
        ctx = {"quality_score": 0.95, "missing_pct": 2, "outlier_count": 3}
        assert quality_threshold(ctx) == "high_quality"

    def test_acceptable_quality(self):
        ctx = {"quality_score": 0.75, "missing_pct": 10, "outlier_count": 20}
        assert quality_threshold(ctx) == "acceptable"

    def test_needs_cleanup(self):
        ctx = {"quality_score": 0.5, "missing_pct": 40}
        assert quality_threshold(ctx) == "needs_cleanup"

    def test_defaults_need_cleanup(self):
        assert quality_threshold({}) == "needs_cleanup"

    def test_outliers_downgrade_high_quality(self):
        ctx = {"quality_score": 0.95, "missing_pct": 2, "outlier_count": 50}
        assert quality_threshold(ctx) == "acceptable"


class TestModelSelectionCriteria:
    def test_no_models(self):
        assert model_selection_criteria({"evaluation_results": []}) == "no_models"

    def test_excellent_model(self):
        ctx = {"evaluation_results": [{"score": 0.97}, {"score": 0.6}]}
        assert model_selection_criteria(ctx) == "excellent"

    def test_good_model(self):
        ctx = {"evaluation_results": [{"score": 0.88}]}
        assert model_selection_criteria(ctx) == "good"

    def test_acceptable_model(self):
        ctx = {"evaluation_results": [{"score": 0.75}]}
        assert model_selection_criteria(ctx) == "acceptable"

    def test_needs_improvement(self):
        ctx = {"evaluation_results": [{"score": 0.4}]}
        assert model_selection_criteria(ctx) == "needs_improvement"


class TestAnalysisConfidence:
    def test_small_sample_is_low_confidence(self):
        ctx = {"sample_size": 50, "confidence_score": 0.9}
        assert analysis_confidence(ctx) == "low"

    def test_many_uncertainties_is_low_confidence(self):
        ctx = {"sample_size": 1000, "uncertainty_areas": ["a"] * 6}
        assert analysis_confidence(ctx) == "low"

    def test_high_confidence(self):
        ctx = {"sample_size": 1000, "confidence_score": 0.9, "uncertainty_areas": ["a"]}
        assert analysis_confidence(ctx) == "high"

    def test_medium_confidence(self):
        ctx = {"sample_size": 500, "confidence_score": 0.6, "uncertainty_areas": ["a", "b", "c"]}
        assert analysis_confidence(ctx) == "medium"


class TestTransforms:
    def test_merge_parallel_stats_collects_results(self):
        ctx = {
            "statistics": {"mean": 5.0},
            "correlation_matrix": {"a_b": 0.7},
            "anomalies": ["row_42"],
        }
        merged = merge_parallel_stats(ctx)

        assert merged["merged"] is True
        assert merged["statistics"] == {"mean": 5.0}
        assert merged["anomalies"] == ["row_42"]

    def test_merge_parallel_stats_defaults(self):
        merged = merge_parallel_stats({})

        assert merged["statistics"] == {}
        assert merged["anomalies"] == []

    def test_aggregate_model_results_picks_best(self):
        ctx = {
            "rf_model": {"metrics": {"accuracy": 0.85}, "status": "ok"},
            "xgb_model": {"metrics": {"accuracy": 0.92}, "status": "ok"},
        }
        aggregated = aggregate_model_results(ctx)

        assert aggregated["best_model_name"] == "xgb_model"
        assert aggregated["best_model_score"] == 0.92
        assert len(aggregated["all_models"]) == 2

    def test_aggregate_model_results_no_models(self):
        aggregated = aggregate_model_results({})

        assert aggregated["best_model"] is None
        assert aggregated["best_model_score"] == 0


class TestRegistries:
    def test_conditions_registry_complete(self):
        assert set(CONDITIONS) == {
            "should_retry_cleaning",
            "should_tune_more",
            "quality_threshold",
            "model_selection_criteria",
            "analysis_confidence",
        }
        assert all(callable(fn) for fn in CONDITIONS.values())

    def test_transforms_registry_complete(self):
        assert set(TRANSFORMS) == {
            "merge_parallel_stats",
            "aggregate_model_results",
        }
        assert all(callable(fn) for fn in TRANSFORMS.values())
