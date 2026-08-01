# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the enhanced data analysis conversation manager and context tracking."""

from victor.framework.extensions import TurnType

from victor_dataanalysis.conversation_enhanced import (
    DataAnalysisContext,
    EnhancedDataAnalysisConversationManager,
)


class TestDataAnalysisContext:
    def test_to_dict_round_trip(self):
        ctx = DataAnalysisContext()
        ctx.datasets_loaded.append("sales.csv")
        ctx.insights_found.append("Q4 revenue spike")

        data = ctx.to_dict()

        assert set(data) == {
            "datasets_loaded",
            "analyses_performed",
            "visualizations_created",
            "insights_found",
        }
        assert data["datasets_loaded"] == ["sales.csv"]
        assert data["insights_found"] == ["Q4 revenue spike"]


class TestEnhancedConversationManager:
    def test_add_message_returns_turn_id(self):
        manager = EnhancedDataAnalysisConversationManager()

        turn_id = manager.add_message("user", "analyze sales.csv", TurnType.USER)

        assert turn_id
        history = manager.get_history()
        assert any(m["content"] == "analyze sales.csv" for m in history)

    def test_track_dataset_deduplicates(self):
        manager = EnhancedDataAnalysisConversationManager()

        manager.track_dataset("sales.csv")
        manager.track_dataset("sales.csv")
        manager.track_dataset("orders.parquet")

        summary = manager.get_dataanalysis_summary()
        assert summary.count("sales.csv") == 1
        assert "orders.parquet" in summary

    def test_track_analysis_records_result(self):
        manager = EnhancedDataAnalysisConversationManager()

        manager.track_analysis("correlation", "price and demand correlate at -0.7")

        obs = manager.get_observability_data()
        analyses = obs["dataanalysis_context"]["analyses_performed"]
        assert analyses == [
            {"analysis": "correlation", "result": "price and demand correlate at -0.7"}
        ]

    def test_summary_sections(self):
        manager = EnhancedDataAnalysisConversationManager()
        manager.track_dataset("sales.csv")
        manager.track_insight("Weekend sales are 2x weekday sales")

        summary = manager.get_dataanalysis_summary()

        assert "## Datasets Loaded" in summary
        assert "## Key Insights" in summary
        assert "- Weekend sales are 2x weekday sales" in summary

    def test_empty_summary_is_empty(self):
        manager = EnhancedDataAnalysisConversationManager()

        assert manager.get_dataanalysis_summary() == ""

    def test_observability_data_tags_vertical(self):
        manager = EnhancedDataAnalysisConversationManager()

        obs = manager.get_observability_data()

        assert obs["vertical"] == "dataanalysis"
        assert "dataanalysis_context" in obs

    def test_stats_available(self):
        manager = EnhancedDataAnalysisConversationManager()
        manager.add_message("user", "hello", TurnType.USER)

        stats = manager.get_stats()

        assert stats is not None
