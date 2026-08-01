# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for victor.extension.protocols implementations in victor-dataanalysis."""

from victor_dataanalysis.protocols import (
    DataAnalysisPromptProvider,
    DataAnalysisSafetyProvider,
    DataAnalysisToolProvider,
    DataAnalysisToolSelectionStrategy,
    DataAnalysisWorkflowProvider,
)


class TestToolProvider:
    def test_tool_list_contract(self):
        tools = DataAnalysisToolProvider().get_tools()

        assert len(tools) == len(set(tools)), "duplicate tools"
        for expected in ("read", "write", "csv_read", "statistics_describe", "ml_train"):
            assert expected in tools

    def test_covers_all_analysis_domains(self):
        tools = set(DataAnalysisToolProvider().get_tools())

        assert tools & {"csv_read", "excel_read", "json_read"}, "no data loading tools"
        assert tools & {"data_clean", "data_transform"}, "no processing tools"
        assert tools & {"statistics_correlation", "statistics_regression"}, "no stats tools"
        assert tools & {"plot_line", "plot_scatter", "plot_heatmap"}, "no plotting tools"
        assert tools & {"ml_train", "ml_predict", "ml_evaluate"}, "no ML tools"


class TestToolSelectionStrategy:
    def test_stage_specific_tools(self):
        strategy = DataAnalysisToolSelectionStrategy()

        assert "csv_read" in strategy.get_tools_for_stage("load", "eda")
        assert "statistics_regression" in strategy.get_tools_for_stage("analyze", "statistics")
        assert "plot_heatmap" in strategy.get_tools_for_stage("visualize", "eda")

    def test_unknown_stage_falls_back_to_exploration_basics(self):
        strategy = DataAnalysisToolSelectionStrategy()

        assert strategy.get_tools_for_stage("unknown_stage", "any") == [
            "read",
            "csv_read",
            "statistics_describe",
        ]


class TestSafetyProvider:
    def test_dangerous_sql_patterns_registered(self):
        provider = DataAnalysisSafetyProvider()

        patterns = [p["pattern"] for p in provider.get_bash_patterns()]
        assert "DELETE FROM" in patterns
        assert "DROP TABLE" in patterns
        assert "TRUNCATE" in patterns

    def test_tool_restrictions_guard_destructive_sql(self):
        restrictions = DataAnalysisSafetyProvider().get_tool_restrictions()

        assert set(restrictions["database_query"]) == {"DELETE", "DROP", "TRUNCATE"}
        assert "--force" in restrictions["data_export_csv"]

    def test_no_file_patterns(self):
        assert DataAnalysisSafetyProvider().get_file_patterns() == []


class TestPromptProvider:
    def test_system_prompt_sections(self):
        sections = DataAnalysisPromptProvider().get_system_prompt_sections()

        assert {"role", "expertise", "methodology", "best_practices", "communication"} <= set(
            sections
        )
        assert "Data Analysis" in sections["role"]
        assert "pandas" in sections["expertise"]

    def test_task_type_hints_have_budgets(self):
        hints = DataAnalysisPromptProvider().get_task_type_hints()

        assert {"eda", "statistics", "ml", "visualization"} <= set(hints)
        for task_type, hint in hints.items():
            assert hint["hint"], f"{task_type} hint empty"
            assert hint["tool_budget"] > 0

    def test_no_extra_contributors(self):
        assert DataAnalysisPromptProvider().get_prompt_contributors() == []


class TestWorkflowProviderProtocol:
    def test_workflow_lookup_roundtrip(self):
        provider = DataAnalysisWorkflowProvider()

        names = provider.list_workflows()
        assert "exploratory_analysis" in names
        workflow = provider.get_workflow("exploratory_analysis")
        assert workflow["stages"] == ["load", "clean", "explore", "visualize"]

    def test_ml_pipeline_defined(self):
        workflow = DataAnalysisWorkflowProvider().get_workflow("ml_pipeline")

        assert workflow["name"] == "Machine Learning Pipeline"

    def test_unknown_workflow_returns_none(self):
        assert DataAnalysisWorkflowProvider().get_workflow("nope") is None
