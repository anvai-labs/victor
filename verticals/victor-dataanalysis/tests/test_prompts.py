# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the data analysis prompt contributor."""

from victor_contracts import TaskTypeHint
from victor_contracts.verticals.protocols import PromptContributorProtocol

from victor_dataanalysis.prompts import (
    DATA_ANALYSIS_TASK_TYPE_HINTS,
    DataAnalysisPromptContributor,
)


class TestTaskTypeHints:
    def test_classifier_aligned_hints_present(self):
        # Keys that must align with TaskTypeClassifier task types
        assert "data_analysis" in DATA_ANALYSIS_TASK_TYPE_HINTS
        assert "visualization" in DATA_ANALYSIS_TASK_TYPE_HINTS
        assert "general" in DATA_ANALYSIS_TASK_TYPE_HINTS

    def test_granular_method_hints_present(self):
        for method in (
            "data_profiling",
            "statistical_analysis",
            "correlation_analysis",
            "regression",
            "clustering",
            "time_series",
        ):
            assert method in DATA_ANALYSIS_TASK_TYPE_HINTS

    def test_hint_shapes(self):
        for task_type, hint in DATA_ANALYSIS_TASK_TYPE_HINTS.items():
            assert isinstance(hint, TaskTypeHint)
            assert hint.task_type == task_type
            assert hint.hint.strip(), f"{task_type} hint is empty"
            assert hint.tool_budget > 0
            assert hint.priority_tools, f"{task_type} has no priority tools"

    def test_precision_tasks_use_low_temperature(self):
        """Statistical tasks need deterministic output."""
        for task_type in ("data_profiling", "statistical_analysis", "correlation_analysis"):
            assert DATA_ANALYSIS_TASK_TYPE_HINTS[task_type].temperature_override <= 0.2


class TestPromptContributor:
    def test_implements_protocol(self):
        contributor = DataAnalysisPromptContributor()

        assert isinstance(contributor, PromptContributorProtocol)

    def test_get_task_type_hints_returns_copy(self):
        contributor = DataAnalysisPromptContributor()

        hints = contributor.get_task_type_hints()
        assert hints == DATA_ANALYSIS_TASK_TYPE_HINTS
        hints.pop("data_analysis")
        assert "data_analysis" in DATA_ANALYSIS_TASK_TYPE_HINTS, "mutation leaked into module dict"

    def test_system_prompt_section_references_core_libraries(self):
        section = DataAnalysisPromptContributor().get_system_prompt_section()

        assert "Python Libraries Reference" in section
        assert "import pandas as pd" in section
        assert "sklearn" in section

    def test_grounding_rules_forbid_fabrication(self):
        rules = DataAnalysisPromptContributor().get_grounding_rules()

        assert "GROUNDING" in rules
        assert "fabricate" in rules

    def test_priority_is_medium(self):
        assert DataAnalysisPromptContributor().get_priority() == 5

    def test_context_hints_for_known_task_type(self):
        contributor = DataAnalysisPromptContributor()

        hint = contributor.get_context_hints("regression")
        assert hint == DATA_ANALYSIS_TASK_TYPE_HINTS["regression"].hint

    def test_context_hints_for_unknown_task_type(self):
        contributor = DataAnalysisPromptContributor()

        assert contributor.get_context_hints("cooking") is None
        assert contributor.get_context_hints(None) is None
