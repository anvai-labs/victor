# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the data analysis YAML workflow provider surface."""

import re

import pytest

from victor.workflows.definition import WorkflowDefinition

from victor_dataanalysis.workflows import DataAnalysisWorkflowProvider


@pytest.fixture
def provider() -> DataAnalysisWorkflowProvider:
    return DataAnalysisWorkflowProvider()


class TestWorkflowDiscovery:
    def test_all_yaml_workflows_discovered(self, provider):
        names = provider.get_workflow_names()

        for expected in (
            "eda_pipeline",
            "eda_quick",
            "data_cleaning",
            "data_cleaning_quick",
            "statistical_analysis",
            "ml_pipeline",
            "ml_quick",
        ):
            assert expected in names, f"missing workflow {expected}"

    def test_get_workflow_returns_definition(self, provider):
        workflow = provider.get_workflow("eda_pipeline")

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.nodes) > 0

    def test_get_workflow_unknown_returns_none(self, provider):
        assert provider.get_workflow("nonexistent_workflow") is None

    def test_get_workflows_map_matches_names(self, provider):
        workflows = provider.get_workflows()

        assert set(provider.get_workflow_names()) == set(workflows)
        assert all(isinstance(wf, WorkflowDefinition) for wf in workflows.values())


class TestAutoWorkflowTriggers:
    def test_patterns_are_valid_regexes(self, provider):
        for pattern, target in provider.get_auto_workflows():
            re.compile(pattern)  # raises on invalid regex
            assert target, f"pattern {pattern!r} has empty target"

    def test_eda_prompts_trigger_eda_workflow(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "run exploratory data analysis on sales.csv")
        ]

        assert "eda_workflow" in matched

    def test_ml_prompts_trigger_ml_pipeline(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "train a model on this dataset")
        ]

        assert "ml_pipeline" in matched

    def test_cleaning_prompts_trigger_data_cleaning(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "clean the data and handle missing values")
        ]

        assert "data_cleaning" in matched


class TestTaskTypeMappings:
    def test_task_types_route_to_workflows(self, provider):
        mappings = provider.TASK_TYPE_MAPPINGS

        assert mappings["eda"] == "eda_workflow"
        assert mappings["cleaning"] == "data_cleaning"
        assert mappings["statistics"] == "statistical_analysis"
        assert mappings["ml"] == "ml_pipeline"


class TestEscapeHatchWiring:
    def test_escape_hatches_load(self, provider):
        conditions, transforms = provider._load_escape_hatches()

        assert "quality_threshold" in conditions
        assert "should_retry_cleaning" in conditions
        assert "merge_parallel_stats" in transforms
        assert "aggregate_model_results" in transforms
