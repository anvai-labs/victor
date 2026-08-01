# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the research YAML workflow provider surface."""

import re

import pytest

from victor.workflows.definition import WorkflowDefinition

from victor_research.workflows import ResearchWorkflowProvider


@pytest.fixture
def provider() -> ResearchWorkflowProvider:
    return ResearchWorkflowProvider()


class TestWorkflowDiscovery:
    def test_all_yaml_workflows_discovered(self, provider):
        names = provider.get_workflow_names()

        for expected in (
            "deep_research",
            "quick_research",
            "fact_check",
            "literature_review",
            "competitive_analysis",
            "competitive_scan",
        ):
            assert expected in names, f"missing workflow {expected}"

    def test_get_workflow_returns_definition(self, provider):
        workflow = provider.get_workflow("fact_check")

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

    def test_auto_targets_are_shipped_workflows(self, provider):
        """Unlike devops, every research auto-trigger targets a real workflow."""
        names = set(provider.get_workflow_names())

        for pattern, target in provider.get_auto_workflows():
            assert target in names, f"{pattern!r} targets unknown workflow {target}"

    def test_fact_check_prompts_trigger_fact_check(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "please fact check this statement")
        ]

        assert "fact_check" in matched

    def test_deep_research_prompts_trigger_deep_research(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "do a deep research on quantum computing")
        ]

        assert "deep_research" in matched


class TestTaskTypeMappings:
    def test_task_types_map_to_shipped_workflows(self, provider):
        names = set(provider.get_workflow_names())

        for task_type, workflow in ResearchWorkflowProvider.TASK_TYPE_MAPPINGS.items():
            assert workflow in names, f"task {task_type} maps to unknown workflow {workflow}"

    def test_key_task_type_routes(self, provider):
        mappings = ResearchWorkflowProvider.TASK_TYPE_MAPPINGS

        assert mappings["research"] == "deep_research"
        assert mappings["verification"] == "fact_check"
        assert mappings["academic"] == "literature_review"
        assert mappings["market"] == "competitive_analysis"


class TestEscapeHatchWiring:
    def test_escape_hatches_module_path(self, provider):
        assert provider._get_escape_hatches_module() == "victor_research.escape_hatches"

    def test_escape_hatches_load(self, provider):
        conditions, transforms = provider._load_escape_hatches()

        assert "source_coverage_check" in conditions
        assert "fact_verdict" in conditions
        assert "merge_search_results" in transforms

    def test_capability_provider_module_path(self, provider):
        assert provider._get_capability_provider_module() == "victor_research.capabilities"
