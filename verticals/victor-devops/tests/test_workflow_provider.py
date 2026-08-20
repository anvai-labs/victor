# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the DevOps YAML workflow provider surface."""

import re

import pytest

from victor.workflows.definition import WorkflowDefinition

from victor_devops.workflows import DevOpsWorkflowProvider


@pytest.fixture
def provider() -> DevOpsWorkflowProvider:
    return DevOpsWorkflowProvider()


class TestWorkflowDiscovery:
    def test_all_yaml_workflows_discovered(self, provider):
        names = provider.get_workflow_names()

        for expected in ("deploy", "cicd", "container_setup", "container_quick"):
            assert expected in names, f"missing workflow {expected}"

    def test_get_workflow_returns_definition(self, provider):
        workflow = provider.get_workflow("deploy")

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

    def test_container_prompts_trigger_container_setup(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "please containerize my app")
        ]

        assert "container_setup" in matched

    def test_cicd_prompts_trigger_pipeline(self, provider):
        matched = [
            target
            for pattern, target in provider.get_auto_workflows()
            if re.search(pattern, "set up ci/cd for this repo")
        ]

        assert "cicd_pipeline" in matched


class TestEscapeHatchWiring:
    def test_escape_hatches_module_path(self, provider):
        assert provider._get_escape_hatches_module() == "victor_devops.escape_hatches"

    def test_escape_hatches_load(self, provider):
        conditions, transforms = provider._load_escape_hatches()

        assert "deployment_ready" in conditions
        assert "merge_deployment_results" in transforms

    def test_capability_provider_module_path(self, provider):
        assert provider._get_capability_provider_module() == "victor_devops.capabilities"
