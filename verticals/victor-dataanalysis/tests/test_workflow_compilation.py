# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compile shipped data analysis YAML workflows through the unified compiler."""

import pytest

from victor_dataanalysis.workflows import DataAnalysisWorkflowProvider


@pytest.fixture(scope="module")
def provider() -> DataAnalysisWorkflowProvider:
    return DataAnalysisWorkflowProvider()


def _file_backed_workflow_names():
    """Workflow names that map 1:1 to a YAML file (compile_workflow contract).

    Workflows defined inside multi-workflow YAML files (e.g. eda_quick inside
    eda_pipeline.yaml) are covered via get_workflow() in
    test_workflow_provider.py.
    """
    provider = DataAnalysisWorkflowProvider()
    yaml_stems = {p.stem for p in provider._get_workflows_directory().glob("*.yaml")}
    return sorted(set(provider.get_workflow_names()) & yaml_stems)


class TestUnifiedCompilation:
    def test_at_least_one_file_backed_workflow(self):
        names = _file_backed_workflow_names()

        assert "eda_pipeline" in names
        assert "ml_pipeline" in names

    @pytest.mark.parametrize("workflow_name", _file_backed_workflow_names())
    def test_workflow_compiles(self, provider, workflow_name):
        compiled = provider.compile_workflow(workflow_name)

        assert compiled is not None
        assert compiled.workflow_name == workflow_name
        assert compiled.compiled_graph is not None
        assert compiled.source_path.exists()

    def test_compiler_is_unified_workflow_compiler(self, provider):
        compiler = provider.get_compiler()

        assert type(compiler).__name__ == "UnifiedWorkflowCompiler"

    def test_compilation_caches_by_source(self, provider):
        first = provider.compile_workflow("eda_pipeline")
        second = provider.compile_workflow("eda_pipeline")

        assert first.cache_key == second.cache_key

    def test_unknown_workflow_raises(self, provider):
        with pytest.raises(Exception):
            provider.compile_workflow("does_not_exist")


class TestWorkflowStructure:
    def test_data_cleaning_workflow_has_condition_nodes(self, provider):
        """The cleaning workflow must gate on escape-hatch conditions."""
        workflow = provider.get_workflow("data_cleaning")

        node_types = {type(node).__name__ for node in workflow.nodes.values()}
        assert any("Condition" in t for t in node_types), node_types

    def test_each_workflow_has_nodes(self, provider):
        for name in provider.get_workflow_names():
            workflow = provider.get_workflow(name)
            assert workflow.nodes, f"workflow {name} has no nodes"
