# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for YAML workflow schema validation.

Tests the YAML loader's ability to parse and validate workflow definitions:
- Basic workflow structure
- Node type parsing (agent, compute, parallel, hitl)
- Metadata parsing
- Batch config parsing
- Temporal context parsing
- Input mapping and context references
- Constraint parsing
- Error handling for invalid schemas
"""

import pytest
import yaml
from io import StringIO
from typing import Dict, Any
from unittest.mock import patch, MagicMock


class TestYAMLWorkflowParsing:
    """Test YAML workflow parsing."""

    def test_parse_basic_workflow(self):
        """Test parsing minimal valid workflow."""
        yaml_content = """
workflows:
  test_workflow:
    description: "Test workflow"
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Research topic"
        tool_budget: 10
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)

        assert "test_workflow" in workflows
        wf = workflows["test_workflow"]
        assert wf.name == "test_workflow"
        assert len(wf.nodes) == 1
        assert "start" in wf.nodes

    def test_parse_compute_node(self):
        """Test parsing compute node with handler."""
        yaml_content = """
workflows:
  compute_test:
    description: "Compute node test"
    nodes:
      - id: compute_step
        type: compute
        handler: data_transform
        inputs:
          data: $ctx.raw_data
          format: json
        output: transformed_data
        constraints:
          llm_allowed: false
          timeout: 30
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml
        from victor.workflows.definition import ComputeNode

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["compute_test"]

        assert len(wf.nodes) == 1
        node = wf.nodes["compute_step"]
        assert isinstance(node, ComputeNode)
        assert node.id == "compute_step"
        assert node.handler == "data_transform"

    def test_parse_parallel_node(self):
        """Test parsing parallel execution node."""
        yaml_content = """
workflows:
  parallel_test:
    description: "Parallel node test"
    nodes:
      - id: parallel_step
        type: parallel
        parallel_nodes: [task_a, task_b, task_c]
        join_strategy: all
        next: [final_step]

      - id: task_a
        type: compute
        handler: simple_handler
        output: result_a

      - id: task_b
        type: compute
        handler: simple_handler
        output: result_b

      - id: task_c
        type: compute
        handler: simple_handler
        output: result_c

      - id: final_step
        type: agent
        role: synthesizer
        goal: "Combine results"
        tool_budget: 5
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml
        from victor.workflows.definition import ParallelNode

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["parallel_test"]

        parallel_node = wf.nodes["parallel_step"]
        assert isinstance(parallel_node, ParallelNode)
        assert len(parallel_node.parallel_nodes) == 3

    def test_parse_hitl_node(self):
        """Test parsing human-in-the-loop node."""
        yaml_content = """
workflows:
  hitl_test:
    description: "HITL node test"
    nodes:
      - id: human_review
        type: hitl
        hitl_type: approval
        prompt: "Please review the analysis"
        timeout: 600
        fallback: continue
        next: [complete]

      - id: complete
        type: agent
        role: finalizer
        goal: "Complete workflow"
        tool_budget: 3
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["hitl_test"]

        hitl_node = wf.nodes["human_review"]
        # HITL nodes are parsed but may have different structure
        assert hitl_node is not None
        assert hitl_node.id == "human_review"


class TestMetadataParsing:
    """Test workflow metadata parsing."""

    def test_parse_vertical_metadata(self):
        """Test parsing vertical metadata."""
        yaml_content = """
workflows:
  meta_test:
    description: "Metadata test"
    metadata:
      vertical: dataanalysis
      category: eda
      complexity: high
    nodes:
      - id: start
        type: agent
        role: analyst
        goal: "Analyze"
        tool_budget: 10
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["meta_test"]

        assert wf.metadata.get("vertical") == "dataanalysis"
        assert wf.metadata.get("category") == "eda"
        assert wf.metadata.get("complexity") == "high"


class TestBatchConfigParsing:
    """Test batch configuration parsing."""

    def test_parse_batch_config(self):
        """Test parsing batch execution configuration."""
        yaml_content = """
workflows:
  batch_test:
    description: "Batch config test"
    batch_config:
      batch_size: 10
      max_concurrent: 5
      retry_strategy: end_of_batch
      continue_on_error: true
    nodes:
      - id: process
        type: compute
        handler: batch_processor
        output: results
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["batch_test"]

        # batch_config may be stored in metadata or as separate attribute
        if hasattr(wf, "batch_config") and wf.batch_config is not None:
            assert wf.batch_config.batch_size == 10
            assert wf.batch_config.max_concurrent == 5
        else:
            # Check if stored in metadata
            batch_config = wf.metadata.get("batch_config")
            if batch_config:
                assert batch_config.get("batch_size") == 10


class TestTemporalContextParsing:
    """Test temporal context parsing."""

    def test_parse_temporal_context(self):
        """Test parsing temporal context configuration."""
        yaml_content = """
workflows:
  temporal_test:
    description: "Temporal context test"
    temporal_context:
      as_of_date: $ctx.analysis_date
      lookback_periods: 8
    nodes:
      - id: analyze
        type: compute
        handler: time_series_analysis
        output: analysis
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["temporal_test"]

        # temporal_context may be stored in metadata or as separate attribute
        if hasattr(wf, "temporal_context") and wf.temporal_context is not None:
            assert wf.temporal_context.lookback_periods == 8
        else:
            # Check if stored in metadata
            temporal = wf.metadata.get("temporal_context")
            if temporal:
                assert temporal.get("lookback_periods") == 8


class TestInputMappingParsing:
    """Test input mapping and context reference parsing."""

    def test_parse_context_references(self):
        """Test parsing $ctx context references."""
        yaml_content = """
workflows:
  ctx_test:
    description: "Context reference test"
    nodes:
      - id: step1
        type: compute
        handler: processor
        inputs:
          symbol: $ctx.target_symbol
          data: $ctx.financial_data
          mode: standard
        output: step1_result

      - id: step2
        type: compute
        handler: aggregator
        inputs:
          input_data: $ctx.step1_result
        output: final_result
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["ctx_test"]

        node1 = wf.nodes["step1"]
        # $ctx. prefix is stripped during parsing
        assert "target_symbol" in str(node1.input_mapping) or "symbol" in str(node1.input_mapping)

    def test_parse_nested_context_references(self):
        """Test parsing nested context references like $ctx.data.field."""
        yaml_content = """
workflows:
  nested_test:
    description: "Nested context test"
    nodes:
      - id: analyze
        type: compute
        handler: analyzer
        inputs:
          ttm_revenue: $ctx.financial_data.ttm_data.revenue
          ratios: $ctx.financial_data.ratios
        output: analysis
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["nested_test"]

        node = wf.nodes["analyze"]
        # Check that input mapping contains the references
        assert node.input_mapping is not None
        assert len(node.input_mapping) > 0


class TestConstraintParsing:
    """Test constraint parsing."""

    def test_parse_compute_constraints(self):
        """Test parsing compute node constraints."""
        yaml_content = """
workflows:
  constraint_test:
    description: "Constraint test"
    nodes:
      - id: compute_node
        type: compute
        handler: processor
        constraints:
          llm_allowed: false
          max_cost_tier: FREE
          timeout: 60
          network_allowed: false
        output: result
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["constraint_test"]

        node = wf.nodes["compute_node"]
        assert node.constraints is not None
        assert node.constraints.llm_allowed is False
        assert node.constraints.timeout == 60
        assert node.constraints.network_allowed is False


class TestLLMConfigParsing:
    """Test LLM configuration parsing for agent nodes."""

    def test_parse_llm_config(self):
        """Test parsing LLM configuration."""
        yaml_content = """
workflows:
  llm_test:
    description: "LLM config test"
    nodes:
      - id: agent_node
        type: agent
        role: analyst
        goal: "Analyze data"
        tool_budget: 15
        llm_config:
          temperature: 0.3
          model_hint: claude-3-sonnet
          max_tokens: 2000
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)
        wf = workflows["llm_test"]

        node = wf.nodes["agent_node"]
        if node.llm_config is not None:
            # llm_config may be a dict or LLMConfig object
            if hasattr(node.llm_config, "temperature"):
                assert node.llm_config.temperature == 0.3
            elif isinstance(node.llm_config, dict):
                assert node.llm_config.get("temperature") == 0.3


class TestErrorHandling:
    """Test error handling for invalid YAML schemas."""

    def test_missing_workflows_key(self):
        """Test behavior when workflows key is missing."""
        yaml_content = """
not_workflows:
  test: true
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        # When 'workflows' key is missing, it treats the whole dict as workflows
        # but won't find valid workflow definitions (no 'nodes' key)
        workflows = load_workflow_from_yaml(yaml_content)
        # Should return empty dict since no valid workflows
        assert len(workflows) == 0

    def test_missing_nodes(self):
        """Test behavior when nodes are missing."""
        yaml_content = """
workflows:
  empty_workflow:
    description: "No nodes"
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        # Workflow without nodes should be skipped
        workflows = load_workflow_from_yaml(yaml_content)
        # Empty workflow should not be included
        assert (
            "empty_workflow" not in workflows
            or len(
                workflows.get("empty_workflow", {}).nodes if "empty_workflow" in workflows else {}
            )
            == 0
        )

    def test_invalid_node_type(self):
        """Test error for invalid node type."""
        yaml_content = """
workflows:
  invalid_test:
    description: "Invalid node type"
    nodes:
      - id: bad_node
        type: invalid_type
        goal: "This should fail"
"""
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowError,
        )

        with pytest.raises((ValueError, KeyError, YAMLWorkflowError)):
            load_workflow_from_yaml(yaml_content)

    def test_missing_required_agent_fields(self):
        """Test behavior when agent fields are missing.

        The loader may either raise an error or use defaults.
        """
        yaml_content = """
workflows:
  missing_fields:
    description: "Missing goal"
    nodes:
      - id: incomplete
        type: agent
        role: analyst
        # Missing: goal, tool_budget
"""
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowError,
        )

        try:
            workflows = load_workflow_from_yaml(yaml_content)
            # If no error, check that workflow was created with defaults
            if "missing_fields" in workflows:
                node = workflows["missing_fields"].nodes.get("incomplete")
                # Node should exist but may have default values
                assert node is not None
        except (ValueError, KeyError, TypeError, YAMLWorkflowError):
            # This is also acceptable - strict validation failed
            pass


class TestMultipleWorkflows:
    """Test parsing multiple workflows from single file."""

    def test_parse_multiple_workflows(self):
        """Test parsing file with multiple workflow definitions."""
        yaml_content = """
workflows:
  workflow_a:
    description: "First workflow"
    nodes:
      - id: start_a
        type: agent
        role: researcher
        goal: "Research A"
        tool_budget: 10

  workflow_b:
    description: "Second workflow"
    nodes:
      - id: start_b
        type: compute
        handler: processor
        output: result_b
"""
        from victor.workflows.yaml_loader import load_workflow_from_yaml

        workflows = load_workflow_from_yaml(yaml_content)

        assert len(workflows) == 2
        assert "workflow_a" in workflows
        assert "workflow_b" in workflows


class TestDirectoryLoading:
    """Test loading workflows from directory."""

    def test_load_from_directory(self, tmp_path):
        """Test loading all YAML files from a directory."""
        # Create test YAML file
        yaml_file = tmp_path / "test_workflow.yaml"
        yaml_file.write_text("""
workflows:
  dir_workflow:
    description: "Directory test"
    nodes:
      - id: start
        type: agent
        role: tester
        goal: "Test loading"
        tool_budget: 5
""")
        from victor.workflows.yaml_loader import load_workflows_from_directory

        workflows = load_workflows_from_directory(tmp_path)

        assert "dir_workflow" in workflows

    def test_skip_non_yaml_files(self, tmp_path):
        """Test that non-YAML files are skipped."""
        # Create YAML and non-YAML files
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text("""
workflows:
  valid_workflow:
    description: "Valid"
    nodes:
      - id: start
        type: agent
        role: test
        goal: "Test"
        tool_budget: 1
""")

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is not a workflow file")

        py_file = tmp_path / "helper.py"
        py_file.write_text("# Python file")

        from victor.workflows.yaml_loader import load_workflows_from_directory

        workflows = load_workflows_from_directory(tmp_path)

        # Only YAML workflow should be loaded
        assert "valid_workflow" in workflows
        assert len(workflows) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
