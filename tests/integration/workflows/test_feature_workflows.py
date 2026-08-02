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

"""Integration tests for feature workflows YAML.

Tests the YAML workflow definition for creating Python features.
"""

import pytest
import tempfile
from pathlib import Path

from victor.core.utils.text_normalizer import (
    normalize_for_filename,
    normalize_for_git_branch,
    normalize_for_test_filename,
)
from victor.tools.scaffold_tool import scaffold
from victor.workflows.yaml_loader import load_workflow_from_yaml


class TestFeatureWorkflowYAML:
    """Integration tests for feature_workflows.yaml."""

    def test_yaml_file_exists(self):
        """Test that feature_workflows.yaml file exists."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        assert yaml_path.exists()
        assert yaml_path.is_file()

    def test_yaml_loads_successfully(self):
        """Test that YAML file can be loaded without errors."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")

        try:
            yaml_content = yaml_path.read_text()
            workflow_def = load_workflow_from_yaml(yaml_content)
            assert workflow_def is not None
            # load_workflow_from_yaml returns dict of WorkflowDefinition objects
            assert isinstance(workflow_def, dict)
            assert len(workflow_def) > 0
        except Exception as e:
            pytest.fail(f"Failed to load YAML: {e}")

    def test_yaml_has_python_feature_workflow(self):
        """Test that YAML contains python_feature workflow."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)

        assert "python_feature" in workflow_def
        workflow = workflow_def["python_feature"]

        # Check workflow structure - it's a WorkflowDefinition object
        assert workflow.description
        assert workflow.metadata
        assert workflow.nodes

    def test_python_feature_workflow_structure(self):
        """Test that python_feature workflow has correct structure."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)
        workflow = workflow_def["python_feature"]

        # Check workflow has nodes
        assert workflow.nodes
        assert len(workflow.nodes) > 0

        # Check first node structure - nodes is a dict of {node_id: WorkflowNode}
        first_node_id = list(workflow.nodes.keys())[0]
        first_node = workflow.nodes[first_node_id]
        assert first_node.id
        # WorkflowNode objects have different attributes than YAML dicts
        assert hasattr(first_node, "role") or hasattr(first_node, "type")

    def test_python_feature_with_git_workflow_exists(self):
        """Test that YAML contains python_feature_with_git workflow."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)

        assert "python_feature_with_git" in workflow_def
        workflow = workflow_def["python_feature_with_git"]

        # This workflow should have git branch creation
        node_ids = list(workflow.nodes.keys())
        assert "create_git_branch" in node_ids


class TestTextNormalizationInWorkflow:
    """Tests for text normalization functions used in workflow."""

    def test_normalize_for_git_branch(self):
        """Test git branch name normalization."""
        branch = normalize_for_git_branch("User Authentication", prefix="feature/")
        assert branch == "feature/user-authentication"

    def test_normalize_for_filename(self):
        """Test filename normalization."""
        filename = normalize_for_filename("User Authentication", extension=".py")
        assert filename == "user_authentication.py"

    def test_normalize_for_test_filename(self):
        """Test test filename normalization."""
        test_file = normalize_for_test_filename("User Authentication")
        assert test_file == "test_user_authentication.py"


class TestScaffoldFromTemplate:
    """Tests for scaffold from-template operation used in workflow."""

    @pytest.mark.asyncio
    async def test_scaffold_python_feature_template(self):
        """Test creating files from python_feature template."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                variables = {
                    "feature_name": "Test Feature",
                    "feature_filename": "test_feature.py",
                    "test_filename": "test_test_feature.py",
                    "feature_module": "test_feature",
                }

                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables=variables,
                )

                assert result["success"] is True
                assert result["count"] == 2

                # Verify files were created
                feature_file = Path("features/test_feature.py")
                test_file = Path("tests/test_test_feature.py")

                assert feature_file.exists()
                assert test_file.exists()

                # Verify content
                feature_content = feature_file.read_text()
                assert "Test Feature" in feature_content

            finally:
                os.chdir(old_cwd)


class TestWorkflowIntegration:
    """End-to-end integration tests for workflow components."""

    @pytest.mark.asyncio
    async def test_complete_feature_creation_workflow(self):
        """Test complete workflow: normalize → create files."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Step 1: Normalize names
                feature_name = "Data Processor"
                branch_name = normalize_for_git_branch(feature_name, prefix="feature/")
                source_filename = normalize_for_filename(feature_name, extension=".py")
                test_filename = normalize_for_test_filename(feature_name)
                feature_module = normalize_for_filename(feature_name)

                assert branch_name == "feature/data-processor"
                assert source_filename == "data_processor.py"
                assert test_filename == "test_data_processor.py"
                assert feature_module == "data_processor"

                # Step 2: Create files using scaffold
                variables = {
                    "feature_name": feature_name,
                    "feature_filename": source_filename,
                    "test_filename": test_filename,
                    "feature_module": feature_module,
                }

                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables=variables,
                )

                assert result["success"] is True

                # Step 3: Verify files created
                feature_file = Path(f"features/{source_filename}")
                test_file = Path(f"tests/{test_filename}")

                assert feature_file.exists()
                assert test_file.exists()

                # Step 4: Verify content is correctly interpolated
                feature_content = feature_file.read_text()
                assert "Data Processor" in feature_content
                assert "data_processor" in feature_content

                test_content = test_file.read_text()
                assert "test_data_processor" in test_content

            finally:
                os.chdir(old_cwd)

    def test_workflow_yaml_valid_structure(self):
        """Test that workflow YAML follows expected structure."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)

        # Check top-level structure - returns dict of WorkflowDefinition objects
        assert isinstance(workflow_def, dict)
        assert len(workflow_def) > 0

        # Check each workflow has required fields
        for workflow_name, workflow in workflow_def.items():
            # workflow is a WorkflowDefinition object
            assert workflow.description, f"{workflow_name} missing description"
            assert workflow.nodes, f"{workflow_name} missing nodes"
            assert isinstance(workflow.nodes, dict), f"{workflow_name} nodes must be a dict"

            # Check each node has required fields
            for node_id, node in workflow.nodes.items():
                assert node.id, f"{workflow_name} node missing id"
                # WorkflowNode objects don't have a 'type' attribute directly
                # they're instances of specific node types (AgentNode, ComputeNode, etc.)


class TestWorkflowNodeDefinitions:
    """Tests for individual workflow node definitions."""

    def test_create_feature_files_node(self):
        """Test create_feature_files node definition."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)
        workflow = workflow_def["python_feature"]

        # Get first node from the nodes dict
        first_node_id = list(workflow.nodes.keys())[0]
        create_node = workflow.nodes[first_node_id]
        assert create_node.id == "create_feature_files"
        # AgentNode has role attribute
        assert hasattr(create_node, "role")
        assert hasattr(create_node, "goal")
        # allowed_tools is a list on AgentNode
        if hasattr(create_node, "allowed_tools") and create_node.allowed_tools:
            assert "scaffold" in create_node.allowed_tools

    def test_workflow_with_git_node_order(self):
        """Test that python_feature_with_git has correct node order."""
        yaml_path = Path("victor/workflows/feature_workflows.yaml")
        yaml_content = yaml_path.read_text()
        workflow_def = load_workflow_from_yaml(yaml_content)
        workflow = workflow_def["python_feature_with_git"]

        # Check node order
        node_ids = list(workflow.nodes.keys())
        assert "create_git_branch" in node_ids
        assert "create_feature_files" in node_ids

        # Check that create_git_branch comes before create_feature_files
        assert node_ids.index("create_git_branch") < node_ids.index("create_feature_files")
