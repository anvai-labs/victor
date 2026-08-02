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

"""Workflow loader mixin with common workflow utilities.

This module provides WorkflowLoaderMixin, a mixin class with utility
methods for working with YAML workflows that verticals can use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowLoaderMixin:
    """Mixin class providing workflow loading utilities.

    Provides utility methods for working with YAML workflow files
    that verticals can use alongside BaseWorkflowProvider.

    This class is designed to be used as a mixin:

        class MyVerticalWorkflowProvider(BaseWorkflowProvider, WorkflowLoaderMixin):
            ...
    """

    @staticmethod
    def validate_workflow_structure(
        workflow_def: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate the structure of a workflow definition.

        Args:
            workflow_def: Workflow definition to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for required top-level fields
        if "nodes" not in workflow_def:
            errors.append("Missing required field: 'nodes'")

        # Validate nodes if present
        if "nodes" in workflow_def:
            nodes = workflow_def["nodes"]
            if not isinstance(nodes, list):
                errors.append("'nodes' must be a list")
            else:
                for i, node in enumerate(nodes):
                    if not isinstance(node, dict):
                        errors.append(f"Node {i}: must be a dict")
                        continue

                    # Check required node fields
                    if "id" not in node:
                        errors.append(f"Node {i}: missing required field 'id'")
                    if "type" not in node:
                        errors.append(f"Node {i}: missing required field 'type'")

                    # Validate node type
                    if "type" in node:
                        valid_types = ["agent", "compute", "handler", "passthrough"]
                        if node["type"] not in valid_types:
                            errors.append(
                                f"Node {i}: invalid type '{node['type']}', "
                                f"must be one of {valid_types}"
                            )

        return (len(errors) == 0, errors)

    @staticmethod
    def find_workflow_files(
        directories: List[str],
        pattern: str = "*.yaml",
    ) -> List[Path]:
        """Find all workflow files in the given directories.

        Args:
            directories: List of directory paths to search
            pattern: File pattern to match (default: "*.yaml")

        Returns:
            List of Path objects for found workflow files
        """
        import glob

        found = []
        for directory in directories:
            expanded_dir = Path(directory).expanduser()
            if not expanded_dir.exists():
                continue

            search_pattern = str(expanded_dir / pattern)
            found.extend(Path(p) for p in glob.glob(search_pattern))

        return sorted(set(found))

    @staticmethod
    def load_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a YAML file into a dictionary.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML content or None if loading failed
        """
        import yaml

        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    @staticmethod
    def extract_workflows(
        yaml_content: Dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract workflow definitions from YAML content.

        Args:
            yaml_content: Parsed YAML content
            defaults: Optional default values to apply

        Returns:
            Dict mapping workflow names to definitions
        """
        workflows = {}

        if not yaml_content:
            return workflows

        # Determine format
        if isinstance(yaml_content, dict):
            if "workflows" in yaml_content:
                workflow_defs = yaml_content["workflows"]
            else:
                # Treat entire file as single workflow
                workflow_defs = yaml_content
        else:
            logger.warning("Unexpected YAML format: not a dict")
            return workflows

        # Extract workflows
        defaults = defaults or {}
        for name, definition in workflow_defs.items():
            if isinstance(definition, dict):
                workflows[name] = {**defaults, **definition}

        return workflows

    @staticmethod
    def create_simple_workflow(
        name: str,
        description: str = "",
        nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a simple workflow definition.

        Args:
            name: Workflow name
            description: Workflow description
            nodes: List of workflow nodes

        Returns:
            Workflow definition dictionary
        """
        return {
            "name": name,
            "description": description,
            "nodes": nodes or [],
        }

    @staticmethod
    def create_agent_node(
        node_id: str,
        agent_type: str,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create an agent node definition.

        Args:
            node_id: Node identifier
            agent_type: Type of agent to use
            prompt: Prompt for the agent
            **kwargs: Additional node properties

        Returns:
            Agent node definition
        """
        return {
            "id": node_id,
            "type": "agent",
            "agent_type": agent_type,
            "prompt": prompt,
            **kwargs,
        }

    @staticmethod
    def create_compute_node(
        node_id: str,
        tools: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a compute node definition.

        Args:
            node_id: Node identifier
            tools: List of tools to allow
            constraints: Optional constraints dict
            **kwargs: Additional node properties

        Returns:
            Compute node definition
        """
        node = {
            "id": node_id,
            "type": "compute",
            "tools": tools,
            **kwargs,
        }

        if constraints:
            node["constraints"] = constraints

        return node

    @staticmethod
    def merge_workflow_defaults(
        workflows: Dict[str, Dict[str, Any]],
        defaults: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge default values into workflow definitions.

        Args:
            workflows: Workflow definitions
            defaults: Default values to apply

        Returns:
            Updated workflow definitions
        """
        merged = {}
        for name, definition in workflows.items():
            merged[name] = {**defaults, **definition}
        return merged

    def get_workflow_summary(self, workflow_def: Dict[str, Any]) -> str:
        """Get a human-readable summary of a workflow.

        Args:
            workflow_def: Workflow definition

        Returns:
            Summary string
        """
        parts = []

        name = workflow_def.get("name", "unnamed")
        description = workflow_def.get("description", "")
        parts.append(f"Workflow: {name}")

        if description:
            parts.append(f"Description: {description}")

        nodes = workflow_def.get("nodes", [])
        parts.append(f"Nodes: {len(nodes)}")

        if nodes:
            parts.append("Node types:")
            node_types = {}
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1

            for node_type, count in sorted(node_types.items()):
                parts.append(f"  - {node_type}: {count}")

        return "\n".join(parts)


__all__ = [
    "WorkflowLoaderMixin",
]
