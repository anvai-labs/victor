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

"""Example JSON workflow compiler plugin.

This demonstrates how to create a custom workflow compiler plugin
for Victor using JSON format instead of YAML.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin


class JsonCompilerPlugin(WorkflowCompilerPlugin):
    """JSON workflow compiler plugin.

    Loads and compiles workflows from JSON files instead of YAML.
    This is an example of how to create custom compiler plugins.

    Example:
        from victor.workflows.compiler_registry import WorkflowCompilerRegistry
        from examples.plugins.json_compiler_plugin import JsonCompilerPlugin

        registry = WorkflowCompilerRegistry()
        registry.register('json', JsonCompilerPlugin)

        compiler = registry.create('json://')
        compiled = compiler.compile('workflow.json')
        result = await compiled.invoke({'input': 'data'})
    """

    supported_schemes = ["json", "json+file"]

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile JSON workflow.

        Args:
            source: JSON file path or JSON string content
            workflow_name: Name of workflow to compile
            validate: Whether to validate before compilation

        Returns:
            CompiledGraphProtocol instance

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        # Load JSON
        data = self._load_json(source)

        # Validate if requested
        if validate:
            self._validate_json(data)

        # Convert JSON to Victor's workflow format
        workflow_def = self._convert_to_workflow_def(data)

        # Compile using Victor's compiler
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        compiler = YAMLToStateGraphCompiler(
            orchestrator=None,  # Will be set by execution context
            orchestrators=None,  # Will be set by execution context
        )

        return compiler.compile(workflow_def, workflow_name=workflow_name)

    @property
    def supported_schemes(self) -> List[str]:
        """Return supported URI schemes."""
        return self.supported_schemes

    def _load_json(self, source: str) -> Dict:
        """Load JSON from file or string.

        Args:
            source: File path or JSON string

        Returns:
            Parsed JSON dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        # Check if source is a file path
        if Path(source).exists():
            with open(source, "r") as f:
                return json.load(f)
        else:
            # Parse as JSON string
            return json.loads(source)

    def _validate_json(self, data: Dict) -> None:
        """Validate JSON structure.

        Args:
            data: Parsed JSON data

        Raises:
            ValueError: If JSON structure is invalid
        """
        required_keys = ["workflows"]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Validate workflows section
        if not isinstance(data["workflows"], dict):
            raise ValueError("'workflows' must be a dictionary")

        # Validate each workflow
        for workflow_name, workflow_def in data["workflows"].items():
            if not isinstance(workflow_def, dict):
                raise ValueError(f"Workflow '{workflow_name}' must be a dictionary")

            if "nodes" not in workflow_def:
                raise ValueError(f"Workflow '{workflow_name}' missing 'nodes' key")

            if not isinstance(workflow_def["nodes"], list):
                raise ValueError(f"Workflow '{workflow_name}' 'nodes' must be a list")

    def _convert_to_workflow_def(self, data: Dict) -> Dict:
        """Convert JSON to Victor workflow definition format.

        Args:
            data: Parsed JSON data

        Returns:
            Workflow definition in Victor's format
        """
        # For this example, we assume JSON structure matches Victor's format
        # In a real plugin, you might need to transform the structure

        # Ensure workflow has metadata
        for workflow_name, workflow_def in data["workflows"].items():
            if "metadata" not in workflow_def:
                workflow_def["metadata"] = {}

            if "version" not in workflow_def["metadata"]:
                workflow_def["metadata"]["version"] = "1.0.0"

        return data

    def validate_source(self, source: str) -> bool:
        """Validate JSON source.

        Args:
            source: JSON string to validate

        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json.loads(source)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for JSON source.

        Args:
            source: JSON source

        Returns:
            Cache key string
        """
        import hashlib

        # Generate hash of JSON content for cache key
        return hashlib.md5(source.encode()).hexdigest()


__all__ = [
    "JsonCompilerPlugin",
]
