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

"""DevOps vertical workflows.

This package provides workflow definitions for common DevOps tasks:
- Infrastructure deployment
- Container management
- CI/CD pipeline setup
- Monitoring configuration

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = DevOpsWorkflowProvider()

    # Compile and execute (recommended - uses UnifiedWorkflowCompiler with caching)
    result = await provider.run_compiled_workflow("deploy", {"env": "prod"})

    # Stream execution with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("deploy", context):
        print(f"Completed: {node_id}")

Available workflows (all YAML-defined):
- deploy: Safe deployment with validation and rollback
- cicd: CI/CD pipeline with security scanning
- container_setup: Container setup with Dockerfile optimization
- container_quick: Quick container build
"""

from typing import Optional

from victor_contracts.workflow_runtime import BaseYAMLWorkflowProvider


class DevOpsWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides DevOps-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading with two-level caching
    - UnifiedWorkflowCompiler integration for consistent execution
    - Checkpointing support for resumable workflows
    - Auto-workflow triggers via class attributes

    Example:
        provider = DevOpsWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Execute with caching (recommended)
        result = await provider.run_compiled_workflow("deploy", {"env": "prod"})

        # Stream with real-time progress
        async for node_id, state in provider.stream_compiled_workflow("deploy", {}):
            print(f"Completed: {node_id}")
    """

    # Auto-workflow triggers for DevOps tasks
    AUTO_WORKFLOW_PATTERNS = [
        (r"deploy\s+infrastructure", "deploy_infrastructure"),
        (r"terraform\s+apply", "deploy_infrastructure"),
        (r"container(ize)?", "container_setup"),
        (r"docker(file)?", "container_setup"),
        (r"ci/?cd", "cicd_pipeline"),
        (r"pipeline", "cicd_pipeline"),
        (r"github\s+actions", "cicd_pipeline"),
    ]

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for DevOps escape hatches.

        Returns:
            Module path to victor_devops.escape_hatches
        """
        return "victor_devops.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        """Return the module path for the DevOps capability provider.

        Returns:
            Module path string for DevOpsCapabilityProvider
        """
        return "victor_devops.capabilities"


__all__ = [
    # YAML-first workflow provider
    "DevOpsWorkflowProvider",
]
