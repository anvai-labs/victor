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

"""Mode-based workflows for agent operational modes.

This module provides declarative YAML-based workflows for the three
agent operational modes: BUILD, PLAN, and EXPLORE.

Design Philosophy:
- Declarative workflow definitions in YAML
- Mode-specific tool budgets and allowed tools
- Sandbox enforcement for non-build modes
- Multi-agent team support for complex tasks

Usage:
    from victor.workflows.mode_workflows import (
        get_mode_workflow,
        get_mode_workflow_provider,
        ModeWorkflowProvider,
    )

    # Get a specific mode workflow
    workflow = get_mode_workflow("explore")

    # Get the provider for integration
    provider = get_mode_workflow_provider()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type

from victor.workflows.definition import WorkflowDefinition
from victor.workflows.yaml_loader import (
    YAMLWorkflowProvider,
    load_workflow_from_file,
    load_workflows_from_directory,
)
from victor.core.verticals.protocols import WorkflowProviderProtocol

logger = logging.getLogger(__name__)

# Path to the mode workflows YAML file
MODE_WORKFLOWS_PATH = Path(__file__).parent / "mode_workflows.yaml"

# Cache for loaded workflows
_mode_workflows_cache: Optional[Dict[str, WorkflowDefinition]] = None


class ModeWorkflowProvider(WorkflowProviderProtocol):
    """Provider for mode-based workflows.

    Implements WorkflowProviderProtocol to provide explore, plan, and build
    workflows from the YAML definition.

    Example:
        provider = ModeWorkflowProvider()
        workflows = provider.get_workflows()
        explore_workflow = workflows.get("explore")
    """

    def __init__(self, yaml_path: Optional[Path] = None):
        """Initialize the provider.

        Args:
            yaml_path: Optional custom path to workflows YAML.
                      Defaults to the built-in mode_workflows.yaml.
        """
        self._yaml_path = yaml_path or MODE_WORKFLOWS_PATH
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Load workflows from YAML file.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances.
        """
        if self._workflows is not None:
            return self._workflows

        try:
            from victor.workflows.yaml_loader import load_workflow_from_file

            # Load all workflows from the YAML file
            self._workflows = load_workflow_from_file(str(self._yaml_path))
            if isinstance(self._workflows, WorkflowDefinition):
                # Single workflow - wrap in dict
                self._workflows = {self._workflows.name: self._workflows}

            logger.debug(f"Loaded {len(self._workflows)} mode workflows")
            return self._workflows

        except Exception as e:
            logger.warning(f"Failed to load mode workflows: {e}")
            self._workflows = {}
            return self._workflows

    def get_workflows(self) -> Dict[str, Type]:
        """Get workflow classes for mode-based workflows.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances.
            (Note: Returns WorkflowDefinition instances, not classes, for YAML workflows)
        """
        return self._load_workflows()

    def get_auto_workflows(self) -> list:
        """Get automatically triggered workflows.

        Mode workflows are not auto-triggered; they are explicitly activated
        via CLI --mode flag or agent mode selection.

        Returns:
            Empty list (no auto-triggers for mode workflows)
        """
        return []

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name.

        Args:
            name: Workflow name (e.g., "explore", "plan", "build")

        Returns:
            WorkflowDefinition or None if not found
        """
        workflows = self._load_workflows()
        return workflows.get(name)

    def list_workflows(self) -> list:
        """List available workflow names.

        Returns:
            List of workflow names
        """
        return list(self._load_workflows().keys())


def get_mode_workflow(mode: str) -> Optional[WorkflowDefinition]:
    """Get workflow for a specific mode.

    Args:
        mode: Mode name ("explore", "plan", or "build")

    Returns:
        WorkflowDefinition for the mode, or None if not found

    Example:
        workflow = get_mode_workflow("explore")
        if workflow:
            executor = WorkflowExecutor()
            result = await executor.execute(workflow, context)
    """
    global _mode_workflows_cache

    if _mode_workflows_cache is None:
        try:
            _mode_workflows_cache = load_workflow_from_file(str(MODE_WORKFLOWS_PATH))
            if isinstance(_mode_workflows_cache, WorkflowDefinition):
                _mode_workflows_cache = {_mode_workflows_cache.name: _mode_workflows_cache}
        except Exception as e:
            logger.warning(f"Failed to load mode workflows: {e}")
            _mode_workflows_cache = {}

    return _mode_workflows_cache.get(mode.lower())


def get_mode_workflow_provider() -> ModeWorkflowProvider:
    """Get the mode workflow provider.

    Returns:
        ModeWorkflowProvider instance for integration with framework
    """
    return ModeWorkflowProvider()


def get_all_mode_workflows() -> Dict[str, WorkflowDefinition]:
    """Get all mode workflows.

    Returns:
        Dict mapping mode names to WorkflowDefinition instances
    """
    global _mode_workflows_cache

    if _mode_workflows_cache is None:
        try:
            _mode_workflows_cache = load_workflow_from_file(str(MODE_WORKFLOWS_PATH))
            if isinstance(_mode_workflows_cache, WorkflowDefinition):
                _mode_workflows_cache = {_mode_workflows_cache.name: _mode_workflows_cache}
        except Exception as e:
            logger.warning(f"Failed to load mode workflows: {e}")
            _mode_workflows_cache = {}

    return _mode_workflows_cache.copy()


__all__ = [
    "ModeWorkflowProvider",
    "get_mode_workflow",
    "get_mode_workflow_provider",
    "get_all_mode_workflows",
    "MODE_WORKFLOWS_PATH",
]
