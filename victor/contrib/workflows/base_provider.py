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

"""Base workflow provider for Victor verticals.

This module provides BaseWorkflowProvider, a reusable base class that
implements YAML workflow support using the framework's workflow system.
Verticals can inherit from this base class to get common workflow
functionality while adding vertical-specific workflows.

Design Pattern: Template Method
- Base class provides common workflow infrastructure
- Verticals override get_vertical_name() and get_workflow_directories()
- Verticals can optionally override workflow validation and registration
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseWorkflowProvider:
    """Base workflow provider for Victor verticals.

    Provides common workflow functionality:
    - YAML workflow discovery from multiple directories
    - Workflow registration and validation
    - Common workflow access patterns
    - Vertical-specific workflow overrides

    Verticals should:
    1. Inherit from BaseWorkflowProvider
    2. Implement get_vertical_name() to return vertical identifier
    3. Implement get_workflow_directories() to return workflow search paths
    4. Optionally override get_workflow_defaults() for default configuration
    5. Optionally override validate_workflow() for custom validation

    Example:
        class CodingWorkflowProvider(BaseWorkflowProvider):
            def get_vertical_name(self) -> str:
                return \"coding\"

            def get_workflow_directories(self) -> List[str]:
                return [
                    \"/usr/local/lib/victor-workflows/common\",
                    \"~/.victor/workflows/coding\",
                ]
    """

    def __init__(self, auto_load: bool = True):
        """Initialize the workflow provider.

        Args:
            auto_load: If True, automatically load workflows on initialization
        """
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._workflow_paths: Dict[str, Path] = {}

        if auto_load:
            self.load_workflows()

        logger.info(
            f"{self.__class__.__name__} initialized for '{self.get_vertical_name()}' "
            f"with {len(self._workflows)} workflows"
        )

    # ==========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ==========================================================================

    @abstractmethod
    def get_vertical_name(self) -> str:
        """Get the vertical name for workflow namespacing.

        Returns:
            Vertical name (e.g., "devops", "rag", "research")
        """
        ...

    @abstractmethod
    def get_workflow_directories(self) -> List[str]:
        """Get workflow search directories.

        Returns:
            List of directory paths to search for YAML workflows
        """
        ...

    # ==========================================================================
    # Template Methods - Can be overridden by subclasses
    # ==========================================================================

    def get_workflow_defaults(self) -> Dict[str, Any]:
        """Get default workflow configuration.

        Returns:
            Dict with default workflow configuration
        """
        return {}

    def validate_workflow(self, workflow_name: str, workflow_def: Dict[str, Any]) -> bool:
        """Validate a workflow definition.

        Args:
            workflow_name: Name of the workflow
            workflow_def: Workflow definition dictionary

        Returns:
            True if workflow is valid, False otherwise
        """
        # Basic validation: check for required fields
        required_fields = ["nodes"]
        return all(field in workflow_def for field in required_fields)

    def get_workflow_file_pattern(self) -> str:
        """Get the file pattern for workflow discovery.

        Returns:
            File pattern (e.g., "*.yaml", "*.yml", "workflow_*.yaml")
        """
        return "*.yaml"

    # ==========================================================================
    # Workflow Management - Common for all verticals
    # ==========================================================================

    def load_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Load workflows from all configured directories.

        Returns:
            Dict mapping workflow names to workflow definitions
        """
        import glob
        import os

        loaded = {}
        pattern = self.get_workflow_file_pattern()

        for directory in self.get_workflow_directories():
            # Expand user path
            expanded_dir = Path(directory).expanduser()
            if not expanded_dir.exists():
                logger.debug(f"Workflow directory does not exist: {directory}")
                continue

            # Find all workflow files
            search_pattern = str(expanded_dir / pattern)
            for workflow_file in glob.glob(search_pattern):
                try:
                    workflows = self._load_workflow_file(workflow_file)
                    loaded.update(workflows)
                except Exception as e:
                    logger.warning(f"Failed to load workflows from {workflow_file}: {e}")

        self._workflows = loaded
        logger.info(
            f"Loaded {len(loaded)} workflows for '{self.get_vertical_name()}' "
            f"from {len(self.get_workflow_directories())} directories"
        )
        return loaded

    def get_workflow(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by name.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Workflow definition or None if not found
        """
        return self._workflows.get(workflow_name)

    def list_workflows(self) -> List[str]:
        """List all available workflow names.

        Returns:
            List of workflow names
        """
        return sorted(self._workflows.keys())

    def has_workflow(self, workflow_name: str) -> bool:
        """Check if a workflow exists.

        Args:
            workflow_name: Name of the workflow

        Returns:
            True if workflow exists
        """
        return workflow_name in self._workflows

    def get_workflow_path(self, workflow_name: str) -> Optional[Path]:
        """Get the file path for a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Path to workflow file or None if not found
        """
        return self._workflow_paths.get(workflow_name)

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _load_workflow_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load workflows from a single YAML file.

        Args:
            file_path: Path to the workflow file

        Returns:
            Dict mapping workflow names to workflow definitions
        """
        import yaml

        workflows = {}
        path = Path(file_path)

        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)

            if not content:
                return workflows

            # Extract workflows from content
            # Support both dict format (workflows: {...}) and direct format
            if isinstance(content, dict):
                if "workflows" in content:
                    workflow_defs = content["workflows"]
                else:
                    # File itself is a single workflow
                    workflow_defs = {path.stem: content}
            else:
                logger.warning(f"Unexpected YAML format in {file_path}")
                return workflows

            # Validate and register workflows
            for name, definition in workflow_defs.items():
                if self.validate_workflow(name, definition):
                    # Apply defaults
                    full_def = {**self.get_workflow_defaults(), **definition}
                    workflows[name] = full_def
                    self._workflow_paths[name] = path
                else:
                    logger.warning(f"Workflow '{name}' failed validation in {file_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading workflow file {file_path}: {e}")

        return workflows

    def reload_workflows(self) -> int:
        """Reload all workflows from disk.

        Returns:
            Number of workflows loaded
        """
        self._workflows.clear()
        self._workflow_paths.clear()
        self.load_workflows()
        return len(self._workflows)


__all__ = [
    "BaseWorkflowProvider",
]
