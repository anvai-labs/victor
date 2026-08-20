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

"""Workflow registry for managing workflow definitions.

Provides centralized storage and discovery of workflow definitions,
including support for dynamic loading and caching.
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from victor.workflows.definition import (
    WorkflowDefinition,
    get_registered_workflows,
)
from victor.framework.tool_naming import canonicalize_tool_list, validate_tool_names

logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetadata:
    """Metadata about a registered workflow.

    Attributes:
        name: Workflow name
        description: Human-readable description
        agent_count: Number of agent nodes
        total_budget: Sum of agent tool budgets
        tags: Categorization tags
        created_at: When workflow was registered
        version: Workflow version
    """

    name: str
    description: str = ""
    agent_count: int = 0
    total_budget: int = 0
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    @classmethod
    def from_definition(cls, defn: WorkflowDefinition) -> "WorkflowMetadata":
        """Create metadata from a workflow definition."""
        return cls(
            name=defn.name,
            description=defn.description,
            agent_count=defn.get_agent_count(),
            total_budget=defn.get_total_budget(),
            tags=set(defn.metadata.get("tags", [])),
            version=defn.metadata.get("version", "1.0.0"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent_count": self.agent_count,
            "total_budget": self.total_budget,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


class WorkflowRegistry:
    """Central registry for workflow definitions.

    Manages workflow registration, discovery, and lifecycle.
    Supports both eager (definition) and lazy (factory) registration.

    Example:
        registry = WorkflowRegistry()

        # Register a definition directly
        registry.register(my_workflow)

        # Register a factory function
        registry.register_factory("lazy_workflow", create_lazy_workflow)

        # Get and execute
        workflow = registry.get("my_workflow")
    """

    def __init__(self):
        """Initialize the registry."""
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._factories: Dict[str, Callable[[], WorkflowDefinition]] = {}
        self._metadata: Dict[str, WorkflowMetadata] = {}
        self._loaded_modules: Set[str] = set()

    def register(
        self,
        workflow: WorkflowDefinition,
        *,
        replace: bool = False,
    ) -> None:
        """Register a workflow definition.

        Tool names in workflow nodes are automatically canonicalized
        to ensure consistency across verticals.

        Args:
            workflow: The workflow to register
            replace: If True, replace existing workflow with same name

        Raises:
            ValueError: If workflow with name exists and replace=False
        """
        if workflow.name in self._definitions and not replace:
            raise ValueError(f"Workflow '{workflow.name}' already registered")

        # Canonicalize tool names in workflow nodes
        self._canonicalize_workflow_tools(workflow)

        errors = workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        self._definitions[workflow.name] = workflow
        self._metadata[workflow.name] = WorkflowMetadata.from_definition(workflow)
        logger.info(f"Registered workflow: {workflow.name}")

    def _canonicalize_workflow_tools(self, workflow: WorkflowDefinition) -> None:
        """Canonicalize tool names in a workflow definition.

        Ensures consistent tool naming across all verticals by converting
        legacy names (e.g., 'read_file') to canonical names (e.g., 'read').

        Args:
            workflow: Workflow definition to update
        """
        for node in workflow.nodes:
            # Check for allowed_tools on node
            allowed_tools = getattr(node, "allowed_tools", None)
            if allowed_tools and isinstance(allowed_tools, list):
                # Validate and warn about legacy names
                legacy = validate_tool_names(
                    allowed_tools, context=f"workflow {workflow.name}", warn=True
                )

                # Canonicalize tool names
                canonical_tools = canonicalize_tool_list(allowed_tools)

                # Update node's allowed_tools
                try:
                    node.allowed_tools = canonical_tools
                except AttributeError:
                    if legacy:
                        logger.warning(
                            f"Cannot update frozen node tools in {workflow.name}. "
                            f"Consider using canonical names: {legacy}"
                        )

    def register_factory(
        self,
        name: str,
        factory: Callable[[], WorkflowDefinition],
        *,
        replace: bool = False,
    ) -> None:
        """Register a workflow factory function.

        The factory will be called lazily when the workflow is first accessed.

        Args:
            name: Workflow name
            factory: Function that creates the workflow
            replace: If True, replace existing factory
        """
        if name in self._factories and not replace:
            raise ValueError(f"Workflow factory '{name}' already registered")

        self._factories[name] = factory
        logger.debug(f"Registered workflow factory: {name}")

    def get(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a workflow by name.

        If workflow is registered as a factory, it will be instantiated
        and cached.

        Args:
            name: Workflow name

        Returns:
            WorkflowDefinition or None if not found
        """
        # Check cached definitions first
        if name in self._definitions:
            return self._definitions[name]

        # Try factory
        if name in self._factories:
            try:
                workflow = self._factories[name]()
                self.register(workflow, replace=True)
                return workflow
            except Exception as e:
                logger.error(f"Failed to create workflow '{name}': {e}")
                return None

        return None

    def get_metadata(self, name: str) -> Optional[WorkflowMetadata]:
        """Get workflow metadata.

        Args:
            name: Workflow name

        Returns:
            WorkflowMetadata or None
        """
        # Ensure workflow is loaded
        if name not in self._metadata and name in self._factories:
            self.get(name)  # This will populate metadata
        return self._metadata.get(name)

    def list_workflows(self) -> List[str]:
        """List all registered workflow names.

        Returns:
            List of workflow names
        """
        names = set(self._definitions.keys())
        names.update(self._factories.keys())
        return sorted(names)

    def list_metadata(self, materialize: bool = False) -> List[WorkflowMetadata]:
        """List metadata for all workflows.

        Args:
            materialize: If True, instantiate all factory-registered workflows.
                        If False (default), only return cached metadata.

        Note: With materialize=False, factories that haven't been accessed
              won't appear in the results.

        Returns:
            List of WorkflowMetadata
        """
        if materialize:
            # Original behavior: ensure all factories are instantiated
            for name in self._factories:
                if name not in self._definitions:
                    self.get(name)

        return list(self._metadata.values())

    def search(
        self,
        *,
        tags: Optional[Set[str]] = None,
        min_agents: Optional[int] = None,
        max_budget: Optional[int] = None,
    ) -> List[WorkflowDefinition]:
        """Search workflows by criteria.

        Args:
            tags: Filter by tags (workflows must have all)
            min_agents: Minimum number of agent nodes
            max_budget: Maximum total tool budget

        Returns:
            List of matching workflows
        """
        results = []

        for name in self.list_workflows():
            workflow = self.get(name)
            if not workflow:
                continue

            meta = self._metadata.get(name)
            if not meta:
                continue

            # Filter by tags
            if tags and not tags.issubset(meta.tags):
                continue

            # Filter by agent count
            if min_agents is not None and meta.agent_count < min_agents:
                continue

            # Filter by budget
            if max_budget is not None and meta.total_budget > max_budget:
                continue

            results.append(workflow)

        return results

    def unregister(self, name: str) -> bool:
        """Remove a workflow from the registry.

        Args:
            name: Workflow name

        Returns:
            True if removed, False if not found
        """
        removed = False

        if name in self._definitions:
            del self._definitions[name]
            removed = True

        if name in self._factories:
            del self._factories[name]
            removed = True

        if name in self._metadata:
            del self._metadata[name]

        if removed:
            logger.info(f"Unregistered workflow: {name}")

        return removed

    def clear(self) -> None:
        """Remove all workflows from the registry."""
        self._definitions.clear()
        self._factories.clear()
        self._metadata.clear()
        logger.info("Cleared workflow registry")

    def load_decorator_registered(self) -> int:
        """Load workflows registered via @workflow decorator.

        Returns:
            Number of workflows loaded
        """
        registered = get_registered_workflows()
        count = 0

        for name, factory in registered.items():
            if name not in self._factories and name not in self._definitions:
                self.register_factory(name, factory)
                count += 1

        logger.info(f"Loaded {count} decorator-registered workflows")
        return count

    def load_from_module(self, module_name: str) -> int:
        """Load workflows from a Python module.

        Imports the module and registers any @workflow decorated functions.

        Args:
            module_name: Fully qualified module name

        Returns:
            Number of workflows loaded
        """
        if module_name in self._loaded_modules:
            return 0

        # Validate the module name as a dotted Python identifier before import
        # so a caller-supplied string cannot reach unexpected import machinery.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", module_name):
            logger.error(f"Refusing to import invalid module name: {module_name!r}")
            return 0

        try:
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            importlib.import_module(module_name)
            self._loaded_modules.add(module_name)
            return self.load_decorator_registered()
        except ImportError as e:
            logger.error(f"Failed to import module '{module_name}': {e}")
            return 0

    def load_from_directory(self, path: Path) -> int:
        """Load workflows from Python files in a directory.

        Args:
            path: Directory containing workflow files

        Returns:
            Number of workflows loaded
        """
        if not path.is_dir():
            logger.warning(f"Not a directory: {path}")
            return 0

        count = 0
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            # Convert path to module name
            module_name = py_file.stem
            try:
                # This is a simplified loader - in production you'd want
                # proper module path resolution
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    count += self.load_decorator_registered()
            except Exception as e:
                logger.error(f"Failed to load {py_file}: {e}")

        logger.info(f"Loaded {count} workflows from {path}")
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry state to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "workflows": {name: workflow.to_dict() for name, workflow in self._definitions.items()},
            "factories": list(self._factories.keys()),
            "metadata": {name: meta.to_dict() for name, meta in self._metadata.items()},
        }


# Global registry instance
_global_registry: Optional[WorkflowRegistry] = None


def get_global_registry() -> WorkflowRegistry:
    """Get the global workflow registry instance.

    Returns:
        Global WorkflowRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = WorkflowRegistry()
    return _global_registry


__all__ = [
    "WorkflowMetadata",
    "WorkflowRegistry",
    "get_global_registry",
]
