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

"""Dynamic workflow discovery - DIP-compliant workflow loading.

This module implements dynamic workflow discovery to eliminate
hardcoded workflow imports in the orchestrator.

Design Goals:
- Dependency Inversion: Orchestrator depends on abstractions, not concrete workflows
- Open/Closed: Add new workflows without modifying orchestrator
- Single Responsibility: Discovery logic centralized here

Usage:
    from victor.workflows.discovery import discover_workflows, register_builtin_workflows

    # Register all built-in workflows dynamically
    registry = WorkflowRegistry()
    register_builtin_workflows(registry)

    # Or discover from a package
    workflows = discover_workflows("victor.workflows")
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from victor.workflows.base import BaseWorkflow, WorkflowRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Registration Decorator
# =============================================================================

_pending_workflows: List[Type[BaseWorkflow]] = []


def workflow_class(cls: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
    """Decorator to mark a class for auto-registration.

    Usage:
        @workflow_class
        class MyWorkflow(BaseWorkflow):
            name = "my_workflow"
            ...
    """
    _pending_workflows.append(cls)
    return cls


# =============================================================================
# Discovery Functions
# =============================================================================


def discover_workflows(
    package_path: str = "victor.workflows",
    exclude_modules: Optional[List[str]] = None,
) -> List[BaseWorkflow]:
    """Discover all workflow classes in a package.

    Scans the package for classes that inherit from BaseWorkflow
    and instantiates them.

    Args:
        package_path: Python package path to scan (default: victor.workflows)
        exclude_modules: Module names to exclude from scanning

    Returns:
        List of workflow instances
    """
    exclude = set(exclude_modules or [])
    exclude.update(
        {
            "base",
            "registry",
            "executor",
            "definition",
            "builder",
            "yaml_loader",
            "cache",
            "protocols",
            "adapters",
            "trigger_registry",
            "graph",
            "graph_dsl",
            "hitl",
            "discovery",  # Don't scan ourselves
        }
    )

    workflows: List[BaseWorkflow] = []

    try:
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        package = importlib.import_module(package_path)
        package_dir = Path(package.__file__).parent

        for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
            if is_pkg or module_name in exclude:
                continue

            try:
                # Module names come from scanning our own package directory,
                # not from user input.
                # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
                module = importlib.import_module(f"{package_path}.{module_name}")

                # Find all BaseWorkflow subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseWorkflow)
                        and obj is not BaseWorkflow
                        and not inspect.isabstract(obj)
                    ):
                        try:
                            instance = obj()
                            workflows.append(instance)
                            logger.debug(f"Discovered workflow: {instance.name}")
                        except Exception as e:
                            logger.warning(f"Failed to instantiate workflow {name}: {e}")

            except Exception as e:
                logger.debug(f"Failed to import workflow module {module_name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to discover workflows from {package_path}: {e}")

    return workflows


def register_builtin_workflows(registry: WorkflowRegistry) -> int:
    """Register all built-in workflows dynamically.

    This replaces the hardcoded imports in the orchestrator,
    following the Dependency Inversion Principle.

    Args:
        registry: The workflow registry to register to

    Returns:
        Number of workflows registered
    """
    count = 0

    # First, register any decorated workflows
    for workflow_cls in _pending_workflows:
        try:
            if registry.get(workflow_cls().name) is None:
                registry.register(workflow_cls())
                count += 1
        except Exception as e:
            logger.warning(f"Failed to register decorated workflow {workflow_cls}: {e}")

    # Then discover from the workflows package
    workflows = discover_workflows()
    for workflow in workflows:
        try:
            if registry.get(workflow.name) is None:
                registry.register(workflow)
                count += 1
                logger.debug(f"Registered workflow: {workflow.name}")
        except ValueError:
            # Already registered
            pass
        except Exception as e:
            logger.warning(f"Failed to register workflow {workflow.name}: {e}")

    return count


def get_workflow_by_name(
    name: str,
    registry: Optional[WorkflowRegistry] = None,
) -> Optional[BaseWorkflow]:
    """Get a workflow by name, with lazy discovery.

    Args:
        name: Workflow name
        registry: Optional registry to use

    Returns:
        Workflow instance or None
    """
    if registry:
        workflow = registry.get(name)
        if workflow:
            return workflow

    # Try to discover
    workflows = discover_workflows()
    for workflow in workflows:
        if workflow.name == name:
            return workflow

    return None


__all__ = [
    "workflow_class",
    "discover_workflows",
    "register_builtin_workflows",
    "get_workflow_by_name",
]
