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

"""Workflow Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for workflow management.
Following ISP, these protocols are focused on a single responsibility:
providing and managing vertical-specific workflows.

Usage:
    from victor.core.verticals.protocols.workflow_provider import (
        WorkflowProviderProtocol,
        VerticalWorkflowProviderProtocol,
    )

    class CodingWorkflowProvider(WorkflowProviderProtocol):
        def get_workflows(self) -> Dict[str, Any]:
            return {
                "feature": feature_implementation_workflow(),
                "bugfix": bug_fix_workflow(),
            }
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# =============================================================================
# Workflow Provider Protocol
# =============================================================================


@runtime_checkable
class WorkflowProviderProtocol(Protocol):
    """Protocol for providing vertical-specific workflows.

    Workflows are named sequences of operations that can be
    triggered by user commands or automatically detected.

    Note: Return type uses Any to support both workflow classes (Type)
    and WorkflowDefinition instances. Implementations typically return
    Dict[str, WorkflowDefinition] from WorkflowBuilder.

    Example:
        class CodingWorkflowProvider(WorkflowProviderProtocol):
            def get_workflows(self) -> Dict[str, Any]:
                return {
                    "feature": feature_implementation_workflow(),
                    "bugfix": bug_fix_workflow(),
                }
    """

    @abstractmethod
    def get_workflows(self) -> Dict[str, Any]:
        """Get workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
            or workflow classes (Type). Most implementations return
            WorkflowDefinition from WorkflowBuilder.
        """
        ...

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatically triggered workflows.

        Returns:
            List of (pattern, workflow_name) tuples for auto-triggering
        """
        return []


# =============================================================================
# Vertical Workflow Provider Protocol
# =============================================================================


@runtime_checkable
class VerticalWorkflowProviderProtocol(Protocol):
    """Protocol for verticals providing workflow definitions.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical workflows with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalWorkflowProviderProtocol):
            @classmethod
            def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
                return CodingWorkflowProvider()
    """

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get the workflow provider for this vertical.

        Returns:
            WorkflowProviderProtocol implementation or None
        """
        ...


__all__ = [
    "WorkflowProviderProtocol",
    "VerticalWorkflowProviderProtocol",
]
