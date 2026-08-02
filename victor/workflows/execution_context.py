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

"""Execution context for workflow nodes.

Provides context (orchestrator, settings, services) to node executors.
This is a stub during the SOLID refactoring migration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WorkflowNodeContext:
    """Execution context for workflow nodes.

    Responsibility (SRP):
    - Provide orchestrator access for agent nodes
    - Provide execution settings
    - Provide service container access
    - Track execution metadata

    Non-responsibility:
    - State management (handled by StateGraph)
    - Node execution logic (handled by node executors)

    Note:
        Renamed from ExecutionContext (2026-04-19) to avoid name collision
        with victor.runtime.context.RuntimeExecutionContext. Use WorkflowNodeContext
        for workflow node execution contexts.
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        settings: Optional[Any] = None,
        services: Optional[Any] = None,
    ):
        """Initialize the execution context.

        Args:
            orchestrator: AgentOrchestrator instance
            settings: Application settings
            services: Service container
        """
        self._orchestrator = orchestrator
        self._settings = settings
        self._services = services

    @property
    def orchestrator(self) -> Any:
        """Get orchestrator."""
        return self._orchestrator

    @property
    def settings(self) -> Any:
        """Get settings."""
        return self._settings

    @property
    def services(self) -> Any:
        """Get services."""
        return self._services


# Backward compatibility alias
ExecutionContext = WorkflowNodeContext


__all__ = ["WorkflowNodeContext", "ExecutionContext"]
