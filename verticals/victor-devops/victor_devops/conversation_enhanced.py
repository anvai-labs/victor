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

"""Enhanced conversation management for victor-devops using ConversationCoordinator.

This module provides DevOps-specific conversation management features using
the framework's ConversationCoordinator for better context tracking and
summarization.

Design Pattern: Extension + Delegation
- Provides DevOps-specific conversation management
- Delegates to framework ConversationCoordinator
- Tracks DevOps-specific context (deployments, infrastructure changes, etc.)

Integration Point:
    Use in DevOpsAssistant for enhanced conversation tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor_contracts import (
    ConversationCoordinator,
    ConversationStats,
    TurnType,
)
from victor_contracts.verticals import VerticalConversationManager

logger = logging.getLogger(__name__)


@dataclass
class DevOpsContext:
    """DevOps-specific conversation context.

    Tracks:
    - Deployments performed
    - Infrastructure changes
    - Docker/Kubernetes operations
    - CI/CD pipeline runs
    - System changes made
    - Alerts and incidents

    Attributes:
        deployments: List of deployments performed
        infrastructure_changes: List of infrastructure changes
        container_operations: List of Docker/K8s operations
        pipeline_runs: List of CI/CD pipeline executions
        system_changes: List of system configuration changes
        alerts: List of alerts or incidents encountered
    """

    deployments: List[Dict[str, Any]] = field(default_factory=list)
    infrastructure_changes: List[Dict[str, Any]] = field(default_factory=list)
    container_operations: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_runs: List[Dict[str, Any]] = field(default_factory=list)
    system_changes: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployments": self.deployments,
            "infrastructure_changes": self.infrastructure_changes,
            "container_operations": self.container_operations,
            "pipeline_runs": self.pipeline_runs,
            "system_changes": self.system_changes,
            "alerts": self.alerts,
        }

    def add_deployment(
        self,
        service: str,
        environment: str,
        status: str,
        version: Optional[str] = None,
    ) -> None:
        """Record a deployment.

        Args:
            service: Service being deployed
            environment: Target environment
            status: Deployment status
            version: Optional version deployed
        """
        self.deployments.append(
            {
                "service": service,
                "environment": environment,
                "status": status,
                "version": version,
            }
        )
        logger.debug(f"Recorded deployment: {service} to {environment}")

    def add_infrastructure_change(
        self,
        change_type: str,
        resource: str,
        tool: str,
    ) -> None:
        """Record an infrastructure change.

        Args:
            change_type: Type of change (create, update, delete)
            resource: Resource affected
            tool: Tool used (terraform, kubectl, docker, etc.)
        """
        self.infrastructure_changes.append(
            {
                "type": change_type,
                "resource": resource,
                "tool": tool,
            }
        )
        logger.debug(f"Recorded infrastructure change: {change_type} on {resource}")

    def add_container_operation(
        self,
        operation: str,
        container_type: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Record a container operation.

        Args:
            operation: Operation performed
            container_type: Type (docker, kubernetes, etc.)
            resource_id: Optional resource ID
        """
        self.container_operations.append(
            {
                "operation": operation,
                "type": container_type,
                "resource_id": resource_id,
            }
        )
        logger.debug(f"Recorded container operation: {operation} on {container_type}")

    def add_pipeline_run(
        self,
        pipeline: str,
        status: str,
        duration: Optional[float] = None,
    ) -> None:
        """Record a pipeline run.

        Args:
            pipeline: Pipeline name
            status: Run status
            duration: Optional duration in seconds
        """
        self.pipeline_runs.append(
            {
                "pipeline": pipeline,
                "status": status,
                "duration": duration,
            }
        )
        logger.debug(f"Recorded pipeline run: {pipeline} (status={status})")


class EnhancedDevOpsConversationManager(VerticalConversationManager):
    """Coordinator-backed conversation manager with the devops context."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        super().__init__(
            DevOpsContext,
            "devops_context",
            "devops",
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )

    def get_devops_context(self) -> DevOpsContext:
        return self.get_context()

    def track_deployment(
        self,
        service: str,
        environment: str,
        status: str,
        version: Optional[str] = None,
    ) -> None:
        """Track a deployment.

        Args:
            service: Service being deployed
            environment: Target environment
            status: Deployment status
            version: Optional version deployed
        """
        self._context.add_deployment(service, environment, status, version)

    def track_infrastructure_change(
        self,
        change_type: str,
        resource: str,
        tool: str,
    ) -> None:
        """Track an infrastructure change.

        Args:
            change_type: Type of change (create, update, delete)
            resource: Resource affected
            tool: Tool used (terraform, kubectl, docker, etc.)
        """
        self._context.add_infrastructure_change(change_type, resource, tool)

    def track_container_operation(
        self,
        operation: str,
        container_type: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Track a container operation.

        Args:
            operation: Operation performed
            container_type: Type (docker, kubernetes, etc.)
            resource_id: Optional resource ID
        """
        self._context.add_container_operation(operation, container_type, resource_id)

    def track_pipeline_run(
        self,
        pipeline: str,
        status: str,
        duration: Optional[float] = None,
    ) -> None:
        """Track a pipeline run.

        Args:
            pipeline: Pipeline name
            status: Run status
            duration: Optional duration in seconds
        """
        self._context.add_pipeline_run(pipeline, status, duration)

    # =========================================================================
    # Summarization
    # =========================================================================

    def needs_summarization(self) -> bool:
        """Check if conversation needs summarization.

        Returns:
            True if summarization is recommended
        """
        return self._conversation_coordinator.needs_summarization()

    def add_summary(self, summary: str) -> None:
        """Add a conversation summary.

        Args:
            summary: Summary text
        """
        self._conversation_coordinator.add_summary(summary)

    def get_devops_summary(self) -> str:
        """Get a DevOps-focused conversation summary.

        Returns:
            Formatted summary of DevOps work done
        """
        parts = []

        ctx = self._context

        # Deployments
        if ctx.deployments:
            parts.append("## Deployments")
            for dep in ctx.deployments:
                version = f" (v{dep['version']})" if dep.get("version") else ""
                parts.append(
                    f"- {dep['service']} to {dep['environment']}{version} - " f"{dep['status']}"
                )
            parts.append("")

        # Infrastructure changes
        if ctx.infrastructure_changes:
            parts.append("## Infrastructure Changes")
            for change in ctx.infrastructure_changes:
                parts.append(f"- {change['type']} {change['resource']} ({change['tool']})")
            parts.append("")

        # Container operations
        if ctx.container_operations:
            parts.append("## Container Operations")
            for op in ctx.container_operations:
                resource = f" on {op['resource_id']}" if op.get("resource_id") else ""
                parts.append(f"- {op['operation']} {op['type']}{resource}")
            parts.append("")

        # Pipeline runs
        if ctx.pipeline_runs:
            success = sum(1 for p in ctx.pipeline_runs if p.get("status") == "success")
            total = len(ctx.pipeline_runs)
            parts.append("## Pipeline Runs")
            parts.append(f"- Results: {success}/{total} successful")
            parts.append("")

        # Conversation stats
        stats = self._conversation_coordinator.get_stats()
        parts.append("## Conversation Stats")
        parts.append(f"- Total turns: {stats.total_turns}")
        parts.append(f"- User turns: {stats.user_turns}")
        parts.append(f"- Assistant turns: {stats.assistant_turns}")
        parts.append(f"- Tool calls: {stats.tool_calls}")

        return "\n".join(parts)


__all__ = [
    "DevOpsContext",
    "EnhancedDevOpsConversationManager",
]
