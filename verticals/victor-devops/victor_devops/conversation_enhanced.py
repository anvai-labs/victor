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


class EnhancedDevOpsConversationManager:
    """Enhanced conversation manager for DevOps using ConversationCoordinator.

    Provides:
    - Standard conversation tracking via ConversationCoordinator
    - DevOps-specific context tracking (deployments, infrastructure, containers)
    - Automatic summarization of DevOps work
    - Operations-focused conversation history

    Example:
        manager = EnhancedDevOpsConversationManager()

        # Add a user message
        manager.add_message("user", "Deploy the new version to staging", TurnType.USER)

        # Track deployment
        manager.track_deployment("my-service", "staging", "success", "v1.2.3")

        # Get conversation summary
        summary = manager.get_devops_summary()
    """

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        """Initialize the enhanced conversation manager.

        Args:
            max_history_turns: Maximum turns to keep in history
            summarization_threshold: Turns before triggering summarization
            enable_deduplication: Whether to enable message deduplication
            enable_statistics: Whether to track conversation statistics
        """
        self._conversation_coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )

        self._devops_context = DevOpsContext()

        logger.info(
            f"EnhancedDevOpsConversationManager initialized with " f"max_turns={max_history_turns}"
        )

    # =========================================================================
    # Message Management (delegates to ConversationCoordinator)
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            turn_type: Type of turn
            metadata: Optional metadata
            tool_calls: Optional tool calls made in this turn

        Returns:
            Turn ID for the added message
        """
        return self._conversation_coordinator.add_message(
            role, content, turn_type, metadata, tool_calls
        )

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of turns to return
            include_system: Whether to include system messages
            include_tool: Whether to include tool messages

        Returns:
            List of message dictionaries
        """
        return self._conversation_coordinator.get_history(max_turns, include_system, include_tool)

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_summaries: Whether to keep conversation summaries
        """
        self._conversation_coordinator.clear_history(keep_summaries)
        if not keep_summaries:
            self._devops_context = DevOpsContext()
        logger.info("Conversation history cleared")

    # =========================================================================
    # DevOps-Specific Context Tracking
    # =========================================================================

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
        self._devops_context.add_deployment(service, environment, status, version)

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
        self._devops_context.add_infrastructure_change(change_type, resource, tool)

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
        self._devops_context.add_container_operation(operation, container_type, resource_id)

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
        self._devops_context.add_pipeline_run(pipeline, status, duration)

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

        ctx = self._devops_context

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

    # =========================================================================
    # Statistics and Observability
    # =========================================================================

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics.

        Returns:
            ConversationStats object
        """
        return self._conversation_coordinator.get_stats()

    def get_devops_context(self) -> DevOpsContext:
        """Get the DevOps context.

        Returns:
            DevOpsContext object
        """
        return self._devops_context

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration.

        Returns:
            Dictionary with observability data
        """
        conv_obs = self._conversation_coordinator.get_observability_data()

        return {
            **conv_obs,
            "devops_context": self._devops_context.to_dict(),
            "vertical": "devops",
        }

    def get_conversation_coordinator(self) -> ConversationCoordinator:
        """Get the underlying ConversationCoordinator.

        Returns:
            ConversationCoordinator instance
        """
        return self._conversation_coordinator


__all__ = [
    "DevOpsContext",
    "EnhancedDevOpsConversationManager",
]
