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

"""Factory for creating coordinators with protocol-based dependencies.

This module provides a factory for creating coordinators with protocol-based
dependency injection, reducing direct coupling to AgentOrchestrator.

Design Patterns:
- Factory Pattern: Centralized object creation
- Dependency Injection: Inject protocol-based dependencies
- Builder Pattern: Construct complex objects step-by-step

Example:
    >>> from victor.agent.coordinators.coordinator_factory import CoordinatorFactory
    >>> from victor.core import get_container
    >>>
    >>> factory = CoordinatorFactory(get_container())
    >>> prompt_coordinator = factory.create_system_prompt_state_passed_coordinator()
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from victor.agent.coordinators.factory_support import (
    create_exploration_coordinator as build_exploration_coordinator,
    create_exploration_state_passed_coordinator as build_exploration_state_passed_coordinator,
    create_safety_state_passed_coordinator as build_safety_state_passed_coordinator,
    create_system_prompt_state_passed_coordinator as build_system_prompt_state_passed_coordinator,
)
from victor.agent.coordinators.protocol_dependencies import OrchestratorProtocolAdapter
from victor.agent.orchestrator_protocols import (
    IAgentOrchestrator,
)

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = [
    "CoordinatorFactory",
]


class CoordinatorFactory:
    """
    Factory for creating coordinators with protocol-based dependencies.

    This factory resolves coordinator dependencies from the DI container
    and provides protocol-based adapters for orchestrator functionality.

    Supported surfaces:
        - ExplorationCoordinator: Canonical read-only exploration runtime
        - ExplorationStatePassedCoordinator: Snapshot/transition exploration wrapper
        - SystemPromptStatePassedCoordinator: Canonical state-passed prompt classification
        - AgentMetricsService: Metrics collection
        - SafetyCoordinator: SDK-owned compatibility / extension surface
        - SafetyStatePassedCoordinator: Canonical state-passed safety wrapper
        - ConversationCoordinator: SDK-owned compatibility / extension surface

    Example:
        >>> factory = CoordinatorFactory(container)
        >>>
        >>> # Create canonical state-passed prompt coordination
        >>> prompt_coordinator = factory.create_system_prompt_state_passed_coordinator()
    """

    def __init__(self, container: "ServiceContainer"):
        """
        Initialize factory.

        Args:
            container: DI container instance
        """
        self._container = container
        self._orchestrator_adapter: Optional[OrchestratorProtocolAdapter] = None

    def create_exploration_coordinator(self) -> Any:
        """
        Create the canonical read-only exploration coordinator.

        Returns:
            ExplorationCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            coordinator = build_exploration_coordinator()

            logger.debug("CoordinatorFactory: Created ExplorationCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create ExplorationCoordinator: {e}")
            raise RuntimeError(f"Failed to create ExplorationCoordinator: {e}") from e

    def create_exploration_state_passed_coordinator(
        self,
        project_root: Optional["Path"] = None,
        max_results: int = 5,
    ) -> Any:
        """
        Create the canonical state-passed exploration coordinator.

        Returns:
            ExplorationStatePassedCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            coordinator = build_exploration_state_passed_coordinator(
                settings=getattr(self._container, "_settings", None),
                project_root=project_root,
                max_results=max_results,
            )

            logger.debug("CoordinatorFactory: Created ExplorationStatePassedCoordinator")
            return coordinator

        except Exception as e:
            logger.error(
                "CoordinatorFactory: Failed to create ExplorationStatePassedCoordinator: %s",
                e,
            )
            raise RuntimeError(f"Failed to create ExplorationStatePassedCoordinator: {e}") from e

    def create_system_prompt_state_passed_coordinator(
        self,
        task_analyzer: Optional[Any] = None,
    ) -> Any:
        """
        Create the canonical state-passed system prompt coordinator.

        Returns:
            SystemPromptStatePassedCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            coordinator = build_system_prompt_state_passed_coordinator(
                container=self._container,
                task_analyzer=task_analyzer,
            )

            logger.debug("CoordinatorFactory: Created SystemPromptStatePassedCoordinator")
            return coordinator

        except Exception as e:
            logger.error(
                "CoordinatorFactory: Failed to create SystemPromptStatePassedCoordinator: %s",
                e,
            )
            raise RuntimeError(f"Failed to create SystemPromptStatePassedCoordinator: {e}") from e

    def create_agent_metrics_service(self) -> Any:
        """
        Create metrics coordinator with protocol dependencies.

        Returns:
            AgentMetricsService instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.services.metrics_service import AgentMetricsService

            # Resolve orchestrator for metrics access
            orchestrator = self._resolve_orchestrator()

            # Create coordinator
            coordinator = AgentMetricsService(orchestrator=orchestrator)

            logger.debug("CoordinatorFactory: Created AgentMetricsService")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create AgentMetricsService: {e}")
            raise RuntimeError(f"Failed to create AgentMetricsService: {e}") from e

    def create_safety_coordinator(self) -> Any:
        """
        Create the deprecated SDK-owned safety compatibility surface.

        Returns:
            SafetyCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            self._warn_sdk_compatibility_surface(
                "create_safety_coordinator()",
                "Use create_safety_state_passed_coordinator() for agent runtime "
                "behavior, or instantiate victor_contracts.safety.SafetyCoordinator "
                "directly for extensions.",
            )
            from victor_contracts.safety import SafetyCoordinator

            coordinator = SafetyCoordinator()

            logger.debug("CoordinatorFactory: Created SafetyCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create SafetyCoordinator: {e}")
            raise RuntimeError(f"Failed to create SafetyCoordinator: {e}") from e

    def create_safety_state_passed_coordinator(self) -> Any:
        """
        Create the canonical state-passed safety coordinator.

        Returns:
            SafetyStatePassedCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            coordinator = build_safety_state_passed_coordinator()

            logger.debug("CoordinatorFactory: Created SafetyStatePassedCoordinator")
            return coordinator

        except Exception as e:
            logger.error(
                "CoordinatorFactory: Failed to create SafetyStatePassedCoordinator: %s",
                e,
            )
            raise RuntimeError(f"Failed to create SafetyStatePassedCoordinator: {e}") from e

    def create_conversation_coordinator(self) -> Any:
        """
        Create the deprecated SDK-owned conversation compatibility surface.

        Returns:
            ConversationCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            self._warn_sdk_compatibility_surface(
                "create_conversation_coordinator()",
                "Conversation coordination is not a canonical victor.agent "
                "runtime surface; instantiate "
                "victor_contracts.conversation.ConversationCoordinator directly for "
                "extensions.",
            )
            from victor_contracts.conversation import ConversationCoordinator

            coordinator = ConversationCoordinator()

            logger.debug("CoordinatorFactory: Created ConversationCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create ConversationCoordinator: {e}")
            raise RuntimeError(f"Failed to create ConversationCoordinator: {e}") from e

    def _warn_sdk_compatibility_surface(self, method_name: str, guidance: str) -> None:
        """Warn when legacy factory methods expose SDK-owned compatibility seams."""
        warnings.warn(
            f"CoordinatorFactory.{method_name} is a deprecated SDK-owned "
            f"compatibility surface. {guidance}",
            DeprecationWarning,
            stacklevel=2,
        )

    def _resolve_orchestrator(self) -> IAgentOrchestrator:
        """Resolve orchestrator from container or create adapter."""
        # Try container first
        orchestrator = self._container.get_optional(IAgentOrchestrator)
        if orchestrator is not None:
            return orchestrator

        # Create adapter from orchestrator in container
        if self._orchestrator_adapter is None:
            # Get orchestrator from container (by class lookup or type)
            from victor.agent.orchestrator import AgentOrchestrator

            orchestrator_impl = self._container.get_optional(AgentOrchestrator)
            if orchestrator_impl is not None:
                self._orchestrator_adapter = OrchestratorProtocolAdapter(orchestrator_impl)
            else:
                logger.warning("CoordinatorFactory: No orchestrator found in container")
                raise RuntimeError("Orchestrator not found in container")

        return self._orchestrator_adapter
