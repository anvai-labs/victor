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

"""Framework Extensions Protocol (ISP: Interface Segregation Principle).

This module contains the focused protocol for framework-related extensions.
Following ISP, this protocol contains ONLY framework-related fields:
- workflow_provider: Workflow management
- rl_config_provider: Reinforcement learning configuration
- team_spec_provider: Multi-agent team specifications

This protocol splits the fat VerticalExtensions interface into a focused
subset, allowing components to depend only on the framework extensions
they actually need.

Usage:
    from victor.core.verticals.protocols.focused.framework_extensions import (
        FrameworkExtensionsProtocol,
    )

    class WorkflowEngine:
        def __init__(self, extensions: FrameworkExtensionsProtocol):
            # Can access workflow, rl_config, and team_spec providers
            self.workflow = extensions.workflow_provider
            self.rl_config = extensions.rl_config_provider
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.core.verticals.protocols.rl_provider import RLConfigProviderProtocol
from victor.core.verticals.protocols.team_provider import TeamSpecProviderProtocol
from victor.core.verticals.protocols.workflow_provider import WorkflowProviderProtocol

# =============================================================================
# Framework Extensions Protocol
# =============================================================================


@runtime_checkable
class FrameworkExtensionsProtocol(Protocol):
    """Focused protocol for framework-related vertical extensions.

    This protocol contains ONLY framework-related fields extracted from
    the fat VerticalExtensions interface for ISP compliance:
    - workflow_provider: Workflow definitions and management
    - rl_config_provider: RL learner configurations and quality thresholds
    - team_spec_provider: Multi-agent team specifications

    Components that only need framework extensions can depend on this
    protocol instead of the full VerticalExtensions, reducing coupling
    and adhering to the Interface Segregation Principle.

    Example:
        class WorkflowIntegration:
            def __init__(
                self,
                extensions: FrameworkExtensionsProtocol,
            ):
                # Only access framework-related fields
                self.workflow = extensions.workflow_provider
                self.rl_config = extensions.rl_config_provider
                self.team_spec = extensions.team_spec_provider

    Note:
        All fields are Optional to support verticals that don't provide
        all framework extensions.
    """

    workflow_provider: Optional[WorkflowProviderProtocol]
    """Workflow provider for vertical-specific workflows.

    Provides workflow definitions that can be triggered by user commands
    or automatically detected. Enables YAML-first workflow architecture.
    """

    rl_config_provider: Optional[RLConfigProviderProtocol]
    """RL configuration provider for adaptive behavior.

    Provides RL learner configurations, task type mappings, and quality
    thresholds for vertical-specific reinforcement learning.
    """

    team_spec_provider: Optional[TeamSpecProviderProtocol]
    """Team specification provider for multi-agent coordination.

    Provides team definitions for complex task execution requiring
    multiple specialized agents.
    """

    def get_workflows(self) -> Dict[str, Any]:
        """Get workflow definitions from the workflow provider.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances.
            Returns empty dict if no workflow provider is available.
        """
        if self.workflow_provider is not None:
            return self.workflow_provider.get_workflows()
        return {}

    def get_auto_workflows(self) -> List[Any]:
        """Get automatically triggered workflow patterns.

        Returns:
            List of (pattern, workflow_name) tuples for auto-triggering.
            Returns empty list if no workflow provider is available.
        """
        if self.workflow_provider is not None:
            return self.workflow_provider.get_auto_workflows()
        return []

    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration from the RL config provider.

        Returns:
            Dict with RL configuration including active_learners,
            quality_thresholds, and task_type_mappings.
            Returns empty dict if no RL config provider is available.
        """
        if self.rl_config_provider is not None:
            return self.rl_config_provider.get_rl_config()
        return {}

    def get_rl_hooks(self) -> Optional[Any]:
        """Get RL hooks from the RL config provider.

        Returns:
            RLHooks instance or None if not available.
        """
        if self.rl_config_provider is not None:
            return self.rl_config_provider.get_rl_hooks()
        return None

    def get_team_specs(self) -> Dict[str, Any]:
        """Get team specifications from the team spec provider.

        Returns:
            Dict mapping team names to TeamSpec instances.
            Returns empty dict if no team spec provider is available.
        """
        if self.team_spec_provider is not None:
            return self.team_spec_provider.get_team_specs()
        return {}

    def get_default_team(self) -> Optional[str]:
        """Get the default team name from the team spec provider.

        Returns:
            Default team name or None if not available.
        """
        if self.team_spec_provider is not None:
            return self.team_spec_provider.get_default_team()
        return None


__all__ = [
    "FrameworkExtensionsProtocol",
]
