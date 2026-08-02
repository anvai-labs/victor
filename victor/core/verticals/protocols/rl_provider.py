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

"""RL Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for RL (Reinforcement Learning)
configuration. Following ISP, these protocols are focused on a single
responsibility: providing RL learner configurations.

Usage:
    from victor.core.verticals.protocols.rl_provider import (
        RLConfigProviderProtocol,
        VerticalRLProviderProtocol,
    )

    class CodingRLConfigProvider(RLConfigProviderProtocol):
        def get_rl_config(self) -> Dict[str, Any]:
            return {
                "active_learners": ["tool_selection", "semantic_threshold"],
                "quality_thresholds": {"code_review": 0.8, "bugfix": 0.85},
            }
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable

# =============================================================================
# RL Config Provider Protocol
# =============================================================================


@runtime_checkable
class RLConfigProviderProtocol(Protocol):
    """Protocol for providing RL (Reinforcement Learning) configuration.

    Enables verticals to configure RL learners, task type mappings,
    and quality thresholds for adaptive behavior.

    Example:
        class CodingRLConfigProvider(RLConfigProviderProtocol):
            def get_rl_config(self) -> Dict[str, Any]:
                return {
                    "active_learners": ["tool_selection", "semantic_threshold"],
                    "quality_thresholds": {"code_review": 0.8, "bugfix": 0.85},
                }
    """

    @abstractmethod
    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration for this vertical.

        Returns:
            Dict with RL configuration including:
            - active_learners: List of learner types to enable
            - quality_thresholds: Task-specific quality thresholds
            - task_type_mappings: Map task types to learner configs
        """
        ...

    def get_rl_hooks(self) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        return None


# =============================================================================
# Vertical RL Provider Protocol
# =============================================================================


@runtime_checkable
class VerticalRLProviderProtocol(Protocol):
    """Protocol for verticals providing RL configuration.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical RL configuration with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalRLProviderProtocol):
            @classmethod
            def get_rl_config_provider(cls) -> Optional[RLConfigProviderProtocol]:
                return CodingRLConfigProvider()

            @classmethod
            def get_rl_hooks(cls) -> Optional[Any]:
                return CodingRLHooks()
    """

    @classmethod
    def get_rl_config_provider(cls) -> Optional[RLConfigProviderProtocol]:
        """Get the RL configuration provider for this vertical.

        Returns:
            RLConfigProviderProtocol implementation or None
        """
        ...

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        ...


__all__ = [
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
]
