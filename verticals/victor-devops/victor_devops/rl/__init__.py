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

"""RL integration for DevOps vertical.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from victor_contracts.constants import ToolNames
from victor_contracts.rl import BaseRLConfig


@dataclass
class DevOpsRLConfig(BaseRLConfig):
    """RL configuration for DevOps vertical.

    Inherits common RL configuration from BaseRLConfig and extends
    with DevOps-specific task types and quality thresholds.
    """

    # active_learners inherited from BaseRLConfig
    # default_patience inherited from BaseRLConfig

    # Uses canonical ToolNames constants for consistency
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "deployment": [
                ToolNames.SHELL,
                ToolNames.DOCKER,
                ToolNames.GIT,
                ToolNames.READ,
                ToolNames.EDIT,
            ],
            "containerization": [
                ToolNames.DOCKER,
                ToolNames.SHELL,
                ToolNames.READ,
                ToolNames.WRITE,
            ],
            "monitoring": [
                ToolNames.SHELL,
                ToolNames.READ,
                ToolNames.WRITE,
                ToolNames.GREP,
            ],
            "configuration": [
                ToolNames.READ,
                ToolNames.WRITE,
                ToolNames.EDIT,
                ToolNames.GREP,
            ],
            "troubleshooting": [
                ToolNames.SHELL,
                ToolNames.READ,
                ToolNames.GREP,
                ToolNames.DOCKER,
            ],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "deployment": 0.90,  # High bar for deployments
            "containerization": 0.85,
            "monitoring": 0.80,
            "configuration": 0.85,
            "troubleshooting": 0.80,
        }
    )

    # default_patience inherited from BaseRLConfig
    # Methods get_tools_for_task, get_quality_threshold, get_patience,
    # is_learner_active, get_rl_config, __repr__ all inherited


class DevOpsRLHooks:
    """RL recording hooks for DevOps middleware."""

    def __init__(self, config: Optional[DevOpsRLConfig] = None):
        self._config = config or DevOpsRLConfig()

    @property
    def config(self) -> DevOpsRLConfig:
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        return self._config.get_quality_threshold(task_type)

    def __repr__(self) -> str:
        return f"DevOpsRLHooks(config={self._config})"


_default_config: DevOpsRLConfig | None = None
_hooks_instance: DevOpsRLHooks | None = None


def get_default_config() -> DevOpsRLConfig:
    global _default_config
    if _default_config is None:
        _default_config = DevOpsRLConfig()
    return _default_config


def get_devops_rl_hooks() -> DevOpsRLHooks:
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = DevOpsRLHooks()
    return _hooks_instance


__all__ = [
    "DevOpsRLConfig",
    "DevOpsRLHooks",
    "get_default_config",
    "get_devops_rl_hooks",
]
