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

"""RL configuration for coding vertical.

Provides coding-specific configuration for the RL system including
learner activations, task type mappings, and quality thresholds.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from victor_contracts.constants import ToolNames
from victor_contracts.rl import BaseRLConfig, LearnerType


@dataclass
class CodingRLConfig(BaseRLConfig):
    """RL configuration for coding vertical.

    Inherits common RL configuration from BaseRLConfig and extends
    with coding-specific task types, quality thresholds, and tool conflicts.

    This configuration customizes RL behavior for software development
    tasks, including which learners are active, tool recommendations
    for different task types, and quality thresholds.

    Attributes:
        active_learners: Learners to activate for coding tasks
        task_type_mappings: Maps task types to recommended tools
        quality_thresholds: Quality thresholds by task type
        default_patience: Default continuation patience by provider
        exploration_bonus: Bonus weight for less-used tools
        conflicting_tools: Tools that should not be recommended together

    Example:
        config = CodingRLConfig()
        config.get_tools_for_task("debugging")
        # Returns: ["read", "grep", "shell", "run_tests", "git_log"]
    """

    # Coding uses additional learners (MODE_TRANSITION, QUALITY_WEIGHTS)
    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.CONTINUATION_PATIENCE,
            LearnerType.GROUNDING_THRESHOLD,
            LearnerType.MODE_TRANSITION,
            LearnerType.QUALITY_WEIGHTS,
        ]
    )

    # Task type to tool mappings for tool selection learning
    # Uses canonical ToolNames constants for consistency
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # Analysis tasks
            "refactoring": [
                ToolNames.RENAME,
                ToolNames.EXTRACT,
                ToolNames.EDIT,
                ToolNames.READ,
            ],
            "debugging": [
                ToolNames.READ,
                ToolNames.GREP,
                ToolNames.SHELL,
                ToolNames.TEST,
                ToolNames.GIT,
                ToolNames.SYMBOL,
                ToolNames.REFS,
            ],
            "exploration": [
                ToolNames.READ,
                ToolNames.GREP,
                ToolNames.CODE_SEARCH,
                ToolNames.OVERVIEW,
                ToolNames.SYMBOL,
                ToolNames.LS,
            ],
            # Implementation tasks
            "feature": [
                ToolNames.READ,
                ToolNames.WRITE,
                ToolNames.EDIT,
                ToolNames.SHELL,
                ToolNames.GIT,
            ],
            "implementation": [
                ToolNames.READ,
                ToolNames.WRITE,
                ToolNames.EDIT,
                ToolNames.SHELL,
                ToolNames.TEST,
            ],
            "testing": [
                ToolNames.TEST,
                ToolNames.SHELL,
                ToolNames.READ,
                ToolNames.WRITE,
            ],
            # Documentation tasks
            "documentation": [
                ToolNames.READ,
                ToolNames.WRITE,
                ToolNames.EDIT,
                ToolNames.GREP,
            ],
            # Review tasks
            "review": [
                ToolNames.READ,
                ToolNames.GREP,
                ToolNames.GIT,
                ToolNames.REFS,
            ],
        }
    )

    # Quality thresholds by task type (higher = stricter)
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "refactoring": 0.90,  # High bar for refactoring
            "debugging": 0.85,
            "feature": 0.80,
            "implementation": 0.80,
            "exploration": 0.70,  # Lower bar for exploration
            "testing": 0.85,
            "documentation": 0.75,
            "review": 0.80,
        }
    )

    # Coding-specific: different patience values (3 for cloud, 7 for local)
    # This overrides the BaseRLConfig default
    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 3,
            "openai": 3,
            "google": 3,
            "deepseek": 5,  # More patient with DeepSeek
            "ollama": 7,  # Most patient with local models
            "lmstudio": 7,
            "vllm": 7,
        }
    )

    # exploration_bonus inherited from BaseRLConfig (0.15)

    # Tools that should never be recommended together (conflicting)
    # Uses canonical ToolNames constants for consistency
    conflicting_tools: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            ToolNames.WRITE: {ToolNames.EDIT},  # Use one or the other
            ToolNames.EDIT: {ToolNames.WRITE},
        }
    )

    # Methods get_tools_for_task, get_quality_threshold, get_patience,
    # is_learner_active, get_rl_config, __repr__ all inherited


# Default singleton instance
_default_config: CodingRLConfig | None = None


def get_default_config() -> CodingRLConfig:
    """Get the default coding RL configuration.

    Returns:
        Default CodingRLConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = CodingRLConfig()
    return _default_config


__all__ = [
    "CodingRLConfig",
    "get_default_config",
]
