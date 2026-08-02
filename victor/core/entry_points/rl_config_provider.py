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

"""RL Config Provider Protocol.

This protocol defines the interface that vertical packages must implement
to register their reinforcement learning configuration with the Victor framework.

Verticals register RL configs via the `victor.rl_configs` entry point group.
Each registration function should conform to this protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class RLConfigProvider(Protocol):
    """Protocol for RL configuration provider factory functions.

    RL configuration provider factory functions are called by the framework to
    obtain vertical-specific RL configurations. These functions are registered
    via the `victor.rl_configs` entry point group.

    The function signature must be:
        def get_rl_config() -> Optional[Dict[str, Any]]

    Example:
        # In victor_coding/rl/config.py:
        def get_rl_config() -> Optional[Dict[str, Any]]:
            \"\"\"Return RL configuration for coding vertical.\"\"\"
            return {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "tool_rewards": {
                    "read": 0.1,
                    "write": 0.2,
                    "edit": 0.3,
                },
            }

        # In victor-coding/pyproject.toml:
        [project.entry-points."victor.rl_configs"]
        coding = "victor_coding.rl.config:get_rl_config"

        # Framework usage:
        from importlib.metadata import entry_points
        eps = entry_points(group="victor.rl_configs")
        for ep in eps:
            if ep.name == "coding":
                config_factory = ep.load()
                rl_config = config_factory()
                if rl_config:
                    apply_rl_config(rl_config)
    """

    def __call__(self) -> Optional[Dict[str, Any]]:
        """Return an RL configuration dictionary for the vertical.

        Returns:
            A dictionary containing RL configuration parameters, or None if
            the vertical doesn't use RL.
        """
        ...
