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

"""RL integration for coding vertical.

This package provides reinforcement learning integration for the coding
vertical, enabling adaptive tool selection, continuation patience,
and quality thresholds.

Example:
    from victor_coding.rl import CodingRLHooks, CodingRLConfig

    # Create hooks with custom config
    config = CodingRLConfig()
    hooks = CodingRLHooks(config=config)

    # Record tool execution
    hooks.on_tool_success(
        tool_name="edit_files",
        task_type="refactoring",
        provider="anthropic",
        model="claude-3-opus",
        duration_ms=1500.0,
    )

    # Get recommendations
    tools = hooks.get_tool_recommendation(
        task_type="debugging",
        available_tools=["read", "grep", "shell"],
    )
"""

from victor_coding.rl.config import (
    CodingRLConfig,
    get_default_config,
)
from victor_coding.rl.hooks import (
    CodingRLHooks,
    get_coding_rl_hooks,
)

__all__ = [
    # Config
    "CodingRLConfig",
    "get_default_config",
    # Hooks
    "CodingRLHooks",
    "get_coding_rl_hooks",
]
