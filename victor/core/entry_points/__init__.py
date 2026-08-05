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

"""Entry point protocol definitions for Victor extensibility.

This module defines the protocols that vertical packages must implement
to register their capabilities with the Victor framework via entry points.

Entry Point Groups:
    victor.plugins            - Canonical plugin packages that register verticals
    victor.verticals          - Legacy raw-class compatibility path
    victor.safety_rules        - Safety rule registration functions
    victor.tool_dependencies  - Tool dependency provider factories
    victor.rl_configs          - RL configuration provider factories
    victor.escape_hatches     - Escape hatch registration functions
    victor.commands            - CLI command registration functions

Example:
    # Canonical plugin registration
    [project.entry-points."victor.plugins"]
    coding = "victor_coding:plugin"

    # In victor-coding/pyproject.toml:
    [project.entry-points."victor.safety_rules"]
    coding = "victor_coding.safety:register_coding_safety_rules"

    # In victor_coding/safety.py:
    def register_coding_safety_rules(enforcer: SafetyEnforcer) -> None:
        enforcer.register_rule(CodingFileOperationRule())
        enforcer.register_rule(CodingGitOperationRule())
"""

from victor.core.entry_points.command_provider import CommandProvider
from victor.core.entry_points.escape_hatch_provider import EscapeHatchProvider
from victor.core.entry_points.rl_config_provider import RLConfigProvider
from victor.core.entry_points.safety_rule_provider import SafetyRuleProvider
from victor.core.entry_points.tool_dependency_provider import ToolDependencyProvider

__all__ = [
    "SafetyRuleProvider",
    "ToolDependencyProvider",
    "RLConfigProvider",
    "EscapeHatchProvider",
    "CommandProvider",
]
