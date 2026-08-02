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

"""Focused Vertical Extensions - ISP-compliant splits.

This package provides focused extension composites that replace the monolithic
VerticalExtensions class. Each composite groups related protocols:

- ToolExtensions: Middleware and tool dependencies
- PromptExtensions: Prompt contributors and enrichment strategies
- SafetyExtensions: Safety patterns and validators
- ConfigExtensions: Mode, RL, and team configurations

This follows the Interface Segregation Principle (ISP) - clients only depend
on the interfaces they actually use, rather than being forced to know about
all 10+ protocols bundled in VerticalExtensions.

Usage:
    from victor.core.verticals.extensions import (
        ToolExtensions,
        PromptExtensions,
        SafetyExtensions,
        ConfigExtensions,
    )

    # Vertical can provide only what it needs
    tool_ext = ToolExtensions(
        middleware=[CodeValidationMiddleware()],
        tool_dependencies=[ToolDependency("edit", ["read"])],
    )

    prompt_ext = PromptExtensions(
        prompt_contributors=[CodingPromptContributor()],
    )

Migration from VerticalExtensions:
    # Old way (ISP violation - bundles all protocols)
    extensions = VerticalExtensions(
        middleware=[...],
        safety_extensions=[...],
        prompt_contributors=[...],
        mode_config_provider=...,
    )

    # New way (focused composites)
    tool_ext = ToolExtensions(middleware=[...], tool_dependencies=[...])
    prompt_ext = PromptExtensions(prompt_contributors=[...])
    safety_ext = SafetyExtensions(safety_patterns=[...])
    config_ext = ConfigExtensions(mode_config=..., rl_config=..., team_specs={...})
"""

from victor.core.verticals.extensions.tool_extensions import ToolExtensions
from victor.core.verticals.extensions.prompt_extensions import PromptExtensions
from victor.core.verticals.extensions.safety_extensions import SafetyExtensions
from victor.core.verticals.extensions.config_extensions import ConfigExtensions

__all__ = [
    "ToolExtensions",
    "PromptExtensions",
    "SafetyExtensions",
    "ConfigExtensions",
]
