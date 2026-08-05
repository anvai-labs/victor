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

"""Mode configuration helpers for Victor verticals.

Provides base classes and utilities for implementing mode configuration
in verticals. Builds on top of the framework's ModeConfigRegistry to
provide vertical-specific convenience methods.

Usage:
    from victor.contrib.mode_config import BaseModeConfigProvider

    class MyVerticalModeConfig(BaseModeConfigProvider):
        def get_vertical_name(self) -> str:
            return \"myvertical\"

        def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
            return {
                \"fast\": ModeDefinition(name=\"fast\", tool_budget=5, ...),
                \"thorough\": ModeDefinition(name=\"thorough\", tool_budget=30, ...),
            }
"""

from victor.contrib.mode_config.base_provider import BaseModeConfigProvider
from victor.contrib.mode_config.mode_helpers import ModeHelperMixin

__all__ = [
    "BaseModeConfigProvider",
    "ModeHelperMixin",
]
