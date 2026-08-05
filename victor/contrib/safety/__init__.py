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

"""Safety extensions for Victor verticals.

Provides base classes and utilities for implementing enhanced safety
in verticals. Verticals can inherit from BaseSafetyExtension to get
common safety functionality while adding vertical-specific rules.

Usage:
    from victor.contrib.safety import BaseSafetyExtension, SafetyContext

    class MyVerticalSafetyExtension(BaseSafetyExtension):
        def get_vertical_name(self) -> str:
            return "myvertical"

        def get_vertical_rules(self) -> List[SafetyPattern]:
            return [
                SafetyPattern(
                    name="myvertical-dangerous-operation",
                    pattern=r"rm -rf /",
                    description="Dangerous file deletion",
                    severity=SafetySeverity.DANGER,
                    allowed_envs=["testing"],
                )
            ]
"""

from victor.contrib.safety.base_extension import BaseSafetyExtension
from victor.contrib.safety.safety_context import SafetyContext
from victor.contrib.safety.vertical_mixin import VerticalSafetyMixin

__all__ = [
    "BaseSafetyExtension",
    "SafetyContext",
    "VerticalSafetyMixin",
]
