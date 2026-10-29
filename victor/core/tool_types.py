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

"""Core import path for the tool-dependency types.

The types are defined once, in the SDK (``victor_contracts``); this module
re-exports them so the long-standing ``victor.core.tool_types`` import path
keeps working and yields the same objects. It used to hold parallel copies
of all three.

Note:
    These types are also re-exported from `victor.core.verticals.protocols`.
"""

from victor_contracts.verticals.protocols.tools import (
    ToolDependency as ToolDependency,
    ToolDependencyProviderProtocol as ToolDependencyProviderProtocol,
)
from victor_contracts.verticals.tool_dependencies import (
    EmptyToolDependencyProvider as EmptyToolDependencyProvider,
)

__all__ = [
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "EmptyToolDependencyProvider",
]
