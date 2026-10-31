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

"""Tool Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for tool selection and
tool dependency management. Following ISP, these protocols are focused
on a single responsibility: tool configuration and selection.

Usage:
    from victor.core.verticals.protocols.tool_provider import (
        ToolSelectionStrategyProtocol,
        ToolSelectionContext,
        ToolSelectionResult,
        VerticalToolSelectionProviderProtocol,
    )

    class CodingToolStrategy(ToolSelectionStrategyProtocol):
        def select_tools(self, context: ToolSelectionContext) -> ToolSelectionResult:
            if context.task_type == "edit":
                return ToolSelectionResult(
                    priority_tools=["read", "edit"],
                    budget_override=10,
                )
            return ToolSelectionResult()
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    ToolSelectionStrategyProtocol as ToolSelectionStrategyProtocol,
    VerticalToolSelectionProviderProtocol as VerticalToolSelectionProviderProtocol,
    TieredToolConfigProviderProtocol as TieredToolConfigProviderProtocol,
    VerticalTieredToolProviderProtocol as VerticalTieredToolProviderProtocol,
)
from victor_contracts.verticals.protocols.promoted_types import (
    ToolSelectionContext as ToolSelectionContext,
    ToolSelectionResult as ToolSelectionResult,
)

# The vertical-specific name is kept as an alias; ToolSelectionContext is the
# SDK's own name for the same type.
VerticalToolSelectionContext = ToolSelectionContext

__all__ = [
    "ToolSelectionContext",
    "ToolSelectionResult",
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
]
