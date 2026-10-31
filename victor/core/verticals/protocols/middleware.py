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

"""Middleware Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for tool execution middleware.
Following ISP, these protocols are focused on a single responsibility:
intercepting and modifying tool calls.

Usage:
    from victor.core.verticals.protocols.middleware import (
        MiddlewareProtocol,
    )

    class CodeValidationMiddleware(MiddlewareProtocol):
        async def before_tool_call(
            self, tool_name: str, arguments: Dict[str, Any]
        ) -> MiddlewareResult:
            if tool_name == "write_file":
                # Validate code before writing
                ...
            return MiddlewareResult()
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    MiddlewareProtocol as MiddlewareProtocol,
)
from victor_contracts.verticals.protocols.promoted_types import (
    MiddlewarePriority as MiddlewarePriority,
    MiddlewareResult as MiddlewareResult,
)

__all__ = [
    "MiddlewareProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
]
