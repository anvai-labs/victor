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

"""Tool usage accounting for ``ToolService``."""

from __future__ import annotations

from typing import Any, Dict


class ToolUsageStats:
    """Track successful and failed tool executions."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    @property
    def counts(self) -> Dict[str, int]:
        """Return the mutable backing counts for compatibility surfaces."""
        return self._counts

    @counts.setter
    def counts(self, value: Dict[str, int]) -> None:
        self._counts = dict(value)

    def record(self, tool_name: str, *, success: bool) -> None:
        """Record one tool execution outcome."""
        key = tool_name if success else f"error:{tool_name}"
        self._counts[key] = self._counts.get(key, 0) + 1

    def clear(self) -> None:
        """Clear all accumulated usage statistics."""
        self._counts.clear()

    def get_tool_call_count(self, tool_name: str) -> int:
        """Return successful call count for a tool."""
        return self._counts.get(tool_name, 0)

    def get_tool_error_count(self, tool_name: str) -> int:
        """Return failed call count for a tool."""
        return self._counts.get(f"error:{tool_name}", 0)

    def snapshot(self, *, budget_remaining: int, budget_used: int) -> Dict[str, Any]:
        """Return the public ToolService usage-stat payload."""
        total_calls = sum(self._counts.values())
        successful_calls = sum(
            count for tool, count in self._counts.items() if not tool.startswith("error:")
        )

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 1.0,
            "by_tool": self._counts.copy(),
            "budget_remaining": budget_remaining,
            "budget_used": budget_used,
        }


__all__ = ["ToolUsageStats"]
