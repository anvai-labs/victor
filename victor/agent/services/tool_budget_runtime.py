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

"""Tool budget runtime helpers for ``ToolService``."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class BudgetManager:
    """Manages executable tool budget for the current turn."""

    def __init__(self, max_budget: int = 100):
        self.max_budget = max_budget
        self.calls_made = 0

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.calls_made >= self.max_budget

    def record_usage(self, count: int = 1) -> None:
        """Record tool usage."""
        self.calls_made += count

    def get_remaining(self) -> int:
        """Get remaining budget."""
        return max(0, self.max_budget - self.calls_made)

    def reset(self) -> None:
        """Reset usage to the start of the current budget window."""
        self.calls_made = 0


class ToolBudgetRuntime:
    """Coordinates local tool budget state with an optional bound pipeline."""

    def __init__(
        self,
        budget_manager: BudgetManager,
        get_tool_pipeline: Callable[[], Optional[Any]],
    ):
        self._budget_manager = budget_manager
        self._get_tool_pipeline = get_tool_pipeline

    def get_limit(self) -> int:
        """Return the active tool budget ceiling."""
        tool_pipeline = self._get_tool_pipeline()
        if tool_pipeline is not None:
            pipeline_budget = getattr(tool_pipeline, "tool_budget", None)
            if isinstance(pipeline_budget, int):
                return max(0, pipeline_budget)
            pipeline_config = getattr(tool_pipeline, "config", None)
            budget = getattr(pipeline_config, "tool_budget", None)
            if isinstance(budget, int):
                return max(0, budget)
        return max(0, self._budget_manager.max_budget)

    def get_used(self) -> int:
        """Return the active count of tool calls already spent this turn."""
        tool_pipeline = self._get_tool_pipeline()
        if tool_pipeline is not None:
            pipeline_used = getattr(tool_pipeline, "calls_used", None)
            if isinstance(pipeline_used, int):
                return max(0, pipeline_used)
        return max(0, self._budget_manager.calls_made)

    def sync_from_runtime(self) -> None:
        """Mirror the active runtime budget into the local compatibility manager."""
        self._budget_manager.max_budget = self.get_limit()
        self._budget_manager.calls_made = self.get_used()

    def set_limit(self, budget: int) -> None:
        """Set the active tool budget ceiling across bound runtime owners."""
        if budget < 0:
            raise ValueError(f"Tool budget must be non-negative: {budget}")

        self._budget_manager.max_budget = budget

        tool_pipeline = self._get_tool_pipeline()
        if tool_pipeline is None:
            return

        set_tool_budget = getattr(tool_pipeline, "set_tool_budget", None)
        if callable(set_tool_budget):
            set_tool_budget(budget)
            return

        pipeline_config = getattr(tool_pipeline, "config", None)
        if pipeline_config is not None and hasattr(pipeline_config, "tool_budget"):
            pipeline_config.tool_budget = budget

    def consume(self, amount: int = 1) -> None:
        """Record tool usage against the active runtime budget."""
        if amount < 0:
            raise ValueError(f"Cannot consume negative budget: {amount}")

        tool_pipeline = self._get_tool_pipeline()
        if tool_pipeline is not None:
            consume_budget = getattr(tool_pipeline, "consume_budget", None)
            if callable(consume_budget):
                consume_budget(amount)
            else:
                tool_pipeline._calls_used = self.get_used() + amount

        self._budget_manager.record_usage(amount)

    def start_new_turn(self) -> None:
        """Reset per-turn budget state while preserving cumulative stats elsewhere."""
        tool_pipeline = self._get_tool_pipeline()
        if tool_pipeline is not None:
            start_new_turn = getattr(tool_pipeline, "start_new_turn", None)
            if callable(start_new_turn):
                start_new_turn()
            else:
                tool_pipeline._calls_used = 0
        self._budget_manager.reset()
        self.sync_from_runtime()

    def get_remaining(self) -> int:
        """Return remaining executable tool calls."""
        return max(0, self.get_limit() - self.get_used())

    def is_exhausted(self) -> bool:
        """Return whether no executable budget remains."""
        return self.get_remaining() <= 0

    def get_info(self) -> Dict[str, int]:
        """Return budget max/used/remaining details."""
        return {
            "max": self.get_limit(),
            "used": self.get_used(),
            "remaining": self.get_remaining(),
        }


__all__ = ["BudgetManager", "ToolBudgetRuntime"]
