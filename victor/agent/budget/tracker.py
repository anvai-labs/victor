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

"""Budget consumption tracking.

This module provides BudgetTracker, which handles budget consumption
tracking and state management. Extracted from BudgetManager to follow
the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from victor.agent.protocols import (
    BudgetConfig,
    BudgetStatus,
    BudgetType,
    IBudgetTracker,
)

logger = logging.getLogger(__name__)


@dataclass
class BudgetState:
    """Internal state for a budget type.

    Attributes:
        current: Current usage count
        base_maximum: Base maximum before multipliers
        last_tool: Last tool that consumed this budget
    """

    current: int = 0
    base_maximum: int = 0
    last_tool: Optional[str] = None


class BudgetTracker(IBudgetTracker):
    """Tracks budget consumption and state.

    This class is responsible for:
    - Tracking budget consumption for all budget types
    - Getting budget status
    - Checking if budgets are exhausted
    - Resetting budgets
    - Providing diagnostic information

    SRP Compliance: Focuses only on budget tracking, delegating
    multiplier calculation, mode completion, and tool classification
    to specialized components.

    Attributes:
        _config: Budget configuration with base values
        _budgets: Internal state for each budget type
        _multiplier_calculator: Component for calculating effective maximums
        _on_exhausted: Optional callback when budget exhausted
    """

    def __init__(
        self,
        config: BudgetConfig,
        multiplier_calculator: Optional[Any] = None,
        on_exhausted: Optional[Callable[[BudgetType], None]] = None,
    ):
        """Initialize the budget tracker.

        Args:
            config: Budget configuration with base values
            multiplier_calculator: Component for calculating effective maximums
            on_exhausted: Optional callback when budget exhausted
        """
        self._config = config
        self._multiplier_calculator = multiplier_calculator
        self._on_exhausted = on_exhausted
        self._budgets: Dict[BudgetType, BudgetState] = {}
        self._initialize_budgets()

    def _initialize_budgets(self) -> None:
        """Set up initial budget states from config."""
        self._budgets = {
            BudgetType.TOOL_CALLS: BudgetState(
                current=0, base_maximum=self._config.base_tool_calls
            ),
            BudgetType.ITERATIONS: BudgetState(
                current=0, base_maximum=self._config.base_iterations
            ),
            BudgetType.EXPLORATION: BudgetState(
                current=0, base_maximum=self._config.base_exploration
            ),
            BudgetType.ACTION: BudgetState(current=0, base_maximum=self._config.base_action),
        }

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Args:
            budget_type: Type of budget to check

        Returns:
            BudgetStatus with current usage and limits
        """
        state = self._budgets.get(budget_type)
        if state is None:
            return BudgetStatus(
                budget_type=budget_type,
                is_exhausted=True,
            )

        # Get effective maximum from multiplier calculator
        if self._multiplier_calculator:
            effective_max = self._multiplier_calculator.calculate_effective_max(
                budget_type, state.base_maximum
            )
            model_multiplier = self._multiplier_calculator.model_multiplier
            mode_multiplier = self._multiplier_calculator.mode_multiplier
            productivity_multiplier = self._multiplier_calculator.productivity_multiplier
        else:
            effective_max = state.base_maximum
            model_multiplier = 1.0
            mode_multiplier = 1.0
            productivity_multiplier = 1.0

        current = state.current
        is_exhausted = current >= effective_max

        return BudgetStatus(
            budget_type=budget_type,
            current=current,
            base_maximum=state.base_maximum,
            effective_maximum=effective_max,
            is_exhausted=is_exhausted,
            model_multiplier=model_multiplier,
            mode_multiplier=mode_multiplier,
            productivity_multiplier=productivity_multiplier,
        )

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget for an operation.

        Args:
            budget_type: Type of budget to consume
            amount: Amount to consume (default 1)

        Returns:
            True if budget was available, False if exhausted
        """
        state = self._budgets.get(budget_type)
        if state is None:
            logger.warning(f"Unknown budget type: {budget_type}")
            return False

        # Get effective maximum from multiplier calculator
        if self._multiplier_calculator:
            effective_max = self._multiplier_calculator.calculate_effective_max(
                budget_type, state.base_maximum
            )
        else:
            effective_max = state.base_maximum

        was_available = state.current < effective_max

        state.current += amount

        is_now_exhausted = state.current >= effective_max
        if is_now_exhausted and was_available:
            logger.info(
                f"Budget {budget_type.value} exhausted: "
                f"{state.current}/{effective_max} (base={state.base_maximum})"
            )
            if self._on_exhausted:
                self._on_exhausted(budget_type)

        return was_available

    def is_exhausted(self, budget_type: BudgetType) -> bool:
        """Check if a budget is exhausted.

        Args:
            budget_type: Type of budget to check

        Returns:
            True if budget is fully consumed
        """
        status = self.get_status(budget_type)
        return status.is_exhausted

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budget(s) to initial state.

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        if budget_type is None:
            self._initialize_budgets()
            logger.debug("BudgetTracker: all budgets reset")
        else:
            state = self._budgets.get(budget_type)
            if state:
                state.current = 0
                state.last_tool = None
                logger.debug(f"BudgetTracker: {budget_type.value} budget reset")

    def set_base_budget(self, budget_type: BudgetType, base: int) -> None:
        """Set the base budget for a type.

        Args:
            budget_type: Type of budget to adjust
            base: New base value
        """
        state = self._budgets.get(budget_type)
        if state:
            state.base_maximum = max(1, base)
            logger.debug(f"BudgetTracker: {budget_type.value} base set to {base}")

    def set_on_exhausted(self, callback: Callable[[BudgetType], None]) -> None:
        """Set callback for when a budget is exhausted.

        Args:
            callback: Function called with budget type when exhausted
        """
        self._on_exhausted = callback

    def get_prompt_budget_info(self) -> Dict[str, Any]:
        """Get budget information for system prompts.

        Returns:
            Dictionary with budget info for prompt building
        """
        tool_status = self.get_status(BudgetType.TOOL_CALLS)
        exploration_status = self.get_status(BudgetType.EXPLORATION)
        action_status = self.get_status(BudgetType.ACTION)

        multiplier_info = {}
        if self._multiplier_calculator:
            multiplier_info = {
                "model_multiplier": self._multiplier_calculator.model_multiplier,
                "mode_multiplier": self._multiplier_calculator.mode_multiplier,
                "productivity_multiplier": self._multiplier_calculator.productivity_multiplier,
            }
        else:
            multiplier_info = {
                "model_multiplier": 1.0,
                "mode_multiplier": 1.0,
                "productivity_multiplier": 1.0,
            }

        return {
            "tool_budget": tool_status.effective_maximum,
            "tool_calls_used": tool_status.current,
            "tool_calls_remaining": tool_status.remaining,
            "exploration_budget": exploration_status.effective_maximum,
            "exploration_used": exploration_status.current,
            "exploration_remaining": exploration_status.remaining,
            "action_budget": action_status.effective_maximum,
            "action_used": action_status.current,
            **multiplier_info,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about all budgets.

        Returns:
            Dictionary with detailed budget state
        """
        diagnostics: Dict[str, Any] = {
            "budgets": {},
        }

        if self._multiplier_calculator:
            diagnostics["multipliers"] = {
                "model": self._multiplier_calculator.model_multiplier,
                "mode": self._multiplier_calculator.mode_multiplier,
                "productivity": self._multiplier_calculator.productivity_multiplier,
                "combined": (
                    self._multiplier_calculator.model_multiplier
                    * self._multiplier_calculator.mode_multiplier
                    * self._multiplier_calculator.productivity_multiplier
                ),
            }
        else:
            diagnostics["multipliers"] = {
                "model": 1.0,
                "mode": 1.0,
                "productivity": 1.0,
                "combined": 1.0,
            }

        for budget_type in BudgetType:
            status = self.get_status(budget_type)
            state = self._budgets.get(budget_type)
            diagnostics["budgets"][budget_type.value] = {
                "current": status.current,
                "base_maximum": status.base_maximum,
                "effective_maximum": status.effective_maximum,
                "remaining": status.remaining,
                "utilization": f"{status.utilization:.1%}",
                "is_exhausted": status.is_exhausted,
                "last_tool": state.last_tool if state else None,
            }

        return diagnostics

    def update_last_tool(self, budget_type: BudgetType, tool_name: str) -> None:
        """Update the last tool that consumed a budget.

        Args:
            budget_type: Type of budget to update
            tool_name: Name of the tool
        """
        state = self._budgets.get(budget_type)
        if state:
            state.last_tool = tool_name
