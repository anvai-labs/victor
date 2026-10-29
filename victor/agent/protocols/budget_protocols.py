"""Protocol definitions for budget protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ModeControllerProtocol",
    "BudgetType",
    "BudgetStatus",
    "BudgetConfig",
    "IBudgetManager",
    "IBudgetTracker",
    "IMultiplierCalculator",
    "IModeCompletionChecker",
]


@runtime_checkable
class ModeControllerProtocol(Protocol):
    """Protocol for agent mode control.

    Controls agent modes (BUILD, PLAN, EXPLORE) that modify agent behavior
    for different operational contexts.
    """

    @property
    def current_mode(self) -> Any:
        """Get the current agent mode."""
        ...

    @property
    def config(self) -> Any:
        """Get the current mode configuration."""
        ...

    def switch_mode(self, new_mode: Any) -> bool:
        """Switch to a new mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful
        """
        ...

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed
        """
        ...

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        ...

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt text for current mode."""
        ...


class BudgetType(str, Enum):
    """Types of budgets tracked by the budget manager.

    Attributes:
        TOOL_CALLS: Total tool calls allowed per session
        ITERATIONS: Total LLM iterations allowed
        EXPLORATION: Read/search operations (counted toward exploration limit)
        ACTION: Write/modify operations (separate from exploration)
    """

    TOOL_CALLS = "tool_calls"
    ITERATIONS = "iterations"
    EXPLORATION = "exploration"
    ACTION = "action"


@dataclass
class BudgetStatus:
    """Status of a specific budget.

    Attributes:
        budget_type: Type of budget
        current: Current usage count
        base_maximum: Base maximum before multipliers
        effective_maximum: Maximum after multipliers applied
        is_exhausted: Whether budget is fully consumed
        model_multiplier: Model-specific multiplier
        mode_multiplier: Mode-specific multiplier
        productivity_multiplier: Productivity-based multiplier
    """

    budget_type: BudgetType
    current: int = 0
    base_maximum: int = 0
    effective_maximum: int = 0
    is_exhausted: bool = False
    model_multiplier: float = 1.0
    mode_multiplier: float = 1.0
    productivity_multiplier: float = 1.0

    @property
    def remaining(self) -> int:
        """Get remaining budget."""
        return max(0, self.effective_maximum - self.current)

    @property
    def utilization(self) -> float:
        """Get budget utilization as a percentage (0.0-1.0)."""
        if self.effective_maximum == 0:
            return 0.0
        return min(1.0, self.current / self.effective_maximum)


@dataclass
class BudgetConfig:
    """Configuration for budget manager.

    Attributes:
        base_tool_calls: Base tool call budget
        base_iterations: Base iteration budget
        base_exploration: Base exploration iterations
        base_action: Base action iterations
    """

    base_tool_calls: int = 30
    base_iterations: int = 50
    base_exploration: int = 8
    base_action: int = 12


@runtime_checkable
class IBudgetManager(Protocol):
    """Protocol for unified budget management.

    Centralizes all budget tracking with consistent multiplier composition:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier

    Replaces scattered budget tracking in:
    - unified_task_tracker.py: exploration_iterations, action_iterations
    - orchestrator.py: tool_budget, complexity_tool_budget
    - intelligent_prompt_builder.py: recommended_tool_budget
    """

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Args:
            budget_type: Type of budget to check

        Returns:
            BudgetStatus with current usage and limits
        """
        ...

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget for an operation.

        Args:
            budget_type: Type of budget to consume
            amount: Amount to consume (default 1)

        Returns:
            True if budget was available, False if exhausted
        """
        ...

    def is_exhausted(self, budget_type: BudgetType) -> bool:
        """Check if a budget is exhausted.

        Args:
            budget_type: Type of budget to check

        Returns:
            True if budget is fully consumed
        """
        ...

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set the model-specific multiplier.

        Model multipliers vary by model capability:
        - GPT-4o: 1.0 (baseline)
        - Claude Opus: 1.2 (more capable)
        - DeepSeek: 1.3 (needs more exploration)
        - Ollama local: 1.5 (needs more attempts)

        Args:
            multiplier: Model multiplier value
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Mode multipliers:
        - BUILD: 2.0 (reading before writing)
        - PLAN: 2.5 (thorough analysis)
        - EXPLORE: 3.0 (exploration is primary goal)

        Args:
            multiplier: Mode multiplier value
        """
        ...

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Productivity multipliers (from RL learning):
        - High productivity session: 0.8 (less budget needed)
        - Normal: 1.0
        - Low productivity: 1.2-2.0 (more attempts needed)

        Args:
            multiplier: Productivity multiplier value
        """
        ...

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budget(s) to initial state.

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        ...

    def get_prompt_budget_info(self) -> Dict[str, Any]:
        """Get budget information for system prompts.

        Returns:
            Dictionary with budget info for prompt building
        """
        ...

    def record_tool_call(self, tool_name: str, is_write_operation: bool = False) -> bool:
        """Record a tool call and consume appropriate budget.

        Automatically routes to EXPLORATION or ACTION budget based
        on whether the operation is a write operation.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write/modify operation

        Returns:
            True if budget was available
        """
        ...


class IBudgetTracker(Protocol):
    """Protocol for budget tracking.

    Defines interface for tracking and consuming budget.
    Core budget functionality, separated from other concerns.
    """

    def consume(self, budget_type: Any, amount: int) -> bool:
        """Consume from budget.

        Args:
            budget_type: Type of budget to consume from
            amount: Amount to consume

        Returns:
            True if consumption succeeded, False if exhausted
        """
        ...

    def get_status(self, budget_type: Any) -> Any:
        """Get current budget status.

        Args:
            budget_type: Type of budget to query

        Returns:
            BudgetStatus instance
        """
        ...

    def reset(self) -> None:
        """Reset all budgets."""
        ...


class IMultiplierCalculator(Protocol):
    """Protocol for budget multiplier calculation.

    Defines interface for calculating effective budget with multipliers.
    Separated from IBudgetTracker to follow ISP.
    """

    def calculate_effective_max(self, base_max: int) -> int:
        """Calculate effective maximum with multipliers.

        Args:
            base_max: Base maximum budget

        Returns:
            Effective maximum after applying multipliers
        """
        ...

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set model-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-1.5)
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set mode-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-3.0)
        """
        ...


class IModeCompletionChecker(Protocol):
    """Protocol for mode completion detection.

    Defines interface for checking if mode should complete early.
    Separated from budget tracking to follow ISP.
    """

    def should_early_exit(self, mode: str, response: str) -> Tuple[bool, str]:
        """Check if should exit mode early.

        Args:
            mode: Current mode
            response: Response to check

        Returns:
            Tuple of (should_exit, reason)
        """
        ...
