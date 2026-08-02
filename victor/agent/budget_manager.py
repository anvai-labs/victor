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

"""Unified Budget Manager.

This module provides centralized budget management with consistent multiplier
composition across all budget types.

Design:
- Single source of truth for all budget tracking
- Consistent multiplier composition: effective_max = base × model × mode × productivity
- Separate tracking for exploration vs action operations
- Mode-specific early exit criteria for graceful completion
- Integration with UnifiedTaskTracker and prompt builder

Refactored to use composition with specialized components:
- BudgetTracker: Budget consumption and state tracking
- MultiplierCalculator: Budget multiplier calculation
- ToolCallClassifier: Tool operation classification
- ModeCompletionChecker: Mode-specific completion criteria

Usage:
    manager = BudgetManager(config=BudgetConfig())
    manager.set_mode_multiplier(2.5)  # PLAN mode

    if manager.consume(BudgetType.EXPLORATION):
        # Performed exploration operation
        pass

    if manager.is_exhausted(BudgetType.EXPLORATION):
        # Force synthesis/completion
        pass

    # Check for mode-based early exit
    if manager.should_early_exit("PLAN", response_text):
        # Mode objectives met, can stop
        pass

Issue Reference: workflow-test-issues-v2.md Issue #6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from victor.agent.budget import (
    BudgetTracker,
    BudgetState,
    ModeCompletionChecker,
    ModeCompletionConfig,
    ModeObjective,
    MultiplierCalculator,
    ToolCallClassifier,
)
from victor.agent.budget.tool_call_classifier import DEFAULT_WRITE_TOOLS
from victor.agent.protocols import (
    BudgetConfig,
    BudgetStatus,
    BudgetType,
    IBudgetManager,
)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.tools.tool_names import get_canonical_name

# Type alias for backward compatibility
ModeCompletionCriteria = ModeCompletionChecker

logger = logging.getLogger(__name__)

# =============================================================================
# Backward Compatibility Exports
# =============================================================================

# Export for backward compatibility
WRITE_TOOLS: Set[str] = DEFAULT_WRITE_TOOLS


def is_write_tool(tool_name: str) -> bool:
    """Check if a tool is a write/action operation.

    Backward compatibility wrapper. Use ToolCallClassifier for new code.

    Args:
        tool_name: Name of the tool

    Returns:
        True if this is a write/modify operation
    """
    classifier = ToolCallClassifier()
    return classifier.is_write_operation(tool_name)


# =============================================================================
# Budget Manager Implementation (Refactored with Composition)
# =============================================================================


@dataclass
class BudgetManager(IBudgetManager, ModeAwareMixin):
    """Unified budget management with multiplier composition.

    Centralizes all budget tracking with consistent multiplier application:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier

    SRP Compliance: Delegates to specialized components:
    - BudgetTracker: Budget consumption and state tracking
    - MultiplierCalculator: Budget multiplier calculation
    - ToolCallClassifier: Tool operation classification
    - ModeCompletionChecker: Mode-specific completion criteria

    Attributes:
        config: Budget configuration with base values
        _tracker: Budget consumption tracking component
        _multiplier_calc: Multiplier calculation component
        _tool_classifier: Tool call classification component
    """

    config: BudgetConfig = field(default_factory=BudgetConfig)
    _tracker: Optional[BudgetTracker] = None
    _multiplier_calc: Optional[MultiplierCalculator] = None
    _tool_classifier: Optional[ToolCallClassifier] = None

    def __post_init__(self) -> None:
        """Initialize specialized components (SRP)."""
        # Initialize multiplier calculator first (tracker depends on it)
        self._multiplier_calc = MultiplierCalculator(
            model_multiplier=1.0,
            mode_multiplier=1.0,
            productivity_multiplier=1.0,
        )

        # Initialize budget tracker with multiplier calculator
        self._tracker = BudgetTracker(
            config=self.config,
            multiplier_calculator=self._multiplier_calc,
            on_exhausted=None,  # Can be set later via set_on_exhausted
        )

        # Initialize tool call classifier
        self._tool_classifier = ToolCallClassifier()

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Delegates to BudgetTracker (SRP).

        Args:
            budget_type: Type of budget to check

        Returns:
            BudgetStatus with current usage and limits
        """
        return self._tracker.get_status(budget_type)

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget for an operation.

        Delegates to BudgetTracker (SRP).

        Args:
            budget_type: Type of budget to consume
            amount: Amount to consume (default 1)

        Returns:
            True if budget was available, False if exhausted
        """
        return self._tracker.consume(budget_type, amount)

    def is_exhausted(self, budget_type: BudgetType) -> bool:
        """Check if a budget is exhausted.

        Delegates to BudgetTracker (SRP).

        Args:
            budget_type: Type of budget to check

        Returns:
            True if budget is fully consumed
        """
        return self._tracker.is_exhausted(budget_type)

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set the model-specific multiplier.

        Delegates to MultiplierCalculator (SRP).

        Model multipliers vary by model capability:
        - GPT-4o: 1.0 (baseline)
        - Claude Opus: 1.2 (more capable, fewer retries)
        - DeepSeek: 1.3 (needs more exploration)
        - Ollama local: 1.5 (needs more attempts)

        Args:
            multiplier: Model multiplier value
        """
        self._multiplier_calc.set_model_multiplier(multiplier)

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Delegates to MultiplierCalculator (SRP).

        Mode multipliers:
        - BUILD: 2.0 (reading before writing)
        - PLAN: 2.5 (thorough analysis)
        - EXPLORE: 3.0 (exploration is primary goal)

        Args:
            multiplier: Mode multiplier value
        """
        self._multiplier_calc.set_mode_multiplier(multiplier)

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Delegates to MultiplierCalculator (SRP).

        Productivity multipliers (from RL learning):
        - High productivity session: 0.8 (less budget needed)
        - Normal: 1.0
        - Low productivity: 1.2-2.0 (more attempts needed)

        Args:
            multiplier: Productivity multiplier value
        """
        self._multiplier_calc.set_productivity_multiplier(multiplier)

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budget(s) to initial state.

        Delegates to BudgetTracker (SRP).

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        self._tracker.reset(budget_type)

    def get_prompt_budget_info(self) -> Dict[str, Any]:
        """Get budget information for system prompts.

        Delegates to BudgetTracker (SRP).

        Returns:
            Dictionary with budget info for prompt building
        """
        return self._tracker.get_prompt_budget_info()

    def record_tool_call(self, tool_name: str, is_write_operation: bool = False) -> bool:
        """Record a tool call and consume appropriate budget.

        Uses ToolCallClassifier for classification (SRP/OCP).
        Automatically routes to EXPLORATION or ACTION budget.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write/modify operation
                              (if not specified, auto-detected from tool name)

        Returns:
            True if budget was available
        """
        # Auto-detect write operation if not specified
        if not is_write_operation:
            is_write_operation = self._tool_classifier.is_write_operation(tool_name)

        # Always consume from tool calls budget
        tool_available = self.consume(BudgetType.TOOL_CALLS)

        # Track in appropriate category
        if is_write_operation:
            category_available = self.consume(BudgetType.ACTION)
            budget_type = BudgetType.ACTION
        else:
            category_available = self.consume(BudgetType.EXPLORATION)
            budget_type = BudgetType.EXPLORATION

        # Update last tool
        self._tracker.update_last_tool(budget_type, tool_name)

        logger.debug(
            f"BudgetManager: tool_call={tool_name}, "
            f"is_write={is_write_operation}, "
            f"tool_available={tool_available}, "
            f"category_available={category_available}"
        )

        return tool_available and category_available

    def set_base_budget(self, budget_type: BudgetType, base: int) -> None:
        """Set the base budget for a type.

        Delegates to BudgetTracker (SRP).

        Args:
            budget_type: Type of budget to adjust
            base: New base value
        """
        self._tracker.set_base_budget(budget_type, base)

    def set_on_exhausted(self, callback: Callable[[BudgetType], None]) -> None:
        """Set callback for when a budget is exhausted.

        Delegates to BudgetTracker (SRP).

        Args:
            callback: Function called with budget type when exhausted
        """
        self._tracker.set_on_exhausted(callback)

    def update_from_mode(self) -> None:
        """Update multiplier from current mode controller.

        Convenience method that reads mode from ModeAwareMixin.
        """
        multiplier = self.exploration_multiplier
        self.set_mode_multiplier(multiplier)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about all budgets.

        Delegates to BudgetTracker (SRP).

        Returns:
            Dictionary with detailed budget state
        """
        return self._tracker.get_diagnostics()

    # =============================================================================
    # Backward Compatibility Properties
    # =============================================================================

    @property
    def _budgets(self):
        """Backward compatibility property for _budgets.

        Deprecated: Use get_status() or get_diagnostics() instead.
        """
        return self._tracker._budgets if self._tracker else {}

    @property
    def _model_multiplier(self) -> float:
        """Backward compatibility property for _model_multiplier.

        Deprecated: Multiplier is now managed by MultiplierCalculator.
        """
        return self._multiplier_calc.model_multiplier if self._multiplier_calc else 1.0

    @property
    def _mode_multiplier(self) -> float:
        """Backward compatibility property for _mode_multiplier.

        Deprecated: Multiplier is now managed by MultiplierCalculator.
        """
        return self._multiplier_calc.mode_multiplier if self._multiplier_calc else 1.0

    @property
    def _productivity_multiplier(self) -> float:
        """Backward compatibility property for _productivity_multiplier.

        Deprecated: Multiplier is now managed by MultiplierCalculator.
        """
        return self._multiplier_calc.productivity_multiplier if self._multiplier_calc else 1.0

    @property
    def _on_exhausted(self):
        """Backward compatibility property for _on_exhausted.

        Deprecated: Callback is now managed by BudgetTracker.
        """
        return self._tracker._on_exhausted if self._tracker else None

    @_on_exhausted.setter
    def _on_exhausted(self, value):
        """Backward compatibility setter for _on_exhausted.

        Deprecated: Use set_on_exhausted() instead.
        """
        if self._tracker:
            self._tracker._on_exhausted = value

    def _calculate_effective_max(self, budget_type: BudgetType) -> int:
        """Calculate effective maximum (backward compatibility).

        Deprecated: Use get_status(budget_type).effective_maximum instead.

        Args:
            budget_type: Type of budget

        Returns:
            Effective maximum after multipliers
        """
        status = self.get_status(budget_type)
        return status.effective_maximum


# =============================================================================
# Extended Budget Manager with Mode Completion
# =============================================================================


@dataclass
class ExtendedBudgetManager(BudgetManager):
    """BudgetManager with integrated mode completion criteria.

    Extends BudgetManager to include mode-specific early exit detection,
    providing a unified interface for both budget tracking and mode
    completion checking.

    SRP Compliance: Uses ModeCompletionChecker for mode completion logic.

    Attributes:
        _mode_checker: Mode completion criteria checker
        _files_read: Count of files read in current session
        _files_written: Count of files written in current session
        _current_mode: Current operating mode
    """

    _mode_checker: Optional[ModeCompletionChecker] = None
    _files_read: int = 0
    _files_written: int = 0
    _current_mode: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize with mode criteria checker (SRP)."""
        super().__post_init__()
        if self._mode_checker is None:
            self._mode_checker = ModeCompletionChecker()

    def set_mode(self, mode: str) -> None:
        """Set current operating mode.

        Args:
            mode: Mode name (EXPLORE, PLAN, BUILD)
        """
        self._current_mode = mode.upper()
        logger.debug(f"ExtendedBudgetManager: mode set to {self._current_mode}")

    def record_file_read(self) -> None:
        """Record a file read operation."""
        self._files_read += 1

    def record_file_write(self) -> None:
        """Record a file write operation."""
        self._files_written += 1

    def record_tool_call(self, tool_name: str, is_write_operation: bool = False) -> bool:
        """Record tool call with file tracking.

        Extends parent to track file read/write operations.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write operation

        Returns:
            True if budget was available
        """
        # Track file operations
        tool_lower = get_canonical_name(tool_name.lower())
        if tool_lower == "read":
            self.record_file_read()
        elif self._tool_classifier.is_write_operation(tool_name):
            self.record_file_write()

        return super().record_tool_call(tool_name, is_write_operation)

    def should_early_exit(
        self,
        response_text: str,
        mode: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if mode objectives are met for early exit.

        Delegates to ModeCompletionChecker (SRP).

        Args:
            response_text: Agent's response text to analyze
            mode: Mode to check (uses current mode if not specified)

        Returns:
            Tuple of (should_exit, reason)
        """
        check_mode = mode or self._current_mode
        if not check_mode:
            return False, "No mode set"

        if self._mode_checker is None:
            return False, "No mode criteria configured"

        # Get current iteration from exploration budget
        exploration_status = self.get_status(BudgetType.EXPLORATION)
        iterations = exploration_status.current

        return self._mode_checker.check_early_exit(
            mode=check_mode,
            files_read=self._files_read,
            files_written=self._files_written,
            iterations=iterations,
            response_text=response_text,
        )

    def get_mode_progress(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get progress towards mode completion.

        Delegates to ModeCompletionChecker (SRP).

        Args:
            mode: Mode to check (uses current mode if not specified)

        Returns:
            Progress information dictionary
        """
        check_mode = mode or self._current_mode
        if not check_mode or self._mode_checker is None:
            return {}

        progress = self._mode_checker.get_progress(check_mode)
        progress["files_read"] = self._files_read
        progress["files_written"] = self._files_written
        return progress

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budgets and file counters.

        Delegates to parent for budgets, resets local counters (SRP).

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        super().reset(budget_type)
        if budget_type is None:
            self._files_read = 0
            self._files_written = 0
            if self._mode_checker:
                self._mode_checker.reset()


# =============================================================================
# Factory Functions
# =============================================================================


def create_budget_manager(
    config: Optional[BudgetConfig] = None,
    model_multiplier: float = 1.0,
    mode_multiplier: float = 1.0,
    productivity_multiplier: float = 1.0,
) -> BudgetManager:
    """Create a configured BudgetManager instance.

    Factory function for DI registration.

    Args:
        config: Budget configuration with base values
        model_multiplier: Initial model multiplier
        mode_multiplier: Initial mode multiplier
        productivity_multiplier: Initial productivity multiplier

    Returns:
        Configured BudgetManager instance
    """
    manager = BudgetManager(config=config or BudgetConfig())
    manager.set_model_multiplier(model_multiplier)
    manager.set_mode_multiplier(mode_multiplier)
    manager.set_productivity_multiplier(productivity_multiplier)
    return manager


def create_extended_budget_manager(
    config: Optional[BudgetConfig] = None,
    mode: Optional[str] = None,
    model_multiplier: float = 1.0,
    mode_multiplier: float = 1.0,
) -> ExtendedBudgetManager:
    """Create a configured ExtendedBudgetManager with mode completion support.

    Factory function for DI registration.

    Args:
        config: Budget configuration with base values
        mode: Initial operating mode (EXPLORE, PLAN, BUILD)
        model_multiplier: Initial model multiplier
        mode_multiplier: Initial mode multiplier

    Returns:
        Configured ExtendedBudgetManager instance
    """
    manager = ExtendedBudgetManager(config=config or BudgetConfig())
    manager.set_model_multiplier(model_multiplier)
    manager.set_mode_multiplier(mode_multiplier)
    if mode:
        manager.set_mode(mode)
    return manager


def create_mode_completion_criteria(
    custom_criteria: Optional[Dict[str, ModeCompletionConfig]] = None,
) -> ModeCompletionCriteria:
    """Create a ModeCompletionCriteria instance.

    Factory function for DI registration.

    Args:
        custom_criteria: Override default criteria for specific modes

    Returns:
        Configured ModeCompletionCriteria instance
    """
    return ModeCompletionCriteria(custom_criteria=custom_criteria)
