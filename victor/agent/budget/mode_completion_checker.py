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

"""Mode completion detection.

This module provides ModeCompletionChecker, which handles mode-specific
completion criteria and early exit detection. Extracted from BudgetManager
to follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.agent.protocols import IModeCompletionChecker

logger = logging.getLogger(__name__)


class ModeObjective(Enum):
    """Defines primary objectives for each mode."""

    EXPLORE = "understand_codebase"
    PLAN = "provide_implementation_plan"
    BUILD = "create_or_modify_files"


@dataclass
class ModeCompletionConfig:
    """Configuration for mode-specific completion criteria.

    Attributes:
        min_files_read: Minimum files to read before considering complete
        min_files_written: Minimum files to write (BUILD mode)
        max_iterations: Maximum iterations before forcing completion
        completion_signals: Phrases indicating task completion
        required_sections: Required sections in output (PLAN mode)
    """

    min_files_read: int = 1
    min_files_written: int = 0
    max_iterations: int = 20
    completion_signals: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)


class ModeCompletionChecker(IModeCompletionChecker):
    """Checks mode-specific completion criteria.

    This class is responsible for:
    - Defining completion criteria per mode
    - Checking if mode objectives are met for early exit
    - Tracking iteration counts per mode
    - Getting progress towards mode completion

    SRP Compliance: Focuses only on mode completion checking, delegating
    budget tracking, multiplier calculation, and tool classification to
    specialized components.

    Attributes:
        _custom_criteria: Optional custom criteria overrides
        _iteration_counts: Iteration counts per mode
    """

    # Default criteria per mode
    CRITERIA = {
        ModeObjective.EXPLORE: ModeCompletionConfig(
            min_files_read=1,
            min_files_written=0,
            max_iterations=15,
            completion_signals=[
                "here's what",
                "the file",
                "this is",
                "here's an overview",
                "this module",
                "the codebase",
                "i found",
                "the structure",
            ],
            required_sections=[],
        ),
        ModeObjective.PLAN: ModeCompletionConfig(
            min_files_read=1,
            min_files_written=0,
            max_iterations=20,
            completion_signals=[
                "implementation plan",
                "steps to",
                "here's how",
                "here's the plan",
                "proposed approach",
                "implementation steps",
                "the plan",
            ],
            required_sections=[
                "step",
                "file",
            ],
        ),
        ModeObjective.BUILD: ModeCompletionConfig(
            min_files_read=0,
            min_files_written=1,
            max_iterations=30,
            completion_signals=[
                "created",
                "implemented",
                "written",
                "has been created",
                "successfully created",
                "file created",
                "implementation complete",
            ],
            required_sections=[],
        ),
    }

    def __init__(self, custom_criteria: Optional[Dict[str, ModeCompletionConfig]] = None):
        """Initialize with optional custom criteria.

        Args:
            custom_criteria: Override default criteria for specific modes
        """
        self._custom_criteria = custom_criteria or {}
        self._iteration_counts: Dict[str, int] = {}

    def get_criteria(self, mode: str) -> ModeCompletionConfig:
        """Get completion criteria for mode.

        Args:
            mode: Mode name (EXPLORE, PLAN, BUILD)

        Returns:
            Completion configuration for the mode
        """
        # Check custom criteria first
        if mode.upper() in self._custom_criteria:
            return self._custom_criteria[mode.upper()]

        # Get from default criteria
        try:
            objective = ModeObjective[mode.upper()]
            return self.CRITERIA.get(objective, ModeCompletionConfig())
        except KeyError:
            logger.warning(f"Unknown mode: {mode}, using default criteria")
            return ModeCompletionConfig()

    def check_early_exit(
        self,
        mode: str,
        files_read: int,
        files_written: int,
        iterations: int,
        response_text: str,
    ) -> tuple[bool, str]:
        """Check if mode objectives are met for early exit.

        Args:
            mode: Current mode (EXPLORE, PLAN, BUILD)
            files_read: Number of files read so far
            files_written: Number of files written so far
            iterations: Current iteration count
            response_text: Agent's response text

        Returns:
            Tuple of (should_exit, reason)
        """
        criteria = self.get_criteria(mode)

        # Track iterations
        self._iteration_counts[mode] = iterations

        # Check maximum iterations exceeded
        if iterations >= criteria.max_iterations:
            logger.info(f"Mode {mode}: max iterations ({criteria.max_iterations}) reached")
            return True, f"Maximum iterations ({criteria.max_iterations}) reached"

        # Check minimum requirements by mode
        mode_upper = mode.upper()

        if mode_upper == "BUILD":
            # BUILD mode requires file(s) to be written
            if files_written < criteria.min_files_written:
                return (
                    False,
                    f"Need {criteria.min_files_written - files_written} more file(s) written",
                )
        else:
            # EXPLORE and PLAN require files to be read
            if files_read < criteria.min_files_read:
                return (
                    False,
                    f"Need {criteria.min_files_read - files_read} more file(s) read",
                )

        # Check for completion signals in response
        response_lower = response_text.lower()
        signals = criteria.completion_signals

        found_signal = None
        for signal in signals:
            if signal in response_lower:
                found_signal = signal
                break

        if not found_signal:
            return False, "No completion signal detected"

        # For PLAN mode, check required sections
        if mode_upper == "PLAN" and criteria.required_sections:
            missing_sections = []
            for section in criteria.required_sections:
                # Check for section headers or keywords
                if not re.search(rf"\b{section}\b", response_lower):
                    missing_sections.append(section)

            if missing_sections:
                return False, f"Missing required sections: {missing_sections}"

        reason = f"Mode objectives complete: '{found_signal}' signal detected"
        logger.info(f"Mode {mode}: early exit - {reason}")
        return True, reason

    def reset(self, mode: Optional[str] = None) -> None:
        """Reset iteration counts.

        Args:
            mode: Specific mode to reset, or None for all
        """
        if mode is None:
            self._iteration_counts.clear()
        elif mode.upper() in self._iteration_counts:
            del self._iteration_counts[mode.upper()]

    def get_progress(self, mode: str) -> Dict[str, Any]:
        """Get progress towards mode completion.

        Args:
            mode: Mode to check

        Returns:
            Progress information dictionary
        """
        criteria = self.get_criteria(mode)
        iterations = self._iteration_counts.get(mode.upper(), 0)

        return {
            "mode": mode,
            "iterations": iterations,
            "max_iterations": criteria.max_iterations,
            "progress_pct": min(100, (iterations / criteria.max_iterations) * 100),
            "min_files_read": criteria.min_files_read,
            "min_files_written": criteria.min_files_written,
        }

    def set_custom_criteria(self, mode: str, criteria: ModeCompletionConfig) -> None:
        """Set custom completion criteria for a mode.

        Args:
            mode: Mode name (EXPLORE, PLAN, BUILD)
            criteria: Custom completion configuration
        """
        self._custom_criteria[mode.upper()] = criteria
        logger.debug(f"ModeCompletionChecker: set custom criteria for {mode}")
