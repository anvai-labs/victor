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

"""Budget multiplier calculation.

This module provides MultiplierCalculator, which handles budget multiplier
calculation and management. Extracted from BudgetManager to follow the
Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Optional

from victor.agent.protocols import BudgetType, IMultiplierCalculator

logger = logging.getLogger(__name__)


class MultiplierCalculator(IMultiplierCalculator):
    """Calculates and manages budget multipliers.

    This class is responsible for:
    - Managing model, mode, and productivity multipliers
    - Calculating effective maximum with multipliers applied
    - Updating multipliers based on mode or model changes

    SRP Compliance: Focuses only on multiplier calculation, delegating
    budget tracking, mode completion, and tool classification to
    specialized components.

    Attributes:
        _model_multiplier: Model-specific multiplier (1.0-1.5)
        _mode_multiplier: Mode-specific multiplier (1.0-3.0)
        _productivity_multiplier: Productivity-based multiplier (0.8-2.0)
    """

    def __init__(
        self,
        model_multiplier: float = 1.0,
        mode_multiplier: float = 1.0,
        productivity_multiplier: float = 1.0,
    ):
        """Initialize the multiplier calculator.

        Args:
            model_multiplier: Initial model multiplier (default 1.0)
            mode_multiplier: Initial mode multiplier (default 1.0)
            productivity_multiplier: Initial productivity multiplier (default 1.0)
        """
        self._model_multiplier = model_multiplier
        self._mode_multiplier = mode_multiplier
        self._productivity_multiplier = productivity_multiplier

    def calculate_effective_max(
        self,
        budget_type: BudgetType,
        base_max: int,
    ) -> int:
        """Calculate effective maximum with all multipliers applied.

        Formula: effective_max = base × model × mode × productivity

        Args:
            budget_type: Type of budget (for future extensions)
            base_max: Base maximum before multipliers

        Returns:
            Effective maximum after multipliers
        """
        combined = self._model_multiplier * self._mode_multiplier * self._productivity_multiplier
        return max(1, int(base_max * combined))

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set the model-specific multiplier.

        Model multipliers vary by model capability:
        - GPT-4o: 1.0 (baseline)
        - Claude Opus: 1.2 (more capable, fewer retries)
        - DeepSeek: 1.3 (needs more exploration)
        - Ollama local: 1.5 (needs more attempts)

        Args:
            multiplier: Model multiplier value
        """
        old_multiplier = self._model_multiplier
        self._model_multiplier = max(0.5, min(3.0, multiplier))

        if old_multiplier != self._model_multiplier:
            logger.debug(f"MultiplierCalculator: model_multiplier={self._model_multiplier}")

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Mode multipliers:
        - BUILD: 2.0 (reading before writing)
        - PLAN: 2.5 (thorough analysis)
        - EXPLORE: 3.0 (exploration is primary goal)

        Args:
            multiplier: Mode multiplier value
        """
        old_multiplier = self._mode_multiplier
        self._mode_multiplier = max(0.5, min(5.0, multiplier))

        if old_multiplier != self._mode_multiplier:
            logger.debug(f"MultiplierCalculator: mode_multiplier={self._mode_multiplier}")

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Productivity multipliers (from RL learning):
        - High productivity session: 0.8 (less budget needed)
        - Normal: 1.0
        - Low productivity: 1.2-2.0 (more attempts needed)

        Args:
            multiplier: Productivity multiplier value
        """
        old_multiplier = self._productivity_multiplier
        self._productivity_multiplier = max(0.5, min(3.0, multiplier))

        if old_multiplier != self._productivity_multiplier:
            logger.debug(
                f"MultiplierCalculator: productivity_multiplier={self._productivity_multiplier}"
            )

    @property
    def model_multiplier(self) -> float:
        """Get current model multiplier.

        Returns:
            Current model multiplier
        """
        return self._model_multiplier

    @property
    def mode_multiplier(self) -> float:
        """Get current mode multiplier.

        Returns:
            Current mode multiplier
        """
        return self._mode_multiplier

    @property
    def productivity_multiplier(self) -> float:
        """Get current productivity multiplier.

        Returns:
            Current productivity multiplier
        """
        return self._productivity_multiplier

    def get_combined_multiplier(self) -> float:
        """Get combined multiplier.

        Returns:
            Combined multiplier (model × mode × productivity)
        """
        return self._model_multiplier * self._mode_multiplier * self._productivity_multiplier
