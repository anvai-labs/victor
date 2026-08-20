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

"""Budget management components.

This package provides focused components for budget management,
extracted from BudgetManager to follow the Single Responsibility
Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

from victor.agent.budget.mode_completion_checker import (
    ModeCompletionChecker,
    ModeCompletionConfig,
    ModeObjective,
)
from victor.agent.budget.multiplier_calculator import MultiplierCalculator
from victor.agent.budget.tool_call_classifier import ToolCallClassifier
from victor.agent.budget.tracker import BudgetState, BudgetTracker

__all__ = [
    "BudgetTracker",
    "BudgetState",
    "MultiplierCalculator",
    "ToolCallClassifier",
    "ModeCompletionChecker",
    "ModeCompletionConfig",
    "ModeObjective",
]
