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

"""Task-related protocols for framework services.

This module defines protocols for task classification and budgeting.
These are framework-level services that can be used by any vertical.

Design Principles:
- SRP: Each protocol has a single responsibility
- DIP: Agent layer depends on protocols, not implementations
- ISP: Protocols are minimal and focused
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable


class TaskComplexity(Enum):
    """Task complexity levels for budgeting and guidance."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    GENERATION = "generation"
    ACTION = "action"
    ANALYSIS = "analysis"


@dataclass
class TaskClassification:
    """Result of task complexity classification.

    Attributes:
        complexity: The classified complexity level
        tool_budget: Recommended number of tool calls
        confidence: Classification confidence (0.0 to 1.0)
        matched_patterns: Pattern names that matched (for debugging)
    """

    complexity: TaskComplexity
    tool_budget: int
    confidence: float
    matched_patterns: List[str]

    def should_force_completion_after(self, tool_calls: int) -> bool:
        """Check if task should complete based on tool call count."""
        return tool_calls >= self.tool_budget


@runtime_checkable
class TaskClassifierProtocol(Protocol):
    """Protocol for task complexity classification.

    Implementations classify user messages into complexity levels
    and provide appropriate tool budgets.
    """

    def classify(self, message: str) -> TaskClassification:
        """Classify a message's task complexity.

        Args:
            message: User message to classify

        Returns:
            TaskClassification with complexity and budget
        """
        ...

    def get_budget(self, complexity: TaskComplexity) -> int:
        """Get tool budget for a complexity level.

        Args:
            complexity: The complexity level

        Returns:
            Recommended tool budget
        """
        ...


@runtime_checkable
class TaskBudgetProviderProtocol(Protocol):
    """Protocol for providing task budgets.

    Separate from classification for cases where only budget is needed.
    """

    def get_budget(self, complexity: TaskComplexity) -> int:
        """Get tool budget for a complexity level."""
        ...

    def update_budget(self, complexity: TaskComplexity, budget: int) -> None:
        """Update budget for a complexity level."""
        ...
