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

"""Protocol definitions for task completion detection.

This module provides protocol-based abstractions for task completion detection,
following the Dependency Inversion Principle (DIP) and Interface Seguration
Principle (ISP) from SOLID design principles.

Protocols allow for:
1. Runtime type checking with isinstance()
2. Multiple implementations without inheritance
3. Clear separation of concerns
4. Testability through dependency injection
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Protocol, runtime_checkable


class ResponsePhase(Enum):
    """Phases of agent response for completion detection.

    This enum helps distinguish between "thinking" (exploration/synthesis)
    and "output" (final delivery) phases of agent responses.
    """

    EXPLORATION = "exploration"  # Reading files, searching codebase
    SYNTHESIS = "synthesis"  # Summarizing, planning, preparing output
    FINAL_OUTPUT = "final_output"  # Delivering answer, completed work
    BLOCKED = "blocked"  # Cannot complete, needs user input


class CompletionConfidence(Enum):
    """Confidence levels for task completion detection.

    This enum provides a graded assessment of completion likelihood:
    - HIGH: Active signal detected (deterministic)
    - MEDIUM: File modifications + passive signal
    - LOW: Only passive phrase detected
    - NONE: No completion signals detected
    """

    HIGH = "high"  # Rare active marker detected - deterministic
    MEDIUM = "medium"  # File modifications + passive signal
    LOW = "low"  # Only passive phrase detected
    NONE = "none"  # No completion signal detected


@runtime_checkable
class TaskCompletionProtocol(Protocol):
    """Protocol for task completion detection (ISP compliance).

    This protocol defines the essential methods for task completion detection
    without being bloated (Interface Segregation Principle). Components depend
    on this abstraction rather than concrete implementations (Dependency Inversion
    Principle).

    Methods are designed for single responsibility and clear separation of concerns.
    """

    def detect_response_phase(self, response_text: str) -> ResponsePhase:
        """Detect the current phase of the agent's response.

        This helps distinguish between "thinking" (exploration/synthesis) and
        "output" (final delivery) phases.

        Args:
            response_text: The agent's response text

        Returns:
            Detected ResponsePhase
        """
        ...

    def get_completion_confidence(self) -> CompletionConfidence:
        """Get the current completion confidence level.

        This provides a graded assessment of completion likelihood:
        - HIGH: Active signal detected (deterministic)
        - MEDIUM: File modifications + passive signal
        - LOW: Only passive phrase detected
        - NONE: No completion signals detected

        Returns:
            Current CompletionConfidence level
        """
        ...

    def analyze_response(self, response_text: str) -> None:
        """Detect completion signals in response text with priority ordering.

        Priority:
        1. ACTIVE SIGNALS: Explicit completion signals instructed in system prompt
        2. PASSIVE PHRASES: Fallback detection for models ignoring instructions

        Args:
            response_text: The agent's response text
        """
        ...

    def should_stop(self) -> bool:
        """Check if task objectives are met and agent should stop.

        Uses priority-based completion detection:
        1. ACTIVE SIGNAL: Explicit completion signal detected (deterministic)
        2. FILE MODS + SIGNAL: File edits + any completion signal
        3. FILE MODS + CONTINUATION: File edits + continuation requests
        4. STANDARD: All expected deliverables met

        Returns:
            True if agent should stop, False otherwise
        """
        ...

    def reset(self) -> None:
        """Reset state for new task."""
        ...
