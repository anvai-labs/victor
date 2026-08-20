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

"""Default task type configurations shared across verticals.

Provides common task type hints, budgets, and patterns that are
applicable to most or all verticals. Verticals can import and extend these
rather than re-defining common patterns.
"""

from typing import Dict, Optional

# =============================================================================
# Common Task Budgets
# =============================================================================

# Default tool budgets by task type (number of tool calls allowed)
# These are reasonable defaults that can be overridden by verticals
COMMON_TASK_BUDGETS: Dict[str, int] = {
    # Quick lookup tasks
    "question": 3,
    "lookup": 3,
    "explain": 5,
    # Standard editing tasks
    "edit": 8,
    "create": 10,
    "update": 8,
    # Complex tasks
    "refactor": 15,
    "debug": 12,
    "analyze": 10,
    # Comprehensive tasks
    "implement": 20,
    "review": 15,
    "test": 12,
    # Default fallback
    "default": 10,
}


# =============================================================================
# Common Task Hints
# =============================================================================

# Task type hints for detecting task types from prompts
# Format: pattern keywords -> task_type
COMMON_TASK_HINTS: Dict[str, str] = {
    # Question/lookup patterns
    "what is": "question",
    "explain": "explain",
    "how does": "question",
    "why does": "question",
    "show me": "lookup",
    "find": "lookup",
    "search": "lookup",
    "list": "lookup",
    # Edit patterns
    "edit": "edit",
    "modify": "edit",
    "change": "edit",
    "update": "update",
    "fix": "edit",
    # Create patterns
    "create": "create",
    "add": "create",
    "write": "create",
    "implement": "implement",
    "build": "implement",
    # Refactor patterns
    "refactor": "refactor",
    "rename": "refactor",
    "extract": "refactor",
    "move": "refactor",
    "reorganize": "refactor",
    # Debug patterns
    "debug": "debug",
    "troubleshoot": "debug",
    "investigate": "debug",
    "why is": "debug",
    "error": "debug",
    # Analysis patterns
    "analyze": "analyze",
    "review": "review",
    "check": "analyze",
    "assess": "analyze",
    # Test patterns
    "test": "test",
    "verify": "test",
    "validate": "test",
}


# =============================================================================
# Lookup Functions
# =============================================================================


def get_task_budget(task_type: str, default: int = 10) -> int:
    """Get the tool budget for a task type.

    Args:
        task_type: The task type to look up
        default: Default budget if task type not found

    Returns:
        Tool budget for the task type
    """
    return COMMON_TASK_BUDGETS.get(task_type, default)


def get_task_hint(text: str) -> Optional[str]:
    """Detect task type from text using common patterns.

    Args:
        text: Text to analyze for task type hints

    Returns:
        Detected task type, or None if no match
    """
    text_lower = text.lower()

    for pattern, task_type in COMMON_TASK_HINTS.items():
        if pattern in text_lower:
            return task_type

    return None


def merge_task_budgets(
    base: Dict[str, int],
    override: Dict[str, int],
) -> Dict[str, int]:
    """Merge task budget definitions.

    Override takes precedence for same task type.

    Args:
        base: Base budget definitions
        override: Override budget definitions

    Returns:
        Merged budget definitions
    """
    result = dict(base)
    result.update(override)
    return result


def merge_task_hints(
    base: Dict[str, str],
    override: Dict[str, str],
) -> Dict[str, str]:
    """Merge task hint definitions.

    Override takes precedence for same pattern.

    Args:
        base: Base hint definitions
        override: Override hint definitions

    Returns:
        Merged hint definitions
    """
    result = dict(base)
    result.update(override)
    return result


__all__ = [
    "COMMON_TASK_BUDGETS",
    "COMMON_TASK_HINTS",
    "get_task_budget",
    "get_task_hint",
    "merge_task_budgets",
    "merge_task_hints",
]
