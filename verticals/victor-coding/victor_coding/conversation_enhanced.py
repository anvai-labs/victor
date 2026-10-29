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

"""Enhanced conversation management for victor-coding using ConversationCoordinator.

This module provides coding-specific conversation management features using
the framework's ConversationCoordinator for better context tracking and
summarization.

Design Pattern: Extension + Delegation
- Provides coding-specific conversation management
- Delegates to framework ConversationCoordinator
- Tracks coding-specific context (files edited, tests run, etc.)

Integration Point:
    Use in CodingAssistant for enhanced conversation tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor_contracts.verticals import VerticalConversationManager

logger = logging.getLogger(__name__)


@dataclass
class CodingContext:
    """Coding-specific conversation context.

    Tracks:
    - Files edited in the conversation
    - Tests run and their results
    - Git operations performed
    - Errors encountered
    - Code snippets discussed

    Attributes:
        files_edited: List of file paths edited
        files_read: List of file paths read
        tests_run: List of tests executed
        git_operations: List of git operations performed
        errors_encountered: List of errors encountered
        code_snippets: List of code snippets discussed
    """

    files_edited: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    tests_run: List[Dict[str, Any]] = field(default_factory=list)
    git_operations: List[Dict[str, Any]] = field(default_factory=list)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    code_snippets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_edited": self.files_edited,
            "files_read": self.files_read,
            "tests_run": self.tests_run,
            "git_operations": self.git_operations,
            "errors_encountered": self.errors_encountered,
            "code_snippets": self.code_snippets,
        }

    def add_file_edit(self, file_path: str, edit_type: str = "edit") -> None:
        """Record a file edit.

        Args:
            file_path: Path to the file
            edit_type: Type of edit (edit, create, delete)
        """
        self.files_edited.append(file_path)
        logger.debug(f"Recorded file edit: {file_path} ({edit_type})")

    def add_file_read(self, file_path: str) -> None:
        """Record a file read.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.files_read:
            self.files_read.append(file_path)
            logger.debug(f"Recorded file read: {file_path}")

    def add_test_run(self, test_name: str, passed: bool, duration: Optional[float] = None) -> None:
        """Record a test run.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration: Optional duration in seconds
        """
        self.tests_run.append(
            {
                "name": test_name,
                "passed": passed,
                "duration": duration,
            }
        )
        logger.debug(f"Recorded test run: {test_name} (passed={passed})")

    def add_git_operation(self, operation: str, target: Optional[str] = None) -> None:
        """Record a git operation.

        Args:
            operation: Git operation (commit, push, pull, etc.)
            target: Optional target (branch, file, etc.)
        """
        self.git_operations.append(
            {
                "operation": operation,
                "target": target,
            }
        )
        logger.debug(f"Recorded git operation: {operation} on {target}")


class EnhancedCodingConversationManager(VerticalConversationManager):
    """Coordinator-backed conversation manager with the coding context."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        super().__init__(
            CodingContext,
            "coding_context",
            "coding",
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear history; a full clear also resets the coding context."""
        super().clear_history(keep_summaries)
        if not keep_summaries:
            self._context = CodingContext()
        logger.info("Conversation history cleared")

    def get_coding_context(self) -> CodingContext:
        return self.get_context()

    def track_file_edit(self, file_path: str, edit_type: str = "edit") -> None:
        """Track a file edit operation.

        Args:
            file_path: Path to the file
            edit_type: Type of edit (edit, create, delete)
        """
        self._context.add_file_edit(file_path, edit_type)

    def track_file_read(self, file_path: str) -> None:
        """Track a file read operation.

        Args:
            file_path: Path to the file
        """
        self._context.add_file_read(file_path)

    def track_test_run(
        self, test_name: str, passed: bool, duration: Optional[float] = None
    ) -> None:
        """Track a test run.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration: Optional duration in seconds
        """
        self._context.add_test_run(test_name, passed, duration)

    def track_git_operation(self, operation: str, target: Optional[str] = None) -> None:
        """Track a git operation.

        Args:
            operation: Git operation (commit, push, pull, etc.)
            target: Optional target (branch, file, etc.)
        """
        self._context.add_git_operation(operation, target)

    # =========================================================================
    # Summarization
    # =========================================================================

    def get_coding_summary(self) -> str:
        """Get a coding-focused conversation summary.

        Returns:
            Formatted summary of coding work done
        """
        parts = []

        ctx = self._context

        # Files edited
        if ctx.files_edited:
            parts.append("## Files Modified")
            for file_path in ctx.files_edited:
                parts.append(f"- {file_path}")
            parts.append("")

        # Tests run
        if ctx.tests_run:
            passed = sum(1 for t in ctx.tests_run if t.get("passed"))
            total = len(ctx.tests_run)
            parts.append("## Tests Run")
            parts.append(f"- Results: {passed}/{total} passed")
            parts.append("")

        # Git operations
        if ctx.git_operations:
            parts.append("## Git Operations")
            for op in ctx.git_operations:
                target = f" on {op['target']}" if op.get("target") else ""
                parts.append(f"- {op['operation']}{target}")
            parts.append("")

        # Conversation stats
        stats = self._conversation_coordinator.get_stats()
        parts.append("## Conversation Stats")
        parts.append(f"- Total turns: {stats.total_turns}")
        parts.append(f"- User turns: {stats.user_turns}")
        parts.append(f"- Assistant turns: {stats.assistant_turns}")
        parts.append(f"- Tool calls: {stats.tool_calls}")

        return "\n".join(parts)

    # =========================================================================
    # Statistics and Observability
    # =========================================================================


__all__ = [
    "CodingContext",
    "EnhancedCodingConversationManager",
]
