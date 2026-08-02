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

"""Protocols for classification components.

This module defines protocols for classifier components following DIP.
Agent layer depends on these protocols, not concrete implementations.

Re-exports from framework/task/protocols.py for convenience, and adds
classification-specific protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

# Re-export core protocols from framework
from victor.framework.task.protocols import (
    TaskClassification,
    TaskClassifierProtocol,
    TaskBudgetProviderProtocol,
    TaskComplexity,
)

from victor.classification.pattern_registry import ClassificationPattern, TaskType


@dataclass
class TaskTypeResult:
    """Result of task type classification.

    Similar to TaskClassification but for fine-grained task types.
    """

    task_type: TaskType
    confidence: float
    top_matches: List[Tuple[str, float]]  # Top matching patterns with scores
    has_file_context: bool  # Whether the prompt mentions specific files
    nudge_applied: Optional[str] = None  # Name of nudge rule applied
    preprocessed_prompt: Optional[str] = None  # Preprocessed prompt used


@runtime_checkable
class SemanticClassifierProtocol(Protocol):
    """Protocol for semantic task type classification.

    Implementations use embedding similarity for classification.
    """

    def classify(self, prompt: str) -> TaskTypeResult:
        """Classify task type from prompt using semantic similarity.

        Args:
            prompt: User message to classify

        Returns:
            TaskTypeResult with task type and confidence
        """
        ...

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...


@runtime_checkable
class PatternMatcherProtocol(Protocol):
    """Protocol for pattern-based classification.

    Implementations use regex patterns for fast classification.
    """

    def match(self, text: str) -> Optional[ClassificationPattern]:
        """Find first matching pattern.

        Args:
            text: Text to match against patterns

        Returns:
            Matching ClassificationPattern or None
        """
        ...

    def match_all(self, text: str) -> List[ClassificationPattern]:
        """Find all matching patterns.

        Args:
            text: Text to match against patterns

        Returns:
            List of all matching patterns
        """
        ...


@runtime_checkable
class NudgeEngineProtocol(Protocol):
    """Protocol for post-classification edge case corrections.

    Nudge engines adjust classification results based on patterns
    that semantic classifiers struggle with.
    """

    def apply(
        self,
        prompt: str,
        embedding_result: TaskType,
        embedding_confidence: float,
        scores: Dict[TaskType, float],
    ) -> Tuple[TaskType, float, Optional[str]]:
        """Apply nudge rules to correct edge cases.

        Args:
            prompt: Original user prompt
            embedding_result: Task type from embedding classification
            embedding_confidence: Confidence from embedding classification
            scores: Scores for all task types from embedding

        Returns:
            Tuple of (final_task_type, final_confidence, nudge_rule_name)
        """
        ...


__all__ = [
    # Re-exported from framework
    "TaskClassification",
    "TaskClassifierProtocol",
    "TaskBudgetProviderProtocol",
    "TaskComplexity",
    # Classification-specific
    "TaskTypeResult",
    "SemanticClassifierProtocol",
    "PatternMatcherProtocol",
    "NudgeEngineProtocol",
    "ClassificationPattern",
    "TaskType",
]
