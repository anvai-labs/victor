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

"""Context-Aware Keyword Detector - Task-type-specific completion signal detection.

This module provides keyword-based completion detection that is aware of:
1. Task type (different patterns for code generation vs analysis vs search)
2. Extracted requirements (validates against what was asked)
3. Response context (code blocks, structure, length)

Design Principles:
1. Task-specific patterns: Different keywords for different task types
2. Requirement-aware: Checks if requirements are addressed in response
3. Multi-signal: Combines completion indicators, code blocks, structure
4. Fallback-safe: Graceful degradation when context is missing

Based on research from:
- arXiv:2603.07379 - Agentic RAG Taxonomy (task type classification)
- Existing patterns in AgenticLoop._is_complete_response()

Example:
    from victor.framework.context_aware_keyword_detector import (
        ContextAwareKeywordDetector,
        CompletionSignal,
    )

    detector = ContextAwareKeywordDetector()

    signal = detector.detect_completion(
        response="Here's the implementation of the fix...",
        task_type=TaskType.DEBUGGING,
        requirements=[requirement_fix],
    )

    if signal.has_completion_indicator:
        print("Task complete!")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.completion_scorer import TaskType
    from victor.framework.perception_integration import Requirement


class CompletionIndicatorType(Enum):
    """Types of completion indicators."""

    EXPLICIT = "explicit"  # Explicit phrases like "task complete", "done"
    DELIVERABLE = "deliverable"  # References to deliverables ("the code above")
    SUMMARY = "summary"  # Summary phrases ("in conclusion")
    CODE = "code"  # Complete code blocks
    STRUCTURE = "structure"  # Structured response (headings, lists)
    REQUIREMENT = "requirement"  # Addresses extracted requirements


@dataclass
class CompletionSignal:
    """Signal from keyword-based completion detection.

    Attributes:
        has_completion_indicator: True if any completion signal detected
        has_complete_code: True if complete code blocks present
        has_structure: True if structured response (headings, lists)
        is_continuation_request: True if model asks for continuation
        confidence: Overall confidence score (0.0-1.0)
        evidence: List of evidence strings explaining the signal
        indicator_types: Types of indicators found
    """

    has_completion_indicator: bool
    has_complete_code: bool
    has_structure: bool
    is_continuation_request: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    indicator_types: List[CompletionIndicatorType] = field(default_factory=list)


class ContextAwareKeywordDetector:
    """Detects completion signals with awareness of task type and requirements.

    This detector improves upon the existing _is_complete_response() and
    _is_continuation_request() methods by:

    1. Using task-type-specific keyword patterns
    2. Checking if response addresses extracted requirements
    3. Providing detailed evidence for the signal
    4. Returning a structured signal object instead of boolean

    Usage:
        detector = ContextAwareKeywordDetector()

        signal = detector.detect_completion(
            response=turn_result.response,
            task_type=TaskType.CODE_GENERATION,
            requirements=perception.requirements,
        )
    """

    def __init__(self):
        """Initialize detector with task-type-specific patterns."""
        # Task-type-specific completion patterns
        self.task_completion_patterns = {
            "code_generation": [
                "here is the implementation",
                "here's the code",
                "the code above",
                "the following code",
                "implementation complete",
                "code is ready",
                "function implemented",
                "class created",
                "added the",
                "implemented the",
            ],
            "testing": [
                "tests pass",
                "all tests passing",
                "test suite complete",
                "coverage",
                "test results",
                "verified",
                "validation complete",
            ],
            "debugging": [
                "the fix is",
                "issue resolved",
                "bug fixed",
                "root cause",
                "problem solved",
                "fix applied",
                "patch implemented",
                "resolved the",
            ],
            "search": [
                "found",
                "located",
                "identified",
                "discovered",
                "the answer is",
                "results show",
                "search complete",
            ],
            "analysis": [
                "in conclusion",
                "to summarize",
                "in summary",
                "key findings",
                "analysis shows",
                "results indicate",
                "findings suggest",
            ],
            "setup": [
                "configured",
                "installed",
                "set up complete",
                "ready to use",
                "environment ready",
                "setup complete",
                "configuration done",
            ],
            "documentation": [
                "documented",
                "added comments",
                "explained",
                "description",
                "documentation complete",
            ],
            "deployment": [
                "deployed",
                "published",
                "released",
                "deployment complete",
                "live",
                "running",
            ],
        }

        # Generic completion patterns (fallback for unknown task types)
        self.generic_completion_patterns = [
            "here is",
            "here's",
            "the answer is",
            "the solution is",
            "in conclusion",
            "to summarize",
            "in summary",
            "the code above",
            "the following code",
            "as shown",
            "task complete",
            "done",
            "finished",
            "completed",
        ]

        # Continuation request patterns
        self.continuation_patterns = [
            "would you like me to",
            "should i continue",
            "do you want me to",
            "shall i proceed",
            "let me know if you'd like",
            "would you prefer i",
            "would you like me to continue",
            "should i proceed with",
        ]

    def detect_completion(
        self,
        response: str,
        task_type: Any,
        requirements: Optional[List[Any]] = None,
    ) -> CompletionSignal:
        """Detect completion signals in response.

        Args:
            response: The agent's response text
            task_type: Type of task (TaskType enum or string)
            requirements: Optional list of extracted requirements

        Returns:
            CompletionSignal with detection results
        """
        if not response:
            return CompletionSignal(
                has_completion_indicator=False,
                has_complete_code=False,
                has_structure=False,
                is_continuation_request=False,
                confidence=0.0,
                evidence=["No response provided"],
            )

        response_lower = response.lower()
        evidence = []
        indicator_types = []

        # 1. Check for continuation requests (negative signal)
        is_continuation = self._detect_continuation_request(response_lower)
        if is_continuation:
            return CompletionSignal(
                has_completion_indicator=False,
                has_complete_code=False,
                has_structure=False,
                is_continuation_request=True,
                confidence=0.3,
                evidence=["Model requested continuation"],
                indicator_types=[],
            )

        # 2. Check for task-type-specific completion patterns
        task_patterns = self._get_task_patterns(task_type)
        matched_patterns = [p for p in task_patterns if p in response_lower]
        has_task_pattern = bool(matched_patterns)
        pattern_match_count = len(matched_patterns)

        if has_task_pattern:
            patterns_str = ", ".join(f"'{p}'" for p in matched_patterns[:3])
            evidence.append(
                f"Found task-specific completion pattern for {task_type}: {patterns_str}"
            )
            indicator_types.append(CompletionIndicatorType.EXPLICIT)

        # 3. Check for deliverable references
        has_deliverable = self._detect_deliverable_references(response_lower)
        if has_deliverable:
            evidence.append("Found deliverable reference (e.g., 'the code above')")
            indicator_types.append(CompletionIndicatorType.DELIVERABLE)

        # 4. Check for summary phrases
        has_summary = self._detect_summary_phrases(response_lower)
        if has_summary:
            evidence.append("Found summary phrase (e.g., 'in conclusion')")
            indicator_types.append(CompletionIndicatorType.SUMMARY)

        # 5. Check for complete code blocks
        has_complete_code = self._detect_complete_code(response)
        if has_complete_code:
            evidence.append("Found complete code blocks (triple backticks)")
            indicator_types.append(CompletionIndicatorType.CODE)

        # 6. Check for structured response
        has_structure = self._detect_structure(response)
        if has_structure:
            evidence.append("Found structured response (headings, lists)")
            indicator_types.append(CompletionIndicatorType.STRUCTURE)

        # 7. Check if requirements are addressed
        if requirements:
            requirements_addressed = self._check_requirements_addressed(response, requirements)
            if requirements_addressed:
                evidence.append(f"Addressed {len(requirements)} requirement(s)")
                indicator_types.append(CompletionIndicatorType.REQUIREMENT)

        # 8. Calculate confidence
        confidence = self._calculate_confidence(
            has_task_pattern,
            has_deliverable,
            has_summary,
            has_complete_code,
            has_structure,
            bool(requirements),
            len(response),
            pattern_match_count,
        )

        # 9. Determine if has completion indicator
        # At least one strong signal (task pattern, deliverable, summary) OR
        # Two moderate signals (code, structure, requirements)
        strong_signals = has_task_pattern or has_deliverable or has_summary
        moderate_signals = sum(
            [
                has_complete_code,
                has_structure,
                bool(requirements),
            ]
        )

        has_completion_indicator = strong_signals or moderate_signals >= 2

        return CompletionSignal(
            has_completion_indicator=has_completion_indicator,
            has_complete_code=has_complete_code,
            has_structure=has_structure,
            is_continuation_request=False,
            confidence=confidence,
            evidence=evidence,
            indicator_types=indicator_types,
        )

    def _detect_continuation_request(self, response_lower: str) -> bool:
        """Detect if response is asking for continuation."""
        return any(pattern in response_lower for pattern in self.continuation_patterns)

    def _get_task_patterns(self, task_type: Any) -> List[str]:
        """Get completion patterns for task type."""
        # Convert task_type to string key
        if isinstance(task_type, str):
            task_key = task_type.lower()
        elif hasattr(task_type, "value"):
            task_key = str(task_type.value).lower()
        else:
            # Handle enum repr like "TaskType.TESTING" → "testing"
            raw = str(task_type).lower()
            task_key = raw.split(".")[-1]

        # Look up task-specific patterns
        if task_key in self.task_completion_patterns:
            return self.task_completion_patterns[task_key]

        # Fallback to generic patterns
        return self.generic_completion_patterns

    def _detect_deliverable_references(self, response_lower: str) -> bool:
        """Detect references to deliverables."""
        deliverable_patterns = [
            "the code above",
            "the following code",
            "as shown above",
            "as shown below",
            "the implementation above",
            "this code",
            "the fix",
            "the solution",
            "the answer",
        ]
        return any(pattern in response_lower for pattern in deliverable_patterns)

    def _detect_summary_phrases(self, response_lower: str) -> bool:
        """Detect summary phrases."""
        summary_patterns = [
            "in conclusion",
            "to summarize",
            "in summary",
            "summary of",
            "key findings",
            "results show",
            "analysis shows",
        ]
        return any(pattern in response_lower for pattern in summary_patterns)

    def _detect_complete_code(self, response: str) -> bool:
        """Detect if response has complete code blocks."""
        # Check for triple backticks (code blocks)
        if "```" not in response:
            return False

        # Count code blocks
        code_block_count = response.count("```")

        # Complete code blocks have even number of backticks (opening + closing)
        # and at least 2 (one complete block)
        return code_block_count >= 2 and code_block_count % 2 == 0

    def _detect_structure(self, response: str) -> bool:
        """Detect if response has structure (headings, lists)."""
        # Check for headings (##, ###)
        has_heading = bool(re.search(r"^#{2,}\s", response, re.MULTILINE))

        # Check for lists (1., -, *)
        has_list = any(
            marker in response for marker in ["\n1.", "\n-", "\n*", "  - ", "  * ", "  1. "]
        )

        return has_heading or has_list

    def _check_requirements_addressed(self, response: str, requirements: List[Any]) -> bool:
        """Check if response addresses extracted requirements."""
        if not requirements:
            return False

        response_lower = response.lower()
        addressed_count = 0

        for req in requirements:
            # Extract keywords from requirement description
            description = getattr(req, "description", "")
            if not description:
                continue

            # Extract meaningful words (4+ characters)
            keywords = re.findall(r"\b\w{4,}\b", description.lower())

            # Check if at least 2 keywords are present in response
            matches = sum(1 for kw in keywords if kw in response_lower)
            if matches >= 2:
                addressed_count += 1

        # Consider requirements addressed if at least 50% are addressed
        return addressed_count >= len(requirements) * 0.5

    def _calculate_confidence(
        self,
        has_task_pattern: bool,
        has_deliverable: bool,
        has_summary: bool,
        has_complete_code: bool,
        has_structure: bool,
        has_requirements: bool,
        response_length: int,
        pattern_match_count: int = 1,
    ) -> float:
        """Calculate confidence score for completion signal."""
        confidence = 0.0

        # Strong signals
        if has_task_pattern:
            # Base weight + bonus for multiple pattern matches
            confidence += 0.50
            if pattern_match_count >= 2:
                confidence += 0.25  # Two or more distinct task patterns → strong signal
        if has_deliverable:
            confidence += 0.35
        if has_summary:
            confidence += 0.25

        # Moderate signals
        if has_complete_code:
            confidence += 0.2
        if has_structure:
            confidence += 0.15
        if has_requirements:
            confidence += 0.1

        # Length bonus (substantial responses are more likely complete)
        if response_length > 1000:
            confidence += 0.1
        elif response_length > 500:
            confidence += 0.05

        # Cap at 1.0
        return min(confidence, 1.0)
