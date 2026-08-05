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

"""Question type classification for smarter asking_input handling.

This module classifies questions into types to determine whether Victor should:
- Auto-continue (rhetorical/politeness questions)
- Return to user (real questions requiring input)

Design Pattern: Strategy (aligns with IntentClassifier pattern)
SOLID: Single Responsibility - only classifies question types

Example:
    classifier = QuestionTypeClassifier.get_instance()
    result = classifier.classify("Should I continue with the implementation?")
    if result.question_type == QuestionType.RHETORICAL:
        # Auto-continue
    else:
        # Return to user for input
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions models may ask."""

    RHETORICAL = "rhetorical"  # Not expecting answer, politeness
    CONTINUATION = "continuation"  # "Should I continue?", "Proceed?"
    CLARIFICATION = "clarification"  # Needs user clarification
    INFORMATION = "information"  # Needs specific user data (API keys, etc.)
    UNKNOWN = "unknown"  # Could not classify


@dataclass
class QuestionClassificationResult:
    """Result of question type classification."""

    question_type: QuestionType
    confidence: float  # 0.0-1.0
    matched_pattern: Optional[str] = None  # Which pattern matched

    @property
    def should_auto_continue(self) -> bool:
        """Whether Victor should auto-continue based on question type."""
        return (
            self.question_type
            in (
                QuestionType.RHETORICAL,
                QuestionType.CONTINUATION,
            )
            and self.confidence >= 0.6
        )


# Compiled patterns for efficient matching
# Pattern format: (regex, question_type, confidence)
QUESTION_PATTERNS: List[Tuple[re.Pattern, QuestionType, float]] = [
    # RHETORICAL/CONTINUATION - auto-continue
    (
        re.compile(r"\bshould\s+i\s+(?:continue|proceed|go\s+ahead)\b", re.IGNORECASE),
        QuestionType.CONTINUATION,
        0.95,
    ),
    (
        re.compile(
            r"\bwould\s+you\s+like\s+(?:me\s+to|to)\s+(?:continue|proceed)\b",
            re.IGNORECASE,
        ),
        QuestionType.CONTINUATION,
        0.95,
    ),
    (
        re.compile(r"\bshall\s+i\s+(?:continue|proceed|go\s+on)\b", re.IGNORECASE),
        QuestionType.CONTINUATION,
        0.95,
    ),
    (
        re.compile(r"\bdo\s+you\s+want\s+me\s+to\s+(?:continue|proceed)\b", re.IGNORECASE),
        QuestionType.CONTINUATION,
        0.90,
    ),
    (
        re.compile(r"\bready\s+(?:to|for)\s+(?:continue|proceed|move\s+on)\?", re.IGNORECASE),
        QuestionType.CONTINUATION,
        0.85,
    ),
    (
        re.compile(
            r"\blet\s+me\s+know\s+(?:if|when)\s+(?:you(?:'re|\s+are)\s+)?ready\b",
            re.IGNORECASE,
        ),
        QuestionType.CONTINUATION,
        0.80,
    ),
    (
        re.compile(
            r"\bany(?:thing)?\s+else\s+(?:you(?:'d|\s+would)\s+like|to\s+add)\?",
            re.IGNORECASE,
        ),
        QuestionType.RHETORICAL,
        0.80,
    ),
    (
        re.compile(
            r"\bdoes\s+(?:this|that)\s+(?:look|sound)\s+(?:good|right|ok)\?",
            re.IGNORECASE,
        ),
        QuestionType.RHETORICAL,
        0.85,
    ),
    (
        re.compile(
            r"\bis\s+(?:this|that)\s+what\s+you\s+(?:wanted|meant|had\s+in\s+mind)\?",
            re.IGNORECASE,
        ),
        QuestionType.RHETORICAL,
        0.80,
    ),
    (re.compile(r"\bmake\s+sense\?", re.IGNORECASE), QuestionType.RHETORICAL, 0.85),
    # CLARIFICATION - needs user input
    (
        re.compile(r"\bwhich\s+(?:one|option|approach|method|file|directory)\b", re.IGNORECASE),
        QuestionType.CLARIFICATION,
        0.90,
    ),
    (
        re.compile(r"\bwhat\s+(?:should|would)\s+(?:i|we|you)\s+(?:name|call)\b", re.IGNORECASE),
        QuestionType.CLARIFICATION,
        0.85,
    ),
    (
        re.compile(
            r"\bhow\s+(?:should|would)\s+you\s+like\s+(?:me|this|it)\s+to\b",
            re.IGNORECASE,
        ),
        QuestionType.CLARIFICATION,
        0.85,
    ),
    (
        re.compile(
            r"\bwhere\s+(?:should|would)\s+(?:i|we)\s+(?:put|place|store|save)\b",
            re.IGNORECASE,
        ),
        QuestionType.CLARIFICATION,
        0.85,
    ),
    (
        re.compile(
            r"\bdo\s+you\s+(?:prefer|want)\s+(?:me\s+to\s+use|to\s+use|using)\b",
            re.IGNORECASE,
        ),
        QuestionType.CLARIFICATION,
        0.80,
    ),
    # INFORMATION - needs specific user data
    (
        re.compile(
            r"\bwhat(?:'s|\s+is)\s+(?:your|the)\s+(?:api|access|secret)\s*(?:key|token)\b",
            re.IGNORECASE,
        ),
        QuestionType.INFORMATION,
        0.95,
    ),
    (
        re.compile(
            r"\bwhat(?:'s|\s+is)\s+(?:your|the)\s+(?:username|password|credentials)\b",
            re.IGNORECASE,
        ),
        QuestionType.INFORMATION,
        0.95,
    ),
    (
        re.compile(
            r"\bwhat(?:'s|\s+is)\s+(?:your|the)\s+(?:database|db|server)\s+(?:url|host|connection)\b",
            re.IGNORECASE,
        ),
        QuestionType.INFORMATION,
        0.90,
    ),
    (
        re.compile(
            r"\bwhat(?:'s|\s+is)\s+(?:your|the)\s+(?:email|phone|contact)\b",
            re.IGNORECASE,
        ),
        QuestionType.INFORMATION,
        0.85,
    ),
    (
        re.compile(r"\bcan\s+you\s+(?:provide|share|give\s+me)\s+(?:your|the)\b", re.IGNORECASE),
        QuestionType.INFORMATION,
        0.80,
    ),
    (
        re.compile(
            r"\bwhat\s+(?:version|framework|library)\s+(?:are\s+you|should\s+(?:i|we))\s+us(?:e|ing)\b",
            re.IGNORECASE,
        ),
        QuestionType.CLARIFICATION,
        0.80,
    ),
]


class QuestionTypeClassifier:
    """Classifies questions to determine if user input is needed.

    Uses heuristic pattern matching similar to IntentClassifier.
    Implements Singleton pattern for efficiency.

    SOLID Principles:
    - Single Responsibility: Only classifies question types
    - Open/Closed: New patterns can be added without modifying class
    - Dependency Inversion: Uses abstract QuestionType enum
    """

    _instance: Optional["QuestionTypeClassifier"] = None

    def __init__(self, decision_service=None):
        """Initialize the classifier.

        Args:
            decision_service: Optional LLMDecisionService for low-confidence augmentation
        """
        self._patterns = QUESTION_PATTERNS
        self._decision_service = decision_service
        logger.debug(f"QuestionTypeClassifier initialized with {len(self._patterns)} patterns")

    @classmethod
    def get_instance(cls) -> "QuestionTypeClassifier":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None

    def classify(self, text: str) -> QuestionClassificationResult:
        """Classify a question from model response.

        Args:
            text: The model's response text (may contain questions)

        Returns:
            QuestionClassificationResult with type and confidence
        """
        if not text or not text.strip():
            return QuestionClassificationResult(
                question_type=QuestionType.UNKNOWN,
                confidence=0.0,
            )

        # Check if text even contains a question
        if "?" not in text:
            return QuestionClassificationResult(
                question_type=QuestionType.UNKNOWN,
                confidence=0.0,
                matched_pattern="no_question_mark",
            )

        # Try pattern matching
        best_match: Optional[Tuple[QuestionType, float, str]] = None

        for pattern, q_type, confidence in self._patterns:
            if pattern.search(text):
                if best_match is None or confidence > best_match[1]:
                    best_match = (q_type, confidence, pattern.pattern)

        if best_match:
            logger.debug(
                f"Question classified as {best_match[0].value} "
                f"(confidence: {best_match[1]:.2f})"
            )
            return QuestionClassificationResult(
                question_type=best_match[0],
                confidence=best_match[1],
                matched_pattern=best_match[2][:50],  # Truncate for logging
            )

        # No pattern matched - consult LLM if available
        if self._decision_service is not None:
            try:
                from victor.agent.decisions.schemas import DecisionType

                decision = self._decision_service.decide_sync(
                    DecisionType.QUESTION_CLASSIFICATION,
                    context={"question_text": text[-300:]},
                    heuristic_confidence=0.5,
                )
                if decision.source == "llm" and hasattr(decision.result, "question_type"):
                    type_map = {
                        "rhetorical": QuestionType.RHETORICAL,
                        "continuation": QuestionType.CONTINUATION,
                        "clarification": QuestionType.CLARIFICATION,
                        "info": QuestionType.INFORMATION,
                    }
                    mapped = type_map.get(decision.result.question_type)
                    if mapped is not None:
                        logger.debug(
                            "LLM classified question as %s (conf=%.2f)",
                            mapped.value,
                            decision.confidence,
                        )
                        return QuestionClassificationResult(
                            question_type=mapped,
                            confidence=decision.confidence,
                            matched_pattern="llm_classification",
                        )
            except Exception:
                logger.debug("LLM question classification failed", exc_info=True)

        # Default to CLARIFICATION if contains question
        # This is conservative - prefer returning to user
        return QuestionClassificationResult(
            question_type=QuestionType.CLARIFICATION,
            confidence=0.5,
            matched_pattern="default_question",
        )

    def should_auto_continue(self, text: str) -> bool:
        """Quick check if response should trigger auto-continue.

        Args:
            text: Model response text

        Returns:
            True if should auto-continue, False if needs user input
        """
        result = self.classify(text)
        return result.should_auto_continue


# Module-level convenience functions
def classify_question(text: str) -> QuestionClassificationResult:
    """Convenience function to classify a question."""
    return QuestionTypeClassifier.get_instance().classify(text)


def should_auto_continue(text: str) -> bool:
    """Convenience function to check if should auto-continue."""
    return QuestionTypeClassifier.get_instance().should_auto_continue(text)
