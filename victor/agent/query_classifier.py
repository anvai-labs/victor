"""Query classifier for agentic execution quality.

Classifies user queries by type and complexity to drive planning,
sub-agent delegation, continuation budgets, and task-aware prompting.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from victor.core.task.protocols import TaskClassification, TaskComplexity

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Semantic type of the user's query."""

    QUICK_QUESTION = "quick_question"
    EXPLORATION = "exploration"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    DEBUGGING = "debugging"


# Budget hints per query type
_BUDGET_HINTS = {
    QueryType.QUICK_QUESTION: 2,
    QueryType.EXPLORATION: 8,
    QueryType.IMPLEMENTATION: 6,
    QueryType.REVIEW: 5,
    QueryType.DEBUGGING: 6,
}

# Patterns: list of (compiled regex, QueryType) in priority order
_QUERY_PATTERNS: List[Tuple[re.Pattern, QueryType]] = [
    (
        re.compile(
            r"\b(what\s+is|how\s+does|explain|describe|tell\s+me\s+about|who\s+is|where\s+is|when\s+did)\b",
            re.IGNORECASE,
        ),
        QueryType.QUICK_QUESTION,
    ),
    (
        re.compile(
            r"\b(explore|map|find\s+all|survey|list\s+all|trace|walk\s+through|scan)\b",
            re.IGNORECASE,
        ),
        QueryType.EXPLORATION,
    ),
    (
        re.compile(
            r"\b(fix|debug|error|bug|crash|exception|traceback|failing|broken|not\s+working)\b",
            re.IGNORECASE,
        ),
        QueryType.DEBUGGING,
    ),
    (
        re.compile(r"\b(review|evaluate|check|assess|audit|inspect|examine)\b", re.IGNORECASE),
        QueryType.REVIEW,
    ),
    (
        re.compile(
            r"\b(implement|create|add|build|write|generate|make|develop|refactor)\b",
            re.IGNORECASE,
        ),
        QueryType.IMPLEMENTATION,
    ),
]


@dataclass
class QueryClassification:
    """Result of classifying a user query."""

    query_type: QueryType
    complexity: TaskComplexity
    should_plan: bool
    should_use_subagents: bool
    continuation_budget_hint: int
    confidence: float


class QueryClassifier:
    """Classifies user queries to drive agentic execution decisions."""

    def __init__(self, complexity_service=None):
        self._complexity_service = complexity_service

    def _get_complexity_service(self):
        if self._complexity_service is not None:
            return self._complexity_service
        from victor.agent.task_complexity import TaskComplexityService

        self._complexity_service = TaskComplexityService(use_semantic=False, use_rl=False)
        return self._complexity_service

    def classify(self, message: str) -> QueryClassification:
        """Classify a user message into a QueryClassification."""
        query_type = self._classify_query_type(message)
        task_classification = self._classify_complexity(message)
        complexity = task_classification.complexity
        confidence = task_classification.confidence

        should_plan = self._derive_should_plan(query_type, complexity)
        should_use_subagents = self._derive_should_use_subagents(query_type, complexity)
        budget_hint = _BUDGET_HINTS.get(query_type, 4)

        result = QueryClassification(
            query_type=query_type,
            complexity=complexity,
            should_plan=should_plan,
            should_use_subagents=should_use_subagents,
            continuation_budget_hint=budget_hint,
            confidence=confidence,
        )
        logger.debug("Query classified: %s", result)
        return result

    def _classify_query_type(self, message: str) -> QueryType:
        """Determine the semantic type via regex patterns."""
        from victor.core.task.direct_response import (
            classify_direct_response_prompt,
        )

        if classify_direct_response_prompt(message).is_direct_response:
            return QueryType.QUICK_QUESTION

        for pattern, query_type in _QUERY_PATTERNS:
            if pattern.search(message):
                return query_type
        return QueryType.IMPLEMENTATION  # default fallback

    def _classify_complexity(self, message: str) -> TaskClassification:
        """Delegate complexity classification to TaskComplexityService."""
        service = self._get_complexity_service()
        return service.classify(message)

    @staticmethod
    def _derive_should_plan(query_type: QueryType, complexity: TaskComplexity) -> bool:
        """Derive whether planning should be used."""
        if complexity in (TaskComplexity.COMPLEX, TaskComplexity.ANALYSIS):
            return True
        if complexity == TaskComplexity.MEDIUM and query_type == QueryType.IMPLEMENTATION:
            return True
        if query_type == QueryType.EXPLORATION:
            return True
        return False

    @staticmethod
    def _derive_should_use_subagents(query_type: QueryType, complexity: TaskComplexity) -> bool:
        """Derive whether sub-agents should be used."""
        if query_type == QueryType.EXPLORATION and complexity in (
            TaskComplexity.COMPLEX,
            TaskComplexity.ANALYSIS,
        ):
            return True
        return False
