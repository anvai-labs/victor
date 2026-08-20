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

"""LLM-based task classification for accurate task type detection.

Uses the fast edge model (qwen3.5:2b) to classify task types more
accurately than keyword-based heuristics. This improves routing
decisions for both PlanningGate and ParadigmRouter.

The edge model is called with a micro-prompt to classify tasks
into standard task types with confidence scores.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Standard task types that align with framework capabilities
STANDARD_TASK_TYPES = [
    "create_simple",
    "edit",
    "debug",
    "search",
    "design",
    "analysis_deep",
    "exploration",
    "refactor",
    "test",
    "action",
    "code_generation",
    "review",
    "document",
]


@dataclass
class TaskClassification:
    """Task classification result.

    Attributes:
        task_type: Primary task type
        confidence: Confidence in classification (0-1)
        alternatives: Alternative task types with scores
        reasoning: Human-readable explanation
        latency_ms: Time taken to classify
    """

    task_type: str
    confidence: float
    alternatives: Dict[str, float]
    reasoning: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


class TaskClassifier:
    """Classifies tasks using edge model.

    Uses the fast edge model (qwen3.5:2b) to provide more accurate
    task classification than keyword-based heuristics.

    Micro-prompt (~80 tokens) ensures fast response (<150ms typically).

    Example:
        classifier = TaskClassifier()
        classification = await classifier.classify("create a new file")
        # Returns: task_type="create_simple", confidence=0.95
    """

    # Task classification micro-prompt
    CLASSIFICATION_PROMPT = """Classify this query into a task type:

Query: {query}

Valid task types: {task_types}

Respond with:
TASK_TYPE: <primary task type>
CONFIDENCE: <0.0-1.0>
ALTERNATIVES: <type1>:<score>, <type2>:<score>
REASONING: <one sentence>"""

    def __init__(
        self,
        enabled: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """Initialize the task classifier.

        Args:
            enabled: Whether the classifier is enabled
            use_cache: Whether to cache classifications
            cache_ttl: Cache time-to-live in seconds
        """
        self.enabled = enabled
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[TaskClassification, float]] = {}
        self._classification_count = 0
        self._cache_hits = 0

    async def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        valid_types: Optional[List[str]] = None,
    ) -> TaskClassification:
        """Classify query using edge model.

        Args:
            query: User's query
            context: Optional execution context
            valid_types: Optional list of valid task types

        Returns:
            TaskClassification with task type and confidence
        """
        import time

        self._classification_count += 1
        start_time = time.time()

        if not self.enabled:
            # Fallback to heuristic-based classification
            return self._heuristic_classify(query, start_time)

        # Check cache
        if self.use_cache:
            cached = self._get_from_cache(query)
            if cached:
                self._cache_hits += 1
                logger.debug("[TaskClassifier] Cache hit for query")
                return cached

        # Use edge model for classification
        try:
            classification = await self._edge_model_classify(
                query, context, valid_types, start_time
            )

            # Cache the result
            if self.use_cache:
                self._add_to_cache(query, classification)

            return classification

        except Exception as e:
            logger.warning(f"[TaskClassifier] Edge model failed: {e}, using heuristic")
            return self._heuristic_classify(query, start_time)

    def _heuristic_classify(self, query: str, start_time: float) -> TaskClassification:
        """Fallback heuristic-based task classification.

        Args:
            query: User's query
            start_time: Classification start time

        Returns:
            TaskClassification based on heuristics
        """
        import time

        query_lower = query.lower()
        task_type = "unknown"
        confidence = 0.6

        # Simple keyword-based classification
        if any(word in query_lower for word in ["create", "write", "new file", "generate"]):
            task_type = "create_simple"
            confidence = 0.7
        elif any(word in query_lower for word in ["fix", "debug", "error", "bug"]):
            task_type = "debug"
            confidence = 0.7
        elif any(word in query_lower for word in ["edit", "change", "modify", "update"]):
            task_type = "edit"
            confidence = 0.7
        elif any(word in query_lower for word in ["find", "search", "look for", "locate"]):
            task_type = "search"
            confidence = 0.7
        elif any(word in query_lower for word in ["run", "execute", "list", "show"]):
            task_type = "action"
            confidence = 0.7
        elif any(word in query_lower for word in ["design", "architecture"]):
            task_type = "design"
            confidence = 0.7
        elif any(word in query_lower for word in ["analyze", "understand", "review"]):
            task_type = "analysis_deep"
            confidence = 0.6

        latency_ms = (time.time() - start_time) * 1000

        return TaskClassification(
            task_type=task_type,
            confidence=confidence,
            alternatives={},
            reasoning="Heuristic-based classification",
            latency_ms=latency_ms,
        )

    async def _edge_model_classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        valid_types: Optional[List[str]],
        start_time: float,
    ) -> TaskClassification:
        """Use edge model to classify task.

        Args:
            query: User's query
            context: Optional execution context
            valid_types: Optional list of valid task types
            start_time: Classification start time

        Returns:
            TaskClassification from edge model
        """
        import time

        # Import here to avoid circular dependency
        from victor.framework.agentic_loop import decide_sync

        # Prepare valid types list
        types = valid_types or STANDARD_TASK_TYPES
        types_str = ", ".join(types)

        # Prepare prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            query=query,
            task_types=types_str,
        )

        # Call edge model
        try:
            response = decide_sync(
                decision_type="task_classification",
                context={"query": query, "prompt": prompt, "valid_types": types},
            )

            # Parse response
            task_type = self._extract_task_type(response, types)
            confidence = self._extract_confidence(response)
            alternatives = self._extract_alternatives(response, types)
            reasoning = self._extract_reasoning(response)

            latency_ms = (time.time() - start_time) * 1000

            classification = TaskClassification(
                task_type=task_type,
                confidence=confidence,
                alternatives=alternatives,
                reasoning=reasoning,
                latency_ms=latency_ms,
            )

            logger.info(
                f"[TaskClassifier] Edge model: task_type={task_type}, "
                f"confidence={confidence:.2f}, latency={latency_ms:.1f}ms"
            )

            return classification

        except Exception as e:
            logger.error(f"[TaskClassifier] Edge model error: {e}")
            raise

    def _extract_task_type(self, response: str, valid_types: List[str]) -> str:
        """Extract task type from response."""
        import re

        match = re.search(r"TASK_TYPE:\s*(\w+)", response, re.IGNORECASE)
        if match:
            task_type = match.group(1).lower()
            # Validate against valid types
            if task_type in valid_types:
                return task_type
        return "unknown"  # Default if not found or invalid

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence from response."""
        import re

        match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.8  # Default

    def _extract_alternatives(self, response: str, valid_types: List[str]) -> Dict[str, float]:
        """Extract alternative task types from response."""
        import re

        alternatives = {}
        match = re.search(r"ALTERNATIVES:\s*(.+)", response, re.IGNORECASE)
        if match:
            try:
                for alt_str in match.group(1).split(","):
                    if ":" in alt_str:
                        task_type, score = alt_str.split(":", 1)
                        task_type = task_type.strip().lower()
                        score = float(score.strip())
                        if task_type in valid_types:
                            alternatives[task_type] = score
            except (ValueError, AttributeError):
                pass
        return alternatives

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        import re

        match = re.search(r"REASONING:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "No reasoning provided"

    def _get_from_cache(self, query: str) -> Optional[TaskClassification]:
        """Get classification from cache if available."""
        import time

        if query in self._cache:
            classification, timestamp = self._cache[query]
            if time.time() - timestamp < self.cache_ttl:
                return classification
            else:
                # Expired, remove from cache
                del self._cache[query]
        return None

    def _add_to_cache(self, query: str, classification: TaskClassification) -> None:
        """Add classification to cache."""
        import time

        self._cache[query] = (classification, time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics.

        Returns:
            Dict with classification counts and cache statistics
        """
        cache_hit_rate = (
            (self._cache_hits / self._classification_count)
            if self._classification_count > 0
            else 0.0
        )

        return {
            "total_classifications": self._classification_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._classification_count = 0
        self._cache_hits = 0
        self._cache.clear()


# Singleton instance
_task_classifier_instance: Optional[TaskClassifier] = None


def get_task_classifier() -> TaskClassifier:
    """Get the singleton TaskClassifier instance.

    Returns:
        TaskClassifier singleton instance
    """
    global _task_classifier_instance
    if _task_classifier_instance is None:
        _task_classifier_instance = TaskClassifier()
    return _task_classifier_instance


__all__ = [
    "TaskClassifier",
    "TaskClassification",
    "STANDARD_TASK_TYPES",
    "get_task_classifier",
]
