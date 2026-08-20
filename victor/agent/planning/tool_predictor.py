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

"""Tool Predictor with ensemble prediction methods.

Predicts which tools will be needed for a task using ensemble voting from:
1. Keyword matching (30% weight) - Fast pattern matching on task description
2. Semantic similarity (40% weight) - Embedding-based similarity
3. Co-occurrence patterns (20% weight) - Historical tool sequences
4. Success rate multiplier (10% weight) - Boost high-success tools

Usage:
    from victor.agent.planning.tool_predictor import ToolPredictor
    from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker

    cooccurrence = CooccurrenceTracker()
    predictor = ToolPredictor(cooccurrence_tracker=cooccurrence)

    predictions = predictor.predict_tools(
        task_description="Fix the authentication bug",
        current_step="exploration",
        recent_tools=[],
        task_type="bugfix",
    )
    # Returns: [ToolPrediction(tool="search", probability=0.85, ...), ...]
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolPrediction:
    """Prediction for a tool with confidence metadata.

    Attributes:
        tool_name: Name of the predicted tool
        probability: Confidence score (0.0-1.0)
        source: Which classifier(s) contributed to this prediction
        success_rate: Historical success rate for this tool
        confidence_level: HIGH, MEDIUM, or LOW based on probability
    """

    tool_name: str
    probability: float
    source: str = "ensemble"
    success_rate: float = 0.5
    confidence_level: str = "MEDIUM"

    def __post_init__(self):
        """Determine confidence level from probability."""
        if self.probability >= 0.7:
            self.confidence_level = "HIGH"
        elif self.probability >= 0.4:
            self.confidence_level = "MEDIUM"
        else:
            self.confidence_level = "LOW"

        # Clamp probability to [0, 1]
        self.probability = max(0.0, min(1.0, self.probability))


@dataclass
class ToolPredictorConfig:
    """Configuration for tool predictor.

    Attributes:
        keyword_weight: Weight for keyword matching classifier
        semantic_weight: Weight for semantic similarity classifier
        cooccurrence_weight: Weight for co-occurrence patterns
        success_weight: Weight for success rate boosting
        min_confidence: Minimum confidence threshold for predictions
        top_k: Maximum number of predictions to return
        enable_keyword_matching: Enable keyword-based prediction
        enable_semantic_matching: Enable semantic similarity prediction
        enable_cooccurrence: Enable co-occurrence pattern prediction
    """

    keyword_weight: float = 0.3
    semantic_weight: float = 0.4
    cooccurrence_weight: float = 0.2
    success_weight: float = 0.1
    min_confidence: float = 0.1
    top_k: int = 5
    enable_keyword_matching: bool = True
    enable_semantic_matching: bool = True
    enable_cooccurrence: bool = True


class ToolPredictor:
    """Predicts which tools will be needed using ensemble methods.

    Ensemble Components:
    1. Keyword Matching: Fast pattern matching on task description
    2. Semantic Similarity: Embedding-based similarity (requires embedding_fn)
    3. Co-occurrence Patterns: Historical tool sequences (requires CooccurrenceTracker)
    4. Success Rate Boosting: Boost tools with high historical success

    The predictor combines these signals using configurable weights to
    generate ranked predictions with confidence scores.
    """

    # Keyword patterns for common tool categories
    TOOL_KEYWORDS = {
        "search": [
            r"\b(search|find|locate|look for|grep)\b",
            r"\bwhere\s+is\b",
            r"\bshow\s+me\b",
        ],
        "read": [
            r"\b(read|open|view|show|display)\b",
            r"\bfile\b",
            r"\bcontent\b",
        ],
        "edit": [
            r"\b(edit|modify|change|fix|update)\b",
            r"\breplace\b",
            r"\bcorrect\b",
        ],
        "write": [
            r"\b(write|create|add|new|implement)\b",
            r"\bbuild\b",
            r"\bgenerate\b",
        ],
        "delete": [
            r"\b(delete|remove|drop|clear)\b",
            r"\bclean\b",
        ],
        "test": [
            r"\b(test|verify|check|validate)\b",
            r"\brun\s+test\b",
            r"\bunit\s+test\b",
        ],
        "run": [
            r"\b(run|execute|start|launch)\b",
            r"\bbuild\b",
            r"\bcompile\b",
        ],
        "git": [
            r"\b(commit|push|pull|clone|branch|merge)\b",
            r"\brepository\b",
            r"\bversion\s+control\b",
        ],
        "analyze": [
            r"\b(analyze|inspect|examine|review)\b",
            r"\bcheck\b",
            r"\binvestigate\b",
        ],
        "plan": [
            r"\b(plan|design|outline|breakdown)\b",
            r"\bstrategy\b",
            r"\bapproach\b",
        ],
    }

    def __init__(
        self,
        cooccurrence_tracker: Optional[Any] = None,
        config: Optional[ToolPredictorConfig] = None,
    ):
        """Initialize the tool predictor.

        Args:
            cooccurrence_tracker: Optional CooccurrenceTracker for pattern-based prediction
            config: Optional predictor configuration
        """
        self.config = config or ToolPredictorConfig()
        self._cooccurrence_tracker = cooccurrence_tracker

        # Compile keyword patterns for efficiency
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for tool, patterns in self.TOOL_KEYWORDS.items():
            self._compiled_patterns[tool] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        logger.info(
            f"ToolPredictor initialized (weights: "
            f"keyword={self.config.keyword_weight}, "
            f"semantic={self.config.semantic_weight}, "
            f"cooccurrence={self.config.cooccurrence_weight}, "
            f"success={self.config.success_weight})"
        )

    @property
    def cooccurrence_tracker(self) -> Optional[Any]:
        """Compatibility accessor for the internal co-occurrence tracker."""
        return self._cooccurrence_tracker

    def predict_tools(
        self,
        task_description: str,
        current_step: str = "exploration",
        recent_tools: Optional[List[str]] = None,
        task_type: str = "default",
        embedding_fn: Optional[Callable[[List[str], str], List[float]]] = None,
    ) -> List[ToolPrediction]:
        """Predict which tools will be needed for the task.

        Uses ensemble voting from multiple classifiers:
        1. Keyword matching on task description
        2. Semantic similarity (if embedding_fn provided)
        3. Co-occurrence patterns (if tracker available)
        4. Success rate boosting

        Args:
            task_description: Description of the task
            current_step: Current step in the workflow
            recent_tools: Tools used recently in this session
            task_type: Type of task for task-specific predictions
            embedding_fn: Optional function(texts, query) -> similarity scores

        Returns:
            List of ToolPrediction objects, sorted by probability
        """
        recent_tools = recent_tools or []

        # Collect predictions from multiple sources
        predictions: Dict[str, float] = {}

        # Source 1: Keyword matching
        if self.config.enable_keyword_matching:
            keyword_predictions = self._predict_from_keywords(task_description, current_step)
            for tool, score in keyword_predictions.items():
                predictions[tool] = predictions.get(tool, 0) + score * self.config.keyword_weight

        # Source 2: Semantic similarity
        if self.config.enable_semantic_matching and embedding_fn:
            semantic_predictions = self._predict_from_semantic(task_description, embedding_fn)
            for tool, score in semantic_predictions.items():
                predictions[tool] = predictions.get(tool, 0) + score * self.config.semantic_weight

        # Source 3: Co-occurrence patterns
        if self.config.enable_cooccurrence and self._cooccurrence_tracker:
            cooccurrence_predictions = self._predict_from_cooccurrence(recent_tools, task_type)
            for tool, score in cooccurrence_predictions.items():
                predictions[tool] = (
                    predictions.get(tool, 0) + score * self.config.cooccurrence_weight
                )

        # Apply success rate boosting
        boosted_predictions = self._apply_success_boosting(predictions, task_type)

        # Normalize probabilities
        total = sum(boosted_predictions.values())
        if total > 0:
            normalized = {
                k: v / total
                for k, v in boosted_predictions.items()
                if v / total >= self.config.min_confidence
            }
        else:
            normalized = {}

        # Sort and convert to ToolPrediction objects
        sorted_predictions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[
            : self.config.top_k
        ]

        result = []
        for tool_name, probability in sorted_predictions:
            success_rate = self._get_tool_success_rate(tool_name, task_type)
            source = self._determine_prediction_source(
                tool_name, task_description, recent_tools, task_type
            )

            result.append(
                ToolPrediction(
                    tool_name=tool_name,
                    probability=probability,
                    source=source,
                    success_rate=success_rate,
                )
            )

        logger.debug(
            f"Tool predictions: {[p.tool_name for p in result]} "
            f"(task={task_type}, step={current_step})"
        )

        return result

    def _predict_from_keywords(
        self,
        task_description: str,
        current_step: str,
    ) -> Dict[str, float]:
        """Predict tools using keyword matching.

        Args:
            task_description: Task description text
            current_step: Current workflow step

        Returns:
            Dictionary of {tool_name: score}
        """
        scores: Dict[str, float] = defaultdict(float)

        text = f"{task_description} {current_step}".lower()

        for tool, patterns in self._compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1

            if matches > 0:
                # Score based on number of matching patterns
                scores[tool] = min(matches / len(patterns), 1.0)

        return dict(scores)

    def _predict_from_semantic(
        self,
        task_description: str,
        embedding_fn: Callable[[List[str], str], List[float]],
    ) -> Dict[str, float]:
        """Predict tools using semantic similarity.

        Args:
            task_description: Task description text
            embedding_fn: Function that computes similarity scores

        Returns:
            Dictionary of {tool_name: score}
        """
        try:
            # Get all available tools
            all_tools = list(self.TOOL_KEYWORDS.keys())

            # Compute semantic similarities
            similarities = embedding_fn(all_tools, task_description)

            # Convert to dictionary
            return dict(zip(all_tools, similarities))

        except Exception as e:
            logger.warning(f"Semantic prediction failed: {e}")
            return {}

    def _predict_from_cooccurrence(
        self,
        recent_tools: List[str],
        task_type: str,
    ) -> Dict[str, float]:
        """Predict tools using co-occurrence patterns.

        Args:
            recent_tools: Tools used recently
            task_type: Type of task

        Returns:
            Dictionary of {tool_name: score}
        """
        if not self._cooccurrence_tracker or not recent_tools:
            return {}

        try:
            predictions = self._cooccurrence_tracker.predict_next_tools(
                current_tools=recent_tools,
                task_type=task_type,
                top_k=self.config.top_k,
            )

            return {p.tool_name: p.probability for p in predictions}

        except Exception as e:
            logger.warning(f"Co-occurrence prediction failed: {e}")
            return {}

    def _apply_success_boosting(
        self,
        predictions: Dict[str, float],
        task_type: str,
    ) -> Dict[str, float]:
        """Apply success rate boosting to predictions.

        Args:
            predictions: Raw prediction scores
            task_type: Task type

        Returns:
            Boosted predictions
        """
        boosted = {}

        for tool, score in predictions.items():
            success_rate = self._get_tool_success_rate(tool, task_type)

            # Boost factor: 0.8x for low success, 1.5x for high success
            boost_factor = 0.8 + (success_rate * 0.7)

            boosted[tool] = score * boost_factor

        return boosted

    def _get_tool_success_rate(self, tool: str, task_type: str) -> float:
        """Get success rate for a tool.

        Args:
            tool: Tool name
            task_type: Task type

        Returns:
            Success rate (0.0-1.0)
        """
        if self._cooccurrence_tracker:
            try:
                return self._cooccurrence_tracker._get_tool_success_rate(tool, task_type)
            except Exception:
                pass

        return 0.5  # Default if no tracker or error

    def _determine_prediction_source(
        self,
        tool_name: str,
        task_description: str,
        recent_tools: List[str],
        task_type: str,
    ) -> str:
        """Determine which classifier(s) contributed to prediction.

        Args:
            tool_name: Predicted tool
            task_description: Task description
            recent_tools: Recent tools used
            task_type: Task type

        Returns:
            Source string (e.g., "keyword+semantic", "cooccurrence", "ensemble")
        """
        sources = []

        # Check keyword contribution
        keyword_score = self._predict_from_keywords(task_description, "exploration").get(
            tool_name, 0
        )
        if keyword_score > 0:
            sources.append("keyword")

        # Check co-occurrence contribution
        if recent_tools and self._cooccurrence_tracker:
            cooccurrence_scores = self._predict_from_cooccurrence(recent_tools, task_type)
            if tool_name in cooccurrence_scores:
                sources.append("cooccurrence")

        if not sources:
            return "ensemble"
        elif len(sources) == 1:
            return sources[0]
        else:
            return "+".join(sources)

    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics.

        Returns:
            Dictionary with predictor metrics
        """
        stats = {
            "config": {
                "keyword_weight": self.config.keyword_weight,
                "semantic_weight": self.config.semantic_weight,
                "cooccurrence_weight": self.config.cooccurrence_weight,
                "success_weight": self.config.success_weight,
                "min_confidence": self.config.min_confidence,
                "top_k": self.config.top_k,
            },
            "has_cooccurrence_tracker": self._cooccurrence_tracker is not None,
            "available_tools": len(self.TOOL_KEYWORDS),
        }

        if self._cooccurrence_tracker:
            stats["cooccurrence_stats"] = self._cooccurrence_tracker.get_statistics()

        return stats


__all__ = [
    "ToolPredictor",
    "ToolPrediction",
    "ToolPredictorConfig",
]
