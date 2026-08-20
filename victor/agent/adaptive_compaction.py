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

"""Adaptive compaction threshold based on conversation patterns.

Analyzes conversation patterns to dynamically adjust the compaction threshold:
- Rapid topic switches → lower threshold (35-40%)
- Deep reasoning on one topic → higher threshold (65-70%)
- Q&A style → lower threshold (35-40%)
- Multi-step problem solving → higher threshold (65-70%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConversationPattern(Enum):
    """Detected conversation patterns."""

    QA_SHORT = "qa_short"
    """Short independent Q&A questions."""

    QA_DEEP = "qa_deep"
    """In-depth questions on various topics."""

    TASK_SINGLE = "task_single"
    """Single focused task with back-and-forth."""

    TASK_MULTI = "task_multi"
    """Multi-step problem building on context."""

    TOPIC_SWITCH_RAPID = "topic_switch_rapid"
    """Frequent topic changes."""

    TOPIC_STABLE_LONG = "topic_stable_long"
    """Extended discussion on same topic."""

    MIXED = "mixed"
    """Alternating patterns."""


@dataclass
class PatternAnalysis:
    """Result of conversation pattern analysis."""

    pattern: ConversationPattern
    """Primary detected pattern."""

    confidence: float
    """Confidence score (0-1)."""

    topic_coherence: float
    """How coherent topics are (0-1)."""

    context_dependency: float
    """How much turns depend on previous context (0-1)."""

    topic_switches_per_turn: float
    """Average topic switches per turn."""

    recommended_threshold: float
    """Recommended compaction threshold (0-1)."""

    reasoning: str
    """Human-readable explanation."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis data."""


class AdaptiveCompactionThreshold:
    """Adaptive compaction threshold based on conversation patterns.

    Analyzes recent conversation history to detect patterns and adjust
    the compaction threshold dynamically for optimal context retention
    and token efficiency.

    Configuration:
        min_threshold: Minimum threshold (default 0.35)
        max_threshold: Maximum threshold (default 0.70)
        analysis_window: Number of recent messages to analyze (default 20)
        update_frequency: How often to re-analyze (default every 5 turns)
    """

    # Threshold ranges by pattern type
    PATTERN_THRESHOLDS = {
        ConversationPattern.QA_SHORT: (0.35, 0.40),
        ConversationPattern.QA_DEEP: (0.40, 0.50),
        ConversationPattern.TASK_SINGLE: (0.50, 0.60),
        ConversationPattern.TASK_MULTI: (0.60, 0.70),
        ConversationPattern.TOPIC_SWITCH_RAPID: (0.35, 0.40),
        ConversationPattern.TOPIC_STABLE_LONG: (0.65, 0.70),
        ConversationPattern.MIXED: (0.45, 0.55),
    }

    def __init__(
        self,
        min_threshold: float = 0.35,
        max_threshold: float = 0.70,
        analysis_window: int = 20,
        update_frequency: int = 5,
        extra_domain_keywords: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize adaptive threshold system.

        Args:
            min_threshold: Minimum threshold (for rapid topic switches)
            max_threshold: Maximum threshold (for deep reasoning)
            analysis_window: Messages to analyze for pattern detection
            update_frequency: Re-analyze every N turns
            extra_domain_keywords: Additional domain→keywords mapping contributed
                by plugins via ``DomainKeywordsProviderProtocol``. Keys that
                collide with built-in domains override the built-in keywords.
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.analysis_window = analysis_window
        self.update_frequency = update_frequency

        # Plugin-contributed domain keywords (merged into built-ins at detection time)
        self._extra_domain_keywords: Dict[str, List[str]] = dict(extra_domain_keywords or {})

        # State tracking
        self._turn_count = 0
        self._last_analysis: Optional[PatternAnalysis] = None
        self._current_threshold = 0.50  # Start at balanced default
        self._topic_history: List[str] = []

        # Statistics
        self._threshold_history: List[Tuple[datetime, float, str]] = []

    def set_extra_domain_keywords(self, extra: Dict[str, List[str]]) -> None:
        """Replace the plugin-contributed domain keyword mapping.

        Called by ``ContextCompactor.set_adaptive_threshold`` after a
        ``DomainKeywordsProviderProtocol`` is found in the capability registry.

        Args:
            extra: Domain→keywords mapping to merge over built-ins.
        """
        self._extra_domain_keywords = dict(extra)

    def calculate_threshold(
        self,
        messages: List[Any],
        force_update: bool = False,
    ) -> float:
        """Calculate adaptive threshold based on conversation pattern.

        Args:
            messages: Current conversation messages
            force_update: Force re-analysis even if update_frequency not met

        Returns:
            Recommended compaction threshold (0-1)
        """
        self._turn_count += 1

        # Only re-analyze every update_frequency turns (or if forced)
        if (
            not force_update
            and self._last_analysis is not None
            and self._turn_count % self.update_frequency != 0
        ):
            return self._current_threshold

        # Analyze conversation pattern
        analysis = self._analyze_pattern(messages)
        self._last_analysis = analysis

        # Calculate threshold based on pattern
        threshold = self._calculate_threshold_for_pattern(analysis)
        self._current_threshold = threshold

        # Log for debugging
        logger.info(
            "[AdaptiveCompaction] Pattern=%s confidence=%.2f " "threshold=%.2f (%s)",
            analysis.pattern.value,
            analysis.confidence,
            threshold,
            analysis.reasoning,
        )

        # Track history
        self._threshold_history.append((datetime.now(), threshold, analysis.pattern.value))

        return threshold

    def _analyze_pattern(self, messages: List[Any]) -> PatternAnalysis:
        """Analyze conversation to detect pattern.

        Args:
            messages: Conversation messages to analyze

        Returns:
            PatternAnalysis with detected pattern and recommendation
        """
        if len(messages) < 4:
            # Not enough data, use balanced default
            return PatternAnalysis(
                pattern=ConversationPattern.MIXED,
                confidence=0.0,
                topic_coherence=0.5,
                context_dependency=0.5,
                topic_switches_per_turn=0.0,
                recommended_threshold=0.50,
                reasoning="Insufficient conversation history, using balanced threshold",
            )

        # Extract recent messages for analysis
        recent_messages = messages[-self.analysis_window :]
        user_messages = [m for m in recent_messages if getattr(m, "role", None) == "user"]

        if len(user_messages) < 2:
            return PatternAnalysis(
                pattern=ConversationPattern.MIXED,
                confidence=0.3,
                topic_coherence=0.5,
                context_dependency=0.5,
                topic_switches_per_turn=0.0,
                recommended_threshold=0.50,
                reasoning="Limited user messages, using balanced threshold",
            )

        # Analyze patterns
        topic_coherence = self._calculate_topic_coherence(user_messages)
        context_dependency = self._calculate_context_dependency(recent_messages)
        topic_switches = self._detect_topic_switches(user_messages)
        avg_turn_length = self._calculate_average_turn_length(recent_messages)

        # Classify pattern
        pattern, confidence = self._classify_pattern(
            topic_coherence,
            context_dependency,
            topic_switches,
            avg_turn_length,
        )

        # Calculate recommended threshold
        threshold_range = self.PATTERN_THRESHOLDS[pattern]
        # Adjust within range based on confidence
        threshold = threshold_range[0] + (threshold_range[1] - threshold_range[0]) * confidence

        # Generate reasoning
        reasoning = self._generate_reasoning(
            pattern,
            topic_coherence,
            context_dependency,
            topic_switches,
            avg_turn_length,
        )

        return PatternAnalysis(
            pattern=pattern,
            confidence=confidence,
            topic_coherence=topic_coherence,
            context_dependency=context_dependency,
            topic_switches_per_turn=(topic_switches / len(user_messages) if user_messages else 0),
            recommended_threshold=threshold,
            reasoning=reasoning,
            metadata={
                "avg_turn_length": avg_turn_length,
                "topic_switch_count": topic_switches,
                "user_message_count": len(user_messages),
            },
        )

    def _calculate_topic_coherence(self, user_messages: List[Any]) -> float:
        """Calculate how coherent topics are across conversation.

        Uses simple keyword overlap and semantic cues.
        Higher = more coherent (same topic throughout).
        Lower = topic switching.

        Args:
            user_messages: List of user messages

        Returns:
            Coherence score (0-1)
        """
        if len(user_messages) < 2:
            return 0.5

        # Extract keywords from each message
        keywords_list = []
        for msg in user_messages:
            content = getattr(msg, "content", "")
            words = set(content.lower().split())
            # Remove common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "what",
                "how",
                "can",
                "could",
                "would",
                "should",
                "why",
                "where",
                "when",
                "who",
                "which",
                "that",
                "this",
                "it",
                "to",
                "for",
                "in",
                "on",
                "at",
                "by",
                "with",
                "from",
                "of",
                "and",
                "or",
                "but",
                "so",
                "if",
                "then",
                "than",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "again",
                "further",
                "once",
                "here",
                "there",
                "all",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "too",
                "very",
                "just",
            }
            keywords = words - stop_words
            keywords_list.append(keywords)

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(keywords_list) - 1):
            words1 = keywords_list[i]
            words2 = keywords_list[i + 1]

            if not words1 or not words2:
                similarities.append(0.0)
                continue

            overlap = len(words1 & words2)
            union = len(words1 | words2)
            similarity = overlap / union if union > 0 else 0.0
            similarities.append(similarity)

        # Average similarity across consecutive turns
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.5

        return avg_similarity

    def _calculate_context_dependency(self, messages: List[Any]) -> float:
        """Calculate how much each turn depends on previous context.

        Uses linguistic markers of reference to previous content:
        - "it", "that", "the above", "previous", "earlier"
        - Reference back to code, files, concepts mentioned before

        Args:
            messages: All messages in conversation

        Returns:
            Dependency score (0-1)
        """
        if len(messages) < 4:
            return 0.5

        # Focus on assistant responses (they show if they understood context)
        assistant_messages = [m for m in messages if getattr(m, "role", None) == "assistant"]

        reference_markers = [
            "mentioned",
            "earlier",
            "above",
            "previous",
            "that",
            "it",
            "the file",
            "the code",
            "as you said",
            "you asked",
            "regarding",
            "continuing",
            "building on",
            "following up",
            "as discussed",
            "based on",
            "referring to",
            "going back",
        ]

        dependent_count = 0
        total_count = len(assistant_messages)

        for msg in assistant_messages:
            content = getattr(msg, "content", "").lower()
            if any(marker in content for marker in reference_markers):
                dependent_count += 1

        dependency_ratio = dependent_count / total_count if total_count > 0 else 0.5

        return dependency_ratio

    def _detect_topic_switches(self, user_messages: List[Any]) -> int:
        """Count number of topic switches in conversation.

        A topic switch is detected when:
        - Semantic similarity drops below threshold
        - Different domain keywords appear (e.g., from python to javascript)

        Args:
            user_messages: List of user messages

        Returns:
            Number of topic switches detected
        """
        if len(user_messages) < 2:
            return 0

        # Built-in domain keywords for common programming topics.
        # Plugin-contributed keywords (self._extra_domain_keywords) are merged in
        # below so verticals can extend vocabulary without modifying core code.
        domain_keywords = {
            "python": ["python", "pip", "django", "flask", "pandas", "numpy"],
            "javascript": [
                "javascript",
                "js",
                "node",
                "react",
                "vue",
                "angular",
                "typescript",
            ],
            "rust": ["rust", "cargo", "crate", "borrow", "lifetime"],
            "go": ["go", "golang", "goroutine", "channel"],
            "database": ["sql", "database", "query", "table", "index", "migration"],
            "web": ["html", "css", "http", "api", "endpoint", "request"],
            "testing": ["test", "mock", "assert", "fixture", "spec"],
            "devops": ["deploy", "docker", "kubernetes", "ci", "cd", "pipeline"],
        }
        if self._extra_domain_keywords:
            domain_keywords.update(self._extra_domain_keywords)

        switches = 0
        previous_domain = None

        for msg in user_messages:
            content = getattr(msg, "content", "").lower()

            # Detect current domain
            current_domain = None
            max_matches = 0
            for domain, keywords in domain_keywords.items():
                matches = sum(1 for kw in keywords if kw in content)
                if matches > max_matches:
                    max_matches = matches
                    current_domain = domain

            # Check if domain switched
            if (
                previous_domain is not None
                and current_domain is not None
                and current_domain != previous_domain
            ):
                switches += 1

            previous_domain = current_domain

        return switches

    def _calculate_average_turn_length(self, messages: List[Any]) -> float:
        """Calculate average turn length in characters.

        Args:
            messages: Conversation messages

        Returns:
            Average characters per message
        """
        if not messages:
            return 0.0

        total_chars = sum(len(getattr(m, "content", "")) for m in messages)
        return total_chars / len(messages)

    def _classify_pattern(
        self,
        topic_coherence: float,
        context_dependency: float,
        topic_switches: int,
        avg_turn_length: float,
    ) -> Tuple[ConversationPattern, float]:
        """Classify conversation pattern based on metrics.

        Args:
            topic_coherence: How coherent topics are (0-1)
            context_dependency: How much context is referenced (0-1)
            topic_switches: Number of topic switches
            avg_turn_length: Average message length

        Returns:
            Tuple of (pattern, confidence)
        """
        # Classification rules
        if topic_switches >= 3:
            # Rapid topic switching
            return (
                ConversationPattern.TOPIC_SWITCH_RAPID,
                0.8,
            )
        elif topic_coherence > 0.7 and context_dependency > 0.6:
            # Deep, coherent discussion
            if avg_turn_length > 200:
                return (ConversationPattern.TOPIC_STABLE_LONG, 0.75)
            else:
                return (ConversationPattern.QA_DEEP, 0.70)
        elif context_dependency > 0.6:
            # Building on previous context
            return (ConversationPattern.TASK_MULTI, 0.70)
        elif topic_coherence > 0.6:
            # Single focused topic
            return (ConversationPattern.TASK_SINGLE, 0.65)
        elif avg_turn_length < 100:
            # Short questions
            return (ConversationPattern.QA_SHORT, 0.75)
        else:
            # Mixed pattern
            return (ConversationPattern.MIXED, 0.50)

    def _calculate_threshold_for_pattern(self, analysis: PatternAnalysis) -> float:
        """Calculate threshold based on pattern analysis.

        Args:
            analysis: Pattern analysis result

        Returns:
            Recommended threshold (0-1)
        """
        # Get threshold range for pattern
        range_min, range_max = self.PATTERN_THRESHOLDS.get(
            analysis.pattern,
            (0.45, 0.55),  # Default to mixed
        )

        # Apply bounds
        range_min = max(range_min, self.min_threshold)
        range_max = min(range_max, self.max_threshold)

        # Adjust within range based on confidence and coherence
        base_threshold = range_min
        adjustment = (range_max - range_min) * analysis.confidence

        # Fine-tune based on topic coherence
        if analysis.topic_coherence > 0.7:
            # Very coherent, can afford higher threshold
            adjustment += 0.05
        elif analysis.topic_coherence < 0.3:
            # Not coherent, should compact more
            adjustment -= 0.05

        threshold = base_threshold + adjustment
        return max(self.min_threshold, min(self.max_threshold, threshold))

    def _generate_reasoning(
        self,
        pattern: ConversationPattern,
        topic_coherence: float,
        context_dependency: float,
        topic_switches: int,
        avg_turn_length: float,
    ) -> str:
        """Generate human-readable explanation for threshold choice.

        Args:
            pattern: Detected pattern
            topic_coherence: Topic coherence score
            context_dependency: Context dependency score
            topic_switches: Number of topic switches
            avg_turn_length: Average message length

        Returns:
            Human-readable reasoning string
        """
        parts = []

        # Pattern description
        pattern_descriptions = {
            ConversationPattern.QA_SHORT: "Short independent Q&A questions",
            ConversationPattern.QA_DEEP: "In-depth questions on various topics",
            ConversationPattern.TASK_SINGLE: "Single focused task",
            ConversationPattern.TASK_MULTI: "Multi-step problem solving",
            ConversationPattern.TOPIC_SWITCH_RAPID: f"Rapid topic switches ({topic_switches} detected)",
            ConversationPattern.TOPIC_STABLE_LONG: "Extended discussion on same topic",
            ConversationPattern.MIXED: "Mixed conversation patterns",
        }
        parts.append(f"Pattern: {pattern_descriptions.get(pattern, pattern.value)}")

        # Coherence explanation
        if topic_coherence > 0.7:
            parts.append("High topic coherence (stable discussion)")
        elif topic_coherence < 0.3:
            parts.append("Low topic coherence (frequent changes)")

        # Context dependency explanation
        if context_dependency > 0.6:
            parts.append("High context dependency (building on previous turns)")

        # Length explanation
        if avg_turn_length < 100:
            parts.append("Short turns (Q&A style)")
        elif avg_turn_length > 300:
            parts.append("Long turns (detailed discussion)")

        return ", ".join(parts)

    def get_current_analysis(self) -> Optional[PatternAnalysis]:
        """Get the most recent pattern analysis.

        Returns:
            PatternAnalysis or None if not yet analyzed
        """
        return self._last_analysis

    def get_threshold_history(self) -> List[Dict[str, Any]]:
        """Get history of threshold calculations.

        Returns:
            List of dicts with timestamp, threshold, and pattern
        """
        return [
            {
                "timestamp": ts.isoformat(),
                "threshold": threshold,
                "pattern": pattern,
            }
            for ts, threshold, pattern in self._threshold_history
        ]

    def reset(self) -> None:
        """Reset adaptive state."""
        self._turn_count = 0
        self._last_analysis = None
        self._current_threshold = 0.50
        self._topic_history = []
        self._threshold_history = []
