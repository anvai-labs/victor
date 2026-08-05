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

"""Thinking pattern detection for breaking circular reasoning loops.

This module detects repetitive thinking patterns in agent responses and
provides guidance to break out of unproductive loops.

Issue Reference: workflow-test-issues-v2.md Issue #4
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


class ThinkingCategory(str, Enum):
    """Thinking pattern category for loop detection."""

    FILE_READ = "file_read"  # Reading files
    SEARCH = "search"  # Searching for information
    ANALYSIS = "analysis"  # Analyzing code/problems
    IMPLEMENTATION = "implementation"  # Implementing solutions
    GENERAL = "general"  # General thinking


# Try to import native extensions for faster pattern detection
_NATIVE_AVAILABLE = False
_native = None

try:
    import victor_native as _native

    _NATIVE_AVAILABLE = True
    logger.debug(f"Native thinking detector loaded (v{_native.__version__})")
except ImportError:
    logger.debug("Native extensions not available, using Python thinking detector")


@dataclass
class ThinkingPattern:
    """Represents a detected thinking pattern."""

    content_hash: str
    keywords: Set[str]
    iteration: int
    timestamp: float
    length: int
    category: str = "general"


@dataclass
class PatternAnalysis:
    """Analysis result for a thinking block."""

    is_loop: bool
    similarity_score: float
    matching_patterns: int
    guidance: str
    category: str


# Common filler words to exclude from keyword extraction
STOPWORDS: frozenset = frozenset(
    {
        "let",
        "me",
        "i",
        "the",
        "a",
        "an",
        "to",
        "and",
        "of",
        "in",
        "for",
        "is",
        "it",
        "this",
        "that",
        "with",
        "be",
        "on",
        "as",
        "at",
        "by",
        "from",
        "or",
        "but",
        "not",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "now",
        "just",
        "also",
        "very",
        "well",
        "here",
        "there",
        "when",
        "where",
        "what",
        "which",
        "who",
        "how",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
    }
)

# Patterns indicating circular thinking
CIRCULAR_PATTERNS: List[re.Pattern] = [
    re.compile(r"let me (read|check|look at|examine|see) (the|this) (file|code)", re.I),
    re.compile(r"i need to (read|check|look at|examine|see)", re.I),
    re.compile(r"(first|now) let me", re.I),
    re.compile(r"let me (first|start by)", re.I),
    re.compile(r"i('ll| will) (need to|have to)", re.I),
    # DeepSeek-specific stalling patterns
    re.compile(r"let me (actually |)use the", re.I),
    re.compile(r"i('ll| will| need to) (actually |)(read|examine|check|use)", re.I),
    re.compile(r"now (let me|i('ll| will))", re.I),
    re.compile(r"i should (read|examine|check|look)", re.I),
    re.compile(r"let me (continue|proceed)", re.I),
]

# Stalling patterns - thinking without action (common in DeepSeek)
STALLING_PATTERNS: List[re.Pattern] = [
    re.compile(r"^let me\b", re.I),  # Starts with "let me"
    re.compile(r"^i('ll| will| need to| should)\b", re.I),  # Starts with intent
    re.compile(r"^now\b", re.I),  # Starts with "now"
    re.compile(r"^first\b", re.I),  # Starts with "first"
]


@runtime_checkable
class IThinkingDetector(Protocol):
    """Protocol for thinking pattern detection."""

    def record_thinking(self, content: str) -> Tuple[bool, str]:
        """Record thinking block and detect loops."""
        ...

    def reset(self) -> None:
        """Reset detector state."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        ...


class ThinkingPatternDetector:
    """Detects repetitive thinking patterns to break loops.

    Features:
    - Exact content hash detection
    - Semantic similarity via keyword overlap
    - Circular phrase pattern detection
    - Configurable thresholds

    Usage:
        detector = ThinkingPatternDetector()

        for thinking_block in agent_responses:
            is_loop, guidance = detector.record_thinking(thinking_block)
            if is_loop:
                # Inject guidance to break the loop
                agent.inject_guidance(guidance)
    """

    # Configuration
    REPETITION_THRESHOLD = 3  # Same pattern 3 times = loop
    SIMILARITY_THRESHOLD = 0.65  # 65% keyword overlap = similar
    WINDOW_SIZE = 10  # Track last 10 thinking blocks
    MIN_KEYWORD_LENGTH = 4  # Minimum word length for keywords

    # Stalling threshold - detect stalling earlier than regular loops
    STALLING_THRESHOLD = 2  # 2 consecutive stalls = trigger action

    def __init__(
        self,
        repetition_threshold: int = REPETITION_THRESHOLD,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        window_size: int = WINDOW_SIZE,
        stalling_threshold: int = STALLING_THRESHOLD,
        presentation: Optional["PresentationProtocol"] = None,
        decision_service: Optional[Any] = None,
        runtime_intelligence: Optional[Any] = None,
    ):
        """Initialize the thinking detector.

        Args:
            repetition_threshold: Count threshold for exact repetition
            similarity_threshold: Jaccard similarity threshold
            window_size: Number of recent patterns to track
            stalling_threshold: Count for consecutive stalling detection
            presentation: Optional presentation adapter for icons (creates default if None)
            decision_service: Optional LLMDecisionService for ambiguous loop detection
            runtime_intelligence: Optional canonical runtime-intelligence service
        """
        self._history: Deque[ThinkingPattern] = deque(maxlen=window_size)
        self._pattern_counts: Dict[str, int] = {}
        self._repetition_threshold = repetition_threshold
        self._similarity_threshold = similarity_threshold
        self._stalling_threshold = stalling_threshold
        self._iteration = 0
        self._consecutive_stalls = 0  # Track consecutive stalling
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence

        # Statistics
        self._total_analyzed = 0
        self._loops_detected = 0
        self._exact_matches = 0
        self._similar_matches = 0
        self._stalling_detected = 0

        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    def _has_decision_support(self) -> bool:
        """Return whether LLM-backed loop detection is available."""
        if self._runtime_intelligence is not None:
            return True
        return self._decision_service is not None

    def _decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Route decisions through the canonical runtime-intelligence service when present."""
        if self._runtime_intelligence is not None:
            return self._runtime_intelligence.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        if self._decision_service is not None:
            return self._decision_service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        return None

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract significant keywords from thinking block.

        Args:
            text: Thinking block text

        Returns:
            Set of significant keywords
        """
        # Normalize and tokenize
        text_lower = text.lower()

        # Remove punctuation and split
        words = re.findall(r"\b[a-z]+\b", text_lower)

        # Filter stopwords and short words
        keywords = {w for w in words if len(w) >= self.MIN_KEYWORD_LENGTH and w not in STOPWORDS}

        return keywords

    def _compute_similarity(self, kw1: Set[str], kw2: Set[str]) -> float:
        """Compute Jaccard similarity between keyword sets.

        Args:
            kw1: First keyword set
            kw2: Second keyword set

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not kw1 or not kw2:
            return 0.0

        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)

        return intersection / union if union > 0 else 0.0

    def _detect_circular_phrases(self, text: str) -> bool:
        """Detect circular thinking phrases.

        Uses native Aho-Corasick implementation for 6x speedup when available.

        Args:
            text: Thinking block text

        Returns:
            True if circular phrases detected
        """
        # Use native implementation when available (6x faster)
        if _NATIVE_AVAILABLE:
            return _native.detect_circular_phrases(text)

        # Python fallback
        for pattern in CIRCULAR_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _detect_stalling(self, text: str) -> bool:
        """Detect stalling patterns (intent without action).

        Common in DeepSeek where model says "Let me..." repeatedly
        without actually executing tools.

        Args:
            text: Thinking block text

        Returns:
            True if stalling pattern detected
        """
        # Normalize to first sentence/line
        first_line = text.split("\n")[0].strip()
        first_sentence = first_line.split(".")[0].strip()

        for pattern in STALLING_PATTERNS:
            if pattern.match(first_sentence):
                return True
        return False

    def _categorize_thinking(self, text: str) -> ThinkingCategory:
        """Categorize the thinking block.

        Args:
            text: Thinking block text

        Returns:
            ThinkingCategory enum value
        """
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["read", "file", "content"]):
            return ThinkingCategory.FILE_READ
        elif any(kw in text_lower for kw in ["search", "find", "look for"]):
            return ThinkingCategory.SEARCH
        elif any(kw in text_lower for kw in ["understand", "analyze", "examine"]):
            return ThinkingCategory.ANALYSIS
        elif any(kw in text_lower for kw in ["implement", "create", "write"]):
            return ThinkingCategory.IMPLEMENTATION
        else:
            return ThinkingCategory.GENERAL

    def record_thinking(self, content: str) -> Tuple[bool, str]:
        """Record thinking block and detect loops.

        Args:
            content: Thinking block content

        Returns:
            Tuple of (is_loop_detected, guidance_message)
        """
        import time

        self._total_analyzed += 1
        self._iteration += 1

        # Compute content hash
        content_normalized = " ".join(content.split())  # Normalize whitespace
        content_hash = hashlib.md5(content_normalized.encode()).hexdigest()[:12]

        # Extract keywords
        keywords = self._extract_keywords(content)

        # Create pattern record
        pattern = ThinkingPattern(
            content_hash=content_hash,
            keywords=keywords,
            iteration=self._iteration,
            timestamp=time.time(),
            length=len(content),
            category=self._categorize_thinking(content),
        )

        # Check for stalling patterns first (DeepSeek-specific)
        if self._detect_stalling(content):
            self._consecutive_stalls += 1
            if self._consecutive_stalls >= self._stalling_threshold:
                self._loops_detected += 1
                self._stalling_detected += 1
                self._history.append(pattern)

                guidance = self._generate_guidance(
                    "stalling",
                    self._consecutive_stalls,
                    pattern.category,
                )
                logger.warning(
                    f"Stalling pattern detected (count: {self._consecutive_stalls}) - "
                    "model is stating intent without executing tools"
                )
                return True, guidance
        else:
            # Reset stalling counter when we see non-stalling content
            self._consecutive_stalls = 0

        # Check for exact repetition
        self._pattern_counts[content_hash] = self._pattern_counts.get(content_hash, 0) + 1

        if self._pattern_counts[content_hash] >= self._repetition_threshold:
            self._loops_detected += 1
            self._exact_matches += 1
            self._history.append(pattern)

            guidance = self._generate_guidance(
                "exact_repetition",
                self._pattern_counts[content_hash],
                pattern.category,
            )
            logger.warning(
                f"Exact repetition loop detected (count: {self._pattern_counts[content_hash]})"
            )
            return True, guidance

        # Check for semantic similarity with recent patterns
        similar_count = 0
        max_similarity = 0.0

        for prev in self._history:
            similarity = self._compute_similarity(keywords, prev.keywords)
            max_similarity = max(max_similarity, similarity)

            if similarity >= self._similarity_threshold:
                similar_count += 1

        self._history.append(pattern)

        if similar_count >= self._repetition_threshold - 1:
            self._loops_detected += 1
            self._similar_matches += 1

            guidance = self._generate_guidance(
                "semantic_similarity",
                similar_count + 1,
                pattern.category,
                max_similarity,
            )
            logger.warning(
                f"Semantic similarity loop detected "
                f"(similar: {similar_count}, max_sim: {max_similarity:.2f})"
            )
            return True, guidance

        # Check for circular phrases
        if self._detect_circular_phrases(content) and len(self._history) > 3:
            # Only warn if we've been going for a while
            recent_circular = sum(
                1 for p in list(self._history)[-3:] if self._detect_circular_phrases(content)
            )
            if recent_circular >= 2:
                logger.debug("Circular phrases detected but not yet a loop")

        # LLM augmentation: if ambiguous (near threshold), consult LLM
        if self._has_decision_support() and len(self._history) > 2:
            try:
                from victor.agent.decisions.schemas import DecisionType

                decision = self._decide_sync(
                    DecisionType.LOOP_DETECTION,
                    context={
                        "content_excerpt": content[:300],
                        "recent_blocks": str(len(self._history)),
                    },
                    heuristic_confidence=0.4,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and decision.source == "llm"
                    and hasattr(decision.result, "is_loop")
                ):
                    if decision.result.is_loop:
                        self._loops_detected += 1
                        guidance = self._generate_guidance(
                            decision.result.loop_type,
                            1,
                            pattern.category,
                        )
                        logger.debug("LLM detected loop: %s", decision.result.loop_type)
                        return True, guidance
            except Exception:
                logger.debug("LLM loop detection failed", exc_info=True)

        return False, ""

    def _generate_guidance(
        self,
        loop_type: str,
        count: int,
        category: str,
        similarity: float = 0.0,
    ) -> str:
        """Generate guidance to break the loop.

        Args:
            loop_type: Type of loop detected
            count: Number of repetitions
            category: Thinking category
            similarity: Similarity score (for semantic loops)

        Returns:
            Guidance message string
        """
        loop_type = getattr(loop_type, "value", loop_type)
        category = getattr(category, "value", category)

        # Different base message for stalling vs repetition
        if loop_type == "stalling":
            warning_icon = self._presentation.icon("warning", with_color=False)
            base_guidance = (
                f"{warning_icon} STALLING DETECTED: "
                f"You've stated your intent {count} times without taking action. "
            )
        else:
            refresh_icon = self._presentation.icon("refresh", with_color=False)
            base_guidance = (
                f"{refresh_icon} LOOP DETECTED ({loop_type}): "
                f"You've repeated this thought pattern {count} times. "
            )

        # Category-specific advice
        category_advice = {
            "file_read": (
                "You've already read this file. Use the content you have. "
                "If you need specific information, state what you're looking for."
            ),
            "search": (
                "You've already searched for this. "
                "Either use the results you have or try a different search query."
            ),
            "analysis": (
                "You've analyzed this enough. "
                "Proceed with your current understanding and take action."
            ),
            "implementation": (
                "Stop planning and start implementing. "
                "Write the code now based on what you know."
            ),
            "general": (
                "Take a different approach or proceed with action. "
                "Repeated thinking without progress is unproductive."
            ),
        }

        # Stalling-specific advice overrides
        if loop_type == "stalling":
            stalling_advice = {
                "file_read": (
                    "STOP saying 'let me read' - EXECUTE the read tool NOW. "
                    "If you've already read the file, use that content."
                ),
                "search": (
                    "STOP saying 'let me search' - EXECUTE the search tool NOW. "
                    "State your query and run the search."
                ),
                "analysis": (
                    "STOP saying 'let me analyze' - provide your analysis NOW. "
                    "Use the information you have."
                ),
                "implementation": (
                    "STOP planning - EXECUTE the edit/write tool NOW. "
                    "Write the code based on what you know."
                ),
                "general": (
                    "STOP stating intent - TAKE ACTION NOW. "
                    "Execute a tool or provide your response."
                ),
            }
            advice = stalling_advice.get(category, stalling_advice["general"])
        else:
            advice = category_advice.get(category, category_advice["general"])

        return base_guidance + advice

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics.

        Returns:
            Dictionary with detection statistics
        """
        return {
            "total_analyzed": self._total_analyzed,
            "loops_detected": self._loops_detected,
            "exact_matches": self._exact_matches,
            "similar_matches": self._similar_matches,
            "stalling_detected": self._stalling_detected,
            "consecutive_stalls": self._consecutive_stalls,
            "detection_rate": (
                self._loops_detected / self._total_analyzed if self._total_analyzed > 0 else 0.0
            ),
            "history_size": len(self._history),
            "unique_patterns": len(self._pattern_counts),
        }

    def get_recent_patterns(self) -> List[Dict[str, Any]]:
        """Get recent pattern information.

        Returns:
            List of recent pattern info dictionaries
        """
        return [
            {
                "iteration": p.iteration,
                "category": p.category,
                "keywords_count": len(p.keywords),
                "length": p.length,
            }
            for p in self._history
        ]

    def reset(self) -> None:
        """Reset detector state for new task."""
        self._history.clear()
        self._pattern_counts.clear()
        self._iteration = 0
        self._consecutive_stalls = 0
        logger.debug("Thinking pattern detector reset")

    def clear_stats(self) -> None:
        """Clear statistics but keep pattern history."""
        self._total_analyzed = 0
        self._loops_detected = 0
        self._exact_matches = 0
        self._similar_matches = 0
        self._stalling_detected = 0


def create_thinking_detector(
    repetition_threshold: int = 3,
    similarity_threshold: float = 0.65,
    prefer_native: bool = True,
    runtime_intelligence: Optional[Any] = None,
) -> ThinkingPatternDetector:
    """Factory function for creating ThinkingPatternDetector.

    Uses native Rust implementation when available for 6x faster pattern detection.

    Args:
        repetition_threshold: Count for exact repetition detection
        similarity_threshold: Jaccard similarity threshold
        prefer_native: Use native detector when available (default True)

    Returns:
        Configured ThinkingPatternDetector instance
    """
    # Log native status
    if _NATIVE_AVAILABLE and prefer_native:
        logger.debug("Using native-accelerated thinking detector")
    else:
        logger.debug("Using Python thinking detector")

    return ThinkingPatternDetector(
        repetition_threshold=repetition_threshold,
        similarity_threshold=similarity_threshold,
        runtime_intelligence=runtime_intelligence,
    )


def is_native_thinking_detector_available() -> bool:
    """Check if native thinking detector is available.

    Returns:
        True if native extensions are loaded
    """
    return _NATIVE_AVAILABLE
