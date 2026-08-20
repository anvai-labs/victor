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

"""Unified task classification with ensemble voting and contextual awareness.

This module provides robust task classification by combining:
- Keyword-based classification (fast, always available)
- Negation detection (prevents false positives)
- Confidence scoring (enables ensemble voting)
- Contextual classification (uses conversation history)
- Semantic classification integration (when embeddings available)

Design Principles:
- Graceful degradation: Works without embeddings, improves with them
- Extensible: Easy to add new patterns and classifiers
- Testable: Pure functions with clear inputs/outputs
- Observable: Provides detailed classification metadata

Example Usage:
    classifier = UnifiedTaskClassifier()

    # Simple classification
    result = classifier.classify("Analyze the codebase for issues")
    print(f"Type: {result.task_type}, Confidence: {result.confidence}")

    # With context
    result = classifier.classify_with_context(
        "Continue with the review",
        history=[{"role": "user", "content": "Analyze the auth module"}]
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.embeddings.task_classifier import TaskTypeClassifier
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.services.tiered_decision_service import TieredDecisionService

# Import native availability from shared location
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# Import fuzzy matching for robust typo-tolerant classification
try:
    from victor.storage.embeddings.fuzzy_matcher import match_keywords_cascading

    _FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    _FUZZY_MATCHING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log native availability for classifier
if _NATIVE_AVAILABLE:
    version = getattr(_native, "__version__", "unknown")
    logger.debug(f"Native classifier loaded (v{version})")
else:
    logger.debug("Native extensions not available, using Python classifier")

# Cache configuration
CLASSIFICATION_CACHE_SIZE = 256  # Max cached classifications
CLASSIFICATION_CACHE_TTL = 300  # 5 minutes TTL


class ClassifierTaskType(Enum):
    """Task types for unified classification output.

    Coarse-grained types for general task classification.

    Renamed from TaskType to be semantically distinct:
    - TaskType (victor.classification.pattern_registry): Canonical prompt classification
    - TrackerTaskType: Progress tracking with milestones
    - LoopDetectorTaskType: Loop detection thresholds
    - ClassifierTaskType: Unified classification output
    - FrameworkTaskType: Framework-level task abstraction
    """

    ANALYSIS = "analysis"  # Explore, review, understand codebase
    ACTION = "action"  # Execute, run, deploy
    GENERATION = "generation"  # Create, write, generate code
    SEARCH = "search"  # Find, locate, grep
    EDIT = "edit"  # Modify, refactor, fix existing code
    DEFAULT = "default"  # Ambiguous or conversational

    def to_canonical(self) -> str:
        """Canonical task-type name in the TaskTypeRegistry (tool-supply P7).

        Collapses this adapter enum onto the single registry taxonomy; coarse values
        (analysis/generation/default) resolve through the registry's alias map.
        """
        from victor.framework.task_types import canonicalize_task_type

        return canonicalize_task_type(self.value)

    @classmethod
    def from_canonical(cls, name: str) -> "Optional[ClassifierTaskType]":
        """Best-effort reverse map: a canonical name -> the member that yields it."""
        from victor.framework.task_types import canonicalize_task_type

        target = canonicalize_task_type(name)
        for member in cls:
            if member.to_canonical() == target:
                return member
        return None


@dataclass
class KeywordMatch:
    """Details of a keyword match for debugging."""

    keyword: str
    category: str
    position: int
    negated: bool = False
    weight: float = 1.0


@dataclass
class ClassificationResult:
    """Unified classification result with confidence and metadata.

    Provides a single source of truth for task classification combining
    keyword, semantic, and contextual signals.
    """

    # Primary classification
    task_type: ClassifierTaskType
    confidence: float  # 0.0 - 1.0

    # Detailed flags (backward compatible with _classify_task_keywords)
    is_action_task: bool = False
    is_analysis_task: bool = False
    is_generation_task: bool = False
    needs_execution: bool = False

    # Classification source tracking
    source: str = "keyword"  # "keyword", "semantic", "ensemble", "context"
    keyword_confidence: float = 0.0
    semantic_confidence: float = 0.0
    context_boost: float = 0.0

    # Debugging metadata
    matched_keywords: List[KeywordMatch] = field(default_factory=list)
    negated_keywords: List[KeywordMatch] = field(default_factory=list)
    context_signals: List[str] = field(default_factory=list)

    # Actionable outputs
    recommended_tool_budget: int = 20
    temperature_adjustment: float = 0.0

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy _classify_task_keywords format for backward compatibility."""
        coarse_type = self.task_type.value
        if coarse_type == "generation":
            coarse_type = "action"  # Legacy mapping
        elif coarse_type not in ("analysis", "action"):
            coarse_type = "default"

        return {
            "is_action_task": self.is_action_task or self.task_type == ClassifierTaskType.ACTION,
            "is_analysis_task": self.is_analysis_task
            or self.task_type == ClassifierTaskType.ANALYSIS,
            "needs_execution": self.needs_execution,
            "coarse_task_type": coarse_type,
            # New fields (ignored by legacy code)
            "confidence": self.confidence,
            "source": self.source,
            "task_type": self.task_type.value,
        }


# =============================================================================
# Keyword Patterns with Weights
# =============================================================================

# Action keywords with weights (higher = stronger signal)
ACTION_KEYWORDS: List[Tuple[str, float]] = [
    ("execute", 1.0),
    ("apply", 1.0),  # Strong action signal - "apply the fix", "apply changes"
    ("run", 0.9),
    ("deploy", 1.0),
    ("build", 0.8),
    ("start", 0.7),
    ("stop", 0.7),
    ("restart", 0.8),
    ("install", 0.8),
    ("commit", 0.9),
    ("push", 0.9),
    ("pull", 0.8),
    ("merge", 0.8),
]

GENERATION_KEYWORDS: List[Tuple[str, float]] = [
    ("create", 0.9),
    ("generate", 1.0),
    ("write", 0.8),
    ("make", 0.7),
    ("implement", 0.8),
    ("add", 0.6),
    ("scaffold", 1.0),
]

ANALYSIS_KEYWORDS: List[Tuple[str, float]] = [
    ("analyze", 1.0),
    ("analysis", 1.0),
    ("review", 0.9),
    ("examine", 0.9),
    ("understand", 0.8),
    ("explore", 0.8),
    ("audit", 1.0),
    ("inspect", 0.9),
    ("investigate", 0.9),
    ("survey", 0.8),
    ("comprehensive", 0.9),
    ("thorough", 0.9),
    ("entire codebase", 1.0),
    ("all files", 0.8),
    ("full analysis", 1.0),
    ("deep dive", 0.9),
    ("architecture", 0.8),
    ("structure", 0.7),
]

ANALYSIS_QUESTION_PATTERNS: List[Tuple[str, float]] = [
    ("what are the", 0.8),
    ("how does", 0.8),
    ("how do", 0.8),
    ("explain the", 0.9),
    ("describe the", 0.8),
    ("key components", 0.8),
    ("what is the purpose", 0.8),
    ("why does", 0.7),
]

SEARCH_KEYWORDS: List[Tuple[str, float]] = [
    ("find", 0.8),
    ("search", 0.9),
    ("locate", 0.9),
    ("where is", 0.8),
    ("grep", 1.0),
    ("look for", 0.7),
    ("list", 0.7),  # "list the directory structure"
    ("show", 0.6),  # "show me the files"
    ("directory structure", 0.9),
    ("folder structure", 0.9),
    ("project structure", 0.8),
]

EDIT_KEYWORDS: List[Tuple[str, float]] = [
    ("fix", 0.9),
    ("refactor", 1.0),
    ("modify", 0.9),
    ("change", 0.7),
    ("update", 0.7),
    ("rename", 0.8),
    ("move", 0.7),
    ("delete", 0.8),
    ("remove", 0.7),
    ("address", 0.9),  # "address issues", "address the problem"
    ("resolve", 0.9),  # "resolve the issue"
    ("remediate", 0.9),  # "remediate the findings"
    ("apply", 0.8),  # "apply the fix"
]

# Execution-specific keywords (subset of action)
EXECUTION_KEYWORDS: List[Tuple[str, float]] = [
    ("execute", 1.0),
    ("run", 0.9),
    ("start", 0.7),
]

# =============================================================================
# Negation Detection Patterns
# =============================================================================

# Patterns that negate keywords (compiled for performance)
NEGATION_PATTERNS: List[re.Pattern] = [
    # "don't/do not analyze"
    re.compile(
        r"\b(don't|do\s+not|no\s+need\s+to|skip|without|avoid)\s+(\w+\s+)*",
        re.IGNORECASE,
    ),
    # "not to analyze"
    re.compile(r"\bnot\s+to\s+(\w+)", re.IGNORECASE),
    # "instead of analyzing"
    re.compile(r"\binstead\s+of\s+(\w+ing)", re.IGNORECASE),
    # "rather than analyze"
    re.compile(r"\brather\s+than\s+(\w+)", re.IGNORECASE),
    # "just X, don't Y" pattern
    re.compile(r"\bjust\s+\w+[,;]\s*(don't|not)\s+", re.IGNORECASE),
]

# Positive override patterns that cancel earlier negation
# These indicate a shift from "don't X" to "do Y" in the same sentence
POSITIVE_OVERRIDE_PATTERNS: List[re.Pattern] = [
    # "but do/but please" - signals positive intent after negation
    re.compile(r"\bbut\s+(do|please|just|actually)\b", re.IGNORECASE),
    # "just X" after comma/semicolon - "don't analyze, just run"
    re.compile(r"[,;]\s*just\s+", re.IGNORECASE),
    # "instead X" - "don't analyze, instead run"
    re.compile(r"[,;]\s*instead\s+", re.IGNORECASE),
    # "actually X" - "don't analyze, actually run"
    re.compile(r"[,;]\s*actually\s+", re.IGNORECASE),
    # "only X" - "don't analyze, only run"
    re.compile(r"[,;]\s*only\s+", re.IGNORECASE),
]

# Window size for negation detection (chars before keyword)
NEGATION_WINDOW = 30


def _has_action_keywords_fast(message: str) -> bool:
    """Fast check for action keywords using native extension.

    Used as early-exit optimization before detailed analysis.

    Args:
        message: Message to check

    Returns:
        True if action keywords are likely present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_action_keywords(message)
    # Fallback to quick string check
    message_lower = message.lower()
    quick_action = ["run", "execute", "deploy", "build", "test", "commit", "push"]
    return any(kw in message_lower for kw in quick_action)


def _has_analysis_keywords_fast(message: str) -> bool:
    """Fast check for analysis keywords using native extension.

    Args:
        message: Message to check

    Returns:
        True if analysis keywords are likely present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_analysis_keywords(message)
    message_lower = message.lower()
    quick_analysis = ["analyze", "explore", "review", "understand", "explain"]
    return any(kw in message_lower for kw in quick_analysis)


def _has_negation_fast(message: str) -> bool:
    """Fast check for negation using native extension.

    Args:
        message: Message to check

    Returns:
        True if negation patterns are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_negation(message)
    message_lower = message.lower()
    negations = ["don't", "do not", "not", "never", "skip", "without", "avoid"]
    return any(neg in message_lower for neg in negations)


def _is_keyword_negated(message: str, keyword: str, position: int) -> bool:
    """Check if a keyword at a given position is negated.

    Looks for negation patterns in a window before the keyword.
    Also checks for positive override patterns that cancel negation
    (e.g., "don't analyze but do run" - "run" is NOT negated).

    Args:
        message: Full message text
        keyword: The keyword found
        position: Position of keyword in message

    Returns:
        True if keyword appears to be negated
    """
    # Get text window before keyword
    start = max(0, position - NEGATION_WINDOW)
    window = message[start:position].lower()

    # First, check for positive override patterns ANYWHERE in window
    # These patterns (like "but do", "just", "instead") signal positive intent
    # and override any earlier negation for THIS keyword
    for pattern in POSITIVE_OVERRIDE_PATTERNS:
        match = pattern.search(window)
        if match:
            # Check if the positive pattern is closer to keyword than any negation
            positive_pos = match.end()
            # If positive pattern ends close to keyword, it overrides negation
            if len(window) - positive_pos < 10:  # Within 10 chars of keyword
                logger.debug(
                    f"Keyword '{keyword}' has positive override '{match.group()}' close by"
                )
                return False

    # Now check for negation patterns
    for pattern in NEGATION_PATTERNS:
        match = pattern.search(window)
        if match:
            # For simple negation patterns (don't, skip, without), check proximity
            _negation_start = match.start()
            negation_text = match.group()

            # Skip if there's a positive override between negation and keyword
            text_after_negation = window[match.end() :]
            for override_pattern in POSITIVE_OVERRIDE_PATTERNS:
                if override_pattern.search(text_after_negation):
                    logger.debug(
                        f"Keyword '{keyword}' negation cancelled by override in: '{text_after_negation}'"
                    )
                    return False

            # Check if negation is close enough (within 15 chars after negation ends)
            if len(window) - match.end() < 15:
                logger.debug(
                    f"Keyword '{keyword}' negated by '{negation_text}' in window: '{window}'"
                )
                return True

    return False


def _find_keywords_with_positions(
    message: str,
    keywords: List[Tuple[str, float]],
    use_fuzzy: bool = True,
    min_similarity: float = 0.75,
) -> List[KeywordMatch]:
    """Find all keyword matches with positions and weights.

    Args:
        message: Message to search
        keywords: List of (keyword, weight) tuples
        use_fuzzy: Whether to use fuzzy matching for typos (default True)
        min_similarity: Minimum similarity ratio for fuzzy matching (default 0.75)

    Returns:
        List of KeywordMatch objects
    """
    message_lower = message.lower()
    matches = []

    # Try fuzzy matching first if enabled and available
    if use_fuzzy and _FUZZY_MATCHING_AVAILABLE:
        keywords_dict = dict(keywords)
        fuzzy_matches, stats = match_keywords_cascading(
            message_lower,
            keywords_dict,
            use_fuzzy=True,
            min_similarity_ratio=min_similarity,
        )

        # If fuzzy matches found, use them
        if fuzzy_matches and stats["method"] == "fuzzy":
            # Create KeywordMatch objects for fuzzy matches
            for matched_keyword in fuzzy_matches:
                weight = keywords_dict[matched_keyword]
                # Find position in message (use first occurrence)
                pattern = rf"\b{re.escape(matched_keyword)}\b"
                match_obj = re.search(pattern, message_lower, re.IGNORECASE)
                if match_obj:
                    pos = match_obj.start()
                    negated = _is_keyword_negated(message, matched_keyword, pos)
                    matches.append(
                        KeywordMatch(
                            keyword=matched_keyword,
                            category="",  # Set by caller
                            position=pos,
                            negated=negated,
                            weight=weight,
                        )
                    )
            return matches

    # Fall back to exact matching (original behavior)
    for keyword, weight in keywords:
        # Use word boundary matching for single words
        if " " not in keyword:
            pattern = rf"\b{re.escape(keyword)}\b"
        else:
            pattern = re.escape(keyword)

        for match in re.finditer(pattern, message_lower, re.IGNORECASE):
            pos = match.start()
            negated = _is_keyword_negated(message, keyword, pos)
            matches.append(
                KeywordMatch(
                    keyword=keyword,
                    category="",  # Set by caller
                    position=pos,
                    negated=negated,
                    weight=weight,
                )
            )

    return matches


def _calculate_category_score(matches: List[KeywordMatch]) -> Tuple[float, int]:
    """Calculate weighted score for a category, accounting for negation.

    Args:
        matches: List of keyword matches for a category

    Returns:
        Tuple of (score, non_negated_count)
    """
    score = 0.0
    non_negated = 0

    for match in matches:
        if match.negated:
            # Negated keywords reduce score
            score -= match.weight * 0.5
        else:
            score += match.weight
            non_negated += 1

    return max(0.0, score), non_negated


class UnifiedTaskClassifier:
    """Unified task classifier combining keyword, semantic, and contextual signals.

    This classifier provides robust task detection through:
    1. Keyword matching with negation detection
    2. Confidence scoring for ensemble decisions
    3. Contextual boosting from conversation history
    4. Optional semantic classification integration
    """

    def __init__(
        self,
        task_analyzer: Optional["TaskAnalyzer"] = None,
        enable_semantic: bool = True,
        semantic_confidence_threshold: float = 0.85,
        context_boost_factor: float = 0.15,
        decision_service: Optional[Any] = None,
        runtime_intelligence: Optional[Any] = None,
        tiered_decision_service: Optional["TieredDecisionService"] = None,
    ):
        """Initialize the unified classifier.

        Args:
            task_analyzer: Optional TaskAnalyzer for semantic classification
            enable_semantic: Whether to use semantic classification when available
            semantic_confidence_threshold: Min confidence to trust semantic over keyword
            context_boost_factor: How much to boost confidence from context (0-1)
            decision_service: Optional LLMDecisionService for low-confidence augmentation
            runtime_intelligence: Optional canonical runtime-intelligence service
            tiered_decision_service: Optional TieredDecisionService for confidence-based triage
        """
        self._task_analyzer = task_analyzer
        self._enable_semantic = enable_semantic
        self._semantic_threshold = semantic_confidence_threshold
        self._context_boost = context_boost_factor
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence
        self._tiered_decision_service = tiered_decision_service

        # Lazy-loaded semantic classifier
        self._semantic_classifier: Optional["TaskTypeClassifier"] = None

        # Classification cache with TTL
        self._cache: Dict[str, Tuple[ClassificationResult, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _has_decision_support(self) -> bool:
        """Return whether low-confidence LLM task classification is available."""
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
        """Delegate low-confidence task classification to the canonical runtime service."""
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

    @property
    def semantic_classifier(self) -> Optional["TaskTypeClassifier"]:
        """Get semantic classifier, loading lazily."""
        if self._semantic_classifier is None and self._enable_semantic:
            try:
                from victor.storage.embeddings.task_classifier import TaskTypeClassifier

                self._semantic_classifier = TaskTypeClassifier.get_instance()
            except (ImportError, Exception) as e:
                logger.debug(f"Semantic classifier not available: {e}")
                self._semantic_classifier = None
        return self._semantic_classifier

    def _get_cache_key(self, message: str) -> str:
        """Generate a cache key for a message.

        Args:
            message: Message to hash

        Returns:
            Hash string suitable for cache key
        """
        # Use MD5 for speed (not security-sensitive)
        return hashlib.md5(message.encode()).hexdigest()

    def _check_cache(self, message: str) -> Optional[ClassificationResult]:
        """Check cache for a classification result.

        Args:
            message: Message to look up

        Returns:
            Cached result if valid, None if miss or expired
        """
        key = self._get_cache_key(message)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < CLASSIFICATION_CACHE_TTL:
                self._cache_hits += 1
                logger.debug(f"Cache hit for message hash {key[:8]}...")
                return result
            else:
                # Expired, remove from cache
                del self._cache[key]

        self._cache_misses += 1
        return None

    def _add_to_cache(self, message: str, result: ClassificationResult) -> None:
        """Add a classification result to cache.

        Args:
            message: Original message
            result: Classification result to cache
        """
        # Evict oldest entries if cache is full
        if len(self._cache) >= CLASSIFICATION_CACHE_SIZE:
            # Remove oldest entry (first inserted)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._get_cache_key(message)
        self._cache[key] = (result, time.time())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache hit/miss rates and size
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "max_size": CLASSIFICATION_CACHE_SIZE,
            "ttl_seconds": CLASSIFICATION_CACHE_TTL,
        }

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Classification cache cleared")

    def classify(self, message: str, use_cache: bool = True) -> ClassificationResult:
        """Classify a message using keyword analysis with negation detection.

        Args:
            message: User message to classify
            use_cache: Whether to use caching (default True)

        Returns:
            ClassificationResult with task type and confidence
        """
        # Check cache first
        if use_cache:
            cached = self._check_cache(message)
            if cached is not None:
                return cached

        # Note: Native extensions accelerate keyword detection within _find_keywords_with_positions
        # and _has_*_keywords_fast functions, but we don't short-circuit the full analysis
        # to maintain consistent output format with all_matches populated.

        # Find all keyword matches
        action_matches = _find_keywords_with_positions(message, ACTION_KEYWORDS)
        for m in action_matches:
            m.category = "action"

        gen_matches = _find_keywords_with_positions(message, GENERATION_KEYWORDS)
        for m in gen_matches:
            m.category = "generation"

        analysis_matches = _find_keywords_with_positions(message, ANALYSIS_KEYWORDS)
        analysis_matches.extend(_find_keywords_with_positions(message, ANALYSIS_QUESTION_PATTERNS))
        for m in analysis_matches:
            m.category = "analysis"

        search_matches = _find_keywords_with_positions(message, SEARCH_KEYWORDS)
        for m in search_matches:
            m.category = "search"

        edit_matches = _find_keywords_with_positions(message, EDIT_KEYWORDS)
        for m in edit_matches:
            m.category = "edit"

        exec_matches = _find_keywords_with_positions(message, EXECUTION_KEYWORDS)
        for m in exec_matches:
            m.category = "execution"

        # Calculate scores for each category
        action_score, action_count = _calculate_category_score(action_matches)
        gen_score, gen_count = _calculate_category_score(gen_matches)
        analysis_score, analysis_count = _calculate_category_score(analysis_matches)
        search_score, search_count = _calculate_category_score(search_matches)
        edit_score, edit_count = _calculate_category_score(edit_matches)
        exec_score, exec_count = _calculate_category_score(exec_matches)

        # Collect all matches for debugging
        all_matches = (
            action_matches + gen_matches + analysis_matches + search_matches + edit_matches
        )
        negated = [m for m in all_matches if m.negated]
        non_negated = [m for m in all_matches if not m.negated]

        # Determine winner
        scores = {
            ClassifierTaskType.ACTION: action_score,
            ClassifierTaskType.GENERATION: gen_score,
            ClassifierTaskType.ANALYSIS: analysis_score,
            ClassifierTaskType.SEARCH: search_score,
            ClassifierTaskType.EDIT: edit_score,
        }

        # Get best score
        best_type = ClassifierTaskType.DEFAULT
        best_score = 0.0
        for task_type, score in scores.items():
            if score > best_score:
                best_type = task_type
                best_score = score

        # Calculate confidence (normalized)
        total_score = sum(scores.values())
        if total_score > 0:
            active_categories = sum(1 for score in scores.values() if score > 0)
            if active_categories == 1:
                # A single clear category should not be penalized below the triage band.
                confidence = min(0.5 + (best_score * 0.15), 0.75)
            else:
                confidence = best_score / (total_score + 1.0)  # +1 for smoothing
                confidence = min(confidence, 0.95)  # Cap at 0.95 for keyword-only
        else:
            confidence = 0.3  # Low confidence for default

        # Special case: when both analysis AND action/edit/gen detected, use position heuristic
        # If action keywords appear AFTER analysis keywords, action is the end goal
        # (e.g., "analyze the codebase and apply the fix" → ACTION)
        # (e.g., "analyze the logs" → ANALYSIS)
        action_positions = [
            m.position for m in action_matches + gen_matches + edit_matches if not m.negated
        ]
        analysis_positions = [m.position for m in analysis_matches if not m.negated]

        if analysis_positions and action_positions:
            max_analysis_pos = max(analysis_positions)
            max_action_pos = max(action_positions)

            # If action keywords appear after analysis keywords, action is the goal
            if max_action_pos > max_analysis_pos:
                # Action/edit/generation takes precedence - it's the end goal
                combined_action_score = action_score + gen_score + edit_score
                if combined_action_score >= analysis_score * 0.5:  # Action strong enough
                    if gen_score >= action_score and gen_score >= edit_score:
                        best_type = ClassifierTaskType.GENERATION
                    elif edit_score >= action_score:
                        best_type = ClassifierTaskType.EDIT
                    else:
                        best_type = ClassifierTaskType.ACTION
                    confidence = min(confidence + 0.15, 0.95)
            else:
                # Analysis appears last - it's the primary task
                if analysis_score >= (action_score + gen_score + edit_score) * 0.5:
                    best_type = ClassifierTaskType.ANALYSIS
                    confidence = min(confidence + 0.1, 0.95)

        # Determine tool budget based on type
        budget_map = {
            ClassifierTaskType.ANALYSIS: 200,
            ClassifierTaskType.ACTION: 50,
            ClassifierTaskType.GENERATION: 10,
            ClassifierTaskType.SEARCH: 30,
            ClassifierTaskType.EDIT: 40,
            ClassifierTaskType.DEFAULT: 20,
        }

        # Temperature adjustment
        temp_adjustment = 0.2 if best_type == ClassifierTaskType.ANALYSIS else 0.0

        # Determine boolean flags (any non-negated match counts)
        has_action = any(not m.negated for m in action_matches)
        has_gen = any(not m.negated for m in gen_matches)
        has_analysis = any(not m.negated for m in analysis_matches)
        has_search = any(not m.negated for m in search_matches)
        has_execution = any(not m.negated for m in exec_matches)

        result = ClassificationResult(
            task_type=best_type,
            confidence=confidence,
            is_action_task=has_action or has_gen,  # Generation is a form of action
            is_analysis_task=has_analysis or has_search,  # Search is exploratory like analysis
            is_generation_task=has_gen,
            needs_execution=has_execution,
            source="keyword",
            keyword_confidence=confidence,
            matched_keywords=non_negated,
            negated_keywords=negated,
            recommended_tool_budget=budget_map[best_type],
            temperature_adjustment=temp_adjustment,
        )

        # Tiered triage: if confidence is low and tiered service available
        # Priority: tiered triage > legacy LLM augmentation
        if self._tiered_decision_service is not None and confidence < 0.8:
            try:
                from victor.agent.decisions.schemas import DecisionType
                from victor.framework.runtime_evaluation_policy import (
                    RuntimeEvaluationPolicy,
                )
                from victor.agent.services.tiered_decision_service import (
                    ClassificationTriage,
                )

                runtime_policy = RuntimeEvaluationPolicy()

                # Only use triage if confidence is in grey area (below high threshold)
                triage_result = self._tiered_decision_service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message_excerpt": message[:300]},
                    heuristic_result=best_type,
                    heuristic_confidence=confidence,
                    runtime_policy=runtime_policy,
                )

                if triage_result.triage_outcome == ClassificationTriage.ACCEPT:
                    # High confidence from tiered service - use it
                    if hasattr(triage_result.result, "task_type"):
                        triage_task_type = getattr(
                            triage_result.result.task_type,
                            "value",
                            triage_result.result.task_type,
                        )
                        type_map = {
                            "analysis": ClassifierTaskType.ANALYSIS,
                            "action": ClassifierTaskType.ACTION,
                            "generation": ClassifierTaskType.GENERATION,
                            "search": ClassifierTaskType.SEARCH,
                            "edit": ClassifierTaskType.EDIT,
                        }
                        triage_type = type_map.get(triage_task_type)
                        if triage_type is not None:
                            result = ClassificationResult(
                                task_type=triage_type,
                                confidence=triage_result.confidence,
                                is_action_task=triage_type
                                in (
                                    ClassifierTaskType.ACTION,
                                    ClassifierTaskType.GENERATION,
                                ),
                                is_analysis_task=triage_type
                                in (
                                    ClassifierTaskType.ANALYSIS,
                                    ClassifierTaskType.SEARCH,
                                ),
                                is_generation_task=triage_type == ClassifierTaskType.GENERATION,
                                needs_execution=has_execution,
                                source="triage",
                                keyword_confidence=confidence,
                                matched_keywords=non_negated,
                                negated_keywords=negated,
                                recommended_tool_budget=budget_map.get(triage_type, 20),
                                temperature_adjustment=(
                                    0.2 if triage_type == ClassifierTaskType.ANALYSIS else 0.0
                                ),
                            )
                            logger.debug(
                                "Tiered triage classified task type as %s (conf=%.2f, outcome=%s)",
                                triage_type.value,
                                triage_result.confidence,
                                triage_result.triage_outcome,
                            )
                elif triage_result.triage_outcome == ClassificationTriage.REJECT:
                    # Low confidence - use conservative DEFAULT
                    result.task_type = ClassifierTaskType.DEFAULT
                    result.confidence = max(confidence * 0.8, 0.3)
                    result.source = "triage_rejected"
                    logger.debug(
                        "Tiered triage rejected classification (conf=%.2f), using DEFAULT",
                        confidence,
                    )
                # VERIFY outcome: keep original result, verification already considered

            except Exception:
                logger.debug("Tiered triage failed, using base result", exc_info=True)
        elif self._has_decision_support() and confidence < 0.7:
            # Legacy LLM augmentation (fallback when tiered service not available)
            try:
                from victor.agent.decisions.schemas import DecisionType

                decision = self._decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message_excerpt": message[:300]},
                    heuristic_result=best_type,
                    heuristic_confidence=confidence,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and decision.source == "llm"
                    and hasattr(decision.result, "task_type")
                ):
                    llm_task_type = getattr(
                        decision.result.task_type, "value", decision.result.task_type
                    )
                    type_map = {
                        "analysis": ClassifierTaskType.ANALYSIS,
                        "action": ClassifierTaskType.ACTION,
                        "generation": ClassifierTaskType.GENERATION,
                        "search": ClassifierTaskType.SEARCH,
                        "edit": ClassifierTaskType.EDIT,
                    }
                    llm_type = type_map.get(llm_task_type)
                    if llm_type is not None and decision.confidence > confidence:
                        result = ClassificationResult(
                            task_type=llm_type,
                            confidence=decision.confidence,
                            is_action_task=llm_type
                            in (
                                ClassifierTaskType.ACTION,
                                ClassifierTaskType.GENERATION,
                            ),
                            is_analysis_task=llm_type
                            in (ClassifierTaskType.ANALYSIS, ClassifierTaskType.SEARCH),
                            is_generation_task=llm_type == ClassifierTaskType.GENERATION,
                            needs_execution=has_execution,
                            source="llm",
                            keyword_confidence=confidence,
                            matched_keywords=non_negated,
                            negated_keywords=negated,
                            recommended_tool_budget=budget_map.get(llm_type, 20),
                            temperature_adjustment=(
                                0.2 if llm_type == ClassifierTaskType.ANALYSIS else 0.0
                            ),
                        )
                        logger.debug(
                            "LLM classified task type as %s (conf=%.2f)",
                            llm_type.value,
                            decision.confidence,
                        )
            except Exception:
                logger.debug("LLM task type classification failed", exc_info=True)

        # Cache the result
        if use_cache:
            self._add_to_cache(message, result)

        return result

    def classify_with_context(
        self,
        message: str,
        history: List[Dict[str, Any]],
        max_history: int = 5,
    ) -> ClassificationResult:
        """Classify with conversation context for improved accuracy.

        Uses recent conversation history to boost confidence when current
        message is ambiguous but context suggests a task type.

        Args:
            message: Current user message
            history: Conversation history (list of {"role": str, "content": str})
            max_history: Max recent messages to consider

        Returns:
            ClassificationResult with context boosting applied
        """
        # Base classification
        result = self.classify(message)

        if not history:
            return result

        # Analyze recent history
        recent = history[-max_history:]
        context_signals = []
        history_types: Dict[ClassifierTaskType, int] = {}

        for msg in recent:
            if msg.get("role") == "user":
                hist_result = self.classify(msg.get("content", ""))
                if hist_result.task_type != ClassifierTaskType.DEFAULT:
                    history_types[hist_result.task_type] = (
                        history_types.get(hist_result.task_type, 0) + 1
                    )

        # Find dominant type in history
        dominant_type = None
        dominant_count = 0
        for task_type, count in history_types.items():
            if count > dominant_count:
                dominant_type = task_type
                dominant_count = count

        # Apply context boosting
        if dominant_type and result.confidence < 0.7:
            # Low confidence + clear context = boost toward context type
            if dominant_count >= 2:  # At least 2 recent messages of same type
                context_signals.append(f"history_dominant:{dominant_type.value}:{dominant_count}")

                # If current result matches context, boost confidence
                if result.task_type == dominant_type:
                    boost = self._context_boost * (dominant_count / max_history)
                    result.confidence = min(result.confidence + boost, 0.95)
                    result.context_boost = boost
                    result.source = "context"
                # If current is DEFAULT but context is clear, consider switching
                elif result.task_type == ClassifierTaskType.DEFAULT and dominant_count >= 3:
                    result.task_type = dominant_type
                    result.confidence = 0.5 + (self._context_boost * dominant_count / max_history)
                    result.context_boost = result.confidence - 0.5
                    result.source = "context"
                    context_signals.append("type_switched_from_default")

        result.context_signals = context_signals
        return result

    def classify_with_ensemble(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> ClassificationResult:
        """Classify using ensemble of keyword, semantic, and context.

        Combines all available signals with weighted voting.

        Args:
            message: User message to classify
            history: Optional conversation history

        Returns:
            ClassificationResult with ensemble decision
        """
        # Get keyword classification (always available)
        keyword_result = self.classify(message)

        # Get context-boosted classification
        if history:
            context_result = self.classify_with_context(message, history)
        else:
            context_result = keyword_result

        # Try semantic classification
        semantic_type = None
        semantic_confidence = 0.0

        if self.semantic_classifier:
            try:
                sem_result = self.semantic_classifier.classify(message)
                semantic_type = self._map_semantic_to_unified(sem_result.task_type)
                semantic_confidence = sem_result.confidence
            except Exception as e:
                logger.debug(f"Semantic classification failed: {e}")

        # Ensemble voting
        if semantic_type and semantic_confidence >= self._semantic_threshold:
            # High-confidence semantic wins
            final_result = ClassificationResult(
                task_type=semantic_type,
                confidence=semantic_confidence,
                is_action_task=semantic_type == ClassifierTaskType.ACTION,
                is_analysis_task=semantic_type == ClassifierTaskType.ANALYSIS,
                is_generation_task=semantic_type == ClassifierTaskType.GENERATION,
                needs_execution=context_result.needs_execution,
                source="semantic",
                keyword_confidence=keyword_result.confidence,
                semantic_confidence=semantic_confidence,
                context_boost=context_result.context_boost,
                matched_keywords=keyword_result.matched_keywords,
                negated_keywords=keyword_result.negated_keywords,
                context_signals=context_result.context_signals,
                recommended_tool_budget=keyword_result.recommended_tool_budget,
                temperature_adjustment=keyword_result.temperature_adjustment,
            )
        elif (
            context_result.source == "context"
            and context_result.confidence > keyword_result.confidence
        ):
            # Context-boosted wins
            final_result = context_result
            final_result.semantic_confidence = semantic_confidence
        else:
            # Keyword wins (with semantic info attached)
            final_result = keyword_result
            final_result.semantic_confidence = semantic_confidence
            if semantic_type and semantic_confidence > 0.5:
                final_result.source = "ensemble"
                # Slight boost if semantic agrees
                if semantic_type == keyword_result.task_type:
                    final_result.confidence = min(final_result.confidence + 0.1, 0.95)

        return final_result

    def _map_semantic_to_unified(self, semantic_type: Any) -> ClassifierTaskType:
        """Map semantic TaskType to unified ClassifierTaskType.

        Args:
            semantic_type: TaskType from embeddings.task_classifier

        Returns:
            Unified ClassifierTaskType
        """
        # The semantic TaskType enum values
        type_map = {
            "edit": ClassifierTaskType.EDIT,
            "search": ClassifierTaskType.SEARCH,
            "create": ClassifierTaskType.GENERATION,
            "analyze": ClassifierTaskType.ANALYSIS,
            "design": ClassifierTaskType.ANALYSIS,  # Design is analysis-like
        }
        return type_map.get(str(semantic_type.value).lower(), ClassifierTaskType.DEFAULT)


# =============================================================================
# Module-level convenience functions
# =============================================================================

_classifier: Optional[UnifiedTaskClassifier] = None


def get_unified_classifier() -> UnifiedTaskClassifier:
    """Get or create the global unified classifier.

    Returns:
        Global UnifiedTaskClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = UnifiedTaskClassifier()
    return _classifier


def classify_task(message: str) -> ClassificationResult:
    """Convenience function for quick classification.

    Args:
        message: User message to classify

    Returns:
        ClassificationResult
    """
    return get_unified_classifier().classify(message)


def classify_task_with_context(message: str, history: List[Dict[str, Any]]) -> ClassificationResult:
    """Convenience function for context-aware classification.

    Args:
        message: User message to classify
        history: Conversation history

    Returns:
        ClassificationResult with context boosting
    """
    return get_unified_classifier().classify_with_context(message, history)
