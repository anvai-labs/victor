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

"""Intent classification using semantic embeddings.

This module provides intent classification for detecting:
- Continuation intent: Model wants to continue exploring/analyzing
- Completion intent: Model has finished and is providing a structured response

Uses semantic similarity with embeddings instead of hardcoded phrase matching.
This reduces false positives and handles variations better.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.core.completion_markers import detect_active_completion_marker
from victor.storage.embeddings.collections import (
    CollectionItem,
    StaticEmbeddingCollection,
)
from victor.storage.embeddings.service import EmbeddingService, get_embedding_service

# Import fuzzy matching for robust typo-tolerant intent classification
try:
    from victor.storage.embeddings.fuzzy_matcher import match_keywords_cascading

    _FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    _FUZZY_MATCHING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Module-level optional reference to LLMDecisionService (set during bootstrap)
_decision_service: Optional[Any] = None


def set_decision_service(service: Optional[Any]) -> None:
    """Set the module-level decision service reference.

    Called during bootstrap when the LLM decision service is available.
    """
    global _decision_service
    _decision_service = service


# Compiled regex patterns for explicit continuation detection
# These are strong signals that the model wants to continue working
CONTINUATION_HEURISTIC_PATTERNS = [
    # "Let me read/check/examine/look at..."
    re.compile(
        r"\blet\s+me\s+(read|check|examine|look\s+at|analyze|review|explore|investigate)\b",
        re.IGNORECASE,
    ),
    # "First, let me..." or "Now let me..."
    re.compile(r"\b(first|now|next),?\s+let\s+me\b", re.IGNORECASE),
    # "I'll read/check/examine..."
    re.compile(
        r"\bi['']ll\s+(read|check|examine|look\s+at|analyze|review|explore)\b",
        re.IGNORECASE,
    ),
    # "I need to read/check..."
    re.compile(
        r"\bi\s+need\s+to\s+(read|check|examine|look\s+at|analyze|review)\b",
        re.IGNORECASE,
    ),
    # "I should read/check..."
    re.compile(r"\bi\s+should\s+(read|check|examine|look\s+at|analyze|review)\b", re.IGNORECASE),
    # "Let me start by..."
    re.compile(r"\blet\s+me\s+start\s+by\b", re.IGNORECASE),
]


# Patterns that indicate the model is STUCK in a loop (says what it will do but doesn't)
# These should be detected to prevent continuation loops
STUCK_LOOP_HEURISTIC_PATTERNS = [
    # "I'm going to" without tool call
    re.compile(r"\bi[''`]?m\s+going\s+to\s+(read|examine|check|call|use)\b", re.IGNORECASE),
    # "I will now" without action
    re.compile(r"\bi\s+will\s+now\s+(read|examine|check|call|use)\b", re.IGNORECASE),
    # Long preamble about what will be done (indicates stalling)
    re.compile(
        r"(first|to\s+begin|to\s+start),?\s+i[''`]?(ll|'ll|m\s+going\s+to)",
        re.IGNORECASE,
    ),
    # Repeating tool name like "I'll use search... I'll use read..."
    re.compile(r"\bi[''`]?ll\s+use\s+\w+.*\bi[''`]?ll\s+use\s+\w+", re.IGNORECASE | re.DOTALL),
]


def _has_stuck_loop_heuristic(text: str) -> bool:
    """Check if text indicates model is stuck in a loop (planning but not executing).

    This detects patterns where the model keeps saying what it will do
    but never actually makes tool calls.

    The detection requires either:
    - 3+ planning statements (strong indicator of stalling)
    - The repeating pattern "I'll use X... I'll use Y" (already indicates multiple plans)
    - 2+ planning statements from different pattern types

    This threshold balances:
    - Too low (1-2): False positives when model legitimately plans before acting
    - Too high (4+): Misses actual stuck loops where model can't execute

    Args:
        text: Text to check

    Returns:
        True if text shows stuck loop patterns
    """
    # Check if multiple planning statements exist (strong stuck signal)
    planning_count = 0
    has_repeating_pattern = False
    matched_pattern_indices = set()

    for i, pattern in enumerate(STUCK_LOOP_HEURISTIC_PATTERNS):
        matches = pattern.findall(text)
        match_count = len(matches)
        planning_count += match_count

        if match_count > 0:
            matched_pattern_indices.add(i)

        # Pattern 3 (index 3) is the "repeating tool use" pattern - strongest indicator
        # This pattern already implies 2+ planning statements in one match
        if i == 3 and match_count > 0:
            has_repeating_pattern = True

    # The repeating pattern is a strong signal on its own (implies 2+ I'll use statements)
    if has_repeating_pattern:
        logger.debug("Stuck loop heuristic: found repeating 'I'll use X... I'll use Y' pattern")
        return True

    # Require 3+ planning statements OR 2+ from different pattern types
    # This prevents false positives for legitimate planning responses
    if planning_count >= 3:
        logger.debug(
            f"Stuck loop heuristic: found {planning_count} planning statements (threshold=3)"
        )
        return True

    # 2+ statements from different pattern types indicates diverse planning without action
    if planning_count >= 2 and len(matched_pattern_indices) >= 2:
        logger.debug(
            f"Stuck loop heuristic: found {planning_count} statements from {len(matched_pattern_indices)} pattern types"
        )
        return True

    return False


def _has_continuation_heuristic_fuzzy(text: str) -> bool:
    """Check for continuation patterns with fuzzy matching for typos.

    This function provides a fallback that uses fuzzy matching to detect
    continuation intent even when the user makes typos in keywords.

    Args:
        text: Text to check

    Returns:
        True if continuation patterns detected with fuzzy matching
    """
    if not _FUZZY_MATCHING_AVAILABLE:
        return False

    # Keywords from CONTINUATION_HEURISTIC_PATTERNS
    continuation_keywords = {
        "continue": 1.5,
        "read": 1.3,
        "check": 1.3,
        "examine": 1.3,
        "analyze": 1.3,
        "review": 1.3,
        "explore": 1.2,
        "investigate": 1.2,
        "look": 1.1,
        "next": 1.4,
        "proceed": 1.4,
    }

    matches, stats = match_keywords_cascading(text, continuation_keywords, use_fuzzy=True)
    if not matches:
        return False

    lower = text.lower()
    matched = set(matches)
    has_action_word = bool(
        matched
        & {
            "read",
            "check",
            "examine",
            "analyze",
            "review",
            "explore",
            "investigate",
            "look",
        }
    )
    has_sequence_word = bool(matched & {"continue", "next", "proceed"})
    has_planning_prefix = any(
        phrase in lower
        for phrase in (
            "let me",
            "i'll",
            "ill ",
            "i need to",
            "i should",
        )
    )

    # Fuzzy continuation should require either:
    # - an explicit planning prefix plus an action term, or
    # - a sequencing term plus an action term, or
    # - several matched planning/action terms together.
    if has_planning_prefix and has_action_word:
        return True
    if has_sequence_word and has_action_word:
        return True
    return len(matched) >= 3 and has_action_word


def _has_stuck_loop_heuristic_fuzzy(text: str) -> bool:
    """Check for stuck loop patterns with fuzzy matching for typos.

    This function provides a fallback that uses fuzzy matching to detect
    stuck loop patterns even when the user makes typos in keywords.

    Args:
        text: Text to check

    Returns:
        True if stuck loop patterns detected with fuzzy matching
    """
    if not _FUZZY_MATCHING_AVAILABLE:
        return False

    # Keywords from STUCK_LOOP_HEURISTIC_PATTERNS
    stuck_loop_keywords = {
        "going": 1.2,
        "will": 1.1,
        "read": 1.3,
        "examine": 1.3,
        "check": 1.3,
        "call": 1.2,
        "use": 1.2,
        "begin": 1.1,
        "start": 1.1,
    }

    matches, stats = match_keywords_cascading(text, stuck_loop_keywords, use_fuzzy=True)

    # Need multiple planning keywords to indicate stuck loop
    return len(matches) >= 2


def _has_asking_input_heuristic_fuzzy(text: str) -> bool:
    """Check for asking input patterns with fuzzy matching for typos.

    This function provides a fallback that uses fuzzy matching to detect
    asking input intent even when the user makes typos in keywords.

    Args:
        text: Text to check

    Returns:
        True if asking input patterns detected with fuzzy matching
    """
    if not _FUZZY_MATCHING_AVAILABLE:
        return False

    # Keywords from ASKING_INPUT_HEURISTIC_PATTERNS
    asking_input_keywords = {
        "should": 1.3,
        "prefer": 1.3,
        "like": 1.2,
        "confirm": 1.3,
        "approve": 1.3,
        "would": 1.2,
        "want": 1.2,
        "know": 1.1,
    }

    matches, stats = match_keywords_cascading(text, asking_input_keywords, use_fuzzy=True)
    return len(matches) > 0


# Compiled regex patterns for explicit asking-for-input detection
# These are strong signals that should override semantic classification
ASKING_INPUT_HEURISTIC_PATTERNS = [
    # "Would you like me to..." with question context
    re.compile(r"\bwould\s+you\s+like\s+(me\s+to|to\s+see)\b", re.IGNORECASE),
    # "Do you want me to..."
    re.compile(r"\bdo\s+you\s+want\s+(me\s+to|to\s+see)\b", re.IGNORECASE),
    # "Should I..." at start of sentence or after newline/period
    re.compile(r"(?:^|[.\n])\s*should\s+i\s+\w+", re.IGNORECASE | re.MULTILINE),
    # "Shall I..."
    re.compile(r"(?:^|[.\n])\s*shall\s+i\s+\w+", re.IGNORECASE | re.MULTILINE),
    # "Let me know if you..."
    re.compile(r"\blet\s+me\s+know\s+if\s+you\b", re.IGNORECASE),
    # "Which.*would you like" or "What.*would you like"
    re.compile(r"\b(which|what)\s+.*\bwould\s+you\s+like\b", re.IGNORECASE),
    # "How would you like me to..."
    re.compile(r"\bhow\s+would\s+you\s+like\s+(me\s+to|to)\b", re.IGNORECASE),
]


# Compiled regex patterns for detecting "task complete + optional elaboration" offers
# These questions indicate the task is DONE and model is just offering to go deeper
# Should be treated as COMPLETION, not ASKING_INPUT
COMPLETION_OFFER_HEURISTIC_PATTERNS = [
    # "Would you like me to elaborate/expand/go deeper..."
    re.compile(
        r"\bwould\s+you\s+like\s+(me\s+to\s+)?(elaborate|expand|go\s+deeper|dive\s+deeper|explain\s+further|provide\s+more\s+detail)\b",
        re.IGNORECASE,
    ),
    # "Do you want me to elaborate/expand..."
    re.compile(
        r"\bdo\s+you\s+want\s+(me\s+to\s+)?(elaborate|expand|go\s+deeper|dive\s+deeper|explain\s+further)\b",
        re.IGNORECASE,
    ),
    # "Let me know if you'd like more details on any of these"
    re.compile(
        r"\blet\s+me\s+know\s+if\s+you.*(more\s+details?|elaborate|go\s+deeper)\b",
        re.IGNORECASE,
    ),
    # "Would you like more details on any of..."
    re.compile(
        r"\bwould\s+you\s+like\s+more\s+(details?|information)\s+on\s+(any|these|this)\b",
        re.IGNORECASE,
    ),
    # "I can elaborate on any of these if you'd like"
    re.compile(
        r"\bi\s+can\s+(elaborate|expand|explain\s+further|provide\s+more\s+detail)\b",
        re.IGNORECASE,
    ),
]


def _has_completion_marker_heuristic(text: str) -> bool:
    """Check if text contains explicit rare completion markers.

    These markers are instructed in the system prompt and are the primary
    mechanism for deterministic task completion detection.

    Args:
        text: Text to check

    Returns:
        True if text contains completion markers
    """
    marker = detect_active_completion_marker(text)
    if marker:
        logger.debug(f"Completion-marker heuristic matched: {marker}")
        return True
    return False


def _has_completion_offer_heuristic(text: str) -> bool:
    """Check if text contains 'task complete + optional elaboration' patterns.

    These patterns indicate the model has finished the task and is offering
    to elaborate further. This should be treated as COMPLETION, not ASKING_INPUT,
    because the core task is done.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains completion offer patterns
    """
    # Quick check: must have a question mark or "if you" to be offering something
    if "?" not in text and "if you" not in text.lower():
        return False

    # Check all heuristic patterns
    for pattern in COMPLETION_OFFER_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Completion-offer heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


def _has_continuation_heuristic(text: str) -> bool:
    """Check if text contains strong continuation heuristic patterns.

    These patterns are unambiguous indicators that the model wants to continue
    exploring/reading, so they should override semantic classification when detected.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains strong continuation patterns
    """
    # Check all heuristic patterns
    for pattern in CONTINUATION_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Continuation heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


def _has_asking_input_heuristic(text: str) -> bool:
    """Check if text contains strong asking-for-input heuristic patterns.

    These patterns are unambiguous indicators that the model is asking for user input,
    so they should override semantic classification when detected.

    Args:
        text: Text to check (usually last ~500 chars of response)

    Returns:
        True if text contains strong asking-for-input patterns
    """
    # Quick check: must have a question mark to be asking something
    if "?" not in text:
        return False

    # Check all heuristic patterns
    for pattern in ASKING_INPUT_HEURISTIC_PATTERNS:
        if pattern.search(text):
            logger.debug(f"Asking-input heuristic matched: {pattern.pattern[:40]}...")
            return True

    return False


class IntentType(Enum):
    """Types of intent that can be classified."""

    CONTINUATION = "continuation"  # Model wants to continue (future-looking)
    COMPLETION = "completion"  # Model has finished (summary/conclusion)
    ASKING_INPUT = "asking_input"  # Model is asking for user input/confirmation
    STUCK_LOOP = "stuck_loop"  # Model is stuck in a loop (planning but not executing)
    NEUTRAL = "neutral"  # Neither clear continuation nor completion


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float  # 0-1 confidence score
    top_matches: List[Tuple[str, float]]  # Top matching intents with scores


# Canonical phrases for continuation intent (future-looking)
CONTINUATION_PHRASES = [
    # "Let me" patterns
    "Let me examine the code",
    "Let me look at the file",
    "Let me read this",
    "Let me check the implementation",
    "Let me analyze this further",
    "Let me investigate",
    "Let me start by reading",
    "Let me begin with",
    "Let me review the file",
    "Let me review the contents",
    "Let me fetch the contents",
    # "I'll" patterns
    "I'll examine the code next",
    "I'll look at the implementation",
    "I'll read the file",
    "I'll check this",
    "I'll analyze this",
    "I'll investigate further",
    "I'll review the contents",
    "I'll review the file",
    "I'll fetch the contents",
    "I'll review this",
    # "Now let's" patterns
    "Now let's look at",
    "Now let's examine",
    "Now let's check",
    "Now let's analyze",
    # "Next" patterns
    "Next I'll examine",
    "Next, I'll look at",
    "Next, let me check",
    # In-progress patterns
    "I'm going to read",
    "I'm going to examine",
    "I'm going to check",
    "I need to look at",
    "I should examine",
]

# Canonical phrases for completion intent (finished/summarizing)
COMPLETION_PHRASES = [
    # Summary patterns
    "In summary, here are the findings",
    "To summarize the analysis",
    "Here is a summary of",
    "In conclusion, the analysis shows",
    # "Here are" patterns
    "Here are the main issues",
    "Here are the improvements",
    "Here are my recommendations",
    "Here are the key findings",
    "Here is the complete analysis",
    # List/structured patterns
    "The following improvements are recommended",
    "These are the main areas for improvement",
    "The key points are",
    "Based on my analysis, here are",
    # Conclusion patterns
    "Overall, the code shows",
    "In total, there are",
    "The analysis reveals",
    "After reviewing the code",
    # Numbered section starts (common in structured responses)
    "## 1. First finding",
    "## 2. Second finding",
    "1. The first issue is",
    "2. The second issue is",
]

# Canonical phrases for asking for user input (questions/confirmation requests)
ASKING_INPUT_PHRASES = [
    # "Would you like" patterns
    "Would you like me to help you",
    "Would you like me to implement",
    "Would you like me to continue",
    "Would you like me to develop",
    "Would you like me to create",
    "Would you like me to proceed",
    "Would you like me to explore",
    "Would you like me to explain",
    "Would you like more details",
    "Would you like to see",
    # "Do you want" patterns
    "Do you want me to help",
    "Do you want me to implement",
    "Do you want me to continue",
    "Do you want me to proceed",
    "Do you want me to create",
    "Do you want me to develop",
    "Do you want more information",
    # "Should I" patterns
    "Should I continue with",
    "Should I implement",
    "Should I proceed",
    "Should I help you",
    "Should I develop",
    "Should I create",
    # "Shall I" patterns
    "Shall I continue",
    "Shall I implement",
    "Shall I proceed",
    "Shall I help",
    # Question patterns asking for direction
    "What would you like me to focus on",
    "Which component should I start with",
    "How would you like me to proceed",
    "Where should I begin",
    "What area should I explore first",
    # "Let me know" patterns
    "Let me know if you would like",
    "Let me know if you want",
    "Let me know if you need",
    "Please let me know",
    # Offer patterns
    "I can help you with",
    "I can implement this if you",
    "I can proceed with",
]


def _classify_with_heuristics(
    text: str,
    best_continuation: float,
    best_completion: float,
    best_asking_input: float,
    top_matches: List[Tuple[str, float]],
    continuation_threshold: float,
    completion_threshold: float,
    asking_input_threshold: float,
) -> IntentResult:
    """Shared heuristic + semantic classification logic.

    Used by both classify_intent (async) and classify_intent_sync to avoid
    duplicating the heuristic override chain.

    Heuristic priority order:
    0. Rare completion markers (VICTOR_*::) - highest, deterministic
    1. Stuck loop patterns - model planning but not executing
    2. Continuation patterns - "let me read", "let me check"
    3. Completion offer patterns - task done + offering to elaborate
    4. Asking input patterns - "would you like", "should I"
    5. Semantic similarity (fallback)

    Args:
        text: Text to classify
        best_continuation: Best semantic score for continuation
        best_completion: Best semantic score for completion
        best_asking_input: Best semantic score for asking input
        top_matches: Debug info for top matches
        continuation_threshold: Min similarity for continuation
        completion_threshold: Min similarity for completion
        asking_input_threshold: Min similarity for asking input

    Returns:
        IntentResult with classified intent
    """
    # HEURISTIC OVERRIDE 0 (HIGHEST PRIORITY): Rare completion markers
    # VICTOR_*:: markers are definitive completion signals instructed in the
    # system prompt, checked first because they are unambiguous.
    if _has_completion_marker_heuristic(text):
        top_matches.append(("heuristic:completion_marker", 0.95))
        logger.debug("COMPLETION detected via marker heuristic (conf=0.95)")
        return IntentResult(
            intent=IntentType.COMPLETION,
            confidence=0.95,
            top_matches=top_matches,
        )

    # HEURISTIC OVERRIDE 1: Check for stuck loop patterns
    if _has_stuck_loop_heuristic(text) or _has_stuck_loop_heuristic_fuzzy(text):
        top_matches.append(("heuristic:stuck_loop_pattern", 0.85))
        logger.debug("STUCK_LOOP detected via heuristic override (conf=0.85)")
        return IntentResult(
            intent=IntentType.STUCK_LOOP,
            confidence=0.85,
            top_matches=top_matches,
        )

    # HEURISTIC OVERRIDE 2: Check for explicit continuation patterns
    if _has_continuation_heuristic(text) or _has_continuation_heuristic_fuzzy(text):
        heuristic_confidence = max(best_continuation, 0.75)
        top_matches.append(("heuristic:continuation_pattern", heuristic_confidence))
        logger.debug(
            f"CONTINUATION detected via heuristic override (conf={heuristic_confidence:.2f})"
        )
        return IntentResult(
            intent=IntentType.CONTINUATION,
            confidence=heuristic_confidence,
            top_matches=top_matches,
        )

    # HEURISTIC OVERRIDE 3: Check for "task done + optional elaboration" patterns
    if _has_completion_offer_heuristic(text):
        heuristic_confidence = max(best_completion, 0.85)
        top_matches.append(("heuristic:completion_offer_pattern", heuristic_confidence))
        logger.debug(
            f"COMPLETION detected via elaboration-offer heuristic (conf={heuristic_confidence:.2f})"
        )
        return IntentResult(
            intent=IntentType.COMPLETION,
            confidence=heuristic_confidence,
            top_matches=top_matches,
        )

    # HEURISTIC OVERRIDE 4: Check for explicit asking-for-input patterns
    if _has_asking_input_heuristic(text) or _has_asking_input_heuristic_fuzzy(text):
        heuristic_confidence = max(best_asking_input, 0.75)
        top_matches.append(("heuristic:asking_input_pattern", heuristic_confidence))
        logger.debug(
            f"ASKING_INPUT detected via heuristic override (conf={heuristic_confidence:.2f})"
        )
        return IntentResult(
            intent=IntentType.ASKING_INPUT,
            confidence=heuristic_confidence,
            top_matches=top_matches,
        )

    # Semantic similarity fallback
    TIE_THRESHOLD = 0.05
    max_score = max(best_continuation, best_completion, best_asking_input)
    scores_are_tied = (
        abs(best_continuation - max_score) < TIE_THRESHOLD
        and abs(best_completion - max_score) < TIE_THRESHOLD
    )

    if scores_are_tied:
        return IntentResult(
            intent=IntentType.NEUTRAL,
            confidence=max_score,
            top_matches=top_matches,
        )

    if (
        best_asking_input >= asking_input_threshold
        and best_asking_input >= best_continuation
        and best_asking_input >= best_completion
    ):
        return IntentResult(
            intent=IntentType.ASKING_INPUT,
            confidence=best_asking_input,
            top_matches=top_matches,
        )
    elif best_continuation >= continuation_threshold and best_continuation > best_completion:
        return IntentResult(
            intent=IntentType.CONTINUATION,
            confidence=best_continuation,
            top_matches=top_matches,
        )
    elif best_completion >= completion_threshold and best_completion > best_continuation:
        return IntentResult(
            intent=IntentType.COMPLETION,
            confidence=best_completion,
            top_matches=top_matches,
        )
    else:
        # LLM augmentation: if result is NEUTRAL and decision service is available
        neutral_confidence = max(best_continuation, best_completion, best_asking_input)
        if _decision_service is not None and neutral_confidence < 0.7:
            try:
                from victor.agent.decisions.schemas import DecisionType

                decision = _decision_service.decide_sync(
                    DecisionType.INTENT_CLASSIFICATION,
                    context={
                        "text_tail": text[-300:],
                        "has_tool_calls": "false",
                    },
                    heuristic_result=IntentType.NEUTRAL,
                    heuristic_confidence=neutral_confidence,
                )
                if decision.source == "llm" and hasattr(decision.result, "intent"):
                    intent_map = {
                        "continuation": IntentType.CONTINUATION,
                        "completion": IntentType.COMPLETION,
                        "asking_input": IntentType.ASKING_INPUT,
                        "stuck_loop": IntentType.STUCK_LOOP,
                    }
                    mapped = intent_map.get(decision.result.intent)
                    if mapped is not None:
                        top_matches.append(("llm:intent", decision.confidence))
                        logger.debug(
                            "LLM classified intent as %s (conf=%.2f)",
                            mapped.value,
                            decision.confidence,
                        )
                        return IntentResult(
                            intent=mapped,
                            confidence=decision.confidence,
                            top_matches=top_matches,
                        )
            except Exception:
                logger.debug("LLM intent classification failed", exc_info=True)

        return IntentResult(
            intent=IntentType.NEUTRAL,
            confidence=neutral_confidence,
            top_matches=top_matches,
        )


class IntentClassifier:
    """Semantic intent classifier using embeddings.

    Replaces hardcoded phrase matching with semantic similarity.
    Benefits:
    - Handles variations and paraphrases
    - Reduces false positives
    - More robust to model output variations

    Usage:
        # Use singleton to avoid duplicate initialization
        classifier = IntentClassifier.get_instance()
        await classifier.initialize()

        result = await classifier.classify_intent(
            "Let me start by examining the orchestrator.py file"
        )
        print(result.intent)  # IntentType.CONTINUATION
        print(result.confidence)  # 0.85
    """

    _instance: Optional["IntentClassifier"] = None
    _lock = __import__("threading").Lock()

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        continuation_threshold: float = 0.30,
        completion_threshold: float = 0.30,
        asking_input_threshold: float = 0.35,
    ) -> "IntentClassifier":
        """Get or create the singleton IntentClassifier instance.

        Args:
            cache_dir: Directory for cache files (only used on first call)
            embedding_service: Shared embedding service (only used on first call)
            continuation_threshold: Min similarity for continuation (only used on first call)
            completion_threshold: Min similarity for completion (only used on first call)
            asking_input_threshold: Min similarity for asking input (only used on first call)

        Returns:
            The singleton IntentClassifier instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache_dir=cache_dir,
                        embedding_service=embedding_service,
                        continuation_threshold=continuation_threshold,
                        completion_threshold=completion_threshold,
                        asking_input_threshold=asking_input_threshold,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            logger.debug("Reset IntentClassifier singleton")

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
        continuation_threshold: float = 0.30,
        completion_threshold: float = 0.30,
        asking_input_threshold: float = 0.35,
    ):
        """Initialize intent classifier.

        Args:
            cache_dir: Directory for cache files
            embedding_service: Shared embedding service
            continuation_threshold: Minimum similarity for continuation intent (default 0.30)
                Note: Semantic similarity scores typically range 0.3-0.9 for related texts.
                Lower values (0.25-0.35) catch more matches but may have false positives.
                Higher values (0.5+) are more conservative but may miss legitimate cases.
            completion_threshold: Minimum similarity for completion intent (default 0.30)
            asking_input_threshold: Minimum similarity for asking input intent (default 0.35)
                Slightly higher threshold to avoid false positives on questions.
        """
        from victor.config.settings import get_project_paths

        self.cache_dir = cache_dir or get_project_paths().global_embeddings_dir
        self.embedding_service = embedding_service or get_embedding_service()
        self.continuation_threshold = continuation_threshold
        self.completion_threshold = completion_threshold
        self.asking_input_threshold = asking_input_threshold

        # Create collections
        self._continuation_collection = StaticEmbeddingCollection(
            name="intent_continuation",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )
        self._completion_collection = StaticEmbeddingCollection(
            name="intent_completion",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )
        self._asking_input_collection = StaticEmbeddingCollection(
            name="intent_asking_input",
            cache_dir=self.cache_dir,
            embedding_service=self.embedding_service,
        )

        self._initialized = False
        self._tiered_service: Optional[Any] = None  # lazily resolved on first classify

    def _get_tiered_service(self) -> Optional[Any]:
        """Return TieredDecisionService if tiered classification is enabled."""
        from victor.agent.services.classification_backend import (
            use_tiered_classification,
        )

        if not use_tiered_classification():
            return None
        if self._tiered_service is None:
            try:
                from victor.core import get_container
                from victor.agent.services.tiered_decision_service import (
                    TieredDecisionService,
                )

                self._tiered_service = get_container().get(TieredDecisionService)
            except Exception:
                pass
        return self._tiered_service

    @property
    def is_initialized(self) -> bool:
        """Check if classifier is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize intent collections with canonical phrases."""
        if self._initialized:
            return

        # Build continuation collection
        continuation_items = [
            CollectionItem(
                id=f"cont_{i}",
                text=phrase,
                metadata={"intent": "continuation"},
            )
            for i, phrase in enumerate(CONTINUATION_PHRASES)
        ]
        await self._continuation_collection.initialize(continuation_items)

        # Build completion collection
        completion_items = [
            CollectionItem(
                id=f"comp_{i}",
                text=phrase,
                metadata={"intent": "completion"},
            )
            for i, phrase in enumerate(COMPLETION_PHRASES)
        ]
        await self._completion_collection.initialize(completion_items)

        # Build asking_input collection
        asking_input_items = [
            CollectionItem(
                id=f"ask_{i}",
                text=phrase,
                metadata={"intent": "asking_input"},
            )
            for i, phrase in enumerate(ASKING_INPUT_PHRASES)
        ]
        await self._asking_input_collection.initialize(asking_input_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation, "
            f"{len(COMPLETION_PHRASES)} completion, and {len(ASKING_INPUT_PHRASES)} asking_input phrases"
        )

    def initialize_sync(self) -> None:
        """Initialize intent collections (sync version)."""
        if self._initialized:
            return

        # Build continuation collection
        continuation_items = [
            CollectionItem(
                id=f"cont_{i}",
                text=phrase,
                metadata={"intent": "continuation"},
            )
            for i, phrase in enumerate(CONTINUATION_PHRASES)
        ]
        self._continuation_collection.initialize_sync(continuation_items)

        # Build completion collection
        completion_items = [
            CollectionItem(
                id=f"comp_{i}",
                text=phrase,
                metadata={"intent": "completion"},
            )
            for i, phrase in enumerate(COMPLETION_PHRASES)
        ]
        self._completion_collection.initialize_sync(completion_items)

        # Build asking_input collection
        asking_input_items = [
            CollectionItem(
                id=f"ask_{i}",
                text=phrase,
                metadata={"intent": "asking_input"},
            )
            for i, phrase in enumerate(ASKING_INPUT_PHRASES)
        ]
        self._asking_input_collection.initialize_sync(asking_input_items)

        self._initialized = True
        logger.info(
            f"IntentClassifier initialized with {len(CONTINUATION_PHRASES)} continuation, "
            f"{len(COMPLETION_PHRASES)} completion, and {len(ASKING_INPUT_PHRASES)} asking_input phrases"
        )

    async def classify_intent(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text.

        Uses a hybrid approach:
        1. Semantic similarity matching against canonical phrases
        2. Heuristic pattern matching as override for unambiguous cases

        Args:
            text: Text to classify (usually model's response start)
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            await self.initialize()

        # Search all collections
        continuation_results = await self._continuation_collection.search(text, top_k=top_k)
        completion_results = await self._completion_collection.search(text, top_k=top_k)
        asking_input_results = await self._asking_input_collection.search(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0
        best_asking_input = asking_input_results[0][1] if asking_input_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))
        for item, score in asking_input_results[:2]:
            top_matches.append((f"ask:{item.text[:50]}", score))

        # Classify using shared heuristic + semantic logic
        return _classify_with_heuristics(
            text=text,
            best_continuation=best_continuation,
            best_completion=best_completion,
            best_asking_input=best_asking_input,
            top_matches=top_matches,
            continuation_threshold=self.continuation_threshold,
            completion_threshold=self.completion_threshold,
            asking_input_threshold=self.asking_input_threshold,
        )

    def classify_intent_sync(
        self,
        text: str,
        top_k: int = 3,
    ) -> IntentResult:
        """Classify the intent of text (sync version).

        Uses a hybrid approach:
        1. Semantic similarity matching against canonical phrases
        2. Heuristic pattern matching as override for unambiguous cases

        Args:
            text: Text to classify
            top_k: Number of top matches to return

        Returns:
            IntentResult with classified intent, confidence, and top matches
        """
        if not self._initialized:
            self.initialize_sync()

        # Search all collections
        continuation_results = self._continuation_collection.search_sync(text, top_k=top_k)
        completion_results = self._completion_collection.search_sync(text, top_k=top_k)
        asking_input_results = self._asking_input_collection.search_sync(text, top_k=top_k)

        # Get best scores
        best_continuation = continuation_results[0][1] if continuation_results else 0.0
        best_completion = completion_results[0][1] if completion_results else 0.0
        best_asking_input = asking_input_results[0][1] if asking_input_results else 0.0

        # Build top matches for debugging
        top_matches = []
        for item, score in continuation_results[:2]:
            top_matches.append((f"cont:{item.text[:50]}", score))
        for item, score in completion_results[:2]:
            top_matches.append((f"comp:{item.text[:50]}", score))
        for item, score in asking_input_results[:2]:
            top_matches.append((f"ask:{item.text[:50]}", score))

        # Classify using shared heuristic + semantic logic
        base_result = _classify_with_heuristics(
            text=text,
            best_continuation=best_continuation,
            best_completion=best_completion,
            best_asking_input=best_asking_input,
            top_matches=top_matches,
            continuation_threshold=self.continuation_threshold,
            completion_threshold=self.completion_threshold,
            asking_input_threshold=self.asking_input_threshold,
        )

        # Tiered triage: delegate to classify_with_triage if the feature is enabled.
        tiered_svc = self._get_tiered_service()
        if tiered_svc is not None:
            intent, confidence, _source = self.classify_with_triage(
                text=text,
                context={"has_tool_calls": False},
                tiered_service=tiered_svc,
            )
            return IntentResult(
                intent=intent,
                confidence=confidence,
                top_matches=top_matches,
            )

        return base_result

    def classify_with_triage(
        self,
        text: str,
        context: Dict[str, Any],
        tiered_service: Optional[Any] = None,
        runtime_policy: Optional[Any] = None,
    ) -> Tuple[IntentType, float, str]:
        """Classify intent with confidence-based triage.

        Extends the base intent classification with tiered triage:
        - High confidence (≥0.8): Accept immediately
        - Medium confidence (0.5-0.8): Verify with edge LLM
        - Low confidence (<0.5): Reject and return CONTINUATION

        Args:
            text: Text to classify (usually model's response tail)
            context: Additional context (has_tool_calls, etc.)
            tiered_service: Optional TieredDecisionService for triage
            runtime_policy: Optional RuntimeEvaluationPolicy for thresholds

        Returns:
            Tuple of (intent, confidence, source) where source is one of:
            - "keyword", "semantic", "llm", "edge_verification", "heuristic_fallback"
        """
        from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
        from victor.agent.services.tiered_decision_service import ClassificationTriage

        if runtime_policy is None:
            runtime_policy = RuntimeEvaluationPolicy()

        # Get base classification
        base_result = self.classify_intent_sync(text)
        base_confidence = base_result.confidence

        # High confidence: Accept immediately
        if base_confidence >= runtime_policy.high_confidence_threshold:
            logger.debug(
                "Intent classification: high confidence (%.2f), accepting %s immediately",
                base_confidence,
                base_result.intent.value,
            )
            return base_result.intent, base_confidence, "semantic"

        # No tiered service: Return base result
        if tiered_service is None:
            logger.debug(
                "Intent classification: no tiered service, using base result (conf=%.2f)",
                base_confidence,
            )
            return base_result.intent, base_confidence, "semantic"

        # Medium/Low confidence: Use tiered triage
        try:
            from victor.agent.decisions.schemas import DecisionType

            triage_result = tiered_service.classify_with_triage(
                DecisionType.INTENT_CLASSIFICATION,
                context={
                    "text_tail": text[-300:],
                    "has_tool_calls": str(context.get("has_tool_calls", False)).lower(),
                    **{k: v for k, v in context.items() if k != "has_tool_calls"},
                },
                heuristic_result=base_result.intent,
                heuristic_confidence=base_confidence,
                runtime_policy=runtime_policy,
            )

            if triage_result.triage_outcome == ClassificationTriage.REJECT:
                # Conservative fallback: CONTINUATION
                logger.debug(
                    "Intent classification: low confidence rejected, using CONTINUATION fallback"
                )
                return IntentType.CONTINUATION, 0.4, "heuristic_fallback"

            # Map triage result back to IntentType
            verified_intent = self._map_to_intent_type(triage_result.result)
            return verified_intent, triage_result.confidence, triage_result.source

        except Exception:
            logger.debug("Intent triage failed, using base result", exc_info=True)
            return base_result.intent, base_confidence, "semantic"

    def _map_to_intent_type(self, result: Any) -> IntentType:
        """Map triage result to IntentType.

        Args:
            result: Result from tiered triage (string, IntentType, or object with intent attribute)

        Returns:
            IntentType enum value
        """
        # If already an IntentType, return it
        if isinstance(result, IntentType):
            return result

        # If it's a string, try to map it
        if isinstance(result, str):
            intent_map = {
                "continuation": IntentType.CONTINUATION,
                "completion": IntentType.COMPLETION,
                "asking_input": IntentType.ASKING_INPUT,
                "stuck_loop": IntentType.STUCK_LOOP,
                "neutral": IntentType.NEUTRAL,
            }
            return intent_map.get(result.lower(), IntentType.NEUTRAL)

        # If it's an object with intent attribute
        if hasattr(result, "intent"):
            return self._map_to_intent_type(result.intent)

        # Default fallback
        return IntentType.NEUTRAL

    def intends_to_continue(self, text: str) -> bool:
        """Quick check if text indicates continuation intent.

        This is a convenience method for the orchestrator to use
        instead of hardcoded phrase matching.

        Args:
            text: Text to check

        Returns:
            True if text indicates continuation intent
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.CONTINUATION

    def is_complete_response(self, text: str) -> bool:
        """Quick check if text indicates a complete response.

        Args:
            text: Text to check

        Returns:
            True if text indicates completion intent
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.COMPLETION

    def is_asking_for_input(self, text: str) -> bool:
        """Quick check if text indicates the model is asking for user input.

        This detects patterns like "Would you like me to...", "Should I...", etc.
        Useful for auto-continuing in one-shot mode.

        Args:
            text: Text to check

        Returns:
            True if text indicates asking for user input
        """
        result = self.classify_intent_sync(text)
        return result.intent == IntentType.ASKING_INPUT

    def clear_cache(self) -> None:
        """Clear cached collections."""
        self._continuation_collection.clear_cache()
        self._completion_collection.clear_cache()
        self._asking_input_collection.clear_cache()
        self._initialized = False
        logger.info("IntentClassifier cache cleared")
