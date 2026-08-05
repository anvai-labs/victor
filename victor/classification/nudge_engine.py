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

"""Nudge engine for post-classification edge case corrections.

This module provides rule-based adjustments for known edge cases where
semantic similarity struggles. Nudge rules are applied AFTER embedding
classification to correct misclassifications.

Design Principles:
- Single Responsibility: Only handles post-classification corrections
- Open/Closed: Uses patterns from pattern_registry.py
- Dependency Inversion: Implements NudgeEngineProtocol
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from victor.classification.pattern_registry import (
    PATTERNS,
    ClassificationPattern,
    TaskType,
    get_patterns_sorted_by_priority,
)

logger = logging.getLogger(__name__)


@dataclass
class NudgeRule:
    """A rule-based nudge for correcting edge case classifications.

    NudgeRule is a simplified view of ClassificationPattern focused on
    the nudging behavior. For most cases, use ClassificationPattern directly.

    Attributes:
        name: Rule identifier for debugging
        pattern: Compiled regex pattern to match (on original prompt)
        target_type: Task type to nudge towards
        min_confidence_boost: Minimum boost to apply when not overriding
        override: If True, always override regardless of scores
    """

    name: str
    pattern: re.Pattern
    target_type: TaskType
    min_confidence_boost: float = 0.1
    override: bool = False

    @classmethod
    def from_classification_pattern(cls, cp: ClassificationPattern) -> "NudgeRule":
        """Create NudgeRule from ClassificationPattern.

        Args:
            cp: ClassificationPattern to convert

        Returns:
            NudgeRule with same behavior
        """
        return cls(
            name=cp.name,
            pattern=cp.compiled_pattern,
            target_type=cp.task_type,
            min_confidence_boost=1.0 - cp.confidence,
            override=cp.override,
        )


class NudgeEngine:
    """Post-classification edge case corrections.

    The NudgeEngine applies rule-based corrections to semantic classification
    results. It handles cases where semantic similarity produces incorrect
    results due to:
    - Ambiguous phrasing ("read and explain" → ANALYZE not EDIT)
    - Command-like syntax ("edit path/to/file.py" → EDIT)
    - Domain-specific patterns (GitHub issue format → BUG_FIX)

    Usage:
        engine = NudgeEngine()
        final_type, confidence, rule_name = engine.apply(
            prompt="read and explain auth.py",
            embedding_result=TaskType.EDIT,
            embedding_confidence=0.7,
            scores={TaskType.EDIT: 0.7, TaskType.ANALYZE: 0.5}
        )
        # Returns: (TaskType.ANALYZE, 1.0, "read_and_explain")
    """

    def __init__(self, patterns: Optional[Dict[str, ClassificationPattern]] = None):
        """Initialize NudgeEngine.

        Args:
            patterns: Optional pattern dictionary. Uses global PATTERNS if None.
        """
        self._patterns = patterns or PATTERNS
        self._rules = self._build_rules()

    def _build_rules(self) -> List[NudgeRule]:
        """Build NudgeRule list from patterns.

        Returns:
            List of NudgeRule sorted by pattern priority
        """
        rules = []
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.priority,
            reverse=True,
        )

        for pattern in sorted_patterns:
            rules.append(NudgeRule.from_classification_pattern(pattern))

        logger.debug(f"Built {len(rules)} nudge rules from patterns")
        return rules

    def apply(
        self,
        prompt: str,
        embedding_result: TaskType,
        embedding_confidence: float,
        scores: Optional[Dict[TaskType, float]] = None,
    ) -> Tuple[TaskType, float, Optional[str]]:
        """Apply nudge rules to correct edge cases.

        Args:
            prompt: Original user prompt
            embedding_result: Task type from embedding classification
            embedding_confidence: Confidence from embedding classification
            scores: Optional scores for all task types from embedding

        Returns:
            Tuple of (final_task_type, final_confidence, nudge_rule_name)
            If no nudge applied, returns (embedding_result, embedding_confidence, None)
        """
        scores = scores or {}

        for rule in self._rules:
            if rule.pattern.search(prompt):
                # Rule matched - determine if we should apply it
                target_score = scores.get(rule.target_type, 0.0)

                if rule.override:
                    # Override rules always apply
                    logger.debug(
                        f"Nudge override: {embedding_result.value} → {rule.target_type.value} "
                        f"(rule: {rule.name})"
                    )
                    return rule.target_type, 1.0, rule.name

                elif target_score + rule.min_confidence_boost >= embedding_confidence:
                    # Apply nudge if target score + boost beats embedding result
                    new_confidence = min(1.0, target_score + rule.min_confidence_boost)
                    logger.debug(
                        f"Nudge applied: {embedding_result.value} → {rule.target_type.value} "
                        f"(rule: {rule.name}, confidence: {new_confidence:.2f})"
                    )
                    return rule.target_type, new_confidence, rule.name

        # No nudge applied
        return embedding_result, embedding_confidence, None

    def get_matching_rules(self, prompt: str) -> List[NudgeRule]:
        """Get all rules that match a prompt.

        Useful for debugging classification behavior.

        Args:
            prompt: Prompt to check against rules

        Returns:
            List of matching NudgeRule
        """
        matches = []
        for rule in self._rules:
            if rule.pattern.search(prompt):
                matches.append(rule)
        return matches

    def get_rule_by_name(self, name: str) -> Optional[NudgeRule]:
        """Get a specific rule by name.

        Args:
            name: Rule name to look up

        Returns:
            NudgeRule if found, None otherwise
        """
        for rule in self._rules:
            if rule.name == name:
                return rule
        return None


class PatternMatcher:
    """Fast pattern-based classification without embeddings.

    PatternMatcher provides regex-based classification that can be used
    as a fast path before more expensive embedding classification, or
    as a fallback when embedding service is unavailable.

    Usage:
        matcher = PatternMatcher()
        pattern = matcher.match("fix the bug in auth.py")
        if pattern:
            print(f"Matched: {pattern.name} → {pattern.task_type}")
    """

    def __init__(self, patterns: Optional[Dict[str, ClassificationPattern]] = None):
        """Initialize PatternMatcher.

        Args:
            patterns: Optional pattern dictionary. Uses global PATTERNS if None.
        """
        self._patterns = patterns or PATTERNS
        self._sorted_patterns: Optional[List[ClassificationPattern]] = None

    @property
    def sorted_patterns(self) -> List[ClassificationPattern]:
        """Get patterns sorted by priority (cached)."""
        if self._sorted_patterns is None:
            self._sorted_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.priority,
                reverse=True,
            )
        return self._sorted_patterns

    def match(self, text: str) -> Optional[ClassificationPattern]:
        """Find first matching pattern in priority order.

        Args:
            text: Text to match against patterns

        Returns:
            First matching ClassificationPattern or None
        """
        for pattern in self.sorted_patterns:
            if pattern.compiled_pattern.search(text):
                logger.debug(f"Pattern matched: {pattern.name}")
                return pattern
        return None

    def match_all(self, text: str) -> List[ClassificationPattern]:
        """Find all matching patterns.

        Args:
            text: Text to match against patterns

        Returns:
            List of all matching patterns sorted by priority
        """
        matches = []
        for pattern in self._patterns.values():
            if pattern.compiled_pattern.search(text):
                matches.append(pattern)
        return sorted(matches, key=lambda p: p.priority, reverse=True)

    def classify_fast(self, text: str) -> Optional[Tuple[TaskType, float, str]]:
        """Fast classification using only pattern matching.

        Args:
            text: Text to classify

        Returns:
            Tuple of (task_type, confidence, pattern_name) or None
        """
        pattern = self.match(text)
        if pattern:
            return pattern.task_type, pattern.confidence, pattern.name
        return None


# Singleton instances for convenience
_nudge_engine: Optional[NudgeEngine] = None
_pattern_matcher: Optional[PatternMatcher] = None


def get_nudge_engine() -> NudgeEngine:
    """Get singleton NudgeEngine instance."""
    global _nudge_engine
    if _nudge_engine is None:
        _nudge_engine = NudgeEngine()
    return _nudge_engine


def get_pattern_matcher() -> PatternMatcher:
    """Get singleton PatternMatcher instance."""
    global _pattern_matcher
    if _pattern_matcher is None:
        _pattern_matcher = PatternMatcher()
    return _pattern_matcher


def reset_singletons():
    """Reset singleton instances. Useful for testing."""
    global _nudge_engine, _pattern_matcher
    _nudge_engine = None
    _pattern_matcher = None
