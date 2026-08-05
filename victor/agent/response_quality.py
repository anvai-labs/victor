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

"""Response quality scoring for LLM outputs.

This module provides quality assessment for LLM responses, enabling
automatic quality gates and feedback loops for improved outputs.

Design Pattern: Strategy + Observer
==================================
Multiple scoring strategies contribute to an overall quality score.
Quality events can be observed for logging, metrics, and adaptive
behavior.

Scoring Dimensions:
- Relevance: How well the response addresses the query
- Completeness: Coverage of requested information
- Accuracy: Factual correctness (via grounding verification)
- Conciseness: Information density vs verbosity
- Actionability: Clarity of next steps (for code tasks)

Usage:
    scorer = ResponseQualityScorer()
    result = await scorer.score(
        query="How do I add authentication?",
        response="To add authentication, you'll need to...",
        context={"files_read": ["auth.py"]},
    )

    if result.overall_score < 0.6:
        # Consider regenerating or improving response
        pass
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Re-export from canonical location for backward compatibility
from victor.core.shared_types import (  # noqa: F401
    DimensionScore,
    QualityResult,
    ResponseQualityDimension,
)


@dataclass
class ScorerConfig:
    """Configuration for quality scorer.

    Attributes:
        min_threshold: Minimum score to pass
        dimension_weights: Custom weights per dimension
        enable_code_quality: Score code snippets
        enable_grounding: Check factual accuracy
        strict_mode: Fail on any dimension below threshold
    """

    min_threshold: float = 0.6
    dimension_weights: Dict[ResponseQualityDimension, float] = field(default_factory=dict)
    enable_code_quality: bool = True
    enable_grounding: bool = True
    strict_mode: bool = False
    dimension_threshold: float = 0.4  # Min score per dimension in strict mode


class ResponseQualityScorer:
    """Scores LLM response quality across multiple dimensions.

    Provides objective quality metrics and improvement suggestions
    for better response generation.
    """

    # Default weights for each dimension
    DEFAULT_WEIGHTS = {
        ResponseQualityDimension.RELEVANCE: 1.5,
        ResponseQualityDimension.COMPLETENESS: 1.2,
        ResponseQualityDimension.ACCURACY: 1.3,
        ResponseQualityDimension.CONCISENESS: 0.8,
        ResponseQualityDimension.ACTIONABILITY: 1.0,
        ResponseQualityDimension.COHERENCE: 0.9,
        ResponseQualityDimension.CODE_QUALITY: 1.1,
    }

    # Patterns that indicate poor quality
    LOW_QUALITY_PATTERNS = [
        (r"I don't know", "Uncertainty"),
        (r"I'm not sure", "Uncertainty"),
        (r"I cannot", "Inability"),
        (r"As an AI", "Meta-commentary"),
        (r"I apologize", "Unnecessary apology"),
        (r"Let me think", "Thinking aloud"),
        (r"TODO|FIXME|XXX", "Incomplete content"),
        (r"\.{3,}", "Placeholder content"),
        (r"etc\.", "Incomplete enumeration"),
    ]

    # Patterns that indicate good quality
    HIGH_QUALITY_PATTERNS = [
        (r"^(?:1\.|Step 1|First,)", "Structured approach"),
        (r"```\w+", "Code examples"),
        (r"(?:For example|e\.g\.)", "Concrete examples"),
        (r"(?:In summary|To summarize)", "Summary provided"),
        (r"(?:Note:|Warning:|Important:)", "Key points highlighted"),
    ]

    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        grounding_verifier: Optional[Any] = None,
    ):
        """Initialize the quality scorer.

        Args:
            config: Scorer configuration
            grounding_verifier: Optional GroundingVerifier for accuracy checking
        """
        self.config = config or ScorerConfig()
        self.grounding_verifier = grounding_verifier

        # Merge custom weights with defaults
        self.weights = {**self.DEFAULT_WEIGHTS, **self.config.dimension_weights}

        # Quality observers
        self._observers: List[Callable[[QualityResult], None]] = []

    def add_observer(self, observer: Callable[[QualityResult], None]) -> None:
        """Add an observer for quality events."""
        self._observers.append(observer)

    def _notify_observers(self, result: QualityResult) -> None:
        """Notify all observers of a quality result."""
        for observer in self._observers:
            try:
                observer(result)
            except Exception as e:
                logger.debug(f"Observer error: {e}")

    def _emit_quality_assessed_event(
        self,
        result: QualityResult,
        query: str,
        response: str,
    ) -> None:
        """Emit RL event for quality assessment.

        This activates the quality_weights learner to learn optimal
        dimension weights based on quality outcomes.

        Args:
            result: Quality assessment result
            query: Original query
            response: Response that was scored
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Extract dimension scores for metadata
            dimension_data = {ds.dimension.value: ds.score for ds in result.dimension_scores}

            event = RLEvent(
                type=RLEventType.QUALITY_ASSESSED,
                success=result.passes_threshold,
                quality_score=result.overall_score,
                metadata={
                    "dimensions": dimension_data,
                    "passes_threshold": result.passes_threshold,
                    "query_length": len(query),
                    "response_length": len(response),
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Quality assessed event emission failed: {e}")

    async def score(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> QualityResult:
        """Score a response's quality.

        Args:
            query: Original query/prompt
            response: LLM response to score
            context: Optional context (files read, tool results, etc.)

        Returns:
            QualityResult with scores and suggestions
        """
        context = context or {}
        dimension_scores: List[DimensionScore] = []

        # Score each dimension
        dimension_scores.append(await self._score_relevance(query, response))
        dimension_scores.append(await self._score_completeness(query, response, context))
        dimension_scores.append(await self._score_accuracy(response, context))
        dimension_scores.append(await self._score_conciseness(response))
        dimension_scores.append(await self._score_actionability(query, response))
        dimension_scores.append(await self._score_coherence(response))

        if self.config.enable_code_quality and self._has_code(response):
            dimension_scores.append(await self._score_code_quality(response))

        # Calculate weighted average
        total_weight = sum(ds.weight for ds in dimension_scores)
        overall_score = (
            sum(ds.score * ds.weight for ds in dimension_scores) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Check threshold
        passes = overall_score >= self.config.min_threshold

        # In strict mode, check each dimension
        if self.config.strict_mode and passes:
            for ds in dimension_scores:
                if ds.score < self.config.dimension_threshold:
                    passes = False
                    break

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(dimension_scores)

        result = QualityResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            passes_threshold=passes,
            improvement_suggestions=suggestions,
            metadata={
                "query_length": len(query),
                "response_length": len(response),
                "has_code": self._has_code(response),
                "context_size": len(context),
            },
        )

        # Notify observers
        self._notify_observers(result)

        # Emit RL event for quality assessment
        self._emit_quality_assessed_event(result, query, response)

        logger.debug(
            f"Quality score: {overall_score:.2f} "
            f"(passes={passes}, dimensions={len(dimension_scores)})"
        )

        return result

    async def _score_relevance(self, query: str, response: str) -> DimensionScore:
        """Score how relevant the response is to the query."""
        query_lower = query.lower()
        response_lower = response.lower()

        # Extract key terms from query
        query_terms = set(re.findall(r"\b\w{4,}\b", query_lower))
        query_terms -= {"what", "how", "when", "where", "which", "that", "this", "with"}

        if not query_terms:
            return DimensionScore(
                dimension=ResponseQualityDimension.RELEVANCE,
                score=0.7,
                weight=self.weights[ResponseQualityDimension.RELEVANCE],
                feedback="Query too short to assess relevance",
            )

        # Count how many query terms appear in response
        term_matches = sum(1 for term in query_terms if term in response_lower)
        term_coverage = term_matches / len(query_terms) if query_terms else 0

        # Check for semantic relevance indicators
        semantic_score = 0.0

        # Response mentions the query topic
        if any(term in response_lower for term in query_terms):
            semantic_score += 0.3

        # Response has reasonable length for query
        query_words = len(query.split())
        response_words = len(response.split())
        if response_words >= query_words * 2:  # At least 2x expansion
            semantic_score += 0.2

        # Combine scores
        score = min(1.0, (term_coverage * 0.5) + semantic_score + 0.3)

        evidence = []
        if term_coverage > 0.5:
            evidence.append(f"Good term coverage: {term_coverage:.0%}")
        else:
            evidence.append(f"Low term coverage: {term_coverage:.0%}")

        return DimensionScore(
            dimension=ResponseQualityDimension.RELEVANCE,
            score=score,
            weight=self.weights[ResponseQualityDimension.RELEVANCE],
            feedback="Improve topic coverage" if score < 0.7 else "Good relevance",
            evidence=evidence,
        )

    async def _score_completeness(
        self, query: str, response: str, context: Dict[str, Any]
    ) -> DimensionScore:
        """Score how complete the response is."""
        score = 0.5  # Base score
        evidence = []

        # Check for multi-part queries
        query_parts = re.split(r"[,;]|(?:\band\b)", query.lower())
        addressed_parts = 0

        for part in query_parts:
            part_terms = set(re.findall(r"\b\w{4,}\b", part))
            if any(term in response.lower() for term in part_terms):
                addressed_parts += 1

        if query_parts:
            part_coverage = addressed_parts / len(query_parts)
            score = 0.3 + (part_coverage * 0.5)
            evidence.append(f"Query parts addressed: {addressed_parts}/{len(query_parts)}")

        # Check for expected sections in longer responses
        response_words = len(response.split())
        if response_words > 100:
            # Longer responses should have structure
            has_structure = bool(
                re.search(r"(?:^|\n)(?:\d+\.|[-*]|\#{1,3})", response, re.MULTILINE)
            )
            if has_structure:
                score += 0.1
                evidence.append("Has structured formatting")

        # Check if code was expected but missing
        code_expected = any(
            kw in query.lower()
            for kw in ["code", "implement", "write", "create", "function", "example"]
        )
        has_code = self._has_code(response)

        if code_expected and not has_code:
            score -= 0.2
            evidence.append("Code expected but not provided")
        elif code_expected and has_code:
            score += 0.1
            evidence.append("Code example provided")

        # Context utilization
        files_read = context.get("files_read", [])
        if files_read:
            files_mentioned = sum(
                1 for f in files_read if f.split("/")[-1].split(".")[0] in response.lower()
            )
            if files_mentioned > 0:
                score += 0.1
                evidence.append(f"Referenced {files_mentioned} read files")

        return DimensionScore(
            dimension=ResponseQualityDimension.COMPLETENESS,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.COMPLETENESS],
            feedback="Add more detail" if score < 0.6 else "Good completeness",
            evidence=evidence,
        )

    async def _score_accuracy(self, response: str, context: Dict[str, Any]) -> DimensionScore:
        """Score factual accuracy of the response."""
        score = 0.7  # Default score when no verification possible
        evidence = []

        # Use grounding verifier if available
        if self.grounding_verifier and self.config.enable_grounding:
            try:
                grounding_result = await self.grounding_verifier.verify(response, context)
                score = grounding_result.confidence
                evidence.append(f"Grounding confidence: {grounding_result.confidence:.2f}")

                if grounding_result.issues:
                    evidence.append(f"Grounding issues: {len(grounding_result.issues)}")

            except Exception as e:
                logger.debug(f"Grounding verification failed: {e}")
                evidence.append("Grounding verification failed")

        # Check for hedging language that may indicate uncertainty
        hedging_patterns = [
            r"\bmight\b",
            r"\bpossibly\b",
            r"\bperhaps\b",
            r"\bmaybe\b",
            r"\bcould be\b",
            r"\bi think\b",
        ]
        hedging_count = sum(
            len(re.findall(pattern, response.lower())) for pattern in hedging_patterns
        )
        if hedging_count > 3:
            score -= 0.1
            evidence.append(f"High hedging language ({hedging_count} instances)")

        return DimensionScore(
            dimension=ResponseQualityDimension.ACCURACY,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.ACCURACY],
            feedback="Verify claims" if score < 0.7 else "Good accuracy",
            evidence=evidence,
        )

    async def _score_conciseness(self, response: str) -> DimensionScore:
        """Score response conciseness (information density)."""
        words = response.split()
        word_count = len(words)

        evidence = []
        score = 0.7  # Default

        # Very short responses might be too brief
        if word_count < 20:
            score = 0.5
            evidence.append("Response may be too brief")

        # Very long responses might be verbose
        elif word_count > 1000:
            # Check for repetition
            unique_words = {w.lower() for w in words if len(w) > 3}
            repetition_ratio = len(unique_words) / len(words)

            if repetition_ratio < 0.3:
                score = 0.5
                evidence.append("High repetition detected")
            else:
                score = 0.7  # Long but varied content
                evidence.append("Long but content-rich")

        else:
            score = 0.8
            evidence.append(f"Good length ({word_count} words)")

        # Check for filler phrases
        filler_patterns = [
            r"as mentioned earlier",
            r"it's worth noting",
            r"it should be noted",
            r"let me explain",
            r"as you can see",
            r"basically",
            r"essentially",
        ]
        filler_count = sum(
            len(re.findall(pattern, response.lower())) for pattern in filler_patterns
        )
        if filler_count > 2:
            score -= 0.1 * min(filler_count - 2, 3)
            evidence.append(f"Filler phrases: {filler_count}")

        return DimensionScore(
            dimension=ResponseQualityDimension.CONCISENESS,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.CONCISENESS],
            feedback="Be more concise" if score < 0.6 else "Good conciseness",
            evidence=evidence,
        )

    async def _score_actionability(self, query: str, response: str) -> DimensionScore:
        """Score how actionable the response is."""
        evidence = []
        score = 0.6  # Base score

        query_lower = query.lower()

        # Check if query expects actionable response
        action_keywords = [
            "how",
            "implement",
            "create",
            "fix",
            "solve",
            "make",
            "add",
            "remove",
            "change",
            "update",
        ]
        expects_action = any(kw in query_lower for kw in action_keywords)

        if not expects_action:
            return DimensionScore(
                dimension=ResponseQualityDimension.ACTIONABILITY,
                score=0.8,  # Not applicable, give good default
                weight=self.weights[ResponseQualityDimension.ACTIONABILITY],
                feedback="Action not expected for this query",
                evidence=["Informational query"],
            )

        # Check for action indicators in response
        has_steps = bool(
            re.search(
                r"(?:^|\n)\s*(?:\d+[.):]|Step\s+\d+|First|Second|Then|Finally)",
                response,
            )
        )
        if has_steps:
            score += 0.2
            evidence.append("Contains step-by-step instructions")

        has_code = self._has_code(response)
        if has_code:
            score += 0.15
            evidence.append("Includes code example")

        # Check for command examples
        has_commands = bool(re.search(r"```(?:bash|sh|shell|console)", response))
        if has_commands:
            score += 0.1
            evidence.append("Includes command examples")

        # Check for file paths
        has_file_refs = bool(
            re.search(
                r"(?:in|edit|modify|create)\s+[`\"']?[\w./]+\.(?:py|js|ts|json)",
                response,
            )
        )
        if has_file_refs:
            score += 0.05
            evidence.append("References specific files")

        return DimensionScore(
            dimension=ResponseQualityDimension.ACTIONABILITY,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.ACTIONABILITY],
            feedback="Add concrete steps" if score < 0.7 else "Good actionability",
            evidence=evidence,
        )

    async def _score_coherence(self, response: str) -> DimensionScore:
        """Score logical coherence and structure."""
        evidence = []
        score = 0.7  # Base score

        # Check for logical connectors
        connectors = [
            "therefore",
            "because",
            "however",
            "although",
            "additionally",
            "furthermore",
            "in contrast",
            "similarly",
        ]
        connector_count = sum(1 for c in connectors if c in response.lower())
        if connector_count > 2:
            score += 0.1
            evidence.append(f"Good use of connectors ({connector_count})")

        # Check for consistent formatting
        bullet_lists = len(re.findall(r"^[-*]\s", response, re.MULTILINE))
        numbered_lists = len(re.findall(r"^\d+[.)]", response, re.MULTILINE))

        if bullet_lists > 0 and numbered_lists > 0:
            # Mixed list styles might be confusing
            score -= 0.05
            evidence.append("Mixed list styles")
        elif bullet_lists > 3 or numbered_lists > 3:
            score += 0.1
            evidence.append("Well-structured lists")

        # Check for proper code block formatting
        code_blocks = re.findall(r"```\w*\n.*?```", response, re.DOTALL)
        unclosed_code = response.count("```") % 2 != 0
        if unclosed_code:
            score -= 0.2
            evidence.append("Unclosed code block")
        elif code_blocks:
            evidence.append(f"{len(code_blocks)} properly formatted code blocks")

        return DimensionScore(
            dimension=ResponseQualityDimension.COHERENCE,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.COHERENCE],
            feedback="Improve structure" if score < 0.6 else "Good coherence",
            evidence=evidence,
        )

    async def _score_code_quality(self, response: str) -> DimensionScore:
        """Score quality of code snippets in response."""
        evidence = []
        score = 0.6  # Base score

        # Extract code blocks
        code_blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)

        if not code_blocks:
            return DimensionScore(
                dimension=ResponseQualityDimension.CODE_QUALITY,
                score=0.5,
                weight=self.weights[ResponseQualityDimension.CODE_QUALITY],
                feedback="No code blocks found",
                evidence=["Inline code only"],
            )

        for lang, code in code_blocks:
            code = code.strip()

            # Check for language tag
            if lang:
                score += 0.05
                evidence.append(f"Language specified: {lang}")

            # Check for basic syntax issues
            if lang in ("python", "py"):
                # Check for basic Python issues
                if "import" in code and not code.strip().startswith("import"):
                    if "def " in code or "class " in code:
                        # Imports should typically be at the top
                        pass  # This is okay for snippets
                if code.count("(") != code.count(")"):
                    score -= 0.1
                    evidence.append("Unbalanced parentheses")

            elif lang in ("javascript", "js", "typescript", "ts"):
                if code.count("{") != code.count("}"):
                    score -= 0.1
                    evidence.append("Unbalanced braces")

            # Check for placeholder code
            placeholder_patterns = [
                r"# TODO",
                r"// TODO",
                r"pass\s*$",
                r"\.{3}",
                r"// \.\.\.",
            ]
            for pattern in placeholder_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    score -= 0.1
                    evidence.append("Contains placeholder code")
                    break

            # Check for proper indentation
            lines = code.split("\n")
            has_indentation = any(line.startswith(("  ", "\t")) for line in lines if line.strip())
            if has_indentation:
                evidence.append("Proper indentation")

        evidence.append(f"{len(code_blocks)} code block(s)")

        return DimensionScore(
            dimension=ResponseQualityDimension.CODE_QUALITY,
            score=min(1.0, max(0.0, score)),
            weight=self.weights[ResponseQualityDimension.CODE_QUALITY],
            feedback="Improve code quality" if score < 0.6 else "Good code quality",
            evidence=evidence,
        )

    def _has_code(self, response: str) -> bool:
        """Check if response contains code."""
        return bool(re.search(r"```\w*\n", response))

    def _generate_suggestions(self, dimension_scores: List[DimensionScore]) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []

        for ds in sorted(dimension_scores, key=lambda x: x.score):
            if ds.score < 0.6:
                if ds.dimension == ResponseQualityDimension.RELEVANCE:
                    suggestions.append("Address the specific question more directly")
                elif ds.dimension == ResponseQualityDimension.COMPLETENESS:
                    suggestions.append("Provide more comprehensive coverage of all aspects")
                elif ds.dimension == ResponseQualityDimension.ACCURACY:
                    suggestions.append("Verify claims against actual codebase content")
                elif ds.dimension == ResponseQualityDimension.CONCISENESS:
                    suggestions.append("Remove filler phrases and unnecessary content")
                elif ds.dimension == ResponseQualityDimension.ACTIONABILITY:
                    suggestions.append("Add concrete steps or code examples")
                elif ds.dimension == ResponseQualityDimension.COHERENCE:
                    suggestions.append("Improve logical flow and formatting")
                elif ds.dimension == ResponseQualityDimension.CODE_QUALITY:
                    suggestions.append("Fix syntax issues and remove placeholder code")

        return suggestions[:5]  # Top 5 suggestions


class QualityGate:
    """Quality gate for enforcing response standards.

    Provides a simple interface for checking if responses meet
    quality requirements before presenting to users.
    """

    def __init__(
        self,
        scorer: ResponseQualityScorer,
        min_score: float = 0.6,
        required_dimensions: Optional[List[ResponseQualityDimension]] = None,
    ):
        """Initialize quality gate.

        Args:
            scorer: Quality scorer instance
            min_score: Minimum overall score to pass
            required_dimensions: Dimensions that must pass individually
        """
        self.scorer = scorer
        self.min_score = min_score
        self.required_dimensions = required_dimensions or []

    async def check(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, QualityResult]:
        """Check if response passes quality gate.

        Args:
            query: Original query
            response: Response to check
            context: Optional context

        Returns:
            Tuple of (passes, quality_result)
        """
        result = await self.scorer.score(query, response, context)

        # Check overall score
        if result.overall_score < self.min_score:
            return False, result

        # Check required dimensions
        for dim in self.required_dimensions:
            dim_score = result.get_dimension_score(dim)
            if dim_score is not None and dim_score < self.scorer.config.dimension_threshold:
                return False, result

        return True, result
