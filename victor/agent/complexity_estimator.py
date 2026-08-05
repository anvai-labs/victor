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

"""Query complexity estimation using edge model.

Uses the fast edge model (qwen3.5:2b) to estimate query complexity
more accurately than simple heuristics. This improves routing decisions
for both PlanningGate and ParadigmRouter.

The edge model is called with a micro-prompt to score complexity on
a scale of 0-1, considering:
- Task complexity (number of steps, dependencies)
- Domain knowledge required
- Reasoning depth needed
- Ambiguity or uncertainty
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ComplexityBand(str, Enum):
    """Complexity bands for categorization.

    - TRIVIAL: 0.0-0.2 (single action, no dependencies)
    - SIMPLE: 0.2-0.4 (few steps, minimal dependencies)
    - MODERATE: 0.4-0.6 (multiple steps, some dependencies)
    - COMPLEX: 0.6-0.8 (many steps, significant dependencies)
    - EXPERT: 0.8-1.0 (requires deep expertise, extensive reasoning)
    """

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

    @classmethod
    def from_score(cls, score: float) -> "ComplexityBand":
        """Get complexity band from score."""
        if score < 0.2:
            return cls.TRIVIAL
        elif score < 0.4:
            return cls.SIMPLE
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.COMPLEX
        else:
            return cls.EXPERT


@dataclass
class ComplexityEstimate:
    """Complexity estimate for a query.

    Attributes:
        score: Complexity score (0-1, higher is more complex)
        band: Complexity band categorization
        confidence: Confidence in estimate (0-1)
        reasoning: Human-readable explanation
        latency_ms: Time taken to estimate
    """

    score: float
    band: ComplexityBand
    confidence: float
    reasoning: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "band": self.band.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


class ComplexityEstimator:
    """Estimates query complexity using edge model.

    Uses the fast edge model (qwen3.5:2b) to provide more accurate
    complexity estimates than simple heuristics.

    Micro-prompt (~60 tokens) ensures fast response (<100ms typically).

    Example:
        estimator = ComplexityEstimator()
        estimate = await estimator.estimate("create a new file")
        # Returns: score=0.1, band=TRIVIAL, confidence=0.95
    """

    # Complexity estimation micro-prompt
    COMPLEXITY_PROMPT = """Rate the complexity of this query on a scale of 0-1:

Query: {query}

Consider:
- Number of steps required
- Dependencies on other components
- Domain knowledge needed
- Reasoning depth required
- Ambiguity or uncertainty

Respond with:
COMPLEXITY: <0.0-1.0>
BAND: <TRIVIAL|SIMPLE|MODERATE|COMPLEX|EXPERT>
CONFIDENCE: <0.0-1.0>
REASONING: <one sentence>"""

    def __init__(
        self,
        enabled: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """Initialize the complexity estimator.

        Args:
            enabled: Whether the estimator is enabled
            use_cache: Whether to cache estimates
            cache_ttl: Cache time-to-live in seconds
        """
        self.enabled = enabled
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[ComplexityEstimate, float]] = {}
        self._estimate_count = 0
        self._cache_hits = 0

    async def estimate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComplexityEstimate:
        """Estimate query complexity using edge model.

        Args:
            query: User's query
            context: Optional execution context

        Returns:
            ComplexityEstimate with score, band, and reasoning
        """
        import time

        self._estimate_count += 1
        start_time = time.time()

        if not self.enabled:
            # Fallback to heuristic-based estimation
            return self._heuristic_estimate(query, start_time)

        # Check cache
        if self.use_cache:
            cached = self._get_from_cache(query)
            if cached:
                self._cache_hits += 1
                logger.debug("[ComplexityEstimator] Cache hit for query")
                return cached

        # Use edge model for estimation
        try:
            estimate = await self._edge_model_estimate(query, context, start_time)

            # Cache the result
            if self.use_cache:
                self._add_to_cache(query, estimate)

            return estimate

        except Exception as e:
            logger.warning(f"[ComplexityEstimator] Edge model failed: {e}, using heuristic")
            return self._heuristic_estimate(query, start_time)

    def _heuristic_estimate(self, query: str, start_time: float) -> ComplexityEstimate:
        """Fallback heuristic-based complexity estimation.

        Args:
            query: User's query
            start_time: Estimation start time

        Returns:
            ComplexityEstimate based on heuristics
        """
        import time

        # Simple heuristics
        query_lower = query.lower()
        score = 0.3  # Base complexity

        # Increase complexity for certain patterns
        if any(word in query_lower for word in ["design", "architecture", "system"]):
            score += 0.3
        if any(word in query_lower for word in ["debug", "fix", "error", "bug"]):
            score += 0.2
        if any(word in query_lower for word in ["analyze", "understand", "review"]):
            score += 0.2
        if len(query) > 200:
            score += 0.1

        # Decrease for simple actions
        if any(word in query_lower for word in ["create", "write", "run", "list"]):
            score -= 0.1

        score = max(0.0, min(1.0, score))

        latency_ms = (time.time() - start_time) * 1000

        return ComplexityEstimate(
            score=score,
            band=ComplexityBand.from_score(score),
            confidence=0.7,  # Lower confidence for heuristics
            reasoning="Heuristic-based estimation",
            latency_ms=latency_ms,
        )

    async def _edge_model_estimate(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        start_time: float,
    ) -> ComplexityEstimate:
        """Use edge model to estimate complexity.

        Args:
            query: User's query
            context: Optional execution context
            start_time: Estimation start time

        Returns:
            ComplexityEstimate from edge model
        """
        import time

        # Import here to avoid circular dependency
        from victor.framework.agentic_loop import decide_sync

        # Prepare prompt
        prompt = self.COMPLEXITY_PROMPT.format(query=query)

        # Call edge model
        try:
            response = decide_sync(
                decision_type="complexity_estimation",
                context={"query": query, "prompt": prompt},
            )

            # Parse response
            score = self._extract_score(response)
            band = self._extract_band(response)
            confidence = self._extract_confidence(response)
            reasoning = self._extract_reasoning(response)

            latency_ms = (time.time() - start_time) * 1000

            estimate = ComplexityEstimate(
                score=score,
                band=band,
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency_ms,
            )

            logger.info(
                f"[ComplexityEstimator] Edge model: score={score:.2f}, "
                f"band={band.value}, latency={latency_ms:.1f}ms"
            )

            return estimate

        except Exception as e:
            logger.error(f"[ComplexityEstimator] Edge model error: {e}")
            raise

    def _extract_score(self, response: str) -> float:
        """Extract complexity score from response."""
        import re

        match = re.search(r"COMPLEXITY:\s*([0-9.]+)", response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5  # Default if not found

    def _extract_band(self, response: str) -> ComplexityBand:
        """Extract complexity band from response."""
        import re

        match = re.search(r"BAND:\s*(\w+)", response, re.IGNORECASE)
        if match:
            band_str = match.group(1).upper()
            try:
                return ComplexityBand(band_str)
            except ValueError:
                pass
        return ComplexityBand.MODERATE  # Default

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

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        import re

        match = re.search(r"REASONING:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "No reasoning provided"

    def _get_from_cache(self, query: str) -> Optional[ComplexityEstimate]:
        """Get estimate from cache if available."""
        import time

        if query in self._cache:
            estimate, timestamp = self._cache[query]
            if time.time() - timestamp < self.cache_ttl:
                return estimate
            else:
                # Expired, remove from cache
                del self._cache[query]
        return None

    def _add_to_cache(self, query: str, estimate: ComplexityEstimate) -> None:
        """Add estimate to cache."""
        import time

        self._cache[query] = (estimate, time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get estimator statistics.

        Returns:
            Dict with estimation counts and cache statistics
        """
        cache_hit_rate = (
            (self._cache_hits / self._estimate_count) if self._estimate_count > 0 else 0.0
        )

        return {
            "total_estimates": self._estimate_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._estimate_count = 0
        self._cache_hits = 0
        self._cache.clear()


# Singleton instance
_complexity_estimator_instance: Optional[ComplexityEstimator] = None


def get_complexity_estimator() -> ComplexityEstimator:
    """Get the singleton ComplexityEstimator instance.

    Returns:
        ComplexityEstimator singleton instance
    """
    global _complexity_estimator_instance
    if _complexity_estimator_instance is None:
        _complexity_estimator_instance = ComplexityEstimator()
    return _complexity_estimator_instance


__all__ = [
    "ComplexityEstimator",
    "ComplexityEstimate",
    "ComplexityBand",
    "get_complexity_estimator",
]
