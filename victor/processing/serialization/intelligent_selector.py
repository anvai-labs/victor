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

"""Intelligent format selector with learning from historical metrics.

Uses past serialization performance to make better format decisions:
- Tracks which formats work best for each tool
- Adjusts thresholds based on context (token budget, urgency)
- Learns from provider-specific success patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
    DataCharacteristics,
)
from victor.processing.serialization.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class FormatScore:
    """Score for a format candidate."""

    format: SerializationFormat
    base_score: float  # From encoder suitability
    historical_score: float  # From past performance
    context_score: float  # From current context
    final_score: float = 0.0

    def __post_init__(self):
        # Weighted combination
        self.final_score = (
            self.base_score * 0.4 + self.historical_score * 0.35 + self.context_score * 0.25
        )


@dataclass
class SelectionContext:
    """Extended context for intelligent format selection."""

    # Basic context
    tool_name: Optional[str] = None
    tool_operation: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    # Token budget context
    remaining_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    response_token_reserve: int = 4096

    # Data context
    data_size_bytes: int = 0
    estimated_json_tokens: int = 0

    # Urgency flags
    prefer_speed: bool = False  # Minimize encoding time
    prefer_savings: bool = False  # Maximize token savings
    prefer_readability: bool = False  # Prefer human-readable formats

    @property
    def context_pressure(self) -> float:
        """Calculate context pressure (0.0 = plenty of room, 1.0 = critical).

        Returns:
            Float between 0 and 1 indicating context pressure
        """
        if self.remaining_context_tokens is None or self.max_context_tokens is None:
            return 0.5  # Default moderate pressure

        available = self.remaining_context_tokens - self.response_token_reserve
        if available <= 0:
            return 1.0

        # Pressure increases as available tokens decrease
        ratio = available / self.max_context_tokens
        if ratio > 0.5:
            return 0.0  # Plenty of room
        elif ratio > 0.3:
            return 0.3  # Some pressure
        elif ratio > 0.15:
            return 0.6  # Moderate pressure
        else:
            return 0.9  # High pressure


class IntelligentFormatSelector:
    """Intelligent format selector that learns from historical data.

    Uses a multi-factor scoring approach:
    1. Base score: From encoder's suitability for the data
    2. Historical score: From past performance with similar contexts
    3. Context score: From current context (token pressure, urgency)

    The selector also dynamically adjusts thresholds based on:
    - Context pressure (more aggressive when tokens are tight)
    - Tool-specific patterns (learned from metrics)
    - Provider-specific preferences (from capabilities)
    """

    # Format preference order for fallback
    DEFAULT_FORMAT_PRIORITY = [
        SerializationFormat.TOON,
        SerializationFormat.CSV,
        SerializationFormat.JSON_MINIFIED,
        SerializationFormat.MARKDOWN_TABLE,
        SerializationFormat.JSON,
    ]

    # Minimum samples for historical data to be considered reliable
    MIN_SAMPLES_FOR_LEARNING = 5

    def __init__(self):
        """Initialize selector."""
        self._format_cache: Dict[str, SerializationFormat] = {}
        self._tool_format_stats: Dict[str, Dict[str, float]] = {}

    def select_best_format(
        self,
        data: Any,
        characteristics: DataCharacteristics,
        config: SerializationConfig,
        context: SelectionContext,
        encoder_scores: Dict[SerializationFormat, float],
    ) -> Tuple[SerializationFormat, str]:
        """Select the best format based on all available information.

        Args:
            data: Data to serialize
            characteristics: Data analysis results
            config: Serialization configuration
            context: Extended selection context
            encoder_scores: Suitability scores from encoders

        Returns:
            Tuple of (selected format, selection reason)
        """
        # Check for forced format
        if config.preferred_format:
            return config.preferred_format, "config_preferred"

        # Build candidate list from allowed formats
        candidates = self._build_candidates(
            encoder_scores,
            config,
            context,
            characteristics,
        )

        if not candidates:
            return config.fallback_format, "no_candidates"

        # Select best candidate
        best = max(candidates, key=lambda c: c.final_score)

        # Apply threshold check
        adjusted_threshold = self._get_adjusted_threshold(config, context)
        if best.final_score < adjusted_threshold:
            # Score too low, use fallback
            return config.fallback_format, f"below_threshold_{best.final_score:.2f}"

        return best.format, f"selected_score_{best.final_score:.2f}"

    def _build_candidates(
        self,
        encoder_scores: Dict[SerializationFormat, float],
        config: SerializationConfig,
        context: SelectionContext,
        characteristics: DataCharacteristics,
    ) -> List[FormatScore]:
        """Build scored candidate list.

        Args:
            encoder_scores: Base suitability scores from encoders
            config: Configuration
            context: Selection context
            characteristics: Data characteristics

        Returns:
            List of FormatScore candidates
        """
        candidates = []

        for fmt in config.allowed_formats:
            if fmt in config.disabled_formats:
                continue

            base_score = encoder_scores.get(fmt, 0.0)
            if base_score <= 0:
                continue

            historical_score = self._get_historical_score(fmt, context)
            context_score = self._get_context_score(fmt, context, characteristics)

            candidates.append(
                FormatScore(
                    format=fmt,
                    base_score=base_score,
                    historical_score=historical_score,
                    context_score=context_score,
                )
            )

        return candidates

    def _get_historical_score(
        self,
        fmt: SerializationFormat,
        context: SelectionContext,
    ) -> float:
        """Get historical performance score for format.

        Args:
            fmt: Format to score
            context: Selection context

        Returns:
            Historical score between 0 and 1
        """
        if not context.tool_name:
            return 0.5  # Neutral if no tool context

        # Try to get cached stats
        cache_key = f"{context.tool_name}:{context.provider or 'any'}"
        if cache_key not in self._tool_format_stats:
            self._tool_format_stats[cache_key] = self._load_tool_format_stats(
                context.tool_name, context.provider
            )

        stats = self._tool_format_stats.get(cache_key, {})
        format_key = fmt.value

        if format_key in stats:
            # Normalize savings to 0-1 scale
            avg_savings = stats[format_key].get("avg_savings", 0)
            usage_count = stats[format_key].get("count", 0)

            if usage_count >= self.MIN_SAMPLES_FOR_LEARNING:
                # More samples = more confidence
                confidence = min(usage_count / 20, 1.0)
                return (avg_savings / 100) * confidence

        return 0.5  # Neutral score for unknown

    def _load_tool_format_stats(
        self,
        tool_name: str,
        provider: Optional[str],
    ) -> Dict[str, Dict[str, float]]:
        """Load format statistics for a tool from metrics.

        Args:
            tool_name: Tool name
            provider: Provider name

        Returns:
            Dictionary of format -> stats
        """
        try:
            collector = get_metrics_collector()

            # Get per-format stats for this tool
            with collector._get_connection() as conn:
                query = """
                    SELECT
                        format_selected,
                        COUNT(*) as count,
                        AVG(token_savings_percent) as avg_savings,
                        AVG(encoding_time_ms) as avg_time
                    FROM serialization_metrics
                    WHERE tool_name = ?
                """
                params = [tool_name]

                if provider:
                    query += " AND provider = ?"
                    params.append(provider)

                query += " GROUP BY format_selected"

                cursor = conn.execute(query, params)
                stats = {}
                for row in cursor.fetchall():
                    stats[row["format_selected"]] = {
                        "count": row["count"],
                        "avg_savings": row["avg_savings"] or 0,
                        "avg_time": row["avg_time"] or 0,
                    }
                return stats

        except Exception as e:
            logger.debug(f"Failed to load tool format stats: {e}")
            return {}

    def _get_context_score(
        self,
        fmt: SerializationFormat,
        context: SelectionContext,
        characteristics: DataCharacteristics,
    ) -> float:
        """Get context-based score for format.

        Args:
            fmt: Format to score
            context: Selection context
            characteristics: Data characteristics

        Returns:
            Context score between 0 and 1
        """
        score = 0.5  # Start neutral

        # Adjust based on context pressure
        pressure = context.context_pressure

        if pressure > 0.7:
            # High pressure: favor most compact formats
            compact_formats = {
                SerializationFormat.CSV: 0.9,
                SerializationFormat.TOON: 0.85,
                SerializationFormat.JSON_MINIFIED: 0.7,
                SerializationFormat.REFERENCE_ENCODED: 0.75,
            }
            score = compact_formats.get(fmt, 0.3)

        elif pressure > 0.4:
            # Moderate pressure: balanced approach
            balanced_formats = {
                SerializationFormat.TOON: 0.8,
                SerializationFormat.CSV: 0.75,
                SerializationFormat.JSON_MINIFIED: 0.7,
                SerializationFormat.MARKDOWN_TABLE: 0.6,
            }
            score = balanced_formats.get(fmt, 0.5)

        # Adjust for urgency flags
        if context.prefer_speed:
            # Simple formats are faster
            fast_formats = {
                SerializationFormat.JSON_MINIFIED: 0.1,
                SerializationFormat.JSON: 0.1,
                SerializationFormat.CSV: 0.05,
            }
            score += fast_formats.get(fmt, 0.0)

        if context.prefer_savings:
            # Compact formats get bonus
            savings_bonus = {
                SerializationFormat.CSV: 0.15,
                SerializationFormat.TOON: 0.12,
                SerializationFormat.REFERENCE_ENCODED: 0.1,
            }
            score += savings_bonus.get(fmt, 0.0)

        if context.prefer_readability:
            # Readable formats get bonus
            readability_bonus = {
                SerializationFormat.MARKDOWN_TABLE: 0.15,
                SerializationFormat.JSON: 0.1,
            }
            score += readability_bonus.get(fmt, 0.0)

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    def _get_adjusted_threshold(
        self,
        config: SerializationConfig,
        context: SelectionContext,
    ) -> float:
        """Get dynamically adjusted threshold based on context.

        Args:
            config: Configuration with base threshold
            context: Selection context

        Returns:
            Adjusted threshold
        """
        base_threshold = config.min_savings_threshold
        pressure = context.context_pressure

        # Lower threshold when context is tight (more aggressive)
        if pressure > 0.7:
            return base_threshold * 0.5  # Much more aggressive
        elif pressure > 0.4:
            return base_threshold * 0.75  # Somewhat more aggressive
        else:
            return base_threshold

    def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """Invalidate cached statistics.

        Args:
            tool_name: Specific tool to invalidate, or None for all
        """
        if tool_name:
            keys_to_remove = [k for k in self._tool_format_stats if k.startswith(tool_name)]
            for key in keys_to_remove:
                del self._tool_format_stats[key]
        else:
            self._tool_format_stats.clear()


# Global instance
_selector: Optional[IntelligentFormatSelector] = None


def get_intelligent_selector() -> IntelligentFormatSelector:
    """Get global intelligent selector.

    Returns:
        IntelligentFormatSelector instance
    """
    global _selector
    if _selector is None:
        _selector = IntelligentFormatSelector()
    return _selector


def reset_intelligent_selector() -> None:
    """Reset global selector (for testing)."""
    global _selector
    _selector = None
