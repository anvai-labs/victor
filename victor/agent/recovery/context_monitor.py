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

"""Context window monitor for proactive compaction.

This module monitors context window usage and provides:
- Early warning when approaching limits
- Proactive compaction recommendations
- Smart truncation strategies
- Token estimation and tracking

Proactive compaction helps prevent:
- Model confusion from overly long contexts
- Stuck loops caused by context overflow
- Degraded response quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContextHealth(Enum):
    """Context window health status."""

    HEALTHY = auto()  # <50% capacity
    WARNING = auto()  # 50-70% capacity
    CRITICAL = auto()  # 70-85% capacity
    OVERFLOW = auto()  # >85% capacity


class CompactionStrategy(Enum):
    """Strategies for context compaction."""

    NONE = auto()  # No compaction needed
    TRUNCATE_OLD_MESSAGES = auto()  # Remove oldest messages
    SUMMARIZE_TOOL_OUTPUTS = auto()  # Compress tool outputs
    SUMMARIZE_CONVERSATION = auto()  # Full conversation summary
    REMOVE_REDUNDANT = auto()  # Remove duplicate/redundant content
    AGGRESSIVE = auto()  # Combination of all strategies


@dataclass
class ContextMetrics:
    """Metrics for context window usage."""

    total_tokens: int = 0
    max_tokens: int = 100000
    message_count: int = 0
    tool_output_tokens: int = 0
    system_prompt_tokens: int = 0
    user_message_tokens: int = 0
    assistant_message_tokens: int = 0

    @property
    def usage_ratio(self) -> float:
        """Context usage as a ratio (0-1)."""
        return self.total_tokens / max(self.max_tokens, 1)

    @property
    def health(self) -> ContextHealth:
        """Current health status."""
        ratio = self.usage_ratio
        if ratio < 0.5:
            return ContextHealth.HEALTHY
        elif ratio < 0.7:
            return ContextHealth.WARNING
        elif ratio < 0.85:
            return ContextHealth.CRITICAL
        else:
            return ContextHealth.OVERFLOW

    @property
    def available_tokens(self) -> int:
        """Tokens available for new content."""
        return max(0, self.max_tokens - self.total_tokens)


@dataclass
class CompactionRecommendation:
    """Recommendation for context compaction."""

    strategy: CompactionStrategy
    urgency: ContextHealth
    target_reduction_tokens: int
    reason: str
    specific_actions: List[str] = field(default_factory=list)


class ContextWindowMonitor:
    """Monitor for context window health and proactive compaction.

    Features:
    - Real-time token tracking
    - Proactive compaction recommendations
    - Health status monitoring
    - Integration with compaction system

    Follows:
    - Single Responsibility: Only monitors context, doesn't compact
    - Interface Segregation: Clear boundaries with compaction system
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "warning": 0.5,  # 50% - start monitoring closely
        "critical": 0.7,  # 70% - recommend compaction
        "overflow": 0.85,  # 85% - force compaction
    }

    # Token estimation (chars per token varies by model)
    TOKEN_RATIOS = {
        "claude": 3.5,  # ~3.5 chars per token
        "gpt": 4.0,  # ~4 chars per token
        "llama": 3.8,
        "qwen": 3.5,
        "default": 4.0,
    }

    def __init__(
        self,
        max_context_tokens: int = 100000,
        response_reserve: int = 4096,
        model_name: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        compaction_callback: Optional[Callable[[], int]] = None,
    ):
        self._max_tokens = max_context_tokens - response_reserve
        self._response_reserve = response_reserve
        self._model_name = model_name or "default"
        self._thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        self._compaction_callback = compaction_callback

        # Current metrics
        self._metrics = ContextMetrics(max_tokens=self._max_tokens)

        # History for trend analysis
        self._token_history: List[Tuple[int, int]] = []  # (timestamp_ms, tokens)
        self._compaction_history: List[Dict[str, Any]] = []

    def _get_token_ratio(self) -> float:
        """Get token ratio for current model."""
        model_lower = self._model_name.lower()
        for model_key, ratio in self.TOKEN_RATIOS.items():
            if model_key in model_lower:
                return ratio
        return self.TOKEN_RATIOS["default"]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        return int(len(text) / self._get_token_ratio())

    def update_metrics(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> ContextMetrics:
        """Update metrics from message list.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt

        Returns:
            Updated ContextMetrics
        """
        metrics = ContextMetrics(max_tokens=self._max_tokens)

        # System prompt tokens
        if system_prompt:
            metrics.system_prompt_tokens = self.estimate_tokens(system_prompt)

        # Process messages
        metrics.message_count = len(messages)

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle multimodal content
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))

            tokens = self.estimate_tokens(str(content))

            if role == "user":
                metrics.user_message_tokens += tokens
            elif role == "assistant":
                metrics.assistant_message_tokens += tokens
            elif role == "tool":
                metrics.tool_output_tokens += tokens

        metrics.total_tokens = (
            metrics.system_prompt_tokens
            + metrics.user_message_tokens
            + metrics.assistant_message_tokens
            + metrics.tool_output_tokens
        )

        self._metrics = metrics

        # Record history
        import time

        self._token_history.append((int(time.time() * 1000), metrics.total_tokens))
        if len(self._token_history) > 100:
            self._token_history.pop(0)

        return metrics

    def get_health(self) -> ContextHealth:
        """Get current context health status."""
        return self._metrics.health

    def get_metrics(self) -> ContextMetrics:
        """Get current metrics."""
        return self._metrics

    def should_compact(self) -> Tuple[bool, str]:
        """Check if compaction should be triggered.

        Returns:
            Tuple of (should_compact, reason)
        """
        health = self._metrics.health

        if health == ContextHealth.OVERFLOW:
            return True, f"Context overflow ({self._metrics.usage_ratio:.1%} capacity)"

        if health == ContextHealth.CRITICAL:
            # Check if growing quickly
            if self._is_growing_fast():
                return (
                    True,
                    f"Context critical and growing fast ({self._metrics.usage_ratio:.1%})",
                )
            return True, f"Context critical ({self._metrics.usage_ratio:.1%})"

        if health == ContextHealth.WARNING and self._is_growing_fast():
            return (
                True,
                f"Context warning with rapid growth ({self._metrics.usage_ratio:.1%})",
            )

        return False, "Context healthy"

    def _is_growing_fast(self) -> bool:
        """Check if context is growing rapidly."""
        if len(self._token_history) < 3:
            return False

        # Compare last 3 readings
        recent = [t[1] for t in self._token_history[-3:]]
        growth_rate = (recent[-1] - recent[0]) / max(recent[0], 1)

        # Growing more than 10% in last few operations
        return growth_rate > 0.1

    def get_compaction_recommendation(self) -> CompactionRecommendation:
        """Get recommendation for context compaction."""
        health = self._metrics.health
        metrics = self._metrics

        if health == ContextHealth.HEALTHY:
            return CompactionRecommendation(
                strategy=CompactionStrategy.NONE,
                urgency=health,
                target_reduction_tokens=0,
                reason="Context is healthy",
            )

        # Calculate target reduction
        target_ratio = self._thresholds["warning"] * 0.8  # Target 40% capacity
        target_tokens = int(metrics.max_tokens * target_ratio)
        reduction_needed = metrics.total_tokens - target_tokens

        if health == ContextHealth.WARNING:
            # Light compaction
            strategy = CompactionStrategy.TRUNCATE_OLD_MESSAGES
            actions = [
                "Remove oldest non-essential messages",
                "Truncate verbose tool outputs",
            ]
        elif health == ContextHealth.CRITICAL:
            # Medium compaction
            strategy = CompactionStrategy.SUMMARIZE_TOOL_OUTPUTS
            actions = [
                "Summarize tool outputs to key findings",
                "Remove older conversation turns",
                "Compress repetitive content",
            ]
        else:  # OVERFLOW
            # Aggressive compaction
            strategy = CompactionStrategy.AGGRESSIVE
            actions = [
                "Full conversation summarization",
                "Keep only last 3-5 messages",
                "Remove all intermediate tool outputs",
                "Generate condensed context summary",
            ]

        return CompactionRecommendation(
            strategy=strategy,
            urgency=health,
            target_reduction_tokens=reduction_needed,
            reason=f"Context at {metrics.usage_ratio:.1%} capacity",
            specific_actions=actions,
        )

    def trigger_compaction(self) -> Optional[int]:
        """Trigger compaction if callback is registered.

        Returns:
            Tokens freed, or None if no callback
        """
        if not self._compaction_callback:
            return None

        before_tokens = self._metrics.total_tokens
        freed = self._compaction_callback()

        # Record compaction
        import time

        self._compaction_history.append(
            {
                "timestamp": time.time(),
                "before_tokens": before_tokens,
                "freed_tokens": freed,
                "health_before": self._metrics.health.name,
            }
        )

        logger.info(f"Compaction freed {freed} tokens (was {before_tokens})")
        return freed

    def register_compaction_callback(
        self,
        callback: Callable[[], int],
    ) -> None:
        """Register a callback for compaction.

        Callback should return the number of tokens freed.
        """
        self._compaction_callback = callback

    def get_token_breakdown(self) -> Dict[str, Any]:
        """Get detailed token breakdown."""
        metrics = self._metrics
        return {
            "total": metrics.total_tokens,
            "max": metrics.max_tokens,
            "available": metrics.available_tokens,
            "usage_ratio": metrics.usage_ratio,
            "health": metrics.health.name,
            "breakdown": {
                "system_prompt": metrics.system_prompt_tokens,
                "user_messages": metrics.user_message_tokens,
                "assistant_messages": metrics.assistant_message_tokens,
                "tool_outputs": metrics.tool_output_tokens,
            },
            "message_count": metrics.message_count,
        }

    def get_growth_trend(self) -> Dict[str, Any]:
        """Get context growth trend analysis."""
        if len(self._token_history) < 2:
            return {"trend": "insufficient_data", "samples": len(self._token_history)}

        # Calculate growth metrics
        tokens = [t[1] for t in self._token_history]
        times = [t[0] for t in self._token_history]

        growth_total = tokens[-1] - tokens[0]
        time_span_ms = times[-1] - times[0]
        growth_rate_per_sec = growth_total / (time_span_ms / 1000) if time_span_ms > 0 else 0

        # Determine trend
        if growth_rate_per_sec > 100:
            trend = "rapid_growth"
        elif growth_rate_per_sec > 50:
            trend = "moderate_growth"
        elif growth_rate_per_sec > 0:
            trend = "slow_growth"
        elif growth_rate_per_sec < -50:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "growth_rate_tokens_per_sec": growth_rate_per_sec,
            "total_growth": growth_total,
            "time_span_seconds": time_span_ms / 1000,
            "samples": len(self._token_history),
        }

    def predict_overflow_time(self) -> Optional[float]:
        """Predict when context will overflow (in seconds).

        Returns:
            Estimated seconds until overflow, or None if not growing
        """
        trend = self.get_growth_trend()

        if trend["trend"] in ("stable", "decreasing", "insufficient_data"):
            return None

        growth_rate = trend["growth_rate_tokens_per_sec"]
        if growth_rate <= 0:
            return None

        _remaining_tokens = self._metrics.available_tokens  # noqa: F841
        overflow_threshold = self._metrics.max_tokens * self._thresholds["overflow"]
        tokens_until_overflow = overflow_threshold - self._metrics.total_tokens

        if tokens_until_overflow <= 0:
            return 0.0

        return tokens_until_overflow / growth_rate

    def reset(self) -> None:
        """Reset monitor state."""
        self._metrics = ContextMetrics(max_tokens=self._max_tokens)
        self._token_history.clear()
