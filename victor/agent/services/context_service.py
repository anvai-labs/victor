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

"""Context service implementation.

Extracts context management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Context size monitoring and metrics
- Context overflow detection and prevention
- Context compaction and optimization
- Message history management
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.agent.runtime.context import AgentRuntimeContext

logger = logging.getLogger(__name__)

# Message is a dict with 'role' and 'content' keys
Message = Dict[str, Any]
_VALID_COMPACTION_STRATEGIES = frozenset({"simple", "tiered", "semantic", "hybrid"})
_LOCAL_PROVIDER_NAMES = frozenset({"ollama", "lmstudio", "llamacpp"})


@dataclass(frozen=True)
class ContextCompactionRun:
    """Outcome from a context-service-owned compaction policy check."""

    handled: bool
    should_compact: bool = False
    messages_removed: int = 0
    recommendation: Optional[Dict[str, Any]] = None


async def compact_context_if_recommended(
    context_service: Any,
    *,
    strategy: str = "tiered",
    min_messages: int = 6,
    task_complexity: Optional[str] = None,
) -> ContextCompactionRun:
    """Run the common ContextService recommendation and compaction flow.

    Returns handled=False when the object does not expose the required
    ContextService-like API so callers can fall back to legacy compaction.
    """
    recommendation_getter = getattr(context_service, "get_compaction_recommendation", None)
    compact_context = getattr(context_service, "compact_context", None)
    if not callable(recommendation_getter) or not callable(compact_context):
        return ContextCompactionRun(handled=False)

    recommendation = recommendation_getter()
    if inspect.isawaitable(recommendation):
        recommendation = await recommendation
    if not isinstance(recommendation, dict):
        return ContextCompactionRun(handled=True)

    should_compact = bool(recommendation.get("should_compact", False))
    if not should_compact:
        return ContextCompactionRun(
            handled=True,
            should_compact=False,
            recommendation=recommendation,
        )

    removed = compact_context(strategy=strategy, min_messages=min_messages)
    if inspect.isawaitable(removed):
        removed = await removed
    return ContextCompactionRun(
        handled=True,
        should_compact=True,
        messages_removed=int(removed or 0),
        recommendation=recommendation,
    )


def _normalize_compaction_strategy(strategy: str) -> str:
    """Normalize arbitrary strategy values to supported compaction strategies."""
    normalized = str(strategy or "").strip().lower()
    return normalized if normalized in _VALID_COMPACTION_STRATEGIES else "tiered"


def _resolve_tool_output_compaction_strategy(
    *,
    default_strategy: str,
    provider_name: str,
    model_name: str,
    task_type: str,
    estimated_output_tokens: int,
) -> str:
    """Resolve a compaction strategy for large incoming tool-output payloads."""
    strategy = _normalize_compaction_strategy(default_strategy)
    provider = str(provider_name or "").strip().lower()
    model = str(model_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    is_analysis_like = "analysis" in task or "review" in task or task == "analyze"

    if "deepseek" in provider or "deepseek" in model:
        return "hybrid"
    if is_analysis_like and estimated_output_tokens >= 3000:
        return "semantic"
    if provider in _LOCAL_PROVIDER_NAMES and estimated_output_tokens >= 3000:
        return "simple"
    return strategy


def _build_tool_output_compaction_decision(
    *,
    estimated_output_tokens: int,
    remaining_tokens: int,
    utilization_percent: float,
    recommendation: Optional[Dict[str, Any]],
    provider_name: str,
    model_name: str,
    task_type: str,
    default_strategy: str,
) -> Dict[str, Any]:
    """Build a structured decision for pre-tool-output compaction."""
    estimated = max(0, int(estimated_output_tokens or 0))
    remaining = max(0, int(remaining_tokens or 0))
    utilization = max(0.0, float(utilization_percent or 0.0))
    recommendation = recommendation if isinstance(recommendation, dict) else {}

    should_compact = False
    policy_reason = ""
    if estimated > 0 and bool(recommendation.get("should_compact", False)):
        should_compact = True
        policy_reason = "base_context_pressure"
    elif estimated > 0 and utilization >= 75.0 and estimated >= 1200:
        should_compact = True
        policy_reason = "high_utilization_large_tool_output"
    elif estimated > 0 and remaining > 0 and estimated >= max(1, int(remaining * 0.6)):
        should_compact = True
        policy_reason = "tool_output_exceeds_remaining_budget"

    return {
        "should_compact": should_compact,
        "compacted": False,
        "messages_removed": 0,
        "saved_tokens": 0,
        "estimated_output_tokens": estimated,
        "remaining_tokens": remaining,
        "utilization_percent": utilization,
        "strategy": _resolve_tool_output_compaction_strategy(
            default_strategy=default_strategy,
            provider_name=provider_name,
            model_name=model_name,
            task_type=task_type,
            estimated_output_tokens=estimated,
        ),
        "reason": "pre_tool_output",
        "policy_reason": policy_reason,
    }


class UtilizationTrend(str, Enum):
    """Context utilization trend direction."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


class CompactionFrequency(str, Enum):
    """Context compaction frequency category."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CompactionUrgency(str, Enum):
    """Context compaction urgency level."""

    NOW = "now"
    SOON = "soon"
    WITHIN_5_MINUTES = "within_5_minutes"
    NO_URGENCY = "no_urgency"


class ContextServiceConfig:
    """Configuration for ContextService.

    Attributes:
        max_tokens: Maximum context tokens
        min_messages_to_keep: Minimum messages to retain after compaction
        default_compaction_strategy: Default compaction strategy
        overflow_threshold_percent: Threshold for overflow warning
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        min_messages_to_keep: int = 6,
        default_compaction_strategy: str = "tiered",
        overflow_threshold_percent: float = 90.0,
    ):
        self.max_tokens = max_tokens
        self.min_messages_to_keep = min_messages_to_keep
        self.default_compaction_strategy = default_compaction_strategy
        self.overflow_threshold_percent = overflow_threshold_percent


class ContextMetricsImpl:
    """Implementation of context metrics."""

    def __init__(
        self,
        total_tokens: int,
        message_count: int,
        user_message_count: int,
        assistant_message_count: int,
        tool_result_count: int,
        system_prompt_tokens: int,
        max_tokens: int,
    ):
        self.total_tokens = total_tokens
        self.message_count = message_count
        self.user_message_count = user_message_count
        self.assistant_message_count = assistant_message_count
        self.tool_result_count = tool_result_count
        self.system_prompt_tokens = system_prompt_tokens
        self._max_tokens = max_tokens

    @property
    def utilization_percent(self) -> float:
        """Context utilization as percentage."""
        return (self.total_tokens / self._max_tokens * 100) if self._max_tokens > 0 else 0


class ContextService:
    """[CANONICAL] Service for context and state management.

    The target implementation for context operations following the
    state-passed architectural pattern. Supersedes logic previously
    managed by ConversationController and ContextCompactor.

    This service follows SOLID principles:
    - SRP: Only handles context operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements ContextServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        config = ContextServiceConfig()
        service = ContextService(config=config)

        metrics = await service.get_context_metrics()
        if await service.check_context_overflow():
            await service.compact_context()
    """

    def __init__(self, config: ContextServiceConfig):
        """Initialize the context service.

        Args:
            config: Service configuration
        """
        self._config = config
        self._messages: List["Message"] = []
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

        # Initialize metrics tracking
        self._metrics: Dict[str, Any] = {
            "compaction_count": 0,
            "overflow_count": 0,
            "last_compaction_time": None,
            "last_overflow_time": None,
            "utilization_history": [],
            "avg_compaction_time": 0.0,
            "operation_count": 0,
            "cache_hit_rate": 0.0,
            "last_compaction_saved": 0,
            "total_tokens_saved": 0,
        }

    @staticmethod
    def _message_value(message: Any, key: str, default: Any = None) -> Any:
        """Read a field from either dict-backed or attribute-backed messages."""
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    async def get_context_metrics(self) -> ContextMetricsImpl:
        """Get current context metrics.

        Returns:
            ContextMetrics with current context information
        """
        total_tokens = sum(
            self._estimate_tokens(self._message_value(m, "content", "")) for m in self._messages
        )

        user_count = sum(1 for m in self._messages if self._message_value(m, "role", "") == "user")
        assistant_count = sum(
            1 for m in self._messages if self._message_value(m, "role", "") == "assistant"
        )
        tool_count = sum(1 for m in self._messages if self._message_value(m, "role", "") == "tool")

        system_tokens = self._estimate_tokens(
            next(
                (
                    self._message_value(m, "content", "")
                    for m in self._messages
                    if self._message_value(m, "role", "") == "system"
                ),
                "",
            )
        )

        return ContextMetricsImpl(
            total_tokens=total_tokens,
            message_count=len(self._messages),
            user_message_count=user_count,
            assistant_message_count=assistant_count,
            tool_result_count=tool_count,
            system_prompt_tokens=system_tokens,
            max_tokens=self._config.max_tokens,
        )

    async def check_context_overflow(self) -> bool:
        """Check if context exceeds limits.

        Returns:
            True if context exceeds limits, False otherwise
        """
        metrics = await self.get_context_metrics()
        return metrics.total_tokens > self._config.max_tokens

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context to fit within limits.

        Args:
            strategy: Compaction strategy
            min_messages: Minimum messages to retain

        Returns:
            Number of messages removed
        """
        started_at = time.time()
        original_count = len(self._messages)
        original_tokens = self.get_context_size()

        if original_count <= min_messages:
            return 0

        # Keep the most recent messages
        self._messages = self._messages[-min_messages:]

        removed = original_count - len(self._messages)
        saved_tokens = max(0, original_tokens - self.get_context_size())
        self._metrics["operation_count"] = int(self._metrics.get("operation_count", 0) or 0) + 1
        if removed > 0:
            self._metrics["compaction_count"] = (
                int(self._metrics.get("compaction_count", 0) or 0) + 1
            )
            self._metrics["last_compaction_time"] = started_at
            self._metrics["last_compaction_saved"] = saved_tokens
            self._metrics["total_tokens_saved"] = (
                int(self._metrics.get("total_tokens_saved", 0) or 0) + saved_tokens
            )
            compaction_duration = max(0.0, time.time() - started_at)
            previous_avg = float(self._metrics.get("avg_compaction_time", 0.0) or 0.0)
            previous_count = max(
                0,
                int(self._metrics.get("compaction_count", 1) or 1) - 1,
            )
            self._metrics["avg_compaction_time"] = (
                (previous_avg * previous_count) + compaction_duration
            ) / max(previous_count + 1, 1)
            self._record_utilization_snapshot()

        self._logger.info(
            "Compacted context: removed %s messages with strategy=%s (saved ~%s tokens)",
            removed,
            _normalize_compaction_strategy(strategy),
            saved_tokens,
        )

        return removed

    async def prepare_for_tool_output_injection(
        self,
        estimated_output_tokens: int,
        *,
        provider_name: str = "",
        model_name: str = "",
        task_type: str = "",
        min_messages: int = 6,
        default_strategy: str = "tiered",
    ) -> Dict[str, Any]:
        """Compact proactively before injecting large tool-output payloads."""
        decision = _build_tool_output_compaction_decision(
            estimated_output_tokens=estimated_output_tokens,
            remaining_tokens=self.get_remaining_tokens(),
            utilization_percent=self.get_context_utilization(),
            recommendation=self.get_compaction_recommendation(),
            provider_name=provider_name,
            model_name=model_name,
            task_type=task_type,
            default_strategy=default_strategy,
        )
        if not decision["should_compact"]:
            return decision

        removed = await self.compact_context(
            strategy=str(decision["strategy"] or default_strategy),
            min_messages=min_messages,
        )
        decision["messages_removed"] = removed
        decision["compacted"] = removed > 0
        if removed > 0:
            decision["saved_tokens"] = int(self._metrics.get("last_compaction_saved", 0) or 0)
        return decision

    def add_message(self, message: Optional["Message"] = None, **kwargs: Any) -> None:
        """Add a message to the context.

        Args:
            message: Message to add
        """
        if message is None:
            if not kwargs:
                raise ValueError("message or keyword fields are required")
            message = kwargs
        elif kwargs:
            raise TypeError("Use either a message object or keyword fields, not both")
        self._messages.append(message)

    def add_messages(self, messages: List["Message"]) -> None:
        """Add multiple messages to the context.

        Args:
            messages: Messages to add
        """
        self._messages.extend(messages)

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List["Message"]:
        """Get messages from context.

        Args:
            limit: Maximum number of messages to return
            role: Filter by message role

        Returns:
            List of messages
        """
        messages = self._messages

        if role:
            messages = [m for m in messages if self._message_value(m, "role", "") == role]

        if limit:
            messages = messages[-limit:]

        return messages

    def clear_messages(self, retain_system: bool = True) -> None:
        """Clear messages from context.

        Args:
            retain_system: If True, retain system prompt
        """
        if retain_system:
            system_messages = [
                m for m in self._messages if self._message_value(m, "role", "") == "system"
            ]
            self._messages = system_messages
        else:
            self._messages.clear()

    def get_max_tokens(self) -> int:
        """Get the maximum context token limit.

        Returns:
            Maximum tokens allowed in context
        """
        return self._config.max_tokens

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum context token limit.

        Args:
            max_tokens: Maximum tokens allowed

        Raises:
            ValueError: If max_tokens is negative
        """
        if max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative: {max_tokens}")

        self._config.max_tokens = max_tokens
        self._logger.info(f"Max tokens updated to {max_tokens}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 characters per token
        return len(str(text or "")) // 4

    def is_healthy(self) -> bool:
        """Check if the context service is healthy.

        Returns:
            True if the service is healthy
        """
        return self._config.max_tokens > 0

    # ==========================================================================
    # Context Monitoring and Threshold Methods
    # ==========================================================================

    def get_context_size(self) -> int:
        """Get current context size in tokens.

        Returns the total number of tokens in the current context
        including all messages.

        Returns:
            Total token count

        Example:
            size = service.get_context_size()
            # Returns: 15000
        """
        total = 0
        for message in self._messages:
            content = self._message_value(message, "content", "")
            total += self._estimate_tokens(content)
        return total

    def get_context_utilization(self) -> float:
        """Get context utilization as percentage.

        Returns the percentage of the max token limit that is currently used.

        Returns:
            Utilization percentage (0-100+)

        Example:
            utilization = service.get_context_utilization()
            # Returns: 75.5 (75.5% used)
        """
        if self._config.max_tokens <= 0:
            return 0.0
        current = self.get_context_size()
        return (current / self._config.max_tokens) * 100

    def get_remaining_tokens(self) -> int:
        """Get remaining token capacity.

        Returns the number of tokens that can still be added before
        hitting the max token limit.

        Returns:
            Remaining token count

        Example:
            remaining = service.get_remaining_tokens()
            # Returns: 85000
        """
        current = self.get_context_size()
        remaining = self._config.max_tokens - current
        return max(0, remaining)

    def is_context_nearly_full(self, threshold_percent: float = 90.0) -> bool:
        """Check if context is nearly full.

        Returns True if context utilization exceeds the given threshold.
        Useful for triggering proactive compaction before overflow.

        Args:
            threshold_percent: Utilization threshold (default 90%)

        Returns:
            True if context is nearly full, False otherwise

        Example:
            if service.is_context_nearly_full(threshold_percent=90):
                # Trigger compaction
        """
        utilization = self.get_context_utilization()
        return utilization >= threshold_percent

    def get_token_count_by_role(self) -> Dict[str, int]:
        """Get token count breakdown by message role.

        Returns a dictionary mapping role names to their token counts.

        Returns:
            Dictionary with role -> token count mapping

        Example:
            counts = service.get_token_count_by_role()
            # {"user": 5000, "assistant": 8000, "system": 1000, "tool": 2000}
        """
        counts = {}
        for message in self._messages:
            role = self._message_value(message, "role", "unknown")
            content = self._message_value(message, "content", "")
            tokens = self._estimate_tokens(content)
            counts[role] = counts.get(role, 0) + tokens
        return counts

    def should_compact(self) -> bool:
        """Check if context should be compacted.

        Returns True if context exceeds the overflow threshold
        configured in the service config.

        Returns:
            True if compaction recommended, False otherwise

        Example:
            if service.should_compact():
                await service.compact_context()
        """
        return self.is_context_nearly_full(self._config.overflow_threshold_percent)

    def get_compaction_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for context compaction.

        Returns detailed information about whether compaction is needed
        and how many messages should be removed.

        Returns:
            Dictionary with compaction recommendation

        Example:
            rec = service.get_compaction_recommendation()
            # {
            #   "should_compact": True,
            #   "current_tokens": 95000,
            #   "utilization_percent": 95.0,
            #   "recommended_removal": 4
            # }
        """
        current_tokens = self.get_context_size()
        utilization = self.get_context_utilization()
        should = self.should_compact()

        # Estimate messages to remove to get below threshold
        messages_to_remove = 0
        if should:
            target_utilization = self._config.overflow_threshold_percent - 10  # 10% buffer
            target_tokens = (target_utilization / 100) * self._config.max_tokens
            excess_tokens = current_tokens - target_tokens

            if excess_tokens > 0:
                avg_tokens_per_message = (
                    current_tokens / len(self._messages) if self._messages else 1
                )
                messages_to_remove = int(excess_tokens / avg_tokens_per_message) + 1

        return {
            "should_compact": should,
            "current_tokens": current_tokens,
            "max_tokens": self._config.max_tokens,
            "utilization_percent": utilization,
            "threshold_percent": self._config.overflow_threshold_percent,
            "recommended_removal": messages_to_remove,
            "message_count": len(self._messages),
        }

    def _estimate_tokens(self, text: str) -> int:
        """Internal token estimation."""
        return self.estimate_tokens(text)

    # ==========================================================================
    # Historical Metrics Tracking
    # ==========================================================================

    def get_historical_metrics(self, sample_count: int = 100) -> Dict[str, Any]:
        """Get historical context metrics for trend analysis.

        Returns historical data about context usage patterns,
        including utilization trends, compaction frequency, etc.

        Args:
            sample_count: Number of historical samples to return

        Returns:
            Dictionary with historical metrics

        Example:
            history = service.get_historical_metrics(sample_count=50)
            # {
            #   "samples": 50,
            #   "average_utilization": 65.5,
            #   "peak_utilization": 92.3,
            #   "utilization_trend": "increasing",
            #   "compaction_count": 5,
            #   "overflow_count": 2,
            # }
        """
        return {
            "samples": sample_count,
            "average_utilization": self._calculate_average_utilization(),
            "peak_utilization": self._get_peak_utilization(),
            "utilization_trend": self._calculate_utilization_trend(),
            "compaction_count": self._metrics.get("compaction_count", 0),
            "overflow_count": self._metrics.get("overflow_count", 0),
            "last_compaction_time": self._metrics.get("last_compaction_time"),
            "last_overflow_time": self._metrics.get("last_overflow_time"),
        }

    def get_utilization_history(self) -> List[Dict[str, Any]]:
        """Get detailed utilization history.

        Returns time-series data of context utilization over time,
        useful for visualization and analysis.

        Returns:
            List of utilization snapshots with timestamps

        Example:
            history = service.get_utilization_history()
            # [
            #   {"timestamp": "2025-04-17T10:00:00", "utilization": 45.2, "tokens": 45200},
            #   {"timestamp": "2025-04-17T10:01:00", "utilization": 52.1, "tokens": 52100},
            # ]
        """
        return self._metrics.get("utilization_history", [])

    # ==========================================================================
    # Trend Analysis
    # ==========================================================================

    def analyze_context_trends(self) -> Dict[str, Any]:
        """Analyze context usage trends over time.

        Identifies patterns in context growth, compaction needs,
        and predicts future requirements.

        Returns:
            Dictionary with trend analysis results

        Example:
            trends = service.analyze_context_trends()
            # {
            #   "growth_rate": 15.5,  # tokens per minute
            #   "predicted_overflow_time": "2025-04-17T10:30:00",
            #   "compaction_frequency": "high",
            #   "optimization_score": 0.75,
            #   "recommendations": ["Consider more aggressive compaction"],
            # }
        """
        current_utilization = self.get_context_utilization()
        history = self.get_utilization_history()

        if len(history) < 2:
            return {
                "growth_rate": 0.0,
                "predicted_overflow_time": None,
                "compaction_frequency": "unknown",
                "optimization_score": 0.5,
                "recommendations": ["Insufficient data for analysis"],
            }

        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(history)

        # Predict overflow time
        predicted_overflow = self._predict_overflow_time(current_utilization, growth_rate)

        # Determine compaction frequency
        compaction_freq = self._determine_compaction_frequency()

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_utilization,
            growth_rate,
            compaction_freq,
            optimization_score,
        )

        return {
            "growth_rate": growth_rate,
            "predicted_overflow_time": predicted_overflow,
            "compaction_frequency": compaction_freq,
            "optimization_score": optimization_score,
            "recommendations": recommendations,
        }

    # ==========================================================================
    # Predictive Compaction
    # ==========================================================================

    def should_compact_soon(self, lookahead_minutes: int = 5) -> bool:
        """Predict if compaction will be needed soon.

        Analyzes growth trends to determine if context will exceed
        thresholds within the lookahead window.

        Args:
            lookahead_minutes: Minutes to look ahead for prediction

        Returns:
            True if compaction will be needed within lookahead window

        Example:
            if service.should_compact_soon(lookahead_minutes=5):
                # Proactively compact now
                await service.compact_context()
        """
        current_util = self.get_context_utilization()
        trends = self.analyze_context_trends()

        if current_util >= self._config.overflow_threshold_percent:
            return True

        # Predict utilization in lookahead window
        growth_rate = trends["growth_rate"]  # tokens per minute
        current_tokens = self.get_context_size()
        max_tokens = self.get_max_tokens()

        # Simple linear prediction
        predicted_tokens = current_tokens + (growth_rate * lookahead_minutes)
        predicted_util = (predicted_tokens / max_tokens * 100) if max_tokens > 0 else 0

        return predicted_util >= self._config.overflow_threshold_percent

    def get_predictive_compaction_plan(self) -> Dict[str, Any]:
        """Get predictive compaction recommendations.

        Returns detailed recommendations for when and how to compact
        the context based on usage patterns.

        Returns:
            Dictionary with compaction plan

        Example:
            plan = service.get_predictive_compaction_plan()
            # {
            #   "should_compact_now": False,
            #   "recommended_compaction_time": "2025-04-17T10:15:00",
            #   "urgency": "low",
            #   "suggested_strategy": "tiered",
            #   "estimated_tokens_after_compaction": 35000,
            # }
        """
        should_compact = self.should_compact_soon()
        trends = self.analyze_context_trends()
        current_tokens = self.get_context_size()

        # Determine urgency
        current_util = self.get_context_utilization()
        if current_util >= 95:
            urgency = "critical"
        elif current_util >= 85:
            urgency = "high"
        elif current_util >= 75:
            urgency = "medium"
        else:
            urgency = "low"

        # Suggest strategy based on urgency
        if urgency == "critical":
            strategy = "aggressive"
        elif urgency == "high":
            strategy = "tiered"
        else:
            strategy = self._config.default_compaction_strategy

        # Estimate tokens after compaction
        estimated_after = int(current_tokens * 0.6)  # Assume 40% reduction

        return {
            "should_compact_now": should_compact,
            "recommended_compaction_time": self._calculate_compaction_time(),
            "urgency": urgency,
            "suggested_strategy": strategy,
            "estimated_tokens_after_compaction": estimated_after,
            "current_utilization": current_util,
            "predicted_utilization_5min": trends.get("predicted_overflow_time") is not None,
        }

    # ==========================================================================
    # Advanced Overflow Detection
    # ==========================================================================

    def get_overflow_risk_score(self) -> float:
        """Calculate overflow risk score (0.0-1.0).

        Analyzes current utilization, growth rate, and historical patterns
        to determine risk of context overflow.

        Returns:
            Risk score from 0.0 (no risk) to 1.0 (critical risk)

        Example:
            risk = service.get_overflow_risk_score()
            # 0.75 = high risk
        """
        current_util = self.get_context_utilization()
        trends = self.analyze_context_trends()

        # Base risk from current utilization
        util_risk = min(current_util / 100.0, 1.0)

        # Growth rate risk (faster growth = higher risk)
        growth_rate = trends.get("growth_rate", 0)
        growth_risk = min(growth_rate / 100.0, 1.0)  # Normalize

        # Historical overflow risk
        overflow_count = self._metrics.get("overflow_count", 0)
        history_risk = min(overflow_count / 10.0, 1.0)  # Cap at 10 overflows

        # Weighted average
        risk_score = (util_risk * 0.5) + (growth_risk * 0.3) + (history_risk * 0.2)

        return min(risk_score, 1.0)

    def get_overflow_alerts(self) -> List[Dict[str, Any]]:
        """Get active overflow alerts.

        Returns list of alerts related to context overflow risk,
        including warnings and recommendations.

        Returns:
            List of alert dictionaries

        Example:
            alerts = service.get_overflow_alerts()
            # [
            #   {"level": "warning", "message": "Context at 85% capacity", "action": "consider_compaction"},
            #   {"level": "info", "message": "Growth rate elevated", "action": "monitor_closely"},
            # ]
        """
        alerts = []
        current_util = self.get_context_utilization()
        risk_score = self.get_overflow_risk_score()

        # Critical alerts
        if current_util >= 95:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Context at {current_util:.1f}% capacity",
                    "action": "compact_immediately",
                    "urgency": "critical",
                }
            )

        # Warning alerts
        elif current_util >= 85:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Context at {current_util:.1f}% capacity",
                    "action": "consider_compaction",
                    "urgency": "high",
                }
            )

        # Info alerts
        elif current_util >= 75:
            alerts.append(
                {
                    "level": "info",
                    "message": f"Context at {current_util:.1f}% capacity",
                    "action": "monitor_closely",
                    "urgency": "medium",
                }
            )

        # Risk-based alerts
        if risk_score >= 0.7:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"High overflow risk score: {risk_score:.2f}",
                    "action": "review_growth_patterns",
                    "urgency": "high",
                }
            )

        # Growth rate alerts
        trends = self.analyze_context_trends()
        growth_rate = trends.get("growth_rate", 0)
        if growth_rate > 50:  # tokens per minute
            alerts.append(
                {
                    "level": "info",
                    "message": f"Elevated growth rate: {growth_rate:.1f} tokens/min",
                    "action": "monitor_closely",
                    "urgency": "medium",
                }
            )

        return alerts

    # ==========================================================================
    # Context Optimization Recommendations
    # ==========================================================================

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get context optimization recommendations.

        Analyzes current context state and provides actionable
        recommendations for optimization.

        Returns:
            Dictionary with optimization recommendations

        Example:
            recs = service.get_optimization_recommendations()
            # {
            #   "priority_actions": ["compact_context", "reduce_system_prompt"],
            #   "savings_potential": 15000,  # tokens
            #   "optimization_score": 0.65,
            #   "detailed_recommendations": [...],
            # }
        """
        current_tokens = self.get_context_size()
        current_util = self.get_context_utilization()

        priority_actions = []
        savings_potential = 0
        detailed_recs = []

        # Analyze system prompt size
        metrics = self.get_token_count_by_role()
        system_tokens = metrics.get("system", 0)

        if system_tokens > 5000:
            priority_actions.append("reduce_system_prompt")
            savings_potential += system_tokens - 3000
            detailed_recs.append(
                {
                    "action": "reduce_system_prompt",
                    "description": "System prompt is large",
                    "current_tokens": system_tokens,
                    "suggested_tokens": 3000,
                    "potential_savings": system_tokens - 3000,
                }
            )

        # Analyze message count
        message_count = len(self.get_messages())
        if message_count > 20:
            priority_actions.append("compact_messages")
            estimated_savings = int(current_tokens * 0.3)
            savings_potential += estimated_savings
            detailed_recs.append(
                {
                    "action": "compact_messages",
                    "description": f"High message count: {message_count}",
                    "current_messages": message_count,
                    "suggested_messages": 10,
                    "potential_savings": estimated_savings,
                }
            )

        # Analyze tool results
        tool_results = metrics.get("tool", 0)
        if tool_results > 10000:
            priority_actions.append("truncate_tool_results")
            savings_potential += tool_results - 5000
            detailed_recs.append(
                {
                    "action": "truncate_tool_results",
                    "description": "Large tool result tokens",
                    "current_tokens": tool_results,
                    "suggested_tokens": 5000,
                    "potential_savings": tool_results - 5000,
                }
            )

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score()

        return {
            "priority_actions": priority_actions,
            "savings_potential": savings_potential,
            "optimization_score": optimization_score,
            "current_utilization": current_util,
            "target_utilization": 70.0,  # Aim for 70%
            "detailed_recommendations": detailed_recs,
        }

    # ==========================================================================
    # Performance Metrics
    # ==========================================================================

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get context-related performance metrics.

        Returns metrics about compaction performance,
        context operations efficiency, etc.

        Returns:
            Dictionary with performance metrics

        Example:
            perf = service.get_performance_metrics()
            # {
            #   "average_compaction_time": 0.5,  # seconds
            #   "compaction_efficiency": 0.85,  # tokens removed / total
            #   "operation_count": 150,
            #   "cache_hit_rate": 0.92,
            # }
        """
        return {
            "average_compaction_time": self._metrics.get("avg_compaction_time", 0.0),
            "compaction_efficiency": self._calculate_compaction_efficiency(),
            "operation_count": self._metrics.get("operation_count", 0),
            "cache_hit_rate": self._metrics.get("cache_hit_rate", 0.0),
            "last_compaction_saved_tokens": self._metrics.get("last_compaction_saved", 0),
            "total_tokens_saved": self._metrics.get("total_tokens_saved", 0),
        }

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _calculate_average_utilization(self) -> float:
        """Calculate average context utilization."""
        history = self.get_utilization_history()
        if not history:
            return self.get_context_utilization()

        total_util = sum(h.get("utilization", 0) for h in history)
        return total_util / len(history) if history else 0.0

    def _record_utilization_snapshot(self) -> None:
        """Record a bounded utilization sample for trend-aware reporting."""
        history = list(self._metrics.get("utilization_history", []) or [])
        history.append(
            {
                "timestamp": time.time(),
                "utilization": self.get_context_utilization(),
                "tokens": self.get_context_size(),
            }
        )
        if len(history) > 200:
            history = history[-200:]
        self._metrics["utilization_history"] = history

    def _get_peak_utilization(self) -> float:
        """Get peak utilization from history."""
        history = self.get_utilization_history()
        if not history:
            return self.get_context_utilization()

        return max(h.get("utilization", 0) for h in history)

    def _calculate_utilization_trend(self) -> UtilizationTrend:
        """Calculate utilization trend direction."""
        history = self.get_utilization_history()
        if len(history) < 5:
            return UtilizationTrend.UNKNOWN

        # Compare recent vs older samples
        recent = history[-5:]
        recent_avg = sum(h.get("utilization", 0) for h in recent) / len(recent)

        older = history[:5] if len(history) >= 10 else history[: len(history) // 2]
        older_avg = sum(h.get("utilization", 0) for h in older) / len(older)

        if recent_avg > older_avg + 5:
            return UtilizationTrend.INCREASING
        elif recent_avg < older_avg - 5:
            return UtilizationTrend.DECREASING
        else:
            return UtilizationTrend.STABLE

    def _calculate_growth_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calculate context growth rate in tokens per minute."""
        if len(history) < 2:
            return 0.0

        # Simple linear regression
        first = history[0]
        last = history[-1]

        token_diff = last.get("tokens", 0) - first.get("tokens", 0)
        # Assume samples are 1 minute apart (simplified)
        time_diff = len(history)

        return token_diff / time_diff if time_diff > 0 else 0.0

    def _predict_overflow_time(self, current_util: float, growth_rate: float) -> Optional[str]:
        """Predict when context will overflow."""
        if growth_rate <= 0:
            return None

        current_tokens = self.get_context_size()
        max_tokens = self.get_max_tokens()
        remaining_tokens = max_tokens - current_tokens

        if remaining_tokens <= 0:
            return "now"

        # Time to overflow in minutes
        minutes_to_overflow = remaining_tokens / growth_rate

        if minutes_to_overflow < 1:
            return "soon"
        elif minutes_to_overflow < 10:
            return f"in_{int(minutes_to_overflow)}_minutes"
        else:
            return None

    def _determine_compaction_frequency(self) -> CompactionFrequency:
        """Determine compaction frequency category."""
        compact_count = self._metrics.get("compaction_count", 0)
        history = self.get_utilization_history()

        if len(history) == 0:
            return CompactionFrequency.UNKNOWN

        # Compactions per sample
        freq = compact_count / len(history)

        if freq > 0.1:
            return CompactionFrequency.HIGH
        elif freq > 0.05:
            return CompactionFrequency.MEDIUM
        else:
            return CompactionFrequency.LOW

    def _calculate_optimization_score(self) -> float:
        """Calculate context optimization score (0.0-1.0)."""
        current_util = self.get_context_utilization()

        # Lower utilization = better optimization
        util_score = 1.0 - (current_util / 100.0)

        # Fewer compactions = better optimization
        compact_count = self._metrics.get("compaction_count", 0)
        compact_score = max(1.0 - (compact_count / 20.0), 0.0)

        # Weighted average
        return (util_score * 0.7) + (compact_score * 0.3)

    def _generate_recommendations(
        self,
        current_util: float,
        growth_rate: float,
        compaction_freq: str,
        optimization_score: float,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if current_util > 80:
            recommendations.append("Consider aggressive compaction")

        if growth_rate > 50:
            recommendations.append("High growth rate detected - monitor closely")

        if compaction_freq == "high":
            recommendations.append("Frequent compactions - consider reducing context input")

        if optimization_score < 0.5:
            recommendations.append("Low optimization score - review context management")

        if not recommendations:
            recommendations.append("Context usage is optimal")

        return recommendations

    def _calculate_compaction_time(self) -> CompactionUrgency:
        """Calculate recommended compaction time."""
        risk_score = self.get_overflow_risk_score()

        if risk_score > 0.8:
            return CompactionUrgency.NOW
        elif risk_score > 0.6:
            return CompactionUrgency.SOON
        elif risk_score > 0.4:
            return CompactionUrgency.WITHIN_5_MINUTES
        else:
            return CompactionUrgency.NO_URGENCY

    def _calculate_compaction_efficiency(self) -> float:
        """Calculate compaction efficiency (tokens removed / total)."""
        last_saved = self._metrics.get("last_compaction_saved", 0)
        current_tokens = self.get_context_size()

        if current_tokens == 0:
            return 0.0

        return last_saved / (last_saved + current_tokens)


class ContextServiceRegistry:
    """Manage runtime-scoped ``ContextService`` instances for multiple agents."""

    def __init__(self, config: ContextServiceConfig):
        self._config = config
        self._services: Dict[str, ContextService] = {}

    def get_or_create(self, runtime_context: AgentRuntimeContext) -> ContextService:
        """Return the context service scoped to one agent runtime identity."""
        scope_key = self._scope_key(runtime_context)
        service = self._services.get(scope_key)
        if service is None:
            service = ContextService(config=self._config)
            self._services[scope_key] = service
        return service

    @staticmethod
    def _scope_key(runtime_context: AgentRuntimeContext) -> str:
        """Return the in-memory isolation key for one runtime context."""
        return "\x1f".join(
            (
                str(runtime_context.session_id or ""),
                str(runtime_context.agent_id or ""),
            )
        )

    async def compact_if_needed(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        strategy: Optional[str] = None,
        min_messages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compact only the context for the provided agent runtime session."""
        service = self.get_or_create(runtime_context)
        metrics_before = await service.get_context_metrics()
        should_compact = (
            metrics_before.total_tokens > self._config.max_tokens
            or metrics_before.utilization_percent >= self._config.overflow_threshold_percent
        )
        if not should_compact:
            return {
                **runtime_context.identity_metadata(),
                "compacted": False,
                "messages_removed": 0,
                "utilization_before": metrics_before.utilization_percent,
                "utilization_after": metrics_before.utilization_percent,
            }

        removed = await service.compact_context(
            strategy=strategy or self._config.default_compaction_strategy,
            min_messages=min_messages or self._config.min_messages_to_keep,
        )
        metrics_after = await service.get_context_metrics()
        return {
            **runtime_context.identity_metadata(),
            "compacted": removed > 0,
            "messages_removed": removed,
            "utilization_before": metrics_before.utilization_percent,
            "utilization_after": metrics_after.utilization_percent,
        }
