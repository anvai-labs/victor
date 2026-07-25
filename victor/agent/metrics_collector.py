# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Metrics collection for agent orchestration.

This module provides centralized metrics collection for:
- Stream metrics (time-to-first-token, duration, tokens/second)
- Tool selection statistics (semantic/keyword/fallback methods)
- Tool execution statistics (success rate, timing, cost tiers)
- Callback handling for decomposed components

Extracted from AgentOrchestrator to improve modularity and testability.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.agent.stream_handler import StreamMetrics
from victor.tools.base import CostTier

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.agent.streaming_controller import StreamingSession
    from victor.agent.debug_logger import DebugLogger
    from victor.observability.analytics.streaming_metrics import (
        StreamingMetricsCollector,
    )

logger = logging.getLogger(__name__)


@dataclass
class ToolSelectionStats:
    """Statistics for tool selection methods."""

    semantic_selections: int = 0
    keyword_selections: int = 0
    fallback_selections: int = 0
    total_tools_selected: int = 0
    total_tools_executed: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return {
            "semantic_selections": self.semantic_selections,
            "keyword_selections": self.keyword_selections,
            "fallback_selections": self.fallback_selections,
            "total_tools_selected": self.total_tools_selected,
            "total_tools_executed": self.total_tools_executed,
        }


@dataclass
class ClassificationStats:
    """Statistics for task classification.

    Tracks how tasks are being classified and which patterns
    are being triggered, useful for understanding user intent
    patterns and classification accuracy.
    """

    # Task type distribution
    action_tasks: int = 0
    analysis_tasks: int = 0
    generation_tasks: int = 0
    search_tasks: int = 0
    edit_tasks: int = 0
    default_tasks: int = 0

    # Classification source distribution
    keyword_classifications: int = 0
    semantic_classifications: int = 0
    context_classifications: int = 0
    ensemble_classifications: int = 0

    # Negation detection
    negations_detected: int = 0
    positive_overrides: int = 0

    # Confidence tracking
    total_confidence: float = 0.0
    classification_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        avg_confidence = (
            self.total_confidence / self.classification_count
            if self.classification_count > 0
            else 0.0
        )
        return {
            "task_types": {
                "action": self.action_tasks,
                "analysis": self.analysis_tasks,
                "generation": self.generation_tasks,
                "search": self.search_tasks,
                "edit": self.edit_tasks,
                "default": self.default_tasks,
            },
            "sources": {
                "keyword": self.keyword_classifications,
                "semantic": self.semantic_classifications,
                "context": self.context_classifications,
                "ensemble": self.ensemble_classifications,
            },
            "negation_handling": {
                "negations_detected": self.negations_detected,
                "positive_overrides": self.positive_overrides,
            },
            "confidence": {
                "average": avg_confidence,
                "total_classifications": self.classification_count,
            },
        }


@dataclass
class CostTracking:
    """Tracks cost by tier for tool executions."""

    total_cost_weight: float = 0.0
    cost_by_tier: Dict[str, float] = field(
        default_factory=lambda: {"free": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0}
    )
    calls_by_tier: Dict[str, int] = field(
        default_factory=lambda: {"free": 0, "low": 0, "medium": 0, "high": 0}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cost_weight": self.total_cost_weight,
            "cost_by_tier": self.cost_by_tier.copy(),
            "calls_by_tier": self.calls_by_tier.copy(),
        }


@dataclass
class MetricsCollectorConfig:
    """Configuration for MetricsCollector."""

    model: str = "unknown"
    provider: str = "unknown"
    analytics_enabled: bool = False


class MetricsCollector:
    """Centralized metrics collection for agent operations.

    Collects and aggregates metrics for:
    - Streaming sessions (TTFT, duration, throughput)
    - Tool selection (method distribution)
    - Tool execution (success rate, timing, cost)

    This class is designed to be injected into the orchestrator and
    receive callbacks from ToolPipeline and StreamingController.
    """

    def __init__(
        self,
        config: MetricsCollectorConfig,
        usage_logger: Any,
        debug_logger: Optional["DebugLogger"] = None,
        streaming_metrics_collector: Optional["StreamingMetricsCollector"] = None,
        tool_cost_lookup: Optional[Callable[[str], Optional[CostTier]]] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            config: Configuration for the collector
            usage_logger: Logger for usage events
            debug_logger: Optional debug logger for tool calls
            streaming_metrics_collector: Optional analytics collector
            tool_cost_lookup: Callback to get tool cost tier by name
        """
        self.config = config
        self.usage_logger = usage_logger
        self.debug_logger = debug_logger
        self.streaming_metrics_collector = streaming_metrics_collector
        self._tool_cost_lookup = tool_cost_lookup or (lambda _: CostTier.FREE)

        # Stream metrics
        self._current_stream_metrics: Optional[StreamMetrics] = None

        # Tool selection stats
        self._selection_stats = ToolSelectionStats()

        # Classification stats
        self._classification_stats = ClassificationStats()

        # Tool usage stats (per-tool breakdown)
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}

        # Cost tracking
        self._cost_tracking = CostTracking()

        # Ad-hoc metrics recorded via the MetricsCollectorProtocol adapter
        # (``record_metric``). The typed ``record_*`` methods above cover the
        # known agent metrics; this store backs generic, name/value observations.
        self._adhoc_metrics: Dict[str, float] = {}

    # =========================================================================
    # MetricsCollectorProtocol adapter (FEP-0014)
    # =========================================================================

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single generic metric observation (protocol adapter).

        Complements the typed ``record_*`` methods with a name/value surface so
        this collector satisfies ``MetricsCollectorProtocol``. Stores the value
        under ``name`` (tag-qualified when tags are supplied).

        Args:
            name: Metric name.
            value: Numeric value to record.
            **tags: Optional string-valued dimensional tags, appended to the key.
        """
        key = name
        if tags:
            tag_suffix = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{name}{{{tag_suffix}}}"
        self._adhoc_metrics[key] = float(value)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time snapshot of collected metrics (protocol adapter).

        Merges the aggregated tool-usage view from :meth:`get_tool_usage_stats`
        with any ad-hoc metrics recorded via :meth:`record_metric`.

        Returns:
            A mapping of the collector's current metric state.
        """
        snapshot: Dict[str, Any] = dict(self.get_tool_usage_stats())
        snapshot["adhoc_metrics"] = dict(self._adhoc_metrics)
        return snapshot

    # =========================================================================
    # Stream Metrics
    # =========================================================================

    def init_stream_metrics(self) -> StreamMetrics:
        """Initialize fresh stream metrics for a new streaming session.

        Returns:
            New StreamMetrics instance
        """
        self._current_stream_metrics = StreamMetrics(start_time=time.time())
        return self._current_stream_metrics

    def record_first_token(self) -> None:
        """Record the time of first token received."""
        if self._current_stream_metrics:
            if self._current_stream_metrics.first_token_time is None:
                self._current_stream_metrics.first_token_time = time.time()

    def finalize_stream_metrics(
        self,
        usage_data: Optional[Dict[str, int]] = None,
        provider_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session.

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       If provided, this will be recorded to enable actual
                       token counts instead of estimates.
            provider_diagnostics: Optional aggregated Sandhi transport
                       diagnostics (retry attempts, usage completeness,
                       upstream request ids) for the turn.

        Returns:
            Finalized StreamMetrics or None if no active session
        """
        if not self._current_stream_metrics:
            return None

        # Record actual token usage from provider API if available
        # This enables accurate token counts instead of estimates
        if usage_data:
            self._current_stream_metrics.record_usage(usage_data)

        if provider_diagnostics:
            self._current_stream_metrics.metadata["sandhi_usage"] = dict(provider_diagnostics)

        # Calculate cost using provider capabilities
        try:
            from victor.config.metrics_capabilities import get_metrics_capabilities

            capabilities = get_metrics_capabilities(self.config.provider, self.config.model)
            self._current_stream_metrics.calculate_cost(capabilities)
        except Exception as e:
            logger.debug(f"Failed to calculate cost: {e}")

        self._current_stream_metrics.end_time = time.time()
        metrics = self._current_stream_metrics

        # Log stream metrics including cost if available. Token fields are the dominant cost
        # term — emit them on the canonical "stream_completed" record so the usage sink (which
        # the RL trace-miner and cost analysis read) actually carries per-turn token accounting,
        # not just timing.
        log_data: Dict[str, Any] = {
            "ttft": metrics.time_to_first_token,
            "total_duration": metrics.total_duration,
            "tokens_per_second": metrics.tokens_per_second,
            "total_chunks": metrics.total_chunks,
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_tokens": metrics.prompt_tokens + metrics.completion_tokens,
            "cache_read_tokens": metrics.cache_read_tokens,
            "cache_write_tokens": metrics.cache_write_tokens,
        }
        if metrics.cost_calculated:
            log_data["total_cost"] = metrics.total_cost
        if provider_diagnostics:
            # Sandhi transport truth: retry attempts, usage completeness, and
            # upstream request ids — durable on the canonical usage sink so
            # cost analysis and the RL trace-miner can see retry burn and
            # partial-usage turns.
            log_data["sandhi_usage"] = dict(provider_diagnostics)

        self.usage_logger.log_event("stream_completed", log_data)

        # Record to streaming metrics collector if available
        if self.streaming_metrics_collector:
            try:
                from victor.observability.analytics.streaming_metrics import (
                    StreamMetrics as AnalyticsMetrics,
                )

                # Use actual token count from API when available, fall back to estimate
                # StreamMetrics.effective_total_tokens handles this logic
                total_tokens = metrics.effective_total_tokens

                # Convert to analytics format and record
                analytics_metrics = AnalyticsMetrics(
                    request_id=str(uuid.uuid4()),
                    start_time=metrics.start_time,
                    first_token_time=metrics.first_token_time,
                    last_token_time=metrics.end_time,
                    total_chunks=metrics.total_chunks,
                    total_tokens=total_tokens,
                    model=self.config.model,
                    provider=self.config.provider,
                )
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self.streaming_metrics_collector.record_metrics(analytics_metrics)
                    )
                except RuntimeError:
                    # No event loop running
                    logger.debug("No event loop, skipping metrics recording")
                except Exception as e:
                    logger.debug(f"Failed to record metrics: {e}")

                # Log source of token data for debugging
                if metrics.has_actual_usage:
                    logger.debug(f"Metrics using actual API tokens: {total_tokens}")
                else:
                    logger.debug(f"Metrics using estimated tokens: {total_tokens}")
            except Exception as e:
                logger.debug(f"Failed to record to metrics collector: {e}")

        return metrics

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Returns:
            Last StreamMetrics or None
        """
        return self._current_stream_metrics

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled
        """
        if not self.streaming_metrics_collector:
            return None

        summary = self.streaming_metrics_collector.get_summary()
        if hasattr(summary, "__dict__"):
            return vars(summary)
        return summary  # type: ignore[return-value]

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        if not self.streaming_metrics_collector:
            return []

        metrics_list = self.streaming_metrics_collector.get_recent_metrics(count=limit)
        return [vars(m) if hasattr(m, "__dict__") else m for m in metrics_list]  # type: ignore[misc]

    # =========================================================================
    # Tool Selection Stats
    # =========================================================================

    def record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        if method == "semantic":
            self._selection_stats.semantic_selections += 1
        elif method == "keyword":
            self._selection_stats.keyword_selections += 1
        elif method == "fallback":
            self._selection_stats.fallback_selections += 1

        self._selection_stats.total_tools_selected += num_tools
        self.usage_logger.log_event("tool_selection", {"method": method, "tool_count": num_tools})

        logger.debug(
            f"Tool selection: method={method}, num_tools={num_tools}, "
            f"stats={self._selection_stats.to_dict()}"
        )

    # =========================================================================
    # Classification Stats
    # =========================================================================

    def record_classification(
        self,
        task_type: str,
        source: str,
        confidence: float,
        negated_count: int = 0,
        had_positive_override: bool = False,
    ) -> None:
        """Record task classification statistics.

        Args:
            task_type: The classified task type ('action', 'analysis', etc.)
            source: Classification source ('keyword', 'semantic', 'context', 'ensemble')
            confidence: Confidence score (0-1)
            negated_count: Number of negated keywords detected
            had_positive_override: Whether a positive override was used
        """
        # Track task type distribution
        task_type_lower = task_type.lower()
        if task_type_lower == "action":
            self._classification_stats.action_tasks += 1
        elif task_type_lower == "analysis":
            self._classification_stats.analysis_tasks += 1
        elif task_type_lower == "generation":
            self._classification_stats.generation_tasks += 1
        elif task_type_lower == "search":
            self._classification_stats.search_tasks += 1
        elif task_type_lower == "edit":
            self._classification_stats.edit_tasks += 1
        else:
            self._classification_stats.default_tasks += 1

        # Track source distribution
        source_lower = source.lower()
        if source_lower == "keyword":
            self._classification_stats.keyword_classifications += 1
        elif source_lower == "semantic":
            self._classification_stats.semantic_classifications += 1
        elif source_lower == "context":
            self._classification_stats.context_classifications += 1
        elif source_lower == "ensemble":
            self._classification_stats.ensemble_classifications += 1

        # Track negation handling
        self._classification_stats.negations_detected += negated_count
        if had_positive_override:
            self._classification_stats.positive_overrides += 1

        # Track confidence
        self._classification_stats.total_confidence += confidence
        self._classification_stats.classification_count += 1

        # Log event
        self.usage_logger.log_event(
            "task_classification",
            {
                "task_type": task_type,
                "source": source,
                "confidence": confidence,
                "negated_count": negated_count,
            },
        )

        logger.debug(
            f"Classification: type={task_type}, source={source}, "
            f"confidence={confidence:.2f}, negated={negated_count}"
        )

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics.

        Returns:
            Dictionary with classification metrics including:
            - Task type distribution
            - Classification source distribution
            - Negation handling stats
            - Confidence metrics
        """
        return self._classification_stats.to_dict()

    # =========================================================================
    # Tool Execution Stats
    # =========================================================================

    def record_tool_execution(self, tool_name: str, success: bool, elapsed_ms: float) -> None:
        """Record tool execution statistics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
        """
        # Get tool cost tier
        tool_cost = self._tool_cost_lookup(tool_name)
        cost_tier = tool_cost if tool_cost else CostTier.FREE
        cost_weight = cost_tier.weight

        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
                "cost_tier": cost_tier.value,
                "total_cost_weight": 0.0,
            }

        stats = self._tool_usage_stats[tool_name]
        # Use .get() for robustness — stats may have been pre-populated by
        # ToolPipeline's on_tool_complete callback with different keys
        stats["total_calls"] = stats.get("total_calls", 0) + 1
        stats["successful_calls"] = stats.get("successful_calls", 0) + (1 if success else 0)
        stats["failed_calls"] = stats.get("failed_calls", 0) + (0 if success else 1)
        stats["total_time_ms"] = stats.get("total_time_ms", 0.0) + elapsed_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / max(stats["total_calls"], 1)
        stats["min_time_ms"] = min(stats.get("min_time_ms", float("inf")), elapsed_ms)
        stats["max_time_ms"] = max(stats.get("max_time_ms", 0.0), elapsed_ms)
        stats["total_cost_weight"] = stats.get("total_cost_weight", 0.0) + cost_weight

        # Update global cost tracking
        self._cost_tracking.total_cost_weight += cost_weight
        self._cost_tracking.cost_by_tier[cost_tier.value] += cost_weight
        self._cost_tracking.calls_by_tier[cost_tier.value] += 1

        self._selection_stats.total_tools_executed += 1

        logger.debug(
            f"Tool executed: {tool_name} "
            f"(success={success}, time={elapsed_ms:.1f}ms, "
            f"total_calls={stats['total_calls']}, "
            f"success_rate={stats['successful_calls']/stats['total_calls']*100:.1f}%)"
        )

    def get_tool_usage_stats(
        self, conversation_state_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Args:
            conversation_state_summary: Optional conversation state to include

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        result: Dict[str, Any] = {
            "selection_stats": self._selection_stats.to_dict(),
            "classification_stats": self._classification_stats.to_dict(),
            "tool_stats": self._tool_usage_stats.copy(),
            "cost_tracking": self._cost_tracking.to_dict(),
            "top_tools_by_usage": sorted(
                [(name, stats["total_calls"]) for name, stats in self._tool_usage_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "top_tools_by_time": sorted(
                [(name, stats["total_time_ms"]) for name, stats in self._tool_usage_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "top_tools_by_cost": sorted(
                [
                    (name, stats.get("total_cost_weight", 0.0))
                    for name, stats in self._tool_usage_stats.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

        if conversation_state_summary:
            result["conversation_state"] = conversation_state_summary

        return result

    # =========================================================================
    # Component Callbacks
    # =========================================================================

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], iteration: int = 0) -> None:
        """Callback when tool execution starts (from ToolPipeline).

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            iteration: Current iteration count
        """
        if self.debug_logger:
            self.debug_logger.log_tool_call(tool_name, arguments, iteration)

    def on_tool_complete(self, result: "ToolCallResult") -> None:
        """Callback when tool execution completes (from ToolPipeline).

        Args:
            result: The tool call result
        """
        # Update tool usage stats (simplified version, full stats via record_tool_execution)
        if result.tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[result.tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
                "cost_tier": "free",
                "total_cost_weight": 0.0,
            }
        stats = self._tool_usage_stats[result.tool_name]
        stats["total_calls"] = stats.get("total_calls", 0) + 1
        if result.success:
            stats["successful_calls"] = stats.get("successful_calls", 0) + 1
        else:
            stats["failed_calls"] = stats.get("failed_calls", 0) + 1
        stats["total_time_ms"] = stats.get("total_time_ms", 0.0) + result.execution_time_ms

    def on_streaming_session_complete(self, session: "StreamingSession") -> None:
        """Callback when streaming session completes (from StreamingController).

        Args:
            session: The completed streaming session
        """
        self.usage_logger.log_event(
            "stream_completed",
            {
                "session_id": session.session_id,
                "model": session.model,
                "provider": session.provider,
                "duration": session.duration,
                "cancelled": session.cancelled,
            },
        )

    # =========================================================================
    # Configuration Updates
    # =========================================================================

    def update_model_info(self, model: str, provider: str) -> None:
        """Update model and provider info (e.g., after hot-swap).

        Args:
            model: New model name
            provider: New provider name
        """
        self.config.model = model
        self.config.provider = provider

    def reset_stats(self) -> None:
        """Reset all statistics (e.g., after conversation reset)."""
        self._selection_stats = ToolSelectionStats()
        self._classification_stats = ClassificationStats()
        self._tool_usage_stats.clear()
        self._cost_tracking = CostTracking()
        self._current_stream_metrics = None
