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

"""Usage analytics for data-driven optimization.

This module provides comprehensive analytics for tool usage, provider
performance, and conversation patterns. Data is used to optimize:
- Tool selection (boost frequently successful tools)
- Provider switching (prefer faster/more reliable providers)
- Context management (optimize for typical conversation lengths)

Design Pattern: Singleton + Observer
====================================
UsageAnalytics is a singleton that observes tool executions, provider
calls, and conversation events to build optimization insights.

Usage:
    analytics = UsageAnalytics.get_instance()

    # Record tool execution
    analytics.record_tool_execution(
        tool_name="read",
        success=True,
        execution_time_ms=150,
        error_type=None,
        context_tokens=5000,
    )

    # Get optimization insights
    insights = analytics.get_tool_insights("read")
    # Returns: {"success_rate": 0.95, "avg_execution_ms": 120, ...}

    # Export metrics for dashboard
    metrics = analytics.export_prometheus_metrics()
"""

import asyncio
from victor.core.json_utils import json_dumps
import logging
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution."""

    timestamp: float
    success: bool
    execution_time_ms: float
    error_type: Optional[str] = None
    context_tokens: int = 0
    query_hash: Optional[str] = None


@dataclass
class ProviderCallRecord:
    """Record of a single provider API call."""

    timestamp: float
    provider_name: str
    model: str
    success: bool
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    error_type: Optional[str] = None


@dataclass
class ConversationStats:
    """Statistics for a conversation session."""

    start_time: float
    end_time: Optional[float] = None
    turn_count: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    provider_switches: int = 0


@dataclass
class AnalyticsConfig:
    """Configuration for usage analytics.

    Attributes:
        max_records_per_tool: Maximum execution records to keep per tool
        max_records_per_provider: Maximum records to keep per provider
        persistence_interval_seconds: How often to persist to disk
        cache_dir: Directory for analytics cache
        enable_prometheus_export: Enable Prometheus metrics export
    """

    max_records_per_tool: int = 1000
    max_records_per_provider: int = 500
    persistence_interval_seconds: int = 300  # 5 minutes
    cache_dir: Optional[Path] = None
    enable_prometheus_export: bool = True


class UsageAnalytics:
    """Comprehensive usage analytics for data-driven optimization.

    Singleton that tracks tool executions, provider calls, and
    conversation patterns to provide optimization insights.

    Features:
    - Tool success rate and latency tracking
    - Provider performance comparison
    - Conversation pattern analysis
    - Time-series data for trend analysis
    - Prometheus metrics export

    Thread-safe for concurrent access.
    """

    _instance: ClassVar[Optional["UsageAnalytics"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize analytics (use get_instance() instead)."""
        self.config = config or AnalyticsConfig()

        # Tool execution records: tool_name -> list of records
        self._tool_records: Dict[str, List[ToolExecutionRecord]] = defaultdict(list)

        # Provider call records: provider_name -> list of records
        self._provider_records: Dict[str, List[ProviderCallRecord]] = defaultdict(list)

        # Session statistics
        self._current_session: Optional[ConversationStats] = None
        self._session_history: List[ConversationStats] = []

        # Aggregated metrics for quick access
        self._tool_aggregates: Dict[str, Dict[str, Any]] = {}
        self._provider_aggregates: Dict[str, Dict[str, Any]] = {}

        # Shell output condensation savings: condenser name -> counters
        self._condensation_stats: Dict[str, Dict[str, int]] = {}

        # Thread safety
        self._data_lock = threading.RLock()

        # Persistence
        self._last_persist_time = time.time()
        self._cache_file: Optional[Path] = None
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.config.cache_dir / "usage_analytics.pkl"
            self._load_from_cache()

        logger.debug("UsageAnalytics initialized")

    @classmethod
    def get_instance(cls, config: Optional[AnalyticsConfig] = None) -> "UsageAnalytics":
        """Get the singleton instance.

        Args:
            config: Optional configuration (only used on first call)

        Returns:
            UsageAnalytics singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    # ========================================================================
    # Recording Methods
    # ========================================================================

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        error_type: Optional[str] = None,
        context_tokens: int = 0,
        query_hash: Optional[str] = None,
        error: Optional[Any] = None,
    ) -> None:
        """Record a tool execution for analytics.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            execution_time_ms: Execution time in milliseconds
            error_type: Error type if failed (e.g., "ValueError", "Timeout")
            context_tokens: Number of context tokens at time of call
            query_hash: Hash of the query (for pattern detection)
            error: Backward-compatible alias for ``error_type``
        """
        if error_type is None and error is not None:
            error_type = error if isinstance(error, str) else type(error).__name__

        record = ToolExecutionRecord(
            timestamp=time.time(),
            success=success,
            execution_time_ms=execution_time_ms,
            error_type=error_type,
            context_tokens=context_tokens,
            query_hash=query_hash,
        )

        with self._data_lock:
            records = self._tool_records[tool_name]
            records.append(record)

            # Trim if over limit
            if len(records) > self.config.max_records_per_tool:
                self._tool_records[tool_name] = records[-self.config.max_records_per_tool :]

            # Update session
            if self._current_session:
                self._current_session.tool_calls += 1

            # Update aggregates
            self._update_tool_aggregate(tool_name)

        # Check persistence
        self._maybe_persist()

        logger.debug(
            f"Recorded tool execution: {tool_name} "
            f"(success={success}, time={execution_time_ms:.1f}ms)"
        )

    def record_provider_call(
        self,
        provider_name: str,
        model: str,
        success: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a provider API call for analytics.

        Args:
            provider_name: Name of the provider
            model: Model used
            success: Whether call succeeded
            latency_ms: Total latency in milliseconds
            tokens_in: Input tokens
            tokens_out: Output tokens
            error_type: Error type if failed
        """
        record = ProviderCallRecord(
            timestamp=time.time(),
            provider_name=provider_name,
            model=model,
            success=success,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error_type=error_type,
        )

        with self._data_lock:
            records = self._provider_records[provider_name]
            records.append(record)

            # Trim if over limit
            if len(records) > self.config.max_records_per_provider:
                self._provider_records[provider_name] = records[
                    -self.config.max_records_per_provider :
                ]

            # Update session
            if self._current_session:
                self._current_session.total_tokens += tokens_in + tokens_out

            # Update aggregates
            self._update_provider_aggregate(provider_name)

        # Check persistence
        self._maybe_persist()

    def start_session(self) -> None:
        """Start a new conversation session."""
        with self._data_lock:
            # Archive current session if exists
            if self._current_session:
                self._current_session.end_time = time.time()
                self._session_history.append(self._current_session)

                # Keep last 100 sessions
                if len(self._session_history) > 100:
                    self._session_history = self._session_history[-100:]

            self._current_session = ConversationStats(start_time=time.time())

        logger.debug("Started new analytics session")

    def end_session(self) -> Optional[ConversationStats]:
        """End the current conversation session.

        Returns:
            Stats for the ended session, or None if no session
        """
        with self._data_lock:
            if self._current_session:
                self._current_session.end_time = time.time()
                stats = self._current_session
                self._session_history.append(stats)
                self._current_session = None
                return stats
        return None

    def record_output_condensation(
        self,
        condenser: str,
        original_chars: int,
        condensed_chars: int,
    ) -> None:
        """Record one shell-output condensation event.

        Args:
            condenser: Name of the condenser that fired (e.g. "pytest")
            original_chars: Output size before condensation
            condensed_chars: Output size after condensation
        """
        with self._data_lock:
            stats = self._condensation_stats.setdefault(
                condenser,
                {"count": 0, "original_chars": 0, "condensed_chars": 0},
            )
            stats["count"] += 1
            stats["original_chars"] += original_chars
            stats["condensed_chars"] += condensed_chars

        self._maybe_persist()

    def get_condensation_stats(self) -> Dict[str, Any]:
        """Get cumulative shell-output condensation savings (rtk `gain` analogue).

        Token estimates use the chars/4 heuristic; percentages are reliable,
        absolute token numbers are approximate.
        """
        with self._data_lock:
            per_condenser = {name: dict(stats) for name, stats in self._condensation_stats.items()}
        total_original = sum(s["original_chars"] for s in per_condenser.values())
        total_condensed = sum(s["condensed_chars"] for s in per_condenser.values())
        chars_saved = total_original - total_condensed
        return {
            "per_condenser": per_condenser,
            "total_events": sum(s["count"] for s in per_condenser.values()),
            "total_original_chars": total_original,
            "total_condensed_chars": total_condensed,
            "chars_saved": chars_saved,
            "est_tokens_saved": chars_saved // 4,
            "savings_pct": (100.0 * chars_saved / total_original) if total_original else 0.0,
        }

    def record_turn(self) -> None:
        """Record a conversation turn."""
        with self._data_lock:
            if self._current_session:
                self._current_session.turn_count += 1

    def record_provider_switch(self) -> None:
        """Record a provider switch."""
        with self._data_lock:
            if self._current_session:
                self._current_session.provider_switches += 1

    def record_tool_selection(
        self,
        method: str,
        num_tools: int,
        selection_time_ms: float = 0,
        task_type: Optional[str] = None,
        query_hash: Optional[str] = None,
    ) -> None:
        """Record tool selection for analytics.

        This tracks how tools are being selected to optimize the process.

        Args:
            method: Selection method used ("semantic", "keyword", "hybrid")
            num_tools: Number of tools selected
            selection_time_ms: Time taken for selection in milliseconds
            task_type: Type of task (for correlation analysis)
            query_hash: Hash of query (for pattern detection)
        """
        with self._data_lock:
            # Track selection method distribution
            if "_selection_stats" not in self.__dict__:
                self._selection_stats: Dict[str, Dict[str, Any]] = {}

            if method not in self._selection_stats:
                self._selection_stats[method] = {
                    "count": 0,
                    "total_tools_selected": 0,
                    "total_time_ms": 0,
                    "task_types": {},
                }

            stats = self._selection_stats[method]
            stats["count"] += 1
            stats["total_tools_selected"] += num_tools
            stats["total_time_ms"] += selection_time_ms

            if task_type:
                stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1

        logger.debug(
            f"Recorded tool selection: method={method}, tools={num_tools}, "
            f"time={selection_time_ms:.1f}ms"
        )

    def record_first_token(
        self,
        provider_name: str,
        model: str,
        time_to_first_token_ms: float,
    ) -> None:
        """Record time to first token for latency tracking.

        This helps optimize perceived responsiveness.

        Args:
            provider_name: Name of the provider
            model: Model used
            time_to_first_token_ms: Time from request to first token
        """
        with self._data_lock:
            # Track TTFT stats per provider/model
            if "_ttft_stats" not in self.__dict__:
                self._ttft_stats: Dict[str, List[float]] = {}

            key = f"{provider_name}:{model}"
            if key not in self._ttft_stats:
                self._ttft_stats[key] = []

            self._ttft_stats[key].append(time_to_first_token_ms)

            # Keep last 100 samples
            if len(self._ttft_stats[key]) > 100:
                self._ttft_stats[key] = self._ttft_stats[key][-100:]

        logger.debug(
            f"Recorded first token: {provider_name}/{model}, "
            f"TTFT={time_to_first_token_ms:.1f}ms"
        )

    def record_streaming_metrics(
        self,
        provider_name: str,
        model: str,
        total_tokens: int,
        total_duration_ms: float,
        time_to_first_token_ms: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
    ) -> None:
        """Record comprehensive streaming metrics.

        Args:
            provider_name: Name of the provider
            model: Model used
            total_tokens: Total tokens generated
            total_duration_ms: Total streaming duration
            time_to_first_token_ms: Time to first token
            tokens_per_second: Token generation rate
        """
        with self._data_lock:
            # Track streaming stats
            if "_streaming_stats" not in self.__dict__:
                self._streaming_stats: Dict[str, Dict[str, Any]] = {}

            key = f"{provider_name}:{model}"
            if key not in self._streaming_stats:
                self._streaming_stats[key] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_duration_ms": 0,
                    "ttft_samples": [],
                    "tps_samples": [],
                }

            stats = self._streaming_stats[key]
            stats["count"] += 1
            stats["total_tokens"] += total_tokens
            stats["total_duration_ms"] += total_duration_ms

            if time_to_first_token_ms is not None:
                stats["ttft_samples"].append(time_to_first_token_ms)
                if len(stats["ttft_samples"]) > 100:
                    stats["ttft_samples"] = stats["ttft_samples"][-100:]

            if tokens_per_second is not None:
                stats["tps_samples"].append(tokens_per_second)
                if len(stats["tps_samples"]) > 100:
                    stats["tps_samples"] = stats["tps_samples"][-100:]

        logger.debug(
            f"Recorded streaming metrics: {provider_name}/{model}, "
            f"tokens={total_tokens}, duration={total_duration_ms:.1f}ms"
        )

    def get_selection_insights(self) -> Dict[str, Any]:
        """Get insights about tool selection patterns.

        Returns:
            Dictionary with selection statistics and recommendations
        """
        with self._data_lock:
            if not hasattr(self, "_selection_stats") or not self._selection_stats:
                return {"status": "no_data"}

            insights = {}
            for method, stats in self._selection_stats.items():
                avg_tools = (
                    stats["total_tools_selected"] / stats["count"] if stats["count"] > 0 else 0
                )
                avg_time = stats["total_time_ms"] / stats["count"] if stats["count"] > 0 else 0

                insights[method] = {
                    "count": stats["count"],
                    "avg_tools_selected": avg_tools,
                    "avg_selection_time_ms": avg_time,
                    "task_distribution": stats.get("task_types", {}),
                }

            # Add recommendations
            recommendations = []
            if "semantic" in insights and "keyword" in insights:
                if insights["semantic"]["avg_selection_time_ms"] > 100:
                    recommendations.append(
                        "Semantic selection is slow. Consider caching embeddings."
                    )

            insights["recommendations"] = recommendations
            insights["status"] = "ok"

            return insights

    def get_ttft_insights(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get time-to-first-token insights.

        Args:
            provider_name: Optional filter by provider

        Returns:
            Dictionary with TTFT statistics
        """
        with self._data_lock:
            if not hasattr(self, "_ttft_stats") or not self._ttft_stats:
                return {"status": "no_data"}

            insights = {}
            for key, samples in self._ttft_stats.items():
                prov, model = key.split(":", 1)
                if provider_name and prov != provider_name:
                    continue

                if not samples:
                    continue

                sorted_samples = sorted(samples)
                p50_idx = int(len(sorted_samples) * 0.50)
                p95_idx = min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)

                insights[key] = {
                    "samples": len(samples),
                    "avg_ms": sum(samples) / len(samples),
                    "p50_ms": sorted_samples[p50_idx],
                    "p95_ms": sorted_samples[p95_idx],
                    "min_ms": sorted_samples[0],
                    "max_ms": sorted_samples[-1],
                }

            insights["status"] = "ok" if insights else "no_data"
            return insights

    # ========================================================================
    # Aggregate Updates
    # ========================================================================

    def _update_tool_aggregate(self, tool_name: str) -> None:
        """Update aggregated metrics for a tool."""
        records = self._tool_records[tool_name]
        if not records:
            return

        successes = sum(1 for r in records if r.success)
        total = len(records)

        # Calculate averages
        avg_time = sum(r.execution_time_ms for r in records) / total
        avg_tokens = sum(r.context_tokens for r in records) / total

        # Recent trend (last 10 vs overall)
        recent = records[-10:]
        recent_success_rate = sum(1 for r in recent if r.success) / len(recent) if recent else 0

        # Error distribution
        errors = [r.error_type for r in records if r.error_type]
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1

        self._tool_aggregates[tool_name] = {
            "total_executions": total,
            "success_count": successes,
            "success_rate": successes / total,
            "avg_execution_ms": avg_time,
            "avg_context_tokens": avg_tokens,
            "recent_success_rate": recent_success_rate,
            "error_distribution": error_counts,
            "last_execution": records[-1].timestamp if records else 0,
        }

    def _update_provider_aggregate(self, provider_name: str) -> None:
        """Update aggregated metrics for a provider."""
        records = self._provider_records[provider_name]
        if not records:
            return

        successes = sum(1 for r in records if r.success)
        total = len(records)

        # Calculate averages
        avg_latency = sum(r.latency_ms for r in records) / total
        total_tokens_in = sum(r.tokens_in for r in records)
        total_tokens_out = sum(r.tokens_out for r in records)

        # P50, P95, P99 latencies
        sorted_latencies = sorted(r.latency_ms for r in records)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)

        self._provider_aggregates[provider_name] = {
            "total_calls": total,
            "success_count": successes,
            "success_rate": successes / total,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": sorted_latencies[p50_idx] if sorted_latencies else 0,
            "p95_latency_ms": sorted_latencies[p95_idx] if sorted_latencies else 0,
            "p99_latency_ms": sorted_latencies[p99_idx] if sorted_latencies else 0,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "last_call": records[-1].timestamp if records else 0,
        }

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_tool_insights(self, tool_name: str) -> Dict[str, Any]:
        """Get optimization insights for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with insights and recommendations
        """
        with self._data_lock:
            if tool_name not in self._tool_aggregates:
                return {"status": "no_data", "tool_name": tool_name}

            agg = self._tool_aggregates[tool_name]
            insights = dict(agg)
            if "execution_count" not in insights:
                insights["execution_count"] = agg.get("total_executions", 0)

            # Add recommendations
            recommendations = []

            if agg["success_rate"] < 0.8:
                recommendations.append(
                    f"Low success rate ({agg['success_rate']:.0%}). "
                    f"Review error types: {agg['error_distribution']}"
                )

            if agg["avg_execution_ms"] > 1000:
                recommendations.append(
                    f"Slow execution (avg {agg['avg_execution_ms']:.0f}ms). "
                    "Consider caching or optimization."
                )

            if agg["recent_success_rate"] < agg["success_rate"] - 0.1:
                recommendations.append("Recent performance degraded. Check for issues.")

            insights["recommendations"] = recommendations
            insights["status"] = "ok"

            return insights

    def get_provider_insights(self, provider_name: str) -> Dict[str, Any]:
        """Get optimization insights for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with insights and recommendations
        """
        with self._data_lock:
            if provider_name not in self._provider_aggregates:
                return {"status": "no_data", "provider_name": provider_name}

            agg = self._provider_aggregates[provider_name]
            insights = dict(agg)

            # Add recommendations
            recommendations = []

            if agg["success_rate"] < 0.95:
                recommendations.append(
                    f"Reliability below 95% ({agg['success_rate']:.1%}). "
                    "Consider fallback provider."
                )

            if agg["p99_latency_ms"] > 5000:
                recommendations.append(
                    f"High tail latency (P99: {agg['p99_latency_ms']:.0f}ms). "
                    "May cause timeout issues."
                )

            insights["recommendations"] = recommendations
            insights["status"] = "ok"

            return insights

    def get_top_tools(self, metric: str = "usage", limit: int = 10) -> List[Tuple[str, float]]:
        """Get top tools by a given metric.

        Args:
            metric: One of "usage", "success_rate", "avg_time"
            limit: Maximum tools to return

        Returns:
            List of (tool_name, value) tuples sorted by metric
        """
        with self._data_lock:
            if metric == "usage":
                key = "total_executions"
            elif metric == "success_rate":
                key = "success_rate"
            elif metric == "avg_time":
                key = "avg_execution_ms"
            else:
                key = "total_executions"

            sorted_tools = sorted(
                self._tool_aggregates.items(),
                key=lambda x: x[1].get(key, 0),
                reverse=True if metric != "avg_time" else False,
            )

            return [(name, agg.get(key, 0)) for name, agg in sorted_tools[:limit]]

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of conversation sessions.

        Returns:
            Dictionary with session statistics
        """
        with self._data_lock:
            if not self._session_history:
                return {"status": "no_sessions"}

            total_sessions = len(self._session_history)
            avg_turns = sum(s.turn_count for s in self._session_history) / total_sessions
            avg_tools = sum(s.tool_calls for s in self._session_history) / total_sessions
            avg_tokens = sum(s.total_tokens for s in self._session_history) / total_sessions
            avg_duration = (
                sum((s.end_time or s.start_time) - s.start_time for s in self._session_history)
                / total_sessions
            )

            return {
                "total_sessions": total_sessions,
                "avg_turns_per_session": avg_turns,
                "avg_tool_calls_per_session": avg_tools,
                "avg_tokens_per_session": avg_tokens,
                "avg_session_duration_seconds": avg_duration,
            }

    def persist_to_rl_database(self, repo_id: Optional[str] = None) -> bool:
        """Persist the current session summary to the RL outcomes database.

        Bridges in-memory UsageAnalytics data into the RL framework so that
        MetaLearningCoordinator can query historical trends without re-implementing
        session aggregation.

        Args:
            repo_id: Optional repository identifier for isolation

        Returns:
            True if persisted successfully, False if nothing to persist or error
        """
        summary = self.get_session_summary()
        if summary.get("status") == "no_sessions":
            return False

        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator
            from victor.framework.rl.base import RLOutcome
            import json

            coord = get_rl_coordinator()
            if repo_id:
                coord.set_repo_context(repo_id)

            outcome = RLOutcome(
                provider="usage_analytics",
                model="aggregation",
                task_type="session_summary",
                success=True,
                quality_score=None,
                metadata={
                    "feedback_source": "auto",
                    "session_summary": json_dumps(summary),
                    "total_sessions": summary.get("total_sessions", 0),
                    "avg_turns_per_session": summary.get("avg_turns_per_session", 0.0),
                    "avg_tool_calls_per_session": summary.get("avg_tool_calls_per_session", 0.0),
                },
                vertical="general",
            )
            coord.record_outcome("tool_selector", outcome)
            return True
        except Exception as e:
            logger.debug("UsageAnalytics: persist_to_rl_database failed: %s", e)
            return False

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get actionable optimization recommendations.

        Returns:
            List of recommendations with priority and actions
        """
        recommendations = []

        with self._data_lock:
            # Analyze tool performance
            for tool_name, agg in self._tool_aggregates.items():
                if agg["success_rate"] < 0.7:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "tool_reliability",
                            "tool": tool_name,
                            "issue": f"Low success rate: {agg['success_rate']:.0%}",
                            "action": "Review error logs and consider deprecation",
                        }
                    )

                if agg["avg_execution_ms"] > 2000:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "tool_performance",
                            "tool": tool_name,
                            "issue": f"Slow execution: {agg['avg_execution_ms']:.0f}ms avg",
                            "action": "Consider caching or async execution",
                        }
                    )

            # Analyze provider performance
            for provider, agg in self._provider_aggregates.items():
                if agg["success_rate"] < 0.9:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "provider_reliability",
                            "provider": provider,
                            "issue": f"Low reliability: {agg['success_rate']:.1%}",
                            "action": "Configure automatic fallback",
                        }
                    )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return recommendations

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        lines.append("# HELP victor_tool_executions_total Total tool executions")
        lines.append("# TYPE victor_tool_executions_total counter")

        with self._data_lock:
            for tool_name, agg in self._tool_aggregates.items():
                lines.append(
                    f'victor_tool_executions_total{{tool="{tool_name}"}} '
                    f'{agg["total_executions"]}'
                )

            lines.append("")
            lines.append("# HELP victor_tool_success_rate Tool success rate")
            lines.append("# TYPE victor_tool_success_rate gauge")

            for tool_name, agg in self._tool_aggregates.items():
                lines.append(
                    f'victor_tool_success_rate{{tool="{tool_name}"}} ' f'{agg["success_rate"]:.4f}'
                )

            lines.append("")
            lines.append("# HELP victor_tool_execution_ms Tool execution time in ms")
            lines.append("# TYPE victor_tool_execution_ms gauge")

            for tool_name, agg in self._tool_aggregates.items():
                lines.append(
                    f'victor_tool_execution_ms{{tool="{tool_name}"}} '
                    f'{agg["avg_execution_ms"]:.2f}'
                )

            lines.append("")
            lines.append("# HELP victor_provider_latency_ms Provider latency in ms")
            lines.append("# TYPE victor_provider_latency_ms gauge")

            for provider, agg in self._provider_aggregates.items():
                lines.append(
                    f'victor_provider_latency_ms{{provider="{provider}",quantile="0.5"}} '
                    f'{agg["p50_latency_ms"]:.2f}'
                )
                lines.append(
                    f'victor_provider_latency_ms{{provider="{provider}",quantile="0.95"}} '
                    f'{agg["p95_latency_ms"]:.2f}'
                )
                lines.append(
                    f'victor_provider_latency_ms{{provider="{provider}",quantile="0.99"}} '
                    f'{agg["p99_latency_ms"]:.2f}'
                )

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export all analytics as JSON.

        Returns:
            JSON string with all analytics data
        """
        with self._data_lock:
            data = {
                "exported_at": time.time(),
                "tool_aggregates": self._tool_aggregates,
                "provider_aggregates": self._provider_aggregates,
                "session_summary": self.get_session_summary(),
                "recommendations": self.get_optimization_recommendations(),
            }

        return json_dumps(data, indent=2, default=str)

    # ========================================================================
    # Persistence
    # ========================================================================

    def _maybe_persist(self) -> None:
        """Persist to disk if enough time has passed."""
        if not self._cache_file:
            return

        current_time = time.time()
        if current_time - self._last_persist_time > self.config.persistence_interval_seconds:
            self._persist_to_cache()
            self._last_persist_time = current_time

    def _persist_to_cache(self) -> None:
        """Persist analytics to disk cache."""
        if not self._cache_file:
            return

        try:
            with self._data_lock:
                data = {
                    "tool_records": dict(self._tool_records),
                    "provider_records": dict(self._provider_records),
                    "session_history": self._session_history,
                    "tool_aggregates": self._tool_aggregates,
                    "provider_aggregates": self._provider_aggregates,
                    "condensation_stats": self._condensation_stats,
                }

            with open(self._cache_file, "wb") as f:
                pickle.dump(data, f)

            logger.debug(f"Persisted analytics to {self._cache_file}")

        except Exception as e:
            logger.warning(f"Failed to persist analytics: {e}")

    def _load_from_cache(self) -> None:
        """Load analytics from disk cache."""
        if not self._cache_file or not self._cache_file.exists():
            return

        try:
            with open(self._cache_file, "rb") as f:
                data = pickle.load(f)

            self._tool_records = defaultdict(list, data.get("tool_records", {}))
            self._provider_records = defaultdict(list, data.get("provider_records", {}))
            self._session_history = data.get("session_history", [])
            self._tool_aggregates = data.get("tool_aggregates", {})
            self._provider_aggregates = data.get("provider_aggregates", {})
            self._condensation_stats = data.get("condensation_stats", {})

            logger.info(
                f"Loaded analytics from cache: "
                f"{len(self._tool_records)} tools, "
                f"{len(self._provider_records)} providers"
            )

        except Exception as e:
            logger.warning(f"Failed to load analytics cache: {e}")

    def flush(self) -> None:
        """Force persist to disk."""
        self._persist_to_cache()

    def clear(self) -> None:
        """Clear all analytics data."""
        with self._data_lock:
            self._tool_records.clear()
            self._provider_records.clear()
            self._session_history.clear()
            self._tool_aggregates.clear()
            self._provider_aggregates.clear()
            self._current_session = None

        logger.debug("Analytics data cleared")

    # =========================================================================
    # Async API + Write Buffering
    # =========================================================================

    def _ensure_buffer_state(self) -> None:
        """Initialize buffer state if not already done."""
        if not hasattr(self, "_tool_buffer"):
            self._tool_buffer: List[Dict[str, Any]] = []
            self._provider_buffer: List[Dict[str, Any]] = []
            self._selection_buffer: List[Dict[str, Any]] = []
            self._streaming_buffer: List[Dict[str, Any]] = []
            self._buffer_lock = threading.Lock()
            self._buffer_size = 100
            self._flush_interval = 5.0
            self._flush_task: Optional[asyncio.Task] = None
            self._auto_flush_running = False
            self._total_buffered = 0
            self._total_flushed = 0

    def buffer_tool_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        error_type: Optional[str] = None,
        context_tokens: int = 0,
        query_hash: Optional[str] = None,
    ) -> None:
        """Buffer a tool execution record for later batch writing.

        Non-blocking - queues the record for later writing.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            execution_time_ms: Execution time in milliseconds
            error_type: Error type if failed
            context_tokens: Number of context tokens
            query_hash: Hash of the query
        """
        self._ensure_buffer_state()
        record = {
            "tool_name": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error_type": error_type,
            "context_tokens": context_tokens,
            "query_hash": query_hash,
        }

        with self._buffer_lock:
            self._tool_buffer.append(record)
            self._total_buffered += 1

            # Auto-flush if buffer is full
            if len(self._tool_buffer) >= self._buffer_size:
                self._flush_tool_buffer_internal()

    def buffer_provider_call(
        self,
        provider_name: str,
        model: str,
        success: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """Buffer a provider call record for later batch writing.

        Non-blocking - queues the record for later writing.

        Args:
            provider_name: Name of the provider
            model: Model used
            success: Whether call succeeded
            latency_ms: Total latency
            tokens_in: Input tokens
            tokens_out: Output tokens
            error_type: Error type if failed
        """
        self._ensure_buffer_state()
        record = {
            "provider_name": provider_name,
            "model": model,
            "success": success,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error_type": error_type,
        }

        with self._buffer_lock:
            self._provider_buffer.append(record)
            self._total_buffered += 1

            # Auto-flush if buffer is full
            if len(self._provider_buffer) >= self._buffer_size:
                self._flush_provider_buffer_internal()

    def buffer_tool_selection(
        self,
        method: str,
        num_tools: int,
        selection_time_ms: float = 0,
        task_type: Optional[str] = None,
        query_hash: Optional[str] = None,
    ) -> None:
        """Buffer a tool selection record.

        Args:
            method: Selection method
            num_tools: Number of tools selected
            selection_time_ms: Time taken
            task_type: Task type
            query_hash: Query hash
        """
        self._ensure_buffer_state()
        record = {
            "method": method,
            "num_tools": num_tools,
            "selection_time_ms": selection_time_ms,
            "task_type": task_type,
            "query_hash": query_hash,
        }

        with self._buffer_lock:
            self._selection_buffer.append(record)
            self._total_buffered += 1

    def buffer_streaming_metrics(
        self,
        provider_name: str,
        model: str,
        total_tokens: int,
        total_duration_ms: float,
        time_to_first_token_ms: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
    ) -> None:
        """Buffer streaming metrics.

        Args:
            provider_name: Provider name
            model: Model used
            total_tokens: Tokens generated
            total_duration_ms: Total duration
            time_to_first_token_ms: TTFT
            tokens_per_second: Generation rate
        """
        self._ensure_buffer_state()
        record = {
            "provider_name": provider_name,
            "model": model,
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration_ms,
            "time_to_first_token_ms": time_to_first_token_ms,
            "tokens_per_second": tokens_per_second,
        }

        with self._buffer_lock:
            self._streaming_buffer.append(record)
            self._total_buffered += 1

    def _flush_tool_buffer_internal(self) -> int:
        """Flush tool buffer (called under lock)."""
        if not self._tool_buffer:
            return 0

        records = self._tool_buffer.copy()
        self._tool_buffer.clear()

        count = 0
        for record in records:
            try:
                self.record_tool_execution(**record)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to flush tool record: {e}")

        self._total_flushed += count
        return count

    def _flush_provider_buffer_internal(self) -> int:
        """Flush provider buffer (called under lock)."""
        if not self._provider_buffer:
            return 0

        records = self._provider_buffer.copy()
        self._provider_buffer.clear()

        count = 0
        for record in records:
            try:
                self.record_provider_call(**record)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to flush provider record: {e}")

        self._total_flushed += count
        return count

    def _flush_selection_buffer_internal(self) -> int:
        """Flush selection buffer (called under lock)."""
        if not self._selection_buffer:
            return 0

        records = self._selection_buffer.copy()
        self._selection_buffer.clear()

        count = 0
        for record in records:
            try:
                self.record_tool_selection(**record)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to flush selection record: {e}")

        self._total_flushed += count
        return count

    def _flush_streaming_buffer_internal(self) -> int:
        """Flush streaming buffer (called under lock)."""
        if not self._streaming_buffer:
            return 0

        records = self._streaming_buffer.copy()
        self._streaming_buffer.clear()

        count = 0
        for record in records:
            try:
                self.record_streaming_metrics(**record)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to flush streaming record: {e}")

        self._total_flushed += count
        return count

    def flush_buffers_sync(self) -> int:
        """Flush all buffers synchronously.

        Returns:
            Total number of records flushed
        """
        self._ensure_buffer_state()

        with self._buffer_lock:
            total = 0
            total += self._flush_tool_buffer_internal()
            total += self._flush_provider_buffer_internal()
            total += self._flush_selection_buffer_internal()
            total += self._flush_streaming_buffer_internal()

            if total > 0:
                logger.debug(f"UsageAnalytics: Flushed {total} buffered records")

            return total

    async def flush_buffers_async(self) -> int:
        """Flush all buffers asynchronously.

        Runs the sync flush in a thread pool to avoid blocking.

        Returns:
            Total number of records flushed
        """
        return await asyncio.to_thread(self.flush_buffers_sync)

    async def start_auto_flush(self, interval_seconds: Optional[float] = None) -> None:
        """Start background task that periodically flushes buffers.

        Args:
            interval_seconds: Override flush interval (default 5.0)
        """
        self._ensure_buffer_state()

        if self._auto_flush_running:
            return

        if interval_seconds is not None:
            self._flush_interval = interval_seconds

        self._auto_flush_running = True

        async def _flush_loop() -> None:
            while self._auto_flush_running:
                await asyncio.sleep(self._flush_interval)
                if self._auto_flush_running:
                    try:
                        await self.flush_buffers_async()
                    except Exception as e:
                        logger.warning(f"Auto-flush error: {e}")

        self._flush_task = asyncio.create_task(_flush_loop())
        logger.debug(f"UsageAnalytics: Started auto-flush (interval={self._flush_interval}s)")

    async def stop_auto_flush(self) -> None:
        """Stop background flush task and flush remaining records."""
        self._auto_flush_running = False

        if hasattr(self, "_flush_task") and self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self.flush_buffers_async()
        logger.debug("UsageAnalytics: Stopped auto-flush")

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dict with buffer sizes and flush counts
        """
        self._ensure_buffer_state()

        with self._buffer_lock:
            return {
                "total_buffered": self._total_buffered,
                "total_flushed": self._total_flushed,
                "pending_tool_records": len(self._tool_buffer),
                "pending_provider_records": len(self._provider_buffer),
                "pending_selection_records": len(self._selection_buffer),
                "pending_streaming_records": len(self._streaming_buffer),
                "auto_flush_running": self._auto_flush_running,
            }


def create_usage_analytics(cache_dir: Optional[Path] = None) -> UsageAnalytics:
    """Factory function to create configured UsageAnalytics.

    Args:
        cache_dir: Optional cache directory for persistence

    Returns:
        Configured UsageAnalytics instance
    """
    if cache_dir is None:
        from victor.config.settings import get_project_paths

        cache_dir = get_project_paths().global_cache_dir / "analytics"

    config = AnalyticsConfig(cache_dir=cache_dir)
    return UsageAnalytics.get_instance(config)
