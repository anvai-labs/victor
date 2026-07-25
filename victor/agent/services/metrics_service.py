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

"""Service-owned metrics runtime for agent orchestration.

This module provides AgentMetricsService, a service-owned facade for
metrics collection, cost tracking, and usage analytics in the orchestrator.

The coordinator wraps:
- MetricsCollector: Stream metrics, tool selection, execution stats
- SessionCostTracker: Per-request and session-cumulative cost tracking
- Token usage tracking for evaluation/benchmarking

The legacy `victor.agent.coordinators.metrics_coordinator` module now re-exports
this implementation for compatibility.
"""

import logging
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.session_cost_tracker import SessionCostTracker
    from victor.agent.stream_handler import StreamMetrics
    from victor.evaluation.protocol import TokenUsage

logger = logging.getLogger(__name__)

_PROMPT_CACHING_PROVIDER_NAMES = frozenset(
    {
        "anthropic",
        "openai",
        "google",
        "xai",
        "deepseek",
        "groq",
        "bedrock",
        "vertex",
        "azure_openai",
        "openrouter",
        "fireworks",
        "together",
        "cerebras",
        "moonshot",
    }
)


@dataclass(frozen=True)
class TaskExecutionReport:
    """First-class per-task token and cost report."""

    task_id: str
    description: str
    task_type: str
    started_at: float
    finished_at: float
    duration_seconds: float
    success: bool
    request_count: int
    api_prompt_tokens: int
    api_completion_tokens: int
    api_total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_hit_rate: float
    tool_schema_tokens: int
    compaction_saved_tokens: int
    compaction_messages_removed: int
    tokens_per_successful_task: float
    total_cost_usd: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class _TaskUsageSnapshot:
    """Cumulative metrics snapshot captured at task boundaries."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_cost_usd: float
    request_count: int


@dataclass
class _ActiveTaskReport:
    """Task report state held while execution is in flight."""

    task_id: str
    description: str
    task_type: str
    started_at: float
    metadata: Dict[str, Any]
    snapshot: _TaskUsageSnapshot


class AgentMetricsService:
    """Coordinates all metrics collection and tracking for the orchestrator.

    This class serves as a facade for:
    - Stream metrics (time-to-first-token, duration, tokens/second)
    - Tool selection and execution statistics
    - Cost tracking (per-request and session-cumulative)
    - Token usage for evaluation tracking

    The coordinator delegates to specialized components:
    - MetricsCollector: Real-time metrics collection
    - SessionCostTracker: Cost tracking and export

    Example:
        coordinator = AgentMetricsService(
            metrics_collector=collector,
            session_cost_tracker=cost_tracker,
            cumulative_token_usage={"prompt_tokens": 0, ...}
        )

        # After streaming
        metrics = coordinator.finalize_stream_metrics(usage_data)

        # Get cost summary
        summary = coordinator.get_session_cost_summary()
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        session_cost_tracker: "SessionCostTracker",
        cumulative_token_usage: Dict[str, int],
    ) -> None:
        """Initialize the metrics coordinator.

        Args:
            metrics_collector: The metrics collector instance
            session_cost_tracker: The session cost tracker instance
            cumulative_token_usage: Mutable dict for cumulative token tracking
        """
        self._metrics_collector = metrics_collector
        self._session_cost_tracker = session_cost_tracker
        self._cumulative_token_usage = cumulative_token_usage
        self._active_task_report: Optional[_ActiveTaskReport] = None
        self._last_task_report: Optional[TaskExecutionReport] = None
        self._task_report_history: List[TaskExecutionReport] = []
        self._successful_task_count = 0
        self._successful_task_token_total = 0
        self._last_tool_strategy_event: Optional[Dict[str, Any]] = None

    # ========================================================================
    # Stream Metrics
    # ========================================================================

    def finalize_stream_metrics(
        self,
        usage_data: Optional[Dict[str, int]] = None,
        provider_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional["StreamMetrics"]:
        """Finalize stream metrics at end of streaming session.

        This method:
        1. Finalizes stream metrics via MetricsCollector
        2. Records to SessionCostTracker for cumulative tracking
        3. Returns the complete metrics

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.
            provider_diagnostics: Optional aggregated Sandhi transport
                       diagnostics (attempts, usage completeness, upstream
                       request ids) for the turn.

        Returns:
            StreamMetrics with complete session data or None
        """
        metrics = self._metrics_collector.finalize_stream_metrics(
            usage_data, provider_diagnostics=provider_diagnostics
        )

        # Record to session cost tracker for cumulative tracking
        if metrics:
            self._session_cost_tracker.record_request(
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                cache_read_tokens=metrics.cache_read_tokens,
                cache_write_tokens=metrics.cache_write_tokens,
                duration_seconds=metrics.total_duration,
                tool_calls=metrics.tool_calls_count,
            )

        return metrics

    def get_last_stream_metrics(self) -> Optional["StreamMetrics"]:
        """Get metrics from the last streaming session.

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_collector.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_collector.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_collector.get_streaming_metrics_history(limit)

    # ========================================================================
    # Cost Tracking
    # ========================================================================

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Returns:
            Dictionary with session cost statistics including:
            - Total tokens and costs by type
            - Per-request breakdown
            - Session totals
        """
        return self._session_cost_tracker.get_summary()

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        return self._session_cost_tracker.format_inline_cost()

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        output_path = Path(path)
        if format == "csv":
            self._session_cost_tracker.export_csv(output_path)
        else:
            self._session_cost_tracker.export_json(output_path)

    # ========================================================================
    # Token Usage (for Evaluation/Benchmarking)
    # ========================================================================

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        from victor.evaluation.protocol import TokenUsage

        return TokenUsage(
            input_tokens=self._cumulative_token_usage.get("prompt_tokens", 0),
            output_tokens=self._cumulative_token_usage.get("completion_tokens", 0),
            total_tokens=self._cumulative_token_usage.get("total_tokens", 0),
            cached_tokens=self._cumulative_token_usage.get("cached_tokens", 0),
            cache_miss_tokens=self._cumulative_token_usage.get("cache_miss_tokens", 0),
            reasoning_tokens=self._cumulative_token_usage.get("reasoning_tokens", 0),
            cost_usd_micros=self._cumulative_token_usage.get("cost_usd_micros", 0),
        )

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Call this at the start of a new evaluation task to get fresh counts.
        """
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    def update_cumulative_token_usage(self, usage_data: Dict[str, int]) -> None:
        """Update cumulative token usage from provider response.

        Args:
            usage_data: Token usage dict with prompt_tokens, completion_tokens, etc.
        """
        for key, value in usage_data.items():
            if key in self._cumulative_token_usage:
                self._cumulative_token_usage[key] += value

    # ========================================================================
    # Tool Usage Statistics
    # ========================================================================

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
        return self._metrics_collector.get_tool_usage_stats(
            conversation_state_summary=conversation_state_summary
        )

    # ========================================================================
    # Tool Strategy Observability
    # ========================================================================

    def emit_tool_strategy_event(
        self,
        *,
        strategy: str,
        tool_count: int,
        tool_tokens: int,
        context_window: int,
        provider: Any,
        model: str,
        reason: str,
        tools: Optional[List[Any]] = None,
        v2_enabled: bool = False,
    ) -> None:
        """Emit tool-strategy observability logs and metrics.

        Args:
            strategy: Strategy name (session_lock, semantic_selection, etc.)
            tool_count: Number of tools selected
            tool_tokens: Total tool tokens
            context_window: Context window size
            provider: Provider instance or provider name string
            model: Active model name
            reason: Reason for strategy selection
            tools: Optional list of selected tools
            v2_enabled: Whether tool strategy v2 is active
        """
        from victor.config.tool_tiers import get_tool_tier

        provider_name = self._normalize_provider_name(provider)
        max_tool_tokens = int(context_window * 0.25)
        context_utilization = (tool_tokens / max_tool_tokens) if max_tool_tokens > 0 else 0
        provider_category = self.get_provider_category(provider)

        tier_distribution: Dict[str, int] = {}
        if tools:
            for tool in tools:
                tool_name = getattr(tool, "name", None)
                if not tool_name:
                    continue
                tier = get_tool_tier(tool_name)
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        event_data = {
            "event_type": "tool.strategy_chosen",
            "strategy": strategy,
            "tool_count": tool_count,
            "tool_tokens": tool_tokens,
            "context_window": context_window,
            "max_tool_tokens": max_tool_tokens,
            "context_utilization": round(context_utilization, 3),
            "provider": provider_name,
            "provider_category": provider_category,
            "model": model,
            "reason": reason,
            "tier_distribution": tier_distribution,
            "v2_enabled": v2_enabled,
        }

        logger.info(
            f"Tool strategy v2={event_data['v2_enabled']}: "
            f"{strategy} ({tool_count} tools, {tool_tokens} tokens, "
            f"{context_utilization:.1%} context utilization, "
            f"provider={provider_name}, category={provider_category}, "
            f"reason={reason})"
        )

        if tier_distribution:
            tier_summary = ", ".join(f"{k}:{v}" for k, v in sorted(tier_distribution.items()))
            logger.debug(f"Tool tier distribution: {tier_summary}")

        self._last_tool_strategy_event = event_data
        self.emit_tool_strategy_metrics(event_data)

    def get_last_tool_strategy_event(self) -> Optional[Dict[str, Any]]:
        """Return the most recent tool-strategy selection event."""
        if self._last_tool_strategy_event is None:
            return None
        return dict(self._last_tool_strategy_event)

    # ========================================================================
    # Task Reports
    # ========================================================================

    def start_task_report(
        self,
        description: str,
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Capture a cumulative snapshot for the start of a task."""
        task_id = str(uuid.uuid4())
        self._active_task_report = _ActiveTaskReport(
            task_id=task_id,
            description=description,
            task_type=task_type or "default",
            started_at=time.time(),
            metadata=dict(metadata or {}),
            snapshot=self._snapshot_task_usage(),
        )
        return task_id

    def finish_task_report(
        self,
        success: bool,
        *,
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tool_schema_tokens: Optional[int] = None,
        compaction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finalize the active task report and return a structured summary."""
        active = self._active_task_report
        if active is None:
            logger.debug("finish_task_report called without an active task")
            return {}

        finished_at = time.time()
        final_snapshot = self._snapshot_task_usage()
        merged_metadata = dict(active.metadata)
        if metadata:
            merged_metadata.update(metadata)
        self._normalize_workspace_report_metadata(merged_metadata)

        prompt_delta = max(0, final_snapshot.prompt_tokens - active.snapshot.prompt_tokens)
        completion_delta = max(
            0, final_snapshot.completion_tokens - active.snapshot.completion_tokens
        )
        total_delta = max(0, final_snapshot.total_tokens - active.snapshot.total_tokens)
        cached_delta = max(0, final_snapshot.cached_tokens - active.snapshot.cached_tokens)
        cache_read_delta = max(
            0, final_snapshot.cache_read_tokens - active.snapshot.cache_read_tokens
        )
        cache_write_delta = max(
            0, final_snapshot.cache_write_tokens - active.snapshot.cache_write_tokens
        )
        request_delta = max(0, final_snapshot.request_count - active.snapshot.request_count)
        total_cost_delta = max(0.0, final_snapshot.total_cost_usd - active.snapshot.total_cost_usd)

        cache_input_tokens = cached_delta or cache_read_delta
        cache_hit_rate = (
            cache_input_tokens / max(cache_input_tokens + prompt_delta, 1)
            if (cache_input_tokens + prompt_delta) > 0
            else 0.0
        )

        compaction = compaction or {}
        compaction_saved_tokens = int(compaction.get("saved_tokens", 0) or 0)
        compaction_messages_removed = int(compaction.get("messages_removed", 0) or 0)
        if "occurred" in compaction:
            merged_metadata["compaction_occurred"] = bool(compaction["occurred"])
        if compaction.get("summary"):
            merged_metadata["compaction_summary"] = str(compaction["summary"])
        if compaction.get("strategy"):
            merged_metadata["compaction_strategy"] = str(compaction["strategy"])
        if compaction.get("reason"):
            merged_metadata["compaction_reason"] = str(compaction["reason"])
        if compaction.get("policy_reason"):
            merged_metadata["compaction_policy_reason"] = str(compaction["policy_reason"])

        effective_task_type = task_type or active.task_type
        report = TaskExecutionReport(
            task_id=active.task_id,
            description=active.description,
            task_type=effective_task_type,
            started_at=active.started_at,
            finished_at=finished_at,
            duration_seconds=max(0.0, finished_at - active.started_at),
            success=success,
            request_count=request_delta,
            api_prompt_tokens=prompt_delta,
            api_completion_tokens=completion_delta,
            api_total_tokens=total_delta,
            cache_read_tokens=cache_read_delta,
            cache_write_tokens=cache_write_delta,
            cache_hit_rate=cache_hit_rate,
            tool_schema_tokens=int(tool_schema_tokens or 0),
            compaction_saved_tokens=compaction_saved_tokens,
            compaction_messages_removed=compaction_messages_removed,
            tokens_per_successful_task=0.0,
            total_cost_usd=total_cost_delta,
            error=error,
            metadata=merged_metadata,
        )

        if success:
            self._successful_task_count += 1
            self._successful_task_token_total += total_delta

        tokens_per_successful_task = (
            self._successful_task_token_total / self._successful_task_count
            if self._successful_task_count > 0
            else 0.0
        )
        report = TaskExecutionReport(
            **{
                **report.to_dict(),
                "tokens_per_successful_task": tokens_per_successful_task,
            }
        )

        self._last_task_report = report
        self._task_report_history.append(report)
        if len(self._task_report_history) > 100:
            self._task_report_history = self._task_report_history[-100:]
        self._active_task_report = None

        logger.info(
            "Task report complete: success=%s task_type=%s total_tokens=%s cache_hit_rate=%.2f",
            success,
            effective_task_type,
            total_delta,
            cache_hit_rate,
        )

        return report.to_dict()

    def get_last_task_report(self) -> Optional[Dict[str, Any]]:
        """Return the most recent completed task report."""
        if self._last_task_report is None:
            return None
        return self._last_task_report.to_dict()

    def get_task_report_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent completed task reports."""
        if limit <= 0:
            return []
        return [report.to_dict() for report in self._task_report_history[-limit:]]

    def get_provider_category(self, provider: Any) -> str:
        """Classify provider for tool-strategy metrics."""
        supports_caching = getattr(provider, "supports_prompt_caching", None)
        if callable(supports_caching) and supports_caching():
            return "caching"

        provider_name = self._normalize_provider_name(provider)
        return "caching" if provider_name.lower() in _PROMPT_CACHING_PROVIDER_NAMES else "local"

    def emit_tool_strategy_metrics(self, event_data: Dict[str, Any]) -> None:
        """Emit tool-strategy metrics in Prometheus-compatible log format."""
        try:
            labels = (
                f'provider="{event_data["provider"]}",'
                f'category="{event_data["provider_category"]}",'
                f'strategy="{event_data["strategy"]}",'
                f'model="{event_data["model"]}"'
            )

            logger.debug(f"METRIC: victor_tool_strategy decisions{{{labels}}} 1")
            logger.debug(f'METRIC: victor_tool_count{{{labels}}} {event_data["tool_count"]}')
            logger.debug(f'METRIC: victor_tool_tokens{{{labels}}} {event_data["tool_tokens"]}')
            logger.debug(
                f'METRIC: victor_context_utilization{{{labels}}} {event_data["context_utilization"]:.3f}'
            )

            v2_labels = f'provider="{event_data["provider"]}"'
            logger.debug(
                f'METRIC: victor_tool_strategy_v2_enabled{{{v2_labels}}} {int(event_data["v2_enabled"])}'
            )

            for tier, count in event_data.get("tier_distribution", {}).items():
                tier_labels = f'provider="{event_data["provider"]}",' f'tier="{tier}"'
                logger.debug(f"METRIC: victor_tool_tier_count{{{tier_labels}}} {count}")

        except Exception as e:
            logger.debug(f"Failed to emit tool strategy metrics: {e}")

    @staticmethod
    def _normalize_provider_name(provider: Any) -> str:
        """Normalize provider object/string to a stable provider name."""
        if isinstance(provider, str):
            return provider

        provider_name = getattr(provider, "name", None)
        if isinstance(provider_name, str) and provider_name:
            return provider_name

        return "unknown"

    def _normalize_workspace_report_metadata(self, metadata: Dict[str, Any]) -> None:
        """Promote workspace policy/diagnostics into stable task-report fields."""
        diagnostics = self._normalize_workspace_diagnostics(
            metadata.get("workspace_isolation_diagnostics")
        )
        if diagnostics:
            metadata["workspace_isolation_diagnostics"] = diagnostics
            metadata["workspace_isolation_diagnostic_count"] = len(diagnostics)
            metadata["workspace_isolation_diagnostic_reasons"] = dict(
                Counter(
                    str(diagnostic.get("reason"))
                    for diagnostic in diagnostics
                    if diagnostic.get("reason")
                )
            )
            metadata["workspace_isolation_diagnostic_operations"] = dict(
                Counter(
                    str(diagnostic.get("operation"))
                    for diagnostic in diagnostics
                    if diagnostic.get("operation")
                )
            )

        raw_policy = metadata.get("workspace_isolation_policy")
        if not isinstance(raw_policy, Mapping):
            raw_policy = metadata.get("workspace_policy")
        if isinstance(raw_policy, Mapping):
            policy = dict(raw_policy)
            metadata["workspace_isolation_policy"] = policy
            for key in (
                "mode",
                "worktree_isolation",
                "materialize_worktrees",
                "dry_run_worktrees",
                "auto_merge_worktrees",
                "allow_risky_worktree_merge",
                "preserve_merge_workspace",
                "cleanup_worktrees",
            ):
                if key in policy:
                    metadata[f"workspace_policy_{key}"] = policy[key]

    @staticmethod
    def _normalize_workspace_diagnostics(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            diagnostic = dict(item)
            reason = str(
                diagnostic.get("reason")
                or diagnostic.get("blocked_reason")
                or diagnostic.get("type")
                or "workspace_isolation"
            ).strip()
            message = str(diagnostic.get("message") or diagnostic.get("error") or reason).strip()
            operation = str(diagnostic.get("operation") or "workspace_isolation").strip()
            severity = str(diagnostic.get("severity") or "warning").strip()
            details = diagnostic.get("details")
            diagnostic["reason"] = reason or "workspace_isolation"
            diagnostic["message"] = message or diagnostic["reason"]
            diagnostic["operation"] = operation or "workspace_isolation"
            diagnostic["severity"] = severity or "warning"
            diagnostic["details"] = dict(details) if isinstance(details, Mapping) else {}
            normalized.append(diagnostic)
        return normalized

    def _snapshot_task_usage(self) -> _TaskUsageSnapshot:
        """Capture the cumulative usage counters used for task deltas."""
        summary: Dict[str, Any] = {}
        tracker = self._session_cost_tracker
        if hasattr(tracker, "get_summary"):
            try:
                summary = tracker.get_summary() or {}
            except Exception as exc:
                logger.debug("Failed to read session cost summary for task report: %s", exc)

        token_summary = summary.get("tokens", {}) if isinstance(summary, dict) else {}
        cost_summary = summary.get("cost", {}) if isinstance(summary, dict) else {}
        request_count = summary.get("request_count", None) if isinstance(summary, dict) else None

        tracker_requests = getattr(tracker, "requests", None)
        if request_count is None and isinstance(tracker_requests, list):
            request_count = len(tracker_requests)

        # Source token counts from the SessionCostTracker summary — the same authoritative
        # cumulative the cache/cost fields below already read, and the one `finalize_stream_metrics`
        # actually updates per turn via `record_request`. The legacy `_cumulative_token_usage`
        # dict has no writer in the service path (`update_cumulative_token_usage` is unused), so
        # reading it produced total_tokens=0 task reports despite real usage. Fall back to it only
        # if the tracker summary is unavailable.
        return _TaskUsageSnapshot(
            prompt_tokens=int(
                token_summary.get("prompt", self._cumulative_token_usage.get("prompt_tokens", 0))
                or 0
            ),
            completion_tokens=int(
                token_summary.get(
                    "completion",
                    self._cumulative_token_usage.get("completion_tokens", 0),
                )
                or 0
            ),
            total_tokens=int(
                token_summary.get("total", self._cumulative_token_usage.get("total_tokens", 0)) or 0
            ),
            cached_tokens=int(self._cumulative_token_usage.get("cached_tokens", 0) or 0),
            cache_read_tokens=int(
                token_summary.get(
                    "cache_read",
                    getattr(tracker, "total_cache_read_tokens", 0),
                )
                or 0
            ),
            cache_write_tokens=int(
                token_summary.get(
                    "cache_write",
                    getattr(tracker, "total_cache_write_tokens", 0),
                )
                or 0
            ),
            total_cost_usd=float(
                cost_summary.get("total", getattr(tracker, "total_cost", 0.0)) or 0.0
            ),
            request_count=int(request_count or 0),
        )

    # ========================================================================
    # Metrics Collector Delegation
    # ========================================================================

    @property
    def metrics_collector(self) -> "MetricsCollector":
        """Get the underlying metrics collector.

        Returns:
            The MetricsCollector instance
        """
        return self._metrics_collector

    @property
    def session_cost_tracker(self) -> "SessionCostTracker":
        """Get the underlying session cost tracker.

        Returns:
            The SessionCostTracker instance
        """
        return self._session_cost_tracker

    # Convenience methods for common metrics operations

    def record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        self._metrics_collector.record_tool_selection(method, num_tools)

    def record_tool_execution(self, tool_name: str, success: bool, elapsed_ms: float) -> None:
        """Record tool execution statistics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
        """
        self._metrics_collector.record_tool_execution(tool_name, success, elapsed_ms)

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], iteration: int = 0) -> None:
        """Callback when tool execution starts (from ToolPipeline).

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            iteration: Current iteration count
        """
        self._metrics_collector.on_tool_start(tool_name, arguments, iteration)

    def on_tool_complete(self, result: Any) -> None:
        """Callback when tool execution completes (from ToolPipeline).

        Args:
            result: The tool call result
        """
        self._metrics_collector.on_tool_complete(result)

    def on_streaming_session_complete(self, session: Any) -> None:
        """Callback when streaming session completes (from StreamingController).

        Args:
            session: The completed streaming session
        """
        self._metrics_collector.on_streaming_session_complete(session)

    def record_first_token(self) -> None:
        """Record the time of first token received."""
        self._metrics_collector.record_first_token()

    def init_stream_metrics(self) -> "StreamMetrics":
        """Initialize fresh stream metrics for a new streaming session.

        Returns:
            New StreamMetrics instance
        """
        return self._metrics_collector.init_stream_metrics()

    def update_model_info(self, model: str, provider: str) -> None:
        """Update model and provider info (e.g., after hot-swap).

        Args:
            model: New model name
            provider: New provider name
        """
        self._metrics_collector.update_model_info(model, provider)

    def get_optimization_status(
        self,
        context_compactor: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        code_correction_middleware: Optional[Any] = None,
        safety_checker: Optional[Any] = None,
        auto_committer: Optional[Any] = None,
        search_router: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Provides visibility into the health and statistics of all optimization
        components for debugging, monitoring, and observability.

        Args:
            context_compactor: Optional context compactor instance
            usage_analytics: Optional usage analytics instance
            sequence_tracker: Optional sequence tracker instance
            code_correction_middleware: Optional code correction middleware
            safety_checker: Optional safety checker instance
            auto_committer: Optional auto committer instance
            search_router: Optional search router instance

        Returns:
            Dictionary with component status and statistics
        """
        status: Dict[str, Any] = {
            "timestamp": time.time(),
            "components": {},
        }

        # Context Compactor
        if context_compactor:
            status["components"]["context_compactor"] = context_compactor.get_statistics()

        # Usage Analytics
        if usage_analytics:
            try:
                status["components"]["usage_analytics"] = {
                    "session_active": usage_analytics._current_session is not None,
                    "tool_records_count": len(usage_analytics._tool_records),
                    "provider_records_count": len(usage_analytics._provider_records),
                }
            except Exception:
                status["components"]["usage_analytics"] = {"status": "error"}

        # Sequence Tracker
        if sequence_tracker:
            try:
                status["components"]["sequence_tracker"] = sequence_tracker.get_statistics()
            except Exception:
                status["components"]["sequence_tracker"] = {"status": "error"}

        # Code Correction Middleware
        status["components"]["code_correction"] = {
            "enabled": code_correction_middleware is not None,
        }
        if code_correction_middleware:
            if hasattr(code_correction_middleware, "config"):
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": code_correction_middleware.config.auto_fix,
                    "max_iterations": code_correction_middleware.config.max_iterations,
                }
            else:
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": getattr(code_correction_middleware, "auto_fix", True),
                    "max_iterations": getattr(code_correction_middleware, "max_iterations", 1),
                }

        # Safety Checker
        status["components"]["safety_checker"] = {
            "enabled": safety_checker is not None,
            "has_confirmation_callback": (
                safety_checker.confirmation_callback is not None if safety_checker else False
            ),
        }

        # Auto Committer
        status["components"]["auto_committer"] = {
            "enabled": auto_committer is not None,
        }
        if auto_committer:
            status["components"]["auto_committer"]["auto_commit"] = auto_committer.auto_commit

        # Search Router
        status["components"]["search_router"] = {
            "enabled": search_router is not None,
        }

        # Overall health
        enabled_count = sum(
            1
            for c in status["components"].values()
            if c.get("enabled", True) and c.get("status") != "error"
        )
        status["health"] = {
            "enabled_components": enabled_count,
            "total_components": len(status["components"]),
            "status": "healthy" if enabled_count >= 4 else "degraded",
        }

        return status

    def reset_stats(self) -> None:
        """Reset all statistics (e.g., after conversation reset)."""
        self._metrics_collector.reset_stats()

    def record_tool_execution_full(
        self,
        tool_name: str,
        success: bool,
        elapsed_ms: float,
        error_type: Optional[str] = None,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        tool_selector: Optional[Any] = None,
        rl_coordinator: Optional[Any] = None,
        conversation_controller: Optional[Any] = None,
        provider_name: str = "unknown",
        model_name: str = "unknown",
        task_type: str = "general",
        vertical_name: Optional[str] = None,
    ) -> None:
        """Record tool execution across all analytics subsystems.

        Fans out to metrics collector, usage analytics, sequence tracker,
        tool selector, and RL coordinator.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
            error_type: Type of error if execution failed
            usage_analytics: Optional UsageAnalytics for data-driven optimization
            sequence_tracker: Optional ToolSequenceTracker for pattern learning
            tool_selector: Optional SemanticToolSelector for confidence boosting
            rl_coordinator: Optional RLCoordinator for Q-learning
            conversation_controller: Optional ConversationController for context metrics
            provider_name: Current provider name for RL context
            model_name: Current model name for RL context
            task_type: Current task type for RL context
            vertical_name: Current vertical name for RL context
        """
        # Core metrics recording
        self._metrics_collector.record_tool_execution(tool_name, success, elapsed_ms)

        # Record to UsageAnalytics for data-driven optimization
        if usage_analytics:
            context_tokens = 0
            if conversation_controller:
                context_metrics = conversation_controller.get_context_metrics()
                context_tokens = context_metrics.estimated_tokens
            usage_analytics.record_tool_execution(
                tool_name=tool_name,
                success=success,
                execution_time_ms=elapsed_ms,
                error_type=error_type,
                context_tokens=context_tokens,
            )

        # Record to ToolSequenceTracker for intelligent next-tool suggestions
        if sequence_tracker:
            sequence_tracker.record_execution(
                tool_name=tool_name,
                success=success,
                execution_time=elapsed_ms / 1000.0,  # Convert to seconds
            )

        # Record to SemanticToolSelector for confidence boosting
        # Enables 15-20% accuracy improvement via workflow pattern detection
        if tool_selector and hasattr(tool_selector, "record_tool_execution"):
            tool_selector.record_tool_execution(tool_name, success=success)

        # Record to RL tool_selector learner for Q-learning optimization
        if rl_coordinator:
            try:
                from victor.framework.rl.base import RLOutcome

                tool_outcome = RLOutcome(
                    success=success,
                    quality_score=1.0 if success else 0.0,
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                    metadata={
                        "tool_name": tool_name,
                        "execution_time_ms": elapsed_ms,
                        "error_type": error_type,
                    },
                )
                rl_coordinator.record_outcome("tool_selector", tool_outcome, vertical_name)
            except ImportError:
                logger.debug("RLOutcome not available, skipping RL recording")
            except KeyError as e:
                logger.debug(f"RL learner not registered: {e}")
            except ValueError as e:
                logger.warning(f"Invalid outcome data for RL recording: {e}")

    def send_rl_reward_signal(
        self,
        session: Any,
        rl_coordinator: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
    ) -> None:
        """Send reward signal to RL model selector for Q-value updates.

        Converts streaming session data into RLOutcome and updates Q-values
        based on session outcome (success, latency, throughput, tool usage).

        Args:
            session: StreamingSession with metrics/outcome data
            rl_coordinator: Optional RLCoordinator for recording outcomes
            vertical_context: Optional vertical context for vertical name
        """
        try:
            from victor.framework.rl.base import RLOutcome

            if not rl_coordinator:
                return

            # Extract metrics from session
            token_count = 0
            if session.metrics:
                token_count = session.metrics.total_chunks or 0

            # Get tool execution count from metrics collector
            tool_calls_made = 0
            if self._metrics_collector:
                tool_calls_made = self._metrics_collector._selection_stats.total_tools_executed

            # Determine success: no error and not cancelled
            success = session.error is None and not session.cancelled

            # Compute quality score (0-1) based on success and metrics
            quality_score = 0.5
            if success:
                quality_score = 0.8
                # Bonus for fast responses
                if session.duration < 10:
                    quality_score += 0.1
                # Bonus for tool usage
                if tool_calls_made > 0:
                    quality_score += 0.1
            quality_score = min(1.0, quality_score)

            # Create outcome
            outcome = RLOutcome(
                provider=session.provider,
                model=session.model,
                task_type=getattr(session, "task_type", "unknown"),
                success=success,
                quality_score=quality_score,
                metadata={
                    "latency_seconds": session.duration,
                    "token_count": token_count,
                    "tool_calls_made": tool_calls_made,
                    "session_id": session.session_id,
                },
                vertical=getattr(vertical_context, "vertical_name", None) or "default",
            )

            # Record outcome for model selector
            vertical_name = getattr(vertical_context, "vertical_name", None) or "default"
            rl_coordinator.record_outcome("model_selector", outcome, vertical_name)

            logger.debug(
                f"RL feedback: provider={session.provider} success={success} "
                f"quality={quality_score:.2f} duration={session.duration:.1f}s"
            )

        except ImportError:
            # RL module not available - skip silently
            pass
        except (KeyError, AttributeError) as e:
            # RL coordinator not properly initialized
            logger.debug(f"RL reward signal skipped (not configured): {e}")
        except (ValueError, TypeError) as e:
            # Invalid reward data
            logger.warning(f"Failed to send RL reward signal (invalid data): {e}")

    def flush_analytics(
        self,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        tool_cache: Optional[Any] = None,
    ) -> Dict[str, bool]:
        """Flush all analytics and cached data to persistent storage.

        Call this method before shutdown or when you need to ensure
        all analytics data is persisted to disk.

        Args:
            usage_analytics: Optional UsageAnalytics instance
            sequence_tracker: Optional ToolSequenceTracker instance
            tool_cache: Optional tool cache instance

        Returns:
            Dictionary indicating success/failure for each component
        """
        results: Dict[str, bool] = {}

        # Flush usage analytics
        if usage_analytics:
            try:
                usage_analytics.flush()
                results["usage_analytics"] = True
                logger.debug("UsageAnalytics flushed to disk")
            except Exception as e:
                logger.warning(f"Failed to flush usage analytics: {e}")
                results["usage_analytics"] = False
        else:
            results["usage_analytics"] = False

        # Flush sequence tracker patterns
        if sequence_tracker:
            try:
                stats = sequence_tracker.get_statistics()
                results["sequence_tracker"] = True
                logger.debug(
                    f"SequenceTracker has {stats.get('unique_transitions', 0)} learned patterns"
                )
            except Exception as e:
                logger.warning(f"Failed to get sequence tracker stats: {e}")
                results["sequence_tracker"] = False
        else:
            results["sequence_tracker"] = False

        # Flush tool cache
        if tool_cache:
            try:
                results["tool_cache"] = True
            except Exception as e:
                logger.warning(f"Failed to access tool cache: {e}")
                results["tool_cache"] = False
        else:
            results["tool_cache"] = False

        logger.info(f"Analytics flush complete: {results}")
        return results


def create_agent_metrics_service(
    metrics_collector: "MetricsCollector",
    session_cost_tracker: "SessionCostTracker",
    cumulative_token_usage: Optional[Dict[str, int]] = None,
) -> AgentMetricsService:
    """Factory function to create a AgentMetricsService.

    Args:
        metrics_collector: The metrics collector instance
        session_cost_tracker: The session cost tracker instance
        cumulative_token_usage: Optional token usage dict (creates new if None)

    Returns:
        Configured AgentMetricsService instance
    """
    if cumulative_token_usage is None:
        cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    return AgentMetricsService(
        metrics_collector=metrics_collector,
        session_cost_tracker=session_cost_tracker,
        cumulative_token_usage=cumulative_token_usage,
    )
