"""Metrics facade for the orchestrator.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition):
stream-metrics, session-cost, preload, token/stage bookkeeping — all thin
delegations to _metrics_coordinator (AgentMetricsService).
_create_background_task and search routing stay in the class body.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.stream_handler import StreamMetrics
from victor.evaluation.protocol import TokenUsage

logger = logging.getLogger(__name__)


class OrchestratorMetricsMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    def _finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.
        """
        return self._metrics_coordinator.finalize_stream_metrics(usage_data)

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_coordinator.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_coordinator.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_coordinator.get_streaming_metrics_history(limit)

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with session cost statistics
        """
        return self._metrics_coordinator.get_session_cost_summary()

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        return self._metrics_coordinator.get_session_cost_formatted()

    def get_last_task_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent canonical per-task metrics report."""
        return self._metrics_coordinator.get_last_task_report()

    def get_task_report_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent canonical per-task metrics reports."""
        return self._metrics_coordinator.get_task_report_history(limit)

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        self._metrics_coordinator.export_session_costs(path, format)

    async def _preload_embeddings(self) -> None:
        """Preload tool embeddings in background to avoid blocking first query.

        This is called asynchronously during initialization if semantic tool
        selection is enabled. Errors are logged but don't crash the app.

        Note: The ToolSelector owns the _embeddings_initialized state to avoid
        DRY violations and consistency issues.
        """
        if not self.semantic_selector:
            return
        # ToolSelector owns the initialization state
        if self.tool_selector._embeddings_initialized:
            return

        try:
            logger.debug("Starting background embedding preload...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            # Mark initialization complete in ToolSelector (single source of truth)
            self.tool_selector._embeddings_initialized = True
            logger.debug(
                f"{self._presentation.icon('success')} Tool embeddings preloaded successfully in background"
            )
        except Exception as e:
            logger.warning(
                f"Failed to preload embeddings in background: {e}. "
                "Will load on first query (may add ~5s latency)."
            )

    async def _run_runtime_preload(self) -> None:
        """Run feature-flagged runtime preload tasks via PreloadManager."""
        from victor.agent.preload import (
            PreloadManager,
            PreloadPriority,
            preload_configuration,
        )

        preload_manager = PreloadManager(
            enable_parallel=bool(getattr(self.settings, "framework_preload_parallel", True))
        )
        preload_manager.add_task(
            "configuration",
            preload_configuration,
            priority=PreloadPriority.CRITICAL.value,
            required=False,
        )

        if self.use_semantic_selection:
            preload_manager.add_task(
                "tool_embeddings",
                self._preload_embeddings,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if getattr(self.settings, "http_connection_pool_enabled", False):

            async def _preload_http_pool() -> None:
                from victor.tools.http_pool import ConnectionPoolConfig, get_http_pool

                await get_http_pool(
                    ConnectionPoolConfig(
                        max_connections=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections",
                                100,
                            )
                        ),
                        max_connections_per_host=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections_per_host",
                                10,
                            )
                        ),
                        connection_timeout=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_connection_timeout",
                                30,
                            )
                        ),
                        total_timeout=int(
                            getattr(self.settings, "http_connection_pool_total_timeout", 60)
                        ),
                    )
                )

            preload_manager.add_task(
                "http_pool",
                _preload_http_pool,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if not preload_manager.list_tasks():
            return

        if getattr(self.settings, "framework_preload_parallel", True):
            stats = await preload_manager.preload_parallel()
        else:
            stats = await preload_manager.preload_all()
        logger.debug(
            "Runtime preload completed: %s/%s tasks in %.2fs",
            stats.completed_tasks,
            stats.total_tasks,
            stats.duration,
        )

    def start_embedding_preload(self) -> None:
        """Start background embedding preload task.

        Should be called after orchestrator initialization to avoid blocking
        the main thread. Safe to call multiple times (no-op if already started).

        Embedding preload is deferred by default to avoid +694MB memory spike
        on first task. Set `preload_embeddings=True` in settings to eagerly load.
        The embedding model will still lazy-load on first semantic query via
        EmbeddingService._ensure_model_loaded().
        """
        if getattr(self.settings, "framework_preload_enabled", False):
            if self._runtime_preload_task is not None:
                return
            task = self._create_background_task(self._run_runtime_preload(), name="runtime_preload")
            if task:
                self._runtime_preload_task = task
                logger.debug("Started runtime preload task")
            return

        # Only preload embeddings if explicitly requested (default: deferred)
        if not getattr(self.settings, "preload_embeddings", False):
            return

        if not self.use_semantic_selection or self._embedding_preload_task:
            return

        task = self._create_background_task(self._preload_embeddings(), name="embedding_preload")
        if task:
            self._embedding_preload_task = task
            logger.debug("Started background task for embedding preload")

    def _record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        self._metrics_coordinator.record_tool_selection(method, num_tools)

    def _record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        elapsed_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Record tool execution statistics. Delegates to AgentMetricsService."""
        self._metrics_coordinator.record_tool_execution_full(
            tool_name=tool_name,
            success=success,
            elapsed_ms=elapsed_ms,
            error_type=error_type,
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_selector=getattr(self, "tool_selector", None),
            rl_coordinator=self._rl_coordinator,
            conversation_controller=getattr(self, "_conversation_controller", None),
            provider_name=getattr(self.current_provider, "name", "unknown"),
            model_name=getattr(self, "_current_model", "unknown"),
            task_type=getattr(self, "_task_type", "general"),
            vertical_name=getattr(self, "_vertical_name", None),
        )

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return self._metrics_coordinator.get_tool_usage_stats(
            conversation_state_summary=self.conversation_state.get_state_summary()
        )

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        return self._metrics_coordinator.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Call this at the start of a new evaluation task to get fresh counts.
        """
        # Reset through coordinator (which updates the cumulative dict)
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    def get_conversation_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        return self.conversation_state.get_stage()

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for the current conversation stage.

        Returns:
            Set of tool names recommended for current stage
        """
        return self.conversation_state.get_stage_tools()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Delegates to AgentMetricsService which owns the reporting logic.
        """
        return self._metrics_coordinator.get_optimization_status(
            context_compactor=self._context_compactor,
            usage_analytics=self._usage_analytics,
            sequence_tracker=self._sequence_tracker,
            code_correction_middleware=self._code_correction_middleware,
            safety_checker=self._safety_checker,
            auto_committer=self._auto_committer,
            search_router=self.search_router,
        )

    def flush_analytics(self) -> Dict[str, bool]:
        """Flush all analytics and cached data. Delegates to AgentMetricsService."""
        return self._metrics_coordinator.flush_analytics(
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_cache=getattr(self, "tool_cache", None),
        )
