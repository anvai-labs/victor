"""Mirror tests for victor.agent.orchestrator_metrics.

Pins the metrics facade moved verbatim out of victor/agent/orchestrator.py
(facade decomposition): one-line AgentMetricsService delegations, the
preload trio's branches, tool-execution recording kwargs, and token/stage
bookkeeping.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def make_orch():
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    """Patch class-level descriptors (own or inherited) with value properties."""
    stack = ExitStack()
    for name, value in values.items():
        owner = next(b for b in type(orch).__mro__ if name in vars(b))
        stack.enter_context(patch.object(owner, name, property(lambda self, v=value: v)))
    return stack


def coord_orch(return_value=None):
    orch = make_orch()
    orch._metrics_coordinator = MagicMock()
    return orch


class TestStreamMetricsDelegation:
    def test_finalize(self):
        orch = coord_orch()
        AgentOrchestrator._finalize_stream_metrics(orch, {"input": 5})
        orch._metrics_coordinator.finalize_stream_metrics.assert_called_once_with({"input": 5})

    def test_last_and_summary_and_history(self):
        orch = coord_orch()
        AgentOrchestrator.get_last_stream_metrics(orch)
        orch._metrics_coordinator.get_last_stream_metrics.assert_called_once_with()
        AgentOrchestrator.get_streaming_metrics_summary(orch)
        orch._metrics_coordinator.get_streaming_metrics_summary.assert_called_once_with()
        AgentOrchestrator.get_streaming_metrics_history(orch, 3)
        orch._metrics_coordinator.get_streaming_metrics_history.assert_called_once_with(3)

    def test_costs_reports_and_export(self):
        orch = coord_orch()
        AgentOrchestrator.get_session_cost_summary(orch)
        AgentOrchestrator.get_session_cost_formatted(orch)
        AgentOrchestrator.get_last_task_report(orch)
        AgentOrchestrator.get_task_report_history(orch, 2)
        orch._metrics_coordinator.export_session_costs.assert_not_called()
        AgentOrchestrator.export_session_costs(orch, "/tmp/c.json", "csv")
        orch._metrics_coordinator.export_session_costs.assert_called_once_with("/tmp/c.json", "csv")


class TestPreloadEmbeddings:
    async def test_noop_without_selector(self):
        orch = make_orch()
        orch.semantic_selector = None
        await AgentOrchestrator._preload_embeddings(orch)  # must not raise

    async def test_noop_when_already_initialized(self):
        orch = make_orch()
        orch.semantic_selector = MagicMock()
        orch.tool_selector = SimpleNamespace_initialized()
        await AgentOrchestrator._preload_embeddings(orch)
        orch.semantic_selector.initialize_tool_embeddings.assert_not_called()

    async def test_success_marks_initialized(self):
        orch = make_orch()
        orch.semantic_selector = MagicMock()
        orch.semantic_selector.initialize_tool_embeddings = AsyncMock()
        orch.tools = MagicMock()
        orch.tool_selector = MagicMock()
        orch.tool_selector._embeddings_initialized = False
        orch._presentation = MagicMock()
        await AgentOrchestrator._preload_embeddings(orch)
        assert orch.tool_selector._embeddings_initialized is True

    async def test_failure_is_swallowed(self):
        # Negative: preload errors are logged, never raised.
        orch = make_orch()
        orch.semantic_selector = MagicMock()
        orch.semantic_selector.initialize_tool_embeddings = AsyncMock(
            side_effect=RuntimeError("no network")
        )
        orch.tools = MagicMock()
        orch.tool_selector = MagicMock()
        orch.tool_selector._embeddings_initialized = False
        await AgentOrchestrator._preload_embeddings(orch)  # must not raise
        assert orch.tool_selector._embeddings_initialized is False


def SimpleNamespace_initialized():
    ns = MagicMock()
    ns._embeddings_initialized = True
    return ns


class TestRuntimePreload:
    async def test_parallel_path(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch.use_semantic_selection = False
        manager = MagicMock()
        manager.list_tasks = MagicMock(return_value=["configuration"])
        stats = MagicMock(completed_tasks=1, total_tasks=1, duration=0.01)
        manager.preload_parallel = AsyncMock(return_value=stats)
        manager.preload_all = AsyncMock(return_value=stats)
        import victor.agent.preload as preload_mod

        with ExitStack() as stack:
            stack.enter_context(patch.object(preload_mod, "PreloadManager", return_value=manager))
            stack.enter_context(patch.object(preload_mod, "preload_configuration", lambda: None))
            await AgentOrchestrator._run_runtime_preload(orch)
        manager.preload_parallel.assert_called_once()
        manager.preload_all.assert_not_called()

    async def test_serial_path(self):
        orch = make_orch()
        orch.settings = MagicMock()
        # getattr(settings, 'framework_preload_parallel', True) -> False
        orch.settings.framework_preload_parallel = False
        orch.use_semantic_selection = False
        manager = MagicMock()
        manager.list_tasks = MagicMock(return_value=["configuration"])
        stats = MagicMock(completed_tasks=1, total_tasks=1, duration=0.01)
        manager.preload_parallel = AsyncMock(return_value=stats)
        manager.preload_all = AsyncMock(return_value=stats)
        import victor.agent.preload as preload_mod

        with ExitStack() as stack:
            stack.enter_context(patch.object(preload_mod, "PreloadManager", return_value=manager))
            stack.enter_context(patch.object(preload_mod, "preload_configuration", lambda: None))
            await AgentOrchestrator._run_runtime_preload(orch)
        manager.preload_all.assert_called_once()


class TestStartEmbeddingPreload:
    def _orch(self, **settings):
        orch = make_orch()
        orch.settings = MagicMock()
        for k, v in settings.items():
            setattr(orch.settings, k, v)
        orch.use_semantic_selection = True
        orch._runtime_preload_task = None
        orch._embedding_preload_task = None
        return orch

    def test_runtime_preload_started(self):
        orch = self._orch(framework_preload_enabled=True)
        orch._create_background_task = MagicMock(return_value="TASK")
        AgentOrchestrator.start_embedding_preload(orch)
        assert orch._runtime_preload_task == "TASK"

    def test_runtime_preload_not_restarted(self):
        orch = self._orch(framework_preload_enabled=True)
        orch._runtime_preload_task = "EXISTING"
        orch._create_background_task = MagicMock()
        AgentOrchestrator.start_embedding_preload(orch)
        orch._create_background_task.assert_not_called()

    def test_default_defers_embedding_preload(self):
        # Negative: without preload_embeddings=True nothing starts.
        orch = self._orch(framework_preload_enabled=False, preload_embeddings=False)
        orch._create_background_task = MagicMock()
        AgentOrchestrator.start_embedding_preload(orch)
        orch._create_background_task.assert_not_called()

    def test_explicit_preload_starts_task(self):
        orch = self._orch(framework_preload_enabled=False, preload_embeddings=True)
        orch._create_background_task = MagicMock(return_value="EMB")
        AgentOrchestrator.start_embedding_preload(orch)
        assert orch._embedding_preload_task == "EMB"


class TestRecordingAndStats:
    def test_record_tool_selection(self):
        orch = coord_orch()
        AgentOrchestrator._record_tool_selection(orch, "semantic", 5)
        orch._metrics_coordinator.record_tool_selection.assert_called_once_with("semantic", 5)

    def test_record_tool_execution_full_kwargs(self):
        orch = make_orch()
        orch._metrics_coordinator = MagicMock()
        orch._rl_coordinator = None
        with patched_state(orch, current_provider=SimpleNamespace(name="prov")):
            AgentOrchestrator._record_tool_execution(orch, "read", True, 12.5, error_type=None)
        _, kwargs = orch._metrics_coordinator.record_tool_execution_full.call_args
        assert kwargs["tool_name"] == "read"
        assert kwargs["success"] is True
        assert kwargs["elapsed_ms"] == 12.5
        assert kwargs["rl_coordinator"] is None
        assert kwargs["provider_name"] == "prov"
        assert kwargs["model_name"] == "unknown"

    def test_usage_stats_and_token_usage(self):
        orch = make_orch()
        orch._metrics_coordinator = MagicMock()
        orch.conversation_state = MagicMock()
        AgentOrchestrator.get_tool_usage_stats(orch)
        orch._metrics_coordinator.get_tool_usage_stats.assert_called_once()
        AgentOrchestrator.get_token_usage(orch)
        orch._metrics_coordinator.get_token_usage.assert_called_once_with()

    def test_reset_token_usage_zeroes_cumulative(self):
        orch = make_orch()
        with patched_state(orch, _cumulative_token_usage={"input": 5, "output": 2}):
            AgentOrchestrator.reset_token_usage(orch)
            assert orch._cumulative_token_usage == {"input": 0, "output": 0}

    def test_stage_helpers(self):
        orch = make_orch()
        orch.conversation_state = MagicMock()
        AgentOrchestrator.get_conversation_stage(orch)
        orch.conversation_state.get_stage.assert_called_once()
        AgentOrchestrator.get_stage_recommended_tools(orch)
        orch.conversation_state.get_stage_tools.assert_called_once()

    def test_optimization_status_and_flush(self):
        orch = make_orch()
        orch._metrics_coordinator = MagicMock()
        for attr in (
            "_context_compactor",
            "_usage_analytics",
            "_sequence_tracker",
            "_code_correction_middleware",
            "_safety_checker",
            "_auto_committer",
            "search_router",
            "tool_cache",
        ):
            setattr(orch, attr, MagicMock())
        AgentOrchestrator.get_optimization_status(orch)
        orch._metrics_coordinator.get_optimization_status.assert_called_once()
        AgentOrchestrator.flush_analytics(orch)
        orch._metrics_coordinator.flush_analytics.assert_called_once()


class TestSurfacePins:
    def test_methods_live_on_mixin(self):
        # Negative pin: moved methods must not reappear in the class body.
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "_finalize_stream_metrics",
            "_run_runtime_preload",
            "flush_analytics",
            "get_optimization_status",
        ):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_metrics.py"), (name, path)

    def test_background_task_and_search_routing_stay(self):
        # Deliberate stays: asyncio patch adjacency + non-metrics surface.
        class_src = inspect.getsource(AgentOrchestrator)
        assert "def _create_background_task(" in class_src
        assert "def route_search_query(" in class_src
