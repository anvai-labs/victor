"""Tests for ToolPipeline, ToolCallResult, ToolPipelineConfig."""

from unittest.mock import AsyncMock, MagicMock, patch
import logging

import pytest

from victor.agent.tool_call_tracker import ToolCallTracker
from victor.agent.tool_pipeline import (
    ToolCallResult,
    ToolPipeline,
    ToolPipelineConfig,
    PipelineExecutionResult,
    LRUToolCache,
)
from victor.agent.tool_executor import ToolExecutionResult
from victor.tools.enums import AccessMode


@pytest.fixture
def log_capture():
    """Fixture to capture log messages."""
    log_messages = []

    class LogHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(self.format(record))

    handler = LogHandler()
    logger = logging.getLogger("victor.agent.tool_pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    yield log_messages

    logger.removeHandler(handler)


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.is_tool_enabled.return_value = True
    registry.get.return_value.access_mode = AccessMode.READONLY
    return registry


@pytest.fixture
def mock_tool_executor():
    from victor.agent.tool_executor import ToolExecutionResult

    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read",
            success=True,
            result="file contents",
            execution_time=0.01,
        )
    )
    return executor


@pytest.fixture
def pipeline(mock_tool_registry, mock_tool_executor):
    config = ToolPipelineConfig(
        tool_budget=5,
        enable_caching=False,
        enable_output_aggregation=False,
        enable_synthesis_checkpoints=False,
        enable_semantic_caching=False,
    )
    return ToolPipeline(
        tool_registry=mock_tool_registry,
        tool_executor=mock_tool_executor,
        config=config,
    )


class TestToolPipelineConfig:
    def test_default_budget(self):
        config = ToolPipelineConfig()
        assert config.tool_budget > 0

    def test_custom_budget(self):
        config = ToolPipelineConfig(tool_budget=10)
        assert config.tool_budget == 10


class TestToolPipelineInit:
    def test_initial_state(self, pipeline):
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 5
        assert pipeline.executed_tools == []

    def test_is_valid_tool_name(self, pipeline):
        assert pipeline.is_valid_tool_name("read") is True
        assert pipeline.is_valid_tool_name("read_file") is True
        assert pipeline.is_valid_tool_name("") is False
        assert pipeline.is_valid_tool_name("Read") is False  # must be lowercase
        assert pipeline.is_valid_tool_name("123abc") is False

    def test_reset(self, pipeline):
        pipeline._calls_used = 3
        pipeline._executed_tools.append("read")
        pipeline.reset()
        assert pipeline.calls_used == 0
        assert pipeline.executed_tools == []


class TestExecuteToolCalls:
    @pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
    async def test_uncertain_write_survives_middleware_without_fallback(self, pipeline, failure):
        from victor.core.verticals.protocols import MiddlewareResult

        pipeline.tools.get.return_value.access_mode = AccessMode.WRITE
        committed = []

        async def submit(**kwargs):
            committed.append("receipt")
            raise failure("File not found: original\nDid you mean:\n - replacement")

        pipeline.executor.execute.side_effect = submit
        chain = MagicMock()
        chain.process_before = AsyncMock(return_value=MiddlewareResult(proceed=True))
        chain.process_after = AsyncMock(return_value="transformed output")
        pipeline.middleware_chain = chain
        # Invoke the common execution path directly; before-policy behavior has its own owner.
        result = await pipeline._execute_single_call({"name": "submit", "arguments": {}}, {})

        assert committed == ["receipt"]
        assert result.success is False
        assert result.retryable is False
        assert result.error_info.details["execution_outcome"] == "unknown"
        assert result.error_info.details["reconciliation_required"] is True
        assert result.result == "transformed output"

    @pytest.mark.parametrize("boundary", ["inner", "outer", "cancel"])
    async def test_real_execution_boundary_does_not_replay_committed_write(self, boundary):
        import asyncio
        from types import SimpleNamespace
        from victor.agent.services.tool_retry import ToolRetryExecutor
        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.decorators import tool
        from victor.tools.registry import ToolRegistry

        committed = []
        started = asyncio.Event()

        @tool(access_mode=AccessMode.WRITE)
        async def submit_record(_exec_ctx=None):
            committed.append("receipt")
            started.set()
            if boundary == "inner":
                raise TimeoutError("response lost")
            await asyncio.Event().wait()

        registry = ToolRegistry()
        registry.register(submit_record)
        executor = ToolExecutor(tool_registry=registry, retry_delay=0)
        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                enable_caching=False,
                enable_semantic_caching=False,
                per_tool_timeout_seconds=0.02 if boundary == "outer" else 60,
            ),
        )
        retry = ToolRetryExecutor(
            SimpleNamespace(
                retry_enabled=True,
                max_retry_attempts=3,
                retry_base_delay=0,
                retry_max_delay=0,
            ),
            pipeline,
        )
        task = asyncio.create_task(retry.execute_tool_with_retry("submit_record", {}, {}))
        await asyncio.wait_for(started.wait(), 2)
        if boundary == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result, success, _ = await task
            assert success is False and result.retryable is False
            assert result.error_info.details["execution_outcome"] == "unknown"
        assert committed == ["receipt"]

    @pytest.mark.parametrize(
        "failure",
        [
            "selection",
            "scope",
            "hook",
            "malformed",
            "chain",
            "decision",
            "arguments",
            "outer_decision",
        ],
    )
    async def test_middleware_failure_never_dispatches(self, pipeline, failure):
        from victor.agent.middleware_chain import MiddlewareChain
        from victor.core.verticals.protocols import MiddlewarePriority, MiddlewareResult

        chain = MiddlewareChain()
        middleware = MagicMock()
        middleware.get_priority.return_value = MiddlewarePriority.NORMAL
        middleware.get_applicable_tools.return_value = None
        middleware.before_tool_call = AsyncMock(side_effect=RuntimeError("private details"))
        if failure == "selection":
            middleware.get_applicable_tools.side_effect = AttributeError("private details")
        elif failure == "scope":
            middleware.get_applicable_tools.return_value = "read"
        elif failure == "malformed":
            middleware.before_tool_call = AsyncMock(return_value=None)
        elif failure == "chain":
            chain.process_before = AsyncMock(side_effect=TypeError("private details"))
        elif failure in {"decision", "outer_decision"}:
            method = AsyncMock(return_value=MiddlewareResult(proceed="false"))
            if failure == "outer_decision":
                chain.process_before = method
            else:
                middleware.before_tool_call = method
        elif failure == "arguments":
            middleware.before_tool_call = AsyncMock(
                return_value=MiddlewareResult(modified_arguments=[])
            )
        chain.add(middleware)
        pipeline.middleware_chain = chain

        outcome = await pipeline.execute_tool_calls([{"name": "write", "arguments": {}}])

        pipeline.executor.execute.assert_not_called()
        result = outcome.results[0]
        assert not result.success and result.skipped
        assert result.block_source == "middleware_chain"
        assert result.retryable is False
        assert "private details" not in (result.skip_reason or "")

    async def test_empty_middleware_arguments_replace_original(self, pipeline):
        from victor.agent.middleware_chain import MiddlewareChain
        from victor.core.verticals.protocols import MiddlewarePriority, MiddlewareResult

        middleware = MagicMock()
        middleware.get_priority.return_value = MiddlewarePriority.NORMAL
        middleware.get_applicable_tools.return_value = None
        middleware.before_tool_call = AsyncMock(
            return_value=MiddlewareResult(modified_arguments={})
        )
        middleware.after_tool_call = AsyncMock(return_value=None)
        chain = MiddlewareChain()
        chain.add(middleware)
        pipeline.middleware_chain = chain
        await pipeline.execute_tool_calls([{"name": "write", "arguments": {"content": "old"}}])
        assert pipeline.executor.execute.call_args.kwargs["arguments"] == {}

    async def test_single_successful_call(self, pipeline):
        tool_calls = [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert isinstance(result, PipelineExecutionResult)
        assert result.total_calls == 1
        assert result.budget_exhausted is False

    async def test_invalid_tool_call_structure(self, pipeline):
        tool_calls = ["not_a_dict"]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].success is False
        assert result.results[0].skipped is True

    async def test_missing_tool_name(self, pipeline):
        tool_calls = [{"arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True
        assert "missing name" in result.results[0].skip_reason.lower()

    async def test_budget_enforcement(self, pipeline):
        pipeline.config.tool_budget = 2
        pipeline._calls_used = 2
        tool_calls = [{"name": "read", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True
        assert "budget" in result.results[0].skip_reason.lower()
        assert result.results[0].outcome_kind == "budget_exhausted"
        assert result.results[0].block_source == "tool_budget"
        assert result.results[0].retryable is False

    async def test_unknown_tool_skipped(self, pipeline):
        pipeline.tools.is_tool_enabled.return_value = False
        tool_calls = [{"name": "nonexistent", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True

    async def test_camel_case_provider_tool_name_normalized_before_unknown_check(self, pipeline):
        pipeline.tools.is_tool_enabled.return_value = False

        result = await pipeline.execute_tool_calls(
            [{"name": "setGlobalAxisManager", "arguments": {}}]
        )

        call_result = result.results[0]
        assert call_result.skipped is True
        assert call_result.tool_name == "set_global_axis_manager"
        assert call_result.outcome_kind == "tool_unavailable"
        assert "set_global_axis_manager" in call_result.skip_reason

    async def test_camel_case_provider_tool_name_executes_as_normalized_name(
        self, pipeline, mock_tool_executor
    ):
        await pipeline.execute_tool_calls(
            [{"name": "readFile", "arguments": {"path": "/tmp/f.py"}}]
        )

        mock_tool_executor.execute.assert_awaited()
        assert mock_tool_executor.execute.await_args.kwargs["tool_name"] == "read_file"

    async def test_callbacks_invoked(self, pipeline):
        on_start = MagicMock()
        on_complete = MagicMock()
        pipeline.on_tool_start = on_start
        pipeline.on_tool_complete = on_complete
        tool_calls = [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
        await pipeline.execute_tool_calls(tool_calls)
        on_start.assert_called()
        on_complete.assert_called()

    async def test_legacy_read_file_uses_exact_dedup_key(self, pipeline):
        await pipeline.execute_tool_calls(
            [{"name": "read_file", "arguments": {"path": "/tmp/f.py"}}]
        )
        assert "/tmp/f.py:None:None" in pipeline._read_file_timestamps

    async def test_write_file_invalidates_read_dedup_key(self, pipeline, mock_tool_executor):
        await pipeline.execute_tool_calls(
            [{"name": "read_file", "arguments": {"path": "/tmp/f.py"}}]
        )
        assert "/tmp/f.py:None:None" in pipeline._read_file_timestamps

        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="write",
            success=True,
            result="updated",
            execution_time=0.01,
        )
        await pipeline.execute_tool_calls(
            [{"name": "write_file", "arguments": {"path": "/tmp/f.py", "content": "x"}}]
        )

        assert "/tmp/f.py:None:None" not in pipeline._read_file_timestamps

    async def test_write_file_marks_code_search_index_stale(self, pipeline, mock_tool_executor):
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="write",
            success=True,
            result="updated",
            execution_time=0.01,
        )

        # Patch load_code_search_module at the source to return a mock with mark_index_cache_stale_for_path
        mock_module = MagicMock()
        mock_module.mark_index_cache_stale_for_path = MagicMock(return_value=2)
        with patch(
            "victor.core.utils.capability_loader.load_code_search_module",
            return_value=mock_module,
        ):
            await pipeline.execute_tool_calls(
                [
                    {
                        "name": "write_file",
                        "arguments": {"path": "/tmp/f.py", "content": "x"},
                    }
                ]
            )

        mock_module.mark_index_cache_stale_for_path.assert_called_once()

    async def test_duplicate_read_sets_structured_skip_metadata(self, pipeline):
        with patch.object(pipeline, "_is_duplicate_read", return_value=True):
            result = await pipeline.execute_tool_calls(
                [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
            )

        call_result = result.results[0]
        assert call_result.skipped is True
        assert call_result.success is True
        assert call_result.outcome_kind == "duplicate_read"
        assert call_result.block_source == "session_read_dedup"
        assert call_result.retryable is True
        assert "already read with the same offset/limit" in call_result.user_message

    async def test_failed_read_suggestion_redirects_next_identical_path(
        self,
        pipeline,
        mock_tool_executor,
    ):
        error = (
            "File not found: src/compute/distance/mod.rs\n"
            "Did you mean one of these?\n"
            "  - src/compute/distance_computation/mod.rs"
        )
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult("read", success=False, result=None, error=error),
            ToolExecutionResult("read", success=False, result=None, error=error),
            ToolExecutionResult("read", success=True, result="correct file"),
        ]

        first = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "src/compute/distance/mod.rs"}}]
        )
        second = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "src/compute/distance/mod.rs"}}]
        )

        assert first.results[0].success is False
        assert second.results[0].success is True
        assert second.results[0].arguments["path"] == "src/compute/distance_computation/mod.rs"
        assert (
            mock_tool_executor.execute.await_args_list[-1].kwargs["arguments"]["path"]
            == "src/compute/distance_computation/mod.rs"
        )

    async def test_successful_recovery_result_does_not_record_failed_signature(
        self,
        pipeline,
        mock_tool_executor,
    ):
        error = (
            "File not found: src/compute/distance/mod.rs\n"
            "Did you mean one of these?\n"
            "  - src/compute/distance_computation/mod.rs"
        )
        bad_args = {"path": "src/compute/distance/mod.rs"}
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult("read", success=False, result=None, error=error),
            ToolExecutionResult("read", success=True, result="correct file"),
        ]

        result = await pipeline.execute_tool_calls([{"name": "read", "arguments": bad_args}])

        assert result.results[0].success is True
        assert pipeline.is_known_failure("read", bad_args) is False
        assert result.results[0].arguments["path"] == "src/compute/distance_computation/mod.rs"

    async def test_failed_read_self_suggestion_does_not_redirect(
        self,
        pipeline,
        mock_tool_executor,
    ):
        error = "File not found: Cargo.toml\n" "Did you mean one of these?\n" "  - Cargo.toml"
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            "read",
            success=False,
            result=None,
            error=error,
        )

        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "Cargo.toml"}}])

        assert pipeline._failed_path_redirects == {}

    async def test_log_message_shows_actual_skip_reasons(self, pipeline, log_capture):
        """Test that log messages show actual skip reasons, not hardcoded text."""
        # Force budget exhaustion
        pipeline.config.tool_budget = 1
        pipeline._calls_used = 1

        # Execute a tool call that will be skipped due to budget
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "/tmp/f.py"}}])

        # Check that the log message contains the actual skip reason
        log_text = " ".join(log_capture)
        assert "Tool budget exhausted" in log_text or "budget" in log_text.lower()

    async def test_log_message_multiple_skip_reasons(self, pipeline, log_capture):
        """Test that log messages show multiple different skip reasons."""
        # First call exhausts budget
        pipeline.config.tool_budget = 1
        pipeline._calls_used = 1

        # Execute multiple tool calls that will be skipped
        await pipeline.execute_tool_calls(
            [
                {"name": "read", "arguments": {"path": "/tmp/f1.py"}},
                {"name": "ls", "arguments": {"path": "/tmp"}},
            ]
        )

        # Check that the log message contains skip reason information
        log_text = " ".join(log_capture)
        # Should mention tools were skipped
        assert "skipped" in log_text.lower()

    async def test_write_clears_search_dedup_history_for_verification(self, mock_tool_registry):
        executor = MagicMock()
        executor.execute = AsyncMock(
            side_effect=[
                ToolExecutionResult(
                    tool_name="code_search",
                    success=True,
                    result="before",
                    execution_time=0.01,
                ),
                ToolExecutionResult(
                    tool_name="write",
                    success=True,
                    result="updated",
                    execution_time=0.01,
                ),
                ToolExecutionResult(
                    tool_name="code_search",
                    success=True,
                    result="after",
                    execution_time=0.01,
                ),
            ]
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                enable_idempotent_caching=False,
                enable_output_aggregation=False,
                enable_synthesis_checkpoints=False,
                enable_semantic_caching=False,
            ),
            deduplication_tracker=ToolCallTracker(),
        )

        search_call = [{"name": "code_search", "arguments": {"query": "node_ids"}}]
        write_call = [{"name": "write", "arguments": {"path": "/tmp/f.py", "content": "x"}}]

        first = await pipeline.execute_tool_calls(search_call, {})
        assert first.results[0].skipped is False

        # Patch load_code_search_module at the source to return a mock with mark_index_cache_stale_for_path
        mock_module = MagicMock()
        mock_module.mark_index_cache_stale_for_path = MagicMock(return_value=1)
        with patch(
            "victor.core.utils.capability_loader.load_code_search_module",
            return_value=mock_module,
        ):
            await pipeline.execute_tool_calls(write_call, {})

        second = await pipeline.execute_tool_calls(search_call, {})
        assert second.results[0].skipped is False
        assert second.results[0].result == "after"
        assert executor.execute.await_count == 3


class TestCodeIntelligenceFirstNavigation:
    async def test_build_mode_routes_broad_code_read_to_diagnostics(
        self,
        pipeline,
        mock_tool_executor,
    ):
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="lsp",
            success=True,
            result={"diagnostics": [{"file_path": "/tmp/service.py", "line": 88}]},
            execution_time=0.01,
        )

        result = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "/tmp/service.py"}}],
            {"mode": "build"},
        )

        call_result = result.results[0]
        assert call_result.tool_name == "lsp"
        assert call_result.arguments == {
            "action": "diagnostics",
            "file_path": "/tmp/service.py",
        }
        assert "Steered broad read(path=...)" in call_result.user_message
        mock_tool_executor.execute.assert_awaited_once_with(
            tool_name="lsp",
            arguments={"action": "diagnostics", "file_path": "/tmp/service.py"},
            context={"mode": "build"},
        )

    async def test_build_mode_follow_up_broad_read_uses_diagnostic_range(
        self,
        pipeline,
        mock_tool_executor,
    ):
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="lsp",
                success=True,
                result={"diagnostics": [{"file_path": "/tmp/service.py", "line": 88}]},
                execution_time=0.01,
            ),
            ToolExecutionResult(
                tool_name="read",
                success=True,
                result="targeted contents",
                execution_time=0.01,
            ),
        ]

        await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "/tmp/service.py"}}],
            {"mode": "build"},
        )
        result = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "/tmp/service.py"}}],
            {"mode": "build"},
        )

        call_result = result.results[0]
        assert call_result.tool_name == "read"
        assert call_result.arguments == {
            "path": "/tmp/service.py",
            "offset": 48,
            "limit": 120,
        }
        assert "line 88" in call_result.user_message
        mock_tool_executor.execute.assert_awaited_with(
            tool_name="read",
            arguments={"path": "/tmp/service.py", "offset": 48, "limit": 120},
            context={"mode": "build"},
        )


class TestLRUToolCache:
    def test_set_and_get(self):
        cache = LRUToolCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_eviction_on_max_size(self):
        cache = LRUToolCache(max_size=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("c") == 3

    def test_clear(self):
        cache = LRUToolCache(max_size=10, ttl_seconds=60)
        cache.set("k", "v")
        cache.clear()
        assert len(cache) == 0


class TestNavigationHintLoopDetection:
    """Test navigation hint loop detection and expiration mechanisms."""

    async def test_explicit_offset_limit_supersedes_navigation_hints(
        self, pipeline, mock_tool_executor
    ):
        """When agent provides explicit offset/limit, navigation hints should be ignored."""
        import time
        from victor.agent.file_state import normalize_file_path

        now = time.monotonic()
        # Set up a navigation hint for line 100 using normalized path
        test_file = "/tmp/test.py"
        normalized_path = normalize_file_path(test_file)
        pipeline._recent_code_navigation_hints[normalized_path] = {
            "source_tool": "refs",
            "file_path": test_file,
            "line": 100,
            "use_count": 0,
            "created_at": now,
            "last_result_hash": None,
        }

        # Call with explicit offset/limit that differs from hint
        result = pipeline._select_follow_up_code_read_rewrite(
            test_file,
            offset=500,
            limit=200,
            context={},
        )

        # Should return None (don't rewrite) when explicit params provided
        assert result is None

    async def test_navigation_hint_expires_after_max_uses(self, pipeline, mock_tool_executor):
        """Navigation hints should expire after max_uses is reached."""
        import time
        from victor.agent.file_state import normalize_file_path

        now = time.monotonic()
        test_file = "/tmp/test.py"
        normalized_path = normalize_file_path(test_file)
        # Set up a navigation hint with use_count at max
        pipeline._recent_code_navigation_hints[normalized_path] = {
            "source_tool": "refs",
            "file_path": test_file,
            "line": 100,
            "use_count": 3,  # At max_uses
            "created_at": now,
            "last_result_hash": None,
        }

        # Call should clear the hint
        result = pipeline._select_follow_up_code_read_rewrite(
            test_file,
            offset=None,
            limit=None,
            context={},
        )

        # Hint should be cleared and return None
        assert result is None
        assert normalized_path not in pipeline._recent_code_navigation_hints

    async def test_navigation_hint_increments_use_count(self, pipeline, mock_tool_executor):
        """Using a navigation hint should increment its use_count."""
        import time
        from victor.agent.file_state import normalize_file_path

        now = time.monotonic()
        test_file = "/tmp/test.py"
        normalized_path = normalize_file_path(test_file)
        # Set up a navigation hint with use_count=0
        pipeline._recent_code_navigation_hints[normalized_path] = {
            "source_tool": "refs",
            "file_path": test_file,
            "line": 100,
            "use_count": 0,
            "created_at": now,
            "last_result_hash": None,
        }

        # Call should return rewrite and increment use_count
        result = pipeline._select_follow_up_code_read_rewrite(
            test_file,
            offset=None,
            limit=None,
            context={},
        )

        assert result is not None
        assert result[0] == "read"
        assert result[1]["offset"] == 60  # max(100 - 40, 0)
        assert result[1]["limit"] == 120
        assert pipeline._recent_code_navigation_hints[normalized_path]["use_count"] == 1

    async def test_no_hint_returns_none(self, pipeline, mock_tool_executor):
        """When no navigation hint exists, should return None."""
        result = pipeline._select_follow_up_code_read_rewrite(
            "/tmp/nonexistent.py",
            offset=None,
            limit=None,
            context={},
        )

        assert result is None

    async def test_none_line_number_skips_rewrite(self, pipeline, mock_tool_executor):
        """Navigation hints with None line number should not trigger rewrite."""
        import time
        from victor.agent.file_state import normalize_file_path

        now = time.monotonic()
        test_file = "/tmp/test.py"
        normalized_path = normalize_file_path(test_file)
        pipeline._recent_code_navigation_hints[normalized_path] = {
            "source_tool": "project_overview",
            "file_path": test_file,
            "line": None,
            "use_count": 0,
            "created_at": now,
            "last_result_hash": None,
        }

        result = pipeline._select_follow_up_code_read_rewrite(
            test_file,
            offset=None,
            limit=None,
            context={},
        )

        assert result is None

    async def test_different_files_maintain_separate_hints(self, pipeline, mock_tool_executor):
        """Navigation hints should be tracked independently per file."""
        import time
        from victor.agent.file_state import normalize_file_path

        now = time.monotonic()
        file1 = "/tmp/file1.py"
        file2 = "/tmp/file2.py"
        normalized_path1 = normalize_file_path(file1)
        normalized_path2 = normalize_file_path(file2)
        pipeline._recent_code_navigation_hints[normalized_path1] = {
            "source_tool": "refs",
            "file_path": file1,
            "line": 100,
            "use_count": 0,
            "created_at": now,
            "last_result_hash": None,
        }
        pipeline._recent_code_navigation_hints[normalized_path2] = {
            "source_tool": "symbol",
            "file_path": file2,
            "line": 200,
            "use_count": 1,
            "created_at": now,
            "last_result_hash": None,
        }

        result1 = pipeline._select_follow_up_code_read_rewrite(
            file1,
            offset=None,
            limit=None,
            context={},
        )
        result2 = pipeline._select_follow_up_code_read_rewrite(
            file2,
            offset=None,
            limit=None,
            context={},
        )

        assert result1 is not None
        assert result1[1]["offset"] == 60  # 100 - 40
        assert result2 is not None
        assert result2[1]["offset"] == 160  # 200 - 40
        assert pipeline._recent_code_navigation_hints[normalized_path1]["use_count"] == 1
        assert pipeline._recent_code_navigation_hints[normalized_path2]["use_count"] == 2

    async def test_navigation_hint_expires_after_ttl(self, pipeline, mock_tool_executor):
        """Navigation hints should expire after TTL seconds."""
        import time
        from victor.agent.file_state import normalize_file_path

        test_file = "/tmp/test.py"
        normalized_path = normalize_file_path(test_file)
        # Set up an expired hint (older than TTL)
        expired_time = time.monotonic() - 700  # Older than 600s TTL
        pipeline._recent_code_navigation_hints[normalized_path] = {
            "source_tool": "refs",
            "file_path": test_file,
            "line": 100,
            "use_count": 0,
            "created_at": expired_time,
            "last_result_hash": None,
        }

        result = pipeline._select_follow_up_code_read_rewrite(
            test_file,
            offset=None,
            limit=None,
            context={},
        )

        # Expired hint should be cleared
        assert result is None
        assert normalized_path not in pipeline._recent_code_navigation_hints
