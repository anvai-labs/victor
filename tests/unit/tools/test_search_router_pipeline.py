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

"""Tests for ToolPipeline."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.search_router import SearchRouter
from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ToolCallResult,
)
from victor.agent.tool_executor import ToolExecutionResult


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock()
    registry.is_tool_enabled = MagicMock(return_value=True)
    return registry


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result={"output": "test result"},
            error=None,
        )
    )
    return executor


@pytest.fixture
def pipeline(mock_tool_registry, mock_tool_executor):
    """Create a tool pipeline for testing."""
    return ToolPipeline(
        tool_registry=mock_tool_registry,
        tool_executor=mock_tool_executor,
        config=ToolPipelineConfig(tool_budget=10),
    )


class TestToolPipelineConfig:
    """Tests for ToolPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolPipelineConfig()
        assert config.tool_budget == 25
        assert config.enable_caching is True
        assert config.enable_analytics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ToolPipelineConfig(tool_budget=50, enable_caching=False)
        assert config.tool_budget == 50
        assert config.enable_caching is False


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""

    def test_success_result(self):
        """Test successful tool call result."""
        result = ToolCallResult(
            tool_name="read_file",
            arguments={"path": "test.py"},
            success=True,
            result="file contents",
        )
        assert result.success is True
        assert result.skipped is False

    def test_skipped_result(self):
        """Test skipped tool call result."""
        result = ToolCallResult(
            tool_name="unknown_tool",
            arguments={},
            success=False,
            skipped=True,
            skip_reason="Unknown tool",
        )
        assert result.skipped is True
        assert result.skip_reason == "Unknown tool"


class TestToolPipeline:
    """Tests for ToolPipeline class."""

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 10

    def test_is_valid_tool_name(self, pipeline):
        """Test tool name validation."""
        # Valid names
        assert pipeline.is_valid_tool_name("read_file") is True
        assert pipeline.is_valid_tool_name("code_search") is True
        assert pipeline.is_valid_tool_name("git") is True

        # Invalid names
        assert pipeline.is_valid_tool_name("") is False
        assert pipeline.is_valid_tool_name(None) is False
        assert pipeline.is_valid_tool_name("Invalid-Name") is False
        assert pipeline.is_valid_tool_name("123_tool") is False
        assert pipeline.is_valid_tool_name("Tool") is False  # Must start lowercase

    def test_is_valid_tool_name_length_limit(self, pipeline):
        """Test tool name length limit."""
        long_name = "a" * 100
        assert pipeline.is_valid_tool_name(long_name) is False

    @pytest.mark.asyncio
    async def test_execute_single_tool_call(self, pipeline, mock_tool_executor):
        """Test executing a single tool call."""
        tool_calls = [{"name": "read_file", "arguments": {"path": "test.py"}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.total_calls == 1
        assert result.successful_calls == 1
        assert result.failed_calls == 0
        assert pipeline.calls_used == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(self, pipeline, mock_tool_executor):
        """Test executing multiple tool calls."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "test.py"}},
            {"name": "code_search", "arguments": {"query": "test"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.total_calls == 2
        assert result.successful_calls == 2
        assert pipeline.calls_used == 2

    @pytest.mark.asyncio
    async def test_skip_invalid_tool_name(self, pipeline):
        """Test that invalid tool names are skipped."""
        tool_calls = [{"name": "Invalid-Tool", "arguments": {}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert result.results[0].skipped is True
        assert "Invalid tool name" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_skip_unknown_tool(self, pipeline, mock_tool_registry):
        """Test that unknown tools are skipped."""
        mock_tool_registry.is_tool_enabled.return_value = False
        tool_calls = [{"name": "unknown_tool", "arguments": {}}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "Unknown or disabled" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, pipeline, mock_tool_executor):
        """Test that tool budget is enforced."""
        pipeline.config.tool_budget = 2
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool3", "arguments": {}},  # Should be skipped
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.budget_exhausted is True
        assert pipeline.calls_used == 2
        # All 3 calls produce results: 2 executed + 1 skip result (budget_exhausted)
        assert len(result.results) == 3
        assert result.results[2].skipped is True
        assert result.results[2].outcome_kind == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_skip_repeated_failures(self, pipeline, mock_tool_executor):
        """Test that repeated failing calls are skipped.

        Note: With batch deduplication enabled (default), identical tool calls
        in the same batch are deduplicated first, so the skip reason will be
        'Deduplicated' rather than 'Repeated failing'. The 'Repeated failing'
        behavior still applies to subsequent batches/iterations.
        """
        # Make the tool fail
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="failing_tool",
            success=False,
            result=None,
            error="Test error",
        )

        tool_calls = [
            {"name": "failing_tool", "arguments": {"x": 1}},
            {"name": "failing_tool", "arguments": {"x": 1}},  # Same signature
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        # First call fails, second is skipped (deduplicated in same batch)
        assert result.failed_calls == 1
        assert result.skipped_calls == 1
        # With batch deduplication enabled, duplicates in same batch are deduped first
        assert "Deduplicated" in result.results[1].skip_reason

    @pytest.mark.asyncio
    async def test_file_not_found_recovery_retries_with_suggested_path(
        self, pipeline, mock_tool_executor
    ):
        """Read failures with concrete suggestions should retry transparently."""
        mock_tool_executor.execute = AsyncMock(
            side_effect=[
                ToolExecutionResult(
                    tool_name="read",
                    success=False,
                    result=None,
                    error=(
                        "File not found: victor/core/registry.py\n"
                        "Did you mean one of these?\n"
                        "  - victor/core/registry/base.py\n"
                        "  - victor/core/registry/__init__.py"
                    ),
                ),
                ToolExecutionResult(
                    tool_name="read",
                    success=True,
                    result="[File: victor/core/registry/base.py]",
                    error=None,
                ),
            ]
        )

        tool_calls = [
            {
                "name": "read",
                "arguments": {
                    "path": "victor/core/registry.py",
                    "limit": 100,
                    "offset": 0,
                },
            }
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.total_calls == 1
        assert result.successful_calls == 1
        assert result.failed_calls == 0
        assert result.results[0].success is True
        assert result.results[0].arguments["path"] == "victor/core/registry/base.py"
        assert result.results[0].result == "[File: victor/core/registry/base.py]"
        assert mock_tool_executor.execute.await_count == 2
        retry_call = mock_tool_executor.execute.await_args_list[1]
        assert retry_call.kwargs["tool_name"] == "read"
        assert retry_call.kwargs["arguments"]["path"] == "victor/core/registry/base.py"

    @pytest.mark.asyncio
    async def test_argument_normalization(self, pipeline, mock_tool_executor):
        """Test that string arguments are normalized."""
        tool_calls = [{"name": "test_tool", "arguments": '{"path": "test.py"}'}]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        # Should succeed with normalized arguments
        assert result.successful_calls == 1
        call_args = mock_tool_executor.execute.call_args
        assert isinstance(call_args.kwargs["arguments"], dict)

    @pytest.mark.asyncio
    async def test_duplicate_read_skip_result_includes_next_step_guidance(self, pipeline):
        """Duplicate read skips should tell the model how to recover productively."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/tools/graph_tool.py"}}]

        with patch.object(pipeline, "_is_duplicate_read", return_value=True):
            result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        skipped = result.results[0]
        assert skipped.skipped is True
        assert skipped.skip_reason == "Duplicate file read within session"
        assert "Do not repeat the same read(path, offset, limit)" in skipped.result
        assert "symbol" in skipped.result
        assert "refs" in skipped.result
        assert "lsp" in skipped.result
        assert "code_search" in skipped.result
        assert "project_overview" in skipped.result
        assert pipeline.last_batch_all_skipped is True
        assert pipeline.last_batch_effectively_blocked is False

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_broad_code_reads_to_lsp_diagnostics(
        self, pipeline, mock_tool_executor
    ):
        """Plan-like modes should rewrite broad reads to a structure-aware tool when possible."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="lsp",
            success=True,
            result={"diagnostics": [{"message": "warning"}]},
            error=None,
        )

        result = await pipeline.execute_tool_calls(tool_calls, {"mode": "plan"})

        assert result.successful_calls == 1
        assert result.skipped_calls == 0
        rewritten = result.results[0]
        assert rewritten.tool_name == "lsp"
        assert rewritten.arguments == {
            "action": "diagnostics",
            "file_path": "victor/agent/tool_pipeline.py",
        }
        assert rewritten.user_message is not None
        assert "Steered broad read(path=...)" in rewritten.user_message
        mock_tool_executor.execute.assert_called_once()
        call_kwargs = mock_tool_executor.execute.call_args.kwargs
        assert call_kwargs["tool_name"] == "lsp"
        assert call_kwargs["arguments"] == {
            "action": "diagnostics",
            "file_path": "victor/agent/tool_pipeline.py",
        }
        assert pipeline.last_batch_all_skipped is False
        assert pipeline.last_batch_effectively_blocked is False

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_broad_code_reads_to_project_overview_when_lsp_unavailable(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Broad read steering should fall back to project overview when diagnostics are unavailable."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "project_overview",
            "symbol",
        }
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="project_overview",
            success=True,
            result={"entries": []},
            error=None,
        )

        result = await pipeline.execute_tool_calls(tool_calls, {"mode": "plan"})

        assert result.successful_calls == 1
        assert result.skipped_calls == 0
        rewritten = result.results[0]
        assert rewritten.tool_name == "project_overview"
        assert rewritten.arguments == {
            "path": "victor/agent",
            "max_depth": 2,
        }
        call_kwargs = mock_tool_executor.execute.call_args.kwargs
        assert call_kwargs["tool_name"] == "project_overview"
        assert call_kwargs["arguments"] == {
            "path": "victor/agent",
            "max_depth": 2,
        }

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_broad_code_reads_to_symbol_when_context_has_target_symbol(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Known symbol intent should prefer direct symbol navigation over generic diagnostics."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "lsp",
            "project_overview",
            "symbol",
        }
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="symbol",
            success=True,
            result={"symbol_name": "ToolPipeline"},
            error=None,
        )

        result = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )

        assert result.successful_calls == 1
        rewritten = result.results[0]
        assert rewritten.tool_name == "symbol"
        assert rewritten.arguments == {
            "file_path": "victor/agent/tool_pipeline.py",
            "symbol_name": "ToolPipeline",
        }
        call_kwargs = mock_tool_executor.execute.call_args.kwargs
        assert call_kwargs["tool_name"] == "symbol"
        assert call_kwargs["arguments"]["symbol_name"] == "ToolPipeline"

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_broad_code_reads_to_refs_when_symbol_tool_unavailable(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Known symbol intent should prefer refs over generic diagnostics when symbol lookup is unavailable."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "lsp",
            "project_overview",
            "refs",
        }
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="refs",
            success=True,
            result={"references": ["victor/agent/tool_pipeline.py:1627"]},
            error=None,
        )

        result = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )

        assert result.successful_calls == 1
        rewritten = result.results[0]
        assert rewritten.tool_name == "refs"
        assert rewritten.arguments == {
            "symbol_name": "ToolPipeline",
            "search_path": "victor/agent",
        }
        call_kwargs = mock_tool_executor.execute.call_args.kwargs
        assert call_kwargs["tool_name"] == "refs"
        assert call_kwargs["arguments"] == {
            "symbol_name": "ToolPipeline",
            "search_path": "victor/agent",
        }

    @pytest.mark.asyncio
    async def test_plan_mode_falls_back_from_empty_symbol_lookup_to_refs(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Low-signal symbol rewrites should keep narrowing toward references."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "symbol",
            "refs",
            "project_overview",
        }
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="symbol",
                success=True,
                result={"matches": []},
                error=None,
            ),
            ToolExecutionResult(
                tool_name="refs",
                success=True,
                result={"references": ["victor/agent/tool_pipeline.py:1627"]},
                error=None,
            ),
        ]

        result = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )

        assert result.successful_calls == 1
        rewritten = result.results[0]
        assert rewritten.tool_name == "refs"
        assert rewritten.arguments == {
            "symbol_name": "ToolPipeline",
            "search_path": "victor/agent",
        }
        assert "Symbol lookup was empty" in (rewritten.user_message or "")
        assert mock_tool_executor.execute.call_count == 2
        first_call = mock_tool_executor.execute.call_args_list[0].kwargs
        second_call = mock_tool_executor.execute.call_args_list[1].kwargs
        assert first_call["tool_name"] == "symbol"
        assert second_call["tool_name"] == "refs"

    @pytest.mark.asyncio
    async def test_plan_mode_falls_back_from_empty_lsp_diagnostics_to_project_overview(
        self, pipeline, mock_tool_executor
    ):
        """Low-signal diagnostic rewrites should keep narrowing automatically."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="lsp",
                success=True,
                result={"diagnostics": []},
                error=None,
            ),
            ToolExecutionResult(
                tool_name="project_overview",
                success=True,
                result={"entries": [{"path": "victor/agent/tool_pipeline.py"}]},
                error=None,
            ),
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {"mode": "plan"})

        assert result.successful_calls == 1
        rewritten = result.results[0]
        assert rewritten.tool_name == "project_overview"
        assert rewritten.arguments == {
            "path": "victor/agent",
            "max_depth": 2,
        }
        assert "LSP diagnostics were empty" in (rewritten.user_message or "")
        assert mock_tool_executor.execute.call_count == 2
        first_call = mock_tool_executor.execute.call_args_list[0].kwargs
        second_call = mock_tool_executor.execute.call_args_list[1].kwargs
        assert first_call["tool_name"] == "lsp"
        assert second_call["tool_name"] == "project_overview"

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_follow_up_broad_read_to_targeted_window_after_refs(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Concrete refs hits should narrow the next broad read to a local window."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "refs",
            "project_overview",
        }
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="refs",
                success=True,
                result={"references": ["victor/agent/tool_pipeline.py:1627"]},
                error=None,
            ),
            ToolExecutionResult(
                tool_name="read",
                success=True,
                result="[File: victor/agent/tool_pipeline.py (offset=1587, limit=120)]",
                error=None,
            ),
        ]

        first = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )
        second = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )

        assert first.successful_calls == 1
        assert first.results[0].tool_name == "refs"
        assert second.successful_calls == 1
        rewritten = second.results[0]
        assert rewritten.tool_name == "read"
        assert rewritten.arguments == {
            "path": "victor/agent/tool_pipeline.py",
            "offset": 1587,
            "limit": 120,
        }
        assert "line 1627" in (rewritten.user_message or "")
        assert mock_tool_executor.execute.call_count == 2
        first_call = mock_tool_executor.execute.call_args_list[0].kwargs
        second_call = mock_tool_executor.execute.call_args_list[1].kwargs
        assert first_call["tool_name"] == "refs"
        assert second_call["tool_name"] == "read"
        assert second_call["arguments"]["offset"] == 1587
        assert second_call["arguments"]["limit"] == 120

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_follow_up_broad_read_to_lsp_range_window(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Standard LSP ranges should become targeted read windows on later broad reads."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "lsp",
        }
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="lsp",
                success=True,
                result={
                    "diagnostics": [
                        {
                            "message": "Possible undefined name",
                            "range": {
                                "start": {"line": 211, "character": 8},
                                "end": {"line": 211, "character": 23},
                            },
                        }
                    ]
                },
                error=None,
            ),
            ToolExecutionResult(
                tool_name="read",
                success=True,
                result="[File: victor/agent/tool_pipeline.py (offset=172, limit=120)]",
                error=None,
            ),
        ]

        first = await pipeline.execute_tool_calls(tool_calls, {"mode": "review"})
        second = await pipeline.execute_tool_calls(tool_calls, {"mode": "review"})

        assert first.successful_calls == 1
        assert first.results[0].tool_name == "lsp"
        assert second.successful_calls == 1
        rewritten = second.results[0]
        assert rewritten.tool_name == "read"
        assert rewritten.arguments == {
            "path": "victor/agent/tool_pipeline.py",
            "offset": 172,
            "limit": 120,
        }
        assert "line 212" in (rewritten.user_message or "")
        assert mock_tool_executor.execute.call_count == 2
        second_call = mock_tool_executor.execute.call_args_list[1].kwargs
        assert second_call["tool_name"] == "read"
        assert second_call["arguments"]["offset"] == 172
        assert second_call["arguments"]["limit"] == 120

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_follow_up_broad_read_to_symbol_range_window(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Symbol location ranges should become targeted read windows on later broad reads."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "symbol",
        }
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="symbol",
                success=True,
                result={
                    "locations": [
                        {
                            "file_path": "victor/agent/tool_pipeline.py",
                            "range": {
                                "start": {"line": 702, "character": 0},
                                "end": {"line": 830, "character": 0},
                            },
                        }
                    ]
                },
                error=None,
            ),
            ToolExecutionResult(
                tool_name="read",
                success=True,
                result="[File: victor/agent/tool_pipeline.py (offset=663, limit=120)]",
                error=None,
            ),
        ]

        first = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "build", "target_symbol": "ToolPipeline"},
        )
        second = await pipeline.execute_tool_calls(
            tool_calls,
            {"mode": "build", "target_symbol": "ToolPipeline"},
        )

        assert first.successful_calls == 1
        assert first.results[0].tool_name == "symbol"
        assert second.successful_calls == 1
        rewritten = second.results[0]
        assert rewritten.tool_name == "read"
        assert rewritten.arguments == {
            "path": "victor/agent/tool_pipeline.py",
            "offset": 663,
            "limit": 120,
        }
        assert "line 703" in (rewritten.user_message or "")

    @pytest.mark.asyncio
    async def test_plan_mode_rewrites_follow_up_broad_read_to_symbol_after_project_overview(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """Recent overview-only navigation should keep narrowing toward symbol lookup."""
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "project_overview",
            "symbol",
        }
        mock_tool_executor.execute.side_effect = [
            ToolExecutionResult(
                tool_name="project_overview",
                success=True,
                result={"entries": [{"path": "victor/agent/tool_pipeline.py"}]},
                error=None,
            ),
            ToolExecutionResult(
                tool_name="symbol",
                success=True,
                result={"symbol_name": "ToolPipeline"},
                error=None,
            ),
        ]

        first = await pipeline.execute_tool_calls(
            [
                {
                    "name": "project_overview",
                    "arguments": {"path": "victor/agent", "max_depth": 2},
                }
            ],
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )
        second = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}],
            {"mode": "plan", "target_symbol": "ToolPipeline"},
        )

        assert first.successful_calls == 1
        assert first.results[0].tool_name == "project_overview"
        assert second.successful_calls == 1
        rewritten = second.results[0]
        assert rewritten.tool_name == "symbol"
        assert rewritten.arguments == {
            "file_path": "victor/agent/tool_pipeline.py",
            "symbol_name": "ToolPipeline",
        }
        assert "recent project_overview" in (rewritten.user_message or "")
        assert mock_tool_executor.execute.call_count == 2
        first_call = mock_tool_executor.execute.call_args_list[0].kwargs
        second_call = mock_tool_executor.execute.call_args_list[1].kwargs
        assert first_call["tool_name"] == "project_overview"
        assert second_call["tool_name"] == "symbol"

    @pytest.mark.asyncio
    async def test_plan_mode_returns_recovery_skip_when_no_auto_rewrite_tool_is_available(
        self, pipeline, mock_tool_registry, mock_tool_executor
    ):
        """If no safe rewrite target is enabled, the pipeline should keep the actionable skip."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_registry.is_tool_enabled.side_effect = lambda name: name in {
            "read",
            "symbol",
        }

        result = await pipeline.execute_tool_calls(tool_calls, {"mode": "plan"})

        assert result.skipped_calls == 1
        skipped = result.results[0]
        assert skipped.skipped is True
        assert skipped.skip_reason == "Broad code-file read before structure-aware navigation"
        assert skipped.result["success"] is False
        assert "Do not start with a broad read(path=...)" in skipped.result["error"]
        suggestions = skipped.result["metadata"]["follow_up_suggestions"]
        assert [item["tool"] for item in suggestions] == ["symbol"]
        assert suggestions[0]["arguments"]["file_path"] == "victor/agent/tool_pipeline.py"
        assert mock_tool_executor.execute.call_count == 0
        assert pipeline.last_batch_all_skipped is True
        assert pipeline.last_batch_effectively_blocked is False

    @pytest.mark.asyncio
    async def test_build_mode_rewrites_initial_broad_code_reads(self, pipeline, mock_tool_executor):
        """Build mode should use code intelligence before initial broad code reads."""
        tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/tool_pipeline.py"}}]
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="lsp",
            success=True,
            result={"diagnostics": [{"message": "warning"}]},
            error=None,
        )

        result = await pipeline.execute_tool_calls(tool_calls, {"mode": "build"})

        assert result.successful_calls == 1
        assert result.skipped_calls == 0
        assert mock_tool_executor.execute.call_count == 1
        rewritten = result.results[0]
        assert rewritten.tool_name == "lsp"
        assert rewritten.arguments == {
            "action": "diagnostics",
            "file_path": "victor/agent/tool_pipeline.py",
        }

    @pytest.mark.asyncio
    async def test_redundant_code_search_skip_includes_recovery_payload(self, mock_tool_registry):
        """Repeated code_search calls should return actionable follow-up suggestions."""
        from victor.agent.tool_call_tracker import (
            ToolCallTracker as ToolDeduplicationTracker,
        )

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="code_search",
                success=True,
                result={"matches": []},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_executor,
            config=ToolPipelineConfig(enable_idempotent_caching=False),
            deduplication_tracker=ToolDeduplicationTracker(),
        )
        tool_calls = [
            {
                "name": "code_search",
                "arguments": {
                    "query": "graph_indexer codebase_graph build_symbol_graph",
                    "path": "victor/core",
                    "mode": "semantic",
                },
            }
        ]

        first = await pipeline.execute_tool_calls(tool_calls, {})
        second = await pipeline.execute_tool_calls(tool_calls, {})

        assert first.results[0].skipped is False
        assert second.skipped_calls == 1
        skipped = second.results[0]
        assert skipped.skipped is True
        assert skipped.success is True
        assert skipped.skip_reason == "Redundant call (semantic overlap with recent operations)"
        assert skipped.result["success"] is False
        assert "Do not repeat the same search" in skipped.result["error"]
        suggestions = skipped.result["metadata"]["follow_up_suggestions"]
        assert [item["tool"] for item in suggestions] == ["overview", "graph"]
        assert suggestions[1]["arguments"]["mode"] == "search"
        assert mock_executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_redundant_refs_skip_includes_symbol_specific_recovery(self, mock_tool_registry):
        """Repeated refs calls should steer the model toward graph or text follow-up."""
        from victor.agent.tool_call_tracker import (
            ToolCallTracker as ToolDeduplicationTracker,
        )

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="refs",
                success=True,
                result=[],
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_executor,
            config=ToolPipelineConfig(enable_idempotent_caching=False),
            deduplication_tracker=ToolDeduplicationTracker(),
        )
        tool_calls = [
            {
                "name": "refs",
                "arguments": {"symbol_name": "BaseProvider", "search_path": "victor"},
            }
        ]

        first = await pipeline.execute_tool_calls(tool_calls, {})
        second = await pipeline.execute_tool_calls(tool_calls, {})

        assert first.results[0].skipped is False
        assert second.skipped_calls == 1
        skipped = second.results[0]
        assert skipped.skipped is True
        assert skipped.result["success"] is False
        assert "Do not repeat the same refs() scan" in skipped.result["error"]
        suggestions = skipped.result["metadata"]["follow_up_suggestions"]
        assert [item["tool"] for item in suggestions] == ["graph", "code_search"]
        assert suggestions[0]["arguments"]["mode"] == "callers|callees"
        assert suggestions[1]["arguments"]["query"] == "BaseProvider"
        assert mock_executor.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_repeated_failing_read_skip_includes_recovery_payload(self, mock_tool_registry):
        """Repeated failing calls should expose a real recovery payload instead of a blank skip."""
        mock_executor = MagicMock()
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_executor,
            config=ToolPipelineConfig(enable_idempotent_caching=False),
        )
        tool_args = {"path": "victor/core/container.py", "offset": 0, "limit": 100}
        signature = pipeline._get_call_signature("read", tool_args)
        pipeline._failed_signatures.add(signature)

        result = await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": tool_args}],
            {},
        )

        assert result.skipped_calls == 1
        skipped = result.results[0]
        assert skipped.skipped is True
        assert skipped.skip_reason == "Repeated failing call with same arguments"
        assert skipped.error is not None
        assert "already failed earlier in this session" in skipped.error
        assert skipped.result["success"] is False
        suggestions = skipped.result["metadata"]["follow_up_suggestions"]
        assert suggestions
        assert suggestions[0]["tool"] in {"overview", "graph"}
        assert mock_executor.execute.call_count == 0
        assert pipeline.last_batch_all_skipped is True
        assert pipeline.last_batch_effectively_blocked is False

    @pytest.mark.asyncio
    async def test_unknown_tool_skip_still_counts_as_effectively_blocked(
        self, pipeline, mock_tool_registry
    ):
        """Non-informative skipped batches should continue to count as blocked."""
        mock_tool_registry.is_tool_enabled.return_value = False

        result = await pipeline.execute_tool_calls(
            [{"name": "unknown_tool", "arguments": {"x": 1}}],
            {},
        )

        assert result.skipped_calls == 1
        assert result.results[0].result is None
        assert pipeline.last_batch_all_skipped is True
        assert pipeline.last_batch_effectively_blocked is True

    def test_reset(self, pipeline):
        """Test resetting pipeline state."""
        pipeline._calls_used = 5
        pipeline._executed_tools = ["tool1", "tool2"]
        pipeline._failed_signatures.add(("test", "{}"))

        pipeline.reset()

        assert pipeline.calls_used == 0
        assert len(pipeline.executed_tools) == 0
        assert len(pipeline._failed_signatures) == 0

    def test_get_analytics(self, pipeline):
        """Test getting analytics."""
        analytics = pipeline.get_analytics()

        assert "total_calls" in analytics
        assert "budget" in analytics
        assert "remaining" in analytics
        assert "tools" in analytics

    @pytest.mark.asyncio
    async def test_callbacks(self, pipeline, mock_tool_executor):
        """Test that callbacks are invoked."""
        start_called = []
        complete_called = []

        def on_start(name, args):
            start_called.append((name, args))

        def on_complete(result):
            complete_called.append(result)

        pipeline.on_tool_start = on_start
        pipeline.on_tool_complete = on_complete

        tool_calls = [{"name": "test_tool", "arguments": {"x": 1}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        assert len(start_called) == 1
        assert len(complete_called) == 1
        assert start_called[0][0] == "test_tool"

    @pytest.mark.asyncio
    async def test_skip_missing_name(self, pipeline):
        """Test skipping tool call without name."""
        tool_calls = [{"arguments": {"x": 1}}]  # Missing name

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "missing name" in result.results[0].skip_reason

    @pytest.mark.asyncio
    async def test_skip_invalid_structure(self, pipeline):
        """Test skipping invalid tool call structure."""
        tool_calls = ["not a dict"]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.skipped_calls == 1
        assert "not a dict" in result.results[0].skip_reason

    def test_executed_tools_tracking(self, pipeline):
        """Test that executed tools are tracked."""
        assert pipeline.executed_tools == []

    def test_clear_failed_signatures(self, pipeline):
        """Test clearing failed signatures."""
        pipeline._failed_signatures.add(("test", "{}"))
        assert len(pipeline._failed_signatures) == 1

        pipeline.clear_failed_signatures()
        assert len(pipeline._failed_signatures) == 0


class TestToolPipelineParallelExecution:
    """Co-design review item 16: allowlisted, opt-in parallel tool dispatch.

    Uses real read-only-allowlisted tool names ("read", "grep", "ls") so
    _compute_parallel_eligible_tools() actually admits them, and a write
    tool ("write") which is always excluded via get_write_tool_names().
    """

    @pytest.fixture
    def concurrency_tracker(self):
        """Tracks (name, start, end) for every executed call plus the max
        number of calls simultaneously in flight — the ground truth for
        proving concurrency actually happened (or didn't)."""

        class Tracker:
            def __init__(self):
                self.spans = []
                self.in_flight = 0
                self.max_in_flight = 0
                self.start_order = []

        return Tracker()

    def _make_executor(self, tracker, delay: float = 0.05):
        """A tool executor whose calls overlap in wall-clock time when
        dispatched concurrently, and record that overlap for assertions."""
        executor = MagicMock()

        async def _execute(tool_name, arguments, context=None):
            tracker.start_order.append(tool_name)
            tracker.in_flight += 1
            tracker.max_in_flight = max(tracker.max_in_flight, tracker.in_flight)
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(delay)
            end = asyncio.get_event_loop().time()
            tracker.in_flight -= 1
            tracker.spans.append((tool_name, start, end))
            return ToolExecutionResult(tool_name=tool_name, success=True, result=f"{tool_name}-ok")

        executor.execute = AsyncMock(side_effect=_execute)
        return executor

    @pytest.fixture
    def registry(self):
        registry = MagicMock()
        registry.is_tool_enabled = MagicMock(return_value=True)
        registry.get_tool = MagicMock(return_value=None)
        return registry

    def _make_pipeline(self, registry, executor, *, budget=25, max_concurrent=5):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        config = ToolPipelineConfig(
            tool_budget=budget,
            parallel_tool_execution=True,
            max_concurrent_tools=max_concurrent,
        )
        return ToolPipeline(tool_registry=registry, tool_executor=executor, config=config)

    @pytest.mark.asyncio
    async def test_default_off_in_fresh_settings(self):
        """The flag ships OFF: fresh Settings/AgentConfig must not enable it."""
        from victor.config.settings import Settings
        from victor.agent.config import UnifiedAgentConfig
        from victor.framework.config import AgentConfig

        assert Settings().parallel_tool_execution is False
        assert AgentConfig().enable_parallel_tools is False
        assert UnifiedAgentConfig().enable_parallel_tools is False

    @pytest.mark.asyncio
    async def test_default_config_is_sequential(self, registry):
        """ToolPipelineConfig() with no override stays off."""
        from victor.agent.tool_pipeline import ToolPipelineConfig

        assert ToolPipelineConfig().parallel_tool_execution is False

    @pytest.mark.asyncio
    async def test_read_only_batch_runs_concurrently(self, registry, concurrency_tracker):
        """Allowlisted read-only calls in one batch actually overlap in
        wall-clock time — the positive case this feature exists for."""
        executor = self._make_executor(concurrency_tracker, delay=0.05)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "grep", "arguments": {"path": "/b"}},
            {"id": "c3", "name": "ls", "arguments": {"path": "/c"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert concurrency_tracker.max_in_flight >= 2
        assert result.parallel_execution_used is True
        assert [r.success for r in result.results] == [True, True, True]
        # Original order preserved regardless of dispatch/completion order.
        assert [r.tool_name for r in result.results] == ["read", "grep", "ls"]

    @pytest.mark.asyncio
    async def test_write_tool_never_gathered_concurrently(self, registry, concurrency_tracker):
        """A write-capable tool must never run inside a concurrent gather —
        even alongside read-only calls in the same batch."""
        executor = self._make_executor(concurrency_tracker, delay=0.05)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "write", "arguments": {"path": "/b"}},
            {"id": "c3", "name": "grep", "arguments": {"path": "/c"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        write_span = next(s for s in concurrency_tracker.spans if s[0] == "write")
        for name, start, end in concurrency_tracker.spans:
            if name == "write":
                continue
            # No other call's interval overlaps the write call's interval.
            overlaps = start < write_span[2] and end > write_span[1]
            assert not overlaps, f"{name} overlapped with write: {(start, end)} vs {write_span}"
        assert [r.tool_name for r in result.results] == ["read", "write", "grep"]
        assert all(r.success for r in result.results)

    @pytest.mark.asyncio
    async def test_unknown_tool_runs_sequential(self, registry, concurrency_tracker):
        """Soundness negative: an unrecognized tool name must NOT be treated
        as parallel-eligible just because it isn't in the write-tools set —
        the allowlist is an intersection, not the authorizer's inverse."""
        executor = self._make_executor(concurrency_tracker, delay=0.05)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "totally_unclassified_tool", "arguments": {}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        unknown_span = next(
            s for s in concurrency_tracker.spans if s[0] == "totally_unclassified_tool"
        )
        read_span = next(s for s in concurrency_tracker.spans if s[0] == "read")
        overlaps = read_span[1] < unknown_span[2] and read_span[2] > unknown_span[1]
        assert not overlaps

    @pytest.mark.asyncio
    async def test_shell_never_parallel_eligible_even_if_idempotent_like(
        self, registry, concurrency_tracker
    ):
        """shell is hard-excluded regardless of any read-only-looking
        classification (co-design review item 16's explicit safety floor)."""
        executor = self._make_executor(concurrency_tracker, delay=0.05)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "shell", "arguments": {"cmd": "ls"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        shell_span = next(s for s in concurrency_tracker.spans if s[0] == "shell")
        read_span = next(s for s in concurrency_tracker.spans if s[0] == "read")
        overlaps = read_span[1] < shell_span[2] and read_span[2] > shell_span[1]
        assert not overlaps

    @pytest.mark.asyncio
    async def test_duplicate_signatures_collapse_before_dispatch(
        self, registry, concurrency_tracker
    ):
        """Two calls with the identical (tool, args) signature are
        deduplicated upstream of dispatch — the real executor only runs
        once regardless of parallel_tool_execution."""
        executor = self._make_executor(concurrency_tracker, delay=0.01)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "read", "arguments": {"path": "/a"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert len(concurrency_tracker.spans) == 1
        assert len(result.results) == 2
        assert sum(1 for r in result.results if r.cached) == 1

    @pytest.mark.asyncio
    async def test_mixed_batch_preserves_original_order(self, registry, concurrency_tracker):
        """Reads run in parallel, writes run sequentially, but the final
        result list always matches the input's original order."""
        executor = self._make_executor(concurrency_tracker, delay=0.02)
        pipeline = self._make_pipeline(registry, executor)

        tool_calls = [
            {"id": "c1", "name": "write", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "read", "arguments": {"path": "/b"}},
            {"id": "c3", "name": "grep", "arguments": {"path": "/c"}},
            {"id": "c4", "name": "write", "arguments": {"path": "/d"}},
            {"id": "c5", "name": "ls", "arguments": {"path": "/e"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert [r.tool_name for r in result.results] == [
            "write",
            "read",
            "grep",
            "write",
            "ls",
        ]
        assert [r.tool_call_id for r in result.results] == ["c1", "c2", "c3", "c4", "c5"]

    @pytest.mark.asyncio
    async def test_budget_exhaustion_computed_before_dispatch(self, registry, concurrency_tracker):
        """Budget is a hard cap on the whole batch even under concurrency:
        exactly `tool_budget` calls execute, the rest are skipped, and the
        skip decision doesn't overshoot from a concurrent-check race."""
        executor = self._make_executor(concurrency_tracker, delay=0.03)
        pipeline = self._make_pipeline(registry, executor, budget=2)

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "grep", "arguments": {"path": "/b"}},
            {"id": "c3", "name": "ls", "arguments": {"path": "/c"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.budget_exhausted is True
        assert len(concurrency_tracker.spans) == 2
        skipped = [r for r in result.results if r.skipped]
        assert len(skipped) == 1
        assert skipped[0].outcome_kind == "budget_exhausted"
        assert [r.tool_name for r in result.results] == ["read", "grep", "ls"]

    @pytest.mark.asyncio
    async def test_max_concurrent_tools_bounds_in_flight_count(self, registry, concurrency_tracker):
        """The semaphore actually caps concurrency at max_concurrent_tools,
        not just "more than one"."""
        executor = self._make_executor(concurrency_tracker, delay=0.05)
        pipeline = self._make_pipeline(registry, executor, budget=25, max_concurrent=2)

        tool_calls = [
            {"id": f"c{i}", "name": "read", "arguments": {"path": f"/{i}"}} for i in range(6)
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert concurrency_tracker.max_in_flight <= 2
        assert all(r.success for r in result.results)

    @pytest.mark.asyncio
    async def test_first_call_always_attempts_even_when_budget_already_exhausted(
        self, registry, concurrency_tracker
    ):
        """Regression guard: when the session's tool_budget was already
        exhausted by a PRIOR turn (not by this batch), the batch's FIRST
        call must still be dispatched and hit _execute_single_call's own
        internal budget check — not this method's pre-dispatch tail-skip.
        The two produce different, caller-visible results (retryable=False
        + no explicit user_message from the internal check, vs.
        retryable=True + an explicit message from the tail-skip), and only
        the internal-check outcome matches the pre-parallelism behavior for
        a batch's first call. A second call in the SAME already-exhausted
        batch, by contrast, must get the pre-dispatch tail-skip, since by
        then admitted > 0."""
        executor = self._make_executor(concurrency_tracker, delay=0.01)
        pipeline = self._make_pipeline(registry, executor, budget=2)
        pipeline._calls_used = 2  # already exhausted before this batch starts

        tool_calls = [
            {"id": "c1", "name": "read", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "grep", "arguments": {"path": "/b"}},
        ]

        result = await pipeline.execute_tool_calls(tool_calls, {})

        first, second = result.results
        assert first.skipped is True
        assert first.outcome_kind == "budget_exhausted"
        assert first.retryable is False
        assert first.skip_reason == "Tool budget exhausted"

        assert second.skipped is True
        assert second.outcome_kind == "budget_exhausted"
        assert second.retryable is True
        assert second.skip_reason == "Tool budget exhausted — call not executed"
        assert second.user_message == (
            "The tool budget was exhausted before this call could be executed."
        )

        # Neither call actually ran the tool.
        assert len(concurrency_tracker.spans) == 0


class TestLRUToolCacheConcurrency:
    """Co-design review item 16: LRUToolCache under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_get_set_does_not_corrupt_state(self):
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=10, ttl_seconds=60)
        assert isinstance(cache.lock, asyncio.Lock)

        async def _writer(i: int):
            for _ in range(20):
                cache.set(f"key-{i}", i)
                cache.get(f"key-{i}")
                await asyncio.sleep(0)

        await asyncio.gather(*(_writer(i) for i in range(20)))

        # No corruption: cache stayed within its bound and every remaining
        # entry is internally consistent (key matches the value it was set
        # with).
        assert len(cache) <= 10
        for key, value in cache.items():
            assert key == f"key-{value}"

    @pytest.mark.asyncio
    async def test_concurrent_cache_result_for_same_signature_is_wasteful_not_corrupting(
        self, mock_tool_registry
    ):
        """Pins the real risk this lock exists for: deduplicate_tool_calls()
        signatures RAW args while get_cached_result()/cache_result()
        signature NORMALIZED args, so two calls that survive dedup as
        "different" (e.g. because a parameter enforcer defaults one's
        missing argument to the value the other already had) can still
        converge on the same cache signature and be dispatched
        concurrently. Reproduced directly at the cache-method level (two
        concurrent cache_result() calls for the identical tool+args, as
        real convergence would produce) rather than via the full
        normalization pipeline: confirms the race is benign — one write
        clobbers the other, but both represent the exact same idempotent
        operation, so the surviving cached value is correct either way —
        without the LRUToolCache lock ever needing to be held for it.
        """
        from victor.agent.tool_pipeline import ToolCallResult, ToolPipeline, ToolPipelineConfig

        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=MagicMock(),
            config=ToolPipelineConfig(tool_budget=25),
        )
        args = {"path": "/a"}

        async def _race_writer(i: int) -> None:
            # Same tool+args as every other writer -> identical signature.
            result = ToolCallResult(tool_name="read", arguments=args, success=True, result=i)
            pipeline.cache_result("read", args, result)
            await asyncio.sleep(0)
            cached = pipeline.get_cached_result("read", args)
            assert cached is not None
            assert cached.success is True

        await asyncio.gather(*(_race_writer(i) for i in range(10)))

        # Exactly one entry for the shared signature — no corruption, no
        # duplicate/split entries from the concurrent unlocked writes.
        assert len(pipeline._idempotent_cache) == 1
        final = pipeline.get_cached_result("read", args)
        assert final is not None
        assert final.success is True


class TestToolPipelineNormalization:
    """Tests for argument normalization."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a tool pipeline for testing."""
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
        )

    def test_normalize_string_json_arguments(self, pipeline):
        """Test normalizing JSON string arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", '{"path": "file.py"}')
        assert args == {"path": "file.py"}

    def test_normalize_string_python_literal(self, pipeline):
        """Test normalizing Python literal string arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", "{'path': 'file.py'}")
        assert args == {"path": "file.py"}

    def test_normalize_invalid_string(self, pipeline):
        """Test normalizing invalid string falls back to value wrapper."""
        args, strategy = pipeline._normalize_arguments("test_tool", "just a string")
        assert args == {"value": "just a string"}

    def test_normalize_none_arguments(self, pipeline):
        """Test normalizing None arguments."""
        args, strategy = pipeline._normalize_arguments("test_tool", None)
        assert args == {}

    def test_normalize_dict_arguments(self, pipeline):
        """Test that dict arguments are passed through normalizer."""
        args, strategy = pipeline._normalize_arguments("test_tool", {"x": 1})
        assert "x" in args

    def test_normalize_tool_call_adds_bug_mode_to_code_search(self, pipeline):
        """Bug-style code_search queries should get mode='bugs' automatically."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "code_search",
            {"query": "json parsing crash on empty payload"},
        )

        assert tool_name == "code_search"
        assert args["mode"] == "bugs"
        assert strategy is not None

    def test_normalize_tool_call_defaults_shell_to_readonly(self, pipeline):
        """Canonical shell calls should default readonly=False."""
        tool_name, args, strategy = pipeline._normalize_tool_call(
            "shell",
            {"cmd": "git status"},
        )

        assert tool_name == "shell"
        assert args["readonly"] is False
        assert strategy is not None

    def test_normalize_tool_call_does_not_expand_shell_readonly(self, pipeline):
        """Legacy shell_readonly should no longer be rewritten by the pipeline."""
        tool_name, args, strategy = pipeline._normalize_tool_call(
            "shell_readonly",
            {"cmd": "git status"},
        )

        assert tool_name == "shell_readonly"
        assert "readonly" not in args
        assert strategy is not None

    def test_normalize_tool_call_reroutes_semantic_bug_search(self, pipeline):
        """Bug-style semantic_code_search queries should route to code_search."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "find similar regressions in auth"},
        )

        assert tool_name == "code_search"
        assert args["mode"] == "bugs"
        assert strategy is not None

    def test_normalize_tool_call_reroutes_call_graph_query_to_graph(self, pipeline):
        """Call-graph semantic queries should route to graph with traversal args."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "who calls parse_json"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "callers"
        assert args["node"] == "parse_json"
        assert args["depth"] == 2
        assert strategy is not None

    def test_normalize_tool_call_reroutes_neighbors_query_to_graph(self, pipeline):
        """Neighbor semantic queries should route to graph neighbor analysis."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "show neighbors of BaseProvider"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "neighbors"
        assert args["node"] == "BaseProvider"
        assert args["depth"] == 1
        assert strategy is not None

    def test_normalize_tool_call_reroutes_path_query_to_graph(self, pipeline):
        """Dependency-path semantic queries should route to graph path analysis."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "find dependency path between Parser and Provider"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "path"
        assert args["source"] == "Parser"
        assert args["target"] == "Provider"
        assert strategy is not None

    def test_normalize_tool_call_reroutes_module_pagerank_query_to_graph(self, pipeline):
        """Architecture-hotspot queries should route to graph module pagerank analysis."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "show the top 4 most important modules"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "module_pagerank"
        assert args["top_k"] == 4
        assert args["only_runtime"] is True
        assert args["include_callsites"] is True
        assert args["max_callsites"] == 3
        assert strategy is not None

    def test_normalize_tool_call_reroutes_file_dependency_query_to_graph(self, pipeline):
        """File-dependency semantic queries should route to graph file dependency analysis."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "show file dependencies for victor/agent/orchestrator.py"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "file_deps"
        assert args["file"] == "victor/agent/orchestrator.py"
        assert strategy is not None

    def test_normalize_tool_call_reroutes_module_dependents_query_to_graph(self, pipeline):
        """Reverse module-dependency queries should route to graph neighbors."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "what modules depend on auth"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "neighbors"
        assert args["node"] == "auth"
        assert args["depth"] == 1
        assert args["direction"] == "in"
        assert args["modules_only"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_module_dependencies_query_to_graph(self, pipeline):
        """Forward module-dependency queries should route to graph neighbors."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "what modules does auth depend on"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "neighbors"
        assert args["node"] == "auth"
        assert args["depth"] == 1
        assert args["direction"] == "out"
        assert args["modules_only"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_architecture_summary_query_to_graph(self, pipeline):
        """Architecture summary queries should route to structured graph ranking."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "summarize the architecture"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "module_pagerank"
        assert args["top_k"] == 5
        assert args["only_runtime"] is True
        assert args["include_callsites"] is True
        assert args["max_callsites"] == 3
        assert args["structured"] is True
        assert args["include_modules"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_downstream_dependency_query_to_graph(self, pipeline):
        """Deep downstream dependency queries should route to graph neighbors."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "show downstream dependencies of auth to depth 3"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "neighbors"
        assert args["node"] == "auth"
        assert args["depth"] == 3
        assert args["direction"] == "out"
        assert args["modules_only"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_scoped_architecture_summary_query_to_graph(
        self, pipeline
    ):
        """Scoped architecture summary queries should route to structured graph neighbors."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "summarize architecture of auth"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "neighbors"
        assert args["node"] == "auth"
        assert args["depth"] == 2
        assert args["direction"] == "both"
        assert args["structured"] is True
        assert args["include_modules"] is True
        assert args["include_symbols"] is True
        assert args["include_calls"] is True
        assert args["include_refs"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_file_dependents_query_to_graph(self, pipeline):
        """Reverse file-dependency queries should route to graph file dependencies."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "what files depend on victor/agent/orchestrator.py"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "file_deps"
        assert args["file"] == "victor/agent/orchestrator.py"
        assert args["direction"] == "in"
        assert strategy is not None

    def test_normalize_tool_call_reroutes_forward_file_dependencies_query_to_graph(self, pipeline):
        """Forward file-dependency queries should route to graph file dependencies."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "what files does victor/agent/orchestrator.py depend on"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "file_deps"
        assert args["file"] == "victor/agent/orchestrator.py"
        assert args["direction"] == "out"
        assert strategy is not None

    def test_normalize_tool_call_reroutes_file_architecture_summary_query_to_graph(self, pipeline):
        """File-scoped architecture queries should route to structured file graph output."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "summarize architecture around victor/agent/orchestrator.py"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "file_deps"
        assert args["file"] == "victor/agent/orchestrator.py"
        assert args["direction"] == "both"
        assert args["structured"] is True
        assert args["include_modules"] is True
        assert args["include_symbols"] is True
        assert args["include_calls"] is True
        assert args["include_refs"] is True
        assert strategy is not None

    def test_normalize_tool_call_reroutes_pagerank_query_to_graph(self, pipeline):
        """Centrality semantic queries should route to graph pagerank analysis."""
        pipeline.search_router = SearchRouter()

        tool_name, args, strategy = pipeline._normalize_tool_call(
            "semantic_code_search",
            {"query": "show the top 3 most central symbols"},
        )

        assert tool_name == "graph"
        assert args["mode"] == "pagerank"
        assert args["top_k"] == 3
        assert strategy is not None

    def test_get_call_signature_json(self, pipeline):
        """Test generating call signature."""
        sig = pipeline._get_call_signature("test_tool", {"a": 1, "b": 2})
        # Signature is a string (either hex hash from native or tool:args from fallback)
        assert isinstance(sig, str)
        assert len(sig) > 0
        # Same args should produce same signature
        sig2 = pipeline._get_call_signature("test_tool", {"a": 1, "b": 2})
        assert sig == sig2

    def test_get_call_signature_non_serializable(self, pipeline):
        """Test generating call signature with non-serializable args."""

        class NonSerializable:
            pass

        sig = pipeline._get_call_signature("test_tool", {"obj": NonSerializable()})
        # Signature is a string even with non-serializable args (uses str() fallback)
        assert isinstance(sig, str)
        assert len(sig) > 0


class TestToolPipelineAnalytics:
    """Tests for analytics tracking."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a pipeline with analytics enabled."""
        config = ToolPipelineConfig(enable_analytics=True)
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_analytics_updated_on_success(self, pipeline, mock_tool_executor):
        """Test that analytics are updated on successful execution."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert analytics["tools"]["test_tool"]["calls"] == 1
        assert analytics["tools"]["test_tool"]["successes"] == 1
        assert analytics["tools"]["test_tool"]["failures"] == 0

    @pytest.mark.asyncio
    async def test_analytics_updated_on_failure(self, pipeline, mock_tool_executor):
        """Test that analytics are updated on failed execution."""
        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert analytics["tools"]["test_tool"]["failures"] == 1

    @pytest.mark.asyncio
    async def test_analytics_tracks_time(self, pipeline, mock_tool_executor):
        """Test that analytics track execution time."""
        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        assert "total_time_ms" in analytics["tools"]["test_tool"]

    @pytest.mark.asyncio
    async def test_analytics_disabled(self, mock_tool_registry, mock_tool_executor):
        """Test that analytics are not updated when disabled."""
        config = ToolPipelineConfig(enable_analytics=False)
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=config,
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        await pipeline.execute_tool_calls(tool_calls, {})

        analytics = pipeline.get_analytics()
        # Tools dict should be empty since analytics disabled
        assert len(analytics["tools"]) == 0


class TestToolPipelineCallbacks:
    """Tests for callback error handling."""

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a tool pipeline for testing."""
        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
        )

    @pytest.mark.asyncio
    async def test_on_tool_start_exception_handled(self, pipeline, mock_tool_executor):
        """Test that exceptions in on_tool_start are handled."""

        def failing_start(name, args):
            raise Exception("Start callback error")

        pipeline.on_tool_start = failing_start

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        # Should not raise - exception is logged but execution continues
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1


class TestToolPipelineSearchRouting:
    """Tests for search router integration in tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_semantic_bug_query(
        self, mock_tool_registry, mock_tool_executor
    ):
        """semantic_code_search bug queries should execute as code_search(mode='bugs')."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="code_search",
                success=True,
                result={"matches": ["src/parser.py"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "semantic_code_search",
                    "arguments": {"query": "json parsing crash on empty payload"},
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "code_search"
        assert executed_kwargs["arguments"]["query"] == "json parsing crash on empty payload"
        assert executed_kwargs["arguments"]["mode"] == "bugs"
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_search_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Call-graph search queries should execute as graph tool calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 2, "results": ["main", "worker"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "who calls parse_json",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "callers"
        assert executed_kwargs["arguments"]["node"] == "parse_json"
        assert executed_kwargs["arguments"]["depth"] == 2
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_neighbors_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Neighbor search queries should execute as graph neighbor calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 2, "results": ["BaseModel", "ProviderConfig"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "show neighbors of BaseProvider",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "neighbors"
        assert executed_kwargs["arguments"]["node"] == "BaseProvider"
        assert executed_kwargs["arguments"]["depth"] == 1
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_path_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Dependency-path search queries should execute as graph path calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"found": True, "path": ["Parser", "Resolver", "Provider"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "find dependency path between Parser and Provider",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "path"
        assert executed_kwargs["arguments"]["source"] == "Parser"
        assert executed_kwargs["arguments"]["target"] == "Provider"
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_module_pagerank_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Architecture-hotspot search queries should execute as module graph ranking calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 4,
                    "results": ["agent", "providers", "storage", "framework"],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "show the top 4 most important modules",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "module_pagerank"
        assert executed_kwargs["arguments"]["top_k"] == 4
        assert executed_kwargs["arguments"]["only_runtime"] is True
        assert executed_kwargs["arguments"]["include_callsites"] is True
        assert executed_kwargs["arguments"]["max_callsites"] == 3
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_file_dependency_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """File-dependency search queries should execute as graph file dependency calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 3,
                    "results": [
                        "victor/providers/base.py",
                        "victor/tools/code_search_tool.py",
                    ],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "show file dependencies for victor/agent/orchestrator.py",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "file_deps"
        assert executed_kwargs["arguments"]["file"] == "victor/agent/orchestrator.py"
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_module_dependents_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Reverse module-dependency queries should execute as graph neighbor calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 2, "results": ["ui", "api"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "what modules depend on auth",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "neighbors"
        assert executed_kwargs["arguments"]["node"] == "auth"
        assert executed_kwargs["arguments"]["direction"] == "in"
        assert executed_kwargs["arguments"]["modules_only"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_module_dependencies_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Forward module-dependency queries should execute as graph neighbor calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 2, "results": ["storage", "providers"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "what modules does auth depend on",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "neighbors"
        assert executed_kwargs["arguments"]["node"] == "auth"
        assert executed_kwargs["arguments"]["direction"] == "out"
        assert executed_kwargs["arguments"]["modules_only"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_architecture_summary_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Architecture summary queries should execute as structured module graph ranking."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 5,
                    "results": ["agent", "providers", "storage", "framework", "tools"],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "summarize the architecture",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "module_pagerank"
        assert executed_kwargs["arguments"]["structured"] is True
        assert executed_kwargs["arguments"]["include_modules"] is True
        assert executed_kwargs["arguments"]["only_runtime"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_downstream_dependency_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Deep downstream dependency queries should execute as graph neighbor calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 3, "results": ["storage", "providers", "ui"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "show downstream dependencies of auth to depth 3",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "neighbors"
        assert executed_kwargs["arguments"]["node"] == "auth"
        assert executed_kwargs["arguments"]["depth"] == 3
        assert executed_kwargs["arguments"]["direction"] == "out"
        assert executed_kwargs["arguments"]["modules_only"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_scoped_architecture_summary_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Scoped architecture summary queries should execute as structured graph neighbors."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 4,
                    "results": ["auth", "storage", "providers", "tests"],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "summarize architecture of auth",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "neighbors"
        assert executed_kwargs["arguments"]["node"] == "auth"
        assert executed_kwargs["arguments"]["structured"] is True
        assert executed_kwargs["arguments"]["include_modules"] is True
        assert executed_kwargs["arguments"]["include_symbols"] is True
        assert executed_kwargs["arguments"]["include_calls"] is True
        assert executed_kwargs["arguments"]["include_refs"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_file_dependents_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Reverse file-dependency queries should execute as graph file dependencies."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 2,
                    "results": [
                        "victor/tools/code_search_tool.py",
                        "victor/providers/base.py",
                    ],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "what files depend on victor/agent/orchestrator.py",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "file_deps"
        assert executed_kwargs["arguments"]["file"] == "victor/agent/orchestrator.py"
        assert executed_kwargs["arguments"]["direction"] == "in"
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_forward_file_dependencies_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Forward file-dependency queries should execute as graph file dependencies."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 2,
                    "results": [
                        "victor/providers/base.py",
                        "victor/tools/code_search_tool.py",
                    ],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "what files does victor/agent/orchestrator.py depend on",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "file_deps"
        assert executed_kwargs["arguments"]["file"] == "victor/agent/orchestrator.py"
        assert executed_kwargs["arguments"]["direction"] == "out"
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_file_architecture_summary_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """File-scoped architecture queries should execute as structured file graph output."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={
                    "count": 4,
                    "results": ["orchestrator", "providers", "tools", "framework"],
                },
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "summarize architecture around victor/agent/orchestrator.py",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "file_deps"
        assert executed_kwargs["arguments"]["file"] == "victor/agent/orchestrator.py"
        assert executed_kwargs["arguments"]["direction"] == "both"
        assert executed_kwargs["arguments"]["structured"] is True
        assert executed_kwargs["arguments"]["include_modules"] is True
        assert executed_kwargs["arguments"]["include_symbols"] is True
        assert executed_kwargs["arguments"]["include_calls"] is True
        assert executed_kwargs["arguments"]["include_refs"] is True
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_reroutes_pagerank_query_to_graph(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Centrality search queries should execute as graph pagerank calls."""
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="graph",
                success=True,
                result={"count": 3, "results": ["auth", "router", "storage"]},
                error=None,
            )
        )
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            search_router=SearchRouter(),
        )

        result = await pipeline.execute_tool_calls(
            [
                {
                    "name": "code_search",
                    "arguments": {
                        "query": "show the top 3 most central symbols",
                        "directory": "src",
                    },
                }
            ],
            {},
        )

        mock_tool_executor.execute.assert_awaited_once()
        executed_kwargs = mock_tool_executor.execute.await_args.kwargs
        assert executed_kwargs["tool_name"] == "graph"
        assert executed_kwargs["arguments"]["mode"] == "pagerank"
        assert executed_kwargs["arguments"]["top_k"] == 3
        assert "directory" not in executed_kwargs["arguments"]
        assert result.successful_calls == 1

    @pytest.mark.asyncio
    async def test_on_tool_complete_exception_handled(self, pipeline, mock_tool_executor):
        """Test that exceptions in on_tool_complete are handled."""

        def failing_complete(result):
            raise Exception("Complete callback error")

        pipeline.on_tool_complete = failing_complete

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        # Should not raise - exception is logged but execution continues
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1


class TestPipelineExecutionResult:
    """Tests for PipelineExecutionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        from victor.agent.tool_pipeline import PipelineExecutionResult

        result = PipelineExecutionResult()

        assert result.results == []
        assert result.total_calls == 0
        assert result.successful_calls == 0
        assert result.failed_calls == 0
        assert result.skipped_calls == 0
        assert result.budget_exhausted is False
        assert result.parallel_execution_used is False
        assert result.parallel_speedup == 1.0


class TestFallbackOnToolFailure:
    """Tests for error recovery fallback in tool pipeline."""

    @pytest.fixture
    def fallback_pipeline(self, mock_tool_registry):
        """Create a pipeline with a failing executor that falls back."""
        executor = MagicMock()

        # First call fails (semantic_code_search), second succeeds (code_search)
        fail_result = ToolExecutionResult(
            tool_name="semantic_code_search",
            success=False,
            result=None,
            error="Tool 'semantic_code_search': dependencies missing for victor-coding",
        )
        success_result = ToolExecutionResult(
            tool_name="code_search",
            success=True,
            result={"matches": ["file.py:10: class MyClass"]},
            error=None,
        )
        executor.execute = AsyncMock(side_effect=[fail_result, success_result])

        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=10),
        )

    async def test_fallback_on_tool_failure(self, fallback_pipeline):
        """Test that failed tool call triggers fallback to alternative tool."""
        tool_calls = [{"name": "semantic_code_search", "arguments": {"query": "MyClass"}}]
        result = await fallback_pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1
        assert result.results[0].success is True
        assert result.results[0].tool_name == "code_search"

    async def test_no_fallback_when_tool_succeeds(self, mock_tool_registry, mock_tool_executor):
        """Test that successful tools don't trigger fallback."""
        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=ToolPipelineConfig(tool_budget=10),
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1
        # Executor should only be called once (no fallback)
        assert mock_tool_executor.execute.call_count == 1

    async def test_fallback_not_triggered_when_disabled(self, mock_tool_registry):
        """Test that fallback doesn't trigger when fallback tool is not enabled."""
        executor = MagicMock()
        fail_result = ToolExecutionResult(
            tool_name="semantic_code_search",
            success=False,
            result=None,
            error="dependencies missing for victor-coding",
        )
        executor.execute = AsyncMock(return_value=fail_result)

        registry = MagicMock()
        # semantic_code_search is enabled, but code_search (fallback) is not
        registry.is_tool_enabled = MagicMock(
            side_effect=lambda name: name == "semantic_code_search"
        )

        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=10),
        )

        tool_calls = [{"name": "semantic_code_search", "arguments": {"query": "test"}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.failed_calls == 1
        # Only one execute call (no fallback attempted)
        assert executor.execute.call_count == 1


class TestPerToolTimeout:
    """Tests for per-tool-call timeout in serial execution path."""

    async def test_tool_timeout_fires(self, mock_tool_registry):
        """Test that slow tool execution triggers timeout."""
        executor = MagicMock()

        async def slow_execute(**kwargs):
            await asyncio.sleep(10)  # Simulate slow tool
            return ToolExecutionResult(
                tool_name="slow_tool", success=True, result="done", error=None
            )

        executor.execute = slow_execute

        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                tool_budget=10,
                per_tool_timeout_seconds=0.1,  # 100ms timeout
            ),
        )

        tool_calls = [{"name": "slow_tool", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.failed_calls == 1
        assert result.results[0].success is False
        assert "timed out" in result.results[0].error

    async def test_tool_timeout_prefers_tool_timeout_property(self, mock_tool_registry):
        """Test per-tool timeout property overrides pipeline default."""

        class _Tool:
            timeout = 1.0

        async def slow_execute(**kwargs):
            await asyncio.sleep(0.1)
            return ToolExecutionResult(
                tool_name="slow_tool", success=True, result="done", error=None
            )

        executor = MagicMock()
        executor.get_tool_function = MagicMock(return_value=_Tool())
        executor.execute = slow_execute

        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                tool_budget=10,
                per_tool_timeout_seconds=0.05,  # Smaller than tool-specific timeout
            ),
        )

        tool_calls = [{"name": "slow_tool", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1
        assert result.results[0].success is True

    async def test_tool_timeout_prefers_decorator_attr(self, mock_tool_registry):
        """Test _tool_timeout attribute from tool wrapper is respected."""

        class _Tool:
            _tool_timeout = 0.05

        async def slow_execute(**kwargs):
            await asyncio.sleep(0.1)
            return ToolExecutionResult(
                tool_name="slow_tool", success=True, result="done", error=None
            )

        executor = MagicMock()
        executor.get_tool_function = MagicMock(return_value=_Tool())
        executor.execute = slow_execute

        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                tool_budget=10,
                per_tool_timeout_seconds=0.2,  # Larger than tool-specific timeout
            ),
        )

        tool_calls = [{"name": "slow_tool", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.failed_calls == 1
        assert result.results[0].success is False
        assert "timed out" in result.results[0].error

    async def test_fast_tool_not_affected_by_timeout(self, mock_tool_registry):
        """Test that fast tools complete normally within timeout."""
        executor = MagicMock()

        async def fast_execute(**kwargs):
            return ToolExecutionResult(
                tool_name="fast_tool", success=True, result="quick", error=None
            )

        executor.execute = fast_execute

        pipeline = ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=executor,
            config=ToolPipelineConfig(
                tool_budget=10,
                per_tool_timeout_seconds=5.0,
            ),
        )

        tool_calls = [{"name": "fast_tool", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls, {})

        assert result.successful_calls == 1
        assert result.results[0].success is True

    def test_config_default_timeout(self):
        """Test that default per_tool_timeout_seconds is 60."""
        config = ToolPipelineConfig()
        assert config.per_tool_timeout_seconds == 60.0

    def test_config_custom_timeout(self):
        """Test custom per_tool_timeout_seconds."""
        config = ToolPipelineConfig(per_tool_timeout_seconds=15.0)
        assert config.per_tool_timeout_seconds == 15.0
